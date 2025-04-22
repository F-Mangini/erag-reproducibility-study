import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, PreTrainedTokenizer # More general type hint
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class QA_Dataset_FiD(Dataset):
    """
    PyTorch Dataset for Fusion-in-Decoder (FiD) models.

    Each item consists of a query, a list of retrieved documents, and a gold answer.
    The __getitem__ method tokenizes the query paired with each document individually.
    """
    def __init__(
        self,
        augmented_data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer, # Use base class for flexibility
        max_input_length: int = 512,
        max_target_length: int = 128,
        max_docs_per_item: Optional[int] = None # Optional: Limit docs loaded per item
    ):
        """
        Args:
            augmented_data: A list of dictionaries, where each dictionary has
                            'query' (str), 'retrieved_docs' (List[str]),
                            and 'gold_answer' (str).
            tokenizer: The tokenizer (e.g., T5Tokenizer).
            max_input_length: Maximum sequence length for encoder inputs (query + context).
            max_target_length: Maximum sequence length for decoder outputs (answer).
            max_docs_per_item: If specified, only use the top N documents from
                               'retrieved_docs' for each item during __getitem__.
                               Useful if the data contains more docs than needed for training.
        """
        self.data = augmented_data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.max_docs_per_item = max_docs_per_item
        logger.info(f"Initialized QA_Dataset_FiD with {len(self.data)} samples.")
        if self.max_docs_per_item:
            logger.info(f"Limiting documents per item to a maximum of {self.max_docs_per_item}.")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves and preprocesses the idx-th sample.

        Returns:
            A dictionary containing:
            - 'input_ids_list': List of token ID lists for each (query + doc) pair.
            - 'attention_mask_list': List of attention mask lists for each pair.
            - 'labels': List of token IDs for the target answer.
        """
        item = self.data[idx]
        query = item["query"]
        gold_answer = item["gold_answer"]
        retrieved_docs = item["retrieved_docs"]

        # Apply max_docs_per_item limit if specified
        if self.max_docs_per_item is not None:
             docs_to_process = retrieved_docs[:self.max_docs_per_item]
        else:
             docs_to_process = retrieved_docs

        input_encodings_list = []
        attention_mask_list = []

        if not docs_to_process:
             logger.warning(f"No documents found for item index {idx}, query: '{query[:50]}...'. This might cause issues in collation.")
             # Return dummy/empty values to avoid crashing collate_fn, needs careful handling there.
             # Or, could raise an error, or filter such items during data loading.
             # Let's return at least one dummy entry for now, collate_fn handles padding.
             docs_to_process.append("No context available.") # Add a placeholder

        for doc in docs_to_process:
            # Format expected by T5/FiD: "question: ... context: ..."
            input_text = f"question: {query} context: {doc}"
            # Tokenize without padding here; padding happens in collate_fn
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_input_length,
                padding=False # Padding will be done in collate_fn
            )
            input_encodings_list.append(input_encoding['input_ids'])
            attention_mask_list.append(input_encoding['attention_mask'])

        # Tokenize the target answer
        target_encoding = self.tokenizer(
            gold_answer,
            truncation=True,
            max_length=self.max_target_length,
            padding=False # Padding will be done in collate_fn
        )

        return {
            'input_ids_list': input_encodings_list,
            'attention_mask_list': attention_mask_list,
            'labels': target_encoding['input_ids']
        }


def collate_fn_fid(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_docs_per_item: int, # Max docs allowed *per item in the batch*
    max_input_length: int = 512,
    max_target_length: int = 128
    ) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of samples from QA_Dataset_FiD for FiD training/evaluation.

    Pads each document's input_ids/attention_mask to max_input_length.
    Pads the number of documents per item up to max_docs_per_item.
    Pads the labels to max_target_length.

    Args:
        batch: A list of dictionaries, where each dict is an output of QA_Dataset_FiD.__getitem__.
        tokenizer: The tokenizer used.
        max_docs_per_item: The target number of documents per item in the output tensors.
                           Items with fewer documents will be padded.
        max_input_length: The target length for each document's input_ids/attention_mask.
        max_target_length: The target length for the labels.

    Returns:
        A dictionary containing padded batch tensors:
        - 'input_ids': Tensor of shape (batch_size, max_docs_per_item, max_input_length).
        - 'attention_mask': Tensor of shape (batch_size, max_docs_per_item, max_input_length).
        - 'labels': Tensor of shape (batch_size, max_target_length).
    """
    if not batch: # Handle empty batch case
        return {
            'input_ids': torch.empty(0, max_docs_per_item, max_input_length, dtype=torch.long),
            'attention_mask': torch.empty(0, max_docs_per_item, max_input_length, dtype=torch.long),
            'labels': torch.empty(0, max_target_length, dtype=torch.long)
        }

    pad_token_id = tokenizer.pad_token_id
    # T5 uses -100 for ignored label tokens during loss calculation
    label_pad_token_id = -100

    all_batch_input_ids = []
    all_batch_attention_masks = []
    all_batch_labels = []

    for item in batch:
        item_input_ids_padded_docs = []
        item_attention_masks_padded_docs = []

        input_ids_list = item['input_ids_list']
        attention_mask_list = item['attention_mask_list']
        labels = item['labels']

        num_docs_in_item = len(input_ids_list)
        num_docs_to_process = min(num_docs_in_item, max_docs_per_item)

        # 1. Pad or truncate each document's tokens to max_input_length
        for i in range(max_docs_per_item):
            if i < num_docs_to_process:
                # Get tokens and mask for the current doc
                input_ids = input_ids_list[i]
                attention_mask = attention_mask_list[i]

                # Pad input_ids
                padding_length_input = max_input_length - len(input_ids)
                padded_input_ids = input_ids + ([pad_token_id] * padding_length_input)

                # Pad attention_mask (usually with 0 for padding)
                padding_length_mask = max_input_length - len(attention_mask)
                padded_attention_mask = attention_mask + ([0] * padding_length_mask)

            else:
                # If item has fewer than max_docs_per_item, pad with dummy docs
                padded_input_ids = [pad_token_id] * max_input_length
                padded_attention_mask = [0] * max_input_length

            item_input_ids_padded_docs.append(torch.tensor(padded_input_ids, dtype=torch.long))
            item_attention_masks_padded_docs.append(torch.tensor(padded_attention_mask, dtype=torch.long))

        # Stack the padded docs for this item
        # Shape: (max_docs_per_item, max_input_length)
        all_batch_input_ids.append(torch.stack(item_input_ids_padded_docs))
        all_batch_attention_masks.append(torch.stack(item_attention_masks_padded_docs))

        # 2. Pad labels to max_target_length
        label_padding_length = max_target_length - len(labels)
        padded_labels = labels + ([label_pad_token_id] * label_padding_length)
        all_batch_labels.append(torch.tensor(padded_labels, dtype=torch.long))

    # 3. Stack all items in the batch
    # Final shapes:
    # (batch_size, max_docs_per_item, max_input_length)
    # (batch_size, max_docs_per_item, max_input_length)
    # (batch_size, max_target_length)
    batch_input_ids = torch.stack(all_batch_input_ids)
    batch_attention_masks = torch.stack(all_batch_attention_masks)
    batch_labels = torch.stack(all_batch_labels)

    return {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_masks,
        'labels': batch_labels
    }

# Example usage (optional, for testing the file directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("--- Testing FiD Dataset and Collate Function ---")

    # Dummy data
    dummy_augmented = [
        {'query': 'q1', 'retrieved_docs': ['d1a', 'd1b long doc'], 'gold_answer': 'ans1'},
        {'query': 'q2', 'retrieved_docs': ['d2a short'], 'gold_answer': 'answer two is longer'},
        {'query': 'q3', 'retrieved_docs': ['d3a', 'd3b', 'd3c', 'd3d'], 'gold_answer': 'a3'},
    ]

    # Dummy tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Create dataset
    dataset = QA_Dataset_FiD(
        dummy_augmented,
        tokenizer,
        max_input_length=32,
        max_target_length=16,
        max_docs_per_item=3 # Limit to 3 docs for this test
        )

    # Test __getitem__
    sample = dataset[0]
    logger.info(f"Sample 0 from dataset:")
    logger.info(f"  Input IDs List Length: {len(sample['input_ids_list'])}")
    logger.info(f"  Attention Mask List Length: {len(sample['attention_mask_list'])}")
    logger.info(f"  Labels: {sample['labels']}")
    logger.info(f"  Decoded Labels: {tokenizer.decode(sample['labels'], skip_special_tokens=True)}")

    # Test collate_fn
    # Create a dummy batch
    batch_samples = [dataset[i] for i in range(len(dataset))]

    MAX_DOCS_COLLATE = 3 # Use the same limit as dataset or different for testing collate padding
    MAX_INPUT_LEN_COLLATE = 32
    MAX_TARGET_LEN_COLLATE = 16

    collated_batch = collate_fn_fid(
        batch_samples,
        tokenizer,
        max_docs_per_item=MAX_DOCS_COLLATE,
        max_input_length=MAX_INPUT_LEN_COLLATE,
        max_target_length=MAX_TARGET_LEN_COLLATE
        )

    logger.info("\nCollated Batch Shapes:")
    logger.info(f"  Input IDs: {collated_batch['input_ids'].shape}")
    logger.info(f"  Attention Mask: {collated_batch['attention_mask'].shape}")
    logger.info(f"  Labels: {collated_batch['labels'].shape}")

    # Verify shapes
    assert collated_batch['input_ids'].shape == (len(batch_samples), MAX_DOCS_COLLATE, MAX_INPUT_LEN_COLLATE)
    assert collated_batch['attention_mask'].shape == (len(batch_samples), MAX_DOCS_COLLATE, MAX_INPUT_LEN_COLLATE)
    assert collated_batch['labels'].shape == (len(batch_samples), MAX_TARGET_LEN_COLLATE)

    logger.info("Shapes verified successfully.")

    # Decode one example from the batch
    item_idx = 0
    doc_idx = 0
    decoded_input = tokenizer.decode(collated_batch['input_ids'][item_idx, doc_idx, :], skip_special_tokens=False) # Show padding tokens
    decoded_label = tokenizer.decode(collated_batch['labels'][item_idx, :], skip_special_tokens=False)
    logger.info(f"\nDecoded Example (Item {item_idx}, Doc {doc_idx}):")
    logger.info(f"  Input: {decoded_input}")
    logger.info(f"  Label: {decoded_label}")
