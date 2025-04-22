import torch
import time
import logging
from typing import Dict, List, Any
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def T5_text_generator(
    queries_and_documents: Dict[str, List[str]],
    model: T5ForConditionalGeneration, # Type hint for clarity
    tokenizer: T5Tokenizer,           # Type hint for clarity
    device: torch.device,
    max_input_len: int = 512,
    max_output_len: int = 128,
    num_beams: int = 4,
    max_docs_per_query: int = 50, # Add limit for safety/consistency
    **generate_kwargs: Any
    ) -> Dict[str, str]:
    """
    Generates answers for multiple queries using a fine-tuned T5 FiD model.

    Processes each query individually, encoding its associated documents,
    running the encoder, reshaping, and then running the decoder's generate method.

    Args:
        queries_and_documents: Dictionary where keys are query strings and
                               values are lists of retrieved document strings.
        model: The loaded fine-tuned T5ForConditionalGeneration model (on device).
        tokenizer: The loaded corresponding T5Tokenizer.
        device: The torch.device where the model is located.
        max_input_len: Max sequence length for each (query + doc) encoder input.
        max_output_len: Max sequence length for the generated answer (decoder).
        num_beams: Number of beams for beam search generation.
        max_docs_per_query: The maximum number of documents to actually use per
                            query, even if more are provided in the input dict.
                            Helps manage memory/compute during inference.
        **generate_kwargs: Additional keyword arguments passed to model.generate().

    Returns:
        Dictionary where keys are the input query strings and values are the
        corresponding generated answer strings (or an error message).
    """
    model.eval() # Ensure model is in evaluation mode
    results: Dict[str, str] = {}
    pad_token_id = tokenizer.pad_token_id # Used? Not directly here, but good practice

    num_queries = len(queries_and_documents)
    logger.info(f"Starting T5-FiD generation for {num_queries} queries...")
    start_time_total = time.time()

    processed_queries = 0
    # Iterate through each query and its associated documents
    for query, retrieved_docs in tqdm(queries_and_documents.items(), desc="Generating Answers"):
        processed_queries += 1
        # logger.debug(f"Processing query {processed_queries}/{num_queries}: \"{query[:60]}...\"") # Can be verbose

        if not retrieved_docs:
            logger.warning(f"No documents provided for query {processed_queries}. Assigning error message.")
            results[query] = "Error: No documents provided for this query."
            continue

        # Limit the number of documents used per query
        docs_to_process = retrieved_docs[:max_docs_per_query]
        num_docs = len(docs_to_process)
        # logger.debug(f"  Using {num_docs} documents (max {max_docs_per_query}).")


        all_input_ids = []
        all_attention_masks = []

        # --- Core FiD Generation Logic (applied per query) ---

        # 1. Preprocess and Tokenize each document for the CURRENT query
        for doc in docs_to_process:
            input_text = f"question: {query} context: {doc}"
            # Use padding='max_length' for consistent tensor shapes for encoder batching
            encoding = tokenizer(
                input_text,
                truncation=True,
                max_length=max_input_len,
                padding="max_length", # Pad encoder inputs
                return_attention_mask=True,
                add_special_tokens=True # Add EOS token etc.
            )
            all_input_ids.append(torch.tensor(encoding['input_ids']))
            all_attention_masks.append(torch.tensor(encoding['attention_mask']))

        if not all_input_ids:
             logger.warning(f"Tokenization resulted in empty inputs for query {processed_queries}. Skipping.")
             results[query] = "Error: Failed to tokenize input documents."
             continue

        # 2. Stack inputs and move to device for CURRENT query's documents
        # Shape: (num_docs, max_input_len)
        input_ids_stacked = torch.stack(all_input_ids).to(device)
        attention_mask_stacked = torch.stack(all_attention_masks).to(device)
        # logger.debug(f"  Encoder input shape: {input_ids_stacked.shape}")

        # Use no_grad context for efficiency during inference
        with torch.no_grad():
            # 3. Encoder Pass for CURRENT query documents
            # Input shape: (num_docs, max_input_len)
            # Output shape: (num_docs, max_input_len, hidden_size)
            raw_encoder_outputs = model.encoder(
                input_ids=input_ids_stacked,
                attention_mask=attention_mask_stacked,
                return_dict=True
            )
            encoder_last_hidden_state = raw_encoder_outputs.last_hidden_state

            # 4. Reshape Encoder Outputs for Generate method (FiD fusion)
            # Concatenate hidden states along sequence length dimension
            # New shape: (1, num_docs * max_input_len, hidden_size)
            bsz = 1 # We process one query at a time for generation
            encoder_hidden_states_reshaped = encoder_last_hidden_state.view(
                bsz,
                num_docs * max_input_len,
                encoder_last_hidden_state.size(-1) # hidden_size
            )
            # logger.debug(f"  Reshaped encoder hidden state shape: {encoder_hidden_states_reshaped.shape}")

            # Create a BaseModelOutput object for the generate method
            encoder_outputs_for_generate = BaseModelOutput(
                last_hidden_state=encoder_hidden_states_reshaped
                # Other outputs like hidden_states, attentions are not needed for basic generation
            )

            # 5. Prepare Cross-Attention Mask for FiD decoder
            # Concatenate attention masks along sequence length dimension
            # New shape: (1, num_docs * max_input_len)
            cross_attention_mask_reshaped = attention_mask_stacked.view(bsz, num_docs * max_input_len)
            # logger.debug(f"  Reshaped cross-attention mask shape: {cross_attention_mask_reshaped.shape}")


            # 6. Generation (Decoder) using model.generate()
            try:
                generated_ids = model.generate(
                    encoder_outputs=encoder_outputs_for_generate, # Use the reshaped outputs
                    attention_mask=cross_attention_mask_reshaped, # This is the crucial mask for the decoder
                    max_length=max_output_len,
                    num_beams=num_beams,
                    early_stopping=True,
                    # Add other kwargs like temperature, top_k, top_p if needed
                    **generate_kwargs
                )
                # logger.debug(f"  Generated token IDs shape: {generated_ids.shape}")

                # 7. Decode the generated token IDs back to text
                # generated_ids shape is usually (bsz, seq_len), bsz=1 here
                generated_text = tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                results[query] = generated_text.strip()

            except Exception as e:
                 logger.error(f"Error during model.generate() for query {processed_queries}: {e}")
                 results[query] = "Error: Generation failed."
                 # Consider adding more specific error handling if needed

    end_time_total = time.time()
    logger.info(f"Finished generating all answers in {end_time_total - start_time_total:.2f} seconds.")

    # Return the dictionary containing {query: answer} pairs
    return results

# Example of how to run this module directly (optional)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Dummy Data and Model Setup for Testing ---
    # NOTE: This requires a valid T5 model/tokenizer path and a device.
    #       Replace "path/to/your/finetuned_model" with the actual path.
    #       This is just a basic structure check.
    model_path = "models/t5_fid_finetuned" # CHANGE THIS
    if not os.path.exists(model_path):
         print(f"Model path '{model_path}' not found. Skipping direct execution example.")
    else:
        try:
            logger.info("--- Running Direct Execution Example ---")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")

            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            model.to(device)
            model.eval()

            dummy_queries_docs = {
                "What is the capital of France?": ["France is a country in Europe.", "Paris is known for the Eiffel Tower.", "The capital is Paris, located on the Seine river."],
                "Who wrote Hamlet?": ["Hamlet is a famous play.", "William Shakespeare wrote many tragedies.", "The author is William Shakespeare."]
            }

            generated_answers = T5_text_generator(
                queries_and_documents=dummy_queries_docs,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_input_len=512,
                max_output_len=64,
                num_beams=2,
                max_docs_per_query=3 # Use only 3 docs for this test
            )

            logger.info("--- Example Generation Results ---")
            for q, a in generated_answers.items():
                print(f"Query: {q}")
                print(f"Answer: {a}\n")

        except Exception as e:
            print(f"\nError running direct execution example: {e}")
            print("Ensure the model path is correct and dependencies are installed.")
