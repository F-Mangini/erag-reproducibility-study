import torch
import faiss
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import time
from tqdm.auto import tqdm # For progress bars

def load_contriever_model(model_name: str = "facebook/contriever", device: torch.device = None) -> tuple[AutoModel, AutoTokenizer]:
    """
    Loads the Contriever model and tokenizer.

    Args:
        model_name: The name of the Contriever model on Hugging Face Hub.
        device: The torch device to load the model onto (e.g., 'cuda', 'cpu').
                If None, defaults to cuda if available, else cpu.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Contriever tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading Contriever model: {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval() # Set to evaluation mode
    print(f"Contriever model loaded successfully on device: {device}")
    return model, tokenizer

def encode_documents(docs: list[str], model: AutoModel, tokenizer: AutoTokenizer, device: torch.device, batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """
    Encodes a list of documents into normalized embeddings using Contriever.

    Args:
        docs: A list of document strings.
        model: The loaded Contriever model.
        tokenizer: The loaded Contriever tokenizer.
        device: The torch device the model is on.
        batch_size: The batch size for encoding.
        max_length: The maximum sequence length for the tokenizer.

    Returns:
        A NumPy array containing the normalized document embeddings.
    """
    all_embeddings = []
    model.eval() # Ensure model is in eval mode
    print(f"Encoding {len(docs)} documents...")
    with torch.no_grad():
        for i in tqdm(range(0, len(docs), batch_size), desc="Encoding Batches"):
            batch_docs = docs[i:i+batch_size]
            inputs = tokenizer(batch_docs, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # Mean pooling (weighted by attention mask)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            # Normalize embeddings (L2 norm) - Crucial for Inner Product
            normalized_embeddings = F.normalize(mean_embeddings, p=2, dim=1)

            all_embeddings.append(normalized_embeddings.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Document encoding complete. Shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(doc_embeddings: np.ndarray, save_path: str, use_gpu: bool = False) -> faiss.Index:
    """
    Builds a Faiss index (IndexFlatIP) from document embeddings and saves it.

    Args:
        doc_embeddings: A NumPy array of document embeddings. Embeddings should
                        ideally be L2-normalized *before* calling this function,
                        but normalization is applied again for safety.
        save_path: Path to save the Faiss index file.
        use_gpu: Whether to attempt building/using the index on GPU (requires faiss-gpu).

    Returns:
        The constructed Faiss index object.
    """
    embedding_dim = doc_embeddings.shape[1]
    print(f"Building Faiss IndexFlatIP with dimension {embedding_dim}...")

    # Faiss requires float32
    if doc_embeddings.dtype != np.float32:
        print("Converting embeddings to float32 for Faiss...")
        doc_embeddings = doc_embeddings.astype(np.float32)

    # Normalize embeddings *again* just before adding (safety measure for IP)
    print("Normalizing embeddings for Faiss (redundant but safe)...")
    faiss.normalize_L2(doc_embeddings)

    index = faiss.IndexFlatIP(embedding_dim)

    if use_gpu and faiss.get_num_gpus() > 0:
        print(f"Moving Faiss index to GPU...")
        try:
            res = faiss.StandardGpuResources() # Use default GPU resources
            index = faiss.index_cpu_to_gpu(res, 0, index) # Move to GPU 0
            print("Faiss index successfully moved to GPU.")
        except Exception as e:
            print(f"Warning: Failed to move Faiss index to GPU: {e}. Using CPU.")
            use_gpu = False # Fallback to CPU
    else:
        print("Using CPU for Faiss index.")

    print(f"Adding {doc_embeddings.shape[0]} documents to Faiss index...")
    start_time = time.time()
    index.add(doc_embeddings)
    add_time = time.time() - start_time
    print(f"Documents added in {add_time:.2f} seconds. Index size: {index.ntotal}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving Faiss index to {save_path}...")
    # If index is on GPU, must move back to CPU to save
    if use_gpu and hasattr(index, 'index'): # Check if it's a GpuIndex
         index_to_save = faiss.index_gpu_to_cpu(index)
    else:
         index_to_save = index
    faiss.write_index(index_to_save, save_path)
    print("Faiss index saved successfully.")

    return index # Return the potentially GPU index for immediate use


def load_faiss_index(index_path: str, use_gpu: bool = False) -> faiss.Index:
    """
    Loads a Faiss index from disk.

    Args:
        index_path: Path to the saved Faiss index file.
        use_gpu: Whether to attempt moving the loaded index to GPU (requires faiss-gpu).

    Returns:
        The loaded Faiss index object (potentially on GPU).

    Raises:
        FileNotFoundError: If the index file does not exist.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Faiss index file not found at: {index_path}")

    print(f"Loading Faiss index from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"Faiss index loaded successfully. Index size: {index.ntotal}")

    if use_gpu and faiss.get_num_gpus() > 0:
        print(f"Attempting to move loaded Faiss index to GPU...")
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("Faiss index successfully moved to GPU.")
        except Exception as e:
            print(f"Warning: Failed to move loaded Faiss index to GPU: {e}. Keeping on CPU.")
    else:
        print("Using CPU for loaded Faiss index.")

    return index


def dense_retrieve(query: str, index: faiss.Index, model: AutoModel, tokenizer: AutoTokenizer, device: torch.device, original_docs: list[str], k: int = 5, max_length: int = 256) -> tuple[list[int], list[str], list[float]]:
    """
    Retrieves the top-k documents for a query using Contriever and a Faiss index (IP).

    Args:
        query: The search query string.
        index: The loaded Faiss index object (expecting normalized vectors and IP metric).
        model: The loaded Contriever model.
        tokenizer: The loaded Contriever tokenizer.
        device: The torch device the model is on.
        original_docs: List of original document strings.
        k: The number of top documents to retrieve.
        max_length: Max sequence length for tokenizing the query.

    Returns:
        A tuple containing:
        - top_indices: List of integer indices of the top documents in original_docs.
        - top_docs: List of the top document strings.
        - top_distances: List of the scores (inner product/cosine similarity) for the top documents.
    """
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        # Mean pooling for query
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask

        # Normalize the query embedding
        normalized_embedding = F.normalize(mean_embedding, p=2, dim=1)
        query_embedding_np = normalized_embedding.cpu().numpy()

        # Faiss IP search requires float32
        if query_embedding_np.dtype != np.float32:
             query_embedding_np = query_embedding_np.astype(np.float32)

    # Faiss search expects normalized query for IndexFlatIP search
    # Although we normalized above, doing it again ensures compatibility if Faiss requires it internally
    faiss.normalize_L2(query_embedding_np)

    # Ensure k is not larger than the number of documents in the index
    actual_k = min(k, index.ntotal)
    if actual_k == 0:
        print(f"Warning: Index for query '{query[:50]}...' is empty. Returning empty.")
        return [], [], []
    if actual_k < k:
         print(f"Warning: Requested k={k} but only {index.ntotal} documents in index. Retrieving {actual_k}.")


    # Search the Faiss index
    distances, indices = index.search(query_embedding_np, actual_k)

    # Indices is usually shape (1, k), distances is shape (1, k)
    top_indices = indices[0].tolist()
    top_distances = distances[0].tolist()
    top_docs = [original_docs[i] for i in top_indices] # Need original docs mapping

    return top_indices, top_docs, top_distances
