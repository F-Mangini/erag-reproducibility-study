import json
import nltk
import string
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import os
import time

# --- NLTK Setup ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK 'stopwords' not found. Downloading...")
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

try:
    # Test if punkt is available, avoid unnecessary download
    word_tokenize("test sentence")
except LookupError:
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
# Create the punctuation translation table once
punctuation_table = str.maketrans('', '', string.punctuation)
# --- End NLTK Setup ---


def preprocess_text(text: str) -> list[str]:
    """
    Applies preprocessing steps specific to BM25:
    lowercase, remove punctuation, tokenize, remove stopwords, stem.

    Args:
        text: The input text string.

    Returns:
        A list of processed (stemmed) tokens.
    """
    if not isinstance(text, str):
        # Handle potential non-string inputs gracefully
        print(f"Warning: Received non-string input in preprocess_text: {type(text)}. Returning empty list.")
        return []
    lowered = text.lower()
    no_punct = lowered.translate(punctuation_table)
    tokens = word_tokenize(no_punct)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens


def build_bm25_index(documents: list[str], save_path: str) -> BM25Okapi:
    """
    Builds a BM25Okapi index from a list of documents and saves it to disk.

    Args:
        documents: A list of document strings.
        save_path: Path to save the pickled BM25 index object.

    Returns:
        The constructed BM25Okapi index object.
    """
    print("Preprocessing documents for BM25...")
    start_time = time.time()
    tokenized_docs = [preprocess_text(doc) for doc in documents]
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing finished in {preprocessing_time:.2f} seconds.")

    print("Building BM25 index...")
    start_time = time.time()
    bm25_index = BM25Okapi(tokenized_docs)
    building_time = time.time() - start_time
    print(f"BM25 index built in {building_time:.2f} seconds.")

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving BM25 index to {save_path}...")
    with open(save_path, "wb") as f_out:
        pickle.dump(bm25_index, f_out)
    print("BM25 index saved successfully.")

    return bm25_index


def load_bm25_index(index_path: str) -> BM25Okapi:
    """
    Loads a pickled BM25Okapi index from disk.

    Args:
        index_path: Path to the pickled BM25 index object.

    Returns:
        The loaded BM25Okapi index object.

    Raises:
        FileNotFoundError: If the index file does not exist.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"BM25 index file not found at: {index_path}")

    print(f"Loading BM25 index from {index_path}...")
    with open(index_path, "rb") as f_in:
        bm25_index = pickle.load(f_in)
    print("BM25 index loaded successfully.")
    return bm25_index


def bm25_retrieve(query: str, bm25_index: BM25Okapi, original_docs: list[str], k: int = 5) -> tuple[list[int], list[str], list[float]]:
    """
    Retrieves the top-k documents for a query using a BM25 index.

    Args:
        query: The search query string.
        bm25_index: The loaded/built BM25Okapi index object.
        original_docs: The list of original document strings (used to return actual text).
        k: The number of top documents to retrieve.

    Returns:
        A tuple containing:
        - top_indices: List of integer indices of the top documents in original_docs.
        - top_docs: List of the top document strings.
        - top_scores: List of the BM25 scores for the top documents.
    """
    # Apply the same preprocessing to the query
    tokenized_query = preprocess_text(query)
    if not tokenized_query:
        print(f"Warning: Query '{query[:50]}...' resulted in empty tokens after preprocessing.")
        return [], [], []

    # Get scores for all documents
    try:
        scores = bm25_index.get_scores(tokenized_query)
    except ValueError as e:
        # Handle potential errors if query terms are not in vocabulary (less common with stemming)
         print(f"Warning: Error getting scores for query '{query[:50]}...': {e}. Returning empty.")
         return [], [], []

    # Get the indices of the top-k scores
    # Ensure k is not larger than the number of documents
    num_docs_total = len(original_docs) # or bm25_index.corpus_size if available and reliable
    actual_k = min(k, num_docs_total)
    if actual_k < k:
         print(f"Warning: Requested k={k} but only {num_docs_total} documents in index. Retrieving {actual_k}.")

    # Get indices of top scores. Use argpartition for efficiency if k << num_docs_total
    # For simplicity and smaller k, argsort is fine.
    if actual_k > 0:
        top_n_indices = np.argsort(scores)[::-1][:actual_k]
    else:
        top_n_indices = np.array([], dtype=int)


    # Retrieve the actual documents and their scores
    top_docs = [original_docs[i] for i in top_n_indices]
    top_scores = [scores[i] for i in top_n_indices]

    return top_n_indices.tolist(), top_docs, top_scores
