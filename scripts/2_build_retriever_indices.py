# scripts/2_build_retriever_indices.py

import argparse
import yaml
import logging
import os
import sys
import json
import torch
import numpy as np # For saving embeddings if needed

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import from src
from retrieval.bm25_retriever import build_bm25_index
from retrieval.contriever_retriever import (
    load_contriever_model,
    encode_documents,
    build_faiss_index
)
from utils.logging_utils import setup_basic_logging # Optional

# Setup logging
# setup_basic_logging() # Or configure directly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_documents(filepath: str) -> List[str]:
    """Loads documents from the JSONL file created by preprocessing."""
    documents = []
    if not os.path.exists(filepath):
        logger.error(f"Processed documents file not found: {filepath}")
        return documents
    logger.info(f"Loading processed documents from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    documents.append(data['document'])
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"Skipping invalid line: {line.strip()}")
                    continue
        logger.info(f"Loaded {len(documents)} documents.")
    except IOError as e:
        logger.error(f"Error reading documents file {filepath}: {e}")
    return documents

def main():
    parser = argparse.ArgumentParser(description="Build retriever indices (BM25, Contriever/Faiss).")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=['bm25', 'contriever', 'all'],
        default='all',
        help="Which retriever index to build ('bm25', 'contriever', or 'all')."
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    logger.info(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {args.config}: {e}")
        sys.exit(1)

    # --- Load Processed Documents ---
    try:
        processed_docs_path = config['data_paths']['wiki_processed_subset_path']
    except KeyError:
        logger.error("Missing 'data_paths.wiki_processed_subset_path' in config.")
        sys.exit(1)

    documents = load_processed_documents(processed_docs_path)
    if not documents:
        logger.error("No documents loaded. Cannot build indices. Ensure '1_preprocess_wikipedia.py' ran successfully.")
        sys.exit(1)

    # --- Device Setup (for Contriever) ---
    try:
        device_cfg = config['hardware']['device']
        faiss_gpu_cfg = config['hardware']['faiss_use_gpu']
    except KeyError as e:
        logger.error(f"Missing hardware configuration key: {e}")
        sys.exit(1)

    if device_cfg:
        device = torch.device(device_cfg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for Contriever operations.")


    # --- Build BM25 Index ---
    if args.retriever in ['bm25', 'all']:
        logger.info("\n--- Building BM25 Index ---")
        try:
            bm25_dir = config['model_paths']['bm25_index_dir']
            bm25_filename = config['model_paths']['bm25_index_filename']
            bm25_save_path = os.path.join(bm25_dir, bm25_filename)
            logger.info(f"Target BM25 index path: {bm25_save_path}")
            build_bm25_index(documents, bm25_save_path)
            logger.info("BM25 index building complete.")
        except KeyError as e:
            logger.error(f"Missing BM25 configuration key: {e}")
        except Exception as e:
            logger.exception(f"An error occurred during BM25 index building: {e}")
            # Decide whether to exit or continue with Contriever if 'all'
            if args.retriever == 'bm25': sys.exit(1)

    # --- Build Contriever Index ---
    if args.retriever in ['contriever', 'all']:
        logger.info("\n--- Building Contriever/Faiss Index ---")
        try:
            contriever_model_name = config['model_paths']['contriever_model_name']
            faiss_dir = config['model_paths']['contriever_index_dir']
            faiss_filename = config['model_paths']['contriever_index_filename']
            faiss_save_path = os.path.join(faiss_dir, faiss_filename)
            # Optional: embedding path
            embeddings_filename = config['model_paths'].get('contriever_embeddings_filename') # Use .get for optional
            embeddings_save_path = os.path.join(faiss_dir, embeddings_filename) if embeddings_filename else None

            # 1. Load Model
            logger.info("Loading Contriever model and tokenizer...")
            contriever_model, contriever_tokenizer = load_contriever_model(contriever_model_name, device)

            # 2. Encode Documents
            logger.info("Encoding documents with Contriever...")
            # Get batch size from config if specified, e.g., under a 'contriever' section
            batch_size = config.get('contriever', {}).get('encoding_batch_size', 32)
            max_length = config.get('contriever', {}).get('encoding_max_length', 256)
            doc_embeddings = encode_documents(
                docs=documents,
                model=contriever_model,
                tokenizer=contriever_tokenizer,
                device=device,
                batch_size=batch_size,
                max_length=max_length
            )

            # Optional: Save embeddings
            if embeddings_save_path:
                 os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
                 logger.info(f"Saving document embeddings to {embeddings_save_path}...")
                 np.save(embeddings_save_path, doc_embeddings)
                 logger.info("Embeddings saved.")

            # 3. Build Faiss Index
            logger.info(f"Building Faiss index at {faiss_save_path}...")
            build_faiss_index(
                doc_embeddings=doc_embeddings,
                save_path=faiss_save_path,
                use_gpu=faiss_gpu_cfg
            )
            logger.info("Contriever/Faiss index building complete.")

        except KeyError as e:
            logger.error(f"Missing Contriever configuration key: {e}")
        except ImportError:
             logger.error("Faiss library not found. Please install 'faiss-cpu' or 'faiss-gpu'.")
        except Exception as e:
            logger.exception(f"An error occurred during Contriever/Faiss index building: {e}")
            if args.retriever == 'contriever': sys.exit(1)

    logger.info("\nRetriever index building script finished.")

if __name__ == "__main__":
    main()
