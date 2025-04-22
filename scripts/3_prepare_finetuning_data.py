# scripts/3_prepare_finetuning_data.py

import argparse
import yaml
import logging
import os
import sys
import json # Needed for loading retrieval results if in JSON

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import from src
from data_processing.dataset_utils import load_all_nq_expected_outputs, augment_with_retrieved_documents
from utils.logging_utils import setup_basic_logging # Optional
# Reuse or define a function to load retrieval results
# Let's define it here for simplicity if not using utils everywhere
def load_retrieval_results(filepath: str) -> dict:
    """Loads pre-computed retrieval results from a file (assuming JSON)."""
    results = {}
    if not os.path.exists(filepath):
        logger.error(f"Retrieval results file not found: {filepath}")
        return results
    logger.info(f"Loading retrieval results from {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"Loaded retrieval results for {len(results)} queries.")
    except Exception as e:
        logger.error(f"Error loading retrieval results from {filepath}: {e}")
    return results


# Setup logging
# setup_basic_logging() # Or configure directly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare augmented data for T5-FiD fine-tuning.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=['bm25', 'contriever'],
        default='bm25',
        help="Which retriever's results to use for augmentation ('bm25' or 'contriever')."
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

    # --- Extract Parameters ---
    try:
        # Which NQ split to use for getting queries/gold answers for the *training* data
        # Note: The notebook used 'nq_dev_path'. For proper training, use 'nq_train_path'.
        # Let's make it configurable, defaulting to train path.
        nq_input_path = config['data_paths'].get('nq_train_path_for_finetuning', config['data_paths']['nq_train_path'])
        logger.info(f"Using NQ data for training preparation from: {nq_input_path}")

        # Path to the pre-computed retrieval results for the NQ input split
        # This needs to exist in the config. Example:
        # bm25_retrieved_docs_train_path: "models/bm25/retrieved_docs_train.json"
        retrieval_results_path_key = f"{args.retriever}_retrieved_docs_train_path"
        if retrieval_results_path_key not in config['model_paths']:
             # Fallback to dev if train isn't specified (matching notebook behavior)
             retrieval_results_path_key = f"{args.retriever}_retrieved_docs_dev_path"
             if retrieval_results_path_key not in config['model_paths']:
                 logger.error(f"Retrieval results path ('{args.retriever}_retrieved_docs_train_path' or '{args.retriever}_retrieved_docs_dev_path') not found in config['model_paths']")
                 sys.exit(1)
             else:
                  logger.warning(f"Using DEV retrieval results for training data augmentation: {config['model_paths'][retrieval_results_path_key]}")
        retrieval_results_path = config['model_paths'][retrieval_results_path_key]


        augmented_output_path = config['data_paths']['augmented_data_path']
        num_docs_to_use = config['retrieval']['retrieval_k_for_training_data']

    except KeyError as e:
        logger.error(f"Missing key in configuration file: {e}")
        sys.exit(1)

    # --- Load Data ---
    logger.info("Loading NQ expected outputs...")
    expected_outputs = load_all_nq_expected_outputs(nq_input_path)
    if not expected_outputs:
        logger.error(f"Failed to load any data from {nq_input_path}. Aborting.")
        sys.exit(1)

    logger.info(f"Loading {args.retriever} retrieval results...")
    retrieval_results = load_retrieval_results(retrieval_results_path)
    if not retrieval_results:
        logger.error(f"Failed to load any retrieval results from {retrieval_results_path}. Aborting.")
        sys.exit(1)

    # --- Align Data (Optional but Recommended) ---
    # Ensure we only augment queries present in both sets
    common_queries = set(expected_outputs.keys()) & set(retrieval_results.keys())
    expected_outputs_aligned = {q: expected_outputs[q] for q in common_queries}
    # Retrieval results usually cover all queries if generated from the same input set
    logger.info(f"Found {len(common_queries)} queries common to NQ data and retrieval results.")
    if not common_queries:
        logger.error("No common queries found between NQ data and retrieval results. Cannot augment.")
        sys.exit(1)


    # --- Augment Data ---
    logger.info(f"Starting data augmentation using top {num_docs_to_use} documents...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(augmented_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Ensured output directory exists: {output_dir}")

        augment_with_retrieved_documents(
            nq_expected_outputs=expected_outputs_aligned,
            retrieval_results=retrieval_results, # Pass the full dict, function will select query
            output_file=augmented_output_path,
            num_retrieved_docs_to_use=num_docs_to_use
        )
        logger.info(f"Augmented data preparation script finished successfully. Output: {augmented_output_path}")

    except Exception as e:
        logger.exception(f"An error occurred during data augmentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
