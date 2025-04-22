# scripts/5_run_evaluation.py

import argparse
import yaml
import logging
import os
import sys
import time

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import from src
from evaluation.evaluate_pipeline import run_full_evaluation
from utils.logging_utils import setup_basic_logging # Optional

# Setup logging
# setup_basic_logging() # Or configure directly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the full eRAG evaluation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to the YAML configuration file."
    )
    # Optional: Add argument to specify which retriever results to use if not hardcoded in evaluate_pipeline
    # parser.add_argument(
    #     "--retriever",
    #     type=str,
    #     choices=['bm25', 'contriever'],
    #     required=True, # Make it required if evaluate_pipeline needs it explicitly
    #     help="Which retriever's results to evaluate ('bm25' or 'contriever')."
    # )
    args = parser.parse_args()

    logger.info(f"Starting full evaluation script with config: {args.config}")
    start_time = time.time()

    try:
        # Check if essential paths exist from config before running
        # (evaluate_pipeline also does checks, but early check is good)
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model_dir = config.get('model_paths', {}).get('t5_finetuned_dir')
        # Add check for retrieval results path used in evaluate_pipeline
        # retrieval_path = config.get('model_paths', {}).get(f"{args.retriever}_retrieved_docs_test_path", None) # Example

        if not model_dir or not os.path.exists(model_dir):
             logger.error(f"Fine-tuned model directory specified in config ('{model_dir}') not found.")
             sys.exit(1)
        # Add similar check for retrieval results path if needed

        # Call the main evaluation function from the src module
        run_full_evaluation(config_path=args.config)

        end_time = time.time()
        logger.info(f"Evaluation script finished successfully in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
         logger.error(f"Configuration or data/model file not found: {e}. Please check paths in {args.config}.")
         sys.exit(1)
    except KeyError as e:
         logger.error(f"Missing required key in configuration file {args.config}: {e}")
         sys.exit(1)
    except ValueError as e:
         logger.error(f"Configuration or data error: {e}")
         sys.exit(1)
    except ImportError as e:
         logger.error(f"Import error: {e}. Make sure all dependencies are installed and src path is correct.")
         sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
