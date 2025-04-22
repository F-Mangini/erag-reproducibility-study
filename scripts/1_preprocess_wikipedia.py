# scripts/1_preprocess_wikipedia.py

import argparse
import yaml
import logging
import os
import sys

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import from src
from data_processing.kilt_preprocessing import run_kilt_preprocessing
from utils.logging_utils import setup_basic_logging # Optional: use the setup function

# Setup logging
# setup_basic_logging() # Or configure directly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Preprocess KILT Wikipedia source based on configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to the YAML configuration file."
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

    # Extract parameters
    try:
        url = config['data_paths']['kilt_url']
        output_file = config['data_paths']['wiki_processed_subset_path']
        num_records = config['wiki_preprocessing']['num_records_to_process']
        max_words = config['wiki_preprocessing']['passage_max_words']
    except KeyError as e:
        logger.error(f"Missing key in configuration file: {e}")
        sys.exit(1)

    # --- Run Preprocessing ---
    logger.info("Starting Wikipedia preprocessing script...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir: # Only create if path includes a directory
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Ensured output directory exists: {output_dir}")

        run_kilt_preprocessing(
            url=url,
            output_file=output_file,
            num_records_to_process=num_records,
            passage_max_words=max_words
        )
        logger.info("Wikipedia preprocessing script finished successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during Wikipedia preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
