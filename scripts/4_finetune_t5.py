# scripts/4_finetune_t5.py

import argparse
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
from training.train_t5_fid import train_model
from utils.logging_utils import setup_basic_logging # Optional

# Setup logging
# setup_basic_logging() # Or configure directly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run T5-FiD fine-tuning using a configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    logger.info(f"Starting T5-FiD fine-tuning script with config: {args.config}")
    start_time = time.time()

    try:
        # Call the main training function from the src module
        train_model(config_path=args.config)
        end_time = time.time()
        logger.info(f"T5-FiD fine-tuning script finished successfully in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
         logger.error(f"Configuration or data file not found: {e}. Please check paths in {args.config}.")
         sys.exit(1)
    except ValueError as e:
         logger.error(f"Configuration error: {e}")
         sys.exit(1)
    except ImportError as e:
         logger.error(f"Import error: {e}. Make sure all dependencies in requirements.txt are installed.")
         sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during fine-tuning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
