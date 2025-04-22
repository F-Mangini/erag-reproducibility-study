# src/utils/logging_utils.py

import json
import pickle
import os
import logging
from typing import Any, Dict, List, Union

# Setup a logger specific to this utility module
logger = logging.getLogger(__name__)

# --- JSON Handling ---

def save_to_json(data: Union[Dict, List], filepath: str, indent: int = 2, ensure_ascii: bool = False):
    """
    Saves data (dictionary or list) to a JSON file.

    Args:
        data: The data to save.
        filepath: The path to the output JSON file.
        indent: Indentation level for pretty printing. Default is 2.
        ensure_ascii: If False, allows non-ASCII characters. Default is False.
    """
    try:
        # Ensure the directory exists
        dirpath = os.path.dirname(filepath)
        if dirpath: # Check if dirname is not empty (e.g., for relative paths in root)
            os.makedirs(dirpath, exist_ok=True)

        logger.debug(f"Saving data to JSON file: {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        logger.debug(f"Successfully saved data to {filepath}")
        return True
    except IOError as e:
        logger.error(f"IOError saving JSON to {filepath}: {e}")
        return False
    except TypeError as e:
        logger.error(f"TypeError saving JSON (data might not be serializable): {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving JSON to {filepath}: {e}")
        return False

def load_from_json(filepath: str) -> Any:
    """
    Loads data from a JSON file.

    Args:
        filepath: The path to the input JSON file.

    Returns:
        The loaded data (typically dict or list), or None if loading fails.
    """
    if not os.path.exists(filepath):
        logger.error(f"JSON file not found: {filepath}")
        return None
    try:
        logger.debug(f"Loading data from JSON file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded data from {filepath}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return None
    except IOError as e:
        logger.error(f"IOError reading JSON from {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading JSON from {filepath}: {e}")
        return None

# --- Pickle Handling (for Checkpoints etc.) ---

def save_checkpoint(data: Any, filepath: str):
    """
    Saves arbitrary Python object(s) using pickle (often used for checkpoints).

    Args:
        data: The Python object to save.
        filepath: The path to the output pickle file.
    """
    try:
        # Ensure the directory exists
        dirpath = os.path.dirname(filepath)
        if dirpath:
             os.makedirs(dirpath, exist_ok=True)

        logger.debug(f"Saving checkpoint data to pickle file: {filepath}")
        with open(filepath, 'wb') as f: # Use 'wb' for binary writing
            pickle.dump(data, f)
        logger.debug(f"Successfully saved checkpoint to {filepath}")
        return True
    except IOError as e:
        logger.error(f"IOError saving pickle to {filepath}: {e}")
        return False
    except pickle.PicklingError as e:
        logger.error(f"PicklingError saving data: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving pickle to {filepath}: {e}")
        return False

def load_checkpoint(filepath: str) -> Any:
    """
    Loads Python object(s) from a pickle file.

    Args:
        filepath: The path to the input pickle file.

    Returns:
        The loaded Python object, or None if loading fails or file not found.
    """
    if not os.path.exists(filepath):
        logger.info(f"Checkpoint file not found: {filepath}. Returning None.")
        return None
    try:
        logger.debug(f"Loading checkpoint data from pickle file: {filepath}")
        with open(filepath, 'rb') as f: # Use 'rb' for binary reading
            data = pickle.load(f)
        logger.debug(f"Successfully loaded checkpoint from {filepath}")
        return data
    except IOError as e:
        logger.error(f"IOError reading pickle from {filepath}: {e}")
        return None
    except pickle.UnpicklingError as e:
        logger.error(f"UnpicklingError loading data (file might be corrupt or incompatible): {e}")
        return None
    except EOFError as e:
        logger.error(f"EOFError reading pickle (file might be empty or truncated): {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading pickle from {filepath}: {e}")
        return None


# --- Basic Logging Configuration (Optional Here) ---
# It's often better to configure logging in the main script entry points
# using basicConfig, but you could add a helper here if needed.

def setup_basic_logging(level=logging.INFO):
    """Sets up basic console logging."""
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Basic logging configured at level {logging.getLevelName(level)}")


# Example usage (optional)
if __name__ == "__main__":
    # Configure logging for the example
    setup_basic_logging(level=logging.DEBUG)

    logger.info("--- Testing Logging Utilities ---")

    # Test JSON
    json_file = "temp_test.json"
    json_data = {"key1": "value1", "list": [1, 2, {"nested": True}], "unicode": "éàçü"}
    logger.info(f"Attempting to save JSON to {json_file}")
    save_success = save_to_json(json_data, json_file)
    if save_success:
        logger.info(f"Attempting to load JSON from {json_file}")
        loaded_json = load_from_json(json_file)
        if loaded_json:
            logger.info(f"Loaded JSON data: {loaded_json}")
            assert json_data == loaded_json # Basic check
        if os.path.exists(json_file):
             os.remove(json_file)

    # Test Pickle
    pickle_file = "temp_test.pkl"
    pickle_data = {"config": {"lr": 0.01}, "epoch": 5, "scores": [0.5, 0.6]}
    logger.info(f"\nAttempting to save Pickle to {pickle_file}")
    save_success_pkl = save_checkpoint(pickle_data, pickle_file)
    if save_success_pkl:
        logger.info(f"Attempting to load Pickle from {pickle_file}")
        loaded_pickle = load_checkpoint(pickle_file)
        if loaded_pickle:
             logger.info(f"Loaded Pickle data: {loaded_pickle}")
             assert pickle_data == loaded_pickle # Basic check
        if os.path.exists(pickle_file):
            os.remove(pickle_file)

    logger.info("--- Logging Utilities Test Complete ---")
