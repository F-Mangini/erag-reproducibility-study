import json
import requests
import logging
import os
from tqdm.auto import tqdm # For progress visualization

logger = logging.getLogger(__name__)

def split_into_passages(text: str, max_words: int = 100) -> list[str]:
    """Splits text into passages of up to max_words words (without overlap)."""
    if not isinstance(text, str):
        logger.warning(f"Received non-string input in split_into_passages: {type(text)}. Returning empty list.")
        return []
    words = text.split()
    passages = []
    if not words: # Handle empty text
         return []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        passages.append(chunk)
    return passages


def process_kilt_page(page_json: dict, max_words: int = 100) -> list[str]:
    """
    Given a KILT knowledge source page (record as dict),
    segments the "text" field (list of paragraphs) into passages of max_words.
    Returns a list of documents combining the title and passage.
    Format: "Title [SEP] Passage"
    """
    title = page_json.get("wikipedia_title", "").strip()
    paragraphs = page_json.get("text", [])
    docs = []
    if not title:
        logger.debug("Page JSON missing 'wikipedia_title'.")
        # Decide how to handle: skip, use default title, etc.
        # Using empty title for now, results in "[SEP] Passage"
        # Consider adding title = "Unknown Title"

    if not isinstance(paragraphs, list):
        logger.warning(f"Expected 'text' field to be a list, got {type(paragraphs)}. Skipping page.")
        return []

    for para in paragraphs:
        if not isinstance(para, str):
            logger.debug(f"Skipping non-string paragraph: {type(para)}")
            continue
        para = para.strip()
        if not para:
            continue
        passages = split_into_passages(para, max_words=max_words)
        for p in passages:
            # Ensure passage is not empty after potential splitting issues
            if p:
                doc = f"{title} [SEP] {p}"
                docs.append(doc)
    return docs

def run_kilt_preprocessing(url: str, output_file: str, num_records_to_process: int = -1, passage_max_words: int = 100):
    """
    Streams the KILT knowledge source, processes records, and saves passages.

    Args:
        url: URL to the kilt_knowledgesource.json file.
        output_file: Path to save the output JSONL file.
        num_records_to_process: Max number of records to process (-1 for all).
        passage_max_words: Maximum words per passage segment.
    """
    logger.info(f"Starting KILT preprocessing from {url}")
    logger.info(f"Saving processed passages to {output_file}")
    if num_records_to_process > 0:
        logger.info(f"Processing a maximum of {num_records_to_process} records.")
    else:
         logger.info(f"Processing all available records.")
    logger.info(f"Max words per passage: {passage_max_words}")

    count = 0
    processed_count = 0
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with requests.get(url, stream=True) as r, open(output_file, 'w', encoding='utf-8') as fout:
            r.raise_for_status() # Check for download errors
            # Use tqdm for progress, assuming total is unknown or very large
            # Total size can be fetched from headers if needed for a more accurate bar
            # total_size = int(r.headers.get('content-length', 0))
            # line_iterator = tqdm(r.iter_lines(), desc="Processing KILT Lines", total=?) # Hard to estimate lines

            for line in tqdm(r.iter_lines(), desc="Processing KILT Lines"):
                if num_records_to_process > 0 and count >= num_records_to_process:
                    logger.info(f"Reached record limit ({num_records_to_process}). Stopping.")
                    break
                count += 1
                if line: # Filter out keep-alive new lines
                    try:
                        page = json.loads(line)
                        docs = process_kilt_page(page, max_words=passage_max_words)
                        for doc in docs:
                            # Save each document as JSON Lines
                            json.dump({"document": doc}, fout)
                            fout.write("\n")
                        processed_count += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON on record {count}: {e}")
                        continue
                    except Exception as e:
                         logger.error(f"Error processing record {count}: {e}")
                         continue

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading KILT source from {url}: {e}")
        raise # Re-raise the exception after logging
    except IOError as e:
        logger.error(f"Error writing to output file {output_file}: {e}")
        raise # Re-raise the exception

    logger.info(f"Finished KILT preprocessing. Processed {processed_count} records out of {count} lines read.")
    logger.info(f"Output saved in '{output_file}'.")

# Example of how to run this module directly (optional)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Example usage if run as a script
    DEFAULT_URL = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"
    DEFAULT_OUTPUT = "data/wikipedia_passages_sample.jsonl"
    DEFAULT_NUM_RECORDS = 5903 # From notebook example

    # Could use argparse here if making it a proper script
    run_kilt_preprocessing(DEFAULT_URL, DEFAULT_OUTPUT, DEFAULT_NUM_RECORDS)

    # Verify output
    if os.path.exists(DEFAULT_OUTPUT):
         with open(DEFAULT_OUTPUT, 'r') as f:
              line_count = sum(1 for _ in f)
         print(f"Verification: Output file '{DEFAULT_OUTPUT}' created with {line_count} lines.")
         # Print first few lines
         with open(DEFAULT_OUTPUT, 'r') as f:
              print("First 5 lines:")
              for i, line in enumerate(f):
                   if i >= 5: break
                   print(f"  {line.strip()}")
    else:
         print(f"Error: Output file '{DEFAULT_OUTPUT}' not found after processing.")
