import json
import logging
import os
from typing import Dict, List, Any # For type hinting

logger = logging.getLogger(__name__)

def load_all_nq_expected_outputs(filename: str) -> Dict[str, List[str]]:
    """
    Loads all queries and their gold answers from a KILT NQ file (JSONL format).

    Args:
        filename: Path to the KILT NQ JSONL file (e.g., nq-dev-kilt.jsonl).

    Returns:
        A dictionary mapping query strings to a list of gold answer strings.
        Returns an empty dict if the file cannot be read or processed.
    """
    expected_outputs: Dict[str, List[str]] = {}
    if not os.path.exists(filename):
         logger.error(f"Input file not found: {filename}")
         return expected_outputs

    logger.info(f"Loading NQ expected outputs from: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    query = record.get("input", "").strip()
                    # Ensure 'output' exists and is a list
                    output_list = record.get("output", [])
                    if not isinstance(output_list, list):
                        logger.warning(f"Invalid 'output' format in line {i+1}. Expected list, got {type(output_list)}. Skipping record.")
                        continue

                    # Extract answers, ensuring 'answer' key exists
                    golds = [
                        out["answer"].strip()
                        for out in output_list
                        if isinstance(out, dict) and "answer" in out and isinstance(out["answer"], str)
                    ]

                    if query and golds: # Only add if query and answers are valid
                        # Use set to handle potential duplicate answers within a record easily
                        unique_golds = sorted(list(set(golds)))
                        expected_outputs[query] = unique_golds
                    elif query and not golds:
                         logger.debug(f"Query found but no valid answers in line {i+1}: '{query[:50]}...'")
                    elif not query:
                         logger.warning(f"Missing 'input' (query) in line {i+1}. Skipping record.")

                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON on line {i+1} in {filename}")
                    continue
                except KeyError as e:
                     logger.error(f"Missing expected key {e} on line {i+1} in {filename}")
                     continue
                except Exception as e:
                     logger.error(f"Unexpected error processing line {i+1} in {filename}: {e}")
                     continue
    except IOError as e:
        logger.error(f"Could not read file {filename}: {e}")
        return {} # Return empty on file read error

    logger.info(f"Loaded {len(expected_outputs)} queries with expected outputs from {filename}.")
    return expected_outputs


def augment_with_retrieved_documents(
    nq_expected_outputs: Dict[str, List[str]],
    retrieval_results: Dict[str, List[str]],
    output_file: str,
    num_retrieved_docs_to_use: int = 50 # Specify how many retrieved docs to include per query
    ) -> List[Dict[str, Any]]:
    """
    Augments NQ data with retrieved documents for fine-tuning.
    Takes the *first* gold answer as the target for simplicity (as in notebook).

    Args:
        nq_expected_outputs: Dict mapping query -> list of gold answers.
        retrieval_results: Dict mapping query -> list of retrieved document strings.
        output_file: Path to save the augmented dataset (JSON format).
        num_retrieved_docs_to_use: The number of top retrieved documents to include
                                  in the augmented data for each query.

    Returns:
        A list of dictionaries, each representing an augmented sample:
        {'query': str, 'retrieved_docs': List[str], 'gold_answer': str}
    """
    logger.info(f"Augmenting {len(nq_expected_outputs)} NQ samples with retrieval results.")
    logger.info(f"Using top {num_retrieved_docs_to_use} retrieved documents per query.")
    augmented_data: List[Dict[str, Any]] = []
    queries_missing_retrieval = 0

    for query, gold_answers in nq_expected_outputs.items():
        # Retrieve corresponding documents for the query
        retrieved_docs = retrieval_results.get(query)

        if retrieved_docs is None:
            logger.warning(f"No retrieval results found for query: '{query[:50]}...'. Skipping.")
            queries_missing_retrieval += 1
            continue

        if not gold_answers:
            logger.warning(f"No gold answers found for query: '{query[:50]}...'. Skipping.")
            continue

        # Use only the specified number of top documents
        docs_to_include = retrieved_docs[:num_retrieved_docs_to_use]

        # Target answer (take the first gold answer as per notebook example)
        target_text = gold_answers[0]

        augmented_data.append({
            "query": query,
            "retrieved_docs": docs_to_include,
            "gold_answer": target_text
        })

    if queries_missing_retrieval > 0:
        logger.warning(f"{queries_missing_retrieval} queries were skipped due to missing retrieval results.")

    logger.info(f"Created augmented dataset with {len(augmented_data)} samples.")

    # Save the augmented dataset to a new file
    try:
         # Ensure directory exists
         os.makedirs(os.path.dirname(output_file), exist_ok=True)
         logger.info(f"Saving augmented dataset to: {output_file}")
         with open(output_file, 'w', encoding='utf-8') as f:
             # Use JSON Lines format for potentially large files? Or just JSON?
             # Notebook used json.dump (single JSON object), let's stick to that for now.
             json.dump(augmented_data, f, indent=4) # indent for readability
         logger.info("Augmented dataset saved successfully.")
    except IOError as e:
         logger.error(f"Could not write augmented data to {output_file}: {e}")
    except Exception as e:
         logger.error(f"Unexpected error saving augmented data: {e}")


    return augmented_data


# Example of how to run this module directly (optional)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Example for load_all_nq_expected_outputs ---
    # Create a dummy NQ file for testing
    dummy_nq_file = "dummy_nq_dev.jsonl"
    dummy_data = [
        {"id": "1", "input": "query one", "output": [{"answer": "answer 1a"}, {"answer": "answer 1b"}]},
        {"id": "2", "input": "query two", "output": [{"answer": "answer 2a"}]},
        {"id": "3", "input": "query three"} # No output
    ]
    with open(dummy_nq_file, "w") as f:
        for item in dummy_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n--- Testing load_all_nq_expected_outputs ---")
    expected = load_all_nq_expected_outputs(dummy_nq_file)
    print("Loaded expected outputs:")
    print(json.dumps(expected, indent=2))
    os.remove(dummy_nq_file) # Clean up dummy file

    # --- Example for augment_with_retrieved_documents ---
    print(f"\n--- Testing augment_with_retrieved_documents ---")
    dummy_expected = {
        "query one": ["answer 1a", "answer 1b"],
        "query two": ["answer 2a"]
    }
    dummy_retrieval = {
        "query one": ["doc1 for q1", "doc2 for q1", "doc3 for q1"],
        "query two": ["doc1 for q2"]
        # query three is missing retrieval results
    }
    dummy_output_augmented = "dummy_augmented_data.json"
    augmented = augment_with_retrieved_documents(
        dummy_expected,
        dummy_retrieval,
        dummy_output_augmented,
        num_retrieved_docs_to_use=2 # Use only top 2 docs
        )

    print("\nGenerated augmented data:")
    print(json.dumps(augmented, indent=2))

    if os.path.exists(dummy_output_augmented):
         print(f"\nAugmented data saved to {dummy_output_augmented}")
         os.remove(dummy_output_augmented) # Clean up
    else:
         print(f"\nError: Augmented data file {dummy_output_augmented} not created.")
