import string
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def normalize_answer(s: str) -> str:
    """
    Normalizes a string by lowercasing, removing punctuation, articles ('a', 'an', 'the'),
    and extra whitespace. Matches common QA evaluation practices.

    Args:
        s: The input string.

    Returns:
        The normalized string.
    """
    if not isinstance(s, str):
        logger.warning(f"Received non-string input in normalize_answer: {type(s)}. Returning empty string.")
        return ""

    def remove_articles(text):
        return ' '.join([word for word in text.split() if word not in ['a', 'an', 'the']])

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_metric(generated_outputs: Dict[str, str],
                        expected_outputs: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Computes Exact Match (EM) scores between generated and expected outputs.

    Compares the normalized generated answer against all normalized gold answers
    for each query.

    Args:
        generated_outputs: A dictionary mapping query strings to the generated answer string.
        expected_outputs: A dictionary mapping query strings to a list of possible
                          gold standard answer strings.

    Returns:
        A dictionary mapping query strings to their EM score (1 if any gold answer
        matches, 0 otherwise).
    """
    em_scores: Dict[str, int] = {}
    logger.info(f"Calculating Exact Match for {len(generated_outputs)} generated outputs.")
    matched_count = 0

    for query, generated_answer in generated_outputs.items():
        if query not in expected_outputs:
            logger.warning(f"Query '{query[:50]}...' found in generated outputs but not in expected outputs. Assigning EM score 0.")
            em_scores[query] = 0
            continue

        if not generated_answer: # Handle empty generated answer
             em_scores[query] = 0
             continue

        normalized_gen_ans = normalize_answer(generated_answer)
        gold_answers = expected_outputs[query]

        match = 0
        if not gold_answers: # Handle case where expected outputs exist but list is empty
             logger.warning(f"No gold answers provided for query '{query[:50]}...' in expected outputs. Assigning EM score 0.")
             em_scores[query] = 0
             continue

        for gold_ans in gold_answers:
             normalized_gold = normalize_answer(gold_ans)
             if normalized_gold == normalized_gen_ans:
                 match = 1
                 matched_count +=1
                 break # Found a match, no need to check further golds for this query

        em_scores[query] = match

    total_queries = len(generated_outputs)
    overall_em = (matched_count / total_queries) * 100 if total_queries > 0 else 0
    logger.info(f"Exact Match calculation complete. Overall EM: {overall_em:.2f}% ({matched_count}/{total_queries})")
    return em_scores


# Example usage (optional)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    gen = {
        "query1": "  The   Eiffel Tower is in Paris! ",
        "query2": "shakespeare wrote hamlet",
        "query3": "answer C",
        "query4": "missing gold",
        "query5": "empty gen"
    }
    exp = {
        "query1": ["Paris", "eiffel tower located in paris"],
        "query2": ["william shakespeare"],
        "query3": ["answer c", "C"],
        "query5": ["some answer"]
        # query4 is missing
    }

    scores = exact_match_metric(gen, exp)
    print("\nExact Match Scores:")
    print(json.dumps(scores, indent=2))
    # Expected: query1: 1, query2: 0, query3: 1, query4: 0, query5: 0
