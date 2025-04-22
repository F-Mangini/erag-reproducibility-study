import json
import os
import pickle
import time
import logging
import yaml
import torch
import scipy.stats as stats
import erag # Assumes the erag package is installed
from typing import Dict, List, Any, Callable, Set
from functools import partial
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel

# Import functions from other modules within the src package
from .metrics import exact_match_metric
from ..generation.t5_fid_generator import T5_text_generator
from ..data_processing.dataset_utils import load_all_nq_expected_outputs
# If retrieval is done here, import retriever functions
# from ..retrieval.bm25_retriever import load_bm25_index, bm25_retrieve
# from ..retrieval.contriever_retriever import load_contriever_model, load_faiss_index, dense_retrieve


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_retrieval_results(filepath: str) -> Dict[str, List[str]]:
    """Loads pre-computed retrieval results from a file."""
    # Assuming results are stored as a JSON mapping query -> list of docs
    logger.info(f"Loading retrieval results from {filepath}")
    if not os.path.exists(filepath):
        logger.error(f"Retrieval results file not found: {filepath}")
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"Loaded retrieval results for {len(results)} queries.")
        return results
    except Exception as e:
        logger.error(f"Error loading retrieval results from {filepath}: {e}")
        return {}

def run_full_evaluation(config_path: str):
    """
    Orchestrates the eRAG and end-to-end evaluation pipeline based on config.

    Args:
        config_path: Path to the YAML configuration file.
    """
    # --- Load Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data_paths']
    model_cfg = config['model_paths']
    eval_cfg = config['evaluation']
    hardware_cfg = config['hardware']
    gen_cfg = config['t5_fid_training'] # Get generation params like max_len

    # --- Device Setup ---
    if hardware_cfg['device']:
        device = torch.device(hardware_cfg['device'])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Fine-tuned Model & Tokenizer ---
    logger.info(f"Loading fine-tuned T5 model from: {model_cfg['t5_finetuned_dir']}")
    if not os.path.exists(model_cfg['t5_finetuned_dir']):
         logger.error(f"Fine-tuned model directory not found: {model_cfg['t5_finetuned_dir']}")
         return
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_cfg['t5_finetuned_dir'])
        model = T5ForConditionalGeneration.from_pretrained(model_cfg['t5_finetuned_dir'])
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        return

    # --- Load Test Data ---
    logger.info("Loading test data...")
    # Load expected outputs (gold answers) for the test set
    test_expected_outputs = load_all_nq_expected_outputs(data_cfg['nq_dev_path']) # Using dev as test for now
    if not test_expected_outputs:
         logger.error("Failed to load expected outputs for testing. Aborting.")
         return

    # Load pre-computed retrieval results for the test set queries
    # We need separate files for BM25 and Contriever if evaluating both
    # For now, let's assume BM25 results are needed, path defined in config.
    # Need to add a path for retrieval results in params.yaml
    # Example: Add `bm25_retrieved_docs_test_path: "models/bm25/retrieved_docs_dev.json"`
    retrieval_results_path = "models/bm25/retrieved_docs_dev.json" # Get this from config eventually
    test_retrieval_results = load_retrieval_results(retrieval_results_path)
    if not test_retrieval_results:
         logger.error(f"Failed to load retrieval results from {retrieval_results_path}. Aborting.")
         return

    # Align data: Only keep queries present in both expected and retrieval results
    common_queries = set(test_expected_outputs.keys()) & set(test_retrieval_results.keys())
    test_expected_outputs = {q: test_expected_outputs[q] for q in common_queries}
    test_retrieval_results = {q: test_retrieval_results[q] for q in common_queries}
    test_queries = list(common_queries) # Use a fixed order for correlation
    logger.info(f"Aligned test data: {len(test_queries)} queries common to expected outputs and retrieval results.")
    if not test_queries:
         logger.error("No common queries between expected outputs and retrieval results. Aborting.")
         return

    # --- Load Checkpoint ---
    checkpoint_file = eval_cfg['checkpoint_file']
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True) # Ensure directory exists
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_file}.")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Starting fresh.")
            checkpoint = {}
    else:
        checkpoint = {}
        logger.info("No checkpoint found, starting evaluation from scratch.")

    # --- Evaluation Loop ---
    k_values = config['retrieval']['retrieval_k_values_for_evaluation']
    # For now, only implement BM25 as in the preliminary run
    retriever_methods = ['BM25'] # Add 'Contriever' later
    os.makedirs(config['data_paths']['logs_dir'], exist_ok=True) # Ensure logs dir exists

    correlations = checkpoint # Use checkpoint directly

    for method in retriever_methods:
        logger.info(f"\n===== Starting Evaluation for Retriever: {method} =====")
        if method not in correlations:
            correlations[method] = {}

        # Load the appropriate retrieval results if evaluating different methods
        # current_retrieval_results = load_retrieval_results(config['paths'][f'{method}_retrieval_path'])
        current_retrieval_results = test_retrieval_results # Using BM25 results for now

        for k in k_values:
            if k in correlations.get(method, {}):
                logger.info(f"Skipping {method} with K={k} (results found in checkpoint).")
                continue

            logger.info(f"\n--- Processing {method} with K = {k} ---")
            start_time_k = time.time()

            # Define retrieval metrics based on k
            retrieval_metrics_set: Set[str] = set()
            for template in eval_cfg['erag_metrics_templates']:
                retrieval_metrics_set.add(template.format(k=k))
            logger.info(f"Using eRAG aggregation metrics: {retrieval_metrics_set}")

            # Create the partial generator function specific to this k for end-to-end
            t5_generator_partial_k = partial(
                T5_text_generator,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_input_len=gen_cfg['max_input_length'],
                max_output_len=gen_cfg['max_target_length'],
                num_beams=eval_cfg['t5_generation_num_beams'],
                max_docs_per_query=k # Use k docs for generation
            )

            try:
                # 1. Evaluate using erag.eval
                logger.info(f"Running erag.eval for K={k}...")
                # Note: erag.eval itself might call the text_generator internally
                # The generator passed here should ideally *not* limit docs to k,
                # as erag handles iteration. Let's create one without max_docs limit for erag.
                t5_generator_for_erag = partial(
                     T5_text_generator,
                     model=model,
                     tokenizer=tokenizer,
                     device=device,
                     max_input_len=gen_cfg['max_input_length'],
                     max_output_len=gen_cfg['max_target_length'],
                     num_beams=eval_cfg['t5_generation_num_beams']
                     # No max_docs_per_query limit here, erag handles iteration
                )
                erag_results = erag.eval(
                    retrieval_results=current_retrieval_results, # Dict query -> list of docs
                    expected_outputs=test_expected_outputs,     # Dict query -> list of answers
                    text_generator=t5_generator_for_erag,       # Function (query_dict) -> answer_dict
                    downstream_metric=exact_match_metric,       # Function (gen_dict, exp_dict) -> score_dict
                    retrieval_metrics=retrieval_metrics_set,    # Set of metric names like {'P_10', 'success_10'}
                    retrieval_k=k                               # Explicitly pass k here
                )
                logger.info(f"erag.eval completed.")

                # Save per-query results
                per_input_file = os.path.join(config['data_paths']['logs_dir'], f"per_input_{method}_K{k}.json")
                with open(per_input_file, "w", encoding="utf-8") as f:
                    json.dump(erag_results['per_input'], f, ensure_ascii=False, indent=2)
                logger.info(f"Saved per-input eRAG results to {per_input_file}.")

                # Save aggregated results
                aggregated_file = os.path.join(config['data_paths']['logs_dir'], f"aggregated_{method}_K{k}.json")
                with open(aggregated_file, "w", encoding="utf-8") as f:
                    json.dump(erag_results['aggregated'], f, ensure_ascii=False, indent=2)
                logger.info(f"Saved aggregated eRAG results to {aggregated_file}.")

                # 2. Generate end-to-end responses using top k documents
                logger.info(f"Running end-to-end generation for K={k}...")
                # Prepare input dict with only top k docs per query
                e2e_input_dict = {q: docs[:k] for q, docs in current_retrieval_results.items() if q in test_queries}
                end_to_end_generated = t5_generator_partial_k(e2e_input_dict) # Use the k-limited generator
                logger.info(f"End-to-end generation complete.")

                # Calculate end-to-end scores
                logger.info(f"Calculating end-to-end Exact Match scores...")
                e2e_scores_dict = exact_match_metric(end_to_end_generated, test_expected_outputs)

                # Save end-to-end scores
                e2e_file = os.path.join(config['data_paths']['logs_dir'], f"end_to_end_{method}_K{k}.json")
                with open(e2e_file, "w", encoding="utf-8") as f:
                    json.dump(e2e_scores_dict, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved end-to-end scores to {e2e_file}.")

                # 3. Compute correlations
                logger.info(f"Calculating correlations between eRAG metrics and end-to-end EM for K={k}...")
                # Ensure scores are in the same order (using test_queries list)
                end_to_end_scores_list = [e2e_scores_dict.get(query, 0) for query in test_queries]

                local_corr_k = {}
                for metric_key in retrieval_metrics_set:
                    # Extract eRAG scores for the current metric, handling potential missing keys
                    eRAG_scores_list = [
                        erag_results['per_input'].get(query, {}).get(metric_key)
                        for query in test_queries
                    ]
                    # Filter out None values if a query somehow didn't get a score
                    valid_indices = [i for i, score in enumerate(eRAG_scores_list) if score is not None]
                    if len(valid_indices) < len(test_queries):
                         logger.warning(f"Missing eRAG scores for metric {metric_key} for {len(test_queries) - len(valid_indices)} queries. Calculating correlation on {len(valid_indices)} pairs.")

                    # Align scores based on valid indices
                    aligned_erag_scores = [eRAG_scores_list[i] for i in valid_indices]
                    aligned_e2e_scores = [end_to_end_scores_list[i] for i in valid_indices]

                    if len(aligned_erag_scores) < 2:
                        logger.warning(f"Not enough valid score pairs ({len(aligned_erag_scores)}) to calculate correlation for metric {metric_key}. Skipping.")
                        continue

                    # Compute correlations only if there's variance in both lists
                    if np.std(aligned_erag_scores) > 1e-6 and np.std(aligned_e2e_scores) > 1e-6:
                        spearman_corr, spearman_p = stats.spearmanr(aligned_erag_scores, aligned_e2e_scores)
                        kendall_corr, kendall_p = stats.kendalltau(aligned_erag_scores, aligned_e2e_scores)

                        local_corr_k[metric_key] = {
                            'spearman_rho': spearman_corr,
                            'spearman_p': spearman_p,
                            'kendall_tau': kendall_corr,
                            'kendall_p': kendall_p,
                            'num_queries': len(aligned_erag_scores)
                        }
                        logger.info(f"  Metric {metric_key}:")
                        logger.info(f"    Spearman Rho: {spearman_corr:.4f} (p={spearman_p:.3g})")
                        logger.info(f"    Kendall Tau:  {kendall_corr:.4f} (p={kendall_p:.3g})")
                    else:
                        logger.warning(f"Skipping correlation for {metric_key} due to zero variance in scores.")
                        local_corr_k[metric_key] = {
                            'spearman_rho': None, 'spearman_p': None,
                            'kendall_tau': None, 'kendall_p': None,
                            'num_queries': len(aligned_erag_scores),
                            'notes': 'Zero variance detected in scores'
                        }


                # Update checkpoint for this k
                correlations[method][k] = local_corr_k
                logger.info(f"Updating checkpoint for {method}, K={k}...")
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(correlations, f)
                logger.info(f"Checkpoint saved.")

            except Exception as e:
                logger.exception(f"Error during evaluation loop for {method}, K={k}: {e}") # Log stack trace
                # Optionally add a sleep or continue logic
                continue # Continue to next k value or method

            end_time_k = time.time()
            logger.info(f"--- Finished {method} with K = {k} in {end_time_k - start_time_k:.2f} seconds ---")


    logger.info("\n===== Evaluation Complete =====")
    logger.info("Final Correlation Summary (from checkpoint):")
    # Pretty print the final results
    print(json.dumps(correlations, indent=2))


# --- Main Execution Block ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run eRAG evaluation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    run_full_evaluation(args.config)
