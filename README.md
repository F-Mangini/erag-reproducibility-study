# Reproducibility Study: Evaluating Retrieval Quality in Retrieval-Augmented Generation (eRAG)

[![Status](https://img.shields.io/badge/status-Work%20In%20Progress-yellow.svg)](https://github.com/F-Mangini/erag-reproducibility-study)

## Introduction

This repository contains a work-in-progress attempt to reproduce the key findings from the paper **"Evaluating Retrieval Quality in Retrieval-Augmented Generation"** by Alireza Salemi and Hamed Zamani. The eRAG paper proposes a novel method for evaluating the retriever component in Retrieval-Augmented Generation (RAG) systems by using the downstream generator LLM itself to assess the utility of individual retrieved documents.

The primary goal of this project is to verify the paper's central claim: that eRAG evaluation scores exhibit a higher correlation with the end-to-end performance of the RAG system compared to traditional retrieval metrics or baseline methods.

## Original Paper

**Title:** Evaluating Retrieval Quality in Retrieval-Augmented Generation
**Authors:** Alireza Salemi, Hamed Zamani
**arXiv:** [https://arxiv.org/abs/2404.13781](https://arxiv.org/abs/2404.13781) (Link to the 2024 paper)

## Project Status: Work In Progress

⚠️ **Important:** This reproducibility study is currently incomplete and faces significant limitations primarily due to computational resource constraints. The results presented are **preliminary** and were obtained under conditions that differ substantially from the original paper:

*   **Hardware:** Experiments run on Google Colab (Free Tier) with an Nvidia T4 GPU (~15GB VRAM), compared to the Nvidia A100 used in the original study.
*   **Fine-tuning:** The T5-Small-FiD generator was fine-tuned for only **1 epoch**, compared to the 10 epochs used originally. This significantly impacts generator performance.
*   **Corpus Subsetting:** Due to memory and time constraints, only a small subset (~1/1000, the first 5903 records) of the KILT Wikipedia knowledge source was used for indexing (BM25) and retrieval.
*   **Data Split:** Preliminary experiments used a split of the NQ *development* set for both training and testing for efficiency, deviating from the standard practice of using the official KILT *training* set.

These limitations, especially the insufficient fine-tuning and limited corpus, drastically affect both the absolute performance of the reproduced RAG system and likely the magnitude of the observed correlations.

## Key Components Implemented

*   **Data Acquisition & Preprocessing:** Scripts to download the NQ dataset (KILT format) and preprocess the (subsetted) KILT Wikipedia corpus into passages.
*   **Retrieval:**
    *   BM25 indexing and retrieval using `rank_bm25` and `nltk`.
    *   Contriever encoding and Faiss (CPU) indexing (`IndexFlatIP`). *Note: Contriever evaluation runs were deferred in the preliminary experiments.*
*   **Generation:** T5-Small model with Fusion-in-Decoder (FiD) architecture implemented for fine-tuning and inference using `transformers`.
*   **Evaluation:**
    *   Integration with the official `erag` package's `eval` function.
    *   Custom Exact Match (EM) implementation for downstream task evaluation.
    *   End-to-end RAG pipeline execution.
    *   Correlation analysis (Spearman's Rho, Kendall's Tau) using `scipy`.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/erag-reproducibility-study.git
    cd erag-reproducibility-study
    ```
    *(Replace `your-username`)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    The first time you run the BM25 indexing code (Cell 9 in the notebook), it should prompt NLTK to download necessary data (`punkt`, `stopwords`). If not, you can run this in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

5.  **API Keys:**
    *   The code uses the Google Generative AI API (Gemini) in one cell (Cell 16) for a baseline RAG example. You'll need a Google API key. The key is currently hardcoded in the notebook (intended for temporary Colab use) - **it's recommended to replace this with an environment variable or a more secure method for actual use.**
    *   *Note: The main T5-FiD fine-tuning and evaluation do NOT require the Gemini API key.*

## Data Acquisition

The notebook (`eRAG_Consegna_11_04.ipynb`) includes cells to automatically download the necessary Natural Questions dataset files (dev, train, test) from the KILT repository into a `data/` directory.

```bash
# Commands executed within the notebook (Cell 3):
!mkdir -p data
!wget -O data/nq-dev-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl
!wget -O data/nq-train-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl
!wget -O data/nq-test_without_answers-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-test_without_answers-kilt.jsonl
```

The KILT Wikipedia knowledge source (`kilt_knowledgesource.json`) is processed *streamed* from its URL in Cell 5 to avoid downloading the entire large file directly. Only a subset is processed and saved to `wikipedia_passages_sample.jsonl` in the current preliminary setup. **Running on the full corpus requires significant changes and resources.**

## Running the Code

1.  Ensure all setup steps are complete.
2.  Open and run the Jupyter Notebook: `eRAG_Consegna_11_04.ipynb`.
3.  **Execution Order:** Run the cells sequentially.
4.  **Resource Requirements:**
    *   A GPU is highly recommended, especially for Contriever encoding and T5 fine-tuning/inference. The notebook was developed and tested on a Google Colab T4 GPU.
    *   Significant RAM is needed for BM25 indexing if using a larger corpus subset.
    *   Running the full pipeline, especially fine-tuning and evaluation loops, can be time-consuming.

## Preliminary Results (Summary)

The initial experiment (BM25 retriever, T5-Small-FiD fine-tuned 1 epoch on NQ dev split, evaluated on NQ dev split, K=40, ~1/1000 corpus) yielded:

*   **End-to-End Exact Match:** ~1.06% (very low generator performance).
*   **eRAG P@40:** ~0.0047
*   **eRAG Success@40:** ~0.074
*   **Correlation (eRAG vs End-to-End EM):**
    *   Kendall's Tau (P@40): ~0.232 (p ≈ 0.000)
    *   Kendall's Tau (Success@40): ~0.234 (p ≈ 0.000)

While statistically significant, the observed correlations are much lower than reported in the eRAG paper (~0.52-0.53). This is strongly believed to be a consequence of the extremely low absolute performance caused by the resource limitations (insufficient fine-tuning and poor retrieval from the corpus subset).

Detailed per-query results and aggregated scores from this preliminary run are saved in the `logs/` directory upon running the final evaluation cell in the notebook. See the accompanying PDF report (`eRAG_Reproducibility_Report.pdf`) for a detailed discussion.

## Challenges & Future Work

*   **Primary Challenge:** Lack of adequate computational resources (GPU VRAM, compute time) compared to the original study.
*   **Future Work:**
    1.  Secure access to more powerful GPUs (e.g., A100 or equivalent).
    2.  Process and index the **full** KILT Wikipedia corpus for BM25 and Contriever.
    3.  Perform T5-FiD fine-tuning for the **full 10 epochs** using the **official NQ training set**.
    4.  Run evaluations using the **Contriever** retriever.
    5.  Reproduce experiments varying the retrieval depth `k`.
    6.  Conduct computational efficiency comparisons.
    7.  Extend evaluation to other KILT datasets (FEVER, WoW) if resources permit.

## Citation

If you use this work, please cite the original eRAG paper:

```bibtex
@misc{salemi2024evaluating,
      title={Evaluating Retrieval Quality in Retrieval-Augmented Generation},
      author={Alireza Salemi and Hamed Zamani},
      year={2024},
      eprint={2404.13781},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
