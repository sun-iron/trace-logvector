# Trace-LogVector (TLV)

This repository contains the official reference implementation and dataset for the paper: **"Trace-LogVector-Based Relational Retrieval for Conversational System Log Analysis"**.

We propose **Trace-LogVector (TLV)**, a relational log representation method designed to improve Retrieval-Augmented Generation (RAG) performance in system log analysis. This project demonstrates how **CARD (Chunk as a Relational Data)**-based multi-chunk strategies significantly outperform traditional single-chunk approaches in retrieving execution traces.
<img width="794" height="402" alt="image" src="https://github.com/user-attachments/assets/cd58f122-fc76-4a20-92b9-9908a5f9979d" />


## Repository Structure
```text
Trace-LogVector
├── dataset
│   ├── 0-trace_log_data.csv           # Raw trace log data derived from service call analysis
│   ├── logvector_cards_s.csv          # Generated Single-chunk TLV representations
│   └── logvector_cards_m.csv          # Generated Multi-chunk (CARD-based) TLV representations
├── script
│   ├── single_card_from_csv.py        # Script to generate Single-chunk representations
│   ├── multi_card_from_csv.py         # Script to generate Multi-chunk (CARD) representations
│   ├── eval_logvector_rag_single.py   # Evaluation script for Single-chunk strategy
│   └── eval_logvector_rag_multi.py    # Evaluation script for Multi-chunk strategy
|
└── baselines_eval                     # Baseline experiments & Ablation study (Added for Revision)
    ├── 1_tfidf_baseline_sklearn.py    # TF-IDF lexical retriever baseline (2x2 Ablation Study)
    ├── 2_sliding_window_from_csv.py   # Sliding window chunking evaluation on structured CSV
    ├── 3_raw_sliding_window_eval.py   # Standard sliding window evaluation directly on raw logs
    └── 4-1_run_sliding_sweep.py       # Sliding window parameter sweep & Noise metric analysis
```
# Evaluation Baselines & Ablation Study (baselines_eval/)
To rigorously validate the effectiveness of the TLV and CARD representations, we provide additional baseline experiments and a 2x2 ablation study. These scripts evaluate the structural representation against standard chunking methods and traditional lexical retrievers:

TF-IDF vs. Transformer (Ablation Study): Compares the semantic matching of the dense Transformer model against a traditional TF-IDF sparse retriever to prove that performance gains are driven by the structural representation (CARD).

Sliding Window Chunking vs. Relational Chunking: Evaluates standard naive sliding-window strategies across various chunk sizes. It introduces a Noise Metric (average number of distinct interleaved rows per retrieved chunk) to quantify context pollution and demonstrate why structural isolation is necessary for highly concurrent log environments.
