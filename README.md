# Trace-LogVector (TLV)

Official repository for the paper:

**Trace-LogVector-Based Relational Retrieval for Conversational System Log Analysis**  
Sun-Chul Park, Young-Han Kim  
*Sensors* 2026, 26(6), 1806  
DOI: [10.3390/s26061806](https://doi.org/10.3390/s26061806)

This repository provides the reference implementation and experimental dataset for **Trace-LogVector (TLV)**, a relational log representation designed for conversational system log analysis in Retrieval-Augmented Generation (RAG) settings.

## Overview

Conventional document-centric chunking is often insufficient for system log analysis because it breaks execution flow and obscures entity relationships.  
**Trace-LogVector (TLV)** addresses this limitation by representing logs as relational retrieval units that preserve:

- execution flow,
- entity relationships,
- and trace-level operational context.

This repository demonstrates how **CARD (Chunk as Relational Data)**-based multi-chunk construction improves retrieval performance over conventional single-chunk representations.

## Key Contributions

- **Trace-LogVector (TLV):** relational log representation for conversational system analysis
- **CARD design principle:** chunk construction based on relational execution units
- **Reproducible evaluation:** comparison of single-chunk vs. multi-chunk retrieval strategies
- **Additional baselines:** TF-IDF lexical retrieval, sliding-window chunking, and ablation analysis

## Repository Structure

```text
Trace-LogVector
├── dataset
│   ├── 0-trace_log_data.csv
│   ├── logvector_cards_s.csv
│   └── logvector_cards_m.csv
├── script
│   ├── single_card_from_csv.py
│   ├── multi_card_from_csv.py
│   ├── eval_logvector_rag_single.py
│   └── eval_logvector_rag_multi.py
└── baselines_eval
    ├── 1_tfidf_baseline_sklearn.py
    ├── 2_sliding_window_from_csv.py
    ├── 3_raw_sliding_window_eval.py
    └── 4-1_run_sliding_sweep.py
