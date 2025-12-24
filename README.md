# Trace-LogVector (TLV)

This repository contains the official reference implementation and dataset for the paper: **"Trace-LogVector-Based Relational Retrieval for Conversational System Log Analysis"**.

We propose **Trace-LogVector (TLV)**, a relational log representation method designed to improve Retrieval-Augmented Generation (RAG) performance in system log analysis. This project demonstrates how **CARD (Chunk as a Relational Data)**-based multi-chunk strategies significantly outperform traditional single-chunk approaches in retrieving execution traces.

## Repository Structure
```text
Trace-LogVector
├── dataset
│   ├── 0-trace_log_data.csv       # Raw trace log data derived from service call analysis
│   ├── logvector_cards_s.csv      # Generated Single-chunk TLV representations
│   └── logvector_cards_m.csv      # Generated Multi-chunk (CARD-based) TLV representations
└── script
    ├── single_card_from_csv.py    # Script to generate Single-chunk representations
    ├── multi_card_from_csv.py     # Script to generate Multi-chunk (CARD) representations
    ├── eval_logvector_rag_single.py # Evaluation script for Single-chunk strategy
    └── eval_logvector_rag_multi.py  # Evaluation script for Multi-chunk strategy
