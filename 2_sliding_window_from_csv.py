import argparse
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

def sliding_window_chunking(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap({chunk_overlap}) must be strictly less than chunk_size({chunk_size}).")
        
    chunks = []
    step_size = chunk_size - chunk_overlap
    
    if len(text) <= chunk_size:
        return [text]
        
    for i in range(0, len(text) - chunk_overlap, step_size):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks

def evaluate_hit_mrr(I: np.ndarray, chunk_to_original_id: List[str], expected_ids: List[List[str]], k: int) -> Tuple[float, float]:
    hit = 0
    rr = []

    for qi in range(len(expected_ids)):
        expected = set(expected_ids[qi])
        
        # 1. FAISS search result
        retrieved_original = [chunk_to_original_id[idx] for idx in I[qi, :] if idx < len(chunk_to_original_id)]
        
        retrieved_unique = []
        for r in retrieved_original:
            if r not in retrieved_unique:
                retrieved_unique.append(r)
            if len(retrieved_unique) == k: 
                break

        # 2. Hit & MRR 
        if any(r in expected for r in retrieved_unique):
            hit += 1

        rank = 0
        for j, r in enumerate(retrieved_unique):
            if r in expected:
                rank = j + 1
                break
        rr.append(1.0 / rank if rank > 0 else 0.0)

    return hit / len(expected_ids), float(np.mean(rr))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards-single", default="./dataset/logvector_cards_s.csv", help="Single-CARD CSV path")
    ap.add_argument("--model-name", default="all-MiniLM-L6-v2", help="HuggingFace model name")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    args = ap.parse_args()

    search_k = args.top_k * 10 
    _, I = index.search(query_embeddings, search_k)

    # 6. Hit@K & MRR@K
    hit, mrr = evaluate_hit_mrr(I, chunk_to_original_id, expected, k=args.top_k)
    
if __name__ == "__main__":
    main()