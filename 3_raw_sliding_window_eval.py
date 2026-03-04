import argparse
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple


EVAL_MULTI: List[Dict] = [
    {
        "query": "Which functions access the wp_usermeta table?",
        "expected_keywords": ["wp_usermeta"],
        "description": "Table-centric search",
    },
    {
        "query": "Which table(s) does the function WP_Post::get_instance() access?",
        "expected_keywords": ["WP_Post::get_instance()", "wp_posts"],
        "description": "Function-centric search",
    },
    {
        "query": "Which table(s) are used by the function AutomatticWAARTSDataStore::get_taxes()?",
        "expected_keywords": ["AutomatticWAARTSDataStore::get_taxes()", "wp_woocommerce_tax_rates"],
        "description": "Bidirectional function–table search",
    },
    {
        "query": "List all functions related to the wp_wc_admin_notes table.",
        "expected_keywords": ["wp_wc_admin_notes", "AutomatticWANDataStore->read()"],
        "description": "Bidirectional table–function search",
    },
    {
        "query": "What is the data-flow information related to the WooCommerce admin note actions (wp_wc_admin_note_actions)?",
        "expected_keywords": ["wp_wc_admin_note_actions", "AutomatticWANDataStore->read_actions()"],
        "description": "Keyword-based relational search",
    },
]

def load_raw_csv_to_text(path: str) -> str:
    df = pd.read_csv(path).fillna("")
    
    lines = df.apply(lambda row: ','.join(row.values.astype(str)), axis=1).tolist()
    full_text = '\n'.join(lines)
    return full_text

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

def evaluate_hit_mrr_by_text(I: np.ndarray, chunks: List[str], expected_list: List[List[str]], k: int) -> Tuple[float, float]:
    hit = 0
    rr = []

    for qi in range(len(expected_list)):
        expected_keywords = expected_list[qi]
        
        retrieved_chunks = []
        for idx in I[qi, :]:
            if idx < len(chunks) and chunks[idx] not in retrieved_chunks:
                retrieved_chunks.append(chunks[idx])
            if len(retrieved_chunks) == k:
                break

        is_hit = False
        rank = 0
        
        for j, chunk_text in enumerate(retrieved_chunks):
            chunk_text_lower = chunk_text.lower()
            
            if all(kw.lower() in chunk_text_lower for kw in expected_keywords):
                is_hit = True
                rank = j + 1
                break 

        if is_hit:
            hit += 1
            rr.append(1.0 / rank)
        else:
            rr.append(0.0)

    return hit / len(expected_list), float(np.mean(rr))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", default="./dataset/0-trace_log_data.csv", help="Raw log CSV path")
    ap.add_argument("--model-name", default="all-MiniLM-L6-v2", help="HuggingFace model name")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    args = ap.parse_args()

    print(f"1. Load Raw data: {args.raw_csv}")
    full_log_text = load_raw_csv_to_text(args.raw_csv)
    print(f"   -> Total text length: {len(full_log_text):,} 문자")

    print(f"2. Chucking(Sliding Window) (Size: {args.chunk_size}, Overlap: {args.chunk_overlap})")
    all_chunks = sliding_window_chunking(full_log_text, args.chunk_size, args.chunk_overlap)
    print(f"   -> Total {len(all_chunks)} Chucking created")

    print(f"3. Load Model : {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    chunk_embeddings = model.encode(all_chunks, show_progress_bar=True)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')
    
    queries = [d["query"] for d in EVAL_MULTI]
    expected_keywords = [d["expected_keywords"] for d in EVAL_MULTI]
    
    query_embeddings = model.encode(queries)
    query_embeddings = np.array(query_embeddings).astype('float32')

    print("4. FAISS Searching (Cosine Similarity)...")
    faiss.normalize_L2(chunk_embeddings)
    faiss.normalize_L2(query_embeddings)

    embed_dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(embed_dim)
    index.add(chunk_embeddings)

    search_k = args.top_k * 5 
    _, I = index.search(query_embeddings, search_k)

    # 6. Eveluation(Sub-string Match)
    hit, mrr = evaluate_hit_mrr_by_text(I, all_chunks, expected_keywords, k=args.top_k)
    
    print("=" * 72)
    print(f"[Raw Log + Sliding Window + Transformer Baseline]")
    print(f"File = {args.raw_csv}")
    print(f"Chunk Size = {args.chunk_size}, Overlap = {args.chunk_overlap}")
    print(f"Top-K = {args.top_k}")
    print(f"Hit@{args.top_k} = {hit:.4f}")
    print(f"MRR@{args.top_k} = {mrr:.4f}")
    print("=" * 72)

    print("\n[ Top-1  (Noise check)]")
    for qi, d in enumerate(EVAL_MULTI):
        expected_kws = d["expected_keywords"]
        
        retrieved_chunks = []
        for idx in I[qi, :]:
            if idx < len(all_chunks) and all_chunks[idx] not in retrieved_chunks:
                retrieved_chunks.append(all_chunks[idx])
            if len(retrieved_chunks) == args.top_k:
                break
                
        top1_chunk = retrieved_chunks[0]
        
        print(f"\nQ{qi+1}: {d['query']}")
        print(f"-> Keyword: {expected_kws}")
        print("-" * 40)
        print(top1_chunk)
        print("-" * 40)
        
if __name__ == "__main__":
    main()