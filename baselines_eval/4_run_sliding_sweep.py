import argparse
import numpy as np
import pandas as pd
import faiss
import torch
import random
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

EVAL_SINGLE: List[Dict] = [
    {"query": "Which functions access the wp_usermeta table?", "expected_keywords": ["wp_usermeta"]},
    {"query": "Which table(s) does the function WP_Post::get_instance() access?", "expected_keywords": ["WP_Post::get_instance", "wp_posts"]},
    {"query": "Which table(s) are used by the function AutomatticWAARTSDataStore::get_taxes()?", "expected_keywords": ["get_taxes", "wp_woocommerce_tax_rates"]},
    {"query": "List all functions related to the wp_wc_admin_notes table.", "expected_keywords": ["wp_wc_admin_notes", "AutomatticWANDataStore->read"]},
    {"query": "What is the data-flow information related to the WooCommerce admin note actions (wp_wc_admin_note_actions)?", "expected_keywords": ["wp_wc_admin_note_actions", "AutomatticWANDataStore->read_actions"]},
]

def load_and_prepare_data(csv_path: str) -> Tuple[str, List[str], List[List[str]]]:
    print(f"-> Data loading: {csv_path}")
    df = pd.read_csv(csv_path).fillna("")
    
    df["card_text"] = df["card_text"].astype(str)
    df["id"] = df["id"].astype(str)
    
    full_text = ""
    char_to_row_id = []
    
    for _, row in df.iterrows():
        r_id = row['id']
        text = row['card_text']
        full_text += text + "\n"
        char_to_row_id.extend([r_id] * (len(text) + 1))
        
    assert len(full_text) == len(char_to_row_id), "Mismatch Text length and ID"
        
    print("\n-> Dynamic keyword (Ground Truth) ID created...")
    expected_ids_list = []
    
    with open("gt_report.txt", "w", encoding="utf-8") as f:
        f.write("=== Ground Truth (GT) Generation Report ===\n")
        f.write("Rule: AND logical matching on card_text\n\n")
        
        for q_idx, eval_dict in enumerate(EVAL_SINGLE):
            kws = eval_dict["expected_keywords"]
            
            mask = np.ones(len(df), dtype=bool)
            for kw in kws:
                mask = mask & df['card_text'].str.contains(kw, case=False, regex=False)
            
            matching_ids = df[mask]['id'].tolist()
            expected_ids_list.append(matching_ids)
            
            log_str = f"Q{q_idx+1} (Keyword: {kws}) -> {len(matching_ids)} Row mapping"
            print("   " + log_str)
            
            f.write(f"Q{q_idx+1}: {eval_dict['query']}\n")
            f.write(f" - Keywords (AND): {kws}\n")
            f.write(f" - Total GT Rows: {len(matching_ids)}\n")
            f.write(f" - Sample IDs (up to 5): {matching_ids[:5]}\n\n")
            
    print("\n-> End label report file created: 'gt_report.txt'")

    return full_text, char_to_row_id, expected_ids_list

def strict_sliding_window(full_text: str, char_to_row_id: List[str], size: int, overlap: int):
    if overlap >= size:
        raise ValueError(f"overlap({overlap}) must be strictly less than size({size})")

    chunks = []
    chunk_row_ids = []
    
    step = size - overlap
    for i in range(0, len(full_text) - overlap, step):
        chunk = full_text[i : i + size]
        if not chunk.strip(): continue
        
        chunks.append(chunk)
        unique_ids_in_chunk = list(dict.fromkeys(char_to_row_id[i : i + size]))
        chunk_row_ids.append(unique_ids_in_chunk)
        
        if i + size >= len(full_text):
            break
    return chunks, chunk_row_ids

def evaluate_strict(I: np.ndarray, chunk_row_ids: List[List[str]], expected_list: List[List[str]], k: int) -> Tuple[float, float, float, List[str]]:
    hit = 0
    rr = []
    noise_counts = []
    debug_logs = []

    for qi in range(len(expected_list)):
        expected = set(expected_list[qi])
        
        if not expected:
            continue
            
        retrieved_unique_ids = []
        seen = set() 
        
        for idx in I[qi, :]:
            if idx < 0 or idx >= len(chunk_row_ids):
                continue
            for r_id in chunk_row_ids[idx]:
                if r_id not in seen:
                    seen.add(r_id)
                    retrieved_unique_ids.append(r_id)
            if len(retrieved_unique_ids) >= k:
                break
                
        if any(r in expected for r in retrieved_unique_ids[:k]):
            hit += 1
            
        rank = 0
        for j, r in enumerate(retrieved_unique_ids[:k]):
            if r in expected:
                rank = j + 1
                break
        rr.append(1.0 / rank if rank > 0 else 0.0)
        
        top_k_chunks_idx = [idx for idx in I[qi, :k] if 0 <= idx < len(chunk_row_ids)]
        if top_k_chunks_idx:
            noise_in_top_chunks = [len(chunk_row_ids[idx]) for idx in top_k_chunks_idx]
            noise_counts.append(np.mean(noise_in_top_chunks))
        else:
            noise_counts.append(0.0)

        # Debug out
        top1_idx = I[qi, 0] if len(I[qi]) > 0 else -1
        if 0 <= top1_idx < len(chunk_row_ids):
            debug_logs.append(f"Q{qi+1} Top-1 Chunk contains {len(chunk_row_ids[top1_idx])} distinct rows")

    valid_q_count = sum(1 for ids in expected_list if len(ids) > 0)
    final_hit = hit / valid_q_count if valid_q_count > 0 else 0.0
    final_mrr = float(np.mean(rr)) if rr else 0.0
    final_noise = float(np.mean(noise_counts)) if noise_counts else 0.0

    return final_hit, final_mrr, final_noise, debug_logs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./dataset/logvector_cards_s.csv")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    sweep_params = [(100, 20), (300, 50), (500, 50), (1000, 100)]
    results = []

    print("\n========================================================")
    print("1. Data and Model load")
    print("========================================================")
    full_text, char_to_row_id, expected_list = load_and_prepare_data(args.csv)
    
    valid_q = sum(1 for ids in expected_list if len(ids) > 0)
    
    print("\n-> 임베딩 모델 로드 중...")
    model = SentenceTransformer(args.model)
    queries = [d["query"] for d in EVAL_SINGLE]
    query_embeddings = np.array(model.encode(queries, show_progress_bar=False)).astype('float32')
    faiss.normalize_L2(query_embeddings)

    print("\n========================================================")
    print("2. Parameter sweepy (Size / Overlap)")
    print("========================================================\n")
    
    for size, overlap in sweep_params:
        chunks, chunk_row_ids = strict_sliding_window(full_text, char_to_row_id, size, overlap)
        chunk_embeddings = np.array(model.encode(chunks, show_progress_bar=False)).astype('float32')
        faiss.normalize_L2(chunk_embeddings)
        
        index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
        index.add(chunk_embeddings)
        _, I = index.search(query_embeddings, args.k * 10)

        hit, mrr, avg_noise, debug_logs = evaluate_strict(I, chunk_row_ids, expected_list, args.k)
        
        results.append({
            "Size": size, "Overlap": overlap, "Hit@5": hit, "MRR@5": mrr, "Noise": avg_noise, "DebugLogs": debug_logs
        })

    print("\n" + "=" * 80)
    print(f"* Evaluated Queries: {valid_q} / {len(EVAL_SINGLE)} (Skipped empty GTs)")
    print("-" * 80)
    print(f"{'Size':<6} | {'Overlap':<7} | {'Hit@5':<6} | {'MRR@5':<6} | {'Noise (# distinct rows/chunk)':<25}")
    print("-" * 80)
    for r in results:
        print(f"{r['Size']:<6} | {r['Overlap']:<7} | {r['Hit@5']:<6.4f} | {r['MRR@5']:<6.4f} | {r['Noise']:<25.2f}")
    print("=" * 80)
    
    print("\n[Qualitative Evidence: Top-1 Chunk Row Mixing]")
    for r in results:
        print(f"\n--- Size: {r['Size']} ---")
        for log in r['DebugLogs'][:2]:
            print(log)

if __name__ == "__main__":
    main()
