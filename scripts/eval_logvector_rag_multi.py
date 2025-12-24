#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_logvector_rag_multi.py
- Evaluates RAG search performance by loading multi-card indexes and chunks.
- Modifies EVAL_DATASET to accommodate table-centric (table:ID) and function-centric (Call_Func:ID) multi-grouping chunks.
- Calculates Hit Rate@K and MRR@K metrics for performance comparison.
- Displays the actual rank at which the correct answer was found (Found at Rank R) when a Top-1 failure occurs.
"""

import argparse
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
import os
import re
import sys

# -----------------------------
# Text Normalization/Masking
# -----------------------------
SQL_QUOTE_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
NUM_RE = re.compile(r'\b\d+(\.\d+)?\b')
WS_RE = re.compile(r'\s+')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_RE = re.compile(r'\+?\d[\d\-]{7,}\d')

def mask_pii(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = EMAIL_RE.sub("email:***@***", s)
    s = PHONE_RE.sub("phone:***", s)
    return s

def normalize_sql(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = SQL_QUOTE_RE.sub("<STR>", s)
    s = NUM_RE.sub("<NUM>", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def normalize_generic(s: str, sql_mode: bool=False, do_mask: bool=False) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.strip()
    if do_mask: s = mask_pii(s)
    if sql_mode:
        s = normalize_sql(s)
    else:
        s = SQL_QUOTE_RE.sub("<STR>", s)
        s = NUM_RE.sub("<NUM>", s)
        s = WS_RE.sub(" ", s).strip()
    return s

# -----------------------------
# Search (Query Encoding)
# -----------------------------
def encode_query(query: str, meta: dict, model_name: str, 
                 normalize_sql_flag: bool, mask_flag: bool) -> np.ndarray:
    
    processed_query = normalize_generic(query, sql_mode=normalize_sql_flag, do_mask=mask_flag)
    embed_type = meta.get('type')
    
    if embed_type == 'st':
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            X = model.encode([processed_query], normalize_embeddings=True)
            return np.asarray(X).astype("float32")
        except ImportError:
            raise ImportError("SentenceTransformer not found. `pip install sentence-transformers`")
        except Exception as e:
            raise Exception(f"SentenceTransformer Encoding failed: {e}")
    
    elif embed_type == 'tfidf':
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec_char = meta.get('vectorizer_char')
            vec_word = meta.get('vectorizer_word')
            
            if vec_char is None or vec_word is None:
                 raise KeyError("TF-IDF vectorizer object is not in metadata")
                 
            Xc = vec_char.transform([processed_query])
            Xw = vec_word.transform([processed_query])
            Xs = hstack([Xc, Xw]).astype("float32")
            
            from sklearn.preprocessing import normalize
            Xs = normalize(Xs, norm="l2", copy=False)
            return Xs.toarray().astype("float32")
        except Exception as e:
            raise Exception(f"TF-IDF Encoding failed: {e}")
    else:
        raise ValueError(f"[ERR] Unknown Embedding type: {embed_type}")


# -----------------------------
# Evaluation dataset
# -----------------------------
# The correct answer ID uses the new ID format (table:table name, Call_Func:function name).
EVAL_DATASET: List[Dict] = [
    {
        "query": "wp_usermeta 테이블에 접근하는 함수는 무엇인가?", 
        "expected_ids": ["table:wp_usermeta"], 
        "description": "테이블 중심 검색 (단일 정답)"
    },
    {
        "query": "WP_Post::get_instance() 함수가 접근하는 테이블은 무엇인가?",
        "expected_ids": ["Call_Func:WP_Post::get_instance()"],
        "description": "함수 중심 검색 (단일 정답)"
    },
    {
        "query": "AutomatticWAARTSDataStore::get_taxes() 함수가 사용하는 테이블을 알려줘.",
        "expected_ids": ["Call_Func:AutomatticWAARTSDataStore::get_taxes()", "table:wp_woocommerce_tax_rates"],
        "description": "함수-테이블 양방향 검색 (복수 정답)"
    },
    {
        "query": "wp_wc_admin_notes 테이블과 관련된 함수들을 모두 알려줘.",
        "expected_ids": ["table:wp_wc_admin_notes", "Call_Func:AutomatticWANDataStore->read()"],
        "description": "테이블-함수 양방향 검색 (복수 정답)"
    },
    {
        "query": "우커머스의 관리자 노트 액션(wp_wc_admin_note_actions)과 관련된 데이터 흐름 정보는?",
        "expected_ids": ["table:wp_wc_admin_note_actions", "Call_Func:AutomatticWANDataStore->read_actions()"],
        "description": "키워드 기반 관계 검색 (복수 정답)"
    },
]


# -----------------------------
# Calculating evaluation metrics
# -----------------------------
def evaluate_retrieval(D: np.ndarray, I: np.ndarray, expected_ids: List[List[str]], ids: List[str], k: int) -> Tuple[float, float]:
    
    # Calculate Hit Rate@K and MRR@K.
    num_queries = len(expected_ids)
    hit_count = 0
    reciprocal_ranks = []
    
    for i in range(num_queries):
        expected = set(expected_ids[i])
        retrieved_local_indices = I[i, :k]
        
        # Convert to actual document ID
        retrieved_ids = [ids[idx] for idx in retrieved_local_indices if idx < len(ids)]
        
        # 1. Calculate Hit Rate@K
        # If any of the K searched items are included in the correct answer, it is a HIT
        if any(retrieved_id in expected for retrieved_id in retrieved_ids):
            hit_count += 1
        
        # 2. Calculate MRR@K
        rank = 0
        for j, retrieved_id in enumerate(retrieved_ids):
            if retrieved_id in expected:
                rank = j + 1
                break
        
        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        
    hit_rate = hit_count / num_queries
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    return hit_rate, mrr


# -----------------------------
# Pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-path", default="./data/logvector_cards_m.index", help="FAISS Index path")
    ap.add_argument("--chunks-path", default="./data/logvector_cards_m.pkl", help="Pickle path")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer Model name")
    ap.add_argument("--top-k", type=int, default=5, help="Top number of documents to search")
    ap.add_argument("--normalize-sql", action="store_true", help="SEnable SQL normalization")
    ap.add_argument("--mask-pii", action="store_true", help="Enable PII masking")

    args = ap.parse_args()

    # 1. Load Index and Chunk file
    try:
        index = faiss.read_index(args.index_path)
        with open(args.chunks_path, "rb") as f:
            chunk_data = pickle.load(f)
        
        cards: List[str] = chunk_data['cards']
        ids: List[str] = chunk_data['ids']
        meta: dict = chunk_data['embed_meta']
        
    except FileNotFoundError:
        print(f"[FATAL] Fine not found(Index or Chunk): {args.index_path}, {args.chunks_path}")
        return
    except Exception as e:
        print(f"[FATAL] Load Failed(Index or Chunk): {e}")
        return

    print(f"[INFO] Index Loaded: Dimension={index.d}, Type={meta.get('type')}")
    print(f"[INFO] Total Chunks: {len(cards)} (Using Multi-CARD")
    print(f"[INFO] Evaluation Queries: {len(EVAL_DATASET)}")

    # 2. Query processing and retrieval
    query_vectors = []
    
    for item in EVAL_DATASET:
        try:
            q_embed = encode_query(
                item['query'], 
                meta, 
                args.model, 
                args.normalize_sql, 
                args.mask_pii
            )
            query_vectors.append(q_embed[0])
        except Exception as e:
            print(f"[WARN] Load Query failed ({item['query'][:20]}...): {e}")
            query_vectors.append(None)

    valid_query_vectors_list = [v for v in query_vectors if v is not None]
    valid_query_vectors = np.array(valid_query_vectors_list).astype('float32') if valid_query_vectors_list else np.array([])
    
    valid_expected_ids = [item['expected_ids'] for i, item in enumerate(EVAL_DATASET) if query_vectors[i] is not None]
    valid_eval_dataset = [item for i, item in enumerate(EVAL_DATASET) if query_vectors[i] is not None]


    if len(valid_query_vectors) == 0:
        print("[FAIL] Filed to make Embedding Vector.")
        return

    # 3. FAISS Searching
    print(f"[INFO] Searching FAISS Index (k={args.top_k})...")
    D, I = index.search(valid_query_vectors, args.top_k)
    
    # 4. Performance Evaluation
    hit_rate, mrr = evaluate_retrieval(D, I, valid_expected_ids, ids, args.top_k)
    
    # 5. Output results
    print("\n" + "="*70)
    print("      Multi-CARD RAG Retrieval Performance Metrics")
    print("="*70)
    print(f"  Embedding Type: {meta.get('type', 'N/A').upper()}")
    print(f"  Top-K Value: {args.top_k}")
    print(f"  Total Queries Evaluated: {len(valid_expected_ids)}")
    print("-" * 70)
    print(f"  [1] Hit Rate@{args.top_k}: {hit_rate:.4f} (Probability that the correct document is included within K)")
    print(f"  [2] MRR@{args.top_k}:      {mrr:.4f} (The average of how high the correct document was ranked in the search results)")
    print("="*70)

    # 6. Example of individual query results (ranking output logic other than Top-1)
    print("\n[INFO] Retrieval Analysis:")
    for i, item in enumerate(valid_eval_dataset):
        if i >= len(I): continue 
        
        expected = set(item['expected_ids'])
        retrieved_indices = I[i, :args.top_k]
        
        # 1. Find the first rank where the correct answer was found
        found_rank = -1
        for j, local_index in enumerate(retrieved_indices):
            if local_index < len(ids):
                retrieved_id = ids[local_index]
                if retrieved_id in expected:
                    found_rank = j + 1
                    break
        
        # 2. Top-1 Information
        retrieved_top1_id = ids[I[i, 0]] if I[i, 0] < len(ids) else "N/A"
        retrieved_top1_card = cards[I[i, 0]][:100] + "..." if I[i, 0] < len(ids) else "N/A"

        # 3. Set status tags
        if found_rank == 1:
            status_tag = "[OK] Found at Top 1"
        elif found_rank > 1:
            status_tag = f"[Hit] Found at Rank {found_rank}" # Top-1 failed, but Hit succeeded
        else:
            status_tag = f"[NG] Not found in Top-{args.top_k}"
            
        
        print(f"\n{status_tag} Query ({item['description']}): {item['query']}")
        print(f"  - Expected IDs: {item['expected_ids']}")
        print(f"  - Top 1 ID:     {retrieved_top1_id}")
        
        # If it is not the Top-1 correct answer (found_rank > 1) or not found at all (found_rank == -1), print the Top-1 card and use it for analysis.
        if found_rank != 1:
            print(f"  - Top 1 Card:   {retrieved_top1_card}")
            
        # If it is not Top-1, print the card where the actual correct answer was found (for comparison analysis with the Top-1 card)
        if found_rank > 1:
            correct_local_index = I[i, found_rank - 1]
            correct_id = ids[correct_local_index]
            correct_card = cards[correct_local_index][:100] + "..."
            print(f"  - Correct Rank {found_rank} ID: {correct_id}")
            print(f"  - Correct Rank {found_rank} Card: {correct_card}")
        
if __name__ == "__main__":
    main()