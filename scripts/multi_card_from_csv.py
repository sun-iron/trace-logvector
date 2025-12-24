#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
multi_card_from_csv.py
- Create a multi CARD LogVector index from CSV log/schema data (card text → embedding → FAISS storage)
- Logs are grouped by two criteria: table and Call_Func , and chunks are created and merged.
- The final RAG index is optimized for both function-centric and table-centric searches.
- Use SentenceTransformer (default), fallback to TF-IDF if unavailable (option --use-tfidf)

USAGE :
  python multi_card_from_csv.py \
    --csv /path/to/0-trace_log_data.csv \
    --index-out ./data/logvector_m.index \
    --chunks-out ./data/logvector_m.pkl \
    --cards-out ./data/logvector_cards_m.csv
"""

import argparse
import os
import re
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import faiss
except Exception as e:
    raise RuntimeError("FAISS is required. Install with `pip install faiss-cpu`.") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass


# -----------------------------
# Text Normalization/Masking
# -----------------------------
SQL_QUOTE_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
NUM_RE = re.compile(r'\b\d+(\.\d+)?\b')
WS_RE = re.compile(r'\s+')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_RE = re.compile(r'\+?\d[\d\-]{7,}\d')

def normalize_sql(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = SQL_QUOTE_RE.sub("<STR>", s)
    s = NUM_RE.sub("<NUM>", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def mask_pii(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = EMAIL_RE.sub("email:***@***", s)
    s = PHONE_RE.sub("phone:***", s)
    return s

def normalize_generic(s: str, sql_mode: bool=False, do_mask: bool=False) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if do_mask:
        s = mask_pii(s)
    if sql_mode:
        s = normalize_sql(s)
    else:
        s = SQL_QUOTE_RE.sub("<STR>", s)
        s = NUM_RE.sub("<NUM>", s)
        s = NUM_RE.sub("<NUM>", s) # Second-order numeric normalization
        s = WS_RE.sub(" ", s).strip()
    return s


# -------------------------------------
# Grouping-based chunk generation function (supports both 'Call_Func' and 'table')
# -------------------------------------
def create_relationship_card(group_df: pd.DataFrame, group_col: str, normalize_sql_flag: bool, mask_flag: bool) -> str:
    # Aggregate all information from the grouped DataFrame to produce a single long card text.
    
    # 1. Group Entity value
    group_value = group_df[group_col].iloc[0]
    
    # 2. Extract all unique tables (table)
    tables = group_df['table'].dropna().astype(str).unique()
    tables_normalized = [normalize_generic(t, False, mask_flag) for t in tables]
    tables_str = " ".join(tables_normalized)
    
    # 3. Extract all unique call functions (Call_Func)
    call_funcs = group_df['Call_Func'].astype(str).unique()
    call_funcs_normalized = [normalize_generic(f, False, mask_flag) for f in call_funcs[:30]] # Max 30
    call_funcs_str = " ".join(call_funcs_normalized) 
    
    # 4. Extract all unique call filename (Call_File)
    call_files = group_df['Call_File'].astype(str).unique()
    call_files_normalized = [normalize_generic(f, False, mask_flag) for f in call_files]
    call_files_str = " ".join(call_files_normalized)

    # 5. Generate final chunk text (specify relationships)
    if group_col == 'table':
        # Table-centered chunk
        card_text = (
            f"ENTITY_TYPE:TABLE ENTITY_VALUE:{group_value} "
            f"ACCESSED_BY_FUNCTIONS:{call_funcs_str} "
            f"ACCESSED_BY_FILES:{call_files_str}"
        )
    elif group_col == 'Call_Func':
        # Funtion-centered chunk
        card_text = (
            f"ENTITY_TYPE:FUNCTION ENTITY_VALUE:{group_value} "
            f"ACCESSES_TABLES:{tables_str} "
            f"CALLED_FROM_FILES:{call_files_str}"
        )
    else:
        # etc
        card_text = f"CONTEXT_ENTITY:{group_col}:{group_value} TABLES:{tables_str} FUNCTIONS:{call_funcs_str}"
        
    return card_text

# -----------------------------
# Embedding & Indexing
# -----------------------------
def build_embeddings(cards: List[str], model_name: str, use_tfidf: bool=False) -> Tuple[np.ndarray, dict]:
    """
    returns: (embeddings[n, d], meta)
    """
    meta = {}
    if not use_tfidf:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            X = model.encode(cards, show_progress_bar=True, normalize_embeddings=True)
            meta = {"type": "st", "model": model_name}
            return np.asarray(X).astype("float32"), meta
        except Exception as e:
            print(f"[WARN] SentenceTransformer load failed({e}); Change to TF-IDF engine")
            use_tfidf = True

    # TF-IDF fallback (character + word n-gram combination)
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    
    print("[INFO] Falling back to TF-IDF vectorization...")

    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    Xc = vec_char.fit_transform(cards)
    vec_word = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xw = vec_word.fit_transform(cards)
    Xs = hstack([Xc, Xw]).astype("float32")

    # L2 regularization
    Xs = normalize(Xs, norm="l2", copy=False)
    meta = {"type": "tfidf", "vectorizer_char": vec_char, "vectorizer_word": vec_word}
    
    X_dense = Xs.toarray().astype("float32")
    return X_dense, meta

def build_faiss_index(embeds: np.ndarray):
    d = embeds.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeds)
    return index


# -----------------------------
#  Pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="./0-trace_log_data.csv", help="Input CSV file path")
    ap.add_argument("--sep", default=",", help="CSV Separator")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer Model name")
    ap.add_argument("--index-out", default="./data/logvector_m.index", help="FAISS Index Output path")
    ap.add_argument("--chunks-out", default="./data/logvector_m.pkl", help="Pickle Output path")
    ap.add_argument("--cards-out", default="./data/logvector_cards_m.csv", help="CARD text CSV output path")
    ap.add_argument("--normalize-sql", action="store_true", help="Enable SQL normalization")
    ap.add_argument("--mask-pii", action="store_true", help="Masking text")
    ap.add_argument("--max-rows", type=int, default=200000, help="Maximum processed rows")
    ap.add_argument("--use-tfidf", action="store_true", help="Forced use of TF-IDF")    
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)

    # 1) Load CSV (attempts to automatically guess delimiter)
    df = None
    tried = [args.sep, None, ",", "|", "\t", ";"]
    for sep in tried:
        if sep is None and args.sep is not None:
            continue
        try:
            df = pd.read_csv(args.csv, engine="python", sep=sep)
            break
        except Exception:
            continue
    if df is None:
        df = pd.read_csv(args.csv, engine="python", on_bad_lines="skip")

    if len(df) > args.max_rows:
        df = df.iloc[:args.max_rows].copy()

    # 2) Generating card text: Table and function-centric grouping
    GROUP_COLS = ['table', 'Call_Func']
    
    all_cards: List[str] = []
    all_ids: List[str] = []
    
    for group_col in GROUP_COLS:
        # Exclude rows without a grouping criteria column
        df_cleaned = df.dropna(subset=[group_col]).copy()
        
        if group_col not in df_cleaned.columns:
            print(f"[WARN] Skipping grouping criteria column '{group_col}' as it does not exist in the data.")
            continue
        
        print(f"[INFO] Grouping criteria: {group_col}")
        grouped = df_cleaned.groupby(group_col)
        
        cards: List[str] = []
        ids: List[str] = []
        
        for group_name, group_df in grouped:
            # Use the group name as the new ID
            new_id = str(group_name) 
            
            card = create_relationship_card(
                group_df, 
                group_col=group_col,
                normalize_sql_flag=args.normalize_sql, 
                mask_flag=args.mask_pii
            )
            
            cards.append(card)
            # Ensure uniqueness by prefixing the grouping criteria to the ID (e.g.table:wp_posts, func:WP_Post::get_instance())
            ids.append(f"{group_col}:{new_id}") 
        
        print(f"[INFO] Create {group_col} chunks based count is {len(cards)}.")
        all_cards.extend(cards)
        all_ids.extend(ids)
        
    print(f"[INFO] Total Chunks (Cards) Finally Generated. Count is {len(all_cards)}")

    # 3) Embedding (using combined data)
    embeds, meta = build_embeddings(all_cards, model_name=args.model, use_tfidf=args.use_tfidf)
    print(f"[INFO] embeddings: shape={embeds.shape}, type={meta.get('type')}")

    # 4) FAISS Index
    index = build_faiss_index(embeds)

    # 5) Store
    faiss.write_index(index, args.index_out)
    with open(args.chunks_out, "wb") as f:
        # Stores columns used as grouping criteria instead of schema_cols
        pickle.dump({"cards": all_cards, "ids": all_ids, "schema_cols": GROUP_COLS, "embed_meta": meta}, f)

    # Card CSV (for review)
    out_df = pd.DataFrame({"id": all_ids, "card_text": all_cards})
    out_df.to_csv(args.cards_out, index=False, encoding="utf-8")

    print(f"[DONE] index → {args.index_out}")
    print(f"[DONE] pickle → {args.chunks_out}")
    print(f"[DONE] cards → {args.cards_out}")


if __name__ == "__main__":
    main()