#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single_card_from_csv.py
- Create a single CARD LogVector index from CSV log/schema data (card text → embedding → FAISS storage)
- Use SentenceTransformer (default), fallback to TF-IDF if unavailable (option --use-tfidf)

USAGE :
  python single_card_from_csv.py \
    --csv /path/to/0-trace_log_data.csv \
    --index-out ./data/logvector_s.index \
    --chunks-out ./data/logvector_s.pkl \
    --cards-out ./data/logvector_cards_s.csv
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
        s = WS_RE.sub(" ", s).strip()
    return s


# -------------------------------------
# Schema hint column auto-detection/card generation
# -------------------------------------
HINT_PAT = re.compile(r"(table|func|file|query|sql|svc|service|action|msg|message|path|module|class)", re.I)

def detect_schema_columns(df: pd.DataFrame, manual: str = "auto") -> List[str]:
    if manual and manual.lower() != "auto":
        cols = [c.strip() for c in manual.split(",") if c.strip() in df.columns]
        if cols:
            return cols
    
    hint_cols = [c for c in df.columns if HINT_PAT.search(c)]
    
    if not hint_cols:
        texty = [c for c in df.columns if df[c].dtype == "object"]
        hint_cols = texty[:5]
    return hint_cols

def row_to_card(row: pd.Series, cols: List[str], normalize_sql_flag: bool, mask_flag: bool) -> str:
    parts = []
    for c in cols:
        v = row.get(c, "")
        if pd.isna(v): 
            v = ""
        v_str = str(v)
        sql_mode = True if re.search(r"(query|sql|stmt|where|select|from)", c, re.I) else False
        v_str = normalize_generic(v_str, sql_mode=sql_mode and normalize_sql_flag, do_mask=mask_flag)
        if v_str:
            parts.append(f"{c}:{v_str}")
    return " ".join(parts).strip()


# -----------------------------
# Embedding & Indexing
# -----------------------------
def build_embeddings(cards: List[str], model_name: str, use_tfidf: bool=False) -> Tuple[np.ndarray, dict]:
    """
    returns: (embeddings[n, d], meta)
    meta = {"type": "st" or "tfidf", "model": model_name, "vectorizer": Optional[sklearn object]}
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

    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    Xc = vec_char.fit_transform(cards)
    vec_word = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xw = vec_word.fit_transform(cards)
    Xs = hstack([Xc, Xw]).astype("float32")

    # L2 regularization
    from sklearn.preprocessing import normalize
    Xs = normalize(Xs, norm="l2", copy=False)
    meta = {"type": "tfidf", "vectorizer_char": vec_char, "vectorizer_word": vec_word}
    X_dense = Xs.toarray().astype("float32")
    return X_dense, meta

def build_faiss_index(embeds: np.ndarray):
    # Cosine similarity = inner product(IP) + vector normalization assumed

    d = embeds.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product
    index.add(embeds)
    return index


# -----------------------------
# Pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="./0-trace_log_data.csv", help="Input CSV file path")
    ap.add_argument("--sep", default=",", help="CSV Separator")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer Model name")
    ap.add_argument("--index-out", default="./data/logvector_s.index", help="FAISS Index Output path")
    ap.add_argument("--chunks-out", default="./data/logvector_s.pkl", help="Pickle Output path")
    ap.add_argument("--cards-out", default="./data/logvector_cards_s.csv", help="CARD text CSV output path")
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

    # 2) Schema hint column determination
    cols = detect_schema_columns(df, args.schema_hint)
    if not cols:
        raise ValueError("Could not find column for card generation. Please specify with --schema-hint..")
    print(f"[INFO] Card Configuration Column: {cols}")

    # 3) Generate CARD text
    cards: List[str] = []
    ids: List[str] = []
    for i, row in df.iterrows():
        card = row_to_card(row, cols, normalize_sql_flag=args.normalize_sql, mask_flag=args.mask_pii)
        cards.append(card)
        if args.id_col and args.id_col in df.columns:
            ids.append(str(row[args.id_col]))
        else:
            ids.append(str(i))

    # 4) Embedding
    embeds, meta = build_embeddings(cards, model_name=args.model, use_tfidf=args.use_tfidf)
    print(f"[INFO] embeddings: shape={embeds.shape}, type={meta.get('type')}")

    # 5) FAISS Index build
    index = build_faiss_index(embeds)

    # 6) Store
    faiss.write_index(index, args.index_out)
    with open(args.chunks_out, "wb") as f:
        pickle.dump({"cards": cards, "ids": ids, "schema_cols": cols, "embed_meta": meta}, f)

    # Card CSV (for review)
    out_df = pd.DataFrame({"id": ids, "card_text": cards})
    out_df.to_csv(args.cards_out, index=False, encoding="utf-8")

    print(f"[DONE] index → {args.index_out}")
    print(f"[DONE] pickle → {args.chunks_out}")
    print(f"[DONE] cards → {args.cards_out}")


if __name__ == "__main__":
    main()
