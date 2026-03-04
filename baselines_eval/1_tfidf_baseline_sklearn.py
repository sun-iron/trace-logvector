import argparse
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# -----------------------------
# Text Normalization/Masking  
# -----------------------------
SQL_QUOTE_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
WS_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-]{7,}\d")


def mask_pii(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = EMAIL_RE.sub("email:***@***", s)
    s = PHONE_RE.sub("phone:***", s)
    return s


def normalize_sql(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = SQL_QUOTE_RE.sub("", s)
    s = NUM_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def normalize_generic(s: str, sql_mode: bool = False, do_mask: bool = False) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if do_mask:
        s = mask_pii(s)
    if sql_mode:
        s = normalize_sql(s)
    else:
        s = SQL_QUOTE_RE.sub("", s)
        s = NUM_RE.sub("", s)
        s = WS_RE.sub(" ", s).strip()
    return s


# -----------------------------
# Evaluation datasets 
# - Multi-CARD: Using 5 query defined in eval_logvector_rag_multi.py
# - Single-CARD: Using 5 query defined in eval_logvector_rag_single.py
# -----------------------------
EVAL_MULTI: List[Dict] = [
    {
        "query": "Which functions access the wp_usermeta table?",
        "expected_ids": ["table:wp_usermeta"],
        "description": "Table-centric search (single answer)",
    },
    {
        "query": "Which table(s) does the function WP_Post::get_instance() access?",
        "expected_ids": ["Call_Func:WP_Post::get_instance()"],
        "description": "Function-centric search (single answer)",
    },
    {
        "query": "Which table(s) are used by the function AutomatticWAARTSDataStore::get_taxes()?",
        "expected_ids": [
            "Call_Func:AutomatticWAARTSDataStore::get_taxes()",
            "table:wp_woocommerce_tax_rates",
        ],
        "description": "Bidirectional function–table search (multiple answers)",
    },
    {
        "query": "List all functions related to the wp_wc_admin_notes table.",
        "expected_ids": ["table:wp_wc_admin_notes", "Call_Func:AutomatticWANDataStore->read()"],
        "description": "Bidirectional table–function search (multiple answers)",
    },
    {
        "query": "What is the data-flow information related to the WooCommerce admin note actions (wp_wc_admin_note_actions)?",
        "expected_ids": ["table:wp_wc_admin_note_actions", "Call_Func:AutomatticWANDataStore->read_actions()"],
        "description": "Keyword-based relational search (multiple answers)",
    },
]

EVAL_SINGLE: List[Dict] = [
    {"query": EVAL_MULTI[0]["query"], "expected_ids": ["0"], "description": "Single-chunk (row)"},
    {"query": EVAL_MULTI[1]["query"], "expected_ids": ["1"], "description": "Single-chunk (row)"},
    {"query": EVAL_MULTI[2]["query"], "expected_ids": ["2"], "description": "Single-chunk (row)"},
    {"query": EVAL_MULTI[3]["query"], "expected_ids": ["3", "5"], "description": "Single-chunk (row, multi)"},
    {"query": EVAL_MULTI[4]["query"], "expected_ids": ["4", "6"], "description": "Single-chunk (row, multi)"},
]


def load_cards_csv(path: str) -> Tuple[List[str], List[str]]:
    """
    CSV columns: id, card_text (레포 data/logvector_cards_{s,m}.csv 형태)
    """
    df = pd.read_csv(path)
    if "id" not in df.columns or "card_text" not in df.columns:
        raise ValueError(f"CSV must have columns ['id','card_text']. Got: {df.columns.tolist()}")
    ids = df["id"].astype(str).tolist()
    cards = df["card_text"].fillna("").astype(str).tolist()
    return cards, ids


def build_tfidf_matrix(cards: List[str]) -> Tuple[csr_matrix, Dict]:
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xc = vec_char.fit_transform(cards)
    vec_word = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xw = vec_word.fit_transform(cards)

    X = hstack([Xc, Xw]).astype("float32")
    X = normalize(X, norm="l2", copy=False)

    meta = {"type": "tfidf", "vectorizer_char": vec_char, "vectorizer_word": vec_word}
    return X, meta


def encode_queries(queries: List[str], meta: Dict, normalize_sql_flag: bool, mask_flag: bool) -> csr_matrix:
    vec_char: TfidfVectorizer = meta["vectorizer_char"]
    vec_word: TfidfVectorizer = meta["vectorizer_word"]

    processed = [normalize_generic(q, sql_mode=normalize_sql_flag, do_mask=mask_flag) for q in queries]

    Qc = vec_char.transform(processed)
    Qw = vec_word.transform(processed)
    Q = hstack([Qc, Qw]).astype("float32")
    Q = normalize(Q, norm="l2", copy=False)
    return Q


def topk_retrieve(Q: csr_matrix, X: csr_matrix, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine Similarity
    sims = Q @ X.T
    """
    sims = (Q @ X.T).toarray()  # (nq, nd)
    I = np.argsort(-sims, axis=1)[:, :topk]
    D = np.take_along_axis(sims, I, axis=1)
    return D, I


def evaluate_hit_mrr(I: np.ndarray, ids: List[str], expected_ids: List[List[str]], k: int) -> Tuple[float, float]:
    hit = 0
    rr = []

    for qi in range(len(expected_ids)):
        expected = set(expected_ids[qi])
        retrieved = [ids[idx] for idx in I[qi, :k] if idx < len(ids)]

        if any(r in expected for r in retrieved):
            hit += 1

        rank = 0
        for j, r in enumerate(retrieved):
            if r in expected:
                rank = j + 1
                break
        rr.append(1.0 / rank if rank > 0 else 0.0)

    return hit / len(expected_ids), float(np.mean(rr))


def run_case(cards_csv: str, eval_dataset: List[Dict], topk: int, normalize_sql_flag: bool, mask_flag: bool) -> None:
    cards, ids = load_cards_csv(cards_csv)
    X, meta = build_tfidf_matrix(cards)

    queries = [d["query"] for d in eval_dataset]
    expected = [d["expected_ids"] for d in eval_dataset]

    Q = encode_queries(queries, meta, normalize_sql_flag, mask_flag)
    _, I = topk_retrieve(Q, X, topk=topk)

    hit, mrr = evaluate_hit_mrr(I, ids, expected, k=topk)
    print("=" * 72)
    print(f"[TF-IDF Baseline] cards={cards_csv}")
    print(f"Top-K = {topk}")
    print(f"Hit@{topk} = {hit:.4f}")
    print(f"MRR@{topk} = {mrr:.4f}")
    print("=" * 72)

    # Print out Top-1
    for qi, d in enumerate(eval_dataset):
        top1_id = ids[I[qi, 0]]
        print(f"- Q{qi+1} {d['description']}: {d['query']}")
        print(f"  expected={d['expected_ids']}")
        print(f"  top1={top1_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards-single", default="./dataset/logvector_cards_s.csv", help="Single-CARD CSV path")
    ap.add_argument("--cards-multi", default="./dataset/logvector_cards_m.csv", help="Multi-CARD CSV path")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--normalize-sql", action="store_true")
    ap.add_argument("--mask-pii", action="store_true")
    ap.add_argument("--mode", choices=["single", "multi", "both"], default="both")
    args = ap.parse_args()

    if args.mode in ("single", "both"):
        run_case(args.cards_single, EVAL_SINGLE, args.top_k, args.normalize_sql, args.mask_pii)

    if args.mode in ("multi", "both"):
        run_case(args.cards_multi, EVAL_MULTI, args.top_k, args.normalize_sql, args.mask_pii)


if __name__ == "__main__":

    main()
