#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 2B: Retrieve BM25 candidates from the Lucene index (Pyserini)

Usage
-----
python -m src.retrieval.step2b_retrieve_bm25 \
  --queries-json outputs/QADataset_fixed/ObliQA_MultiPassage_test.json \
  --index-dir outputs/indexes/bm25 \
  --out outputs/runs/bm25_test.trec

The input JSON must be a list of objects containing at least:
  - "Question" (text)
  - "QuestionID" (unique id)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pyserini.search.lucene import LuceneSearcher


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BM25 retrieval with Pyserini (outputs TREC run).")
    ap.add_argument("--queries-json", required=True, help="Path to split JSON with questions.")
    ap.add_argument("--index-dir", required=True, help="Path to BM25 (Lucene) index.")
    ap.add_argument("--out", required=True, help="Path to write TREC run file.")
    ap.add_argument("--hits", type=int, default=1000, help="Top-k hits per query.")
    ap.add_argument("--bm25-k1", type=float, default=0.9, help="BM25 k1.")
    ap.add_argument("--bm25-b", type=float, default=0.4, help="BM25 b.")
    ap.add_argument("--remove-dup-docs", action="store_true",
                    help="If a doc id appears multiple times for a query (shouldn't), keep first.")
    ap.add_argument("--tag", default="bm25", help="Run tag for the TREC file (column 6).")
    return ap.parse_args()


# ------------------------
# I/O Helpers
# ------------------------

def load_queries(path: Path) -> List[Tuple[str, str]]:
    """Return list of (qid, query_text)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} should be a list of examples")
    out: List[Tuple[str, str]] = []
    seen = set()
    for ex in data:
        q = (ex.get("Question") or "").strip()
        qid = str(ex.get("QuestionID") or "").strip()
        if not q or not qid:
            continue
        if qid in seen:
            # deterministic dedup
            continue
        seen.add(qid)
        out.append((qid, q))
    if not out:
        raise ValueError(f"No (QuestionID, Question) pairs found in {path}")
    return out


def write_trec(run_path: Path, rows: Iterable[Tuple[str, str, int, float, str]]) -> None:
    """Write lines: qid Q0 docid rank score tag"""
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("w", encoding="utf-8") as f:
        for qid, docid, rank, score, tag in rows:
            f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")


# ------------------------
# Retrieval
# ------------------------

def main() -> None:
    args = parse_args()
    queries_path = Path(args.queries_json)
    index_dir = Path(args.index_dir)
    out_path = Path(args.out)

    queries = load_queries(queries_path)
    print(f"[INFO] Loaded {len(queries)} queries from {queries_path}")

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    print(f"[INFO] BM25 params: k1={args.bm25_k1} b={args.bm25_b}")

    rows = []
    for qid, qtext in queries:
        hits = searcher.search(qtext, k=args.hits)
        seen_docs = set()
        rank = 1
        for h in hits:
            docid = h.docid
            if args.remove_dup_docs:
                if docid in seen_docs:
                    continue
                seen_docs.add(docid)
            rows.append((qid, docid, rank, float(h.score), args.tag))
            rank += 1

    write_trec(out_path, rows)
    print(f"[OK] Wrote TREC run: {out_path}  ({len(rows)} lines)")


if __name__ == "__main__":
    main()
