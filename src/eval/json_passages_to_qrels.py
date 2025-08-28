#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create TREC qrels from ObliQA-MP JSON where *all* Passages are positives.

Each example must have:
- "QuestionID": str
- "Passages": list of passage dicts with an "ID" (the same IDs used in docs.jsonl)

Output format (per line):
qid 0 docid 1
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", required=True, help="Path to split JSON (train/val/test)")
    ap.add_argument("--out-qrels", required=True, help="Path to write qrels")
    return ap.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.split_json)
    out_path = Path(args.out_qrels)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.load(open(in_path, "r", encoding="utf-8"))
    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ex in data:
            qid = str(ex.get("QuestionID", "")).strip()
            passages = ex.get("Passages") or []
            for p in passages:
                # ID is what we exported to docs.jsonl (via graph) → guaranteed by step0_fix_json_ids
                did = str(p.get("ID") or p.get("ContextID") or "").strip()
                if not qid or not did:
                    continue
                out.write(f"{qid} 0 {did} 1\n")
                n_rows += 1
    print(f"[OK] Wrote {n_rows} qrels rows to {out_path}")

if __name__ == "__main__":
    main()
