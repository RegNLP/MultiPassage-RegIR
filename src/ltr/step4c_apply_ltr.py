#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply trained LTR model to a feature CSV and write a TREC run.
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import List
import lightgbm as lgb

def read_feature_list(path: Path) -> List[str]:
    names = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if ":" in line: _, line = line.split(":", 1)
            names.append(line.strip())
    return names

def load_features_csv(path: Path, feature_names: List[str]):
    qids, docids, feats = [], [], []
    import csv
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            qids.append(row["qid"]); docids.append(row["docid"])
            feats.append([float(row[name]) for name in feature_names])
    return qids, docids, feats

def write_trec(out_path: Path, qids, docids, scores, tag: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for q, d, s in zip(qids, docids, scores):
        rows.setdefault(q, []).append((d, float(s)))
    with out_path.open("w", encoding="utf-8") as f:
        for q in sorted(rows):
            ranked = sorted(rows[q], key=lambda kv: (-kv[1], kv[0]))
            for rank, (docid, sc) in enumerate(ranked, start=1):
                f.write(f"{q} Q0 {docid} {rank} {sc:.6f} ltr\n")
    print(f"[OK] Wrote TREC run: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--feature-csv", required=True)
    ap.add_argument("--feature-list", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    feat_names = read_feature_list(Path(args.feature_list))
    qids, docids, X = load_features_csv(Path(args.feature_csv), feat_names)

    model = lgb.Booster(model_file=args.model)
    scores = model.predict(X)
    write_trec(Path(args.out), qids, docids, scores, tag="ltr")

if __name__ == "__main__":
    main()
