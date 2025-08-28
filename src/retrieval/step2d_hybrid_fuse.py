#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import defaultdict
import math

def read_trec_run(path: Path):
    scores = defaultdict(dict)
    ranks = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts[:6]
            score = float(score)
            scores[qid][docid] = score
    for qid, ds in scores.items():
        sorted_docs = sorted(ds.items(), key=lambda x: (-x[1], x[0]))
        for i, (docid, _) in enumerate(sorted_docs, start=1):
            ranks[qid][docid] = i
    return scores, ranks

def trec_write(path: Path, fused, tag="hybrid"):
    with path.open("w", encoding="utf-8") as out:
        for qid in sorted(fused.keys()):
            ranked = sorted(fused[qid].items(), key=lambda x: (-x[1], x[0]))
            for rank, (docid, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")

def per_query_norm(scores, method="zscore"):
    normed = {}
    for qid, ds in scores.items():
        vals = list(ds.values())
        if not vals:
            normed[qid] = {}
            continue
        if method == "zscore":
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
            std = math.sqrt(var) if var > 0 else 1.0
            normed[qid] = {d: (v - mu) / std for d, v in ds.items()}
        elif method == "minmax":
            vmin, vmax = min(vals), max(vals)
            rng = (vmax - vmin) if vmax > vmin else 1.0
            normed[qid] = {d: (v - vmin) / rng for d, v in ds.items()}
        else:
            raise ValueError(f"Unknown norm method: {method}")
    return normed

def weighted_fusion(bm25_scores, dense_scores, alpha=0.5, norm="zscore", k=None):
    bm25_n = per_query_norm(bm25_scores, method=norm)
    dense_n = per_query_norm(dense_scores, method=norm)
    fused = defaultdict(dict)
    all_qids = set(bm25_n.keys()) | set(dense_n.keys())
    for qid in all_qids:
        docs = set(bm25_n.get(qid, {}).keys()) | set(dense_n.get(qid, {}).keys())
        for d in docs:
            s_b = bm25_n.get(qid, {}).get(d, 0.0)
            s_d = dense_n.get(qid, {}).get(d, 0.0)
            fused[qid][d] = (1 - alpha) * s_b + alpha * s_d
        if k:
            ranked = sorted(fused[qid].items(), key=lambda x: (-x[1], x[0]))[:k]
            fused[qid] = dict(ranked)
    return fused

def rrf_fusion(bm25_ranks, dense_ranks, k=1000, c=60):
    fused = defaultdict(dict)
    all_qids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    for qid in all_qids:
        docs = set(bm25_ranks.get(qid, {}).keys()) | set(dense_ranks.get(qid, {}).keys())
        for d in docs:
            s = 0.0
            if d in bm25_ranks.get(qid, {}):
                s += 1.0 / (c + bm25_ranks[qid][d])
            if d in dense_ranks.get(qid, {}):
                s += 1.0 / (c + dense_ranks[qid][d])
            fused[qid][d] = s
        ranked = sorted(fused[qid].items(), key=lambda x: (-x[1], x[0]))[:k]
        fused[qid] = dict(ranked)
    return fused

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--bm25-name", required=True)   # <-- underscore
    ap.add_argument("--dense-name", required=True)  # <-- underscore
    ap.add_argument("--out-dir", default="outputs/runs")
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--method", choices=["weighted", "rrf"], default="weighted")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--norm", choices=["zscore", "minmax"], default="zscore")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--rrf-c", type=int, default=60)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        bm25_path = runs_dir / f"{args.bm25_name}_{split}.trec"
        dense_path = runs_dir / f"{args.dense_name}_{split}.trec"
        if not bm25_path.exists() or not dense_path.exists():
            miss = []
            if not bm25_path.exists(): miss.append(bm25_path.name)
            if not dense_path.exists(): miss.append(dense_path.name)
            print(f"[SKIP] {split}: missing {', '.join(miss)}")
            continue

        bm25_scores, bm25_ranks = read_trec_run(bm25_path)
        dense_scores, dense_ranks = read_trec_run(dense_path)

        if args.method == "weighted":
            fused = weighted_fusion(bm25_scores, dense_scores, alpha=args.alpha, norm=args.norm, k=args.k)
            out = out_dir / f"{args.bm25_name}+{args.dense_name}_{split}.trec"
        else:
            fused = rrf_fusion(bm25_ranks, dense_ranks, k=args.k, c=args.rrf_c)
            out = out_dir / f"rrf_{args.bm25_name}+{args.dense_name}_{split}.trec"

        trec_write(out, fused, tag=args.tag)
        print(f"[OK] Wrote fused run: {out}")

    print("Done.")

if __name__ == "__main__":
    main()
