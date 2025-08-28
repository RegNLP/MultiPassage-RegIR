#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 4A: Feature extraction for LTR.

Inputs:
- BM25, Dense, and (optionally) RRF TREC runs for the SAME query set
- Graph (data/graph.gpickle)
- Doc length map (data/collections/bm25_jsonl/docids.tsv)

Outputs:
- CSV with per (qid, docid) row and features
- feature_list.txt listing feature columns for LTR training
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import networkx as nx

# -----------------------------
# TREC helpers
# -----------------------------

def read_trec(path: Path) -> Dict[str, List[Tuple[str, int, float]]]:
    """qid -> [(docid, rank, score), ...]"""
    per_q: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts[:6]
            try:
                r = int(rank)
                s = float(score)
            except Exception:
                continue
            per_q[qid].append((docid, r, s))
    # sort deterministically by rank then docid
    for qid, rows in per_q.items():
        rows.sort(key=lambda t: (t[1], t[0]))
    return per_q


def index_run(per_q: Dict[str, List[Tuple[str, int, float]]]) -> Dict[str, Dict[str, Tuple[int, float]]]:
    """qid -> docid -> (rank, score)"""
    out: Dict[str, Dict[str, Tuple[int, float]]] = defaultdict(dict)
    for qid, rows in per_q.items():
        for docid, r, s in rows:
            if docid not in out[qid]:  # keep first (best) rank
                out[qid][docid] = (r, s)
    return out


# -----------------------------
# Graph features
# -----------------------------

def load_graph(path: Path) -> nx.DiGraph:
    with path.open("rb") as f:
        return pickle.load(f)

def graph_basic_features(G: nx.DiGraph, docid: str) -> Dict[str, float]:
    """Compute lightweight per-passage graph features. Node is a passage uid (docid)."""
    if docid not in G:
        return {
            "deg_out": 0, "deg_in": 0, "deg_total": 0,
            "uses_term_cnt": 0, "mentions_cnt": 0,
            "cites_out": 0, "cited_by_in": 0,
            "parent_depth": 0,
        }
    deg_out = G.out_degree(docid)
    deg_in  = G.in_degree(docid)
    uses_term = 0
    mentions = 0
    cites_out = 0
    cited_by_in = 0
    for _, v, d in G.out_edges(docid, data=True):
        t = d.get("type")
        if t == "USES_TERM": uses_term += 1
        elif t == "MENTIONS": mentions += 1
        elif t == "CITES": cites_out += 1
    for u, _, d in G.in_edges(docid, data=True):
        t = d.get("type")
        if t == "CITED_BY": cited_by_in += 1

    # parent depth from dot-delimited PassageID stored on node
    node_passage_id = (G.nodes[docid].get("PassageID") or "")
    parent_depth = node_passage_id.count(".") if node_passage_id else 0

    return {
        "deg_out": float(deg_out),
        "deg_in": float(deg_in),
        "deg_total": float(deg_out + deg_in),
        "uses_term_cnt": float(uses_term),
        "mentions_cnt": float(mentions),
        "cites_out": float(cites_out),
        "cited_by_in": float(cited_by_in),
        "parent_depth": float(parent_depth),
    }


def load_doclen_map(path: Path) -> Dict[str, int]:
    """docids.tsv: 'docid<TAB>char_len' written by step2a."""
    m: Dict[str, int] = {}
    if not path.exists():
        return m
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2: continue
            did, ln = parts
            try:
                m[did] = int(ln)
            except Exception:
                pass
    return m


# -----------------------------
# Feature extraction
# -----------------------------

def per_query_minmax(ranks_or_scores: Dict[str, Tuple[int, float]], use_score: bool) -> Dict[str, float]:
    """Normalize ranks or scores to [0,1] per query. If all equal, return zeros."""
    vals = [ (s if use_score else r) for (r, s) in ranks_or_scores.values() ]
    if not vals:
        return {}
    vmin, vmax = min(vals), max(vals)
    rng = (vmax - vmin) if vmax > vmin else 1.0
    out = {}
    for docid, (r, s) in ranks_or_scores.items():
        v = s if use_score else r
        out[docid] = (v - vmin) / rng if vmax > vmin else 0.0
    return out


def extract_features(
    bm25_idx: Dict[str, Dict[str, Tuple[int, float]]],
    dense_idx: Dict[str, Dict[str, Tuple[int, float]]],
    rrf_idx: Dict[str, Dict[str, Tuple[int, float]]],
    G: nx.DiGraph,
    doclen: Dict[str, int],
    out_csv: Path,
    feature_list_out: Path,
    tag_bm25: str = "bm25",
    tag_dense: str = "dense",
    tag_rrf: str = "rrf",
    topk: int = 1000,
) -> None:

    features = [
        # ranks (lower is better) and scores
        f"{tag_bm25}_rank", f"{tag_bm25}_score",
        f"{tag_dense}_rank", f"{tag_dense}_score",
        f"{tag_rrf}_rank",  f"{tag_rrf}_score",
        # per-query minmax
        f"{tag_bm25}_rank_minmax", f"{tag_bm25}_score_minmax",
        f"{tag_dense}_rank_minmax", f"{tag_dense}_score_minmax",
        f"{tag_rrf}_rank_minmax",  f"{tag_rrf}_score_minmax",
        # graph
        "deg_out", "deg_in", "deg_total",
        "uses_term_cnt", "mentions_cnt",
        "cites_out", "cited_by_in",
        "parent_depth",
        # doc length
        "doc_char_len",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["qid", "docid"] + features)  # header

        all_qids = set(bm25_idx.keys()) | set(dense_idx.keys()) | set(rrf_idx.keys())
        for qid in sorted(all_qids):
            # union of candidate docs across runs
            docs = set()
            docs |= set(bm25_idx.get(qid, {}).keys())
            docs |= set(dense_idx.get(qid, {}).keys())
            docs |= set(rrf_idx.get(qid, {}).keys())

            # per-query minmax maps
            bm25_minmax_rank = per_query_minmax(bm25_idx.get(qid, {}), use_score=False)
            bm25_minmax_score = per_query_minmax(bm25_idx.get(qid, {}), use_score=True)
            dense_minmax_rank = per_query_minmax(dense_idx.get(qid, {}), use_score=False)
            dense_minmax_score = per_query_minmax(dense_idx.get(qid, {}), use_score=True)
            rrf_minmax_rank   = per_query_minmax(rrf_idx.get(qid, {}),   use_score=False)
            rrf_minmax_score  = per_query_minmax(rrf_idx.get(qid, {}),   use_score=True)

            for docid in docs:
                # base ranks/scores
                b_r, b_s = bm25_idx.get(qid, {}).get(docid, (topk+1, 0.0))
                d_r, d_s = dense_idx.get(qid, {}).get(docid, (topk+1, 0.0))
                r_r, r_s = rrf_idx.get(qid, {}).get(docid, (topk+1, 0.0))

                # graph feats
                g = graph_basic_features(G, docid)

                # doc length
                L = float(doclen.get(docid, 0))

                row = [
                    qid, docid,
                    b_r, b_s,
                    d_r, d_s,
                    r_r, r_s,
                    bm25_minmax_rank.get(docid, 0.0), bm25_minmax_score.get(docid, 0.0),
                    dense_minmax_rank.get(docid, 0.0), dense_minmax_score.get(docid, 0.0),
                    rrf_minmax_rank.get(docid, 0.0),  rrf_minmax_score.get(docid, 0.0),
                    g["deg_out"], g["deg_in"], g["deg_total"],
                    g["uses_term_cnt"], g["mentions_cnt"],
                    g["cites_out"], g["cited_by_in"],
                    g["parent_depth"],
                    L,
                ]
                w.writerow(row)

    # write feature list (for LTR tools)
    with feature_list_out.open("w", encoding="utf-8") as fl:
        for i, name in enumerate(features, start=1):
            fl.write(f"{i}:{name}\n")

    print(f"[OK] Features written: {out_csv}")
    print(f"[OK] Feature list written: {feature_list_out}")
    print(f"[INFO] Total features: {len(features)}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Extract features from run files and graph.")
    ap.add_argument("--bm25-run", required=True, help="Path to BM25 TREC run")
    ap.add_argument("--dense-run", required=True, help="Path to dense TREC run")
    ap.add_argument("--rrf-run",   required=True, help="Path to RRF TREC run")
    ap.add_argument("--graph", default="data/graph.gpickle", help="Path to graph.gpickle")
    ap.add_argument("--doclen-map", default="data/collections/bm25_jsonl/docids.tsv", help="docids.tsv from step2a")
    ap.add_argument("--out-csv", default="outputs/features/features_test.csv", help="Output CSV path")
    ap.add_argument("--feature-list", default="outputs/ltr/feature_list_ltr_demo.txt", help="Feature list path")
    ap.add_argument("--topk", type=int, default=1000, help="Default rank for missing docs")
    return ap.parse_args()


def main():
    args = parse_args()
    bm25 = index_run(read_trec(Path(args.bm25_run)))
    dense = index_run(read_trec(Path(args.dense_run)))
    rrf   = index_run(read_trec(Path(args.rrf_run)))
    G     = load_graph(Path(args.graph))
    dlen  = load_doclen_map(Path(args.doclen_map))
    out_csv = Path(args.out_csv)
    feat_list = Path(args.feature_list)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feat_list.parent.mkdir(parents=True, exist_ok=True)

    extract_features(bm25, dense, rrf, G, dlen, out_csv, feat_list, topk=args.topk)


if __name__ == "__main__":
    main()
