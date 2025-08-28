#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 4B: Train an LTR model (LightGBM LambdaMART) from feature CSVs + qrels.

Inputs
------
--train-csv       CSV from step4a with header: qid,docid,<features...>
--val-csv         (optional) same format; used for validation reporting
--qrels           TREC qrels for TRAIN split (supports "qid 0 docid rel" or "qid docid rel")
--val-qrels       (optional) TREC qrels for VAL split; if omitted, falls back to --qrels
--feature-list    Plain list of feature names (or indexed "1:name" per line)

Outputs
-------
--out-model          LightGBM text model (default: outputs/ltr/ltr_model.txt)
--out-feature-list   The exact feature list used during training

Notes
-----
- Feature CSV rows must be grouped by qid (contiguous). step4a writes them grouped.
- Labels are aligned from qrels; missing (qid,docid) pairs default to 0.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import lightgbm as lgb


# -----------------------------
# I/O Helpers
# -----------------------------

def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """Read qrels; accept 4-col 'qid 0 docid rel' or 3-col 'qid docid rel' and skip malformed rows."""
    qrels: Dict[str, Dict[str, int]] = {}
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
            elif len(parts) == 3:
                qid, docid, rel = parts
            else:
                bad += 1
                continue
            try:
                reli = int(rel)
            except Exception:
                bad += 1
                continue
            qrels.setdefault(qid, {})[docid] = reli
    if bad:
        print(f"[WARN] Skipped {bad} malformed qrels rows in {path}")
    return qrels


def read_feature_list(path: Path) -> List[str]:
    """Allow either '1:name' lines or plain 'name' lines."""
    names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            if ":" in t:
                _, t = t.split(":", 1)
            names.append(t.strip())
    if not names:
        raise ValueError(f"No features found in {path}")
    return names


def load_features_csv(path: Path, feature_names: List[str]) -> Tuple[List[str], List[str], np.ndarray]:
    """Return (qids, docids, X[float32]) for the requested feature columns."""
    qids: List[str] = []
    docids: List[str] = []
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        missing = [n for n in feature_names if n not in rdr.fieldnames]
        if missing:
            raise ValueError(f"Missing feature columns in {path.name}: {missing}")
        if "qid" not in rdr.fieldnames or "docid" not in rdr.fieldnames:
            raise ValueError(f"{path.name} must have 'qid' and 'docid' columns")
        for row in rdr:
            qids.append(row["qid"])
            docids.append(row["docid"])
            rows.append([float(row[n]) for n in feature_names])
    X = np.asarray(rows, dtype=np.float32)
    return qids, docids, X


def make_group(qids: List[str]) -> List[int]:
    """LightGBM expects group sizes per query; requires qids contiguous."""
    if not qids:
        return []
    group: List[int] = []
    last = qids[0]
    cnt = 0
    for q in qids:
        if q == last:
            cnt += 1
        else:
            group.append(cnt)
            last = q
            cnt = 1
    group.append(cnt)
    return group


def check_contiguous_groups(qids: List[str]) -> bool:
    """Return True iff each qid appears in a single contiguous block."""
    first_seen: Dict[str, int] = {}
    last_seen: Dict[str, int] = {}
    for i, q in enumerate(qids):
        if q not in first_seen:
            first_seen[q] = i
        last_seen[q] = i
    # If any qid appears, its indices must form a single block
    for q in first_seen:
        start, end = first_seen[q], last_seen[q]
        # Check no other qid interleaves within [start, end]
        if any(qids[i] != q for i in range(start, end + 1)):
            return False
    return True


def align_labels(qids: List[str], docids: List[str], qrels: Dict[str, Dict[str, int]]) -> np.ndarray:
    """Map (qid, docid) pairs to integer relevance using qrels; default 0."""
    labels = [int(qrels.get(q, {}).get(d, 0)) for q, d in zip(qids, docids)]
    return np.asarray(labels, dtype=np.int32)


# -----------------------------
# Simple eval (sanity)
# -----------------------------

def eval_map_at_k(y_true: np.ndarray, y_pred: np.ndarray, qids: List[str], k: int = 10) -> float:
    """Quick MAP@k for sanity (not pytrec_eval). Assumes qids are contiguous."""
    assert y_true.shape[0] == y_pred.shape[0] == len(qids)
    n = len(qids)
    if n == 0:
        return 0.0
    # compute group sizes
    group = make_group(qids)
    idx = 0
    aps: List[float] = []
    for g in group:
        scores = y_pred[idx:idx+g]
        labels = y_true[idx:idx+g]
        order = np.argsort(-scores)
        hits = 0
        ap = 0.0
        denom = int((labels > 0).sum())
        if denom == 0:
            aps.append(0.0)
        else:
            for rank, r in enumerate(order[:k], start=1):
                if labels[r] > 0:
                    hits += 1
                    ap += hits / rank
            aps.append(ap / denom)
        idx += g
    return float(np.mean(aps)) if aps else 0.0


# -----------------------------
# CLI / Training
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train LambdaMART (LightGBM) for regulatory RAG LTR.")
    ap.add_argument("--train-csv", required=True, help="features_train.csv")
    ap.add_argument("--val-csv", required=False, help="features_val.csv (optional)")
    ap.add_argument("--qrels", required=True, help="TRAIN qrels")
    ap.add_argument("--val-qrels", required=False, help="VAL qrels (optional; falls back to TRAIN qrels)")
    ap.add_argument("--feature-list", required=True, help="feature_list.txt")
    ap.add_argument("--out-model", default="outputs/ltr/ltr_model.txt")
    ap.add_argument("--out-feature-list", default="outputs/ltr/feature_list_used.txt")
    # model params (sane defaults; tweak in README if needed)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--lr", type=float, default=0.07)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--n-estimators", type=int, default=500)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_feature_list).parent.mkdir(parents=True, exist_ok=True)

    # Load supervision & feature config
    train_qrels = read_qrels(Path(args.qrels))
    val_qrels   = read_qrels(Path(args.val_qrels)) if args.val_qrels else train_qrels
    feat_names  = read_feature_list(Path(args.feature_list))
    print(f"[INFO] Loaded TRAIN qrels for {len(train_qrels)} queries")
    if args.val_qrels:
        print(f"[INFO] Loaded VAL   qrels for {len(val_qrels)} queries")
    print(f"[INFO] Using {len(feat_names)} features")

    # Load train
    tr_qids, tr_docids, tr_X = load_features_csv(Path(args.train_csv), feat_names)
    if not check_contiguous_groups(tr_qids):
        print("[WARN] Train qids are not contiguous; ensure features are grouped by qid.")
    tr_labels = align_labels(tr_qids, tr_docids, train_qrels)
    tr_group  = make_group(tr_qids)
    print(f"[INFO] Train: X={tr_X.shape}, labels={tr_labels.shape}, groups={len(tr_group)}")

    # Optional validation
    valid_sets = []
    valid_names = []
    va_qids = va_X = va_labels = None
    if args.val_csv:
        va_qids, va_docids, va_X = load_features_csv(Path(args.val_csv), feat_names)
        if not check_contiguous_groups(va_qids):
            print("[WARN] Val qids are not contiguous; ensure features are grouped by qid.")
        va_labels = align_labels(va_qids, va_docids, val_qrels)
        va_group  = make_group(va_qids)
        print(f"[INFO] Val:   X={va_X.shape}, labels={va_labels.shape}, groups={len(va_group)}")
        valid_sets.append(lgb.Dataset(
            va_X, label=va_labels, group=va_group, free_raw_data=False, feature_name=feat_names
        ))
        valid_names.append("valid")

    # LightGBM params
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        boosting="gbdt",
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        verbosity=-1,
    )

    # Train
    model = lgb.train(
        params,
        lgb.Dataset(tr_X, label=tr_labels, group=tr_group, free_raw_data=False, feature_name=feat_names),
        num_boost_round=args.n_estimators,
        valid_sets=valid_sets if valid_sets else [lgb.Dataset(tr_X, label=tr_labels, group=tr_group, free_raw_data=False, feature_name=feat_names)],
        valid_names=valid_names if valid_names else ["train"],
        keep_training_booster=True,
    )

    # Save artifacts
    model.save_model(args.out_model)
    with open(args.out_feature_list, "w", encoding="utf-8") as f:
        for i, n in enumerate(feat_names, 1):
            f.write(f"{i}:{n}\n")
    print(f"[OK] Model saved to {args.out_model}")
    print(f"[OK] Feature list saved to {args.out_feature_list}")

    # Quick sanity metrics
    tr_pred = model.predict(tr_X)
    tr_map10 = eval_map_at_k(tr_labels, tr_pred, tr_qids, k=10)
    print(f"[TRAIN] MAP@10 ≈ {tr_map10:.4f}")
    if args.val_csv:
        va_pred = model.predict(va_X)
        va_map10 = eval_map_at_k(va_labels, va_pred, va_qids, k=10)
        print(f"[VALID] MAP@10 ≈ {va_map10:.4f}")


if __name__ == "__main__":
    main()
