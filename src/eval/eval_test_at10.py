#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate all *_test.trec runs at @10 only (Recall@10, MAP@10, nDCG@10).

Examples
--------
# All runs
python -m src.eval.eval_test_at10

# Only reranked
python -m src.eval.eval_test_at10 --pattern 'sr_*_test.trec'

# Only hybrids
python -m src.eval.eval_test_at10 --pattern 'bm25+*_test.trec'
"""

import argparse
import glob
import gzip
import io
import json
from pathlib import Path
import pytrec_eval

def _open_any(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open("r", encoding="utf-8")

def build_qrels(qa_path: Path):
    """
    Build qrels dict {qid: {docid: rel}} using passage UIDs from the fixed JSON.
    Prefer p['ID'] then fallback to p['ContextID'].
    All listed passages are treated as relevant (rel=1).
    """
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    qrels, missing = {}, 0
    for item in data:
        qid = str(item["QuestionID"])
        qrels[qid] = {}
        for p in item.get("Passages", []):
            uid = p.get("ID") or p.get("ContextID") or ""
            if uid:
                qrels[qid][str(uid)] = 1
            else:
                missing += 1
    if missing:
        print(f"[WARN] build_qrels: {missing} gold passages with missing IDs (ignored)")
    return qrels

def read_run(path: Path):
    """Read a TREC run into {qid: {docid: score}}, keep highest score if duplicates."""
    run = {}
    with _open_any(path) as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, score, _ = parts[:6]
            try:
                score = float(score)
            except ValueError:
                continue
            d = run.setdefault(qid, {})
            if docid not in d or score > d[docid]:
                d[docid] = score
    return run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-test",
                    default="outputs/QADataset_fixed/ObliQA_MultiPassage_test.json",
                    help="Path to *fixed* test QA JSON")
    ap.add_argument("--runs-dir", default="outputs/runs",
                    help="Folder with run files")
    ap.add_argument("--pattern", default="*_test.trec",
                    help="Glob pattern to match runs (e.g., 'sr_*_test.trec')")
    ap.add_argument("--out-tsv", default="outputs/eval/test_at10_summary.tsv",
                    help="Path to save TSV summary")
    args = ap.parse_args()

    qa_path = Path(args.qa_test)
    runs_dir = Path(args.runs_dir)
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    # qrels from fixed JSON so IDs match the runs
    qrels = build_qrels(qa_path)
    metric_keys = {"map_cut_10", "ndcg_cut_10", "recall_10"}  # underscores!
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metric_keys)

    run_paths = sorted(glob.glob(str(runs_dir / args.pattern)))
    if not run_paths:
        print(f"[WARN] No runs found under {runs_dir} matching pattern '{args.pattern}'")
        return

    rows = []
    for rp in run_paths:
        rp = Path(rp)
        run_name = rp.name.replace("_test.trec", "").replace(".gz", "")
        run = read_run(rp)
        res = evaluator.evaluate(run)
        if not res:
            print(f"[SKIP] {run_name}: no overlapping qids/docids with qrels ({len(run)} scored docs total)")
            continue
        n = len(res)
        agg = {
            "Recall@10": sum(v["recall_10"]   for v in res.values()) / n,
            "MAP@10":    sum(v["map_cut_10"]  for v in res.values()) / n,
            "nDCG@10":   sum(v["ndcg_cut_10"] for v in res.values()) / n,
        }
        rows.append((run_name, agg["Recall@10"], agg["MAP@10"], agg["nDCG@10"], n))

    rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))

    total_q = len(qrels)
    print(f"\nAll test runs (evaluated on {total_q} queries):\n")
    header = f"{'Method':40s}  {'Recall@10':>9s}  {'MAP@10':>7s}  {'nDCG@10':>8s}"
    print(header)
    print("-" * len(header))
    for name, r10, map10, ndcg10, n in rows:
        print(f"{name:40s}  {r10:9.4f}  {map10:7.4f}  {ndcg10:8.4f}")

    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("Method\tRecall@10\tMAP@10\tnDCG@10\tEvaluatedQueries\n")
        for name, r10, map10, ndcg10, n in rows:
            f.write(f"{name}\t{r10:.6f}\t{map10:.6f}\t{ndcg10:.6f}\t{n}\n")
    print(f"\n[OK] Wrote summary TSV -> {out_tsv}")

if __name__ == "__main__":
    main()
