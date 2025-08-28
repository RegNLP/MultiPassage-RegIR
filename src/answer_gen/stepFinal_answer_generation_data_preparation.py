#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stepFinal_answer_generation_data_preparation.py

Purpose
-------
Prepare JSON inputs for answer generation by:
- Reading only WHITELISTED .trec runs from outputs/runs/
- Inferring split from filename suffix (*_test.trec / *_val.trec / *_train.trec)
- Loading the regulatory graph (pickled NetworkX graph) and building a robust PID->text index from Passage nodes
- Reading the matching ObliQA-MP split (data/QADataset/*.json)
- Resolving retrieved PIDs to passage text, filling up to TOP_K by scanning deeper ranks
- Writing two JSON files per run under outputs/answers/:
    1) <run_basename>.json              (unfiltered; current behavior)
    2) <run_basename>_filtered.json     (NEW: score-based filtered variant)

New Filtering Method
--------------------
For each query, we apply per-query score min-max normalization to [0,1] and then:
- Enforce a minimum normalized score >= FILTER_MIN_NORM_SCORE (default 0.7)
- Early-stop when a consecutive drop >= FILTER_DROP_THRESHOLD (default 0.2) is detected
- Always keep at least top-1 item to avoid empty lists

Run
---
    python src/stepFinal_answer_generation_data_preparation.py
"""

import os
import re
import glob
import json
import pickle
from collections import defaultdict, OrderedDict
from typing import Dict

# ---------------- Paths & Config ----------------
RUNS_DIR = os.path.join("outputs", "runs")
OUTPUT_DIR = os.path.join("outputs", "answers")
DATASET_DIR = os.path.join("data", "QADataset")
SPLIT_FILE = {
    "train": "ObliQA_MultiPassage_train.json",
    "val":   "ObliQA_MultiPassage_val.json",
    "test":  "ObliQA_MultiPassage_test.json",
}

# Strict graph locations (pickled NetworkX graph)
GRAPH_CANDIDATES = [
    "graph.gpickle"
]

TOP_K = 10
ENCODING = "utf-8"

# >>>>>>>>>>> EDIT THIS: whitelist of exact .trec filenames to process
WHITELIST = [
    "bm25_test.trec",  # BM25 (sparse baseline)
    "dense_FT_BGE_test.trec",  # Best Dense retriever: FT_BGE
    "rrf_bm25+dense_FT_BGE_test.trec",  # Best simple fusion: RRF(BM25 + FT_BGE)
    "bm25+dense_FT_E5_a0.5_test.trec",  # Best weighted hybrid: BM25 + FT_E5 (α = 0.5)
    "bm25+dense_FT_MSMARCO_a0.5+sr_bge_base_en_v1.5_test.trec",  # Hybrid + Secondary Signals (SR)
    "ltr_lgbm_allfeat_test.trec"  # My method: LTR (LGBM, all features)
]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ---------------- Filtering config (NEW) ----------------
FILTERING_ENABLED = True
FILTER_MIN_NORM_SCORE = 0.7      # keep items with normalized score >= 0.7
FILTER_DROP_THRESHOLD = 0.2      # stop when consecutive drop >= 0.2
FILTER_NORMALIZE = True          # per-query min-max to [0,1]

# ---------------- Regex helpers ----------------
INVISIBLE_CHARS_RE = re.compile(r"[\u200e\u200f\u202a-\u202e]")
SPLIT_SUFFIX_RE = re.compile(r"_(test|val|train)\.trec$", re.IGNORECASE)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = INVISIBLE_CHARS_RE.sub("", s)
    return s.strip()


# ---------------- Graph loading & PID index ----------------
def load_graph_any():
    """
    Loads the saved regulatory graph strictly from GRAPH_CANDIDATES via pickle.load.
    """
    for cand in GRAPH_CANDIDATES:
        if os.path.isfile(cand):
            try:
                with open(cand, "rb") as f:
                    G = pickle.load(f)
                print(f"[INFO] Graph loaded (pickle) from: {cand}")
                return G
            except Exception as e:
                print(f"[WARN] Failed to load graph at '{cand}': {e}")
    raise FileNotFoundError(f"Could not find/load a graph pickle. Tried: {GRAPH_CANDIDATES}")


def build_pid_index_from_graph(G) -> Dict[str, str]:
    """
    Build pid->text map from Passage nodes in the graph.

    Expected Passage node attributes:
      - node id = uid (stable unique ID)
      - attributes: ID (often same as uid), PassageID (e.g., "6.1.2"), DocumentID, text
    We index:
      - uid (node id) and attrs['ID'] if present
      - combos of (DocumentID, PassageID): {did_pid, Ddid_pid, did::pid, ...}
    """
    pid2text: Dict[str, str] = {}

    def add_key(k: str, text: str):
        if not k:
            return
        if k not in pid2text:
            pid2text[k] = text
        kn = re.sub(r"\s+", "", k)
        if kn and kn not in pid2text:
            pid2text[kn] = text

    added_uid = 0
    added_combo = 0

    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") != "Passage":
            continue
        text = clean_text(attrs.get("text", "") or "")
        if not text:
            continue

        # 1) Passage UID keys
        add_key(str(node_id), text)
        added_uid += 1
        if "ID" in attrs and attrs["ID"]:
            add_key(str(attrs["ID"]), text)

        # 2) Combos from DocumentID + PassageID (fallbacks)
        did = attrs.get("DocumentID")
        pid = attrs.get("PassageID")
        if did is not None and pid:
            did_s = str(did).strip()
            pid_s = str(pid).strip()
            combos = {
                f"{did_s}_{pid_s}",
                f"D{did_s}_{pid_s}",
                f"{did_s}::{pid_s}",
                f"D{did_s}::{pid_s}",
                f"{did_s}-{pid_s}",
                f"D{did_s}-{pid_s}",
                pid_s,  # sometimes only PassageID is used
            }
            for c in combos:
                add_key(c, text)
                added_combo += 1

    print(f"[INFO] PID index from graph: {len(pid2text)} keys (UIDs added: {added_uid}, combos added: {added_combo})")
    return pid2text


# ---------------- Dataset loading (for questions) ----------------
def load_questions_for_split(split: str) -> Dict[str, str]:
    path = os.path.join(DATASET_DIR, SPLIT_FILE[split])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset split file not found: {path}")
    with open(path, "r", encoding=ENCODING) as f:
        data = json.load(f)
    qid2question = {}
    for item in data:
        qid = item.get("QuestionID")
        qtext = item.get("Question")
        if qid is not None and qtext:
            qid2question[str(qid)] = clean_text(qtext)
    print(f"[INFO] ({split.upper()}) Loaded questions: {len(qid2question)}")
    return qid2question


# ---------------- Run file discovery & split inference ----------------
def infer_split_from_filename(filename: str) -> str:
    m = SPLIT_SUFFIX_RE.search(filename)
    if not m:
        return "test"
    return m.group(1).lower()


def find_run_path(filename: str) -> str:
    cand = os.path.join(RUNS_DIR, filename)
    if os.path.isfile(cand):
        return cand
    for p in glob.glob(os.path.join(RUNS_DIR, "*.trec")):
        if os.path.basename(p) == filename:
            return p
    return ""


# ---------------- TREC parsing ----------------
def read_trec_run_all(path: str):
    """
    Parse a TREC run file into: {qid: [(pid, score, rank), ...]} (NO top-k cut here).
    We'll do 'fill to K' later when resolving passages.
    """
    per_q = defaultdict(list)
    with open(path, "r", encoding=ENCODING) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _Q0, pid, rank, score, _tag = parts[:6]
            try:
                rank_i = int(rank)
                score_f = float(score)
            except Exception:
                continue
            per_q[qid].append((pid, score_f, rank_i))
    # de-dup by pid (keep best rank) + sort by rank
    out = {}
    for qid, triples in per_q.items():
        best_by_pid = {}
        for pid, score, rank in triples:
            cur = best_by_pid.get(pid)
            if cur is None or rank < cur[2]:
                best_by_pid[pid] = (pid, score, rank)
        cleaned = list(best_by_pid.values())
        cleaned.sort(key=lambda x: x[2])
        out[qid] = cleaned
    return out


# ---------------- Filtering helpers (NEW) ----------------
def _minmax_normalize(vals):
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return [1.0 for _ in vals]
    rng = (vmax - vmin)
    return [(v - vmin) / rng for v in vals]


def filter_items_by_score(items,
                          min_score: float = FILTER_MIN_NORM_SCORE,
                          drop_thr: float = FILTER_DROP_THRESHOLD,
                          normalize: bool = FILTER_NORMALIZE):
    """
    items: list[{"pid","score","rank","text"}] ordered by rank (best first).
    Returns: list of 'text' after filtering.
    """
    if not items:
        return []
    scores = [it["score"] for it in items]
    ns = _minmax_normalize(scores) if normalize else scores

    keep_idx = []
    prev = None
    for i, _ in enumerate(items):
        s = ns[i]
        if i == 0:
            keep_idx.append(i)
            prev = s
            continue
        # minimum score gate
        if s < min_score:
            break
        # drop detection
        if prev - s >= drop_thr:
            break
        keep_idx.append(i)
        prev = s

    if not keep_idx:
        keep_idx = [0]  # ensure at least top-1
    return [items[i]["text"] for i in keep_idx]


# ---------------- Processing ----------------
def process_single_run(run_filename: str, G, pid2text: Dict[str, str]):
    run_path = find_run_path(run_filename)
    if not run_path:
        print(f"[ERROR] Run file not found under {RUNS_DIR}: {run_filename}")
        return

    split = infer_split_from_filename(run_filename)
    print(f"[INFO] Processing: {run_filename}  |  split={split}")

    qid2question = load_questions_for_split(split)
    qid2picks = read_trec_run_all(run_path)

    # Diagnostics: QID overlap
    qids_in_run = set(qid2picks.keys())
    qids_in_data = set(qid2question.keys())
    overlap = len(qids_in_run & qids_in_data)
    print(f"[INFO] QIDs: run={len(qids_in_run)} | dataset={len(qids_in_data)} | overlap={overlap}")

    outputs = []
    filtered_outputs = []
    unresolved = []

    for qid, triples in qid2picks.items():
        question = qid2question.get(qid)
        if question is None:
            unresolved.append(("missing_question", qid))
            continue

        items = []  # keep pid, score, rank, text for top-K
        for pid, score, rank in triples:
            if len(items) >= TOP_K:
                break
            # try direct and normalized keys
            text = pid2text.get(pid) or pid2text.get(re.sub(r"\s+", "", pid), None)
            if text is None:
                # try basic combos (in case run used doc_pid style)
                tokens = re.split(r"[:_\-]", pid)
                tokens = [t for t in tokens if t]
                if len(tokens) >= 2:
                    doc = tokens[0]
                    rest = "_".join(tokens[1:])
                    for c in (f"{doc}_{rest}", f"D{doc}_{rest}", f"{doc}::{rest}", f"D{doc}::{rest}",
                              f"{doc}-{rest}", f"D{doc}-{rest}", rest):
                        text = pid2text.get(c) or pid2text.get(re.sub(r"\s+", "", c))
                        if text:
                            break
            if text is None:
                unresolved.append(("missing_passage", qid, pid))
                continue
            items.append({"pid": pid, "score": score, "rank": rank, "text": text})

        # Unfiltered (current behavior)
        retrieved_passages = [it["text"] for it in items]

        # Filtered (new behavior)
        if FILTERING_ENABLED:
            retrieved_passages_filtered = filter_items_by_score(items)
        else:
            retrieved_passages_filtered = retrieved_passages

        answer = ""  # generation stub

        outputs.append(OrderedDict([
            ("QuestionID", qid),
            ("Question", question),
            ("RetrievedPassages", retrieved_passages),
            ("Answer", answer),
        ]))

        filtered_outputs.append(OrderedDict([
            ("QuestionID", qid),
            ("Question", question),
            ("RetrievedPassages", retrieved_passages_filtered),
            ("Answer", answer),
        ]))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base = os.path.splitext(os.path.basename(run_filename))[0]
    if unresolved:
        dbg_path = os.path.join(OUTPUT_DIR, f"{base}.unresolved.debug.jsonl")
        with open(dbg_path, "w", encoding=ENCODING) as f:
            for rec in unresolved:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[WARN] Unresolved logged -> {dbg_path} "
              f"(questions={sum(1 for x in unresolved if x[0]=='missing_question')}, "
              f"passages={sum(1 for x in unresolved if x[0]=='missing_passage')})")

    # Unfiltered
    out_path = os.path.join(OUTPUT_DIR, f"{base}.json")
    with open(out_path, "w", encoding=ENCODING) as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote {len(outputs)} QA entries -> {out_path}")

    # Filtered
    filt_path = os.path.join(OUTPUT_DIR, f"{base}_filtered.json")
    with open(filt_path, "w", encoding=ENCODING) as f:
        json.dump(filtered_outputs, f, ensure_ascii=False, indent=2)

    # quick stats
    orig_avg = sum(len(x["RetrievedPassages"]) for x in outputs) / max(1, len(outputs))
    filt_avg = sum(len(x["RetrievedPassages"]) for x in filtered_outputs) / max(1, len(filtered_outputs))
    reduction = (orig_avg - filt_avg) if orig_avg > 0 else 0.0
    print(f"[INFO] Filtered saved -> {filt_path} | avg passages/query: original={orig_avg:.2f}, "
          f"filtered={filt_avg:.2f} (Δ={reduction:.2f})")


def main():
    if not WHITELIST:
        print("[ERROR] WHITELIST is empty. Please add .trec filenames you want to process.")
        return

    # Check whitelist presence
    any_found = False
    for fname in WHITELIST:
        path = find_run_path(fname)
        print(f"[CHECK] {fname} -> {'FOUND' if path else 'NOT FOUND'}")
        if path:
            any_found = True
    if not any_found:
        print(f"[ERROR] None of the WHITELIST files were found under {RUNS_DIR}.")
        return

    # Load regulatory graph & build pid index
    G = load_graph_any()
    pid2text = build_pid_index_from_graph(G)

    # Process runs one-by-one
    for fname in WHITELIST:
        process_single_run(fname, G, pid2text)


if __name__ == "__main__":
    main()
