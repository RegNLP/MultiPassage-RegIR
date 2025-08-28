#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step0_fix_json_ids.py

Purpose
-------
Normalize and backfill IDs inside ObliQA-MP JSON split files (train/val/test).
- Ensures every example has a unique QuestionID.
- Normalizes PassageID strings (removes zero-width chars, collapses dots).
- Fills missing 'ID' for each passage using a {DocumentID}:{PassageID} pattern.
- Optionally hashes an ID when components are missing.
- Writes corrected JSON files to an output directory, preserving other fields.

Typical usage
-------------
python -m src.prep.step0_fix_json_ids \
  --qa-dir data/QADataset \
  --out-dir outputs/QADataset_fixed \
  --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F]")
DOT_RUN_RE = re.compile(r"\.+")  # collapse runs of dots to single '.'

SPLIT_FILES_DEFAULT = ("ObliQA_MultiPassage_train.json",
                       "ObliQA_MultiPassage_val.json",
                       "ObliQA_MultiPassage_test.json")


# ------------------------------
# Normalization helpers
# ------------------------------
def strip_zero_width(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return ZERO_WIDTH_RE.sub("", s)


def norm_passage_id(pid: Any) -> str:
    """
    Make PassageID uniform:
      - cast to str
      - remove zero-width chars
      - strip spaces
      - collapse '.....' -> '.'
      - trim leading/trailing dots
    """
    s = "" if pid is None else str(pid)
    s = strip_zero_width(s).strip()
    s = DOT_RUN_RE.sub(".", s)
    return s.strip(".")


def make_hashed_id(prefix: str, *parts: Any, n: int = 8) -> str:
    h = hashlib.md5("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:n]}"


def make_composite_id(doc_id: Any, passage_id: Any, template: str) -> str:
    return template.format(DocumentID=str(doc_id), PassageID=str(passage_id))


# ------------------------------
# I/O helpers
# ------------------------------
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------------
# Core processing
# ------------------------------
def process_split(
    data: List[Dict[str, Any]],
    id_template: str,
    hash_when_missing: bool = True,
    verbose: bool = True,
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process one split (list of examples). Returns (fixed_data, stats).
    """
    stats = {
        "examples": 0,
        "passages": 0,
        "qid_missing": 0,
        "qid_duplicates": 0,
        "pid_fixed": 0,
        "id_backfilled": 0,
        "id_hashed": 0,
        "id_already_ok": 0,
    }

    seen_qids: set[str] = set()
    qid_dup_count = 0

    fixed: List[Dict[str, Any]] = []
    for ex in data:
        stats["examples"] += 1

        # --- QuestionID ---
        qid = ex.get("QuestionID")
        if not qid:
            stats["qid_missing"] += 1
            qid = make_hashed_id("Q", ex.get("Question", ""), stats["examples"])
            ex["QuestionID"] = qid
            if verbose:
                print(f"[FIX] Missing QuestionID -> set to {qid}")

        qid = str(qid)
        if qid in seen_qids:
            qid_dup_count += 1
            # Disambiguate deterministically: append a short hash of content + idx
            new_qid = f"{qid}__{make_hashed_id('D', ex.get('Question', ''), stats['examples'])}"
            if verbose:
                print(f"[WARN] Duplicate QuestionID '{qid}' -> '{new_qid}'")
            ex["QuestionID"] = new_qid
            qid = new_qid
        seen_qids.add(qid)

        # --- Passages ---
        passages = ex.get("Passages", [])
        if not isinstance(passages, list):
            # keep structure but convert to empty list if malformed
            if verbose:
                print(f"[WARN] Example {qid}: 'Passages' not a list; coercing to []")
            passages = []
            ex["Passages"] = passages

        new_passages = []
        for p in passages:
            stats["passages"] += 1
            if not isinstance(p, dict):
                if verbose:
                    print(f"[WARN] Example {qid}: passage is not dict; skipping")
                continue

            # Normalize PassageID
            raw_pid = p.get("PassageID")
            npid = norm_passage_id(raw_pid)
            if raw_pid != npid:
                stats["pid_fixed"] += 1
                p["PassageID"] = npid
                if verbose:
                    print(f"[FIX] Example {qid}: PassageID '{raw_pid}' -> '{npid}'")

            # Backfill ID if missing/empty
            curr_id = p.get("ID")
            if curr_id is None or str(curr_id).strip() == "":
                doc_id = p.get("DocumentID")
                if doc_id is not None and npid:
                    new_id = make_composite_id(doc_id, npid, id_template)
                    p["ID"] = new_id
                    stats["id_backfilled"] += 1
                    if verbose:
                        print(f"[FIX] Example {qid}: backfilled ID -> {new_id}")
                elif hash_when_missing:
                    new_id = make_hashed_id("P", qid, npid or "NO_PID", doc_id or "NO_DOC")
                    p["ID"] = new_id
                    stats["id_hashed"] += 1
                    if verbose:
                        print(f"[FIX] Example {qid}: hashed ID -> {new_id}")
                else:
                    # leave as-is; possibly handled downstream
                    if strict:
                        raise ValueError(
                            f"Example {qid}: cannot backfill passage ID (missing DocumentID/PassageID)"
                        )
                    if verbose:
                        print(f"[WARN] Example {qid}: cannot backfill passage ID (missing parts)")
            else:
                stats["id_already_ok"] += 1

            new_passages.append(p)

        ex["Passages"] = new_passages
        fixed.append(ex)

    stats["qid_duplicates"] = qid_dup_count
    return fixed, stats


# ------------------------------
# CLI
# ------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Normalize/backfill IDs for ObliQA-MP splits.")
    ap.add_argument("--qa-dir", type=Path, required=True,
                    help="Directory containing split JSON files (train/val/test).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write corrected split JSON files.")
    ap.add_argument("--splits", nargs="*", default=list(SPLIT_FILES_DEFAULT),
                    help="Specific split filenames to process (default: train/val/test).")
    ap.add_argument("--id-format", default="{DocumentID}:{PassageID}",
                    help="Template for backfilled passage IDs.")
    ap.add_argument("--no-hash-missing", action="store_true",
                    help="Do not hash IDs when DocumentID/PassageID are missing.")
    ap.add_argument("--strict", action="store_true",
                    help="Fail on any passage that cannot get a valid ID.")
    ap.add_argument("--quiet", action="store_true",
                    help="Reduce logging.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    qa_dir: Path = args.qa_dir
    out_dir: Path = args.out_dir

    if not qa_dir.exists():
        raise FileNotFoundError(f"--qa-dir not found: {qa_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_totals = {
        "files": 0,
        "examples": 0,
        "passages": 0,
        "qid_missing": 0,
        "qid_duplicates": 0,
        "pid_fixed": 0,
        "id_backfilled": 0,
        "id_hashed": 0,
        "id_already_ok": 0,
    }

    for fname in args.splits:
        in_path = qa_dir / fname
        if not in_path.exists():
            print(f"[WARN] Split not found, skipping: {in_path}")
            continue

        print(f"\n[INFO] Processing {in_path}")
        data = read_json(in_path)
        if not isinstance(data, list):
            raise ValueError(f"{in_path} does not contain a list of examples")

        fixed, stats = process_split(
            data,
            id_template=args.id_format,
            hash_when_missing=not args.no_hash_missing,
            verbose=not args.quiet,
            strict=args.strict,
        )

        out_path = out_dir / fname
        write_json(fixed, out_path)
        print(f"[OK] Wrote: {out_path}")

        # roll up totals
        all_totals["files"] += 1
        for k in (k for k in all_totals.keys() if k != "files"):
            all_totals[k] += stats.get(k, 0)

        # per-file summary
        print(
            "[STATS] "
            f"examples={stats['examples']} passages={stats['passages']} | "
            f"qid_missing={stats['qid_missing']} dup_qid={stats['qid_duplicates']} | "
            f"pid_fixed={stats['pid_fixed']} id_backfilled={stats['id_backfilled']} "
            f"id_hashed={stats['id_hashed']} id_ok={stats['id_already_ok']}"
        )

    # grand total summary
    print("\n===== SUMMARY =====")
    print(
        f"files={all_totals['files']} examples={all_totals['examples']} "
        f"passages={all_totals['passages']}"
    )
    print(
        f"QuestionIDs -> missing fixed={all_totals['qid_missing']}, "
        f"duplicates resolved={all_totals['qid_duplicates']}"
    )
    print(
        f"PassageIDs -> normalized={all_totals['pid_fixed']}; "
        f"IDs backfilled={all_totals['id_backfilled']}, "
        f"hashed={all_totals['id_hashed']}, "
        f"already_ok={all_totals['id_already_ok']}"
    )


if __name__ == "__main__":
    main()
