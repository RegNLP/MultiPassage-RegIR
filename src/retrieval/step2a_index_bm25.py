#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 2A: Build BM25 index directly from graph.gpickle
- Exports Passage nodes as a Pyserini JsonCollection (docs.jsonl)
- (Optionally) builds a Lucene index via Pyserini CLI

Usage
-----
python -m src.retrieval.step2a_index_bm25 \
  --graph data/graph.gpickle \
  --collection-dir data/collections/bm25_jsonl \
  --index-dir outputs/indexes/bm25 \
  --build-index
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export passages from graph and (optionally) build a BM25 index.")
    ap.add_argument("--graph", required=True, help="Path to graph.gpickle")
    ap.add_argument("--collection-dir", required=True, help="Directory to store JSONL collection (JsonCollection)")
    ap.add_argument("--index-dir", required=True, help="Directory to store BM25 index (Lucene)")
    ap.add_argument("--min-chars", type=int, default=5, help="Minimum passage length to index (after strip())")
    ap.add_argument("--threads", type=int, default=8, help="Number of indexing threads")
    ap.add_argument("--build-index", action="store_true", help="Build the BM25 index after JSONL export")
    ap.add_argument("--no-store-raw", action="store_true", help="Do not store raw text in the index")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing collection/index")
    ap.add_argument("--max-docs", type=int, default=0, help="If >0, cap number of exported docs (deterministic)")
    return ap.parse_args()


def _load_graph(graph_path: Path):
    if not graph_path.exists():
        raise FileNotFoundError(f"[ERR] Graph not found: {graph_path}")
    with graph_path.open("rb") as f:
        return pickle.load(f)


def _iter_passages(G) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (uid, data) for Passage nodes only."""
    for uid, data in G.nodes(data=True):
        if isinstance(data, dict) and data.get("type") == "Passage":
            yield str(uid), data


def export_collection(graph_path: Path, collection_dir: Path, min_chars: int, max_docs: int = 0, overwrite: bool = False) -> Path:
    """Export Passage nodes from the graph as JSONL (JsonCollection-compatible)."""
    print(f"[INFO] Loading graph from {graph_path} ...")
    G = _load_graph(graph_path)

    collection_dir = collection_dir.resolve()
    if collection_dir.exists() and any(collection_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"[ERR] Collection dir not empty: {collection_dir} (use --overwrite)")
    collection_dir.mkdir(parents=True, exist_ok=True)

    out_fp = collection_dir / "docs.jsonl"
    map_fp = collection_dir / "docids.tsv"  # id \t char_len

    passages = list(_iter_passages(G))
    # Deterministic export order by stable ID on node; fallback to node id
    passages.sort(key=lambda it: (str(it[1].get("ID") or it[0])))

    written, skipped_short, skipped_empty = 0, 0, 0
    with out_fp.open("w", encoding="utf-8") as out_f, map_fp.open("w", encoding="utf-8") as map_f:
        for uid, data in passages:
            text = (data.get("text") or "").strip()
            if not text:
                skipped_empty += 1
                continue
            if len(text) < min_chars:
                skipped_short += 1
                continue

            # Use the stable passage ID we stored on the node; fall back to node_id
            doc_id = str(data.get("ID") or uid)

            # Normalize newlines to spaces (Pyserini handles, but keeps doc tidy)
            contents = " ".join(text.splitlines()).strip()

            out_f.write(json.dumps({"id": doc_id, "contents": contents}, ensure_ascii=False) + "\n")
            map_f.write(f"{doc_id}\t{len(contents)}\n")
            written += 1

            if max_docs and written >= max_docs:
                break

    print(f"[OK] Exported {written} passages to {out_fp} "
          f"(skipped: empty={skipped_empty}, short<{min_chars}={skipped_short})")
    print(f"[OK] DocID map written to {map_fp}")
    return out_fp


def build_bm25_index(collection_dir: Path, index_dir: Path, threads: int, store_raw: bool, overwrite: bool = False) -> None:
    """Invoke Pyserini's CLI to build a Lucene index from the JsonCollection."""
    index_dir = index_dir.resolve()
    if index_dir.exists() and any(index_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"[ERR] Index dir not empty: {index_dir} (use --overwrite)")
    index_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--input", str(collection_dir),
        "--index", str(index_dir),
        "--storePositions",
        "--storeDocvectors",
    ]
    if store_raw:
        cmd.append("--storeRaw")

    print("[CMD]", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError as e:
        print("[ERROR] Could not invoke Pyserini. Is it installed? `pip install pyserini`")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Pyserini indexer failed with exit code {e.returncode}")
        raise
    print(f"[DONE] BM25 index built at {index_dir}")


def main() -> None:
    args = parse_args()
    store_raw = not args.no_store_raw

    export_collection(
        Path(args.graph),
        Path(args.collection_dir),
        args.min_chars,
        max_docs=args.max_docs,
        overwrite=args.overwrite,
    )

    if args.build_index:
        build_bm25_index(
            Path(args.collection_dir),
            Path(args.index_dir),
            args.threads,
            store_raw,
            overwrite=args.overwrite,
        )
    else:
        print("[INFO] Skipping index build (no --build-index).")


if __name__ == "__main__":
    main()
