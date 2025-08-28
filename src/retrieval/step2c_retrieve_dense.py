#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 2C: Dense retrieval with sentence-transformers (E5) + Faiss

- Reads collection from data/collections/bm25_jsonl/docs.jsonl (id, contents)
- Encodes passages and builds a Faiss index (saved under outputs/indexes/dense/)
- Encodes queries from a split JSON (QuestionID, Question)
- Searches top-k dense hits and writes TREC run

Usage:
python -m src.retrieval.step2c_retrieve_dense \
  --queries-json outputs/QADataset_fixed/ObliQA_MultiPassage_test.json \
  --collection data/collections/bm25_jsonl/docs.jsonl \
  --index-dir outputs/indexes/dense \
  --out outputs/runs/dense_test.trec
"""

from __future__ import annotations

import argparse
import faiss
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dense retrieval with E5 + Faiss, outputs TREC.")
    ap.add_argument("--queries-json", required=True, help="Split JSON with QuestionID/Question.")
    ap.add_argument("--collection", required=True, help="Path to docs.jsonl (id, contents).")
    ap.add_argument("--index-dir", required=True, help="Dir to save/load Faiss index + mapping.")
    ap.add_argument("--out", required=True, help="Output TREC run path.")
    ap.add_argument("--model", default="intfloat/e5-base-v2", help="Sentence-Transformer model.")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for encoding.")
    ap.add_argument("--hits", type=int, default=1000, help="Top-k results per query.")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild embeddings/index.")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (cosine).")
    return ap.parse_args()


def load_collection(path: Path) -> Tuple[List[str], List[str]]:
    ids, texts = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            did = str(obj["id"])
            txt = (obj.get("contents") or "").strip()
            if not txt:
                continue
            ids.append(did)
            texts.append(txt)
    return ids, texts


def load_queries(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out, seen = [], set()
    for ex in data:
        qid = str(ex.get("QuestionID") or "").strip()
        q = (ex.get("Question") or "").strip()
        if not qid or not q or qid in seen:
            continue
        seen.add(qid)
        out.append((qid, q))
    if not out:
        raise ValueError(f"No queries in {path}")
    return out


def e5_format_queries(queries: List[str]) -> List[str]:
    # E5 expects "query: ..." / "passage: ..." formatting
    return [f"query: {q}" for q in queries]


def e5_format_passages(texts: List[str]) -> List[str]:
    return [f"passage: {t}" for t in texts]


def encode_texts(model: SentenceTransformer, texts: List[str], batch: int, normalize: bool) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    embs = model.encode(
        texts,
        batch_size=batch,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embs.astype("float32")


def build_or_load_index(index_dir: Path, dim: int, rebuild: bool) -> faiss.IndexFlatIP:
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = index_dir / "dense.index"
    if faiss_path.exists() and not rebuild:
        index = faiss.read_index(str(faiss_path))
        return index
    # fresh index
    index = faiss.IndexFlatIP(dim)
    return index


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    map_path = index_dir / "docids.npy"
    ids_path = index_dir / "ids.txt"

    # Load collection
    doc_ids, doc_texts = load_collection(Path(args.collection))
    print(f"[INFO] Loaded collection: {len(doc_ids)} docs")

    # Model
    model = SentenceTransformer(args.model)
    # For cosine similarity with Faiss IP, normalize both sides
    normalize = True if args.normalize else True

    # Build/load passage embeddings
    emb_path = index_dir / "doc_emb.npy"
    if emb_path.exists() and not args.rebuild:
        doc_emb = np.load(emb_path)
        print(f"[INFO] Loaded embeddings: {doc_emb.shape}")
    else:
        print("[INFO] Encoding passages...")
        p_texts = e5_format_passages(doc_texts)
        doc_emb = encode_texts(model, p_texts, args.batch, normalize=normalize)
        np.save(emb_path, doc_emb)
        print(f"[OK] Saved embeddings: {emb_path}")

    # Build/load Faiss index
    index = build_or_load_index(index_dir, doc_emb.shape[1], args.rebuild)
    if index.ntotal == 0 or args.rebuild:
        index.add(doc_emb)
        faiss.write_index(index, str(index_dir / "dense.index"))
        np.save(map_path, np.arange(len(doc_ids), dtype=np.int32))
        with ids_path.open("w", encoding="utf-8") as f:
            for did in doc_ids:
                f.write(did + "\n")
        print(f"[OK] Built Faiss index with {index.ntotal} vectors")

    # Load queries
    pairs = load_queries(Path(args.queries_json))
    qids = [qid for qid, _ in pairs]
    qtexts = [q for _, q in pairs]

    # Encode queries
    print("[INFO] Encoding queries...")
    q_texts = e5_format_queries(qtexts)
    q_emb = encode_texts(model, q_texts, args.batch, normalize=normalize)

    # Search
    topk = args.hits
    scores, inds = index.search(q_emb, topk)

    # Write TREC
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, qid in enumerate(qids):
            for rank, j in enumerate(inds[i], start=1):
                if j < 0:
                    continue
                docid = doc_ids[int(j)]
                score = float(scores[i][rank - 1])
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} dense\n")

    print(f"[OK] Wrote dense TREC run: {out_path} (queries={len(qids)}, k={topk})")


if __name__ == "__main__":
    main()
