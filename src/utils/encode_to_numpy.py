#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Encode a JSONL passage collection (id + text/contents) into a NumPy matrix for FAISS.

Outputs:
  - <out-dir>/embeddings.npy  : float32 [num_docs, dim] (L2-normalized)
  - <out-dir>/docids.txt      : one docid per line (aligned with rows)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Prefer sentence-transformers if available; otherwise fall back to HF
try:
    from sentence_transformers import SentenceTransformer
    USE_ST = True
except Exception:
    USE_ST = False


def read_docs(jsonl_path: Path):
    """Read docs.jsonl exported by step2c; return (docids, texts)."""
    docids, texts = [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            did = str(obj["id"])
            txt = obj.get("text")
            if txt is None:
                txt = obj.get("contents", "")
            docids.append(did)
            texts.append("" if txt is None else str(txt))
    return docids, texts


def encode_with_sentence_transformers(model_name: str, texts, batch_size: int = 32, device: str = "cpu"):
    """Encode using sentence-transformers with normalization."""
    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize
        show_progress_bar=True,
    )
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32, copy=False)
    return embs


def encode_with_hf(model_name: str, texts, batch_size: int = 16, device: str = "cpu", max_length: int = 512):
    """Encode using plain HF (CLS pooling) with normalization."""
    import torch  # local import so it's only required for this path
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()

    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = mdl(**enc)
            # CLS pooling
            last_hidden = out.last_hidden_state  # (B, T, H)
            cls = last_hidden[:, 0, :]          # (B, H)
            # L2 normalize for IP similarity
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)

        all_vecs.append(cls.cpu().numpy())

    embs = np.vstack(all_vecs).astype(np.float32, copy=False)
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    ap.add_argument("--input-jsonl", required=True, help="docs.jsonl (with id + text/contents)")
    ap.add_argument("--out-dir", required=True, help="Output dir for embeddings.npy & docids.txt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    jsonl_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docids, texts = read_docs(jsonl_path)
    if not docids:
        raise SystemError(f"No documents read from {jsonl_path}")

    print(f"[INFO] Encoding {len(docids)} docs with "
          f"{'sentence-transformers' if USE_ST else 'HF AutoModel'} on {args.device} ...")

    if USE_ST:
        embs = encode_with_sentence_transformers(args.model, texts, batch_size=args.batch_size, device=args.device)
    else:
        embs = encode_with_hf(args.model, texts, batch_size=args.batch_size, device=args.device, max_length=args.max_length)

    # Save outputs
    np.save(out_dir / "embeddings.npy", embs)
    with (out_dir / "docids.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(docids))

    print(f"[OK] Saved: {out_dir/'embeddings.npy'}  shape={embs.shape} dtype={embs.dtype}")
    print(f"[OK] Saved: {out_dir/'docids.txt'}     n={len(docids)}")


if __name__ == "__main__":
    # CPU stability env (optional but helpful on macOS)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
