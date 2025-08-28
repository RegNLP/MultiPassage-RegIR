#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1: Build full regulatory knowledge graph (Documents → Passages, PARENT_OF,
global NamedEntity/DefinedTerm nodes, and optional cross-references).

Usage
-----
python -m src.prep.step1_build_graph \
  --crossref data/Documents/CrossReferenceData.csv \
  --out data/graph.gpickle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import pandas as pd

# --- Import manifest (works both as module and script) ---
try:
    # when called as module: python -m src.prep.step1_build_graph
    from .doc_manifest import DOCUMENTS  # type: ignore
except Exception:
    # fallback if run directly from src/prep
    from doc_manifest import DOCUMENTS  # type: ignore

# --- Repo root (portable paths, no user-specific absolute paths) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../MultiPassage-RegulatoryRAG


# =========================
# Normalization Utilities
# =========================

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F]")
DOT_RUN_RE = re.compile(r"\.+")


def strip_zero_width(s: str) -> str:
    return ZERO_WIDTH_RE.sub("", s)


def norm_pid(pid: Any) -> str:
    """
    Normalize PassageID:
      - cast to str, remove zero-width chars
      - strip spaces
      - collapse runs of '.' to single '.'
      - trim leading/trailing '.'
    """
    if pid is None:
        return ""
    s = strip_zero_width(str(pid)).strip()
    s = DOT_RUN_RE.sub(".", s)
    return s.strip(".")


def norm_term(s: Any) -> str:
    if s is None:
        return ""
    # collapse whitespace, strip zero-width
    s2 = " ".join(strip_zero_width(str(s)).split())
    return s2


def safe_uid(prefix: str, term: str) -> str:
    """
    Build a safe, deterministic UID for NE/DT nodes.
    - Keep [A-Za-z0-9_.:-], replace others with '_'
    - Truncate to 80 chars; if truncated/changed, add short md5 tail
    """
    raw = term or ""
    base = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw)
    changed = (base != raw)
    if len(base) > 80:
        base = base[:80]
        changed = True
    if changed:
        tail = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
        base = f"{base}_{tail}"
    return f"{prefix}_{base}"


def extract_list(obj: Any) -> List[Dict[str, Any]]:
    """If obj is a dict with 'Passages', return that; if list, return list; else empty list."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get("Passages", []) or []
    return []


def get_ne_fields(ent: Any) -> Tuple[str, str]:
    """Return (term, desc) for a NamedEntity item (dict or str)."""
    if isinstance(ent, dict):
        term = ent.get("Term") or ent.get("ContextID") or ent.get("ID")
        desc = ent.get("Description") or ent.get("Meaning", "")
    elif isinstance(ent, str):
        term, desc = ent, ""
    else:
        return "", ""
    return norm_term(term), ("" if desc is None else str(desc))


def get_dt_fields(item: Any) -> Tuple[str, str]:
    """Return (term, desc) for a DefinedTerm item (dict or str)."""
    if isinstance(item, dict):
        term = item.get("Term") or item.get("ContextID") or item.get("ID")
        desc = item.get("Description") or item.get("Meaning", "")
    elif isinstance(item, str):
        term, desc = item, ""
    else:
        return "", ""
    return norm_term(term), ("" if desc is None else str(desc))


# =========================
# Graph Construction
# =========================

def build_graph(crossref_path: Path) -> nx.DiGraph:
    print("[INFO] Building regulatory graph from manifest...")
    G = nx.DiGraph()

    for doc in DOCUMENTS:
        doc_id = doc["DocumentID"]
        title = doc.get("title", f"Document {doc_id}")
        json_path = Path(doc.get("json_file_path", ""))

        # Resolve path relative to repo if needed
        if not json_path.is_absolute():
            json_path = (PROJECT_ROOT / json_path).resolve()

        doc_node_id = f"D{doc_id}"

        # Document node
        G.add_node(
            doc_node_id,
            ID=str(doc_node_id),
            type="Document",
            title=title,
            DocumentID=doc_id,
            source_file=str(json_path),
        )

        print(f"\n📄 Processing {title}")
        if not json_path.exists():
            print(f"[WARN] Missing file: {json_path}")
            continue

        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        passages = extract_list(raw)
        pid2uid: Dict[str, str] = {}

        # ---- Passages + NE/DT + CONTAINS ----
        for p in passages:
            if not isinstance(p, dict):
                continue

            uid = p.get("ID") or p.get("ContextID")
            if not uid:
                # skip if no stable unique ID
                continue

            pid = norm_pid(p.get("PassageID") or p.get("ID") or p.get("ContextID"))
            text = (p.get("Passage") or p.get("Text") or "").strip()

            # Passage node
            G.add_node(
                uid,
                ID=str(uid),
                type="Passage",
                PassageID=pid,
                DocumentID=doc_id,
                text=text,
            )
            if pid:
                pid2uid[pid] = uid

            # Document -> Passage
            G.add_edge(doc_node_id, uid, type="CONTAINS")

            # Named Entities (GLOBAL)
            for ent in (p.get("NamedEntities") or []):
                term, desc = get_ne_fields(ent)
                if not term:
                    continue
                ne_uid = safe_uid("NE", term)
                if ne_uid not in G:
                    G.add_node(ne_uid, ID=ne_uid, type="NamedEntity", term=term, description=desc)
                G.add_edge(uid, ne_uid, type="MENTIONS")

            # Defined Terms (GLOBAL)
            for item in (p.get("DefinedTerms") or []):
                term, desc = get_dt_fields(item)
                if not term:
                    continue
                dt_uid = safe_uid("DT", term)
                if dt_uid not in G:
                    G.add_node(dt_uid, ID=dt_uid, type="DefinedTerm", term=term, description=desc)
                G.add_edge(uid, dt_uid, type="USES_TERM")

        # ---- PARENT_OF hierarchy (dot-delimited PassageIDs) ----
        for pid, uid in pid2uid.items():
            parts = pid.split(".")
            while len(parts) > 1:
                parts.pop()
                parent_pid = ".".join(parts)
                parent_uid = pid2uid.get(parent_pid)
                if parent_uid:
                    G.add_edge(parent_uid, uid, type="PARENT_OF")
                    break  # only link closest parent

    # ---- Cross-References (CITES / CITED_BY) ----
    if crossref_path and crossref_path.exists():
        try:
            df = pd.read_csv(crossref_path)
            # Expect at least: SourceID, TargetID (passage UIDs)
            added, missed = 0, 0
            for _, row in df.iterrows():
                src_uid = str(row.get("SourceID", "")).strip()
                tgt_uid = str(row.get("TargetID", "")).strip()
                if not src_uid or not tgt_uid or src_uid.lower() == "nan" or tgt_uid.lower() == "nan":
                    continue
                if src_uid in G and tgt_uid in G:
                    G.add_edge(src_uid, tgt_uid, type="CITES")
                    G.add_edge(tgt_uid, src_uid, type="CITED_BY")
                    added += 2
                else:
                    missed += 1
            print(f"\n🔗 Cross-reference edges added: {added} (missed rows: {missed})")
        except Exception as e:
            print(f"⚠️ Error loading cross-reference file '{crossref_path}': {e}")
    else:
        print(f"⚠️ No cross-reference file found at {crossref_path}")

    return G


# =========================
# Summary
# =========================

def print_graph_summary(G: nx.DiGraph) -> None:
    print("\n📊 Graph Summary")
    print(f"🔹 Total Nodes: {G.number_of_nodes()}")
    print(f"🔹 Total Edges: {G.number_of_edges()}")

    node_types = Counter(nx.get_node_attributes(G, "type").values())
    print("   📌 Nodes by type:")
    for t, count in sorted(node_types.items(), key=lambda x: (-x[1], x[0])):
        print(f"      - {t}: {count}")

    edge_types = Counter(nx.get_edge_attributes(G, "type").values())
    print("   📌 Edges by type:")
    for t, count in sorted(edge_types.items(), key=lambda x: (-x[1], x[0])):
        print(f"      - {t}: {count}")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the regulatory knowledge graph.")
    ap.add_argument(
        "--crossref",
        default=str(PROJECT_ROOT / "data" / "Documents" / "CrossReferenceData.csv"),
        help="CSV with columns {SourceID, TargetID} (optional).",
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "data" / "graph.gpickle"),
        help="Path to save the graph (.gpickle).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    crossref_path = Path(args.crossref)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    G = build_graph(crossref_path)

    # ---- Save with graceful fallback ----
    try:
        write_gpickle = getattr(nx, "write_gpickle", None)
        if callable(write_gpickle):
            write_gpickle(G, out_path)
        else:
            import pickle
            with open(out_path, "wb") as f:
                pickle.dump(G, f)
    except Exception:
        # Last-resort fallback
        import pickle
        with open(out_path, "wb") as f:
            pickle.dump(G, f)

    print(f"\n✅ Graph saved to: {out_path}")

    # Summary
    print_graph_summary(G)



if __name__ == "__main__":
    main()
