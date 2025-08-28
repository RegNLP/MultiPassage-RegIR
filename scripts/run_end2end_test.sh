#!/usr/bin/env bash
set -euo pipefail

# --------------- config ---------------
SPLIT="${1:-test}"                              # allow: ./run_end2end_test.sh [train|val|test]
QA_DIR="data/QADataset"                         # raw JSONs (train/val/test)
FIX_DIR="outputs/QADataset_fixed"
GRAPH="data/graph.gpickle"
CROSSREF="data/Documents/CrossReferenceData.csv"

COLL_DIR="data/collections/bm25_jsonl"
BM25_INDEX="outputs/indexes/bm25"
DENSE_INDEX_DIR="outputs/indexes/dense"
RUN_DIR="outputs/runs"
FEAT_DIR="outputs/features"
LTR_DIR="outputs/ltr"
EVAL_DIR="outputs/eval"

# Stability on macOS/Conda for dense encoding (harmless elsewhere)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${FIX_DIR}" "${RUN_DIR}" "${FEAT_DIR}" "${LTR_DIR}" "${COLL_DIR}" \
         "$(dirname "${GRAPH}")" "${EVAL_DIR}"

# --------------- preflight ---------------
echo "==> Preflight checks"
if python - <<'PY' >/dev/null 2>&1
from importlib import import_module as I
I("pyserini"); I("pytrec_eval"); I("sentence_transformers"); I("faiss")
PY
then
  :  # ok
else
  echo "[ERR] Missing one of: pyserini, pytrec_eval, sentence-transformers, faiss." >&2
  echo "     Activate env or run scripts/setup_env.sh" >&2
  exit 1
fi

if [[ -z "${JAVA_HOME:-}" ]]; then
  echo "[WARN] JAVA_HOME is not set. If pyserini indexing fails, export Java 17/21:"
  echo '  export JAVA_HOME="$("/usr/libexec/java_home" -v 17)"'
fi

# --------------- 0) fix dataset IDs ---------------
echo "==> Step0: fixing dataset IDs -> ${FIX_DIR}"
python -m src.prep.step0_fix_json_ids \
  --qa-dir "${QA_DIR}" \
  --out-dir "${FIX_DIR}"

FIX_JSON="${FIX_DIR}/ObliQA_MultiPassage_${SPLIT}.json"
if [[ ! -s "${FIX_JSON}" ]]; then
  echo "[ERR] Expected fixed JSON not found: ${FIX_JSON}" >&2
  echo "     Make sure raw data exists under ${QA_DIR}/${SPLIT} or copy the JSON there." >&2
  exit 1
fi

# --------------- 1) build graph ---------------
echo "==> Step1: building graph -> ${GRAPH}"
python -m src.prep.step1_build_graph \
  --crossref "${CROSSREF}" \
  --out "${GRAPH}"

# --------------- 2a) export + BM25 index ---------------
echo "==> Step2a: JsonCollection export + BM25 index"
rm -rf "${BM25_INDEX}"
python -m src.retrieval.step2a_index_bm25 \
  --graph "${GRAPH}" \
  --collection-dir "${COLL_DIR}" \
  --index-dir "${BM25_INDEX}" \
  --build-index

# --------------- 2b) BM25 retrieve ---------------
echo "==> Step2b: BM25 retrieval (${SPLIT})"
python -m src.retrieval.step2b_retrieve_bm25 \
  --queries-json "${FIX_JSON}" \
  --index-dir "${BM25_INDEX}" \
  --out "${RUN_DIR}/bm25_${SPLIT}.trec" \
  --hits 1000 --bm25-k1 0.9 --bm25-b 0.4 --tag bm25

# --------------- 2c) Dense retrieve ---------------
echo "==> Step2c: Dense retrieval (${SPLIT})"
mkdir -p "${DENSE_INDEX_DIR}"
rm -f "${DENSE_INDEX_DIR}/doc_emb.npy" "${DENSE_INDEX_DIR}/faiss.index" 2>/dev/null || true
python -m src.retrieval.step2c_retrieve_dense \
  --queries-json "${FIX_JSON}" \
  --collection "${COLL_DIR}/docs.jsonl" \
  --index-dir "${DENSE_INDEX_DIR}" \
  --out "${RUN_DIR}/dense_${SPLIT}.trec" \
  --batch 8

# --------------- 2d) RRF fusion ---------------
echo "==> Step2d: RRF fusion (${SPLIT})"
python -m src.retrieval.step2d_hybrid_fuse \
  --runs-dir "${RUN_DIR}" \
  --bm25-name bm25 \
  --dense-name dense \
  --method rrf \
  --k 1000 --rrf-c 60 --tag rrf

# --------------- 3) qrels from JSON ---------------
echo "==> Step3: building qrels (${SPLIT})"
python -m src.eval.json_passages_to_qrels \
  --split-json "${FIX_JSON}" \
  --out-qrels "data/qrels_${SPLIT}.txt"

# --------------- 4) features (test) ---------------
echo "==> Step4a: extracting features (${SPLIT})"
python -m src.features.step4a_extract_features \
  --bm25-run "${RUN_DIR}/bm25_${SPLIT}.trec" \
  --dense-run "${RUN_DIR}/dense_${SPLIT}.trec" \
  --rrf-run   "${RUN_DIR}/rrf_bm25+dense_${SPLIT}.trec" \
  --graph "${GRAPH}" \
  --doclen-map "${COLL_DIR}/docids.tsv" \
  --out-csv "${FEAT_DIR}/features_${SPLIT}.csv" \
  --feature-list "${LTR_DIR}/feature_list.txt"

# --------------- 5) evaluate @10 ---------------
echo "==> Step4d: evaluating on ${SPLIT} (BM25 / Dense / RRF) @10"
python -m src.eval.eval_test_at10 \
  --qa-test "${FIX_JSON}" \
  --runs-dir "${RUN_DIR}" \
  --pattern "*_${SPLIT}.trec" \
  --out-tsv "${EVAL_DIR}/${SPLIT}_at10_summary.tsv"

echo "✅ Done. See ${EVAL_DIR}/${SPLIT}_at10_summary.tsv"
