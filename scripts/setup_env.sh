#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# Config (change if you like)
# -------------------------------
ENV_NAME="regulatoryrag"
CHANNEL="conda-forge"
PY_VERSION="3.10"
PYSERINI_VERSION="0.22.0"   # Java 17–compatible

RECREATE=0
if [[ "${1:-}" == "--recreate" ]]; then
  RECREATE=1
fi

# -------------------------------
# Helpers
# -------------------------------
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }; }
say()  { echo -e "\033[1;36m==>\033[0m $*"; }

need conda
eval "$(conda shell.bash hook)"

# -------------------------------
# Create / recreate env
# -------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ $RECREATE -eq 1 ]]; then
    say "Removing existing env '$ENV_NAME'..."
    conda env remove -n "$ENV_NAME" -y
  else
    say "Conda env '$ENV_NAME' already exists. Use '--recreate' to rebuild."
    exit 0
  fi
fi

say "Creating conda env '$ENV_NAME'..."
conda create -y -n "$ENV_NAME" -c "$CHANNEL" \
  "python=${PY_VERSION}" \
  openjdk=17 \
  faiss \
  "numpy<2" \
  pandas \
  networkx \
  lightgbm \
  pip \
  tqdm

say "Activating env..."
conda activate "$ENV_NAME"

# -------------------------------
# Pip installs (no nmslib headaches)
# -------------------------------
say "Installing pip packages..."
python -m pip install --upgrade pip

# Core retrieval / encoders / eval
python -m pip install --no-cache-dir \
  sentence-transformers>=2.7.0 \
  transformers \
  torch \
  pytrec-eval \
  pyjnius>=1.6.1 \
  onnxruntime

# Pyserini WITHOUT optional deps (avoids nmslib build on macOS ARM)
python -m pip install --no-cache-dir "pyserini==${PYSERINI_VERSION}" --no-deps

# -------------------------------
# Activation hooks (Java + stability)
# -------------------------------
say "Writing env activation hooks..."
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/java_vars.sh" <<'EOS'
# Java from conda's openjdk
export JAVA_HOME="$CONDA_PREFIX"
export JVM_PATH="$CONDA_PREFIX/lib/server/libjvm.dylib"
export PATH="$JAVA_HOME/bin:$PATH"

# Stable defaults for tokenizers/BLAS (user can override)
: "${OMP_NUM_THREADS:=1}"; export OMP_NUM_THREADS
: "${MKL_NUM_THREADS:=1}"; export MKL_NUM_THREADS
: "${TOKENIZERS_PARALLELISM:=false}"; export TOKENIZERS_PARALLELISM
EOS

cat > "$DEACTIVATE_DIR/java_vars.sh" <<'EOS'
# best-effort cleanup
unset JAVA_HOME
unset JVM_PATH
EOS

# Also export for THIS shell so checks below work now
export JAVA_HOME="$CONDA_PREFIX"
export JVM_PATH="$CONDA_PREFIX/lib/server/libjvm.dylib"
export PATH="$JAVA_HOME/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# -------------------------------
# Sanity checks
# -------------------------------
say "Verifying Pyserini, JVM, FAISS..."
python - <<'PY'
import os, importlib
print("JAVA_HOME:", os.environ.get("JAVA_HOME"))
try:
    import pyserini
    print("pyserini:", pyserini.__version__)
    importlib.import_module("pyserini.index.lucene")
    print("pyserini.index.lucene: OK")
except Exception as e:
    print("Pyserini check FAILED:", e)

try:
    import faiss
    print("faiss:", getattr(faiss, '__version__', 'import ok'))
except Exception as e:
    print("FAISS check FAILED:", e)

try:
    from jnius import autoclass
    System = autoclass('java.lang.System')
    print("java.version:", System.getProperty("java.version"))
except Exception as e:
    print("JVM check FAILED:", e)
PY

cat <<'MSG'

✅ Environment is ready.

Next steps (every new shell):
  conda activate regulatoryrag

Run the quick test pipeline (if you have it):
  chmod +x scripts/run_end2end_test.sh
  ./scripts/run_end2end_test.sh

Notes:
- We pin Pyserini to 0.22.0 (Java 17). If you upgrade Pyserini, switch to JDK 21 and update JAVA_HOME.
- If torch fails via pip on your platform, try: conda install pytorch -c pytorch -c conda-forge
MSG
