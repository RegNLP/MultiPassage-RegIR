#!/usr/bin/env bash
set -euo pipefail

# Removes a Conda environment (default: regulatoryrag).
# Usage:
#   bash scripts/teardown_env.sh
#   bash scripts/teardown_env.sh --name otherenv
#   bash scripts/teardown_env.sh --force

ENV_NAME="regulatoryrag"
FORCE=0

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name) ENV_NAME="${2:-}"; shift 2 ;;
    -y|--yes|--force) FORCE=1; shift ;;
    *) echo "Usage: $0 [--name NAME] [--force]"; exit 1 ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }; }
say()  { echo -e "\033[1;31m==>\033[0m $*"; }

need conda
# enable 'conda activate' in script
eval "$(conda shell.bash hook)"

# ---- check existence ----
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' not found; nothing to do."
  exit 0
fi

# ---- deactivate if active ----
if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  say "Deactivating active env '$ENV_NAME'..."
  conda deactivate
fi

# ---- confirm ----
if [[ $FORCE -eq 0 ]]; then
  read -r -p "Remove conda env '$ENV_NAME'? [y/N] " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

# ---- remove ----
say "Removing env '$ENV_NAME'..."
conda env remove -n "$ENV_NAME" -y

say "Done."
