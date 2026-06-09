#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$PROJECT_ROOT/.conda/sdxl-lora}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$CONDA_ENV_PREFIX" "python=$PYTHON_VERSION"
fi

"$CONDA_ENV_PREFIX/bin/python" -m pip install --upgrade pip
"$CONDA_ENV_PREFIX/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"

echo "conda env ready: $CONDA_ENV_PREFIX"
