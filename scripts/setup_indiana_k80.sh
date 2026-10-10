#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN=${PYTHON_BIN:-python3.10}
"$PYTHON_BIN" -m venv .venv-k80
.venv-k80/bin/python -m pip install 'pip<25'
.venv-k80/bin/python -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
.venv-k80/bin/python -m pip install -r scripts/requirements-k80.txt
mkdir -p .cache
if [[ ! -d .cache/dinov2/.git ]]; then
    git clone https://github.com/facebookresearch/dinov2.git .cache/dinov2
fi
git -C .cache/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
printf 'Setup complete. Run: bash scripts/run_indiana_k80.sh\n'
