#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    printf 'Activate your Python 3.10 Conda environment first.\n' >&2
    exit 1
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "K80 setup requires Python 3.10"'
python -m pip install -r requirements.txt
mkdir -p .cache
if [[ ! -d .cache/dinov2/.git ]]; then
    git clone https://github.com/facebookresearch/dinov2.git .cache/dinov2
fi
git -C .cache/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
printf 'Setup complete. Run: bash scripts/run_indiana_k80.sh\n'
