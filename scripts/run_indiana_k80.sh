#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export XFORMERS_DISABLED=1
export CLEAR_DINOV2_REPO="$PWD/.cache/dinov2"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=${OUT:-results/indiana_$(date -u +%Y%m%dT%H%M%SZ)_$$}
mkdir -p "$OUT"
.venv-k80/bin/python -u scripts/test_indiana.py --download --output "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
