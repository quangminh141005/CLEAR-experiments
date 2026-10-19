#!/usr/bin/env bash
# Server-side only. Installs into the active Conda environment.
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROFILE=auto
DINO_REV=7764ea0f912e53c92e82eb78a2a1631e92725fc8
while (($#)); do
    case "$1" in
        --profile) PROFILE=$2; shift 2 ;;
        -h|--help) printf 'Usage: bash setup.sh [--profile auto|k80|cuda|cpu|prompts]\nActivate a dedicated Python 3.10 Conda environment first.\n'; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ -z "${CONDA_PREFIX:-}" || ! -x "$CONDA_PREFIX/bin/python" ]]; then
    printf 'Activate a Python 3.10 Conda environment first, for example:\nconda create -n clear-k80 python=3.10 pip -y\nconda activate clear-k80\n' >&2
    exit 1
fi
ENV_PREFIX=$CONDA_PREFIX
PY="$ENV_PREFIX/bin/python"
"$PY" -c 'import pathlib, sys; assert sys.version_info[:2] == (3,10), "This reproducible stack requires Python 3.10"; print("Installing into Conda environment:", pathlib.Path(sys.prefix).resolve())'
GPU_NAMES=""
if command -v nvidia-smi >/dev/null; then
    nvidia-smi || true
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
else
    printf 'NVIDIA driver tools unavailable. CPU mode remains available.\n'
fi
if [[ "$PROFILE" == auto ]]; then
    if [[ "$GPU_NAMES" == *K80* ]]; then PROFILE=k80
    elif [[ -n "$GPU_NAMES" ]]; then PROFILE=cuda
    else PROFILE=cpu; fi
fi
case "$PROFILE" in k80|cuda|cpu|prompts) ;; *) printf 'Unsupported profile %s\n' "$PROFILE" >&2; exit 2 ;; esac
if [[ "$PROFILE" == cuda && "$GPU_NAMES" == *K80* ]]; then
    printf 'K80 detected: use --profile k80. The modern CUDA stack cannot run sm_37.\n' >&2; exit 1
fi
"$PY" -m pip install pip==24.3.1 setuptools==77.0.3 wheel==0.44.0
if [[ "$PROFILE" == k80 ]]; then
    # Kepler 3.7: retain the existing repository's CUDA 10.2 / PyTorch 1.12 stack.
    "$PY" -m pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
elif [[ "$PROFILE" == cuda ]]; then
    "$PY" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
else
    "$PY" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
fi
"$PY" -m pip install -r "$REPO_DIR/requirements-evaluation.txt"
if [[ "$PROFILE" == prompts ]]; then
    # Separate modern CPU environment for the 7B SFR model; no K80 dependency upgrade.
    "$PY" -m pip install transformers==4.46.3 tokenizers==0.20.3 sentencepiece==0.2.0 safetensors==0.4.5 accelerate==0.34.2
fi
if [[ "$PROFILE" != prompts ]]; then
    command -v git >/dev/null || { printf 'Install git manually before setting up DINOv2.\n' >&2; exit 1; }
    DINO_DIR="$REPO_DIR/.cache/clear-evaluation/dinov2-$DINO_REV"
    if [[ ! -d "$DINO_DIR" ]]; then
        git clone https://github.com/facebookresearch/dinov2.git "$DINO_DIR"
        git -C "$DINO_DIR" checkout --detach "$DINO_REV"
    fi
    [[ $(git -C "$DINO_DIR" rev-parse HEAD) == "$DINO_REV" ]] || { printf 'DINOv2 revision mismatch\n' >&2; exit 1; }
else
    DINO_DIR=""
fi
{
    printf 'export PYTHONPATH=%q${PYTHONPATH:+:$PYTHONPATH}\n' "$REPO_DIR/src"
    printf 'export XFORMERS_DISABLED=1\n'
    if [[ -n "$DINO_DIR" ]]; then printf 'export CLEAR_DINOV2_REPO=%q\n' "$DINO_DIR"; fi
    printf 'export CLEAR_ASSET_DIR=%q\n' "$REPO_DIR/assets/clear"
} > "$ENV_PREFIX/clear-env.sh"
"$PY" -m pip freeze > "$ENV_PREFIX/requirements-resolved.txt"
"$PY" -c 'import platform, torch, torchvision; print("Python",platform.python_version(),"Torch",torch.__version__,"Torchvision",torchvision.__version__,"wheel CUDA",torch.version.cuda,"GPU available",torch.cuda.is_available()); print("compiled architectures",torch.cuda.get_arch_list() if torch.cuda.is_available() else []); print("GPUs",[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
if [[ "$PROFILE" == k80 ]]; then
    printf 'K80: compute capability 3.7, FP32 ordinary attention, batch 1 suggested. Driver must support CUDA 10.2 AND your Kepler card. No BF16, FlashAttention or xFormers. Verify sm_37 in architecture output on this server.\n'
fi
printf 'Setup finished for Conda environment %s. Configure CLEAR in the current shell with:\nsource %q\n' "$ENV_PREFIX" "$ENV_PREFIX/clear-env.sh"
printf 'After opening another SSH shell, conda activate this environment and source its clear-env.sh again.\n'
printf 'No model/dataset downloaded by setup. Full concept bank needs substantial host RAM (32–64 GiB recommended).\n'
