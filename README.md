# CLEAR

Concept-Level Embeddings for Auditable Radiology (CLEAR) packages the tooling behind the paper
*CLEAR: An Auditable Foundation Model for Radiology Grounded in Clinical Concepts*. The code base
follows a standard Python package layout so it is straightforward to install, extend, and import
from other projects.

## Features

- Thin wrapper around CLIP checkpoints with reliable download and tokenisation utilities.
- Concept-bank utilities for loading the radiological observation space (CLIP and LLM embeddings).
- A streaming zero-shot pipeline that projects images into the LLM concept space without requiring
  the full concept matrix in accelerator memory.
- Metric helpers for threshold search, F1, Matthews correlation coefficient analysis, and
  bootstrapped confidence intervals.
- Example scripts for benchmarking, model auditing, and concept bottleneck experiments.

## Installation

1. Ensure you are running Python 3.9 or newer.
2. (Optional) Create and activate a virtual environment.
3. Install the package in editable mode:

   ```bash
   pip install -e .[visualization]
   ```

   The optional `visualization` extra pulls in matplotlib for plotting utilities.

## Quick Start

The example below loads the CLEAR vision-language backbone and computes
similarity scores between a chest X-ray and radiological text prompts.
Model weights are available at
[huggingface.co/peterhan91/CLEAR](https://huggingface.co/peterhan91/CLEAR).

```python
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from clear.model import CLIP
from clear import tokenize
from examples.train import load_clip

# 1. Load the CLEAR checkpoint (DINOv2 ViT-B/14 + text encoder).
model = load_clip(
    model_path="checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
    use_dinov2=True,
    dinov2_model_name="dinov2_vitb14",
)
model.eval()

# 2. Preprocess a chest X-ray.
preprocess = Compose([
    Resize(448, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(448),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
device = next(model.parameters()).device
image = preprocess(Image.open("path/to/cxr.jpg").convert("RGB")).unsqueeze(0).to(device)

# 3. Encode image and text.
with torch.no_grad():
    image_features = model.encode_image(image)
    text_tokens = tokenize(["pleural effusion", "no pleural effusion"]).to(device)
    text_features = model.encode_text(text_tokens)

    # 4. Cosine similarity → softmax probability.
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = (image_features @ text_features.T).softmax(dim=-1)

print(logits)  # [[prob_positive, prob_negative]]
```

The full CLEAR pipeline projects these concept similarity scores through
LLM-derived semantic embeddings (SFR-Embedding-Mistral) for auditable
zero-shot classification. See `examples/` for benchmarking, concept
bottleneck models, and model auditing scripts.

## Repository Layout

```
src/clear/        Core library modules (CLIP loader, concept bank, zero-shot pipeline, metrics).
examples/         Research scripts and notebooks rewritten to import the packaged modules.
scripts/          Standalone data preparation and legacy utilities.
pyproject.toml    Packaging metadata for building or installing CLEAR.
README.md         This document.
LICENSE           Apache-2.0 licence text.
```

## License

All code in this repository is released under the Apache-2.0 License. Portions derived from the OpenAI
CLIP repository retain the original copyright notice.
