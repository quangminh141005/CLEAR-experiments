"""Run a single-image CLEAR backbone and concept-retrieval demonstration.

The default path downloads only the released model checkpoint and compares an
image with positive/negative text prompts. Passing ``--top-concepts`` also
downloads the 768-dimensional concept matrix and reports the highest-scoring
radiological observations. The latter artifact is approximately 1.1 GB.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence
import warnings

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from clear import load_pretrained, tokenize


DEFAULT_REPO_ID = "peterhan91/CLEAR"


def _load_concepts(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {path}")
        field = "concept" if "concept" in reader.fieldnames else reader.fieldnames[0]
        return [str(row[field]) for row in reader]


def _load_tensor(path: str | Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions before weights_only.
        warnings.warn(
            "This PyTorch version cannot use weights-only loading; continue only "
            f"if you trust {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        payload = torch.load(path, map_location="cpu")
    if not torch.is_tensor(payload) or payload.ndim != 2:
        raise ValueError(f"Expected one two-dimensional tensor in {path}")
    return payload.float()


@torch.inference_mode()
def _rank_concepts(
    image_features: torch.Tensor,
    concept_embeddings: torch.Tensor,
    concepts: Sequence[str],
    *,
    top_k: int,
    chunk_size: int,
) -> list[tuple[str, float]]:
    if concept_embeddings.shape[0] != len(concepts):
        raise ValueError(
            "Concept rows and embedding rows are misaligned: "
            f"{len(concepts)} concepts versus {concept_embeddings.shape[0]} embeddings"
        )

    device = image_features.device
    best_scores = torch.empty(0, device=device)
    best_indices = torch.empty(0, dtype=torch.long, device=device)

    for start in range(0, concept_embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, concept_embeddings.shape[0])
        chunk = F.normalize(concept_embeddings[start:stop].to(device), dim=-1)
        scores = (image_features @ chunk.T).squeeze(0)
        local_k = min(top_k, scores.numel())
        values, indices = scores.topk(local_k)
        indices = indices + start

        best_scores = torch.cat((best_scores, values))
        best_indices = torch.cat((best_indices, indices))
        keep = min(top_k, best_scores.numel())
        best_scores, order = best_scores.topk(keep)
        best_indices = best_indices[order]

    return [
        (concepts[index], float(score))
        for index, score in zip(best_indices.cpu().tolist(), best_scores.cpu().tolist())
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a chest X-ray image")
    parser.add_argument(
        "--prompts",
        nargs=2,
        default=("pleural effusion", "no pleural effusion"),
        metavar=("POSITIVE", "NEGATIVE"),
        help="Positive and negative text prompts for the backbone demonstration",
    )
    parser.add_argument(
        "--top-concepts",
        type=int,
        default=0,
        metavar="N",
        help="Also retrieve the N highest-scoring CLEAR observations",
    )
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    args = parser.parse_args()
    if args.top_concepts < 0:
        parser.error("--top-concepts must be non-negative")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def main() -> None:
    args = _parse_args()
    model, preprocess = load_pretrained(repo_id=args.repo_id, device=args.device)
    with Image.open(args.image) as source_image:
        image = preprocess(source_image).unsqueeze(0).to(args.device)

    with torch.inference_mode():
        image_features = F.normalize(model.encode_image(image), dim=-1)
        prompt_tokens = tokenize(list(args.prompts)).to(args.device)
        prompt_features = F.normalize(model.encode_text(prompt_tokens), dim=-1)
        pair_scores = (image_features @ prompt_features.T).softmax(dim=-1).squeeze(0)

    print("Backbone paired-prompt scores (not calibrated clinical probabilities):")
    for prompt, score in zip(args.prompts, pair_scores.cpu().tolist()):
        print(f"  {prompt}: {score:.4f}")

    if not args.top_concepts:
        return

    concepts_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="mimic_concepts.csv",
    )
    embeddings_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="concept_embeddings_368294.pt",
    )
    concepts = _load_concepts(concepts_path)
    concept_embeddings = _load_tensor(embeddings_path)
    ranked = _rank_concepts(
        image_features,
        concept_embeddings,
        concepts,
        top_k=args.top_concepts,
        chunk_size=args.chunk_size,
    )

    print(f"\nTop {args.top_concepts} CLEAR radiological observations:")
    for rank, (concept, score) in enumerate(ranked, start=1):
        print(f"  {rank:>2}. {concept} ({score:.4f})")


if __name__ == "__main__":
    main()
