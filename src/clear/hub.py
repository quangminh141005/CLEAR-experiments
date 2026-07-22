"""Official loading and chest-X-ray preprocessing for released CLEAR weights."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import warnings

import torch
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Lambda,
    Normalize,
    Resize,
    ToTensor,
)

from .model import CLIP

__all__ = ["build_cxr_preprocess", "load_pretrained"]


DEFAULT_REPO_ID = "peterhan91/CLEAR"
DEFAULT_CHECKPOINT_FILENAME = "best_model.pt"
CXR_IMAGE_SIZE = 448
CXR_PIXEL_MEAN = 101.48761
CXR_PIXEL_STD = 83.43944


class _DinoV2Visual(nn.Module):
    """Expose DINOv2 under the names used by the published state dictionary."""

    def __init__(self, backbone: nn.Module, output_dim: int = 768) -> None:
        super().__init__()
        backbone_dim = getattr(backbone, "embed_dim", None)
        if backbone_dim is None:
            backbone_dim = getattr(backbone, "num_features", None)
        if backbone_dim is None:
            raise ValueError("Could not determine the DINOv2 backbone embedding dimension")

        self.backbone = backbone
        self.projection = nn.Linear(int(backbone_dim), output_dim)
        self.input_resolution = CXR_IMAGE_SIZE
        self.output_dim = output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(images))

    @property
    def conv1(self) -> nn.Module:
        # ``CLIP.dtype`` uses ``visual.conv1.weight.dtype``. The compatibility
        # property keeps that API intact after replacing the original vision
        # tower with DINOv2.
        return self.projection


def _convert_rgb(image: Any) -> Any:
    if not hasattr(image, "convert"):
        raise TypeError("CLEAR preprocessing expects a PIL image")
    return image.convert("RGB")


def build_cxr_preprocess(image_size: int = CXR_IMAGE_SIZE) -> Compose:
    """Build the preprocessing used for CLEAR chest-X-ray experiments.

    The original experiments normalized 0--255 image tensors with a mean of
    ``101.48761`` and standard deviation of ``83.43944``. ``ToTensor`` scales
    PIL pixels to 0--1, so the values are divided by 255 here. Images are
    resized to an exact square rather than aspect-ratio resized and cropped.
    """

    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    mean = (CXR_PIXEL_MEAN / 255.0,) * 3
    std = (CXR_PIXEL_STD / 255.0,) * 3
    return Compose(
        [
            Lambda(_convert_rgb),
            Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        # Compatibility with PyTorch releases predating ``weights_only``. This
        # warning is intentionally prominent because the fallback uses pickle.
        if "weights_only" not in str(exc):
            raise
        warnings.warn(
            "This PyTorch version does not support safe weights-only loading; "
            f"only continue if you trust checkpoint {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("CLEAR checkpoint must contain a PyTorch state dictionary")

    payload: Mapping[Any, Any] = checkpoint
    for field in ("state_dict", "model_state_dict"):
        candidate = payload.get(field)
        if isinstance(candidate, Mapping):
            payload = candidate
            break

    if not payload or not all(isinstance(key, str) for key in payload):
        raise ValueError("CLEAR state-dictionary keys must be non-empty strings")
    if not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError("CLEAR state dictionary may contain tensors only")

    state_dict = dict(payload)
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    required = {
        "visual.backbone.register_tokens",
        "visual.backbone.patch_embed.proj.weight",
        "visual.projection.weight",
        "positional_embedding",
        "text_projection",
        "token_embedding.weight",
        "ln_final.weight",
    }
    missing = sorted(required.difference(state_dict))
    if missing:
        raise ValueError(
            "Checkpoint is not the released CLEAR DINOv2-register architecture; "
            f"missing keys: {', '.join(missing)}"
        )
    return state_dict


def _text_config(state_dict: Mapping[str, torch.Tensor]) -> dict[str, int]:
    positional_embedding = state_dict["positional_embedding"]
    text_projection = state_dict["text_projection"]
    token_embedding = state_dict["token_embedding.weight"]
    ln_final = state_dict["ln_final.weight"]

    if positional_embedding.ndim != 2 or text_projection.ndim != 2:
        raise ValueError("CLEAR text positional and projection weights must be matrices")
    if token_embedding.ndim != 2 or ln_final.ndim != 1:
        raise ValueError("CLEAR token embedding or final normalization has an invalid shape")

    context_length, transformer_width = map(int, positional_embedding.shape)
    vocab_size, token_width = map(int, token_embedding.shape)
    projection_width, embed_dim = map(int, text_projection.shape)
    if token_width != transformer_width or projection_width != transformer_width:
        raise ValueError("CLEAR checkpoint contains inconsistent text-encoder dimensions")
    if int(ln_final.shape[0]) != transformer_width or transformer_width % 64:
        raise ValueError("CLEAR checkpoint contains an unsupported text-transformer width")

    layer_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("transformer.resblocks.") and key.count(".") >= 3
    }
    if not layer_indices or layer_indices != set(range(max(layer_indices) + 1)):
        raise ValueError("CLEAR checkpoint text-transformer layers are not contiguous")

    return {
        "embed_dim": embed_dim,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "transformer_width": transformer_width,
        "transformer_heads": transformer_width // 64,
        "transformer_layers": len(layer_indices),
    }


def _cached_dinov2_repo() -> Optional[Path]:
    candidate = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    return candidate if (candidate / "hubconf.py").is_file() else None


def _build_dinov2_backbone(*, local_files_only: bool) -> nn.Module:
    cached_repo = _cached_dinov2_repo()
    if cached_repo is not None:
        return torch.hub.load(
            str(cached_repo),
            "dinov2_vitb14_reg",
            source="local",
            pretrained=False,
            verbose=False,
        )
    if local_files_only:
        raise FileNotFoundError(
            "DINOv2 source code is not present in the torch hub cache and "
            "local_files_only=True prevents downloading it"
        )
    return torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14_reg",
        pretrained=False,
        trust_repo=True,
        verbose=False,
    )


def _build_model(
    state_dict: Mapping[str, torch.Tensor],
    *,
    local_files_only: bool,
) -> CLIP:
    config = _text_config(state_dict)
    if config["embed_dim"] != 768:
        raise ValueError(
            "The released DINOv2 ViT-B/14 checkpoint must use 768-dimensional features; "
            f"got {config['embed_dim']}"
        )

    # CLIP owns the text encoder. A deliberately small temporary vision tower
    # avoids allocating a second ViT-B while CLIP is initialized; it is replaced
    # immediately by the published DINOv2-register architecture.
    model = CLIP(
        embed_dim=config["embed_dim"],
        image_resolution=14,
        vision_layers=1,
        vision_width=64,
        vision_patch_size=14,
        context_length=config["context_length"],
        vocab_size=config["vocab_size"],
        transformer_width=config["transformer_width"],
        transformer_heads=config["transformer_heads"],
        transformer_layers=config["transformer_layers"],
    )
    model.visual = _DinoV2Visual(
        _build_dinov2_backbone(local_files_only=local_files_only),
        output_dim=config["embed_dim"],
    )
    return model


def load_pretrained(
    checkpoint_path: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_CHECKPOINT_FILENAME,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    strict: bool = True,
) -> Tuple[CLIP, Callable[[Any], torch.Tensor]]:
    """Load the published CLEAR backbone and its official CXR preprocessing.

    When ``checkpoint_path`` is omitted, ``best_model.pt`` is downloaded from
    ``peterhan91/CLEAR`` on Hugging Face. The checkpoint fully supplies the
    DINOv2 ViT-B/14-register weights; no separate pretrained DINOv2 weights are
    downloaded. ``local_files_only=True`` requires both the checkpoint and the
    DINOv2 torch-hub source to already be cached.
    """

    if checkpoint_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Loading CLEAR from Hugging Face requires huggingface-hub; "
                "install the project dependencies or pass checkpoint_path"
            ) from exc
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
        resolved_checkpoint = Path(downloaded)
    else:
        resolved_checkpoint = Path(checkpoint_path).expanduser()

    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(resolved_checkpoint)

    state_dict = _extract_state_dict(_safe_torch_load(resolved_checkpoint))
    model = _build_model(state_dict, local_files_only=local_files_only)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        warnings.warn(
            "CLEAR checkpoint was loaded non-strictly; missing keys: "
            f"{incompatible.missing_keys}; unexpected keys: {incompatible.unexpected_keys}",
            RuntimeWarning,
            stacklevel=2,
        )

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(target_device)
    model.eval()
    return model, build_cxr_preprocess()
