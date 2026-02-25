"""
Minimal training utilities for loading CLEAR checkpoints.

This module provides the ``load_clip`` helper used by the example scripts
to instantiate a CLIP model (optionally with a DINOv2 vision encoder) and
load pretrained weights.
"""

import torch
from torch import nn

from clear import clip
from clear.model import CLIP


def load_clip(
    model_path=None,
    pretrained=False,
    context_length=77,
    use_dinov2=False,
    dinov2_model_name="dinov2_vitb14",
    freeze_dinov2=False,
):
    """Load a CLIP model, optionally replacing the vision encoder with DINOv2.

    Parameters
    ----------
    model_path : str, optional
        Path to a ``.pt`` checkpoint.  When provided the state dict is loaded
        into the model after construction.
    pretrained : bool, optional
        If *True* **and** ``use_dinov2`` is *False*, load the OpenAI ViT-B/32
        pretrained CLIP weights.
    context_length : int, optional
        Maximum token length for the text encoder (default 77).
    use_dinov2 : bool, optional
        Replace the default visual encoder with a DINOv2 backbone loaded from
        ``torch.hub``.
    dinov2_model_name : str, optional
        DINOv2 variant name, e.g. ``"dinov2_vitb14"`` (default).
    freeze_dinov2 : bool, optional
        Freeze the DINOv2 backbone parameters.

    Returns
    -------
    model : CLIP
        The constructed (and optionally weight-loaded) CLIP model on the
        appropriate device.
    """
    params = {
        "embed_dim": 768,
        "image_resolution": 320,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 16,
        "context_length": context_length,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if pretrained and not use_dinov2:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else:
        model = CLIP(**params)

        if use_dinov2:
            dinov2_backbone = torch.hub.load(
                "facebookresearch/dinov2",
                dinov2_model_name + "_reg",
                pretrained=True,
            )
            dinov2_backbone = dinov2_backbone.to(device)

            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                features = dinov2_backbone(dummy_input)
                backbone_dim = features.shape[-1]

            class DinoV2Visual(nn.Module):
                def __init__(self, backbone, backbone_dim, output_dim):
                    super().__init__()
                    self.backbone = backbone
                    self.projection = nn.Linear(backbone_dim, output_dim)

                def forward(self, x):
                    features = self.backbone(x)
                    return self.projection(features)

                @property
                def conv1(self):
                    return self.projection

            model.visual = DinoV2Visual(
                dinov2_backbone, backbone_dim, params["embed_dim"]
            )

            if freeze_dinov2:
                for param in model.visual.backbone.parameters():
                    param.requires_grad = False

            print(f"Loaded CLIP model with DinoV2 vision encoder: {dinov2_model_name}")
        else:
            print("Loaded in clip model.")

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    return model
