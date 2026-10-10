from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

import clear.hub as hub


def _synthetic_state_dict() -> dict[str, torch.Tensor]:
    return {
        "visual.backbone.register_tokens": torch.zeros(1, 4, 768),
        "visual.backbone.patch_embed.proj.weight": torch.zeros(768, 3, 14, 14),
        "visual.projection.weight": torch.zeros(768, 768),
        "positional_embedding": torch.zeros(8, 64),
        "text_projection": torch.zeros(64, 768),
        "token_embedding.weight": torch.zeros(16, 64),
        "ln_final.weight": torch.zeros(64),
        "transformer.resblocks.0.attn.in_proj_weight": torch.zeros(192, 64),
    }


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.loaded_state_dict = None
        self.loaded_strict = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def _mock_model_loading(monkeypatch, checkpoint):
    fake_model = _FakeModel()
    captured = {}

    monkeypatch.setattr(hub, "_safe_torch_load", lambda path: checkpoint)

    def fake_build_model(state_dict, *, local_files_only):
        captured["state_dict"] = state_dict
        captured["local_files_only"] = local_files_only
        return fake_model

    monkeypatch.setattr(hub, "_build_model", fake_build_model)
    return fake_model, captured


def test_extract_state_dict_unwraps_and_removes_data_parallel_prefix() -> None:
    state_dict = _synthetic_state_dict()
    checkpoint = {
        "state_dict": {f"module.{key}": value for key, value in state_dict.items()}
    }

    extracted = hub._extract_state_dict(checkpoint)

    assert extracted.keys() == state_dict.keys()
    assert extracted["text_projection"] is state_dict["text_projection"]


def test_load_pretrained_uses_local_checkpoint_without_hub_download(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.touch()
    checkpoint = {
        "model_state_dict": {
            f"module.{key}": value for key, value in _synthetic_state_dict().items()
        }
    }
    model, captured = _mock_model_loading(monkeypatch, checkpoint)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: pytest.fail("local checkpoint must not access Hugging Face"),
    )

    loaded_model, preprocess = hub.load_pretrained(
        checkpoint_path,
        device="cpu",
        local_files_only=True,
        strict=False,
    )

    assert loaded_model is model
    assert model.loaded_strict is False
    assert model.training is False
    assert model.anchor.device.type == "cpu"
    assert captured["local_files_only"] is True
    assert "module.text_projection" not in captured["state_dict"]
    assert callable(preprocess)


def test_load_pretrained_forwards_hugging_face_download_options(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "downloaded.pt"
    checkpoint_path.touch()
    _mock_model_loading(monkeypatch, _synthetic_state_dict())
    captured = {}

    def fake_hf_hub_download(**kwargs):
        captured.update(kwargs)
        return str(checkpoint_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    hub.load_pretrained(
        repo_id="example/CLEAR",
        filename="weights.pt",
        revision="publication",
        cache_dir=tmp_path / "cache",
        device="cpu",
        local_files_only=True,
    )

    assert captured == {
        "repo_id": "example/CLEAR",
        "filename": "weights.pt",
        "revision": "publication",
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
    }


def test_cxr_preprocess_converts_grayscale_and_uses_published_normalization() -> None:
    preprocess = hub.build_cxr_preprocess(image_size=32)

    output = preprocess(Image.new("L", (12, 20), color=101))

    assert output.shape == (3, 32, 32)
    expected = (101.0 - hub.CXR_PIXEL_MEAN) / hub.CXR_PIXEL_STD
    torch.testing.assert_close(output, torch.full_like(output, expected), atol=1e-6, rtol=0)


@pytest.mark.parametrize("image_size", [0, -1, 32.5, True])
def test_cxr_preprocess_rejects_invalid_sizes(image_size) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        hub.build_cxr_preprocess(image_size=image_size)


def test_legacy_attention_matches_scaled_dot_product():
    from types import SimpleNamespace
    torch.manual_seed(42)
    attention = SimpleNamespace(
        qkv=nn.Linear(16, 48), num_heads=4, scale=0.5,
        attn_drop=0.0, training=False, proj=nn.Linear(16, 16),
        proj_drop=nn.Identity(),
    )
    x = torch.randn(2, 9, 16)
    q, k, v = attention.qkv(x).reshape(2, 9, 3, 4, 4).permute(2, 0, 3, 1, 4).unbind(0)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    reference = attention.proj(reference.transpose(1, 2).reshape(2, 9, 16))
    torch.testing.assert_close(hub._legacy_attention(attention, x), reference)
