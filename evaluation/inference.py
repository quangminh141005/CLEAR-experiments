"""Adapter over the released ConceptBank and ZeroShotPipeline; FP32, no fitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .storage import file_hash, stable_hash


ASSET_FILES = ["best_model.pt", "mimic_concepts.csv", "concept_embeddings_368294.pt",
               "concepts_embeddings_sfr_mistral.pickle"]


def validate_matrix(matrix, dimension, name, chunk_size=8192):
    if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N,{dimension}); got {matrix.shape}")
    for start in range(0, len(matrix), chunk_size):
        block = matrix[start:start + chunk_size]
        if not np.isfinite(block).all() or (np.linalg.norm(block, axis=1) <= 1e-8).any():
            raise ValueError(f"{name} contains nonfinite or zero embeddings")


def observation_list(path):
    import pandas as pd
    frame = pd.read_csv(path)
    if "concept" not in frame:
        raise ValueError("Observation CSV must contain an explicit concept column")
    if frame.concept.isna().any() or frame.concept.astype(str).str.strip().eq("").any():
        raise ValueError("Missing observation text")
    if "concept_idx" in frame:
        values = frame.concept_idx.to_numpy()
        if not np.array_equal(values, np.arange(len(frame))):
            raise ValueError("concept_idx must be 0..N-1 in file order; refusing to reorder assets")
    concepts = frame.concept.astype(str).tolist()
    if len(set(concepts)) != len(concepts):
        raise ValueError("Duplicate observation text in bank")
    return concepts


def inspect_assets(root, mode="clear", checkpoint=None):
    root = Path(root)
    manifest_path = root / "assets.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing assets.json. Run scripts/prepare_clear_assets.py on the server first.")
    manifest = json.loads(manifest_path.read_text())
    names = ASSET_FILES if mode == "clear" else ASSET_FILES[:1]
    hashes = {}
    for name in names:
        path = Path(checkpoint) if name == "best_model.pt" and checkpoint else root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = file_hash(path)
        expected = manifest["sha256"].get(name)
        if not expected or expected != digest:
            raise ValueError(f"Asset checksum mismatch for {path}; register the correct official snapshot")
        hashes[name] = digest
    return {"repository": manifest["repository"], "revision": manifest["revision"], "sha256": hashes,
            "observation_order_sha256": manifest.get("observation_order_sha256"),
            "expected_concepts": manifest.get("expected_concepts", 368294)}


def read_prompt_cache(path, pairs):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        texts = data["texts"].tolist()
        embeddings = data["embeddings"].astype(np.float32)
    if metadata.get("model") != "Salesforce/SFR-Embedding-Mistral" or metadata.get("pooling") != "last-token-l2":
        raise ValueError("Prompt cache must use SFR-Embedding-Mistral last-token pooling")
    revision = metadata.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("Prompt cache must record the immutable SFR model revision")
    if metadata.get("format") not in {"raw", "instruction"}:
        raise ValueError("Prompt cache must record its raw/instruction input recipe")
    if len(texts) != len(set(texts)) or len(texts) != len(embeddings):
        raise ValueError("Duplicate/misaligned prompt cache rows")
    validate_matrix(embeddings, 4096, "prompt embeddings")
    if metadata.get("texts_sha256") != stable_hash(texts):
        raise ValueError("Prompt text order checksum differs")
    lookup = dict(zip(texts, embeddings))
    missing = [text for pair in pairs for text in pair if text not in lookup]
    if missing:
        raise ValueError(f"Prompt cache lacks {missing[:5]}; regenerate it for this cohort's findings")
    return np.stack([lookup[p] for p, _ in pairs]), np.stack([lookup[n] for _, n in pairs]), metadata


class InferenceEngine:
    def __init__(self, *, assets_dir, labels, pairs, device="cpu", mode="clear", chunk_size=512,
                 prompt_cache=None, allow_unsafe_pickle=False, checkpoint=None):
        import torch
        from clear import load_pretrained, tokenize
        from clear.concepts import ConceptBank, _load_array
        from clear.pipeline import ZeroShotPipeline
        from torch.nn.functional import normalize

        if chunk_size < 1:
            raise ValueError("concept-chunk-size must be positive")
        self.torch, self.normalize = torch, normalize
        self.device, self.labels, self.mode = torch.device(device), labels, mode
        root = Path(assets_dir)
        # inspect_assets is performed by the CLI before opening a resumable run.
        self.model, self.preprocess = load_pretrained(checkpoint or root / "best_model.pt",
                                                     device=self.device, local_files_only=True)
        self.model.float().eval()
        if mode == "clear":
            concepts = observation_list(root / "mimic_concepts.csv")
            manifest = json.loads((root / "assets.json").read_text())
            if len(concepts) != manifest["expected_concepts"] or stable_hash(concepts) != manifest["observation_order_sha256"]:
                raise ValueError("Observation count or ordering differs from registered asset snapshot")
            clip = _load_array(root / "concept_embeddings_368294.pt")
            llm = _load_array(root / "concepts_embeddings_sfr_mistral.pickle", allow_unsafe_pickle=allow_unsafe_pickle)
            validate_matrix(clip, 768, "CLEAR observation embeddings")
            validate_matrix(llm, 4096, "SFR observation embeddings")
            bank = ConceptBank(concepts, clip_embeddings=clip, llm_embeddings=llm)
            self.pipeline = ZeroShotPipeline(self.model, bank, concept_chunk_size=chunk_size, device=self.device)
            pos, neg, self.prompt_metadata = read_prompt_cache(prompt_cache, pairs)
            self.positive = normalize(torch.from_numpy(pos).to(self.device), dim=-1)
            self.negative = normalize(torch.from_numpy(neg).to(self.device), dim=-1)
        else:
            self.prompt_metadata = {"encoder": "CLEAR paired text tower; direct baseline"}
            def encode_texts(texts):
                chunks = []
                for start in range(0, len(texts), 16):
                    tokens = tokenize(texts[start:start + 16]).to(self.device)
                    chunks.append(normalize(self.model.encode_text(tokens).float(), dim=-1))
                return torch.cat(chunks)
            with torch.no_grad():
                self.positive = encode_texts([p for p, _ in pairs])
                self.negative = encode_texts([n for _, n in pairs])

    def predict(self, images):
        torch = self.torch
        with torch.no_grad():
            if self.mode == "clear":
                features = self.pipeline.encode_images(images)
                representation = self.pipeline.project_to_concept_space(features)
            else:
                representation = self.normalize(self.model.encode_image(images.to(self.device)).float(), dim=-1)
            positive = representation @ self.positive.T
            negative = representation @ self.negative.T
            scores = torch.stack((positive, negative), dim=-1).softmax(dim=-1)[..., 0]
        return scores.cpu().numpy(), positive.cpu().numpy(), negative.cpu().numpy()


def load_image(path, preprocess, image_policy="8bit"):
    """Reject implicit clipping of native 16-bit PadChest PNGs."""
    from PIL import Image
    with Image.open(path) as source:
        if source.mode.startswith("I") or source.mode == "F":
            array = np.asarray(source)
            if image_policy != "uint16-linear":
                raise ValueError(f"High-bit-depth image {path}; explicitly select --image-policy uint16-linear or prepare documented 8-bit inputs")
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 65535:
                raise ValueError(f"Image {path} is outside unsigned 16-bit range")
            source = Image.fromarray(np.rint(array.astype(np.float32) / 257).astype(np.uint8))
        return preprocess(source.convert("RGB"))
