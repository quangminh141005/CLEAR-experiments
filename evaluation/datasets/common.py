"""Shared annotation semantics and deterministic patient grouping."""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re

import numpy as np
import pandas as pd


def normalized_name(value):
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def canonical(value, aliases=None):
    value = normalized_name(value)
    mapping = {"pulmonary atelectasis": "atelectasis", "effusion": "pleural effusion"}
    mapping.update(aliases or {})
    return mapping.get(value, value)


def tag_list(value):
    """None means missing; [] means an explicitly supplied empty annotation list."""
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    text = str(value).strip()
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, (list, tuple)) or not all(isinstance(x, str) for x in parsed):
            raise ValueError(f"Invalid annotation list: {text[:120]}")
        return list(parsed)
    return [part.strip() for part in text.split(";") if part.strip()]


def tag_targets(tags, labels, *, unmentioned="unknown", normal="unknown", aliases=None):
    if tags is None:
        return {label: None for label in labels}
    present = {canonical(x, aliases) for x in tags}
    assume_negative = unmentioned == "negative" or (normal == "negative" and bool(present & {"normal", "no finding"}))
    return {label: 1 if label in present else (0 if assume_negative else None) for label in labels}


def numeric_target(value, uncertain="ignore", missing="ignore"):
    if pd.isna(value) or str(value).strip() == "":
        return 0 if missing == "negative" else None
    number = float(value)
    if number == -1:
        return {"ignore": None, "negative": 0, "positive": 1}[uncertain]
    if number not in (0, 1):
        raise ValueError(f"Unexpected numeric label {value!r}; expected 0, 1, -1 or blank")
    return int(number)


@dataclass
class Record:
    sample_id: str
    patient_id: str
    study_id: str
    image_path: str
    labels: dict
    source_labels: object = None
    view: str = ""


@dataclass
class Cohort:
    records: list[Record]
    labels: list[str]
    metadata: dict = field(default_factory=dict)

    def select(self, *, split="all", validation_fraction=0.2, seed=42, subset_size=None):
        if not 0 < validation_fraction < 1:
            raise ValueError("validation-fraction must be between 0 and 1")
        if split != "all" and not self.metadata.get("patient_ids_verified", False):
            raise ValueError("Patient identifiers are not verified. Supply --patient-map before constructing splits.")
        selected = self.records
        if split != "all":
            def is_validation(record):
                key = f"{seed}:{record.patient_id}".encode()
                return int(hashlib.sha256(key).hexdigest()[:16], 16) / 2**64 < validation_fraction
            selected = [r for r in selected if is_validation(r) == (split == "validation")]
        # Sample groups, then retain all images from chosen groups (size is a patient budget).
        if subset_size is not None:
            if subset_size < 1:
                raise ValueError("subset-size must be positive")
            patients = sorted({r.patient_id for r in selected})
            rng = np.random.default_rng(seed)
            chosen = set(rng.choice(patients, min(subset_size, len(patients)), replace=False))
            selected = [r for r in selected if r.patient_id in chosen]
        selected = sorted(selected, key=lambda r: r.sample_id)
        if not selected:
            raise ValueError("No images remain after cohort selection")
        if len({r.sample_id for r in selected}) != len(selected):
            raise ValueError("Duplicate sample IDs in annotation file")
        return Cohort(selected, self.labels, {**self.metadata, "split": split,
            "validation_fraction": validation_fraction, "seed": seed, "subset_patient_budget": subset_size,
            "selected_images": len(selected), "selected_patients_or_groups": len({r.patient_id for r in selected})})

    def payload(self):
        return {"labels": self.labels, "metadata": self.metadata, "records": [asdict(r) for r in self.records]}


class ImageIndex:
    """Resolve archive prefixes by suffix; ambiguous basenames are never guessed."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.by_name = {}
        self.by_suffix = {}
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                self.by_name.setdefault(path.name, []).append(path)
                parts = path.relative_to(self.root).parts
                for start in range(len(parts) - 1):
                    key = "/".join(parts[start:])
                    self.by_suffix[key] = None if key in self.by_suffix else path

    def resolve(self, relative):
        relative = str(relative).replace("\\", "/")
        path = Path(relative)
        direct = self.root / path
        if direct.is_file() and direct.resolve().is_relative_to(self.root):
            return str(direct.resolve())
        candidates = self.by_name.get(path.name, [])
        parts = path.parts
        # Never drop a CheXpert patient ID to guess from another patient's study
        # or common view filename. Missing subset images should remain missing.
        patient_start = next((i for i, part in enumerate(parts) if re.fullmatch(r"patient[^/]+", part)), None)
        limit = patient_start + 1 if patient_start is not None else max(0, len(parts) - 1)
        for start in range(limit):
            suffix = "/".join(parts[start:])
            if suffix in self.by_suffix:
                match = self.by_suffix[suffix]
                if match is None:
                    raise ValueError(f"Ambiguous image suffix {suffix}")
                return str(match)
        if patient_start is not None:
            return None
        if len(candidates) == 1:
            return str(candidates[0])
        if candidates:
            raise ValueError(f"Ambiguous image path {relative}: {len(candidates)} candidates")
        return None


def find_annotation(root, pattern, explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    paths = sorted(Path(root).rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} under {root}, found {len(paths)}. Supply --annotations.")
    return paths[0]


def patient_mapping(path):
    if not path:
        return None
    frame = pd.read_csv(path, dtype=str)
    if frame.study_id.duplicated().any() or frame.patient_id.isna().any():
        raise ValueError("patient-map requires unique study_id and nonempty patient_id")
    return dict(zip(frame.study_id, frame.patient_id))


def finish(records, labels, metadata, options):
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("No valid finding names; choose --labels or verify native annotations")
    missing = metadata.get("missing_images", 0)
    if missing and options.get("missing_images", "error") == "error":
        raise FileNotFoundError(f"{missing} annotated images are absent. For a sample archive use --missing-images skip.")
    if not records:
        raise ValueError("No annotated images found; inspect annotation and image roots")
    return Cohort(records, labels, metadata)
