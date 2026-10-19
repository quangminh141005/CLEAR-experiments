"""Atomic metadata and batch checkpoints with strict resume identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def stable_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class PredictionStore:
    """One writer per output directory. Interrupted .tmp files are ignored."""
    def __init__(self, root, identity, *, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = self.root / "identity.json"
        digest = stable_hash(identity)
        if metadata.exists():
            if not resume:
                raise FileExistsError(f"Existing run at {root}; use --resume or a fresh output directory")
            previous = json.loads(metadata.read_text())
            if previous["fingerprint"] != digest:
                raise ValueError("Resume fingerprint differs (cohort, configuration, assets or source code changed)")
        else:
            if any(self.root.iterdir()):
                raise FileExistsError(f"Refusing to overwrite nonempty directory {root}")
            write_json(metadata, {"fingerprint": digest, "identity": identity})
        self.parts = self.root / "prediction_parts"
        self.parts.mkdir(exist_ok=True)
        self.rows = self.load()
        self.next_part = max((int(path.stem.removeprefix("part-")) for path in self.parts.glob("part-*.json")), default=-1) + 1
        self.completed = {row["sample_id"] for row in self.rows}
        if len(self.completed) != len(self.rows):
            raise ValueError("Duplicate sample IDs in prediction parts")

    def load(self):
        rows = []
        for path in sorted(self.parts.glob("part-*.json")):
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
        return rows

    def save_batch(self, rows):
        ids = [r["sample_id"] for r in rows]
        if len(set(ids)) != len(ids) or self.completed.intersection(ids):
            raise ValueError("Batch contains duplicate or already completed sample IDs")
        # Do not rescan all previous files per batch: that becomes quadratic on
        # large CheXpert runs with batch size one.
        write_json(self.parts / f"part-{self.next_part:08d}.json", rows)
        self.next_part += 1
        self.rows.extend(rows)
        self.completed.update(ids)
