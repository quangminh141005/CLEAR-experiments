"""Utilities for loading and working with radiological concept banks."""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import csv
import json
import pickle
import warnings

import numpy as np

__all__ = ["ConceptBank"]


class _NumPyCompatUnpickler(pickle.Unpickler):
    """Read NumPy 2 pickles on NumPy 1.x after the caller explicitly opts in."""

    def find_class(self, module: str, name: str) -> Any:
        # NumPy 2 renamed ``numpy.core`` to ``numpy._core``. The published SFR
        # concept file uses the new path, while CLEAR still supports NumPy 1.x.
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def _load_concept_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".jsonl", ".json"}:
        concepts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = json.loads(line.replace("'", '"'))
                if isinstance(payload, dict):
                    value = payload.get("concept") or payload.get("text") or payload.get("observation")
                    if value is None:
                        raise ValueError(f"JSON line must contain a 'concept' field: {line[:80]}")
                    concepts.append(str(value))
                else:
                    concepts.append(str(payload))
        return concepts

    if path.suffix in {".csv", ".tsv"}:
        delimiter = "," if path.suffix == ".csv" else "\t"
        concepts = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            field = reader.fieldnames[0] if reader.fieldnames else None
            if field is None:
                raise ValueError(f"No header row found in {path}")
            for row in reader:
                concepts.append(str(row[field]))
        return concepts

    # default to newline-delimited text file
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _safe_torch_load(path: Path) -> Any:
    """Load tensor-only PyTorch files without enabling arbitrary unpickling."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a core dependency
        raise ImportError("Loading .pt embeddings requires PyTorch") from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        # ``weights_only`` was added after the oldest PyTorch version originally
        # supported by CLEAR. Retain compatibility, but make the security
        # difference explicit instead of silently falling back to pickle.
        if "weights_only" not in str(exc):
            raise
        warnings.warn(
            "This PyTorch version does not support safe weights-only loading; "
            f"only continue if you trust {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu")


def _stack_indexed_embeddings(payload: Mapping[Any, Any], path: Path) -> np.ndarray:
    """Convert the released ``{concept_idx: vector}`` pickle into a matrix."""

    if not payload:
        raise ValueError(f"Embedding dictionary in {path} is empty")
    if not all(isinstance(key, Integral) and not isinstance(key, bool) for key in payload):
        raise ValueError(
            f"Embedding dictionary in {path} must use integer concept indices as keys"
        )

    keys = sorted(int(key) for key in payload)
    missing = next((index for index, key in enumerate(keys) if index != key), None)
    if missing is not None:
        raise ValueError(
            f"Embedding dictionary in {path} must have contiguous keys starting at 0; "
            f"first missing index is {missing}"
        )

    vectors: List[np.ndarray] = []
    vector_dim: Optional[int] = None
    for index in keys:
        vector = np.asarray(payload[index], dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(
                f"Embedding at index {index} in {path} must be one-dimensional; "
                f"got shape {vector.shape}"
            )
        if vector_dim is None:
            vector_dim = int(vector.shape[0])
            if vector_dim == 0:
                raise ValueError(f"Embedding vectors in {path} cannot be empty")
        elif vector.shape[0] != vector_dim:
            raise ValueError(
                f"Embedding at index {index} in {path} has dimension {vector.shape[0]}; "
                f"expected {vector_dim}"
            )
        vectors.append(vector)

    return np.stack(vectors, axis=0)


def _coerce_embedding_payload(payload: Any, path: Path) -> np.ndarray:
    """Extract one numeric embedding matrix from a supported file payload."""

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a core dependency
        torch = None

    if torch is not None and torch.is_tensor(payload):
        array = payload.detach().cpu().numpy()
    elif isinstance(payload, np.ndarray):
        array = payload
    elif isinstance(payload, Mapping):
        for field in ("embeddings", "concept_embeddings"):
            if field in payload:
                return _coerce_embedding_payload(payload[field], path)
        return _stack_indexed_embeddings(payload, path)
    else:
        raise ValueError(
            f"Unsupported embedding payload in {path}: expected a tensor, array, "
            "or an integer-indexed embedding dictionary"
        )

    try:
        array = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Embeddings in {path} must be numeric") from exc
    if array.ndim != 2:
        raise ValueError(
            f"Embeddings in {path} must be a two-dimensional matrix; got shape {array.shape}"
        )
    if 0 in array.shape:
        raise ValueError(f"Embedding matrix in {path} cannot have an empty dimension")
    return array


def _load_array(
    path: Optional[Path],
    *,
    allow_unsafe_pickle: bool = False,
) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return _coerce_embedding_payload(_safe_torch_load(path), path)

    if suffix in {".pickle", ".pkl"}:
        if not allow_unsafe_pickle:
            raise ValueError(
                f"Refusing to unpickle {path} because pickle files can execute arbitrary "
                "code. Pass allow_unsafe_pickle=True only for a file you trust."
            )
        warnings.warn(
            f"Unpickling {path}; only trusted pickle files should be loaded.",
            RuntimeWarning,
            stacklevel=2,
        )
        with path.open("rb") as handle:
            payload = _NumPyCompatUnpickler(handle).load()
        return _coerce_embedding_payload(payload, path)

    # ``allow_pickle=False`` is intentional. It preserves safe loading for
    # NumPy arrays and also supports extensionless legacy .npy/.npz paths.
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            if not data.files:
                raise ValueError(f"NumPy archive {path} contains no arrays")
            if "embeddings" in data:
                array = data["embeddings"]
            else:
                # Preserve the original behavior for unnamed archives.
                array = data[data.files[0]]
        finally:
            data.close()
    else:
        array = data
    return _coerce_embedding_payload(array, path)


@dataclass(slots=True)
class ConceptBank:
    """Container for radiological concepts and associated embeddings.

    Parameters
    ----------
    concepts:
        Sequence of concept strings ordered consistently with the embedding matrices.
    clip_embeddings:
        Optional matrix of CLIP text-encoder features for each concept. Shape must
        be ``(num_concepts, clip_dim)`` when provided.
    llm_embeddings:
        Optional matrix of LLM embeddings for each concept. Shape must be
        ``(num_concepts, llm_dim)`` when provided.
    metadata:
        Optional dictionary of auxiliary metadata (e.g., dataset statistics).
    """

    concepts: Sequence[str]
    clip_embeddings: Optional[np.ndarray] = None
    llm_embeddings: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.concepts = tuple(self.concepts)
        n_concepts = len(self.concepts)

        if self.clip_embeddings is not None:
            self.clip_embeddings = _coerce_embedding_payload(
                self.clip_embeddings, Path("<clip_embeddings>")
            )
            if self.clip_embeddings.shape[0] != n_concepts:
                raise ValueError(
                    "clip_embeddings rows must match number of concepts: "
                    f"got {self.clip_embeddings.shape[0]} rows for {n_concepts} concepts"
                )
        if self.llm_embeddings is not None:
            self.llm_embeddings = _coerce_embedding_payload(
                self.llm_embeddings, Path("<llm_embeddings>")
            )
            if self.llm_embeddings.shape[0] != n_concepts:
                raise ValueError(
                    "llm_embeddings rows must match number of concepts: "
                    f"got {self.llm_embeddings.shape[0]} rows for {n_concepts} concepts"
                )

    @classmethod
    def from_files(
        cls,
        concepts_path: str | Path,
        *,
        clip_embeddings_path: str | Path | None = None,
        llm_embeddings_path: str | Path | None = None,
        metadata: Optional[dict] = None,
        allow_unsafe_pickle: bool = False,
    ) -> "ConceptBank":
        """Load a concept bank from disk.

        ``concepts_path`` may be a JSON Lines file containing objects with a
        ``concept`` field, a CSV/TSV file, or a simple newline-delimited text
        file. Embedding paths may point to ``.npy``/``.npz`` arrays, a
        tensor-only PyTorch ``.pt`` file, or the released integer-indexed
        ``.pickle`` dictionary. Because pickle can execute arbitrary code,
        loading a pickle requires the explicit ``allow_unsafe_pickle=True``
        opt-in and should only be used with a trusted file.
        """

        concepts = _load_concept_list(Path(concepts_path))
        clip_embeddings = (
            _load_array(
                Path(clip_embeddings_path),
                allow_unsafe_pickle=allow_unsafe_pickle,
            )
            if clip_embeddings_path
            else None
        )
        llm_embeddings = (
            _load_array(
                Path(llm_embeddings_path),
                allow_unsafe_pickle=allow_unsafe_pickle,
            )
            if llm_embeddings_path
            else None
        )
        return cls(
            concepts=concepts,
            clip_embeddings=clip_embeddings,
            llm_embeddings=llm_embeddings,
            metadata=metadata or {},
        )

    @property
    def num_concepts(self) -> int:
        return len(self.concepts)

    def has_clip_embeddings(self) -> bool:
        return self.clip_embeddings is not None

    def has_llm_embeddings(self) -> bool:
        return self.llm_embeddings is not None

    def ensure_clip_embeddings(self) -> np.ndarray:
        if self.clip_embeddings is None:
            raise ValueError("ConceptBank does not contain CLIP embeddings")
        return self.clip_embeddings

    def ensure_llm_embeddings(self) -> np.ndarray:
        if self.llm_embeddings is None:
            raise ValueError("ConceptBank does not contain LLM embeddings")
        return self.llm_embeddings

    def subset(self, indices: Sequence[int]) -> "ConceptBank":
        """Return a new ``ConceptBank`` containing only ``indices``."""

        concepts = [self.concepts[i] for i in indices]
        clip_embeddings = None
        llm_embeddings = None
        if self.clip_embeddings is not None:
            clip_embeddings = self.clip_embeddings[np.asarray(indices)]
        if self.llm_embeddings is not None:
            llm_embeddings = self.llm_embeddings[np.asarray(indices)]
        metadata = dict(self.metadata)
        metadata["subset_indices"] = list(indices)
        return ConceptBank(
            concepts=concepts,
            clip_embeddings=clip_embeddings,
            llm_embeddings=llm_embeddings,
            metadata=metadata,
        )

    def iter_chunks(self, chunk_size: int) -> Iterable[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Yield concept embedding chunks to manage memory usage."""

        clip = self.ensure_clip_embeddings()
        llm = self.llm_embeddings
        for start in range(0, self.num_concepts, chunk_size):
            stop = min(start + chunk_size, self.num_concepts)
            clip_chunk = clip[start:stop]
            llm_chunk = llm[start:stop] if llm is not None else None
            yield clip_chunk, llm_chunk
