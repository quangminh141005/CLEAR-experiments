import pytest
import torch

from examples.quickstart import _rank_concepts


def test_rank_concepts_streams_chunks_and_returns_global_top_k() -> None:
    image_features = torch.tensor([[1.0, 0.0]])
    concept_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [-1.0, 0.0],
        ]
    )
    concepts = ["first", "second", "third", "fourth"]

    ranked = _rank_concepts(
        image_features,
        concept_embeddings,
        concepts,
        top_k=2,
        chunk_size=2,
    )

    assert [concept for concept, _ in ranked] == ["first", "third"]
    assert ranked[0][1] == pytest.approx(1.0)


def test_rank_concepts_rejects_misaligned_artifacts() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        _rank_concepts(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            ["first", "second"],
            top_k=1,
            chunk_size=1,
        )
