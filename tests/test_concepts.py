import pickle

import numpy as np
import pytest
import torch

from clear import ConceptBank


def _write_concepts(path) -> None:
    path.write_text("concept,concept_idx\npleural effusion,0\nno pleural effusion,1\n")


@pytest.mark.parametrize("suffix", [".npy", ".npz", ".pt"])
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")
def test_concept_bank_loads_safe_embedding_formats(tmp_path, suffix: str) -> None:
    concepts_path = tmp_path / "mimic_concepts.csv"
    embeddings_path = tmp_path / f"concept_embeddings{suffix}"
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    _write_concepts(concepts_path)

    if suffix == ".npy":
        np.save(embeddings_path, expected)
    elif suffix == ".npz":
        np.savez(embeddings_path, embeddings=expected)
    else:
        torch.save(torch.from_numpy(expected), embeddings_path)

    bank = ConceptBank.from_files(
        concepts_path,
        clip_embeddings_path=embeddings_path,
    )

    assert bank.concepts == ("pleural effusion", "no pleural effusion")
    np.testing.assert_array_equal(bank.ensure_clip_embeddings(), expected)


def test_pickle_loading_requires_explicit_opt_in(tmp_path) -> None:
    concepts_path = tmp_path / "mimic_concepts.csv"
    embeddings_path = tmp_path / "concepts_embeddings_sfr_mistral.pickle"
    _write_concepts(concepts_path)
    with embeddings_path.open("wb") as handle:
        pickle.dump({0: [1.0, 2.0], 1: [3.0, 4.0]}, handle)

    with pytest.raises(ValueError, match="pickle"):
        ConceptBank.from_files(
            concepts_path,
            llm_embeddings_path=embeddings_path,
        )

    with pytest.warns(RuntimeWarning, match="trusted pickle"):
        bank = ConceptBank.from_files(
            concepts_path,
            llm_embeddings_path=embeddings_path,
            allow_unsafe_pickle=True,
        )
    np.testing.assert_array_equal(
        bank.ensure_llm_embeddings(),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_pickle_mapping_indices_must_be_contiguous(tmp_path) -> None:
    concepts_path = tmp_path / "mimic_concepts.csv"
    embeddings_path = tmp_path / "concepts_embeddings_sfr_mistral.pickle"
    _write_concepts(concepts_path)
    with embeddings_path.open("wb") as handle:
        pickle.dump({0: [1.0, 2.0], 2: [3.0, 4.0]}, handle)

    with pytest.warns(RuntimeWarning, match="trusted pickle"):
        with pytest.raises(ValueError, match="contiguous"):
            ConceptBank.from_files(
                concepts_path,
                llm_embeddings_path=embeddings_path,
                allow_unsafe_pickle=True,
            )
