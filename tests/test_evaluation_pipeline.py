"""Synthetic-only tests. No network, downloaded weights, or real model execution."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.datasets import chexpert, indiana, padchest
from evaluation.datasets.common import canonical, numeric_target, tag_targets
from evaluation.inference import observation_list, read_prompt_cache, validate_matrix
from evaluation.metrics import binary_metrics, evaluate_rows
from evaluation.reporting import SECTIONS, generate
from evaluation.storage import PredictionStore, stable_hash


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), "gray").save(path)


@pytest.fixture
def iu(tmp_path):
    root = tmp_path / "iu"
    root.mkdir()
    pd.DataFrame([{"uid": "1", "Problems": "Pulmonary Atelectasis;Cardiomegaly"},
                  {"uid": "2", "Problems": "normal"}, {"uid": "3", "Problems": None}]).to_csv(root / "indiana_reports.csv", index=False)
    views = []
    for uid in ("1", "2", "3"):
        for view in ("Frontal", "Lateral"):
            name = f"{uid}-{view}.png"
            image(root / "images" / name)
            views.append({"uid": uid, "filename": name, "projection": view})
    pd.DataFrame(views).to_csv(root / "indiana_projections.csv", index=False)
    return root


def test_indiana_mapping_and_missing_semantics(iu):
    cohort = indiana.load(iu, labels=["atelectasis", "pneumonia"])
    assert len(cohort.records) == 6
    assert cohort.records[0].labels == {"atelectasis": 1, "pneumonia": None}
    negative = indiana.load(iu, labels=["atelectasis"], unmentioned_policy="negative")
    assert {r.labels["atelectasis"] for r in negative.records if r.study_id == "2"} == {0}
    assert {r.labels["atelectasis"] for r in negative.records if r.study_id == "3"} == {None}
    assert canonical("Pulmonary Atelectasis") == "atelectasis"
    assert tag_targets(None, ["atelectasis"], unmentioned="negative") == {"atelectasis": None}


def test_patient_split_and_subset(iu, tmp_path):
    with pytest.raises(ValueError, match="Patient identifiers"):
        indiana.load(iu).select(split="test")
    mapping = tmp_path / "patients.csv"
    pd.DataFrame({"study_id": ["1", "2", "3"], "patient_id": ["p1", "p1", "p2"]}).to_csv(mapping, index=False)
    cohort = indiana.load(iu, patient_map=mapping)
    a, b = cohort.select(subset_size=1), cohort.select(subset_size=1)
    assert a.payload() == b.payload()
    selected = {r.patient_id for r in a.records}
    assert len(selected) == 1
    assert len(a.records) == sum(r.patient_id in selected for r in cohort.records)
    # Use many synthetic groups so both deterministic split partitions have support.
    from evaluation.datasets.common import Cohort, Record
    grouped = Cohort([Record(str(i), f"p{i//2}", str(i), "mock.png", {"a": 0}) for i in range(80)], ["a"], {"patient_ids_verified": True})
    validation = grouped.select(split="validation")
    test = grouped.select(split="test")
    assert not ({r.patient_id for r in validation.records} & {r.patient_id for r in test.records})
    assert len(validation.records) + len(test.records) == len(grouped.records)


def test_padchest_subset_and_annotations(tmp_path):
    image(tmp_path / "nested" / "a.png")
    image(tmp_path / "nested" / "b.png")
    path = tmp_path / "labels.csv"
    pd.DataFrame([{"ImageID": "a.png", "PatientID": "p1", "Labels": "['atelectasis']"},
                  {"ImageID": "b.png", "PatientID": "p2", "Labels": None},
                  {"ImageID": "absent.png", "PatientID": "p3", "Labels": "[]"}]).to_csv(path, index=False)
    cohort = padchest.load(tmp_path, ["atelectasis", "cardiomegaly"], missing_images="skip", unmentioned_policy="negative")
    assert cohort.metadata["missing_images"] == 1
    assert cohort.records[0].labels == {"atelectasis": 1, "cardiomegaly": 0}
    assert cohort.records[1].labels == {"atelectasis": None, "cardiomegaly": None}
    with pytest.raises(FileNotFoundError):
        padchest.load(tmp_path, ["atelectasis"])


def test_chexpert_paths_and_uncertainty(tmp_path):
    relative = "CheXpert-v1.0-small/valid/patient00001/study1/view1_frontal.jpg"
    image(tmp_path / "valid/patient00001/study1/view1_frontal.jpg")
    pd.DataFrame([{"Path": relative, "Frontal/Lateral": "Frontal", "Atelectasis": -1,
                  "Cardiomegaly": 0, "Edema": None}]).to_csv(tmp_path / "valid.csv", index=False)
    cohort = chexpert.load(tmp_path, ["atelectasis", "cardiomegaly", "edema"])
    assert cohort.records[0].patient_id == "patient00001"
    assert cohort.records[0].labels == {"atelectasis": None, "cardiomegaly": 0, "edema": None}
    assert numeric_target(-1, "positive") == 1
    assert numeric_target(-1, "negative") == 0
    assert numeric_target(None, missing="negative") == 0
    with pytest.raises(ValueError):
        numeric_target(2)


def test_chexpert_missing_patient_path_never_matches_another_patient(tmp_path):
    from evaluation.datasets.common import ImageIndex
    image(tmp_path / "wrapper/valid/patient00001/study1/view1_frontal.jpg")
    index = ImageIndex(tmp_path)
    assert index.resolve("CheXpert/valid/patient00001/study1/view1_frontal.jpg") is not None
    assert index.resolve("CheXpert/valid/patient99999/study1/view1_frontal.jpg") is None


def test_metric_correctness_and_masking():
    result = binary_metrics([0, 0, 1, 1, None], [.1, .8, .4, .9, .99])
    assert result["auroc"] == .75
    assert result["ap"] == pytest.approx(5 / 6)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["f1"] == result["recall"] == result["specificity"] == .5
    assert result["missing"] == 1
    assert binary_metrics([1, 1], [.2, .8])["auroc"] is None
    assert binary_metrics([0, 0], [.2, .8])["ap"] is None
    assert binary_metrics([None], [.2])["auroc"] is None
    with pytest.raises(ValueError):
        binary_metrics([-1, 0], [.2, .8])


def rows_fixture():
    return [{"sample_id": str(i), "patient_id": f"p{i}", "study_id": str(i), "image_path": "synthetic.png",
             "labels": {"a": i % 2, "empty": None}, "scores": {"a": .1 if i % 2 == 0 else .9, "empty": .4}}
            for i in range(12)]


def test_cluster_bootstrap_and_undefined_macro():
    rows = rows_fixture()
    first = evaluate_rows(rows, ["a", "empty"], bootstrap=30, seed=3)
    assert first == evaluate_rows(rows, ["a", "empty"], bootstrap=30, seed=3)
    assert first["macro_auroc"] == 1
    assert first["valid_auroc_classes"] == ["a"]
    assert first["per_class"]["empty"]["ci95"]["auroc"]["low"] is None
    assert first["macro_auroc_ci95"]["low"] == first["macro_auroc_ci95"]["high"] == 1
    assert evaluate_rows(rows, ["empty"], bootstrap=0)["macro_auroc"] is None


def test_prediction_resume_and_identity(tmp_path):
    root = tmp_path / "run"
    store = PredictionStore(root, {"seed": 42})
    rows = rows_fixture()
    store.save_batch(rows[:2])
    (root / "prediction_parts/part-00000001.json.tmp").write_text("interrupted")
    resumed = PredictionStore(root, {"seed": 42}, resume=True)
    assert resumed.rows == rows[:2]
    assert resumed.completed == {"0", "1"}
    resumed.save_batch(rows[2:])
    assert PredictionStore(root, {"seed": 42}, resume=True).rows == rows
    with pytest.raises(ValueError, match="fingerprint"):
        PredictionStore(root, {"seed": 43}, resume=True)
    with pytest.raises(FileExistsError):
        PredictionStore(root, {"seed": 42})
    with pytest.raises(ValueError, match="duplicate"):
        resumed.save_batch(rows[:1])


def test_observation_dimensions_and_alignment(tmp_path):
    path = tmp_path / "concepts.csv"
    pd.DataFrame({"concept_idx": [0, 1], "concept": ["first", "second"]}).to_csv(path, index=False)
    assert observation_list(path) == ["first", "second"]
    pd.DataFrame({"concept_idx": [1, 0], "concept": ["second", "first"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="refusing to reorder"):
        observation_list(path)
    with pytest.raises(ValueError):
        validate_matrix(np.ones((2, 768)), 4096, "SFR")
    with pytest.raises(ValueError):
        validate_matrix(np.zeros((2, 4096)), 4096, "SFR")
    from clear.concepts import ConceptBank, _stack_indexed_embeddings
    with pytest.raises(ValueError, match="rows must match"):
        ConceptBank(["a"], clip_embeddings=np.ones((2, 3)))
    aligned = _stack_indexed_embeddings({1: np.array([0, 1]), 0: np.array([1, 0])}, path)
    np.testing.assert_array_equal(aligned, np.eye(2))
    with pytest.raises(ValueError, match="contiguous"):
        _stack_indexed_embeddings({1: np.ones(2)}, path)


def test_cached_prompt_text_alignment(tmp_path):
    path = tmp_path / "prompts.npz"
    texts = ["no a", "a"]
    embeddings = np.ones((2, 4096), dtype=np.float32)
    embeddings[1] *= 2
    metadata = {"model": "Salesforce/SFR-Embedding-Mistral", "pooling": "last-token-l2", "texts_sha256": stable_hash(texts),
                "revision": "a" * 40, "format": "raw"}
    np.savez(path, texts=texts, embeddings=embeddings, metadata=json.dumps(metadata))
    positive, negative, _ = read_prompt_cache(path, [("a", "no a")])
    assert positive[0, 0] == 2 and negative[0, 0] == 1
    with pytest.raises(ValueError, match="lacks"):
        read_prompt_cache(path, [("b", "no b")])


def test_official_chunk_projection_matches_dense_formula():
    import torch
    from clear.concepts import ConceptBank
    from clear.pipeline import ZeroShotPipeline

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def encode_image(self, inputs):
            return inputs

    clip = np.array([[1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
    llm = np.array([[1., 2., 3.], [3., 0., 2.], [2., 3., 1.]], dtype=np.float32)
    bank = ConceptBank(["a", "b", "c"], clip, llm)
    model = MockModel()
    image_features = torch.nn.functional.normalize(torch.tensor([[1., 2.], [3., 4.]]), dim=-1)
    expected = torch.nn.functional.normalize((image_features @ torch.nn.functional.normalize(torch.from_numpy(clip), dim=-1).T) @ torch.from_numpy(llm), dim=-1)
    for size in (1, 2, 5):
        pipeline = ZeroShotPipeline(model, bank, device=torch.device("cpu"), concept_chunk_size=size)
        torch.testing.assert_close(pipeline.project_to_concept_space(image_features), expected)


def test_full_adapter_uses_sfr_prompts_and_all_observations(tmp_path, monkeypatch):
    import pickle
    import torch
    import clear
    from evaluation.inference import InferenceEngine

    concepts = ["observation a", "observation b", "observation c"]
    pd.DataFrame({"concept_idx": [0, 1, 2], "concept": concepts}).to_csv(tmp_path / "mimic_concepts.csv", index=False)
    clip = np.zeros((3, 768), dtype=np.float32)
    clip[0, 0], clip[1, 1], clip[2, :2] = 1, 1, [1, 1]
    llm = np.zeros((3, 4096), dtype=np.float32)
    llm[0, :3], llm[1, :3], llm[2, :3] = [1, 0, 1], [0, 1, 1], [1, 1, 0]
    torch.save(torch.from_numpy(clip), tmp_path / "concept_embeddings_368294.pt")
    with (tmp_path / "concepts_embeddings_sfr_mistral.pickle").open("wb") as handle:
        pickle.dump({2: llm[2], 0: llm[0], 1: llm[1]}, handle)
    (tmp_path / "assets.json").write_text(json.dumps({"expected_concepts": 3, "observation_order_sha256": stable_hash(concepts)}))
    prompts = np.zeros((2, 4096), dtype=np.float32)
    prompts[0, 0], prompts[1, 1] = 1, 1
    texts = ["a", "no a"]
    metadata = {"model": "Salesforce/SFR-Embedding-Mistral", "revision": "a" * 40,
                "pooling": "last-token-l2", "format": "raw", "texts_sha256": stable_hash(texts)}
    cache = tmp_path / "prompts.npz"
    np.savez(cache, texts=texts, embeddings=prompts, metadata=json.dumps(metadata))

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
        def encode_image(self, images):
            return images
        def encode_text(self, tokens):
            raise AssertionError("Full CLEAR diagnostic prompts must not use the 768-d text tower")

    monkeypatch.setattr(clear, "load_pretrained", lambda *a, **k: (MockModel(), None))
    with pytest.warns(RuntimeWarning, match="Unpickling"):
        engine = InferenceEngine(assets_dir=tmp_path, labels=["a"], pairs=[("a", "no a")],
                                 prompt_cache=cache, allow_unsafe_pickle=True, chunk_size=1)
    inputs = torch.zeros(1, 768)
    inputs[0, :2] = torch.tensor([1., 2.])
    scores, positive, negative = engine.predict(inputs)
    normalized = torch.nn.functional.normalize(inputs, dim=-1)
    expected_representation = torch.nn.functional.normalize(
        (normalized @ torch.nn.functional.normalize(torch.from_numpy(clip), dim=-1).T) @ torch.from_numpy(llm), dim=-1)
    expected_logits = expected_representation[:, :2]
    np.testing.assert_allclose(positive[:, 0], expected_logits[:, 0].numpy(), rtol=1e-5)
    np.testing.assert_allclose(negative[:, 0], expected_logits[:, 1].numpy(), rtol=1e-5)
    np.testing.assert_allclose(scores[:, 0], expected_logits.softmax(dim=-1)[:, 0].numpy(), rtol=1e-5)


def test_high_bit_depth_preprocessing_is_explicit(tmp_path):
    from evaluation.inference import load_image
    path = tmp_path / "native16.png"
    Image.fromarray(np.array([[0, 65535]], dtype=np.uint16)).save(path)
    with pytest.raises(ValueError, match="High-bit-depth"):
        load_image(path, lambda x: np.asarray(x))
    converted = load_image(path, lambda x: np.asarray(x), "uint16-linear")
    np.testing.assert_array_equal(converted[0, :, 0], [0, 255])


def test_threshold_provenance_and_overlap(tmp_path):
    from evaluation.calibration import load_thresholds
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"dataset": "chexpert", "inference_signature": "same",
                               "thresholds": {"a": .6}, "validation_patient_ids": ["p1"]}))
    run = {"dataset": "chexpert", "inference_signature": "same"}
    assert load_thresholds(path, run, [{"patient_id": "p2"}])[0] == {"a": .6}
    with pytest.raises(ValueError, match="overlap"):
        load_thresholds(path, run, [{"patient_id": "p1"}])
    with pytest.raises(ValueError, match="different"):
        load_thresholds(path, {**run, "inference_signature": "other"}, [{"patient_id": "p2"}])


def test_missing_report_generation(tmp_path):
    from docx import Document
    markdown, word = generate({}, tmp_path / "reports")
    text = markdown.read_text()
    assert all(title in text for title in SECTIONS)
    assert "Chưa thực hiện" in text
    assert "0.000" not in text
    assert len(Document(word).tables) >= 2
    with pytest.raises(FileExistsError):
        generate({}, tmp_path / "reports")


def test_cli_mocked_inference_resume_and_report(iu, tmp_path, monkeypatch):
    import torch
    import evaluate
    output = tmp_path / "run"
    calls = []

    class MockEngine:
        fail = True
        def __init__(self, **kwargs):
            self.labels = kwargs["labels"]
            self.preprocess = lambda image: torch.zeros(3, 12, 12)

        def predict(self, images):
            calls.append(len(images))
            if self.fail and len(calls) == 2:
                raise RuntimeError("synthetic interruption")
            scores = np.full((len(images), len(self.labels)), .6)
            return scores, scores, 1 - scores

    monkeypatch.setattr(evaluate, "InferenceEngine", MockEngine)
    monkeypatch.setattr(evaluate, "inspect_assets", lambda *a: {"revision": "mock", "sha256": {}})
    monkeypatch.setattr(evaluate, "environment_info", lambda *a: {"python": "synthetic", "packages": {}})
    arguments = ["--dataset", "indiana", "--data-dir", str(iu), "--output-dir", str(output), "--labels", "atelectasis",
                 "--inference-mode", "direct", "--unmentioned-policy", "negative", "--batch-size", "2", "--bootstrap", "0", "--no-plots"]
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        evaluate.main(arguments)
    assert len(list((output / "prediction_parts").glob("*.json"))) == 1
    MockEngine.fail = False
    evaluate.main([*arguments, "--resume"])
    rows = json.loads((output / "predictions.json").read_text())
    assert len(rows) == len({r["sample_id"] for r in rows}) == 6
    assert json.loads((output / "run.json").read_text())["status"] == "completed"
    markdown, word = generate({"indiana": output}, tmp_path / "report")
    assert "Đã thực hiện" in markdown.read_text()
    assert word.is_file()
    evaluate.main(["--output-dir", str(output), "--metrics-only", "--bootstrap", "0", "--no-plots"])


def test_published_reference_shape_and_protocol():
    from evaluation.comparison import references
    assert len(references("indiana")) == 30
    assert len(references("padchest")) == 55
    assert len(references("chexpert")) == 13
    assert set(references("chexpert").experiment) == {"supervised linear-probe"}
    row = references("indiana").set_index("disease").loc["atelectasis"]
    assert row.auroc == .776 and row.positive_support == 616
