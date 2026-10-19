#!/usr/bin/env python3
"""CLEAR zero-shot evaluation. All network acquisition is an explicit separate step."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import pandas as pd

from evaluation.calibration import load_thresholds
from evaluation.comparison import paper_labels, prompt_texts, save_comparison
from evaluation.datasets import LOADERS
from evaluation.datasets.common import canonical, normalized_name
from evaluation.inference import InferenceEngine, inspect_assets, load_image, read_prompt_cache
from evaluation.metrics import evaluate_rows, metrics_frame
from evaluation.storage import PredictionStore, file_hash, stable_hash, write_json


def parser_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    defaults = {}
    if known.config:
        import yaml
        defaults = yaml.safe_load(Path(known.config).read_text()) or {}
        if not isinstance(defaults, dict):
            raise ValueError("Config must be a YAML mapping")
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--dataset", choices=list(LOADERS))
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--projections", help="Indiana projections CSV")
    parser.add_argument("--patient-map", help="CSV with study_id,patient_id; required for verified Indiana splits")
    parser.add_argument("--label-column")
    parser.add_argument("--label-map", help="JSON mapping raw label names to canonical names")
    parser.add_argument("--labels", help="Comma-separated canonical findings; default paper findings")
    parser.add_argument("--all-findings", action="store_true", help="All observed/native finding labels, not a test-selected class subset")
    parser.add_argument("--unmentioned-policy", choices=["unknown", "negative"], default="unknown")
    parser.add_argument("--normal-policy", choices=["unknown", "negative"], default="unknown")
    parser.add_argument("--uncertain-policy", choices=["ignore", "negative", "positive"], default="ignore")
    parser.add_argument("--missing-policy", choices=["ignore", "negative"], default="ignore")
    parser.add_argument("--missing-images", choices=["error", "skip"], default="error")
    parser.add_argument("--image-policy", choices=["8bit", "uint16-linear"], default="8bit")
    parser.add_argument("--view", choices=["all", "frontal"], default="all")
    parser.add_argument("--split", choices=["all", "validation", "test"], default="all")
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--subset-size", type=int, help="Maximum patient/groups; retain all images for each selected group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...; one GPU per process")
    parser.add_argument("--inference-mode", choices=["clear", "direct"], default="clear")
    parser.add_argument("--assets-dir", default=os.environ.get("CLEAR_ASSET_DIR", "assets/clear"))
    parser.add_argument("--checkpoint", help="Optional local path to the registered official checkpoint")
    parser.add_argument("--prompt-cache", help="SFR .npz; default ASSETS/prompts/DATASET.npz")
    parser.add_argument("--concept-chunk-size", type=int, default=512)
    parser.add_argument("--allow-unsafe-pickle", action="store_true", help="Trust the registered official SFR pickle")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--threshold-file", help="Thresholds selected by evaluation.calibration on independent validation patients")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Inspect cohort/paths/label support, without loading torch/models")
    parser.add_argument("--metrics-only", action="store_true", help="Recalculate saved predictions without model/data access")
    parser.add_argument("--no-plots", action="store_true")
    allowed = {action.dest for action in parser._actions}
    unknown = set(defaults) - allowed
    if unknown:
        parser.error(f"Unknown config keys: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.concept_chunk_size < 1 or args.bootstrap < 0:
        parser.error("batch-size/chunk-size must be positive, bootstrap nonnegative")
    if not np.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be finite in [0,1]")
    if args.all_findings and args.labels:
        parser.error("--all-findings and --labels are mutually exclusive")
    if not args.metrics_only and (not args.dataset or not args.data_dir):
        parser.error("--dataset and --data-dir required except for --metrics-only")
    return args


def environment_info(torch=None):
    packages = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "pillow", "transformers", "python-docx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version, "platform": platform.platform(), "packages": packages}
    if torch is not None:
        result.update(cuda_runtime=torch.version.cuda, gpu_available=torch.cuda.is_available(),
                      cudnn=torch.backends.cudnn.version(),
                      gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                      gpu_compute_capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())],
                      compiled_cuda_arches=torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
    try:
        result["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        result["nvidia_smi"] = None
    return result


def source_hashes():
    paths = [ROOT / "evaluate.py", *sorted((ROOT / "evaluation").rglob("*.py")),
             *sorted((ROOT / "src" / "clear").glob("*.py"))]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def architecture_identity():
    """Record the actual local architecture source, including uncommitted changes."""
    path = os.environ.get("CLEAR_DINOV2_REPO")
    if not path:
        return {"source": "loader torch-hub cache discovery; set CLEAR_DINOV2_REPO for fully recorded source"}
    root = Path(path).resolve()
    if not (root / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 source not found: {root}")
    return {"source": str(root), "python_sha256":
            {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))}}


def validate_prediction_rows(rows, cohort):
    expected = {r["sample_id"]: r for r in cohort["records"]}
    if len(rows) != len(expected) or {r["sample_id"] for r in rows} != set(expected):
        raise ValueError("Predictions must contain each cohort sample exactly once")
    for row in rows:
        record = expected[row["sample_id"]]
        if any(row[key] != record[key] for key in ("patient_id", "study_id", "labels", "image_path")):
            raise ValueError(f"Prediction reference metadata differs for {row['sample_id']}")
        if set(row["scores"]) != set(cohort["labels"]):
            raise ValueError("Prediction classes differ from the saved cohort")


def cohort_summary(cohort):
    support = {}
    for label in cohort.labels:
        values = [r.labels[label] for r in cohort.records]
        support[label] = {"positive": values.count(1), "negative": values.count(0), "missing": values.count(None)}
    return {"metadata": cohort.metadata, "per_class_support": support,
            "warnings": ["Unknown labels are masked; zero negatives implies undefined AUROC.",
                         "Indiana uid is a study proxy unless a verified patient map is supplied.",
                         "No inference has been performed by a prepare-only run."]}


def export_predictions(root, rows, labels):
    write_json(root / "predictions.json", rows)
    records = []
    for row in rows:
        for label in labels:
            records.append({"sample_id": row["sample_id"], "patient_id": row["patient_id"], "study_id": row["study_id"],
                "image_path": row["image_path"], "disease": label, "y_true": row["labels"][label],
                "score": row["scores"][label], "positive_logit": row.get("positive_logits", {}).get(label),
                "negative_logit": row.get("negative_logits", {}).get(label)})
    pd.DataFrame(records).to_csv(root / "predictions.csv", index=False)


def finalize(root, run, rows, args):
    labels = run["cohort"]["labels"]
    validate_prediction_rows(rows, run["cohort"])
    if args.threshold_file:
        thresholds, provenance = load_thresholds(args.threshold_file, run, rows)
        threshold_info = {"type": "independent_validation", "provenance": provenance}
    else:
        thresholds = dict.fromkeys(labels, args.threshold)
        threshold_info = {"type": "fixed", "value": args.threshold, "note": "Uncalibrated paired cosine scores; threshold not fit on evaluation labels"}
    metrics = evaluate_rows(rows, labels, thresholds=thresholds, bootstrap=args.bootstrap, seed=args.seed)
    metrics["threshold_policy"] = threshold_info
    metrics["dataset"] = run["dataset"]
    metrics["inference_mode"] = run["inference_mode"]
    metrics["predictions_sha256"] = file_hash(root / "predictions.json")
    write_json(root / "metrics.json", metrics)
    metrics_frame(metrics).to_csv(root / "metrics.csv", index=False)
    pd.DataFrame([{k: metrics[k] for k in ("dataset", "inference_mode", "macro_auroc", "images", "patients_or_groups", "number_of_classes")},
                 ]).to_csv(root / "summary.csv", index=False)
    errors = []
    for label in labels:
        for kind, truth in (("FP", 0), ("FN", 1)):
            selected = [r for r in rows if r["labels"].get(label) == truth and
                        (r["scores"][label] >= thresholds[label]) != bool(truth)]
            selected.sort(key=lambda r: r["scores"][label], reverse=(kind == "FP"))
            for row in selected[:20]:
                errors.append({"disease": label, "error": kind, "sample_id": row["sample_id"],
                    "patient_id": row["patient_id"], "image_path": row["image_path"], "score": row["scores"][label],
                    "threshold": thresholds[label], "reference_label": truth})
    pd.DataFrame(errors, columns=["disease", "error", "sample_id", "patient_id", "image_path", "score", "threshold", "reference_label"]).to_csv(root / "errors.csv", index=False)
    save_comparison(root, run, metrics)
    if not args.no_plots:
        from evaluation.visualization import plot_outputs
        plot_outputs(rows, labels, metrics, root / "figures")
    logging.info("Macro AUROC: %s (%d valid classes)", metrics["macro_auroc"], len(metrics["valid_auroc_classes"]))


def main(argv=None):
    args = parser_args(argv)
    root = Path(args.output_dir)
    if args.metrics_only:
        run = json.loads((root / "run.json").read_text())
        if run["status"] != "completed":
            raise ValueError("Run incomplete; resume inference before recalculating full-cohort metrics")
        rows = json.loads((root / "predictions.json").read_text())
        if run.get("predictions_sha256") != file_hash(root / "predictions.json"):
            raise ValueError("Saved prediction checksum differs from the completed run")
        finalize(root, run, rows, args)
        return
    aliases = json.loads(Path(args.label_map).read_text()) if args.label_map else {}
    if not isinstance(aliases, dict):
        raise ValueError("label-map must be a JSON object")
    aliases = {normalized_name(k): normalized_name(v) for k, v in aliases.items()}
    # PadChest native spelling aliases; exact leaf labels remain distinct.
    if args.dataset == "padchest":
        aliases = {"descending aortic elongation": "descendent aortic elongation", "goitre": "goiter", **aliases}
    labels = [canonical(x, aliases) for x in args.labels.split(",")] if args.labels else (None if args.all_findings else paper_labels(args.dataset))
    if labels is not None and len(labels) != len(set(labels)):
        raise ValueError("Label names collapse to duplicates after mapping")
    options = vars(args).copy()
    options["aliases"] = aliases
    cohort = LOADERS[args.dataset](args.data_dir, labels=labels, **{k: v for k, v in options.items() if k != "labels"})
    cohort = cohort.select(split=args.split, validation_fraction=args.validation_fraction, seed=args.seed, subset_size=args.subset_size)
    pairs = prompt_texts(args.dataset, cohort.labels)
    if args.prepare_only:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Use a fresh preparation directory")
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "cohort.json", cohort.payload())
        write_json(root / "data_check.json", cohort_summary(cohort))
        write_json(root / "required_prompts.json", dict(zip(cohort.labels, pairs)))
        print(f"Prepared {len(cohort.records)} images; inspect {root / 'data_check.json'}. No model was loaded.")
        return
    assets = inspect_assets(args.assets_dir, args.inference_mode, args.checkpoint)
    prompt_cache = Path(args.prompt_cache or Path(args.assets_dir) / "prompts" / f"{args.dataset}.npz")
    prompt_identity = None
    if args.inference_mode == "clear":
        if not args.allow_unsafe_pickle:
            raise ValueError("The official SFR asset is a pickle. Supply --allow-unsafe-pickle only after verifying trusted provenance.")
        _, _, prompt_metadata = read_prompt_cache(prompt_cache, pairs)
        prompt_identity = {"sha256": file_hash(prompt_cache), "metadata": prompt_metadata}
    code_identity = source_hashes()
    inference_config = {"assets": assets, "prompts": prompt_identity, "pairs": pairs, "labels": cohort.labels,
        "mode": args.inference_mode, "chunk_size": args.concept_chunk_size, "image_policy": args.image_policy,
        "architecture": architecture_identity(),
        "inference_code_sha256": {name: digest for name, digest in code_identity.items()
                                  if name.startswith("src/clear/") or name == "evaluation/inference.py"},
        "label_policies": {k: options[k] for k in ("unmentioned_policy", "normal_policy", "uncertain_policy", "missing_policy", "aliases")}}
    # Include input sizes and mtimes without re-reading every X-ray during resume.
    image_identity = [{"path": r.image_path, "size": Path(r.image_path).stat().st_size,
                       "mtime_ns": Path(r.image_path).stat().st_mtime_ns} for r in cohort.records]
    identity = {"cohort": cohort.payload(), "images": image_identity, "inference": inference_config,
                "source_sha256": code_identity, "device": args.device, "batch_size": args.batch_size,
                "runtime": environment_info()}
    store = PredictionStore(root, identity, resume=args.resume)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(root / "run.log", encoding="utf-8")], force=True)
    run = {"schema_version": 1, "dataset": args.dataset, "inference_mode": args.inference_mode,
           "inference_signature": stable_hash(inference_config), "configuration": vars(args), "assets": assets,
           "prompts": prompt_identity, "cohort": cohort.payload(), "status": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # Check threshold independence before spending time on model inference.
    if args.threshold_file:
        load_thresholds(args.threshold_file, run, [asdict(r) for r in cohort.records])
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    run["environment"] = environment_info(torch)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose --device cpu or fix the driver/wheel")
        device = torch.device(args.device)
        capability = torch.cuda.get_device_capability(device)
        arch = f"sm_{capability[0]}{capability[1]}"
        if capability[0] == 3 and arch not in torch.cuda.get_arch_list():
            raise RuntimeError(f"Wheel lacks {arch} kernels. Tesla K80 needs the legacy cu102 stack, not modern PyTorch.")
    write_json(root / "run.json", run)
    write_json(root / "cohort.json", cohort.payload())
    write_json(root / "data_check.json", cohort_summary(cohort))
    selected_ids = {r.sample_id for r in cohort.records}
    if not store.completed.issubset(selected_ids):
        raise ValueError("Prediction parts contain samples outside this cohort")
    pending = [record for record in cohort.records if record.sample_id not in store.completed]
    started = time.monotonic()
    try:
        if pending:
            engine = InferenceEngine(assets_dir=args.assets_dir, labels=cohort.labels, pairs=pairs, device=args.device,
                mode=args.inference_mode, chunk_size=args.concept_chunk_size, prompt_cache=prompt_cache,
                allow_unsafe_pickle=args.allow_unsafe_pickle, checkpoint=args.checkpoint)
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                images = torch.stack([load_image(r.image_path, engine.preprocess, args.image_policy) for r in batch])
                scores, positive, negative = engine.predict(images)
                expected_shape = (len(batch), len(cohort.labels))
                if any(array.shape != expected_shape or not np.isfinite(array).all() for array in (scores, positive, negative)):
                    raise ValueError("Nonfinite or misaligned model outputs; no batch saved")
                if np.any((scores < 0) | (scores > 1)):
                    raise ValueError("Model scores outside [0,1]; no batch saved")
                rows = []
                for i, record in enumerate(batch):
                    rows.append({**asdict(record), "scores": dict(zip(cohort.labels, map(float, scores[i]))),
                        "positive_logits": dict(zip(cohort.labels, map(float, positive[i]))),
                        "negative_logits": dict(zip(cohort.labels, map(float, negative[i])))})
                store.save_batch(rows)
                logging.info("Saved %d/%d images", len(store.completed), len(cohort.records))
        rows = sorted(store.rows, key=lambda r: r["sample_id"])
        export_predictions(root, rows, cohort.labels)
        finalize(root, run, rows, args)
        run.update(status="completed", current_session_seconds=time.monotonic() - started,
                   predictions_sha256=file_hash(root / "predictions.json"),
                   finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(root / "run.json", run)
    except BaseException as exc:
        run.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc),
                   completed_images=len(store.completed), current_session_seconds=time.monotonic() - started)
        write_json(root / "run.json", run)
        logging.exception("Run stopped; saved batches can be resumed with the same arguments and --resume")
        raise


if __name__ == "__main__":
    main()
