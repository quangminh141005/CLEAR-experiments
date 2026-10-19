"""Choose thresholds only from a separately recorded validation run."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

from .storage import file_hash, write_json


def select_thresholds(run_dir, output):
    root = Path(run_dir)
    run = json.loads((root / "run.json").read_text())
    if run["status"] != "completed" or run["cohort"]["metadata"]["split"] != "validation":
        raise ValueError("Threshold selection requires a completed --split validation run")
    if file_hash(root / "predictions.json") != run.get("predictions_sha256"):
        raise ValueError("Validation predictions differ from the completed run checksum")
    rows = json.loads((root / "predictions.json").read_text())
    selected = {}
    for label in run["cohort"]["labels"]:
        known = [r for r in rows if r["labels"].get(label) is not None]
        y = np.array([r["labels"][label] for r in known])
        s = np.array([r["scores"][label] for r in known])
        if len(np.unique(y)) != 2:
            raise ValueError(f"Validation class {label} lacks positive/negative support; use fixed thresholds or another cohort")
        p, recall, thresholds = precision_recall_curve(y, s)
        f1 = 2 * p[:-1] * recall[:-1] / np.maximum(p[:-1] + recall[:-1], 1e-12)
        selected[label] = float(thresholds[np.argmax(f1)])
    write_json(output, {"thresholds": selected, "selection": "maximum validation F1; smallest threshold breaks ties",
        "dataset": run["dataset"], "inference_signature": run["inference_signature"],
        "validation_patient_ids": sorted({r["patient_id"] for r in rows}),
        "validation_predictions_sha256": file_hash(root / "predictions.json"),
        "validation_run": str(root.resolve())})


def load_thresholds(path, run, rows):
    payload = json.loads(Path(path).read_text())
    if payload["dataset"] != run["dataset"] or payload["inference_signature"] != run["inference_signature"]:
        raise ValueError("Validation thresholds use a different dataset/inference/label policy")
    if set(payload["validation_patient_ids"]) & {r["patient_id"] for r in rows}:
        raise ValueError("Validation and evaluation patients overlap")
    return payload["thresholds"], payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    select_thresholds(args.validation_run, args.output)


if __name__ == "__main__":
    main()
