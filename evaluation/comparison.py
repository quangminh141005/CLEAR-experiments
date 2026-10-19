"""Published reference comparisons with explicit methodological differences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .storage import file_hash, write_json

PAPER_URL = "https://doi.org/10.1038/s41551-026-01741-4"
SUPPLEMENT_URL = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41551-026-01741-4/MediaObjects/41551_2026_1741_MOESM1_ESM.pdf"
REFERENCE = Path(__file__).parent / "references" / "paper_auroc.csv"


def references(dataset):
    frame = pd.read_csv(REFERENCE)
    return frame[frame.dataset == dataset].copy()


def paper_labels(dataset):
    return references(dataset).disease.tolist()


def prompt_texts(dataset, labels):
    names = dict(zip(references(dataset).disease, references(dataset).paper_finding.str.lower()))
    return [(names.get(label, label), "no " + names.get(label, label)) for label in labels]


def compare(run, metrics):
    dataset = run["dataset"]
    ref = references(dataset).set_index("disease")
    reason = {"indiana": "cohort/Problems labels and patient identity not independently matched to paper",
              "padchest": "Kaggle sample differs from full physician-labelled PadChest benchmark",
              "chexpert": "zero-shot versus supervised linear-probe; Kaggle cohort differs from expert test set"}[dataset]
    if run["inference_mode"] == "direct":
        reason += "; direct baseline omits CLEAR concept projection"
    rows = []
    for label in dict.fromkeys([*ref.index, *metrics["per_class"]]):
        published = ref.loc[label] if label in ref.index else None
        local = metrics["per_class"].get(label, {})
        original = float(published.auroc) if published is not None else None
        reproduced = local.get("auroc")
        ci = local.get("ci95", {}).get("auroc", {})
        rows.append({"disease": label, "paper_auroc": original, "reproduced_auroc": reproduced,
            "absolute_difference": abs(reproduced - original) if original is not None and reproduced is not None else None,
            "signed_difference": reproduced - original if original is not None and reproduced is not None else None,
            "positive": local.get("positive"), "negative": local.get("negative"), "missing": local.get("missing"),
            "ci_low": ci.get("low"), "ci_high": ci.get("high"),
            "paper_ci_low": float(published.ci_low) if published is not None else None,
            "paper_ci_high": float(published.ci_high) if published is not None else None,
            "paper_positive_support": int(published.positive_support) if published is not None and pd.notna(published.positive_support) else None,
            "paper_table": published.table if published is not None else None,
            "paper_experiment": published.experiment if published is not None else None,
            "comparability": "context_only" if dataset == "chexpert" else "conditional_not_verified",
            "reason": reason if published is not None else "No published per-class reference",
            "paper_source": SUPPLEMENT_URL})
    common = [r for r in rows if r["paper_auroc"] is not None and r["reproduced_auroc"] is not None]
    summary = {"dataset": dataset, "paper_macro_from_rounded_per_class": float(ref.auroc.mean()),
        "paper_classes": len(ref), "reproduced_macro_auroc": metrics["macro_auroc"],
        "matched_valid_classes": len(common),
        "paper_macro_on_matched_classes": sum(r["paper_auroc"] for r in common) / len(common) if common else None,
        "reproduced_macro_on_matched_classes": sum(r["reproduced_auroc"] for r in common) / len(common) if common else None,
        "exact_reproduction_verified": False, "reason": reason, "source": SUPPLEMENT_URL,
        "reference_csv_sha256": file_hash(REFERENCE)}
    return rows, summary


def save_comparison(root, run, metrics):
    rows, summary = compare(run, metrics)
    root = Path(root)
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    write_json(root / "comparison.json", {"summary": summary, "per_class": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    save_comparison(root, json.loads((root / "run.json").read_text()), json.loads((root / "metrics.json").read_text()))


if __name__ == "__main__":
    main()
