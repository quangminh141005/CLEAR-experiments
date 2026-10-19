"""Masked binary metrics and patient-cluster percentile bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def _ratio(a, b):
    return float(a / b) if b else None


def binary_metrics(y, scores, threshold=0.5):
    y, scores = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    if y.shape != scores.shape or y.ndim != 1:
        raise ValueError("Labels and scores must be aligned vectors")
    if np.any(np.isfinite(y) & ~np.isin(y, [0, 1])):
        raise ValueError("Reference labels must be binary or missing")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite in [0,1]")
    mask = np.isfinite(y)
    y, scores = y[mask].astype(int), scores[mask]
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    n_pos, n_neg = int(tp + fn), int(tn + fp)
    return {"auroc": float(roc_auc_score(y, scores)) if n_pos and n_neg else None,
            "ap": float(average_precision_score(y, scores)) if n_pos else None,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "specificity": _ratio(tn, tn + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "positive": n_pos, "negative": n_neg, "known": int(len(y)), "missing": int((~mask).sum()),
            "threshold": float(threshold), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def interval(values, requested):
    valid = [v for v in values if v is not None and np.isfinite(v)]
    bounds = [float(v) for v in np.percentile(valid, [2.5, 97.5])] if len(valid) >= 2 else [None, None]
    return {"low": bounds[0], "high": bounds[1], "valid_resamples": len(valid), "requested_resamples": requested}


def evaluate_rows(rows, labels, *, thresholds=None, bootstrap=1000, seed=42):
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    thresholds = thresholds or {label: 0.5 for label in labels}
    if set(thresholds) != set(labels) or any(not np.isfinite(x) or not 0 <= x <= 1 for x in thresholds.values()):
        raise ValueError("One finite threshold in [0,1] required per class")
    if not rows:
        raise ValueError("No predictions available")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate samples would bias metrics")
    y = np.array([[r["labels"].get(label) for label in labels] for r in rows], dtype=float)
    scores = np.array([[r["scores"][label] for label in labels] for r in rows], dtype=float)
    per_class = {label: binary_metrics(y[:, i], scores[:, i], thresholds[label]) for i, label in enumerate(labels)}
    valid_classes = [label for label in labels if per_class[label]["auroc"] is not None]
    macro = float(np.mean([per_class[label]["auroc"] for label in valid_classes])) if valid_classes else None
    # Resample entire patients/groups, preserving correlation between images and classes.
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["patient_id"], []).append(i)
    group_indices = list(groups.values())
    rng = np.random.default_rng(seed)
    keys = ["auroc", "ap", "precision", "recall", "specificity", "f1", "tn", "fp", "fn", "tp", "positive", "negative"]
    distributions = {label: {key: [] for key in keys} for label in labels}
    macro_samples = []
    for _ in range(bootstrap):
        picked = rng.integers(0, len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[i] for i in picked])
        values = {}
        for i, label in enumerate(labels):
            result = binary_metrics(y[indices, i], scores[indices, i], thresholds[label])
            values[label] = result["auroc"]
            for key in keys:
                distributions[label][key].append(result[key])
        # Keep the macro class set fixed; do not silently average a changing subset.
        macro_samples.append(float(np.mean([values[label] for label in valid_classes]))
                             if valid_classes and all(values[label] is not None for label in valid_classes) else None)
    for label in labels:
        per_class[label]["ci95"] = {key: interval(values, bootstrap) for key, values in distributions[label].items()}
        # Undefined point metrics remain undefined even if a resample happened to yield a value.
        for key in keys:
            if per_class[label][key] is None:
                per_class[label]["ci95"][key].update(low=None, high=None)
    return {"per_class": per_class, "macro_auroc": macro, "macro_auroc_ci95": interval(macro_samples, bootstrap),
            "valid_auroc_classes": valid_classes, "number_of_classes": len(labels), "images": len(rows),
            "patients_or_groups": len(groups), "bootstrap_unit": "patient_id cluster", "bootstrap_seed": seed,
            "bootstrap_replicates": bootstrap, "ci_method": "95% percentile; degenerate replicates excluded; counts reported",
            "confusion_matrix_order": "rows true [0,1]; columns predicted [0,1]"}


def metrics_frame(metrics):
    records = []
    for label, item in metrics["per_class"].items():
        row = {"disease": label, **{k: v for k, v in item.items() if k not in {"ci95", "confusion_matrix"}}}
        for key, bounds in item["ci95"].items():
            row.update({f"{key}_ci_low": bounds["low"], f"{key}_ci_high": bounds["high"],
                        f"{key}_bootstrap_valid": bounds["valid_resamples"]})
        records.append(row)
    return pd.DataFrame(records)
