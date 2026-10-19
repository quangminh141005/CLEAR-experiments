"""Standalone scientific figures; never interpolate missing metrics as zero."""
from pathlib import Path

import numpy as np


def plot_outputs(rows, labels, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for i, label in enumerate(labels):
        y = np.array([r["labels"].get(label) for r in rows], dtype=float)
        scores = np.array([r["scores"][label] for r in rows])
        keep = np.isfinite(y)
        y, scores = y[keep], scores[keep]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            precision, recall, _ = precision_recall_curve(y, scores)
            axes[0].plot(fpr, tpr)
            axes[0].plot([0, 1], [0, 1], "--", color="gray")
            axes[1].plot(recall, precision)
            axes[1].axhline(y.mean(), linestyle="--", color="gray")
        else:
            for ax in axes[:2]:
                ax.text(.5, .5, "Undefined: insufficient label support", ha="center", wrap=True)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC — {label}", xlim=(0, 1), ylim=(0, 1))
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall", xlim=(0, 1), ylim=(0, 1))
        matrix = np.array(metrics["per_class"][label]["confusion_matrix"])
        axes[2].imshow(matrix, cmap="Blues")
        for (a, b), value in np.ndenumerate(matrix):
            axes[2].text(b, a, str(value), ha="center", va="center", color="black")
        axes[2].set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Reference", title="Confusion matrix")
        fig.savefig(output / f"class-{i:03d}.png", dpi=160)
        plt.close(fig)
    for metric in ("auroc", "ap", "f1"):
        valid = [(name, metrics["per_class"][name][metric]) for name in labels
                 if metrics["per_class"][name][metric] is not None]
        # Bound bitmap height even for --all-findings; the complete CSV remains
        # authoritative if a very large native vocabulary makes the chart dense.
        fig, ax = plt.subplots(figsize=(9, min(50, max(3, len(valid) * .25))), constrained_layout=True)
        if valid:
            names, values = zip(*valid)
            ax.barh(names, values, color="#276486")
            ax.invert_yaxis()
        else:
            ax.text(.5, .5, "No defined metrics", ha="center")
        ax.set(xlim=(0, 1), xlabel=metric.upper(), title="Per-class metrics (defined values only)")
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)
