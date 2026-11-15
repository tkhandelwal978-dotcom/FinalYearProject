"""Evaluate the trained checkpoint on the held-out test split.

The test split is only ever touched here, after training and validation-based
threshold tuning have finished.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

from ml.config import CLASS_NAMES, PathsConfig, save_json
from ml.inference.predictor import load_checkpoint
from ml.preprocessing.transforms import build_eval_transform
from ml.training.data import FireImageDataset, make_loader
from ml.training.trainer import predict_probabilities, select_device

logger = logging.getLogger(__name__)


def evaluate(paths: PathsConfig, batch_size: int = 32, num_workers: int = 4) -> dict:
    device = select_device()
    model, metadata = load_checkpoint(paths.checkpoint_path, device)
    image_size = metadata["preprocessing"]["image_size"][0]
    threshold = float(metadata["decision_threshold"])

    test_ds = FireImageDataset(paths.dataset_dir / "test", build_eval_transform(image_size))
    loader = make_loader(test_ds, batch_size, False, num_workers, seed=0)
    probs, labels, test_loss = predict_probabilities(model, loader, device, use_amp=True)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "model_version": metadata["model_version"],
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "test_samples": int(len(labels)),
        "test_class_counts": test_ds.class_counts(),
        "decision_threshold": threshold,
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "false_negatives_missed_fires": int(fn),
        "false_positives_false_alarms": int(fp),
        "confusion_matrix": {"labels": list(CLASS_NAMES), "matrix": cm.tolist(),
                             "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "metrics_at_0_5": {
            "accuracy": float(accuracy_score(labels, probs >= 0.5)),
            "recall": float(recall_score(labels, probs >= 0.5, zero_division=0)),
            "precision": float(precision_score(labels, probs >= 0.5, zero_division=0)),
        },
    }
    report = classification_report(labels, preds, target_names=list(CLASS_NAMES), digits=4, output_dict=True)
    metrics["classification_report"] = report

    paths.ensure()
    save_json(metrics, paths.metrics_dir / "test_metrics.json")
    (paths.reports_dir / "classification_report.txt").write_text(
        f"FireGuard AI - test set classification report (threshold {threshold:.2f})\n\n"
        + classification_report(labels, preds, target_names=list(CLASS_NAMES), digits=4),
        encoding="utf-8")
    pd.DataFrame({"path": [str(p) for p, _ in test_ds.samples], "label": labels, "fire_probability": probs,
                  "predicted": preds}).to_csv(paths.metrics_dir / "test_predictions.csv", index=False)

    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    save_json({"roc": {"fpr": _downsample(fpr), "tpr": _downsample(tpr)},
               "pr": {"precision": _downsample(precision), "recall": _downsample(recall)}},
              paths.metrics_dir / "curves.json")
    _plot_all(cm, fpr, tpr, precision, recall, metrics, paths)
    logger.info("Test metrics: acc %.4f prec %.4f recall %.4f f1 %.4f auc %.4f", metrics["accuracy"],
                metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["roc_auc"])
    return metrics


def _downsample(values: np.ndarray, max_points: int = 200) -> list[float]:
    """Keep curve JSON small enough for the dashboard while preserving its shape."""
    if len(values) <= max_points:
        return [round(float(v), 5) for v in values]
    idx = np.linspace(0, len(values) - 1, max_points).round().astype(int)
    return [round(float(values[i]), 5) for i in idx]


def _plot_all(cm, fpr, tpr, precision, recall, metrics: dict, paths: PathsConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    names = ["No fire", "Fire"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix (threshold {metrics['decision_threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(paths.plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#ef4444", lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (fire recall)")
    ax.set_title("ROC curve - test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.plots_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#8b5cf6", lw=2, label=f"PR (AP = {metrics['average_precision']:.4f})")
    ax.set_xlabel("Recall (fire)")
    ax.set_ylabel("Precision (fire)")
    ax.set_title("Precision-Recall curve - test set")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.plots_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
