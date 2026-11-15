"""Reads model metadata and evaluation outputs from the training artifacts on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.schemas.system import ModelInfoOut
from app.services.model_service import ModelService

PLOT_FILES = {
    "training_accuracy": "training_accuracy.png",
    "training_loss": "training_loss.png",
    "confusion_matrix": "confusion_matrix.png",
    "roc_curve": "roc_curve.png",
    "precision_recall_curve": "precision_recall_curve.png",
    "class_distribution": "class_distribution.png",
}


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build_model_info(models: ModelService, artifacts_dir: Path) -> ModelInfoOut:
    meta = models.metadata or _read_json(artifacts_dir / "model" / "training_metadata.json") or {}
    history_path = artifacts_dir / "metrics" / "training_history.csv"
    history = pd.read_csv(history_path).round(5).to_dict(orient="records") if history_path.is_file() else []
    plots = {key: f"/artifacts/plots/{name}" for key, name in PLOT_FILES.items()
             if (artifacts_dir / "plots" / name).is_file()}
    training = {k: meta.get(k) for k in ("epochs_run", "best_epoch", "best_val_loss", "training_started_utc",
                                         "training_finished_utc", "training_duration_minutes", "pos_weight",
                                         "total_parameters", "trainable_parameters_fine_tune", "train_class_counts",
                                         "validation_class_counts", "threshold_tuning")} if meta else None
    return ModelInfoOut(
        model_loaded=models.loaded,
        model_name=meta.get("model_name"),
        model_version=meta.get("model_version"),
        architecture=meta.get("architecture"),
        framework=meta.get("framework"),
        device=models.device or meta.get("device"),
        image_size=(meta.get("preprocessing") or {}).get("image_size"),
        decision_threshold=meta.get("decision_threshold"),
        class_names=meta.get("class_names"),
        preprocessing=meta.get("preprocessing"),
        hyperparameters=meta.get("hyperparameters"),
        training=training,
        dataset=_read_json(artifacts_dir / "metrics" / "dataset_stats.json") or meta.get("dataset"),
        test_metrics=_read_json(artifacts_dir / "metrics" / "test_metrics.json"),
        training_history=history,
        curves=_read_json(artifacts_dir / "metrics" / "curves.json"),
        plots=plots,
    )
