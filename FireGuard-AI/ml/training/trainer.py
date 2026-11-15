"""Two-phase transfer-learning trainer for the fire classifier.

Phase 1 trains only the classification head on a frozen ImageNet backbone.
Phase 2 unfreezes the last ``fine_tune_blocks`` backbone stages and fine-tunes
them with a lower learning rate. Both phases use Adam, BCE loss with a class
weight for imbalance, ReduceLROnPlateau scheduling, and early stopping on
validation loss. The best checkpoint (lowest validation loss) is kept.

After training, the decision threshold is tuned on the *validation* set (never
the test set) to keep fire recall at or above ``min_fire_recall``.
"""

from __future__ import annotations

import logging
import platform
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch import nn

from ml.config import CLASS_DISPLAY_NAMES, CLASS_NAMES, PathsConfig, TrainConfig, load_json, save_json
from ml.preprocessing.transforms import build_eval_transform, build_train_transform, preprocessing_config
from ml.training.data import FireImageDataset, make_loader
from ml.training.model import ARCHITECTURE_NAME, FireClassifier

logger = logging.getLogger(__name__)
MODEL_VERSION = "1.0.0"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_probabilities(model: nn.Module, loader, device: torch.device, use_amp: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (P(fire), labels, mean BCE loss) for every sample in ``loader``."""
    model.eval()
    probs, labels, total_loss, n = [], [], 0.0, 0
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
        logits = logits.float()
        total_loss += loss_fn(logits, targets).item()
        n += targets.numel()
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(int), total_loss / max(n, 1)


def tune_threshold(probs: np.ndarray, labels: np.ndarray, min_recall: float) -> dict:
    """Highest-F1 threshold whose fire recall is still >= ``min_recall`` (validation data only)."""
    best = {"threshold": 0.5, "f1": -1.0, "recall": 0.0, "met_recall_target": False}
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        preds = (probs >= threshold).astype(int)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        if recall >= min_recall and f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": float(f1), "recall": float(recall), "met_recall_target": True}
    if not best["met_recall_target"]:
        preds = (probs >= 0.5).astype(int)
        best.update(f1=float(f1_score(labels, preds, zero_division=0)),
                    recall=float(recall_score(labels, preds, zero_division=0)))
        logger.warning("No threshold reached fire recall >= %.2f on validation; using 0.5", min_recall)
    return best


class Trainer:
    def __init__(self, cfg: TrainConfig, paths: PathsConfig) -> None:
        self.cfg, self.paths = cfg, paths
        paths.ensure()
        seed_everything(cfg.seed)
        self.device = select_device()
        self.use_amp = cfg.use_amp and self.device.type == "cuda"

        self.train_ds = FireImageDataset(paths.dataset_dir / "train", build_train_transform(cfg.image_size))
        self.val_ds = FireImageDataset(paths.dataset_dir / "validation", build_eval_transform(cfg.image_size))
        self.train_loader = make_loader(self.train_ds, cfg.batch_size, True, cfg.num_workers, cfg.seed)
        self.val_loader = make_loader(self.val_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)

        counts = self.train_ds.class_counts()
        # Class imbalance: weight the positive (fire) term by n_negative / n_positive.
        self.pos_weight = counts["non_fire"] / max(counts["fire"], 1)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight, device=self.device))

        self.model = FireClassifier(dropout=cfg.dropout).to(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epoch = 0
        logger.info("Device: %s | train %s | val %s | pos_weight %.3f", self.device, counts,
                    self.val_ds.class_counts(), self.pos_weight)

    # --- training loop ---------------------------------------------------------------

    def _train_one_epoch(self, optimizer) -> tuple[float, float]:
        self.model.train()
        self.model.set_frozen_bn_eval()
        total_loss, correct, n = 0.0, 0, 0
        for images, targets in self.train_loader:
            images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits.float(), targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item() * targets.numel()
            correct += ((logits.detach() >= 0).float() == targets).sum().item()
            n += targets.numel()
        return total_loss / n, correct / n

    def _run_phase(self, phase: str, epochs: int, optimizer) -> None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.cfg.lr_factor, patience=self.cfg.lr_patience)
        epochs_without_improvement = 0
        logger.info("Phase '%s': %d epochs, %s trainable params", phase, epochs, f"{self.model.trainable_parameters():,}")
        for _ in range(epochs):
            self.epoch += 1
            start = time.time()
            train_loss, train_acc = self._train_one_epoch(optimizer)
            probs, labels, val_loss = predict_probabilities(self.model, self.val_loader, self.device, self.use_amp)
            preds = (probs >= 0.5).astype(int)
            row = {
                "epoch": self.epoch,
                "phase": phase,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": float((preds == labels).mean()),
                "val_fire_recall": float(recall_score(labels, preds, zero_division=0)),
                "val_roc_auc": float(roc_auc_score(labels, probs)),
                "seconds": round(time.time() - start, 1),
            }
            self.history.append(row)
            scheduler.step(val_loss)

            improved = val_loss < self.best_val_loss - 1e-4
            if improved:
                self.best_val_loss, self.best_epoch = val_loss, self.epoch
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.paths.model_dir / "best_weights.tmp.pt")
            else:
                epochs_without_improvement += 1
            logger.info(
                "Epoch %02d [%s] train_loss %.4f acc %.4f | val_loss %.4f acc %.4f recall %.4f auc %.4f | %.0fs%s",
                self.epoch, phase, train_loss, train_acc, val_loss, row["val_accuracy"], row["val_fire_recall"],
                row["val_roc_auc"], row["seconds"], "  *best*" if improved else "")
            if epochs_without_improvement >= self.cfg.early_stopping_patience:
                logger.info("Early stopping: no val_loss improvement for %d epochs", epochs_without_improvement)
                break

    def fit(self) -> dict:
        started = datetime.now(timezone.utc)
        cfg = self.cfg

        self.model.freeze_backbone()
        head_optimizer = torch.optim.Adam(self.model.head.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay)
        self._run_phase("head", cfg.head_epochs, head_optimizer)

        # Resume fine-tuning from the best head-only weights.
        self.model.load_state_dict(torch.load(self.paths.model_dir / "best_weights.tmp.pt", map_location=self.device))
        self.model.unfreeze_last_blocks(cfg.fine_tune_blocks)
        ft_params = [p for p in self.model.parameters() if p.requires_grad]
        ft_optimizer = torch.optim.Adam(ft_params, lr=cfg.fine_tune_lr, weight_decay=cfg.weight_decay)
        self._run_phase("fine_tune", cfg.fine_tune_epochs, ft_optimizer)

        tmp = self.paths.model_dir / "best_weights.tmp.pt"
        self.model.load_state_dict(torch.load(tmp, map_location=self.device))
        tmp.unlink()

        probs, labels, _ = predict_probabilities(self.model, self.val_loader, self.device, self.use_amp)
        threshold = tune_threshold(probs, labels, cfg.min_fire_recall)
        logger.info("Validation-tuned threshold: %s", threshold)

        finished = datetime.now(timezone.utc)
        metadata = self._metadata(started, finished, threshold)
        torch.save({"state_dict": self.model.state_dict(), "metadata": metadata}, self.paths.checkpoint_path)
        self._save_artifacts(metadata)
        return metadata

    # --- artifacts ---------------------------------------------------------------------

    def _metadata(self, started: datetime, finished: datetime, threshold: dict) -> dict:
        dataset_stats_path = self.paths.metrics_dir / "dataset_stats.json"
        return {
            "model_name": "FireGuard EfficientNet-B0",
            "model_version": MODEL_VERSION,
            "architecture": ARCHITECTURE_NAME,
            "framework": f"PyTorch {torch.__version__}",
            "class_names": list(CLASS_NAMES),
            "class_display_names": CLASS_DISPLAY_NAMES,
            "positive_class": "fire",
            "decision_threshold": threshold["threshold"],
            "threshold_tuning": {"source": "validation set", "min_fire_recall": self.cfg.min_fire_recall, **threshold},
            "preprocessing": preprocessing_config(self.cfg.image_size),
            "hyperparameters": self.cfg.to_dict(),
            "pos_weight": self.pos_weight,
            "total_parameters": self.model.total_parameters(),
            "trainable_parameters_fine_tune": self.model.trainable_parameters(),
            "train_class_counts": self.train_ds.class_counts(),
            "validation_class_counts": self.val_ds.class_counts(),
            "epochs_run": self.epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else platform.processor() or "CPU",
            "training_started_utc": started.isoformat(),
            "training_finished_utc": finished.isoformat(),
            "training_duration_minutes": round((finished - started).total_seconds() / 60, 2),
            "dataset": load_json(dataset_stats_path) if dataset_stats_path.is_file() else None,
        }

    def _save_artifacts(self, metadata: dict) -> None:
        history = pd.DataFrame(self.history)
        history.to_csv(self.paths.metrics_dir / "training_history.csv", index=False)
        save_json(metadata, self.paths.model_dir / "training_metadata.json")
        save_json({"class_names": list(CLASS_NAMES), "display_names": CLASS_DISPLAY_NAMES, "positive_index": 1},
                  self.paths.model_dir / "class_labels.json")
        save_json(metadata["preprocessing"], self.paths.model_dir / "preprocessing.json")
        plot_training_curves(history, self.paths)


def plot_training_curves(history: pd.DataFrame, paths: PathsConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fine_tune_start = history.loc[history["phase"] == "fine_tune", "epoch"].min()
    for metric, title, filename in (("accuracy", "Training vs Validation Accuracy", "training_accuracy.png"),
                                    ("loss", "Training vs Validation Loss", "training_loss.png")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["epoch"], history[f"train_{metric}"], marker="o", label=f"Train {metric}", color="#3b82f6")
        ax.plot(history["epoch"], history[f"val_{metric}"], marker="o", label=f"Validation {metric}", color="#f97316")
        if pd.notna(fine_tune_start):
            ax.axvline(fine_tune_start - 0.5, linestyle="--", color="gray", alpha=0.7)
            ax.text(fine_tune_start - 0.4, ax.get_ylim()[1], " fine-tuning", va="top", fontsize=8, color="gray")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.plots_dir / filename, dpi=150)
        plt.close(fig)
