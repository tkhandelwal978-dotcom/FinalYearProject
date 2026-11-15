"""Central configuration for the FireGuard AI ML pipeline.

Every path and hyperparameter used by the data, training, evaluation and
inference code is defined here so the whole pipeline is reproducible from a
single place. Paths can be overridden through environment variables (or a
project-level ``.env`` file), which lets large raw data live outside the repo.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Class index 1 is the positive ("fire") class; the model's sigmoid output is P(fire).
CLASS_NAMES: tuple[str, str] = ("non_fire", "fire")
CLASS_DISPLAY_NAMES = {"fire": "FIRE DETECTED", "non_fire": "NO FIRE DETECTED"}
SPLITS: tuple[str, str, str] = ("train", "validation", "test")

# ImageNet statistics: EfficientNet-B0 was pretrained with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file (existing vars win)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass
class PathsConfig:
    raw_dir: Path = field(default_factory=lambda: _env_path("FIREGUARD_RAW_DIR", PROJECT_ROOT / "data_raw"))
    dataset_dir: Path = field(default_factory=lambda: _env_path("FIREGUARD_DATASET_DIR", PROJECT_ROOT / "dataset"))
    artifacts_dir: Path = field(default_factory=lambda: _env_path("FIREGUARD_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / "model"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def plots_dir(self) -> Path:
        return self.artifacts_dir / "plots"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "fireguard_efficientnet_b0.pt"

    def ensure(self) -> None:
        for d in (self.raw_dir, self.dataset_dir, self.model_dir, self.metrics_dir, self.plots_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # Images smaller than this (either side) are treated as unusable.
    min_side: int = 64
    # Stored copies are resized so the longest side is at most this (keeps the dataset compact).
    max_stored_side: int = 512
    # Perceptual-hash Hamming distance at or below which two images count as near-duplicates.
    # Chosen by inspecting pairs: at <= 8 they were crops / rescales of one photo (some across
    # datasets); from 10 upwards they were different scenes.
    phash_threshold: int = 8


@dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    # Phase 1: train only the classification head on a frozen backbone.
    head_epochs: int = 5
    head_lr: float = 1e-3
    # Phase 2: unfreeze the last `fine_tune_blocks` EfficientNet stages and fine-tune.
    fine_tune_epochs: int = 15
    fine_tune_lr: float = 1e-4
    fine_tune_blocks: int = 3
    dropout: float = 0.3
    weight_decay: float = 1e-5
    lr_patience: int = 2
    lr_factor: float = 0.3
    early_stopping_patience: int = 5
    # Decision threshold is tuned on the validation set so fire recall stays at or above this.
    min_fire_recall: float = 0.95
    use_amp: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
