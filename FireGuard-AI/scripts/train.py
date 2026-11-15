"""Train the FireGuard EfficientNet-B0 fire classifier.

Usage:
    python scripts/train.py                              # defaults from ml/config.py
    python scripts/train.py --batch-size 16 --fine-tune-epochs 8 --num-workers 0   # e.g. CPU-only machines
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import PathsConfig, TrainConfig  # noqa: E402
from ml.training.trainer import Trainer  # noqa: E402


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in dataclasses.fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        if field.type in ("bool", bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=getattr(defaults, field.name))
        else:
            caster = {"int": int, "float": float}.get(str(field.type), str)
            parser.add_argument(flag, type=caster, default=getattr(defaults, field.name))
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    paths = PathsConfig()
    paths.ensure()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(paths.reports_dir / "training_log.txt", mode="w")],
    )
    metadata = Trainer(cfg, paths).fit()
    print(f"\nSaved model to {paths.checkpoint_path}")
    print(f"Best epoch {metadata['best_epoch']} / {metadata['epochs_run']}, best val loss {metadata['best_val_loss']:.4f}")
    print(f"Decision threshold (validation-tuned): {metadata['decision_threshold']:.2f}")


if __name__ == "__main__":
    main()
