"""Evaluate the trained model on the unseen test split and write metrics, plots and reports.

Usage:
    python scripts/evaluate.py [--batch-size 32] [--num-workers 4]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import PathsConfig  # noqa: E402
from ml.evaluation.evaluate import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = PathsConfig()
    metrics = evaluate(paths, args.batch_size, args.num_workers)
    print("\nTest set results")
    for key in ("test_samples", "decision_threshold", "accuracy", "precision", "recall", "f1_score", "roc_auc",
                "specificity", "false_negatives_missed_fires", "false_positives_false_alarms"):
        value = metrics[key]
        print(f"  {key:<30} {value:.4f}" if isinstance(value, float) else f"  {key:<30} {value}")
    print(f"\nArtifacts written to {paths.artifacts_dir}")


if __name__ == "__main__":
    main()
