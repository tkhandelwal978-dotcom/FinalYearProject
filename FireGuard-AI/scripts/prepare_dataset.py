"""Clean, de-duplicate and split the downloaded images into dataset/{train,validation,test}/{fire,non_fire}.

Usage:
    python scripts/prepare_dataset.py [--seed 42] [--val 0.15] [--test 0.15]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import DataConfig, PathsConfig  # noqa: E402
from ml.preprocessing.prepare import prepare_dataset  # noqa: E402


def main() -> None:
    defaults = DataConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--val", type=float, default=defaults.val_fraction)
    parser.add_argument("--test", type=float, default=defaults.test_fraction)
    parser.add_argument("--phash-threshold", type=int, default=defaults.phash_threshold)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = DataConfig(seed=args.seed, val_fraction=args.val, test_fraction=args.test,
                     phash_threshold=args.phash_threshold)
    stats = prepare_dataset(PathsConfig(), cfg, workers=args.workers)
    print(json.dumps({k: stats[k] for k in ("final_images", "class_totals", "split_counts",
                                            "duplicates_removed", "corrupted_or_too_small_removed",
                                            "conflicting_label_images_removed")}, indent=2))


if __name__ == "__main__":
    main()
