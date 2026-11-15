"""Download the public fire / non-fire datasets used to train FireGuard AI.

Usage:
    python scripts/download_dataset.py                 # download every registered source
    python scripts/download_dataset.py --only deepfire # download a single source
    python scripts/download_dataset.py --register deepfire=D:/downloads/forest_fire
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import PathsConfig  # noqa: E402
from ml.data.download import download_all, register_manual  # noqa: E402
from ml.data.sources import SOURCES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=[s.key for s in SOURCES], help="Subset of sources to download")
    parser.add_argument("--register", metavar="KEY=FOLDER", help="Register a manually downloaded dataset folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = PathsConfig()

    if args.register:
        key, folder = args.register.split("=", 1)
        manifest = register_manual(paths, key, Path(folder))
    else:
        manifest = download_all(paths, args.only)

    print("\nAvailable sources:")
    for key, info in manifest.items():
        print(f"  {key:<20} {info['license']:<32} {info['path']}")


if __name__ == "__main__":
    main()
