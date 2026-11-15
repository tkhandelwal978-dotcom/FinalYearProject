"""Resumable download of the public fire datasets via kagglehub.

kagglehub downloads public datasets anonymously and resumes from a partial
``.incomplete`` file when re-invoked, so wrapping it in a retry loop gives us a
robust downloader on unreliable connections.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from ml.config import PathsConfig, load_json, save_json
from ml.data.sources import SOURCES, DatasetSource

logger = logging.getLogger(__name__)

MANIFEST_NAME = "sources_manifest.json"


def manifest_path(paths: PathsConfig) -> Path:
    return paths.raw_dir / MANIFEST_NAME


def load_manifest(paths: PathsConfig) -> dict:
    path = manifest_path(paths)
    return load_json(path) if path.is_file() else {}


def download_source(source: DatasetSource, paths: PathsConfig, max_attempts: int = 20) -> Path:
    # kagglehub reads its cache location from the environment at import time.
    os.environ["KAGGLEHUB_CACHE"] = str(paths.raw_dir / "kagglehub_cache")
    import kagglehub

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Downloading %s (%s), attempt %d/%d", source.key, source.kaggle_slug, attempt, max_attempts)
            return Path(kagglehub.dataset_download(source.kaggle_slug))
        except Exception as exc:  # network errors surface as several exception types
            wait = min(60, 5 * attempt)
            logger.warning("Download of %s failed (%s). Retrying in %ds...", source.key, exc, wait)
            time.sleep(wait)
    raise RuntimeError(
        f"Could not download {source.kaggle_slug} after {max_attempts} attempts. "
        f"Download it manually from {source.url}, extract it, and pass its folder with "
        f"`python scripts/download_dataset.py --register {source.key}=<folder>`."
    )


def download_all(paths: PathsConfig, keys: list[str] | None = None) -> dict:
    paths.ensure()
    manifest = load_manifest(paths)
    for source in SOURCES:
        if keys and source.key not in keys:
            continue
        existing = manifest.get(source.key)
        if existing and Path(existing["path"]).is_dir():
            logger.info("%s already available at %s", source.key, existing["path"])
            continue
        location = download_source(source, paths)
        manifest[source.key] = {"path": str(location), "slug": source.kaggle_slug, "license": source.license}
        save_json(manifest, manifest_path(paths))
    return manifest


def register_manual(paths: PathsConfig, key: str, folder: Path) -> dict:
    """Record a manually downloaded and extracted dataset folder."""
    from ml.data.sources import get_source

    source = get_source(key)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    paths.ensure()
    manifest = load_manifest(paths)
    manifest[key] = {"path": str(folder.resolve()), "slug": source.kaggle_slug, "license": source.license}
    save_json(manifest, manifest_path(paths))
    return manifest
