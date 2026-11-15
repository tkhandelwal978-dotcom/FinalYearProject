"""Build the train / validation / test dataset from the downloaded raw sources.

Pipeline:
1. Collect every image from each registered source and label it from its folder.
2. Decode each image fully; drop corrupted or too-small files.
3. Group exact duplicates (MD5) and near-duplicates (perceptual hash) with union-find.
4. Drop groups whose members carry conflicting labels; keep one image per group.
5. Stratified split of the de-duplicated images with a fixed seed.
6. Write resized RGB JPEGs into ``dataset/<split>/<class>/`` plus a manifest and stats.

Because de-duplication happens *before* splitting, no image (or near-copy of an
image) can appear in more than one split, which prevents train/test leakage.
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml.config import CLASS_NAMES, SPLITS, DataConfig, PathsConfig, save_json
from ml.data.download import load_manifest
from ml.data.sources import SOURCES, label_for_path
from ml.preprocessing.image_ops import InvalidImageError, file_md5, is_image_file, load_rgb

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    source: str
    path: Path
    label: str
    md5: str = ""
    phash: int = 0
    width: int = 0
    height: int = 0
    error: str = ""


def collect_candidates(paths: PathsConfig) -> tuple[list[Candidate], dict]:
    manifest = load_manifest(paths)
    if not manifest:
        raise FileNotFoundError("No downloaded sources found. Run `python scripts/download_dataset.py` first.")

    candidates: list[Candidate] = []
    unlabeled: dict[str, int] = {}
    for source in SOURCES:
        info = manifest.get(source.key)
        if not info:
            logger.warning("Source %s not downloaded; skipping it.", source.key)
            continue
        root = Path(info["path"])
        skipped = 0
        for file in sorted(root.rglob("*")):
            if not file.is_file() or not is_image_file(file):
                continue
            label = label_for_path(source, file.relative_to(root).parts)
            if label is None:
                skipped += 1
                continue
            candidates.append(Candidate(source=source.key, path=file, label=label))
        unlabeled[source.key] = skipped
        logger.info("%s: %d labelled images, %d unlabelled skipped", source.key,
                    sum(c.source == source.key for c in candidates), skipped)
    return candidates, unlabeled


def _inspect(candidate: Candidate, min_side: int) -> Candidate:
    try:
        img = load_rgb(candidate.path, min_side=min_side)
        candidate.width, candidate.height = img.size
        candidate.md5 = file_md5(candidate.path)
        candidate.phash = int(str(imagehash.phash(img)), 16)
    except InvalidImageError as exc:
        candidate.error = str(exc)
    return candidate


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_duplicates(items: list[Candidate], threshold: int) -> list[int]:
    """Return a group id per item; exact and perceptual near-duplicates share a group."""
    uf = _UnionFind(len(items))
    first_by_md5: dict[str, int] = {}
    for i, item in enumerate(items):
        if item.md5 in first_by_md5:
            uf.union(first_by_md5[item.md5], i)
        else:
            first_by_md5[item.md5] = i

    hashes = np.array([item.phash for item in items], dtype=np.uint64)
    for i in tqdm(range(len(items)), desc="Near-duplicate search", unit="img"):
        xor = np.bitwise_xor(hashes[i + 1:], hashes[i])
        # Hamming distance = popcount of XOR, computed via the uint8 byte view.
        distances = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for offset in np.nonzero(distances <= threshold)[0]:
            uf.union(i, i + 1 + int(offset))
    return [uf.find(i) for i in range(len(items))]


def split_groups(df: pd.DataFrame, cfg: DataConfig) -> pd.Series:
    """Stratified train/validation/test assignment of the de-duplicated images."""
    holdout = cfg.val_fraction + cfg.test_fraction
    train_idx, rest_idx = train_test_split(
        df.index, test_size=holdout, stratify=df["label"], random_state=cfg.seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=cfg.test_fraction / holdout, stratify=df.loc[rest_idx, "label"], random_state=cfg.seed)
    split = pd.Series("", index=df.index)
    split[train_idx], split[val_idx], split[test_idx] = "train", "validation", "test"
    return split


def _write_image(row: pd.Series, dataset_dir: Path, max_side: int) -> str:
    from ml.preprocessing.image_ops import resize_max_side

    img = resize_max_side(load_rgb(Path(row["source_path"])), max_side)
    out = dataset_dir / row["split"] / row["label"] / f"{row['source']}_{row['md5'][:16]}.jpg"
    img.save(out, format="JPEG", quality=95)
    return str(out.relative_to(dataset_dir))


def prepare_dataset(paths: PathsConfig, cfg: DataConfig, workers: int = 8) -> dict:
    paths.ensure()
    candidates, unlabeled = collect_candidates(paths)
    logger.info("Inspecting %d images (decode, size check, hashing)...", len(candidates))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inspected = list(tqdm(pool.map(lambda c: _inspect(c, cfg.min_side), candidates),
                              total=len(candidates), desc="Validating", unit="img"))

    corrupted = [c for c in inspected if c.error]
    valid = [c for c in inspected if not c.error]
    for c in corrupted:
        logger.warning("Dropping %s: %s", c.path, c.error)

    groups = group_duplicates(valid, cfg.phash_threshold)
    df = pd.DataFrame({
        "source": [c.source for c in valid],
        "source_path": [str(c.path) for c in valid],
        "label": [c.label for c in valid],
        "md5": [c.md5 for c in valid],
        "phash": [f"{c.phash:016x}" for c in valid],
        "width": [c.width for c in valid],
        "height": [c.height for c in valid],
        "group": groups,
    })

    labels_per_group = df.groupby("group")["label"].nunique()
    conflicting_groups = set(labels_per_group[labels_per_group > 1].index)
    conflicting = df[df["group"].isin(conflicting_groups)]
    df = df[~df["group"].isin(conflicting_groups)]
    # Keep the highest-resolution member of each duplicate group as its representative.
    df = df.assign(pixels=df["width"] * df["height"]).sort_values(["group", "pixels"], ascending=[True, False])
    duplicates_removed = len(df) - df["group"].nunique()
    df = df.drop_duplicates("group").drop(columns="pixels").reset_index(drop=True)

    df["split"] = split_groups(df, cfg)

    if paths.dataset_dir.exists():
        for split in SPLITS:
            shutil.rmtree(paths.dataset_dir / split, ignore_errors=True)
    for split in SPLITS:
        for cls in CLASS_NAMES:
            (paths.dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for _, row in df.iterrows()]
        df["dataset_path"] = list(tqdm(pool.map(lambda r: _write_image(r, paths.dataset_dir, cfg.max_stored_side), rows),
                                       total=len(rows), desc="Writing dataset", unit="img"))

    _portable_manifest(df, paths.raw_dir).to_csv(paths.metrics_dir / "dataset_manifest.csv", index=False)
    counts = {split: {cls: int(((df["split"] == split) & (df["label"] == cls)).sum()) for cls in CLASS_NAMES}
              for split in SPLITS}
    stats = {
        "seed": cfg.seed,
        "split_fractions": {"train": 1 - cfg.val_fraction - cfg.test_fraction,
                            "validation": cfg.val_fraction, "test": cfg.test_fraction},
        "sources": [
            {"key": s.key, "title": s.title, "license": s.license, "url": s.url, "citation": s.citation,
             "images_used": int((df["source"] == s.key).sum())}
            for s in SOURCES
        ],
        "raw_images_found": len(candidates),
        "unlabelled_skipped": unlabeled,
        "corrupted_or_too_small_removed": len(corrupted),
        "duplicates_removed": int(duplicates_removed),
        "conflicting_label_images_removed": int(len(conflicting)),
        "final_images": int(len(df)),
        "class_totals": {cls: int((df["label"] == cls).sum()) for cls in CLASS_NAMES},
        "split_counts": counts,
        "phash_threshold": cfg.phash_threshold,
        "max_stored_side": cfg.max_stored_side,
    }
    save_json(stats, paths.metrics_dir / "dataset_stats.json")
    _plot_distribution(counts, paths.plots_dir / "class_distribution.png")
    return stats


def _portable_manifest(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Store source paths relative to the raw-data folder so the manifest is machine-independent."""
    def relative(path: str) -> str:
        try:
            return Path(path).relative_to(raw_dir).as_posix()
        except ValueError:  # manually registered source outside raw_dir
            return Path(path).name
    return df.assign(source_path=df["source_path"].map(relative), dataset_path=df["dataset_path"].str.replace("\\", "/"))


def _plot_distribution(counts: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(SPLITS))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (cls, color) in enumerate((("fire", "#ef4444"), ("non_fire", "#22c55e"))):
        values = [counts[s][cls] for s in SPLITS]
        bars = ax.bar(x + (i - 0.5) * 0.38, values, 0.38, label=cls, color=color)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, [s.capitalize() for s in SPLITS])
    ax.set_ylabel("Images")
    ax.set_title("Class distribution per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
