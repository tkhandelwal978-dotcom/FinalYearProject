"""Registry of the public datasets FireGuard AI trains on.

Each source declares how its folder names map onto our two classes. Folder
matching is done on lower-cased path components, so the mapping stays robust to
the slightly different layouts each publisher uses. Any image whose path does
not match a rule is skipped (and counted) rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSource:
    key: str
    kaggle_slug: str
    title: str
    license: str
    url: str
    citation: str
    # Ordered (folder-name, class) rules; the first rule whose folder name appears
    # in the image's relative path decides its label.
    label_rules: tuple[tuple[str, str], ...]
    # Fallback (filename-prefix, class) rules for folders that mix both classes.
    filename_rules: tuple[tuple[str, str], ...] = ()


SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        key="fire_dataset",
        kaggle_slug="phylake1337/fire-dataset",
        title="FIRE Dataset (Saied, 2018)",
        license="CC0: Public Domain",
        url="https://www.kaggle.com/datasets/phylake1337/fire-dataset",
        citation="A. Saied, FIRE Dataset, Kaggle, 2018.",
        label_rules=(("non_fire_images", "non_fire"), ("fire_images", "fire")),
    ),
    DatasetSource(
        key="deepfire",
        kaggle_slug="alik05/forest-fire-dataset",
        title="Forest Fire Dataset / DeepFire (Khan et al., 2022)",
        license="CC BY 4.0",
        url="https://www.kaggle.com/datasets/alik05/forest-fire-dataset",
        citation=(
            "A. Khan, B. Hassan, S. Khan, R. Ahmed, A. Abuassba, 'DeepFire: A Novel Dataset and Deep "
            "Transfer Learning Benchmark for Forest Fire Detection', Mobile Information Systems, 2022."
        ),
        label_rules=(("nofire", "non_fire"), ("non_fire", "non_fire"), ("fire", "fire")),
        # Its "Testing" folder is flat; labels are encoded as fire_XXXX.jpg / nofire_XXXX.jpg.
        filename_rules=(("nofire", "non_fire"), ("fire", "fire")),
    ),
    DatasetSource(
        key="wildfire_detection",
        kaggle_slug="brsdincer/wildfire-detection-image-data",
        title="Wildfire Detection Image Data",
        license="Open Database License (ODbL)",
        url="https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data",
        citation="B. Dincer, Wildfire Detection Image Data, Kaggle.",
        label_rules=(("nofire", "non_fire"), ("no_fire", "non_fire"), ("fire", "fire")),
    ),
)


def get_source(key: str) -> DatasetSource:
    for source in SOURCES:
        if source.key == key:
            return source
    raise KeyError(f"Unknown dataset source '{key}'. Known: {[s.key for s in SOURCES]}")


def label_for_path(source: DatasetSource, relative_parts: tuple[str, ...]) -> str | None:
    """Return the class label for an image given its path components inside the source."""
    parts = [p.lower() for p in relative_parts[:-1]]  # directories only, not the filename
    for folder, label in source.label_rules:
        if folder in parts:
            return label
    filename = relative_parts[-1].lower()
    for prefix, label in source.filename_rules:
        if filename.startswith(prefix):
            return label
    return None
