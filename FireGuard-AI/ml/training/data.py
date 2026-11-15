"""Dataset loading for training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ml.config import CLASS_NAMES
from ml.preprocessing.image_ops import is_image_file, load_rgb


class FireImageDataset(Dataset):
    """Reads ``<split_dir>/<class>/*.jpg``; label 1 = fire, 0 = non_fire."""

    def __init__(self, split_dir: Path, transform) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for index, cls in enumerate(CLASS_NAMES):
            class_dir = split_dir / cls
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder {class_dir}. Run scripts/prepare_dataset.py first.")
            self.samples += [(p, index) for p in sorted(class_dir.iterdir()) if is_image_file(p)]
        if not self.samples:
            raise RuntimeError(f"No images found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return self.transform(load_rgb(path)), torch.tensor(label, dtype=torch.float32)

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def class_counts(self) -> dict[str, int]:
        return {cls: self.labels.count(i) for i, cls in enumerate(CLASS_NAMES)}


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
