"""Stores uploaded images, thumbnails and Grad-CAM renders under the storage directory.

Files are named by prediction ID (never by the user's filename) and re-encoded
from decoded pixels, which strips metadata and any non-image payload.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ml.preprocessing.image_ops import resize_max_side

MEDIA_PREFIX = "/media"


class StorageService:
    def __init__(self, storage_dir: Path) -> None:
        self.root = storage_dir
        self.uploads = storage_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)

    def _save(self, image: Image.Image, name: str, max_side: int | None = None, quality: int = 92) -> str:
        if max_side:
            image = resize_max_side(image, max_side)
        path = self.uploads / name
        image.convert("RGB").save(path, format="JPEG", quality=quality)
        return path.relative_to(self.root).as_posix()

    def save_upload(self, prediction_id: str, image: Image.Image) -> tuple[str, str]:
        original = self._save(image, f"{prediction_id}.jpg", max_side=2048)
        thumbnail = self._save(image, f"{prediction_id}_thumb.jpg", max_side=256, quality=85)
        return original, thumbnail

    def save_gradcam(self, prediction_id: str, heatmap: Image.Image, overlay: Image.Image) -> tuple[str, str]:
        return (self._save(heatmap, f"{prediction_id}_heatmap.jpg", max_side=1024),
                self._save(overlay, f"{prediction_id}_overlay.jpg", max_side=1024))

    def absolute(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("Path escapes the storage directory")
        return path

    def delete(self, *relative_paths: str | None) -> None:
        for relative in relative_paths:
            if relative:
                self.absolute(relative).unlink(missing_ok=True)

    @staticmethod
    def url(relative: str | None) -> str | None:
        return f"{MEDIA_PREFIX}/{relative}" if relative else None
