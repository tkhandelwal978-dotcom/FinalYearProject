"""Low-level image helpers shared by dataset preparation and inference."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class InvalidImageError(ValueError):
    """Raised when bytes or a file cannot be decoded into a usable RGB image."""


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(source: Path | bytes, min_side: int = 1) -> Image.Image:
    """Decode an image fully, honour EXIF orientation and return it as RGB.

    ``Image.verify`` catches structural corruption, but truncated files only fail
    once pixel data is decoded, so the image is reopened and ``load()``-ed too.
    """
    try:
        opener = (lambda: Image.open(io.BytesIO(source))) if isinstance(source, bytes) else (lambda: Image.open(source))
        with opener() as probe:
            probe.verify()
        with opener() as img:
            img.load()
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "P"):
                # Composite transparency onto white so transparent pixels don't become black.
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                img = background
            else:
                img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"Unreadable image: {exc}") from exc

    if min(img.size) < min_side:
        raise InvalidImageError(f"Image too small: {img.size[0]}x{img.size[1]} (minimum side {min_side}px)")
    return img


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    scale = max_side / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
