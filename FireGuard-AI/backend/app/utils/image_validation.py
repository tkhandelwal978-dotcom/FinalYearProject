"""Upload validation: extension, declared type, size and actual decoded format."""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath

from PIL import Image

from ml.preprocessing.image_ops import InvalidImageError, load_rgb

# Formats we accept, keyed by what Pillow detects from the bytes themselves.
ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs early


class UploadValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ValidatedImage:
    image: Image.Image
    detected_format: str
    safe_filename: str
    size_bytes: int


def sanitize_filename(filename: str | None, max_length: int = 120) -> str:
    """Return a display-safe base filename (no directories, control or exotic characters)."""
    name = PurePath((filename or "").replace("\\", "/")).name
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    if not name:
        name = "upload"
    if len(name) > max_length:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: max_length - len(ext) - 1] + dot + ext) if dot else name[:max_length]
    return name


def validate_upload(data: bytes, filename: str | None, content_type: str | None,
                    allowed_extensions: tuple[str, ...], max_bytes: int) -> ValidatedImage:
    safe_name = sanitize_filename(filename)
    extension = PurePath(safe_name).suffix.lower()
    if extension not in allowed_extensions:
        raise UploadValidationError(
            f"Unsupported file type '{extension or 'none'}'. Allowed: {', '.join(allowed_extensions)}", 415)
    if content_type and content_type not in (*ALLOWED_FORMATS.values(), "application/octet-stream"):
        raise UploadValidationError(f"Unsupported content type '{content_type}'", 415)
    if not data:
        raise UploadValidationError("The uploaded file is empty")
    if len(data) > max_bytes:
        raise UploadValidationError(f"File is too large ({len(data) / 1_048_576:.1f} MB). "
                                    f"Maximum is {max_bytes / 1_048_576:.0f} MB", 413)

    try:
        with Image.open(io.BytesIO(data)) as probe:
            detected = probe.format
    except Exception as exc:
        raise UploadValidationError("The file is not a valid image") from exc
    if detected not in ALLOWED_FORMATS:
        raise UploadValidationError(f"Image format '{detected}' is not supported", 415)

    try:
        image = load_rgb(data, min_side=32)
    except InvalidImageError as exc:
        raise UploadValidationError(str(exc)) from exc
    return ValidatedImage(image=image, detected_format=detected, safe_filename=safe_name, size_bytes=len(data))
