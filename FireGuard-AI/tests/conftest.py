"""Shared fixtures. Model-dependent tests use the real trained checkpoint and skip if it is missing."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from ml.config import PathsConfig  # noqa: E402

PATHS = PathsConfig()
CHECKPOINT = PATHS.checkpoint_path
requires_model = pytest.mark.skipif(not CHECKPOINT.is_file(), reason="trained checkpoint not found; run scripts/train.py")


def image_bytes(color=(200, 80, 20), size=(256, 192), fmt="JPEG", mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def test_split_images(cls: str, limit: int) -> list[Path]:
    folder = PATHS.dataset_dir / "test" / cls
    return sorted(folder.glob("*.jpg"))[:limit] if folder.is_dir() else []


test_split_images.__test__ = False  # helper, not a test


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        storage_dir=tmp_path / "storage",
        model_path=CHECKPOINT,
        max_upload_mb=2,
        log_level="WARNING",
    )


@pytest.fixture
def client(settings: Settings):
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as test_client:
        yield test_client
