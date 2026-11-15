"""Environment-driven backend settings (prefix ``FIREGUARD_``, optional ``backend/.env``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import PROJECT_ROOT

BACKEND_DIR = PROJECT_ROOT / "backend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIREGUARD_", env_file=BACKEND_DIR / ".env", extra="ignore")

    app_name: str = "FireGuard AI"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'storage' / 'fireguard.db').as_posix()}"
    storage_dir: Path = BACKEND_DIR / "storage"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    model_path: Path = PROJECT_ROOT / "artifacts" / "model" / "fireguard_efficientnet_b0.pt"
    device: str | None = None  # "cpu" | "cuda"; auto-detected when unset

    max_upload_mb: float = 10.0
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    enable_gradcam: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
