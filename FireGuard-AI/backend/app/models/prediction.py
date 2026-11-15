from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255))  # sanitized original filename (display only)
    image_path: Mapped[str] = mapped_column(String(512))  # path relative to the storage dir
    thumbnail_path: Mapped[str] = mapped_column(String(512))
    heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    predicted_class: Mapped[str] = mapped_column(String(16), index=True)
    fire_probability: Mapped[float] = mapped_column(Float)
    no_fire_probability: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    inference_ms: Mapped[float] = mapped_column(Float)
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    __table_args__ = (Index("ix_predictions_class_created", "predicted_class", "created_at"),)
