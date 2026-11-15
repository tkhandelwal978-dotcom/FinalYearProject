from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PredictedClass = Literal["fire", "non_fire"]


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    predicted_class: PredictedClass
    label: str = Field(description="Human-readable result, e.g. 'FIRE DETECTED'")
    fire_probability: float
    no_fire_probability: float
    confidence: float
    threshold: float
    inference_ms: float
    image_width: int
    image_height: int
    file_size_bytes: int
    model_version: str
    created_at: datetime
    image_url: str
    thumbnail_url: str
    heatmap_url: str | None = None
    overlay_url: str | None = None
    report_url: str


class PaginatedPredictions(BaseModel):
    items: list[PredictionOut]
    total: int
    page: int
    page_size: int
    pages: int


class DeleteResponse(BaseModel):
    id: str
    deleted: bool = True
