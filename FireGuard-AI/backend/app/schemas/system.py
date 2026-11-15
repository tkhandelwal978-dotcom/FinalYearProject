from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.schemas.prediction import PredictionOut


class HealthOut(BaseModel):
    status: str
    app_name: str
    app_version: str
    model_loaded: bool
    model_version: str | None
    device: str | None
    database: str
    model_error: str | None = None


class DailyTrend(BaseModel):
    date: str
    fire: int
    non_fire: int
    total: int


class ConfidenceBucket(BaseModel):
    range: str
    fire: int
    non_fire: int


class AnalyticsOut(BaseModel):
    total_predictions: int
    fire_detections: int
    non_fire_detections: int
    fire_detection_rate: float
    average_confidence: float | None
    average_inference_ms: float | None
    predictions_last_24h: int
    fire_last_24h: int
    daily_trend: list[DailyTrend]
    confidence_distribution: list[ConfidenceBucket]
    recent_predictions: list[PredictionOut]


class ModelInfoOut(BaseModel):
    model_loaded: bool
    model_name: str | None
    model_version: str | None
    architecture: str | None
    framework: str | None
    device: str | None
    image_size: list[int] | None
    decision_threshold: float | None
    class_names: list[str] | None
    preprocessing: dict[str, Any] | None
    hyperparameters: dict[str, Any] | None
    training: dict[str, Any] | None
    dataset: dict[str, Any] | None
    test_metrics: dict[str, Any] | None
    training_history: list[dict[str, Any]]
    curves: dict[str, Any] | None
    plots: dict[str, str]
