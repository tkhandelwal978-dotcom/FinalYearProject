"""Runs inference for an upload and manages persisted prediction records."""

from __future__ import annotations

import logging
import math
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.prediction import PredictionRecord
from app.schemas.prediction import PaginatedPredictions, PredictionOut
from app.services.model_service import ModelService
from app.services.storage_service import StorageService
from app.utils.image_validation import ValidatedImage
from ml.config import CLASS_DISPLAY_NAMES

logger = logging.getLogger(__name__)


def to_schema(record: PredictionRecord) -> PredictionOut:
    return PredictionOut(
        id=record.id,
        filename=record.filename,
        predicted_class=record.predicted_class,
        label=CLASS_DISPLAY_NAMES[record.predicted_class],
        fire_probability=record.fire_probability,
        no_fire_probability=record.no_fire_probability,
        confidence=record.confidence,
        threshold=record.threshold,
        inference_ms=record.inference_ms,
        image_width=record.image_width,
        image_height=record.image_height,
        file_size_bytes=record.file_size_bytes,
        model_version=record.model_version,
        created_at=record.created_at,
        image_url=StorageService.url(record.image_path),
        thumbnail_url=StorageService.url(record.thumbnail_path),
        heatmap_url=StorageService.url(record.heatmap_path),
        overlay_url=StorageService.url(record.overlay_path),
        report_url=f"/api/history/{record.id}/report",
    )


class PredictionService:
    def __init__(self, db: Session, models: ModelService, storage: StorageService) -> None:
        self.db, self.models, self.storage = db, models, storage

    def create(self, upload: ValidatedImage, with_gradcam: bool) -> PredictionRecord:
        predictor = self.models.require()
        prediction_id = str(uuid.uuid4())
        heatmap_path = overlay_path = None

        if with_gradcam:
            result, cam = predictor.predict_with_gradcam(upload.image)
        else:
            result, cam = predictor.predict(upload.image), None

        image_path, thumbnail_path = self.storage.save_upload(prediction_id, upload.image)
        if cam is not None:
            heatmap_path, overlay_path = self.storage.save_gradcam(prediction_id, cam.heatmap, cam.overlay)

        record = PredictionRecord(
            id=prediction_id,
            filename=upload.safe_filename,
            image_path=image_path,
            thumbnail_path=thumbnail_path,
            heatmap_path=heatmap_path,
            overlay_path=overlay_path,
            predicted_class=result.predicted_class,
            fire_probability=round(result.fire_probability, 6),
            no_fire_probability=round(result.no_fire_probability, 6),
            confidence=round(result.confidence, 6),
            threshold=result.threshold,
            inference_ms=round(result.inference_ms, 2),
            image_width=upload.image.width,
            image_height=upload.image.height,
            file_size_bytes=upload.size_bytes,
            model_version=predictor.metadata["model_version"],
        )
        try:
            self.db.add(record)
            self.db.commit()
        except Exception:
            self.db.rollback()
            self.storage.delete(image_path, thumbnail_path, heatmap_path, overlay_path)
            raise
        logger.info("Prediction %s: %s (P(fire)=%.4f) for %s", prediction_id, result.predicted_class,
                    result.fire_probability, upload.safe_filename)
        return record

    def get(self, prediction_id: str) -> PredictionRecord | None:
        return self.db.get(PredictionRecord, prediction_id)

    def list(self, page: int, page_size: int, predicted_class: str | None = None,
             search: str | None = None, sort: str = "newest") -> PaginatedPredictions:
        query = select(PredictionRecord)
        if predicted_class:
            query = query.where(PredictionRecord.predicted_class == predicted_class)
        if search:
            pattern = f"%{search.strip()}%"
            query = query.where(PredictionRecord.filename.ilike(pattern) | PredictionRecord.id.ilike(pattern))
        total = self.db.scalar(select(func.count()).select_from(query.subquery())) or 0
        order = {
            "newest": PredictionRecord.created_at.desc(),
            "oldest": PredictionRecord.created_at.asc(),
            "confidence": PredictionRecord.confidence.desc(),
            "fire_probability": PredictionRecord.fire_probability.desc(),
        }[sort]
        rows = self.db.scalars(query.order_by(order).offset((page - 1) * page_size).limit(page_size)).all()
        return PaginatedPredictions(items=[to_schema(r) for r in rows], total=total, page=page,
                                    page_size=page_size, pages=max(1, math.ceil(total / page_size)))

    def all_filtered(self, predicted_class: str | None = None) -> list[PredictionRecord]:
        query = select(PredictionRecord).order_by(PredictionRecord.created_at.desc())
        if predicted_class:
            query = query.where(PredictionRecord.predicted_class == predicted_class)
        return list(self.db.scalars(query).all())

    def delete(self, record: PredictionRecord) -> None:
        paths = (record.image_path, record.thumbnail_path, record.heatmap_path, record.overlay_path)
        self.db.delete(record)
        self.db.commit()
        self.storage.delete(*paths)
