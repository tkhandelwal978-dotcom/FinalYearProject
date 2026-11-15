from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from app.services.storage_service import StorageService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    yield from request.app.state.database.session()


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_storage(request: Request) -> StorageService:
    return request.app.state.storage


def get_prediction_service(
    db: Session = Depends(get_db),
    models: ModelService = Depends(get_model_service),
    storage: StorageService = Depends(get_storage),
) -> PredictionService:
    return PredictionService(db, models, storage)
