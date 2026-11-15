"""HTTP routes for FireGuard AI (mounted under /api)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_model_service, get_prediction_service, get_settings_dep, get_storage
from app.core.config import Settings
from app.schemas.prediction import DeleteResponse, PaginatedPredictions, PredictionOut
from app.schemas.system import AnalyticsOut, HealthOut, ModelInfoOut
from app.services.analytics_service import compute_analytics
from app.services.model_info_service import build_model_info
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService, to_schema
from app.services.report_service import build_history_csv, build_pdf_report
from app.services.storage_service import StorageService
from app.utils.image_validation import validate_upload

router = APIRouter(prefix="/api")
ClassFilter = Literal["fire", "non_fire"]


@router.post("/predict", response_model=PredictionOut, status_code=201, tags=["prediction"],
             summary="Analyse an image for fire")
async def predict(
    file: UploadFile = File(..., description="JPG, JPEG, PNG or WEBP image"),
    gradcam: bool = Query(True, description="Also compute a Grad-CAM explanation"),
    settings: Settings = Depends(get_settings_dep),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionOut:
    service.models.require()  # fail fast with 503 before reading the upload
    data = await file.read(settings.max_upload_bytes + 1)
    upload = validate_upload(data, file.filename, file.content_type, settings.allowed_extensions,
                             settings.max_upload_bytes)
    record = await run_in_threadpool(service.create, upload, gradcam and settings.enable_gradcam)
    return to_schema(record)


@router.get("/health", response_model=HealthOut, tags=["system"])
def health(
    db: Session = Depends(get_db),
    models: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings_dep),
) -> HealthOut:
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:  # pragma: no cover - only on a broken database
        database = f"error: {exc}"
    return HealthOut(
        status="ok" if models.loaded and database == "ok" else "degraded",
        app_name=settings.app_name,
        app_version=settings.app_version,
        model_loaded=models.loaded,
        model_version=(models.metadata or {}).get("model_version"),
        device=models.device,
        database=database,
        model_error=models.error,
    )


@router.get("/analytics", response_model=AnalyticsOut, tags=["analytics"])
def analytics(days: int = Query(14, ge=1, le=365), db: Session = Depends(get_db)) -> AnalyticsOut:
    return compute_analytics(db, days)


@router.get("/history", response_model=PaginatedPredictions, tags=["history"])
def history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    predicted_class: ClassFilter | None = Query(None),
    search: str | None = Query(None, max_length=100, description="Match filename or prediction ID"),
    sort: Literal["newest", "oldest", "confidence", "fire_probability"] = "newest",
    service: PredictionService = Depends(get_prediction_service),
) -> PaginatedPredictions:
    return service.list(page, page_size, predicted_class, search, sort)


@router.get("/history/export", tags=["history"], summary="Download prediction history as CSV")
def export_history(
    predicted_class: ClassFilter | None = Query(None),
    service: PredictionService = Depends(get_prediction_service),
) -> Response:
    return Response(build_history_csv(service.all_filtered(predicted_class)), media_type="text/csv",
                     headers={"Content-Disposition": 'attachment; filename="fireguard_history.csv"'})


def _get_or_404(service: PredictionService, prediction_id: str):
    record = service.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Prediction '{prediction_id}' not found")
    return record


@router.get("/history/{prediction_id}", response_model=PredictionOut, tags=["history"])
def get_prediction(prediction_id: str, service: PredictionService = Depends(get_prediction_service)) -> PredictionOut:
    return to_schema(_get_or_404(service, prediction_id))


@router.delete("/history/{prediction_id}", response_model=DeleteResponse, tags=["history"])
def delete_prediction(prediction_id: str, service: PredictionService = Depends(get_prediction_service)) -> DeleteResponse:
    service.delete(_get_or_404(service, prediction_id))
    return DeleteResponse(id=prediction_id)


@router.get("/history/{prediction_id}/report", tags=["history"], summary="Download a PDF report for a prediction")
def prediction_report(
    prediction_id: str,
    service: PredictionService = Depends(get_prediction_service),
    storage: StorageService = Depends(get_storage),
) -> Response:
    record = _get_or_404(service, prediction_id)
    pdf = build_pdf_report(record, storage, service.models.metadata)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="fireguard_report_{record.id[:8]}.pdf"'})


@router.get("/model-info", response_model=ModelInfoOut, tags=["system"])
def model_info(
    models: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings_dep),
) -> ModelInfoOut:
    return build_model_info(models, settings.artifacts_dir)
