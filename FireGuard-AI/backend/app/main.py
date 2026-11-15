"""FireGuard AI FastAPI application.

Run from the ``backend`` directory:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.database.session import Database
from app.services.model_service import ModelService, ModelUnavailableError
from app.services.storage_service import StorageService
from app.utils.image_validation import UploadValidationError

logger = logging.getLogger("fireguard")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.database = Database(settings.database_url)
        app.state.database.create_tables()
        app.state.storage = StorageService(settings.storage_dir)
        # Load the model exactly once per process; every request reuses it.
        app.state.model_service = ModelService(settings.model_path, settings.device)
        logger.info("FireGuard AI ready (model loaded: %s)", app.state.model_service.loaded)
        yield
        app.state.database.engine.dispose()

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.app_version,
        description="Image-based fire detection with an EfficientNet-B0 model fine-tuned on public fire datasets.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.exception_handler(UploadValidationError)
    async def upload_error(_: Request, exc: UploadValidationError):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable(_: Request, exc: ModelUnavailableError):
        return JSONResponse(status_code=503, content={"detail": f"Model unavailable: {exc}"})

    @app.exception_handler(RequestValidationError)
    async def request_validation(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422,
                            content={"detail": "Invalid request", "errors": jsonable_encoder(exc.errors())})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(router)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "plots").mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=settings.storage_dir), name="media")
    app.mount("/artifacts/plots", StaticFiles(directory=settings.artifacts_dir / "plots"), name="plots")

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": settings.app_name, "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
