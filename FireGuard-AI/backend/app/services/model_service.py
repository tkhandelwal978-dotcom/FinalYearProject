"""Owns the single FirePredictor instance loaded at application start-up."""

from __future__ import annotations

import logging
from pathlib import Path

from ml.inference.predictor import FirePredictor

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.model_path = model_path
        self.predictor: FirePredictor | None = None
        self.error: str | None = None
        try:
            self.predictor = FirePredictor(model_path, device)
        except Exception as exc:  # keep the API up (health reports the problem) even without a model
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("Model failed to load: %s", self.error)

    @property
    def loaded(self) -> bool:
        return self.predictor is not None

    @property
    def metadata(self) -> dict | None:
        return self.predictor.metadata if self.predictor else None

    @property
    def device(self) -> str | None:
        return str(self.predictor.device) if self.predictor else None

    def require(self) -> FirePredictor:
        if self.predictor is None:
            raise ModelUnavailableError(self.error or "Model is not loaded")
        return self.predictor
