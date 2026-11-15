"""Load the trained checkpoint once and run fire predictions (with optional Grad-CAM)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ml.preprocessing.transforms import build_eval_transform
from ml.training.model import FireClassifier

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    predicted_class: str  # "fire" | "non_fire"
    fire_probability: float
    no_fire_probability: float
    confidence: float  # probability of the predicted class
    threshold: float
    inference_ms: float


@dataclass
class GradCamResult:
    heatmap: Image.Image  # colourised activation map, same size as the input image
    overlay: Image.Image  # heatmap blended onto the input image


def load_checkpoint(path: Path, device: torch.device) -> tuple[FireClassifier, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at {path}. Train it with `python scripts/train.py`.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    model = FireClassifier(dropout=metadata["hyperparameters"]["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, metadata


class FirePredictor:
    """Thread-safe wrapper around the trained model for single-image inference."""

    def __init__(self, checkpoint_path: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.metadata = load_checkpoint(checkpoint_path, self.device)
        self.image_size = self.metadata["preprocessing"]["image_size"][0]
        self.threshold = float(self.metadata["decision_threshold"])
        self.transform = build_eval_transform(self.image_size)
        # Grad-CAM temporarily registers hooks on the model, so inference is serialized.
        self._lock = threading.Lock()
        self._warmup()
        logger.info("Loaded %s v%s on %s (threshold %.2f)", self.metadata["model_name"],
                    self.metadata["model_version"], self.device, self.threshold)

    def _warmup(self) -> None:
        with torch.no_grad():
            self.model(torch.zeros(1, 3, self.image_size, self.image_size, device=self.device))

    def _to_prediction(self, fire_probability: float, elapsed_ms: float) -> Prediction:
        is_fire = fire_probability >= self.threshold
        return Prediction(
            predicted_class="fire" if is_fire else "non_fire",
            fire_probability=fire_probability,
            no_fire_probability=1.0 - fire_probability,
            confidence=fire_probability if is_fire else 1.0 - fire_probability,
            threshold=self.threshold,
            inference_ms=elapsed_ms,
        )

    def predict(self, image: Image.Image) -> Prediction:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self._lock, torch.no_grad():
            start = time.perf_counter()
            fire_probability = torch.sigmoid(self.model(tensor)).item()
            elapsed = (time.perf_counter() - start) * 1000
        return self._to_prediction(fire_probability, elapsed)

    def predict_with_gradcam(self, image: Image.Image) -> tuple[Prediction, GradCamResult]:
        """Predict and explain: Grad-CAM over the last conv stage for the predicted class."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        activations: dict[str, torch.Tensor] = {}

        def forward_hook(_module, _inputs, output):
            output.retain_grad()
            activations["value"] = output

        with self._lock:
            handle = self.model.gradcam_layer.register_forward_hook(forward_hook)
            try:
                start = time.perf_counter()
                with torch.enable_grad():
                    logit = self.model(tensor)
                    fire_probability = torch.sigmoid(logit).item()
                    prediction = self._to_prediction(fire_probability, (time.perf_counter() - start) * 1000)
                    # Explain the predicted class: +logit for fire, -logit for "no fire".
                    target = logit if prediction.predicted_class == "fire" else -logit
                    self.model.zero_grad(set_to_none=True)
                    target.sum().backward()
                acts = activations["value"]
                weights = acts.grad.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * acts).sum(dim=1)).squeeze(0).detach().cpu().numpy()
            finally:
                handle.remove()
                self.model.zero_grad(set_to_none=True)

        return prediction, self._render_cam(cam, image)

    @staticmethod
    def _render_cam(cam: np.ndarray, image: Image.Image) -> GradCamResult:
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        rgb = np.asarray(image.convert("RGB"))
        cam_resized = cv2.resize(cam.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0, 1))
        heatmap = cv2.cvtColor(cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb, 0.55, heatmap, 0.45, 0)
        return GradCamResult(heatmap=Image.fromarray(heatmap), overlay=Image.fromarray(overlay))
