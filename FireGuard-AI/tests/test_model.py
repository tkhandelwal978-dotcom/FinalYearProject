import numpy as np
import pytest
import torch

from ml.config import CLASS_NAMES
from ml.inference.predictor import FirePredictor
from ml.preprocessing.image_ops import load_rgb
from ml.training.model import FireClassifier
from ml.training.trainer import tune_threshold
from tests.conftest import CHECKPOINT, image_bytes, requires_model, test_split_images


def test_classifier_outputs_one_logit_per_image():
    model = FireClassifier(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2,)


def test_freeze_and_unfreeze_backbone():
    model = FireClassifier(pretrained=False)
    model.freeze_backbone()
    head_only = model.trainable_parameters()
    assert head_only == sum(p.numel() for p in model.head.parameters())
    model.unfreeze_last_blocks(3)
    assert model.trainable_parameters() > head_only
    assert not any(p.requires_grad for p in model.features[0].parameters())
    assert all(p.requires_grad for p in model.features[-1].parameters())


def test_threshold_tuning_respects_recall_floor():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.4, 0.35])
    result = tune_threshold(probs, labels, min_recall=1.0)
    assert result["met_recall_target"] and result["recall"] == 1.0
    assert result["threshold"] <= 0.3


@pytest.fixture(scope="module")
def predictor() -> FirePredictor:
    return FirePredictor(CHECKPOINT)


@requires_model
def test_checkpoint_loads_with_metadata(predictor):
    meta = predictor.metadata
    assert meta["class_names"] == list(CLASS_NAMES)
    assert 0 < predictor.threshold < 1
    assert meta["preprocessing"]["image_size"] == [224, 224]


@requires_model
def test_predictions_on_held_out_images_are_mostly_correct(predictor):
    correct = total = 0
    for cls in CLASS_NAMES:
        for path in test_split_images(cls, 15):
            result = predictor.predict(load_rgb(path))
            assert abs(result.fire_probability + result.no_fire_probability - 1) < 1e-6
            correct += result.predicted_class == cls
            total += 1
    if total == 0:
        pytest.skip("prepared test split not available")
    assert correct / total >= 0.85


@requires_model
def test_gradcam_matches_plain_prediction(predictor):
    paths = test_split_images("fire", 1)
    image = load_rgb(paths[0]) if paths else load_rgb(image_bytes())
    plain = predictor.predict(image)
    explained, cam = predictor.predict_with_gradcam(image)
    assert abs(plain.fire_probability - explained.fire_probability) < 1e-3
    assert cam.overlay.size == image.size and cam.heatmap.size == image.size
