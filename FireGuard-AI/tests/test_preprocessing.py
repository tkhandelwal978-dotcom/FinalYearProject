import io

import pytest
import torch
from PIL import Image

from app.utils.image_validation import sanitize_filename
from ml.config import IMAGENET_MEAN, IMAGENET_STD
from ml.preprocessing.image_ops import InvalidImageError, load_rgb, resize_max_side
from ml.preprocessing.transforms import build_eval_transform, build_train_transform
from tests.conftest import image_bytes


def test_load_rgb_converts_rgba_and_grayscale():
    assert load_rgb(image_bytes(color=(10, 20, 30, 128), fmt="PNG", mode="RGBA")).mode == "RGB"
    assert load_rgb(image_bytes(color=128, fmt="PNG", mode="L")).mode == "RGB"


def test_load_rgb_rejects_garbage_and_truncated_files():
    with pytest.raises(InvalidImageError):
        load_rgb(b"definitely not an image")
    truncated = image_bytes(size=(400, 400))[:500]
    with pytest.raises(InvalidImageError):
        load_rgb(truncated)


def test_load_rgb_enforces_minimum_size():
    with pytest.raises(InvalidImageError, match="too small"):
        load_rgb(image_bytes(size=(20, 20)), min_side=32)


def test_resize_max_side_keeps_aspect_ratio():
    img = resize_max_side(Image.new("RGB", (1000, 500)), 200)
    assert img.size == (200, 100)
    small = Image.new("RGB", (100, 50))
    assert resize_max_side(small, 200) is small


def test_eval_transform_shape_and_normalization():
    tensor = build_eval_transform(224)(Image.new("RGB", (640, 360), (255, 255, 255)))
    assert tensor.shape == (3, 224, 224)
    expected = torch.tensor([(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
    assert torch.allclose(tensor.mean(dim=(1, 2)), expected, atol=1e-4)


def test_train_transform_is_random_but_shape_stable():
    transform = build_train_transform(224)
    img = Image.effect_noise((300, 300), 64).convert("RGB")
    a, b = transform(img), transform(img)
    assert a.shape == b.shape == (3, 224, 224)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("raw, expected", [
    ("../../etc/passwd.jpg", "passwd.jpg"),
    ("C:\\Users\\me\\fire photo.png", "fire photo.png"),
    ("<script>.jpg", "_script_.jpg"),
    ("", "upload"),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (300, 100))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 degrees
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", exif=exif)
    assert load_rgb(buffer.getvalue()).size == (100, 300)
