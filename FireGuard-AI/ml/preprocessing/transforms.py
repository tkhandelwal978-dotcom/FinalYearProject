"""Image transforms for training and inference.

Augmentations are deliberately mild and colour-preserving: no hue or
saturation jitter, because colour is one of fire's strongest visual cues and
shifting it could teach the model that recoloured flames are not fire.
"""

from __future__ import annotations

from torchvision import transforms

from ml.config import IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        # Random crop + zoom: keeps 70-100% of the image area so flames are rarely cropped out.
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int) -> transforms.Compose:
    # Resize the whole frame (no centre-crop) so fire near the image border is never discarded.
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocessing_config(image_size: int) -> dict:
    """Serializable description of inference preprocessing, saved alongside the model."""
    return {
        "image_size": [image_size, image_size],
        "resize": "direct (no crop), bilinear, antialiased",
        "color_mode": "RGB",
        "scale": "pixel / 255",
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
    }
