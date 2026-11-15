"""EfficientNet-B0 transfer-learning model for binary fire classification.

Architecture:
    Input (3x224x224) -> EfficientNet-B0 features (ImageNet weights)
    -> Global Average Pooling -> BatchNorm -> Dropout -> Dense(1) -> Sigmoid = P(fire)

The network returns raw logits; the sigmoid is applied by the loss during
training (BCEWithLogitsLoss, which is numerically stabler) and explicitly at
inference time.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ARCHITECTURE_NAME = "EfficientNet-B0 + custom head (GAP -> BatchNorm -> Dropout -> Dense -> Sigmoid)"
BACKBONE_FEATURES = 1280


class FireClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features  # 9 sequential stages: stem, 7 MBConv stages, final 1x1 conv
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(BACKBONE_FEATURES),
            nn.Dropout(dropout),
            nn.Linear(BACKBONE_FEATURES, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).squeeze(1)

    # --- freezing helpers -------------------------------------------------------------

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int) -> None:
        """Unfreeze the last ``n_blocks`` of the 9 backbone stages for fine-tuning."""
        self.freeze_backbone()
        n_blocks = max(0, min(n_blocks, len(self.features)))
        for block in list(self.features)[len(self.features) - n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def set_frozen_bn_eval(self) -> None:
        """Keep BatchNorm layers of frozen stages in eval mode.

        Frozen stages should behave exactly as pretrained; letting their BN running
        statistics drift on small fire batches would silently change their features.
        """
        for block in self.features:
            if not any(p.requires_grad for p in block.parameters()):
                for module in block.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last convolutional stage, whose activations Grad-CAM explains."""
        return self.features[-1]
