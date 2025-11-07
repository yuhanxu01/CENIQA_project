"""
Simple CNN baseline model WITHOUT GMM clustering
Direct regression: CNN backbone → MLP → Quality Score
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import CNNBackbone
from regressors import MonotonicMLP


class SimpleCNNModel(nn.Module):
    """
    Simple CNN baseline without GMM clustering.
    Architecture: CNN Backbone → MLP Regressor → Quality Score
    """
    def __init__(self, backbone_name='resnet18', feature_dim=512,
                 hidden_dim=512, dropout=0.3, freeze_backbone=False):
        super().__init__()

        self.backbone = CNNBackbone(
            model_name=backbone_name,
            pretrained=True,
            feature_dim=feature_dim
        )

        if freeze_backbone:
            for param in self.backbone.model.parameters():
                param.requires_grad = False
            for param in self.backbone.proj.parameters():
                param.requires_grad = True

        # Direct regression without GMM
        self.regressor = MonotonicMLP(
            input_dim=feature_dim,  # No GMM posteriors, just features
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x, return_all=False):
        """
        Forward pass.

        Args:
            x: input images [batch_size, 3, H, W]
            return_all: if True, return features as well

        Returns:
            quality_score: [batch_size]
            or dict with 'quality_score' and 'features' if return_all=True
        """
        features = self.backbone(x)
        quality_score = self.regressor(features).squeeze(-1)

        if return_all:
            return {
                'quality_score': quality_score,
                'features': features
            }
        return quality_score


def build_simple_model(backbone='resnet18', feature_dim=512,
                       hidden_dim=512, dropout=0.3):
    """Factory function to build simple model."""
    return SimpleCNNModel(
        backbone_name=backbone,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=False
    )


if __name__ == '__main__':
    # Test the model
    model = SimpleCNNModel()
    x = torch.randn(4, 3, 224, 224)

    # Test forward pass
    output = model(x, return_all=True)
    print(f"Quality score shape: {output['quality_score'].shape}")
    print(f"Features shape: {output['features'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
