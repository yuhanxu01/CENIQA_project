"""Backbone models for feature extraction."""
import torch
import torch.nn as nn
import timm


class CNNBackbone(nn.Module):
    """CNN-based feature extractor using ResNet/EfficientNet."""
    def __init__(self, model_name='resnet50', pretrained=True, feature_dim=768):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.in_features = self.model.num_features
        self.proj = nn.Linear(self.in_features, feature_dim)
        
    def forward(self, x):
        features = self.model(x)
        return self.proj(features)


class ViTBackbone(nn.Module):
    """Vision Transformer backbone."""
    def __init__(self, model_name='vit_small_patch16_224', pretrained=True, feature_dim=768):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.in_features = self.model.num_features
        if self.in_features != feature_dim:
            self.proj = nn.Linear(self.in_features, feature_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        features = self.model(x)
        return self.proj(features)


class SwinBackbone(nn.Module):
    """Swin Transformer backbone."""
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True, feature_dim=768):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.in_features = self.model.num_features
        self.proj = nn.Linear(self.in_features, feature_dim)
    
    def forward(self, x):
        features = self.model(x)
        return self.proj(features)


class UNetBackbone(nn.Module):
    """UNet-based feature extractor for quality assessment."""
    def __init__(self, in_channels=3, feature_dim=768):
        super().__init__()
        
        # Encoder blocks
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1024, feature_dim)
        
    def _conv_block(self, in_channels, out_channels):
        """Convolution block with BN and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(nn.functional.max_pool2d(e4, 2))
        
        # Global pooling
        features = self.global_pool(b).flatten(1)
        return self.proj(features)


def build_backbone(backbone_name: str, pretrained: bool = True, feature_dim: int = 768):
    """Factory function to build backbone."""
    if 'resnet' in backbone_name or 'efficient' in backbone_name:
        return CNNBackbone(backbone_name, pretrained, feature_dim)
    elif 'vit' in backbone_name:
        return ViTBackbone(backbone_name, pretrained, feature_dim)
    elif 'swin' in backbone_name:
        return SwinBackbone(backbone_name, pretrained, feature_dim)
    elif 'unet' in backbone_name:
        return UNetBackbone(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
