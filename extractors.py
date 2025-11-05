"""Feature extraction modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureExtractor(nn.Module):
    """Extract multi-scale features from different resolutions."""
    def __init__(self, base_model, scales=[1.0, 0.75, 0.5]):
        super().__init__()
        self.base_model = base_model
        self.scales = scales
        
    def forward(self, x):
        features = []
        for scale in self.scales:
            if scale != 1.0:
                size = int(x.shape[-1] * scale)
                x_scaled = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            feat = self.base_model(x_scaled)
            features.append(feat)
        return torch.cat(features, dim=-1)


class FrequencyFeatureExtractor(nn.Module):
    """Extract frequency domain features using DCT."""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(64 * 64, feature_dim)
        
    def dct2d(self, x):
        """2D Discrete Cosine Transform."""
        X1 = torch.fft.fft(x, dim=-1)
        X2 = torch.fft.fft(X1, dim=-2)
        return X2.real
    
    def forward(self, x):
        # Convert to grayscale
        if x.shape[1] == 3:
            x_gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        else:
            x_gray = x[:, 0]
        
        # Apply DCT
        dct = self.dct2d(x_gray)
        dct_coeffs = dct[:, :64, :64].flatten(1)
        features = self.fc(dct_coeffs)
        return features


class PatchWiseQualityExtractor(nn.Module):
    """Extract patch-wise quality features."""
    def __init__(self, backbone, patch_size=32, n_patches=9):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.n_patches = n_patches
        
    def extract_patches(self, x):
        """Extract overlapping patches."""
        b, c, h, w = x.shape
        patches = []
        
        # Extract patches in a grid
        stride = (h - self.patch_size) // (int(self.n_patches**0.5) - 1)
        
        for i in range(int(self.n_patches**0.5)):
            for j in range(int(self.n_patches**0.5)):
                h_start = i * stride
                w_start = j * stride
                patch = x[:, :, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
                patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # [B, N_patches, C, H_patch, W_patch]
        return patches
    
    def forward(self, x):
        patches = self.extract_patches(x)
        b, n, c, h, w = patches.shape
        
        # Reshape for backbone
        patches_flat = patches.reshape(b * n, c, h, w)
        features = self.backbone(patches_flat)  # [B*N, feature_dim]
        
        features = features.reshape(b, n, -1)  # [B, N, feature_dim]
        return features.mean(dim=1)  # Average across patches
