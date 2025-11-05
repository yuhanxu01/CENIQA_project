"""Complete CENIQA model."""
import torch
import torch.nn as nn
from backbones import build_backbone
from extractors import MultiScaleFeatureExtractor, FrequencyFeatureExtractor, PatchWiseQualityExtractor
from gmm_module import DifferentiableGMM
from regressors import build_regressor
from config import ModelConfig


class CENIQA(nn.Module):
    """Complete CENIQA model with all components."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Build backbone
        backbone = build_backbone(config.backbone, config.pretrained, config.feature_dim)
        
        # Multi-scale wrapper
        if config.use_multi_scale:
            backbone = MultiScaleFeatureExtractor(backbone)
            backbone_dim = config.feature_dim * 3
        else:
            backbone_dim = config.feature_dim
        
        self.backbone = backbone
        
        # Frequency features
        freq_dim = 0
        if config.use_frequency_features:
            self.freq_extractor = FrequencyFeatureExtractor(256)
            freq_dim = 256
        else:
            self.freq_extractor = None
        
        # Feature projection
        total_feature_dim = backbone_dim + freq_dim
        self.feature_proj = nn.Linear(total_feature_dim, config.hidden_dim)
        
        # GMM clustering
        self.gmm = DifferentiableGMM(
            config.n_clusters,
            config.hidden_dim,
            config.gmm_covariance_type
        )
        
        # Regression head
        regressor_input_dim = config.hidden_dim + config.n_clusters
        self.regressor = build_regressor(
            config.regressor_type,
            regressor_input_dim,
            config.hidden_dim,
            config.dropout_rate
        )
        
        # Auxiliary heads
        self.distortion_classifier = nn.Linear(config.hidden_dim, 10)
        self.ranking_head = nn.Linear(config.hidden_dim, 1)
        
    def extract_features(self, x):
        """Extract and project features."""
        features = [self.backbone(x)]
        
        if self.freq_extractor is not None:
            freq_features = self.freq_extractor(x)
            features.append(freq_features)
        
        features = torch.cat(features, dim=-1)
        features = self.feature_proj(features)
        return features
    
    def forward(self, x, return_all=False):
        """
        Forward pass.
        Args:
            x: Input images [B, C, H, W]
            return_all: Return all intermediate outputs
        Returns:
            quality_score or dict with all outputs
        """
        # Extract features
        features = self.extract_features(x)
        
        # Get GMM posteriors
        posteriors = self.gmm(features)
        
        # Concatenate with posteriors
        combined = torch.cat([features, posteriors], dim=-1)
        
        # Predict quality
        quality_score = self.regressor(combined).squeeze(-1)
        
        if return_all:
            return {
                'quality_score': quality_score,
                'features': features,
                'posteriors': posteriors,
                'distortion_logits': self.distortion_classifier(features),
                'ranking_score': self.ranking_head(features).squeeze(-1)
            }
        
        return quality_score
    
    def get_cluster_assignments(self, x):
        """Get hard cluster assignments."""
        features = self.extract_features(x)
        return self.gmm.get_cluster_assignments(features)
