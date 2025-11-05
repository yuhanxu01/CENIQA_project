"""Configuration management for CENIQA model."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Model and training configuration."""
    # Backbone settings
    backbone: str = 'vit_small_patch16_224'
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Feature extraction
    feature_dim: int = 768
    hidden_dim: int = 512
    use_multi_scale: bool = True
    use_frequency_features: bool = True
    
    # GMM settings
    n_clusters: int = 8
    gmm_covariance_type: str = 'diag'
    gmm_init_method: str = 'kmeans++'
    gmm_update_freq: int = 10
    use_bic_selection: bool = True
    min_clusters: int = 4
    max_clusters: int = 12
    
    # Regression head
    regressor_type: str = 'mlp'  # mlp, kan, transformer, gru, attention
    use_monotonic: bool = True
    dropout_rate: float = 0.2
    use_uncertainty: bool = True
    
    # Training settings
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    
    # Loss weights
    lambda_contrast: float = 1.0
    lambda_consistency: float = 0.5
    lambda_ranking: float = 0.5
    lambda_cluster: float = 0.3
    lambda_quality: float = 1.0
    lambda_monotonic: float = 0.1
    
    # Data settings
    image_size: int = 384
    num_crops: int = 10
    augmentation_strength: float = 0.5
    
    # Experiment settings
    experiment_name: str = 'ceniqa_baseline'
    seed: int = 42
    use_wandb: bool = True
    save_dir: str = './experiments'
    checkpoint_freq: int = 5


def load_config(config_path: str) -> ModelConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return ModelConfig(**config_dict)


def save_config(config: ModelConfig, save_path: str):
    """Save configuration to JSON file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
