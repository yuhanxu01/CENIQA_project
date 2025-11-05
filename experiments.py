"""Experiment configuration and runner."""
import os
import json
from pathlib import Path
from config import ModelConfig


# Experiment configurations
EXPERIMENTS = {
    'baseline_vit': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'mlp',
        'n_clusters': 8,
    },
    'baseline_resnet': {
        'backbone': 'resnet50',
        'regressor_type': 'mlp',
        'n_clusters': 8,
    },
    'baseline_swin': {
        'backbone': 'swin_tiny_patch4_window7_224',
        'regressor_type': 'mlp',
        'n_clusters': 8,
    },
    'baseline_unet': {
        'backbone': 'unet',
        'regressor_type': 'mlp',
        'n_clusters': 8,
    },
    'regressor_kan': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'kan',
        'n_clusters': 8,
    },
    'regressor_transformer': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'transformer',
        'n_clusters': 8,
    },
    'regressor_gru': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'gru',
        'n_clusters': 8,
    },
    'regressor_attention': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'attention',
        'n_clusters': 8,
    },
    'clusters_4': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'mlp',
        'n_clusters': 4,
    },
    'clusters_6': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'mlp',
        'n_clusters': 6,
    },
    'clusters_10': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'mlp',
        'n_clusters': 10,
    },
    'clusters_12': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'mlp',
        'n_clusters': 12,
    },
    'no_multiscale': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'mlp',
        'n_clusters': 8,
        'use_multi_scale': False,
    },
    'no_frequency': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'mlp',
        'n_clusters': 8,
        'use_frequency_features': False,
    },
    'no_monotonic': {
        'backbone': 'vit_small_patch16_224',
        'regressor_type': 'mlp',
        'n_clusters': 8,
        'use_monotonic': False,
    },
}


def create_experiment_config(exp_name, base_config=None):
    """
    Create config for specific experiment.
    Args:
        exp_name: Experiment name
        base_config: Base configuration dict
    Returns:
        ModelConfig instance
    """
    if base_config is None:
        base_config = {}
    
    # Get experiment overrides
    if exp_name in EXPERIMENTS:
        exp_overrides = EXPERIMENTS[exp_name]
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")
    
    # Create config
    config = ModelConfig()
    
    # Apply overrides
    for key, value in base_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    for key, value in exp_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Update experiment name
    config.experiment_name = exp_name
    
    return config


def save_experiment_configs(output_dir='configs'):
    """Save all experiment configs to JSON files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for exp_name in EXPERIMENTS.keys():
        config = create_experiment_config(exp_name)
        
        config_path = Path(output_dir) / f'{exp_name}.json'
        
        config_dict = {
            'backbone': config.backbone,
            'pretrained': config.pretrained,
            'feature_dim': config.feature_dim,
            'hidden_dim': config.hidden_dim,
            'use_multi_scale': config.use_multi_scale,
            'use_frequency_features': config.use_frequency_features,
            'n_clusters': config.n_clusters,
            'regressor_type': config.regressor_type,
            'use_monotonic': config.use_monotonic,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'experiment_name': config.experiment_name,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved {config_path}")


def list_experiments():
    """List all available experiments."""
    print("Available experiments:")
    print("-" * 50)
    for exp_name in sorted(EXPERIMENTS.keys()):
        print(f"  {exp_name}")


if __name__ == '__main__':
    # Save all configs
    save_experiment_configs()
    list_experiments()
