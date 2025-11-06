"""
Configuration file generator for different experiments.
Creates JSON config files for various model architectures and settings.
"""
import json
import os
from pathlib import Path


def create_base_config():
    """Base configuration with common settings."""
    return {
        "experiment_name": "base_experiment",
        "device": "cuda",  # Use GPU
        "seed": 42,

        # Data settings
        "data": {
            "dataset": "cifar10",
            "train_samples": 2000,  # Increased for GPU
            "val_samples": 500,
            "test_samples": 500,
            "batch_size": 128,  # Larger batch for GPU
            "num_workers": 2  # Safe for most systems
        },

        # Model architecture
        "model": {
            "backbone": "resnet18",
            "feature_dim": 512,
            "n_clusters": 8,  # More clusters
            "hidden_dim": 512,  # Larger hidden dim
            "dropout": 0.3,
            "freeze_backbone": False
        },

        # Training settings
        "training": {
            "epochs": 50,  # More epochs for GPU
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "scheduler_params": {
                "mode": "max",
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1e-6
            }
        },

        # Loss weights
        "loss_weights": {
            "quality_loss": 1.0,
            "cluster_loss": 0.1
        },

        # Logging and checkpointing
        "logging": {
            "save_dir": "./experiments",
            "log_interval": 10,
            "save_best": True,
            "save_last": True
        },

        # Visualization
        "visualization": {
            "enabled": True,
            "plot_clusters": True,
            "plot_predictions": True,
            "save_plots": True
        }
    }


def create_resnet18_config():
    """ResNet18 + GMM + MLP configuration."""
    config = create_base_config()
    config["experiment_name"] = "resnet18_gmm_mlp"
    config["model"]["backbone"] = "resnet18"
    config["model"]["feature_dim"] = 512
    config["model"]["n_clusters"] = 8
    return config


def create_resnet50_config():
    """ResNet50 + GMM + MLP configuration (larger model)."""
    config = create_base_config()
    config["experiment_name"] = "resnet50_gmm_mlp"
    config["model"]["backbone"] = "resnet50"
    config["model"]["feature_dim"] = 1024
    config["model"]["n_clusters"] = 12
    config["model"]["hidden_dim"] = 1024
    config["training"]["learning_rate"] = 5e-4  # Lower LR for larger model
    return config


def create_efficientnet_config():
    """EfficientNet + GMM + MLP configuration."""
    config = create_base_config()
    config["experiment_name"] = "efficientnet_gmm_mlp"
    config["model"]["backbone"] = "efficientnet_b0"
    config["model"]["feature_dim"] = 768
    config["model"]["n_clusters"] = 10
    config["training"]["batch_size"] = 96  # Slightly smaller for EfficientNet
    return config


def create_vit_config():
    """Vision Transformer + GMM + MLP configuration."""
    config = create_base_config()
    config["experiment_name"] = "vit_gmm_mlp"
    config["model"]["backbone"] = "vit_small_patch16_224"
    config["model"]["feature_dim"] = 768
    config["model"]["n_clusters"] = 12
    config["data"]["batch_size"] = 64  # ViT needs more memory
    config["training"]["learning_rate"] = 5e-4
    return config


def create_small_clusters_config():
    """ResNet18 with fewer clusters (ablation study)."""
    config = create_base_config()
    config["experiment_name"] = "resnet18_4clusters"
    config["model"]["n_clusters"] = 4
    return config


def create_large_clusters_config():
    """ResNet18 with many clusters (ablation study)."""
    config = create_base_config()
    config["experiment_name"] = "resnet18_16clusters"
    config["model"]["n_clusters"] = 16
    config["model"]["hidden_dim"] = 768
    return config


def create_frozen_backbone_config():
    """ResNet18 with frozen backbone (faster training)."""
    config = create_base_config()
    config["experiment_name"] = "resnet18_frozen"
    config["model"]["freeze_backbone"] = True
    config["training"]["learning_rate"] = 5e-3  # Higher LR for frozen backbone
    config["training"]["epochs"] = 30
    return config


def create_quick_test_config():
    """Quick test configuration (small dataset, few epochs)."""
    config = create_base_config()
    config["experiment_name"] = "quick_test"
    config["data"]["train_samples"] = 500
    config["data"]["val_samples"] = 100
    config["data"]["test_samples"] = 100
    config["training"]["epochs"] = 10
    return config


def save_config(config, filename):
    """Save configuration to JSON file."""
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    filepath = configs_dir / filename
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Saved config: {filepath}")
    return filepath


def generate_all_configs():
    """Generate all experiment configurations."""
    configs = {
        "resnet18_gmm_mlp.json": create_resnet18_config(),
        "resnet50_gmm_mlp.json": create_resnet50_config(),
        "efficientnet_gmm_mlp.json": create_efficientnet_config(),
        "vit_gmm_mlp.json": create_vit_config(),
        "resnet18_4clusters.json": create_small_clusters_config(),
        "resnet18_16clusters.json": create_large_clusters_config(),
        "resnet18_frozen.json": create_frozen_backbone_config(),
        "quick_test.json": create_quick_test_config(),
    }

    print("="*60)
    print("Generating Experiment Configurations")
    print("="*60)

    for filename, config in configs.items():
        save_config(config, filename)

    print(f"\n{len(configs)} configuration files created in 'configs/' directory")
    print("\nTo run an experiment:")
    print("  python train_gpu.py --config configs/resnet18_gmm_mlp.json")


if __name__ == "__main__":
    generate_all_configs()
