#!/usr/bin/env python3
"""
Debug script to inspect checkpoint structure and infer model configuration.
Useful for troubleshooting visualization errors.

Usage:
    python debug_checkpoint.py experiments/resnet18_large_10k/best_model.pth
"""
import torch
import argparse
from pathlib import Path


def debug_checkpoint(checkpoint_path):
    """Print detailed information about a checkpoint file."""
    print("="*70)
    print(f"Checkpoint Debug Info: {checkpoint_path}")
    print("="*70)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # Print top-level keys
    print("Top-level keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    print()

    # Check for model state dict
    if 'model_state_dict' not in checkpoint:
        print("⚠️  Warning: 'model_state_dict' not found in checkpoint")
        return

    state_dict = checkpoint['model_state_dict']

    # Group keys by module
    print(f"Model structure ({len(state_dict)} parameters):")
    modules = {}
    for key in state_dict.keys():
        module = key.split('.')[0]
        if module not in modules:
            modules[module] = []
        modules[module].append(key)

    for module, keys in sorted(modules.items()):
        print(f"\n  {module}: ({len(keys)} parameters)")
        # Show first few keys as examples
        for key in sorted(keys)[:5]:
            shape = tuple(state_dict[key].shape)
            print(f"    - {key}: {shape}")
        if len(keys) > 5:
            print(f"    ... and {len(keys) - 5} more")

    print("\n" + "="*70)
    print("Inferred Configuration:")
    print("="*70)

    # Try to infer configuration
    try:
        # Feature dim
        if 'backbone.proj.weight' in state_dict:
            feature_dim = state_dict['backbone.proj.weight'].shape[0]
            print(f"✅ feature_dim: {feature_dim}")
        else:
            print("❌ Could not infer feature_dim")

        # Number of clusters
        if 'gmm.means' in state_dict:
            n_clusters = state_dict['gmm.means'].shape[0]
            print(f"✅ n_clusters: {n_clusters}")
        else:
            print("❌ Could not infer n_clusters")

        # Hidden dim
        hidden_dim = None
        if 'regressor.fc1.weight' in state_dict:
            hidden_dim = state_dict['regressor.fc1.weight'].shape[0]
            print(f"✅ hidden_dim: {hidden_dim} (from regressor.fc1)")
        elif 'regressor.mlp.0.weight' in state_dict:
            hidden_dim = state_dict['regressor.mlp.0.weight'].shape[0]
            print(f"✅ hidden_dim: {hidden_dim} (from regressor.mlp.0)")
        else:
            regressor_keys = [k for k in state_dict.keys() if k.startswith('regressor.') and 'weight' in k]
            if regressor_keys:
                first_key = sorted(regressor_keys)[0]
                hidden_dim = state_dict[first_key].shape[0]
                print(f"⚠️  hidden_dim: {hidden_dim} (inferred from {first_key})")
            else:
                print("❌ Could not infer hidden_dim")

        # Backbone type
        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.model.')]
        if backbone_keys:
            if any('resnet' in k.lower() for k in str(backbone_keys)):
                print(f"✅ backbone: resnet18 (inferred)")
            else:
                print(f"⚠️  backbone: unknown (showing keys sample)")
                for key in sorted(backbone_keys)[:3]:
                    print(f"     - {key}")
        else:
            print("❌ Could not identify backbone")

    except Exception as e:
        print(f"❌ Error during inference: {e}")

    # Print metrics if available
    if 'metrics' in checkpoint:
        print("\n" + "="*70)
        print("Checkpoint Metrics:")
        print("="*70)
        metrics = checkpoint['metrics']
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Print epoch if available
    if 'epoch' in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch'] + 1}")

    print("\n" + "="*70)
    print("✅ Debug complete!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Debug checkpoint structure')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return

    debug_checkpoint(checkpoint_path)


if __name__ == '__main__':
    main()
