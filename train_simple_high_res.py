"""
Training script for SIMPLE BASELINE (no GMM) with HIGH-RESOLUTION images.
For comparison with GMM-based model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from high_res_distorted_dataset import HighResDistortedDataset
from simple_model import SimpleCNNModel
from train_gpu import validate, save_checkpoint


def train_epoch_simple(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch (simple model, no GMM losses).
    """
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, scores in pbar:
        images = images.to(device)
        scores = scores.to(device)

        predictions = model(images)
        loss = criterion(predictions, scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return {'loss': total_loss / len(dataloader)}


def main():
    parser = argparse.ArgumentParser(description='Train SIMPLE baseline with HIGH-RES images')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='stl10',
                       choices=['stl10', 'imagenet-1k'],
                       help='Dataset to use')
    parser.add_argument('--distortion_strength', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'],
                       help='Distortion strength')
    parser.add_argument('--train_samples', type=int, default=None,
                       help='Max training samples')
    parser.add_argument('--val_samples', type=int, default=None,
                       help='Max validation samples')
    parser.add_argument('--distortions_per_image', type=int, default=5,
                       help='Distortions per image')

    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='CNN backbone')
    parser.add_argument('--feature_dim', type=int, default=512,
                       help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='high_res_simple',
                       help='Experiment name')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset} (distortion strength: {args.distortion_strength})")
    print("\n" + "="*70)

    # Create datasets
    print("Creating HIGH-RESOLUTION training dataset...")
    train_dataset = HighResDistortedDataset(
        dataset_name=args.dataset,
        split='train',
        max_samples=args.train_samples,
        distortions_per_image=args.distortions_per_image,
        include_pristine=True,
        distortion_strength=args.distortion_strength
    )

    print("\nCreating HIGH-RESOLUTION validation dataset...")
    val_dataset = HighResDistortedDataset(
        dataset_name=args.dataset,
        split='test',
        max_samples=args.val_samples,
        distortions_per_image=args.distortions_per_image,
        include_pristine=True,
        distortion_strength=args.distortion_strength
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print("="*70 + "\n")

    # Create model
    model = SimpleCNNModel(
        backbone_name=args.backbone,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.3,
        freeze_backbone=False
    ).to(device)

    print(f"Model created: {args.backbone} + MLP (NO GMM)")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    exp_dir = Path('experiments') / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    best_srcc = -1
    history = {'train': [], 'val': []}

    print("="*70)
    print("Starting training (SIMPLE baseline, no GMM)...")
    print("="*70 + "\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        start_time = time.time()

        # Train
        train_metrics = train_epoch_simple(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        # Print metrics
        print(f"\nEpoch {epoch+1} Results (time: {epoch_time:.1f}s):")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"SRCC: {val_metrics['srcc']:.4f} | "
              f"PLCC: {val_metrics['plcc']:.4f} | "
              f"RMSE: {val_metrics['rmse']:.4f}")

        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save checkpoint
        is_best = val_metrics['srcc'] > best_srcc
        if is_best:
            best_srcc = val_metrics['srcc']
            print(f"  ★ New best SRCC: {best_srcc:.4f}")

        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            exp_dir / 'checkpoint_latest.pth',
            is_best=is_best,
            best_path=exp_dir / 'checkpoint_best.pth'
        )

        # Save history
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print()

    print("="*70)
    print("Training complete!")
    print(f"Best validation SRCC: {best_srcc:.4f}")
    print(f"Results saved to: {exp_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
