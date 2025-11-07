"""
Training script for simple CNN baseline (WITHOUT GMM)
Direct regression: CNN → MLP → Quality Score
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from scipy.stats import spearmanr, pearsonr

from simple_model import SimpleCNNModel
from distorted_dataset import DistortedImageDataset


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        if len(batch) == 3:
            images, scores, _ = batch
        else:
            images, scores = batch

        images = images.to(device)
        scores = scores.to(device)

        # Forward pass
        predictions = model(images)

        # Simple MSE loss (no clustering loss)
        loss = criterion(predictions, scores)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            if len(batch) == 3:
                images, scores, _ = batch
            else:
                images, scores = batch

            images = images.to(device)
            scores = scores.to(device)

            predictions = model(images)
            loss = criterion(predictions, scores)

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    srcc, _ = spearmanr(all_preds, all_targets)
    plcc, _ = pearsonr(all_preds, all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    return {
        'loss': total_loss / len(dataloader),
        'srcc': srcc,
        'plcc': plcc,
        'rmse': rmse
    }


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train simple CNN baseline (no GMM)')
    parser.add_argument('--experiment_name', type=str, default='resnet18_simple_baseline',
                       help='Experiment name')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                       help='Backbone architecture')
    parser.add_argument('--train_samples', type=int, default=1666,
                       help='Number of reference images for training')
    parser.add_argument('--val_samples', type=int, default=166,
                       help='Number of reference images for validation')
    parser.add_argument('--distortions_per_image', type=int, default=5,
                       help='Number of distorted versions per reference image')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--feature_dim', type=int, default=512,
                       help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()

    print("="*60)
    print(f"Training Simple Baseline (NO GMM)")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Architecture: CNN → MLP → Quality Score")
    print(f"Train samples: {args.train_samples} x {args.distortions_per_image + 1} = "
          f"{args.train_samples * (args.distortions_per_image + 1)}")
    print(f"Val samples: {args.val_samples} x {args.distortions_per_image + 1} = "
          f"{args.val_samples * (args.distortions_per_image + 1)}")
    print("="*60)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create experiment directory
    exp_dir = Path('experiments') / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['model_type'] = 'simple_cnn_baseline'
    config['has_gmm'] = False
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Create datasets
    print("\nCreating distorted datasets...")
    train_dataset = DistortedImageDataset(
        split='train',
        max_samples=args.train_samples,
        distortions_per_image=args.distortions_per_image,
        include_pristine=True
    )
    val_dataset = DistortedImageDataset(
        split='test',
        max_samples=args.val_samples,
        distortions_per_image=args.distortions_per_image,
        include_pristine=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Create model
    print("\nInitializing simple baseline model (no GMM)...")
    model = SimpleCNNModel(
        backbone_name=args.backbone,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=False
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    best_srcc = -1
    best_epoch = 0
    training_history = []

    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_metrics['srcc'])
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"SRCC: {val_metrics['srcc']:.4f}")
        print(f"PLCC: {val_metrics['plcc']:.4f}")
        print(f"RMSE: {val_metrics['rmse']:.4f}")
        print(f"LR: {current_lr:.6f}")

        # Save history
        history_entry = {
            'epoch': int(epoch + 1),
            'train_loss': float(train_loss),
            'val_loss': float(val_metrics['loss']),
            'srcc': float(val_metrics['srcc']),
            'plcc': float(val_metrics['plcc']),
            'rmse': float(val_metrics['rmse']),
            'lr': float(current_lr)
        }
        training_history.append(history_entry)

        # Save best model
        if val_metrics['srcc'] > best_srcc:
            best_srcc = val_metrics['srcc']
            best_epoch = epoch + 1
            save_checkpoint(
                model, optimizer, epoch,
                val_metrics,
                exp_dir / 'best_model.pth'
            )
            print(f"[BEST] Model saved (SRCC: {best_srcc:.4f})")

        # Save last model
        save_checkpoint(
            model, optimizer, epoch,
            val_metrics,
            exp_dir / 'last_model.pth'
        )

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best SRCC: {best_srcc:.4f} (Epoch {best_epoch})")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print("="*60)

    # Save training history
    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=4)

    print(f"\nResults saved to: {exp_dir}")


if __name__ == '__main__':
    main()
