"""
Training script using distorted images for realistic IQA.
This version uses the DistortedImageDataset with multiple distortion types.
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

from distorted_dataset import DistortedImageDataset
from train_gpu import SimpleCNNGMMMLPModel, validate, save_checkpoint


def refit_gmm(model, dataloader, device):
    """
    Re-fit GMM with current features from the backbone.

    Args:
        model: The CNN+GMM+MLP model
        dataloader: DataLoader to collect features from
        device: Device to use (cuda or cpu)
    """
    model.eval()
    all_features = []
    all_scores = []

    with torch.no_grad():
        for images, scores in tqdm(dataloader, desc='Re-fitting GMM'):
            images = images.to(device)
            features = model.backbone(images)
            all_features.append(features.cpu().numpy())
            all_scores.append(scores.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    # Re-fit GMM with sklearn
    try:
        model.gmm.fit_sklearn(all_features)
        print(f"GMM re-fitted with {len(all_features)} samples")

        # Analyze cluster distribution
        model.eval()
        with torch.no_grad():
            features_tensor = torch.from_numpy(all_features).float().to(device)
            posteriors = model.gmm(features_tensor)
            cluster_assignments = torch.argmax(posteriors, dim=1).cpu().numpy()

        print("\nCluster distribution after re-fitting:")
        for k in range(model.gmm.n_clusters):
            mask = cluster_assignments == k
            count = mask.sum()
            avg_quality = all_scores[mask].mean() if count > 0 else 0
            print(f"  Cluster {k}: {count:4d} samples (avg quality: {avg_quality:.3f})")

    except Exception as e:
        print(f"Warning: GMM fitting failed: {e}")
        print("Skipping GMM re-fitting for this epoch.")


def train_epoch(model, dataloader, optimizer, criterion, device,
                cluster_loss_weight=0.5, balance_weight=1.0, entropy_weight=0.1):
    """
    Train for one epoch with improved regularization.

    Args:
        cluster_loss_weight: Weight for cluster separation loss
        balance_weight: Weight for uniform cluster distribution regularization
        entropy_weight: Weight for entropy regularization
    """
    model.train()
    total_loss = 0
    total_quality_loss = 0
    total_cluster_loss = 0
    total_balance_loss = 0
    total_entropy_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, scores in pbar:
        images = images.to(device)
        scores = scores.to(device)

        outputs = model(images, return_all=True)
        predictions = outputs['quality_score']
        posteriors = outputs['posteriors']

        # 1. Quality loss (主要任务)
        quality_loss = criterion(predictions, scores)

        # 2. Cluster separation loss (鼓励明确的聚类分配)
        max_posteriors = torch.max(posteriors, dim=1)[0]
        cluster_loss = -torch.mean(max_posteriors)

        # 3. Uniform distribution regularization (鼓励均匀的聚类分布)
        # 计算batch中每个聚类的平均后验概率
        cluster_distribution = torch.mean(posteriors, dim=0)  # [n_clusters]
        n_clusters = posteriors.shape[1]
        uniform_distribution = torch.ones(n_clusters, device=device) / n_clusters

        # 使用KL散度惩罚偏离均匀分布
        balance_loss = torch.sum(
            cluster_distribution * torch.log(
                (cluster_distribution + 1e-10) / (uniform_distribution + 1e-10)
            )
        )

        # 4. Entropy regularization (防止过度自信)
        entropy = -torch.sum(posteriors * torch.log(posteriors + 1e-10), dim=1).mean()
        entropy_loss = -entropy  # 负号因为我们想要更高的熵

        # Combined loss
        loss = (quality_loss +
                cluster_loss_weight * cluster_loss +
                balance_weight * balance_loss +
                entropy_weight * entropy_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_quality_loss += quality_loss.item()
        total_cluster_loss += cluster_loss.item()
        total_balance_loss += balance_loss.item()
        total_entropy_loss += entropy_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'q_loss': f'{quality_loss.item():.4f}',
            'c_loss': f'{cluster_loss.item():.4f}',
            'b_loss': f'{balance_loss.item():.4f}'
        })

    return {
        'total_loss': total_loss / len(dataloader),
        'quality_loss': total_quality_loss / len(dataloader),
        'cluster_loss': total_cluster_loss / len(dataloader),
        'balance_loss': total_balance_loss / len(dataloader),
        'entropy_loss': total_entropy_loss / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(description='Train with distorted images')
    parser.add_argument('--experiment_name', type=str, default='resnet18_distorted',
                       help='Experiment name')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                       help='Backbone architecture')
    parser.add_argument('--n_clusters', type=int, default=8,
                       help='Number of GMM clusters')
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
    parser.add_argument('--refit_interval', type=int, default=0,
                       help='GMM refit interval (epochs), 0=disabled')
    parser.add_argument('--cluster_loss_weight', type=float, default=0.1,
                       help='Weight for cluster separation loss')
    parser.add_argument('--balance_weight', type=float, default=1.0,
                       help='Weight for uniform cluster distribution regularization')
    parser.add_argument('--entropy_weight', type=float, default=0.1,
                       help='Weight for entropy regularization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()

    print("="*60)
    print(f"Training with Distorted Images (Improved)")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Backbone: {args.backbone}")
    print(f"Clusters: {args.n_clusters}")
    print(f"Train samples: {args.train_samples} x {args.distortions_per_image + 1} = "
          f"{args.train_samples * (args.distortions_per_image + 1)}")
    print(f"Val samples: {args.val_samples} x {args.distortions_per_image + 1} = "
          f"{args.val_samples * (args.distortions_per_image + 1)}")
    print("\nRegularization:")
    print(f"  - Cluster separation weight: {args.cluster_loss_weight}")
    print(f"  - Balance weight: {args.balance_weight}")
    print(f"  - Entropy weight: {args.entropy_weight}")
    print(f"  - GMM refit interval: {args.refit_interval if args.refit_interval > 0 else 'Disabled'}")
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
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Create datasets with distortions
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
    print("\nInitializing model...")
    model = SimpleCNNGMMMLPModel(
        backbone_name=args.backbone,
        feature_dim=512,
        n_clusters=args.n_clusters,
        hidden_dim=512,
        dropout=0.3,
        freeze_backbone=False
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize GMM
    print("\nInitializing GMM...")
    refit_gmm(model, train_loader, device)

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

        # Re-fit GMM periodically (only if enabled)
        if args.refit_interval > 0 and epoch > 0 and epoch % args.refit_interval == 0:
            print(f"\n[Epoch {epoch+1}] Re-fitting GMM...")
            refit_gmm(model, train_loader, device)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            cluster_loss_weight=args.cluster_loss_weight,
            balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_metrics['srcc'])
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        print(f"  - Quality Loss: {train_metrics['quality_loss']:.4f}")
        print(f"  - Cluster Loss: {train_metrics['cluster_loss']:.4f}")
        print(f"  - Balance Loss: {train_metrics['balance_loss']:.4f}")
        print(f"  - Entropy Loss: {train_metrics['entropy_loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"SRCC: {val_metrics['srcc']:.4f}")
        print(f"PLCC: {val_metrics['plcc']:.4f}")
        print(f"RMSE: {val_metrics['rmse']:.4f}")
        print(f"LR: {current_lr:.6f}")

        # Save history
        history_entry = {
            'epoch': int(epoch + 1),
            'train_loss': float(train_metrics['total_loss']),
            'quality_loss': float(train_metrics['quality_loss']),
            'cluster_loss': float(train_metrics['cluster_loss']),
            'balance_loss': float(train_metrics['balance_loss']),
            'entropy_loss': float(train_metrics['entropy_loss']),
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

    # Analyze final cluster distribution
    print("\n" + "="*60)
    print("Final Cluster Analysis")
    print("="*60)
    refit_gmm(model, val_loader, device)


if __name__ == '__main__':
    main()
