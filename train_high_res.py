"""
Training script using HIGH-RESOLUTION distorted images for realistic IQA.
This version uses STL-10 (96x96) or ImageNet instead of CIFAR-10 (32x32).

Key improvements:
- Uses HighResDistortedDataset with clearer images
- Supports distortion strength levels (light/medium/heavy)
- Same model architecture and training as before
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

from high_res_distorted_dataset import HighResDistortedDataset
from scipy.stats import spearmanr, pearsonr
from sklearn.mixture import GaussianMixture


class SimpleCNNGMMMLPModel(nn.Module):
    """
    Simple CNN + GMM + MLP model for image quality assessment.
    """
    def __init__(self, backbone_name='resnet18', n_clusters=5, feature_dim=512,
                 hidden_dim=512, dropout=0.3, freeze_backbone=False):
        super().__init__()

        # Backbone
        if backbone_name == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            backbone.fc = nn.Identity()
        elif backbone_name == 'resnet34':
            from torchvision.models import resnet34, ResNet34_Weights
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            backbone.fc = nn.Identity()
        elif backbone_name == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.n_clusters = n_clusters

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # GMM (sklearn, not differentiable)
        self.gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        self.gmm_fitted = False

        # MLP regressor
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim + n_clusters, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Get GMM cluster probabilities
        if self.gmm_fitted:
            cluster_probs = torch.from_numpy(
                self.gmm.predict_proba(features.detach().cpu().numpy())
            ).float().to(x.device)
        else:
            # If GMM not fitted yet, use zeros
            cluster_probs = torch.zeros(x.size(0), self.n_clusters).to(x.device)

        # Concatenate features and cluster probabilities
        combined = torch.cat([features, cluster_probs], dim=1)

        # Predict quality
        quality = self.regressor(combined)

        return quality.squeeze()


def validate(model, dataloader, criterion, device):
    """
    Validate the model and compute metrics.
    """
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, scores in dataloader:
            images = images.to(device)
            scores = scores.to(device)

            preds = model(images)
            loss = criterion(preds, scores)

            total_loss += loss.item()
            predictions.extend(preds.cpu().numpy())
            targets.extend(scores.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    srcc, _ = spearmanr(predictions, targets)
    plcc, _ = pearsonr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return {
        'loss': total_loss / len(dataloader),
        'srcc': srcc,
        'plcc': plcc,
        'rmse': rmse
    }


def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False, best_path=None):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    torch.save(checkpoint, save_path)

    if is_best and best_path is not None:
        torch.save(checkpoint, best_path)


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
        model.gmm.fit(all_features)
        model.gmm_fitted = True
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
        cluster_distribution = torch.mean(posteriors, dim=0)  # [n_clusters]
        n_clusters = posteriors.shape[1]
        uniform_distribution = torch.ones(n_clusters, device=device) / n_clusters

        # 使用KL散度惩罚偏离均匀分布
        balance_loss = torch.sum(
            cluster_distribution * torch.log(
                (cluster_distribution + 1e-10) / (uniform_distribution + 1e-10)
            )
        )

        # 4. Entropy regularization (防止过度自信的预测)
        entropy_loss = -torch.mean(torch.sum(posteriors * torch.log(posteriors + 1e-10), dim=1))

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

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'quality_loss': total_quality_loss / n,
        'cluster_loss': total_cluster_loss / n,
        'balance_loss': total_balance_loss / n,
        'entropy_loss': total_entropy_loss / n
    }


def main():
    parser = argparse.ArgumentParser(description='Train IQA model with HIGH-RES distorted images')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='stl10',
                       choices=['stl10', 'imagenet-1k'],
                       help='Dataset to use (stl10=96x96, imagenet-1k=high-res)')
    parser.add_argument('--distortion_strength', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'],
                       help='Distortion strength: light (0.1-0.4), medium (0.2-0.6), heavy (0.3-1.0)')
    parser.add_argument('--train_samples', type=int, default=None,
                       help='Max training samples (None = use all)')
    parser.add_argument('--val_samples', type=int, default=None,
                       help='Max validation samples (None = use all)')
    parser.add_argument('--distortions_per_image', type=int, default=5,
                       help='Number of distorted versions per reference image')

    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='CNN backbone architecture')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of GMM clusters')
    parser.add_argument('--feature_dim', type=int, default=512,
                       help='Feature dimension from backbone')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for MLP')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    # Loss weights (improved defaults)
    parser.add_argument('--cluster_loss_weight', type=float, default=0.1,
                       help='Weight for cluster loss (reduced from 0.5)')
    parser.add_argument('--balance_weight', type=float, default=1.0,
                       help='Weight for uniform cluster balance loss')
    parser.add_argument('--entropy_weight', type=float, default=0.1,
                       help='Weight for entropy regularization')

    # GMM refitting (disabled by default)
    parser.add_argument('--refit_interval', type=int, default=0,
                       help='Epochs between GMM refitting (0 = disabled)')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='high_res_gmm',
                       help='Experiment name for saving results')

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
    model = SimpleCNNGMMMLPModel(
        backbone_name=args.backbone,
        n_clusters=args.n_clusters,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.3,
        freeze_backbone=False
    ).to(device)

    print(f"Model created: {args.backbone} + GMM({args.n_clusters} clusters) + MLP")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Initialize GMM
    print("Initializing GMM with training data...")
    refit_gmm(model, train_loader, device)
    print()

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    exp_dir = Path('experiments') / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = vars(args)
    config['device'] = str(device)
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    best_srcc = -1
    history = {'train': [], 'val': []}

    print("="*70)
    print("Starting training...")
    print("="*70 + "\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            cluster_loss_weight=args.cluster_loss_weight,
            balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        # Print detailed metrics
        print(f"\nEpoch {epoch+1} Results (time: {epoch_time:.1f}s):")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Q_loss: {train_metrics['quality_loss']:.4f} | "
              f"C_loss: {train_metrics['cluster_loss']:.4f} | "
              f"B_loss: {train_metrics['balance_loss']:.4f}")
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

        # Refit GMM if enabled
        if args.refit_interval > 0 and (epoch + 1) % args.refit_interval == 0:
            print(f"\n  Re-fitting GMM at epoch {epoch+1}...")
            refit_gmm(model, train_loader, device)

        print()

    print("="*70)
    print("Training complete!")
    print(f"Best validation SRCC: {best_srcc:.4f}")
    print(f"Results saved to: {exp_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
