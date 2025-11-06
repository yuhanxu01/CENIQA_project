"""
GPU-optimized training script for CNN+GMM+MLP experiments.
Supports configuration files and extensive logging.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

from backbones import CNNBackbone
from gmm_module import DifferentiableGMM
from regressors import MonotonicMLP


class SimpleCNNGMMMLPModel(nn.Module):
    """CNN + GMM + MLP model."""
    def __init__(self, backbone_name='resnet18', feature_dim=512,
                 n_clusters=8, hidden_dim=512, dropout=0.3,
                 freeze_backbone=False):
        super().__init__()

        self.backbone = CNNBackbone(
            model_name=backbone_name,
            pretrained=True,
            feature_dim=feature_dim
        )

        if freeze_backbone:
            for param in self.backbone.model.parameters():
                param.requires_grad = False
            for param in self.backbone.proj.parameters():
                param.requires_grad = True

        self.gmm = DifferentiableGMM(
            n_clusters=n_clusters,
            feature_dim=feature_dim,
            covariance_type='diag'
        )

        regressor_input_dim = feature_dim + n_clusters
        self.regressor = MonotonicMLP(
            input_dim=regressor_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x, return_all=False):
        features = self.backbone(x)
        posteriors = self.gmm(features)
        combined = torch.cat([features, posteriors], dim=-1)
        quality_score = self.regressor(combined).squeeze(-1)

        if return_all:
            return {
                'quality_score': quality_score,
                'features': features,
                'posteriors': posteriors
            }
        return quality_score


class HuggingFaceImageDataset(Dataset):
    """Dataset loader for HuggingFace datasets."""
    def __init__(self, split='train', max_samples=None):
        try:
            from datasets import load_dataset
            from torchvision import transforms
            from PIL import Image

            print(f"Loading dataset (split={split}, max_samples={max_samples})...")

            dataset = load_dataset("cifar10", split=split)

            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            self.images = []
            self.scores = []

            print(f"Processing {len(dataset)} images...")
            for item in tqdm(dataset, desc=f"Loading {split}"):
                img = item['img']
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_array = np.array(img).astype(np.float32)

                # Enhanced quality indicators
                brightness = np.mean(img_array)
                contrast = np.std(img_array)
                grad_x = np.abs(np.diff(img_array, axis=1)).mean()
                grad_y = np.abs(np.diff(img_array, axis=0)).mean()
                sharpness = (grad_x + grad_y) / 2
                color_variety = np.mean([np.std(img_array[:, :, i]) for i in range(3)])

                synthetic_score = (
                    (brightness / 255 * 25) +
                    (min(contrast / 70, 1) * 25) +
                    (min(sharpness / 30, 1) * 30) +
                    (min(color_variety / 70, 1) * 20)
                )
                synthetic_score = min(100, max(0, synthetic_score))

                self.images.append(img)
                self.scores.append(synthetic_score)

            print(f"Dataset loaded: {len(self.images)} images")
            print(f"Score range: {min(self.scores):.2f} - {max(self.scores):.2f}")

        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_tensor = self.transform(self.images[idx])
        score_normalized = self.scores[idx] / 100.0
        return img_tensor, torch.tensor(score_normalized, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, criterion, device, cluster_loss_weight=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_quality_loss = 0
    total_cluster_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, scores in pbar:
        images = images.to(device)
        scores = scores.to(device)

        outputs = model(images, return_all=True)
        predictions = outputs['quality_score']
        posteriors = outputs['posteriors']

        # Quality loss
        quality_loss = criterion(predictions, scores)

        # Cluster separation loss (optional)
        cluster_loss = -torch.mean(torch.max(posteriors, dim=1)[0])

        # Combined loss
        loss = quality_loss + cluster_loss_weight * cluster_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_quality_loss += quality_loss.item()
        total_cluster_loss += cluster_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'q_loss': f'{quality_loss.item():.4f}'
        })

    return {
        'total_loss': total_loss / len(dataloader),
        'quality_loss': total_quality_loss / len(dataloader),
        'cluster_loss': total_cluster_loss / len(dataloader)
    }


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    from scipy.stats import spearmanr, pearsonr

    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_posteriors = []

    with torch.no_grad():
        for images, scores in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            scores = scores.to(device)

            outputs = model(images, return_all=True)
            predictions = outputs['quality_score']
            posteriors = outputs['posteriors']

            loss = criterion(predictions, scores)
            total_loss += loss.item()

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())
            all_posteriors.append(posteriors.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_posteriors = np.concatenate(all_posteriors, axis=0)

    srcc, _ = spearmanr(all_preds, all_targets)
    plcc, _ = pearsonr(all_preds, all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    return {
        'loss': total_loss / len(dataloader),
        'srcc': srcc,
        'plcc': plcc,
        'rmse': rmse,
        'predictions': all_preds,
        'targets': all_targets,
        'posteriors': all_posteriors
    }


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train CNN+GMM+MLP model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print("="*60)
    print(f"Experiment: {config['experiment_name']}")
    print("="*60)
    print(json.dumps(config, indent=2))
    print("="*60)

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create experiment directory
    exp_dir = Path(config['logging']['save_dir']) / config['experiment_name']
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config to experiment directory
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = HuggingFaceImageDataset(
        split='train',
        max_samples=config['data']['train_samples']
    )
    val_dataset = HuggingFaceImageDataset(
        split='test',
        max_samples=config['data']['val_samples']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # Create model
    print("\nInitializing model...")
    model = SimpleCNNGMMMLPModel(
        backbone_name=config['model']['backbone'],
        feature_dim=config['model']['feature_dim'],
        n_clusters=config['model']['n_clusters'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {config['model']['backbone']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize GMM
    print("\nInitializing GMM...")
    model.eval()
    all_features = []
    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc='Collecting features'):
            images = images.to(device)
            features = model.backbone(images)
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    model.gmm.fit_sklearn(all_features)
    print(f"GMM initialized with {len(all_features)} samples")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    if config['training']['scheduler'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config['training']['scheduler_params']
        )
    else:
        scheduler = None

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    best_srcc = -1
    best_epoch = 0
    training_history = []

    start_time = time.time()

    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            cluster_loss_weight=config['loss_weights']['cluster_loss']
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        if scheduler:
            scheduler.step(val_metrics['srcc'])

        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"SRCC: {val_metrics['srcc']:.4f}")
        print(f"PLCC: {val_metrics['plcc']:.4f}")
        print(f"RMSE: {val_metrics['rmse']:.4f}")
        print(f"LR: {current_lr:.6f}")

        # Save history (convert numpy types to Python types for JSON)
        history_entry = {
            'epoch': int(epoch + 1),
            'train_loss': float(train_metrics['total_loss']),
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
            if config['logging']['save_best']:
                save_checkpoint(
                    model, optimizer, epoch,
                    val_metrics,
                    exp_dir / 'best_model.pth'
                )
                print(f"[BEST] Model saved (SRCC: {best_srcc:.4f})")

        # Save last model
        if config['logging']['save_last']:
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
