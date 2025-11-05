"""
Demo script: CNN + GMM + MLP pipeline
Uses a small HuggingFace dataset to test the pipeline.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm

# Import project modules
from backbones import CNNBackbone
from gmm_module import DifferentiableGMM
from regressors import MonotonicMLP


class SimpleCNNGMMMLPModel(nn.Module):
    """Simplified CNN + GMM + MLP model for demo."""
    def __init__(self,
                 backbone_name='resnet18',
                 feature_dim=512,
                 n_clusters=4,
                 hidden_dim=256):
        super().__init__()

        # CNN Feature Extractor (using ResNet18 for speed)
        self.backbone = CNNBackbone(
            model_name=backbone_name,
            pretrained=True,
            feature_dim=feature_dim
        )

        # GMM for clustering
        self.gmm = DifferentiableGMM(
            n_clusters=n_clusters,
            feature_dim=feature_dim,
            covariance_type='diag'
        )

        # MLP for regression
        regressor_input_dim = feature_dim + n_clusters
        self.regressor = MonotonicMLP(
            input_dim=regressor_input_dim,
            hidden_dim=hidden_dim,
            dropout=0.2
        )

    def forward(self, x, return_all=False):
        # Extract features with CNN
        features = self.backbone(x)

        # Get GMM cluster posteriors
        posteriors = self.gmm(features)

        # Concatenate features and posteriors
        combined = torch.cat([features, posteriors], dim=-1)

        # Predict quality score with MLP
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
    def __init__(self, split='train', max_samples=200):
        """
        Initialize dataset from HuggingFace.
        We'll use a small subset of CIFAR-10 and create synthetic quality scores
        for demonstration purposes.
        """
        try:
            from datasets import load_dataset
            print(f"Loading HuggingFace dataset (split={split}, max_samples={max_samples})...")

            # Load CIFAR-10 as a demo dataset
            dataset = load_dataset("cifar10", split=split, trust_remote_code=True)

            # Take a subset
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            self.images = []
            self.scores = []

            # Generate synthetic quality scores based on image properties
            print(f"Processing {len(dataset)} images...")
            for idx, item in enumerate(tqdm(dataset)):
                img = item['img']

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Generate synthetic quality score (0-100)
                # Based on image statistics for demo purposes
                img_array = np.array(img)
                brightness = np.mean(img_array)
                variance = np.var(img_array)
                # Normalize to 0-100 range
                synthetic_score = min(100, max(0, (brightness / 255 * 50) + (variance / 10000 * 50)))

                self.images.append(img)
                self.scores.append(synthetic_score)

            print(f"Dataset loaded: {len(self.images)} images")
            print(f"Score range: {min(self.scores):.2f} - {max(self.scores):.2f}")

        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        score = self.scores[idx]

        # Apply transforms
        img_tensor = self.transform(img)

        # Normalize score to 0-1 range
        score_normalized = score / 100.0

        return img_tensor, torch.tensor(score_normalized, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for images, scores in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        scores = scores.to(device)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, scores)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, scores in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            scores = scores.to(device)

            predictions = model(images)
            loss = criterion(predictions, scores)

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

    # Calculate correlation
    from scipy.stats import spearmanr, pearsonr
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    srcc, _ = spearmanr(all_preds, all_targets)
    plcc, _ = pearsonr(all_preds, all_targets)

    return total_loss / len(dataloader), srcc, plcc


def main():
    """Main training loop."""
    print("="*60)
    print("CNN + GMM + MLP Pipeline Demo")
    print("="*60)

    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}\n")

    # Create datasets
    print("Loading datasets...")
    train_dataset = HuggingFaceImageDataset(split='train', max_samples=200)
    val_dataset = HuggingFaceImageDataset(split='test', max_samples=50)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("\nInitializing model...")
    model = SimpleCNNGMMMLPModel(
        backbone_name='resnet18',
        feature_dim=512,
        n_clusters=4,
        hidden_dim=256
    )
    model = model.to(DEVICE)

    # Print model architecture
    print("\nModel Architecture:")
    print(f"- Backbone: ResNet18")
    print(f"- Feature dim: 512")
    print(f"- GMM clusters: 4")
    print(f"- Regressor: MLP (hidden_dim=256)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Initialize GMM with sklearn (optional but recommended)
    print("\nInitializing GMM with training data...")
    model.eval()
    all_features = []
    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc='Collecting features'):
            images = images.to(DEVICE)
            features = model.backbone(images)
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    model.gmm.fit_sklearn(all_features)
    print(f"GMM initialized with {len(all_features)} samples")

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    best_srcc = -1
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_loss, srcc, plcc = validate(model, val_loader, criterion, DEVICE)

        # Step scheduler
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}\n")

        # Save best model
        if srcc > best_srcc:
            best_srcc = srcc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'srcc': srcc,
                'plcc': plcc,
            }, 'best_model_demo.pth')
            print(f"[BEST] Model saved (SRCC: {srcc:.4f})\n")

    print("="*60)
    print("Training Complete!")
    print(f"Best SRCC: {best_srcc:.4f}")
    print("="*60)

    # Demo inference
    print("\n" + "="*60)
    print("Demo Inference on a Few Samples")
    print("="*60 + "\n")

    model.eval()
    with torch.no_grad():
        images, scores = next(iter(val_loader))
        images = images.to(DEVICE)

        outputs = model(images[:5], return_all=True)
        predictions = outputs['quality_score'].cpu().numpy()
        actual = scores[:5].numpy()
        posteriors = outputs['posteriors'].cpu().numpy()

        print("Sample Predictions:")
        for i in range(5):
            print(f"  Sample {i+1}:")
            print(f"    Predicted: {predictions[i]:.4f}")
            print(f"    Actual: {actual[i]:.4f}")
            print(f"    Cluster assignment: {posteriors[i].argmax()}")
            print(f"    Cluster posteriors: {posteriors[i]}")
            print()

    print("Demo completed successfully!")


if __name__ == '__main__':
    main()
