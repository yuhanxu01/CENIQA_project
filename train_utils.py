"""Training utilities."""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import numpy as np
from scipy.stats import spearmanr, pearsonr


class Trainer:
    """Training manager for CENIQA."""
    
    def __init__(self, model, config, device):
        """
        Args:
            model: CENIQA model
            config: Training config
            device: torch device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        elif self.config.scheduler == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=1
            )
        return None
    
    def train_epoch(self, train_loader, criterion, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        all_losses = {}
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            images = batch['image'].to(self.device)
            
            # Prepare targets
            targets = {}
            if 'quality_score' in batch:
                targets['quality_scores'] = batch['quality_score'].to(self.device)
            if 'distortion_label' in batch:
                targets['distortion_labels'] = batch['distortion_label'].to(self.device)
            
            # Forward
            outputs = self.model(images, return_all=True)
            
            # Compute loss
            losses = criterion(outputs, targets, self.model)
            
            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate
            total_loss += losses['total'].item()
            for key, value in losses.items():
                if key not in all_losses:
                    all_losses[key] = 0
                all_losses[key] += value.item()
        
        # Average
        avg_losses = {k: v / len(train_loader) for k, v in all_losses.items()}
        return avg_losses
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                scores = batch['quality_score'].to(self.device)
                
                pred = self.model(images)
                
                predictions.extend(pred.cpu().numpy())
                ground_truth.extend(scores.cpu().numpy())
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Compute metrics
        srcc, _ = spearmanr(predictions, ground_truth)
        plcc, _ = pearsonr(predictions, ground_truth)
        rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
        
        return {
            'srcc': srcc,
            'plcc': plcc,
            'rmse': rmse
        }
    
    def update_gmm(self, train_loader):
        """Update GMM with current features."""
        self.model.eval()
        all_features = []
        
        with torch.no_grad():
            for batch in train_loader:
                images = batch['image'].to(self.device)
                features = self.model.extract_features(images)
                all_features.append(features.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        self.model.gmm.fit_sklearn(all_features)
    
    def save_checkpoint(self, save_path, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']


def compute_metrics(predictions, ground_truth):
    """Compute evaluation metrics."""
    srcc, _ = spearmanr(predictions, ground_truth)
    plcc, _ = pearsonr(predictions, ground_truth)
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    
    return {
        'srcc': srcc,
        'plcc': plcc,
        'rmse': rmse
    }
