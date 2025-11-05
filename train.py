"""Main training script for CENIQA."""
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from config import ModelConfig, save_config, load_config
from model import CENIQA
from dataset import IQADataset, get_train_transform, get_val_transform
from losses import CENIQALoss
from train_utils import Trainer


def set_seed(seed):
    """Set random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    """Main training function."""
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = ModelConfig()
    
    # Override with command line args
    if args.backbone:
        config.backbone = args.backbone
    if args.regressor:
        config.regressor_type = args.regressor
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    set_seed(config.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(config.save_dir) / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / 'config.json')
    
    # Data
    train_transform = get_train_transform(config.image_size)
    val_transform = get_val_transform(config.image_size)
    
    train_dataset = IQADataset(
        root_dir=args.data_root,
        csv_file=args.train_csv,
        transform=train_transform,
        training=True
    )
    
    val_dataset = IQADataset(
        root_dir=args.data_root,
        csv_file=args.val_csv,
        transform=val_transform,
        training=False
    )
    
    test_dataset = IQADataset(
        root_dir=args.data_root,
        csv_file=args.test_csv,
        transform=val_transform,
        training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = CENIQA(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and trainer
    criterion = CENIQALoss(config)
    trainer = Trainer(model, config, device)
    
    # Training loop
    best_srcc = 0
    
    for epoch in range(config.epochs):
        # Train
        train_losses = trainer.train_epoch(train_loader, criterion, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Update scheduler
        if trainer.scheduler:
            trainer.scheduler.step()
        
        # Update GMM
        if epoch % config.gmm_update_freq == 0 and epoch > 0:
            print(f"Updating GMM at epoch {epoch}...")
            trainer.update_gmm(train_loader)
        
        # Log
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_losses['total']:.4f}")
        print(f"Val SRCC: {val_metrics['srcc']:.4f}, PLCC: {val_metrics['plcc']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        
        # Save best
        if val_metrics['srcc'] > best_srcc:
            best_srcc = val_metrics['srcc']
            trainer.save_checkpoint(save_dir / 'best_model.pth', epoch, val_metrics)
            print(f"Saved best model with SRCC: {best_srcc:.4f}")
        
        # Save periodic
        if (epoch + 1) % config.checkpoint_freq == 0:
            trainer.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pth', epoch, val_metrics)
    
    # Test
    test_metrics = trainer.validate(test_loader)
    print(f"\n{'='*50}")
    print(f"Final Test Results:")
    print(f"SRCC: {test_metrics['srcc']:.4f}, PLCC: {test_metrics['plcc']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
    print(f"{'='*50}")
    
    # Save test results
    with open(save_dir / 'test_results.txt', 'w') as f:
        f.write(f"SRCC: {test_metrics['srcc']:.4f}\n")
        f.write(f"PLCC: {test_metrics['plcc']:.4f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CENIQA model')
    parser.add_argument('--config', type=str, default='configs/default.json', help='Config file path')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--train_csv', type=str, default='./data/train.csv', help='Training CSV')
    parser.add_argument('--val_csv', type=str, default='./data/val.csv', help='Validation CSV')
    parser.add_argument('--test_csv', type=str, default='./data/test.csv', help='Test CSV')
    parser.add_argument('--backbone', type=str, default=None, help='Backbone model')
    parser.add_argument('--regressor', type=str, default=None, help='Regressor type')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    
    args = parser.parse_args()
    main(args)
