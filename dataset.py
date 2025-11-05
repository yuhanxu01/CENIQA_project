"""Dataset for Image Quality Assessment."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


class IQADataset(Dataset):
    """IQA dataset with distortion augmentation."""
    
    DISTORTION_TYPES = [
        'gaussian_blur', 'motion_blur', 'defocus_blur',
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'jpeg_compression', 'jpeg2000_compression',
        'brightness', 'contrast'
    ]
    
    def __init__(self, root_dir, csv_file, transform=None, training=True):
        """
        Args:
            root_dir: Directory with all images
            csv_file: CSV file with image paths and MOS scores
            transform: Image transforms
            training: Whether this is training set
        """
        self.root_dir = Path(root_dir)
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.training = training
        
    def __len__(self):
        return len(self.data)
    
    def apply_distortion(self, img, distortion_type, level):
        """Apply synthetic distortion to image."""
        if distortion_type == 'gaussian_blur':
            kernel_size = int(3 + level * 4) * 2 + 1
            img = transforms.GaussianBlur(kernel_size, sigma=1 + level * 2)(img)
        elif distortion_type == 'gaussian_noise':
            noise = torch.randn_like(img) * (0.01 + level * 0.09)
            img = torch.clamp(img + noise, 0, 1)
        elif distortion_type == 'brightness':
            img = transforms.ColorJitter(brightness=level)(img)
        elif distortion_type == 'contrast':
            img = transforms.ColorJitter(contrast=level)(img)
        
        return img
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.root_dir / row['image_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        sample = {
            'image': img,
            'quality_score': torch.tensor(row['mos'] / 100.0, dtype=torch.float32)
        }
        
        if self.training:
            # Apply synthetic distortion
            distortion_type = np.random.choice(self.DISTORTION_TYPES)
            distortion_level = np.random.uniform(0, 1)
            distortion_idx = self.DISTORTION_TYPES.index(distortion_type)
            
            img_distorted = self.apply_distortion(img.clone(), distortion_type, distortion_level)
            
            sample['image_distorted'] = img_distorted
            sample['distortion_label'] = torch.tensor(distortion_idx, dtype=torch.long)
            sample['distortion_level'] = torch.tensor(distortion_level, dtype=torch.float32)
            
            # Create ranking pairs
            if np.random.random() < 0.5:
                level2 = np.random.uniform(0, 1)
                img_distorted2 = self.apply_distortion(img.clone(), distortion_type, level2)
                sample['image_pair'] = img_distorted2
                sample['level_pair'] = torch.tensor(level2, dtype=torch.float32)
        
        return sample


def get_train_transform(image_size=384):
    """Get training transforms."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform(image_size=384):
    """Get validation transforms."""
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
