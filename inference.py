"""Inference script for CENIQA model."""
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from config import load_config
from model import CENIQA


class CENIQAInference:
    """Inference wrapper for CENIQA model."""
    
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file
            device: torch device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config and model
        self.config = load_config(config_path)
        self.model = CENIQA(self.config).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize(self.config.image_size + 32),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        Predict quality score for single image.
        Args:
            image_path: Path to image file
        Returns:
            Quality score (0-1)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            score = self.model(img_tensor).item()
        
        return score
    
    def predict_batch(self, image_paths):
        """
        Predict quality scores for batch of images.
        Args:
            image_paths: List of image paths
        Returns:
            List of quality scores
        """
        # Load and preprocess images
        images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            images.append(img_tensor)
        
        images = torch.stack(images).to(self.device)
        
        # Predict
        with torch.no_grad():
            scores = self.model(images).cpu().numpy()
        
        return scores.tolist()
    
    def get_cluster_info(self, image_path):
        """
        Get cluster assignment and posterior probabilities.
        Args:
            image_path: Path to image file
        Returns:
            Cluster ID and posterior probabilities
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(img_tensor, return_all=True)
            posteriors = outputs['posteriors'][0].cpu().numpy()
            cluster_id = posteriors.argmax()
        
        return {
            'cluster_id': int(cluster_id),
            'posteriors': posteriors.tolist()
        }


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CENIQA inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--image', type=str, required=True, help='Image path or directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = CENIQAInference(args.checkpoint, args.config, args.device)
    
    # Single image or directory
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image
        score = inference.predict(str(image_path))
        info = inference.get_cluster_info(str(image_path))
        print(f"Image: {image_path.name}")
        print(f"Quality Score: {score:.4f}")
        print(f"Cluster ID: {info['cluster_id']}")
        print(f"Posteriors: {[f'{p:.4f}' for p in info['posteriors']]}")
    
    elif image_path.is_dir():
        # Directory of images
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
        scores = inference.predict_batch([str(f) for f in image_files])
        
        for img_file, score in zip(image_files, scores):
            print(f"{img_file.name}: {score:.4f}")


if __name__ == '__main__':
    main()
