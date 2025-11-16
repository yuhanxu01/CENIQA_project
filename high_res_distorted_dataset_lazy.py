"""
High-Resolution Distorted Image Quality Assessment Dataset - LAZY LOADING VERSION
Loads images on-the-fly to avoid memory overflow
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2


class HighResDistortedDatasetLazy(Dataset):
    """
    Lazy loading dataset that applies distortions on-the-fly.

    Key differences from the original:
    - Does NOT preload all images into memory
    - Generates distortions dynamically in __getitem__
    - Uses fixed random seed per sample for reproducibility
    - Much lower memory usage, suitable for large datasets

    Supported datasets:
    - STL-10: 96x96 images (much clearer than CIFAR-10's 32x32)
    - ImageNet-1k: variable size, typically 200x200+
    """

    DISTORTION_TYPES = [
        'gaussian_blur',
        'motion_blur',
        'gaussian_noise',
        'jpeg_compression',
        'color_saturation',
        'contrast',
        'brightness',
        'pixelation'
    ]

    def __init__(self,
                 dataset_name='stl10',
                 split='train',
                 max_samples=None,
                 distortions_per_image=5,
                 include_pristine=True,
                 distortion_strength='medium',
                 random_seed=42):
        """
        Args:
            dataset_name: 'stl10' or 'imagenet-1k'
            split: 'train' or 'test'
            max_samples: max number of reference images
            distortions_per_image: number of distorted versions per reference
            include_pristine: whether to include pristine (undistorted) images
            distortion_strength: 'light' (0.1-0.4), 'medium' (0.2-0.6), 'heavy' (0.3-1.0)
            random_seed: seed for reproducible distortion generation
        """
        self.dataset_name = dataset_name
        self.distortion_strength = distortion_strength
        self.distortions_per_image = distortions_per_image
        self.include_pristine = include_pristine
        self.random_seed = random_seed

        # Set distortion level range based on strength
        if distortion_strength == 'light':
            self.min_level, self.max_level = 0.1, 0.4
        elif distortion_strength == 'medium':
            self.min_level, self.max_level = 0.2, 0.6
        else:  # heavy
            self.min_level, self.max_level = 0.3, 1.0

        print(f"\n{'='*70}")
        print(f"Creating LAZY-LOADING dataset")
        print(f"Dataset: {dataset_name} (split={split})")
        print(f"Distortion strength: {distortion_strength} (level range: {self.min_level:.1f}-{self.max_level:.1f})")
        print(f"Max samples: {max_samples if max_samples else 'ALL'}")
        print(f"{'='*70}\n")

        # Load dataset based on name
        if dataset_name == 'stl10':
            # STL-10: 96x96 images, 10 classes - use torchvision
            import torchvision.datasets as datasets_tv
            dataset_split = 'train' if split == 'train' else 'test'
            print(f"Loading STL-10 from torchvision (split={dataset_split})...")
            self.base_dataset = datasets_tv.STL10(root='./data', split=dataset_split, download=True)
            print(f"✓ Loaded {len(self.base_dataset)} reference images from STL-10")

        elif dataset_name == 'imagenet-1k':
            # ImageNet-1k: high-res images
            from datasets import load_dataset
            try:
                self.base_dataset = load_dataset("imagenet-1k", split=split, trust_remote_code=True)
            except:
                print("WARNING: Full ImageNet-1k not available, using tiny-imagenet instead...")
                self.base_dataset = load_dataset("Maysee/tiny-imagenet", split=split)
            print(f"✓ Loaded {len(self.base_dataset)} reference images")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Limit number of reference images if requested
        if max_samples:
            self.num_reference_images = min(max_samples, len(self.base_dataset))
        else:
            self.num_reference_images = len(self.base_dataset)

        # Calculate total dataset size
        images_per_reference = distortions_per_image + (1 if include_pristine else 0)
        self.total_size = self.num_reference_images * images_per_reference

        print(f"\nDataset configuration:")
        print(f"  Reference images: {self.num_reference_images}")
        print(f"  Distortions per image: {distortions_per_image}")
        print(f"  Include pristine: {include_pristine}")
        print(f"  Total samples: {self.total_size}")
        print(f"  Memory mode: LAZY (on-the-fly loading)")
        print(f"{'='*70}\n")

        # Setup image transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.total_size

    def _get_reference_image(self, ref_idx):
        """Load and preprocess reference image."""
        if self.dataset_name == 'stl10':
            # torchvision STL10 returns (image, label)
            img, _ = self.base_dataset[ref_idx]
        else:
            # HuggingFace dataset
            item = self.base_dataset[ref_idx]
            if 'image' in item:
                img = item['image']
            elif 'img' in item:
                img = item['img']
            else:
                raise KeyError(f"Cannot find image key. Available: {item.keys()}")

        # Ensure RGB PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to 224x224 for model input
        img = img.resize((224, 224), Image.LANCZOS)

        return img

    def __getitem__(self, idx):
        """
        Dynamically generate distorted image on-the-fly.

        Index mapping:
        - If include_pristine=True and distortions_per_image=5:
          idx 0, 6, 12, ... -> pristine images
          idx 1-5, 7-11, 13-17, ... -> distorted versions
        """
        images_per_reference = self.distortions_per_image + (1 if self.include_pristine else 0)

        # Determine which reference image and which variant
        ref_idx = idx // images_per_reference
        variant_idx = idx % images_per_reference

        # Load reference image from disk
        img = self._get_reference_image(ref_idx)

        # Determine if this should be pristine or distorted
        if self.include_pristine and variant_idx == 0:
            # Pristine image
            distorted_img = img
            quality_score = 100.0
        else:
            # Distorted image
            # Adjust variant_idx if pristine is included
            if self.include_pristine:
                variant_idx -= 1

            # Use deterministic random seed for reproducibility
            # Same ref_idx + variant_idx will always get same distortion
            seed = self.random_seed + ref_idx * 1000 + variant_idx
            rng = np.random.RandomState(seed)

            # Randomly select distortion type and level (but deterministic per seed)
            distortion_type = rng.choice(self.DISTORTION_TYPES)
            distortion_level = rng.uniform(self.min_level, self.max_level)

            # Apply distortion
            distorted_img, quality_score = self.apply_distortion(
                img, distortion_type, distortion_level, rng
            )

        # Transform to tensor
        img_tensor = self.transform(distorted_img)
        score_normalized = quality_score / 100.0

        return img_tensor, torch.tensor(score_normalized, dtype=torch.float32)

    def apply_distortion(self, img, distortion_type, level, rng=None):
        """
        Apply specific distortion to image and calculate quality score.

        Args:
            img: PIL Image (224x224)
            distortion_type: type of distortion
            level: distortion intensity
            rng: numpy RandomState for reproducibility

        Returns:
            distorted_img: PIL Image
            quality_score: float (0-100, higher = better quality)
        """
        if rng is None:
            rng = np.random

        img = img.copy()
        img_array = np.array(img)

        if distortion_type == 'gaussian_blur':
            kernel_size = int(3 + level * 10)
            if kernel_size % 2 == 0:
                kernel_size += 1
            distorted = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
            quality_score = 100 - level * 60

        elif distortion_type == 'motion_blur':
            kernel_size = int(5 + level * 15)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            distorted = cv2.filter2D(img_array, -1, kernel)
            quality_score = 100 - level * 65

        elif distortion_type == 'gaussian_noise':
            noise_std = level * 30
            noise = rng.randn(*img_array.shape) * noise_std
            distorted = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            quality_score = 100 - level * 70

        elif distortion_type == 'jpeg_compression':
            quality = int(100 - level * 75)
            quality = max(10, quality)
            img_pil = Image.fromarray(img_array)
            import io
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            distorted_pil = Image.open(buffer)
            distorted_pil.load()
            distorted = np.array(distorted_pil)
            buffer.close()
            quality_score = quality * 0.8 + 20

        elif distortion_type == 'color_saturation':
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
            if level < 0.5:
                saturation_factor = level * 2
                quality_score = 60 + (1 - level) * 40
            else:
                saturation_factor = 1 + (level - 0.5) * 3
                quality_score = 100 - (level - 0.5) * 100

            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            distorted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        elif distortion_type == 'contrast':
            if level < 0.5:
                contrast_factor = 0.4 + level
                quality_score = 50 + level * 100
            else:
                contrast_factor = 1.0 + (level - 0.5) * 1.5
                quality_score = 100 - (level - 0.5) * 80

            mean = img_array.mean()
            distorted = np.clip((img_array - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        elif distortion_type == 'brightness':
            if level < 0.5:
                brightness_shift = -level * 80
                quality_score = 55 + level * 90
            else:
                brightness_shift = (level - 0.5) * 100
                quality_score = 100 - (level - 0.5) * 70

            distorted = np.clip(img_array + brightness_shift, 0, 255).astype(np.uint8)

        elif distortion_type == 'pixelation':
            h, w = img_array.shape[:2]
            block_size = int(2 + level * 12)

            small_h, small_w = max(1, h // block_size), max(1, w // block_size)
            small = cv2.resize(img_array, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

            distorted = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            quality_score = 100 - level * 65

        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

        quality_score = np.clip(quality_score, 0, 100)

        return Image.fromarray(distorted), quality_score


# For backward compatibility, export as main class name
HighResDistortedDataset = HighResDistortedDatasetLazy


if __name__ == '__main__':
    print("="*70)
    print("Testing Lazy-Loading High-Resolution Dataset")
    print("="*70)

    # Test with small dataset
    dataset = HighResDistortedDatasetLazy(
        dataset_name='stl10',
        split='train',
        max_samples=100,
        distortions_per_image=5,
        include_pristine=True,
        distortion_strength='medium'
    )

    print(f"\n{'='*70}")
    print(f"Dataset ready: {len(dataset)} samples")
    print(f"Testing sample loading...")

    # Test loading a few samples
    for i in [0, 1, 10, 50]:
        img_tensor, score = dataset[i]
        print(f"  Sample {i:3d}: shape={img_tensor.shape}, score={score.item():.3f}")

    print(f"\n✓ Lazy loading works! Images loaded on-the-fly.")
    print(f"✓ Memory usage: MINIMAL (only loads one image at a time)")
    print("="*70)
