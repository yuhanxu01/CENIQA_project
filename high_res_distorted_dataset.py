"""
High-Resolution Distorted Image Quality Assessment Dataset
Uses higher resolution datasets (STL-10, ImageNet) for clearer images
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


class HighResDistortedDataset(Dataset):
    """
    Dataset that applies various distortions to high-resolution reference images.

    Supported datasets:
    - STL-10: 96x96 images (much clearer than CIFAR-10's 32x32)
    - ImageNet-1k: variable size, typically 200x200+

    Distortion types (same as before but applied to clearer images):
    1. Gaussian Blur - 高斯模糊
    2. Motion Blur - 运动模糊
    3. Gaussian Noise - 高斯噪声
    4. JPEG Compression - JPEG压缩
    5. Color Saturation - 色彩饱和度
    6. Contrast Change - 对比度变化
    7. Brightness Change - 亮度变化
    8. Pixelation - 像素化/块效应
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
                 dataset_name='stl10',  # 'stl10' or 'imagenet-1k'
                 split='train',
                 max_samples=None,
                 distortions_per_image=5,
                 include_pristine=True,
                 distortion_strength='medium'):  # 'light', 'medium', 'heavy'
        """
        Args:
            dataset_name: 'stl10' or 'imagenet-1k'
            split: 'train' or 'test'
            max_samples: max number of reference images
            distortions_per_image: number of distorted versions per reference
            include_pristine: whether to include pristine (undistorted) images
            distortion_strength: 'light' (0.1-0.4), 'medium' (0.2-0.6), 'heavy' (0.3-1.0)
        """
        try:
            from datasets import load_dataset
            import torchvision.datasets as datasets_tv

            self.dataset_name = dataset_name
            self.distortion_strength = distortion_strength

            # Set distortion level range based on strength
            if distortion_strength == 'light':
                self.min_level, self.max_level = 0.1, 0.4
            elif distortion_strength == 'medium':
                self.min_level, self.max_level = 0.2, 0.6
            else:  # heavy
                self.min_level, self.max_level = 0.3, 1.0

            print(f"Loading {dataset_name} dataset (split={split}, max_samples={max_samples})...")
            print(f"Distortion strength: {distortion_strength} (level range: {self.min_level:.1f}-{self.max_level:.1f})")

            # Load dataset based on name
            if dataset_name == 'stl10':
                # STL-10: 96x96 images, 10 classes - use torchvision
                dataset_split = 'train' if split == 'train' else 'test'
                print(f"Loading STL-10 from torchvision (split={dataset_split})...")
                stl10_dataset = datasets_tv.STL10(root='./data', split=dataset_split, download=True)

                # Convert to list format compatible with the rest of the code
                dataset = [{'image': img, 'label': label} for img, label in stl10_dataset]
                print(f"Loaded {len(dataset)} images from STL-10")
            elif dataset_name == 'imagenet-1k':
                # ImageNet-1k: high-res images (usually 200x200+)
                # Note: This requires authentication, use a subset if not available
                try:
                    dataset = load_dataset("imagenet-1k", split=split, trust_remote_code=True)
                except:
                    print("WARNING: Full ImageNet-1k not available, using tiny-imagenet instead...")
                    dataset = load_dataset("Maysee/tiny-imagenet", split=split)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            if max_samples:
                # Handle both list and HuggingFace Dataset formats
                if hasattr(dataset, 'select'):
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                else:
                    dataset = dataset[:max_samples]

            self.reference_images = []
            self.distorted_images = []
            self.quality_scores = []
            self.distortion_types = []
            self.distortion_levels = []

            total_images = len(dataset) * (distortions_per_image + (1 if include_pristine else 0))
            print(f"Creating distorted images from {len(dataset)} references...")
            print(f"Distortions per image: {distortions_per_image}")
            print(f"Total images: {total_images}")

            for item in tqdm(dataset, desc=f"Processing {split}"):
                # Get image (key may vary by dataset)
                if 'image' in item:
                    img = item['image']
                elif 'img' in item:
                    img = item['img']
                else:
                    raise KeyError(f"Cannot find image key in dataset. Available keys: {item.keys()}")

                # Ensure RGB
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize to 224x224 NOW (before distortion) to maintain quality
                # This ensures distortions are applied to higher-res images
                img = img.resize((224, 224), Image.LANCZOS)

                # Add pristine image (quality score = 100)
                if include_pristine:
                    self.distorted_images.append(img.copy())
                    self.quality_scores.append(100.0)
                    self.distortion_types.append('pristine')
                    self.distortion_levels.append(0)

                # Create distorted versions
                for _ in range(distortions_per_image):
                    # Randomly select distortion type and level
                    distortion_type = np.random.choice(self.DISTORTION_TYPES)
                    distortion_level = np.random.uniform(self.min_level, self.max_level)

                    # Apply distortion
                    distorted_img, quality_score = self.apply_distortion(
                        img.copy(), distortion_type, distortion_level
                    )

                    self.distorted_images.append(distorted_img)
                    self.quality_scores.append(quality_score)
                    self.distortion_types.append(distortion_type)
                    self.distortion_levels.append(distortion_level)

            print(f"\nDataset created:")
            print(f"  Total images: {len(self.distorted_images)}")
            print(f"  Score range: {min(self.quality_scores):.2f} - {max(self.quality_scores):.2f}")
            print(f"  Score mean: {np.mean(self.quality_scores):.2f}")

            # Print distortion type distribution
            print(f"\nDistortion type distribution:")
            from collections import Counter
            dist_counts = Counter(self.distortion_types)
            for dist_type, count in sorted(dist_counts.items()):
                print(f"  {dist_type:20s}: {count:4d} images")

        except ImportError as e:
            raise ImportError(f"Please install required packages: {e}")

        # Setup transforms (no resize needed, already 224x224)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def apply_distortion(self, img, distortion_type, level):
        """
        Apply specific distortion to image and calculate quality score.

        NOTE: Image is already 224x224 at this point, so distortions
        will be applied to high-quality images.

        Args:
            img: PIL Image (224x224)
            distortion_type: type of distortion
            level: distortion intensity (varies by strength setting)

        Returns:
            distorted_img: PIL Image
            quality_score: float (0-100, higher = better quality)
        """
        img = img.copy()
        img_array = np.array(img)

        if distortion_type == 'gaussian_blur':
            # Gaussian blur - scaled for 224x224 images
            kernel_size = int(3 + level * 10)  # 3-13 for light/medium
            if kernel_size % 2 == 0:
                kernel_size += 1
            distorted = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
            quality_score = 100 - level * 60  # 40-100

        elif distortion_type == 'motion_blur':
            # Motion blur
            kernel_size = int(5 + level * 15)  # 5-20
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            distorted = cv2.filter2D(img_array, -1, kernel)
            quality_score = 100 - level * 65  # 35-100

        elif distortion_type == 'gaussian_noise':
            # Additive Gaussian noise - reduced for better visibility
            noise_std = level * 30  # 6-30 (less aggressive than before)
            noise = np.random.randn(*img_array.shape) * noise_std
            distorted = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            quality_score = 100 - level * 70  # 30-100

        elif distortion_type == 'jpeg_compression':
            # JPEG compression artifacts
            quality = int(100 - level * 75)  # 25-98 (less aggressive)
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
            # Change color saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
            if level < 0.5:
                saturation_factor = level * 2  # 0.2-1.0 (desaturation)
                quality_score = 60 + (1 - level) * 40  # 60-100
            else:
                saturation_factor = 1 + (level - 0.5) * 3  # 1.0-2.5 (oversaturation)
                quality_score = 100 - (level - 0.5) * 100  # 50-100

            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            distorted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        elif distortion_type == 'contrast':
            # Contrast change
            if level < 0.5:
                contrast_factor = 0.4 + level  # 0.4-0.9 (low contrast)
                quality_score = 50 + level * 100  # 50-100
            else:
                contrast_factor = 1.0 + (level - 0.5) * 1.5  # 1.0-1.75 (high contrast)
                quality_score = 100 - (level - 0.5) * 80  # 60-100

            mean = img_array.mean()
            distorted = np.clip((img_array - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        elif distortion_type == 'brightness':
            # Brightness change
            if level < 0.5:
                brightness_shift = -level * 80  # -16 to -40 (darker)
                quality_score = 55 + level * 90  # 55-100
            else:
                brightness_shift = (level - 0.5) * 100  # 0 to +50 (brighter)
                quality_score = 100 - (level - 0.5) * 70  # 65-100

            distorted = np.clip(img_array + brightness_shift, 0, 255).astype(np.uint8)

        elif distortion_type == 'pixelation':
            # Pixelation / block artifacts - scaled for 224x224
            h, w = img_array.shape[:2]
            block_size = int(2 + level * 12)  # 2-14 pixels

            # Downsample
            small_h, small_w = max(1, h // block_size), max(1, w // block_size)
            small = cv2.resize(img_array, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

            # Upsample back
            distorted = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            quality_score = 100 - level * 65  # 35-100

        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

        # Ensure quality score is in valid range
        quality_score = np.clip(quality_score, 0, 100)

        return Image.fromarray(distorted), quality_score

    def __len__(self):
        return len(self.distorted_images)

    def __getitem__(self, idx):
        img = self.distorted_images[idx]
        score = self.quality_scores[idx]

        img_tensor = self.transform(img)
        score_normalized = score / 100.0

        return img_tensor, torch.tensor(score_normalized, dtype=torch.float32)

    def get_metadata(self, idx):
        """Get metadata for an image (for analysis)."""
        return {
            'distortion_type': self.distortion_types[idx],
            'distortion_level': self.distortion_levels[idx],
            'quality_score': self.quality_scores[idx]
        }


def visualize_high_res_distortions(dataset_name='stl10', distortion_strength='medium'):
    """
    Helper function to visualize different distortion types on high-res images.
    """
    import matplotlib.pyplot as plt
    from datasets import load_dataset

    # Load one sample image
    print(f"Loading sample image from {dataset_name}...")
    if dataset_name == 'stl10':
        dataset = load_dataset("stl10", split='train')
    elif dataset_name == 'imagenet-1k':
        try:
            dataset = load_dataset("imagenet-1k", split='train')
        except:
            print("Using tiny-imagenet instead...")
            dataset = load_dataset("Maysee/tiny-imagenet", split='train')

    # Get image
    if 'image' in dataset[0]:
        img = dataset[0]['image']
    elif 'img' in dataset[0]:
        img = dataset[0]['img']

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to 224x224
    img = img.resize((224, 224), Image.LANCZOS)
    print(f"Image loaded and resized: size={img.size}, mode={img.mode}")

    # Create dataset instance for distortion application
    temp_dataset = HighResDistortedDataset.__new__(HighResDistortedDataset)
    temp_dataset.distortion_strength = distortion_strength

    # Set level ranges
    if distortion_strength == 'light':
        temp_dataset.min_level, temp_dataset.max_level = 0.1, 0.4
    elif distortion_strength == 'medium':
        temp_dataset.min_level, temp_dataset.max_level = 0.2, 0.6
    else:
        temp_dataset.min_level, temp_dataset.max_level = 0.3, 1.0

    # Test each distortion type at different levels
    fig, axes = plt.subplots(len(HighResDistortedDataset.DISTORTION_TYPES) + 1, 4,
                             figsize=(16, 3.5 * (len(HighResDistortedDataset.DISTORTION_TYPES) + 1)))

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'Original (224x224)\nQuality: 100.0', fontsize=10)
    axes[0, 0].axis('off')
    for i in range(1, 4):
        axes[0, i].axis('off')

    success_count = 0
    failure_count = 0

    # Each distortion type
    print(f"\nApplying distortions (strength={distortion_strength}):")
    for row, dist_type in enumerate(HighResDistortedDataset.DISTORTION_TYPES, start=1):
        print(f"\n  {row}. {dist_type}:")
        for col, level in enumerate([0.2, 0.4, 0.6, 0.8]):
            try:
                img_copy = img.copy()
                distorted_img, quality_score = temp_dataset.apply_distortion(
                    img_copy, dist_type, level
                )
                axes[row, col].imshow(distorted_img)
                axes[row, col].set_title(f'{dist_type}\nlevel={level:.1f}, Q={quality_score:.1f}',
                                        fontsize=9)
                axes[row, col].axis('off')
                print(f"    ✓ level={level:.1f} -> Q={quality_score:.1f}")
                success_count += 1
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'ERROR\n{str(e)[:30]}...',
                                   ha='center', va='center', fontsize=6, color='red',
                                   transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
                print(f"    ✗ level={level:.1f} FAILED: {str(e)}")
                failure_count += 1

    plt.tight_layout()
    filename = f'high_res_distortions_{dataset_name}_{distortion_strength}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print(f"Visualization complete!")
    print(f"  Success: {success_count}/{success_count + failure_count}")
    print(f"\nSaved to: {filename}")
    print("="*60)
    plt.close()


if __name__ == '__main__':
    # Test the dataset
    print("="*60)
    print("Testing High-Resolution Distorted Dataset")
    print("="*60)

    print("\n1. Creating STL-10 dataset (medium strength)...")
    dataset = HighResDistortedDataset(
        dataset_name='stl10',
        split='train',
        max_samples=10,
        distortions_per_image=5,
        include_pristine=True,
        distortion_strength='medium'
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"\nSample metadata:")
    for i in range(min(12, len(dataset))):
        meta = dataset.get_metadata(i)
        print(f"  Image {i:2d}: {meta['distortion_type']:20s} "
              f"level={meta['distortion_level']:.2f} "
              f"quality={meta['quality_score']:.1f}")

    # Visualize distortions
    print("\n2. Generating distortion visualization...")
    visualize_high_res_distortions(dataset_name='stl10', distortion_strength='medium')

    print("\n" + "="*60)
    print("Test complete! Check 'high_res_distortions_stl10_medium.png'")
    print("="*60)
