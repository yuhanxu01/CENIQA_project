"""
Distorted Image Quality Assessment Dataset
Creates images with various types of distortions for IQA training
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFilter
import cv2
from tqdm import tqdm


class DistortedImageDataset(Dataset):
    """
    Dataset that applies various distortions to reference images.

    Distortion types:
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

    def __init__(self, split='train', max_samples=None,
                 distortions_per_image=5, include_pristine=True):
        """
        Args:
            split: 'train' or 'test'
            max_samples: max number of reference images
            distortions_per_image: number of distorted versions per reference
            include_pristine: whether to include pristine (undistorted) images
        """
        try:
            from datasets import load_dataset
            from torchvision import transforms

            print(f"Loading dataset (split={split}, max_samples={max_samples})...")
            dataset = load_dataset("cifar10", split=split)

            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            self.reference_images = []
            self.distorted_images = []
            self.quality_scores = []
            self.distortion_types = []
            self.distortion_levels = []

            print(f"Creating distorted images from {len(dataset)} references...")
            print(f"Distortions per image: {distortions_per_image}")
            print(f"Total images: {len(dataset) * (distortions_per_image + (1 if include_pristine else 0))}")

            for item in tqdm(dataset, desc=f"Processing {split}"):
                img = item['img']
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Add pristine image (quality score = 100)
                if include_pristine:
                    self.distorted_images.append(img)
                    self.quality_scores.append(100.0)
                    self.distortion_types.append('pristine')
                    self.distortion_levels.append(0)

                # Create distorted versions
                for _ in range(distortions_per_image):
                    # Randomly select distortion type and level
                    distortion_type = np.random.choice(self.DISTORTION_TYPES)
                    distortion_level = np.random.uniform(0.2, 1.0)  # 0.2-1.0 (higher = more distortion)

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

        # Setup transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def apply_distortion(self, img, distortion_type, level):
        """
        Apply specific distortion to image and calculate quality score.

        Args:
            img: PIL Image
            distortion_type: type of distortion
            level: distortion intensity (0.2-1.0, higher = more distortion)

        Returns:
            distorted_img: PIL Image
            quality_score: float (0-100, higher = better quality)
        """
        # Make a deep copy to ensure independence
        img = img.copy()
        img_array = np.array(img)

        if distortion_type == 'gaussian_blur':
            # Gaussian blur with varying kernel size
            kernel_size = int(3 + level * 12)  # 3-15
            if kernel_size % 2 == 0:
                kernel_size += 1
            distorted = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
            quality_score = 100 - level * 60  # 40-100

        elif distortion_type == 'motion_blur':
            # Motion blur simulation
            kernel_size = int(5 + level * 20)  # 5-25
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            distorted = cv2.filter2D(img_array, -1, kernel)
            quality_score = 100 - level * 65  # 35-100

        elif distortion_type == 'gaussian_noise':
            # Additive Gaussian noise
            noise_std = level * 40  # 8-40
            noise = np.random.randn(*img_array.shape) * noise_std
            distorted = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            quality_score = 100 - level * 70  # 30-100

        elif distortion_type == 'jpeg_compression':
            # JPEG compression artifacts
            quality = int(100 - level * 80)  # 20-98
            quality = max(5, quality)
            img_pil = Image.fromarray(img_array)
            import io
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            # Load the image data immediately to avoid buffer issues
            distorted_pil = Image.open(buffer)
            distorted_pil.load()  # Force loading image data from buffer
            distorted = np.array(distorted_pil)
            buffer.close()  # Explicitly close buffer after loading
            quality_score = quality * 0.8 + 20  # Approximate quality based on compression

        elif distortion_type == 'color_saturation':
            # Change color saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
            # level < 0.5: desaturation, level > 0.5: oversaturation
            if level < 0.5:
                saturation_factor = level * 2  # 0.4-1.0
                quality_score = 60 + (1 - level) * 40  # 60-100
            else:
                saturation_factor = 1 + (level - 0.5) * 4  # 1.0-3.0
                quality_score = 100 - (level - 0.5) * 100  # 50-100

            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            distorted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        elif distortion_type == 'contrast':
            # Contrast change
            if level < 0.5:
                # Low contrast
                contrast_factor = 0.3 + level  # 0.3-0.8
                quality_score = 50 + level * 100  # 50-100
            else:
                # High contrast
                contrast_factor = 1.0 + (level - 0.5) * 2  # 1.0-2.0
                quality_score = 100 - (level - 0.5) * 80  # 60-100

            mean = img_array.mean()
            distorted = np.clip((img_array - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        elif distortion_type == 'brightness':
            # Brightness change
            if level < 0.5:
                # Darker
                brightness_shift = -level * 100  # -20 to -50
                quality_score = 55 + level * 90  # 55-100
            else:
                # Brighter
                brightness_shift = (level - 0.5) * 120  # 0 to +60
                quality_score = 100 - (level - 0.5) * 70  # 65-100

            distorted = np.clip(img_array + brightness_shift, 0, 255).astype(np.uint8)

        elif distortion_type == 'pixelation':
            # Pixelation / block artifacts
            h, w = img_array.shape[:2]
            block_size = int(2 + level * 14)  # 2-16 pixels

            # Downsample
            small_h, small_w = h // block_size, w // block_size
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


def visualize_distortions():
    """
    Helper function to visualize different distortion types.
    Useful for debugging and understanding the dataset.
    """
    import matplotlib.pyplot as plt
    from datasets import load_dataset

    # Load one sample image
    print("Loading sample image from CIFAR-10...")
    dataset = load_dataset("cifar10", split='train')
    img = dataset[0]['img']
    if img.mode != 'RGB':
        img = img.convert('RGB')
    print(f"Image loaded: size={img.size}, mode={img.mode}")

    # Create dataset instance for distortion application
    temp_dataset = DistortedImageDataset.__new__(DistortedImageDataset)

    # Test each distortion type at different levels
    fig, axes = plt.subplots(len(DistortedImageDataset.DISTORTION_TYPES) + 1, 4,
                             figsize=(12, 3 * (len(DistortedImageDataset.DISTORTION_TYPES) + 1)))

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original (100.0)')
    axes[0, 0].axis('off')
    for i in range(1, 4):
        axes[0, i].axis('off')

    # Track success/failure
    success_count = 0
    failure_count = 0
    failed_distortions = []

    # Each distortion type with error handling
    print("\nApplying distortions:")
    for row, dist_type in enumerate(DistortedImageDataset.DISTORTION_TYPES, start=1):
        print(f"\n  {row}. {dist_type}:")
        for col, level in enumerate([0.3, 0.5, 0.7, 0.9]):
            try:
                # Make a fresh copy of the original image for each distortion
                img_copy = img.copy()
                distorted_img, quality_score = temp_dataset.apply_distortion(
                    img_copy, dist_type, level
                )
                axes[row, col].imshow(distorted_img)
                axes[row, col].set_title(f'{dist_type}\nlevel={level:.1f}, Q={quality_score:.1f}',
                                        fontsize=8)
                axes[row, col].axis('off')
                print(f"    ✓ level={level:.1f} -> Q={quality_score:.1f}")
                success_count += 1
            except Exception as e:
                # Display error in the plot
                axes[row, col].text(0.5, 0.5, f'ERROR\n{str(e)[:30]}...',
                                   ha='center', va='center', fontsize=6, color='red',
                                   transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{dist_type}\nlevel={level:.1f}\nFAILED',
                                        fontsize=8, color='red')
                axes[row, col].axis('off')
                print(f"    ✗ level={level:.1f} FAILED: {str(e)}")
                failure_count += 1
                if dist_type not in failed_distortions:
                    failed_distortions.append(dist_type)

    plt.tight_layout()
    plt.savefig('distortion_examples.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print(f"Visualization complete!")
    print(f"  Success: {success_count}")
    print(f"  Failures: {failure_count}")
    if failed_distortions:
        print(f"  Failed distortion types: {', '.join(failed_distortions)}")
    print(f"\nSaved to: distortion_examples.png")
    print("="*60)
    plt.show()


if __name__ == '__main__':
    # Test the dataset
    print("Creating test dataset...")
    dataset = DistortedImageDataset(
        split='train',
        max_samples=10,
        distortions_per_image=5,
        include_pristine=True
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"\nSample metadata:")
    for i in range(min(10, len(dataset))):
        meta = dataset.get_metadata(i)
        print(f"  Image {i}: {meta['distortion_type']:20s} "
              f"level={meta['distortion_level']:.2f} "
              f"quality={meta['quality_score']:.1f}")

    # Visualize distortions
    print("\nGenerating distortion visualization...")
    visualize_distortions()
