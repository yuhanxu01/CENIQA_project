"""
Quick test script to verify high-resolution dataset works properly.
Generates visualization showing the difference between CIFAR-10 and STL-10.
"""
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

print("="*70)
print("Testing High-Resolution Dataset")
print("="*70)

# Test 1: Create a small high-res dataset
print("\n1. Creating STL-10 high-res dataset (10 samples)...")
try:
    from high_res_distorted_dataset import HighResDistortedDataset

    dataset = HighResDistortedDataset(
        dataset_name='stl10',
        split='train',
        max_samples=10,
        distortions_per_image=5,
        include_pristine=True,
        distortion_strength='medium'
    )

    print(f"   ✓ Dataset created: {len(dataset)} images")
    print(f"   ✓ Image shape: {dataset[0][0].shape}")

    # Print some metadata
    print("\n   Sample metadata:")
    for i in range(min(10, len(dataset))):
        meta = dataset.get_metadata(i)
        print(f"     {i:2d}. {meta['distortion_type']:20s} "
              f"level={meta['distortion_level']:.2f} Q={meta['quality_score']:.1f}")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Visualize distortions
print("\n2. Generating distortion visualization...")
try:
    from high_res_distorted_dataset import visualize_high_res_distortions

    visualize_high_res_distortions(dataset_name='stl10', distortion_strength='medium')
    print("   ✓ Visualization saved: high_res_distortions_stl10_medium.png")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Compare CIFAR-10 vs STL-10 side by side
print("\n3. Creating comparison (CIFAR-10 vs STL-10)...")
try:
    from datasets import load_dataset

    # Load one image from each
    cifar = load_dataset("cifar10", split='train')
    stl = load_dataset("stl10", split='train')

    cifar_img = cifar[0]['img']
    if 'image' in stl[0]:
        stl_img = stl[0]['image']
    else:
        stl_img = stl[0]['img']

    if not isinstance(stl_img, Image.Image):
        stl_img = Image.fromarray(stl_img)

    # Resize both to 224x224 for comparison
    cifar_224 = cifar_img.resize((224, 224), Image.LANCZOS)
    stl_224 = stl_img.resize((224, 224), Image.LANCZOS)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Row 1: Original sizes
    axes[0, 0].imshow(cifar_img)
    axes[0, 0].set_title(f'CIFAR-10 Original\n{cifar_img.size[0]}x{cifar_img.size[1]}', fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(stl_img)
    axes[0, 1].set_title(f'STL-10 Original\n{stl_img.size[0]}x{stl_img.size[1]}', fontsize=10)
    axes[0, 1].axis('off')

    # Show size comparison
    axes[0, 2].text(0.5, 0.5,
                    f'CIFAR-10: {cifar_img.size[0]}x{cifar_img.size[1]}\n(32x32 pixels)\n\n'
                    f'STL-10: {stl_img.size[0]}x{stl_img.size[1]}\n(96x96 pixels)\n\n'
                    f'STL-10 is 3x larger!\n'
                    f'9x more pixels!',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[0, 2].axis('off')

    # Row 2: Both resized to 224x224
    axes[1, 0].imshow(cifar_224)
    axes[1, 0].set_title('CIFAR-10 → 224x224\n(Blurry upscale)', fontsize=10, color='red')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(stl_224)
    axes[1, 1].set_title('STL-10 → 224x224\n(Clearer upscale)', fontsize=10, color='green')
    axes[1, 1].axis('off')

    axes[1, 2].text(0.5, 0.5,
                    'After resizing to 224x224:\n\n'
                    'CIFAR-10: Very blurry\n'
                    '(32→224 = 7x upscale)\n\n'
                    'STL-10: Much clearer\n'
                    '(96→224 = 2.3x upscale)\n\n'
                    '→ STL-10 preserves\nmore details!',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 2].axis('off')

    plt.suptitle('Dataset Comparison: CIFAR-10 vs STL-10', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Comparison saved: dataset_comparison.png")
    plt.close()

except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test Complete!")
print("\nGenerated files:")
print("  1. high_res_distortions_stl10_medium.png - Shows all distortion types")
print("  2. dataset_comparison.png - CIFAR-10 vs STL-10 comparison")
print("\nYou can now use train_high_res.py for training with clearer images!")
print("="*70)
