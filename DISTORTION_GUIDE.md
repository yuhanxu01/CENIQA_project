# Distorted Image Quality Assessment - Guide

## Problem with Previous Approach

**Old approach**: Used pristine CIFAR-10 images and calculated "synthetic quality scores" based on image statistics (brightness, contrast, sharpness).

**Issues**:
1. ❌ No actual image distortions applied
2. ❌ All images were similar quality (pristine)
3. ❌ GMM clusters couldn't learn meaningful quality differences
4. ❌ All samples collapsed into one cluster

## New Approach: Real Distortions

**New dataset** (`distorted_dataset.py`): Creates distorted versions of reference images with **8 types of distortions**:

### 8 Distortion Types Implemented

| Distortion Type | Description | Quality Range | Cluster Expected |
|-----------------|-------------|---------------|------------------|
| **Gaussian Blur** | 高斯模糊 - Simulates out-of-focus | 40-100 | Blur cluster |
| **Motion Blur** | 运动模糊 - Simulates camera shake | 35-100 | Blur cluster |
| **Gaussian Noise** | 高斯噪声 - Random pixel noise | 30-100 | Noise cluster |
| **JPEG Compression** | JPEG压缩 - Compression artifacts | 20-100 | Compression cluster |
| **Color Saturation** | 色彩饱和度 - Over/under saturated | 50-100 | Color cluster |
| **Contrast** | 对比度变化 - High/low contrast | 50-100 | Contrast cluster |
| **Brightness** | 亮度变化 - Too dark/bright | 55-100 | Brightness cluster |
| **Pixelation** | 像素化/块效应 - Block artifacts | 35-100 | Pixelation cluster |
| **Pristine** | 原始图像 - No distortion | 100 | High quality cluster |

### Quality Score Calculation

Each distortion has a **distortion level** (0.2-1.0):
- **Lower level (0.2-0.4)**: Mild distortion → Higher quality score (70-100)
- **Medium level (0.5-0.7)**: Moderate distortion → Medium quality score (50-70)
- **High level (0.8-1.0)**: Severe distortion → Lower quality score (20-50)

## How to Use

### 1. Visualize Distortions First

```python
# Run this to see examples of all distortions
python distorted_dataset.py
```

This generates `distortion_examples.png` showing all 8 distortion types at 4 different levels.

### 2. Train with Distorted Images

**Quick training (for testing):**
```bash
python train_with_distortions.py \
    --experiment_name quick_test_distorted \
    --train_samples 100 \
    --val_samples 50 \
    --distortions_per_image 3 \
    --epochs 20 \
    --batch_size 64
```

**Full training (GPU recommended):**
```bash
python train_with_distortions.py \
    --experiment_name resnet18_distorted \
    --backbone resnet18 \
    --n_clusters 8 \
    --train_samples 500 \
    --val_samples 100 \
    --distortions_per_image 5 \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --refit_interval 10 \
    --cluster_loss_weight 0.5
```

**Large-scale training:**
```bash
python train_with_distortions.py \
    --experiment_name resnet50_distorted_large \
    --backbone resnet50 \
    --n_clusters 12 \
    --train_samples 1000 \
    --val_samples 200 \
    --distortions_per_image 8 \
    --epochs 100 \
    --batch_size 128
```

### 3. Parameters Explained

- `--train_samples 500`: Use 500 reference images
- `--distortions_per_image 5`: Create 5 distorted versions of each reference
- Total training images = 500 × (5 + 1) = **3000 images**
  - 500 pristine images (quality = 100)
  - 2500 distorted images (quality = 20-100)

- `--n_clusters 8`: Number of GMM clusters (should match number of distortion types)
- `--refit_interval 10`: Re-fit GMM every 10 epochs to adapt to changing features
- `--cluster_loss_weight 0.5`: Balance between quality prediction and cluster separation

### 4. Expected Results

With distorted images, you should see:

✅ **Multiple active clusters** (not just one!)
- Each cluster corresponds to a distortion type or quality range
- Example:
  - Cluster 0: Gaussian blur (quality 40-60)
  - Cluster 1: Motion blur (quality 35-55)
  - Cluster 2: Noise (quality 30-50)
  - Cluster 3: JPEG compression (quality 20-60)
  - Cluster 4: Color issues (quality 50-80)
  - Cluster 5: Contrast/brightness (quality 55-90)
  - Cluster 6: Pixelation (quality 35-65)
  - Cluster 7: Pristine images (quality 90-100)

✅ **Better SRCC** - More diverse training data leads to better generalization

✅ **Interpretable clusters** - Each cluster captures specific types of quality degradation

## Comparison: Old vs New

### Old Approach (No Distortions)
```
Dataset: 2000 pristine CIFAR-10 images
Quality scores: Synthetic (based on statistics)
Result: SRCC 0.98, but all in one cluster ❌
```

### New Approach (With Distortions)
```
Dataset: 500 references × 6 versions = 3000 images
  - 500 pristine (Q=100)
  - 2500 distorted (Q=20-100, 8 types)
Quality scores: Based on distortion type and level
Expected result: SRCC 0.85-0.95, 8 active clusters ✅
```

## Testing and Visualization

After training, use the updated test script:

```python
# Upload train_with_distortions.py, distorted_dataset.py to Colab
# Train the model
!python train_with_distortions.py --experiment_name resnet18_distorted

# Test and visualize
!python test_with_viz.py \
    --experiment experiments/resnet18_distorted \
    --skip_tsne

# Display results
%run display_viz_english.py
```

Now your cluster distribution plot should show **multiple clusters**, each capturing different distortion characteristics!

## Advanced: Custom Distortions

You can add more distortion types in `distorted_dataset.py`:

```python
# Add to DISTORTION_TYPES list:
DISTORTION_TYPES = [
    # ... existing types ...
    'rain',              # Rain effect
    'fog',               # Fog/haze
    'lens_distortion',   # Barrel/pincushion distortion
    'chromatic_aberration',  # Color fringing
]

# Implement in apply_distortion():
elif distortion_type == 'rain':
    # Your implementation here
    pass
```

## Troubleshooting

**Q: Still seeing mostly one cluster?**
- Increase `--cluster_loss_weight` to 1.0 or higher
- Decrease `--refit_interval` to 5 (more frequent refitting)
- Use more diverse distortions (`--distortions_per_image 8`)

**Q: Lower SRCC than before?**
- This is expected! More diverse/difficult data = slightly lower SRCC
- But the model is now learning real quality assessment, not just fitting statistics
- SRCC 0.85-0.92 with distortions is better than 0.98 without distortions

**Q: Training too slow?**
- Reduce `--train_samples` to 300-400
- Reduce `--distortions_per_image` to 3-4
- Use smaller backbone (`resnet18` instead of `resnet50`)

## Next Steps

1. ✅ Visualize distortion examples
2. ✅ Train with distorted dataset
3. ✅ Verify multiple clusters are active
4. ✅ Analyze which clusters capture which distortions
5. 📊 Compare performance across different distortion types
6. 🔬 Experiment with custom distortion combinations

---

Now you have a **real image quality assessment system** that learns from actual quality degradations, not just image statistics! 🎉
