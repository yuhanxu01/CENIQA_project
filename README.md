# CENIQA: Cluster-Enhanced Image Quality Assessment

A PyTorch implementation of the CENIQA model for image quality assessment using Gaussian Mixture Models and multiple backbone architectures.

## Quick Start

See [QUICK_START.md](QUICK_START.md) for a beginner-friendly guide.

See [TEST_GMM_GUIDE.md](TEST_GMM_GUIDE.md) for GMM training in Google Colab.

## Project Structure

```
ceniqa_project/
├── Core Model Components
│   ├── backbones.py              # Backbone models (ResNet, ViT, etc.)
│   ├── extractors.py             # Feature extractors
│   ├── gmm_module.py             # Gaussian Mixture Model module
│   ├── regressors.py             # Regression heads (MLP, etc.)
│   ├── model.py                  # Main CENIQA model
│   ├── simple_model.py           # Simplified model (no GMM)
│   └── losses.py                 # Loss functions
│
├── Dataset & Training
│   ├── dataset.py                # Basic dataset loader
│   ├── distorted_dataset.py      # Dataset with distortions
│   ├── high_res_distorted_dataset.py  # High-res datasets (STL10, CIFAR10)
│   ├── train.py                  # Standard training script
│   ├── train_simple_high_res.py  # Simple baseline (no GMM)
│   ├── train_high_res.py         # GMM training on high-res data
│   └── train_utils.py            # Training utilities
│
├── Inference & Visualization
│   ├── inference.py              # Model inference
│   ├── visualize.py              # Visualization tools
│   └── run_comparison_experiments.py  # Compare different models
│
├── Testing
│   ├── test_distortions.py       # Test distortion application
│   └── test_high_res_dataset.py  # Test high-res dataset loading
│
├── Configuration
│   ├── config.py                 # Configuration classes
│   └── configs/default.json      # Default configuration
│
└── Documentation
    ├── README.md                 # This file
    ├── QUICK_START.md            # Quick start guide
    ├── TEST_GMM_GUIDE.md         # GMM training guide (Colab)
    ├── DISTORTION_GUIDE.md       # Distortion types reference
    └── VISUALIZATION_GUIDE_CN.md # Visualization guide (Chinese)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yuhanxu01/CENIQA_project.git
cd CENIQA_project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training on High-Resolution Datasets (Recommended)

**Simple Baseline (No GMM):**
```bash
python train_simple_high_res.py \
  --dataset stl10 \
  --experiment_name stl10_baseline \
  --epochs 50
```

**GMM Model:**
```bash
python train_high_res.py \
  --dataset stl10 \
  --n_clusters 5 \
  --experiment_name stl10_gmm \
  --epochs 50
```

Supported datasets: `stl10`, `cifar10`, `imagenet-1k`

**For Colab users:** Add `!` prefix. See [TEST_GMM_GUIDE.md](TEST_GMM_GUIDE.md) for details.

### Training on Custom Datasets

Create CSV files with image paths and MOS scores:
```
image_path,mos
img1.jpg,85.5
img2.jpg,72.3
...
```

Train the model:
```bash
python train.py \
  --config configs/default.json \
  --data_root ./data \
  --train_csv ./data/train.csv \
  --val_csv ./data/val.csv \
  --test_csv ./data/test.csv
```

### Inference

Single image:
```bash
python inference.py \
  --checkpoint ./experiments/baseline/best_model.pth \
  --config ./experiments/baseline/config.json \
  --image ./test_image.jpg
```

Batch inference:
```bash
python inference.py \
  --checkpoint ./experiments/baseline/best_model.pth \
  --config ./experiments/baseline/config.json \
  --image ./test_images/
```

### Comparison Experiments

Compare different model configurations:
```bash
python run_comparison_experiments.py
```

This will train multiple models and generate comparison visualizations.

## Model Components

### Backbones
- **ResNet**: ResNet18, ResNet34, ResNet50
- **EfficientNet**: EfficientNet-B0 to B7
- **Vision Transformer**: ViT-Small, ViT-Base
- **Swin Transformer**: Swin-Tiny, Swin-Small

### Feature Extractors
- **MultiScaleFeatureExtractor**: Extract features at multiple resolutions
- **FrequencyFeatureExtractor**: DCT-based frequency domain features
- **PatchWiseQualityExtractor**: Patch-based quality assessment

### GMM Module
- **DifferentiableGMM**: Learnable Gaussian Mixture Model
- End-to-end trainable clustering
- Automatic cluster number selection via BIC

### Regression Heads
- **MonotonicMLP**: MLP with monotonic constraints
- **Standard MLP**: Basic multi-layer perceptron
- More variants available in `regressors.py`

## Key Features

### Distortion Types
The model handles 8 types of image distortions:
1. Gaussian Blur
2. Motion Blur
3. Gaussian Noise
4. JPEG Compression
5. Color Saturation
6. Contrast Change
7. Brightness Change
8. Pixelation

See [DISTORTION_GUIDE.md](DISTORTION_GUIDE.md) for details.

### High-Resolution Support
- **STL-10**: 96×96 images (5,000 train, 8,000 test)
- **CIFAR-10**: 32×32 images (50,000 train, 10,000 test)
- **ImageNet**: Variable resolution (200×200+)

All images are resized to 224×224 for training.

### Training Strategies
- **Simple Baseline**: Direct quality prediction without GMM
- **GMM-Enhanced**: Cluster-based quality assessment
- **Comparison Mode**: Automatically compare multiple configurations

## Evaluation Metrics

- **SRCC**: Spearman Rank Correlation Coefficient
- **PLCC**: Pearson Linear Correlation Coefficient
- **RMSE**: Root Mean Square Error

## Visualization

The project includes comprehensive visualization tools:
```bash
python visualize.py --experiment_dir ./experiments/baseline
```

See [VISUALIZATION_GUIDE_CN.md](VISUALIZATION_GUIDE_CN.md) for details.

## Configuration

Key parameters in `config.py`:

```python
# Model architecture
backbone: str = 'resnet18'
n_clusters: int = 5

# Training
epochs: int = 50
batch_size: int = 32
learning_rate: float = 0.001

# Dataset
dataset: str = 'stl10'  # stl10, cifar10, imagenet-1k
distortion_strength: str = 'medium'  # light, medium, heavy
```

## File Organization

### Core Files
- `model.py` - Full CENIQA model with GMM
- `simple_model.py` - Simplified baseline without GMM
- `train_simple_high_res.py` - Train simple baseline
- `train_high_res.py` - Train GMM model
- `high_res_distorted_dataset.py` - High-res dataset with distortions

### Legacy Files (for compatibility)
- `train.py` - Original training script
- `dataset.py` - Original dataset loader
- `distorted_dataset.py` - Original distorted dataset

## Tips

1. **Start simple**: Use `train_simple_high_res.py` for baseline results
2. **Then try GMM**: Use `train_high_res.py` to see if clustering helps
3. **Compare**: Use `run_comparison_experiments.py` to compare different settings
4. **Visualize**: Use `visualize.py` to understand model behavior

## Troubleshooting

**Q: Getting "Dataset doesn't exist on Hub" error?**
A: This is fixed in the latest version. Make sure you pulled the latest code.

**Q: Out of memory error?**
A: Reduce `--batch_size` or use a smaller backbone like `resnet18`.

**Q: Training is slow?**
A: Reduce `--max_samples` for quick experiments, or use fewer `--epochs`.

## Citation

If you use this code, please cite:

```bibtex
@article{ceniqa,
  title={CENIQA: Cluster-Enhanced Neural Image Quality Assessment},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License
