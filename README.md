# CENIQA: Cluster-Enhanced Image Quality Assessment

A PyTorch implementation of the CENIQA model for image quality assessment using Gaussian Mixture Models and multiple backbone architectures.

## Project Structure

```
ceniqa_project/
├── config.py              # Configuration management
├── backbones.py           # Backbone models (CNN, ViT, Swin, UNet)
├── extractors.py          # Feature extractors (multi-scale, frequency)
├── gmm_module.py          # Gaussian Mixture Model
├── regressors.py          # Regression heads (MLP, KAN, Transformer, GRU, Attention)
├── model.py               # Main CENIQA model
├── losses.py              # Loss functions
├── dataset.py             # Dataset class and transforms
├── train_utils.py         # Training utilities
├── train.py               # Main training script
├── inference.py           # Inference script
├── experiments.py         # Experiment configurations
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Installation

```bash
# Clone repo and install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

Create CSV files with image paths and MOS scores:
```
image_path,mos
img1.jpg,85.5
img2.jpg,72.3
...
```

### 2. Training

Basic training:
```bash
python train.py \
  --config configs/default.json \
  --data_root ./data \
  --train_csv ./data/train.csv \
  --val_csv ./data/val.csv \
  --test_csv ./data/test.csv
```

With custom parameters:
```bash
python train.py \
  --backbone vit_small_patch16_224 \
  --regressor mlp \
  --epochs 100 \
  --batch_size 32
```

### 3. Inference

Single image:
```bash
python inference.py \
  --checkpoint ./experiments/baseline_vit/best_model.pth \
  --config ./experiments/baseline_vit/config.json \
  --image ./test_image.jpg
```

Batch inference:
```bash
python inference.py \
  --checkpoint ./experiments/baseline_vit/best_model.pth \
  --config ./experiments/baseline_vit/config.json \
  --image ./test_images/
```

### 4. Experiments

Generate experiment configs:
```bash
python experiments.py
```

This creates JSON configs in `configs/` directory.

Run specific experiment:
```bash
python train.py --config configs/baseline_vit.json
```

## Model Components

### Backbones
- **CNNBackbone**: ResNet, EfficientNet
- **ViTBackbone**: Vision Transformer
- **SwinBackbone**: Swin Transformer
- **UNetBackbone**: Custom UNet architecture

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
- **KANRegressor**: Kolmogorov-Arnold Network
- **TransformerRegressor**: Transformer-based regressor
- **GRURegressor**: GRU-based regressor
- **AttentionRegressor**: Attention-based regressor

## Configuration

Key parameters in `config.py`:

```python
# Backbone
backbone: str = 'vit_small_patch16_224'
feature_dim: int = 768

# GMM
n_clusters: int = 8
use_bic_selection: bool = True

# Regressor
regressor_type: str = 'mlp'  # mlp, kan, transformer, gru, attention

# Training
epochs: int = 100
learning_rate: float = 1e-4
scheduler: str = 'cosine'

# Loss weights
lambda_contrast: float = 1.0
lambda_consistency: float = 0.5
lambda_ranking: float = 0.5
lambda_cluster: float = 0.3
lambda_quality: float = 1.0
lambda_monotonic: float = 0.1
```

## Training Details

### Three-Stage Training

1. **Self-Supervised Pre-training** (20-30 epochs)
   - Train backbone + auxiliary heads
   - No MOS labels needed
   - Uses contrastive + consistency + ranking losses

2. **GMM Fitting** (offline or online)
   - Collect features from training data
   - Fit Gaussian Mixture Model
   - Select optimal cluster number via BIC

3. **End-to-End Fine-tuning** (50-70 epochs)
   - Joint training of all components
   - Use full loss combination
   - Update GMM periodically

### Loss Components

- **Quality Loss**: MSE loss on MOS prediction
- **Contrastive Loss**: Same distortion closer, different distortion further
- **Consistency Loss**: Predict correct distortion type
- **Ranking Loss**: Order predictions by distortion level
- **Cluster Loss**: Encourage compact, well-separated clusters
- **Monotonic Loss**: Ensure monotonic quality prediction

## Evaluation Metrics

- **SRCC**: Spearman Rank Correlation Coefficient
- **PLCC**: Pearson Linear Correlation Coefficient
- **RMSE**: Root Mean Square Error

## Experiments

Available experiments:

- **Backbone comparisons**: ResNet, EfficientNet, ViT, Swin, UNet
- **Regressor comparisons**: MLP, KAN, Transformer, GRU, Attention
- **Cluster number ablation**: K=4,6,8,10,12
- **Feature ablation**: Multi-scale, frequency domain, patch-wise
- **Component ablation**: Monotonic constraints, auxiliary losses

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
