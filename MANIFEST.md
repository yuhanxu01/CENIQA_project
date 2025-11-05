# CENIQA Project - Complete File Manifest

## Package Contents (16 Python files + 4 docs)

### Core Architecture (4 files)
```
config.py              (120 lines)   - Configuration management
                                      ModelConfig, load/save functions
model.py              (140 lines)   - Main CENIQA model
                                      Combines backbone + GMM + regressor
losses.py             (150 lines)   - Training loss functions
                                      Quality, contrastive, consistency, ranking, cluster losses
backbones.py          (180 lines)   - Feature extraction backbones
                                      CNN, ViT, Swin, UNet implementations
```

### Feature Extractors (2 files)
```
extractors.py         (130 lines)   - Feature extraction modules
                                      Multi-scale, frequency, patch-wise
gmm_module.py         (120 lines)   - Gaussian Mixture Model
                                      Differentiable GMM for clustering
```

### Regressors (1 file)
```
regressors.py         (250 lines)   - Quality prediction heads
                                      MLP, KAN, Transformer, GRU, Attention
```

### Data & Training (4 files)
```
dataset.py            (160 lines)   - Dataset and transforms
                                      IQADataset with distortion augmentation
train_utils.py        (220 lines)   - Training utilities
                                      Trainer class, validation, checkpointing
train.py              (180 lines)   - Main training script
                                      Entry point with CLI arguments
experiments.py        (160 lines)   - Experiment configurations
                                      16 predefined experiment configs
```

### Inference (1 file)
```
inference.py          (180 lines)   - Inference wrapper
                                      Single/batch prediction, cluster info
```

### Documentation (4 files)
```
README.md             (220 lines)   - Full documentation
                                      Usage, installation, parameters
QUICKSTART.py         (120 lines)   - Quick start guide
                                      Key concepts and commands
PROJECT_STRUCTURE.txt (300 lines)   - Detailed structure documentation
                                      File descriptions, dataflow, workflow
MANIFEST.md           (This file)   - File inventory and statistics
```

### Configuration (2 items)
```
configs/
├── default.json               - Example configuration
requirements.txt              - Package dependencies
```

## Statistics

**Total Python Code**: ~1,800 lines (excluding documentation)
**Total Documentation**: ~640 lines
**Total Files**: 20

### Breakdown by Category:
- **Architecture**: 550 lines (30%)
- **Training**: 560 lines (31%)
- **Data Handling**: 160 lines (9%)
- **Inference**: 180 lines (10%)
- **Config**: 120 lines (7%)
- **Documentation**: 640 lines (13%)

## Key Features Implemented

### Backbones ✓
- ResNet (CNN-based)
- EfficientNet (CNN-based)
- Vision Transformer (ViT)
- Swin Transformer
- Custom UNet
- Factory pattern for easy swapping

### Feature Extraction ✓
- Multi-scale features (3 scales)
- Frequency domain (DCT-based)
- Patch-wise extraction
- Configurable combinations

### Clustering ✓
- Differentiable Gaussian Mixture Model
- Learnable parameters (means, variances, weights)
- sklearn GMM initialization
- BIC-based cluster selection

### Regressors ✓
- Monotonic MLP (enforced positivity)
- Kolmogorov-Arnold Network (KAN)
- Transformer-based decoder
- GRU-based sequential
- Attention-based weighting

### Loss Functions ✓
- Quality prediction (MSE)
- Contrastive learning
- Distortion consistency
- Pairwise ranking
- Cluster compactness
- Monotonic regularization

### Training Features ✓
- Three-stage training strategy
- Configurable loss weights
- Learning rate scheduling (cosine, onecycle)
- Gradient clipping
- Checkpoint management
- Validation metrics (SRCC, PLCC, RMSE)

### Data Processing ✓
- IQADataset with augmentation
- 10 distortion types
- Synthetic distortion application
- Multi-crop training
- Color jittering augmentation

### Experiments ✓
- 16 predefined experiment configurations
- Backbone ablations (4 variants)
- Regressor ablations (4 variants)
- GMM cluster ablations (4 variants)
- Component ablations (3 variants)

### Inference ✓
- Single image prediction
- Batch prediction
- Cluster assignment retrieval
- Posterior probability output
- Command-line interface

## How to Use

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Generate Configs
```bash
python experiments.py
```

### 3. Training
```bash
python train.py \
  --config configs/baseline_vit.json \
  --data_root ./data \
  --train_csv ./data/train.csv \
  --val_csv ./data/val.csv \
  --test_csv ./data/test.csv
```

### 4. Inference
```bash
python inference.py \
  --checkpoint ./experiments/exp_name/best_model.pth \
  --config ./experiments/exp_name/config.json \
  --image ./test_image.jpg
```

## Dependencies

**Core (required)**:
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0

**Models**:
- timm >= 0.9.0
- einops >= 0.6.0

**Data/ML**:
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- Pillow >= 8.3.0

**Optional**:
- wandb >= 0.12.0 (monitoring)

## Data Format

**CSV files** (train.csv, val.csv, test.csv):
```csv
image_path,mos
img001.jpg,85.5
img002.jpg,72.3
...
```

Where:
- `image_path`: relative path to image file
- `mos`: Mean Opinion Score (0-100)

## Output Structure

```
experiments/
└── {experiment_name}/
    ├── config.json              - Saved configuration
    ├── best_model.pth           - Best model checkpoint
    ├── checkpoint_epoch_*.pth   - Periodic checkpoints
    └── test_results.txt         - Final test metrics
```

## Configuration Parameters

### Model Parameters
- `backbone`: resnet50, efficientnet_b0, vit_small_patch16_224, swin_tiny_patch4_window7_224, unet
- `feature_dim`: 768 (feature dimensionality)
- `hidden_dim`: 512 (hidden layer dimensionality)
- `use_multi_scale`: true/false
- `use_frequency_features`: true/false

### GMM Parameters
- `n_clusters`: 4-12 (number of mixture components)
- `use_bic_selection`: true/false (auto select K)
- `gmm_update_freq`: 10 (update every N epochs)

### Regressor Parameters
- `regressor_type`: mlp, kan, transformer, gru, attention
- `use_monotonic`: true/false (enforce monotonicity)
- `dropout_rate`: 0.2 (dropout probability)

### Training Parameters
- `epochs`: 100 (total epochs)
- `batch_size`: 32
- `learning_rate`: 1e-4
- `scheduler`: cosine, onecycle
- `weight_decay`: 1e-5

### Loss Weights
- `lambda_quality`: 1.0 (MSE loss)
- `lambda_contrast`: 1.0 (contrastive loss)
- `lambda_consistency`: 0.5 (distortion classification)
- `lambda_ranking`: 0.5 (ranking loss)
- `lambda_cluster`: 0.3 (cluster consistency)
- `lambda_monotonic`: 0.1 (monotonic constraint)

## Experiments Included

### Backbone Comparisons
- `baseline_vit`: Vision Transformer (main baseline)
- `baseline_resnet`: ResNet50
- `baseline_swin`: Swin Transformer
- `baseline_unet`: Custom UNet

### Regressor Comparisons
- `regressor_kan`: Kolmogorov-Arnold Network
- `regressor_transformer`: Transformer decoder
- `regressor_gru`: GRU-based
- `regressor_attention`: Attention-based

### Clustering Comparisons
- `clusters_4`: K=4 clusters
- `clusters_6`: K=6 clusters
- `clusters_10`: K=10 clusters
- `clusters_12`: K=12 clusters

### Component Ablations
- `no_multiscale`: disable multi-scale features
- `no_frequency`: disable frequency features
- `no_monotonic`: disable monotonic constraint

## Performance Metrics

Evaluation metrics:
- **SRCC**: Spearman Rank Correlation Coefficient (0-1, higher better)
- **PLCC**: Pearson Linear Correlation Coefficient (0-1, higher better)
- **RMSE**: Root Mean Square Error (lower better)

## Notes

1. **First Run**: Will download pretrained backbone weights automatically
2. **GPU Memory**: Adjust batch_size if OOM errors occur
3. **Data Preparation**: Ensure CSV files have correct format
4. **Path Names**: Use relative paths or absolute paths, not mixed
5. **Random Seed**: Set for reproducibility across runs

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review PROJECT_STRUCTURE.txt for architecture details
3. Check QUICKSTART.py for quick reference
4. Run `python experiments.py` to see available configurations

