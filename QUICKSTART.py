"""
CENIQA Project Quick Start Guide
=================================

Project Structure:
------------------
ceniqa_project/
├── config.py              # ModelConfig dataclass
├── backbones.py           # CNN, ViT, Swin, UNet
├── extractors.py          # Multi-scale, Frequency, Patch-wise features
├── gmm_module.py          # Differentiable Gaussian Mixture Model
├── regressors.py          # MLP, KAN, Transformer, GRU, Attention
├── model.py               # Main CENIQA model
├── losses.py              # Combined training loss
├── dataset.py             # IQADataset with augmentation
├── train_utils.py         # Trainer class
├── train.py               # Main training script
├── inference.py           # Inference wrapper
├── experiments.py         # Experiment configurations
├── requirements.txt       # Dependencies
├── configs/default.json   # Example config
└── README.md             # Full documentation


Quick Start:
-----------

1. Install dependencies:
   $ pip install -r requirements.txt

2. Prepare data (CSV with image_path and mos columns):
   data/
   ├── train.csv
   ├── val.csv
   ├── test.csv
   └── images/
       ├── img1.jpg
       ├── img2.jpg
       └── ...

3. Generate experiment configs:
   $ python experiments.py

4. Train baseline model:
   $ python train.py --config configs/default.json \
                     --data_root ./data \
                     --train_csv ./data/train.csv \
                     --val_csv ./data/val.csv \
                     --test_csv ./data/test.csv

5. Train specific experiment:
   $ python train.py --config configs/baseline_vit.json ...

6. Inference on single image:
   $ python inference.py --checkpoint ./experiments/exp_name/best_model.pth \
                         --config ./experiments/exp_name/config.json \
                         --image ./test_image.jpg

7. Batch inference:
   $ python inference.py --checkpoint ./experiments/exp_name/best_model.pth \
                         --config ./experiments/exp_name/config.json \
                         --image ./test_images/


Key Configuration Parameters:
----------------------------

Backbone choices:
  - resnet50, efficientnet_b0
  - vit_small_patch16_224
  - swin_tiny_patch4_window7_224
  - unet

Regressor choices:
  - mlp (MLP with monotonic constraints)
  - kan (Kolmogorov-Arnold Network)
  - transformer (Transformer decoder)
  - gru (GRU-based)
  - attention (Attention-based)

GMM clusters:
  - n_clusters: 4, 6, 8, 10, 12
  - use_bic_selection: auto select K
  - gmm_update_freq: update every N epochs

Features:
  - use_multi_scale: extract at 3 scales (1.0, 0.75, 0.5)
  - use_frequency_features: DCT-based frequency domain
  - use_monotonic: enforce quality monotonicity


Available Experiments:
---------------------
baseline_vit, baseline_resnet, baseline_swin, baseline_unet
regressor_kan, regressor_transformer, regressor_gru, regressor_attention
clusters_4, clusters_6, clusters_10, clusters_12
no_multiscale, no_frequency, no_monotonic


Module Responsibilities:
------------------------
config.py       → Configuration management
backbones.py    → Feature extraction backbones
extractors.py   → Multi-scale and frequency features
gmm_module.py   → Clustering distortions
regressors.py   → Quality prediction heads
model.py        → Main model combining all
losses.py       → Training loss functions
dataset.py      → Data loading and augmentation
train_utils.py  → Training and validation utilities
train.py        → Training script entry point
inference.py    → Inference and batch prediction
experiments.py  → Experiment configurations


Training Loss Components:
------------------------
- Quality Loss (MSE): predict correct MOS score
- Contrastive Loss: same distortion closer, different further
- Consistency Loss: predict correct distortion type
- Ranking Loss: correct ranking of distortion levels
- Cluster Loss: compact and separated clusters
- Monotonic Loss: enforce monotonic quality prediction


Evaluation Metrics:
-------------------
- SRCC: Spearman Rank Correlation Coefficient
- PLCC: Pearson Linear Correlation Coefficient
- RMSE: Root Mean Square Error


Tips:
-----
1. Start with default config and baseline_vit backbone
2. Adjust batch_size and learning_rate based on GPU memory
3. Use cosine annealing scheduler for best convergence
4. Update GMM every 10 epochs for stable clustering
5. Monitor validation SRCC for early stopping
6. Save best model based on SRCC metric
7. Try different regressors after finding best backbone
8. Use experiment configs for systematic ablations

"""

if __name__ == '__main__':
    print(__doc__)
