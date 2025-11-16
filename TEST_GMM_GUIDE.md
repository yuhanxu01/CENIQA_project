# GMM 训练指南 / GMM Training Guide

## 在 Google Colab 中运行 / Running in Google Colab

**重要提示**: 在 Colab 中运行 shell 命令需要在命令前加 `!` 前缀

### 步骤 1: 训练简单基线（无 GMM）/ Step 1: Train Simple Baseline (No GMM)

在 Colab 单元格中运行：

```python
!python train_simple_high_res.py \
  --dataset stl10 \
  --experiment_name stl10_simple_baseline \
  --epochs 50
```

**参数说明 / Parameters:**
- `--dataset stl10`: 使用 STL10 数据集
- `--experiment_name stl10_simple_baseline`: 实验名称
- `--epochs 50`: 训练 50 个 epoch

---

### 步骤 2: 训练 GMM 模型 / Step 2: Train GMM Model

在 Colab 单元格中运行：

```python
!python train_high_res.py \
  --dataset stl10 \
  --n_clusters 5 \
  --experiment_name stl10_gmm_model \
  --epochs 50
```

**参数说明 / Parameters:**
- `--dataset stl10`: 使用 STL10 数据集
- `--n_clusters 5`: GMM 聚类数量设为 5
- `--experiment_name stl10_gmm_model`: 实验名称
- `--epochs 50`: 训练 50 个 epoch

---

## 在本地终端运行 / Running in Local Terminal

如果在本地终端运行，**不需要** `!` 前缀：

### 训练简单基线 / Train Simple Baseline
```bash
python train_simple_high_res.py \
  --dataset stl10 \
  --experiment_name stl10_simple_baseline \
  --epochs 50
```

### 训练 GMM 模型 / Train GMM Model
```bash
python train_high_res.py \
  --dataset stl10 \
  --n_clusters 5 \
  --experiment_name stl10_gmm_model \
  --epochs 50
```

---

## 常见问题 / FAQ

### Q1: 为什么在 Colab 中出现 SyntaxError？
**A**: 因为没有加 `!` 前缀。Colab 默认是 Python 环境，运行 shell 命令需要 `!`。

### Q2: 如何查看训练结果？
**A**: 训练完成后，结果会保存在 `experiments/` 目录下对应的实验名称文件夹中。

### Q3: 可以修改哪些参数？
**A**:
- `--epochs`: 训练轮数（默认 50）
- `--n_clusters`: GMM 聚类数（仅 GMM 模型，推荐 3-10）
- `--batch_size`: 批次大小（默认 32）
- `--learning_rate`: 学习率（默认 0.001）

---

## 其他数据集 / Other Datasets

### CIFAR-10
```python
# Colab
!python train_simple_high_res.py --dataset cifar10 --experiment_name cifar10_baseline --epochs 50
!python train_high_res.py --dataset cifar10 --n_clusters 5 --experiment_name cifar10_gmm --epochs 50
```

### ImageNet 子集 / ImageNet Subset
```python
# Colab
!python train_simple_high_res.py --dataset imagenet --experiment_name imagenet_baseline --epochs 50
!python train_high_res.py --dataset imagenet --n_clusters 5 --experiment_name imagenet_gmm --epochs 50
```
