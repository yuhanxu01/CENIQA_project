# 高分辨率图像训练指南

## 问题分析

之前的实验中，GMM和简单模型表现差不多，主要原因是：

**CIFAR-10图像质量太差**
- 原始分辨率：32x32像素（非常小！）
- Resize到224x224后：图像非常模糊
- 在模糊图片上应用distortion：效果很差，难以区分不同质量等级

## 解决方案：使用高分辨率数据集

### 数据集对比

| 数据集 | 原始分辨率 | Resize到224 | 图像质量 | 备注 |
|--------|-----------|------------|---------|------|
| **CIFAR-10** | 32×32 | 7倍放大 | ❌ 很模糊 | 之前使用 |
| **STL-10** | 96×96 | 2.3倍放大 | ✅ 清晰 | **推荐** |
| **ImageNet** | 200×200+ | 约1倍 | ✅✅ 非常清晰 | 下载大 |

**推荐使用STL-10**：
- 比CIFAR-10清晰9倍（9倍像素数）
- 数据量适中（训练5000张，测试8000张）
- 容易下载和使用

---

## 快速开始

### 1. 测试高分辨率数据集

```bash
# 测试STL-10数据集并生成可视化对比
python test_high_res_dataset.py
```

这将生成两个图片：
- `dataset_comparison.png` - CIFAR-10 vs STL-10 清晰度对比
- `high_res_distortions_stl10_medium.png` - 所有distortion类型的效果

### 2. 训练GMM模型（高分辨率版本）

```bash
# 完整训练（约2小时）
python train_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_gmm \
  --epochs 50 \
  --batch_size 64

# 快速测试（10分钟）
python train_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_gmm_test \
  --train_samples 500 \
  --val_samples 200 \
  --epochs 10 \
  --batch_size 32
```

### 3. 训练简单基线模型（无GMM）

```bash
# 完整训练
python train_simple_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_simple \
  --epochs 50 \
  --batch_size 64

# 快速测试
python train_simple_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_simple_test \
  --train_samples 500 \
  --val_samples 200 \
  --epochs 10 \
  --batch_size 32
```

---

## Distortion强度设置

新的高分辨率数据集支持三种distortion强度：

| 强度 | Level范围 | 说明 | 适用场景 |
|------|----------|------|---------|
| **light** | 0.1-0.4 | 轻微失真 | 高质量图像评估 |
| **medium** | 0.2-0.6 | 中等失真 | **推荐，更真实** |
| **heavy** | 0.3-1.0 | 严重失真 | 极端情况测试 |

**推荐使用medium**：更接近真实场景的图像质量范围。

---

## 在Colab中运行

### 完整训练流程

```python
%cd /content/CENIQA_project

# 拉取最新代码
!git pull origin claude/resnet18-distorted-images-training-011CUrFBWVpjMy2D1UaHbtMx

# 1. 测试数据集（可选，查看清晰度对比）
!python test_high_res_dataset.py

# 显示对比图
from IPython.display import Image, display
print("CIFAR-10 vs STL-10 清晰度对比：")
display(Image('dataset_comparison.png'))
print("\nSTL-10 distortion效果：")
display(Image('high_res_distortions_stl10_medium.png'))

# 2. 训练GMM模型（高分辨率）
!python train_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_gmm \
  --epochs 50 \
  --batch_size 64

# 3. 训练简单基线（高分辨率）
!python train_simple_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_simple \
  --epochs 50 \
  --batch_size 64

# 4. 对比两个模型
!python compare_experiments.py \
  --exp_simple experiments/stl10_simple \
  --exp_gmm experiments/stl10_gmm \
  --test_samples 1000

# 显示对比结果
display(Image('experiments/comparison/comparison_dashboard.png'))
```

### 快速测试版本（10分钟）

```python
%cd /content/CENIQA_project
!git pull origin claude/resnet18-distorted-images-training-011CUrFBWVpjMy2D1UaHbtMx

# 快速训练（少量数据）
!python train_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_gmm_quick \
  --train_samples 500 --val_samples 200 \
  --epochs 10 --batch_size 32

!python train_simple_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_simple_quick \
  --train_samples 500 --val_samples 200 \
  --epochs 10 --batch_size 32

!python compare_experiments.py \
  --exp_simple experiments/stl10_simple_quick \
  --exp_gmm experiments/stl10_gmm_quick \
  --test_samples 300

from IPython.display import Image, display
display(Image('experiments/comparison/comparison_dashboard.png'))
```

---

## 预期改进

使用高分辨率数据集后，预期会看到：

### ✅ 图像质量提升
- **更清晰的原始图像**：96×96 vs 32×32
- **更明显的distortion效果**：在清晰图像上应用失真，效果更真实
- **更好的质量区分**：模型更容易学习不同质量等级

### ✅ 模型性能提升
- **更高的SRCC/PLCC**：清晰图像提供更多特征信息
- **更低的RMSE**：质量预测更准确
- **GMM聚类效果更好**：清晰图像特征更易于聚类

### ✅ GMM vs Simple差距更明显
- 在CIFAR-10上：GMM ≈ Simple（都很差）
- 在STL-10上：**预期GMM > Simple**（图像清晰后GMM优势显现）

---

## 参数说明

### 数据集参数

```bash
--dataset stl10                    # 数据集选择（stl10 或 imagenet-1k）
--distortion_strength medium       # 失真强度（light/medium/heavy）
--train_samples 500               # 训练样本数（None=全部）
--val_samples 200                 # 验证样本数（None=全部）
--distortions_per_image 5         # 每张图片生成的失真版本数
```

### 模型参数

```bash
--backbone resnet18               # CNN骨干网络
--n_clusters 5                    # GMM聚类数（仅GMM模型）
--feature_dim 512                 # 特征维度
--hidden_dim 512                  # MLP隐藏层维度
```

### 训练参数

```bash
--epochs 50                       # 训练轮数
--batch_size 64                   # 批大小
--lr 1e-4                        # 学习率

# GMM模型专用
--cluster_loss_weight 0.1        # 聚类损失权重
--balance_weight 1.0             # 均匀分布损失权重
--refit_interval 0               # GMM重新拟合间隔（0=禁用）
```

---

## 文件说明

### 新增文件

1. **high_res_distorted_dataset.py** - 高分辨率数据集类
   - 支持STL-10和ImageNet
   - 支持三种distortion强度
   - 在高分辨率上应用distortion，再resize

2. **train_high_res.py** - GMM模型训练（高分辨率）
   - 使用HighResDistortedDataset
   - 与train_with_distortions.py相同的训练逻辑

3. **train_simple_high_res.py** - 简单基线训练（高分辨率）
   - 使用HighResDistortedDataset
   - 无GMM，直接回归

4. **test_high_res_dataset.py** - 数据集测试和可视化
   - 生成CIFAR-10 vs STL-10对比
   - 生成distortion效果展示

### 保留文件（旧版本）

- `distorted_dataset.py` - CIFAR-10版本（低分辨率）
- `train_with_distortions.py` - CIFAR-10训练脚本
- `train_simple_baseline.py` - CIFAR-10简单基线

---

## 疑难解答

### 问题1：STL-10下载失败

```python
# 方案1：使用国内镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 方案2：手动下载
# 访问 https://huggingface.co/datasets/stl10
# 下载到本地后加载
```

### 问题2：显存不足

```bash
# 减小batch size
--batch_size 32  # 或 16, 8

# 减少样本数
--train_samples 500
--val_samples 200
```

### 问题3：训练太慢

```bash
# 使用快速测试模式
--train_samples 500
--val_samples 200
--epochs 10

# 或使用更小的backbone
--backbone resnet18  # 而不是resnet50
```

---

## 下一步

训练完成后：

1. **查看训练曲线**
   ```python
   import json
   import matplotlib.pyplot as plt

   with open('experiments/stl10_gmm/history.json') as f:
       history = json.load(f)

   # 绘制SRCC曲线
   plt.plot([x['srcc'] for x in history['val']])
   plt.xlabel('Epoch')
   plt.ylabel('SRCC')
   plt.title('Validation SRCC')
   plt.show()
   ```

2. **对比GMM vs Simple**
   ```bash
   python compare_experiments.py \
     --exp_simple experiments/stl10_simple \
     --exp_gmm experiments/stl10_gmm \
     --test_samples 1000
   ```

3. **可视化聚类效果**
   ```bash
   python enhanced_visualize.py \
     --checkpoint experiments/stl10_gmm/checkpoint_best.pth \
     --experiment_dir experiments/stl10_gmm \
     --test_samples 600
   ```

---

## 预期训练时间

| 配置 | 数据集大小 | Epochs | 预计时间 |
|------|-----------|--------|---------|
| 快速测试 | 500样本 | 10 | **10分钟** |
| 中等训练 | 2000样本 | 30 | 45分钟 |
| 完整训练 | 全部 | 50 | 2小时 |

*基于V100 GPU，实际时间可能有所不同*

---

## 总结

通过使用**STL-10高分辨率数据集**：
- ✅ 图像从32×32提升到96×96（9倍像素）
- ✅ Distortion效果更真实、更明显
- ✅ 模型学习更有效，性能预期提升
- ✅ GMM聚类效果预期更好
- ✅ GMM vs Simple的差距预期更明显

**立即开始**：`python test_high_res_dataset.py` 查看效果！
