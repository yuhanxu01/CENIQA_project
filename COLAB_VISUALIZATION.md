# Google Colab 可视化完整指南

## 问题：可视化命令失败

如果你看到这些错误：
- `ModuleNotFoundError: No module named 'torch'`
- `experiments/resnet18_large_10k/: No such file or directory`

**原因**：你在一个环境中训练，试图在另一个环境中可视化

## ✅ 解决方案：在同一个环境中运行

### 完整的 Colab Workflow

```python
# ============================================
# 完整的训练和可视化流程（在Google Colab中）
# ============================================

# 1. 克隆仓库（如果还没有）
!git clone https://github.com/yuhanxu01/CENIQA_project.git
%cd CENIQA_project

# 2. 切换到正确的分支
!git checkout claude/resnet18-distorted-images-training-011CUrFBWVpjMy2D1UaHbtMx

# 3. 安装依赖（Colab通常已经有了）
!pip install datasets scikit-learn

# 4. 检查GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 5. 训练模型（如果还没训练）
!python train_with_distortions.py \
  --experiment_name resnet18_large_10k \
  --backbone resnet18 \
  --train_samples 1666 \
  --val_samples 166 \
  --epochs 50 \
  --batch_size 64

# 6. 检查训练结果
!ls -lh experiments/resnet18_large_10k/

# 7. 运行可视化
!python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth \
  --num_images 25 \
  --test_samples 500

# 8. 显示生成的图像
from IPython.display import Image, display
import os

viz_dir = 'experiments/resnet18_large_10k'
images = [
    'enhanced_visualization.png',
    'cluster_analysis.png',
    'prediction_scatter.png'
]

for img in images:
    img_path = os.path.join(viz_dir, img)
    if os.path.exists(img_path):
        print(f"\n{'='*50}")
        print(f"📊 {img}")
        print('='*50)
        display(Image(img_path))
    else:
        print(f"⚠️  {img} not found")

# 9. 查看训练历史
import json
with open('experiments/resnet18_large_10k/training_history.json', 'r') as f:
    history = json.load(f)

# 显示最佳结果
import pandas as pd
df = pd.DataFrame(history)
print("\n" + "="*50)
print("📈 Training Summary")
print("="*50)
print(f"Best SRCC: {df['srcc'].max():.4f} (Epoch {df['srcc'].idxmax() + 1})")
print(f"Best PLCC: {df['plcc'].max():.4f} (Epoch {df['plcc'].idxmax() + 1})")
print(f"Final SRCC: {df['srcc'].iloc[-1]:.4f}")
print(f"Final PLCC: {df['plcc'].iloc[-1]:.4f}")

# 绘制训练曲线
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# SRCC
axes[0, 0].plot(df['epoch'], df['srcc'], marker='o', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('SRCC')
axes[0, 0].set_title('Spearman Correlation')
axes[0, 0].grid(True, alpha=0.3)

# PLCC
axes[0, 1].plot(df['epoch'], df['plcc'], marker='o', linewidth=2, color='orange')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('PLCC')
axes[0, 1].set_title('Pearson Correlation')
axes[0, 1].grid(True, alpha=0.3)

# Loss
axes[1, 0].plot(df['epoch'], df['train_loss'], label='Train', marker='o', linewidth=2)
axes[1, 0].plot(df['epoch'], df['val_loss'], label='Val', marker='s', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Training & Validation Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(df['epoch'], df['lr'], marker='o', linewidth=2, color='green')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/resnet18_large_10k/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ All visualizations complete!")
```

## 🎯 快速调试

### 检查当前状态

```python
# 检查是否在正确的目录
!pwd

# 检查实验目录
!ls -la experiments/ 2>/dev/null || echo "No experiments directory"

# 检查特定实验
!ls -la experiments/resnet18_large_10k/ 2>/dev/null || echo "Experiment not found - need to train first"

# 检查Python包
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("❌ PyTorch not installed")

try:
    import numpy
    print(f"✅ NumPy: {numpy.__version__}")
except ImportError:
    print("❌ NumPy not installed")
```

### 如果训练已完成但可视化失败

```python
# 1. 确认文件存在
import os
exp_dir = 'experiments/resnet18_large_10k'

print("Checking files...")
required_files = [
    'best_model.pth',
    'config.json',
    'training_history.json'
]

for f in required_files:
    path = os.path.join(exp_dir, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "✅" if exists else "❌"
    print(f"{status} {f}: {size/1024/1024:.2f} MB" if size > 0 else f"{status} {f}: missing")

# 2. 手动运行可视化
!python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth \
  --num_images 25 \
  --test_samples 500 \
  --device cuda

# 3. 如果还是失败，查看完整错误
!python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth \
  --num_images 25 \
  --test_samples 500 2>&1 | head -50
```

## 📥 下载结果到本地

如果想在本地查看：

```python
# 在Colab中：打包实验结果
!zip -r resnet18_large_10k_results.zip experiments/resnet18_large_10k/

# 下载
from google.colab import files
files.download('resnet18_large_10k_results.zip')

# 在本地：解压并查看
# unzip resnet18_large_10k_results.zip
# open experiments/resnet18_large_10k/*.png
```

## 🐛 常见错误和解决方案

### 错误 1: ModuleNotFoundError: No module named 'torch'

**原因**：环境不匹配

**解决**：在训练的同一个环境中运行可视化

### 错误 2: experiments/resnet18_large_10k/: No such file or directory

**原因**：还没训练模型

**解决**：
```python
# 运行训练
!python train_with_distortions.py \
  --experiment_name resnet18_large_10k \
  --train_samples 1666 \
  --epochs 50
```

### 错误 3: RuntimeError: CUDA out of memory

**原因**：GPU内存不足

**解决**：
```python
# 减少batch size和测试样本
!python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth \
  --num_images 16 \
  --test_samples 200
```

### 错误 4: FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pth'

**原因**：检查点文件名不对

**解决**：
```python
# 检查可用的检查点
!ls experiments/resnet18_large_10k/*.pth

# 使用正确的文件名
# 如果有 last_model.pth：
!python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint last_model.pth \
  --num_images 25 \
  --test_samples 500
```

## 📊 完整示例输出

成功运行后你会看到：

```
Loading model from: experiments/resnet18_large_10k/best_model.pth
Model configuration:
  - Backbone: resnet18
  - Feature dim: 512
  - Clusters: 8
  - Hidden dim: 512

Loading test dataset...
Processing 500 images...
Loading test: 100% 500/500 [00:00<00:00, 1882.24it/s]

Running inference...
Inference: 100% 4/4 [00:02<00:00,  1.54it/s]

Performance metrics:
  SRCC: 0.7842
  PLCC: 0.7956
  RMSE: 0.1234

Generating visualizations...
✅ Saved: experiments/resnet18_large_10k/enhanced_visualization.png
✅ Saved: experiments/resnet18_large_10k/cluster_analysis.png
✅ Saved: experiments/resnet18_large_10k/prediction_scatter.png

Done!
```

## 🎨 查看可视化结果

```python
from IPython.display import Image, display

# 主可视化
display(Image('experiments/resnet18_large_10k/enhanced_visualization.png'))

# 聚类分析
display(Image('experiments/resnet18_large_10k/cluster_analysis.png'))

# 预测散点图
display(Image('experiments/resnet18_large_10k/prediction_scatter.png'))
```

## 💡 提示

1. **在同一个环境中运行所有步骤**（训练 + 可视化）
2. **使用Colab的持久化存储**（Google Drive）保存结果
3. **定期保存检查点**以防会话断开
4. **先用小数据集测试**（train_samples=500, epochs=10）

## 📚 相关文档

- `TRAINING_IMPROVEMENTS.md` - 训练改进说明
- `RUN_TRAINING_AND_VIZ.md` - 完整训练流程
- `VISUALIZATION_GUIDE_CN.md` - 可视化详细指南
