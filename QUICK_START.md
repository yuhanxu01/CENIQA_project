# 快速开始指南

## Colab/GPU环境快速部署

### 第一步：上传文件

上传以下文件到Colab或你的GPU服务器：

**必需文件**（7个）：
1. `backbones.py`
2. `gmm_module.py`
3. `regressors.py`
4. `config_experiments.py`
5. `train_gpu.py`
6. `test_with_viz.py`
7. `visualize.py`

### 第二步：安装依赖

```python
# 在Colab中运行
!pip install timm datasets tqdm matplotlib seaborn -q
```

### 第三步：生成配置文件

```python
!python config_experiments.py
```

这将生成8个实验配置文件在 `configs/` 目录。

### 第四步：选择实验并训练

#### 快速测试（5-10分钟）
```python
!python train_gpu.py --config configs/quick_test.json
```

#### ResNet18完整训练（~50分钟，GPU）
```python
!python train_gpu.py --config configs/resnet18_gmm_mlp.json
```

#### ResNet50大模型（~100分钟，GPU）
```python
!python train_gpu.py --config configs/resnet50_gmm_mlp.json
```

#### Vision Transformer（~75分钟，GPU）
```python
!python train_gpu.py --config configs/vit_gmm_mlp.json
```

### 第五步：测试和可视化

```python
# 快速测试（跳过t-SNE）
!python test_with_viz.py \
    --experiment experiments/resnet18_gmm_mlp \
    --skip_tsne

# 完整测试（包含t-SNE）
!python test_with_viz.py \
    --experiment experiments/resnet18_gmm_mlp
```

### 第六步：查看结果

```python
# 显示训练指标
import json
with open('experiments/resnet18_gmm_mlp/visualizations/test_results.json', 'r') as f:
    results = json.load(f)
print(results)

# 显示可视化图片
from IPython.display import Image, display

print("预测散点图：")
display(Image('experiments/resnet18_gmm_mlp/visualizations/predictions_scatter.png'))

print("\n聚类分布：")
display(Image('experiments/resnet18_gmm_mlp/visualizations/cluster_distribution.png'))

print("\nPCA可视化：")
display(Image('experiments/resnet18_gmm_mlp/visualizations/features_pca.png'))

print("\n样本预测：")
display(Image('experiments/resnet18_gmm_mlp/visualizations/sample_predictions.png'))

print("\n训练曲线：")
display(Image('experiments/resnet18_gmm_mlp/visualizations/training_curves.png'))
```

## 修改配置参数

### 方法1：直接修改JSON文件

生成配置后，直接编辑 `configs/resnet18_gmm_mlp.json`：

```json
{
    "model": {
        "n_clusters": 12,        // 改变聚类数
        "hidden_dim": 1024       // 增大隐藏层
    },
    "training": {
        "epochs": 100,           // 增加训练轮数
        "learning_rate": 0.0005  // 调整学习率
    },
    "data": {
        "train_samples": 5000    // 增加训练样本
    }
}
```

### 方法2：Python代码创建配置

```python
import json

config = {
    "experiment_name": "my_custom_experiment",
    "device": "cuda",
    "seed": 42,
    "data": {
        "dataset": "cifar10",
        "train_samples": 3000,
        "val_samples": 500,
        "test_samples": 500,
        "batch_size": 256,
        "num_workers": 4
    },
    "model": {
        "backbone": "resnet34",
        "feature_dim": 512,
        "n_clusters": 10,
        "hidden_dim": 768,
        "dropout": 0.3,
        "freeze_backbone": False
    },
    "training": {
        "epochs": 80,
        "learning_rate": 0.0008,
        "weight_decay": 0.0001,
        "optimizer": "adam",
        "scheduler": "reduce_on_plateau",
        "scheduler_params": {
            "mode": "max",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6
        }
    },
    "loss_weights": {
        "quality_loss": 1.0,
        "cluster_loss": 0.1
    },
    "logging": {
        "save_dir": "./experiments",
        "log_interval": 10,
        "save_best": True,
        "save_last": True
    },
    "visualization": {
        "enabled": True,
        "plot_clusters": True,
        "plot_predictions": True,
        "save_plots": True
    }
}

# 保存配置
with open('configs/my_custom_experiment.json', 'w') as f:
    json.dump(config, f, indent=4)

# 运行训练
!python train_gpu.py --config configs/my_custom_experiment.json
```

## 并行运行多个实验

```python
# 按顺序运行多个实验
experiments = [
    "configs/resnet18_gmm_mlp.json",
    "configs/resnet50_gmm_mlp.json",
    "configs/efficientnet_gmm_mlp.json"
]

for config in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {config}")
    print('='*60)
    !python train_gpu.py --config {config}
```

## 可视化对比多个实验

```python
import json
import matplotlib.pyplot as plt

experiments = [
    'experiments/resnet18_gmm_mlp',
    'experiments/resnet50_gmm_mlp',
    'experiments/efficientnet_gmm_mlp'
]

results = {}
for exp in experiments:
    with open(f'{exp}/visualizations/test_results.json', 'r') as f:
        results[exp.split('/')[-1]] = json.load(f)

# 对比SRCC
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
srcc_values = [results[name]['srcc'] for name in names]

ax.bar(names, srcc_values, color='steelblue', alpha=0.7)
ax.set_ylabel('SRCC', fontsize=12)
ax.set_title('Model Comparison - SRCC', fontsize=14)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(srcc_values):
    ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## 常用配置组合

### 高精度配置（ResNet50 + 更多聚类）
```json
{
    "model": {
        "backbone": "resnet50",
        "n_clusters": 16,
        "hidden_dim": 1024
    },
    "training": {
        "epochs": 100,
        "learning_rate": 0.0005
    },
    "data": {
        "train_samples": 5000
    }
}
```

### 快速训练配置（冻结backbone）
```json
{
    "model": {
        "freeze_backbone": true
    },
    "training": {
        "epochs": 30,
        "learning_rate": 0.005
    },
    "data": {
        "train_samples": 1000
    }
}
```

### 内存友好配置（小batch）
```json
{
    "data": {
        "batch_size": 64
    },
    "model": {
        "backbone": "resnet18",
        "hidden_dim": 256
    }
}
```

## 调试技巧

### 打印模型结构
```python
from train_gpu import SimpleCNNGMMMLPModel

model = SimpleCNNGMMMLPModel(
    backbone_name='resnet18',
    feature_dim=512,
    n_clusters=8
)

print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 检查数据加载
```python
from train_gpu import HuggingFaceImageDataset

dataset = HuggingFaceImageDataset(split='train', max_samples=100)
print(f"Dataset size: {len(dataset)}")

img, score = dataset[0]
print(f"Image shape: {img.shape}")
print(f"Score: {score}")
```

### 单步测试
```python
import torch

model = SimpleCNNGMMMLPModel()
model.eval()

# 随机输入
x = torch.randn(4, 3, 224, 224)

with torch.no_grad():
    outputs = model(x, return_all=True)
    print(f"Quality scores: {outputs['quality_score']}")
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Posteriors shape: {outputs['posteriors'].shape}")
```

## 输出文件说明

训练完成后会生成以下文件：

```
experiments/resnet18_gmm_mlp/
├── config.json                 # 实验配置
├── best_model.pth              # 最佳模型（136MB）
├── last_model.pth              # 最后epoch模型
├── training_history.json       # 训练历史数据
└── visualizations/
    ├── test_results.json       # 测试指标
    ├── predictions_scatter.png # 预测散点图
    ├── cluster_distribution.png # 聚类分布
    ├── features_pca.png        # PCA可视化
    ├── features_tsne.png       # t-SNE可视化
    ├── sample_predictions.png  # 样本预测
    └── training_curves.png     # 训练曲线
```

## 下一步

1. 尝试不同的backbone（ResNet18/50, EfficientNet, ViT）
2. 调整聚类数量（4, 8, 12, 16）
3. 实验不同的训练策略（冻结/不冻结backbone）
4. 使用真实的IQA数据集替换CIFAR-10

详细文档请参考：`GPU_EXPERIMENTS_README.md`
