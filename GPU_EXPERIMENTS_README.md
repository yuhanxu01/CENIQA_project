# GPU实验指南 - CNN+GMM+MLP图像质量评估

完整的实验框架，支持多种模型配置、GPU训练和可视化分析。

## 快速开始

### 1. 生成配置文件

```bash
python config_experiments.py
```

这将在 `configs/` 目录下生成8个配置文件：
- `resnet18_gmm_mlp.json` - ResNet18基础模型
- `resnet50_gmm_mlp.json` - ResNet50大模型
- `efficientnet_gmm_mlp.json` - EfficientNet模型
- `vit_gmm_mlp.json` - Vision Transformer
- `resnet18_4clusters.json` - 4个聚类（消融实验）
- `resnet18_16clusters.json` - 16个聚类（消融实验）
- `resnet18_frozen.json` - 冻结backbone（快速训练）
- `quick_test.json` - 快速测试配置

### 2. 训练模型

```bash
# 基础ResNet18模型
python train_gpu.py --config configs/resnet18_gmm_mlp.json

# ResNet50大模型
python train_gpu.py --config configs/resnet50_gmm_mlp.json

# Vision Transformer
python train_gpu.py --config configs/vit_gmm_mlp.json

# 快速测试
python train_gpu.py --config configs/quick_test.json
```

### 3. 测试和可视化

```bash
# 完整测试（包含t-SNE）
python test_with_viz.py --experiment experiments/resnet18_gmm_mlp

# 快速测试（跳过t-SNE）
python test_with_viz.py --experiment experiments/resnet18_gmm_mlp --skip_tsne

# 指定检查点
python test_with_viz.py --experiment experiments/resnet18_gmm_mlp --checkpoint last_model.pth
```

## 配置文件说明

每个配置文件包含以下部分：

### 数据配置
```json
"data": {
    "dataset": "cifar10",
    "train_samples": 2000,    // 训练样本数
    "val_samples": 500,       // 验证样本数
    "test_samples": 500,      // 测试样本数
    "batch_size": 128,        // GPU批大小
    "num_workers": 4          // 数据加载进程数
}
```

### 模型配置
```json
"model": {
    "backbone": "resnet18",    // backbone模型
    "feature_dim": 512,        // 特征维度
    "n_clusters": 8,           // GMM聚类数
    "hidden_dim": 512,         // MLP隐藏层维度
    "dropout": 0.3,            // Dropout率
    "freeze_backbone": false   // 是否冻结backbone
}
```

### 训练配置
```json
"training": {
    "epochs": 50,                  // 训练轮数
    "learning_rate": 0.001,        // 学习率
    "weight_decay": 0.0001,        // 权重衰减
    "optimizer": "adam",           // 优化器
    "scheduler": "reduce_on_plateau",  // 学习率调度器
    "scheduler_params": {
        "mode": "max",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 0.000001
    }
}
```

## 实验输出结构

训练后，每个实验会在 `experiments/` 目录下创建文件夹：

```
experiments/
└── resnet18_gmm_mlp/
    ├── config.json              # 实验配置
    ├── best_model.pth           # 最佳模型
    ├── last_model.pth           # 最后epoch模型
    ├── training_history.json    # 训练历史
    └── visualizations/          # 可视化结果
        ├── test_results.json
        ├── predictions_scatter.png
        ├── cluster_distribution.png
        ├── features_pca.png
        ├── features_tsne.png
        ├── sample_predictions.png
        └── training_curves.png
```

## 可视化说明

### 1. predictions_scatter.png
- 预测值 vs 真实值散点图
- 显示SRCC, PLCC, RMSE指标
- 红色虚线表示完美预测

### 2. cluster_distribution.png
- 左图：每个聚类的样本数量分布
- 右图：每个聚类的质量分数分布（箱线图）
- 红星表示聚类的平均质量分数

### 3. features_pca.png
- PCA降维后的特征可视化
- 左图：按质量分数着色
- 右图：按聚类着色

### 4. features_tsne.png
- t-SNE降维后的特征可视化
- 更好地展示特征的聚类结构
- 计算时间较长

### 5. sample_predictions.png
- 16个样本图像及其预测结果
- 显示预测值、真实值、聚类ID和误差

### 6. training_curves.png
- 训练和验证损失曲线
- SRCC和PLCC变化曲线
- 学习率变化曲线

## 自定义配置

### 方法1：修改现有配置文件

直接编辑 `configs/*.json` 文件中的参数。

### 方法2：创建新配置

在 `config_experiments.py` 中添加新函数：

```python
def create_my_custom_config():
    config = create_base_config()
    config["experiment_name"] = "my_experiment"
    config["model"]["backbone"] = "resnet34"
    config["model"]["n_clusters"] = 10
    config["training"]["epochs"] = 100
    return config
```

然后在 `generate_all_configs()` 中添加：
```python
"my_experiment.json": create_my_custom_config(),
```

## 推荐实验流程

### 1. 快速测试（验证环境）
```bash
python config_experiments.py
python train_gpu.py --config configs/quick_test.json
python test_with_viz.py --experiment experiments/quick_test --skip_tsne
```

### 2. Backbone对比实验
```bash
# ResNet18
python train_gpu.py --config configs/resnet18_gmm_mlp.json

# ResNet50
python train_gpu.py --config configs/resnet50_gmm_mlp.json

# EfficientNet
python train_gpu.py --config configs/efficientnet_gmm_mlp.json

# Vision Transformer
python train_gpu.py --config configs/vit_gmm_mlp.json
```

### 3. 聚类数消融实验
```bash
python train_gpu.py --config configs/resnet18_4clusters.json
python train_gpu.py --config configs/resnet18_gmm_mlp.json  # 8 clusters
python train_gpu.py --config configs/resnet18_16clusters.json
```

### 4. 可视化对比
```bash
for exp in experiments/*/; do
    python test_with_viz.py --experiment "$exp"
done
```

## 性能预期（GPU）

基于2000训练样本 + 500验证样本：

| 模型 | 训练时间/epoch | 总训练时间 | 预期SRCC |
|------|---------------|-----------|----------|
| ResNet18 (frozen) | ~30秒 | ~15分钟 | 0.70-0.75 |
| ResNet18 | ~1分钟 | ~50分钟 | 0.80-0.85 |
| ResNet50 | ~2分钟 | ~100分钟 | 0.82-0.87 |
| EfficientNet-B0 | ~1.5分钟 | ~75分钟 | 0.81-0.86 |
| ViT-Small | ~1.5分钟 | ~75分钟 | 0.82-0.87 |

*基于NVIDIA V100/A100 GPU的估计时间*

## 超参数调优建议

### 学习率
- **ResNet18/34**: 1e-3
- **ResNet50+**: 5e-4
- **ViT/Swin**: 5e-4
- **冻结backbone**: 5e-3

### Batch Size（根据GPU显存）
- **16GB GPU**: 128-256
- **24GB GPU**: 256-512
- **40GB+ GPU**: 512-1024

### 聚类数
- **小数据集** (<1000样本): 4-6
- **中等数据集** (1000-5000): 8-12
- **大数据集** (5000+): 12-16

### 训练轮数
- **快速实验**: 10-20 epochs
- **正常训练**: 50-100 epochs
- **精细调优**: 100-200 epochs

## 常见问题

### Q: GPU内存不足
A: 减小batch_size或使用更小的backbone（如ResNet18而非ResNet50）

### Q: 训练不收敛
A:
1. 降低学习率
2. 增加训练样本数
3. 减少模型复杂度（降低hidden_dim或n_clusters）

### Q: 聚类分布不均
A:
1. 增加cluster_loss权重
2. 调整GMM初始化参数
3. 尝试不同的n_clusters

### Q: t-SNE太慢
A: 使用 `--skip_tsne` 标志跳过，或只可视化部分样本

## 进阶使用

### 使用真实IQA数据集

修改 `train_gpu.py` 中的 `HuggingFaceImageDataset` 类：

```python
# 替换为你的数据集
dataset = load_dataset("your_dataset_name", split=split)
# 或从CSV加载
import pandas as pd
df = pd.read_csv("your_annotations.csv")
```

### 多GPU训练

在 `train_gpu.py` 中添加：

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### 继续训练

修改 `train_gpu.py` 添加resume功能：

```python
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

## 依赖包

确保安装所有依赖：
```bash
pip install torch torchvision timm numpy scipy scikit-learn \
            datasets tqdm matplotlib seaborn Pillow
```

## 联系和贡献

如有问题或改进建议，欢迎反馈！
