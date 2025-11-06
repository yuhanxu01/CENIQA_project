# 文件清单和说明

## 核心文件（必须上传到Colab）

### 1. 模型组件
- **backbones.py** (4 KB) - CNN backbone实现（ResNet, EfficientNet, ViT等）
- **gmm_module.py** (3 KB) - 可微分高斯混合模型
- **regressors.py** (6 KB) - MLP回归头

### 2. 实验框架
- **config_experiments.py** (新建) - 配置文件生成器
  - 生成8种不同的实验配置
  - 包含ResNet18/50, EfficientNet, ViT等

- **train_gpu.py** (新建) - GPU优化的训练脚本
  - 支持配置文件
  - 自动GMM初始化
  - 完整的训练日志
  - 自动保存最佳模型

- **test_with_viz.py** (新建) - 测试和可视化脚本
  - 完整的测试流程
  - 自动生成6种可视化
  - 聚类统计分析

- **visualize.py** (新建) - 可视化工具库
  - 预测散点图
  - 聚类分布图
  - PCA/t-SNE特征可视化
  - 训练曲线
  - 样本预测展示

## 文档文件

- **GPU_EXPERIMENTS_README.md** (新建) - 详细实验指南
  - 完整的使用说明
  - 配置参数详解
  - 性能预期
  - 常见问题解答

- **QUICK_START.md** (新建) - 快速开始指南
  - Colab部署步骤
  - 示例代码
  - 调试技巧

- **PROBLEM_ANALYSIS.md** (之前创建) - 问题分析文档
  - CPU版本问题诊断
  - 性能对比

## 依赖文件

- **requirements.txt** (已更新)
  - 添加了matplotlib, seaborn用于可视化

## 原有文件（项目已有）

- backbones.py
- config.py
- dataset.py
- extractors.py
- gmm_module.py
- losses.py
- model.py
- regressors.py
- train.py
- train_utils.py
- inference.py

## 上传到Colab的最小文件集

### 方案A：完整功能（推荐）
上传这7个文件：
1. backbones.py
2. gmm_module.py
3. regressors.py
4. config_experiments.py ⭐ 新
5. train_gpu.py ⭐ 新
6. test_with_viz.py ⭐ 新
7. visualize.py ⭐ 新

### 方案B：最小化（仅训练）
上传这4个文件：
1. backbones.py
2. gmm_module.py
3. regressors.py
4. train_gpu.py ⭐ 新

手动创建简单配置：
```python
# 在Colab中运行
import json

config = {
    "experiment_name": "test",
    "device": "cuda",
    "seed": 42,
    "data": {"train_samples": 1000, "val_samples": 200, "batch_size": 128, "num_workers": 2},
    "model": {"backbone": "resnet18", "feature_dim": 512, "n_clusters": 8, "hidden_dim": 512, "dropout": 0.3, "freeze_backbone": False},
    "training": {"epochs": 20, "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "adam", "scheduler": "reduce_on_plateau", "scheduler_params": {"mode": "max", "factor": 0.5, "patience": 3, "min_lr": 1e-6}},
    "loss_weights": {"quality_loss": 1.0, "cluster_loss": 0.1},
    "logging": {"save_dir": "./experiments", "log_interval": 10, "save_best": True, "save_last": True},
    "visualization": {"enabled": True}
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)
```

## 生成的文件（训练后）

### 配置文件（自动生成）
```
configs/
├── resnet18_gmm_mlp.json
├── resnet50_gmm_mlp.json
├── efficientnet_gmm_mlp.json
├── vit_gmm_mlp.json
├── resnet18_4clusters.json
├── resnet18_16clusters.json
├── resnet18_frozen.json
└── quick_test.json
```

### 实验输出
```
experiments/
└── {experiment_name}/
    ├── config.json              # 实验配置备份
    ├── best_model.pth           # 最佳模型 (~136MB)
    ├── last_model.pth           # 最后模型 (~136MB)
    ├── training_history.json    # 训练历史
    └── visualizations/
        ├── test_results.json
        ├── predictions_scatter.png
        ├── cluster_distribution.png
        ├── features_pca.png
        ├── features_tsne.png       # 可选
        ├── sample_predictions.png
        └── training_curves.png
```

## 文件大小参考

- Python源文件: 3-15 KB（很小）
- 模型检查点: ~136 MB（ResNet18）
- 可视化图片: 100-500 KB 每张
- 文档文件: 5-10 KB

## 使用流程

### 第一次使用
1. 上传7个核心文件到Colab
2. 安装依赖：`pip install timm datasets tqdm matplotlib seaborn -q`
3. 生成配置：`python config_experiments.py`
4. 选择实验训练：`python train_gpu.py --config configs/xxx.json`
5. 测试可视化：`python test_with_viz.py --experiment experiments/xxx`

### 后续实验
1. 修改现有配置或创建新配置
2. 运行训练
3. 比较结果

## 推荐的实验顺序

### Day 1: 环境验证
- quick_test.json (10分钟)
- 验证所有功能正常

### Day 2: Backbone对比
- resnet18_gmm_mlp.json (50分钟)
- resnet50_gmm_mlp.json (100分钟)
- efficientnet_gmm_mlp.json (75分钟)

### Day 3: 聚类数消融
- resnet18_4clusters.json
- resnet18_gmm_mlp.json (8 clusters)
- resnet18_16clusters.json

### Day 4: 可视化分析
- 运行所有实验的test_with_viz.py
- 对比结果，写报告

## 注意事项

1. **GPU内存**：如果遇到OOM，减小batch_size
2. **训练时间**：基于V100 GPU的估计，实际可能有差异
3. **Colab限制**：注意Colab的运行时限制（12小时）
4. **数据持久化**：训练完记得下载模型和结果
5. **可视化**：t-SNE很慢，可以用--skip_tsne跳过

## 关键改进点

相比demo_cnn_gmm_mlp.py：

1. **更大的模型** - 增加了参数和复杂度
2. **更多数据** - 从200增加到2000训练样本
3. **更长训练** - 从5增加到50 epochs
4. **配置管理** - JSON配置文件，易于修改和复现
5. **完整可视化** - 6种可视化图表
6. **实验对比** - 支持多个实验并行
7. **GPU优化** - 大batch size, pin_memory等

## 获取帮助

- 参考 QUICK_START.md 了解基本用法
- 参考 GPU_EXPERIMENTS_README.md 了解详细配置
- 参考 PROBLEM_ANALYSIS.md 了解之前的问题和解决方案
