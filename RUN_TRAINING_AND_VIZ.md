# 完整的训练和可视化流程指南

## 当前状态

你尝试运行：
```bash
python enhanced_visualize.py --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth --num_images 25 --test_samples 500
```

但遇到问题：
- ❌ 实验目录 `experiments/resnet18_large_10k` 不存在
- ❌ 没有 `best_model.pth` 检查点文件
- ⚠️ 需要先训练模型

## 解决方案：完整的训练流程

### 步骤 1: 安装依赖（如果需要）

```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib seaborn
pip install datasets tqdm Pillow
pip install scikit-learn
```

### 步骤 2: 运行改进的训练脚本

使用我们刚修复的训练脚本（带均匀聚类正则化）：

```bash
python train_with_distortions.py \
  --experiment_name resnet18_large_10k \
  --backbone resnet18 \
  --n_clusters 8 \
  --train_samples 1666 \
  --val_samples 166 \
  --distortions_per_image 5 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --cluster_loss_weight 0.1 \
  --balance_weight 1.0 \
  --entropy_weight 0.1 \
  --refit_interval 0
```

**说明**：
- `train_samples=1666`: 训练集参考图像数量
- `distortions_per_image=5`: 每张图像生成5个失真版本
- 总训练样本数：1666 × 6 = 9996（5个失真 + 1个原图）
- 验证样本数：166 × 6 = 996

训练完成后会创建：
```
experiments/
└── resnet18_large_10k/
    ├── config.json
    ├── best_model.pth          ← 最佳模型检查点
    ├── last_model.pth          ← 最后一个epoch的模型
    └── training_history.json   ← 训练历史
```

### 步骤 3: 运行可视化

训练完成后，运行可视化：

```bash
python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth \
  --num_images 25 \
  --test_samples 500
```

这将生成：
```
experiments/resnet18_large_10k/
├── enhanced_visualization.png  ← 主要可视化图
├── cluster_analysis.png        ← 聚类分析图
└── prediction_scatter.png      ← 预测散点图
```

## 快速开始（最小配置）

如果你只是想快速测试：

```bash
# 1. 快速训练（小数据集，10个epoch）
python train_with_distortions.py \
  --experiment_name quick_test \
  --train_samples 500 \
  --val_samples 100 \
  --epochs 10 \
  --batch_size 32

# 2. 可视化结果
python enhanced_visualize.py \
  --experiment experiments/quick_test \
  --checkpoint best_model.pth \
  --num_images 16 \
  --test_samples 100
```

## 替代方案：使用原始训练脚本

如果你想使用原始的CIFAR10训练脚本（不带失真）：

### 1. 创建配置文件

```bash
cat > configs/resnet18_large_10k.json << 'EOF'
{
  "experiment_name": "resnet18_large_10k",
  "device": "cuda",
  "seed": 42,
  "data": {
    "dataset": "cifar10",
    "train_samples": 10000,
    "val_samples": 1000,
    "batch_size": 128,
    "num_workers": 2
  },
  "model": {
    "backbone": "resnet18",
    "feature_dim": 512,
    "n_clusters": 8,
    "hidden_dim": 512,
    "dropout": 0.3,
    "freeze_backbone": false
  },
  "training": {
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adam",
    "scheduler": "reduce_on_plateau",
    "scheduler_params": {
      "mode": "max",
      "factor": 0.5,
      "patience": 5,
      "min_lr": 1e-06
    }
  },
  "loss_weights": {
    "quality_loss": 1.0,
    "cluster_loss": 0.1
  },
  "logging": {
    "save_dir": "./experiments",
    "log_interval": 10,
    "save_best": true,
    "save_last": true
  }
}
EOF
```

### 2. 运行训练

```bash
python train_gpu.py --config configs/resnet18_large_10k.json
```

### 3. 运行可视化

```bash
python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth \
  --num_images 25 \
  --test_samples 500
```

## 检查训练进度

训练时可以使用另一个终端查看：

```bash
# 查看训练历史
cat experiments/resnet18_large_10k/training_history.json | tail -20

# 查看最佳SRCC
cat experiments/resnet18_large_10k/training_history.json | grep -o '"srcc": [0-9.]*' | sort -t: -k2 -rn | head -1

# 实时监控（如果训练正在运行）
watch -n 5 'ls -lh experiments/resnet18_large_10k/*.pth'
```

## 预期训练时间

在Tesla T4 GPU上：

| 配置 | 样本数 | Epoch时间 | 总时间(50 epochs) |
|-----|--------|----------|------------------|
| Quick Test | 500 | ~15秒 | ~12分钟 |
| Small | 1666×6=10k | ~30秒 | ~25分钟 |
| Large | 10000 | ~2分钟 | ~1.7小时 |

## 预期训练结果

使用改进的训练脚本（带均匀聚类正则化）：

- **SRCC**: 0.75-0.85（稳定提升）
- **PLCC**: 0.75-0.85
- **RMSE**: 0.10-0.15
- **聚类分布**: 每个聚类 1000-1500 样本（均匀）
- **Balance Loss**: < 0.05（接近均匀分布）

## 故障排查

### Q: 实验目录不存在
**A**: 需要先运行训练脚本创建实验目录和检查点

### Q: ModuleNotFoundError: No module named 'torch'
**A**: 安装依赖：
```bash
pip install -r requirements.txt
```

### Q: CUDA out of memory
**A**: 减少batch size：
```bash
--batch_size 32  # 或 16
```

### Q: 训练很慢
**A**:
- 减少训练样本：`--train_samples 500`
- 减少epochs：`--epochs 10`
- 增加num_workers：在GPU服务器上可以设为4-8

### Q: 想使用预训练的检查点
**A**: 如果你在其他地方已经训练过模型：
```bash
# 1. 创建实验目录
mkdir -p experiments/resnet18_large_10k

# 2. 复制检查点文件
cp /path/to/your/best_model.pth experiments/resnet18_large_10k/

# 3. 运行可视化
python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth \
  --num_images 25 \
  --test_samples 500
```

## 可视化选项

完整的可视化命令选项：

```bash
python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \  # 实验目录
  --checkpoint best_model.pth \                  # 检查点文件名
  --num_images 25 \                              # 显示多少张图像
  --test_samples 500 \                           # 测试集大小
  --device cuda                                   # 使用的设备
```

生成的可视化包括：
1. **图像网格**：显示预测vs真实质量分数
2. **聚类分析**：t-SNE降维的聚类可视化
3. **预测散点图**：真实值vs预测值
4. **性能指标**：SRCC, PLCC, RMSE

## 推荐的完整工作流

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型（使用改进的脚本）
python train_with_distortions.py \
  --experiment_name resnet18_large_10k \
  --train_samples 1666 \
  --val_samples 166 \
  --epochs 50 \
  --batch_size 64

# 3. 等待训练完成（约25分钟）
# 监控：watch -n 5 'tail experiments/resnet18_large_10k/training_history.json'

# 4. 可视化结果
python enhanced_visualize.py \
  --experiment experiments/resnet18_large_10k \
  --checkpoint best_model.pth \
  --num_images 25 \
  --test_samples 500

# 5. 查看生成的图像
ls -lh experiments/resnet18_large_10k/*.png
```

## 下一步

训练和可视化完成后：

1. **分析结果**：查看生成的可视化图像
2. **调优参数**：根据结果调整超参数
3. **创建PR**：如果结果满意，提交代码
4. **部署模型**：使用最佳检查点进行推理

## 需要帮助？

如果遇到问题：
1. 查看 `TRAINING_IMPROVEMENTS.md` 了解改进细节
2. 查看 `VISUALIZATION_GUIDE_CN.md` 了解可视化详情
3. 检查 `training_history.json` 查看训练指标
