# GMM vs 无GMM 测试指南

## 概述

本指南说明如何测试和对比带 GMM 和不带 GMM 的两个模型版本。

---

## 快速开始

### 步骤 1：训练简单基线模型（无 GMM）

```bash
python train_simple_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_simple_baseline \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3
```

**关键参数：**
- `--dataset stl10`: 使用高分辨率 STL-10 数据集
- `--distortion_strength medium`: 中等失真强度
- `--experiment_name`: 实验名称，结果保存在 `experiments/stl10_simple_baseline/`
- `--epochs 50`: 训练 50 轮
- `--batch_size 64`: 批次大小
- `--lr 1e-3`: 学习率

**模型结构：**
- ResNet18 骨干网络 → 512 维特征
- 单调 MLP 回归器：512 → 512 → 1
- **无 GMM 聚类模块**

---

### 步骤 2：训练 GMM 模型

```bash
python train_high_res.py \
  --dataset stl10 \
  --distortion_strength medium \
  --experiment_name stl10_gmm_model \
  --n_clusters 5 \
  --cluster_loss_weight 0.1 \
  --balance_weight 0.5 \
  --entropy_weight 0.1 \
  --refit_interval 10 \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3
```

**关键参数：**
- `--n_clusters 5`: GMM 聚类数量（高分辨率数据建议 5）
- `--cluster_loss_weight 0.1`: 聚类损失权重
- `--balance_weight 0.5`: 聚类均衡损失权重
- `--entropy_weight 0.1`: 熵正则化权重
- `--refit_interval 10`: 每 10 个 epoch 重新拟合 GMM

**模型结构：**
- ResNet18 骨干网络 → 512 维特征
- GMM 聚类模块 → 5 维后验概率
- 单调 MLP 回归器：(512+5) → 512 → 1
- **带 GMM 聚类模块**

---

### 步骤 3：对比两个模型

训练完成后，你可以查看以下文件来对比结果：

1. **训练日志：**
   ```bash
   # 简单基线
   cat experiments/stl10_simple_baseline/train_log.txt

   # GMM 模型
   cat experiments/stl10_gmm_model/train_log.txt
   ```

2. **最终指标：**
   ```bash
   # 简单基线
   cat experiments/stl10_simple_baseline/metrics.json

   # GMM 模型
   cat experiments/stl10_gmm_model/metrics.json
   ```

3. **可视化结果：**
   ```bash
   # 查看训练曲线
   ls experiments/stl10_simple_baseline/*.png
   ls experiments/stl10_gmm_model/*.png
   ```

---

## 关键对比指标

| 指标 | 简单基线 | GMM 模型 | 说明 |
|------|---------|---------|------|
| **SRCC** | ？ | ？ | Spearman 相关系数（越高越好） |
| **PLCC** | ？ | ？ | Pearson 相关系数（越高越好） |
| **MSE Loss** | ？ | ？ | 均方误差损失（越低越好） |
| **参数量** | ~11.7M | ~11.7M + GMM | GMM 增加少量参数 |
| **训练时间** | 基线 | 基线 + 10-20% | GMM 增加计算开销 |

---

## 详细参数说明

### 简单基线模型（无 GMM）

**可调参数：**
```bash
--backbone resnet18          # 骨干网络：resnet18, resnet34, resnet50
--hidden_dim 512            # MLP 隐藏层维度
--dropout 0.3               # Dropout 比率
--lr 1e-3                   # 学习率
--weight_decay 1e-4         # 权重衰减
```

### GMM 模型

**可调参数：**
```bash
# GMM 设置
--n_clusters 5              # 聚类数（4-16，推荐 5-8）
--gmm_covariance_type diag  # 协方差类型：diag, full

# 损失权重
--cluster_loss_weight 0.1   # 聚类分离损失（0.0-0.5）
--balance_weight 0.5        # 聚类均衡损失（0.0-1.0）
--entropy_weight 0.1        # 熵正则化（0.0-0.3）

# GMM 更新
--refit_interval 10         # 重新拟合间隔（5-20 epoch）
--use_sklearn_init True     # 使用 sklearn 初始化
```

---

## 方法二：自动对比实验脚本

如果你想自动运行所有实验并生成对比报告，可以使用：

```bash
python run_comparison_experiments.py
```

这个脚本会：
1. 自动训练简单基线模型
2. 自动训练 GMM 模型
3. 生成对比报告和可视化

---

## 预期结果

### 简单基线（无 GMM）
- **优点：** 训练更快，模型更简单
- **缺点：** 可能在复杂失真模式上表现较差
- **适用场景：** 快速原型，基线参考

### GMM 模型
- **优点：** 更好的特征聚类，可能提升性能
- **缺点：** 训练稍慢，需要调整更多超参数
- **适用场景：** 需要更好性能的生产环境

---

## 实验建议

### 1. 数据集选择
- **STL-10（高分辨率）：** 96x96 像素，推荐用于高分辨率测试
- **CIFAR-10（低分辨率）：** 32x32 像素，快速实验

### 2. GMM 聚类数选择
```bash
# 少聚类（快速，可能性能较差）
--n_clusters 3

# 中等聚类（平衡，推荐）
--n_clusters 5

# 多聚类（更细粒度，可能过拟合）
--n_clusters 10
```

### 3. 失真强度选择
```bash
--distortion_strength low     # 轻微失真
--distortion_strength medium  # 中等失真（推荐）
--distortion_strength high    # 严重失真
```

---

## 故障排除

### 问题 1：GMM 聚类不均衡
**现象：** 所有样本集中在少数几个聚类

**解决方案：**
```bash
# 增加均衡权重
--balance_weight 1.0

# 减少聚类数
--n_clusters 3
```

### 问题 2：训练不稳定
**现象：** 损失震荡，指标不收敛

**解决方案：**
```bash
# 降低学习率
--lr 5e-4

# 增加权重衰减
--weight_decay 5e-4

# 减少 GMM 损失权重
--cluster_loss_weight 0.05
```

### 问题 3：过拟合
**现象：** 训练集性能好，验证集性能差

**解决方案：**
```bash
# 增加 dropout
--dropout 0.5

# 增加数据增强
--distortions_per_image 10

# 减少模型复杂度
--hidden_dim 256
```

---

## 快速命令参考

```bash
# 1. 训练简单基线（约 30-60 分钟）
python train_simple_high_res.py --dataset stl10 --experiment_name baseline --epochs 50

# 2. 训练 GMM 模型（约 40-80 分钟）
python train_high_res.py --dataset stl10 --n_clusters 5 --experiment_name gmm --epochs 50

# 3. 查看结果
cat experiments/baseline/metrics.json
cat experiments/gmm/metrics.json

# 4. 可视化对比
python -c "
import json
with open('experiments/baseline/metrics.json') as f:
    baseline = json.load(f)
with open('experiments/gmm/metrics.json') as f:
    gmm = json.load(f)
print(f'Baseline SRCC: {baseline[\"best_srcc\"]:.4f}')
print(f'GMM SRCC: {gmm[\"best_srcc\"]:.4f}')
print(f'Improvement: {(gmm[\"best_srcc\"] - baseline[\"best_srcc\"]) / baseline[\"best_srcc\"] * 100:.2f}%')
"
```

---

## 下一步

1. **运行实验：** 按照上述步骤运行两个模型
2. **分析结果：** 对比 SRCC、PLCC、训练曲线
3. **调整参数：** 根据结果调整 GMM 超参数
4. **深入分析：** 使用 `visualize_gmm_clusters.py` 可视化聚类结果

有问题随时问我！
