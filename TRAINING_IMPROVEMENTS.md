# 训练改进说明

## 问题诊断

### 原始问题

1. **c_loss=-1.0000**: 聚类损失过度优化，模型完全收敛到one-hot分布
2. **性能崩溃**: Epoch 11后SRCC从0.8047骤降到0.0101
3. **GMM重拟合破坏性**: 重新拟合GMM完全改变聚类分配，破坏已学习的特征-质量映射
4. **聚类不均匀**: 某些聚类有2292个样本，某些只有502个样本

### 根本原因

- **cluster_loss_weight=0.5 过高**: 导致模型过度关注聚类任务而非质量预测
- **GMM定期重拟合**: 每10个epoch完全重置聚类中心，MLP需要重新学习映射关系
- **缺乏均匀分布约束**: 没有机制保证聚类分布均匀

## 改进方案

### 1. 均匀聚类分布正则化

**新增Balance Loss** - 使用KL散度惩罚聚类分布偏离均匀分布：

```python
# 计算batch中每个聚类的平均后验概率
cluster_distribution = torch.mean(posteriors, dim=0)  # [n_clusters]

# 目标：均匀分布
uniform_distribution = torch.ones(n_clusters) / n_clusters

# KL散度惩罚
balance_loss = KL(cluster_distribution || uniform_distribution)
```

**效果**:
- 鼓励8个聚类平均分配样本
- 防止某些聚类过度集中或空闲
- 提高模型泛化能力

### 2. 改进的损失函数

**新的总损失**:

```python
Total Loss = quality_loss
           + cluster_loss_weight × cluster_loss      # 降低到0.1
           + balance_weight × balance_loss           # 新增，默认1.0
           + entropy_weight × entropy_loss           # 保留0.1
```

**各组件作用**:

| Loss组件 | 权重 | 作用 | 目标 |
|---------|------|------|------|
| **Quality Loss** | 1.0 | MSE预测损失 | 主要任务：准确预测质量分数 |
| **Cluster Loss** | 0.1 ↓ | 聚类分离 | 鼓励明确的聚类分配（降低权重） |
| **Balance Loss** | 1.0 🆕 | 均匀分布 | 保证聚类均匀分配样本 |
| **Entropy Loss** | 0.1 | 熵正则化 | 防止过度自信 |

### 3. 禁用GMM重拟合

**改进**:
- `refit_interval` 默认值从10改为0（禁用）
- 只在初始化时使用sklearn GMM拟合一次
- 后续通过梯度下降自然更新GMM参数

**原因**:
- GMM重拟合破坏已学习的映射关系
- 导致性能崩溃
- 梯度下降已经足够更新GMM参数

### 4. 调整超参数默认值

| 参数 | 原值 | 新值 | 说明 |
|-----|------|------|------|
| `cluster_loss_weight` | 0.5 | 0.1 | 降低聚类损失权重 |
| `refit_interval` | 10 | 0 | 禁用GMM重拟合 |
| `balance_weight` | - | 1.0 | 新增均匀分布权重 |
| `entropy_weight` | - | 0.1 | 熵正则化权重 |

## 使用方法

### 基本使用（推荐默认参数）

```bash
python train_with_distortions.py \
  --experiment_name resnet18_balanced \
  --backbone resnet18 \
  --n_clusters 8 \
  --epochs 50 \
  --batch_size 64
```

### 调整正则化强度

```bash
# 更强的均匀分布约束
python train_with_distortions.py \
  --experiment_name resnet18_strong_balance \
  --balance_weight 2.0 \
  --cluster_loss_weight 0.05

# 完全移除聚类损失（仅使用均匀分布约束）
python train_with_distortions.py \
  --experiment_name resnet18_balance_only \
  --cluster_loss_weight 0.0 \
  --balance_weight 1.5

# 启用GMM重拟合（不推荐）
python train_with_distortions.py \
  --experiment_name resnet18_with_refit \
  --refit_interval 20 \
  --cluster_loss_weight 0.05
```

### 参数调优指南

**balance_weight** (均匀分布权重):
- 0.5-1.0: 轻度约束，允许自然的聚类分布
- 1.0-2.0: 中等约束，鼓励相对均匀（推荐）
- 2.0+: 强约束，强制均匀分布

**cluster_loss_weight** (聚类分离权重):
- 0.0: 完全移除聚类损失，仅依赖均匀约束
- 0.05-0.1: 轻度鼓励聚类分离（推荐）
- 0.2+: 较强聚类分离（可能过度）

**entropy_weight** (熵正则化权重):
- 0.05-0.1: 轻度防止过度自信（推荐）
- 0.2+: 强制保持不确定性

## 预期改进

### 训练稳定性
- ✅ 避免性能崩溃
- ✅ SRCC持续稳定提升
- ✅ 不会出现c_loss=-1.0000

### 聚类质量
- ✅ 8个聚类相对均匀（每个约1250±200样本）
- ✅ 避免singleton或collapsed聚类
- ✅ GMM拟合不会失败

### 模型性能
- ✅ 预计SRCC稳定在0.75-0.85
- ✅ 更好的泛化能力
- ✅ 更鲁棒的训练过程

## 监控指标

训练时关注以下指标：

```
Train Loss: 0.0234
  - Quality Loss: 0.0180    # 应该持续下降
  - Cluster Loss: -0.8500   # 应该在[-0.9, -0.7]范围
  - Balance Loss: 0.0234    # 应该逐渐减小到<0.05
  - Entropy Loss: -1.8500   # 应该相对稳定
Val Loss: 0.0269
SRCC: 0.7678                # 应该持续提升
```

**健康指标**:
- Balance Loss < 0.05: 聚类分布接近均匀
- Cluster Loss在[-0.95, -0.70]之间: 聚类适度分离
- SRCC持续提升: 质量预测改善

**警告信号**:
- Balance Loss > 0.2: 聚类严重不均匀
- Cluster Loss < -0.99: 过度聚类（考虑降低cluster_loss_weight）
- SRCC震荡或下降: 学习率过高或过拟合

## 理论基础

### 为什么需要均匀聚类？

1. **防止模式坍缩**: 避免所有样本集中到少数几个聚类
2. **充分利用模型容量**: 确保每个聚类都学习有用的表示
3. **提高泛化**: 均匀分布使模型对不同类型的distortion都有鲁棒性
4. **避免GMM拟合失败**: 防止singleton或ill-defined covariance

### Balance Loss vs Cluster Loss

- **Cluster Loss**: 鼓励每个样本明确分配到某个聚类（intra-sample）
- **Balance Loss**: 鼓励所有样本均匀分配到各个聚类（inter-sample）
- 两者互补，共同保证聚类质量

## 下一步优化

可选的进一步改进：

1. **自适应权重**: 根据训练阶段动态调整损失权重
2. **渐进式GMM更新**: 使用EMA而非完全重拟合
3. **对抗训练**: 增加对distortion的鲁棒性
4. **混合精度训练**: 加速训练（使用torch.cuda.amp）

## 问题排查

### Q: Balance Loss一直很高（>0.2）
**A**: 增加balance_weight到2.0-3.0，或降低cluster_loss_weight

### Q: SRCC提升很慢
**A**: 可能balance_weight过高，尝试降低到0.5或完全移除

### Q: 训练不稳定
**A**:
- 检查学习率，尝试降低到5e-4
- 增加batch size到128
- 增加weight decay到1e-3

### Q: 想恢复GMM重拟合
**A**: 不推荐，如果必须：
```bash
--refit_interval 20 --cluster_loss_weight 0.05 --balance_weight 1.5
```
