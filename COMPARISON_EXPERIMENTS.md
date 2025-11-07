# 对比实验指南

## 🎯 目标

对比两个模型的性能：
1. **Simple Baseline** - 简单的CNN→MLP直接回归（**无GMM**）
2. **GMM-based Model** - CNN→GMM→MLP（**带GMM聚类**）

## 📊 实验架构对比

| 特性 | Simple Baseline | GMM-based Model |
|------|----------------|-----------------|
| **架构** | CNN → MLP | CNN → GMM → MLP |
| **聚类** | ❌ 无 | ✅ 8个GMM聚类 |
| **输入维度** | 512 (features) | 520 (features + posteriors) |
| **参数** | ~11.2M | ~11.8M |
| **训练时间** | ~25分钟 | ~30分钟 |

## 🚀 方法1: 自动运行（推荐）

### 在Google Colab中运行

```python
# ============================================================================
# 🔬 自动运行两个实验并对比 - 完整流程
# ============================================================================

%cd /content/CENIQA_project

# 1. 更新代码
!git pull origin claude/resnet18-distorted-images-training-011CUrFBWVpjMy2D1UaHbtMx

# 2. 运行自动对比脚本（会依次训练两个模型并对比）
!python run_comparison_experiments.py

# 3. 显示对比结果
from IPython.display import Image, display

comparison_dir = 'experiments/comparison'

print("="*80)
print("📊 对比结果")
print("="*80)

display(Image(f'{comparison_dir}/comparison_dashboard.png'))

# 4. 显示详细结果
import json

with open(f'{comparison_dir}/comparison_results.json', 'r') as f:
    results = json.load(f)

print("\n" + "="*80)
print("详细指标对比")
print("="*80)

simple = results['simple_baseline']
gmm = results['gmm_based']

print(f"\n{'Metric':<10} {'Simple':<12} {'GMM':<12} {'Winner':<10}")
print("-" * 50)

for metric in ['srcc', 'plcc', 'rmse', 'mae']:
    val_simple = simple[metric]
    val_gmm = gmm[metric]

    if metric in ['srcc', 'plcc']:  # Higher is better
        winner = '🏆 GMM' if val_gmm > val_simple else '🏆 Simple'
        diff = val_gmm - val_simple
    else:  # Lower is better
        winner = '🏆 GMM' if val_gmm < val_simple else '🏆 Simple'
        diff = val_simple - val_gmm

    print(f"{metric.upper():<10} {val_simple:<12.4f} {val_gmm:<12.4f} {winner}")
    print(f"           {'Difference: ' + f'{diff:+.4f}':<12}")

print("\n" + "="*80)
```

## 🔧 方法2: 手动分步运行

### 步骤1: 训练Simple Baseline

```python
%cd /content/CENIQA_project

!python train_simple_baseline.py \
  --experiment_name resnet18_simple_baseline \
  --train_samples 1666 \
  --val_samples 166 \
  --epochs 50 \
  --batch_size 64
```

### 步骤2: 训练GMM-based Model

```python
!python train_with_distortions.py \
  --experiment_name resnet18_large_10k \
  --train_samples 1666 \
  --val_samples 166 \
  --epochs 50 \
  --batch_size 64 \
  --cluster_loss_weight 0.1 \
  --balance_weight 1.0 \
  --refit_interval 0
```

### 步骤3: 对比两个模型

```python
!python compare_experiments.py \
  --exp_simple experiments/resnet18_simple_baseline \
  --exp_gmm experiments/resnet18_large_10k \
  --test_samples 500
```

### 步骤4: 显示结果

```python
from IPython.display import Image, display

display(Image('experiments/comparison/comparison_dashboard.png'))
```

## 🎨 生成的可视化

对比面板包含7个子图：

1. **指标对比表** - SRCC, PLCC, RMSE, MAE的详细对比
2. **Simple Baseline散点图** - 预测 vs 真实
3. **GMM-based散点图** - 预测 vs 真实
4. **误差分布对比** - 两个模型的误差直方图
5. **累积误差曲线** - 显示误差累积分布
6. **箱型图对比** - 误差分布的统计特性
7. **按质量范围对比** - 不同质量区间的MAE

## 📈 预期结果

### 情况A: GMM模型更好

```
Metric     Simple       GMM          Improvement
SRCC       0.7234       0.7842       +0.0608 (+8.4%)
PLCC       0.7156       0.7956       +0.0800 (+11.2%)
RMSE       0.1456       0.1234       -0.0222 (-15.2%)

🏆 GMM-based model wins!
```

**结论**：GMM聚类有效提升了模型性能

### 情况B: 两者接近

```
Metric     Simple       GMM          Improvement
SRCC       0.7842       0.7856       +0.0014 (+0.2%)
PLCC       0.7923       0.7956       +0.0033 (+0.4%)
RMSE       0.1248       0.1234       -0.0014 (-1.1%)

🤝 Models perform similarly
```

**结论**：GMM的额外复杂度没有带来显著提升，可以考虑用更简单的模型

### 情况C: Simple更好（不太可能）

```
Metric     Simple       GMM          Difference
SRCC       0.7956       0.7234       -0.0722 (-9.1%)

⚠️  Simple baseline is better!
```

**结论**：GMM训练可能有问题，需要检查：
- Balance loss是否生效
- GMM重拟合是否破坏了训练
- 聚类数是否合适

## 🔍 快速测试（5分钟）

如果只是想快速看看效果：

```python
# 快速测试版本（少量数据，少量epoch）
%cd /content/CENIQA_project

# 训练Simple Baseline (5分钟)
!python train_simple_baseline.py \
  --experiment_name simple_quick_test \
  --train_samples 500 \
  --val_samples 100 \
  --epochs 10 \
  --batch_size 32

# 训练GMM-based (6分钟)
!python train_with_distortions.py \
  --experiment_name gmm_quick_test \
  --train_samples 500 \
  --val_samples 100 \
  --epochs 10 \
  --batch_size 32 \
  --cluster_loss_weight 0.1 \
  --balance_weight 1.0

# 对比
!python compare_experiments.py \
  --exp_simple experiments/simple_quick_test \
  --exp_gmm experiments/gmm_quick_test \
  --test_samples 200

# 显示
from IPython.display import Image, display
display(Image('experiments/comparison/comparison_dashboard.png'))
```

## 📁 生成的文件结构

```
experiments/
├── resnet18_simple_baseline/      # Simple Baseline实验
│   ├── best_model.pth             # 最佳模型
│   ├── last_model.pth             # 最后一个epoch
│   ├── training_history.json      # 训练历史
│   └── config.json                # 配置
│
├── resnet18_large_10k/            # GMM-based实验
│   ├── best_model.pth
│   ├── last_model.pth
│   ├── training_history.json
│   └── config.json
│
└── comparison/                    # 对比结果
    ├── comparison_dashboard.png   # 对比可视化
    └── comparison_results.json    # 详细结果
```

## 💡 理解对比结果

### GMM的潜在优势

1. **更好的特征表示** - 聚类可以捕捉不同类型的失真
2. **更鲁棒** - 对不同质量范围的图像都有好的表现
3. **可解释性** - 可以看到哪些失真被分到哪个聚类

### Simple Baseline的优势

1. **更简单** - 更少的参数和复杂度
2. **更快** - 训练和推断都更快
3. **更稳定** - 没有GMM崩溃的风险

### 如何选择

- **GMM提升 > 5%** → 使用GMM-based模型
- **GMM提升 < 2%** → 使用Simple Baseline（更简单）
- **GMM更差** → 检查GMM训练配置

## 🐛 故障排查

### 问题1: Simple Baseline训练失败

```bash
# 检查依赖
python -c "from simple_model import SimpleCNNModel; print('OK')"
```

### 问题2: 对比脚本找不到模型

```bash
# 检查两个实验是否都有best_model.pth
ls -lh experiments/resnet18_simple_baseline/*.pth
ls -lh experiments/resnet18_large_10k/*.pth
```

### 问题3: GMM模型性能异常差

检查training_history.json中的balance_loss：
```python
import json
with open('experiments/resnet18_large_10k/training_history.json') as f:
    history = json.load(f)

if 'balance_loss' in history[0]:
    print("✅ Balance loss enabled")
    print(f"Final balance_loss: {history[-1]['balance_loss']:.4f}")
else:
    print("❌ Balance loss not found - using old training script?")
```

## 📚 相关文件

- `simple_model.py` - Simple Baseline模型定义
- `train_simple_baseline.py` - Simple Baseline训练脚本
- `train_with_distortions.py` - GMM-based训练脚本
- `compare_experiments.py` - 对比脚本
- `run_comparison_experiments.py` - 自动运行脚本

## 🎯 下一步

完成对比后：

1. **分析结果** - 看哪个模型更好
2. **调优** - 根据结果调整超参数
3. **论文/报告** - 使用生成的对比图
4. **部署** - 选择性能更好的模型

---

**推荐使用"方法1: 自动运行"，一键完成所有实验！** 🚀
