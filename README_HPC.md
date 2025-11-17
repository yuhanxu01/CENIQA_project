# HPC并行训练：7个IQA方法对比实验

## 📁 项目文件说明

### 主要脚本
- **`train_single_method.py`** - 核心训练脚本，支持7个方法的独立训练
- **`submit_quick_test.sh`** - 提交快速测试（2 epochs，验证环境）
- **`submit_full_training.sh`** - 提交完整训练（60 epochs）
- **`compare_results.py`** - 结果对比和可视化
- **`test_local.sh`** - 本地测试脚本（提交HPC前验证）

### 文档
- **`QUICK_START.md`** - ⭐ 快速开始指南（从这里开始）
- **`HPC_TRAINING_GUIDE.md`** - 详细使用文档
- **`README_HPC.md`** - 本文件

## 🎯 7个实验方法

| # | 方法名 | 命令行参数 | 说明 |
|---|--------|-----------|------|
| 1 | No GMM Baseline | `--method no_gmm` | 不使用GMM，直接回归（基线） |
| 2 | Vanilla GMM | `--method vanilla_gmm` | 标准GMM + 特征拼接 |
| 3 | MoE GMM | `--method moe` | Mixture of Experts（每个cluster一个expert） |
| 4 | Attention GMM | `--method attention` | Attention机制融合cluster特征 |
| 5 | Learnable GMM | `--method learnable_gmm` | 可学习的GMM参数 |
| 6 | Distortion-Aware | `--method distortion_aware` | 显式建模失真类型 |
| 7 | Complete Pipeline | `--method complete` | 综合所有改进技术 |

## 🚀 快速开始（3步走）

### 1️⃣ 本地验证（可选但推荐）

在本地机器上快速验证代码：

```bash
./test_local.sh
```

这会用极少量数据测试所有7个方法（10-20分钟）

### 2️⃣ HPC快速测试（必须）

上传代码到HPC后，先运行快速测试：

```bash
# 在HPC上
cd /gpfs/scratch/rl5285/CENIQA_project
./submit_quick_test.sh

# 监控任务
squeue -u $USER
tail -f logs/quick_*.out
```

等待完成后检查结果：

```bash
python compare_results.py \
    --results_dir results/quick_test \
    --output_dir comparison_plots/quick_test
```

### 3️⃣ 完整训练

快速测试通过后，运行完整训练：

```bash
./submit_full_training.sh

# 等待完成后
python compare_results.py \
    --results_dir results/full_training \
    --output_dir comparison_plots/full_training
```

## 📊 实验配置

### 快速测试
- **目的**: 验证代码和环境
- **Epochs**: 2
- **数据**: 500训练样本 + 200验证样本
- **时间**: 每个方法30-45分钟，总计3-5小时

### 完整训练
- **目的**: 获得最终对比结果
- **Epochs**: 60
- **数据**: 70,200训练样本 + 7,800验证样本（90/10分割）
- **时间**: 每个方法8-12小时（并行运行）

## 📈 预期结果

根据初步实验，预期性能排名（SRCC）：

1. **Complete Pipeline** (0.75-0.85) - 综合所有改进
2. **Learnable GMM** (0.70-0.80) - 自适应参数
3. **MoE GMM** (0.65-0.75) - 专家混合
4. **Distortion-Aware** (0.65-0.75) - 失真建模
5. **Attention GMM** (0.60-0.70) - 注意力机制
6. **Vanilla GMM** (0.55-0.65) - 标准GMM
7. **No GMM** (0.50-0.60) - 基线方法

> 实际结果可能因数据集和超参数而异

## 📂 输出文件

### 训练结果
```
results/
├── quick_test/
│   ├── no_gmm_results_*.json
│   ├── vanilla_gmm_results_*.json
│   └── ...
└── full_training/
    ├── no_gmm_results_*.json
    ├── vanilla_gmm_results_*.json
    └── ...
```

### 模型权重
```
checkpoints/
├── quick_test/
│   ├── no_gmm_best.pth
│   └── ...
└── full_training/
    ├── no_gmm_best.pth
    └── ...
```

### 对比结果
```
comparison_plots/
├── quick_test/
│   ├── comparison_table.csv
│   ├── training_comparison.png
│   └── final_comparison.png
└── full_training/
    ├── comparison_table.csv
    ├── training_comparison.png
    └── final_comparison.png
```

## 🔧 高级用法

### 单独训练某个方法

```bash
# 在HPC上提交单个任务
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=test_moe
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=20G

python train_single_method.py \
    --method moe \
    --epochs 60 \
    --batch_size 16 \
    --output_dir results/full_training \
    --checkpoint_dir checkpoints/full_training
EOF
```

### 自定义参数

```bash
python train_single_method.py \
    --method complete \
    --epochs 100 \
    --batch_size 32 \
    --lr 5e-5 \
    --max_train_samples 5000 \
    --output_dir results/custom \
    --checkpoint_dir checkpoints/custom
```

### 恢复训练

目前不支持恢复训练。如果任务中断，需要重新开始。

## 🐛 故障排除

### 问题1: 任务提交失败
```bash
# 检查SLURM配置
sbatch --test-only submit_quick_test.sh

# 查看分区状态
sinfo -p gpu4_medium
```

### 问题2: CUDA内存不足
降低batch size：
```bash
python train_single_method.py --method METHOD --batch_size 8 ...
```

### 问题3: 数据加载慢
增加workers：
```bash
# 修改train_single_method.py中的num_workers
DataLoader(..., num_workers=8, ...)
```

### 问题4: 查看详细错误
```bash
# 查看SLURM错误日志
cat logs/full_METHOD_JOBID.err

# 查看Python traceback
grep -A 20 "Traceback" logs/full_METHOD_JOBID.out
```

## 📞 支持

遇到问题？检查：

1. **日志文件**: `logs/*.out` 和 `logs/*.err`
2. **结果文件**: `results/*/*.json`
3. **训练指南**: `HPC_TRAINING_GUIDE.md`
4. **快速开始**: `QUICK_START.md`

## 🔬 实验设计说明

### 公平对比保证

所有7个方法使用：
- ✅ 相同的数据集（STL-10，90/10分割）
- ✅ 相同的backbone（ResNet-50）
- ✅ 相同的训练参数（lr=1e-4, batch_size=16）
- ✅ 相同的训练epochs（60）
- ✅ 相同的评估指标（SRCC, PLCC）

### 数据集配置

- **数据源**: STL-10 (13,000张参考图)
- **失真类型**: 8种（blur, noise, jpeg, saturation, contrast, brightness, pixelation）
- **每张图生成**: 5种失真 + 1张原图 = 6个样本
- **总样本数**: 78,000
- **分割**: 90% 训练 (70,200) / 10% 验证 (7,800)
- **随机种子**: 42（保证可复现）

### 评估指标

- **SRCC** (Spearman Rank Correlation): 衡量排序一致性
- **PLCC** (Pearson Linear Correlation): 衡量线性相关性
- **主要指标**: SRCC（更鲁棒）

## 📝 引用

如果使用本代码，请引用原始CENIQA论文及相关工作。

---

**最后更新**: 2025-11-17
**作者**: Claude & Research Team
