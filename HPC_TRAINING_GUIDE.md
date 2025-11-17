# HPC训练指南 - 7个IQA方法对比实验

## 📋 实验概述

本实验对比7个图像质量评估(IQA)方法：

1. **No GMM (Baseline)** - 不使用GMM，直接回归
2. **Vanilla GMM** - 标准GMM + 特征拼接
3. **MoE GMM** - Mixture of Experts
4. **Attention GMM** - Attention-Gated Feature Fusion
5. **Learnable GMM** - 可学习的GMM参数
6. **Distortion-Aware** - 显式建模失真类型
7. **Complete Pipeline** - 完整的Self-Supervised Pipeline

## 📂 文件结构

```
CENIQA_project/
├── train_single_method.py          # 单方法训练脚本
├── submit_quick_test.sh            # 快速测试提交脚本（2 epochs）
├── submit_full_training.sh         # 完整训练提交脚本（60 epochs）
├── compare_results.py              # 结果对比脚本
├── logs/                           # SLURM日志目录
├── results/
│   ├── quick_test/                # 快速测试结果
│   └── full_training/             # 完整训练结果
└── checkpoints/
    ├── quick_test/                # 快速测试模型
    └── full_training/             # 完整训练模型
```

## 🚀 使用步骤

### 步骤1: 快速测试（验证环境和代码）

首先运行快速测试确保所有代码和环境正常：

```bash
# 创建必要的目录
mkdir -p logs results checkpoints

# 提交快速测试任务（1个node，7个方法串行，每个2 epochs）
sbatch submit_quick_test.sh
```

**快速测试配置：**
- 申请资源: 1个GPU node
- 运行方式: 7个方法串行运行（一个接一个）
- Epochs: 2
- 训练样本: 500张参考图 × 6 = 3000样本
- 验证样本: 200张参考图 × 6 = 1200样本
- 预计时间: 每个方法 ~30-45分钟，总计 ~3-5小时

**监控任务：**
```bash
# 查看任务状态（应该只有1个任务）
squeue -u $USER

# 查看实时日志（所有7个方法在一个日志文件中）
tail -f logs/quick_test_all_*.out

# 查看完整日志
cat logs/quick_test_all_*.out
```

### 步骤2: 检查快速测试结果

等待所有快速测试完成后，检查结果：

```bash
# 查看快速测试结果
python compare_results.py \
    --results_dir results/quick_test \
    --output_dir comparison_plots/quick_test
```

这会生成：
- `comparison_table.csv` - 对比表格
- `training_comparison.png` - 训练曲线对比
- `final_comparison.png` - 最终结果对比

**确认检查点：**
- ✅ 所有7个任务都成功完成
- ✅ 没有CUDA错误或内存溢出
- ✅ 训练损失正常下降
- ✅ SRCC和PLCC指标合理（>0.1）

### 步骤3: 完整训练

快速测试通过后，运行完整训练：

```bash
# 给脚本添加执行权限
chmod +x submit_full_training.sh

# 提交完整训练任务（7个node并行，每个方法60 epochs）
./submit_full_training.sh
```

**完整训练配置：**
- 申请资源: 7个GPU nodes
- 运行方式: 7个方法并行运行（同时运行）
- Epochs: 60
- 训练集: ~11,700张参考图 × 6 = ~70,200样本 (90%)
- 验证集: ~1,300张参考图 × 6 = ~7,800样本 (10%)
- Batch size: 16
- Learning rate: 1e-4
- 预计时间: 每个方法 ~8-12小时（并行运行，总时间不变）

**监控任务：**
```bash
# 查看任务状态（应该看到7个任务）
squeue -u $USER

# 查看特定方法的日志
tail -f logs/full_no_gmm_*.out
tail -f logs/full_moe_*.out

# 查看所有方法的进度
watch -n 60 'ls -lh checkpoints/full_training/*.pth'
```

### 步骤4: 对比最终结果

所有完整训练完成后，生成对比报告：

```bash
# 对比所有方法的结果
python compare_results.py \
    --results_dir results/full_training \
    --output_dir comparison_plots/full_training
```

输出文件：
- `comparison_table.csv` - 详细对比表格
- `training_comparison.png` - 训练过程对比（4个子图）
- `final_comparison.png` - 最终结果对比（SRCC vs PLCC）

## 📊 查看结果

### 命令行查看

```bash
# 查看CSV表格
column -t -s, comparison_plots/full_training/comparison_table.csv | less -S

# 查看最佳结果
cat results/full_training/*_results_*.json | grep -A 3 "best_srcc"
```

### 下载结果到本地

```bash
# 在本地机器上运行
scp -r username@hpc:/path/to/CENIQA_project/comparison_plots ./
scp -r username@hpc:/path/to/CENIQA_project/results ./
```

## 🔧 单独训练某个方法

如果需要单独训练某个特定方法：

```bash
# 快速测试模式
python train_single_method.py \
    --method moe \
    --quick_test

# 完整训练模式
python train_single_method.py \
    --method moe \
    --epochs 60 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir results/full_training \
    --checkpoint_dir checkpoints/full_training
```

可用的方法：
- `no_gmm` - No GMM Baseline
- `vanilla_gmm` - Vanilla GMM
- `moe` - MoE GMM
- `attention` - Attention GMM
- `learnable_gmm` - Learnable GMM
- `distortion_aware` - Distortion-Aware
- `complete` - Complete Pipeline

## 📈 预期结果

基于初步测试，预期排名（仅供参考）：

1. **Complete Pipeline** - 综合所有改进
2. **Learnable GMM** - 自适应GMM参数
3. **MoE GMM** - 专家混合模型
4. **Distortion-Aware** - 显式建模失真
5. **Attention GMM** - 注意力机制
6. **Vanilla GMM** - 标准GMM
7. **No GMM** - 无GMM基线

实际结果可能因数据集和超参数设置而异。

## 🐛 常见问题

### Q1: 任务一直在队列中等待
```bash
# 检查队列状态
squeue -u $USER

# 查看分区可用性
sinfo -p gpu4_medium
```

### Q2: 内存不足错误
减少batch size或增加内存：
```bash
# 修改submit脚本中的：
#SBATCH --mem=40G  # 增加到40G
```

### Q3: CUDA错误
检查GPU可用性：
```bash
# 在计算节点上
nvidia-smi
```

### Q4: 重新运行某个失败的方法
```bash
# 单独提交
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=retry_moe
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/retry_moe_%j.out

/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8 train_single_method.py \
    --method moe \
    --epochs 60 \
    --output_dir results/full_training \
    --checkpoint_dir checkpoints/full_training
EOF
```

## 📝 注意事项

1. **确保环境正确**：Python 3.8，已安装所有依赖
2. **检查路径**：修改脚本中的Python路径为你的环境路径
3. **磁盘空间**：确保有足够空间存储模型和结果（~10GB）
4. **GPU资源**：确认分区和GPU类型可用
5. **日志监控**：定期检查日志确保训练正常进行

## 📧 支持

如有问题，请查看：
1. SLURM日志文件：`logs/*.err`
2. Python输出日志：`logs/*.out`
3. 训练结果JSON：`results/*/*.json`

祝训练顺利！🚀
