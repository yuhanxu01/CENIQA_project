# HPC实验指南：7个IQA模型对比

## 📋 实验概述

本实验比较7个图像质量评估(IQA)模型：

### 基线模型
1. **0_NoGMM** - 无GMM基线（仅backbone + regressor）
2. **1_StandardGMM** - 标准GMM（当前实现：sklearn GMM + concatenate）

### GMM改进方案（来自GMM_IMPROVEMENTS.md）
3. **2_MoE** - 方案1: Mixture of Expert Regressors
4. **3_Attention** - 方案2: Attention-Gated Feature Fusion
5. **4_LearnableGMM** - 方案3: Differentiable GMM with Learnable Priors
6. **5_DistortionAware** - 方案4: Distortion-Aware Multi-Expert Architecture
7. **6_Complete** - 方案5: Complete Self-Supervised GMM-IQA Pipeline

---

## 🚀 使用步骤

### 步骤1：上传代码到HPC

```bash
# 本地打包
tar -czf ceniqa_project.tar.gz CENIQA_project/

# 上传到HPC
scp ceniqa_project.tar.gz your_username@hpc_address:/gpfs/scratch/rl5285/

# 在HPC上解压
ssh your_username@hpc_address
cd /gpfs/scratch/rl5285/
tar -xzf ceniqa_project.tar.gz
```

### 步骤2：准备数据

```bash
# 在HPC上
cd /gpfs/scratch/rl5285/CENIQA_project

# 创建数据目录
mkdir -p data/stl10

# 下载STL10数据集（或使用你自己的数据）
# 数据集会在第一次运行时自动下载
```

### 步骤3：测试运行（**重要！**）

在运行完整实验之前，先用测试模式验证代码和环境：

```bash
# 提交测试任务
sbatch run_test_hpc.sh

# 查看任务状态
squeue -u $USER

# 查看测试日志（等任务开始运行后）
tail -f logs/test_*.out

# 检查是否有错误
cat logs/test_*.err
```

**测试参数：**
- 训练样本: 100
- 验证样本: 50
- Epochs: 2
- 预计时间: 10-20分钟

**如果测试失败，检查：**
1. Python环境是否正确
2. 数据路径是否存在
3. GPU是否可用
4. 依赖包是否安装

### 步骤4：运行完整实验

测试通过后，提交完整实验：

```bash
# 提交完整实验任务
sbatch run_full_hpc.sh

# 查看任务状态
squeue -u $USER

# 实时查看输出
tail -f logs/full_*.out

# 查看错误（如果有）
tail -f logs/full_*.err
```

**完整实验参数：**
- 训练样本: 5000
- 验证样本: 1000
- Epochs: 50
- Batch size: 32
- 学习率: 1e-4
- 预计时间: 20-40小时（取决于GPU性能）

---

## 📊 查看结果

### 实时监控

```bash
# 查看当前训练进度
tail -f logs/full_*.out

# 查看GPU使用情况（如果在计算节点上）
nvidia-smi

# 查看作业队列
squeue -u $USER
```

### 实验完成后

结果保存在：
- **JSON结果**: `results/comparison_results_full_*.json`
- **最佳模型**: `checkpoints/*_best.pth`
- **训练日志**: `logs/full_*.out`

查看结果：

```bash
# 查看最新结果文件
ls -lt results/comparison_results_*.json | head -1

# 美化输出JSON
python -m json.tool results/comparison_results_full_*.json | less

# 提取关键结果
python analyze_results.py results/comparison_results_full_*.json
```

---

## 📁 目录结构

```
CENIQA_project/
├── compare_all_methods.py          # 主实验脚本
├── run_test_hpc.sh                 # 测试提交脚本
├── run_full_hpc.sh                 # 完整实验提交脚本
├── analyze_results.py              # 结果分析脚本
├── logs/                           # SLURM日志
│   ├── test_*.out                  # 测试stdout
│   ├── test_*.err                  # 测试stderr
│   ├── full_*.out                  # 完整实验stdout
│   └── full_*.err                  # 完整实验stderr
├── checkpoints/                    # 模型检查点
│   ├── 0_NoGMM_best.pth
│   ├── 1_StandardGMM_best.pth
│   ├── 2_MoE_best.pth
│   ├── 3_Attention_best.pth
│   ├── 4_LearnableGMM_best.pth
│   ├── 5_DistortionAware_best.pth
│   └── 6_Complete_best.pth
└── results/                        # 实验结果
    ├── comparison_results_test_*.json
    └── comparison_results_full_*.json
```

---

## 🔧 自定义配置

如果需要修改实验参数，编辑 `run_full_hpc.sh`:

```bash
# 修改训练参数
/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8 compare_all_methods.py \
    --epochs 100 \              # 训练轮数
    --batch_size 64 \           # batch大小
    --lr 5e-5 \                 # 学习率
    --num_train 10000 \         # 训练样本数
    --num_val 2000 \            # 验证样本数
    --data_root data/stl10 \    # 数据路径
    --output_dir results        # 输出目录
```

或者修改SLURM资源：

```bash
#SBATCH --mem=80G              # 增加内存
#SBATCH --time=72:00:00        # 延长时间限制
#SBATCH --gres=gpu:2           # 使用2个GPU（需要修改代码支持DDP）
```

---

## ⚠️ 常见问题

### 1. 任务被杀死（OOM）

**症状**: `logs/*.err` 显示 "Killed" 或 "Out of memory"

**解决**:
```bash
# 增加内存
#SBATCH --mem=40G  # 改为更大的值

# 或减小batch size
--batch_size 16
```

### 2. GPU不可用

**症状**: 代码运行在CPU上，速度极慢

**解决**:
```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查GPU分配
echo $CUDA_VISIBLE_DEVICES

# 确保SLURM脚本有
#SBATCH --gres=gpu:1
```

### 3. 数据加载慢

**症状**: 训练时GPU利用率低，大部分时间在等待数据

**解决**:
```bash
# 增加num_workers
--num_workers 8  # 在Python脚本中修改DataLoader

# 或增加cache_size
cache_size=500  # 在Python脚本中修改
```

### 4. 任务超时

**症状**: 任务在达到时间限制后被强制终止

**解决**:
```bash
# 延长时间限制
#SBATCH --time=96:00:00  # 4天

# 或减少epochs
--epochs 30
```

---

## 📈 预期结果

根据GMM_IMPROVEMENTS.md的理论分析，预期性能排序（SRCC）：

1. **6_Complete** (0.88-0.92) - 完整Pipeline，结合所有最佳实践
2. **5_DistortionAware** (0.86-0.90) - 显式建模distortion
3. **4_LearnableGMM** (0.85-0.89) - 端到端学习GMM
4. **2_MoE** (0.83-0.87) - Mixture of Experts
5. **3_Attention** (0.82-0.86) - Attention机制
6. **1_StandardGMM** (0.80-0.84) - 当前GMM实现
7. **0_NoGMM** (0.80-0.84) - 无GMM基线

**实际结果可能有所不同，取决于：**
- 数据集特性
- 超参数设置
- 随机初始化
- 训练时长

---

## 📝 结果分析

实验完成后，使用分析脚本：

```bash
# 生成可视化报告
python analyze_results.py results/comparison_results_full_*.json

# 输出包括：
# 1. 性能对比表格
# 2. 训练曲线图
# 3. 模型排名
# 4. 统计显著性检验
```

---

## 🎯 下一步

实验完成后，可以：

1. **论文撰写**: 使用结果和可视化
2. **进一步分析**: 使用保存的模型进行推理
3. **超参数优化**: 针对最佳模型进行调优
4. **扩展实验**: 在更多数据集上测试

---

## 📞 帮助

如有问题，检查：
1. SLURM日志: `logs/*.err` 和 `logs/*.out`
2. Python traceback
3. GPU/内存使用情况

祝实验顺利！🚀
