# HPC快速开始指南

## ⚡ 30秒快速开始

```bash
# 1. 上传代码到HPC（在本地）
tar -czf ceniqa.tar.gz CENIQA_project/
scp ceniqa.tar.gz your_user@hpc:/gpfs/scratch/rl5285/

# 2. SSH到HPC并解压
ssh your_user@hpc
cd /gpfs/scratch/rl5285/
tar -xzf ceniqa.tar.gz
cd CENIQA_project

# 3. 先测试（2 epochs，100样本，~10分钟）
sbatch run_test_hpc.sh

# 4. 查看测试结果
tail -f logs/test_*.out

# 5. 测试通过后运行完整实验（50 epochs，5000样本，~30小时）
sbatch run_full_hpc.sh

# 6. 实时监控
tail -f logs/full_*.out
```

## 📊 实验完成后查看结果

```bash
# 查看结果摘要
python analyze_results.py results/comparison_results_full_*.json

# 生成可视化（包含训练曲线、性能排名、LaTeX表格）
python analyze_results.py results/comparison_results_full_*.json --output_dir plots
```

---

## 🎯 实验内容

### 7个模型对比：

**基线模型：**
1. `0_NoGMM` - 无GMM（仅CNN + Regressor）
2. `1_StandardGMM` - 标准GMM（当前实现）

**GMM改进方案：**
3. `2_MoE` - Mixture of Experts
4. `3_Attention` - Attention-Gated Fusion
5. `4_LearnableGMM` - Learnable GMM
6. `5_DistortionAware` - Distortion-Aware Multi-Expert
7. `6_Complete` - Complete Self-Supervised Pipeline

### 评估指标：
- **SRCC** (Spearman Rank Correlation Coefficient)
- **PLCC** (Pearson Linear Correlation Coefficient)

---

## 📁 重要文件

| 文件 | 说明 |
|------|------|
| `compare_all_methods.py` | 主实验脚本（7个模型） |
| `run_test_hpc.sh` | 测试提交脚本（2 epochs） |
| `run_full_hpc.sh` | 完整实验提交脚本（50 epochs） |
| `analyze_results.py` | 结果分析和可视化 |
| `HPC_EXPERIMENT_GUIDE.md` | 详细实验指南 |

---

## ⚙️ 自定义参数

### 修改训练参数

编辑 `run_full_hpc.sh`，修改这些参数：

```bash
python compare_all_methods.py \
    --epochs 100 \              # 训练轮数
    --batch_size 64 \           # batch大小
    --lr 5e-5 \                 # 学习率
    --num_train 10000 \         # 训练样本数
    --num_val 2000              # 验证样本数
```

### 修改SLURM资源

编辑 `run_full_hpc.sh` 开头的SBATCH参数：

```bash
#SBATCH --mem=40G              # 内存
#SBATCH --time=48:00:00        # 时间限制
#SBATCH --cpus-per-task=7      # CPU核心数
#SBATCH --gres=gpu:1           # GPU数量
```

---

## 🔍 监控和调试

### 查看任务状态
```bash
squeue -u $USER                # 查看队列
scontrol show job <job_id>     # 查看任务详情
scancel <job_id>               # 取消任务
```

### 查看日志
```bash
# 实时查看输出
tail -f logs/full_*.out

# 查看错误
cat logs/full_*.err

# 查看GPU使用（在计算节点上）
nvidia-smi
```

### 常见错误

**Out of Memory (OOM)**
```bash
# 解决：减小batch size或增加内存
--batch_size 16
#SBATCH --mem=80G
```

**任务超时**
```bash
# 解决：延长时间限制或减少epochs
#SBATCH --time=72:00:00
--epochs 30
```

---

## 📈 预期结果

基于理论分析的预期性能排序（SRCC）：

1. 🥇 **6_Complete** (0.88-0.92)
2. 🥈 **5_DistortionAware** (0.86-0.90)
3. 🥉 **4_LearnableGMM** (0.85-0.89)
4. **2_MoE** (0.83-0.87)
5. **3_Attention** (0.82-0.86)
6. **1_StandardGMM** (0.80-0.84)
7. **0_NoGMM** (0.80-0.84)

*实际结果可能因数据集和超参数而异*

---

## 📞 需要帮助？

详细指南请查看：
- `HPC_EXPERIMENT_GUIDE.md` - 完整实验指南
- `GMM_IMPROVEMENTS.md` - 方法理论说明
- `logs/*.err` - 错误日志

祝实验顺利！🚀
