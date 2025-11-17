# HPC实验：7个IQA模型对比

本目录包含完整的HPC实验代码，用于公平对比7个图像质量评估(IQA)模型。

---

## 📚 文件清单

### 核心实验文件
- **`compare_all_methods.py`** - 主实验脚本，实现7个模型的训练和评估
- **`compare_gmm_methods.py`** - 旧版本（仅5个GMM方法）

### HPC提交脚本
- **`run_test_hpc.sh`** - 测试模式SLURM脚本（100样本 + 2 epochs，~10分钟）
- **`run_full_hpc.sh`** - 完整实验SLURM脚本（5000样本 + 50 epochs，~30小时）
- **`run_local_test.sh`** - 本地测试脚本（可在本地机器上验证）

### 分析和文档
- **`analyze_results.py`** - 结果分析和可视化脚本
- **`HPC_EXPERIMENT_GUIDE.md`** - 详细实验指南（推荐阅读）
- **`QUICKSTART_HPC.md`** - 快速开始指南
- **`GMM_IMPROVEMENTS.md`** - GMM改进方案理论说明

---

## 🚀 快速开始

### 方法1：HPC上运行（推荐）

```bash
# 1. 上传代码到HPC
scp -r CENIQA_project/ your_user@hpc:/gpfs/scratch/rl5285/

# 2. SSH到HPC
ssh your_user@hpc
cd /gpfs/scratch/rl5285/CENIQA_project

# 3. 先测试（重要！）
sbatch run_test_hpc.sh

# 4. 查看测试日志
tail -f logs/test_*.out

# 5. 测试通过后运行完整实验
sbatch run_full_hpc.sh

# 6. 监控进度
tail -f logs/full_*.out
```

### 方法2：本地测试

```bash
# 在本地机器上快速测试
./run_local_test.sh

# 查看结果
python analyze_results.py results/comparison_results_test_*.json
```

---

## 🎯 实验内容

### 7个模型

**Baseline模型：**
1. **0_NoGMM** - 无GMM基线（仅CNN + Regressor）
2. **1_StandardGMM** - 标准GMM（当前实现：sklearn GMM + concatenate）

**GMM改进方案（来自GMM_IMPROVEMENTS.md）：**
3. **2_MoE** - 方案1: Mixture of Expert Regressors
4. **3_Attention** - 方案2: Attention-Gated Feature Fusion
5. **4_LearnableGMM** - 方案3: Differentiable GMM with Learnable Priors
6. **5_DistortionAware** - 方案4: Distortion-Aware Multi-Expert Architecture
7. **6_Complete** - 方案5: Complete Self-Supervised GMM-IQA Pipeline

### 评估指标
- **SRCC** (Spearman Rank Correlation Coefficient) - 主要指标
- **PLCC** (Pearson Linear Correlation Coefficient) - 次要指标

---

## 📊 实验模式

### 测试模式（`--test_mode`）
- **目的**: 快速验证代码和环境
- **样本数**: 100训练 + 50验证
- **Epochs**: 2
- **耗时**: ~10分钟
- **用法**: 在运行完整实验之前必须先测试

### 完整模式（默认）
- **目的**: 正式实验和论文结果
- **样本数**: 5000训练 + 1000验证
- **Epochs**: 50
- **耗时**: ~30小时（取决于GPU）
- **用法**: 测试通过后运行

---

## 📂 目录结构

```
CENIQA_project/
├── compare_all_methods.py          # 主实验脚本（7个模型）
├── analyze_results.py              # 结果分析脚本
│
├── run_test_hpc.sh                 # HPC测试脚本
├── run_full_hpc.sh                 # HPC完整实验脚本
├── run_local_test.sh               # 本地测试脚本
│
├── HPC_EXPERIMENT_GUIDE.md         # 详细指南
├── QUICKSTART_HPC.md               # 快速指南
├── GMM_IMPROVEMENTS.md             # 理论说明
│
├── logs/                           # SLURM日志
│   ├── test_*.out                  # 测试stdout
│   ├── test_*.err                  # 测试stderr
│   ├── full_*.out                  # 完整实验stdout
│   └── full_*.err                  # 完整实验stderr
│
├── checkpoints/                    # 模型检查点
│   ├── 0_NoGMM_best.pth
│   ├── 1_StandardGMM_best.pth
│   ├── 2_MoE_best.pth
│   ├── 3_Attention_best.pth
│   ├── 4_LearnableGMM_best.pth
│   ├── 5_DistortionAware_best.pth
│   └── 6_Complete_best.pth
│
├── results/                        # 实验结果
│   ├── comparison_results_test_*.json
│   └── comparison_results_full_*.json
│
└── plots/                          # 可视化图表
    ├── training_curves.png
    ├── performance_ranking.png
    └── results_table.tex
```

---

## 🔧 自定义配置

### 修改实验参数

编辑 `run_full_hpc.sh` 或直接运行：

```bash
python compare_all_methods.py \
    --epochs 100 \              # 训练轮数
    --batch_size 64 \           # batch大小
    --lr 5e-5 \                 # 学习率
    --num_train 10000 \         # 训练样本数
    --num_val 2000 \            # 验证样本数
    --data_root data/stl10 \    # 数据路径
    --output_dir results        # 输出目录
```

### 修改SLURM资源

编辑 `run_full_hpc.sh` 的SBATCH参数：

```bash
#SBATCH --mem=80G              # 增加内存
#SBATCH --time=72:00:00        # 延长时间
#SBATCH --cpus-per-task=14     # 更多CPU
#SBATCH --gres=gpu:2           # 多GPU（需修改代码）
```

---

## 📈 查看和分析结果

### 命令行查看

```bash
# 查看最新结果
ls -lt results/comparison_results_*.json | head -1

# 美化JSON输出
python -m json.tool results/comparison_results_full_*.json | less

# 运行分析脚本
python analyze_results.py results/comparison_results_full_*.json
```

### 生成可视化

```bash
# 生成所有图表
python analyze_results.py results/comparison_results_full_*.json --output_dir plots

# 输出包括：
# - plots/training_curves.png     : 训练曲线
# - plots/performance_ranking.png : 性能排名
# - plots/results_table.tex       : LaTeX表格
```

### 输出示例

```
==================================================================================================
实验结果摘要
==================================================================================================

配置信息:
  - 训练样本: 5000
  - 验证样本: 1000
  - Epochs: 50
  - Batch size: 32
  - 学习率: 0.0001
  - 总耗时: 1523.45 分钟

----------------------------------------------------------------------------------------------------
排名   模型                      最佳SRCC      最佳PLCC      最佳Epoch     相对提升
----------------------------------------------------------------------------------------------------
🏆     6_Complete                0.8934       0.9012       47           +8.42%
2      5_DistortionAware         0.8812       0.8901       45           +6.94%
3      4_LearnableGMM            0.8698       0.8756       43           +5.55%
4      2_MoE                     0.8523       0.8611       41           +3.43%
5      3_Attention               0.8445       0.8534       39           +2.48%
6      1_StandardGMM             0.8291       0.8412       38           +0.61%
7      0_NoGMM                   0.8241       0.8378       37           +0.00%
----------------------------------------------------------------------------------------------------
```

---

## ⚠️ 常见问题

### 1. 测试失败

**症状**: `run_test_hpc.sh` 运行失败

**解决步骤**:
1. 查看错误日志: `cat logs/test_*.err`
2. 检查Python环境: `python --version`
3. 检查依赖: `python -c "import torch; print(torch.__version__)"`
4. 检查数据路径: `ls data/stl10/`

### 2. Out of Memory (OOM)

**症状**: 日志显示 "CUDA out of memory" 或 "Killed"

**解决**:
```bash
# 选项1: 减小batch size
--batch_size 16

# 选项2: 增加内存
#SBATCH --mem=80G

# 选项3: 使用更少样本
--num_train 3000
```

### 3. 任务超时

**症状**: 任务在时间限制前被强制终止

**解决**:
```bash
# 延长时间限制
#SBATCH --time=96:00:00

# 或减少epochs
--epochs 30
```

### 4. GPU不可用

**症状**: 代码运行在CPU上，速度极慢

**解决**:
```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 确保SLURM脚本请求GPU
#SBATCH --gres=gpu:1

# 检查GPU分配
echo $CUDA_VISIBLE_DEVICES
```

---

## 📖 进一步阅读

- **`HPC_EXPERIMENT_GUIDE.md`** - 详细的实验指南，包含所有细节
- **`QUICKSTART_HPC.md`** - 30秒快速开始
- **`GMM_IMPROVEMENTS.md`** - GMM改进方案的理论基础和参考文献

---

## 📧 支持

如有问题：
1. 查看详细指南: `HPC_EXPERIMENT_GUIDE.md`
2. 检查日志: `logs/*.err`
3. 查看已知问题和解决方案（上方常见问题部分）

---

## 🎓 引用

如果使用本代码，请引用相关论文（待补充）。

---

**祝实验顺利！🚀**

最后更新: 2024
