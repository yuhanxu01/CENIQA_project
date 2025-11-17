#!/bin/bash
#SBATCH --job-name=iqa_full
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

# ============================================
# 完整实验：7个模型对比
# No GMM + 标准GMM + 5种GMM改进
# 5000 train, 1000 val, 50 epochs
# ============================================

echo "=========================================="
echo "完整实验：7个IQA模型对比"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 创建必要目录
mkdir -p logs
mkdir -p checkpoints
mkdir -p results

# 激活环境
source /gpfs/scratch/rl5285/miniconda3/bin/activate
conda activate UNSB

# 打印环境信息
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU Memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")')"
else
    echo "WARNING: CUDA not available!"
fi
echo ""

# 切换到项目目录
cd /gpfs/scratch/rl5285/CENIQA_project || exit 1

# 运行完整实验
echo "开始完整实验..."
echo "  - 训练样本: 5000"
echo "  - 验证样本: 1000"
echo "  - Epochs: 50"
echo "  - Batch size: 32"
echo ""

/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8 compare_all_methods.py \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_train 5000 \
    --num_val 1000 \
    --data_root data/stl10 \
    --output_dir results

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ 完整实验完成！"
    echo "✓ 结果保存在: results/comparison_results_full_*.json"
    echo "✓ 模型保存在: checkpoints/*_best.pth"
else
    echo "✗ 实验失败，退出码: $exit_code"
    echo "✗ 请检查错误日志: logs/full_${SLURM_JOB_ID}.err"
fi
echo "End time: $(date)"
echo "=========================================="

exit $exit_code
