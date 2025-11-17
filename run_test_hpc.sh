#!/bin/bash
#SBATCH --job-name=iqa_test
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# ============================================
# 测试模式：快速验证代码和环境
# 少量数据 (100 train, 50 val) + 2 epochs
# ============================================

echo "=========================================="
echo "测试模式：验证代码和环境"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 创建日志目录
mkdir -p logs
mkdir -p checkpoints

# 激活环境
source /gpfs/scratch/rl5285/miniconda3/bin/activate
conda activate UNSB

# 打印环境信息
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# 切换到项目目录
cd /gpfs/scratch/rl5285/CENIQA_project || exit 1

# 运行测试
echo "开始测试..."
/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8 compare_all_methods.py \
    --test_mode \
    --data_root data/stl10 \
    --output_dir results

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ 测试完成！代码和环境正常"
    echo "✓ 可以提交完整实验: sbatch run_full_hpc.sh"
else
    echo "✗ 测试失败，退出码: $exit_code"
    echo "✗ 请检查错误日志: logs/test_${SLURM_JOB_ID}.err"
fi
echo "End time: $(date)"
echo "=========================================="

exit $exit_code
