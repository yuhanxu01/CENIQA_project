#!/bin/bash
# 快速测试所有7个方案 - 串行运行验证代码和环境
# 1个node，7个方法串行，2 epochs

#SBATCH --job-name=quick_test_all
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/quick_test_all_%j.out
#SBATCH --error=logs/quick_test_all_%j.err

echo "=========================================="
echo "快速测试 - 串行运行所有7个方法"
echo "节点: ${SLURM_NODELIST}"
echo "任务ID: ${SLURM_JOB_ID}"
echo "开始时间: $(date)"
echo "=========================================="
echo ""

# 创建必要目录
mkdir -p logs results/quick_test checkpoints/quick_test

# 定义方法列表
METHODS=("no_gmm" "vanilla_gmm" "moe" "attention" "learnable_gmm" "distortion_aware" "complete")
METHOD_NAMES=("No GMM Baseline" "Vanilla GMM" "MoE GMM" "Attention GMM" "Learnable GMM" "Distortion-Aware" "Complete Pipeline")

# 串行运行所有方法
for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    name="${METHOD_NAMES[$i]}"

    echo ""
    echo "=========================================="
    echo "[$((i+1))/7] 测试方法: ${name}"
    echo "方法代码: ${method}"
    echo "开始时间: $(date)"
    echo "=========================================="

    /gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8 train_single_method.py \
        --method ${method} \
        --quick_test \
        --output_dir results/quick_test \
        --checkpoint_dir checkpoints/quick_test

    if [ $? -eq 0 ]; then
        echo "✓ ${name} 测试成功"
    else
        echo "✗ ${name} 测试失败"
        exit 1
    fi

    echo "完成时间: $(date)"
done

echo ""
echo "=========================================="
echo "所有7个方法测试完成！"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "查看对比结果："
echo "python compare_results.py --results_dir results/quick_test --output_dir comparison_plots/quick_test"
