#!/bin/bash
# 快速测试所有7个方案 - 验证代码和环境
# 2 epochs, 500训练样本, 200验证样本

METHODS=("no_gmm" "vanilla_gmm" "moe" "attention" "learnable_gmm" "distortion_aware" "complete")
METHOD_NAMES=("No_GMM_Baseline" "Vanilla_GMM" "MoE_GMM" "Attention_GMM" "Learnable_GMM" "Distortion_Aware" "Complete_Pipeline")

echo "提交快速测试任务..."
echo "================================"

for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    name="${METHOD_NAMES[$i]}"

    echo "提交方法 ${i}: ${name} (${method})"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=quick_${method}
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/quick_${method}_%j.out
#SBATCH --error=logs/quick_${method}_%j.err

echo "=========================================="
echo "快速测试: ${name}"
echo "方法: ${method}"
echo "节点: \${SLURM_NODELIST}"
echo "任务ID: \${SLURM_JOB_ID}"
echo "=========================================="

# 创建日志目录
mkdir -p logs

# 激活环境并运行
/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8 train_single_method.py \\
    --method ${method} \\
    --quick_test \\
    --output_dir results/quick_test \\
    --checkpoint_dir checkpoints/quick_test

echo "完成时间: \$(date)"
EOF

done

echo "================================"
echo "已提交 ${#METHODS[@]} 个快速测试任务"
echo "查看任务状态: squeue -u \$USER"
echo "查看日志: tail -f logs/quick_*.out"
