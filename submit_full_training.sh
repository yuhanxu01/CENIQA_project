#!/bin/bash
# 完整训练所有7个方案
# 60 epochs, 全部STL-10数据集 (90/10分割)

METHODS=("no_gmm" "vanilla_gmm" "moe" "attention" "learnable_gmm" "distortion_aware" "complete")
METHOD_NAMES=("No_GMM_Baseline" "Vanilla_GMM" "MoE_GMM" "Attention_GMM" "Learnable_GMM" "Distortion_Aware" "Complete_Pipeline")

echo "提交完整训练任务..."
echo "================================"

for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    name="${METHOD_NAMES[$i]}"

    echo "提交方法 ${i}: ${name} (${method})"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=full_${method}
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_${method}_%j.out
#SBATCH --error=logs/full_${method}_%j.err

echo "=========================================="
echo "完整训练: ${name}"
echo "方法: ${method}"
echo "节点: \${SLURM_NODELIST}"
echo "任务ID: \${SLURM_JOB_ID}"
echo "=========================================="

# 创建日志和输出目录
mkdir -p logs
mkdir -p results/full_training
mkdir -p checkpoints/full_training

# 激活环境并运行
/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8 train_single_method.py \\
    --method ${method} \\
    --epochs 60 \\
    --batch_size 16 \\
    --lr 1e-4 \\
    --output_dir results/full_training \\
    --checkpoint_dir checkpoints/full_training

echo "完成时间: \$(date)"
EOF

done

echo "================================"
echo "已提交 ${#METHODS[@]} 个完整训练任务"
echo "查看任务状态: squeue -u \$USER"
echo "查看日志: tail -f logs/full_*.out"
