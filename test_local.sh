#!/bin/bash
# 本地测试脚本 - 在提交HPC之前验证代码
# 使用极少量数据快速测试所有方法是否能正常运行

echo "========================================"
echo "本地测试 - 验证所有7个方法"
echo "========================================"
echo ""
echo "这将运行7个方法各1个epoch，使用100个样本"
echo "预计总时间: 10-20分钟（取决于你的GPU）"
echo ""

# 创建测试输出目录
mkdir -p results/local_test
mkdir -p checkpoints/local_test

METHODS=("no_gmm" "vanilla_gmm" "moe" "attention" "learnable_gmm" "distortion_aware" "complete")
METHOD_NAMES=("No GMM" "Vanilla GMM" "MoE" "Attention" "Learnable GMM" "Distortion-Aware" "Complete")

echo "开始测试..."
echo ""

for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    name="${METHOD_NAMES[$i]}"

    echo "----------------------------------------"
    echo "[$((i+1))/7] 测试: ${name} (${method})"
    echo "----------------------------------------"

    python train_single_method.py \
        --method ${method} \
        --epochs 1 \
        --batch_size 8 \
        --max_train_samples 100 \
        --max_val_samples 50 \
        --output_dir results/local_test \
        --checkpoint_dir checkpoints/local_test

    if [ $? -eq 0 ]; then
        echo "✓ ${name} 测试通过"
    else
        echo "✗ ${name} 测试失败"
        exit 1
    fi

    echo ""
done

echo "========================================"
echo "所有测试通过！✓"
echo "========================================"
echo ""
echo "下一步："
echo "1. 在HPC上运行快速测试: ./submit_quick_test.sh"
echo "2. 检查快速测试结果"
echo "3. 运行完整训练: ./submit_full_training.sh"
echo ""
