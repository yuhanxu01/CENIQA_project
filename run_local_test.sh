#!/bin/bash

# ============================================
# 本地测试脚本：快速验证代码
# 用法: ./run_local_test.sh
# ============================================

echo "=========================================="
echo "本地测试：验证代码和环境"
echo "=========================================="
echo ""

# 创建必要目录
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p data

# 检查Python
echo "检查Python环境..."
python --version
echo ""

# 检查依赖
echo "检查依赖包..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
echo ""

# 检查CUDA
echo "检查CUDA..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# 运行测试
echo "开始测试（少量数据 + 2 epochs）..."
echo ""

python compare_all_methods.py \
    --test_mode \
    --data_root data/stl10 \
    --output_dir results 2>&1 | tee logs/local_test.log

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ 测试完成！代码和环境正常"
    echo "✓ 日志保存在: logs/local_test.log"
    echo "✓ 结果保存在: results/comparison_results_test_*.json"
    echo ""
    echo "下一步:"
    echo "  - 查看结果: python analyze_results.py results/comparison_results_test_*.json"
    echo "  - 运行完整实验: python compare_all_methods.py --epochs 50 --num_train 5000 --num_val 1000"
else
    echo "✗ 测试失败，退出码: $exit_code"
    echo "✗ 请检查日志: logs/local_test.log"
fi
echo "=========================================="

exit $exit_code
