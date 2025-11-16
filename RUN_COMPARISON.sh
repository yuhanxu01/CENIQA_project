#!/bin/bash
# 一键运行GMM方法比较脚本

echo "======================================"
echo "  一键比较5种GMM改进方法"
echo "======================================"
echo ""

# 创建必要的目录
mkdir -p checkpoints
mkdir -p data

# 运行比较脚本
python compare_gmm_methods.py

echo ""
echo "完成！查看 gmm_comparison_results_*.json 获取详细结果"
