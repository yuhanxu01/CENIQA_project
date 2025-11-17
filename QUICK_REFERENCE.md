# 快速参考卡片

## ⚡ 两种运行模式对比

| 项目 | 快速测试 | 完整训练 |
|------|---------|---------|
| **提交命令** | `sbatch submit_quick_test.sh` | `./submit_full_training.sh` |
| **申请资源** | 1个GPU node | 7个GPU nodes |
| **运行方式** | 7个方法串行 | 7个方法并行 |
| **Epochs** | 2 | 60 |
| **数据量** | 少量 (3000+1200) | 全部 (70200+7800) |
| **预计时间** | 3-5小时 | 8-12小时 |
| **任务数** | 1个 | 7个 |
| **日志文件** | `quick_test_all_*.out` | `full_METHOD_*.out` |

## 🚀 使用流程

```bash
# 1. 快速测试（必须先做）
sbatch submit_quick_test.sh
tail -f logs/quick_test_all_*.out

# 2. 检查结果
python compare_results.py --results_dir results/quick_test --output_dir comparison_plots/quick_test

# 3. 完整训练
./submit_full_training.sh
squeue -u $USER  # 应该看到7个任务

# 4. 查看最终结果
python compare_results.py --results_dir results/full_training --output_dir comparison_plots/full_training
```

## 📊 7个实验方法

1. `no_gmm` - No GMM Baseline
2. `vanilla_gmm` - Vanilla GMM
3. `moe` - MoE GMM
4. `attention` - Attention GMM
5. `learnable_gmm` - Learnable GMM
6. `distortion_aware` - Distortion-Aware
7. `complete` - Complete Pipeline

## 🔍 监控命令

```bash
# 查看任务状态
squeue -u $USER

# 快速测试日志（1个文件）
tail -f logs/quick_test_all_*.out

# 完整训练日志（7个文件）
tail -f logs/full_moe_*.out
tail -f logs/full_complete_*.out

# 取消任务
scancel JOB_ID          # 取消单个
scancel -u $USER        # 取消所有
```

## 📁 输出结构

```
results/
├── quick_test/         # 快速测试结果
│   ├── no_gmm_results_*.json
│   └── ...
└── full_training/      # 完整训练结果
    ├── no_gmm_results_*.json
    └── ...

checkpoints/
├── quick_test/         # 快速测试模型
│   ├── no_gmm_best.pth
│   └── ...
└── full_training/      # 完整训练模型
    ├── no_gmm_best.pth
    └── ...

comparison_plots/
├── quick_test/
│   ├── comparison_table.csv
│   ├── training_comparison.png
│   └── final_comparison.png
└── full_training/
    ├── comparison_table.csv
    ├── training_comparison.png
    └── final_comparison.png
```

## ⚠️ 重要提示

1. **必须先运行快速测试**验证环境
2. 修改脚本中的Python路径为你的环境
3. 快速测试用 `sbatch` 提交，完整训练用 `./` 执行
4. 快速测试只有1个任务，完整训练有7个任务

## 📖 详细文档

- **QUICK_START.md** - 快速开始指南
- **HPC_TRAINING_GUIDE.md** - 详细使用文档
- **USAGE_SUMMARY.txt** - 使用总结
- **README_HPC.md** - 完整说明
