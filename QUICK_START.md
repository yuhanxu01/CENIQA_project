# 快速开始指南

## 🎯 第一步：快速测试（必须先做！）

在运行完整实验之前，**必须**先运行快速测试来验证代码和环境：

```bash
cd /gpfs/scratch/rl5285/CENIQA_project  # 进入项目目录

# 1. 创建必要目录
mkdir -p logs results checkpoints

# 2. 提交快速测试（1个node，7个方法串行，2 epochs）
sbatch submit_quick_test.sh
```

**快速测试说明：**
- 申请1个GPU node
- 7个方法串行运行（一个接一个）
- 每个方法2 epochs
- 总时间约3-5小时

### 监控快速测试

```bash
# 查看任务状态
squeue -u $USER

# 实时查看所有方法的输出（在一个日志文件中）
tail -f logs/quick_test_all_*.out
```

### 检查快速测试结果

等待所有7个任务完成（约3-5小时），然后：

```bash
# 对比快速测试结果
python compare_results.py \
    --results_dir results/quick_test \
    --output_dir comparison_plots/quick_test

# 查看对比表格
cat comparison_plots/quick_test/comparison_table.csv
```

**✅ 确认检查点：**
- [ ] 所有7个任务成功完成（无错误）
- [ ] 训练损失正常下降
- [ ] SRCC值合理（通常 > 0.1）
- [ ] 无CUDA或内存错误

## 🚀 第二步：完整训练

快速测试通过后，运行完整训练：

```bash
# 提交完整训练（7个node并行，每个方法60 epochs）
./submit_full_training.sh
```

**完整训练说明：**
- 申请7个GPU nodes
- 7个方法并行运行（同时运行）
- 每个方法60 epochs
- 总时间约8-12小时

### 监控完整训练

```bash
# 查看任务状态（应该看到7个任务）
squeue -u $USER

# 查看特定方法的训练进度
tail -f logs/full_moe_*.out
tail -f logs/full_complete_*.out

# 查看已保存的模型
ls -lh checkpoints/full_training/*.pth
```

### 查看最终结果

等待所有任务完成（约8-12小时），然后：

```bash
# 生成对比报告
python compare_results.py \
    --results_dir results/full_training \
    --output_dir comparison_plots/full_training

# 查看排名
cat comparison_plots/full_training/comparison_table.csv
```

## 📊 7个方法说明

| 方法 | 说明 | 关键特点 |
|------|------|----------|
| **no_gmm** | No GMM Baseline | 直接回归，不使用GMM |
| **vanilla_gmm** | Vanilla GMM | 标准GMM + 特征拼接 |
| **moe** | MoE GMM | 每个cluster一个expert |
| **attention** | Attention GMM | Attention机制融合 |
| **learnable_gmm** | Learnable GMM | 可学习的GMM参数 |
| **distortion_aware** | Distortion-Aware | 显式建模失真类型 |
| **complete** | Complete Pipeline | 综合所有改进 |

## 🔧 手动运行单个方法

如果只想训练某个特定方法：

```bash
# 快速测试模式
python train_single_method.py --method moe --quick_test

# 完整训练模式
python train_single_method.py \
    --method moe \
    --epochs 60 \
    --batch_size 16 \
    --output_dir results/full_training \
    --checkpoint_dir checkpoints/full_training
```

## 📈 预期输出

### 快速测试（2 epochs）
```
方法                        最佳SRCC       最佳PLCC
---------------------------------------------------------
Complete Pipeline          0.3-0.5        0.3-0.5
Learnable GMM              0.3-0.5        0.3-0.5
MoE GMM                    0.3-0.4        0.3-0.4
Vanilla GMM                0.2-0.4        0.2-0.4
Attention GMM              0.2-0.3        0.2-0.3
Distortion-Aware           0.1-0.3        0.1-0.3
No GMM (Baseline)          0.1-0.3        0.1-0.3
```

### 完整训练（60 epochs）
```
方法                        最佳SRCC       最佳PLCC
---------------------------------------------------------
Complete Pipeline          0.7-0.85       0.7-0.85
Learnable GMM              0.65-0.80      0.65-0.80
MoE GMM                    0.60-0.75      0.60-0.75
Distortion-Aware           0.60-0.75      0.60-0.75
Attention GMM              0.55-0.70      0.55-0.70
Vanilla GMM                0.50-0.65      0.50-0.65
No GMM (Baseline)          0.45-0.60      0.45-0.60
```

（实际结果可能有所不同）

## ✅ 完整流程总结

```bash
# 1. 快速测试（1个node串行，必须先做！）
sbatch submit_quick_test.sh
# 等待3-5小时，7个方法串行运行

# 2. 检查快速测试结果
python compare_results.py --results_dir results/quick_test --output_dir comparison_plots/quick_test

# 3. 确认无误后，开始完整训练（7个node并行）
./submit_full_training.sh
# 等待8-12小时，7个方法并行运行

# 4. 查看最终结果
python compare_results.py --results_dir results/full_training --output_dir comparison_plots/full_training
```

就这么简单！🎉
