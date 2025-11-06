# 可视化使用指南 (Visualization Guide)

本指南介绍如何使用增强版可视化工具来查看你的图像质量评估模型的结果。

## 快速开始 🚀

### 方法 1: 自动运行（推荐）

```bash
# 自动检测最新的实验并运行可视化
python run_visualization.py

# 显示更多图片（默认25张）
python run_visualization.py --num_images 50

# 使用更多测试样本
python run_visualization.py --test_samples 1000
```

### 方法 2: 手动指定实验

```bash
# 指定实验目录
python enhanced_visualize.py --experiment experiments/resnet18_gmm_mlp --num_images 25

# 完整参数示例
python enhanced_visualize.py \
    --experiment experiments/resnet18_gmm_mlp \
    --checkpoint best_model.pth \
    --num_images 25 \
    --test_samples 500 \
    --batch_size 128
```

## 生成的可视化内容 📊

运行完成后，会在 `experiments/你的实验名/enhanced_visualizations/` 目录下生成以下文件：

### 1. comprehensive_metrics.png - 综合性能指标面板

包含以下信息：

- **整体性能指标**
  - SRCC (Spearman 相关系数): 衡量预测排序的准确性
  - PLCC (Pearson 相关系数): 衡量线性相关性
  - RMSE (均方根误差): 预测误差大小
  - MAE (平均绝对误差): 平均误差

- **预测散点图**: 显示预测值 vs 真实值
- **误差分布**: 显示预测误差的分布情况
- **聚类分布**: 显示各个cluster的样本数量
- **各cluster准确率**: 显示每个cluster的SRCC指标
- **质量分数分布**: 每个cluster的质量评分分布
- **详细统计表**: 包含每个cluster的详细统计信息

### 2. image_grid_detailed.png - 详细图片网格

显示具体的图片样本，每张图片包含：

- **原始图片**: 实际的输入图像
- **Predicted (预测分数)**: 模型预测的质量评分
- **Ground Truth (真实分数)**: 数据集标注的真实质量评分
- **Cluster (所属聚类)**: 图片被分配到哪个cluster
- **Confidence (置信度)**: 模型对该cluster分配的置信度
- **Error (误差)**: 预测值与真实值的差距

**颜色标识**：
- 🟢 **绿色边框**: 误差 < 0.1 (预测很准确)
- 🟠 **橙色边框**: 0.1 ≤ 误差 < 0.2 (预测一般)
- 🔴 **红色边框**: 误差 ≥ 0.2 (预测误差较大)

### 3. cluster_examples.png - 各聚类代表样本

显示每个cluster的代表性样本，帮助你理解：
- 每个cluster包含什么类型的图片
- 各cluster的质量分布特征
- 模型对不同cluster的预测表现

## 命令参数说明 ⚙️

### run_visualization.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--experiment` | 自动检测 | 指定实验目录路径 |
| `--num_images` | 25 | 在网格中显示的图片数量 |
| `--test_samples` | 500 | 用于评估的测试样本总数 |

### enhanced_visualize.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--experiment` | **必需** | 实验目录路径 |
| `--checkpoint` | best_model.pth | 模型checkpoint文件名 |
| `--num_images` | 25 | 详细显示的图片数量 |
| `--test_samples` | 500 | 评估的测试样本数 |
| `--batch_size` | 128 | 推理时的批次大小 |

## 使用场景示例 💡

### 场景 1: 快速查看训练结果

```bash
# 训练完成后立即可视化
python run_visualization.py
```

### 场景 2: 详细分析特定实验

```bash
# 显示更多图片，全面评估
python run_visualization.py --num_images 100 --test_samples 1000
```

### 场景 3: 比较多个实验

```bash
# 可视化实验1
python enhanced_visualize.py --experiment experiments/resnet18_gmm_mlp --num_images 25

# 可视化实验2
python enhanced_visualize.py --experiment experiments/resnet34_gmm_mlp --num_images 25

# 然后比较两个实验的 enhanced_visualizations 目录
```

### 场景 4: 分析特定checkpoint

```bash
# 查看最终模型的表现
python enhanced_visualize.py \
    --experiment experiments/resnet18_gmm_mlp \
    --checkpoint final_model.pth \
    --num_images 50
```

## 理解可视化结果 📖

### 如何评估模型性能？

1. **查看整体指标** (comprehensive_metrics.png)
   - SRCC > 0.9: 非常好
   - SRCC > 0.8: 良好
   - SRCC > 0.7: 一般
   - PLCC 和 SRCC 应该接近

2. **检查预测散点图**
   - 点应该聚集在红色虚线（完美预测线）附近
   - 离散程度越小越好

3. **分析误差分布**
   - 应该以0为中心呈正态分布
   - 如果偏向一边，说明模型有系统性偏差

4. **观察cluster分布**
   - 各cluster样本数不应过度不均衡
   - 每个cluster应该有足够的样本

5. **检查各cluster准确率**
   - 各cluster的SRCC不应差距过大
   - 如果某个cluster准确率很低，说明模型在该类型图片上表现较差

### 如何理解cluster？

模型使用GMM (Gaussian Mixture Model) 将图片分为不同的clusters，每个cluster代表一类具有相似特征的图片。通过查看 `cluster_examples.png`，你可以：

1. 了解每个cluster包含什么样的图片
2. 发现模型是如何对图片进行分组的
3. 找出哪些类型的图片模型预测得好/不好

## 故障排除 🔧

### 问题1: 找不到实验目录

```
Error: No 'experiments' directory found!
```

**解决方法**: 确保已经运行过训练，或者手动指定实验目录：
```bash
python enhanced_visualize.py --experiment path/to/your/experiment
```

### 问题2: 找不到checkpoint

```
Error: No checkpoint found in experiment directory!
```

**解决方法**:
- 确认实验目录中有 `best_model.pth`, `checkpoint.pth` 或 `final_model.pth`
- 或者手动指定checkpoint文件名：
```bash
python enhanced_visualize.py --experiment experiments/xxx --checkpoint your_model.pth
```

### 问题3: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解决方法**: 减小batch size
```bash
python enhanced_visualize.py --experiment experiments/xxx --batch_size 32
```

### 问题4: 可视化图片太小看不清

**解决方法**:
1. 直接打开生成的PNG文件（高分辨率，300 DPI）
2. 减少显示的图片数量，使每张图更大：
```bash
python run_visualization.py --num_images 16  # 4x4网格，每张图更大
```

## 高级用法 🎓

### 自定义可视化

如果你想修改可视化样式，可以编辑 `enhanced_visualize.py` 文件：

- 修改颜色方案: 查找 `colors_cluster` 相关代码
- 修改图片大小: 修改 `figsize` 参数
- 调整误差阈值: 修改 `if error < 0.1` 等条件
- 添加新的图表: 在 `plot_comprehensive_metrics` 函数中添加

### 批量可视化

如果你有多个实验要可视化：

```bash
# 创建批量脚本 batch_visualize.sh
for exp in experiments/*/; do
    echo "Visualizing $exp"
    python enhanced_visualize.py --experiment "$exp" --num_images 25
done
```

### 导出结果用于报告

生成的PNG文件是高质量的（300 DPI），可以直接用于：
- 学术论文
- 技术报告
- 演示文稿（PPT）
- 项目文档

## 常见问题 FAQ ❓

**Q: 可视化需要多长时间？**
A: 通常1-5分钟，取决于：
- 测试样本数量（--test_samples）
- 显示图片数量（--num_images）
- GPU/CPU性能

**Q: 可以在没有GPU的机器上运行吗？**
A: 可以，会自动使用CPU，但速度会慢一些。

**Q: 生成的图片在哪里？**
A: 在 `experiments/你的实验名/enhanced_visualizations/` 目录下

**Q: 可以改变显示的图片吗？**
A: 脚本会随机选择具有代表性的样本。如果想固定随机种子，可以在代码中添加 `np.random.seed(42)`

**Q: SRCC、PLCC是什么意思？**
A:
- SRCC (Spearman Rank Correlation Coefficient): 衡量预测排序与真实排序的一致性
- PLCC (Pearson Linear Correlation Coefficient): 衡量预测值与真实值的线性相关程度
- 两者都越接近1越好

## 示例输出说明 📝

运行成功后，你会看到类似的输出：

```
================================================================================
Enhanced Image Quality Assessment Visualization
================================================================================
Experiment: resnet18_gmm_mlp
Checkpoint: best_model.pth
Number of images to visualize: 25
================================================================================

Device: cuda

Loading model...
Model loaded: resnet18

Loading test dataset...

Running inference...
100%|████████████████████████| 4/4 [00:02<00:00,  1.82it/s]

================================================================================
Generating Enhanced Visualizations
================================================================================

1. Generating comprehensive metrics dashboard...
Saved: experiments/resnet18_gmm_mlp/enhanced_visualizations/comprehensive_metrics.png

2. Generating image grid with 25 samples...
Saved: experiments/resnet18_gmm_mlp/enhanced_visualizations/image_grid_detailed.png

3. Generating cluster-wise examples...
Saved: experiments/resnet18_gmm_mlp/enhanced_visualizations/cluster_examples.png

================================================================================
Visualization Complete!
================================================================================

All visualizations saved to: experiments/resnet18_gmm_mlp/enhanced_visualizations

Generated files:
  - comprehensive_metrics.png : Complete performance dashboard
  - image_grid_detailed.png   : 25 images with scores and clusters
  - cluster_examples.png      : Representative samples from each cluster
================================================================================
```

## 技术支持 📧

如有问题，请检查：
1. 本文档的"故障排除"部分
2. 项目的其他README文件
3. 确保所有依赖包已正确安装（见 requirements.txt）

---

**祝你使用愉快！** 🎉
