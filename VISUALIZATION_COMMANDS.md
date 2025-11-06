# 可视化命令速查表 (Quick Command Reference)

## 🚀 最简单的使用方式

如果你的代码已经训练完成，只需运行：

```bash
python run_visualization.py
```

这会自动：
- ✅ 找到你最新的实验
- ✅ 加载训练好的模型
- ✅ 生成所有可视化图表
- ✅ 保存结果到 `experiments/你的实验名/enhanced_visualizations/`

---

## 📋 所有可用命令

### 1️⃣ 基础可视化（推荐）

```bash
# 自动检测并可视化（最简单）
python run_visualization.py

# 显示更多图片
python run_visualization.py --num_images 50

# 使用更多测试样本（更准确但更慢）
python run_visualization.py --num_images 50 --test_samples 1000
```

### 2️⃣ 高级可视化（更多控制）

```bash
# 完整命令示例
python enhanced_visualize.py \
    --experiment experiments/resnet18_gmm_mlp \
    --checkpoint best_model.pth \
    --num_images 25 \
    --test_samples 500 \
    --batch_size 128

# 使用不同的checkpoint
python enhanced_visualize.py \
    --experiment experiments/resnet18_gmm_mlp \
    --checkpoint final_model.pth \
    --num_images 25

# 可视化所有测试数据（慢但最准确）
python enhanced_visualize.py \
    --experiment experiments/resnet18_gmm_mlp \
    --num_images 100 \
    --test_samples 5000
```

### 3️⃣ 原始可视化工具

```bash
# 使用原始的test_with_viz.py（功能较少）
python test_with_viz.py \
    --experiment experiments/resnet18_gmm_mlp \
    --test_samples 500 \
    --skip_tsne
```

---

## 📊 生成的可视化文件说明

运行后会在 `experiments/你的实验名/enhanced_visualizations/` 生成：

### 📄 comprehensive_metrics.png
**综合性能仪表盘** - 一张图看懂所有指标：
- 整体性能指标（SRCC, PLCC, RMSE, MAE）
- 预测散点图（预测值 vs 真实值）
- 误差分布直方图
- 聚类分布柱状图
- 各聚类准确率对比
- 质量分数分布箱线图
- 详细统计表格

### 🖼️ image_grid_detailed.png
**详细图片网格** - 看到具体图片的表现：
- 每张图片显示：
  - ✨ 原始图像
  - 🎯 预测质量分数
  - ✅ 真实质量分数
  - 🔢 所属Cluster ID
  - 💯 Cluster置信度
  - ❌ 预测误差
- 颜色编码：
  - 🟢 绿色 = 误差小（< 0.1）
  - 🟠 橙色 = 误差中等（0.1-0.2）
  - 🔴 红色 = 误差大（> 0.2）

### 🎨 cluster_examples.png
**聚类代表样本** - 理解每个cluster的特点：
- 每个cluster显示5个代表性样本
- 帮助理解模型如何对图片分组
- 发现哪类图片预测得好/不好

---

## 🎯 常用场景

### 场景1: 刚训练完，想快速看结果
```bash
python run_visualization.py
```

### 场景2: 想看更多图片样例
```bash
python run_visualization.py --num_images 50
```

### 场景3: 需要最准确的评估（会慢一些）
```bash
python run_visualization.py --num_images 100 --test_samples 2000
```

### 场景4: 内存不够，需要小batch
```bash
python enhanced_visualize.py \
    --experiment experiments/resnet18_gmm_mlp \
    --batch_size 32
```

### 场景5: 比较不同的checkpoint
```bash
# 看best_model的表现
python enhanced_visualize.py --experiment experiments/resnet18_gmm_mlp --checkpoint best_model.pth

# 看final_model的表现
python enhanced_visualize.py --experiment experiments/resnet18_gmm_mlp --checkpoint final_model.pth
```

---

## 🔧 参数详解

| 参数 | 默认值 | 说明 | 建议 |
|------|--------|------|------|
| `--experiment` | 自动检测 | 实验目录路径 | 不指定会自动找最新的 |
| `--checkpoint` | best_model.pth | 模型文件名 | 通常用best_model.pth |
| `--num_images` | 25 | 网格显示图片数 | 25-50适中，100以上会很大 |
| `--test_samples` | 500 | 测试样本总数 | 500快速，1000-2000准确 |
| `--batch_size` | 128 | 推理批次大小 | GPU内存小用32或64 |

---

## 📈 如何解读结果

### ✅ 好的结果应该是：

1. **SRCC > 0.85**: 预测排序很准确
2. **PLCC > 0.85**: 预测值很接近真实值
3. **RMSE < 0.15**: 误差很小
4. **误差分布**: 集中在0附近，呈正态分布
5. **各cluster准确率**: 差距不大，都>0.7
6. **绿色边框图片多**: 说明大部分预测都很准

### ⚠️ 需要改进的迹象：

1. **SRCC < 0.7**: 模型需要改进
2. **某个cluster准确率很低**: 该类图片处理不好
3. **误差分布偏斜**: 存在系统性偏差
4. **红色边框图片多**: 很多预测不准

---

## 🐛 常见问题

### Q: 运行报错 "No experiments directory found"
```bash
# 确保在项目根目录运行
cd /path/to/CENIQA_project
python run_visualization.py

# 或手动指定
python enhanced_visualize.py --experiment /完整/路径/to/experiment
```

### Q: CUDA out of memory
```bash
# 减小batch size
python enhanced_visualize.py --experiment experiments/xxx --batch_size 32

# 或减少测试样本
python run_visualization.py --test_samples 200
```

### Q: 想保存更高质量的图片
生成的PNG已经是300 DPI高质量，直接可用于论文/报告

### Q: 能在Colab运行吗？
可以！在Colab中运行：
```python
!python run_visualization.py
```

### Q: 可视化在哪里？
```
experiments/
└── 你的实验名/
    └── enhanced_visualizations/
        ├── comprehensive_metrics.png      # 综合指标
        ├── image_grid_detailed.png        # 图片网格
        └── cluster_examples.png           # 聚类样本
```

---

## 💡 Pro Tips

### Tip 1: 批量可视化多个实验
```bash
#!/bin/bash
for exp in experiments/*/; do
    echo "Processing $exp"
    python enhanced_visualize.py --experiment "$exp" --num_images 25
done
```

### Tip 2: 快速对比两个模型
```bash
# 可视化模型A
python enhanced_visualize.py --experiment experiments/modelA --num_images 50

# 可视化模型B
python enhanced_visualize.py --experiment experiments/modelB --num_images 50

# 对比两个目录下的 comprehensive_metrics.png
```

### Tip 3: 只看几个cluster的样本
编辑 `enhanced_visualize.py`，修改 `select_diverse_samples` 函数

### Tip 4: 生成PPT友好的图片
图片已经是高分辨率（300 DPI），直接插入PPT即可

---

## 📚 更多信息

- 详细指南：`VISUALIZATION_GUIDE_CN.md`
- 原始可视化工具：`visualize.py`
- 测试脚本：`test_with_viz.py`

---

## ⌨️ 复制即用的命令

```bash
# === 最常用的3个命令 ===

# 1. 快速可视化（最简单）
python run_visualization.py

# 2. 显示更多图片
python run_visualization.py --num_images 50

# 3. 完整评估（最准确）
python run_visualization.py --num_images 100 --test_samples 1000
```

**就是这么简单！** 🎉

需要更多帮助？查看 `VISUALIZATION_GUIDE_CN.md` 获取完整文档。
