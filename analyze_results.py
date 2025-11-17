"""分析和可视化7个模型的对比结果"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(json_file):
    """加载结果JSON文件"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def print_summary_table(results_data):
    """打印结果摘要表格"""
    print("\n" + "="*100)
    print("实验结果摘要")
    print("="*100)

    # 打印配置
    config = results_data.get('config', {})
    print("\n配置信息:")
    print(f"  - 训练样本: {config.get('num_train', 'N/A')}")
    print(f"  - 验证样本: {config.get('num_val', 'N/A')}")
    print(f"  - Epochs: {config.get('epochs', 'N/A')}")
    print(f"  - Batch size: {config.get('batch_size', 'N/A')}")
    print(f"  - 学习率: {config.get('lr', 'N/A')}")
    print(f"  - 总耗时: {config.get('total_time_minutes', 0):.2f} 分钟")

    # 结果表格
    results = results_data['results']

    print("\n" + "-"*100)
    print(f"{'排名':<6} {'模型':<25} {'最佳SRCC':<12} {'最佳PLCC':<12} {'最佳Epoch':<12} {'相对提升':<12}")
    print("-"*100)

    # 按SRCC排序
    sorted_models = sorted(results.items(), key=lambda x: x[1]['best_srcc'], reverse=True)

    # 基线SRCC (NoGMM)
    baseline_srcc = None
    for name, res in sorted_models:
        if 'NoGMM' in name:
            baseline_srcc = res['best_srcc']
            break

    for rank, (model_name, res) in enumerate(sorted_models, 1):
        srcc = res['best_srcc']
        plcc = res['best_plcc']
        epoch = res['best_epoch']

        # 计算相对提升
        if baseline_srcc and baseline_srcc > 0:
            improvement = ((srcc - baseline_srcc) / baseline_srcc) * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "N/A"

        marker = "🏆" if rank == 1 else f"{rank}"
        print(f"{marker:<6} {model_name:<25} {srcc:<12.4f} {plcc:<12.4f} {epoch:<12} {improvement_str:<12}")

    print("-"*100)

    # 统计信息
    srrcs = [res['best_srcc'] for res in results.values()]
    plccs = [res['best_plcc'] for res in results.values()]

    print(f"\n统计信息:")
    print(f"  SRCC: 均值={np.mean(srrcs):.4f}, 标准差={np.std(srrcs):.4f}, "
          f"范围=[{np.min(srrcs):.4f}, {np.max(srrcs):.4f}]")
    print(f"  PLCC: 均值={np.mean(plccs):.4f}, 标准差={np.std(plccs):.4f}, "
          f"范围=[{np.min(plccs):.4f}, {np.max(plccs):.4f}]")


def plot_training_curves(results_data, output_dir='plots'):
    """绘制训练曲线"""
    Path(output_dir).mkdir(exist_ok=True)
    results = results_data['results']

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')

    # 1. Training Loss
    ax = axes[0, 0]
    for model_name, res in results.items():
        losses = res['train_losses']
        ax.plot(losses, label=model_name, linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Validation SRCC
    ax = axes[0, 1]
    for model_name, res in results.items():
        srrcs = res['val_srcc']
        ax.plot(srrcs, label=model_name, linewidth=2, alpha=0.8, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('SRCC', fontsize=12)
    ax.set_title('Validation SRCC', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # 3. Validation PLCC
    ax = axes[1, 0]
    for model_name, res in results.items():
        plccs = res['val_plcc']
        ax.plot(plccs, label=model_name, linewidth=2, alpha=0.8, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PLCC', fontsize=12)
    ax.set_title('Validation PLCC', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # 4. Best Scores Comparison
    ax = axes[1, 1]
    model_names = list(results.keys())
    best_srrcs = [results[m]['best_srcc'] for m in model_names]
    best_plccs = [results[m]['best_plcc'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, best_srrcs, width, label='Best SRCC', alpha=0.8)
    bars2 = ax.bar(x + width/2, best_plccs, width, label='Best PLCC', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Best Scores Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # 保存图片
    output_file = f"{output_dir}/training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 训练曲线已保存: {output_file}")

    # 关闭图形以释放内存
    plt.close()


def plot_performance_comparison(results_data, output_dir='plots'):
    """绘制性能对比图"""
    Path(output_dir).mkdir(exist_ok=True)
    results = results_data['results']

    # 按SRCC排序
    sorted_models = sorted(results.items(), key=lambda x: x[1]['best_srcc'], reverse=True)
    model_names = [name for name, _ in sorted_models]
    srrcs = [res['best_srcc'] for _, res in sorted_models]
    plccs = [res['best_plcc'] for _, res in sorted_models]

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance Ranking', fontsize=16, fontweight='bold')

    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

    # 1. SRCC排名
    bars1 = ax1.barh(model_names, srrcs, color=colors, alpha=0.8)
    ax1.set_xlabel('SRCC', fontsize=12)
    ax1.set_title('SRCC Ranking', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars1, srrcs)):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    # 标注排名
    for i, bar in enumerate(bars1):
        rank = i + 1
        marker = "🏆" if rank == 1 else f"#{rank}"
        ax1.text(0.005, bar.get_y() + bar.get_height()/2,
                marker, va='center', ha='left', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 2. PLCC排名
    sorted_by_plcc = sorted(results.items(), key=lambda x: x[1]['best_plcc'], reverse=True)
    plcc_names = [name for name, _ in sorted_by_plcc]
    plcc_values = [res['best_plcc'] for _, res in sorted_by_plcc]

    bars2 = ax2.barh(plcc_names, plcc_values, color=colors, alpha=0.8)
    ax2.set_xlabel('PLCC', fontsize=12)
    ax2.set_title('PLCC Ranking', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars2, plcc_values)):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    # 标注排名
    for i, bar in enumerate(bars2):
        rank = i + 1
        marker = "🏆" if rank == 1 else f"#{rank}"
        ax2.text(0.005, bar.get_y() + bar.get_height()/2,
                marker, va='center', ha='left', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # 保存
    output_file = f"{output_dir}/performance_ranking.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 性能排名已保存: {output_file}")

    plt.close()


def export_latex_table(results_data, output_dir='plots'):
    """导出LaTeX表格"""
    Path(output_dir).mkdir(exist_ok=True)
    results = results_data['results']

    # 按SRCC排序
    sorted_models = sorted(results.items(), key=lambda x: x[1]['best_srcc'], reverse=True)

    # 生成LaTeX表格
    latex = r"""\begin{table}[t]
\centering
\caption{Comparison of IQA Methods}
\label{tab:comparison}
\begin{tabular}{lccc}
\toprule
Model & SRCC $\uparrow$ & PLCC $\uparrow$ & Best Epoch \\
\midrule
"""

    for model_name, res in sorted_models:
        srcc = res['best_srcc']
        plcc = res['best_plcc']
        epoch = res['best_epoch']

        # 高亮最佳结果
        if res == sorted_models[0][1]:
            latex += f"\\textbf{{{model_name}}} & \\textbf{{{srcc:.4f}}} & \\textbf{{{plcc:.4f}}} & {epoch} \\\\\n"
        else:
            latex += f"{model_name} & {srcc:.4f} & {plcc:.4f} & {epoch} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # 保存
    output_file = f"{output_dir}/results_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"✓ LaTeX表格已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析IQA实验结果')
    parser.add_argument('result_file', type=str, help='结果JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='plots', help='输出目录')
    parser.add_argument('--no_plot', action='store_true', help='不生成图表')
    args = parser.parse_args()

    # 加载结果
    print(f"加载结果: {args.result_file}")
    data = load_results(args.result_file)

    # 打印摘要表格
    print_summary_table(data)

    # 生成可视化
    if not args.no_plot:
        try:
            print(f"\n生成可视化图表...")
            plot_training_curves(data, args.output_dir)
            plot_performance_comparison(data, args.output_dir)
            export_latex_table(data, args.output_dir)
            print(f"\n✓ 所有图表已保存到: {args.output_dir}/")
        except Exception as e:
            print(f"\n⚠ 生成图表失败: {e}")
            print("(可能是因为matplotlib未安装或在无GUI环境中)")

    print("\n分析完成！")


if __name__ == '__main__':
    main()
