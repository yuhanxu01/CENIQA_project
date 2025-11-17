"""
对比所有方法的训练结果
读取JSON文件并生成对比表格和可视化
"""
import os
import json
import argparse
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir):
    """加载所有结果JSON文件"""
    json_files = glob(os.path.join(results_dir, '*_results_*.json'))

    if not json_files:
        print(f"错误: 在 {results_dir} 中未找到结果文件")
        return None

    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)

    return results


def create_comparison_table(results):
    """创建对比表格"""
    method_display_names = {
        'no_gmm': 'No GMM (Baseline)',
        'vanilla_gmm': 'Vanilla GMM',
        'moe': 'MoE GMM',
        'attention': 'Attention GMM',
        'learnable_gmm': 'Learnable GMM',
        'distortion_aware': 'Distortion-Aware',
        'complete': 'Complete Pipeline'
    }

    data = []
    for r in results:
        method = r['method']
        data.append({
            '方法': method_display_names.get(method, method),
            '最佳SRCC': r['best_srcc'],
            '最佳PLCC': r['best_plcc'],
            '最佳Epoch': r['best_epoch'],
            '训练Epochs': r['epochs'],
            '训练集大小': r['train_size'],
            '验证集大小': r['val_size']
        })

    df = pd.DataFrame(data)
    df = df.sort_values('最佳SRCC', ascending=False)

    return df


def plot_training_curves(results, output_dir):
    """绘制训练曲线"""
    method_display_names = {
        'no_gmm': 'No GMM (Baseline)',
        'vanilla_gmm': 'Vanilla GMM',
        'moe': 'MoE GMM',
        'attention': 'Attention GMM',
        'learnable_gmm': 'Learnable GMM',
        'distortion_aware': 'Distortion-Aware',
        'complete': 'Complete Pipeline'
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Comparison', fontsize=16)

    # 训练损失
    ax1 = axes[0, 0]
    for r in results:
        method = method_display_names.get(r['method'], r['method'])
        epochs = range(1, len(r['train_losses']) + 1)
        ax1.plot(epochs, r['train_losses'], marker='o', label=method, markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 验证SRCC
    ax2 = axes[0, 1]
    for r in results:
        method = method_display_names.get(r['method'], r['method'])
        epochs = range(1, len(r['val_srcc']) + 1)
        ax2.plot(epochs, r['val_srcc'], marker='o', label=method, markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation SRCC')
    ax2.set_title('Validation SRCC')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 验证PLCC
    ax3 = axes[1, 0]
    for r in results:
        method = method_display_names.get(r['method'], r['method'])
        epochs = range(1, len(r['val_plcc']) + 1)
        ax3.plot(epochs, r['val_plcc'], marker='o', label=method, markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation PLCC')
    ax3.set_title('Validation PLCC')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 最佳SRCC对比条形图
    ax4 = axes[1, 1]
    methods = [method_display_names.get(r['method'], r['method']) for r in results]
    best_srccs = [r['best_srcc'] for r in results]

    # 按SRCC排序
    sorted_data = sorted(zip(methods, best_srccs), key=lambda x: x[1], reverse=True)
    methods_sorted, srccs_sorted = zip(*sorted_data)

    colors = plt.cm.viridis(range(len(methods_sorted)))
    bars = ax4.barh(methods_sorted, srccs_sorted, color=colors)
    ax4.set_xlabel('Best SRCC')
    ax4.set_title('Best SRCC Comparison')
    ax4.grid(True, alpha=0.3, axis='x')

    # 在条形上添加数值
    for i, (bar, val) in enumerate(zip(bars, srccs_sorted)):
        ax4.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()

    # 保存图像
    plot_path = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {plot_path}")

    return plot_path


def plot_final_comparison(results, output_dir):
    """绘制最终结果对比"""
    method_display_names = {
        'no_gmm': 'No GMM',
        'vanilla_gmm': 'Vanilla GMM',
        'moe': 'MoE',
        'attention': 'Attention',
        'learnable_gmm': 'Learnable',
        'distortion_aware': 'Distortion',
        'complete': 'Complete'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = [method_display_names.get(r['method'], r['method']) for r in results]
    srccs = [r['best_srcc'] for r in results]
    plccs = [r['best_plcc'] for r in results]

    # 按SRCC排序
    sorted_indices = sorted(range(len(srccs)), key=lambda i: srccs[i], reverse=True)
    methods_sorted = [methods[i] for i in sorted_indices]
    srccs_sorted = [srccs[i] for i in sorted_indices]
    plccs_sorted = [plccs[i] for i in sorted_indices]

    x = range(len(methods_sorted))
    width = 0.35

    # SRCC对比
    ax1 = axes[0]
    bars1 = ax1.bar([i - width/2 for i in x], srccs_sorted, width, label='SRCC', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], plccs_sorted, width, label='PLCC', alpha=0.8)

    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Correlation', fontsize=12)
    ax1.set_title('Best SRCC & PLCC Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_sorted, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 在条形上添加数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    # 散点图：SRCC vs PLCC
    ax2 = axes[1]
    colors = plt.cm.tab10(range(len(methods_sorted)))

    for i, (method, srcc, plcc) in enumerate(zip(methods_sorted, srccs_sorted, plccs_sorted)):
        ax2.scatter(srcc, plcc, s=200, c=[colors[i]], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax2.annotate(method, (srcc, plcc), fontsize=9, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    ax2.set_xlabel('Best SRCC', fontsize=12)
    ax2.set_ylabel('Best PLCC', fontsize=12)
    ax2.set_title('SRCC vs PLCC', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 添加对角线参考
    min_val = min(min(srccs), min(plccs))
    max_val = max(max(srccs), max(plccs))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)

    plt.tight_layout()

    # 保存图像
    plot_path = os.path.join(output_dir, 'final_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"最终对比图已保存到: {plot_path}")

    return plot_path


def main():
    parser = argparse.ArgumentParser(description='对比所有方法的训练结果')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='结果目录路径')
    parser.add_argument('--output_dir', type=str, default='comparison_plots',
                      help='图像输出目录')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("加载结果文件...")
    print("="*80)

    # 加载结果
    results = load_results(args.results_dir)

    if not results:
        return

    print(f"\n找到 {len(results)} 个方法的结果\n")

    # 创建对比表格
    print("="*80)
    print("结果对比表格")
    print("="*80)

    df = create_comparison_table(results)
    print(df.to_string(index=False))

    # 保存表格
    table_path = os.path.join(args.output_dir, 'comparison_table.csv')
    df.to_csv(table_path, index=False)
    print(f"\n表格已保存到: {table_path}")

    # 绘制训练曲线
    print("\n" + "="*80)
    print("生成训练曲线对比图...")
    print("="*80)
    plot_training_curves(results, args.output_dir)

    # 绘制最终对比
    print("\n" + "="*80)
    print("生成最终结果对比图...")
    print("="*80)
    plot_final_comparison(results, args.output_dir)

    # 显示排名
    print("\n" + "="*80)
    print("方法排名（按SRCC）")
    print("="*80)

    for i, row in df.iterrows():
        print(f"{i+1}. {row['方法']:<25} SRCC={row['最佳SRCC']:.4f}  PLCC={row['最佳PLCC']:.4f}  (Epoch {row['最佳Epoch']})")

    print("\n" + "="*80)
    print("对比完成！")
    print("="*80)


if __name__ == '__main__':
    main()
