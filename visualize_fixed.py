"""
重写的可视化脚本 - 修复所有问题
支持 DistortedImageDataset，正确的聚类分布，清晰的图像显示

Usage:
    python visualize_fixed.py --experiment experiments/resnet18_large_10k --checkpoint best_model.pth
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from train_gpu import SimpleCNNGMMMLPModel
from distorted_dataset import DistortedImageDataset


def denormalize_image(tensor):
    """
    反归一化图像用于显示
    ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    if tensor.device != 'cpu':
        tensor = tensor.cpu()

    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


def load_model_from_checkpoint(checkpoint_path, device):
    """从检查点加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # 推断模型配置
    feature_dim = state_dict['backbone.proj.weight'].shape[0]
    n_clusters = state_dict['gmm.means'].shape[0]

    if 'regressor.fc1.weight' in state_dict:
        hidden_dim = state_dict['regressor.fc1.weight'].shape[0]
    else:
        hidden_dim = 512

    print(f"Model config inferred:")
    print(f"  - feature_dim: {feature_dim}")
    print(f"  - n_clusters: {n_clusters}")
    print(f"  - hidden_dim: {hidden_dim}")

    # 创建模型
    model = SimpleCNNGMMMLPModel(
        backbone_name='resnet18',
        feature_dim=feature_dim,
        n_clusters=n_clusters,
        hidden_dim=hidden_dim,
        dropout=0.3,
        freeze_backbone=False
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, n_clusters


def run_inference(model, dataloader, device):
    """运行推断并收集结果"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_posteriors = []
    all_images = []
    all_distortion_types = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            if len(batch) == 3:
                images, scores, distortion_types = batch
            else:
                images, scores = batch
                distortion_types = None

            images = images.to(device)

            # 前向传播
            outputs = model(images, return_all=True)

            # 收集结果
            all_predictions.append(outputs['quality_score'].cpu().numpy())
            all_targets.append(scores.numpy())
            all_posteriors.append(outputs['posteriors'].cpu().numpy())
            all_images.append(images.cpu())

            if distortion_types is not None:
                all_distortion_types.extend(distortion_types)

    # 合并所有批次
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    posteriors = np.concatenate(all_posteriors)
    images = torch.cat(all_images, dim=0)

    # 获取聚类分配
    cluster_ids = np.argmax(posteriors, axis=1)

    return {
        'predictions': predictions,
        'targets': targets,
        'posteriors': posteriors,
        'cluster_ids': cluster_ids,
        'images': images,
        'distortion_types': all_distortion_types if all_distortion_types else None
    }


def select_diverse_samples(cluster_ids, predictions, targets, n_samples=25):
    """
    选择多样化的样本，确保覆盖所有聚类
    """
    n_clusters = len(np.unique(cluster_ids))
    samples_per_cluster = max(1, n_samples // n_clusters)

    selected_indices = []

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_ids == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            print(f"Warning: Cluster {cluster_id} has no samples")
            continue

        # 从每个聚类中选择样本，尽量覆盖不同的质量范围
        n_select = min(samples_per_cluster, len(cluster_indices))

        # 按预测分数排序，选择均匀分布的样本
        cluster_preds = predictions[cluster_indices]
        sorted_idx = cluster_indices[np.argsort(cluster_preds)]

        # 均匀采样
        step = len(sorted_idx) // n_select if n_select > 0 else 1
        selected = sorted_idx[::step][:n_select]

        selected_indices.extend(selected.tolist())

    # 如果样本数不够，随机补充
    while len(selected_indices) < n_samples and len(selected_indices) < len(predictions):
        remaining = list(set(range(len(predictions))) - set(selected_indices))
        if remaining:
            selected_indices.append(np.random.choice(remaining))
        else:
            break

    return np.array(selected_indices[:n_samples])


def plot_image_grid(images, predictions, targets, cluster_ids, posteriors,
                    distortion_types=None, save_path=None):
    """
    绘制图像网格，显示详细信息
    """
    n_images = len(images)
    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if idx < n_images:
            # 显示图像
            img = denormalize_image(images[idx])
            ax.imshow(img)
            ax.axis('off')

            # 获取信息
            pred = predictions[idx]
            target = targets[idx]
            cluster = cluster_ids[idx]
            confidence = posteriors[idx][cluster]
            error = abs(pred - target)

            # 颜色编码：根据误差
            if error < 0.1:
                color = 'green'
            elif error < 0.2:
                color = 'orange'
            else:
                color = 'red'

            # 标题
            title_parts = [
                f'Pred: {pred:.3f}',
                f'GT: {target:.3f}',
                f'Cluster: {cluster}',
                f'Conf: {confidence:.2f}',
                f'Err: {error:.3f}'
            ]

            if distortion_types and idx < len(distortion_types):
                dist_type = distortion_types[idx]
                if dist_type != 'pristine':
                    title_parts.insert(0, f'[{dist_type}]')

            title = '\n'.join(title_parts)
            ax.set_title(title, fontsize=8, color=color, weight='bold', pad=5)

            # 边框
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)
        else:
            ax.axis('off')

    plt.suptitle(
        'Image Quality Assessment - Diverse Samples from All Clusters\n'
        'Green: Error < 0.1 | Orange: 0.1 ≤ Error < 0.2 | Red: Error ≥ 0.2',
        fontsize=14, weight='bold', y=1.0
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    plt.close()


def plot_cluster_examples(images, predictions, targets, cluster_ids, posteriors,
                          n_clusters, n_per_cluster=6, save_path=None):
    """
    每个聚类显示代表性样本
    """
    fig, axes = plt.subplots(n_clusters, n_per_cluster,
                            figsize=(n_per_cluster*2.5, n_clusters*2.5))

    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_ids == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        cluster_count = len(cluster_indices)

        if cluster_count == 0:
            # 空聚类
            for i in range(n_per_cluster):
                ax = axes[cluster_id, i] if n_clusters > 1 else axes[i]
                ax.text(0.5, 0.5, 'No\nSamples', ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            continue

        # 选择代表性样本
        n_select = min(n_per_cluster, cluster_count)

        # 按质量分数排序后均匀选择
        cluster_preds = predictions[cluster_indices]
        sorted_idx = cluster_indices[np.argsort(cluster_preds)]

        if cluster_count >= n_per_cluster:
            step = cluster_count // n_per_cluster
            selected = sorted_idx[::step][:n_per_cluster]
        else:
            selected = sorted_idx

        for i in range(n_per_cluster):
            ax = axes[cluster_id, i] if n_clusters > 1 else axes[i]

            if i < len(selected):
                idx = selected[i]

                # 显示图像
                img = denormalize_image(images[idx])
                ax.imshow(img)
                ax.axis('off')

                # 信息
                pred = predictions[idx]
                target = targets[idx]
                conf = posteriors[idx][cluster_id]
                error = abs(pred - target)

                color = 'green' if error < 0.1 else 'orange' if error < 0.2 else 'red'

                title = f'P:{pred:.2f} T:{target:.2f}\nConf:{conf:.2f}'
                ax.set_title(title, fontsize=8, color=color)

                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)
            else:
                ax.axis('off')

        # 聚类标签（在第一个子图的左侧）
        ax = axes[cluster_id, 0] if n_clusters > 1 else axes[0]
        cluster_avg_quality = targets[cluster_mask].mean()
        cluster_avg_pred = predictions[cluster_mask].mean()
        ax.text(-0.15, 0.5,
                f'Cluster {cluster_id}\n({cluster_count} samples)\nAvg Q: {cluster_avg_quality:.2f}\nAvg P: {cluster_avg_pred:.2f}',
                transform=ax.transAxes, ha='right', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.suptitle(f'Representative Samples from Each Cluster',
                fontsize=14, weight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    plt.close()


def plot_performance_metrics(predictions, targets, cluster_ids, save_path=None):
    """
    性能指标可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 计算总体指标
    srcc, _ = spearmanr(predictions, targets)
    plcc, _ = pearsonr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))

    # 1. 散点图：预测 vs 真实
    ax = axes[0, 0]
    ax.scatter(targets, predictions, alpha=0.5, s=30, c='steelblue', edgecolors='k', linewidth=0.5)

    # 对角线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('Ground Truth Score', fontsize=12, weight='bold')
    ax.set_ylabel('Predicted Score', fontsize=12, weight='bold')
    ax.set_title('Predictions vs Ground Truth', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加指标文本
    metrics_text = f'SRCC: {srcc:.4f}\nPLCC: {plcc:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2. 误差分布
    ax = axes[0, 1]
    errors = predictions - targets
    ax.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {errors.mean():.4f}')
    ax.set_xlabel('Prediction Error', fontsize=12, weight='bold')
    ax.set_ylabel('Frequency', fontsize=12, weight='bold')
    ax.set_title('Error Distribution', fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. 每个聚类的性能
    ax = axes[1, 0]
    n_clusters = len(np.unique(cluster_ids))
    cluster_metrics = []

    for i in range(n_clusters):
        mask = cluster_ids == i
        if mask.sum() > 0:
            cluster_preds = predictions[mask]
            cluster_targets = targets[mask]
            cluster_srcc, _ = spearmanr(cluster_preds, cluster_targets)
            cluster_metrics.append((i, mask.sum(), cluster_srcc))

    cluster_ids_plot = [x[0] for x in cluster_metrics]
    cluster_counts = [x[1] for x in cluster_metrics]
    cluster_srccs = [x[2] for x in cluster_metrics]

    x_pos = np.arange(len(cluster_ids_plot))
    bars = ax.bar(x_pos, cluster_srccs, color='steelblue', alpha=0.7, edgecolor='black')

    # 颜色编码
    for i, (bar, srcc_val) in enumerate(zip(bars, cluster_srccs)):
        if srcc_val >= 0.7:
            bar.set_color('green')
        elif srcc_val >= 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax.set_xlabel('Cluster ID', fontsize=12, weight='bold')
    ax.set_ylabel('SRCC', fontsize=12, weight='bold')
    ax.set_title('Per-Cluster Performance', fontsize=13, weight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{cid}\n({cnt})' for cid, cnt in zip(cluster_ids_plot, cluster_counts)])
    ax.axhline(y=srcc, color='red', linestyle='--', linewidth=2, label=f'Overall: {srcc:.3f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 聚类分布
    ax = axes[1, 1]
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

    wedges, texts, autotexts = ax.pie(counts, labels=[f'C{c}' for c in unique_clusters],
                                        autopct='%1.1f%%', colors=colors,
                                        startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})

    # 添加图例
    legend_labels = [f'Cluster {c}: {cnt} samples' for c, cnt in zip(unique_clusters, counts)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax.set_title('Cluster Distribution', fontsize=13, weight='bold')

    plt.suptitle(f'Performance Metrics Dashboard\n'
                 f'Overall - SRCC: {srcc:.4f} | PLCC: {plcc:.4f} | RMSE: {rmse:.4f}',
                 fontsize=15, weight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Fixed visualization for distorted images')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment directory')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Checkpoint filename')
    parser.add_argument('--test_samples', type=int, default=500,
                       help='Number of test samples')
    parser.add_argument('--num_display', type=int, default=25,
                       help='Number of images to display in grid')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    checkpoint_path = exp_dir / args.checkpoint

    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return

    print("="*80)
    print("🎨 Fixed Visualization for Distorted Image Quality Assessment")
    print("="*80)
    print(f"Experiment: {exp_dir.name}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 加载模型
    print("\n📦 Loading model...")
    model, n_clusters = load_model_from_checkpoint(checkpoint_path, device)
    print("✅ Model loaded successfully")

    # 加载测试数据集（使用正确的DistortedImageDataset！）
    print("\n📊 Loading test dataset...")
    test_dataset = DistortedImageDataset(
        split='test',
        max_samples=args.test_samples,
        distortions_per_image=5,
        include_pristine=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 运行推断
    results = run_inference(model, test_loader, device)

    # 打印统计信息
    print("\n" + "="*80)
    print("📈 Inference Results")
    print("="*80)
    print(f"Total samples: {len(results['predictions'])}")
    print(f"Number of clusters: {n_clusters}")

    srcc, _ = spearmanr(results['predictions'], results['targets'])
    plcc, _ = pearsonr(results['predictions'], results['targets'])
    rmse = np.sqrt(np.mean((results['predictions'] - results['targets']) ** 2))

    print(f"\nOverall Performance:")
    print(f"  SRCC: {srcc:.4f}")
    print(f"  PLCC: {plcc:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    print(f"\nCluster Distribution:")
    unique_clusters, counts = np.unique(results['cluster_ids'], return_counts=True)
    for cluster_id, count in zip(unique_clusters, counts):
        mask = results['cluster_ids'] == cluster_id
        avg_quality = results['targets'][mask].mean()
        print(f"  Cluster {cluster_id}: {count:4d} samples (avg quality: {avg_quality:.3f})")

    # 创建输出目录
    viz_dir = exp_dir / 'visualizations_fixed'
    viz_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("🎨 Generating Visualizations")
    print("="*80)

    # 1. 性能指标仪表板
    print("\n1. Generating performance metrics dashboard...")
    plot_performance_metrics(
        results['predictions'],
        results['targets'],
        results['cluster_ids'],
        save_path=viz_dir / 'performance_metrics.png'
    )

    # 2. 选择多样化样本并绘制
    print(f"\n2. Selecting {args.num_display} diverse samples from all clusters...")
    selected_indices = select_diverse_samples(
        results['cluster_ids'],
        results['predictions'],
        results['targets'],
        n_samples=args.num_display
    )

    print(f"   Selected samples distribution:")
    for cluster_id in range(n_clusters):
        count = np.sum(results['cluster_ids'][selected_indices] == cluster_id)
        print(f"     Cluster {cluster_id}: {count} samples")

    print("\n3. Generating image grid...")
    plot_image_grid(
        results['images'][selected_indices],
        results['predictions'][selected_indices],
        results['targets'][selected_indices],
        results['cluster_ids'][selected_indices],
        results['posteriors'][selected_indices],
        distortion_types=[results['distortion_types'][i] for i in selected_indices] if results['distortion_types'] else None,
        save_path=viz_dir / 'image_grid_diverse.png'
    )

    # 3. 每个聚类的代表性样本
    print("\n4. Generating cluster examples...")
    plot_cluster_examples(
        results['images'],
        results['predictions'],
        results['targets'],
        results['cluster_ids'],
        results['posteriors'],
        n_clusters=n_clusters,
        n_per_cluster=6,
        save_path=viz_dir / 'cluster_examples_detailed.png'
    )

    print("\n" + "="*80)
    print("✅ Visualization Complete!")
    print("="*80)
    print(f"\nAll visualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    print(f"  1. performance_metrics.png - Overall performance dashboard")
    print(f"  2. image_grid_diverse.png - {args.num_display} diverse samples from all clusters")
    print(f"  3. cluster_examples_detailed.png - Representative samples from each cluster")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
