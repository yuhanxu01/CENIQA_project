"""
Enhanced visualization script with detailed image analysis.
Shows: images, quality scores, cluster assignments, and accuracy metrics.

Usage:
    python enhanced_visualize.py --experiment experiments/resnet18_gmm_mlp --num_images 25
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

from train_gpu import SimpleCNNGMMMLPModel, HuggingFaceImageDataset


def infer_model_config_from_checkpoint(checkpoint_path, device):
    """Infer model configuration from checkpoint state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Infer feature_dim from backbone projection layer
    feature_dim = state_dict['backbone.proj.weight'].shape[0]

    # Infer n_clusters from GMM parameters
    n_clusters = state_dict['gmm.means'].shape[0]

    # Infer hidden_dim from regressor first layer
    hidden_dim = state_dict['regressor.mlp.0.weight'].shape[0]

    # Try to infer backbone name from the checkpoint or use default
    # Check the backbone model structure
    backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.model.')]
    if any('resnet' in k.lower() for k in backbone_keys):
        if 'layer4' in str(backbone_keys):
            # Count layers to determine resnet variant
            backbone_name = 'resnet18'  # default
        else:
            backbone_name = 'resnet18'
    else:
        backbone_name = 'resnet18'  # default fallback

    return {
        'backbone': backbone_name,
        'feature_dim': feature_dim,
        'n_clusters': n_clusters,
        'hidden_dim': hidden_dim,
        'dropout': 0.3,
        'freeze_backbone': False
    }, checkpoint


def load_model(checkpoint_path, config_path, device):
    """Load trained model from checkpoint."""
    # Try to load config file
    config = None
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")

    # If config doesn't exist or is missing keys, infer from checkpoint
    if config is None:
        print("Config file not found, inferring configuration from checkpoint...")
        config, checkpoint = infer_model_config_from_checkpoint(checkpoint_path, device)
    else:
        # Try to get values from config, with fallback to inference
        checkpoint = None
        required_keys = ['backbone', 'feature_dim', 'n_clusters', 'hidden_dim']

        # Check if config has nested 'model' key structure
        if 'model' in config and isinstance(config['model'], dict):
            config = config['model']

        # Check if all required keys are present
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            print(f"Warning: Config missing keys {missing_keys}, inferring from checkpoint...")
            inferred_config, checkpoint = infer_model_config_from_checkpoint(checkpoint_path, device)
            # Merge with existing config, preferring inferred values for missing keys
            for key in missing_keys:
                config[key] = inferred_config[key]

    # Ensure dropout key exists
    if 'dropout' not in config:
        config['dropout'] = config.get('dropout_rate', 0.3)

    # Ensure freeze_backbone key exists
    if 'freeze_backbone' not in config:
        config['freeze_backbone'] = False

    # Create model
    model = SimpleCNNGMMMLPModel(
        backbone_name=config['backbone'],
        feature_dim=config['feature_dim'],
        n_clusters=config['n_clusters'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        freeze_backbone=config['freeze_backbone']
    )

    # Load checkpoint if not already loaded
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def denormalize_image(img_tensor):
    """Denormalize image tensor to displayable numpy array."""
    if torch.is_tensor(img_tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor

    # ImageNet normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Denormalize
    img = img.transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img


def plot_image_grid_with_details(images, predictions, targets, cluster_ids, posteriors,
                                  n_images=25, save_path=None):
    """
    Plot grid of images with detailed information:
    - Original image
    - Predicted quality score
    - Ground truth quality score
    - Cluster assignment
    - Prediction error
    """
    n_images = min(n_images, len(images))

    # Select diverse samples from different clusters
    indices = select_diverse_samples(cluster_ids, predictions, n_images)

    # Calculate grid size
    n_cols = 5
    n_rows = int(np.ceil(n_images / n_cols))

    # Create figure
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4.5))

    for i, idx in enumerate(indices):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # Denormalize and display image
        img = denormalize_image(images[idx])
        ax.imshow(img)
        ax.axis('off')

        # Get information
        pred_score = predictions[idx]
        true_score = targets[idx]
        cluster_id = cluster_ids[idx]
        error = abs(pred_score - true_score)
        confidence = posteriors[idx][cluster_id]  # Cluster confidence

        # Create title with color coding based on error
        if error < 0.1:
            color = 'green'
        elif error < 0.2:
            color = 'orange'
        else:
            color = 'red'

        title = (f'Predicted: {pred_score:.3f}\n'
                f'Ground Truth: {true_score:.3f}\n'
                f'Cluster: {cluster_id} (conf: {confidence:.2f})\n'
                f'Error: {error:.3f}')

        ax.set_title(title, fontsize=10, color=color, weight='bold')

        # Add border based on error
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.suptitle('Image Quality Assessment Results\n'
                 'Green: Error < 0.1 | Orange: 0.1 ≤ Error < 0.2 | Red: Error ≥ 0.2',
                 fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def select_diverse_samples(cluster_ids, predictions, n_samples):
    """Select diverse samples covering different clusters."""
    n_clusters = len(np.unique(cluster_ids))
    samples_per_cluster = max(1, n_samples // n_clusters)

    indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_ids == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) > 0:
            # Sample from this cluster
            n_select = min(samples_per_cluster, len(cluster_indices))
            # Select samples with diverse quality scores
            selected = np.random.choice(cluster_indices, n_select, replace=False)
            indices.extend(selected.tolist())

    # If we don't have enough samples, randomly add more
    while len(indices) < n_samples and len(indices) < len(predictions):
        remaining = list(set(range(len(predictions))) - set(indices))
        if remaining:
            indices.append(np.random.choice(remaining))
        else:
            break

    return indices[:n_samples]


def plot_comprehensive_metrics(predictions, targets, cluster_ids, posteriors, save_path=None):
    """
    Plot comprehensive metrics dashboard:
    - Overall accuracy metrics (SRCC, PLCC, RMSE)
    - Cluster-wise accuracy
    - Error distribution
    - Confusion-style visualization
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Calculate overall metrics
    srcc, _ = spearmanr(predictions, targets)
    plcc, _ = pearsonr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))

    # 1. Overall metrics text box (top center)
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.axis('off')
    metrics_text = (
        f'Overall Performance Metrics\n'
        f'{"="*40}\n\n'
        f'SRCC (Spearman): {srcc:.4f}\n'
        f'PLCC (Pearson): {plcc:.4f}\n'
        f'RMSE: {rmse:.4f}\n'
        f'MAE: {mae:.4f}\n\n'
        f'Total Samples: {len(predictions)}\n'
        f'Number of Clusters: {len(np.unique(cluster_ids))}'
    )
    ax_metrics.text(0.5, 0.5, metrics_text,
                   ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 2. Scatter plot: Predictions vs Ground Truth (top left)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_scatter.scatter(targets, predictions, alpha=0.5, s=30, c='steelblue')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax_scatter.set_xlabel('Ground Truth Score', fontsize=12)
    ax_scatter.set_ylabel('Predicted Score', fontsize=12)
    ax_scatter.set_title('Predictions vs Ground Truth', fontsize=14, weight='bold')
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)

    # 3. Error distribution histogram (top right)
    ax_error = fig.add_subplot(gs[0, 2])
    errors = predictions - targets
    ax_error.hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax_error.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax_error.set_xlabel('Prediction Error', fontsize=12)
    ax_error.set_ylabel('Frequency', fontsize=12)
    ax_error.set_title(f'Error Distribution (Mean: {errors.mean():.4f})', fontsize=14, weight='bold')
    ax_error.grid(True, alpha=0.3)

    # 4. Cluster distribution (middle left)
    ax_cluster_dist = fig.add_subplot(gs[1, 0])
    n_clusters = len(np.unique(cluster_ids))
    cluster_counts = [np.sum(cluster_ids == i) for i in range(n_clusters)]
    colors_cluster = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    bars = ax_cluster_dist.bar(range(n_clusters), cluster_counts, color=colors_cluster, alpha=0.7, edgecolor='black')
    ax_cluster_dist.set_xlabel('Cluster ID', fontsize=12)
    ax_cluster_dist.set_ylabel('Number of Samples', fontsize=12)
    ax_cluster_dist.set_title('Cluster Distribution', fontsize=14, weight='bold')
    ax_cluster_dist.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    total_samples = len(cluster_ids)
    for i, (bar, count) in enumerate(zip(bars, cluster_counts)):
        percentage = (count / total_samples) * 100
        ax_cluster_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    # 5. Cluster-wise accuracy (middle center)
    ax_cluster_acc = fig.add_subplot(gs[1, 1])
    cluster_metrics = []
    for i in range(n_clusters):
        mask = cluster_ids == i
        if mask.sum() > 0:
            cluster_preds = predictions[mask]
            cluster_targets = targets[mask]
            cluster_srcc, _ = spearmanr(cluster_preds, cluster_targets)
            cluster_rmse = np.sqrt(np.mean((cluster_preds - cluster_targets) ** 2))
            cluster_metrics.append({
                'cluster': i,
                'srcc': cluster_srcc,
                'rmse': cluster_rmse,
                'n_samples': mask.sum()
            })

    # Plot cluster SRCCs
    cluster_ids_list = [m['cluster'] for m in cluster_metrics]
    cluster_srccs = [m['srcc'] for m in cluster_metrics]
    bars = ax_cluster_acc.bar(cluster_ids_list, cluster_srccs, color=colors_cluster[:len(cluster_ids_list)],
                              alpha=0.7, edgecolor='black')
    ax_cluster_acc.set_xlabel('Cluster ID', fontsize=12)
    ax_cluster_acc.set_ylabel('SRCC', fontsize=12)
    ax_cluster_acc.set_title('Cluster-wise Accuracy (SRCC)', fontsize=14, weight='bold')
    ax_cluster_acc.axhline(y=srcc, color='red', linestyle='--', linewidth=2, label=f'Overall: {srcc:.3f}')
    ax_cluster_acc.legend()
    ax_cluster_acc.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, srcc_val in zip(bars, cluster_srccs):
        ax_cluster_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{srcc_val:.3f}', ha='center', va='bottom', fontsize=9)

    # 6. Quality score distribution per cluster (middle right)
    ax_quality_dist = fig.add_subplot(gs[1, 2])
    cluster_quality_data = []
    cluster_positions = []
    for i in range(n_clusters):
        mask = cluster_ids == i
        if mask.sum() > 0:
            cluster_quality_data.append(predictions[mask])
            cluster_positions.append(i)

    bp = ax_quality_dist.boxplot(cluster_quality_data, positions=cluster_positions,
                                 widths=0.6, patch_artist=True, showfliers=True)
    for patch, color in zip(bp['boxes'], colors_cluster[cluster_positions]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_quality_dist.set_xlabel('Cluster ID', fontsize=12)
    ax_quality_dist.set_ylabel('Predicted Quality Score', fontsize=12)
    ax_quality_dist.set_title('Quality Score Distribution per Cluster', fontsize=14, weight='bold')
    ax_quality_dist.grid(True, alpha=0.3, axis='y')

    # 7. Cluster statistics table (bottom)
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    # Create table data
    table_data = [['Cluster', 'Samples', '% Total', 'Mean Score', 'Std Score', 'SRCC', 'RMSE']]
    for i in range(n_clusters):
        mask = cluster_ids == i
        if mask.sum() > 0:
            cluster_preds = predictions[mask]
            cluster_targets = targets[mask]
            cluster_srcc, _ = spearmanr(cluster_preds, cluster_targets)
            cluster_rmse = np.sqrt(np.mean((cluster_preds - cluster_targets) ** 2))

            table_data.append([
                f'{i}',
                f'{mask.sum()}',
                f'{mask.sum()/len(cluster_ids)*100:.1f}%',
                f'{cluster_preds.mean():.4f}',
                f'{cluster_preds.std():.4f}',
                f'{cluster_srcc:.4f}',
                f'{cluster_rmse:.4f}'
            ])

    # Create table
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          bbox=[0.1, 0.0, 0.8, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')

    plt.suptitle('Comprehensive Performance Analysis', fontsize=16, weight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_cluster_examples(images, predictions, targets, cluster_ids, posteriors,
                          n_per_cluster=5, save_path=None):
    """
    Plot representative examples from each cluster.
    """
    n_clusters = len(np.unique(cluster_ids))

    fig, axes = plt.subplots(n_clusters, n_per_cluster,
                            figsize=(n_per_cluster * 3, n_clusters * 3))

    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_ids == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        # Select n_per_cluster samples
        n_select = min(n_per_cluster, len(cluster_indices))
        selected = np.random.choice(cluster_indices, n_select, replace=False)

        for i, idx in enumerate(selected):
            ax = axes[cluster_id, i]

            # Display image
            img = denormalize_image(images[idx])
            ax.imshow(img)
            ax.axis('off')

            # Add info
            pred = predictions[idx]
            target = targets[idx]
            confidence = posteriors[idx][cluster_id]
            error = abs(pred - target)

            color = 'green' if error < 0.1 else 'orange' if error < 0.2 else 'red'

            title = (f'Pred: {pred:.3f}\n'
                    f'GT: {target:.3f}\n'
                    f'Conf: {confidence:.2f}\n'
                    f'Err: {error:.3f}')
            ax.set_title(title, fontsize=9, color=color)

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

        # Fill empty slots
        for i in range(n_select, n_per_cluster):
            axes[cluster_id, i].axis('off')

        # Add cluster label
        axes[cluster_id, 0].text(-0.1, 0.5, f'Cluster {cluster_id}',
                                rotation=90, verticalalignment='center',
                                transform=axes[cluster_id, 0].transAxes,
                                fontsize=14, weight='bold')

    plt.suptitle('Representative Examples from Each Cluster', fontsize=16, weight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def run_inference(model, dataloader, device):
    """Run inference and collect all outputs."""
    all_predictions = []
    all_targets = []
    all_features = []
    all_posteriors = []
    all_images = []

    print("Running inference...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)

            outputs = model(images, return_all=True)

            all_predictions.append(outputs['quality_score'].cpu().numpy())
            all_targets.append(targets.numpy())
            all_features.append(outputs['features'].cpu().numpy())
            all_posteriors.append(outputs['posteriors'].cpu().numpy())
            all_images.append(images.cpu())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    features = np.concatenate(all_features)
    posteriors = np.concatenate(all_posteriors)
    images = torch.cat(all_images, dim=0)

    cluster_assignments = np.argmax(posteriors, axis=1)

    return {
        'predictions': predictions,
        'targets': targets,
        'features': features,
        'posteriors': posteriors,
        'cluster_assignments': cluster_assignments,
        'images': images
    }


def main():
    parser = argparse.ArgumentParser(description='Enhanced visualization with detailed image analysis')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment directory (e.g., experiments/resnet18_gmm_mlp)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Checkpoint filename (default: best_model.pth)')
    parser.add_argument('--num_images', type=int, default=25,
                       help='Number of images to visualize in detail (default: 25)')
    parser.add_argument('--test_samples', type=int, default=500,
                       help='Number of test samples to evaluate (default: 500)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference')
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    checkpoint_path = exp_dir / args.checkpoint
    config_path = exp_dir / 'config.json'
    viz_dir = exp_dir / 'enhanced_visualizations'
    viz_dir.mkdir(exist_ok=True)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print("="*80)
    print("Enhanced Image Quality Assessment Visualization")
    print("="*80)
    print(f"Experiment: {exp_dir.name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of images to visualize: {args.num_images}")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading model...")
    model, config = load_model(checkpoint_path, config_path, device)
    print(f"Model loaded: {config['backbone']}")

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = HuggingFaceImageDataset(
        split='test',
        max_samples=args.test_samples
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Run inference
    results = run_inference(model, test_loader, device)

    print("\n" + "="*80)
    print("Generating Enhanced Visualizations")
    print("="*80)

    # 1. Comprehensive metrics dashboard
    print("\n1. Generating comprehensive metrics dashboard...")
    plot_comprehensive_metrics(
        results['predictions'],
        results['targets'],
        results['cluster_assignments'],
        results['posteriors'],
        save_path=viz_dir / 'comprehensive_metrics.png'
    )

    # 2. Image grid with detailed information
    print(f"\n2. Generating image grid with {args.num_images} samples...")
    plot_image_grid_with_details(
        results['images'],
        results['predictions'],
        results['targets'],
        results['cluster_assignments'],
        results['posteriors'],
        n_images=args.num_images,
        save_path=viz_dir / 'image_grid_detailed.png'
    )

    # 3. Cluster examples
    print("\n3. Generating cluster-wise examples...")
    plot_cluster_examples(
        results['images'],
        results['predictions'],
        results['targets'],
        results['cluster_assignments'],
        results['posteriors'],
        n_per_cluster=5,
        save_path=viz_dir / 'cluster_examples.png'
    )

    # Print summary
    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"\nAll visualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    print(f"  - comprehensive_metrics.png : Complete performance dashboard")
    print(f"  - image_grid_detailed.png   : {args.num_images} images with scores and clusters")
    print(f"  - cluster_examples.png      : Representative samples from each cluster")
    print("="*80)


if __name__ == '__main__':
    main()
