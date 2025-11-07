"""
Compare two experiments: Simple Baseline (no GMM) vs GMM-based model
"""
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr

from simple_model import SimpleCNNModel
from train_gpu import SimpleCNNGMMMLPModel
from distorted_dataset import DistortedImageDataset


def load_model_from_checkpoint(checkpoint_path, model_type, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    if model_type == 'simple':
        # Simple baseline: CNN → MLP
        feature_dim = state_dict['backbone.proj.weight'].shape[0]
        hidden_dim = state_dict['regressor.fc1.weight'].shape[0]

        model = SimpleCNNModel(
            backbone_name='resnet18',
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=0.3,
            freeze_backbone=False
        )
    else:
        # GMM-based model: CNN → GMM → MLP
        feature_dim = state_dict['backbone.proj.weight'].shape[0]
        n_clusters = state_dict['gmm.means'].shape[0]
        hidden_dim = state_dict['regressor.fc1.weight'].shape[0]

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

    return model


def evaluate_model(model, dataloader, device, model_name):
    """Evaluate model and return predictions."""
    all_preds = []
    all_targets = []
    all_features = []

    print(f"\nEvaluating {model_name}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'{model_name}'):
            if len(batch) == 3:
                images, scores, _ = batch
            else:
                images, scores = batch

            images = images.to(device)

            outputs = model(images, return_all=True)
            predictions = outputs['quality_score']
            features = outputs['features']

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(scores.numpy())
            all_features.append(features.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    features = np.concatenate(all_features)

    # Calculate metrics
    srcc, _ = spearmanr(preds, targets)
    plcc, _ = pearsonr(preds, targets)
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))

    return {
        'predictions': preds,
        'targets': targets,
        'features': features,
        'srcc': srcc,
        'plcc': plcc,
        'rmse': rmse,
        'mae': mae
    }


def plot_comparison(results_simple, results_gmm, save_dir):
    """Plot comprehensive comparison."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Extract data
    preds_simple = results_simple['predictions']
    preds_gmm = results_gmm['predictions']
    targets = results_simple['targets']  # Same targets for both

    # 1. Metrics comparison table
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')

    metrics_data = [
        ['Metric', 'Simple Baseline', 'GMM-based', 'Improvement'],
        ['SRCC', f"{results_simple['srcc']:.4f}", f"{results_gmm['srcc']:.4f}",
         f"{(results_gmm['srcc'] - results_simple['srcc']):.4f}"],
        ['PLCC', f"{results_simple['plcc']:.4f}", f"{results_gmm['plcc']:.4f}",
         f"{(results_gmm['plcc'] - results_simple['plcc']):.4f}"],
        ['RMSE', f"{results_simple['rmse']:.4f}", f"{results_gmm['rmse']:.4f}",
         f"{(results_simple['rmse'] - results_gmm['rmse']):.4f}"],
        ['MAE', f"{results_simple['mae']:.4f}", f"{results_gmm['mae']:.4f}",
         f"{(results_simple['mae'] - results_gmm['mae']):.4f}"]
    ]

    table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Style improvement column
    for i in range(1, 5):
        cell = table[(i, 3)]
        improvement = float(metrics_data[i][3])
        if i <= 2:  # SRCC, PLCC (higher is better)
            color = '#90EE90' if improvement > 0 else '#FFB6C1'
        else:  # RMSE, MAE (lower is better)
            color = '#90EE90' if improvement < 0 else '#FFB6C1'
        cell.set_facecolor(color)

    ax.set_title('Performance Metrics Comparison', fontsize=14, weight='bold', pad=20)

    # 2. Scatter plot: Simple Baseline
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(targets, preds_simple, alpha=0.5, s=20, c='blue', label='Simple')
    min_val = min(targets.min(), preds_simple.min())
    max_val = max(targets.max(), preds_simple.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Ground Truth', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'Simple Baseline\nSRCC: {results_simple["srcc"]:.4f}', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Scatter plot: GMM-based
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(targets, preds_gmm, alpha=0.5, s=20, c='green', label='GMM')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Ground Truth', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'GMM-based Model\nSRCC: {results_gmm["srcc"]:.4f}', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Error distribution comparison
    ax = fig.add_subplot(gs[1, 0])
    errors_simple = preds_simple - targets
    errors_gmm = preds_gmm - targets

    ax.hist(errors_simple, bins=50, alpha=0.5, label=f'Simple (std={errors_simple.std():.4f})',
            color='blue', edgecolor='black')
    ax.hist(errors_gmm, bins=50, alpha=0.5, label=f'GMM (std={errors_gmm.std():.4f})',
            color='green', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Error Distribution Comparison', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Cumulative error plot
    ax = fig.add_subplot(gs[1, 1])
    abs_errors_simple = np.abs(errors_simple)
    abs_errors_gmm = np.abs(errors_gmm)

    sorted_errors_simple = np.sort(abs_errors_simple)
    sorted_errors_gmm = np.sort(abs_errors_gmm)
    cumulative = np.arange(1, len(targets) + 1) / len(targets) * 100

    ax.plot(sorted_errors_simple, cumulative, label='Simple Baseline', linewidth=2, color='blue')
    ax.plot(sorted_errors_gmm, cumulative, label='GMM-based', linewidth=2, color='green')
    ax.set_xlabel('Absolute Error', fontsize=11)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax.set_title('Cumulative Error Distribution', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Box plot comparison
    ax = fig.add_subplot(gs[1, 2])
    data_to_plot = [abs_errors_simple, abs_errors_gmm]
    bp = ax.boxplot(data_to_plot, labels=['Simple', 'GMM'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel('Absolute Error', fontsize=11)
    ax.set_title('Error Distribution Boxplot', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 7. Per-quality-range comparison
    ax = fig.add_subplot(gs[2, :])

    # Divide into quality bins
    n_bins = 10
    quality_bins = np.linspace(targets.min(), targets.max(), n_bins + 1)
    bin_centers = (quality_bins[:-1] + quality_bins[1:]) / 2

    mae_simple_per_bin = []
    mae_gmm_per_bin = []
    counts_per_bin = []

    for i in range(n_bins):
        mask = (targets >= quality_bins[i]) & (targets < quality_bins[i + 1])
        if mask.sum() > 0:
            mae_simple_per_bin.append(np.abs(errors_simple[mask]).mean())
            mae_gmm_per_bin.append(np.abs(errors_gmm[mask]).mean())
            counts_per_bin.append(mask.sum())
        else:
            mae_simple_per_bin.append(0)
            mae_gmm_per_bin.append(0)
            counts_per_bin.append(0)

    x = np.arange(n_bins)
    width = 0.35

    bars1 = ax.bar(x - width/2, mae_simple_per_bin, width, label='Simple Baseline',
                   color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, mae_gmm_per_bin, width, label='GMM-based',
                   color='green', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Quality Range', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('MAE per Quality Range', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{bin_centers[i]:.2f}' for i in range(n_bins)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add sample counts as text
    for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, counts_per_bin)):
        if count > 0:
            ax.text(i, max(bar1.get_height(), bar2.get_height()) + 0.002,
                   f'n={count}', ha='center', va='bottom', fontsize=8)

    # Overall title
    plt.suptitle('Model Comparison: Simple Baseline vs GMM-based Model',
                fontsize=16, weight='bold', y=0.995)

    # Save
    save_path = save_dir / 'comparison_dashboard.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Simple Baseline vs GMM-based model')
    parser.add_argument('--exp_simple', type=str, required=True,
                       help='Simple baseline experiment directory')
    parser.add_argument('--exp_gmm', type=str, required=True,
                       help='GMM-based experiment directory')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Checkpoint filename')
    parser.add_argument('--test_samples', type=int, default=500,
                       help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    args = parser.parse_args()

    print("="*80)
    print("🔬 Experiment Comparison")
    print("="*80)
    print(f"Simple Baseline: {args.exp_simple}")
    print(f"GMM-based Model: {args.exp_gmm}")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load models
    print("\n📦 Loading models...")
    checkpoint_simple = Path(args.exp_simple) / args.checkpoint
    checkpoint_gmm = Path(args.exp_gmm) / args.checkpoint

    if not checkpoint_simple.exists():
        print(f"❌ Simple baseline checkpoint not found: {checkpoint_simple}")
        return

    if not checkpoint_gmm.exists():
        print(f"❌ GMM-based checkpoint not found: {checkpoint_gmm}")
        return

    model_simple = load_model_from_checkpoint(checkpoint_simple, 'simple', device)
    model_gmm = load_model_from_checkpoint(checkpoint_gmm, 'gmm', device)

    print("✅ Models loaded")

    # Load test dataset
    print("\n📊 Loading test dataset...")
    test_dataset = DistortedImageDataset(
        split='test',
        max_samples=args.test_samples // 6,
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

    # Evaluate both models
    results_simple = evaluate_model(model_simple, test_loader, device, 'Simple Baseline')
    results_gmm = evaluate_model(model_gmm, test_loader, device, 'GMM-based')

    # Print comparison
    print("\n" + "="*80)
    print("📊 Results Comparison")
    print("="*80)

    print(f"\n{'Metric':<10} {'Simple':<12} {'GMM-based':<12} {'Improvement':<12}")
    print("-" * 50)
    print(f"{'SRCC':<10} {results_simple['srcc']:<12.4f} {results_gmm['srcc']:<12.4f} "
          f"{results_gmm['srcc'] - results_simple['srcc']:+.4f}")
    print(f"{'PLCC':<10} {results_simple['plcc']:<12.4f} {results_gmm['plcc']:<12.4f} "
          f"{results_gmm['plcc'] - results_simple['plcc']:+.4f}")
    print(f"{'RMSE':<10} {results_simple['rmse']:<12.4f} {results_gmm['rmse']:<12.4f} "
          f"{results_simple['rmse'] - results_gmm['rmse']:+.4f}")
    print(f"{'MAE':<10} {results_simple['mae']:<12.4f} {results_gmm['mae']:<12.4f} "
          f"{results_simple['mae'] - results_gmm['mae']:+.4f}")

    # Determine winner
    print("\n" + "="*80)
    if results_gmm['srcc'] > results_simple['srcc']:
        improvement_pct = (results_gmm['srcc'] - results_simple['srcc']) / results_simple['srcc'] * 100
        print(f"🏆 GMM-based model is BETTER by {improvement_pct:.2f}%")
    elif results_simple['srcc'] > results_gmm['srcc']:
        degradation_pct = (results_simple['srcc'] - results_gmm['srcc']) / results_gmm['srcc'] * 100
        print(f"⚠️  Simple baseline is BETTER by {degradation_pct:.2f}%")
    else:
        print("🤝 Models perform equally")

    # Generate comparison plots
    print("\n🎨 Generating comparison plots...")
    save_dir = Path('experiments') / 'comparison'
    save_dir.mkdir(exist_ok=True)

    plot_comparison(results_simple, results_gmm, save_dir)

    # Save detailed results
    comparison_results = {
        'simple_baseline': {k: float(v) if isinstance(v, (np.floating, float)) else v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in results_simple.items() if k != 'features'},
        'gmm_based': {k: float(v) if isinstance(v, (np.floating, float)) else v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in results_gmm.items() if k != 'features'}
    }

    with open(save_dir / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)

    print(f"\n✅ Comparison complete! Results saved to: {save_dir}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
