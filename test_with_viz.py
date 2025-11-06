"""
Test script with comprehensive visualization.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from train_gpu import SimpleCNNGMMMLPModel, HuggingFaceImageDataset
from visualize import (
    plot_prediction_scatter,
    plot_cluster_distribution,
    plot_feature_tsne,
    plot_feature_pca,
    plot_training_curves,
    plot_sample_predictions
)


def load_model(checkpoint_path, config_path, device):
    """Load trained model from checkpoint."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = SimpleCNNGMMMLPModel(
        backbone_name=config['model']['backbone'],
        feature_dim=config['model']['feature_dim'],
        n_clusters=config['model']['n_clusters'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        freeze_backbone=config['model']['freeze_backbone']
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def test_model(model, dataloader, device):
    """Run inference and collect all outputs."""
    from scipy.stats import spearmanr, pearsonr

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

    # Calculate metrics
    srcc, _ = spearmanr(predictions, targets)
    plcc, _ = pearsonr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    results = {
        'srcc': srcc,
        'plcc': plcc,
        'rmse': rmse,
        'predictions': predictions,
        'targets': targets,
        'features': features,
        'posteriors': posteriors,
        'cluster_assignments': cluster_assignments,
        'images': images
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Test CNN+GMM+MLP model with visualization')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment directory (e.g., experiments/resnet18_gmm_mlp)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                       help='Checkpoint filename (default: best_model.pth)')
    parser.add_argument('--test_samples', type=int, default=500,
                       help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for testing')
    parser.add_argument('--skip_tsne', action='store_true',
                       help='Skip t-SNE visualization (saves time)')
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    checkpoint_path = exp_dir / args.checkpoint
    config_path = exp_dir / 'config.json'
    viz_dir = exp_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print("="*60)
    print("Testing with Visualization")
    print("="*60)
    print(f"Experiment: {exp_dir.name}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading model...")
    model, config = load_model(checkpoint_path, config_path, device)
    print(f"Model loaded: {config['model']['backbone']}")

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
    results = test_model(model, test_loader, device)

    # Print metrics
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"SRCC: {results['srcc']:.4f}")
    print(f"PLCC: {results['plcc']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print("="*60)

    # Save results
    results_to_save = {
        'srcc': float(results['srcc']),
        'plcc': float(results['plcc']),
        'rmse': float(results['rmse']),
        'n_samples': len(results['predictions'])
    }
    with open(viz_dir / 'test_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=4)

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    # 1. Prediction scatter plot
    print("\n1. Prediction vs Ground Truth scatter plot...")
    plot_prediction_scatter(
        results['predictions'],
        results['targets'],
        save_path=viz_dir / 'predictions_scatter.png',
        title=f"{exp_dir.name} - Predictions vs Ground Truth"
    )

    # 2. Cluster distribution
    print("\n2. Cluster distribution and quality per cluster...")
    plot_cluster_distribution(
        results['posteriors'],
        results['predictions'],
        save_path=viz_dir / 'cluster_distribution.png'
    )

    # 3. PCA visualization (fast)
    print("\n3. PCA visualization of features...")
    plot_feature_pca(
        results['features'],
        labels=results['targets'],
        cluster_assignments=results['cluster_assignments'],
        save_path=viz_dir / 'features_pca.png'
    )

    # 4. t-SNE visualization (slower, optional)
    if not args.skip_tsne:
        print("\n4. t-SNE visualization of features (this may take a while)...")
        # Use subset for t-SNE if dataset is large
        n_samples_tsne = min(1000, len(results['features']))
        indices = np.random.choice(len(results['features']), n_samples_tsne, replace=False)

        plot_feature_tsne(
            results['features'][indices],
            labels=results['targets'][indices],
            cluster_assignments=results['cluster_assignments'][indices],
            save_path=viz_dir / 'features_tsne.png',
            perplexity=30
        )
    else:
        print("\n4. Skipping t-SNE visualization (use --skip_tsne flag to enable)")

    # 5. Sample predictions
    print("\n5. Sample predictions with images...")
    plot_sample_predictions(
        results['images'],
        results['predictions'],
        results['targets'],
        results['cluster_assignments'],
        n_samples=16,
        save_path=viz_dir / 'sample_predictions.png'
    )

    # 6. Training curves (if available)
    print("\n6. Training curves...")
    history_path = exp_dir / 'training_history.json'
    if history_path.exists():
        result = plot_training_curves(
            history_path,
            save_path=viz_dir / 'training_curves.png'
        )
        if result is None:
            print("Training curves plot was skipped due to corrupted history file.")
    else:
        print("Training history not found, skipping training curves")

    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"All visualizations saved to: {viz_dir}")
    print("="*60)

    # Print cluster statistics
    print("\n" + "="*60)
    print("Cluster Statistics")
    print("="*60)

    n_clusters = results['posteriors'].shape[1]
    for cluster_id in range(n_clusters):
        mask = results['cluster_assignments'] == cluster_id
        n_samples = mask.sum()
        if n_samples > 0:
            cluster_scores = results['predictions'][mask]
            mean_score = cluster_scores.mean()
            std_score = cluster_scores.std()
            min_score = cluster_scores.min()
            max_score = cluster_scores.max()

            print(f"\nCluster {cluster_id}:")
            print(f"  Samples: {n_samples} ({n_samples/len(results['predictions'])*100:.1f}%)")
            print(f"  Mean Score: {mean_score:.4f} ± {std_score:.4f}")
            print(f"  Range: [{min_score:.4f}, {max_score:.4f}]")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
