"""
Visualization utilities for model predictions and clustering.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from pathlib import Path


def plot_prediction_scatter(predictions, targets, save_path=None, title="Predictions vs Ground Truth"):
    """
    Plot scatter plot of predictions vs ground truth.

    Args:
        predictions: numpy array of predicted scores
        targets: numpy array of ground truth scores
        save_path: path to save figure
        title: plot title
    """
    from scipy.stats import spearmanr, pearsonr

    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=50)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Calculate metrics
    srcc, _ = spearmanr(predictions, targets)
    plcc, _ = pearsonr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    plt.xlabel('Ground Truth Score', fontsize=14)
    plt.ylabel('Predicted Score', fontsize=14)
    plt.title(title, fontsize=16)

    # Add metrics text
    metrics_text = f'SRCC: {srcc:.4f}\nPLCC: {plcc:.4f}\nRMSE: {rmse:.4f}'
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return plt.gcf()


def plot_cluster_distribution(posteriors, predictions=None, save_path=None):
    """
    Plot cluster distribution and quality scores per cluster.

    Args:
        posteriors: numpy array [N, n_clusters] of cluster posteriors
        predictions: optional numpy array [N] of quality predictions
        save_path: path to save figure
    """
    n_clusters = posteriors.shape[1]
    cluster_assignments = np.argmax(posteriors, axis=1)

    fig, axes = plt.subplots(1, 2 if predictions is not None else 1,
                            figsize=(15 if predictions is not None else 8, 6))

    if predictions is None:
        axes = [axes]

    # Plot 1: Cluster distribution
    cluster_counts = [np.sum(cluster_assignments == i) for i in range(n_clusters)]

    axes[0].bar(range(n_clusters), cluster_counts, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Cluster ID', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Cluster Distribution', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    total_samples = len(cluster_assignments)
    for i, count in enumerate(cluster_counts):
        percentage = (count / total_samples) * 100
        axes[0].text(i, count, f'{percentage:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    # Plot 2: Quality scores per cluster (if predictions provided)
    if predictions is not None:
        cluster_quality = []
        for i in range(n_clusters):
            mask = cluster_assignments == i
            if mask.sum() > 0:
                cluster_quality.append(predictions[mask])
            else:
                cluster_quality.append([])

        # Box plot
        positions = [i for i in range(n_clusters) if len(cluster_quality[i]) > 0]
        data_to_plot = [cluster_quality[i] for i in positions]

        bp = axes[1].boxplot(data_to_plot, positions=positions, widths=0.6,
                            patch_artist=True, showfliers=False)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        for patch, color in zip(bp['boxes'], colors[positions]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_xlabel('Cluster ID', fontsize=12)
        axes[1].set_ylabel('Quality Score', fontsize=12)
        axes[1].set_title('Quality Distribution per Cluster', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add mean labels
        for pos, data in zip(positions, data_to_plot):
            mean_val = np.mean(data)
            axes[1].plot(pos, mean_val, 'r*', markersize=15, zorder=3)
            axes[1].text(pos, mean_val, f'{mean_val:.3f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_feature_tsne(features, labels=None, cluster_assignments=None,
                     save_path=None, perplexity=30):
    """
    Plot t-SNE visualization of features.

    Args:
        features: numpy array [N, feature_dim]
        labels: optional numpy array [N] for color coding (e.g., quality scores)
        cluster_assignments: optional numpy array [N] for cluster visualization
        save_path: path to save figure
        perplexity: t-SNE perplexity parameter
    """
    print("Computing t-SNE... (this may take a while)")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Color by quality scores (if provided)
    if labels is not None:
        scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1],
                                  c=labels, cmap='viridis', s=30, alpha=0.6)
        plt.colorbar(scatter1, ax=axes[0], label='Quality Score')
        axes[0].set_title('t-SNE colored by Quality Score', fontsize=14)
    else:
        axes[0].scatter(features_2d[:, 0], features_2d[:, 1], s=30, alpha=0.6)
        axes[0].set_title('t-SNE Visualization', fontsize=14)

    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)

    # Plot 2: Color by cluster assignments (if provided)
    if cluster_assignments is not None:
        n_clusters = len(np.unique(cluster_assignments))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

        for i in range(n_clusters):
            mask = cluster_assignments == i
            axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors[i]], label=f'Cluster {i}',
                          s=30, alpha=0.6)

        axes[1].legend(loc='best', fontsize=10, ncol=2)
        axes[1].set_title('t-SNE colored by Cluster Assignment', fontsize=14)
    else:
        axes[1].scatter(features_2d[:, 0], features_2d[:, 1], s=30, alpha=0.6)
        axes[1].set_title('t-SNE Visualization', fontsize=14)

    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_feature_pca(features, labels=None, cluster_assignments=None, save_path=None):
    """
    Plot PCA visualization of features (faster than t-SNE).

    Args:
        features: numpy array [N, feature_dim]
        labels: optional numpy array [N] for color coding
        cluster_assignments: optional numpy array [N]
        save_path: path to save figure
    """
    print("Computing PCA...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Color by quality scores
    if labels is not None:
        scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1],
                                  c=labels, cmap='viridis', s=30, alpha=0.6)
        plt.colorbar(scatter1, ax=axes[0], label='Quality Score')
        axes[0].set_title(f'PCA colored by Quality Score\n(Variance explained: {pca.explained_variance_ratio_.sum():.2%})',
                         fontsize=14)
    else:
        axes[0].scatter(features_2d[:, 0], features_2d[:, 1], s=30, alpha=0.6)
        axes[0].set_title('PCA Visualization', fontsize=14)

    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)

    # Plot 2: Color by clusters
    if cluster_assignments is not None:
        n_clusters = len(np.unique(cluster_assignments))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

        for i in range(n_clusters):
            mask = cluster_assignments == i
            axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors[i]], label=f'Cluster {i}',
                          s=30, alpha=0.6)

        axes[1].legend(loc='best', fontsize=10, ncol=2)
        axes[1].set_title('PCA colored by Cluster Assignment', fontsize=14)
    else:
        axes[1].scatter(features_2d[:, 0], features_2d[:, 1], s=30, alpha=0.6)
        axes[1].set_title('PCA Visualization', fontsize=14)

    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_training_curves(history_path, save_path=None):
    """
    Plot training curves from history JSON file.

    Args:
        history_path: path to training_history.json
        save_path: path to save figure
    """
    import json
    import os

    # Check if history file exists and is valid
    if not os.path.exists(history_path):
        print(f"Warning: Training history not found at {history_path}")
        print("Skipping training curves plot.")
        return None

    try:
        with open(history_path, 'r') as f:
            history = json.load(f)

        if not history or len(history) == 0:
            print("Warning: Training history is empty.")
            print("Skipping training curves plot.")
            return None
    except json.JSONDecodeError as e:
        print(f"Warning: Training history file is corrupted: {e}")
        print("Skipping training curves plot.")
        return None

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    srcc = [h['srcc'] for h in history]
    plcc = [h['plcc'] for h in history]
    lr = [h['lr'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # SRCC curve
    axes[0, 1].plot(epochs, srcc, label='SRCC', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('SRCC', fontsize=12)
    axes[0, 1].set_title('Spearman Rank Correlation', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=max(srcc), color='r', linestyle='--', alpha=0.5)
    axes[0, 1].text(0.5, max(srcc), f'Best: {max(srcc):.4f}',
                   ha='left', va='bottom')

    # PLCC curve
    axes[1, 0].plot(epochs, plcc, label='PLCC', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('PLCC', fontsize=12)
    axes[1, 0].set_title('Pearson Linear Correlation', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=max(plcc), color='r', linestyle='--', alpha=0.5)
    axes[1, 0].text(0.5, max(plcc), f'Best: {max(plcc):.4f}',
                   ha='left', va='bottom')

    # Learning rate curve
    axes[1, 1].plot(epochs, lr, color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_sample_predictions(images, predictions, targets, cluster_ids, n_samples=16, save_path=None):
    """
    Plot sample images with their predictions.

    Args:
        images: tensor or numpy array of images [N, C, H, W]
        predictions: numpy array of predictions [N]
        targets: numpy array of targets [N]
        cluster_ids: numpy array of cluster assignments [N]
        n_samples: number of samples to plot
        save_path: path to save figure
    """
    n_samples = min(n_samples, len(images))
    indices = np.random.choice(len(images), n_samples, replace=False)

    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for idx, ax in zip(indices, axes):
        # Convert tensor to numpy if needed
        if torch.is_tensor(images):
            img = images[idx].cpu().numpy()
        else:
            img = images[idx]

        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img.transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.axis('off')

        # Add text
        pred = predictions[idx]
        target = targets[idx]
        cluster = cluster_ids[idx]
        error = abs(pred - target)

        title = f'Pred: {pred:.3f}\nGT: {target:.3f}\nCluster: {cluster}\nErr: {error:.3f}'
        ax.set_title(title, fontsize=9)

    # Remove empty subplots
    for ax in axes[n_samples:]:
        ax.remove()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig
