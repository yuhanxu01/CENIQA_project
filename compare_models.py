"""
Compare training results between GMM and Simple models.
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_history(experiment_dir):
    """Load training history from experiment directory."""
    history_path = Path(experiment_dir) / 'history.json'
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def print_comparison(gmm_history, simple_history, gmm_name, simple_name):
    """Print comparison table of final results."""
    gmm_final = gmm_history['val'][-1]
    simple_final = simple_history['val'][-1]

    # Find best SRCC for each model
    gmm_best_srcc = max([epoch['srcc'] for epoch in gmm_history['val']])
    simple_best_srcc = max([epoch['srcc'] for epoch in simple_history['val']])

    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)

    print(f"\n{'Metric':<20} {'GMM Model':<25} {'Simple Model':<25} {'Winner':<10}")
    print("-" * 80)

    # Final epoch metrics
    metrics = [
        ('Final Val Loss', gmm_final['loss'], simple_final['loss'], 'lower'),
        ('Final Val SRCC', gmm_final['srcc'], simple_final['srcc'], 'higher'),
        ('Final Val PLCC', gmm_final['plcc'], simple_final['plcc'], 'higher'),
        ('Final Val RMSE', gmm_final['rmse'], simple_final['rmse'], 'lower'),
    ]

    for name, gmm_val, simple_val, better in metrics:
        gmm_str = f"{gmm_val:.4f}"
        simple_str = f"{simple_val:.4f}"

        if better == 'lower':
            winner = "GMM ✓" if gmm_val < simple_val else "Simple ✓"
        else:
            winner = "GMM ✓" if gmm_val > simple_val else "Simple ✓"

        print(f"{name:<20} {gmm_str:<25} {simple_str:<25} {winner:<10}")

    print("-" * 80)

    # Best metrics
    print(f"{'Best Val SRCC':<20} {gmm_best_srcc:.4f}{'':<20} {simple_best_srcc:.4f}{'':<20} "
          f"{'GMM ✓' if gmm_best_srcc > simple_best_srcc else 'Simple ✓':<10}")

    print("-" * 80)

    # Training efficiency
    gmm_epochs = len(gmm_history['train'])
    simple_epochs = len(simple_history['train'])

    print(f"{'Epochs Trained':<20} {gmm_epochs:<25} {simple_epochs:<25}")

    # Improvement over baseline
    improvement = ((gmm_final['srcc'] - simple_final['srcc']) / simple_final['srcc']) * 100
    print(f"\nGMM SRCC improvement over Simple: {improvement:+.2f}%")

    print("="*80 + "\n")


def plot_comparison(gmm_history, simple_history, gmm_name, simple_name, save_dir):
    """Plot training curves comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GMM vs Simple Model Comparison', fontsize=16, fontweight='bold')

    epochs_gmm = range(1, len(gmm_history['train']) + 1)
    epochs_simple = range(1, len(simple_history['train']) + 1)

    # Training Loss
    ax = axes[0, 0]
    ax.plot(epochs_gmm, [e['loss'] for e in gmm_history['train']],
            label='GMM Model', marker='o', markersize=3, linewidth=2)
    ax.plot(epochs_simple, [e['loss'] for e in simple_history['train']],
            label='Simple Model', marker='s', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation Loss
    ax = axes[0, 1]
    ax.plot(epochs_gmm, [e['loss'] for e in gmm_history['val']],
            label='GMM Model', marker='o', markersize=3, linewidth=2)
    ax.plot(epochs_simple, [e['loss'] for e in simple_history['val']],
            label='Simple Model', marker='s', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SRCC
    ax = axes[1, 0]
    ax.plot(epochs_gmm, [e['srcc'] for e in gmm_history['val']],
            label='GMM Model', marker='o', markersize=3, linewidth=2)
    ax.plot(epochs_simple, [e['srcc'] for e in simple_history['val']],
            label='Simple Model', marker='s', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SRCC')
    ax.set_title('Validation SRCC Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PLCC
    ax = axes[1, 1]
    ax.plot(epochs_gmm, [e['plcc'] for e in gmm_history['val']],
            label='GMM Model', marker='o', markersize=3, linewidth=2)
    ax.plot(epochs_simple, [e['plcc'] for e in simple_history['val']],
            label='Simple Model', marker='s', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PLCC')
    ax.set_title('Validation PLCC Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = Path(save_dir) / 'model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")

    # Try to show plot
    try:
        plt.show()
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Compare GMM and Simple model results')
    parser.add_argument('--gmm_dir', type=str, required=True,
                       help='Path to GMM model experiment directory')
    parser.add_argument('--simple_dir', type=str, required=True,
                       help='Path to Simple model experiment directory')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                       help='Directory to save comparison plots')

    args = parser.parse_args()

    # Load histories
    print(f"Loading GMM model history from: {args.gmm_dir}")
    gmm_history = load_history(args.gmm_dir)

    print(f"Loading Simple model history from: {args.simple_dir}")
    simple_history = load_history(args.simple_dir)

    # Extract experiment names
    gmm_name = Path(args.gmm_dir).name
    simple_name = Path(args.simple_dir).name

    # Print comparison
    print_comparison(gmm_history, simple_history, gmm_name, simple_name)

    # Plot comparison
    plot_comparison(gmm_history, simple_history, gmm_name, simple_name, args.save_dir)


if __name__ == '__main__':
    main()
