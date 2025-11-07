"""
Automated script to run both experiments and compare them
完整的对比实验流程：训练两个模型 + 对比可视化
"""
import subprocess
import sys
import time
from pathlib import Path
import json


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True, text=True)
        elapsed = time.time() - start_time

        print(f"\n✅ {description} completed in {elapsed/60:.2f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False


def check_checkpoint(exp_dir, checkpoint_name='best_model.pth'):
    """Check if checkpoint exists."""
    checkpoint_path = Path(exp_dir) / checkpoint_name
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024*1024)
        print(f"✅ Checkpoint exists: {checkpoint_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False


def load_training_history(exp_dir):
    """Load and display training history."""
    history_path = Path(exp_dir) / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

        best_srcc = max(h['srcc'] for h in history)
        best_epoch = [h['epoch'] for h in history if h['srcc'] == best_srcc][0]

        print(f"  Best SRCC: {best_srcc:.4f} (Epoch {best_epoch})")
        print(f"  Final SRCC: {history[-1]['srcc']:.4f}")
        print(f"  Total epochs: {len(history)}")
        return True
    else:
        print(f"⚠️  Training history not found")
        return False


def main():
    print("="*80)
    print("🔬 Automated Comparison Experiment")
    print("="*80)
    print("This script will:")
    print("  1. Train Simple Baseline (no GMM)")
    print("  2. Train GMM-based model")
    print("  3. Compare both models")
    print("  4. Generate comparison visualizations")
    print("="*80)

    # Configuration
    config = {
        'train_samples': 4666,
        'val_samples': 466,
        'distortions_per_image': 5,
        'batch_size': 64,
        'epochs': 50,
        'lr': 1e-3
    }

    exp_simple = 'experiments/resnet18_simple_baseline'
    exp_gmm = 'experiments/resnet18_large_10k'

    # Ask user if they want to skip training
    print("\n" + "="*80)
    print("Training Options")
    print("="*80)

    skip_simple = False
    skip_gmm = False

    # Check if experiments already exist
    if Path(exp_simple).exists() and (Path(exp_simple) / 'best_model.pth').exists():
        print(f"\n✅ Simple baseline experiment already exists: {exp_simple}")
        response = input("Skip training simple baseline? (y/n): ").strip().lower()
        skip_simple = (response == 'y')
    else:
        print(f"\n❌ Simple baseline not found, will train")

    if Path(exp_gmm).exists() and (Path(exp_gmm) / 'best_model.pth').exists():
        print(f"\n✅ GMM-based experiment already exists: {exp_gmm}")
        response = input("Skip training GMM-based model? (y/n): ").strip().lower()
        skip_gmm = (response == 'y')
    else:
        print(f"\n❌ GMM-based model not found, will train")

    # Step 1: Train Simple Baseline
    if not skip_simple:
        cmd_simple = [
            'python', 'train_simple_baseline.py',
            '--experiment_name', 'resnet18_simple_baseline',
            '--train_samples', str(config['train_samples']),
            '--val_samples', str(config['val_samples']),
            '--distortions_per_image', str(config['distortions_per_image']),
            '--batch_size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--lr', str(config['lr'])
        ]

        success = run_command(cmd_simple, "Training Simple Baseline (no GMM)")

        if not success:
            print("\n❌ Simple baseline training failed!")
            sys.exit(1)

        # Check checkpoint
        check_checkpoint(exp_simple)
        load_training_history(exp_simple)
    else:
        print("\n⏭️  Skipping simple baseline training")
        load_training_history(exp_simple)

    # Step 2: Train GMM-based model
    if not skip_gmm:
        cmd_gmm = [
            'python', 'train_with_distortions.py',
            '--experiment_name', 'resnet18_large_10k',
            '--train_samples', str(config['train_samples']),
            '--val_samples', str(config['val_samples']),
            '--distortions_per_image', str(config['distortions_per_image']),
            '--batch_size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--lr', str(config['lr']),
            '--cluster_loss_weight', '0',
            '--balance_weight', '0',
            '--entropy_weight', '0',
            '--refit_interval', '0'
        ]

        success = run_command(cmd_gmm, "Training GMM-based Model")

        if not success:
            print("\n❌ GMM-based training failed!")
            sys.exit(1)

        # Check checkpoint
        check_checkpoint(exp_gmm)
        load_training_history(exp_gmm)
    else:
        print("\n⏭️  Skipping GMM-based model training")
        load_training_history(exp_gmm)

    # Step 3: Compare models
    cmd_compare = [
        'python', 'compare_experiments.py',
        '--exp_simple', exp_simple,
        '--exp_gmm', exp_gmm,
        '--test_samples', '1000',
        '--batch_size', '64'
    ]

    success = run_command(cmd_compare, "Comparing Models")

    if not success:
        print("\n❌ Comparison failed!")
        sys.exit(1)

    # Final summary
    print("\n" + "="*80)
    print("✅ All Experiments Complete!")
    print("="*80)

    print(f"\nResults:")
    print(f"  - Simple Baseline: {exp_simple}")
    print(f"  - GMM-based Model: {exp_gmm}")
    print(f"  - Comparison: experiments/comparison")

    print(f"\nGenerated files:")
    print(f"  - experiments/comparison/comparison_dashboard.png")
    print(f"  - experiments/comparison/comparison_results.json")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
