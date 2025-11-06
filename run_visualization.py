"""
Quick visualization launcher - automatically finds your experiment and runs visualization.

This script will:
1. Find your experiment directory
2. Run enhanced visualization showing images, scores, clusters, and accuracy
3. Save all results to the visualizations folder

Usage:
    python run_visualization.py
    python run_visualization.py --num_images 50  # Show more images
"""
import os
import sys
import subprocess
from pathlib import Path
import argparse


def find_experiment_dir():
    """Find the most recent experiment directory."""
    exp_base = Path('experiments')

    if not exp_base.exists():
        print("Error: No 'experiments' directory found!")
        print("Please make sure you have run training first.")
        return None

    # Find all experiment directories
    exp_dirs = [d for d in exp_base.iterdir() if d.is_dir()]

    if not exp_dirs:
        print("Error: No experiments found in 'experiments' directory!")
        return None

    # Find the most recent one (based on modification time)
    latest_exp = max(exp_dirs, key=lambda d: d.stat().st_mtime)

    return latest_exp


def check_checkpoint(exp_dir):
    """Check if checkpoint exists."""
    checkpoints = ['best_model.pth', 'checkpoint.pth', 'final_model.pth']

    for ckpt in checkpoints:
        ckpt_path = exp_dir / ckpt
        if ckpt_path.exists():
            return ckpt

    return None


def main():
    parser = argparse.ArgumentParser(description='Quick visualization launcher')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment directory (default: auto-detect latest)')
    parser.add_argument('--num_images', type=int, default=25,
                       help='Number of images to visualize (default: 25)')
    parser.add_argument('--test_samples', type=int, default=500,
                       help='Number of test samples (default: 500)')
    args = parser.parse_args()

    print("="*80)
    print("Image Quality Assessment - Quick Visualization Launcher")
    print("="*80)

    # Find experiment directory
    if args.experiment:
        exp_dir = Path(args.experiment)
        if not exp_dir.exists():
            print(f"Error: Experiment directory not found: {exp_dir}")
            return
    else:
        print("\nAuto-detecting experiment directory...")
        exp_dir = find_experiment_dir()
        if exp_dir is None:
            return

    print(f"\nUsing experiment: {exp_dir}")

    # Check for checkpoint
    checkpoint = check_checkpoint(exp_dir)
    if checkpoint is None:
        print("\nError: No checkpoint found in experiment directory!")
        print("Looking for: best_model.pth, checkpoint.pth, or final_model.pth")
        return

    print(f"Using checkpoint: {checkpoint}")

    # Check if config exists
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        print("\nError: config.json not found in experiment directory!")
        return

    print("\n" + "="*80)
    print("Running Enhanced Visualization")
    print("="*80)
    print(f"Images to display: {args.num_images}")
    print(f"Test samples: {args.test_samples}")
    print("="*80)

    # Run enhanced visualization
    cmd = [
        'python', 'enhanced_visualize.py',
        '--experiment', str(exp_dir),
        '--checkpoint', checkpoint,
        '--num_images', str(args.num_images),
        '--test_samples', str(args.test_samples)
    ]

    print(f"\nExecuting: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError running visualization: {e}")
        return
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
        return

    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"\nResults saved to: {exp_dir}/enhanced_visualizations/")
    print("\nGenerated visualizations:")
    print("  1. comprehensive_metrics.png - Overall performance metrics and cluster analysis")
    print("  2. image_grid_detailed.png   - Grid of images with predictions and clusters")
    print("  3. cluster_examples.png      - Representative samples from each cluster")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
