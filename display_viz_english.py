"""
Display visualizations for trained model - English version
Use this in Colab after running test_with_viz.py
"""
from IPython.display import Image, display
import json
import os

# Configuration
exp_name = "resnet18_gmm_mlp"
viz_dir = f"experiments/{exp_name}/visualizations"

# Check if visualization directory exists
if not os.path.exists(viz_dir):
    print(f"Error: {viz_dir} not found!")
    print("Please run test_with_viz.py first:")
    print(f"  !python test_with_viz.py --experiment experiments/{exp_name} --skip_tsne")
else:
    # 1. Display test metrics
    print("="*60)
    print("Test Metrics")
    print("="*60)

    results_path = f'{viz_dir}/test_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            print(f"SRCC: {results['srcc']:.4f}")
            print(f"PLCC: {results['plcc']:.4f}")
            print(f"RMSE: {results['rmse']:.4f}")
            print(f"Number of samples: {results['n_samples']}")
    else:
        print(f"Warning: {results_path} not found")

    # 2. Display prediction scatter plot
    print("\n" + "="*60)
    print("Prediction Scatter Plot")
    print("="*60)
    scatter_path = f'{viz_dir}/predictions_scatter.png'
    if os.path.exists(scatter_path):
        display(Image(scatter_path))
    else:
        print(f"Warning: {scatter_path} not found")

    # 3. Display cluster distribution
    print("\n" + "="*60)
    print("Cluster Distribution")
    print("="*60)
    cluster_path = f'{viz_dir}/cluster_distribution.png'
    if os.path.exists(cluster_path):
        display(Image(cluster_path))
    else:
        print(f"Warning: {cluster_path} not found")

    # 4. Display PCA visualization
    print("\n" + "="*60)
    print("PCA Feature Visualization")
    print("="*60)
    pca_path = f'{viz_dir}/features_pca.png'
    if os.path.exists(pca_path):
        display(Image(pca_path))
    else:
        print(f"Warning: {pca_path} not found")

    # 5. Display sample predictions
    print("\n" + "="*60)
    print("Sample Predictions")
    print("="*60)
    samples_path = f'{viz_dir}/sample_predictions.png'
    if os.path.exists(samples_path):
        display(Image(samples_path))
    else:
        print(f"Warning: {samples_path} not found")

    # 6. Display training curves
    print("\n" + "="*60)
    print("Training Curves")
    print("="*60)
    curves_path = f'{viz_dir}/training_curves.png'
    if os.path.exists(curves_path):
        display(Image(curves_path))
    else:
        print(f"Warning: {curves_path} not found")

    # 7. Display t-SNE visualization (optional)
    print("\n" + "="*60)
    print("t-SNE Feature Visualization (Optional)")
    print("="*60)
    tsne_path = f'{viz_dir}/features_tsne.png'
    if os.path.exists(tsne_path):
        display(Image(tsne_path))
    else:
        print("Info: t-SNE visualization not generated (use --skip_tsne flag to skip)")

    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"\nAll visualizations are saved in: {viz_dir}")
    print("\nTo download all visualizations:")
    print("  from google.colab import files")
    print(f"  !zip -r viz_results.zip {viz_dir}")
    print("  files.download('viz_results.zip')")
