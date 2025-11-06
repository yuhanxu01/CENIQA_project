"""
Quick visualization script for already trained models.
Use this when training_history.json failed to save.
"""
from IPython.display import Image, display
import json
import os

def display_all_visualizations(exp_name="resnet18_gmm_mlp"):
    """Display all visualizations from an experiment."""

    viz_dir = f"experiments/{exp_name}/visualizations"

    if not os.path.exists(viz_dir):
        print(f"Error: {viz_dir} not found!")
        print("Please run test_with_viz.py first:")
        print(f"  python test_with_viz.py --experiment experiments/{exp_name} --skip_tsne")
        return

    # Display metrics
    results_path = f"{viz_dir}/test_results.json"
    if os.path.exists(results_path):
        print("="*60)
        print("测试指标")
        print("="*60)
        with open(results_path, 'r') as f:
            results = json.load(f)
            print(f"SRCC: {results['srcc']:.4f}")
            print(f"PLCC: {results['plcc']:.4f}")
            print(f"RMSE: {results['rmse']:.4f}")
            print(f"样本数: {results['n_samples']}")

    # Display all visualization images
    viz_files = {
        "predictions_scatter.png": "1. 预测散点图",
        "cluster_distribution.png": "2. 聚类分布",
        "features_pca.png": "3. PCA可视化",
        "sample_predictions.png": "4. 样本预测",
        "training_curves.png": "5. 训练曲线",
        "features_tsne.png": "6. t-SNE可视化 (可选)"
    }

    for filename, title in viz_files.items():
        filepath = f"{viz_dir}/{filename}"
        if os.path.exists(filepath):
            print("\n" + "="*60)
            print(title)
            print("="*60)
            display(Image(filepath))
        else:
            if "tsne" not in filename:  # t-SNE是可选的
                print(f"\n⚠️ {filename} not found")


if __name__ == "__main__":
    import sys

    # Get experiment name from command line or use default
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "resnet18_gmm_mlp"

    print(f"Displaying visualizations for: {exp_name}")
    display_all_visualizations(exp_name)
