#!/bin/bash
# Setup script for local visualization

echo "=========================================="
echo "Setting up local environment for visualization"
echo "=========================================="

# 1. Install dependencies
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib seaborn scikit-learn
pip install datasets tqdm Pillow

# 2. Check if experiments directory exists
if [ ! -d "experiments/resnet18_large_10k" ]; then
    echo ""
    echo "ERROR: experiments/resnet18_large_10k/ does not exist!"
    echo ""
    echo "You need to either:"
    echo "1. Run training first:"
    echo "   python train_with_distortions.py --experiment_name resnet18_large_10k --train_samples 1666 --epochs 50"
    echo ""
    echo "2. Or copy the experiment directory from your training environment"
    echo ""
    exit 1
fi

# 3. Check if checkpoint exists
if [ ! -f "experiments/resnet18_large_10k/best_model.pth" ]; then
    echo ""
    echo "ERROR: best_model.pth not found!"
    echo ""
    echo "Available files:"
    ls -lh experiments/resnet18_large_10k/
    echo ""
    exit 1
fi

echo ""
echo "✅ Environment ready!"
echo ""
echo "Run visualization with:"
echo "python enhanced_visualize.py --experiment experiments/resnet18_large_10k --checkpoint best_model.pth --num_images 25 --test_samples 500"
