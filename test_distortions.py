"""
Simple test script to verify all distortion types work correctly.
Run this to generate distortion_examples.png and verify all 8 distortion types display.
"""
import sys

def test_all_distortions():
    """Test all distortion types individually."""
    try:
        from distorted_dataset import DistortedImageDataset
        from datasets import load_dataset
        from PIL import Image
        import numpy as np

        print("="*60)
        print("Testing Individual Distortion Types")
        print("="*60)

        # Load a test image
        print("\nLoading test image...")
        dataset = load_dataset("cifar10", split='train')
        img = dataset[0]['img']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        print(f"✓ Image loaded: {img.size}, {img.mode}")

        # Create instance
        temp_dataset = DistortedImageDataset.__new__(DistortedImageDataset)

        # Test each distortion type
        print("\nTesting each distortion type:")
        print("-"*60)

        for i, dist_type in enumerate(DistortedImageDataset.DISTORTION_TYPES, 1):
            try:
                img_copy = img.copy()
                distorted_img, quality_score = temp_dataset.apply_distortion(
                    img_copy, dist_type, 0.5
                )

                # Verify output
                assert isinstance(distorted_img, Image.Image), f"Output is not PIL Image: {type(distorted_img)}"
                assert distorted_img.size == img.size, f"Size mismatch: {distorted_img.size} != {img.size}"
                assert distorted_img.mode == 'RGB', f"Mode is not RGB: {distorted_img.mode}"
                assert 0 <= quality_score <= 100, f"Invalid quality score: {quality_score}"

                print(f"{i}. ✓ {dist_type:20s} - Quality: {quality_score:.1f}")

            except Exception as e:
                print(f"{i}. ✗ {dist_type:20s} - ERROR: {str(e)}")
                return False

        print("-"*60)
        print("✓ All distortion types passed!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_full_visualization():
    """Run the full visualization with all distortion types."""
    print("\n" + "="*60)
    print("Running Full Visualization")
    print("="*60)

    try:
        from distorted_dataset import visualize_distortions
        visualize_distortions()
        return True
    except Exception as e:
        print(f"\n✗ Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DISTORTION TEST SUITE")
    print("="*60)

    # Test 1: Individual distortion types
    test1_passed = test_all_distortions()

    # Test 2: Full visualization
    if test1_passed:
        test2_passed = run_full_visualization()
    else:
        print("\n⚠️ Skipping full visualization due to individual test failures")
        test2_passed = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Individual distortion tests: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Full visualization:          {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("="*60)

    if test1_passed and test2_passed:
        print("\n✓ All tests passed! Check distortion_examples.png for results.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
