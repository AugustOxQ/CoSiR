"""
Test script for text-to-image retrieval functions
Created: 2025-11-19
Purpose: Test retrieve_image_with_condition_fast and visualize_angular_semantics_text_to_image_fast
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.tools import (
    retrieve_image_with_condition_fast,
    visualize_angular_semantics_text_to_image_fast,
    get_representatives_polar_grid,
)


def create_mock_data(num_images=100, num_texts_per_image=5):
    """Create mock data for testing"""
    embed_dim = 512

    # Create random embeddings
    all_img_emb = torch.randn(num_images, embed_dim)
    all_txt_emb = torch.randn(num_images * num_texts_per_image, embed_dim)

    # Normalize embeddings
    all_img_emb = torch.nn.functional.normalize(all_img_emb, p=2, dim=1)
    all_txt_emb = torch.nn.functional.normalize(all_txt_emb, p=2, dim=1)

    # Create text list
    all_raw_text = [f"Caption {i//num_texts_per_image}_{i%num_texts_per_image}"
                    for i in range(num_images * num_texts_per_image)]

    # Create text_to_image_map
    text_to_image_map = []
    for img_idx in range(num_images):
        text_to_image_map += [img_idx] * num_texts_per_image
    text_to_image_map = torch.LongTensor(text_to_image_map)

    # Create mock images (random colored squares for visualization)
    all_raw_image = []
    for i in range(num_images):
        # Create a random colored image
        img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        all_raw_image.append(img_array)

    return all_img_emb, all_txt_emb, all_raw_text, text_to_image_map, all_raw_image


def test_retrieve_image_with_condition_fast():
    """Test the retrieve_image_with_condition_fast function"""
    print("\n" + "="*80)
    print("Testing retrieve_image_with_condition_fast")
    print("="*80)

    # Create mock model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a simple mock model with combine method
    class MockModel:
        def __init__(self):
            self.device = device

        def eval(self):
            pass

        def combine(self, txt_emb, img_emb, condition):
            # Simple modulation: add condition effect
            # In reality, this uses gated combiner
            modulated = txt_emb + 0.1 * torch.randn_like(txt_emb).to(txt_emb.device)
            return modulated

    model = MockModel()

    # Create mock data
    precomputed_data = create_mock_data(num_images=50, num_texts_per_image=5)
    all_img_emb, all_txt_emb, all_raw_text, text_to_image_map, all_raw_image = precomputed_data

    print(f"Data shapes:")
    print(f"  Images: {all_img_emb.shape}")
    print(f"  Texts: {all_txt_emb.shape}")
    print(f"  Raw images: {len(all_raw_image)}")
    print(f"  Text to image map: {text_to_image_map.shape}")

    # Test with different conditions
    test_conditions = [
        torch.tensor([0.0, 0.0]),  # Zero condition
        torch.tensor([1.0, 0.0]),  # Right
        torch.tensor([0.0, 1.0]),  # Up
        torch.tensor([-1.0, 0.0]), # Left
        torch.tensor([0.0, -1.0]), # Down
    ]

    for i, condition in enumerate(test_conditions):
        print(f"\nTest {i+1}: Condition = {condition.numpy()}")

        # Test with fixed query text
        query_text_id = 10
        top_images = retrieve_image_with_condition_fast(
            model=model,
            precomputed_data=precomputed_data,
            condition=condition,
            query_text_id=query_text_id,
            k=5,
            device=device
        )

        print(f"  Query text ID: {query_text_id}")
        print(f"  Query text: '{all_raw_text[query_text_id]}'")
        print(f"  Retrieved {len(top_images)} images")
        print(f"  Image shapes: {[img.shape for img in top_images[:3]]}")

    print("\n✓ retrieve_image_with_condition_fast test passed!")
    return True


def test_visualize_angular_semantics_text_to_image_fast():
    """Test the visualize_angular_semantics_text_to_image_fast function"""
    print("\n" + "="*80)
    print("Testing visualize_angular_semantics_text_to_image_fast")
    print("="*80)

    # Create mock model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MockModel:
        def __init__(self):
            self.device = device

        def eval(self):
            pass

        def combine(self, txt_emb, img_emb, condition):
            # Add some variation based on condition
            angle = torch.atan2(condition[:, 1], condition[:, 0])
            noise = 0.1 * angle.unsqueeze(1) * torch.randn_like(txt_emb).to(txt_emb.device)
            modulated = txt_emb + noise
            return modulated

    model = MockModel()

    # Create mock data
    precomputed_data = create_mock_data(num_images=100, num_texts_per_image=5)

    # Generate polar grid conditions
    conditions_2d = get_representatives_polar_grid(
        learned_conditions=None,
        num_angles=12,
        num_radii=3,
        max_radius=2.0
    )

    print(f"Generated {len(conditions_2d)} conditions")
    print(f"Conditions shape: {conditions_2d.shape}")

    # Create visualization
    save_path = "/project/CoSiR/src/test/20251119_text_to_image_retrieval/test_visualization.png"

    try:
        fig = visualize_angular_semantics_text_to_image_fast(
            conditions_2d=conditions_2d,
            model=model,
            precomputed_data=precomputed_data,
            save_path=save_path,
            device=device
        )

        print(f"\n✓ Visualization created successfully!")
        print(f"  Saved to: {save_path}")

        # Close the figure to free memory
        plt.close(fig)

        return True

    except Exception as e:
        print(f"\n✗ Visualization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Text-to-Image Retrieval Function Tests")
    print("="*80)

    success = True

    # Test 1: retrieve_image_with_condition_fast
    try:
        if not test_retrieve_image_with_condition_fast():
            success = False
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 2: visualize_angular_semantics_text_to_image_fast
    try:
        if not test_visualize_angular_semantics_text_to_image_fast():
            success = False
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Summary
    print("\n" + "="*80)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*80 + "\n")

    return success


if __name__ == "__main__":
    main()
