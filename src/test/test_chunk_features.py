#!/usr/bin/env python3
"""
Test the new get_features_by_chunk method
"""

import sys

sys.path.append(".")
from src.utils import FeatureManager
import torch


def test_get_features_by_chunk():
    """Test the new get_features_by_chunk method"""
    print("Testing get_features_by_chunk method...")

    # Use the existing test data
    feature_manager = FeatureManager(
        "/project/CoSiR/data/test_features_new", chunk_size=20
    )

    # Test loading chunk 0
    print("\nLoading chunk 0...")
    chunk_0_features = feature_manager.get_features_by_chunk(0)
    print(f"Chunk 0 features keys: {chunk_0_features.keys()}")
    print(
        f"Chunk 0 shapes: img={chunk_0_features['img_features'].shape}, txt={chunk_0_features['txt_features'].shape}"
    )

    # Test loading chunk 1
    print("\nLoading chunk 1...")
    chunk_1_features = feature_manager.get_features_by_chunk(1)
    print(
        f"Chunk 1 shapes: img={chunk_1_features['img_features'].shape}, txt={chunk_1_features['txt_features'].shape}"
    )

    # Test with feature filtering
    print("\nLoading chunk 0 with only img_features...")
    img_only = feature_manager.get_features_by_chunk(0, feature_types=["img_features"])
    print(f"Filtered features keys: {img_only.keys()}")
    print(f"Img features shape: {img_only['img_features'].shape}")

    # Test cache hit on second load
    print("\nLoading chunk 0 again (should hit cache)...")
    cache_stats_before = feature_manager.cache.stats.copy()
    chunk_0_again = feature_manager.get_features_by_chunk(0)
    cache_stats_after = feature_manager.cache.stats

    print(f"Cache stats before: {cache_stats_before}")
    print(f"Cache stats after: {cache_stats_after}")
    print(
        f"Cache hit occurred: {cache_stats_after['hits'] > cache_stats_before['hits']}"
    )

    # Compare with get_chunk method
    print("\nComparing with get_chunk method...")
    chunk_data = feature_manager.get_chunk(0)
    print(f"get_chunk returns sample_ids: {chunk_data['sample_ids'][:5]}...")
    print(f"get_chunk features keys: {chunk_data['features'].keys()}")
    print(f"get_chunk img shape: {chunk_data['features']['img_features'].shape}")

    # Verify they return the same data
    assert torch.equal(
        chunk_0_features["img_features"], chunk_data["features"]["img_features"]
    )
    assert torch.equal(
        chunk_0_features["txt_features"], chunk_data["features"]["txt_features"]
    )
    print("✓ get_features_by_chunk and get_chunk return identical feature data")

    # Test error handling
    print("\nTesting error handling...")
    try:
        feature_manager.get_features_by_chunk(999)  # Non-existent chunk
        print("ERROR: Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"✓ Correctly raised error for non-existent chunk: {e}")

    print("\n✅ All get_features_by_chunk tests passed!")


if __name__ == "__main__":
    test_get_features_by_chunk()
