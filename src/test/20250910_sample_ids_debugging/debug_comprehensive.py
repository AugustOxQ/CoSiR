#!/usr/bin/env python3
"""
Comprehensive debug script to reproduce the exact sample ID issue
"""
import sys
import os

sys.path.append("/project/CoSiR/src")
import torch
from src.utils.feature_manager import FeatureManager
from src.utils.embedding_manager import TrainableEmbeddingManager


# Simulate the exact workflow from train_cosir.py
def debug_sample_id_issue():
    print("=== Setting up test environment ===")

    # Use a temporary directory for testing
    test_dir = "/project/CoSiR/debug_features_test"
    os.makedirs(test_dir, exist_ok=True)

    # Create feature manager with same config as train_cosir.py
    feature_config = {
        "storage_dir": test_dir,
        "primary_backend": "chunked",
        "chunked_storage": {
            "enabled": True,
            "chunk_size": 256,
            "compression": False,
        },
        "cache": {
            "l1_size_mb": 256,
            "l2_size_mb": 512,
            "l3_path": None,
        },
    }

    feature_manager = FeatureManager(
        features_dir=test_dir,
        chunk_size=256,
        config=feature_config,
    )

    # Simulate what happens in the training loop
    print("\n=== Simulating feature extraction ===")

    # Simulate a batch from FeatureExtractionDataset
    batch_size = 5
    feature_dim = 512

    # These are the sample_ids that would come from FeatureExtractionDataset.__getitem__
    simulated_sample_ids = [0, 1, 2, 3, 4]  # What the dataset returns

    # Create fake features
    chunk_img = torch.randn(batch_size, feature_dim)
    chunk_txt = torch.randn(batch_size, feature_dim)
    txt_full = torch.randn(batch_size, 77, feature_dim)  # Typical text features

    print(f"Sample IDs to store: {simulated_sample_ids}")
    print(
        f"Feature shapes - img: {chunk_img.shape}, txt: {chunk_txt.shape}, txt_full: {txt_full.shape}"
    )

    # Store features using add_features_chunk (same as train_cosir.py line 157)
    batch_id = 0
    feature_manager.add_features_chunk(
        batch_id,
        chunk_img,
        chunk_txt,
        txt_full,
        simulated_sample_ids,
    )

    # Create sample_ids_list (same as train_cosir.py line 166)
    sample_ids_list = []
    sample_ids_list.extend(simulated_sample_ids)
    print(f"sample_ids_list: {sample_ids_list}")

    print("\n=== Testing feature retrieval ===")

    # Now test what happens when we try to get features back
    try:
        # This would be called during training to get features
        batch_sample_ids = sample_ids_list[:3]  # Get first 3 samples
        print(f"Requesting features for sample IDs: {batch_sample_ids}")

        loaded_features = feature_manager.get_features(batch_sample_ids)
        print(f"Successfully loaded features with keys: {list(loaded_features.keys())}")

        if "img_features" in loaded_features:
            print(f"Loaded img_features shape: {loaded_features['img_features'].shape}")
            print(f"Loaded txt_features shape: {loaded_features['txt_features'].shape}")

    except Exception as e:
        print(f"ERROR loading features: {e}")
        print(f"Error type: {type(e).__name__}")

        # Debug: check what's actually stored in the chunk
        print("\n=== Debugging chunk contents ===")
        try:
            chunk_data = feature_manager.get_chunk(batch_id)
            print(f"Chunk {batch_id} sample_ids: {chunk_data['sample_ids']}")
            print(
                f"Chunk {batch_id} features keys: {list(chunk_data['features'].keys())}"
            )

            # Check if there's a mismatch
            stored_sample_ids = chunk_data["sample_ids"]
            requested_sample_ids = batch_sample_ids

            print(f"Stored sample IDs: {stored_sample_ids}")
            print(f"Requested sample IDs: {requested_sample_ids}")
            print(
                f"Missing sample IDs: {set(requested_sample_ids) - set(stored_sample_ids)}"
            )

        except Exception as e2:
            print(f"Error reading chunk: {e2}")

    print("\n=== Testing embedding manager ===")

    # Test embedding manager with same sample IDs
    try:
        embedding_manager = TrainableEmbeddingManager(
            sample_ids=sample_ids_list,
            embedding_dim=64,
            storage_mode="memory",
        )

        # Test getting embeddings (this should work)
        test_embeddings = embedding_manager.get_embeddings(batch_sample_ids)
        print(f"Successfully got embeddings shape: {test_embeddings.shape}")

    except Exception as e:
        print(f"ERROR with embedding manager: {e}")

    # Clean up
    import shutil

    shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    debug_sample_id_issue()
