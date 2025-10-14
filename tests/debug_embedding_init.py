"""
Debug script to test embedding initialization issue
Created: 2025-01-13
"""

import torch
import numpy as np
from pathlib import Path
from src.utils.embedding_manager import TrainableEmbeddingManager


def test_embedding_initialization():
    """Test if embeddings are properly stored after initialization"""

    print("=" * 80)
    print("Testing Embedding Initialization Storage")
    print("=" * 80)

    # Create test data
    n_samples = 1000
    embedding_dim = 2
    sample_ids = list(range(n_samples))

    # Test directory
    test_dir = Path("test_embeddings_debug")
    test_dir.mkdir(exist_ok=True)

    print(f"\n1. Creating TrainableEmbeddingManager with {n_samples} samples")
    print(f"   Storage mode: disk")
    print(f"   Embedding dim: {embedding_dim}")

    # Create manager with disk storage
    manager = TrainableEmbeddingManager(
        sample_ids=sample_ids,
        embedding_dim=embedding_dim,
        device="cpu",
        initialization_strategy="zeros",
        storage_mode="disk",
        embeddings_dir=str(test_dir),
        chunk_size=100,
        auto_sync=True,
    )

    print(f"\n2. Checking initial embeddings (should be zeros)")
    initial_embeddings = manager.get_embeddings([0, 1, 2])
    print(f"   Sample 0-2 embeddings:\n{initial_embeddings}")
    print(f"   Are all zeros? {torch.allclose(initial_embeddings, torch.zeros_like(initial_embeddings))}")

    print(f"\n3. Manually updating chunk 0 with non-zero values")
    chunk_size = 100
    new_values = torch.randn(chunk_size, embedding_dim) * 5.0  # Random values scaled
    chunk_0_sample_ids = list(range(chunk_size))

    # This simulates what initialize_embeddings_imgtxt does
    manager.update_embeddings_by_chunk(0, chunk_0_sample_ids, new_values)

    print(f"   Updated chunk 0 with random values")
    print(f"   Dirty chunks: {manager.dirty_chunks}")
    print(f"   Number of dirty chunks: {len(manager.dirty_chunks)}")

    # Check in-memory/cache
    print(f"\n4. Checking embeddings immediately after update (from cache)")
    updated_embeddings = manager.get_embeddings([0, 1, 2])
    print(f"   Sample 0-2 embeddings:\n{updated_embeddings}")
    print(f"   Are NOT zeros? {not torch.allclose(updated_embeddings, torch.zeros_like(updated_embeddings))}")

    # Check if synced to disk
    chunk_file = test_dir / "embeddings_chunk_0.pt"
    print(f"\n5. Checking if chunk file exists on disk: {chunk_file}")
    print(f"   Exists? {chunk_file.exists()}")

    if chunk_file.exists():
        chunk_data = torch.load(chunk_file, map_location="cpu")
        print(f"   Number of samples in chunk file: {len(chunk_data)}")
        if 0 in chunk_data:
            print(f"   Sample 0 from disk:\n{chunk_data[0]}")
            print(f"   Is NOT zero on disk? {not torch.allclose(chunk_data[0], torch.zeros_like(chunk_data[0]))}")
        else:
            print(f"   ERROR: Sample 0 not found in chunk file!")

    # Now test the sync
    print(f"\n6. Manually calling sync_dirty_chunks_to_disk()")
    manager.sync_dirty_chunks_to_disk()
    print(f"   Dirty chunks after sync: {manager.dirty_chunks}")

    # Check disk again
    print(f"\n7. Re-checking disk after manual sync")
    if chunk_file.exists():
        chunk_data = torch.load(chunk_file, map_location="cpu")
        if 0 in chunk_data:
            print(f"   Sample 0 from disk after sync:\n{chunk_data[0]}")
            print(f"   Is NOT zero on disk? {not torch.allclose(chunk_data[0], torch.zeros_like(chunk_data[0]))}")

    # Test reloading
    print(f"\n8. Creating NEW manager to test reload from disk")
    manager2 = TrainableEmbeddingManager(
        sample_ids=sample_ids,
        embedding_dim=embedding_dim,
        device="cpu",
        initialization_strategy="zeros",
        storage_mode="disk",
        embeddings_dir=str(test_dir),
        chunk_size=100,
        auto_sync=True,
    )

    reloaded_embeddings = manager2.get_embeddings([0, 1, 2])
    print(f"   Reloaded sample 0-2 embeddings:\n{reloaded_embeddings}")
    print(f"   Match updated values? {torch.allclose(reloaded_embeddings, updated_embeddings, atol=1e-5)}")
    print(f"   Are still zeros? {torch.allclose(reloaded_embeddings, torch.zeros_like(reloaded_embeddings))}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    print("If embeddings are zeros after reload, it means:")
    print("1. update_embeddings_by_chunk() marks chunks as dirty but doesn't sync")
    print("2. The sync only happens when auto_sync batch threshold is reached")
    print("3. initialize_embeddings_imgtxt() needs to call sync at the end")
    print("=" * 80)


if __name__ == "__main__":
    test_embedding_initialization()
