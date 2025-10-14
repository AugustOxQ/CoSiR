"""
Test update_embeddings_by_chunk followed immediately by get_embeddings_by_chunk
Created: 2025-01-13
"""

import torch
from pathlib import Path
import shutil
from src.utils.embedding_manager import TrainableEmbeddingManager


def test_update_then_get():
    """Test that update -> get sequence works correctly"""

    print("=" * 80)
    print("TEST: Update then immediately Get embeddings")
    print("=" * 80)

    # Setup
    test_dir = Path("test_update_get_immediate")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    n_samples = 200
    embedding_dim = 2
    chunk_size = 100

    print(f"\n1. Creating TrainableEmbeddingManager")
    manager = TrainableEmbeddingManager(
        sample_ids=list(range(n_samples)),
        embedding_dim=embedding_dim,
        device="cpu",
        initialization_strategy="zeros",
        storage_mode="disk",
        embeddings_dir=str(test_dir / "embeddings"),
        chunk_size=chunk_size,
        auto_sync=True,
        sync_batch_size=100,
    )

    print(f"\n2. Initial check - chunk 0 embeddings (should be zeros)")
    chunk_0_ids, chunk_0_emb_before = manager.get_embeddings_by_chunk(0)
    print(f"   Sample 0 before update: {chunk_0_emb_before[0]}")
    print(f"   Is zero? {torch.allclose(chunk_0_emb_before[0], torch.zeros_like(chunk_0_emb_before[0]))}")

    print(f"\n3. Creating NEW non-zero embeddings")
    new_embeddings = torch.randn(len(chunk_0_ids), embedding_dim) * 5.0
    print(f"   New embedding for sample 0: {new_embeddings[0]}")

    print(f"\n4. Calling update_embeddings_by_chunk()")
    manager.update_embeddings_by_chunk(0, chunk_0_ids, new_embeddings)
    print(f"   Update complete")

    print(f"\n5. IMMEDIATELY calling get_embeddings_by_chunk() [THE KEY TEST]")
    chunk_0_ids_after, chunk_0_emb_after = manager.get_embeddings_by_chunk(0)
    print(f"   Sample 0 after update: {chunk_0_emb_after[0]}")
    print(f"   Original new value: {new_embeddings[0]}")

    print(f"\n6. Checking if they match")
    matches = torch.allclose(chunk_0_emb_after[0], new_embeddings[0], atol=1e-5)
    is_zero = torch.allclose(chunk_0_emb_after[0], torch.zeros_like(chunk_0_emb_after[0]))

    print(f"   Matches new value? {matches}")
    print(f"   Is still zero? {is_zero}")

    print("\n" + "=" * 80)
    if matches and not is_zero:
        print("✅ SUCCESS! Update -> Get sequence works correctly")
    else:
        print("❌ FAILURE! Get is not returning updated values")
        print("   This means update_embeddings_by_chunk is not working")
    print("=" * 80)

    return matches and not is_zero


if __name__ == "__main__":
    success = test_update_then_get()
    exit(0 if success else 1)
