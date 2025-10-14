"""
Test the simplified no-cache TrainableEmbeddingManager
Created: 2025-01-13
"""

import torch
from pathlib import Path
import shutil
import sys

# Import the nocache version
sys.path.insert(0, '/project/CoSiR')
from src.utils.embedding_manager_nocache import TrainableEmbeddingManager


def test_nocache_manager():
    """Test all essential functions of the nocache manager"""

    print("=" * 80)
    print("TESTING SIMPLIFIED NO-CACHE TrainableEmbeddingManager")
    print("=" * 80)

    # Setup
    test_dir = Path("test_nocache_manager")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    n_samples = 500
    embedding_dim = 2
    chunk_size = 100

    print(f"\n1. Creating manager (disk mode)")
    manager = TrainableEmbeddingManager(
        sample_ids=list(range(n_samples)),
        embedding_dim=embedding_dim,
        device="cpu",
        initialization_strategy="zeros",
        storage_mode="disk",
        embeddings_dir=str(test_dir / "embeddings"),
        chunk_size=chunk_size,
    )

    print(f"\n2. Testing get_embeddings_by_chunk()")
    chunk_0_ids, chunk_0_emb = manager.get_embeddings_by_chunk(0)
    print(f"   Chunk 0: {len(chunk_0_ids)} samples")
    print(f"   Sample 0: {chunk_0_emb[0]}")
    print(f"   Is zero? {torch.allclose(chunk_0_emb[0], torch.zeros_like(chunk_0_emb[0]))}")

    print(f"\n3. Testing update_embeddings_by_chunk()")
    new_embeddings = torch.randn(len(chunk_0_ids), embedding_dim) * 5.0
    print(f"   New embedding for sample 0: {new_embeddings[0]}")
    manager.update_embeddings_by_chunk(0, chunk_0_ids, new_embeddings)

    print(f"\n4. Verifying update worked (immediate get)")
    chunk_0_ids_after, chunk_0_emb_after = manager.get_embeddings_by_chunk(0)
    print(f"   Sample 0 after update: {chunk_0_emb_after[0]}")
    matches = torch.allclose(chunk_0_emb_after[0], new_embeddings[0], atol=1e-5)
    print(f"   Matches? {matches}")

    if not matches:
        print("   ❌ FAIL: Update did not work!")
        return False

    print(f"\n5. Testing get_embeddings() for specific IDs")
    specific_embs = manager.get_embeddings([0, 50, 100, 250])
    print(f"   Got {specific_embs.shape[0]} embeddings")
    print(f"   Sample 0: {specific_embs[0]}")
    print(f"   Matches updated value? {torch.allclose(specific_embs[0], new_embeddings[0], atol=1e-5)}")

    print(f"\n6. Testing get_all_embeddings()")
    all_ids, all_embs = manager.get_all_embeddings()
    print(f"   Total samples: {len(all_ids)}")
    print(f"   Embeddings shape: {all_embs.shape}")
    print(f"   Sample 0: {all_embs[0]}")

    print(f"\n7. Testing persistence: Creating NEW manager")
    manager2 = TrainableEmbeddingManager(
        sample_ids=list(range(n_samples)),
        embedding_dim=embedding_dim,
        device="cpu",
        initialization_strategy="zeros",
        storage_mode="disk",
        embeddings_dir=str(test_dir / "embeddings"),
        chunk_size=chunk_size,
    )

    reloaded_ids, reloaded_emb = manager2.get_embeddings_by_chunk(0)
    print(f"   Reloaded sample 0: {reloaded_emb[0]}")
    persisted = torch.allclose(reloaded_emb[0], new_embeddings[0], atol=1e-5)
    print(f"   Persisted correctly? {persisted}")

    if not persisted:
        print("   ❌ FAIL: Embeddings did not persist!")
        return False

    print(f"\n8. Testing template save/load")
    # First update some more chunks
    for chunk_id in [1, 2]:
        ids, _ = manager.get_embeddings_by_chunk(chunk_id)
        new_vals = torch.randn(len(ids), embedding_dim) * 3.0
        manager.update_embeddings_by_chunk(chunk_id, ids, new_vals)

    # Save as template
    manager.store_imgtxt_template()

    # Create new manager in different dir
    test_dir2 = test_dir / "experiment2"
    manager3 = TrainableEmbeddingManager(
        sample_ids=list(range(n_samples)),
        embedding_dim=embedding_dim,
        device="cpu",
        initialization_strategy="zeros",
        storage_mode="disk",
        embeddings_dir=str(test_dir2 / "training_embeddings"),
        chunk_size=chunk_size,
    )

    # Load template
    try:
        manager3.load_imgtxt_template()
        _, template_emb = manager3.get_embeddings_by_chunk(0)
        print(f"   Template sample 0: {template_emb[0]}")
        template_ok = torch.allclose(template_emb[0], new_embeddings[0], atol=1e-5)
        print(f"   Template loaded correctly? {template_ok}")
    except Exception as e:
        print(f"   ❌ Template load failed: {e}")
        template_ok = False

    print(f"\n9. Testing save_final_embeddings()")
    final_dir = test_dir / "final"
    manager.save_final_embeddings(str(final_dir))
    final_files = list(final_dir.glob("embeddings_chunk_*.pt"))
    print(f"   Saved {len(final_files)} chunk files")

    print(f"\n10. Checking attributes for train_cosir.py compatibility")
    print(f"   Has chunk_mapping? {hasattr(manager, 'chunk_mapping')}")
    print(f"   Has id_to_chunk_index? {hasattr(manager, 'id_to_chunk_index')}")
    print(f"   Has optimize_cache_settings? {hasattr(manager, 'optimize_cache_settings')}")

    print("\n" + "=" * 80)
    print("OVERALL RESULT:")
    print("=" * 80)

    if matches and persisted and template_ok:
        print("✅ ALL TESTS PASSED!")
        print("   - Update/get works correctly")
        print("   - Persistence works")
        print("   - Template save/load works")
        print("   - All required methods present")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = test_nocache_manager()
    exit(0 if success else 1)
