#!/usr/bin/env python3
"""
Test script for template embedding functionality.

This script demonstrates how the template embedding save/load feature works.
"""

import os
import tempfile
import shutil
from pathlib import Path
import torch
from src.utils import TrainableEmbeddingManager


def test_template_embedding():
    """Test the template embedding save and load functionality"""

    print("=" * 80)
    print("Testing Template Embedding Functionality")
    print("=" * 80)

    # Create a temporary directory structure mimicking the actual setup
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create directory structure:
        # tmpdir/CoSiR_Experiment/test_dataset/20251013_test/training_embeddings/
        experiment_dir = tmpdir / "CoSiR_Experiment" / "test_dataset" / "20251013_test"
        embeddings_dir = experiment_dir / "training_embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n1. Created test directory structure:")
        print(f"   {embeddings_dir}")

        # Create a sample embedding manager
        sample_ids = list(range(100))  # 100 samples
        embedding_dim = 2

        print(f"\n2. Creating TrainableEmbeddingManager:")
        print(f"   - Sample IDs: {len(sample_ids)}")
        print(f"   - Embedding dim: {embedding_dim}")
        print(f"   - Storage mode: disk")

        embedding_manager = TrainableEmbeddingManager(
            sample_ids=sample_ids,
            embedding_dim=embedding_dim,
            storage_mode="disk",
            device="cpu",
            initialization_strategy="normal",
            embeddings_dir=str(embeddings_dir),
            chunk_size=20,  # 5 chunks total
        )

        print(f"\n3. Initialized embeddings:")
        print(f"   - Number of chunks: {len(embedding_manager.chunk_mapping)}")
        chunk_files = list(embeddings_dir.glob("embeddings_chunk_*.pt"))
        print(f"   - Chunk files created: {len(chunk_files)}")

        # Get original embeddings
        original_sample_ids, original_embeddings = embedding_manager.get_embeddings_by_chunk(0)
        print(f"\n4. Original embeddings (chunk 0):")
        print(f"   - Sample IDs: {original_sample_ids[:5]}...")
        print(f"   - Embeddings shape: {original_embeddings.shape}")
        print(f"   - Sample values:\n{original_embeddings[:3]}")

        # Store as template
        print(f"\n5. Storing embeddings as template...")
        embedding_manager.store_imgtxt_template()

        # Check template directory
        template_dir = embeddings_dir.parent.parent / "template_embeddings"
        template_files = list(template_dir.glob("embeddings_chunk_*.pt"))
        print(f"\n6. Template directory created:")
        print(f"   - Location: {template_dir}")
        print(f"   - Template files: {len(template_files)}")

        # Modify embeddings to simulate training
        print(f"\n7. Modifying embeddings (simulating training)...")
        for chunk_id in range(len(embedding_manager.chunk_mapping)):
            chunk_sample_ids, chunk_embeddings = embedding_manager.get_embeddings_by_chunk(chunk_id)
            # Add random noise to simulate training
            modified_embeddings = chunk_embeddings + torch.randn_like(chunk_embeddings) * 0.5
            embedding_manager.update_embeddings_by_chunk(chunk_id, chunk_sample_ids, modified_embeddings)

        embedding_manager.sync_dirty_chunks_to_disk()

        # Get modified embeddings
        modified_sample_ids, modified_embeddings = embedding_manager.get_embeddings_by_chunk(0)
        print(f"   - Modified embeddings (chunk 0):\n{modified_embeddings[:3]}")

        # Create a new embedding manager for a different experiment
        print(f"\n8. Creating new experiment with different directory...")
        new_experiment_dir = tmpdir / "CoSiR_Experiment" / "test_dataset" / "20251013_test2"
        new_embeddings_dir = new_experiment_dir / "training_embeddings"
        new_embeddings_dir.mkdir(parents=True, exist_ok=True)

        new_embedding_manager = TrainableEmbeddingManager(
            sample_ids=sample_ids,
            embedding_dim=embedding_dim,
            storage_mode="disk",
            device="cpu",
            initialization_strategy="normal",  # Will be overwritten by template
            embeddings_dir=str(new_embeddings_dir),
            chunk_size=20,
        )

        print(f"   - New embeddings dir: {new_embeddings_dir}")

        # Load from template
        print(f"\n9. Loading template embeddings into new experiment...")
        new_embedding_manager.load_imgtxt_template()

        # Verify loaded embeddings match original
        loaded_sample_ids, loaded_embeddings = new_embedding_manager.get_embeddings_by_chunk(0)
        print(f"\n10. Verifying loaded embeddings match original:")
        print(f"    - Loaded embeddings (chunk 0):\n{loaded_embeddings[:3]}")
        print(f"    - Original embeddings (chunk 0):\n{original_embeddings[:3]}")

        # Check if they match (should match original, not modified)
        embeddings_match = torch.allclose(loaded_embeddings, original_embeddings, rtol=1e-5)
        print(f"\n11. Test Result:")
        if embeddings_match:
            print(f"    ✅ SUCCESS: Loaded embeddings match original template!")
        else:
            print(f"    ❌ FAILURE: Loaded embeddings don't match original!")
            print(f"    Max difference: {(loaded_embeddings - original_embeddings).abs().max():.6f}")

        print("\n" + "=" * 80)
        print("Test completed!")
        print("=" * 80)

        return embeddings_match


if __name__ == "__main__":
    success = test_template_embedding()
    exit(0 if success else 1)
