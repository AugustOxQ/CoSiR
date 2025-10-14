"""
Diagnostic script to check embedding initialization status
Created: 2025-01-13
"""

import torch
from pathlib import Path
import sys


def diagnose_embeddings(embeddings_dir):
    """Diagnose embedding files to check if they contain non-zero values"""

    embeddings_path = Path(embeddings_dir)

    print("=" * 80)
    print(f"DIAGNOSING EMBEDDINGS IN: {embeddings_path}")
    print("=" * 80)

    if not embeddings_path.exists():
        print(f"❌ Directory does not exist: {embeddings_path}")
        return False

    # Find all embedding chunks
    chunk_files = sorted(embeddings_path.glob("embeddings_chunk_*.pt"))

    if not chunk_files:
        print(f"❌ No embedding chunk files found in {embeddings_path}")
        return False

    print(f"\n✅ Found {len(chunk_files)} embedding chunk files")

    # Check each chunk
    all_zeros = True
    all_nonzero = True

    for i, chunk_file in enumerate(chunk_files):
        print(f"\n--- Chunk {i}: {chunk_file.name} ---")

        try:
            chunk_data = torch.load(chunk_file, map_location="cpu")

            if not isinstance(chunk_data, dict):
                print(f"  ⚠️  WARNING: Chunk is not a dict, it's {type(chunk_data)}")
                continue

            print(f"  Number of samples in chunk: {len(chunk_data)}")

            # Check first few samples
            sample_ids = list(chunk_data.keys())[:5]
            print(f"  First 5 sample IDs: {sample_ids}")

            for sid in sample_ids:
                embedding = chunk_data[sid]
                is_zero = torch.allclose(embedding, torch.zeros_like(embedding))

                print(f"    Sample {sid}: shape={embedding.shape}, is_zero={is_zero}")
                print(f"                   values={embedding}")

                if not is_zero:
                    all_zeros = False
                else:
                    all_nonzero = False

        except Exception as e:
            print(f"  ❌ Error loading chunk: {e}")
            continue

    print("\n" + "=" * 80)
    print("DIAGNOSIS RESULT:")
    print("=" * 80)

    if all_zeros:
        print("❌ ALL EMBEDDINGS ARE ZEROS!")
        print("   This means initialization did not work properly.")
        print("   Possible causes:")
        print("   1. initialize_embeddings_imgtxt() was not called")
        print("   2. Sync was not performed after initialization")
        print("   3. Loading from a template that contains zeros")
        return False
    elif all_nonzero:
        print("✅ ALL EMBEDDINGS ARE NON-ZERO!")
        print("   Initialization worked correctly.")
        return True
    else:
        print("⚠️  MIXED: Some embeddings are zero, some are not")
        print("   This indicates partial initialization.")
        return False


def check_template_embeddings(base_dir):
    """Check if template embeddings exist and their status"""

    base_path = Path(base_dir)
    template_dir = base_path / "template_embeddings"

    print("\n" + "=" * 80)
    print("CHECKING TEMPLATE EMBEDDINGS")
    print("=" * 80)

    if not template_dir.exists():
        print(f"✅ No template directory found at: {template_dir}")
        print("   This means training will initialize from scratch")
        return None

    print(f"⚠️  Template directory EXISTS at: {template_dir}")
    print("   Training may load from this template!")

    return diagnose_embeddings(template_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        embeddings_dir = sys.argv[1]
    else:
        # Default path - adjust this to your experiment path
        print("Usage: python diagnose_embeddings.py <embeddings_dir>")
        print("\nExample paths to check:")
        print("  Training embeddings: res/CoSiR_Experiment/dataset_name/exp_id/training_embeddings")
        print("  Template embeddings: res/CoSiR_Experiment/dataset_name/template_embeddings")
        sys.exit(1)

    # Diagnose training embeddings
    diagnose_embeddings(embeddings_dir)

    # Also check if there's a template
    try:
        base_dir = Path(embeddings_dir).parent.parent
        check_template_embeddings(base_dir)
    except:
        pass
