#!/usr/bin/env python3
"""
Debug script to reproduce the EXACT issue mentioned by the user
"""
import sys
import os

sys.path.append("/project/CoSiR/src")
import torch
import json
from src.utils.feature_manager import FeatureManager
from src.utils.embedding_manager import TrainableEmbeddingManager


def debug_actual_issue():
    print("=== Reproducing the actual bug scenario ===")

    # Let's check the actual data files to see what sample IDs exist
    feature_config = {
        "storage_dir": "/project/CoSiR/data/comprehensive_cosir_test/features",
        "sample_ids_path": "/project/CoSiR/data/comprehensive_cosir_test/sample_ids.pt",
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

    # Check if actual data exists
    if os.path.exists(feature_config["sample_ids_path"]):
        print(f"Loading actual sample_ids from: {feature_config['sample_ids_path']}")
        sample_ids_list = torch.load(feature_config["sample_ids_path"])
        print(f"Loaded sample_ids_list: {sample_ids_list[:10]}... (showing first 10)")
        print(f"Total samples: {len(sample_ids_list)}")

        # Check if feature storage directory exists
        if os.path.exists(feature_config["storage_dir"]):
            print(f"\nChecking feature storage at: {feature_config['storage_dir']}")

            # Load feature manager
            feature_manager = FeatureManager(
                feature_config["storage_dir"], config=feature_config
            )
            feature_manager.load_features()

            print(
                f"Feature manager index mapping keys: {list(feature_manager.index_mapping.keys())[:10]}..."
            )
            print(f"Total indexed samples: {len(feature_manager.index_mapping)}")

            # Check for mismatch
            stored_sample_ids = set(feature_manager.index_mapping.keys())
            list_sample_ids = set(sample_ids_list)

            print(f"\nSample ID analysis:")
            print(f"Sample IDs in list: {len(list_sample_ids)}")
            print(f"Sample IDs in storage: {len(stored_sample_ids)}")
            print(f"Missing from storage: {len(list_sample_ids - stored_sample_ids)}")
            print(f"Extra in storage: {len(stored_sample_ids - list_sample_ids)}")

            if list_sample_ids != stored_sample_ids:
                print("\n⚠️  MISMATCH DETECTED!")
                missing = list_sample_ids - stored_sample_ids
                extra = stored_sample_ids - list_sample_ids

                if missing:
                    print(
                        f"Missing from storage: {sorted(list(missing))[:10]}... (showing first 10)"
                    )
                if extra:
                    print(
                        f"Extra in storage: {sorted(list(extra))[:10]}... (showing first 10)"
                    )

            # Test the specific case that fails
            print(f"\n=== Testing embedding manager with these sample IDs ===")

            # This is the line from train_cosir.py that creates the embedding manager
            embedding_manager = TrainableEmbeddingManager(
                sample_ids=list(
                    range(100)
                ),  # Note: This is DIFFERENT from sample_ids_list!
                embedding_dim=64,
                storage_mode="memory",
                auto_sync=False,
                chunk_size=25,
            )

            print(
                f"Embedding manager sample_ids: {embedding_manager.sample_ids[:10]}..."
            )
            print(
                f"Embedding manager total samples: {len(embedding_manager.sample_ids)}"
            )

            # Try to get embeddings for batch_sample_ids from the actual sample_ids_list
            batch_sample_ids = sample_ids_list[:5]  # First 5 from feature manager

            print(f"\n=== Testing the failing line ===")
            print(f"batch_sample_ids: {batch_sample_ids}")

            try:
                label_embeddings = embedding_manager.get_embeddings(batch_sample_ids)
                print(f"✅ Success! Got embeddings shape: {label_embeddings.shape}")
            except Exception as e:
                print(f"❌ FAILED! Error: {e}")
                print(f"Error type: {type(e).__name__}")

                # This is the bug! The embedding manager was created with range(100)
                # but the actual sample IDs from the feature extraction can be different!
                print(f"\n🐛 BUG IDENTIFIED:")
                print(
                    f"   Embedding manager expects: range(100) = {list(range(100))[:10]}..."
                )
                print(f"   But actual sample IDs are: {batch_sample_ids}")
                print(f"   These don't match!")

        else:
            print(
                f"Feature storage directory not found: {feature_config['storage_dir']}"
            )
    else:
        print(f"Sample IDs file not found: {feature_config['sample_ids_path']}")
        print("This suggests features haven't been extracted yet.")


if __name__ == "__main__":
    debug_actual_issue()
