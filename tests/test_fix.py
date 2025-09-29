#!/usr/bin/env python3
"""
Test script to verify the bug fix works
"""
import sys
import os
sys.path.append('/project/CoSiR/src')
import torch
from utils.feature_manager import FeatureManager
from utils.embedding_manager import TrainableEmbeddingManager

def test_fix():
    print("=== Testing the bug fix ===")
    
    # Load the actual sample IDs that were extracted
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
    
    if os.path.exists(feature_config["sample_ids_path"]):
        # Load sample IDs (same as in train_cosir.py line 124)
        sample_ids_list = torch.load(feature_config["sample_ids_path"])
        print(f"Loaded sample_ids_list with {len(sample_ids_list)} samples")
        print(f"First 10 sample IDs: {sample_ids_list[:10]}")
        
        # Create embedding manager with FIXED code (using actual sample_ids_list)
        print("\n=== Creating embedding manager with FIXED code ===")
        embedding_manager = TrainableEmbeddingManager(
            sample_ids=sample_ids_list,  # ✅ FIXED: Use actual sample IDs
            embedding_dim=64,
            storage_mode="memory",
            auto_sync=False,
            chunk_size=25,
        )
        
        print(f"Embedding manager created for {len(embedding_manager.sample_ids)} samples")
        print(f"First 10 embedding sample IDs: {embedding_manager.sample_ids[:10]}")
        
        # Test the line that was failing
        batch_sample_ids = sample_ids_list[:5]
        print(f"\n=== Testing the previously failing line ===")
        print(f"batch_sample_ids: {batch_sample_ids}")
        
        try:
            label_embeddings = embedding_manager.get_embeddings(batch_sample_ids)
            print(f"✅ SUCCESS! Got embeddings shape: {label_embeddings.shape}")
            print("🎉 Bug is FIXED!")
            
            # Test with more sample IDs to be sure
            larger_batch = sample_ids_list[:20] if len(sample_ids_list) >= 20 else sample_ids_list
            larger_embeddings = embedding_manager.get_embeddings(larger_batch)
            print(f"✅ Also tested with {len(larger_batch)} samples: {larger_embeddings.shape}")
            
        except Exception as e:
            print(f"❌ Still failing: {e}")
            return False
            
    else:
        print("Sample IDs file not found - need to run feature extraction first")
        return False
    
    return True

if __name__ == "__main__":
    success = test_fix()
    if success:
        print("\n🎯 BUG FIX VERIFIED SUCCESSFUL!")
    else:
        print("\n❌ Bug fix failed")