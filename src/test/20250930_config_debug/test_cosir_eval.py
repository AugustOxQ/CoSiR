#!/usr/bin/env python3
"""
Test script for CoSiR evaluation using existing experiment

This script tests the CoSiREvaluator with the specified experiment path
instead of creating artificial data.

Usage:
    python src/test/20250930_config_debug/test_cosir_eval.py

Date: 2025-09-30
"""

import sys
sys.path.append('/project/CoSiR')

import torch
from src.hook.eval_cosir import CoSiREvaluator

def test_evaluator_components():
    """Test loading all evaluator components"""
    experiment_path = "res/CoSiR_Experiment/cc3m/20250929_184240_CoSiR_Experiment"

    print("=== Testing CoSiR Evaluator Components ===")
    print(f"Experiment path: {experiment_path}")

    # Initialize evaluator
    print("\n1. Initializing evaluator...")
    evaluator = CoSiREvaluator(experiment_path)
    print(f"✓ Evaluator initialized successfully")

    # Test config loading
    print("\n2. Testing config...")
    print(f"Config type: {type(evaluator.config)}")
    print(f"Config keys: {list(evaluator.config.keys())}")

    # Test feature manager loading
    print("\n3. Testing feature manager loading...")
    try:
        feature_manager = evaluator.load_feature_manager()
        print("✓ Feature manager loaded successfully")
        print(f"Feature manager type: {type(feature_manager)}")
    except Exception as e:
        print(f"❌ Feature manager loading failed: {e}")
        return

    # Test model loading
    print("\n4. Testing model loading...")
    try:
        model = evaluator.load_model()
        print("✓ Model loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    # Test embedding manager loading
    print("\n5. Testing embedding manager loading...")
    try:
        embedding_manager = evaluator.load_embedding_manager()
        print("✓ Embedding manager loaded successfully")
        print(f"Embedding manager type: {type(embedding_manager)}")
    except Exception as e:
        print(f"❌ Embedding manager loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test getting sample embeddings
    print("\n6. Testing sample embedding retrieval...")
    try:
        # Get first few sample IDs
        sample_ids_path = evaluator.config["featuremanager"]["sample_ids_path"]
        sample_ids_list = torch.load(sample_ids_path, map_location="cpu")
        first_5_ids = sample_ids_list[:5]

        print(f"First 5 sample IDs: {first_5_ids}")

        # Get embeddings for first sample
        embedding = evaluator.get_sample_embedding(first_5_ids[0])
        print(f"✓ Sample embedding retrieved")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding device: {embedding.device}")

    except Exception as e:
        print(f"❌ Sample embedding retrieval failed: {e}")
        import traceback
        traceback.print_exc()

    # Test getting representatives
    print("\n7. Testing representative embeddings...")
    try:
        representatives = evaluator.get_representatives(num_representatives=10)
        print(f"✓ Representatives computed")
        print(f"Representatives shape: {representatives.shape}")

    except Exception as e:
        print(f"❌ Representatives computation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_evaluator_components()