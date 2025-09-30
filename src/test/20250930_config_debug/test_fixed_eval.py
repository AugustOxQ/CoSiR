#!/usr/bin/env python3
"""
Test script to verify the fixed CoSiR evaluation config loading

Date: 2025-09-30
"""

import sys
sys.path.append('/project/CoSiR')

from src.hook.eval_cosir import CoSiREvaluator

def test_evaluator_loading():
    """Test that CoSiREvaluator can now load config correctly"""
    experiment_path = "res/CoSiR_Experiment/cc3m/20250929_184240_CoSiR_Experiment"

    print("=== Testing Fixed CoSiREvaluator ===")
    print(f"Experiment path: {experiment_path}")

    try:
        # Initialize evaluator
        evaluator = CoSiREvaluator(experiment_path)
        print("✓ CoSiREvaluator initialized successfully")

        # Check config is now a dict
        print(f"Config type: {type(evaluator.config)}")
        if isinstance(evaluator.config, dict):
            print(f"✓ Config is properly parsed as dictionary")
            print(f"Config keys: {list(evaluator.config.keys())}")

            # Test accessing featuremanager config (the specific error case)
            feature_config = {
                "storage_dir": evaluator.config["featuremanager"]["storage_dir"],
                "sample_ids_path": evaluator.config["featuremanager"]["sample_ids_path"],
                "primary_backend": evaluator.config["featuremanager"]["primary_backend"],
            }
            print("✓ Successfully accessed featuremanager config")
            print(f"Feature config preview: {feature_config}")

        else:
            print(f"❌ Config is still not a dict: {type(evaluator.config)}")

        # Test loading feature manager (the original failing operation)
        print("\n=== Testing Feature Manager Loading ===")
        try:
            feature_manager = evaluator.load_feature_manager()
            print("✓ Feature manager loaded successfully")
        except Exception as e:
            print(f"❌ Feature manager loading failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ CoSiREvaluator initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluator_loading()