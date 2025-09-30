#!/usr/bin/env python3
"""
Debug script for CoSiR evaluation config loading issue

Issue: eval_cosir.py fails with "string indices must be integers" when loading config
from experiment_metadata.json because config is stored as string instead of dict.

Date: 2025-09-30
"""

import sys
sys.path.append('/project/CoSiR')

import json
import ast
from pathlib import Path
from src.hook.eval_cosir import CoSiREvaluator

def debug_config_loading():
    """Debug the config loading issue"""
    experiment_path = "res/CoSiR_Experiment/cc3m/20250929_184240_CoSiR_Experiment"

    print("=== Debugging Config Loading Issue ===")
    print(f"Experiment path: {experiment_path}")

    # Check what files exist
    exp_dir = Path(experiment_path)
    print(f"\nExperiment directory exists: {exp_dir.exists()}")

    # Check config files
    config_path = exp_dir / "configs/config.json"
    metadata_path = exp_dir / "experiment_metadata.json"

    print(f"Config.json exists: {config_path.exists()}")
    print(f"Metadata.json exists: {metadata_path.exists()}")

    # Load and inspect metadata
    if metadata_path.exists():
        print("\n=== Metadata Content ===")
        with open(metadata_path) as f:
            metadata = json.load(f)

        print(f"Metadata keys: {list(metadata.keys())}")
        print(f"Config type: {type(metadata.get('config'))}")
        print(f"Config content (first 200 chars): {str(metadata.get('config'))[:200]}...")

        # Try to parse the config string
        config_str = metadata.get('config')
        if isinstance(config_str, str):
            print("\n=== Attempting to Parse Config String ===")
            try:
                # Try using ast.literal_eval (safer than eval)
                config_dict = ast.literal_eval(config_str)
                print("✓ Successfully parsed config using ast.literal_eval")
                print(f"Parsed config type: {type(config_dict)}")
                print(f"Config keys: {list(config_dict.keys())}")

                # Test accessing featuremanager config
                if 'featuremanager' in config_dict:
                    print(f"Featuremanager keys: {list(config_dict['featuremanager'].keys())}")
                    print(f"Storage dir: {config_dict['featuremanager']['storage_dir']}")
                else:
                    print("❌ No 'featuremanager' key found")

            except Exception as e:
                print(f"❌ Failed to parse config string: {e}")

                # Try eval as last resort (not recommended for production)
                try:
                    config_dict = eval(config_str)
                    print("✓ Successfully parsed config using eval")
                    print(f"Parsed config type: {type(config_dict)}")
                except Exception as e2:
                    print(f"❌ Failed to parse with eval too: {e2}")

    # Test the actual evaluator initialization
    print("\n=== Testing CoSiREvaluator Initialization ===")
    try:
        evaluator = CoSiREvaluator(experiment_path)
        print("✓ CoSiREvaluator initialized successfully")
        print(f"Config type after loading: {type(evaluator.config)}")
        print(f"Config keys: {list(evaluator.config.keys()) if isinstance(evaluator.config, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ CoSiREvaluator initialization failed: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

def test_fixed_config_loading():
    """Test the fixed config loading"""
    experiment_path = "res/CoSiR_Experiment/cc3m/20250929_184240_CoSiR_Experiment"

    print("\n=== Testing Fixed Config Loading ===")

    # Simulate the fixed _load_config method
    exp_dir = Path(experiment_path)
    metadata_path = exp_dir / "experiment_metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        config = metadata.get("config", {})
        print(f"Raw config type: {type(config)}")

        # If config is a string, parse it
        if isinstance(config, str):
            try:
                import ast
                config = ast.literal_eval(config)
                print("✓ Successfully parsed string config")
                print(f"Parsed config type: {type(config)}")

                # Test accessing nested config
                feature_config = {
                    "storage_dir": config["featuremanager"]["storage_dir"],
                    "sample_ids_path": config["featuremanager"]["sample_ids_path"],
                    "primary_backend": config["featuremanager"]["primary_backend"],
                }
                print("✓ Successfully accessed featuremanager config")
                print(f"Feature config: {feature_config}")

            except Exception as e:
                print(f"❌ Failed to parse string config: {e}")

if __name__ == "__main__":
    debug_config_loading()
    test_fixed_config_loading()