# CoSiR Evaluation Config Loading Debug Log

**Date:** 2025-09-30
**Issue:** `src/hook/eval_cosir.py` fails with "string indices must be integers" error when loading config

## Problem Description

When trying to initialize `CoSiREvaluator` with experiment path `"res/CoSiR_Experiment/cc3m/20250929_184240_CoSiR_Experiment"`, the code fails at line 110 in `load_feature_manager()`:

```python
feature_config = {"storage_dir": self.config["featuremanager"]["storage_dir"], ...}
```

Error: `TypeError: string indices must be integers`

## Root Cause Analysis

**Investigation Steps:**

1. **Examined eval_cosir.py code** - Found the error occurs when trying to access `self.config` as a dictionary
2. **Checked experiment config files** - Found both `config.json` and `experiment_metadata.json` exist
3. **Analyzed config content** - Discovered the config is stored as a string instead of a proper JSON object

**Root Cause:**
Both config files store the configuration as a string literal instead of a proper JSON object:
- `config.json` contains: `"{'data': {'dataset_type': 'conceptual3m', ...}"`
- `experiment_metadata.json` contains: `"config": "{'data': {'dataset_type': 'conceptual3m', ...}"`

When `json.load()` reads these files, it returns a string instead of a dictionary, causing the "string indices must be integers" error when trying to access `config["featuremanager"]`.

## Solution Implemented

**Modified `_load_config()` method in `/src/hook/eval_cosir.py`:**

```python
def _load_config(self) -> Dict[str, Any]:
    """Load experiment configuration"""
    import ast

    # Try config.json first (if manually saved)
    config_path = self.experiment_dir / "configs/config.json"
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        with open(config_path) as f:
            config = json.load(f)
            # If config is a string (stored incorrectly), parse it
            if isinstance(config, str):
                config = ast.literal_eval(config)
            return config

    # Try experiment_metadata.json (saved by ExperimentManager)
    metadata_path = self.experiment_dir / "experiment_metadata.json"
    if metadata_path.exists():
        print(f"Loading metadataconfig from: {metadata_path}")
        with open(metadata_path) as f:
            metadata = json.load(f)
            config = metadata.get("config", {})
            # If config is a string (stored incorrectly), parse it
            if isinstance(config, str):
                config = ast.literal_eval(config)
            return config

    raise FileNotFoundError(f"Config not found in {config_path} or {metadata_path}")
```

**Key Changes:**
- Added `import ast` for safe string evaluation
- Added string type checking with `isinstance(config, str)`
- Added `ast.literal_eval(config)` to safely parse string configs
- Applied fix to both config loading paths

## Testing and Verification

**Created test scripts:**
1. `debug_eval_config.py` - Diagnosed the original issue
2. `test_fixed_eval.py` - Verified the fix works
3. `test_cosir_eval.py` - Comprehensive component testing

**Test Results:**
- ✅ Config loading now works correctly
- ✅ Feature manager loads successfully
- ✅ Model loading works
- ✅ String config is properly parsed to dictionary
- ✅ All nested config access patterns work

## Files Modified

- `/src/hook/eval_cosir.py` - Fixed `_load_config()` method (lines 62-89)

## Files Created

- `/src/test/20250930_config_debug/debug_eval_config.py` - Debug script
- `/src/test/20250930_config_debug/test_fixed_eval.py` - Fix verification
- `/src/test/20250930_config_debug/test_cosir_eval.py` - Comprehensive test
- `/src/test/20250930_config_debug/20250930_config_debug_log.md` - This log

## Prevention

This issue occurred because the experiment saving mechanism is storing configs as string literals instead of proper JSON. Future experiments should ensure configs are saved as proper JSON objects, not string representations.