# Condition Space Evaluator Test Suite

## Overview
This directory contains comprehensive tests for the `CoSiRAutomaticEvaluator` class, which provides automatic evaluation metrics for the CoSiR model's condition space.

## Files

### `test_condition_space_evaluator.py`
Comprehensive test suite with 14 test cases covering:
- Basic functionality
- Edge cases (small datasets, single condition, zero radius, collinear conditions)
- All 6 evaluation metrics
- Full evaluation pipeline
- Integration with real CoSiR model

### `20251111_debugging_log.md`
Detailed documentation of bugs found, investigation process, root causes, and solutions implemented.

## Running the Tests

```bash
# Activate the CoSiR environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR

# Run all tests
python src/test/20251111_condition_space_evaluator/test_condition_space_evaluator.py
```

## Test Coverage

### Basic Tests
- ✓ Basic initialization
- ✓ Re-initialization with new data

### Edge Case Tests
- ✓ Small dataset (5 images, 10 texts)
- ✓ Single condition
- ✓ Zero radius conditions (all at origin)
- ✓ Collinear conditions (1D structure)

### Metric Tests
1. ✓ Radius-Effect Correlation
2. ✓ Angular-Semantic Monotonicity
3. ✓ Conditional Retrieval Gain
4. ✓ Retrieval Diversity
5. ✓ Semantic Coherence
6. ✓ Condition Space Quality

### Integration Tests
- ✓ Full evaluation pipeline
- ✓ Integration with real CoSiR model and COCO data

## Expected Results

All 14 tests should pass:
```
======================================================================
Test Results: 14 passed, 0 failed out of 14 tests
======================================================================
```

## Key Improvements Made

1. **Input Validation**: All functions now validate inputs and clamp to valid ranges
2. **Zero Variance Protection**: Correlation functions handle zero variance gracefully
3. **Index Bounds Checking**: Caption indices are validated before access
4. **Empty Data Handling**: Functions return safe defaults when data is insufficient
5. **Improved Clustering**: More conservative requirements for silhouette score computation

## Mock Model

The test suite includes a `MockModel` class that simulates the CoSiR model's behavior:
- Simple combine function that adds perturbation based on condition magnitude
- Allows testing without requiring a fully trained model
- Runs quickly on CPU

## Usage Example

```python
from src.utils.condition_space_evaluator import CoSiRAutomaticEvaluator

# Create evaluator
evaluator = CoSiRAutomaticEvaluator(model, precomputed, conditions)

# Run all evaluations
results = evaluator.evaluate_all(save_path="results.json", verbose=True)

# Or run individual metrics
radius_effect = evaluator.compute_radius_effect_correlation(n_samples=100)
angular_mono = evaluator.compute_angular_semantic_monotonicity(n_angles=12)
```

## Debugging Tips

1. **Warning Messages**: Pay attention to warning messages - they indicate edge cases being handled
2. **Sample Sizes**: Use smaller sample sizes for testing, larger for production
3. **Device**: Tests run on CPU by default, but work with CUDA if available
4. **Data Requirements**: Most metrics need at least 10-20 samples for meaningful results

## Known Limitations

- Silhouette score requires at least 40 conditions for 2 clusters (20 samples per cluster)
- Angular metrics assume 2D condition space
- Diversity metrics need at least 2 conditions
- Correlation metrics need at least 2 samples and non-zero variance

## Future Improvements

- [ ] Add performance benchmarks
- [ ] Add visualization of evaluation results
- [ ] Add parametric tests for statistical significance
- [ ] Add tests for higher-dimensional condition spaces
- [ ] Add logging instead of print statements
- [ ] Add configurable behavior for edge cases
