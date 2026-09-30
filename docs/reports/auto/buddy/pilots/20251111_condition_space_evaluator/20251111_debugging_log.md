# Condition Space Evaluator Debugging Log
**Date:** 2025-11-11
**Component:** `src/utils/condition_space_evaluator.py`

## Problem Description
The `CoSiRAutomaticEvaluator` class had several bugs and edge cases that were not properly handled:

1. **Missing validation** for edge cases (small datasets, zero variance, etc.)
2. **Division by zero** risks in correlation calculations
3. **Index out of bounds** errors when accessing ground truth captions
4. **Empty data handling** insufficient for degenerate cases
5. **Test function misplaced** inside the class definition

## Investigation Steps

### 1. Code Review
Analyzed the entire `CoSiRAutomaticEvaluator` class and identified potential failure points:
- Line 96-97: Pearson correlation without variance check
- Line 173: Spearman correlation without data length check
- Line 212: Ground truth caption indices not validated
- Line 484-486: Silhouette score without proper sample count validation
- Line 736-793: Test function incorrectly placed inside class

### 2. Edge Cases Identified
- **Single condition**: Only 1 condition vector (can't cluster or compute diversity)
- **Zero radius conditions**: All conditions at origin (zero variance)
- **Collinear conditions**: All conditions along a line (1D structure in 2D space)
- **Small datasets**: < 10 samples for various metrics
- **Missing ground truth**: Invalid or out-of-bounds caption indices
- **Empty ranks**: No valid retrieval results

### 3. Test Coverage
Created comprehensive test suite with 14 test cases:
- Basic initialization test
- Edge case: small dataset (5 images, 10 texts)
- Edge case: single condition
- Edge case: zero radius conditions
- Edge case: collinear conditions
- Re-initialization test
- Individual metric tests (6 metrics)
- Full evaluation pipeline test
- Integration test with real CoSiR model

## Root Cause
The evaluator was designed for typical use cases with sufficient data but lacked defensive programming for edge cases. Key issues:

1. **No input validation**: Functions didn't check if inputs were within valid ranges
2. **Assumptions about data**: Assumed all indices valid, sufficient variance, etc.
3. **No graceful degradation**: Failed instead of returning sensible defaults
4. **Test code location**: Test function was inside the class instead of external

## Solution Implemented

### 1. Input Validation (Lines 68-82, 144-146, 226-228, 345-348, 441-443)
```python
# Example from compute_radius_effect_correlation
n_samples = min(n_samples, len(self.conditions))
n_texts_sample = min(n_texts_sample, self.n_texts)

if n_samples < 2:
    print("  ⚠ Warning: Need at least 2 samples for correlation")
    return {
        "correlation": 0.0,
        "p_value": 1.0,
        # ... safe defaults
    }
```

### 2. Zero Variance Protection (Lines 113-117)
```python
# Check for zero variance to avoid division by zero
if radii.std() < 1e-8 or effects.std() < 1e-8:
    correlation, p_value = 0.0, 1.0
else:
    correlation, p_value = pearsonr(radii, effects)
```

### 3. Index Bounds Checking (Lines 245-249, 258-263, 284-289)
```python
# Validate ground truth indices
gt_cap_indices = self.img_to_cap_map.get(img_idx, [])

if not gt_cap_indices:
    continue

for gt_idx in gt_cap_indices:
    if gt_idx >= self.n_texts:
        continue
    # ... safe to use gt_idx
```

### 4. Empty Data Handling (Lines 295-314)
```python
if not all_ranks_baseline or not all_ranks_conditional:
    print("  ⚠ Warning: No valid retrieval results found")
    return {
        "R@1_baseline": 0.0,
        # ... all metrics set to 0
    }
```

### 5. Improved Silhouette Score Computation (Lines 540-556)
```python
n_clusters = min(8, max(2, len(conditions) // 20))  # More conservative
if n_clusters >= 2 and len(conditions) >= 2 * n_clusters:
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(conditions)
        if len(np.unique(labels)) >= 2:
            silhouette = silhouette_score(conditions, labels)
        else:
            silhouette = 0.0
    except Exception as e:
        print(f"  ⚠ Warning: Silhouette score computation failed: {e}")
        silhouette = 0.0
else:
    silhouette = 0.0
    n_clusters = 0
```

### 6. Test Function Removal
Removed the test function from the class (lines 736-796) and created a proper external test suite.

## Test Results
All 14 tests passed successfully:
- ✓ Basic initialization
- ✓ Edge case: small dataset
- ✓ Edge case: single condition
- ✓ Edge case: zero radius conditions
- ✓ Edge case: collinear conditions
- ✓ Re-initialization
- ✓ Radius-effect correlation
- ✓ Angular-semantic monotonicity
- ✓ Conditional retrieval gain
- ✓ Retrieval diversity
- ✓ Semantic coherence
- ✓ Condition space quality
- ✓ Full evaluation pipeline
- ✓ Integration test with real model

## Files Modified
1. `/project/CoSiR/src/utils/condition_space_evaluator.py` - Fixed all bugs and added validation
2. `/project/CoSiR/src/test/20251111_condition_space_evaluator/test_condition_space_evaluator.py` - Created comprehensive test suite

## Impact Assessment
- **Robustness**: Significantly improved handling of edge cases
- **Reliability**: No crashes on degenerate inputs
- **Usability**: Clear warning messages for invalid scenarios
- **Testing**: Comprehensive test coverage for confidence in changes

## Recommendations
1. Always run the test suite before using the evaluator in production
2. Monitor warning messages during evaluation
3. Consider adding more edge case tests as new scenarios are discovered
4. Add type hints and additional documentation for clarity

## Next Steps
- Consider adding logging instead of print statements
- Add configurable behavior for edge cases (fail vs. warn vs. skip)
- Create visualization tools for evaluation results
- Add unit tests for individual helper functions
