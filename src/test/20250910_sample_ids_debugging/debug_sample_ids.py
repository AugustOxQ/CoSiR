#!/usr/bin/env python3
"""
Debug script to reproduce the sample ID mismatch issue
"""

# Reproduce the current sample_ids logic from FeatureExtractionDataset
def current_sample_ids_logic(num_annotations):
    """Current logic from line 30 of cosir_datamodule.py"""
    sample_ids = {i: idx for idx, i in enumerate(range(num_annotations))}
    return sample_ids

# Test with a small dataset
num_annotations = 10
print("=== Current Sample IDs Logic (BUGGY) ===")
sample_ids = current_sample_ids_logic(num_annotations)
print(f"sample_ids mapping: {sample_ids}")
print("What gets returned for each idx:")
for idx in range(min(5, num_annotations)):
    sample_id = sample_ids[idx]
    print(f"  idx={idx} -> sample_id={sample_id}")

print("\n=== Expected Sample IDs Logic (FIXED) ===")
print("sample_ids should be: {0: 0, 1: 1, 2: 2, 3: 3, ...}")
print("What should get returned for each idx:")
for idx in range(min(5, num_annotations)):
    sample_id = idx  # This is what we actually want
    print(f"  idx={idx} -> sample_id={sample_id}")

print("\n=== Analysis ===")
print("Current line 30: self.sample_ids = {i: idx for idx, i in enumerate(range(len(self.annotations)))}")
print("This creates: {0: 0, 1: 1, 2: 2, ...} which is correct")
print("But the variable naming is confusing - let's check what actually happens...")

# Let's trace through the enumerate logic
print("\nTracing enumerate(range(10)):")
for idx, i in enumerate(range(10)):
    print(f"  idx={idx}, i={i}")

print("\nSo {i: idx for idx, i in enumerate(range(10))} creates:")
sample_ids_traced = {i: idx for idx, i in enumerate(range(10))}
print(f"  {sample_ids_traced}")
print("This is actually correct! The issue might be elsewhere...")