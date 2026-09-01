# Requirements

- Clamp weighted CSR samples to the active anchor's own row without altering the uniform branch.
- Cover a float32 rounded-to-row-end boundary and more than 10,000 weighted draws across rows.
- Verify the exact two-epoch RedCaps-150k typed buddy-training smoke path on GPU.
- Confirm the shared helper protects both contrastive and smoothness callers, then commit the verified fix.
