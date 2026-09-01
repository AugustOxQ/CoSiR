# Experiment 15.1 review

## Scope

- Persisted edge provenance uses stable, classifier-key ordering, with a documented
  `uint8` code per `buddy_edges.npy` column.
- Both standalone and embedding-manager template paths save, copy, load, and expose
  the parallel provenance array. Loading an old template removes stale provenance.
- Standalone `distance_mode` defaults to `blend` and is forwarded unchanged.
- Family #1 reads only static CSR tensors; Family #2/#3 retain the refreshable CSR.

## Findings

No critical or warning findings after source review and focused CPU verification.

The required Claude wrapper review could not execute in this root-runner environment:
its CLI rejects `--dangerously-skip-permissions` when invoked as root. The Codex
review wrapper was invoked against the full diff; local verification below is the
authoritative recorded evidence.

## Verification

- `test_buddy_reg.py`: provenance count/alignment/remap and manager persistence.
- `test_buddy_refresh.py`: static Family #1 CSR survives refresh.
- `test_compute_buddy_init_distance_mode.py`: standalone config forwarding.
- Existing buddy graph, compute-buddies, and feature-override tests pass.
- `py_compile` and `git diff --check` pass.
