# Ticket 04: emotion graphs

`pf_graphs.py` implements the index's four interfaces. The emotion branch is a
documented copy of `rs_graphs._partition_aspect`, omitting style and genre;
normalization, sampled SD, mutual neighbors, reciprocal edges, pair ordering,
hashes, and atomic writers reuse frozen helpers. Every top-k receives the requested
k. Fit and validation graphs use independent node sets.

`build` takes dictionaries containing `keys`, `E`, `D`, `I` (extra fields such as
`T` are ignored). `load` returns a dictionary with `fit_pairs`, `dev_pairs`,
`manifest`, and `path`. The index did not prescribe that return type: this is an
interface inference. Manifest `counts` uses `fit` and `dev`; `pairs_sha256` hashes
the `emotion.npz` file bytes; key hashes use the frozen newline-joined convention.
Loads verify keys/order, settings, code, archive hash, array structure, and counts.

`gate_iv()` returns evidence with top-level `ok`, and writes it to
`PF_OUTPUTS/stage0a/graphs.json`. The runner must stop when `ok` is false. Each of
the three named r10 graphs includes counts and seconds per partition, dtype,
shape, exact equality and array-hash checks for all six frozen arrays, source
hash checks for E/D/I, reference archive checksum, and split identity checks.
`build_seconds` excludes input I/O; `input_seconds` is recorded separately (the
two full variants share their input load). The real gate has not been run.

The tests were developed in red/green slices: missing module, then missing
build/load, then missing gate, each followed by passing targeted tests. The
synthetic oracle invokes the frozen builder with all five spaces, k=3/5,
correction on/off, a permuted subset, and node counts below/above the 256-row
block size. k=3 exercises zero-degree lambda fallback. Archive round trips and
mutations cover key order, code/archive/count damage, and gate edge, dtype,
node-order, source, identity and checksum mismatches.

Unit command (20 passed, 0 failed, 1 slow test deselected):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261031_photo_finetune/test_pf_graphs.py -m 'not slow' --basetemp=.scratch/t04-pytest
```

Controller command (CPU, `slow` and `real_data`; may exceed four minutes):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261031_photo_finetune/test_pf_graphs.py::test_gate_iv_real_data --basetemp=.scratch/t04-controller
```

The test sets `PF_OUTPUTS` to its pytest `tmp_path`; the evidence remains under
`.scratch/t04-controller/test_gate_iv_real_data0/stage0a/graphs.json`.

Timing measurements and estimates are recorded below. No GPU, full graph replay,
or writes to frozen stores/outputs were performed. Scratch scripts, timing JSON,
and logs are under this worktree's ignored `.scratch/`; nothing needs deletion.

CPU timings on 2026-11-01, eight OMP/MKL/BLAS threads, float32 2048-dimensional
vectors, correction on, one measurement per size/k:

| k | 8,192 real nodes (s) | 16,384 real nodes (s) | Estimate: 70,000 fit nodes (s) |
|---|---:|---:|---:|
| 20 | 4.1254 | 15.0999 | 267.6 |
| 40 | 4.6726 | 16.1907 | 281.2 |
| 80 | 5.7067 | 19.0604 | 326.3 |

Inputs were loaded by the same CT providers used for full r10, checked against
r10's E/D/I source hashes, then restricted to the first N entries of an rng0
permutation of its fit nodes. Loading/checking/selecting took 57.59s and 55.63s;
that time is excluded from the graph timings. Process peak RSS, including full
provider loading, was 5.239 GiB and 5.223 GiB. Both commands completed in under
two minutes with a 210-second timeout.

For each k, fit `T(n) = a + b*(n/8192)^2` to the two measured times. At k=40,
`a=0.8333`, `b=3.8393` seconds. This estimates 109.5s for the new full r10 k40
emotion build (42,969 fit plus 6,152 dev nodes, summing both partition times).
Its actual timing is deliberately pending the controller's slow gate:
`graphs['full/aspect__k40']['build_seconds']` in the evidence JSON. For context,
r10's archived full corrected k40 build took 151.9428s **for all three aspects**;
that is not a measurement of this new emotion-only builder.

The 70,000-node estimates are for fit only; the same model adds about 1.45/1.87/
2.45s for 4,251 validation nodes at k=20/40/80. They exclude input I/O, assume
unchanged CPU conditions, and extrapolate reciprocal-set and top-k costs together
with dense block multiplication; node geometry, memory bandwidth, and machine
load can change the result. They are estimates, not full-data measurements.

Memory uses three normalized spaces, 256-by-n scores, n-by-k neighbor arrays,
and sparse Python edge sets. At 70,000 nodes/k80 the approximate allocation
budget is 2.14 GiB for the caller's four photo spaces, 1.60 GiB for normalized
E/D/I, about 1 GiB for simultaneous score/top-k buffers, and a few GiB for edge
sets/allocator overhead: below 16 GiB by allocation estimate. Peak RSS at that
size has not been measured. No n-by-n score matrix is constructed at that size.

All requested code and tests are implemented. Exact real-graph equality, actual
full k40 timing, and full-size RSS remain controller validation, as required by
the prohibition on running the slow gate here. No scientific rule was changed.
