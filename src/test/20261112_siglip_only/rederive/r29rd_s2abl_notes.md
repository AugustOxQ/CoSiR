# Independent SC7 ablation re-derivation

Files: `r29rd_s2abl.py`, `r29rd_run_s2abl.sh`, `test_r29rd_s2abl.py`.
All commands run from `/project/CoSiR/src/test/20261112_siglip_only/rederive/`.
The runner sets OMP/MKL/OpenBLAS to eight threads and disables bytecode writes.
No deletion, commit, GPU execution, or r29 held-model scoring during development.

## Protocol

Point AMI and mAP follow the r28 ablation docstring. The exact cosine product detail comes from the permitted rt_metrics module docstring: normalise in float32, multiply in float64, stable descending ranking. AMI does not renormalise stored exports, fits every row, then masks unknown labels.

Each paired interval samples rows/queries first and shared training seeds second using fresh default_rng(0), 2000 AMI / 10000 mAP draws. This follows the task's explicit pairing. r17's generic two-seeded-arm comparison uses independent seed draws instead; its shared-seed retrieval form is the appropriate analogue here. Both readings are documented in output notes; the requested shared-seed reading is implemented.

No read-out pick, LEVEL unit or outcome is assigned. A removal needs all five seeds of all requested aspects present. Held has an explicit `--picks-committed` controller assertion and requires a complete passing frozen point replay for the requested CPU/CUDA backend.

## Commands

Fixtures (less than one minute):

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python -B -m unittest test_r29rd_s2abl
```

Frozen full replay (controller queue if longer than four minutes):

```bash
bash r29rd_run_s2abl.sh replay --device cpu --output r29rd_s2abl_replay.json > r29rd_s2abl_replay_full.log 2>&1
```

This compares all four removals and K0, all three aspects and five seeds, to r28's frozen per-seed values, seed means and deltas. Default interval sanity checks cover instructed_graphs. Other rows' replay intervals can be requested with `--interval-rows instructed_graphs content_correction buddy relational`.

Bounded point-replay work (rerun unchanged to resume caches):

```bash
timeout 225s bash r29rd_run_s2abl.sh replay --points-only --device cpu --time-budget 190 --output r29rd_s2abl_replay.json > r29rd_s2abl_replay_points.log 2>&1
```

Held, controller only after picks are committed and named exports are present:

```bash
bash r29rd_run_s2abl.sh held --rows instructed_graphs content_correction buddy relational --picks-committed --device cpu --replay-record r29rd_s2abl_replay.json --output s2abl_held.json > r29rd_s2abl_held.log 2>&1
```

Name only present rows to produce a partial-row held report. CUDA uses the same commands with `--device cuda`, but first run and pass the frozen replay on CUDA; the held guard requires matching backend. This agent has not run CUDA. AMI and bootstrap remain CPU work with either backend.

## C1 consistency limitation

`c1_rederived.json` contains three AMI seed means but no per-seed AMI values. Its independently saved partition-cache directory is also absent. The controller's held `kstar_check` records the new per-seed values and compares the three means directly; it explicitly marks per-seed target checks incomplete unless the old C1 partition caches are restored. No values are reconstructed from a mean and presented as recorded per-seed targets.

## Provenance and caching

Every JSON report includes `notes`, `replays`, and `inputs` with SHA256 hashes. Seed caches bind source bytes, export/sidecar bytes, label/order identities, batch size and device; cached data have their own checksum. No older scorer module is imported. The two forbidden scorer module files r28_ablation/r20_ret and rt_metrics were inspected through docstrings only; reports hash their files for provenance without executing them. Historical r28 values are only used after scoring, for comparison.

No files have been deleted. Obsolete smoke caches, if any, may be listed for optional future cleanup; retaining them is harmless.

## Verified bounded replay results

17 fixture tests passed in 2.144 seconds. The eight-cell frozen subset has 48 comparison records: AMI maximum absolute difference 0.0, mAP 1.1102230246251565e-16; all pass 1e-9. Coverage: instructed_graphs, all aspects; content_correction, emotion only, all five seeds. The other 16 cells remain for the full controller replay.

The entire instructed_graphs row has 2000-draw AMI and 10000-draw mAP intervals in r29rd_s2abl_replay_intervals.json. All six contain d. Total warm-cache runtime: 85.87696287501603 seconds. Full-precision endpoints and provenance are also collected in r29rd_s2abl_checks.json.

| Command | CPU estimate | GPU estimate (unmeasured) |
|---|---|---|
| Fixture tests | 2 seconds | 2 seconds; CPU fixtures |
| Full replay, cold cache | 15–25 minutes | 13–25 minutes |
| Full replay, present CPU caches | 8–12 minutes | no CUDA cache yet |
| Full held, all four rows | 18–30 minutes | 16–30 minutes |

GPU estimates include CPU clustering and bootstrap; the cosine products must remain float64. All recorded execution used CPU.

Optional obsolete smoke-cache paths are listed in r29rd_s2abl_optional_cleanup.txt; nothing was removed.
