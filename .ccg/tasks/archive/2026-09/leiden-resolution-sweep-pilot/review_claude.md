No Critical findings, and nothing blocks the conclusion. At resolution 1.0 the script reproduces B1's graph-only numbers exactly: 1,148,216 edges, 1423 communities, 1404 below 1%, and 4.773× community lift. The sweep also stays within the brief: `detect_communities` and the shared files are not edited, it uses B1's saved split, and it reuses B1's own `assign_to_train_communities`, `community_pairs`, `lift_result` and `occupancy` functions. The issues are about provenance and how the report reads its own results.

## Critical
None.

## Warning

**W1. The checked-in script is not what produced the report.**
- `run_leiden_resolution_sweep_pilot.py:173` hard-codes `device = torch.device("cpu")`, but the report header says `device=cuda`. The script's docstring (line 8) also still says "Local GPU graph rebuild to match B1".
- So the script was changed after the run, and nobody has run the CPU version.
- The edge-count check at line 193 would stop the script if a CPU rebuild of the graph came out different, so this won't give wrong numbers silently. But the report can't be reproduced as written.
- Fix: either restore B1's `cuda if available else cpu` choice and re-run, or note in the report that it came from the earlier GPU version.

**W2. The report's verdict disagrees with the updated diagnosis doc.**
- The report ends with "next diagnostic step is the graph's density and degree distribution".
- `docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md:63-80` has since established the actual cause. The raw train graph has 1,397 connected components (1,354 isolated nodes, about 40 fragments of 2–3 nodes, and one giant component). Leiden can't merge across components, so 1,397 is a hard floor.
- The sweep's own numbers show this: the count stops at exactly 1397 from resolution 0.05 down.
- Meanwhile the sweep report still recommends a next step the project has already moved past. It also uses the hard-coded constant `B1_RAW_ISOLATES` (line 46) instead of counting components itself.
- Fix: compute `connected_components` in the script, which is cheap, and rewrite the verdict to name the component floor as the cause.

**W3. The pass/fail rule could never pass, and the report hides the more useful reading.**
- The "healthier" test (lines 131-133) uses the below-1% share across all communities, singletons included. With about 1,396 components sitting at the floor, no resolution can pass. The "no" verdict was decided by the metric before the sweep ran.
- The more useful reading: at resolution 1.0 the giant component splits into 1423 − 1396 = **27 communities**, and transfer coverage is exactly 27/1423. So every real community received validation points.
- The effective partition at resolution 1.0 is therefore already about 27-way, close to the ArtELingo scale (~19) the brief mentions. The "~99% occupancy-collapsed" framing is mostly a count of disconnected singletons.
- Fix: add a row set or column that excludes components of 3 nodes or fewer, or reports communities per component.

## Info

- **I1.** The drop in lift at lower resolutions is expected, not a finding. Coverage falls 27 → 14 → 7 → 2 → 1. At coverage 1 the "community" is the whole validation set, sampled down to 2,000 points, so 0.971× is just the null level of about 1.0. The report should say this rather than describe lift as "degrading".
- **I2.** Min and median occupancy are always 1/1.0 because singletons dominate them, so they tell us nothing here. The `subsampled` count is stored but not printed; B1's report does print it.
- **I3.** The sanity text says "within the 1% gate". The match was actually exact on all three numbers, which is worth saying because it is strong evidence the graph matches B1's. `nnz // 2` against B1's `triu(k=1)` edge count is safe because the check passed, which rules out self-loops.
- **I4.** A note for the follow-up rather than this review: `run_b1_repaired_graph_pilot_report.md` shows 71 communities with median occupancy 2.0 after repair. `ensure_min_degree` only fixes nodes with no edges, so the roughly 40 small multi-node fragments very likely remain as their own components. "Repair fixes occupancy" is therefore only partly true, and those fragments should be connected as well.
- **I5.** The brief's checklist is met: the script ran to completion, the resolution-1.0 check passed, and the report has the table and a verdict. No shared files were changed.

**Bottom line:** the sweep is correct and it does refute the resolution hypothesis. Before this report is cited, fix the device mismatch (W1) and rewrite the verdict around the connected-component floor and the effective 27-way partition (W2, W3).
