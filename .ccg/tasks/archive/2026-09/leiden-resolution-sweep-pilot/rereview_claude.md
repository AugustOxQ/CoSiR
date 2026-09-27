The re-review found no remaining Critical issues. Both previous findings are fixed.

**Previous findings**
- **CPU vs CUDA (fixed):** `run_leiden_resolution_sweep_pilot.py:174` now picks CUDA when available, otherwise CPU. That is the same line B1 uses (`run_b1_redcaps_single_teacher_pilot.py:400`), and the device is passed on to `build_graphs` (line 191). The report now says "device=cuda; Leiden and validation metrics run on CPU", which matches the code: Leiden runs in igraph, and the k-NN transfer step uses scikit-learn with `n_jobs=1`.
- **Stale next diagnostic (fixed):** the script now counts connected components and isolated nodes (lines 201–203), and the verdict ends on the 1,397-component floor. The numbers agree with each other: at resolution 0.05 and below the partition has 1,397 communities, the same as the component count. That is expected, because every Leiden community is connected. The 1.8% figure is also correct (1 − 1397/1423).

**Critical:** none.

**Warning:** none.

**Info**
1. **Sanity-check wording is hardcoded** (lines 92–96). The "Passed: reproduced B1's 1423 / 1404 / 4.773×" text prints the fixed B1 constants, not the values the run measured. The check allows a 1% difference on counts and 5% on lift, so a run that differed slightly would still say "reproduced … at reported precision". It is accurate here, since the resolution-1 row shows exactly 1423 / 1404 / 4.773×.
2. **The verdict's closing sentence is not tied to the data** (lines 166–168). "The disconnected-component floor explains the persistent count" is printed even if the smallest partition size doesn't equal the component count. It does equal it here (1397 = 1397). Adding a check or an equality clause would keep a future rerun from overclaiming.
3. **I couldn't confirm from the files alone that the report was regenerated after the device fix.** It has the exact format of the current `write_report` (for example the `{component_count:,}` output), which suggests it was. Also, the graph matches B1's edge count and the resolution-1 partition exactly, so it is effectively the same graph whichever device built it. If you need certainty, compare the report's 14:27:36 timestamp with the script's last change or the run log.
