# 20261029 retrieval run log

- rt_metrics speed, 12,281 x 2,048 random float32 rows, 8 workers, before the fix (float32 products): retrieval qe_k 0 2.9 s, qe_k 5 5.0 s, graded 4.3 s; largest worker 318 MB peak.
- Same, after the fix (float64 similarities, row-wise sums for the nDCG discounts): qe_k 0 3.1 s, qe_k 5 5.7 s, graded 4.2 s; largest worker 437 MB peak (limit 2 GB, no block size change).

## Amendment 1 (PercepT validation source)

The saved full-scale PercepT validation outputs used different mapper weights from the sample-scale held P3 comparator (13,400 fit paintings). Validation probabilities are now exported by rt_percept_val.py using the same per-seed sample-scale mapper as held, with matching weight hashes checked on export and load.

## C13 stage 4 numeric alignment (r11 fix-c13)

**Numeric convention.** Stored encoder/probability rows are converted to float32 and L2-normalized once in float32; plain Qwen's frozen loader already does this, and its raw bytes remain the stage 0b hash input. CRL uses those CLIP image rows and the saved unit float32 text rows, promotes both to float64, computes the cosine product and L2-normalizes the resulting vector in float64, then casts once to float32. Fit means, centered rows, covariance `(X-mu).T @ (X-mu)/(n-1)`, eigenvalues/eigenvectors and epsilon are float64 (no eigenvalue clipping). Every W, including none, operates in float64: whitening projects centered rows first, then divides by `sqrt(lambda + 1e-6*lambda_1)`; L2 normalization is float64, with the single cast to float32 only after normalization. Scoring promotes these stored unit rows to float64. QE uses the query plus the float64 sum of its top-k gallery neighbours (the mean's common factor cancels on normalization), then float64 L2 normalization, with no downcast; both initial and expanded similarities are float64 matrix products. Norm denominators are floored at 1e-12. Rankings use a stable descending sort in gallery row order with self excluded.

### Diagnosis captured before edits

All scratch artifacts below are under `/tmp/claude-0/-project-CoSiR/585c1724-5f23-49f9-94a9-046b675ae2e9/scratchpad/fix_c13/`; the supplied `queue_smoke/` artifacts were read only. `probe.py` / `before_threads.log` reproduce main loading/fitting with eight BLAS threads and the independent worker's one thread. The command exited 1 on the unchanged 1e-6 mAP assertion. `isolate.py` / `isolate.log` change just the product shape/thread count or eigendecomposition thread count. `before.log` additionally compares both fits with one thread. Four cheap convention tests failed before the implementation change (`red_tests.log`, 3.34 s).

1. **Truncated null space (hypothesis 1 confirmed).** The c* emotion seed-0 fit half has 300 rows and 299 eigenvalues above `1e-10 * lambda_1`; eigenvalues 299, 300, 1024 are `6.01553e-7`, `2.12191e-17`, `1.00520e-19`. With identical input rows, eight-thread versus one-thread fits give the same whiten64 mAP (0.235525029026) and whiten256 mAP (0.203732187806), but whiten1024 similarities differ by 0.116061126 and mAP by 2.06334e-5 (0.205061806601 / 0.205041173223). This selects 725 directions from an arbitrary 1,749-dimensional null space; epsilon regularizes their scale, but cannot make a truncated null-space basis unique. Full-width whitening retains the whole null space and is invariant to its rotation before float32 rounding. For plain/emotion, the combined loader/fit differences give a whiten1024 similarity delta 0.169512 and 2,986,662 pair-order reversals in one half. A smoke intended to check whiten1024 needs **at least 2,050 rows, 1,025 per parity half**, and enough actual input rank; no menu, epsilon, comparison or tolerance was changed.

2. **CRL's exact difference is float32 GEMM rounding, not different texts or encoders.** The main loader constructs all 6,152 validation cosine rows before smoke slicing; the independent loader constructs 600. At eight threads, these products differ in 3,518 of 30,000 entries, max 1.0430813e-7. At fixed 600 rows, eight versus one thread differs in 3,488 entries by the same maximum. At one thread, the full and sliced products match exactly. Text rows are the same saved float32 values (max norm error 1.19209e-7), with no extra text normalization in either loader. The loaded normalized CRL vectors differ by up to 7.45058e-8 in the mixed-thread probe. The supplied smoke's CRL `none+k0` mAP delta reaches 4.50571e-6; `none+k1` reaches 2.71636e-4. Computing and normalizing cosine vectors in float64 before one downcast removes dependence on float32 GEMM tiling.

3. **Input and W normalization, plus whitening operation order.** Main plain Qwen `raw` is already normalized by the frozen CT loader, then `embed` normalized it again; the independent loader did it once. On 600 emotion rows, 153,327 elements differ, max 1.49012e-8. For W=none, main normalized in float64 and cast to float32; the independent code returned its float32 input without that step. Centering already used float64 mean/subtraction/normalization followed by a float32 cast in both paths. Whitening differed in float64 association: main pre-scaled eigenvectors before projection; the independent path projected first and scaled afterward, matching the displayed rule. The independent eigendecomposition also clipped negative numerical eigenvalues; main used the actual eigenvalues. The convention removes that clipping. Both QE paths already used the same float64 sum (`query + sum(neighbours)`), norm and similarity product; no dtype or operation-order difference was found inside QE or similarity itself. Differences in its input rankings propagate through neighbour selection. The supplied smoke's instructed `none+k3` mAP delta 2.71387e-6 becomes a score delta 1.95481e-4; plain `whiten64+k0` mAP delta 1.60189e-6 becomes 2.22296e-4. CLIP's R0 baseline delta 2.01322e-7 also shifts scores whose transformed mAPs agree (whiten64+k0 score delta 2.71639e-5).

Per-step example, **instructed / emotion / even score half / 600 rows total**, before edits. A rank swap below counts one reversed unordered pair of gallery paintings across all queries; self is excluded.

| Step | Max abs vector delta | Max abs similarity delta | Rank swaps | Queries with swaps |
|---|---:|---:|---:|---:|
| Loaded input | 0 | 0 | 0 | 0 |
| W=none normalization | 1.4901161e-8 | 2.0502199e-7 | 1 | 1 |
| QE k=3 normalization and scoring | 6.1699430e-9 | 1.0164542e-7 | 2 | 2 |

The resulting half mAPs were main 0.26598525895513725 and independent 0.2659906866853112, delta 5.42773e-6. After the fix, `feedback.py` / `feedback.log` replays these captured inputs: both W and QE vector/similarity deltas and changed rank positions are zero; both half mAPs are 0.26598525895513725. The independent module still imports no `rt_` module and uses its own loader, fit, apply, ranking and metric implementations; only the written convention is shared.

### Validation and reproduction

All commands used `/root/miniconda3/envs/CoSiR/bin/python`, `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, CPU only. The five explicitly targeted files (`test_rt_metrics.py`, `test_rt_readout.py`, `test_rt_data.py`, `test_rt_crl.py`, `test_rt_vh.py`) passed together: **173 passed in 33.12 s**, `tests_final.log`. The final stricter stage 5 convention tests additionally passed with the rest of `test_rt_vh.py`: **7 passed in 1.34 s**, `vh_tests_final.log`. The cheap added tests cover single input normalization, CRL precision through normalization, whitening association and the cast boundary, and W=none normalization in stage 4 and stage 5 (including R0 and none+k3).

`agreement.py ARM [--seed SEED]` loads the main validation rows in full before slicing and independently loads 2,050 rows through `rt_vh.Data`. It uses the real main `fit`, `apply`, `retrieval` and the independent `fit_stats`, `apply_w`, `ap_core`, both parity halves, all three aspects, and every seed 0..4 for c* and PercepT. Main fits/transforms use eight BLAS threads and independent fits/transforms use one, matching their production execution. Queries and galleries use only labelled rows, while fits include every row of their parity half. R0, center+k5 and whiten64+k3 use 600 rows; whiten1024+k0 uses 2,050 rows (sizes capped at arm width by the unchanged menu). `baseline2050.py` also independently checks R0 at 2,050 rows, enabling a score comparison against the matching baseline. `report.py` averages the two halves then seeds, computes the unchanged scores using the supplied smoke ranges, and asserts per-aspect mAP and score differences <=1e-6 and identical guards. No full-menu pick claim is made from these sampled read-outs; the controller reruns the full smoke (at least 2,050 rows for whiten1024) and full-scale stage 4 with the unchanged C13 gate.

Example reproduction (set `FIX_C13` to the scratch directory above, one arm/seed per command):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 timeout 235 /root/miniconda3/envs/CoSiR/bin/python "$FIX_C13/agreement.py" cstar --seed 0 > "$FIX_C13/agreement_cstar_0.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 timeout 235 /root/miniconda3/envs/CoSiR/bin/python "$FIX_C13/baseline2050.py" > "$FIX_C13/baseline2050.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python "$FIX_C13/report.py" > "$FIX_C13/agreement_table.md" 2>&1
```

The original `queue_smoke` files were not changed, and no files were deleted. Candidates for later removal: only this task's `fix_c13/` scratch scripts, captured arrays, JSON results and logs, after the controller retains any evidence it needs.

### Agreement results

All 225 per-seed aspect/read-out checks passed. After averaging seeds, all 105 aspect mAP values, 35 normalized scores and 35 guard comparisons passed. Maximum absolute mAP delta: **5.48490e-10** (maximum before seed averaging: **7.43485e-10**); maximum score delta: **5.15192e-8**. The longest arm/seed command took 155.2 s; the 2,050-row R0 check took 32.15 s. Full precision JSON: `agreement_summary.json`; individual results: `agreement_<arm>_<seed>.json` and `baseline2050.json`, in the scratch directory above.

The table gives cross-fit mAP after averaging all required seeds, main / independent, in emotion/style/genre order. Each score uses R0 on the same row count. Narrow-arm whitening sizes follow the original menu.

| Arm | Rows | Read-out | Emotion main / VH | Style main / VH | Genre main / VH | Max abs delta | Score delta |
|---|---:|---|---|---|---|---:|---:|
| cstar | 600 | none+k0 | 0.278776944230 / 0.278776944230 | 0.472843595301 / 0.472843595301 | 0.706111079452 / 0.706111079452 | 1.11e-16 | 0 |
| cstar | 600 | center+k5 | 0.284219989906 / 0.284219989906 | 0.492516221392 / 0.492516221392 | 0.728858981591 / 0.728858981591 | 0 | 1.51e-14 |
| cstar | 600 | whiten64+k3 | 0.236283648404 / 0.236283648404 | 0.292285902516 / 0.292285902516 | 0.504734067189 / 0.504734067189 | 0 | 1.42e-14 |
| cstar | 2050 | none+k0 | 0.282015147941 / 0.282015147941 | 0.411368883807 / 0.411368883807 | 0.631973533723 / 0.631973533723 | 0 | 0 |
| cstar | 2050 | whiten1024+k0 | 0.209462952497 / 0.209462952349 | 0.144432637876 / 0.144432637986 | 0.245334323300 / 0.245334323300 | 1.49e-10 | 9.08e-10 |
| plain | 600 | none+k0 | 0.246013420747 / 0.246013420747 | 0.308944765390 / 0.308944765390 | 0.597002486614 / 0.597002486614 | 0 | 0 |
| plain | 600 | center+k5 | 0.253866696174 / 0.253866696174 | 0.333146330283 / 0.333146330283 | 0.660444236796 / 0.660444236796 | 0 | 0 |
| plain | 600 | whiten64+k3 | 0.229687977325 / 0.229687977325 | 0.275293740193 / 0.275293740193 | 0.480176404388 / 0.480176404388 | 0 | 0 |
| plain | 2050 | none+k0 | 0.255727552234 / 0.255727552234 | 0.246326160402 / 0.246326160402 | 0.530426482462 / 0.530426482462 | 0 | 0 |
| plain | 2050 | whiten1024+k0 | 0.207908012018 / 0.207908012018 | 0.135512379916 / 0.135512379916 | 0.216044949488 / 0.216044949488 | 0 | 0 |
| instructed | 600 | none+k0 | 0.272272049686 / 0.272272049686 | 0.470700382294 / 0.470700382294 | 0.692637322432 / 0.692637322432 | 5.55e-17 | 0 |
| instructed | 600 | center+k5 | 0.277671192010 / 0.277671192010 | 0.489716896551 / 0.489716896551 | 0.712476025822 / 0.712476025822 | 0 | 7.11e-15 |
| instructed | 600 | whiten64+k3 | 0.240158867527 / 0.240158867527 | 0.359391143921 / 0.359391143921 | 0.540137799346 / 0.540137799346 | 0 | 1.42e-14 |
| instructed | 2050 | none+k0 | 0.281594658706 / 0.281594658706 | 0.411504066567 / 0.411504066567 | 0.626283321984 / 0.626283321984 | 0 | 0 |
| instructed | 2050 | whiten1024+k0 | 0.208675241957 / 0.208675241957 | 0.155053176880 / 0.155053176880 | 0.218347260183 / 0.218347260183 | 0 | 0 |
| clip | 600 | none+k0 | 0.244287528696 / 0.244287528696 | 0.387309952885 / 0.387309952885 | 0.532791353549 / 0.532791353549 | 0 | 0 |
| clip | 600 | center+k5 | 0.251640130126 / 0.251640130126 | 0.435129842807 / 0.435129842807 | 0.587886368912 / 0.587886368912 | 0 | 0 |
| clip | 600 | whiten64+k3 | 0.231768682878 / 0.231768682878 | 0.333636299208 / 0.333636299208 | 0.485763382439 / 0.485763382439 | 0 | 0 |
| clip | 2050 | none+k0 | 0.256412615338 / 0.256412615338 | 0.328618454895 / 0.328618454895 | 0.479665592362 / 0.479665592362 | 0 | 0 |
| clip | 2050 | whiten768+k0 | 0.203613259124 / 0.203613258575 | 0.129587023510 / 0.129587023510 | 0.190480858853 / 0.190480858853 | 5.48e-10 | 5.15e-08 |
| siglip2 | 600 | none+k0 | 0.232126260378 / 0.232126260378 | 0.365309116839 / 0.365309116839 | 0.533831513389 / 0.533831513389 | 0 | 0 |
| siglip2 | 600 | center+k5 | 0.246145286855 / 0.246145286855 | 0.420841191682 / 0.420841191682 | 0.599610215825 / 0.599610215825 | 0 | 0 |
| siglip2 | 600 | whiten64+k3 | 0.226186850639 / 0.226186850639 | 0.346586017894 / 0.346586017894 | 0.499461199007 / 0.499461199007 | 0 | 0 |
| siglip2 | 2050 | none+k0 | 0.249100407281 / 0.249100407281 | 0.317629756422 / 0.317629756422 | 0.490428012997 / 0.490428012997 | 0 | 0 |
| siglip2 | 2050 | whiten1024+k0 | 0.202108444405 / 0.202108444405 | 0.120060928405 / 0.120060928405 | 0.178751394795 / 0.178751394795 | 1.39e-17 | 0 |
| percept | 600 | none+k0 | 0.252705864315 / 0.252705864315 | 0.236791649923 / 0.236791649923 | 0.631404642358 / 0.631404642358 | 0 | 0 |
| percept | 600 | center+k5 | 0.255048172722 / 0.255048172722 | 0.234817368490 / 0.234817368490 | 0.640342006845 / 0.640342006845 | 0 | 0 |
| percept | 600 | whiten40+k3 | 0.231239553481 / 0.231239553481 | 0.215522879398 / 0.215522879398 | 0.477010070996 / 0.477010070996 | 0 | 0 |
| percept | 2050 | none+k0 | 0.256676940395 / 0.256676940395 | 0.203324214395 / 0.203324214395 | 0.560386245033 / 0.560386245033 | 0 | 0 |
| percept | 2050 | whiten40+k0 | 0.233378248321 / 0.233378248321 | 0.179896900024 / 0.179896900024 | 0.421548708082 / 0.421548708082 | 0 | 0 |
| crl | 600 | none+k0 | 0.255261676296 / 0.255261676296 | 0.412120034186 / 0.412120034186 | 0.608124829938 / 0.608124829938 | 0 | 0 |
| crl | 600 | center+k5 | 0.260430061455 / 0.260430061455 | 0.440901834283 / 0.440901834283 | 0.677420342542 / 0.677420342542 | 5.55e-17 | 3.55e-15 |
| crl | 600 | whiten50+k3 | 0.220319695625 / 0.220319695625 | 0.250137307401 / 0.250137307401 | 0.408332766981 / 0.408332766981 | 0 | 0 |
| crl | 2050 | none+k0 | 0.265335628741 / 0.265335628741 | 0.365178667920 / 0.365178667920 | 0.526405006687 / 0.526405006687 | 0 | 0 |
| crl | 2050 | whiten50+k0 | 0.225665908203 / 0.225665908203 | 0.221207352218 / 0.221207352218 | 0.340532072770 / 0.340532072770 | 0 | 0 |

## Build 3: the queue smoke (2026-10-29)

GPU exports (20:22 to 20:23, after the Affection extraction freed the lock), from this folder with
`PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1`, under `flock -o -w 7200 /tmp/gpu0.lock`: `rt_percept_val.py`, then
`rt_crl.py lists`, then `rt_crl.py texts`. CRL lists: 50 values per aspect (one width K = 50); values equal to a label
name: emotion 4, style 3, genre 4. `test_rt_data.py` then passed in full (100, no skips).

Smoke: `rt_queue.sh smoke <scratch>` (every step with `--smoke N`, real derived inputs, outputs in scratch).
- At 600 rows (20:24 to 20:36): every step passed up to `rt_vh.py stage4`, which found 174 of 790 values outside 1e-6
  (picks equal). Causes and fix: the section "C13 stage 4 numeric alignment" above. Whitening to 1,024 needs each
  parity half above 1,024 rows, so the queue's smoke now uses 2,100 rows (`SMOKE_N`).
- At 2,100 rows (20:55 to 21:15), after fix-c13: all 228 tests pass; every step exits 0; `rt_vh` agrees on every
  value: stage 0a 24, stage 1 68, stage 2 47, stage 3 19, stage 4 790, stage 5 47 values, 0 disagreements.
  Step times: stage 0a and 0b under 1 minute each; stages 1 to 3 under 1 minute each; stage 4 11.2 minutes (c* 290 s,
  PercepT 139 s, the others 28 to 57 s); stage 5 under 1 minute; the six `rt_vh` checks 5 minutes together.

## Final review fixes (r11 build 4)

Three findings of the final whole-branch review, fixed on `r11-fix2`.

- F1 (should-fix): a full-mode rerun of `rt_queue.sh` could never pass `commit_ranges`. 0a wrote the same doc, with
  wall-clock times, to `<outputs>/stage0a/ranges.json` and to the tracked `stage0a_ranges.json`, so a rerun changed the
  tracked file and `guard()` refused every held stage. Now `rt_paths.RANGES_KEYS = ('map', 'ndcg', 'inputs', 'smoke')`
  names what the committed file holds; `stage0a` writes the full doc to outputs and only those keys to `RANGES_FILE`.
  `rt_paths.ranges_match(outputs_doc_path)` compares `RANGES_FILE` with those keys of the outputs doc (parsed JSON);
  0a's own assert, `rt_run.guard` and the full-mode assert in `stage4` all call it. `rt_vh.py` imports no `rt_` module
  and never reads `RANGES_FILE` (grep: no match); it reads only `<outputs>/stage0a/ranges.json`. Unchanged.
- F2 (should-fix): no test pinned a held-data guard. New `test_rt_guards.py` (15 tests): `guard()` on a temporary git
  repo (missing, untracked, staged, working copy changed, different from outputs, no picks; allowed when equal, also
  with other timings; smoke skips the git checks), `ranges_match`, `stage4`'s assert, `Loader` and
  `Inputs(validation_only=True)` refusing held reads before any store read, and two `stage0a` calls with different
  timings writing a byte-identical `RANGES_FILE`.
- F3 (note): `rt_queue.sh` runs `rt_vh.py stage4` directly after `rt_run.py stage4`, so C13 confirms the picks before
  stage 5 reads them. Order: 0a, commit_ranges, 0b, exports, stage1, stage2, stage3, stage4, vh_stage4, stage5, diag,
  vh_stage0a, vh_stage1, vh_stage2, vh_stage3, vh_stage5.

Mutation check (scratch copy of `src/test`, each guard removed in turn; the real files were not edited):

| Mutation | Result |
|---|---|
| M1 `guard()` returns '' at once | 8 tests fail (all the refusal tests) |
| M2 `Loader._check` does nothing | `test_loader_refuses_held_reads_before_any_store_read` fails |
| M3 `ranges_match` always true | 3 fail (`test_ranges_match_equal_and_unequal`, `test_stage4_full_mode_asserts_the_same_comparison`, `test_guard_refuses_committed_but_different_from_outputs`) |
| M4 `Inputs.read` validation-only assert removed | `test_inputs_validation_only_refuses_held_reads_before_any_store_read` fails |
| M5 `stage0a` writes the full doc to `RANGES_FILE` | `test_stage0a_writes_a_byte_identical_ranges_file_on_a_rerun` fails |

All five caught. `test_rt_guards.py` and `test_rt_vh.py`: 22 passed. Not run: the rest of the round's tests and the queue
smoke (the controller runs them).

- F4 (must-fix, found by the controller after the re-review): the repo's `.gitignore` ignores `*.json`, so the queue's
  plain `git add` of `stage0a_ranges.json` would fail at `commit_ranges` on the first full run, and every held stage
  would then refuse. `.gitignore` now un-ignores that one path; `test_the_real_ranges_file_is_not_gitignored` pins it
  (it fails with the negation removed). The temporary-repo guard tests could not see it (no `.gitignore` there).

## Amendment 2 (PercepT core at full scale)

Failure: the stage 0b regression check failed on one item. The held emotion AMI of PercepT core P3 (own reading) came out
0.08549055120302276; the frozen reference in `rt_stage0.py` `AMI_REFERENCES` is 0.08742922158584242.

Cause: the round read PercepT core from the sample-scale mapper (fit on 13,400 paintings). The reference, and r10's
comparator for c* (a full-scale model), use the full-scale mapper. The frozen r10 functions run on the full-scale held
files give exactly 0.08742922158584242. For seeds 0 to 4 the full `.pt` hash equals the hash in both the full record and the
full held sidecar.

Path change (`rt_paths.PERCEPT_OUT = CT_OUT / 'stage2/full'`; `rt_paths.PERCEPT_VAL_DIR = 'percept_val_full'`):
- held probabilities and sidecar: `held/percept/core__k40/...` becomes `stage2/full/held/percept/core__k40/...`;
- mapper weights and record (export source): `percept/core/stage2/k40/...` becomes `stage2/full/percept/core/stage2/k40/...`;
- `ct_percept.OUTPUTS` during `infer_stage2` is `CT / 'stage2/full'`;
- `rt_vh.py` has the same two paths as its own constants (it still imports no `rt_` module).

Amendment 1 is kept as is: validation is re-inferred from the same mapper as held, with the same hash checks. Only the
mapper changed.

New export folder: `<outputs>/percept_val_full/`. The old `percept_val/` files are not touched, and `_publish` never
overwrites them. The queue checks `percept_val_full/` and runs the export when it is missing (it needs the GPU).

Tests (`test_rt_data.py`): the fixtures and real-data tests use the new paths. Two new real-data tests:
`test_real_percept_held_sidecar_hash_is_full_scale_weights` (seeds 0 to 4: the held sidecar `rt_data` reads has the hash of
the full-scale `.pt`) and `test_real_percept_core_held_emotion_ami_matches_frozen_reference` (own-reading held emotion AMI
through `rt_stage0.held_ami`, equal to 0.08742922158584242 within 1e-9). Real-data tests that need the validation exports
skip while `percept_val_full/` is missing.

## The full run (r11 run, 2026-10-29)

From `/project/CoSiR` on main, `bash rt_queue.sh full`, CPU (8 threads, 8 workers); the PercepT export under the GPU lock.

- First start (22:02): stage 0a passed; its seed ranges were committed and pushed (`9829741`) before any held retrieval
  number. Stage 0b failed on one of 54 items, PercepT core's held emotion AMI (amendment 2 above). The failed record is
  kept as `<outputs>/stage0b/check_failed_A1_sample_scale.json` and `logs/stage0b_failed_A1_sample_scale.log`.
- Fix: `r11-fix3` (Sonnet `high` worker; controller diff read; 128 tests pass, then the 38 PercepT tests with the new
  exports, no skips), merged as `625eb64`, with the spec's amendment A2 (`1a00ff8`).
- Second start (22:10 to 23:02): every step exit 0. Stage 0a reproduced the committed ranges byte for byte
  (`commit_ranges skipped`); 0b 54 of 54; the PercepT export about 1 minute; stage 1 4 min, 2 1 min, 3 1 min, 4 17 min,
  `rt_vh stage4` 11 min, 5 and diag under 1 min each, the other five `rt_vh` checks 10 min together.
- C13: `rt_vh` agrees on every value (0a 24, stage 1 68, stage 2 47, stage 3 19, stage 4 790, stage 5 47; tolerance 1e-6).
- Cross-check against the build's full-scale validation check: stage 0a's ranges equal; stage 4's mAP values equal
  exactly for six arms; PercepT core differs (largest 0.018), as amendment 2 expects.
- Verdicts: stage 1 NO-GO (SC4, emotion against the instructed image: BELOW); stage 2 switch GO; stage 3 qualified;
  stage 4 picks center+k10 for every arm except CRL (none+k10); stage 5 GO. Numbers: the auto report.
