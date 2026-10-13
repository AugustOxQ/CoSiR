# Ticket 22 hand-back — 2026-10-14

Built `run_r9_preread.py`: rule §6 inputs → refits → seed-42 regression → sensitivity → phase-1 agreement → smoke. Each stop has a marker; a failure records the failed check and every later check as not run. Records and smoke outputs are exclusive writes. `run_r9_held.run_held` refuses a missing, failed or stale pre-read before computing a read.

Existing-file edits and reasons:

- `run_r9_held.py`: the launch gate is necessary to make the stops gate actual held launches; `setup_a28(..., inputs_only=True)` reaches the real-read-set branch and returns only input features/posteriors/A3 codes/masks. Both smoke and inputs-only loads use a scoped, non-label parquet projection through the existing A28 loader. Sentinel label columns satisfy its schema and are discarded at the inputs-only boundary; real labels are never decoded. Smoke names import `AR.SMOKE_RECORDS`.
- `rederive/rd9_compare.py`: phase-1 output now certifies its current pins SHA and verifies its independent input files. `--phase1 --backbone clip_b32` uses the existing CLIP comparison, including the three baseline means. New-backbone comparison understands stage 2's actual namespaced `seed42.arrays` schema. It checks every pinned hash exposed by phase 1, with explicit required primary scorers, cluster/parity/cells and reader inputs. The legacy `array_hashes` schema remains supported. The pre-read's own regression checks all 96 stage-0a array hashes on new backbones, including R1/PM.
- `test_r9_held.py`, `test_r9_held_a28.py`: existing scenarios stub the new gate to continue isolating their original held guards; marker inventories include the independently tested launch gate. No original production guard was weakened.
- `.claude/20261014_log.md`, `20261202_r9_scope_log.md`: required edit logs.

No edit to `run_r9_stage2.py` or any of its 17 closure modules. No helper module was needed.

Choices settled:

- Stage-0a acceptance reads only `stage0a_regression.json`, requires `passed: true`, exact current `r9_module_shas()` and the stage-0a runner SHA. An old pass-counts file supplies no permission. The controller must refresh stage 0a after these runner/module additions; stage 2's certified recipe closure stays unchanged.
- The owner pins check is reused. All declared parts are authenticated before calling it, so even its internal `read_cells` call follows a SHA check; subsequent `load_picks` and `read_cells` follow the completed input gate.
- The phase-1 comparer is **not launched by the pre-read**. It selects `AR.AGREEMENT_DIR/rd9_<B>_phase1_agreement_fix1.json`, then `_crash1`, then the base `rd9_<B>_phase1_agreement.json`, requiring agreement true, phase 1, current pins SHA, exact independent JSON/NPZ identity (including named corrected outputs within the owner output directory), and the independent JSON's current pins binding. Task 25's independent outputs must already exist there.
- Sensitivity is recomputed with `r9_stats.check_diffs`, `sigma_split` and `sensitivity_record`; variance components must equal the owner's file. New-backbone sources are never rewritten. A scratch run copies the exact certified bytes once for the held smoke's results boundary and records that active copy's SHA too; finite-but-changed copies fail the held launch gate. **CLIP reads R6's sensitivity directly; no r9 CLIP sensitivity file is created.** Its SHA is recorded.
- The smoke exercises both ArtELingo and the backbone's A28 path on selection rows, seeds 9001–9003, 64 episodes per pair. A28 uses row-ID-mod-27 languages and the pinned V, shares and language seeds. It also exercises the real-read-set `inputs_only` branch before any synthetic episode/score.
- The end-to-end smoke includes the actual held runner, statistics/Holm/pass, apply step, and descriptive pass. As R6, its phase-2 agreement is an **explicitly labelled smoke stand-in**, not an independent C13 claim. Producer records live below `AR.AGREEMENT_DIR/smoke/<read>_<variant>_<smoke-read>/`; their same-name copies go into that run's scratch smoke folder, which is explicitly passed to the apply step. Real agreement locations stay unchanged.
- The descriptive smoke projects real non-label metadata. It explicitly omits thin-value disclosures requiring real A28 labels; every scored row remains a synthetic ArtELingo selection row.
- The wiring mutation executes a kept, isolated copy of `r9_score.py` on the actual selection smoke bundle. The owner's CF condition-free check must fire with its specific message (`ValueError`, as that owner implements it). The original score module is never edited by the smoke.
- Each read keeps its own immutable smoke proof. A current canonical smoke record and its F copy may be retained for later reads with identical module/runner bytes; a stale canonical record requires a fresh R6 variant. Both carry `passed: true`, current `r9_module_shas()` and the held runner SHA. The proof also records the pre-read runner SHA.
- Pre-read reruns use `_fix1`, `_crash1`, `_reserve`; normal launch selects fix1, then crash1, then the base record. A failed newer proof blocks an older pass. A pre-read fix1 can authorise a first held launch without the held runner's `--fix` flag. Reserve/fix held launches require their respective proofs.
- `--results` defaults to F/results and must remain inside the executing worktree. All build-test outputs stayed below F/results. No main-checkout result, real ledger or held-row score was written. An initial mutant produced a synthetic F smoke copy; the fixture now isolates that copy under tmp_path and the exact test-created invalid artifact was removed before commit.

Validation: **653 passed**, including **62 pre-read cases** and 591 existing held/comparer cases. Forty-one actual temporary guard-deletion mutants plus one active-sigma SHA-binding mutant are recorded in `t22_mutation_results.json`; each source edit was restored in `finally`. One initial scratch-sigma test was shadowed by the later runner refusal; its stub was corrected to reach the next boundary and its final mutant is caught. The wiring control really scores synthetic terms with zero CF gain; its wiring mutation fires. No GPU, full CLIP smoke or full CLIP pre-read was run.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider src/test/20261202_r9_scope/test_r9_preread.py src/test/20261202_r9_scope/test_r9_held.py src/test/20261202_r9_scope/test_r9_held_a28.py src/test/20261202_r9_scope/rederive/rd9_test_compare.py src/test/20261202_r9_scope/rederive/rd9_test_arithmetic.py --basetemp src/test/20261202_r9_scope/results/t22_final_tests
```

Controller work remains: the real CLIP smoke/pass record required by the ticket must be produced by the full command below. Its expected duration exceeds the four-minute agent command limit. First provide the owners' pinned inputs and passed stage certificates in this checkout; its untracked stage certificates/independent outputs were absent during implementation. Refresh stage 0a on final bytes (estimated 6–10 minutes CPU):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage0a.py
```

After task 25's independently prepared/scored files exist, run this **before** the CLIP pre-read (usually seconds; refuses an existing agreement; keep old evidence and give task 25 a fresh `_fix1` agreement output name if necessary; the pre-read selects it automatically):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --phase1 --backbone clip_b32 --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_clip_b32_phase1.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_clip_b32_seed42_arrays.npz --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_clip_b32_phase1_agreement.json
```

Full CLIP pre-read including both smoke paths and F's smoke-record copy, estimated **10–20 minutes CPU**. Run from `/project/CoSiR-wt/r9-t22`:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_preread.py --read H8 --results /project/CoSiR-wt/r9-t22/src/test/20261202_r9_scope/results/t22_controller
```

For the new backbones, compare their existing independent outputs against their current pins (usually seconds each):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --phase1 --backbone qwen3vl_emb_2b --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_phase1.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_seed42_arrays.npz --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_qwen3vl_emb_2b_phase1_agreement.json
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/rederive/rd9_compare.py --phase1 --backbone siglip2_so400m --derived src/test/20261202_r9_scope/rederive/rd9_out/rd9_siglip2_so400m_phase1.json --derived-arrays src/test/20261202_r9_scope/rederive/rd9_out/rd9_siglip2_so400m_seed42_arrays.npz --out src/test/20261202_r9_scope/rederive/rd9_out/rd9_siglip2_so400m_phase1_agreement.json
```

Full pre-reads, estimated **15–30 minutes each for Qwen; 20–40 minutes each for SigLIP**. These commands score selection rows only:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_preread.py --read H6 --results /project/CoSiR-wt/r9-t22/src/test/20261202_r9_scope/results/t22_controller
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_preread.py --read H7 --results /project/CoSiR-wt/r9-t22/src/test/20261202_r9_scope/results/t22_controller
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_preread.py --read H9 --results /project/CoSiR-wt/r9-t22/src/test/20261202_r9_scope/results/t22_controller
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_preread.py --read H10 --results /project/CoSiR-wt/r9-t22/src/test/20261202_r9_scope/results/t22_controller
```

These scratch proofs authorise launches only when the held runner uses the same results root. For production, run the pre-read with its default results root in the production checkout on final bytes, and commit the successful smoke-record copy before the ledger/launch. This task did not authorise a held launch or a ledger change.
