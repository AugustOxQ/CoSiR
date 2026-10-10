# Episode-seed ledger (ArtELingo aspect episodes, selection rows)

Every scoring of an aspect-episode seed on selection rows is one row. Spec §15: seed 45 is scored once, for one
pre-registered A′. Held rows have their own ledger (`held_ledger.md`).

| Seed | Status | Scored by (date, purpose) |
|---|---|---|
| 42 | development and picks | E1 baselines (2026-10-03); E3 grid pick (2026-10-03); E3 post-hoc fixed-λ profile (2026-10-03); method-repair diagnostics: H1 pilot and H3 transfer (2026-10-03, `src/test/20261105_method_repair_diagnostics/`); quick checks D0, N1, N2 (2026-10-04, `src/test/20261108_new_method_quick_checks/`); N6 dev check, exploratory N6 analysis and N6c gate (2026-10-04) |
| 43 | spent | E1 baselines; E3 GO test, K8 and ablation (2026-10-03); E3 post-hoc fixed-λ profile and bootstrap-seed sensitivity (2026-10-03) |
| 44 | MLLM probe | Qwen3-VL-2B-Instruct v1 and v2, 300 episodes per pair (2026-10-03) |
| 45 | test (spent) | N1-nested-A3 fresh-seed test with 47 and 48, NO-GO; A3's matched control scored for `ADDENDUM_1.md` R3 (2026-10-04, `src/test/20261108_new_method_quick_checks/`); N6c's test not run (gate failed) |
| 46 | spent | Qwen3-VL-8B-Instruct MLLM probe, 600 per pair (2026-10-03, node404; addendum `src/test/20261106_mllm_probe_8b/PREREGISTRATION.md`): works = False |
| 47, 48 | test (spent) | as seed 45 (2026-10-04) |
| 49, 50, 51 | test (spent) | Reader-fix round 3 fresh-seed test of AFF (one-sided affect steering on R1) with R1 beside it, built 2026-10-06 21:00 to 21:08 with `run_baselines.py`, hash-checked against seeds 42, 43, 45, 47, 48 (`src/test/20261121_round3_affect_gate/`) |
| 52, 53, 54 | held (H5) | Round 6 held read of AFF on held rows (not selection rows), 4,096 episodes per pair, built 2026-10-10 03:35 to 03:43 by `run_r6_held.py --mode held`; the nine per-pair SHA-256s are in `held_ledger.md` row H5; also the held GPU jobs of rule §8.3 (verbaliser and listing on 52 to 54, reranker on 52) (`src/test/20261125_artelingo_held_test/`) |
| 55 and later | free | none |
| 9001, 9002, 9003 | smoke | Wiring smoke only (64 episodes per pair, `run_baselines.py --smoke`, written to `results/smoke/`), round 3's end-to-end test (2026-10-06); round 6's smoke (2026-10-10, `run_r6_smoke.py`, under that folder's `results/smoke/`); never results |
