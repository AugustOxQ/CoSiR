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
| 49 and later | free | none |
