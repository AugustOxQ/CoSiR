# Episode-seed ledger (ArtELingo aspect episodes, selection rows)

Every scoring of an aspect-episode seed on selection rows is one row. Spec §15: seed 45 is scored once, for one
pre-registered A′. Held rows have their own ledger (`held_ledger.md`).

| Seed | Status | Scored by (date, purpose) |
|---|---|---|
| 42 | development and picks | E1 baselines (2026-10-03); E3 grid pick (2026-10-03); E3 post-hoc fixed-λ profile (2026-10-03); method-repair diagnostics: H1 pilot and H3 transfer (2026-10-03, `src/test/20261105_method_repair_diagnostics/`) |
| 43 | spent | E1 baselines; E3 GO test, K8 and ablation (2026-10-03); E3 post-hoc fixed-λ profile and bootstrap-seed sensitivity (2026-10-03) |
| 44 | MLLM probe | Qwen3-VL-2B-Instruct v1 and v2, 300 episodes per pair (2026-10-03) |
| 45 | reserved: the single A′ GO test | none |
| 46 and later | free; any 8B MLLM probe uses 46 or later | none |
