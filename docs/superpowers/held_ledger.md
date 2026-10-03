# Held-read ledger (CVPR plan spec §10)

Every read of a final test split is one row. Final scripts check this file and refuse to run a second time.
Budget: 1 main + 1 reserve read per dataset for the CVPR paper.

| # | Date | Dataset and split | Purpose | Episode / data SHA-256 | Script SHA-256 | Report |
|---|---|---|---|---|---|---|
| H1 | 2026-09-29 | ArtELingo held rows, value (label) episodes, seed 42, 1,024 per label | repaired-factor condition eval | emotion e62ab41f…, style 3a58cf9d… | see report | [condition eval](../reports/auto/v2/2026-10-12_candidate_a_condition_eval_repaired_factors.md) |
| H2 | 2026-09-30 | ArtELingo held rows, same episodes as H1 | stage (d) final test | as H1 | see report | [stage (d) final](../reports/auto/v2/2026-10-14_candidate_a_stage_d_final.md) |
| H3 | 2026-10-01 | ArtELingo held rows, value episodes, seed 43, 8,192 per label | affect factor-learning held test | emotion abd1ca38…, style ee87686c… | 11d5c73f… | [affect held](../reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md) |
| H4 | 2026-10-02 | CUB standard test split (5,794 images, all 200 species; includes the 50 zero-shot test species) | backbone-check attribute probes and retrieval (diagnostic, no CoSiR model) | n/a | see report | [backbone check](../reports/auto/v2/2026-10-25_backbone_check.md) |

**CVPR budget status:** ArtELingo 0 of 2 used (aspect episodes are new; H1 to H3 were value episodes); CUB 0 of 2
(H4 disclosed); SemArt 0 of 2; GeneCIS 0 of 2.
