# Figures for the affect steering stage report (2026-10-12)

Built by `build_figures.py`; every plotted number is in `figure_data.json`.

- `chain.png`: the line of work in date order, 15 steps coloured by outcome (worked, did not work, comparator or diagnostic). Source: hand-typed from the brief.
- `aff_pipeline.png`: AFF as a condition-free base lane plus a gated steering branch, joined in a fused score. Source: hand-typed diagram, no numbers.
- `held_comparison.png`: on held paintings AFF is far above every external baseline and about 0.6 R@1 points above the condition-free bars (R@1 and condition gain, 95% intervals). Source: `src/test/20261125_artelingo_held_test/results/descriptive.json` (`rows[*].pooled`).
- `frontier.png`: R@1 splits into either rate and condition gain; only R1 and AFF gain on choosing. Source: same held file, either rate = 2 x R@1 - gain (computed).
- `per_pair_fresh_held.png`: AFF minus B'(A0) per aspect pair, fresh (round 3) and held (round 6). Source: round 3 `descriptive.json` (`item2_bar_margin_cells_frozen.AFF_bar_margin_pooled`) and held `descriptive.json` (`bar_margin.AFF.pooled`).
- `random_gate.png`: margin over B'(A0) on fresh episodes for R1, AFF and two random one-sided gate draws; the random gate matches AFF. Source: round 3 `descriptive.json` (`item3_R1_checks`, `item2...`, `item7_random_share_control`).
- `protocol.png`: the three stages (development episodes, fresh test, held read), the conditions to move on, the seven checks and the training split. Source: hand-typed from the brief; no plotted data. Built by `build_figures_2.py`.
- `after_go.png`: development-episode (seed 42) margins of AFF and of seven attempts to improve it, each over its own bar comparator, with the +0.5 bar. Source: `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md`, `2026-11-23_idea3_goemotions.md`, `2026-11-26_design_l_style_source.md` (S1 column); values typed in `build_figures_2.py`.
- `mllm.png`: MLLM and describe-then-score R@1 minus cosine, and condition gain, with AFF off the axis. Sources: probe rows from `2026-11-01_aspect_factor_gonogo.md` §9.2 and `2026-11-08_new_method_quick_checks.md` (8B, seed 46); held rows from the held `descriptive.json`. The held MLLM difference uses cosine on the same seed 52 (13.24), so it is +1.27, not +1.25 (pooled cosine 13.25). Describe-then-score is pooled over seeds 52 to 54 (13.06 against 13.25, −0.19).

Figures 1 to 6 are built by `build_figures.py`, figures 7 to 9 by `build_figures_2.py` (imported by the first; run `build_figures.py` to rebuild all).
