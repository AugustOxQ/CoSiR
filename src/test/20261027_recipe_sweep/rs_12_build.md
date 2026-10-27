# Ticket 12 build notes (rs_vp.py), progress note at the 200k-token mark

Status: code and tests done and green (39 tests); not done: the sections marked OPEN below. Branch r10-rs-12.

## Built
`rs_vp.py` modes: `partitions --machine M [--runs ...]`, `sweep --machine M`, `stage3 --sweep-machine M [--draws N]`,
`ablation --final NAME [--draws N]`, `self-check [--draws N] [--all-arms]`. `--outputs` everywhere. Each writes
`rederive_<check>.json` (pass, items with value, expected, diff, SHA-256 of every input read) beside the file it checks:
scores/M/rederive_partitions.json, sweep/M/rederive_stage{1,2}.json, stage3/rederive_stage3.json,
ablation/rederive_dev.json, <outputs>/rederive_selfcheck.json. Exit 1 on any disagreement.

## Definitions settled (one line of why each)
- r_a, tolerances, guard, grid and tie rules are constants in rs_vp.py with the rule section in a comment (C13).
- Positions for the TOP tie-break are read off rs_sheet.stage1() order (the sheet's listed order), not rs_sheet.positions.
- Part (b) derives twice (scorer's A, my A from rederive_partitions.json) and requires the same decisions; the stage
  file is compared with the scorer-A derivation. Lists the rule fixes no order for (grid knobs, levels, grid points,
  pool) compare as sets; TOP is ordered.
- Stage 3 takes TOP from my own stage 2 derivation (my A values), so `stage3` needs `--sweep-machine`.
- Joint bootstrap: draws are identical for every configuration (own default_rng(0)), so all arms are scored in one pass.
- Verdict layouts for stage3/verdict.json and ablation/dev.json are ASSUMED (configs or results -> name -> Abar, e{e,lo,hi},
  M{M,lo,hi}, qualifies; ablation: e or effects); `norm_config` and `compare_stage3` / `compare_ablation` are the one place to adapt.
- Ablation compares point values e_a (intervals only with --draws > 0); scorer partitions/score files compared when present.
- code_state returns ('no-git', None) when git is missing (the node copy may have no .git).

## Self-check (points only, run 2026-10-27)
`python rs_vp.py self-check --outputs /tmp/claude-0/r10-rs-12`: 57 items, 0 differ (stage A GS-0.02 and R6-5 seed 0 partitions and
A; stage B GS-0.02 vs R6-5 seeds 1-3 partitions, A-bar, per-seed A and e-bar), see rederive_selfcheck.json there.

## OPEN (for the next agent)
- Run self-check with intervals: `rs_vp.py self-check --draws 2000` (expected: per-aspect lo/hi all three aspects and joint lo/hi for emotion and style equal stage B's, within 1e-6; about 10 min on 8 workers; background, log to /tmp/claude-0/r10-rs-12/).
- Untested against real data: `partitions` (needs a real outputs tree), `ablation` wrapper (only `ablation_core` and `compare_ablation` are unit tested).
- Confirm verdict layouts with the controller after tickets 06 and 10 land.
- Merge r10-recipe-sweep and rerun tests before the final hand-back.
