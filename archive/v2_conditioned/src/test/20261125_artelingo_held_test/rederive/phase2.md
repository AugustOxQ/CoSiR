# Round 6, C13 re-derivation, phase 2 (held seeds 52, 53, 54)

**Verdict: AGREE.** 249 quantities compared with the runner's held outputs: 249 equal, 0 differences, nothing
pending. Every point, bound, SE and margin agrees to 0.0, not just within 1e-9. All 43 per-anchor arrays of
`held_arrays.npz` are identical in shape, dtype and values.

Written 2026-10-10 03:56 (Amsterdam) by a Claude subagent standing in for the Codex job of
`.scratch/r6-held-test/run/codex_phase2_brief.md`; Codex had been blocked by a permission check. The brief's rules
applied unchanged. I did not open, grep or import any `r6_*`, `run_r6_*`, `test_r6_*` or `prep_*` file, nor
`final_review/`, `rule_check/`, `design/`, `/project/CoSiR-r6/`, `scripts/run_r6_*` or `das6_sync_r6.py`. The only
`.scratch/` file I opened was the brief.

**Order.** My numbers were written to `phase2_results.json` and `out/rd_held_arrays.npz` first. The SHA-256 of
`phase2_results.json`, `8d353218a509d147ef46ee75dec4a5ccec32287e96243d6f03702190a0213661`, was printed at 03:54.
The first runner held output was opened after that, at 2026-10-10 03:54, starting with `results/held_started.json`.

## What was re-derived (own code; imports as rule §8.4 allows)

| Rule item | Result |
|---|---|
| §5 item 1, held rows | `artelingo_splits(data).held` (61,744 rows) equals `held_codes.npz`'s `held_rows`, the only key read. `groups`, `scorer_train` and `selection` equal `prepare.npz`, and scorer_train ∪ selection equals `split_train`. Train, val and held partition all rows. Held shares no leakage group with train or val, and is disjoint from selection and scorer-train |
| §5 item 2, values | Development sets have 8 emotions, 23 styles and 10 genres, equal to phase 1's. Each is eligible on the held pool (≥ 30 held paintings). The only held-eligible value outside the development sets is New_Realism (style) |
| §5 item 3, episodes | Phase 1's own `values` restriction around `build_aspect_episodes`, on held rows, seeds 52, 53, 54, 4,096 per pair. All pass `validate_aspect_episodes`. Every member is a held row; anchor values and shared example-pair values lie in the development sets. The nine SHA-256s are distinct from each other and from all 24 per-pair hashes in E1's eight non-smoke `baselines_seed*.json`. They equal `held_started.json`, `held_pass.json` and `sensitivity_held.json` (27 items) |
| §6 item 2 / §5 item 5, heads | Refit once with `run_told_oracle.fit_one_head` and `run_n6.fit_heads`, keeping each fitted object. Selection posteriors are bit-equal to `n6_posteriors.npz`, `step1_heads_style.npz` and phase 1's `rd_post.npz`. The kept objects predict held rows with the fitters' own call. All ten coefficient SHA-256s equal `held_started.json`'s `coef_sha256` |
| §5 item 4, frozen picks | B (16, 8 / 8, 2), B0 (16, 16 / 16, 8) and B1 (16, 16 / 16, 8) come from phase 1's own seed-42 cross-fit. Cells are AFF 39/119, CF 149/10, R1 116/119 and R1 counterpart 58/123. RCA λ is 0.5 / 0.5 (tune half 0 / 1, which score parity 1 / 0). They equal the `picks_seed42.json` and `regression_seed42.json` that the held run pinned by SHA-256 in `held_started.json` (12 items, plus 3 file hashes) |
| §5 item 5, assembly | A held context: CLIP features and A3 codes NaN outside held rows. Cosine, T_N1u, and B, B0, B1 through my own nested score z(cos) + λ_u·z(T_N1u) + λ_a·z(T_6u); `nested_scores` is not on §8.4's list. Then the A0 reader, gates, AFF, CF and R1 cells, and RCA fused at λ. Each seed is scored as one 12,288-episode block, and CF stays condition-free per cell (asserted). As a regression, the same function in selection mode on seed 42 reproduces phase 1's 50 per-anchor arrays exactly |
| §8.3 / brief item 3, arrays | AFF, CF, COS, RCA, B, B0, B1 and R1 fused, 5 metrics each, with `cl`, `pair_index` and `seed_index`: 43 arrays, all exact. The key set matches |
| §3, §4, pooled checks | 36,864 episodes and 9,519 clusters. The cluster_bootstrap `ci95` identity is asserted for every check. P1 to P7 each have n_j = 0, the Holm order is P1..P7, and all pass. S1 has n = 258 and S2 n = 1,611, with S1 at Holm k = 1 and S2 at k = 2. Each check's n_j, point, 95% and Holm-level bounds, k, level, boundary flag and pass are equal (111 `held_pass` items) |
| §8.2, sensitivity | σ² comes from phase 1's seed-42 split, with Σ_p M_p² over the pooled anchors. SE, x and x₉₅ for P1 to P7, and SE, x₂ and x₉₅ for S1, S2 all equal `sensitivity_held.json`, as do N and n_paintings. The σ² source file's SHA-256 matches |

Pooled points (R@1 points, my file, equal to the runner's):

| Check | n_j | point | 95% |
|---|---|---|---|
| P1 | 0 | 6.1896 | [5.9612, 6.4030] |
| P2 | 0 | 5.8946 | [5.6714, 6.1113] |
| P3 | 0 | 0.6300 | [0.5076, 0.7521] |
| P4 | 0 | 0.5995 | [0.4629, 0.7337] |
| P5 | 0 | 0.6314 | [0.5016, 0.7606] |
| P6 | 0 | 3.0619 | [2.8691, 3.2547] |
| P7 | 0 | 2.9188 | [2.7025, 3.1375] |
| S1 | 258 | 0.1356 | [−0.0279, 0.3025] |
| S2 | 1,611 | 0.0217 | [−0.0628, 0.1118] |

`held_pass.json` records no pass or fail for S1 and S2: they are tested only after a GO, by the apply step. My own
Holm over S1 and S2 (`rd_held_stats.json`) gives neither passing: S1's count of 258 is above n*₁ = 61, and S2's
1,611 is above n*₂ = 124. Those two pass flags were not compared.

**Boundaries (rule §8.5).** None. Every P check has n_j = 0 against n*_k of 16 to 124, and S1 and S2 are far from 61
and 124.

## Comparison breakdown

held_started 24 (episodes 9, coefficients 10, rule SHA, attempts, 3 pinned seed-42 records), frozen picks 12,
held_pass 111, held_arrays 44 (43 arrays and the key set), sensitivity_held 58.

## Differences

None.

## What I read outside `rederive/`

- `F/DECISION_RULE.md` (whole); the R3 rule's headings, §4 and §8.
- Library and earlier-round code, read only to call it or to mirror its interface:
  - `src/data/artelingo_splits.py`, `src/eval/aspect_episodes.py`, `src/eval/aspect_quick_checks.py`
    (`crossfit_condition_free`, `uniform_probe_scores`), `src/eval/aspect_nested.py`,
    `src/eval/aspect_scorers.py` (`fused_scores`, `crossfit_lambda`), `src/model/aspect_rule.py` (`zscore_rows`),
    `src/eval/aspect_metrics.py` (`cluster_bootstrap`), `src/train/train_factors.py` (`encode_rows`), and
    `src/eval/pair_metric_baselines.py` (signatures);
  - `run_n6.py` (`fit_heads`), `run_checks.py` (`unit`, `model_inputs`), `run_told_oracle.py` (`fit_one_head`),
    `rb_build.py` (`load_readers`), and `run_gonogo.py` (the `EvalContext` class, read so the held context could
    mirror it; not imported).
- Data: `prepare.npz`, `held_codes.npz` (`held_rows` only), and E1's `baselines_seed*.json`, together with every
  SHA-pinned input phase 1 used.
- Runner outputs, from 03:54, after the SHA-256 above: the `*held*` names in the `results/` listing,
  `held_started.json`, `held_pass.json`, `sensitivity_held.json` and `held_arrays.npz`. Also the seed-42 records
  `picks_seed42.json` and `regression_seed42.json` (content) and `sensitivity_seed42.json` (hash only).
- Not opened: `held_episodes_seed5*.npz` (my hashes were compared through the three JSON records), `held_jobs/`, any
  `run_r6_*.log`, any DTS, FT or MLLM output, and `held_verdict.json`, which does not exist.

## Files

Scripts `rd_held_episodes.py`, `rd_held_scores.py`, `rd_held_stats.py` and `rd_compare_held.py`. Step records
`rd_held_episodes.json`, `rd_held_scores.json` and `rd_held_stats.json`, all folded into `phase2_results.json`;
`phase2_compare.json`. In `out/` (gitignored): `rd_held_arrays.npz` (19.8 MB), `rd_held_episodes.npz` (8.9 MB),
and the logs `rd_held_*.log` and `rd_compare_held.log`. `out/` now totals 102 MB with phase 1's files. Nothing was
deleted, and no phase-1 record was changed.
