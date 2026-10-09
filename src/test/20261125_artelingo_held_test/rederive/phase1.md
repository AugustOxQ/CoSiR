# Round 6, C13 re-derivation, phase 1 (seed 42, selection rows)

**Verdict: AGREE.** 389 quantities compared with the runner's result files, 389 equal, 0 differences, 0 files
pending. No held row was read.

Written 2026-10-10 00:05 (Amsterdam) by an agent that did not open or grep the round-6 implementation (`r6_*`,
`run_r6_*`, `test_r6_*`, `prep_episodes_seed42.py`, `final_review/`, `rule_check/`, `design/`, `.scratch/`,
`/project/CoSiR-r6/`, `scripts/run_r6_*`, `das6_sync_r6.py`). The runner's files in `results/` were read only after
my numbers existed.

## What was re-derived (own code, imports as rule §8.4 allows)

| Rule item | Result |
|---|---|
| §6 item 2, head refits | All 8 stored selection posteriors (n6 `affect/image/caption__img/txt`, step 1 `style_csd__img/txt`) reproduced bit for bit; affect head identical to told_oracle arm L |
| §6 item 3, B, B0, B1 | Picks per tune half equal the runner's; mean R@1 exactly 18.341064453125, 18.436686197916664, 18.804931640625 |
| §6 item 4, regression | AFF 19.136555989583336, CF 18.39599609375, bar margin, gain statistic, AFF − R1, R1's margin and gain statistic, AFF − B1: all equal to the rule's values exactly (difference 0.0) |
| §6 item 4, per-anchor arrays | 98 array checks equal: AFF, CF, R1, R1 counterpart (5 metrics each), gates, cells, σ*, P, margins, picks, τ; cosine, RCA and the nine PM scorers against `per_anchor_seed42.npz` |
| §5 item 4, frozen picks | AFF/CF/R1 cells; RCA and PM λ per half (my own cross-fit of the λ rule reproduces the recorded picks); B, B0, B1 picks |
| §5 items 2 and 3, episodes | Value sets 8 emotions, 23 styles, 10 genres (codes and names); my `values` restriction reproduces the per-pair hashes of seeds 42, 9001, 9002, 9003 |
| §3, §4 on seed 42 | P1 to P7: n_j = 0 each, Holm order P1..P7, all pass; S1 n = 54, S2 n = 15, Holm order S2, S1, both pass; no check within one count of a boundary |
| §6 item 5, sensitivity inputs | σ_a², σ_ε² for P1 to P7, S1, S2 equal the runner's; the eight checks shared with round 3 also equal round 3's `sensitivity.json` bit for bit |

The runner's coefficient SHA-256 format is SHA-256 over `coef_` bytes followed by `intercept_` bytes; my refits give the
same ten hashes.

## Comparison breakdown

refit 50, picks 33, value sets 6, regression 171 (including the runner's 96 array verdicts against my own),
pass counts 98, per-episode differences 11 (exact arrays), sensitivity 20. Not in phase 1: DTS picks
(`dts_seed42.json`), held quantities (phase 2).

## Files

Scripts `rd_common.py`, `rd_heads.py`, `rd_episodes.py`, `rd_seed42.py`, `rd_stats.py`, `rd_compare.py`,
`rd_compare_late.py`; step records `rd_heads.json`, `rd_episodes.json`, `rd_seed42.json`, `rd_stats.json` (folded into
`phase1_results.json`); `phase1_compare.json`. Arrays and logs in `out/` (gitignored, 75 MB: `rd_post.npz` 63 MB,
`rd_arrays_seed42.npz` 9.5 MB, `rd_episodes_seed42.npz` 2.9 MB). They serve phase 2; delete after it.
