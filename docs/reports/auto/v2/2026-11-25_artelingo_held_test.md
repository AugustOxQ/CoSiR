# ArtELingo held test of AFF: GO; pooled over three aspect pairs, AFF beats cosine, RCA, B, B′(A0) and its matched control on held paintings (+0.60 to +6.19 R@1 points); against B′(A1) and R1 not established

> Report date 2026-11-25 (a sequence number; written 2026-10-10, Amsterdam; the read ran 2026-10-10 03:33 to 03:44) · Spec: [2026-10-09-r6-held-test-design.md](../../../superpowers/specs/2026-10-09-r6-held-test-design.md) · Decision rule: `src/test/20261125_artelingo_held_test/DECISION_RULE.md` @ c394b60 (never amended) · Run log: [20261125_artelingo_held_test_log.md](../../../../src/test/20261125_artelingo_held_test/20261125_artelingo_held_test_log.md)

## Results brief

**What we tried.** One pre-registered read of AFF (affect steering, the current best method) on ArtELingo's held paintings, which nothing in AFF was fitted, picked or tuned on; a claim test with every weight frozen on development seed 42 ([spec](../../../superpowers/specs/2026-10-09-r6-held-test-design.md)). 12,288 episodes on each of three fresh seeds (52 to 54), 36,864 in all, 4,096 per aspect pair per seed.

**Outcome against the bar.** The verdict is GO: all seven checks pass with Holm, pooled over the 36,864 episodes (R@1 in points, 95% interval; AFF 19.44).

| Comparator | AFF minus it, R@1 |
|---|---|
| cosine (13.25) | +6.19 [+5.96, +6.40] |
| RCA (13.54) | +5.89 [+5.67, +6.11] |
| B (18.81) | +0.63 [+0.51, +0.75] |
| B′(A0) (18.84) | +0.60 [+0.46, +0.73] |
| matched control (18.80) | +0.63 [+0.50, +0.76] |

Condition gain (correct condition minus swapped, same target): AFF +3.06 [+2.87, +3.25] over every condition-free scorer (their gain is 0) and +2.92 [+2.70, +3.14] over RCA. The two secondary checks, against the strongest condition-free scorer B′(A1) (+0.14 [-0.03, +0.30]) and against the reader R1 (+0.02 [-0.06, +0.11]), are both inconclusive.

**What it does not show.** No per-pair claim: on style × genre AFF is below B, B′(A0) and its control (-0.43 [-0.65, -0.21] against B′(A0)), and its margin over B′(A1) is -1.24 there. One dataset (ArtELingo), one backbone (CLIP ViT-B/32). A claim test licenses only the text of the claim (below); AFF was found among about 50 variants on seed 42, so the disclosures of rule §10.4 travel with every AFF number. The margin over B, B′(A0) and the control is small (about 0.6 points) against the +3.06 condition gain. Over the strongest condition-free scorer, B′(A1), it is +0.14 [-0.03, +0.30], not established. Nor does the test show that AFF's affect gate matters: R1, the same reader without it, clears all seven checks at 95% (descriptive), and AFF minus R1 is +0.02 [-0.06, +0.11].

**Options for next.** (a) Design L, a grouping redesign aimed at the style × genre deficit, already chosen as the next loop (user, 2026-10-09); its own spec, rough estimate a few days of CPU work plus DAS6 jobs; it could show whether the style grouping can be made to help instead of leak into genre. A design-L read on ArtELingo held rows would be a new held read (a new ledger row, C5), disclosed as designed after this read. (b) Write the ArtELingo-centred paper now around this claim and the per-pair table; no compute, but the claim is a small pooled margin and the style × genre result must be shown beside it. (c) A second dataset or backbone (CUB, SemArt, a second backbone); needs new data prep and a new held-style read; it could show whether the margin is specific to ArtELingo.

**Your reading.** *(left for the user)*

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The pooled claim holds: GO, and the margins over B, B′(A0) and the control are about +0.6 on each of the three seeds (+0.58 to +0.66). The seeds draw on nearly the same paintings, so this agreement is not an independent replication; the pooled interval, clustered on anchor paintings, already accounts for it. The intervals stay above zero when candidate or target paintings are resampled too (P4's lower bound is +0.45, or +0.40 with targets). The size is modest, and it comes from emotion × style and emotion × genre (+0.59 and +1.64 over B′(A0)); style × genre loses. We are fairly sure of the pooled sign over B, B′(A0) and the control, not of any margin over B′(A1) or R1, and less sure the margin would survive another dataset.

**Our recommendation.** Option (a), design L, since the style × genre loss is the clearest weakness and the Fri 2026-10-16 go/no-go needs to know whether it is fixable, with option (b) drafted in parallel from this report. This is our view; the user decides.

</details>

## What ran

- **Stages and types:** (1) baseline stage on development seed 42: build and tune the describe-then-score comparator, refit the classifiers, store every scorer's picks, reproduce round 3's seed-42 numbers (preparation, no claim); (2) the claim test on held rows (type: claim test, full loop). Re-derivation by an independent agent, then the descriptive pass.
- **Data and seeds:** ArtELingo held rows (the held split, 61,744 rows from 12,281 paintings, disjoint from selection and scorer-train rows); episode seeds 52, 53, 54; seed 42 only for the frozen picks, DTS tuning and the regression check. The seeds are ledgered as held seeds of row H5.
- **Code:** build merged into main at df421a7 (15 tickets, three-part final review, one fix wave, scoped re-review); run commits to b71e7b3. Decision rule SHA-256 prefix 7444a5e338; the runner `run_r6_held.py` a82ca8bc128e equals the SHA recorded in ledger row H5 before the read. The read ran locally on CPU (620 s, one attempt, no flag). DAS6 node401, node402 and node408 ran every GPU job: the seed-42 DTS and smoke jobs (2026-10-09 23:48 to 2026-10-10 03:24), the 40 held jobs (nine GPUs, 03:45 to 07:44: DTS verbaliser and listing, FT features, the seed-52 MLLM reranker) and six rebuilt held listing shards (07:54 to 08:10).
- **Pre-read checks (rule §6), all passed:** refit reproduces eight arrays bit for bit; picks equal their targets; regression 131 items, 0 failed; DTS built within its 24-hour budget with no stop; smoke passed; re-derivation phase 1 (seed 42) 389 of 389 agree.

## Results

All seven checks pass; the verdict follows the rule's integer Holm procedure over 5,000 cluster bootstrap resamples (anchor paintings, seed 42). n_j counts the resample means at or below 0; a check passes when n_j is at most the limit in brackets. All figures are from `held_verdict.json`. Sample: 36,864 episodes, 9,519 anchor paintings.

| ID | Measured (R@1 points) | Number [95%] | Holm-level interval | n_j (limit) | Outcome |
|---|---|---|---|---|---|
| P1 | AFF minus cosine | +6.19 [+5.96, +6.40] | [+5.89, +6.50] | 0 (16) | pass |
| P2 | AFF minus RCA | +5.89 [+5.67, +6.11] | [+5.62, +6.19] | 0 (19) | pass |
| P3 | AFF minus B | +0.63 [+0.51, +0.75] | [+0.47, +0.79] | 0 (24) | pass |
| P4 | AFF minus B′(A0) | +0.60 [+0.46, +0.73] | [+0.42, +0.77] | 0 (30) | pass |
| P5 | AFF minus matched control | +0.63 [+0.50, +0.76] | [+0.47, +0.79] | 0 (40) | pass |
| P6 | condition gain, AFF minus control (also against cosine, B, B′(A0)) | +3.06 [+2.87, +3.25] | [+2.85, +3.29] | 0 (61) | pass |
| P7 | condition gain, AFF minus RCA | +2.92 [+2.70, +3.14] | [+2.70, +3.14] | 0 (124) | pass |
| S1 | AFF minus B′(A1) | +0.14 [-0.03, +0.30] | [-0.06, +0.32] | 258 (61) | inconclusive at a detectable margin of 0.265 points (realised half-width 0.165) |
| S2 | AFF minus R1 | +0.02 [-0.06, +0.11] | [-0.06, +0.11] | 1,611 (124) | inconclusive at a detectable margin of 0.142 points (realised half-width 0.087) |

S1 and S2 are tested only after a GO and never change the verdict. No check is near its boundary (`boundary_report` is empty).

**Verdict under the rule: GO.** Claim licensed, as written in `held_verdict.json`: "AFF beats COS, RCA, B, B′(A0) and its matched control on aspect R@1, and its condition gain exceeds theirs and RCA's, pooled over three aspect pairs, on paintings never used to fit, select or tune it, with Holm across the seven checks. No per-pair margin, no other dataset or backbone. Disclosures of §10.4 accompany every AFF number."

**Disclosures (rule §10.4), with every AFF number:**

1. AFF was found among about 50 label-free variants on seed 42, and rounds 4 and 5 read about 10 more there.
2. DTS was tuned with at most 8 settings on seed 42.
3. The held paintings were read with value episodes in H1 to H3 (earlier held ledger rows).
4. 24 held rows have a training image at CLIP cosine of at least 0.99 (63 at 0.98 or more).
5. One A3 checkpoint and one reader pair; the plan's three training seeds do not apply to frozen AFF.
6. Later designs (design L) are made after this read.
7. Per-pair results are reported beside the pooled claim (Analysis), with the style × genre margin disclosed: on held paintings -0.43 [-0.65, -0.21] over B′(A0) (round 3: -0.580 [-0.793, -0.363]), and design L named as the route to it.

## Analysis

Everything in this section is descriptive and decides nothing (rule §10.5). Source: `descriptive.json`.

**Per pair, the pooled margin is two wins and a loss.** AFF minus each comparator, R@1 points, 12,288 episodes per pair; the last column is the margin over B′(A1).

| Pair | vs B | vs B′(A0) | vs control | gain over control | vs B′(A1) |
|---|---|---|---|---|---|
| emotion × style | +0.86 [+0.64, +1.10] | +0.59 [+0.35, +0.83] | +0.45 [+0.24, +0.67] | +0.86 [+0.56, +1.15] | +0.35 [+0.06, +0.62] |
| emotion × genre | +1.60 [+1.38, +1.81] | +1.64 [+1.40, +1.90] | +1.73 [+1.50, +1.97] | +6.05 [+5.68, +6.41] | +1.30 [+1.00, +1.60] |
| style × genre | -0.57 [-0.75, -0.38] | -0.43 [-0.65, -0.21] | -0.28 [-0.48, -0.09] | +2.28 [+1.96, +2.59] | -1.24 [-1.51, -0.96] |

Our reading (descriptive, untested): the style × genre loss recurs on new paintings at a similar size (-0.43, against round 3's -0.580 over B′(A0)), so it is unlikely to be an accident of one set of paintings or episodes. It is the weakness design L targets. This table is the per-pair result the rule requires beside the pooled claim.

**Per seed, the margins keep their sign and size, on largely shared paintings.** AFF minus each comparator, R@1 points, 12,288 episodes per seed; every seed draws its candidates from nearly the same 9,947 held paintings, so the rows are not independent replications (the bar comparator is B on seed 52, the control on seed 53 and B′(A0) on seed 54):

| Seed | vs cosine | vs B | vs B′(A0) | vs control | gain over control | vs B′(A1) |
|---|---|---|---|---|---|---|
| 52 | +6.21 | +0.60 | +0.60 | +0.63 | +3.07 | +0.09 [-0.18, +0.38] |
| 53 | +6.12 | +0.63 | +0.61 | +0.61 | +2.93 | +0.12 [-0.16, +0.40] |
| 54 | +6.24 | +0.65 | +0.58 | +0.66 | +3.18 | +0.20 [-0.09, +0.47] |

Each of the P3 to P5 per-seed intervals excludes zero (lower bounds +0.35 to +0.44); the S1 intervals include zero on every seed. AFF minus R1 is +0.06 [-0.06, +0.17] on the bar margin and +0.19 [+0.07, +0.32] on gain. R1's own seven checks all have lower bounds above 0 (descriptive only).

**External comparators** (reported, never gating; pooled R@1 on the 36,864 episodes unless noted, with AFF minus each):

| Comparator | R@1 | AFF minus it |
|---|---|---|
| DTS (describe-then-score, fused with cosine) | 13.06 | +6.38 [+6.15, +6.59] |
| DTS-CF (its matched control; its seed-42 λ is 0 on both halves, so it equals cosine) | 13.25 | +6.19 [+5.96, +6.40] |
| DTS-N (told the true aspect name; the rule's ceiling variant, below DTS and cosine on R@1 here) | 12.15 | +7.29 [+7.03, +7.53] |
| FT-LP / FT-LB / FT-LoRA (fine-tuned CLIP) | 15.45 / 15.32 / 15.80 | +3.99 / +4.11 / +3.64 (each interval lower bound above +3.4) |
| MLLM (Qwen3-VL-8B reranker; seed 52 only, 12,288 episodes) | 14.50 | +4.94 [+4.56, +5.33] |
| PM (the nine other condition-free scorers of `run_baselines.py`; gain -0.04 to +0.19) | 12.99 to 13.25 | +6.19 to +6.44 (lower bounds at least +5.96) |

**Swap success** (both targets reorder with the condition; the condition-free scorers are 0 by construction). On it the privileged DTS-N and the MLLM reranker are above AFF, and R1 is level with it:

| Scorer | Swap success (%) | AFF minus it |
|---|---|---|
| AFF | 11.50 [11.25, 11.76] | |
| R1 | 11.67 | -0.17 [-0.33, -0.01] |
| RCA | 3.17 | +8.33 [+8.05, +8.61] |
| DTS | 3.93 | +7.57 [+7.28, +7.86] |
| DTS-N | 16.43 | -4.93 [-5.29, -4.57] |
| MLLM (seed 52) | 13.08 | -1.61 [-2.19, -1.01] |

DTS parsing failures: one short listing in all 36,864 episodes (seed 54, condition a), no empty phrases; DTS-N has none. Phrases that listed fewer than the basis size K = 8 values: 125 (condition a) and 99 (condition b) in total. Our reading: on R@1 and condition gain no external comparator comes near AFF or the condition-free scorers B and B′(A0). On swap success the privileged DTS-N and the MLLM reranker are above AFF, so the paper's MLLM row needs that line. The DTS gain (+0.08) is not distinguishable from zero and DTS-N's is +1.17, so the phrase route is weak on R@1 here. That fits DTS's seed-42 hit count of 6,347 against AFF's 9,406 (the stop did not fire).

**Diagnostics** (descriptive):

- **Gate shares.** At τ_0, AFF's gate opens on 54.50% of episodes (78.76% for condition a, 30.23% for b) and R1's on 99.99%; at τ_2, 35.27% and 50.01%. AFF's frozen cells use τ_0 on one parity half and τ_2 on the other; R1's use τ_2 on both.
- **Pick accuracy** against the told mapping: 51.78% [51.44, 52.15]. Chance is 33.33% for a uniform pick (R3 rule D14); always picking the image grouping, the told answer in four of the six pair and condition cells, would score 66.67% (style × genre condition a: 12.82%).
- **Redundancy.** Affect is the least redundant grouping in both directions on all three seeds (affect 0.35 to 0.38, image and caption 0.62 to 0.72).
- **Two-way bootstrap.** It widens each interval by a factor of 1.05 to 1.08 when candidate paintings are resampled (the 13 candidates' weights averaged), and by 1.39 to 1.45 when the two target paintings are resampled. Under both, the lower bounds stay above zero for P1 to P7 (the smallest, P4: +0.45 and +0.40), and the S1 and S2 intervals still include zero.
- **Item reuse** is high when the seeds are pooled: over the 36,864 episodes a candidate painting appears about 48 times and an anchor painting about 3.9 times. Per seed (12,288 episodes) it is about 16 and 1.8, below development seed 42's 30 and 2.7, because the held pool has about 1.9 times as many paintings; the three seeds reusing one pool is what raises it, which is why the two-way check was run.

Our reading: none of these changes the verdict. The paper should cite the target-resampled interval, the wider one, beside P3 to P5 (P4: +0.60 [+0.40, +0.79]).

## Deviations

No rule amendment: `DECISION_RULE.md` has a single commit (c394b60) and the log records none. Items below are agent defaults, bugs, routing and take-backs.

- **Agent default (build):** the DTS budget is judged on `dts_first_built.json` while a rerun's setting and GPU fingerprints are unchanged (`DTS_CLOCK_START` pinned to 2026-10-09 12:29); a smoke GPU family without outputs fails unless waived (recorded); smoke DTS failures retry seeds 9002 and 9003; a crashed `--fix 1` cannot be rerun (stricter than rule §8.1); the first read refuses after 2026-10-15; `value_sets.json` must be the file the regression record names; each attempt records git HEAD and status.
- **Agent default (run 1):** seed-42 episodes file made fresh by `prep_episodes_seed42.py`; listing jobs split into nine shards with batch-aligned boundaries and earlier outputs as caches (the splitter later counts only uncached items); the chosen verbaliser reuses the tuning subset's W1 answers.
- **Agent default (runs 2 to 4):** held GPU shards (verbaliser 2,048 episodes, reranker 1,024, listing 3 per seed). The agreement record carries extra provenance fields.
- **Routing and take-back:** on the user's instruction of 2026-10-10 03:03 (Claude usage short), re-derivation phase 2 was routed to Codex; the permission check refused the job, so Claude took it back and a fresh Opus subagent that had not read the implementation ran it (agent default: Opus because its numbers gate the verdict). 249 of 249 items agree; the controller checked its 44 tool calls (no implementation file, no held output before its SHA print).
- **Descriptive-pass stop and fix (agent default, not covered by the rule; for the user to ratify, rule §9 last row).** The first descriptive pass stopped (exit 5, 07:51) on 416 listing keys that had different answers across held seeds: each seed's listing job had only the seed-42 listings as caches, so a phrase new to held was generated once per seed in different batches, and greedy answers differ with batch composition. Seeds 53 and 54 were regenerated with the earlier held listings as caches (log rows 07:53 to 08:10), `external_sources.json` was moved aside to `.failed1` (not deleted) and written again, and the pass finished at 08:29. The verdict was untouched: it rests on `held_pass.json`, written before any DTS input was read. The DTS rows of the descriptive table use the rebuilt listings; the first outputs are kept. The fix restores rule §7.2's one listing per (phrase, K): over the 28 listing folders now used, 31,981 keys have one answer each and every held listing input is covered (the old set had 416 keys with two answers). Which answer a shared phrase keeps was fixed by seed order (52 first) before any DTS number existed. The rebuild changed 632 of seed 53's 5,429 and 750 of seed 54's 5,640 listing answers, so DTS's held rows carry listing batch noise that the anchor bootstrap does not measure; no conclusion depends on it. The six rebuilt shards are held GPU jobs (rule §8.3) and are noted in row H5.
- **Bugs:** none affecting the verdict after `held_verdict.json`. The held-listing defect above touches only descriptive DTS rows, so no reserve read (row H5-R) applies.
- **Agent defaults not listed above** (run handoff §2, run-3 and run-4 handoffs, log 03:56): a smoke-scale DTS failure on all three smoke seeds marks DTS missing; any changed AB file stops the run before `held_started.json` (exit 1); `--after-crash`, `--fix 1`, `--reserve` and the smoke are not time-boxed; the smoke seeds are exempt from the `dts_settings.json` copy check; the smoke listing ran unsplit; FT was shipped with its checkpoints to all three nodes; the phase-2 agent read round 3's `EvalContext` class (in `run_gonogo.py`) for its interface without importing it (rule §8.4 excludes that import).

## Files and storage

- **Outputs** (local, gitignored, `src/test/20261125_artelingo_held_test/results/`; SHA-256 prefixes): `held_pass.json` 93e001572172; `held_verdict.json` 55d88be907c4; `rederive_agreement.json` d73b5855d7ce (comparison in `../rederive/phase2_compare.json`); `descriptive.json` 13bbcb68886b; `sensitivity_held.json` 7443ea25df78; `held_arrays.npz` c3f48c0ed633; `refit_check.json` aebc840fa262; `picks_seed42.json` df1fff781136; `regression_seed42.json` ca168eb628f6; `dts_seed42.json` 07cf3001e368; `smoke_record.json` 5b299985088a; `external_sources.json` 5a634197a54c. Ledger: row H5 (held ledger) and the seed ledger.
- **Pulled cluster outputs:** `res/cluster_jobs/` (102 job folders, 3.1 GB in all; the held FT features are 348 MB of the 371 MB pulled on 2026-10-10).
- **Storage:** listed on `.scratch/pending_deletions.md` for the storage summary at the r6 read wrap-up; nothing deleted (unattended run). Also kept: `external_sources.json.failed1`, `run_r6_descriptive.log.failed1`.
