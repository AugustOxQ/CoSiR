# Decision rule: round 6, the ArtELingo held-split paper test of AFF (one read), committed before any code

> Spec: `docs/superpowers/specs/2026-10-09-r6-held-test-design.md` @ f15d42c · Written 2026-10-09 from 04:05
> (Amsterdam) before any implementation script of this round exists; the review's fix wave applied from 04:33, the
> re-review's from 04:46 ·
> Reviewer: a fresh Opus reviewer, report `rule_check/opus_rule_check.md`

**Precedence.** Where this file differs from the spec, from round 3's rule or from the plan, this file governs. Round
3's rule (`src/test/20261121_round3_affect_gate/DECISION_RULE.md`, SHA-256
2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925; "R3 rule" below) supplies the definitions D1 to D15,
the metrics and the bootstrap of its §2, unchanged except where §5 says how they apply to held rows. Statistics,
reading a failure, the regression check and re-derivation follow the constitution (C9, C11, C12, C13) except where
§3 says otherwise. Folder and report dates (`20261125`, `2026-11-25`) are sequence numbers; times are Amsterdam.

**Authorisation.** The user's decisions in the grill of 2026-10-09 (handoff
`docs/superpowers/handoffs/2026-10-09-r6-decide-2-handoff.md` §2; spec §4) and the spec's approval. After its commit
this file changes only with the user's approval. Commits go to `main` and are pushed (`git.md`).

**Status.** One candidate, AFF, read once on ArtELingo's held rows (CLIP ViT-B/32), constitution C5 amendment 2:
held-ledger row H5. Seed 42 is development data for every pick (R3 rule header: AFF was found among about 50 variants
there). Nothing is fitted, picked or tuned on held rows.

## 1. Prior (C8)

Written before any number of this round. AFF's pooled margins should shrink on new paintings, as fresh-seed effects
did before: the margin over B′(A0) about +0.4 R@1 (round 3: +0.591, seed-42 cells +0.585), over B and the matched
counterpart about +0.5, over cosine and RCA about +5, the gain statistic about +3. With about 1.9 times round 3's
anchor paintings we expect 95% half-widths near ±0.10 on the R@1 checks against condition-free scorers and about
±0.13 at Holm's strictest level (our estimate).

So a GO is likely (about 75%); a NO-GO on cosine, RCA or the gain checks would surprise us; a NO-GO on B′(A0) or the
counterpart would not. AFF − B′(A1) about +0.15 (seed 42: +0.332), near a coin flip; AFF − R1 about +0.1 (round 3
with seed-42 cells: +0.090). Style × genre negative again. The describe-then-score comparator below B′(A0) on R@1
with some condition gain (literature check §1: "possible, not likely" to reach AFF).

## 2. Candidates and comparators

| ID | What | Role |
|---|---|---|
| AFF | R3 rule's fused reader (D5, D6, D8), seed-42 cells (§5.4) | candidate |
| CF | AFF's matched counterpart (R3 rule D9), seed-42 cells | matched control, comparator |
| COS, RCA | cosine; RCA fused with cosine at λ frozen (§5.4) | comparators |
| B, B0 | B and B′(A0) (R3 rule D10, D11), seed-42 picks (§5.4) | comparators |
| B1 | B′(A1): `crossfit_condition_free` over the nested cells with T_6u over A1 = (affect, image, caption, csd) (round 4's rule D2, D4), seed-42 picks | secondary comparator |
| R1 | R3 rule's R1 fused reader, seed-42 cells | secondary comparator |
| DTS, DTS-CF, DTS-N | the describe-then-score comparator (§7), its matched control, its true-name ceiling | reported |
| FT-LP, FT-LB, FT-LoRA | the three fine-tuned CLIP baselines of `docs/reports/auto/v2/2026-11-24_clip_lightweight_ft.md`, frozen | reported |
| PM | the other scorers of `run_baselines.py`: diag, diag_relu, bilinear, kissme, xing, wang, probe, tip, value_prototype, each at its seed-42 λ (§5.4) | reported |
| MLLM | the in-context reranker of `src/test/20261106_mllm_probe_8b/`, its recorded setting, held seed 52 only | reported |

## 3. The bar

**Pass checks**, pooled over seeds 52, 53, 54 (R3 rule §2's pooling; 36,864 episodes; clusters = anchor paintings),
each a paired per-anchor difference:

| ID | Quantity |
|---|---|
| P1 | R@1, AFF − COS |
| P2 | R@1, AFF − RCA |
| P3 | R@1, AFF − B |
| P4 | R@1, AFF − B0 |
| P5 | R@1, AFF − CF |
| P6 | gain statistic: condition gain of AFF − that of CF (CF's is 0, asserted); equals the gain difference against COS, B and B0 |
| P7 | condition gain, AFF − RCA |

**Holm across P1 to P7** at a family-wise one-sided α = 0.025 (the two-sided 95% of round 3). One cluster bootstrap
(C9: 5,000 resamples of anchor paintings, seed 42, the same resample indices for every check):

- **Draws.** A copy of `cluster_bootstrap`'s loop (`src/eval/aspect_metrics.py`) that returns the 5,000 resample
  means b_r. Clusters: `numpy.unique(groups[anchor], return_inverse=True)` over the 36,864 pooled episodes (seeds 52,
  53, 54 concatenated in that order). `rng = numpy.random.default_rng(42)`; chunks of 250 rows drawn with
  `rng.integers(0, k, size=(250, k))`; b_r = Σ cluster sums / Σ cluster counts. The copy is asserted to give
  `cluster_bootstrap`'s own `ci95` exactly for every check.
- **Counts.** For check j, n_j = the number of the 5,000 b_r that are ≤ 0, and p_j = (n_j + 1) / 5,001. Every
  per-episode difference is a multiple of 0.25 (R3 rule D8), so "b_r ≤ 0" is decided on the integer 4·Σ cluster sums.
- **Holm.** Order the checks by n_j ascending, ties in the order P1 to P7. The k-th check in this order (k = 1 to 7)
  **passes** if it and every earlier check pass and 40·(n_(k) + 1)·(8 − k) ≤ 5,001, the integer form of
  p ≤ 0.025 / (8 − k).
- **Intervals.** The same draws give each check's 95% interval and its interval at its own Holm level, two-sided
  1 − 0.05 / (8 − k) (`numpy.percentile` of the b_r at 100·0.025/(8 − k) and 100·(1 − 0.025/(8 − k))). The report
  gives both with the point. The integer form is never laxer than the percentile interval; at a one-count boundary they
  can disagree, and the integer form decides.

No minimum size (the user, D7). C10's +0.5 is not used.

## 4. Rules

- **GO** if §6's pre-read checks all passed and all seven pass checks pass (§3).
- **NO-GO** otherwise. Each failed check is read on its own (C11): point above 0, "inconclusive at a detectable margin
  of x" with x from §8.2 and the realised half-width beside it; point at or below 0, "AFF did not beat <name> on new
  paintings" (<name>: cosine, RCA, B, B′(A0), its matched control; "the condition-free scorers on condition gain" for
  P6; "RCA on condition gain" for P7).
  - A check that fails only because an earlier check in the Holm order failed (its own count would pass at its level)
    is reported as "not reached: the Holm procedure stopped at <earlier check>", together with its C11 reading.
  - A NO-GO in which every failed check reads "inconclusive" is reported as **inconclusive**; otherwise as **did not
    beat**.
  - Checks that pass in a NO-GO are reported as passed. No claim is licensed by them.
- **Secondary checks** (never change the verdict): S1 = R@1, AFF − B1; S2 = R@1, AFF − R1; paired, pooled, the same
  bootstrap draws. **Tested only after a GO** (gatekeeping): then Holm across S1 and S2 at one-sided 0.025, exactly as
  §3 with 2 in place of 7: order by n_j ascending, ties S1 first; the k-th passes if it and every earlier one pass and
  40·(n_(k) + 1)·(3 − k) ≤ 5,001. Every "beats" claim of the paper then shares the one family-wise one-sided 0.025
  (two-sided 5%) (the user, spec D16).
  - After a GO, a failed secondary check is read as in the NO-GO line, with <name> = B′(A1) or R1 and x₂ of §8.2 in
    place of x. A passed S1 licenses "AFF also beats B′(A1)"; a passed S2 "AFF also beats R1".
  - After a NO-GO, S1 and S2 are reported (point, 95% interval), not tested and not read.
- **Claim licensed by a GO** (spec §6): AFF beats COS, RCA, B, B′(A0) and its matched control on aspect R@1, and its
  condition gain exceeds theirs and RCA's, pooled over three aspect pairs, on paintings never used to fit, select or
  tune it, with Holm across the seven checks. No per-pair margin, no other dataset or backbone. Disclosures of §10.4
  accompany every AFF number.

## 5. The held pipeline (what is new)

1. **Held rows.** Held = the 20% part of `grouped_split(leakage_groups(...), seed 42)` (`src/data/splits.py`,
   `src/data/artelingo_splits.py`), rows with all three labels known. Asserted, as H3's `recompute_split` did:
   - `groups`, `split.train`, `scorer_train` and `selection` recomputed by `artelingo_splits` equal `prepare.npz`'s
     `groups`, `split_train`, `scorer_train` and `selection` (`src/test/20261013_stage_d_selection/cache/prepare.npz`,
     SHA-256 d30f0281bb521d18c7a4d4adca1a7938689e2789db7fd528ce0322b90ac44c8b);
   - `artelingo_splits(data).held` equals the `held_rows` array of
     `src/test/20261014_stage_d_final/cache/held_codes.npz` (only that key is loaded);
   - held rows share no leakage group with train or val; held ∩ selection = held ∩ scorer-train = ∅; IDs follow C6
     (extraction `sample_ids`, never `range(n)`).
2. **Values.** For each aspect pair, the development value sets are `eligible_values(labels[x], groups, pool_sel, 30)`
   for its two aspects, where `pool_sel` is the selection rows with all three labels known (as
   `build_aspect_episodes` forms its pool). They are written, as label codes with their names, to
   `results/value_sets.json` before the read and asserted to be 8 emotions, 23 styles and 10 genres. On held rows,
   each development value must also pass `eligible_values` on the held pool (≥ 30 held paintings; asserted).
   - A value eligible on held rows but outside the development set (New_Realism) is never the anchor's value on the
     pair's two aspects and never the shared value of an example pair.
   - Its rows stay in the pool. They may fill every other role (a negative; an example-pair item whose shared value
     is on the other aspect; an emotion × genre anchor, where style is the third aspect), as New_Realism's 17
     selection paintings, not eligible there, did in development.
3. **Episodes.** `build_aspect_episodes(labels, groups, held, a, b, 4096, s, third=t, index=index)`, unchanged
   except for one keyword, `values=(V_a, V_b)`. It replaces `ok_a` and `ok_b` with `sorted(set(ok_a) & set(V_a))`
   and the same for b, after asserting that `V_a ⊆ ok_a` and `V_b ⊆ ok_b`. It is the only code change, made in an
   `r6_` copy of the function (the original stays unchanged).
   - The copy is validated with `validate_aspect_episodes` and passes the identity check of §6 item 4 (given
     selection rows and the development sets, it reproduces the stored per-pair `episodes_sha256` of seeds 42, 9001,
     9002 and 9003 exactly).
   - Labels are `artelingo_aspect_labels(data)`, groups are `artelingo_splits(data).groups`,
     `index = PaintingValueIndex(labels, groups)` over all rows, and the pairs and third aspects are
     `run_baselines.PAIRS`. Seeds 52, 53, 54, in that order; every member (anchor, 4 support pairs, 4 contrast pairs,
     13 candidates) a held row.
   - Per pair and seed, the episode SHA-256 is recorded. The nine must differ from each other and from every per-pair
     SHA-256 in every `baselines_seed*.json` of `src/test/20261030_aspect_baselines/results/` (non-smoke); on held
     rows this cannot fail unless the pool is wrong, and it is kept as that guard.
   - Parity halves: by episode index, as `EvalContext` defines them.
4. **Picks, frozen from seed 42.** The pick made on seed-42 tune half h scores the held episodes of parity 1 − h:

   | Scorer | Tune half 0 → held parity 1 | Tune half 1 → held parity 0 | Source |
   |---|---|---|---|
   | AFF | cell 39 (τ_0, λ_u 4, λ_a 16) | cell 119 (τ_2, 0, 16) | R3 rule §5 item 3 |
   | CF | cell 149 (τ_2, 4, 4) | cell 10 (τ_0, 0.5, 0.5) | R3 rule §5 item 3 |
   | R1 fused; R1 counterpart | 116; 58 | 119; 123 | R3 rule §5 item 2 |
   | RCA; each PM | `lambda_picks["0"]` | `lambda_picks["1"]` | `baselines_seed42.json` (the key convention is asserted against `crossfit_lambda`) |
   | B, B0, B1 | recomputed (§6 item 3) | recomputed | `results/picks_seed42.json` |
   | DTS, DTS-CF, DTS-N | §7 | §7 | `results/dts_seed42.json` |

   Assembly: RCA and each PM are `fused_scores(cos, term, λ)` with λ from the half the parity calls for. B, B0 and B1
   are `nested_scores(cos, T_N1u, T_6u, λ_u, λ_a)` with T_6u over their groupings and `crossfit_condition_free`'s
   picks of the half the parity calls for. AFF, CF and R1 use round 3's cell assembly with the cells above. No pick,
   σ*, τ or λ is computed from held episodes. CF and DTS-CF stay condition-free (asserted per cell).
5. **Bundle on held.** As R3 rule §4 item 1, with a held context in place of `EvalContext`: `EvalContext`'s fields
   with held rows in place of selection rows.
   - CLIP features, A3 codes and every posterior are NaN outside held rows and finite on them (asserted, the
     `EvalContext.masked` pattern), and every episode row is a held row (asserted).
   - In the read, each head of §6 item 2 is fitted once; its selection-row posteriors are asserted equal to the stored
     ones again (§6 item 2's check), and the same fitted object then predicts held rows. A SHA-256 of each head's
     coefficients is recorded in §6 item 2's run and in `held_started.json`, and asserted equal. These checks run
     before `held_started.json` is written.
   - Cosine on held CLIP B/32 features. T_N1u is `centered_term(..., uniform=True)` on held A3 codes from the A3
     checkpoint, as R3 rule D10 computes it; its centring is per episode (the mean over the 13 candidates and over
     the 8 example items), so no statistic of other held rows enters it.
   - Posteriors of every grouping (affect, image, caption, affect-km, csd) come from the refit heads, applied to held
     rows' image or caption features. The 18 reader features, the half-reader probabilities, T^c, m^c, π^c and the
     gates are computed exactly as R3 rule D5, D6.
   - RCA and PM use `run_baselines.py`'s scorer-train fits, refit as in §11.
   - R3 rule D1 to D15 apply with these readings: where they speak of a test seed or of the seed's episodes, read a
     held seed (52, 53, 54) and its held episodes; every reference to seed 42 stays seed 42 on selection rows. D7's
     criterion is not recomputed before the verdict, and its held values are descriptive (§10.5). D12's bar comparator
     is chosen per scope over held episodes, for the descriptive bar margin only. D13 is not used.

## 6. Before the read

In this order; each failure stops the work before the read (§9), except item 7's (the smoke), which is fixed and
rerun.

1. **Inputs.** Scripts assert the SHA-256 of this file, of the R3 rule and of every input of R3 rule D15, plus
   `20261116_grouping_step1_style/results/step1_group_style.npz` (b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2),
   `step1_heads_style.npz` (898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b) and
   `20261122_round4_aff_vetoes/results/dev_seed42.json` (fd7b3f480d5997284f9d8cfe29fe25275ff3edd38c1ada0a9392d8396ba9e01f).
   - Code of round 4 or round 5 is reused only as an `r6_` copy (the originals stay read-only). Before the copy is
     used, every "before reuse" item of round 4's final review §5 and N13 and of round 5's final review §4 that
     concerns the copied code is fixed in the copy, and the log lists each item with its fix.
   - B′(A1)'s csd posteriors on held rows come from the refit heads of item 2, never from `r4_bundle.extend_a1`'s
     stored-posterior load.
2. **Head refits.** The affect heads as R3 rule D2 (identity with told_oracle arm L asserted). The image, caption and
   affect-km heads with `run_n6.fit_heads` and the same 60,000-row draw; they must reproduce `n6_posteriors.npz`'s
   `image__img`, `image__txt`, `caption__img`, `caption__txt` and `affect__img`, `affect__txt` (E2's affect-km
   grouping, not D1's affect). The csd heads with `run_told_oracle.fit_one_head` on CLIP B/32 features with
   `step1_group_style.npz`'s `style_csd` labels and step 1's draw; they must reproduce `step1_heads_style.npz`'s
   `style_csd__img` and `style_csd__txt` (not the CSD-feature head `style_csd__img_src`).
   - Every refit posterior on selection rows, cast to float32 as stored, must equal the stored array bit for bit.
     Any difference, however small, stops the work before the read; the differences are logged and the user decides.
   - Items 3 and 4 use the refit posteriors, as the held path does, never the stored ones.
3. **B, B0, B1 picks.** On seed 42 (selection), `crossfit_condition_free` as R3 rule D10, D11 and round 4's D4; the
   picks (λ_u, λ_a) of each tune half and the mean R@1 are written to `results/picks_seed42.json`. The mean R@1 must
   equal 18.341064453125 (B), 18.436686197916664 (B0) and 18.804931640625 (B1) exactly.
4. **Regression (C12).** The held runner's scoring code in selection mode on seed 42, with the frozen picks of §5.4,
   reproduces exactly:
   - AFF fused R@1 19.136555989583336; CF 18.39599609375; bar margin 0.6998697916666667 [0.4598852740816973,
     0.9371680126852968]; gain statistic 3.110758463541667 [2.780005709854805, 3.4559584315470384];
   - AFF − R1 fused R@1 0.21769205729166666 [0.06425880757348419, 0.3709597330984391]; R1's margin against its
     counterpart (R3 rule §5 item 2) 0.4435221354166667 [0.21646171563312194, 0.6735669710776852] and its gain
     statistic 2.667236328125 [2.325087836946873, 3.012361650695922];
   - AFF − B1 0.33162434895833337 [0.048231414333532084, 0.6246158772581268];
   - the per-anchor arrays `aff_fused__*`, `aff_cf__*`, `r1_fused__*`, `r1_cf__*` and the gates `aff_gate__*`,
     `r1_gate__*` of round 3's `results/seed42_arrays.npz` (SHA-256
     5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a); the `cosine__*` and `rca__*` keys and every
     key of the nine PM scorers of §2 in `per_anchor_seed42.npz`;
   - the episodes: the code of §5 item 3, given selection rows, the development value sets and seeds 42, 9001, 9002,
     9003 (4,096, 64, 64, 64 per pair), reproduces the per-pair `episodes_sha256` of `baselines_seed42.json` and of
     `results/smoke/baselines_seed900{1,2,3}.json` (both in `src/test/20261030_aspect_baselines/`).

   A difference is traced; the user decides.
5. **Sensitivity input.** Seed-42 per-episode differences of P1 to P7, S1, S2, split into σ_a² and σ_ε² as R3 rule
   §6.1, written to `results/sensitivity_seed42.json`.
6. **The describe-then-score stop (§7).**
7. **Smoke.** It runs last. The same script bytes run end to end (episodes, bundle, scoring, Holm, `held_pass.json`,
   rule application, descriptive pass) in selection mode on smoke seeds 9001 to 9003 (64 episodes per pair, R3 rule
   §10), printing no metric. One wiring mutation (AFF's gated term where CF expects G_cf) must fire an assertion.
   - In smoke mode the apply step accepts only an agreement record marked `smoke`, which the real mode refuses (a unit
     test shows both). The GPU job scripts run on the smoke episodes too (selection rows, 64 per pair), so the smoke
     covers their key joins.
   - The SHA-256s of the runner, of every `r6_` module and of the DTS settings file are written to
     `results/smoke_record.json`, which is kept, is outside `results/smoke/`, and is never deleted; a copy is
     committed as `smoke_record.json` in this folder. Any change to these files afterwards needs a new smoke.
   - The `r6_` modules that items 2 to 6 ran must have the SHA-256s recorded in those runs. If any changed, those
     items rerun before the smoke. For item 6, the stop is evaluated again from the rerun; cached verbaliser and
     listing outputs may be reused only if the modules that produced them are unchanged.

## 7. The describe-then-score comparator and the stop

Built from the literature check's §1 (`docs/reports/literature/2026-11-25_held_claim_check.md`). Every setting below
is fixed by this file. `results/dts_settings.json` copies them (wordings, basis sizes, listing prompt, token limit,
model revision) and is committed before the first seed-42 call.

1. **Verbaliser.** Qwen3-VL-8B-Instruct (revision recorded), greedy, at most 32 new tokens, one call per episode and
   condition. The message shows "Group A": the 4 support pairs, then "Group B": the 4 contrast pairs, each pair as
   (image, caption) in episode column order, then one wording below. Condition a's supports are `pairs_a_*` and its
   contrasts `pairs_b_*`; condition b swaps them. The phrase is the first line of the answer, normalised: lower-cased,
   whitespace collapsed, leading and trailing punctuation removed. Images are given at the 8B probe's recorded
   processor settings (`src/test/20261106_mllm_probe_8b/`).

   | ID | Wording |
   |---|---|
   | W1 | "Each pair is a painting and a caption. What do the pairs in Group A share that the pairs in Group B lack? Answer with a short phrase." |
   | W2 | "In each pair of Group A, the painting and the caption match in one respect. In each pair of Group B, they match in a different respect. In a short phrase, name the respect in which the pairs of Group A match." |
   | W3 | "The pairs in Group A count as similar; the pairs in Group B do not count as similar under the same criterion. In a short phrase, state the criterion." |
   | W4 | "In at most five words, name the property that links the two items of each pair in Group A but not those in Group B." |

2. **Scorer (CRL; Liu et al., NeurIPS 2025).** Basis size K ∈ {8, 16}.
   - Listing: the same model, text only, greedy, at most 128 new tokens, is asked: "List K distinct values of the
     following criterion for describing paintings: <phrase>. One value per line, no numbering." (K written as a
     number). A leading list marker (`^\s*(\d+[.)]|[-*•])\s*`) is removed before normalising; lines are then
     normalised as phrases are; empty and repeated lines are dropped; the first K are kept. One
     listing per (normalised phrase, K), cached.
   - Parsing failure: a phrase that is empty, or whose listing gives fewer than 2 values, gives T_DTS^c = 0 on that
     episode and condition (the fused score is then cosine). These are counted and reported.
   - Projection: each value is embedded as written by the CLIP B/32 text encoder of the feature cache and
     unit-normalised; the K embeddings form the basis. Each item is mapped to the vector of its cosines with the K
     value embeddings (image items through the cached image features, caption items through the cached text features;
     CRL's R = I·Tᵀ on unit vectors). Query and candidate are scored by the cosine of those vectors: T_DTS^c.
3. **Fusion and controls.** T_DTS^c is fused with cosine by `fused_scores` (`zfuse`), λ by `crossfit_lambda`
   (`LAMBDA_GRID` with its edge extension), as every external baseline.
   - DTS-CF: the mean of z(T_DTS^a) and z(T_DTS^b) per ranking row, averaged in float64 then cast to float32 (as R3
     rule D9 builds G_cf; unlike G_cf, it is then z-scored again inside `zfuse`, as every fused term is), equal in
     both conditions (asserted), fused by `crossfit_lambda` (for a condition-free term its criterion reduces to
     R@1).
   - DTS-N: the phrase replaced by the target aspect's name ("emotion", "style", "genre"; privileged), then listed and
     scored as DTS.
4. **Order on seed 42** (selection rows only).
   - First DTS-N, at K = 8 and K = 16, on all 12,288 episodes. Sanity: its pooled condition gain point (cross-fitted
     fused) must be above 0 at each K; otherwise the pipeline is debugged within the budget of item 7.
   - Then the tuning: the 8 settings (W1 to W4 × K 8, 16), each run on the first 1,024 episodes of each pair and
     scored by (R@1 + condition gain) / 2 of its cross-fitted fused scores on those 3,072 episodes. Ties go to the
     first setting in the order W1 K8, W1 K16, W2 K8, …, W4 K16.
   - The chosen setting runs on all 12,288 episodes. Its λ picks per half, DTS-CF's and DTS-N's (at the chosen K),
     the wordings, the model revision, the parsing-failure counts and every setting's score go to
     `results/dts_seed42.json`. On held episodes DTS, DTS-CF and DTS-N use the chosen setting and these picks.
5. **Built.** DTS counts as built when item 4's sanity has passed and the chosen setting has run on all 12,288
   seed-42 episodes.
6. **The stop.** DTS's hit count is the sum over the 12,288 seed-42 episodes, pooled over the three pairs, of 4·R@1
   of its cross-fitted fused scores (`r2_fusion.as_int4`). AFF's is 9,406 (9,406 / 49,152 = 19.136555989583336%).
   If DTS's hit count is greater than 9,406, the build stops before the read and the user decides (promote it to a
   pass check, or rethink). Only DTS counts, not DTS-N or DTS-CF.
7. **Budget.** 24 hours of wall time from the first DTS commit (Amsterdam time in the log). If DTS is not built
   within it, the stop cannot be evaluated and the read waits for the user.

## 8. The read

1. **Ledger and refusal.** One runner computes every held quantity that enters §3 or §4.
   - Before launch, the run chat commits and pushes ledger row H5 with the date, the purpose, the runner's SHA-256
     and "(pending)" in the episode and report cells.
   - The runner refuses to start if: `results/held_verdict.json` exists, or `results/held_pass.json` exists and
     `--fix 1` is not given; or `results/held_started.json` exists and neither `--after-crash` nor `--fix 1` is
     given; or `--after-crash` is given and `held_started.json` already records two attempts; or row H5 is missing,
     has a script SHA-256 other than its own, or has anything but "(pending)" in its report cell.
   - It asserts that the SHA-256s of itself and of every `r6_` module it imports equal those in the latest smoke
     record of §6 item 7. Its first write, before the held context or any held episode is built, is
     `results/held_started.json` (attempts: Amsterdam time and SHA-256s) and a copy of it, committed as
     `held_started.json` in this folder.
   - The runner appends each seed's per-pair episode SHA-256s to its attempt in `held_started.json` as soon as they
     exist.
   - After a crash before `held_pass.json` exists, one rerun with `--after-crash` is allowed. If the first attempt
     recorded episode SHA-256s, the rerun must reproduce them, and it is recorded in row H5. A second crash stops the
     work until the user decides.
   - If `held_pass.json` exists, `held_verdict.json` does not, and the phase-2 re-derivation of §8.4 has traced a
     disagreement to a bug in the runner, one corrected pass is allowed with `--fix 1`. It needs a new smoke (§6
     item 7) of the corrected code, recorded as `smoke_record_fix1.json` beside the original, and the run chat first
     adds the corrected runner's SHA-256 to row H5 (the refusal checks the latest one).
   - The `--fix 1` pass asserts that the held episodes' SHA-256s equal those recorded in H5. It writes
     `results/held_pass_fix1.json` beside the kept original, and the attempt is recorded in `held_started.json` and in
     row H5. It refuses if `held_pass_fix1.json` exists. Phase 2 is then repeated against the corrected pass, and
     `held_verdict.json` is written from it. Anything further goes to the user.
   - The run chat fills the episode cells once the hashes exist, and fills the report cell with
     `held_verdict.json`'s SHA-256 when it exists.
2. **Sensitivity.** After the held episodes are built and before any held score: per check, SE² = (σ_a²·Σ_p M_p² +
   σ_ε²·N) / N², with M_p the pooled episode count of anchor painting p over the three seeds and N = 36,864, σ² from
   §6 item 5. For P1 to P7: x = 3.532·SE (z at one-sided 0.025/7 plus z at 0.8; the detectable margin at Holm's
   strictest level) and x₉₅ = 2.80·SE. For S1 and S2 (read only after a GO, §4): x₂ = 3.083·SE (z at one-sided
   0.025/2, 2.2414, plus z at 0.8, 0.8416), with x₉₅ beside it. All are written to `results/sensitivity_held.json` and
   used only to read a failed check.
3. **Order of computation.** The runner writes `results/held_pass.json`: every check's n_j, point, 95% and Holm-level
   intervals, the Holm order, each pass check's pass or fail, the SHA-256 of this file and the Amsterdam time. It
   writes no verdict. After the phase-2 agreement of §8.4, `run_r6_apply_rule.py` writes `results/held_verdict.json`
   from `held_pass.json`.
   - Until `held_verdict.json` exists, only these are computed: the held bundles; the per-anchor arrays of AFF, CF,
     COS, RCA, B, B0, B1 and R1 fused; the pooled P1 to P7, S1, S2. No metric of DTS, FT, PM or MLLM and no per-seed
     or per-pair number is computed before it. The descriptive script asserts that `held_verdict.json` exists and
     records the phase-2 agreement.
   - GPU jobs may run once the held episodes exist: the verbaliser and value listing on held episodes, LB and LoRA
     held image features, and the MLLM reranker on held seed 52. Each is part of H5 and listed in its row. Each
     receives row IDs, images and captions only, never labels, aspect names (except DTS-N) or target columns; the
     reranker uses its recorded candidate permutation.
   - Each GPU job writes raw outputs keyed by global row ID, or by episode index and condition, and computes and
     prints no metric. Their outputs are joined to episodes by those keys (C6, asserted) only in the descriptive pass.
4. **Re-derivation (C13).** An agent that has not written or read the implementation re-derives with its own code.
   - Imports allowed: what R3 rule §8 lists, except `EvalContext`, plus `load_artelingo`,
     `artelingo_aspect_labels`, `artelingo_splits`, `eligible_values`,
     `PaintingValueIndex`, `build_aspect_episodes` (with its own implementation of the `values` restriction of §5 item
     3), `validate_aspect_episodes`, `episodes_sha256`, `run_n6.fit_heads`, `run_told_oracle.fit_one_head`,
     `fit_pca_basis`, `fit_pair_scaler` and `rca_term` of `src/eval/pair_metric_baselines.py` (as `run_baselines.py`
     calls them), and `fused_scores`. It writes its own held context, frozen-pick assembly, bootstrap counts and Holm.
   - Phase 1, before the read: §6 items 3 and 4, the Holm counts on seed 42, the sensitivity inputs. Phase 2, after
     the held pass and before `held_verdict.json` is written: episode hashes, the frozen-pick assembly, P1 to P7, S1,
     S2, the n_j, the Holm order and every pass or fail.
   - Agreement as R3 rule §8: discrete quantities identical; points and bounds within 1e-9 R@1 points; per-anchor
     arrays exact. A disagreement is traced before `held_verdict.json` is written; the computation that follows this
     file settles it, and the user is told.
5. **Boundaries.** A check whose n_(k) is within one count of the largest passing count
   n*_k = ⌊5,001 / (40·(8 − k))⌋ − 1 (16, 19, 24, 30, 40, 61, 124 for k = 1 to 7; for S1 and S2 after a GO, 61 and
   124) is reported to the user with the runner's and the re-derivation's counts. The integer rule still decides.

## 9. Outcome-to-action

| Outcome | Action |
|---|---|
| A SHA-256, split, ID, eligibility, hash-distinctness, held-mask or condition-free assertion fails | stop that step; report; nothing improvised |
| §6 item 2 (refit, any difference) or item 4 (C12) fails | stop before the read; cause traced; the user decides |
| §7 stop, or DTS not built within its budget | stop before the read; the user decides |
| All seven pass | GO within §4's claim; secondary checks tested and reported with their outcome |
| Any of the seven fails | NO-GO, reported as inconclusive or did not beat (§4); every read is reported in the paper (C5) |
| Re-derivation disagrees beyond §8.4 before `held_verdict.json` is written | traced; the computation that follows this file settles it (a runner bug: one corrected pass with `--fix 1`, §8.1); the user is told |
| A crash or a bug before `held_verdict.json` is written | code corrected to match this file is not a rule change; a crashed run restarts under §8.1 (`--after-crash`), a runner bug gets at most one corrected pass (`--fix 1`, §8.1); a written results file is never overwritten (new files `_fix1`, both kept, logged); the held episodes are never rebuilt with other seeds |
| A bug found after `held_verdict.json` exists (by the final review, or by the report's checks and confirmed by the final review) | The correction is C5's reserve read: the same held episodes, frozen picks and this file; corrected code only; ledger row H5-R; outputs with the suffix `_reserve`, with the originals kept and reported beside the corrected verdict. It is run by the runner with `--reserve`, after row H5-R is committed. It refuses if `results/held_started_reserve.json` or `results/held_verdict_reserve.json` exists, otherwise under §8.1's rules with `_reserve` names. It is run only after the user has seen the cause, and never to try for a better number. One reserve read only; anything after it goes to the user |
| The read has not started by Thu 2026-10-15 | no read starts; the user decides |
| Anything this file does not cover | the user decides |

## 10. Process

1. **Folder and names.** `src/test/20261125_artelingo_held_test/`; modules `r6_*` (`run_r6_*`, `test_r6_*`); earlier
   round folders read-only, imported by path; results in this folder's `results/` (gitignored); smoke files in
   `results/smoke/`, never results.
2. **Environment.** Every Python call with `PYTHONDONTWRITEBYTECODE=1`; CPU work at `OMP_NUM_THREADS=8
   MKL_NUM_THREADS=8`, at most three processes, `uptime` and `free -g` checked first; tests on the real data shape;
   mutation tests on copies; times from `TZ=Europe/Amsterdam date`. GPU work on DAS6 through the cluster CLI (node401,
   node402, node408; `cluster-selftest gpu` on each first); the local GPU only under its lock. The main session
   launches every real run.
3. **Records.** The log `20261125_artelingo_held_test_log.md`; the auto report
   `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md` with its index row; the user-read report; the held ledger
   row H5; the seed ledger notes seeds 52 to 54 as held seeds of H5.
4. **Disclosures** (with every AFF number): AFF was found among about 50 label-free variants on seed 42, and rounds 4
   and 5 read about 10 more there; DTS was tuned with at most 8 settings on seed 42; the held paintings were read with
   value episodes in H1 to H3; 24 held rows have a training image at CLIP cosine ≥ 0.99 (63 at ≥ 0.98); one A3
   checkpoint and one reader pair (the plan's three training seeds do not apply to frozen AFF); later designs (design
   L) are made after this read. Per-pair results are reported beside the pooled claim, with the style × genre margin
   disclosed (round 3: −0.580 [−0.793, −0.363] over B′(A0)) and design L named as the route to it.
5. **After the verdict** (descriptive, decides nothing): per seed and per pair; the bar margin (R3 rule D12) and AFF −
   B1 per pair; R1's own seven checks; DTS, DTS-CF, DTS-N, FT, PM and MLLM rows with AFF minus each, and DTS's
   parsing-failure counts; the two-way (anchor × candidate) bootstrap with the item-reuse rate; swap success; gate
   shares, pick accuracy and redundancy (R3 rule §7 items 4 to 6).
6. **Time-box.** If the read has not started by Thu 2026-10-15, no read starts; the user decides.

## 11. Frozen and rerun

- **Frozen:** groupings, readers, τ, the affect restriction, the 224-cell family, the A3 checkpoint (R3 rule D1 to
  D15); every pick of §5.4; DTS's setting and λ (§7); the fine-tune checkpoints.
- **Rerun:** the head fits (same draw, §6 item 2); the PCA basis and pair scaler of `run_baselines.py`, refit on the
  same 60,000 scorer-train rows (`fit_rows_sha256` asserted against `baselines_seed42.json`; the fit checked by §6
  item 4's PM and RCA arrays); B, B0 and B1's seed-42 picks (§6 item 3); everything scored on held rows.

## 12. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| C9 | Holm across the seven pass checks, decided by the integer count form of §3, instead of each 95% lower bound (the integer form is never laxer than the percentile interval; at a one-count boundary they can disagree, and the integer form decides) | plan §10's multiplicity rule for paper claims; the user's choice (spec D7) | each at 95%: uncorrected across seven claims |

## 13. ID glossary

| ID | Meaning |
|---|---|
| AFF, CF, R1, B, B0, B1 | §2 |
| COS, RCA, PM, FT, MLLM | §2 |
| DTS, DTS-CF, DTS-N | the describe-then-score comparator, its matched control, its true-name ceiling (§7) |
| W1 to W4; K | the verbaliser's four wordings; the basis size, 8 or 16 (§7) |
| P1 to P7; S1, S2 | pass checks (§3); secondary checks (§4) |
| n_j, p_j | the number of the 5,000 bootstrap resample means ≤ 0, and (n_j + 1) / 5,001 (§3) |
| x, x₉₅, x₂ | detectable margins (§8.2) |
| H5, H5-R | this read's held-ledger row; its reserve-read row, used only under §9 |
| R3 rule | `src/test/20261121_round3_affect_gate/DECISION_RULE.md` |
