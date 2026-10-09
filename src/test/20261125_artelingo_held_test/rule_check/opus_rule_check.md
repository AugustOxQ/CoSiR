# Rule check: round 6 decision rule (ArtELingo held test of AFF)

> Reviewer: Opus, fresh context (I did not write the rule or the spec). Finished 2026-10-09 04:25 (Amsterdam, read
> from `TZ=Europe/Amsterdam date`; I did not record the start time). Drafts reviewed: `DECISION_RULE.md` (SHA-256 6b2e5dff3be4aa8d…) and
> `docs/superpowers/specs/2026-10-09-r6-held-test-design.md` (a60339ba008f600d…).
> Read: the handoff §2 and §5, `design/facts.md`, constitution v2, round 3's rule (all of it), round 4's rule §3 and
> §4, plan §8, §10 and §16, the literature check §1, round 5's final review §4, round 4's final review §5, the decision
> rule template, and the code the rule relies on (`src/eval/aspect_episodes.py`, `aspect_quick_checks.py`,
> `aspect_metrics.cluster_bootstrap`, `aspect_scorers.crossfit_lambda`, `aspect_nested`, `run_n6.fit_heads`,
> `run_told_oracle.fit_one_head`, `run_step1.stage_heads`, `run_gonogo.EvalContext`, `r3_bundle.build_bundle`, H3's
> `run_held.py`).
> Not done: I loaded no held row, built no held episode, scored nothing and ran no model. I hashed files and read only
> stored selection-row results. The only file I wrote is this one.

**Counts: 2 blocking, 19 should-fix, 11 nits.** The statistics are right: the integer Holm form, all three secondary
options, the sensitivity formula and its constant 3.532, and the GO / NO-GO / inconclusive mapping. All 40 SHA-256s and
every seed-42 number the rule cites match their sources. The main risks are in the held pipeline (§5): the episode
definition contradicts itself, a split assertion cannot be implemented as written, the refuse-twice runner contradicts
itself, and the smoke comes before the steps it is meant to cover.

## 1. Blocking

### B1. The held episodes are undefined: "the builder unchanged except for the row pool" admits New_Realism (rule §5 items 2 and 3; spec §3.3, D17)

`build_aspect_episodes` (`src/eval/aspect_episodes.py:81-99`) computes eligibility itself, by
`eligible_values(la, groups, pool, 30)` on the pool it is given. On held rows New_Realism has 37 paintings (facts §B),
so the builder "unchanged except for the row pool" makes it eligible: 24 styles, not 23. Item 2 then excludes it. The
builder has no way to do that, so the implementer has to choose how:

- *(i)* intersect `ok_a` and `ok_b` with the development value sets. New_Realism rows stay in the pool and can still
  be drawn as negatives or as the second item of an example pair, exactly as New_Realism's 17 selection paintings were
  in development;
- *(ii)* remove New_Realism rows from the pool. That also changes the emotion × genre episodes, where style is only
  the third aspect.

The two choices give different episodes, and so different hashes and different verdict numbers. The C13 re-deriver
could choose the other one, and "the computation that follows this file" cannot settle a disagreement, because the
file allows both. The episodes are the one irreversible step of the read, so this must be fixed before the commit.

**Replace rule §5 items 2 and 3 with:**

> 2. **Values.** For each aspect pair, the development value sets are `eligible_values(labels[x], groups, pool_sel,
>    30)` for its two aspects, where `pool_sel` is the selection rows with all three labels known (as
>    `build_aspect_episodes` forms its pool). They are written, as label codes with their names, to
>    `results/value_sets.json` before the read and asserted to be 8 emotions, 23 styles and 10 genres. On held rows,
>    each development value must also pass `eligible_values` on the held pool (≥ 30 held paintings; asserted). A value
>    eligible on held rows but outside the development set (New_Realism) is never an anchor value and never an
>    example-pair value. Its rows stay in the pool, so they may be drawn as negatives or as the second item of an
>    example pair, as New_Realism's selection rows were in development.
> 3. **Episodes.** `build_aspect_episodes(labels, groups, held, a, b, 4096, s, third=t, index=index)`, unchanged
>    except for one keyword, `values=(V_a, V_b)`. It replaces `ok_a` and `ok_b` with `sorted(set(ok_a) & set(V_a))`
>    and the same for b, after asserting that `V_a ⊆ ok_a` and `V_b ⊆ ok_b`. It is the only code change, made in an
>    `r6_` copy of the function (the original stays unchanged). It is validated with `validate_aspect_episodes` and
>    passes the identity check of §6 item 4 (the copy, given selection rows and the development sets, reproduces the
>    per-pair `episodes_sha256` of `baselines_seed42.json` and of smoke seeds 9001 to 9003 exactly). Labels are
>    `artelingo_aspect_labels(data)`, groups are `artelingo_splits(data).groups`,
>    `index = PaintingValueIndex(labels, groups)` over all rows, and the pairs and third aspects are
>    `run_baselines.PAIRS`. Seeds 52, 53, 54, in that order.

(If the user wants (ii) instead, the rule must say so in these terms. (i) is the one that matches development.)

### B2. The level of the secondary checks is still open, and so is its detectable-margin constant (rule §4 "[Q1 pending]", §8.2; spec D16, §5)

This is known: the user settles it at approval. Two things must change with it, or the rule stays incomplete after
its commit:

- Delete the two options not chosen, and fill D16's "decided by" and "when".
- §8.2 gives only x = 3.532·SE (the seven pass checks) and x₉₅ = 2.80·SE. It does not say which constant reads a failed
  S1 or S2. **Add to §8.2:** "For S1 and S2: x₉₅ = 2.80·SE under option (a); under (b) or (c),
  x₂ = 3.083·SE (z at one-sided 0.025/2, 2.2414, plus z at 0.8, 0.8416)."
- Note for the brief: option (a) is the literal form of the user's grill answer ("if its lower bound is above 0 the
  paper may say 'also beats it'", handoff §2), and (b) and (c) are stricter. The spec's Q1 should say this, so that
  choosing (b) or (c) is a visible decision and not a quiet tightening.

## 2. Should-fix

### S1. The held-row assertion cannot be implemented: `prepare.npz` has no held rows (rule §5 item 1)

`src/test/20261013_stage_d_selection/cache/prepare.npz` holds `groups`, `split_train`, `scorer_train`, `selection`,
`img_codes`, `txt_codes`, `factor_scale`, `clip_image`, `clip_caption`, `community`, and no held array. H3's
`recompute_split` (`run_held.py:245-265`) asserted those four arrays and took held from `grouped_split`. It then checked
held against `held_rows` of stage (d)'s `src/test/20261014_stage_d_final/cache/held_codes.npz` (`run_held.py:486-488`).
As written, an unattended implementer either stops (§9, "nothing improvised") or improvises. **Replace** "Asserted: the
held row IDs equal those that H3 asserted (…prepare.npz)" **with:**

> Asserted, as H3's `recompute_split` did: `groups`, `split.train`, `scorer_train` and `selection` recomputed by
> `artelingo_splits` equal `prepare.npz`'s `groups`, `split_train`, `scorer_train` and `selection` (SHA-256
> d30f0281bb521d18c7a4d4adca1a7938689e2789db7fd528ce0322b90ac44c8b). `artelingo_splits(data).held` equals the
> `held_rows` array of `src/test/20261014_stage_d_final/cache/held_codes.npz` (only that key is loaded). Held rows
> share no leakage group with train or val.

### S2. The refuse-twice rules contradict each other, and the guard can be deleted (rule §8.1)

"It refuses to start if `results/held_started.json` … exists" excludes the `--after-crash` rerun that the next sentence
allows. "A completed H5 row" is undefined, and the row is half-filled at launch and filled again "after the build".
`results/` is gitignored, so a storage cleanup that deletes it would also remove two of the three guards. A second
crash is not covered. **Replace §8.1 with:**

> 1. **Ledger and refusal.** One runner computes every held quantity that enters §3 or §4. Before launch, the run
>    chat commits and pushes ledger row H5 with the date, the purpose, the runner's SHA-256 and "(pending)" in the
>    episode and report cells. The runner refuses to start if: `results/held_verdict.json` exists; or
>    `results/held_started.json` exists and `--after-crash` is not given; or `--after-crash` is given and
>    `held_started.json` already records two attempts; or row H5 is missing, has a script SHA-256 other than its own,
>    or has anything but "(pending)" in its report cell. It asserts that the SHA-256s of itself and of every `r6_`
>    module it imports equal those in the smoke record of §6 item 5. Its first write, before any held row is loaded,
>    is `results/held_started.json` (attempts: Amsterdam time and SHA-256s) and a copy of it, committed as
>    `held_started.json` in this folder. After a crash before `held_verdict.json` exists, one rerun with
>    `--after-crash` is allowed. If the first attempt recorded episode SHA-256s, the rerun must reproduce them, and it
>    is recorded in row H5. A second crash stops the work until the user decides. The run chat fills the episode cells
>    once the hashes exist, and fills the report cell with `held_verdict.json`'s SHA-256 when it exists.

### S3. The verdict file comes before the re-derivation, so descriptive numbers can be seen before the verdict is confirmed (rule §8.3, §8.4, §10.5; spec §3.6)

§8.3 has the runner write `held_verdict.json` with the verdict, and allows the descriptive pass once that file exists.
§8.4's phase 2 runs "after the held pass and before the verdict is applied". So per-pair, per-seed, DTS, FT, PM and
MLLM numbers can be computed before the re-derivation agrees. If phase 2 then finds a bug, the corrected verdict is
settled with those numbers already known. Round 3's order was: test pass, re-derivation, rule applied and verdict
written, then the descriptive pass. **Replace** "Until `results/held_verdict.json` is written (verdict, …)" **with:**
"The runner writes `results/held_pass.json` (every check's n_j, point, 95% and Holm-level intervals, the Holm order,
each check's pass or fail, this file's SHA-256 and the Amsterdam time). It writes no verdict. After the phase-2
agreement of §8.4, `run_r6_apply_rule.py` writes `results/held_verdict.json` from `held_pass.json`. Until
`held_verdict.json` exists, only these are computed: …". Then change §8.4's "before the verdict is applied" to "before
`held_verdict.json` is written", and the S2 refusal follows. The descriptive script asserts that `held_verdict.json`
exists and records the phase-2 agreement.

### S4. The smoke runs before the steps it must cover (rule §6 items 5 to 7)

Item 5, the smoke, runs "end to end (…, descriptive pass)" before item 6 writes `sensitivity_seed42.json` and before
item 7 creates `dts_seed42.json`. The read needs both: x is read from the first, and the descriptive DTS rows from the
second. So either the smoke cannot run the full path, or code is added after it and the "same script bytes" check
fails. Round 3 also deleted its smoke files after the pass (its §10), which would delete the SHA-256 record that the
read asserts. **Change §6:** move the smoke to the last item, after the sensitivity input and the DTS stop. **Replace
item 5's last sentence with:** "It runs last. The SHA-256s of the runner, of every `r6_` module and of the DTS setting
file are written to `results/smoke_record.json`, which is kept, is outside `results/smoke/`, and is never deleted. Any
change to these files afterwards needs a new smoke."

### S5. The refit tolerance relaxes the user's stop and is not shown in the brief (rule §6 item 2; spec D20, brief)

The user's decision (handoff §2, minor 8) is: "Refit heads must reproduce the stored selection posteriors; if not, the
build stops before the read." The rule also accepts refits within 1e-6 if the arg-max groups agree and item 4 passes.
That is a reasonable engineering allowance (lbfgs under another thread count can move the last bits), but it is an
agent default that loosens a user stop. It is not in the spec's §4 or in "Check these". **Add to the brief's "Check
these":** "If a refit posterior is not bit-identical, it is still accepted when every difference is at most 1e-6, the
arg-max groups are the same and the regression check (C12) then reproduces round 3's numbers exactly with the refit
posteriors. Agent default; the alternative is to stop on any difference." **Replace** "and item 4 then passes with the
refit posteriors in place of the stored ones" **with** "and items 3 and 4, which always use the refit posteriors (the
held path), pass".

### S6. The reuse pointer is wrong: round 5's §4 items are in `r5_` code; round 4's own list is in its final review §5 (rule §6 item 1; spec D19; handoff §2 minor 6)

The "before reuse" items of round 5's final review §4 concern `r5_stats.cell_text`, `r5_diag.chosen_cells`,
`sharper_term`, `r5_bundle.load_ext`, `extend` and `_check_q`. None of them is round 4's code. Round 4's own list
(`src/test/20261122_round4_aff_vetoes/final_review/final_review.md` §5 and N13) includes "D11 hashes of
r3_bundle/r3_common asserted only inside extend_a1. Move before reuse". That is the function B′(A1) would come from.
Also, `r4_bundle.extend_a1` loads the stored step-1 heads and asserts `z["selection"] == ctx.selection`
(round 4 rule D2), so it cannot run on held rows at all. **Replace the last sentence of §6 item 1 with:**

> Code of round 4 or round 5 is reused only as an `r6_` copy (the originals stay read-only). Before the copy is used,
> every "before reuse" item of round 4's final review §5 and N13 and of round 5's final review §4 that concerns the
> copied code is fixed in the copy, and the log lists each item with its fix. B′(A1)'s csd posteriors on held rows
> come from the refit heads of item 2, never from `r4_bundle.extend_a1`'s stored-posterior load.

The spec's D19 and the handoff wording should be corrected the same way. The user's decision ("round 4's code reviewed
before reuse") stands; only the pointer changes.

### S7. "With 'seed' read as 'held seed'" also rewrites the seed-42 references (rule §5 item 5)

R3's D6 (τ from "R1's 24,576 seed-42 top-two margins"; "on seed 42 τ_0 is the minimum margin"), D7 ("recomputes the
six values on seed 42") and D10, D11 ("Seed 42: R@1 …") name seed 42 itself. The blanket substitution turns them into
nonsense, or invites a recomputation on held rows. **Replace** "Everything of R3 rule D1 to D15 applies, with 'seed'
read as 'held seed'" **with:** "R3 rule D1 to D15 apply with these readings: where they speak of a test seed or of the
seed's episodes, read a held seed (52, 53, 54) and its held episodes; every reference to seed 42 stays seed 42 on
selection rows. D7's criterion is not recomputed before the verdict, and its held values are descriptive (§10.5). D12's
bar comparator is chosen per scope over held episodes, for the descriptive bar margin only. D13 is not used."

### S8. n_j needs the bootstrap draws, which `cluster_bootstrap` does not return (rule §3)

`src/eval/aspect_metrics.cluster_bootstrap` returns only the point and `ci95`. The counts n_j need the 5,000 resample
means, so the implementer has to rewrite the loop, and a different generator call gives different n_j. **Replace**
"For check j, n_j = the number of resamples whose mean difference is ≤ 0 (an integer)" **with:**

> For check j, n_j = the number of the 5,000 resample means b_r ≤ 0, where b_r is computed by a copy of
> `cluster_bootstrap`'s loop that returns them: clusters `numpy.unique(groups[anchor], return_inverse=True)` over the
> 36,864 pooled episodes (seeds 52, 53, 54 concatenated in that order); `rng = numpy.random.default_rng(42)`; chunks of
> 250 rows drawn with `rng.integers(0, k, size=(250, k))`; b_r = Σ cluster sums / Σ cluster counts. The copy is
> asserted to give `cluster_bootstrap`'s own `ci95` exactly for every check. Every per-episode difference is a multiple
> of 0.25 (round 3's D8 item 5), so "b_r ≤ 0" is decided on the integer 4·Σ cluster sums. The same draws give the 95%
> and Holm-level intervals (`numpy.percentile` of the b_r at 100·0.025/(8 − k) and 100·(1 − 0.025/(8 − k))).

### S9. A check that fails only because Holm stopped earlier is not "inconclusive at x" on its own evidence (rule §4 NO-GO line; C11)

In the step-down, once the k-th check fails, every later check fails, even one whose own count would pass at its own
level. C11's reading still applies, but the report has to say which kind of failure it is. **Add after "Each failed
check is read on its own (C11)":** "A check that fails only because an earlier check in the Holm order failed (its own
count would pass at its level) is reported as 'not reached: the Holm procedure stopped at <earlier check>', together
with its C11 reading."

### S10. The held context is not pinned (rule §5 item 5)

"No selection mask; its own assertion that every row is held" lets features of non-held rows stay finite, so an
indexing bug that touches a train row would pass silently. The rule also does not say whether each head is fitted once
and then used on both selection and held rows. **Replace the first sentence of §5 item 5 with:**

> As R3 rule §4 item 1, with a held context in place of `EvalContext`: `EvalContext`'s fields with held rows in place
> of selection rows. CLIP features, A3 codes and every posterior are NaN outside held rows and finite on them
> (asserted, the `EvalContext.masked` pattern), and every episode row is a held row (asserted). Each head of §6 item 2
> is fitted once per process. The same fitted object predicts selection rows (the check of item 2) and held rows (the
> read), and a SHA-256 of its coefficients is recorded in both runs and asserted equal. T_N1u is
> `centered_term(..., uniform=True)` on held A3 codes; its centring is per episode (the mean over the 13 candidates and
> over the 8 example items), so no statistic of other held rows enters it.

### S11. RCA's and the pair metrics' scorer-train fits are not stored, so they are refit, not frozen (rule §5 item 5, §11; spec §3.7)

The PCA basis and `PairScaler` are fitted inside `run_baselines.py:126-132` on
`default_rng(0).choice(scorer_train, 60000)` and never saved. RCA enters two pass checks (P2, P7), so its held path is
verdict code. **Add to §11 "Rerun":** "the PCA basis and pair scaler of `run_baselines.py`, refit on the same 60,000
scorer-train rows (`fit_rows_sha256` asserted against `baselines_seed42.json`)". In §5 item 5, write "RCA and PM use
`run_baselines.py`'s scorer-train fits, refit as in §11". In §5 item 4, name the assembly: RCA and PM are
`fused_scores(cos, term, λ)` with λ from the half the parity calls for; B, B0 and B1 are
`nested_scores(cos, T_N1u, T_6u, λ_u, λ_a)` with `crossfit_condition_free`'s picks of the half the parity calls for;
AFF, CF and R1 use round 3's cell assembly with the cells of §5.4.

### S12. The C12 regression checks means where the stored arrays allow an exact check, and never tests the held episode path (rule §6 item 4)

Round 3's `results/seed42_arrays.npz` (SHA-256 5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a) holds
the per-anchor arrays of AFF's fused reader and counterpart and of R1's, plus the gates. `per_anchor_seed42.npz` holds
every pair metric. Two pre-read checks of the held path are possible on selection rows. **Add to §6 item 4:**

> … and exactly: the per-anchor arrays `aff_fused__*`, `aff_cf__*`, `r1_fused__*`, `r1_cf__*` and the gates
> `aff_gate`, `r1_gate` of round 3's `results/seed42_arrays.npz` (SHA-256 5ea4b09a…ab8a); every PM key of
> `per_anchor_seed42.npz`. The episode code of §5 item 3, given selection rows, the development value sets and
> seeds 42, 9001, 9002, 9003 (4,096, 64, 64, 64 per pair), reproduces the per-pair `episodes_sha256` of
> `baselines_seed42.json` and of `results/smoke/baselines_seed900{1,2,3}.json` exactly.

### S13. The GPU jobs read held rows, against "one runner reads held rows", and "raw outputs only" is not enforceable as written (rule §8.1, §8.3)

The verbaliser and CRL listing on held episodes, the LB and LoRA held image features and the MLLM reranker on held
seed 52 all read held images or captions, outside the one runner. "The descriptive script asserts it" cannot check
what a GPU job printed. The episode files also encode the targets by position (candidate columns 0 and 1).
**Replace the GPU sentence of §8.3 with:**

> GPU jobs may run once the held episodes exist: the verbaliser and value listing on held episodes, LB and LoRA held
> image features, and the MLLM reranker on held seed 52. Each is part of H5 and listed in its row. Each receives row
> IDs, images and captions only, never labels, aspect names (except DTS-N) or target columns; the reranker uses its
> recorded candidate permutation. Each writes raw outputs keyed by global row ID, or by episode index and condition,
> and computes and prints no metric. Their outputs are joined to episodes by those keys (C6, asserted) only in the
> descriptive pass.

### S14. The describe-then-score comparator leaves build choices open that could move the stop (rule §7 items 1, 2, 3, 5)

The stop is evaluated without judgment once DTS has a number. What that number is, though, still depends on choices
the rule leaves open:

- *Settings fixed in advance.* Add to item 2: "The four prompt wordings and the two basis sizes (the number of values
  the model is asked to list, e.g. 8 and 16) are written to `results/dts_settings.json` and committed before the first
  seed-42 DTS call. The verbaliser is shown the 4 support pairs and then the 4 contrast pairs, each pair as
  (image, caption) in episode column order. At most 32 new tokens."
- *Projection.* Replace "projected onto it as CRL does and scored by cosine there" with "each item is mapped to the
  vector of its CLIP B/32 cosines with the K value embeddings (image items through the image encoder's features, caption
  items through the text encoder's features, as CRL's similarity vector), and query and candidate are scored by the
  cosine of those vectors".
- *Parsing failures.* "A phrase whose value list cannot be parsed into at least 2 values gives T_DTS = 0 on that row
  (so the fused score is cosine); these rows are counted and reported." "Normalised phrase" = lower-cased, whitespace
  collapsed, trailing punctuation removed.
- *DTS-CF.* Replace "the two-condition mean of T_DTS" with "the mean of z(T_DTS^a) and z(T_DTS^b) per ranking row (as
  R3 rule D9 builds G_cf), fused by `crossfit_lambda` (for a condition-free term its criterion reduces to R@1)".
- *Order and "built".* Add to item 3: "DTS-N runs first, before item 2's tuning (spec §3.8). DTS counts as built when
  DTS-N's gain point is above 0 and the chosen setting has run on all 12,288 episodes."
- *Budget clock.* "One working day" is undefined in an unattended overnight build. Replace it with "24 hours of wall
  time from the first DTS commit (Amsterdam time in the log)".
- *Exact stop.* Item 4 can be decided on integers: AFF's seed-42 fused R@1 is exactly 9,406 / 49,152 (Σ 4·R@1 =
  9,406). "The stop fires if DTS's Σ over the 12,288 episodes of 4·R@1 (`r2_fusion.as_int4`) is greater than 9,406."

### S15. The brief lists two stop conditions; the rule has four (spec brief "Plan", spec §3.5, rule §6, §7, §9)

The brief says: "The build stops before the read only if the new comparator beats AFF's R@1 on seed 42 (19.14), or the
refit classifiers do not reproduce." The rule also stops when the C12 regression fails and when DTS is not built within
its budget, and on any failed assertion. **Replace that sentence with:** "The build stops before the read if the new
comparator beats AFF's R@1 on seed 42 (19.14), if it cannot be built and run on seed 42 within a day, if the refit
classifiers do not reproduce their stored outputs, or if the code fails to reproduce round 3's seed-42 numbers
exactly (C12). Otherwise nothing waits for you until the reports."

### S16. No pre-registered path for a fix found after the verdict; C5's reserve read is not mentioned (rule §9)

§9 says a re-derivation or final-review disagreement is "traced; the computation that follows this file settles it".
After `held_verdict.json` exists, that means scoring held rows again: under C5 that is the reserve read ("one reserve
read for a pre-registered fix after a final-review finding"). **Add a §9 row:** "A bug found after
`held_verdict.json` exists (by the re-derivation, the final review or the report's checks) | The correction is C5's
reserve read: the same held episodes, frozen picks and this file; corrected code only; ledger row H5-R; outputs
`_fix1`, with the originals kept. It is run only after the user has seen the cause, and never to try for a better
number."

### S17. The user's claim decision asks for the style × genre disclosure with design L as the route; the rule does not carry it (rule §4 claim, §10.4; handoff §2 Q6)

**Add to §10.4:** "per-pair results reported beside the pooled claim, with the style × genre margin disclosed (round 3:
−0.580 [−0.793, −0.363] over B′(A0)) and design L named as the route to it".

### S18. The re-derivation's allowed imports do not cover the held path (rule §8.4)

R3 rule §8's list has `EvalContext` (selection only) but not the episode builder, the head fitters or the RCA fit. As
written, the re-deriver either breaks the list or rewrites the builder and lbfgs fits, and a rewritten builder will not
reproduce the episode hashes. **Replace** "it may import the loaders and frozen components R3 rule §8 lists, and the
held row split" **with:** "it may import what R3 rule §8 lists, except `EvalContext`, plus `artelingo_splits`,
`build_aspect_episodes` (with its own implementation of the `values` restriction of §5 item 3),
`validate_aspect_episodes`, `episodes_sha256`, `run_n6.fit_heads`, `run_told_oracle.fit_one_head`,
`run_baselines.py`'s `fit_pca_basis`, `fit_pair_scaler` and its RCA term, and `fused_scores`. It writes its own held
context, frozen-pick assembly, bootstrap counts and Holm."

### S19. The times in the drafts were estimated, not read from the clock (rule header; spec header, §4 "When")

The rule says "Written 2026-10-09 from 04:50" and the spec "Date: 2026-10-09 04:40". D5, D12, D18 and D24 also say
04:40. `TZ=Europe/Amsterdam date` read 04:23 at the end of this review. This is round 5's lapse (an estimated log
time). **Fix:** take every time from `TZ=Europe/Amsterdam date` at commit.

## 3. Nits

- **N1** (§3): The integer form is up to two counts stricter than the percentile interval. At k = 1 the interval
  `numpy.percentile` index is 17.85, so the interval can clear 0 with n = 17 or 18 while the integer form needs
  n ≤ 16. At k = 7, n = 125 can pass the interval and fail the integer form. This is safe, since the integer form
  decides, but the report could show a Holm-level lower bound above 0 on a failed check. Add: "The integer form is
  never laxer than the percentile interval; at a one-count boundary they can disagree, and the integer form
  decides." Say the same in §12's exception row.
- **N2** (§4 option (a)): "passes if its 95% lower bound is above 0 (… passes if 40·(n_j + 1) ≤ 5,001)" names two tests
  that differ at n_j = 125. Write: "passes if 40·(n_j + 1) ≤ 5,001 (the integer form decides; its 95% interval is
  reported)".
- **N3** (§8.5): "Equality" with the threshold cannot happen (40·x is even and 5,001 is odd). Write: "A check whose
  n_(k) is within one count of the largest passing count n*_k = ⌊5,001 / (40·(8 − k))⌋ − 1 (16, 19, 24, 30, 40, 61,
  124 for k = 1 to 7) is reported to the user with the runner's and the re-derivation's counts."
- **N4** (§5 item 3): "every per-pair SHA-256 in `baselines_seed{42,43,45,46,47,48,49,50,51}.json` that exists": seed
  46 has no such file, and seed 44 is left out. Write "in every `baselines_seed*.json` of
  `20261030_aspect_baselines/results/` (non-smoke)". On held rows the check is trivially satisfied (no row is
  shared); keep it anyway as a guard against a pool bug.
- **N5** (§6 item 2): Name the stored keys. `n6_posteriors.npz`'s `affect__*` is E2's affect-km grouping, not D1's
  affect. "Step 1's fitting code" for csd means `run_told_oracle.fit_one_head` on CLIP B/32 features with
  `step1_group_style.npz`'s `style_csd` labels (the keys `style_csd__img` and `style_csd__txt`), not the CSD-feature
  head `style_csd__img_src`.
- **N6** (§6 item 4): "R1's bar margin against its counterpart" is round 3's wording for round-1 R-c's margin against
  the counterpart. Write "R1's margin against its counterpart (R3 rule §5 item 2)", so it is not read as D12's bar
  margin.
- **N7** (brief "Check these" 2; spec D5): "the scheme round 3 checked (+0.585 …)". Round 3's frozen-cell line froze
  only AFF's and the counterpart's cells; B and B′(A0) were re-picked on each test seed. Add "(B and B′ picks were not
  frozen there; freezing them is new)".
- **N8** (spec §3.2): The list of other pair metrics omits bilinear and diag_relu, which rule §2 includes. Align it.
- **N9** (spec §9 budget): The CRL value listing adds one 8B call per distinct normalised phrase and basis size, which
  is not counted. If most phrases are distinct, that can roughly double the held calls (about 74,000 more).
- **N10** (rule): Round 3's rule had a time-box line; this rule has none. Add: "If the read has not started by Thu
  2026-10-15, no read starts; the user decides."
- **N11** (§4, NO-GO): Holm controls the family-wise error, so the checks that pass in a NO-GO are individually valid.
  Say whether the paper may state them ("AFF beats cosine and RCA on new paintings") or only the GO claim. This is the
  user's call; the rule should not leave it to the report.

## 4. Verified as correct

- **SHA-256s.** The four new ones in §0 and §6 item 1 match `sha256sum`:
  - R3 rule 2d311dbe…5dc5925
  - `step1_group_style.npz` b04d96b4…50b7a20e2
  - `step1_heads_style.npz` 898a3701…aaf8df8b
  - `dev_seed42.json` fd7b3f48…a9e01f

  All 36 SHA-256s of R3 rule D1, D2, D10 and D15 (30 table rows, 5 inline, plus `n6_posteriors.npz` 2ad75026…) still
  match. That covers "every input of R3 rule D15". No `*_seed5[2-9]*` file exists in `20261030_aspect_baselines/results/`.
- **Seed-42 numbers.** B 18.341064453125 and B′(A0) 18.436686197916664 match R3 D10 and D11. B′(A1) 18.804931640625
  matches R4 D4 and `dev_seed42.json` `beside_aff/Bprime_A1_mean_r1`. AFF 19.136555989583336 matches
  `dev_seed42.json` `aff/fused_r1` and R3 §5 item 3. These match R3 §5 items 2 and 3 to the last digit:
  - CF 18.39599609375
  - bar margin 0.6998697916666667 [0.4598852740816973, 0.9371680126852968]
  - gain statistic 3.110758463541667 [2.780005709854805, 3.4559584315470384]
  - AFF − R1 0.21769205729166666 [0.06425880757348419, 0.3709597330984391]
  - R1 0.4435221354166667 [0.21646171563312194, 0.6735669710776852] and 2.667236328125 [2.325087836946873,
    3.012361650695922]

  AFF − B′(A1) 0.33162434895833337 [0.048231414333532084, 0.6246158772581268] matches `dev_seed42.json`
  `beside_aff/AFF_minus_Bprime_A1`.
- **Cells.** 39 (τ_0, 4, 16), 119 (τ_2, 0, 16), 149 (τ_2, 4, 4), 10 (τ_0, 0.5, 0.5), 116 (τ_2, 0, 2), 58 (τ_1, 0, 0.5)
  and 123 (τ_2, 0.5, 1) each equal (t·7 + u)·8 + a on NESTED_U and NESTED_A, and match R3 §5 items 2 and 3.
- **λ keys.** In `crossfit_lambda`, `picks[h]` is chosen on tune half h and applied to parity 1 − h, so the §5.4
  mapping of `lambda_picks["0"]` to held parity 1 is right. RCA's seed-42 λ is 0.5 on both halves.
  `crossfit_condition_free` uses the same convention.
- **Prior.** Round 3's frozen-cell line: AFF − B′(A0) 0.5852593315972222 and AFF − R1 0.09019639756944445
  (`results/descriptive.json`). Round 3's cross-fitted margin was +0.591.
- **Holm, integer form.** (n + 1)/5,001 ≤ 0.025/(8 − k) ⟺ 40·(n + 1)·(8 − k) ≤ 5,001 (0.025·5,001 = 125.025).
  The largest passing counts are 16, 19, 24, 30, 40, 61 and 124. The step-down ("it and every earlier check pass") is
  right. Tie order cannot change an outcome: tied counts face a laxer level later in the order. Counting resample
  means ≤ 0 is the conservative one-sided bootstrap p-value.
- **Secondary options.** (a) 40·(n + 1) ≤ 5,001 is one-sided 0.025. (b) is a correct serial gatekeeping: the second
  family is tested only after every primary hypothesis is rejected, with Holm at 3 − k, and it keeps the family-wise
  error at 0.025 across both families. (c) is a correct Holm within the secondary family (no control across families,
  as stated).
- **Sensitivity.** SE² = (σ_a²·Σ M_p² + σ_ε²·N) / N² is the exact variance of the grand mean under round 3's one-way
  random-effects split, with the realised held counts. 3.532 = z(1 − 0.025/7) + z(0.8) = 2.6901 + 0.8416 = 3.5317.
  2.80 = 1.9600 + 0.8416. It uses episode counts only, which are label-derived, never scores.
- **C11 mapping.** NO-GO with every failed point above 0 is reported as inconclusive; any failed point at or below 0 as
  "did not beat". This matches C11 check by check.
- **Fidelity to the handoff §2.** Every item is carried, except where S5, S6 and S17 say otherwise and B2's open
  question. The frozen picks follow plan §8. D5, D12, D18 and D23 are agent defaults and are flagged in the brief.
- **Held safety of the frozen components.** τ, readers, groupings, A3, partitions and the head draws are all fitted on
  selection seed 42 or scorer-train. z-scores are per ranking row and `centered_term` centres per episode, so nothing
  is fitted on held rows, provided S10 and S11 are applied.
- **Template.** The prior, candidates, bar, rules, frozen/rerun, exceptions and glossary sections are all present; the
  extra sections follow round 3's form.

## 5. Re-review of the fix wave

> Same reviewer, finished 2026-10-09 04:45 (Amsterdam, from the clock). Scope: the diffs between the pre-fix copies
> (`scratchpad/predraft/`) and the current `DECISION_RULE.md` (411 lines) and spec. I read the current rule in full
> and the spec diff with its brief. The controller's decisions are taken as settled: B2 = option (b) (the user's
> choice), B1 = keep New_Realism rows in the pool, S5 = exact bitwise refit, and an integer stop at 9,406. As before:
> no held data, no model run, no edit of the two files.

**Counts for this pass: 0 blocking, 2 should-fix, 8 nits.**

Every blocking and should-fix finding of §1 and §2 is resolved as intended. Every nit of §3 is applied, except N2,
which was rightly dropped with option (a). I re-checked these against the files:

- **Gate keys.** `seed42_arrays.npz` has `aff_gate__a`, `aff_gate__b`, `r1_gate__a`, `r1_gate__b`, and the
  `aff_fused__*`, `aff_cf__*`, `r1_fused__*`, `r1_cf__*` keys.
- **Functions.** `src/eval/pair_metric_baselines.py` defines `fit_pca_basis`, `fit_pair_scaler` and `rca_term`, and
  `run_baselines.py` imports them from there. `r2_fusion.as_int4` exists. `run_baselines.py` has a `__main__` guard,
  so importing its `PAIRS` is safe.
- **Smoke files.** `results/smoke/baselines_seed900{1,2,3}.json` hold 64 episodes per pair, with all three pair
  hashes.
- **Style × genre.** The disclosed margin is −0.579833984375 [−0.7932, −0.3632] over B′(A0) (round 3's
  `descriptive.json`).
- **Builder roles.** The description of New_Realism's roles matches `build_aspect_episodes` line by line: anchor
  values and example-pair shared values come only from the restricted `ok` sets, while p_a's or p_b's other-aspect
  value, negatives, example-pair partners and the third aspect are unrestricted.
- **Secondary thresholds.** 61 and 124 are right for Holm with two checks.

**The deviations from my text are sound:**

- **S2** ("before the held context or any held episode is built"). This is more implementable than "before any held
  row is loaded", since `load_artelingo()` loads every row.
- **S4** (a committed `smoke_record.json`; items 2 to 4 rerun if their modules change). This keeps the C12
  regression tied to the read's code.
- **S12** (`aff_gate__*`, `r1_gate__*`). The names are correct.
- **S16** (narrowed to a final-review finding, ledgered as H5-R). This conforms to C5.
- **S18** (the `pair_metric_baselines` functions). This is correct.

**The describe-then-score text (rule §7) is precise and fair, and it leaks no aspect name.** W1 to W4 and the listing
prompt name only paintings, captions, pairs, a respect, a criterion or a property. W2, W3 and W4 describe the episode
exactly: within a support pair, image and caption share the condition's value; a contrast pair shares the other
aspect and differs on this one. The settings are fixed in this file, the ties are ordered, the empty-phrase fallback
gives z(0) = 0 and so plain cosine, and the ceiling runs at both basis sizes before tuning. The spec's brief is plain
and lists every stop before the read. No user decision of handoff §2 is altered; D16's "why" now shows that (b) is
stricter than the grill's literal answer.

### Should-fix

**R1. The refusal blocks the pre-verdict correction that §8.4 and §9 require (rule §8.1; spec §8 risk row 1).** The
runner refuses if `results/held_pass.json` exists, and `--after-crash` applies only "before `held_pass.json`
exists". But a phase-2 disagreement caused by a runner bug must be settled by "the computation that follows this
file" before `held_verdict.json` is written (§8.4, §9 rows 6 and 7, spec risk row: "before the verdict file: correct
the code …, same episodes"). That computation is a new held pass, and the refusal forbids it. An unattended run then
either stops or improvises. **Add to §8.1, after the `--after-crash` bullet:**

> - If `held_pass.json` exists, `held_verdict.json` does not, and the phase-2 re-derivation of §8.4 has traced a
>   disagreement to a bug in the runner, one corrected pass is allowed with `--fix 1`. It needs a new smoke (§6
>   item 7) of the corrected code. It asserts that the held episodes' SHA-256s equal those recorded in H5. It writes
>   `results/held_pass_fix1.json` beside the kept original, and the attempt is recorded in `held_started.json` and in
>   row H5. It refuses if `held_pass_fix1.json` exists. Phase 2 is then repeated against the corrected pass. Anything
>   further goes to the user.

**And in §8.1's refusal list, replace** "`results/held_pass.json` or `results/held_verdict.json` exists" **with**
"`results/held_verdict.json` exists, or `results/held_pass.json` exists and `--fix 1` is not given".

**R2. The DTS stop and the sensitivity input can drift from the code that is smoked and read (rule §6 item 7, second
bullet).** Only items 2 to 4 rerun when their modules change. If the describe-then-score code is corrected after
item 6 evaluated the stop, `dts_seed42.json` (its hit count and λ picks) comes from code that no longer exists. The
user's stop rule would then rest on stale code. The same holds for item 5's sensitivity input. **Replace the bullet
with:**

> - The `r6_` modules that items 2 to 6 ran must have the SHA-256s recorded in those runs. If any changed, those items
>   rerun before the smoke. For item 6, the stop is evaluated again from the rerun; cached verbaliser and listing
>   outputs may be reused only if the modules that produced them are unchanged.

### Nits

- **R3** (rule §5 item 5, third bullet; §8.1): "A SHA-256 of each head's coefficients is recorded … in the read", but
  the head checks run before `held_started.json`, which §8.1 calls the runner's first write. **Write:** "recorded in
  `held_started.json`".
- **R4** (rule §7 item 3, DTS-CF): `crossfit_lambda` → `fused_scores` → `zfuse` z-scores the control's term again, while
  R3 rule D9's G_cf is not z-scored again. **Add** after "(as R3 rule D9 builds G_cf)": "; unlike G_cf, it is then
  z-scored again inside `zfuse`, as every fused term is".
- **R5** (rule §7 items 1 and 2): Normalisation does not strip list markers. A model that numbers its values despite
  the prompt gives "1. joy", which keeps its "1." and weakens the comparator. The image input is unspecified.
  **Add** to item 2's listing bullet: "a leading list marker (`^\s*(\d+[.)]|[-*•])\s*`) is removed before
  normalising". **Add** to item 1: "images are given at the 8B probe's recorded processor settings
  (`src/test/20261106_mllm_probe_8b/`)".
- **R6** (rule §6 item 7, smoke "rule application, descriptive pass"): In selection mode no phase-2 agreement exists
  for `run_r6_apply_rule.py` to require, and the descriptive pass needs GPU outputs for the smoke episodes. **Add:**
  "In smoke mode the apply step accepts only an agreement record marked `smoke`, which the real mode refuses (a unit
  test shows both). The GPU job scripts run on the smoke episodes too (selection rows, 64 per pair), so the smoke
  covers their key joins."
- **R7** (rule §9, reserve-read row): The reserve read needs its own single-use guard. **Add:** "It is run by the
  runner with `--reserve`, after row H5-R is committed. It refuses if `results/held_started_fix1.json` or
  `results/held_verdict_fix1.json` exists, otherwise under §8.1's rules with `_fix1` names."
- **R8** (rule §8.4 imports): The episodes need the labels. **Add** `load_artelingo` and `artelingo_aspect_labels` to
  the list.
- **R9** (rule §4, secondary checks): "the one family-wise 5%" and §3's "family-wise one-sided α = 0.025" name one
  level two ways. **Write** "the one family-wise one-sided 0.025 (two-sided 5%)".
- **R10** (rule §8.1): "If the first attempt recorded episode SHA-256s" does not say where they are recorded.
  **Write:** "the runner appends each seed's per-pair episode SHA-256s to its attempt in `held_started.json` as soon
  as they exist".
