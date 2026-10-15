# Idea 3: GoEmotions placement of captions on AFF, measured on seed 42, tested on fresh seeds if carried (design)

**Date:** 2026-10-07 (Amsterdam). **Status:** the user settled the open points of the handoff
(`docs/superpowers/handoffs/2026-10-07-idea3-goemotions-handoff.md` §5) one at a time in chat between 07:45 and 08:04
and approved both design sections; this file is the written spec. The binding rule will be
`src/test/20261123_idea3_goemotions/DECISION_RULE.md`, written from this spec with round 4's and round 3's rules as
templates, checked by a fresh Opus reviewer and committed before any code. Where the rule and this spec differ, the
rule governs.

## 1. Why this step

Round 4 (report `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md`) ended in a kill: no veto on AFF's gate beat AFF
on seed 42. AFF, round 3's one-sided affect steering frozen as tested (bar margin +0.591 [+0.462, +0.729] against
B′(A0) on the fresh seeds 49 to 51), stays the current best. Vetoes can only redistribute AFF's steering. Idea 3 of the
brainstorm (`docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md` §3.3) is the one remaining idea that changes what
steering buys: place every caption in the Leiden affect grouping by its own GoEmotions probabilities instead of by the
CLIP caption head. The communities carry emotion strongly (pair lift 2.71 on the groups), the heads weakly (1.145), and
the CLIP caption head reaches 35.72% held-out accuracy over the 41 communities.

Decided by the user before this spec (handoff §2, not reopened): round 4's kill stands; idea 3 comes before the
held-split paper test, so both held reads stay available; a cheap measured step on seed 42 comes before any fresh-seed
round; the process of rounds 3 and 4.

Settled for this step (07:45 to 08:04):

1. **Design L:** idea 3 does **not** count as the parked grouping redesign. It changes how captions are placed into the
   existing 41 communities; the communities, the image head, the image and caption groupings and B stay as they are.
   Design L stays parked as the route to a real style signal.
2. **Placement:** a multinomial logistic head from the 28 GoEmotions probabilities to the 41 communities, with the CLIP
   caption head's own fit recipe and only the input swapped (§2).
3. **What changes:** two candidates in one development family: G-T (the steering term only) and G-TF (the steering term
   and the reader's affect features) (§3).
4. **Bar and round:** round 4's carry on seed 42 (D10 against the rebuilt comparators and a positive paired gain over
   AFF), detection AUCs descriptive only; the same rule pre-registers the test on seeds 52 to 54 (§5).
5. **Comparators:** the strongest of B, B′(A0), B′_G and the candidate's own matched counterpart; B′(A1) beside,
   descriptive. The user's open decision on B′(A1) for the paper test stays open.
6. **GoEmotions compute:** once, on the 32,413 selection captions, on the shared local GPU under the lock if free,
   otherwise on CPU. Held captions are not touched.

## 2. The placement

- **GoEmotions on the selection captions.** `src/data/affect.py::goemotions_probabilities` (SamLowe/roberta-base-go_emotions,
  28 sigmoid labels, batch 256, max_length 64) on the captions of the 32,413 selection rows, joined as round 1's
  post-hoc affect script joins them (`join_captions(data.sample_ids[selection], annotations)`), in ascending selection
  order. Computed once, stored in the round folder's cache with its SHA-256; every later step, the re-derivation
  included, reads that file. This applies the affect model to selection captions for the first time (the affect
  factor-learning line applied it only to scorer-train captions); selection rows are development rows.
- **The GoEmotions caption head (GE head).** `LogisticRegression(C=1.0, max_iter=300)` (lbfgs), the recipe of
  `run_told_oracle.py::fit_one_head`, fitted on the same 60,000 scorer-train rows that recipe draws, with the stored
  scorer-train GoEmotions probabilities (`src/test/20261018_affect_factor_learning/cache/affect_prepare.npz`,
  `affect_probs`) as input and the `partition_L` communities as labels. Input: the raw 28 probabilities, no
  normalisation (the communities were built by Euclidean kNN on the raw probabilities; the CLIP heads unit-normalise
  because CLIP features are compared by cosine). If lbfgs stops at its iteration cap, the rule fixes the fallback,
  decided before any episode number.
- **Output.** `predict_proba` on the 32,413 selection GoEmotions rows: a float32 (32,413, 41) posterior in
  `partition_L` label order, NaN outside selection, that replaces `post["affect"]["txt"]` only. The image head's
  posterior, the image and caption groupings, the CSD heads and B are unchanged.
- **Placement quality,** measured before any episode number: the GE head's held-out accuracy on the same 10,000
  check rows as the CLIP heads, beside the CLIP caption head's 35.72% and the image head's 9.81%.

## 3. The candidates

Both keep AFF's frozen A0 half-readers, the 224 cells (4 τ × 7 λ_u × 8 λ_a), z-scoring per ranking row before the gate,
round 2's integer cross-fits (min-margin for the fused reader, max-R@1 for the counterpart) and a matched counterpart
G_cf under the candidate's own gates.

| Candidate | Reader features | Pick, margin, τ, gate | Steering term T^c |
|---|---|---|---|
| **G-T** | AFF's (CLIP placement) | AFF's exactly: π^c, m^c, τ_0 to τ_3, g^c = 1[m^c ≥ τ]·1[π^c = affect] | Σ_h P^c(h)·s_h with s_affect from the GE placement |
| **G-TF** | the six affect columns (S, C, Δ, two spreads, arg-max match) from the GE placement; image and caption columns unchanged | the frozen half-readers on the new features; τ′_0 to τ′_3 = the 0/25/50/75th percentiles of G-TF's own 24,576 seed-42 margins (label-free), frozen for later seeds; g^c = 1[m^c ≥ τ′]·1[π^c = affect] | Σ_h P^c(h)·s_h with G-TF's P^c and s_affect from the GE placement |

s_affect(q, k) is the image-head posterior of the image times the GE posterior of the caption, in both directions. Every
agreement pairs one image with one caption, so the weak image head (9.81%) bounds how much sharper the term can get.

## 4. Seed 42: regression checks, the development step, the measured diagnostics

**One pipeline,** parameterised by the episode seed: round 3's bundle and round 4's gates, development record, carry and
GO checks, reused by import, with this round's placement swapped in by its own modules (prefix `r5_`) without editing
earlier rounds' files. Run on seed 42, before any candidate number, it must pass:

1. the bundle, R1 and AFF reproduce round 3's seed-42 targets exactly (round 3's rule §5 items 1 to 3);
2. GoEmotions rerun on a fixed sample of scorer-train captions reproduces the stored `affect_probs` within a tolerance
   the rule fixes, and the selection caption join gives 32,413 strings;
3. the placement function, fed the unit-normalised CLIP caption features, reproduces the stored CLIP caption-head
   posteriors exactly (same draws, same recipe);
4. with the CLIP caption posterior passed in place of the GE placement, G-T's and G-TF's code paths reproduce AFF
   exactly, τ′ computed from those margins equals AFF's τ_0 to τ_3, and B′_G equals B′(A0).

Any failure stops the work and goes to the user with its cause traced.

**Development.** Each candidate is evaluated against the strongest (largest mean R@1) of B, B′(A0), B′_G (B′(A0)
rebuilt with the GE placement, round 3's condition-free recipe) and its own matched counterpart. Keeping both B′
versions can only raise the bar.

- **Development bar** (round 4's D10): bar margin point at least +0.5, its lower bound above 0, and the gain statistic's
  lower bound above 0.
- **Carry:** among the candidates that clear the bar and have Δ_k > 0 (the integer net rankings over AFF on seed 42's
  49,152 rankings), carry the one with the largest Δ_k; candidates within 24 rankings of the largest are tied, and a
  tie goes to G-T.
- **Kill:** an empty carry set builds no test seed; the seed-42 results go to the user and AFF stays the current best.
- **Beside:** B′(A1) (18.805 on seed 42) beside AFF and each candidate, descriptive.

**Measured diagnostics** (seed 42, descriptive, decide nothing):

- the emotion-detection AUC of G-TF's P(affect), beside R1's 0.787, by the brainstorm's exact definition
  (`bs_07_detector.py`: positives are condition a of emotion × style and emotion × genre, negatives the other four
  (pair, condition) values, 24,576 values pooled);
- the feature-level AUC of Δ_affect, GE placement against CLIP placement, on the same sets;
- the emotion pair lift through the placement (`run_told_oracle.py::pair_stats_heads`), GE against CLIP, after
  reproducing the stored 1.1445 with the CLIP posteriors;
- the sharper term: per pair and condition, fused minus counterpart, condition gain, either change and the chosen λ_a,
  against AFF's (does the sharper term need less weight for its gain?).

## 5. The test on seeds 52, 53 and 54 (only if a candidate is carried)

- **Sensitivity projection,** after the carry and before the build: round 3's formula on the carried candidate's
  seed-42 per-episode differences, for every GO check; logged, never stops the round.
- **Wiring smoke test** on seeds 9001 to 9003, before the build.
- **Build:** seeds 52, 53 and 54 once each, in one invocation, with `run_baselines.py --episodes-seed <s>`; the per-pair
  episode hashes differ from each other and from seeds 42, 43, 45, 47 to 51; a ledger row. The GoEmotions probabilities
  and the GE head do not depend on the seed and are reused.
- **Frozen from seed 42:** the GE head, the A0 half-readers, AFF's τ_0 to τ_3 and G-TF's τ′_0 to τ′_3, the affect
  restriction, the 224-cell family and its tie rules, the heads and the recipes of B, B′(A0) and B′_G. **Rerun on each
  seed's own parity halves:** B, B′(A0), B′_G, the gates, σ*, both cross-fits and the counterpart.
- **GO** if, pooled over the three seeds (one cluster per anchor painting across seeds; 5,000 resamples, seed 42), all
  nine checks have a 95% lower bound above 0: R@1 of the carried candidate's fused reader minus each of cosine, RCA, B,
  B′(A0), B′_G and its matched counterpart; the gain statistic; the condition gain minus RCA's; and **R@1 minus AFF's
  fused reader, paired per anchor.**
- **Reading a failure:** a failed check with a pooled point above 0 is "inconclusive at a detectable margin of x"; at or
  below 0, "the candidate did not beat <comparator>". If only the AFF check fails: "the candidate works, but no
  improvement over AFF was shown", and AFF stays the current best. A GO makes the candidate the method for the
  held-split paper test.
- **Claim licensed:** the candidate beats AFF and each comparator, pooled over the three aspect pairs, on new episodes
  from the same 6,451 selection paintings; not transfer to new paintings, not a margin on each pair.
- **After the verdict** (descriptive): per-seed and per-pair results; AFF's seven checks on seeds 52 to 54 (a
  replication of round 3); B′(A1) beside AFF and the candidate; gate open shares per pair and condition; the
  frozen-cell line.

## 6. Disclosure and stated prior

- **Disclosure.** AFF was selected on seed 42 among about 50 variants, and both candidates are built on it, so their
  seed-42 numbers are inflated; the fresh seeds are the protection. GoEmotions is the affect grouping's own external
  source and is now applied at inference to every caption (an external RoBERTa pass per caption). ArtELingo captions are
  the annotators' explanations of their emotion, so a caption's GoEmotions probabilities are close to its emotion label;
  the method reads the caption, which is the task's own input (query or candidate), and no evaluation label.
- **Prior,** written into the rule before any number: the gain is small at best, because every agreement multiplies
  the sharper caption posterior by the weak image head. G-TF's frozen readers see a distribution shift; its direction
  is not predictable from what we have. A kill would not surprise us.

## 7. Process

Rounds 3 and 4's process, with round 4's lapses (its report §7) closed:

- The rule is written from this spec with round 4's and round 3's rules as templates, checked by a fresh Opus reviewer,
  fixed, committed before any code and sent to the user. A short implementation plan follows, and the user chooses the
  execution method.
- Subagents implement; the main session launches every real run. GoEmotions on the GPU under
  `flock -n -o -E 75 /tmp/gpu0.lock` after checking `nvidia-smi`, or on CPU; everything else on CPU, at most three
  processes.
- **This round writes its own runners** (`r5_` prefix, no name shared with a module of rounds 1 to 4) and imports
  earlier rounds' pure functions. The round-4 final review's before-reuse items that touch imported functions are closed
  in this round's code or tests; the runner-level ones (a guard that covers every candidate computation, input checks
  on every resume path, a KILL line that says it is pending the re-derivation's agreement) are built in from the start.
- **Test-seed code** (build runner, GO runner, rule application, the wiring smoke test and the rule's tests for them)
  is written only if a candidate is carried, and before any test-seed run. The rule pre-registers this, so it is not a
  departure.
- **Independent re-derivation** with its own code: it imports only what the rule lists (and reads the same hashed
  GoEmotions file, spot-checking a sample of it on CPU); phase 1 on seed 42 before any test seed is built, phase 2 on
  the test seeds before the rule is applied. Its results stay out of the shared ledger and the run log until the
  implementation's runner is written.
- No test the rule lists is skipped without asking the user. Every Python call, `--help` included, runs with
  `PYTHONDONTWRITEBYTECODE=1`. Smoke and dry runs print no metric value.
- The whole-branch final review on the most capable model, one fix wave and a scoped re-review, **before** the full
  report `docs/reports/auto/v2/2026-11-23_idea3_goemotions.md` is committed with its `reports_sum.md` row.
- Folder `src/test/20261123_idea3_goemotions/` (folder and report dates are sequence numbers). Commits go to `main`,
  scoped by explicit path, never pushed.
- Target: the seed-42 verdict on 7 October.
