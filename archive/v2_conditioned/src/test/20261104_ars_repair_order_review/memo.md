# Which repair to run first after the E3 NO-GO: a decision memo for method A′ (CoSiR v2)

Date: 2026-10-03. Status: a proposal for review. Nothing in Sections 5 to 7 has been run. Every number in Sections 2 and
3 comes from the E3 report (Appendix A), whose final review re-derived it from stored per-anchor arrays.

## 0. The decision requested

E3, the pre-registered go/no-go of method A, ended NO-GO on 2026-10-03 (Appendix A). The user chose to attempt a method
repair, called A′, before the project falls back to an analysis paper (branch 3). A′ must pass the same pre-registered
GO test that A failed, on fresh selection episodes, under a new pre-registration written before any A′ run is scored
on them. Four candidate steps are on the table: H1, a nested test-time score; H2, a repair of the training fit; H3, a
label-trained learnability diagnostic; and H4, their combination (Section 4).

The time budget is about six working days. The branch decision was planned for Friday Oct 9, and every day the repair
runs past about Oct 12 comes out of the paper schedule (CVPR abstract Nov 10, paper Nov 16).

The question for this review is **which step to run first, and under which decision rules**, so that:

1. the first step yields the most decision-relevant information per day;
2. no step spends the fresh test episodes or the held rows, or weakens the later pre-registered GO test;
3. a negative outcome stops the repair early rather than late.

**The user leans toward running H3 first.** Section 6 compares that order with three alternatives, Section 8 lists the
weaknesses we already see, and Section 9 lists the questions for the panel.

## 1. Task, method and GO test in brief

**Task** (spec §3, Appendix C). A query (an image or a caption) and a condition c = (S, C) are given. S holds 4
cross-item *support pairs* (an image of one painting with a caption of another) that agree on the wanted *aspect*, and
C holds 4 *contrast pairs* that agree on another aspect. The 13 candidates are in the other modality. One candidate,
p_A, shares the anchor's value on aspect A, one, p_B, shares its value on aspect B, and 11 negatives share neither.
The condition never shows the anchor's value (value-disjoint), and swapping S and C must flip the target. On ArtELingo
the aspects are emotion (8 values), style (23) and genre (10), and the three aspect pairs are pooled.

**Metrics.** *R@1* is the share of rankings whose target ranks strictly first (ties are misses), averaged over two
conditions and two directions. The *other-aspect rate* is how often the other aspect's candidate ranks first.
*Condition gain* is R@1 minus the other-aspect rate; it is exactly 0 for any scorer that ignores the condition. The
*either rate* is R@1 plus the other-aspect rate, so **R@1 = (either rate + gain) / 2**.

**Method A.** Two small encoders map frozen CLIP ViT-B/32 features into 32 non-negative factors per modality. At test
time the training-free *agreement rule* sets w = ReLU(mean over S of a_I ⊙ a_T minus mean over C of a_I ⊙ a_T),
L1-normalised, and the *factor term* is Σ_l w_l q_l c_l. The term is fused with the cosine by per-episode z-scores,
z(cos) + λ·z(term), with λ *cross-fitted*: picked on one parity half of the anchors by the mean of R@1 and gain and
applied to the other half. The *uniform-weight control* uses w = 1/L, which removes the condition. Training uses the C0
factor recipe plus a *pseudo-aspect episode loss* on banks of 65,536 episodes built from three label-free k-means
*pseudo-partitions* (64 clusters each) of the 183,694 scorer-train rows: GoEmotions affect (distant supervision), CLIP
image and CLIP caption.

**GO** (spec §6). On fresh episodes, the picked run beats each of backbone-only cosine, the GO bar (RCA, the best
raw-feature metric-from-pairs baseline, fixed on seed 42 in E1) and its own uniform control, on both R@1 and gain:
the painting-clustered 95% lower bound of the paired difference is above 0 (5,000 resamples, seed 42). Strong GO adds
+4 points on both over the cosine.

**Rows.** Training uses scorer-train rows only. Development episodes use the 32,413 selection rows (6,451 paintings).
Val and held rows are never read. Scorer-train and selection rows come from a grouped split and share no painting.

## 2. What E3 established

| Quantity (seed-43 test episodes unless stated) | Value |
|---|---|
| Backbone-only (CLIP cosine) R@1 | 13.53 (seed 42: 12.96) |
| GO bar RCA, R@1 / gain | 13.52 / −0.06 |
| Picked A3 (λ_aspect 3), cross-fitted, R@1 / gain | 13.76 / 0.26 [−0.04, 0.56] |
| A3 uniform-weight control R@1 (gain 0 by construction) | 16.72 |
| Either rate: cosine / A3 uniform term / A3 cross-fitted / A3 weighted term alone | 27.05 / 33.44 / 27.25 / 21.37 |
| A3 weighted term alone (fixed λ = ∞, post-hoc), gain | 0.97 [0.53, 1.41] (seed 42: 0.99 [0.56, 1.45]) |
| A3 at fixed λ = 8 (fused, post-hoc), gain | 1.14 [0.71, 1.58] |
| Term-only paired gain, A3 minus SE / minus C0 | +0.91 [0.38, 1.47] / +0.80 [0.25, 1.37] |
| Pseudo-aspect loss, last 10 logs vs constant-score value (3.258; 2.565 without swap) | 1.0% to 4.4% below, every run |
| Learned temperature τ (A3) | 0.031 → 0.082, rising in every logged interval |
| Fresh pseudo-aspect episodes (training rows), A3 gain: agreement rule / training score (0.3·cos + term) | 0.36 [−0.12, 0.86] / 0.95 [0.28, 1.62] |
| Winner's curse: A3 gain on the seed-42 pick episodes vs seed 43 | 0.52 → 0.26 |
| K8 (H1 run, no genre partition) | +0.10 [−0.21, 0.40], fails |
| Label-supervised probe reference (aspect-episode spike, diagnostic) | about 23 R@1 |
| AMI of each pseudo-partition with the evaluation labels (E2, scorer-train rows) | affect–emotion 0.20; image–style 0.32, image–genre 0.40; caption–genre 0.16; caption–emotion 0.06, caption–style 0.06 |

The E3 report reads two limits from these numbers (both post-hoc, outside the decision map):

- **Selection costs aspect-finding.** The uniform factor term finds an aspect-sharing candidate more often than the
  cosine (33.44 against 27.05), but the agreement-weighted term, which does select the conditioned aspect, finds one
  less often (21.37). The pre-registered pick criterion, mean(R@1, gain) = either/4 + 3·gain/4, therefore kept the
  cross-fitted λ at 0.25 to 0.5, and the fused gain fell to 0.26.
- **The fit was weak.** The aspect loss barely moved from its constant-score value; the little that was learned seems
  to have transferred (training-task gain about 0.95, labelled-aspect term-only gain about 1.0).

The untested alternative the report names is a *task failure*: the k-means pseudo-partitions may not be learnable as
conditional aspects by this basis at all.

## 3. The GO arithmetic a repair must respect

To beat the uniform control's R@1 of 16.72 on the seed-43 draw:

- at A3's cross-fitted either rate (27.25), a method needs a gain above 2 × 16.72 − 27.25 ≈ 6.2 points;
- with the weighted term alone (21.37), it needs about 12.1;
- a score that keeps the uniform term's either rate (33.44) beats 16.72 with any reliable positive gain, since then
  R@1 = 16.72 + gain / 2.

A fit repair alone (H2) that raises the gain while the either rate stays where it is must therefore reach a gain of 6
to 12 points. A nested score (H1) needs much less, but the margin is still thin. Against the control on R@1 it is about
gain/2 minus half of any lost either rate. In E3 the paired interval of A3 against its control had a half-width of
about 0.33 R@1 points, and the gain intervals had half-widths of about 0.30 (fused) to 0.44 (term alone). A nested
score that keeps the whole either rate would therefore need a fused gain of roughly 0.7 to 1.0 points or more on the
fresh draw to clear both bounds, and any loss of either rate raises that. A3's best measured gains (0.97 term-only,
1.14 at λ = 8) sit at that edge, and seed-42 gains halved on the fresh draw in E3.

## 4. The repair hypotheses and their evidence

**H1. Nested test-time score** (most direct; untested). s = z(cos) + λ_u·z(T_u) + λ_a·z(T_a), with T_u the uniform
factor term, T_a the agreement-weighted term and both λs cross-fitted on a small 2-D grid.
- *For:* T_u finds aspect-sharing candidates (33.44), T_a selects the conditioned one (about +1 point over SE and C0 on
  both draws). In the current score the second displaces the first. A3 and the other checkpoints already exist, so a
  pilot on the seed-42 development episodes costs minutes of CPU and no training.
- *Against:* whether the two z-scored terms add or cancel is unknown. T_a's gain is largest at high λ_a, where
  aspect-finding suffers. The 2-D grid adds pick freedom on seed 42. Changing the test-time score after a failed test
  invites a forking-paths objection; the mitigation is a fresh test seed, a pre-registration before any test number,
  and reporting E3's NO-GO next to A′.
- *Ceiling:* H1 cannot raise T_a's gain beyond what training gave it (about 1 point), so it passes only if that point
  survives fusion on a fresh draw (Section 3).

**H2. Fit repair** (the loss stayed near its constant-score value).
- *Candidates, none established as causes* (§6.3 of Appendix A could not separate them): a fixed or bounded τ; the
  factor term's scale against β·cos in the training score (β = 0.3; A5 with β = 0 was the one run whose fresh-episode
  agreement gain cleared 0, 0.61 [0.07, 1.18]); more episodes per step, steps or a different learning rate; aspect-only
  training or a warm start from C0 or SE.
- *Against:* raising the loss weight did not help (A3's unweighted aspect loss ended closer to the constant-score value
  than A1's: 3.222 against 3.191). Each candidate needs retraining, so a grid of up to about 10 runs, plus a
  pre-registered training-fit gate before any GO test. If the partitions themselves are not learnable as conditional
  aspects, no H2 setting will help.

**H3. Task learnability diagnostic** (decides whether H2 can work at all). Train the same architecture and loss on
episodes built from the real ArtELingo labels of **scorer-train** rows.
- *Why:* the pseudo-partitions are only moderate proxies of the evaluation aspects (AMI 0.20 for affect against
  emotion, 0.32 and 0.40 for image against style and genre, 0.16 for caption against genre), and E3 cannot say whether
  the weak fit came from the proxies or from the architecture and loss. Label-supervised probes reach about 23 R@1 on
  the spike's episodes, so the task is learnable for probes; whether this factor basis with the agreement rule can
  learn it has never been tested.
- *What it is not:* a candidate method. It reads evaluation labels on training rows, which spec §4 C2 forbids for the
  method. It is an upper-bound diagnostic, labelled as such in the log and report, and never reported as the method.

**H4. Combination:** the H1 score on an H2-repaired model, if H1 alone falls short.

## 5. Proposed designs

### 5.1 H3: label-trained learnability diagnostic

- **Bank LAB.** `build_episode_bank` (the builder behind E2's banks) with the evaluation labels of scorer-train rows as
  its partitions: emotion without the catch-all "something else", style, and genre where labelled (−1 elsewhere). Same
  constraints and eligibility rule as E2's banks (at least 30 paintings per value, value-disjoint, cross-item, third
  aspect controlled), 65,536 episodes over the three aspect pairs, a new bank seed. Scorer-train rows only; no
  selection, val or held row enters the bank. Building one E2 bank took about 3.5 minutes of CPU.
- **Runs.** L3 = A3's settings (the picked recipe: λ_aspect 3, λ_swap 1, β 0.3, L 32, 32 episodes per step, 2,000
  steps, model seed 42) trained on LAB. Optionally L5 = A5's settings (β = 0), the one E3 run whose fresh-episode
  agreement gain cleared 0. Both under one GPU lock, about 10 minutes each, in parallel.
- **Measurements.**
  1. *Fit:* the aspect loss over the last 10 logs against its constant-score value, beside E3's 1.0% to 4.4%; the τ
     trajectory.
  2. *In-distribution:* the gain on fresh label episodes over scorer-train rows (a new episode seed), with the
     procedure of `train_fit_diagnostic.py`.
  3. *Transfer to development episodes:* on the seed-42 selection episodes, the term-only gain and either rate (fixed
     λ = ∞), the E3 cross-fitted score with its uniform control, and the nested score of 5.2 if its code exists.
- **Proposed outcome rules,** committed before the run:
  - **No fit:** the loss ends less than 5% below its constant-score value **and** the fresh label-episode gain's lower
    bound is at or below 0. *Reading:* the architecture or the loss is the bottleneck. *Next:* test H2's settings on
    LAB, a fast testbed whose target is known to be learnable, before any pseudo-bank retraining.
  - **Fits, weak transfer:** the run is not "no fit" (the loss ends at least 5% below its constant-score value, or the
    fresh label-episode gain's lower bound is above 0), but the seed-42 term-only gain is below 3.0 points (three
    times A3's 0.99). *Reading:* even with the true aspects, this basis transfers selection across values
    only a little, so a pseudo-partition repair is unlikely to reach GO in the window. *Next:* H1 on the existing
    checkpoints is the only short path; otherwise branch 3.
  - **Fits and transfers:** the run is not "no fit", and the seed-42 term-only gain is at least 3.0 with a lower bound
    above 0. *Reading:* the bottleneck is
    the pseudo-partitions or the fit to them. *Next:* H2 and, if time allows, alternative partitions, scored with H1.
- **Why 3.0:** a GO with a nested score needs a fused gain of roughly 1 point on fresh episodes (Section 3). Seed-42
  gains halved on the fresh draw in E3, fusion keeps only part of a term's gain, and a label-free method should do
  worse than a label-trained one. A label-trained ceiling below about 3 points leaves little room for that chain.

### 5.2 H1: nested-score pilot (a development look)

- **Score:** s = z(cos) + λ_u·z(T_u) + λ_a·z(T_a), per-episode z-scores as in `zfuse`, on the grid λ_u ∈ {0, 0.5, 1,
  2, 4, 8, 16} × λ_a ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16} (56 cells).
- **Its uniform control:** the same formula with T_a replaced by T_u, which reduces to z(cos) + (λ_u + λ_a)·z(T_u),
  cross-fitted on the same grid. This is E3's uniform control on the set of λ sums, so GO against it tests exactly
  whether the conditioned term adds R@1 and gain over the best unconditioned fusion.
- **Cross-fitting:** parity halves, mean(R@1, gain) per half, as in E3; a new 2-D function with unit tests (non-finite
  scores stay misses; the 2-D version equals the 1-D one when λ_u = 0).
- **Models:** A1 to A6 (existing checkpoints, SHA-256 checked against E3's records), with C0 and SE as references for
  whether aspect training matters under the nested score.
- **Outputs:** R@1, gain and either rate per cell and cross-fitted, paired comparisons against the cosine, RCA and the
  nested control, all on seed 42 only.
- **Pilot reading (descriptive):** H1 is *promising* if the best run's cross-fitted nested score beats its nested
  control on seed 42 on both R@1 and gain (lower bounds above 0). The pick for the A′ test would be pre-registered
  separately before any seed-45 number.

### 5.3 H2

Run only after H3 has read out: on LAB if H3 says "no fit", on the pseudo banks if H3 says "fits and transfers", in
both cases behind a pre-registered training-fit gate.

## 6. The orders compared

| Order | First step | First decision-relevant result | What a negative first result tells us | Main risk |
|---|---|---|---|---|
| **O1: H3 first** (the user's leaning), then H1 or H2 by H3's outcome | H3 | about 1 day (bank, runner, 10-minute run, scoring) | the architecture/loss is the bottleneck (no fit), or the whole repair is unlikely in the window (weak transfer) | delays the cheapest test (H1) by a day; H3's reading may be confounded (Section 8) |
| **O2: H1 first,** H3 only if H1 falls short | H1 pilot | about half a day (2-D cross-fit code and tests, minutes of CPU) | the nested score does not carry T_a's gain; H3 then decides between H2 and branch 3 | if H1 looks promising on seed 42 we may skip H3 and test a thin margin (Section 3) on seed 45 without knowing the ceiling |
| **O3: H1 pilot and H3 in parallel** | both | about 1 day for both | both readings at once, with H3 also scored under the nested score | two development looks on seed 42 at once; more code to review in the same day |
| **O4: H2 grid directly** | about 10 retraining runs | about 1.5 to 2 days including a pre-registration | little: a failed grid cannot separate a bad setting from an unlearnable task | spends the most time on the least diagnosed step |

The two diagnostic steps depend on each other in one direction: H1 says whether a fused score can carry T_a's gain,
and H3 says how large T_a's gain could be if training fit the true aspects. H3's transfer measurement is more
informative when the nested scorer of H1 already exists.

Cost estimates (ours, from E2 and E3 timings, not measured for these steps): H1 pilot, half a day of code and tests,
minutes of CPU. H3, half a day of code (a bank built from labels with E2's builder, and a runner variant), about 4
minutes of CPU for the bank, about 10 minutes of GPU per run, minutes of CPU for scoring. H2 grid, about a day
including its pre-registration, under an hour of GPU.

## 7. Constraints that do not move

- **Episode seeds.** Seed 42 is the development and pick draw. Seed 43 is spent (E3's test). Seed 44 was scored by
  both MLLM probe runs. Seed 45 is reserved for the A′ GO test and is untouched (no episode or result file uses
  it). A written ledger records every episode seed scored, by whom and for what.
- **Held rows stay untouched.** The ArtELingo held budget is 0 of 2 used for aspect episodes; the repair spends none.
- **RCA stays the GO bar.** It was named mechanically on seed 42 in E1 and is not re-picked on seed 45; cosine and
  RCA are re-run on seed 45 with the E1 runner before the GO test.
- **Evaluation rules:** 4 support and 4 contrast pairs, 13 candidates, value-disjoint cross-item episodes, at least
  30 paintings per eligible value, the three pairs pooled; R@1, gain, other-aspect rate and swap from the shared
  metrics module; painting-clustered bootstrap, 5,000 resamples, seed 42; ties and non-finite rows are misses.
- **Row scope:** development on selection rows only; training and fitting on scorer-train rows only; val and held rows
  never; NaN masking outside scope, asserted.
- **No evaluation labels in the method.** GoEmotions affect clusters are allowed as distant supervision. H3 is the one
  clearly labelled exception, as a diagnostic.
- **Grid size** of the order of E3's (about 10 runs), pre-registered.
- **Pre-registration first:** the A′ spec revision and pre-registration are committed before any A′ run is scored on
  seed 45. A dated addendum may add descriptive items, never decision rules after results.

## 8. Weaknesses we already see

1. **Granularity confound in H3.** The labels have 8, 23 and 10 values; the pseudo-partitions have 64 clusters each.
   A label-trained success could come from coarser partitions rather than from label semantics, which would point the
   repair at k rather than at the feature spaces. A cheap control would be k-means partitions at matched k (8, 23, 10)
   on E2's feature spaces, at the cost of one more bank and run.
2. **H3 trains on exactly the evaluation aspects.** A pass bounds "selection among trained aspects", the narrowed C2
   after K8's failure, not transfer to an unseen aspect.
3. **An H3 failure is not conclusive about the architecture.** Settings chosen for pseudo banks (τ, β, steps, episodes
   per step) may not suit label banks, so "no fit" may partly reflect the settings.
4. **One model seed per run and one development draw.** E3 showed single-draw differences reversing between seeds 42
   and 43 (the supervision ablation: −0.56 on seed 43, +0.23 on seed 42).
5. **The thresholds are judgement calls** (3.0 points, 5% below constant, lower bound above 0), set without a power
   analysis.
6. **Repeated development looks on seed 42** (the H1 pilot, H3, then the A′ pick) raise the chance that seed 42
   overstates what A′ can do; only the seed-45 test controls that, and only for the final claim.
7. **The window is short.** If both steps take longer than estimated, the repair eats into the branch-1 replication.

## 9. Questions for the panel

1. Which order (O1 to O4) best serves the three aims of Section 0, and why? Is the user's leaning, H3 first,
   methodologically the right first step?
2. Can H3 as designed discriminate its three readings? What would make it more discriminating at little cost (for
   example the matched-k control of Section 8, a second recipe, or another measurement)?
3. Are H3's outcome rules and thresholds sound, and should they be committed before the run, as proposed?
4. Does H3 (evaluation labels on training rows) or the H1 pilot (a 2-D development look on seed 42) threaten the
   validity of the later A′ GO test or of what the paper may claim?
5. What is missing from the H1 pilot design: the control's definition, the grid, the pick rule, or a pilot reading
   that is too lenient given the thin margin of Section 3?

## Appendices

- **Appendix A:** the E3 report, `docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md`, verbatim.
- **Appendix B:** the method-repair handoff, `docs/superpowers/handoffs/2026-10-03-method-repair-handoff.md`, verbatim.
- **Appendix C:** spec §3 (problem) and §6 (method A and the go/no-go) of
  `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`, verbatim.
