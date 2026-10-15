# Handoff: repair method A to turn the E3 NO-GO into a GO

Written 2026-10-03 (morning, after the overnight E0 to E5 run) for a fresh agent. Read this first; it points to
everything else. The user will drive this in a new chat with their research workflow skill.

## The job

**Decision (user, 2026-10-03):** of the three options in the E3 report §11, the user chose **option 3, a method
repair before giving up branch 1** (the method paper). Branch 3 (the analysis paper) stays the fallback; the 8B MLLM
probe (option 2) was not chosen for now.

**Goal:** a method variant (call it **A′**) that passes the **same pre-registered GO test** that A failed, on **fresh
selection episodes**, under a new pre-registration written before any A′ run is scored on them.

**Success criterion (unchanged from E3, spec §6 "Go/no-go"):** against each of three comparators (backbone-only
cosine, the GO bar RCA, and A′'s own uniform-weight control), the painting-clustered 95% lower bound of the paired
difference is above 0 on both R@1 and condition gain. Strong GO: R@1 and gain each at least 4 points above backbone
only.

**Deadlines:** CVPR abstract Tue Nov 10, paper Mon Nov 16, supplementary Mon Nov 23. The original plan wanted the
branch decision on Fri Oct 9 and the branch-1 replication (E5) to start Oct 10, methods frozen Oct 23. A repair that
ends by about Oct 9 to 12 keeps branch 1 on schedule; each later day comes out of that window.

## Read in this order

1. **The E3 report** `docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md`, especially the Summary, §5 (GO
   test), §6 (why it failed; §6.1 is the post-hoc fixed-λ profile, the key evidence), §10 (disclosures) and §11 (the
   repair option, its arithmetic and its risks).
2. **The spec** `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`: §3 (problem), §4
   (claims and branches), §6 (method A and the go/no-go), §10 (statistics and held budget), §14 (review changes).
   A′ changes §6, so the spec needs a dated revision (see "Decisions to take first").
3. **The E3 pre-registration** `src/test/20261101_aspect_factor_gonogo/PREREGISTRATION.md` (with its addendum): the
   template for A′'s pre-registration.
4. **Background only if needed:** E0 `2026-10-29_aspect_eval_setup.md`, E1 `2026-10-30_aspect_baselines.md`, E2
   `2026-10-31_pseudo_partitions.md` (all in `docs/reports/auto/v2/`), and the plan
   `docs/superpowers/plans/2026-10-03-cosir-v2-cvpr-e0-e5.md` (its Mechanism note and Global Constraints still hold).
5. **Project memory:** `~/.claude/projects/-project-CoSiR/memory/project_v2-publication-plan-pending.md`.

## What E3 established (the facts the repair builds on)

All numbers are seed-43 selection episodes (12,288 episodes, 4,575 anchor paintings) unless stated; every number was
re-derived with independent code in the final review.

| Quantity | Value |
|---|---|
| Backbone-only (CLIP B/32 cosine) R@1 | 13.53 (seed 42: 12.96) |
| GO bar RCA, R@1 / gain | 13.52 / −0.06 |
| Picked A3 (λ_aspect 3), cross-fitted, R@1 / gain | 13.76 / 0.26 [−0.04, 0.56] |
| A3 uniform-weight control R@1 | 16.72 (gain 0 by construction) |
| "Either aspect candidate first" rate: cosine / uniform term / A3 weighted term alone | 27.1 / 33.4 / 21.4 |
| A3 weighted term alone (fixed λ = ∞), condition gain | 0.97 [0.53, 1.41] (seed 42: 0.99 [0.56, 1.45]) |
| A3 at fixed λ = 8 (fused), gain | 1.14 [0.71, 1.58] |
| Term-only paired gain, A3 minus SE / minus C0 | +0.91 [0.38, 1.47] / +0.80 [0.25, 1.37] |
| Pseudo-aspect training loss, last 10 logs vs constant-score value (3.258; 2.565 without swap) | 1.0% to 4.4% below, in every run |
| Learned τ (A3) | 0.031 → 0.082, rising in every logged interval |
| Fresh pseudo-aspect episodes (training rows), A3 gain: agreement rule / training score (0.3·cos + term) | 0.36 [−0.12, 0.86] / 0.95 [0.28, 1.62] |
| K8 (H1, no genre partition) | +0.10 [−0.21, 0.40], fails; A1 also misses on genre (0.42 [−0.02, 0.86]) |
| Label-supervised probe reference (aspect spike, diagnostic) | about 23 R@1, so the task is learnable with labels |

**The two limits, in one sentence each.**
- **Selection costs aspect-finding.** The uniform factor term finds an aspect-sharing candidate more often than CLIP,
  but the agreement-weighted term, which does select the conditioned aspect, finds one less often than CLIP; since
  R@1 = (either rate + gain) / 2, the pre-registered cross-fit kept the factor weight small and the fused gain fell
  to 0.26.
- **The fit was weak.** The aspect loss barely moved from its constant-score value; the little that was learned seems
  to have transferred (training-task gain ≈ labelled-aspect term gain ≈ 1).

**The arithmetic the repair must respect.** To beat the uniform control's R@1 of 16.72, a method at A3's cross-fitted
either rate (27.25) needs a condition gain above about 6.2; one at the weighted term's either rate (21.4) needs about
12.1. **A score that keeps the uniform term's 33.4 either rate beats 16.72 with any reliable positive gain**
(R@1 = 16.72 + gain / 2). That is why the nested score below is the first lead.

## Repair hypotheses (ranked by the evidence)

**H1. Nested test-time score (most direct; untested).** Keep the uniform term's aspect-finding and add the agreement
term's selection:
`s = z(cos) + λ_u · z(uniform factor term) + λ_a · z(agreement-weighted factor term)`, with z the per-episode z-score
(as in `src/model/aspect_rule.py::zfuse`), and λ_u, λ_a cross-fitted per parity half on a small 2-D grid.
- The A3 checkpoint already exists, so H1 can be piloted on **seed-42** episodes (development, already used for the
  pick) in minutes, using the stored codes and the scorers in `src/eval/aspect_scorers.py`. Do this pilot before
  writing the pre-registration, and record it as a development look.
- **Define its uniform control carefully:** the same nested formula with the agreement weights replaced by uniform
  weights (the condition removed), with its own cross-fitted λs. GO against that control then tests exactly whether
  the conditional part adds R@1 and gain. Write this definition into the pre-registration.
- Open question: whether the two z-scored terms add or cancel; the profile suggests the weighted term's gain is
  largest at high λ_a, where aspect-finding suffers, so the 2-D grid should include high λ_a with high λ_u.

**H2. Fit repair (the loss stayed near its constant-score value).** Candidates to test, not established causes (§6.3
could not separate them):
- τ: fixed or bounded instead of learned (it rose steadily, at 30% to 52% of the most Adam can move it).
- The factor term's scale against β·cos in the training score (`aspect_beta` 0.3); A5 (β = 0) is the one run whose
  fresh-episode agreement gain cleared 0 (0.61 [0.07, 1.18]).
- Episodes per step (32 now), steps (2,000), learning rate; aspect-only training or a warm start from C0/SE.
- **Counter-evidence:** raising the loss weight did not help (A3's unweighted aspect loss ended closer to the
  constant-score value than A1's: 3.222 vs 3.191).
- Gate before any GO test: the training-fit check of `src/test/20261101_aspect_factor_gonogo/train_fit_diagnostic.py`
  (gain on fresh pseudo-aspect episodes over training rows) must be clearly above 0, at a threshold pre-registered.

**H3. Task learnability (decides whether H2 can work at all).** The k-means pseudo-partitions may not be learnable as
conditional aspects by this basis. A cheap **diagnostic** (never a candidate, never reported as the method): train the
same architecture and loss on episodes built from the real ArtELingo labels of **scorer-train** rows. If even that
cannot fit, the bottleneck is the architecture or the loss, not the pseudo-partitions. This reads evaluation labels on
training rows, which the spec forbids for the method, so label it clearly as an upper-bound diagnostic in the log and
report. Alternative partitions (finer k, other feature spaces) follow only if the label-trained check fits.

**H4. Combine.** If H1 alone falls short, an H1 score on an H2-repaired model.

## Decisions to take first (with the user)

1. **Spec revision.** H1 changes spec §6's test-time score, so A′ is a method change: write a dated spec revision
   (method A′, its score, its control, its grid) before the pre-registration. The paper must report E3's NO-GO of A
   alongside any A′ result.
2. **Fresh episodes for the A′ GO test.** Seed 42 is the development and pick draw; seed 43 is spent (E3's test);
   seed 44 was scored by both MLLM runs. Use a new seed (e.g. 45) on **selection rows** for the A′ test, and seed 42
   (or another dev seed) for picking. This also needs the comparators on the new seed: re-run the E1 runner
   `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed 45` (cosine and RCA per-anchor arrays) before
   the GO test. Keep a written list of every episode seed scored and by whom ("looks" ledger).
3. **Held rows stay untouched.** ArtELingo held budget: 0 of 2 used for aspect episodes; the repair should not spend
   one. The held ledger is `docs/superpowers/held_ledger.md`.
4. **GO bar.** RCA stays the GO baseline (named mechanically on seed 42 in E1); do not re-pick it on the new seed.
5. **Grid size.** E3 allowed about 10 runs; keep A′'s grid of the same order and pre-register it.

## Evaluation rules that must not move

- 4 support + 4 contrast pairs, 13 candidates; value-disjoint, cross-item episodes from `build_aspect_episodes` (spec
  §5.1); at least 30 paintings per eligible value; the three ArtELingo pairs (emotion × style, emotion × genre,
  style × genre) pooled.
- R@1, condition gain, other-aspect rate, swap from `src/eval/aspect_metrics.py::per_anchor`; painting-clustered
  bootstrap, 5,000 resamples, seed 42 (`cluster_bootstrap`, `compare`); ties and non-finite rows are misses.
- λ cross-fitting by parity `np.arange(n) % 2` over the pooled episodes, pick criterion mean(R@1, gain) per half,
  per-half grid extension (`crossfit_lambda`). A 2-D (λ_u, λ_a) version needs its own function, tests and a
  pre-registered grid.
- Row scope: development on **selection** rows only; training and any fitting on **scorer-train** rows only; val and
  held rows never; NaN masking outside scope, asserted (copy the asserts from `run_gonogo.py`).
- No evaluation labels in training (GoEmotions affect clusters are allowed, as distant supervision), except the
  clearly labelled H3 diagnostic.

## Code entry points

| What | Where |
|---|---|
| Episodes, validator, hashing | `src/eval/aspect_episodes.py` |
| Metrics, bootstrap, paired compare | `src/eval/aspect_metrics.py` |
| Agreement weights, z-score, z-fusion | `src/model/aspect_rule.py` |
| Cosine, agreement term (uniform flag), fixed-β score, cross-fit | `src/eval/aspect_scorers.py` |
| Raw-feature baselines incl. RCA | `src/eval/pair_metric_baselines.py` |
| Factor training with the aspect loss (`lambda_aspect`, `aspect_beta`, `lambda_swap`, `aspect_episodes_per_step`) | `src/train/train_factors.py`, `src/train/aspect_loss.py` |
| Pseudo-partitions and banks | `src/train/pseudo_partitions.py`; banks in `src/test/20261031_pseudo_partitions/results/` |
| E3 runner (`--train`, `--select`, `--gonogo`), grid launcher | `src/test/20261101_aspect_factor_gonogo/run_gonogo.py`, `run_grid.sh` |
| E3 checkpoints A1 to A6, H1, S1 | `src/test/20261101_aspect_factor_gonogo/checkpoints/` (local, gitignored) |
| Post-hoc fixed-λ profile (the H1 evidence) | `src/test/20261101_aspect_factor_gonogo/posthoc_lambda_profile.py`, results `results/posthoc_lambda_profile*.{json,npz}` |
| Training-fit diagnostic (the H2 gate) | `src/test/20261101_aspect_factor_gonogo/train_fit_diagnostic.py`, `results/train_fit_diagnostic.json` |
| E1 runner (cosine and RCA arrays for a new seed) | `src/test/20261030_aspect_baselines/run_baselines.py` |
| SE, C0, R3 codes | `run_affect.model_codes` in `src/test/20261018_affect_factor_learning/run_affect.py` (cached codes in `src/test/20261030_aspect_baselines/results/codes_*.npz`) |
| E4 features (Qwen and CLIP) for later experiments | `/data/SSD2/pre_extract/<dataset>/<backbone>/`; check with `scripts/extract_features.py --verify` |

## Environment and rules that bite (all verified 2026-10-03)

- **Python:** `/root/miniconda3/envs/CoSiR/bin/python` (Python 3.11, transformers 5.6.2). Never install into it; use
  `pip install --target /data/SSD2/pyenvs/<name>/`.
- **GPU (local RTX 3090, shared):** check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  first; wrap every GPU job in `flock -n -o -E 75 /tmp/gpu0.lock <cmd>`; one lock holder may run several of your own
  jobs (E3 ran 3 trainings in parallel under one `flock`, about 10 min each, 3.9 GiB each). CPU jobs set
  `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. DAS6 is allowed if the local GPU is blocked (cluster-run skill; the user owns
  reservations).
- **Long jobs** are launched by the main session with `run_in_background`; subagents write and smoke-test scripts.
- **Polling pitfall:** a wait loop like `until ! pgrep -f "foo.py"` never ends, because `pgrep -f` matches the loop's
  own command line. Use the background-task notification, or `pgrep -f "[f]oo.py"`.
- **Git:** main, files staged by explicit path, `bin/` stays untracked; main is in sync with origin as of 74c87f3
  (pushed 2026-10-03 at the user's request); push again only if the user asks.
- **Reports:** every experiment ends with a report in `docs/reports/auto/v2/` (paper-draft style, a real baseline
  beside every number, figures, no dashes as punctuation, untested readings labelled) plus one row in
  `docs/reports/reports_sum.md`, then `python scripts/check_reports_sum.py` must print OK. Report dates are a sequence;
  the next free value is 2026-11-04 (11-02 is the MLLM probe folder, 11-03 the E4 folder).
- **Experiment folders:** `src/test/<sequence date>_<name>/` with the `.gitignore` copied from
  `src/test/20261023_aspect_episode_spike/.gitignore`; each ends with a `_log.md`. Edits to existing source files get
  an entry in `.claude/<yyyymmdd>_log.md` (it is gitignored but tracked; add with `git add -f`).
- **Pre-registration first:** commit the A′ pre-registration before any A′ run is scored on the test seed; a dated
  addendum is fine for descriptive items, never for decision rules after results.
- **Final review:** after the repair's GO test, run one whole-branch review on the most capable model that re-derives
  the load-bearing numbers from stored arrays, then one fix wave and a scoped re-review (`~/.claude/rules/final-review.md`).
  On this project it has found a real defect every time, including twice in E0 to E5.

## Lessons from the overnight run (defects that reviews caught)

- The plan's own scorer turned out-of-scope NaN rows into finite scores (fixed: non-finite stays non-finite). Keep
  `test_nonfinite_scores_are_misses` and the NaN tests green when adding a 2-D fusion.
- A baseline can silently duplicate another (the plan's pair probe equalled the diagonal rule); test new scorers for
  distinctness.
- A diagnostic loaded the wrong checkpoint under the right name (S instead of SE); assert checkpoint SHA-256 prefixes.
- The first E3 explanation of the failure was contradicted by the stored codes; re-derive mechanism claims from data
  before writing them.
- Single-draw ablations can reverse on another episode seed (S1 vs A1: −0.56 on seed 43, +0.23 on seed 42); report
  both draws and one model seed per arm as a limit.

## Suggested first steps

1. Read the E3 report §6.1 and §11 and the post-hoc profile JSON.
2. Pilot H1 on seed-42 episodes with the existing A3 (and A1, A4, A5) checkpoints: a small 2-D (λ_u, λ_a) grid, its
   uniform control, R@1 / gain / either rate. Record it as a development look in a new folder
   (`src/test/20261104_<name>/`).
3. In parallel, the H3 label-trained diagnostic on CPU or a short GPU run, to learn whether the architecture can fit
   aspect selection at all.
4. With the user: decide A′ (H1 alone, or H1 + an H2 training change), write the spec revision, then the
   pre-registration (grid, pick rule on seed 42, the GO test on a fresh seed with its comparators, the training-fit
   gate if training changes).
5. Run, test once, report, final review; then hand the branch decision back to the user.
