# Handoff: fix the aspect reader, with the CSD style grouping in the set (plan (a))

Written 2026-10-06 00:25 (Amsterdam) for a fresh chat. Read this first; it points to everything else. The user drives
the decisions. **Order of work set by the user: an ARS methodology-focus review of this plan comes first. If the review
is smooth, run the plan; if it raises issues, fix the plan (and §5's draft rule) before any code.**

## 1. The job in one paragraph

CoSiR v2 scores an image and a caption under an unnamed aspect shown by 4 support pairs and 4 contrast pairs. Our
label-free pipeline places items into pseudo-aspect *groupings* with cross-modal heads, and a *reader* picks which
grouping the support pairs share. On 5 October we added a style grouping built from CSD style embeddings. Told the right
grouping, the scorer now beats its matched condition-free counterpart by **+2.23 [1.93, 2.56] R@1**, the highest so far;
the label-free reader gets **+0.06 [−0.15, 0.28]**. The grouping is good enough; the reader is the binding part. The plan
replaces the reader's raw "largest Δ" rule with readers that can tell the groupings apart, picks one on development
episodes against a pre-written rule, and tests it once on fresh episode seeds before the go/no-go decision on
**Friday 9 October 2026** (CVPR abstract 10 November).

## 2. Read in this order

1. The draft report `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md` (commit 6b67c7f),
   especially §2 (terms), §9 (step 1) and §10 (the checklist and the open decision). It is the main source.
2. Step-1 plan and log: `src/test/20261116_grouping_step1_style/PLAN.md`, `20261116_grouping_step1_style_log.md`
   (commit c935b09). Step 0 (same commit): `src/test/20261115_grouping_step0_checks/` (PLAN, ADDENDUM_0a, log).
3. The earlier reader handoff `docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md`: its Step 1 (calibrated
   and learned reader), Step 3 (gating), Step 5 (test) and §6 (pitfalls) are the templates this plan adapts.
4. The stage report `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md` §2 (task, metrics,
   R@1 = (either + gain) / 2) and §14 (oracle layers, why reading costs either rate).
5. Literature behind the grouping choices: `src/test/20261114_grouping_research/synthesis.md` (scan T6 on pseudo-task
   transfer matters for R-b below).
6. Project memory: `project_v2-publication-plan-pending.md` (top entries), `v2-matched-control-lesson.md`,
   `feedback_seed-handling-light.md`; global rules in `~/.claude/rules/` (notably `timestamps.md`,
   `shared-resources.md`, `final-review.md`, `agent-routing.md`).

## 3. Decided and not to be reopened

- **Plan (a) chosen by the user (2026-10-06):** reader fix with the CSD grouping in the set. Design L (late-fusion
  refinement of groupings) and a change of course (benchmark or analysis paper) are not pursued now; 9 October decides
  what follows if (a) fails.
- **Groupings fixed:** affect = Leiden default on GoEmotions probabilities (graph k 20, resolution 1.0, 41 groups);
  image and caption = E2 k-means 64; style = CSD Leiden grouping (17 groups). No grouping is re-chosen in this work.
  Gram, sibling-aware agreement, image-side affect head work and Leiden image/caption groupings are stopped (report §10).
- **No grouping choice reads the evaluation labels.** Reader variants are method choices made on development episodes,
  which is allowed; the told mapping is a diagnostic only.
- **Matched controls** for every configuration (memory `v2-matched-control-lesson.md`), with B extended to B′ by every
  new condition-free ingredient.
- **Light seed handling:** develop on episode seed 42; test the one carried configuration on fresh seeds 49, 50, 51
  (free in `docs/superpowers/episode_seed_ledger.md`), each reported and pooled.
- **Timestamps** in Amsterdam local time, plain, no offset: `TZ=Europe/Amsterdam date '+%F %H:%M'`.

## 4. Where things stand (seed 42, development; report §9)

| Configuration (CLIP heads, B = C2 with R@1 18.34) | Told margin | Reader margin | Bar margin | Reader pick accuracy |
|---|---|---|---|---|
| A0: affect L, image k-means 64, caption k-means 64 | +1.64 [1.37, 1.92] | +0.35 [0.15, 0.57] | +0.31 [0.10, 0.53] | 54.7% (3 groupings) |
| A1: A0 + CSD style grouping | +2.23 [1.93, 2.56] | +0.06 [−0.15, 0.28] | +0.01 [−0.23, 0.24] | 43.2% (4 groupings) |
| AR: A0 + random 17-group grouping (control) | +0.67 [0.40, 0.95] | +0.25 [0.09, 0.42] | +0.16 [−0.02, 0.35] | 41.9% |

Terms: *margin* = R@1 of the fused term minus its matched condition-free counterpart (the same term averaged over the two
conditions), both fused on B; *B′* = B rebuilt with the configuration's averaged-heads term (A0 18.44, A1 18.80); *bar
margin* = fused reader minus whichever of B′ and the counterpart has the larger R@1. *Told* gives the scorer the right
grouping (emotion → affect, style → CSD in A1 / image in A0, genre → image); a ceiling, not a method.

**Why the reader fails** (report §9, Figure 3).
1. *Coarse groupings win by noise.* 17 large groups give large agreement values, so Δ (support-pair agreement minus
   contrast-pair agreement) swings widely. The random grouping, which carries nothing, was picked in 27% to 37% of the
   conditions where the right grouping's Δ is weak.
2. *CSD looks like "the visual grouping" for both style and genre.* CSD matches style about as well as genre (AMI 0.34
   against 0.33; pair ratio 0.83, below 1), so genre supports also agree on it, and its coarse groups beat the 64-group
   image grouping on Δ: picked in 70.5% of emotion × genre genre conditions (image, correct, 15.8%) and 55.0% of
   style × genre genre conditions.
3. The extra grouping also lifts the matched comparators (A1: B′ − B +0.46), so condition-free value does not count.

## 5. The plan

### 5.1 Base and reference
Base configuration **A1** (the four groupings, CLIP heads exactly as in step 1). Every reader also runs on **A0** as the
reference, to show whether the CSD grouping helps once the reader works.

### 5.2 Reader candidates (cheapest first)

| Id | Reader | Targets | Cost |
|---|---|---|---|
| **R-a** | *Scaled Δ:* divide each grouping's Δ_h by its spread, then take the arg-max. Spread = root mean square of Δ_h over the seed-42 development episodes, both conditions pooled (Δ under b is −Δ under a, so the mean is 0; reads no label). Frozen from seed 42 for the test | failure 1 | minutes, CPU |
| **R-b** | *Learned reader:* a multinomial logistic regression over per-episode features of **all groupings jointly** (per grouping: mean support agreement S_h, mean contrast agreement C_h, Δ_h, their spread over the 4 pairs, and the share of support pairs whose image and caption arg-max groups coincide), trained to predict which grouping the supports share, on pseudo-aspect bank episodes built from the four groupings (scorer-train rows only). Scored by arg-max (primary) or the expected term Σ P(h)·s_h (secondary); condition b's features come from the swapped inputs | failures 1 and 2: it can learn that CSD and image agreeing together means genre, CSD alone means style | hours, CPU |
| **R-c** | *Confidence gate* on the best of R-a / R-b: s = z(B) + λ·g·z(T), g in [0, 1] from the reader's top-two margin, gate applied after z-scoring, λ and threshold by A′'s min-margin cross-fit; the counterpart applies the same gate to T_cf | the cost of remaining wrong picks | minutes |

**R-b requirements** (from the 2026-10-04 handoff, Step 1.2):
- *Cross-fitted heads for training only:* refit the heads on half of the scorer-train paintings and build the reader's
  training features only from bank episodes on the other half (or swap halves and average), so the reader learns from
  out-of-sample posteriors as it will meet them. At evaluation the reader reads the standard heads of step 1.
- *Bank:* `src/train/pseudo_partitions.py::build_episode_bank` with the four groupings as partitions (one block per
  grouping pair, as E2 built its AIC bank; about 200 s per 65,536 episodes on CPU); bank rows are local scorer-train
  indices (map them as `src/test/20261105_method_repair_diagnostics/common.py` does).
- *Domain-shift check:* report the reader's accuracy on held-out bank episodes beside its seed-42 pick accuracy; if the
  first is high and the second does not move, R-b is killed. Literature (scan T6, CACTUs; our dropped N4) says
  pseudo-task training transfers only partly.

### 5.3 Measured for every candidate, on A1 and A0
Fused reader against its matched counterpart (R@1, gain, either) and against B′; the bar margin; pick accuracy per pair
and condition under the told mapping (diagnostic); per aspect pair; paired differences against the same configuration's
current arg-max reader. 95% intervals from 5,000 painting resamples.

### 5.4 Draft decision rule
To be committed as `src/test/20261117_reader_fix_csd/DECISION_RULE.md` **after the ARS review and before any code**
(20261113, reserved earlier for a reader-fix folder, stays unused). The review may change any item.

1. **Candidates:** R-a, R-b (arg-max and expected), each on A1 and A0; R-c applied to the best of them by bar margin.
2. **Matched counterpart** of each candidate: the same fused score with the reader term replaced by its two-condition
   mean, max-R@1 cross-fit (`crossfit_condition_free`); B′ = B plus the configuration's averaged-heads term.
3. **Development bar:** bar margin ≥ +0.5 with a 95% lower bound above 0, and the reader's gain over its counterpart
   with a lower bound above 0.
4. **Carried configuration:** the largest bar margin among candidates that clear the bar; ties within 0.05 go to the
   simpler (R-a before R-b, no gate before gate, A0 before A1).
5. **Kill:** if no candidate raises pick accuracy by at least 10 points over its configuration's arg-max reader (A1
   43.2%, A0 54.7%) or reaches the bar, no test is built; report to the user.
6. **Test:** build seeds 49, 50, 51 once (`src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`,
   about 200 s each on CPU; check the 9 new episode SHA-256s differ from all earlier ones), rerun every cross-fit on each
   seed's own halves (patterns `src/test/20261108_new_method_quick_checks/run_n6c.py`, `test_seeds.py`). **GO** =
   pooled over the three seeds (one cluster per painting across seeds), R@1 and condition gain both with 95% lower
   bounds above 0 against **each of** cosine, RCA, B′ and the matched counterpart (gain is 0 for all but RCA, so it is
   one number; do not count it several times). Each seed reported alone. Update the seed ledger.

### 5.5 Timeline (Amsterdam)

| Day | Work |
|---|---|
| Tue 6 Oct | ARS methodology-focus review of this plan; fix if needed; commit the rule; implement and run R-a; start R-b's bank and cross-fitted heads |
| Wed 7 Oct | Train and evaluate R-b with the domain-shift check; R-c; controller re-derives every load-bearing number |
| Thu 8 Oct | Apply the rule; if a candidate clears the bar, build seeds 49 to 51 and run the one test |
| Fri 9 Oct | User decides from the test (GO / NO-GO); if nothing cleared the bar, decide between design L and other options |

### 5.6 What the ARS review should look at
Matched counterparts and B′ for each candidate (gating included); leakage in R-b (bank rows, cross-fitted heads,
features only from out-of-sample posteriors); multiplicity (several candidates on a reused seed 42, carried by the
maximum; every fresh-seed test so far roughly halved the development margin); whether R-a's spread estimated on the
development episodes is acceptable or should come from bank episodes; the kill thresholds; the label policy (told
mapping only as a diagnostic); whether the timeline is realistic.

## 6. Code, data and environment

| What | Where |
|---|---|
| Step-1 configuration code (four groupings, per-arm told mapping, generalised `evaluate_general`, `bar_margin`, B′ with averaged heads) | `src/test/20261116_grouping_step1_style/run_step1.py`; results in its `results/` (gitignored): `step1_group_style.npz` (partitions, incl. `style_csd`), `step1_heads_style.npz` (selection-row posteriors of the style groupings; affect L heads are refit bit-identically with `fit_one_head`, image and caption posteriors are the stored N6 ones in `src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz`), `step1_eval_style.{json,npz}` |
| Heads | `src/test/20261111_community_told_oracle/run_told_oracle.py::fit_one_head` (60,000-row draw, `LogisticRegression(C=1, max_iter=300)`, unit CLIP features) |
| Reader, deltas, scores | `src/eval/aspect_quick_checks.py` (`aspect_deltas`, `inferred_scores`, `crossfit_condition_free`); `src/eval/aspect_nested.py` (`crossfit_nested`) |
| Context, B, stored-number assertions | `src/test/20261112_community_sweep/run_sweep.py::setup`; step-0's `run_checks.py` shows the importlib workaround for the clashing `run_checks` module name |
| Groupings on disk | affect L: `src/test/20261111_community_told_oracle/results/per_anchor_told_oracle.npz` (`partition_L`); E2 image/caption: `src/test/20261031_pseudo_partitions/results/partitions.npz`; CSD: step 1's `step1_group_style.npz` |
| Style features | `/data/SSD2/pre_extract/artelingo/style_csd_vitl/` (`embeddings.npy` 61,402 × 768, `meta.json`, `row_to_image.npy`) |
| Bank builder | `src/train/pseudo_partitions.py::build_episode_bank`; E2 example `src/test/20261031_pseudo_partitions/build_partitions.py` |
| Test seeds | `src/test/20261030_aspect_baselines/run_baselines.py`; ledger `docs/superpowers/episode_seed_ledger.md` |

- **Python:** `/root/miniconda3/envs/CoSiR/bin/python`; CPU work with `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
  MKL_NUM_THREADS=8`, at most 3 processes; check `uptime` and `free -g` first. Nothing here needs the GPU or node404.
- **Process the user wants:** subagents for implementation (two in parallel is fine: R-a and R-c in one, R-b's bank and
  reader in the other); the main session reviews every result and re-derives load-bearing numbers with its own code;
  long jobs launched by the main session in the background; a whole-branch final review before calling the work done
  (`~/.claude/rules/final-review.md`).
- **Records:** a dated folder `src/test/20261117_reader_fix_csd/` with `.gitignore` copied from
  `src/test/20261023_aspect_episode_spike/`, plan or rule committed before numbers, a log at the end; report sections or a
  new report under `docs/reports/auto/v2/` with a row in `docs/reports/reports_sum.md` and
  `python scripts/check_reports_sum.py`.
- **Git:** main; stage by explicit path; `bin/`, `docs/paper/` and `.DS_Store` stay untracked; commit only when the
  user asks; never push unless asked.

## 7. Pitfalls already paid for

- A control that removes more than the condition fakes a pass (N1). Every candidate's counterpart and B′ as in §5.4.
- Per-episode z-scoring removes per-episode scale; a gate must multiply after z-scoring.
- Δ under condition b is exactly −Δ under a.
- Any condition-free ingredient (sibling smoothing, a fourth grouping) lifted the method and its counterpart equally;
  judge readers on the bar margin, not on R@1 against B.
- B's cross-fit picks were tuned on the same parity halves the fusions reuse (a small leak shared by every arm).
- Two step-0 rules were design errors (0a's leave-one-out reading, 0c's instance-level gate); have the rule reviewed
  before running, which is why the ARS review comes first.
- Seed 42 has been reused many times; only the fresh-seed test counts.
