# Fixing the aspect reader with a CSD style grouping in the set: a pre-results plan for review (CoSiR v2, plan (a))

Date: 2026-10-06 (Amsterdam). Status: a plan for review. Nothing in it has been run: no reader code, no bank, no
episodes, no decision-rule file. The plan itself is Appendix A, §5 (the handoff of 2026-10-06, verbatim). Every number
quoted from earlier work comes from Appendices B to E, whose controller reviews re-derived them from stored per-anchor
arrays.

## 0. The decision requested

CoSiR v2 scores an image and a caption under an unnamed aspect shown by 4 support pairs and 4 contrast pairs. A
label-free pipeline places items into pseudo-aspect *groupings* through cross-modal logistic *heads*, and a *reader*
picks which grouping the support pairs share. On 5 October a fourth grouping built from CSD style embeddings raised the
told margin (the scorer given the right grouping) to +2.23 [1.93, 2.56] R@1 over its matched condition-free
counterpart, while the label-free reader fell to +0.06 [−0.15, 0.28] (Appendix B §9). The user chose plan (a): replace
the reader's "largest raw Δ" rule with readers that can tell the groupings apart, pick one on the seed-42 development
episodes against a pre-written rule, and test it once on fresh episode seeds 49, 50 and 51 before a go/no-go on
Friday 9 October 2026 (CVPR abstract 10 November).

The question for this review is **whether the plan in Appendix A §5, including its draft decision rule (§5.4), can
discriminate what it claims to and supports the decision it feeds**, specifically:

1. whether each reader candidate (R-a scaled Δ, R-b learned reader, R-c confidence gate) has a matched counterpart and
   B′ that remove only the condition;
2. whether R-b's training can leak evaluation information or in-sample head posteriors into the reader;
3. whether the selection among several candidates on a reused development seed, carried by the maximum, keeps the
   fresh-seed test meaningful;
4. whether R-a's spread should be estimated on the development episodes or elsewhere;
5. whether the kill thresholds, the development bar and the GO rule are coherent with each other and with the label
   policy (the told mapping is a diagnostic only);
6. whether the timeline (Tue 6 to Fri 9 October) is realistic.

The author's order of work: this review first. If it is smooth, the plan runs as written; if it raises issues, the plan
and its §5.4 rule are fixed before any code, and the fixed rule is committed as
`src/test/20261117_reader_fix_csd/DECISION_RULE.md` before any number it governs exists.

## 1. Task, metrics and bars in brief (details: Appendix E)

- **Episode.** A query (one image or one caption of a selection painting), 4 support pairs that each agree within the
  pair on a value of aspect A (four different values, never the query's), 4 contrast pairs that do the same for aspect B,
  and 13 candidates in the other modality: p_A shares the query's value of A, p_B its value of B, 11 negatives share
  neither. Swapping supports and contrasts (condition b) makes p_B the target. Aspects: emotion (8 values, labelled per
  viewer row), style (23) and genre (10) (per painting). Pairs: emotion × style, emotion × genre, style × genre; 4,096
  episodes per pair per episode seed. Each episode yields four rankings (2 conditions × 2 directions).
- **Metrics.** R@1 (target strictly first; ties miss), other-aspect rate, condition gain = R@1 − other-aspect rate,
  either rate = R@1 + other-aspect rate, so R@1 = (either + gain) / 2. A condition-free scorer has gain exactly 0.
- **Comparators.** Cosine (12.96 on seed 42), RCA (13.38, the GO bar), B (the best condition-free score, 18.34), B′ (B
  rebuilt with a configuration's own averaged-heads term) and the configuration's matched counterpart.
- **Margin** = fused reader minus its matched counterpart. **Bar margin** = fused reader minus whichever of B′ and the
  counterpart has the larger R@1. **Development bar** (inherited): bar margin ≥ +0.5 with a 95% lower bound above 0,
  set because every fresh-seed test so far roughly halved the development margin.

## 2. Fixed by the user and not to be reopened (Appendix A §3)

- Plan (a) itself; design L and a change of course are not pursued now.
- The groupings: affect = Leiden on GoEmotions probabilities (41 groups); image and caption = k-means 64 on CLIP
  features; style = CSD Leiden (17 groups). No grouping is re-chosen.
- No grouping choice reads evaluation labels; reader variants are method choices made on development episodes; the told
  mapping is a diagnostic only.
- Matched controls for every configuration; B extended to B′ by every new condition-free ingredient.
- Seed handling (user preference, recorded 2026-10-04): develop and pick on episode seed 42; test the one carried
  configuration on fresh seeds 49, 50 and 51, each reported and pooled. The user considers several fresh test seeds
  sufficient protection and does not want single-look or lucky-seed ceremony; pre-registering the rule before the test
  still stands (Appendix F).

## 3. Implementation facts the plan relies on

Read from the code on 2026-10-06; stated here because the plan refers to these functions by name.

- **Development episodes (seed 42):** 12,288 episodes on 4,602 anchor paintings (anchors drawn with replacement). The
  two cross-fit halves are the episodes' index parity. Fresh episode seeds are new draws on the same 6,451 selection
  paintings.
- **Reader statistic** (`aspect_deltas`): for grouping h, Δ_h = mean over the 4 support pairs of p_h(image) · p_h(caption)
  minus the same mean over the 4 contrast pairs, with p_h the head posteriors. Under condition b, Δ_h is exactly −Δ_h
  under condition a. The current hard reader takes the arg-max of Δ over the configuration's groupings (ties to the
  first) and scores candidates by p_h(query) · p_h(candidate) on the picked grouping.
- **Fused reader** (`crossfit_nested(B, B, T, parity)`): every term is z-scored per ranking row; the score is
  (1 + λ_u)·z(B) + λ_a·z(T) over a 7 × 8 grid (λ_u, λ_a ∈ {0, …, 16}, 56 cells). On each parity half, the cell that
  maximises min(R@1 − R@1 of B, condition gain) is chosen and applied to the other half.
- **Matched counterpart:** T_cf = (T under condition a + T under condition b) / 2, fused by
  `crossfit_condition_free(B, B, T_cf, parity)` over the same 56 cells with the maximum-R@1 rule. That function raises an
  error unless each of its terms is identical under both conditions.
- **B and B′:** B = `crossfit_condition_free(cos, T_N1u, T_6u, parity)`, with T_N1u the centered factor term of A3 and
  T_6u the head agreement averaged over the three E2 groupings (R@1 18.34). B′ is the same call with T_6u averaged over
  the configuration's own groupings (A0 18.44, A1 18.80), so B′ is B rebuilt, not B plus a term. B's own cross-fit
  picks were tuned on the same parity halves that later fusions reuse.
- **Bar margin:** the comparator (B′ or the counterpart) is the one with the larger mean R@1 over all episodes; the
  margin is the paired per-anchor difference; the same comparator is used per aspect pair.
- **Pick accuracy** (diagnostic): per episode, the mean over the two conditions of 1[picked grouping = told grouping];
  A1's told mapping is emotion → affect, style → CSD, genre → image. With four groupings chance is 25%, with three 33%.
- **Heads** (`fit_one_head`): one image head and one caption head per grouping, `LogisticRegression(C=1, max_iter=300)`
  on unit-normalised CLIP ViT-B/32 features of a 60,000-row draw from the 183,694 scorer-train rows (the draw touches
  31,287 of 36,518 scorer-train paintings); posteriors are computed on selection rows. Held-out accuracies, image /
  caption head: affect 9.81 / 35.72, image 92.7 / 22.9, caption 21.6 / 89.7, CSD 85.10 / 40.41.
- **Bank** (`build_episode_bank(partitions, groups, rows, n_per_pair, seed)`): one block per unordered pair of groupings
  (six blocks for four groupings), each built by the same `build_aspect_episodes` that builds evaluation episodes, with
  groupings in place of aspects and group ids in place of values (cross-item pairs, value-disjoint, 13 candidates, at
  least 30 paintings per value). When exactly three groupings are given, the third is controlled on the candidates; with
  four groupings no third grouping is controlled. Bank supports share a group of the conditioned grouping exactly. The
  affect, image and caption groupings assign groups per row; the CSD grouping assigns one group per painting.
- **Intervals:** 5,000 painting resamples of the per-anchor arrays; cross-fit picks are fixed before resampling.
- **Δ scale:** for the E2 groupings, the per-episode spread of Δ was about 0.017 (Appendix D §3); coarse groupings (17
  groups) give larger agreement values than 64-group ones (Appendix B §9).

## 4. What the review should look at (Appendix A §5.6, verbatim)

> Matched counterparts and B′ for each candidate (gating included); leakage in R-b (bank rows, cross-fitted heads,
> features only from out-of-sample posteriors); multiplicity (several candidates on a reused seed 42, carried by the
> maximum; every fresh-seed test so far roughly halved the development margin); whether R-a's spread estimated on the
> development episodes is acceptable or should come from bank episodes; the kill thresholds; the label policy (told
> mapping only as a diagnostic); whether the timeline is realistic.

The object of review is Appendix A §5 (5.1 to 5.6). Appendix A §1 to §4 and §6 to §7 are context. Appendices B to F are
evidence and are not themselves under review.

## Appendices (verbatim)

- **Appendix A.** Handoff of 2026-10-06, `docs/superpowers/handoffs/2026-10-06-reader-fix-with-csd-handoff.md` (the
  plan).
- **Appendix B.** Draft report of 2026-10-05, `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md`
  (groupings, Leiden, step 0, step 1).
- **Appendix C.** Step-1 log, `src/test/20261116_grouping_step1_style/20261116_grouping_step1_style_log.md`, sections
  Results to Controller review.
- **Appendix D.** The earlier reader handoff of 2026-10-04, `docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md`
  (the templates this plan adapts).
- **Appendix E.** Stage report of 2026-10-04, `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`,
  §2 (task, data, metrics, protocol) and §14 (oracle layers and diagnosis).
- **Appendix F.** Two project lessons recorded by the user: the matched-control lesson and the seed-handling preference.


---

# Appendix A. Handoff of 2026-10-06 (the plan; verbatim)

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


---

# Appendix B. Draft report of 2026-10-05: pseudo-aspect groupings, Leiden communities, step 0 and step 1 (verbatim)

# Pseudo-aspect groupings: how good they are, Leiden communities, and the first redesign steps (DRAFT)

**Status: draft, exploratory.** Written 2026-10-05 from a working session; Sections 7 to 10 added the same evening
(redesign research, step 0 and step 1). Every number is on the seed-42 development episodes (12,288 episodes, 4,602
anchor paintings), and none of it decides the GO. Intervals are 95% and resample anchor paintings (5,000 resamples). Run
folders: `src/test/20261110_partition_profile/`, `src/test/20261111_community_told_oracle/` and
`src/test/20261112_community_sweep/` (commit a6287af); `src/test/20261114_grouping_research/` (literature),
`src/test/20261115_grouping_step0_checks/` and `src/test/20261116_grouping_step1_style/` (result files gitignored).
Figures and their data: `docs/reports/assets/2026-11-12_partition_quality_leiden_communities/`.

## Summary

The stage report of 4 October ended with a diagnosis: the label-free reader picks the right pseudo-aspect grouping in
only 52.4% of rankings, so its margin over its matched condition-free counterpart is +0.14 [−0.04, 0.32] R@1, while the
same scorer told the right grouping reaches +1.14 [0.90, 1.41]. We asked what the groupings themselves are worth. A
profile against the evaluation labels showed that the affect grouping (k-means, 64 clusters on GoEmotions probabilities)
does carry emotion as clusters, but almost none of that signal survives the classifiers that place a lone image or
caption into a cluster (same-emotion pairs agree 2.17 times as often as different-emotion pairs on the clusters, 1.11
times through the classifiers). Replacing the affect k-means with Leiden communities on the same signal raised the told
margin to +1.64 [1.37, 1.92] and the reader margin to +0.35 [0.15, 0.57]; k-means with the same number of groups did not
move (told +1.10). A 3 × 3 sweep of the Leiden graph's k and resolution (14 to 118 groups) found Leiden ahead of k-means at
every matched count (told +0.48 to +0.88) and margins that barely depend on the settings; the picked cell reached a reader
margin of +0.50 [0.30, 0.71], the development bar. The discussion then turned to the groupings as a component: their
number of groups was inherited and never tested, every grouping uses the same number, nobody knew whether they were good
before they were used, they are fixed, and their groups do not interact. We decided to redesign that component before
returning to the bars, the reader and the next experiments.

The redesign began with a literature run (six scans and a synthesis) that ranked four designs by where the sources are
merged and recommended the cheapest, one grouping per source (P0), as the first and control design. Three label-light
checks followed (step 0). A label-free placeability score ranked Leiden above k-means at every matched count, as the told
margins do, and was adopted for comparisons at one group count; counting sibling groups as related lifted the reader and
its condition-free counterpart by the same half point, so the margin from reading stayed at +0.35; and the image head
reaches half of the small painting-level ceiling for affect. Step 1 added a style grouping built from CSD style
embeddings (hand-matched) or VGG-19 Gram statistics (generic). CSD gave the first grouping that matches style at least as
well as genre and raised the told margin to +2.23 [1.93, 2.56], the highest so far, with a style × genre gain of +1.00
[0.49, 1.51]; but the label-free reader fell to +0.06 [−0.15, 0.28], below a random fourth grouping (+0.25), because the
extra grouping lifts the matched comparators and the reader picks CSD for genre conditions too. No arm reached the
development bar. Of everything tried in P0, the CSD style grouping is the only change worth continuing, and only together
with a reader fix (Section 10).

## 1. Where this started

The stage report (`docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`, Sections 10, 11 and 14) and
the handoff (`docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md`) placed the block in the reader: the heads of
N6, trained without ArtELingo labels, beat their matched counterpart by +1.14 when told the right grouping and by +0.14
when the reader chose. The plan was to calibrate or learn the reader, add a grouping that separates style from genre, and
test once on fresh seeds 49 to 51.

A design question and answer on the stage report (commit 7b26631, Section 3 of that report) traced how the buddy line
left the pipeline. The Stage 2 topic mapper was dropped by design on 28 September, the affect buddy graph was never built,
and the Block 1 Leiden communities lost to image k-means on 1 October. What survived was the content buddy graph as a
regulariser of method A's factors, which reaches today's numbers only through B, and the GoEmotions signal, as a k-means
grouping. The conditions of an episode therefore come from its example pairs and from three fixed k-means groupings, and
no part of the system discovers or learns them. That prompted the question of this report: do the groupings that define
the pseudo-aspects coincide with the human aspects?

**The affect signal is external and frozen.** The GoEmotions model is the public checkpoint
`SamLowe/roberta-base-go_emotions` (RoBERTa-base fine-tuned on Reddit comments, 28 labels). The scripts we checked,
on the buddy line, on the percept branch's PercepT port and in v2 (`src/data/affect.py`), load it in eval mode and run it
under `no_grad`. The only optimisers in them train PercepT's own autoencoder, and we found no script that fine-tunes it. It reads ArtELingo captions at inference and never trained on them. Our
PercepT port uses the same checkpoint, so that comparison shares the signal; the published PercepT uses ModernBERT-base
and CLIP ViT-L/14. Against cosine, RCA and the MLLM baselines, which get no GoEmotions signal, the method has an extra
external source that the paper must state (its labels name 6 of the 8 evaluation emotions).

## 2. Terms used below

| Term | Meaning |
|---|---|
| grouping (partition) | one way of splitting the 183,694 scorer-train rows into groups without evaluation labels: **affect** (GoEmotions probabilities of the caption), **image** (CLIP image features), **caption** (CLIP caption features); E2 built each with k-means, 64 clusters |
| group | one cell of a grouping (a cluster or community) |
| head | a logistic regression on frozen CLIP features, trained on 60,000 scorer-train rows to predict a row's group from its image alone or its caption alone; one per grouping and modality, six in all (N6) |
| agreement | for an image of one painting and a caption of another, the dot product of the image head's and the caption head's group probabilities |
| reader | N6's label-free rule: for each grouping, support-pair agreement minus contrast-pair agreement (Δ); pick the grouping with the largest Δ and score candidates by agreement with the query on it |
| told | the same scorer given the right grouping from the evaluation labels (emotion to affect, style and genre to image); a diagnostic ceiling for the reader on these groupings and heads |
| B | the best score that ignores the condition: cosine, A3's centered factor term and the agreement averaged over the three groupings, weights cross-fitted; R@1 18.34 [17.97, 18.70] (cosine 12.96, RCA 13.38) |
| matched counterpart | B plus the added term averaged over the two conditions, so it has the same ingredients but cannot follow the condition |
| margin | R@1 of B plus the term (told or reader) minus R@1 of its matched counterpart: what reading the condition adds |
| λ | the added term's weight relative to B (in the code a pair, (1 + λ_u) on z(B) and λ_a on z(term)), chosen on one parity half of the episodes and applied to the other (min-margin rule for the term, max R@1 for the counterpart) |
| B′ | B rebuilt with the averaged-heads term of a configuration's own groupings (the condition-free bar extended by any new ingredient) |
| bar margin | R@1 of the fused reader minus whichever of B′ and the matched counterpart has the larger R@1 (Sections 8 and 9); the development bar asks for at least +0.5 with a lower bound above 0 |

The bars: the GO requires, on fresh seeds 49 to 51 pooled, R@1 and condition gain lower bounds above 0 against cosine,
RCA, B and the matched counterpart. Because every fresh-seed test so far roughly halved the development margin, the
development bar on seed 42 is a reader margin of at least +0.5 with a lower bound above 0. Three oracle layers frame it:
the label-free reader (+0.14), told groupings (+1.14) and told label-probe posteriors built from the evaluation labels
(+6.66).

## 3. How good the groupings are (partition profile)

We profiled the three E2 groupings against the evaluation labels on scorer-train rows, and through the heads on selection
rows. The run took 13.8 s; exact counts agreed with 2,000,000 sampled pairs in all 90 comparisons (largest |z| 2.42), and
the controller re-derived six values with an independent sample.

**Per group.** Every emotion has at least one clearly emotion-coherent affect cluster, but no emotion sits in one cluster.

*Table 1. Affect grouping (k-means 64) by emotion, scorer-train rows.*

| Emotion | Base rate | Best cluster: purity (lift) | Share of the emotion's rows in it | Clusters to cover 80% |
|---|---|---|---|---|
| sadness | 11.8% | 92.6% (7.9×) | 35% | 10 |
| fear | 10.3% | 89.5% (8.7×) | 30% | 11 |
| amusement | 11.2% | 95.9% (8.5×) | 15% | 19 |
| disgust | 5.4% | 89.8% (16.5×) | 12% | 15 |
| excitement | 9.4% | 87.5% (9.4×) | 16% | 17 |
| anger | 1.6% | 61.0% (37×) | 19% | 15 |
| awe | 18.4% | 70.9% (3.9×), 588 rows | 1.4% | 15 |
| contentment | 31.9% | 86.1% (2.7×) | 5.5% | 18 |

The median emotion needs 15 of the 64 clusters to cover 80% of its rows, against 28 for a value spread like all rows.
Sadness splits three ways: 35% in cluster 38 (93% sad), about 21% in five smaller mostly sad clusters (3, 59, 32, 35 and 46; 49 to 78% sad,
plausibly GoEmotions' separate grief, disappointment and remorse labels), and about 19% in two large mixed clusters
(57 and 1, 31,000 rows, whose top emotion reaches only 21 to 27%; plausibly captions that state no emotion). Both
readings of the split are untested. The 18 October affect report had already noted six clusters at least 85% one emotion
and row purity 0.488 against 0.284 for the majority guess (`2026-10-18_candidate_a_affect_factor_learning_selection.md`).

**Per pair, on the groups and through the heads.** For two rows on different paintings, we compared how often they share
a group when they share an aspect value with how often they do when they do not (Figure 1, blue). Through the heads we
compared the mean agreement of an image of one painting with a caption of another under the same split (orange).

![Lift on groups and through heads](../../assets/2026-11-12_partition_quality_leiden_communities/lift_groups_vs_heads.png)

*Figure 1. Ratio of agreement for same-value pairs over different-value pairs, on the groups (perfect placement) and
through the heads. 1.0 means no signal. The second group is the Leiden affect grouping of Section 5 (31 groups).*

The affect grouping carries emotion on the groups (2.17×; 7.67% against 3.53% share a cluster) and almost none through the
heads (1.11×; agreement 0.0446 against 0.0402). The image grouping keeps part of its genre signal (9.75× to 3.00×) and
less of style (4.59× to 1.54×). For the reader the relevant comparison is support-like pairs (same value of the
conditioned aspect, different value of the other) against contrast-like pairs: for the affect grouping under emotion
conditions it is 2.04× on the groups and 1.03 to 1.04× through the heads.

*Table 2. Held-out accuracy of the six heads (10,000 scorer-train rows not used for fitting).*

| Grouping | Image head | Caption head |
|---|---|---|
| affect, k-means 64 | 13.5% | 34.6% |
| image, k-means 64 | 92.7% | 22.9% |
| caption, k-means 64 | 21.6% | 89.7% |

Each grouping is predicted well from the modality it was built from and poorly from the other. The affect grouping is
built from the caption's text through GoEmotions, so even its caption head, which sees only CLIP caption features, reaches
34.6%.

**Why the reader picks as it does.** Through the heads, the mean support-minus-contrast contrast under emotion conditions
is +0.0012 to +0.0018 for affect, against −0.0055 (emotion × style) and −0.0236 (emotion × genre) for the image grouping and
−0.0019 and −0.0171 for caption, with a per-episode spread near 0.017 (handoff Section 3). The reader takes affect for
emotion mainly because the others turn negative. Under the style condition of style × genre the image grouping's contrast
is −0.0205 and affect's about 0, so the reader takes affect, matching the observed 17.1% image picks. Even on the groups
the image grouping's style-minus-genre contrast is −0.033 (0.44×): it carries genre more than style, so no reader can use it
for style.

*Sources: `src/test/20261110_partition_profile/` (log, `profile_partitions.py`); E2 report
`docs/reports/auto/v2/2026-10-31_pseudo_partitions.md`; N6 heads in `src/test/20261108_new_method_quick_checks/run_n6.py`.*

## 4. The told oracle with Leiden affect communities

Plan `src/test/20261111_community_told_oracle/PLAN.md` (written before any number) changed only the affect grouping and
kept the image and caption groupings, the told mapping and B. Arm R0 reproduced the stored diagnostics exactly (told
+1.14, reader +0.14, pick accuracy 52.4%), and a fresh refit of the six heads was bit-identical to the stored posteriors.
Arm L used `src/model/communities.py::detect_communities` at its defaults (kNN union graph with k = 20, modularity, seed 42)
on the 28 GoEmotions probabilities per row, with groups under 200 rows merged (42 communities, 41 after merging). Arm K
used E2's k-means at 41 clusters.

*Table 3. Told and reader margins over the matched counterpart (R@1).*

| Affect grouping | Groups | Told margin | e×s / e×g / s×g | Reader margin | Pick accuracy | Told vs R0 (paired) |
|---|---|---|---|---|---|---|
| R0: k-means | 64 | +1.14 [0.90, 1.41] | +1.14 / +2.62 / −0.33 | +0.14 [−0.04, 0.32] | 52.4% | baseline |
| L: Leiden | 41 | +1.64 [1.37, 1.92] | +2.50 / +3.03 / −0.61 | +0.35 [0.15, 0.57] | 54.7% | +0.50 [0.25, 0.74] |
| K: k-means | 41 | +1.10 [0.81, 1.40] | +1.48 / +2.48 / −0.65 | +0.16 [0.00, 0.33] | 51.3% | −0.04 [−0.22, 0.13] |

At the same 41 groups Leiden beat k-means by +0.54 [0.31, 0.78] told, so the gain came from how Leiden groups the rows
and the number of groups did not explain it. Leiden's groups were more emotion-coherent (pair lift 2.71 against 2.04 for K and 2.17 for
R0), but through the heads the signal stayed small (1.145, 1.120 and 1.109). Most of the told gain came from emotion ×
style, the weakest pair, whose margin doubled. The negative style × genre margins come from the fusion weights; that pair
runs on the unchanged image grouping. By the plan's reading, L was promising (paired lower bound above 0 and a reader
moving up), K not better. The controller re-derived the margins and paired differences from the stored per-anchor arrays
and reran Leiden from scratch (same 42 communities).

## 5. Sweep of the Leiden graph's k and resolution

Plan `src/test/20261112_community_sweep/PLAN.md` (written before any number) fixed a 3 × 3 grid, graph k ∈ {10, 20, 40}
by resolution ∈ {0.25, 1.0, 4.0} with `RBConfigurationVertexPartition`, a k-means control at every distinct group count,
and a pick rule: the largest reader margin, ties within 0.05 to fewer groups. The k = 20, resolution 1.0 cell reproduced
arm L exactly (ARI 1.0). Three CPU processes ran 9 Leiden cells and 8 k-means controls in 7 to 8.5 minutes each; every
process first reproduced R0 and B.

![Margins against the number of groups](../../assets/2026-11-12_partition_quality_leiden_communities/margins_vs_groups.png)

*Figure 2. Told margin (left) and reader margin (right) against the number of affect groups, for the nine Leiden cells
(coloured by graph k, slightly offset horizontally) and k-means at matched counts (grey, including R0 at 64). Bars are 95%
intervals.*

*Table 4. Group count, told margin and reader margin by cell (R@1).*

| Graph k | Resolution 0.25 | Resolution 1.0 | Resolution 4.0 |
|---|---|---|---|
| groups, k = 10 / 20 / 40 | 15 / 14 / 15 | 44 / 41 / 31 | 118 / 95 / 88 |
| told, k = 10 | +1.95 [1.67, 2.24] | +1.83 [1.54, 2.12] | +1.85 [1.57, 2.13] |
| told, k = 20 | +1.70 [1.42, 1.98] | +1.64 [1.37, 1.92] | +1.68 [1.40, 1.95] |
| told, k = 40 | +1.88 [1.58, 2.18] | +1.68 [1.40, 1.95] | +1.65 [1.38, 1.92] |
| reader, k = 10 | +0.38 [0.17, 0.59] | +0.40 [0.18, 0.61] | +0.46 [0.23, 0.69] |
| reader, k = 20 | +0.38 [0.18, 0.58] | +0.35 [0.15, 0.57] | +0.48 [0.25, 0.71] |
| reader, k = 40 | +0.36 [0.16, 0.56] | +0.50 [0.30, 0.71] | +0.46 [0.23, 0.69] |

- **Leiden against k-means at matched counts:** told +0.48 to +0.88 in all nine cells, every lower bound above 0; reader
  +0.17 to +0.33, lower bound above 0 in six cells and between −0.02 and 0.00 in three. k-means stayed at +0.97 to +1.24
  told and +0.14 to +0.18 reader at every count from 14 to 118.
- **Flat in the settings:** all nine told margins lie inside each other's intervals, and so do the reader margins. The
  cluster-level emotion lift rose with the group count (1.99 at 14 groups to 3.45 at 118), but the lift through the heads
  stayed at 1.12 to 1.16, because head accuracy fell as groups multiplied (image head 21.2% at 14 groups, 4.8% at 118).
- **Pick accuracy** stayed at 54.1 to 55.9% (R0 52.4%), so the reader's gain came from stronger evidence when it picked
  right, and picking right more often did not account for it.
- **The pick** (k = 40, resolution 1.0, 31 groups): reader margin +0.50 [0.30, 0.71], made of +1.44 condition gain for an
  either-rate cost of −0.43 (N6 on R0: +0.79 for −0.51); told +1.68 [1.40, 1.95]. It is the first label-free configuration
  whose development margin reaches +0.5. It is the maximum of nine cells and beats the untuned default by only +0.15
  [−0.01, 0.31], so shrinkage on fresh seeds is expected; B′ (B rebuilt with the cell's averaged-heads term) was not
  computed for it (for arm L, B′ was 18.44, below its counterpart). The reader now recovers 30% of the told margin,
  against 12% for R0.

*Sources: `src/test/20261111_community_told_oracle/` and `src/test/20261112_community_sweep/` (PLANs, logs, scripts);
figure data `docs/reports/assets/2026-11-12_partition_quality_leiden_communities/figure_data.json`.*

## 6. The groupings as a component: what is open

The numbers above improved the affect grouping. The discussion that followed questioned the component as a whole.

1. **The number of groups was never tested.** 64 first appears as the k-means setting of the "CLIP clusters" condition
   source in stage (d) (`2026-10-13_candidate_a_stage_d_selection.md`); the affect grouping copied "the settings of the
   image-cluster source" (`2026-10-18_candidate_a_affect_factor_learning_selection.md`), and E2 reused all three. We found
   no stated reason for 64. Today's sweep is the first variation, and only for affect.
2. **Every grouping uses the same number.** Nothing requires that; each grouping could have its own, chosen by a
   criterion that does not read the evaluation labels.
3. **Their quality was unknown before use.** They were judged after the fact by AMI with the evaluation labels (stage
   report Table 4), and the told margin was the first functional test.
4. **They are fixed.** Each is computed once from frozen features; the heads are trained once on them, the reader is a
   rule, and nothing is trained end to end. Method A trained its factors on episode banks built from them, and a bank's
   support pairs share a group by construction, while real same-emotion pairs share an affect cluster only 7.7% of the
   time (3.5% for different emotions).
5. **Groups do not interact.** Agreement counts only probability in the same group, so two sad clusters count as
   unrelated; across groupings only fixed rules connect them (B averages, the reader takes the largest Δ).

**Decision (user, 2026-10-05).** The grouping component is to be re-discussed in detail and reshaped before we return to
the bars, the reader and the margins. The proposed robustness runs (several Leiden seeds for the picked and default
cells; the fixed set R0, k-means at 31 and 41, Leiden at 41 and 31 on the spent episode seeds 45, 47 and 48) are on hold
until then. A parallel discussion of improving the Leiden method and using it in factor learning was opened in a separate
session.

**Questions for that discussion.**
- What should a grouping be judged on before it is used, without the evaluation labels (for example how well a lone image
  and a lone caption can be placed in it, or how stable it is across seeds)?
- One number of groups per grouping, chosen by that criterion, or a soft or hierarchical grouping?
- Should groupings be learned or updated with the heads or the factors instead of fixed in advance?
- Should sibling groups count as partial agreement (a group-to-group similarity in the agreement)?
- Which grouping separates style from genre, and can the caption half be placed in the affect grouping by GoEmotions
  itself instead of by a head?

## 7. The redesign: what the literature says and which designs we considered

**How we decided.** In discussion the user added two further concerns: the sources were chosen to match the evaluation
aspects (CLIP image, CLIP caption, GoEmotions), and no grouping holds a single property. The user's first idea was to
fuse several sources into one structure and then split it into single-property groupings. We wrote a brief
(`src/test/20261114_grouping_research/research_brief.md`) and ran an ARS deep-research pass of six parallel scans (three-way
WHY/HOW/WHAT scans for threads 1 to 4, quick briefs for 5 and 6) and a synthesis (`synthesis.md` in the same folder). The
scans verified each cited paper's existence; most were read at abstract level or through fetch summaries, so the
theory results below need a human read before they enter a paper.

**Designs**, named by where the sources are merged:

| Design | Sources merged at | How the parts are separated | What is trained |
|---|---|---|---|
| P0 | not merged: one grouping per source | not needed | heads only |
| L (late fusion) | groupings: the M groupings are kept and a small model refines them jointly | non-redundancy given the other groupings, placeability, staying close to the own source | a refinement model and heads |
| G | graphs: one layer per source in a multiplex graph, one shared Leiden partition | which layers support each community | heads |
| E | edges: every source's neighbour edges pooled and tagged by source | competing property heads, source tags, modality, non-redundancy | a two-tower trunk and heads |
| consensus (dropped) | one fused partition | nothing records which source grouped which rows | |

**What the literature contributed** (synthesis §1 to §4; reader-inferred where marked there).
- No published method fuses several sources and then splits them while measuring that each part holds one property
  (scan T1). The successes that recover several facets learn them jointly on properties that are independent or sampled
  by attribute (MFCVAE, SCE-Net, DiscoverNet).
- Identifiability results for image and text (Daunhawer et al., ICLR 2023, and related work; scan T2) recover the block of
  factors both modalities share, not modality-specific factors. In our data the shared block is content; style is mostly
  image-only and viewer-level emotion caption-only. An objective or a selection rule that rewards cross-modal
  placeability therefore pulls groupings toward content.
- Losses that reshape a representation and its groups without DEC exist (SCAN, TEMI, SwAV, IIC; scan T3), but the
  cross-modal ones reward what image and caption share.
- The best-supported style encoder not trained on WikiArt style labels is CSD (Somepalli et al. 2024, preprint; trained
  on LAION-Styles); Gram statistics of an ImageNet VGG are the cleanest generic option (scan T4). Long-CLIP has low
  priority because ArtEmis captions average 15.8 words.
- The synthesis ranked P0 first (the control every fusion design needs, every part supported), then L, then G (its first
  step is in effect the dropped consensus partition), then E (against the identifiability results, most expensive, and
  overlapping factor learning).

**Decisions (user, 2026-10-05).** Follow the synthesis order (checks first, then P0, then L); hand-matched sources first,
then a fixed generic source menu written before their results; GoEmotions stays the affect source for now; the affect
grouping is the Leiden default (graph k 20, resolution 1.0, 41 groups), because the sweep's pick was chosen by a reader
margin on labelled episodes, which the rule that no grouping choice reads evaluation labels excludes; the decision point
is 9 October, the date of the CVPR plan's go/no-go.

## 8. Step 0: three label-light checks

Plan `src/test/20261115_grouping_step0_checks/PLAN.md` (written before any number) and an addendum for check 0a
(`ADDENDUM_0a.md`); log in the same folder. The controller re-derived every number below with its own code.

**0a, the same-painting ceiling of the affect grouping.** A painting's rows share one image while each row's affect group
comes from its own caption, so the image can only place a row as well as the painting's viewers agree. Two rows of the
same painting share an affect group 1.54 times (k-means 64) and 1.65 times (Leiden) as often as rows of different
paintings, against 1.00 for random relabellings of the same sizes; in absolute terms only about 6% of same-painting pairs
share a group (13% for the caption grouping). The plan's first reading compared the image head's accuracy with a
leave-one-out "painting majority" accuracy; that predictor ignores how common each group is (7.33% against 11.49% for
always guessing k-means' largest group), so the reading was a design error and was replaced by the addendum. On the 5,231
paintings the head never saw, the image head reaches **half of the calibrated ceiling** for both groupings:

| Grouping | Ceiling R_u (perfect painting-level predictor) | Image head H | Share reached F = (H − 1)/(R_u − 1) | Random control |
|---|---|---|---|---|
| k-means 64 | 1.543 [1.483, 1.600] | 1.272 [1.259, 1.285] | 0.50 [0.46, 0.56] | 1.00 |
| Leiden (41) | 1.652 [1.585, 1.722] | 1.327 [1.314, 1.342] | 0.50 [0.45, 0.56] | 1.00 |

Reading: limited room. A better image head could move Leiden's ratio from 1.33 toward 1.65 at most.

**0b, placeability as a label-free criterion.** Placeability is the adjusted mutual information between the image head's
and the caption head's group assignments of the same row. It ranked each Leiden cell of the sweep above k-means at the
same group count in 9 of 9 pairs (Leiden 0.038 to 0.073, k-means 0.035 to 0.043), the ordering the told margins give, so
it was adopted. It also rises with the number of groups (0.038 at 14 groups, 0.073 at 118) while the told margins stay
flat, so it can compare groupings only at one group count within one source.

**0c, sibling-aware agreement.** Agreement p_imgᵀ S p_txt, with S the centred-centroid cosine similarity between groups,
counts two sad clusters as related. Against plain agreement the margins did not move (told +0.02 [−0.26, 0.30], reader
−0.01 [−0.25, 0.25]). S improved the reader on B from +0.41 to +0.89 R@1 and its pick accuracy from 54.7% to 58.9%, but it
improved the condition-free counterpart by the same amount (+0.05 to +0.55; B′ 18.44 to 19.06): smoothing over siblings
made the heads a better similarity in both uses, not a better reader. The plan's safety gate (same-row against random-pair
separation) penalised smoothing by design and was not the right test; the "do not adopt" reading rests on the margins.

## 9. Step 1: a style grouping beside affect, image and caption

Plan `src/test/20261116_grouping_step1_style/PLAN.md` (written before any number); log in the same folder. The image
grouping carries genre more than style (pair ratio 0.44 for style against genre; told gain on style × genre exactly 0),
so a grouping in which style dominates is the only route to reading style × genre.

**Features and groupings.** CSD ViT-L style embeddings (the authors' release, hash-checked) and VGG-19 Gram statistics
(five layers, a 128-dimension PCA per layer, 640 dimensions) for the 61,402 painting images; Leiden at the default
settings on one node per painting; a random grouping of CSD's sizes as the control for offering the reader a fourth
option. Label-free diagnostics and one disclosed label description (computed after the arms; it chose nothing):

| Grouping | Groups | Overlap with the CLIP image grouping (AMI) | Stability over Leiden seeds | AMI with style / genre | Style-vs-genre pair ratio |
|---|---|---|---|---|---|
| CLIP image k-means 64 (reference) | 64 | 1 | | 0.32 / 0.40 | 0.44 |
| CLIP image Leiden (reference) | 17 | 0.57 | 0.74 | 0.28 / 0.44 | 0.43 |
| **CSD** | 17 | 0.40 | 0.81 | **0.34 / 0.33** | **0.83** |
| Gram | 11 | 0.22 | 0.67 | 0.14 / 0.19 | 0.77 |
| random | 17 | 0.00 | | 0.00 / 0.00 | 1.00 |

CSD carries something the CLIP image grouping does not (overlap 0.40, below the 0.57 of another clustering of the same
CLIP features), is stable, and is the first grouping that matches style at least as well as genre; style still does not
dominate (ratio below 1). Gram is new but weak on both labels.

**Arms** (A0 is the Leiden affect grouping with the image and caption k-means groupings, which reproduced the
told-oracle arm L exactly; the reader picks among the arm's groupings):

| Arm | Told margin | Told on style × genre, minus A0 | Reader margin | Bar margin | Reader minus the random-slot arm |
|---|---|---|---|---|---|
| A0 (baseline) | +1.64 [1.37, 1.92] | | +0.35 [0.15, 0.57] | +0.31 [0.10, 0.53] | |
| AR: + random grouping | +0.67 [0.40, 0.95] | +0.07 [−0.42, 0.57] | +0.25 [0.09, 0.42] | +0.16 [−0.02, 0.35] | |
| A1: + CSD | **+2.23 [1.93, 2.56]** | **+1.00 [0.49, 1.51]** | +0.06 [−0.15, 0.28] | +0.01 [−0.23, 0.24] | −0.19 [−0.44, 0.04] |
| A1s: + CSD, image head on CSD | +2.26 [1.94, 2.61] | +0.96 [0.42, 1.47] | −0.01 [−0.24, 0.22] | −0.01 [−0.26, 0.25] | −0.26 [−0.52, −0.01] |
| A2: + Gram | +1.76 [1.45, 2.08] | +0.61 [0.08, 1.15] | −0.00 [−0.18, 0.17] | −0.04 [−0.25, 0.18] | −0.26 [−0.47, −0.05] |
| A2s: + Gram, image head on Gram | +1.60 [1.26, 1.93] | +0.65 [0.09, 1.21] | +0.10 [−0.04, 0.23] | +0.10 [−0.04, 0.23] | −0.16 [−0.34, 0.02] |
| A3: Leiden image and caption (descriptive) | +1.39 [1.09, 1.69] | | +0.04 [−0.20, 0.27] | +0.04 [−0.20, 0.27] | |

![Step 1: told and reader margins, and the reader's picks under A1](../../assets/2026-11-12_partition_quality_leiden_communities/step1_told_reader.png)

*Figure 3. (a) Told and reader margins over the matched counterpart for every step-1 arm, with 95% intervals. (b) Under
A1, the share of rankings in which the reader picked each grouping, for four conditions; ✓ marks the grouping the told
oracle uses.*

**Readings** (plan §6). R1, the told oracle gains on style × genre: met by all four style arms. R2, the development bar:
met by none. R3, better than a random fourth grouping: met by none. Under the plan no fresh-seed test was built.

**Why the told margin rose and the reader fell.** Told "style goes to CSD, genre goes to image", the oracle no longer reads
style from a genre-dominated grouping, and its style × genre margin rose from −0.61 to +0.39. The label-free reader picks
the grouping with the largest Δ (support-pair agreement minus contrast-pair agreement) and fails in two ways.
1. *A coarse extra grouping wins when the right signal is weak.* Seventeen large groups give large agreement values, so
   Δ swings widely by chance. The random grouping, which carries nothing, was picked in 27% to 35% of emotion conditions
   and 37% of style × genre style conditions, where the right grouping's Δ is weak, against 11% of emotion × style style
   conditions and 5% to 7% of genre conditions, where the image grouping's Δ is strong. The extra grouping also lifts the matched comparators (A1: B′ − B +0.46), so what it adds as a condition-free
   similarity does not count as reading.
2. *CSD looks like "the visual grouping" for both style and genre.* Because CSD follows genre about as much as style, the
   support pairs of a genre condition agree on it too, and its coarse groups beat the 64-group image grouping on Δ. Under
   A1 the reader picked CSD in 73.5% of emotion × style style conditions (correct), but also in 70.5% of emotion × genre
   genre conditions, where the image grouping is correct (15.8%), and in 55.0% of style × genre genre conditions; under
   style × genre style conditions CSD's Δ is about zero (pair ratio 0.83) and the reader drifted to affect (45.8%).

## 10. Where the grouping component stands: what to continue

**Design status.**

| Design | Status | What was tested | Result |
|---|---|---|---|
| P0 | partly tested | Leiden against k-means for affect (Sections 4, 5) | Leiden better at every matched count |
| | | sibling-aware agreement (step 0c) | lifts the reader and its counterpart equally; no reading gain |
| | | placeability as a criterion (step 0b) | adopted, at one group count within one source |
| | | image-side affect ceiling (step 0a) | the head reaches half of a small ceiling |
| | | style grouping from CSD or Gram (step 1) | told up (+2.23 for CSD), reader about 0; bar not reached |
| | | Leiden for image and caption (step 1, A3) | worse (reader −0.32 against A0) |
| L | not tested | | |
| G | not tested | the synthesis reduced it to a quick dominance diagnostic, not run | |
| E | not tested | the synthesis proposed moving it to the factor-learning discussion | |
| consensus | dropped | by reasoning and the percept line's union graph | |

**Checklist for P0.**

| Item | Continue? | Reason |
|---|---|---|
| Leiden affect grouping (default, 41 groups) | keep, as the base | beat k-means at every group count |
| CSD style grouping | **yes, together with a reader fix** | the only change that raised the ceiling (told +2.23 against +1.64; style × genre +1.00) |
| Gram style grouping | drop for now | weak on both labels; a later generic reference |
| sibling-aware agreement | stop | lifts the reader and its control equally |
| image-side affect head work | stop (low priority) | half of a small ceiling is already reached |
| Leiden for image and caption | stop | made the reader worse |

**Why this continuation has the best chance, and how good the chance is.** Against the larger of B′ and the matched
counterpart, the told term with CSD leaves a ceiling of +2.23 R@1 (A0: +1.64); the development bar of +0.5 needs a reader
that recovers a little under a quarter of it. A0's reader recovered about a fifth of its ceiling (+0.31 of +1.64), and
with CSD added the reader recovers nothing (+0.01 of +2.23). The first fix of the reader
handoff targets the failure seen here: dividing each grouping's Δ by its own spread removes the advantage of coarse
groupings, needs no labels and runs in minutes on CPU. The chance is moderate at best: every fresh-seed test so far
roughly halved the development margin, and CSD's genre content (pair ratio 0.83) may still confuse a well-scaled reader.
That second problem is the one design L addresses (non-redundancy given the image grouping would remove what CSD shares
with it).

**Open decision for 9 October (user).** (a) Lift the hold on the reader and fix it with the CSD grouping in the set; or
(b) continue the grouping redesign with design L. A change of course to the benchmark or analysis paper is not on the
table now.

## 11. Limitations

- One episode seed (42), reused many times; one Leiden seed and one head-fit draw per cell; the sweep's pick is the best
  of nine.
- The told mapping was chosen with the labels (from AMI); the profile reads evaluation labels on scorer-train and
  selection rows and is descriptive only. Step 1's told mapping (style to the style grouping) was fixed in its plan
  before any number.
- B's cross-fit was tuned on the same parity halves that the fusions reuse.
- Emotion is labelled per viewer row and style and genre per painting, so the per-group numbers mix two label levels.
- Two step-0 rules were design errors, both disclosed above: the leave-one-out reading of 0a (replaced by the addendum)
  and the instance-level gate of 0c.
- CSD starts from a CLIP model and was trained on web images; overlap of its training images with our paintings cannot
  be ruled out. Its labels contain no WikiArt style annotations.
- The literature run read most papers at abstract level or through fetch summaries; the identifiability results need a
  human read before they are cited.
- No result here was tested on fresh seeds.


---

# Appendix C. Step-1 log, sections Results to Controller review (verbatim)

## Results

### Groupings (label free; PLAN.md §5)

| Grouping | Groups | Rows min / median / max | effN | AMI vs E2 image / E2 caption / affect L | Leiden-seed AMI (42–43, 42–44, 43–44; mean) | Held-out img / txt (majority) | P_ami (P_lift) |
|---|---|---|---|---|---|---|---|
| style_csd | 17 | 532 / 9,287 / 23,979 | 11.7 | 0.4029 / 0.1110 / 0.0197 | 0.827, 0.799, 0.796; **0.807** (seeds 43, 44: 14, 15 groups) | 85.10 / 40.41 (13.08) | 0.2084 (2.511) |
| style_csd, CSD image head | | | | | | 91.67 / (CLIP) | 0.1981 (2.548) |
| style_gram | 11 | 3,153 / 17,461 / 24,951 | 10.3 | 0.2199 / 0.0750 / 0.0128 | 0.695, 0.687, 0.637; **0.673** (11, 13 groups) | 71.53 / 31.92 (13.12) | 0.1497 (1.863) |
| style_gram, Gram image head | | | | | | 89.90 / (CLIP) | 0.1242 (1.891) |
| style_rand | 17 | 567 / 9,343 / 23,374 | 11.7 | 0.0038 / 0.0005 / 0.0000 (vs style_csd 0.0011) | n/a | 15.64 / 13.20 (13.18) | 0.0022 (1.004) |
| image_leiden | 17 | 2,697 / 10,328 / 22,009 | 14.4 | **0.5702** / 0.1651 / 0.0225 | 0.732, 0.712, 0.785; 0.743 | 90.35 / 46.46 (11.92) | 0.2818 (3.664) |
| caption_leiden | 30 (32 raw, 217 rows merged) | 631 / 4,064 / 20,726 | 21.2 | 0.1648 / **0.5120** / 0.0329 | 0.843, 0.848, 0.835; 0.842 | 37.34 / 84.60 (11.68) | 0.2226 (3.346) |

References: E2 image and caption from the stored N6 posteriors have P_ami 0.2416 and 0.1959 and P_lift 5.896 and 4.392;
AMI(E2 image, E2 caption) is 0.1663.

At resolution 1.0, painting-level Leiden gives coarse groupings: 17 groups for CSD and for CLIP image, 11 for Gram, all
far below the 64 of the stored k-means groupings. No style or image community fell under 200 rows. style_csd shares a
fair amount with E2 image (AMI 0.40) and little with caption or affect, so it is not a copy of the image grouping.
style_gram is weaker on every label-free count: AMI with image 0.22, seed stability 0.67, held-out CLIP image accuracy
71.53, P_ami 0.150. The source-feature image heads fit their own groupings better than the CLIP head (CSD 91.67 against
85.10, Gram 89.90 against 71.53), but they place no better against the caption head (P_ami 0.198 against 0.208 for CSD,
0.124 against 0.150 for Gram). image_leiden and caption_leiden agree with their k-means counterparts at AMI 0.57 and 0.51,
so the algorithm change alone moves about half the partition.

### Arms (R@1 in pp; B = C2, R@1 18.34 [17.97, 18.70]; baselines A0 and AR)

| Arm | Told margin | Reader margin | Reader gain margin | Reader − B (T_cf − B) | B′ (B′ − B) | Bar margin (comparator) | Pick accuracy |
|---|---|---|---|---|---|---|---|
| **A0** (baseline) | +1.64 [1.37, 1.92] | +0.35 [0.15, 0.57] | +1.34 [1.03, 1.65] | +0.41 (+0.05) | 18.44 (+0.10) | **+0.31 [0.10, 0.53]** (B′) | 54.7 [54.1, 55.3] |
| **AR** (random slot) | +0.67 [0.40, 0.95] | +0.25 [0.09, 0.42] | +0.71 [0.46, 0.97] | +0.27 (+0.02) | 18.45 (+0.11) | +0.16 [−0.02, 0.35] (B′) | 41.9 [41.3, 42.5] |
| A1 (CSD, CLIP heads) | **+2.23 [1.93, 2.56]** | +0.06 [−0.15, 0.28] | +1.30 [0.96, 1.63] | +0.47 (+0.41) | 18.80 (+0.46) | +0.01 [−0.23, 0.24] (B′) | 43.2 [42.5, 43.9] |
| A1s (CSD, CSD image head) | **+2.26 [1.94, 2.61]** | −0.01 [−0.24, 0.22] | +1.52 [1.16, 1.87] | +0.53 (+0.54) | 18.88 (+0.54) | −0.01 [−0.26, 0.25] (B′) | 43.3 [42.6, 43.9] |
| A2 (Gram, CLIP heads) | +1.76 [1.45, 2.08] | −0.00 [−0.18, 0.17] | +0.56 [0.29, 0.82] | +0.13 (+0.14) | 18.51 (+0.17) | −0.04 [−0.25, 0.18] (B′) | 42.0 [41.3, 42.6] |
| A2s (Gram, Gram image head) | +1.60 [1.26, 1.93] | +0.10 [−0.04, 0.23] | +0.28 [0.08, 0.49] | +0.22 (+0.12) | 18.24 (−0.10) | +0.10 [−0.04, 0.23] (counterpart) | 41.4 [40.8, 42.1] |
| A3 (descriptive: affect L, image Leiden, caption Leiden) | +1.39 [1.09, 1.69] | +0.04 [−0.20, 0.27] | +1.57 [1.22, 1.93] | +0.29 (+0.25) | 18.35 (+0.01) | +0.04 [−0.20, 0.27] (counterpart) | 57.0 [56.3, 57.6] |

Pick accuracy counts a pick as correct when the reader's argmax-Δ grouping equals the arm's told grouping for that
condition. With four groupings the chance level is 25% against 33% for three, so the pick accuracies of A1 to AR are
not directly comparable with A0's or A3's.

Per aspect pair (margin R@1; told | reader):

| Arm | Emotion × style | Emotion × genre | Style × genre |
|---|---|---|---|
| A0 | +2.50 / +0.49 | +3.03 / +0.67 | −0.61 [−0.89, −0.35] / −0.10 |
| AR | −0.27 / +0.09 | +2.81 / +0.56 | −0.54 / +0.11 |
| A1 | +3.27 / +0.47 | +3.04 / +0.45 | **+0.39 [−0.16, 0.91]** / −0.73 [−1.11, −0.34] |
| A1s | +3.45 / +0.56 | +2.98 / +0.19 | +0.35 [−0.20, 0.89] / −0.77 [−1.19, −0.35] |
| A2 | +2.15 / +0.22 | +3.14 / −0.01 | +0.00 [−0.54, 0.52] / −0.21 |
| A2s | +1.64 / −0.11 | +3.12 / +0.20 | +0.04 [−0.53, 0.59] / +0.20 |
| A3 | +2.77 / +0.29 | +2.20 / +0.60 | −0.81 / −0.77 |

Paired differences (per anchor; R@1):

| Arm | Told − A0 | Told s×g − A0 | Reader − A0 | Bar − A0 | Told − AR | Reader − AR | Bar − AR |
|---|---|---|---|---|---|---|---|
| AR | −0.97 [−1.25, −0.68] | +0.07 [−0.42, 0.57] | −0.10 [−0.29, 0.09] | −0.15 [−0.33, 0.02] | | | |
| A1 | +0.59 [0.35, 0.85] | **+1.00 [0.49, 1.51]** | −0.29 [−0.52, −0.06] | −0.31 [−0.55, −0.05] | +1.56 [1.24, 1.89] | **−0.19 [−0.44, 0.04]** | −0.15 [−0.41, 0.11] |
| A1s | +0.62 [0.37, 0.89] | **+0.96 [0.42, 1.47]** | −0.36 [−0.61, −0.12] | −0.32 [−0.59, −0.04] | +1.59 [1.25, 1.94] | **−0.26 [−0.52, −0.01]** | −0.17 [−0.45, 0.12] |
| A2 | +0.12 [−0.14, 0.39] | **+0.61 [0.08, 1.15]** | −0.36 [−0.56, −0.15] | −0.35 [−0.61, −0.09] | +1.09 [0.78, 1.41] | **−0.26 [−0.47, −0.05]** | −0.20 [−0.45, 0.06] |
| A2s | −0.04 [−0.32, 0.24] | **+0.65 [0.09, 1.21]** | −0.26 [−0.47, −0.04] | −0.22 [−0.44, 0.01] | +0.93 [0.61, 1.25] | **−0.16 [−0.34, 0.02]** | −0.06 [−0.27, 0.14] |
| A3 | −0.25 [−0.54, 0.03] | −0.20 [−0.65, 0.28] | −0.32 [−0.57, −0.06] | −0.27 [−0.53, −0.01] | | | |

**What moved and why.** Telling the model "style → style_csd" works. The told margin rose from +1.64 to +2.23 (A1)
and +2.26 (A1s). On style × genre it went from −0.61 to +0.39 and +0.35, a paired gain of about one point. The told
term rose on emotion × style as well (+3.27 against +2.50), while emotion × genre did not move. Gram helped the told
term on style × genre too (+0.61, +0.65 paired), though less than CSD, and the gain did not reach the overall told
margin (A2 +0.12 paired, A2s −0.04). The random slot lowered the told term by 0.97, as expected when style is told to
read a grouping that carries no style.

The reader did not follow. Every style arm's reader margin fell to about zero, between −0.01 and +0.10, against A0's
+0.35, and it fell below AR's +0.25 as well. Measured against B, the A1 reader barely moved (+0.47 against A0's +0.41).
Its condition-free counterpart, however, rose from +0.05 to +0.41, and B′ rose by the same amount (B′ − B +0.46): the
fourth grouping's heads are useful as an unconditioned similarity, and the matched comparators absorb that use.

The pick tables show where the conditioned use fails. Under A1 the reader chose style_csd for 73.5% of emotion × style
style conditions, but also for 70.5% of emotion × genre genre conditions, where the told grouping is image (image was
picked 15.8%). In style × genre it picked style_csd for only 30.2% of style conditions (affect 45.8%) and for 55.0% of
genre conditions (image 20.4%). The reader treats style_csd as the image-appearance slot for both style and genre, so
the style × genre reader margin fell from −0.10 to −0.73. The label description below agrees: style_csd still carries
genre about as much as style.

A3 (descriptive) swaps both k-means 64 groupings for Leiden ones (17 and 30 groups). The told term fell by 0.25
(paired, interval reaching +0.03) and the reader by 0.32 [−0.57, −0.06]. The counterpart rose again (+0.25 against B,
against +0.05 for A0), so the reader's margin fell to +0.04.

## Readings of PLAN.md §6, applied literally

| Reading | Rule | A1 | A1s | A2 | A2s |
|---|---|---|---|---|---|
| R1, style slot readable (told) | told s×g margin, arm − A0 (paired), lower bound > 0 | +1.00 [0.49, 1.51] **met** | +0.96 [0.42, 1.47] **met** | +0.61 [0.08, 1.15] **met** | +0.65 [0.09, 1.21] **met** |
| R2, development bar | bar margin ≥ +0.5 with lower bound > 0, and reader gain margin lower bound > 0 | bar +0.01 [−0.23, 0.24]; gain +1.30 [0.96, 1.63]: **not met** | bar −0.01 [−0.26, 0.25]; gain +1.52 [1.16, 1.87]: **not met** | bar −0.04 [−0.25, 0.18]; gain +0.56 [0.29, 0.82]: **not met** | bar +0.10 [−0.04, 0.23]; gain +0.28 [0.08, 0.49]: **not met** |
| R3, not just a fourth option | reader margin, arm − AR (paired), lower bound > 0 | −0.19 [−0.44, 0.04] **not met** | −0.26 [−0.52, −0.01] **not met** | −0.26 [−0.47, −0.05] **not met** | −0.16 [−0.34, 0.02] **not met** |

R2 for reference and controls: A0's bar margin is +0.31 [0.10, 0.53] (reference; not met). AR's is +0.16 [−0.02, 0.35]
(not met). A3, descriptive, is at +0.04 [−0.20, 0.27] (not met). Every arm's reader gain margin has its lower bound
above 0; every R2 failure comes from the bar clause.

**Default for 9 October (PLAN.md):** no arm among A1, A1s, A2 and A2s meets R2, so no fresh-seed test is built, and
9 October decides between continuing (design L, step 3) and changing course. The user decides.

## Label description (disclosed; computed after both eval runs; chooses nothing)

AMI with style uses all 183,694 scorer-train rows and AMI with genre the 148,956 rows with a genre label. The style ×
genre contrast is profile check 3's measure on different-painting pairs: s_AB = P(same group | same style, different
genre), s_BA = P(same group | same genre, different style), and ratio = s_AB / s_BA (above 1: style dominates).

| Grouping | AMI style | AMI genre | s_AB | s_BA | Ratio |
|---|---|---|---|---|---|
| style_csd | **0.3413** | 0.3302 | 0.1859 | 0.2229 | **0.834** |
| style_gram | 0.1435 | 0.1871 | 0.1354 | 0.1756 | 0.771 |
| style_rand | 0.0018 | 0.0011 | 0.0975 | 0.0976 | 0.999 |
| image_leiden | 0.2844 | 0.4358 | 0.1028 | 0.2384 | 0.431 |
| caption_leiden | 0.0554 | 0.1636 | 0.0540 | 0.1082 | 0.499 |
| E2 image (reference) | 0.3176 | 0.3970 | 0.0257 | 0.0585 | 0.439 |
| E2 caption (reference) | 0.0582 | 0.1613 | 0.0161 | 0.0374 | 0.430 |
| affect L (reference) | 0.0159 | 0.0236 | 0.0376 | 0.0388 | 0.970 |

CSD moves the balance toward style: the ratio rose from 0.439 (E2 image) to 0.834, and style_csd is the only grouping
whose AMI with style exceeds its AMI with genre. Style still does not dominate (ratio below 1). That fits the reader
choosing style_csd for genre conditions. Gram moves the ratio about as much (0.771) but carries little of either label
(AMI 0.14 and 0.19). The random slot sits at 1.0, as it must.

## Choices not fixed by PLAN.md

- **Two sets, one script.** The style features arrived after the descriptive arms ran, so every stage runs per set and
  writes `results/step1_<stage>_<set>.{json,txt[,npz]}` (the brief named `step1_<stage>`). The style eval run recomputes
  A0 as its paired baseline and asserts the reproduction again.
- **Painting vectors.** A painting's vector is that of its first scorer-train row. CLIP image vectors differ inside 309
  paintings by at most 1.05e−5; CSD and Gram vectors never differ. Vectors were unit-normalised again (`run_checks.unit`,
  float32) before the kNN, including the already-unit CSD and Gram vectors.
- **Merge.** Under-200 is counted in rows. Centroids are row-weighted means of the unit painting vectors (rows inherit
  their painting's vector); for caption_leiden they are row means of the unit caption features. No style or image
  community needed merging. caption_leiden merged 2 communities holding 217 rows.
- **Stability.** Seeds 43 and 44 ran on the same graph and were merged the same way. For caption_leiden they used
  ModularityVertexPartition (`detect_communities`' call), with seed 42 asserted equal to `detect_communities`. AMI was
  computed on scorer-train rows (sklearn default, arithmetic). It is not applicable to style_rand.
- **style_rand.** `default_rng(0).permutation` over the 36,518 merged painting labels of style_csd. Painting counts per
  group are kept exactly and row counts approximately (min 567 against 532). No merge was applied.
- **Secondary heads.** These use a copy of `fit_one_head`'s image branch with the source features (verified bit identical
  to `fit_one_head` when given CLIP features). The caption posterior is the same grouping's CLIP caption head, not refit.
- **Bar margin.** The comparator is whichever of B′ and the counterpart has the larger mean R@1 over all episodes (ties
  go to B′); the same comparator is used per aspect pair.
- **Readings.** R2's "the reader's gain over its counterpart" was read as the gain metric of the fused reader minus the
  fused counterpart (`fusedT_vs_fusedTcf.gain`), and "≥ +0.5" was applied to the full-precision point. R1 and R3 were
  applied to A1, A1s, A2 and A2s, and R2 to those four plus AR (A0 as reference, A3 as description). "No arm meets R2"
  in the default was read over A1, A1s, A2 and A2s.
- **Pick accuracy.** A pick is correct when the reader's argmax-Δ grouping equals the arm's told grouping for that
  condition. For AR the told style grouping is style_rand.
- **Label description.** Style AMI on all rows (every row has a style), genre AMI on genre-labelled rows, and the
  contrast on rows labelled for both with exact pair counts. Values for E2 image, E2 caption and affect L were added as
  references.
- **Placeability.** P_ami and P_lift were computed on the 32,413 selection rows, as in step 0b, with references from the
  stored N6 posteriors.

## Caveats

- This is exploratory work on the seed 42 development episodes, which earlier runs have read many times. The groupings,
  heads, arms, told mappings and readings were fixed in PLAN.md before any number.
- Each grouping comes from one Leiden seed and one head draw. Seed stability is moderate for style_csd (0.807) and lower
  for style_gram (0.673). For example, seeds 43 and 44 give 14 and 15 CSD groups instead of 17.
- **Granularity is confounded with source.** Painting-level Leiden at resolution 1.0 gives 11 to 17 groups, against 64 for
  the k-means groupings they sit beside (or replace, in A3). A coarse grouping makes a strong condition-free similarity,
  which raises the counterpart and B′; that is part of why the bar margins fall. PLAN.md fixed the recipe, and no other
  resolution was tried.
- Pick accuracies over four groupings (chance 25%) are not comparable with those over three (33%).
- The told mapping keeps genre → image. The reader's preference for style_csd under genre conditions counts as a wrong
  pick even where style_csd separates genre well, so pick accuracy understates how usable the grouping is for genre. The
  fused R@1 margins do not depend on that mapping.
- B's cross-fit picks were tuned on the same parity halves that the fusion and counterpart cross-fits reuse. This is a
  small second-order leak shared by every arm.
- The script's SHA-256 differs between the descriptive (`78e887e9…`) and style (`f4bea509…`) batches, because of the
  loader change described above. The descriptive outputs do not touch the changed function.
- `results/smoke/` holds the smoke runs (DINOv2-small stand-in for CSD and Gram, 3,000-row heads); their numbers are not
  results.

## Files

- `PLAN.md`: the fixed design (unchanged).
- `run_step1.py`: stages `group`, `heads`, `eval`, `describe`; `--set descriptive|style`; `--smoke`.
- `results/step1_group_{descriptive,style}.{json,txt,npz}`: grouping settings, sizes, merges, graph and Leiden
  statistics, feature provenance and alignment checks; the npz holds every partition (merged, raw, seeds 43/44).
- `results/step1_heads_{descriptive,style}.{json,txt,npz}`: head provenance and held-out accuracies; selection-row
  posteriors (CLIP heads and source-feature image heads).
- `results/step1_eval_{descriptive,style}.{json,txt,npz}`: checks, per-arm blocks (overall, per pair, pick, B′, bar,
  paired differences), readings; the npz holds per-anchor fused, counterpart and B′ arrays for all metrics, reader picks
  and told indices, so every margin can be re-derived.
- `results/step1_describe_{descriptive,style}.{json,txt}`: label-free diagnostics; the style file also holds the label
  description.
- `results/run_*.log`: run logs. `results/extract_full.log` belongs to the GPU extraction, not to this script.
- Everything in `results/` is gitignored: 56 MB in total, including 27 MB of smoke outputs. The largest file is 13.7 MB,
  and nothing over 100 MB was written. The style features (`/data/SSD2/pre_extract/artelingo/style_*`, about 0.5 GB) are
  the extraction's outputs and were only read.

## Controller review (2026-10-05 20:55)

The main session re-derived with its own code (`rederive_step1.py` in the session scratchpad): the CSD style grouping
rebuilt from the raw embeddings with its own kNN graph and Leiden call (17 communities, identical to the stored
partition, AMI 1.0; AMI with E2 image 0.4029); and, from `step1_eval_style.npz` with its own painting bootstrap, the
told, reader and bar margins of A0, AR, A1, A1s, A2 and A2s, R1 on the style × genre episodes and R3. Every point estimate
equals the table above and every interval agrees to within 0.03 (bootstrap draws differ). The readings stand: R1 met by
all four style arms, R2 and R3 by none.


---

# Appendix D. Earlier reader handoff of 2026-10-04 (verbatim)

# Handoff: turn the NO-GO into a GO by fixing the aspect reader (and the style/genre split)

Written 2026-10-04 for a fresh agent; revised after a final review the same day. Read this first; it points to everything
else. The user drives the decisions.

## The job

Make the example-conditioned method pass a fresh-seed GO test before the CVPR abstract (10 November 2026). The last three
days (`docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`) ended with every attempt at or below its
matched condition-free control. Exploratory diagnostics run afterwards (Section 3) located the blocking part: **the
reader that decides which aspect the example pairs show**. Told the right partition, the same heads (trained without
ArtELingo labels) beat their own condition-free counterpart by **+1.14 [0.90, 1.41] R@1** on the development episodes;
with the current reader the margin is +0.14 [−0.04, 0.32]. The plan fixes the reader, then the one aspect pair the
representation cannot separate, then tests once on fresh seeds.

Decided with the user and not to be reopened:

- **Branch decision not taken.** The user wants a working method before choosing between the method paper and the
  analysis paper. Paper writing stays paused.
- **Seed handling light** (project memory `feedback_seed-handling-light.md`): develop on seed 42; test the final
  configuration on fresh episode seeds (next free: **49, 50, 51**), each reported and pooled. No single-look ceremony.
- **Write the decision rule before the test seeds are built**, and every rule before the numbers it governs.
- **Matched controls** (project memory `v2-matched-control-lesson.md`): every configuration is compared with the same
  score with only the condition removed. Two false passes came from controls that removed more than the condition.

## Read in this order

1. The stage report above (self-contained; Sections 2, 9 to 11 and 14 matter most).
2. `docs/reports/auto/v2/2026-11-08_new_method_quick_checks.md` (D0, N1, N2, N6, N6c, matched controls).
3. `src/test/20261109_fix_diagnostics/20261109_fix_diagnostics_log.md`, `results/diagnose_fixes.txt` and
   `results/diagnose_counterparts.txt` (this handoff's evidence).
4. `src/test/20261108_new_method_quick_checks/` rule files (`DECISION_RULE.md`, `ADDENDUM_1.md` to `ADDENDUM_3_N6C.md`)
   as templates for the next rule file.
5. Project memory: `project_v2-publication-plan-pending.md` (top entries), `v2-matched-control-lesson.md`.

## 1. Where things stand (seed 42 development episodes)

| Scorer | R@1 | Gain | Either | Note |
|---|---|---|---|---|
| CLIP cosine | 12.96 | 0 | 25.92 | backbone only |
| RCA (GO bar, best raw pair metric) | 13.38 | 0.10 | 26.66 | |
| **B = best condition-free score** | **18.34** [17.97, 18.70] | 0 | 36.68 | z(cos) + λ_u·z(T_N1u) + λ_a·z(T_6u), max-R@1 cross-fit (N6c's matched control) |
| N6 reader term alone | 16.51 | 4.41 [3.96, 4.87] | 28.61 | reads the condition |
| N6 reader on B | 18.53 | 0.79 | 36.27 | +0.19 [0.01, 0.38] over B; **+0.14 [−0.04, 0.32] over its counterpart** |
| Same heads, partition told, on B | 19.84 | 4.56 | 35.12 | +1.50 [1.22, 1.79] over B; **+1.14 [0.90, 1.41] over its counterpart** |
| Label-probe reference, told, on B (diagnostic) | 30.62 | 20.55 | 40.69 | +6.66 [6.31, 7.00] over its counterpart |

T_N1u is A3's centered uniform factor term; T_6u the mean over the three partition heads; T_6 the reader term (all in
`src/eval/aspect_quick_checks.py`). The *counterpart* of a conditioned term T is T_cf = (T under condition a + T under
condition b) / 2, the same term with the condition removed, fused on B with a max-R@1 cross-fit
(`crossfit_condition_free(B, B, T_cf, parity)`). The told term's counterpart alone already adds +0.35 [0.16, 0.55] to B.

## 2. Why the method fails

R@1 = (either + gain) / 2. A conditioned term passes only if its condition gain outruns the either rate it costs. Even
correct readers pay: the told-partition term loses 2.27 [1.88, 2.66] either against its counterpart, and label probes
told the aspect lose 4.81 against their own condition-free mean (either 40.27 against 45.08). They pass because their
gain is large (4.56; 21.06). N6's reader gains too little (0.79) for what it costs (0.51 either), because it often picks
the wrong partition.

## 3. The evidence (exploratory, seed 42, decides nothing)

- **Pick accuracy** (a label-informed development diagnostic: it reads each episode's aspect identity, which the method
  must infer, and the told mapping emotion to affect, style and genre to image was itself chosen from AMI with the
  labels; never compute it on test seeds before their verdict). N6's hard reader picks the told partition in 52.4% of
  rankings and under both conditions in 28.1% of episodes. Per pair and condition: emotion × style 59.5 / 54.3, emotion ×
  genre 67.9 / 61.9, **style × genre 17.1** / 53.6. Because Δ under condition b is exactly −Δ under a and the told mapping
  sends style and genre to the same partition, the ceiling for an arg-max reader is 83.3% of rankings and 66.7% of
  episodes with both picks right (style × genre can never have both).
- **The told partition's margin is the target.** Over its counterpart: +1.14 [0.90, 1.41] R@1, gain 4.56, either −2.27.
  Most of it comes from emotion × style and emotion × genre (its gain on style × genre is exactly 0).
- **Right picks still cost either.** On the 3,459 episodes where N6's picks were right under both conditions (emotion ×
  style 1,576, emotion × genre 1,883, style × genre 0), the told fusion lost 3.01 [2.09, 3.93] either and still gained
  1.95 R@1; N6's own fusion showed no detectable either loss there (+0.12 [−0.43, 0.66]) only because the cross-fit gave
  it a small weight. A confidence gate can reduce the cost of wrong picks but not remove the cost of reading.
- **What did not help:** contrastive terms (picked partition minus the contrast-like partition: +0.07 over B), top-k
  cascades with the reader (−0.55 to −2.59 R@1), soft weighting (+0.12).
- **Why the reader errs.** Partitions differ in head sharpness (held-out head accuracy, image / caption heads: affect
  13.5 / 34.6, image 92.7 / 22.9, caption 21.6 / 89.7), so raw Δ values are not comparable across partitions. Per aspect
  pair and condition, the affect partition's mean Δ is near 0 (|mean| ≤ 0.003; per-episode sd 0.017) while the image
  partition's mean Δ swings by about ±0.02 to 0.03; the reader mostly follows the image partition's sign (an observation
  from the quick-checks final review, not tested causally).

## 4. The plan

Development on seed 42 throughout. For every variant report: pick accuracy (diagnostic, above), the fused score on B
against B **and against its counterpart** (R@1, gain, either, with intervals), and per pair.

### Step 0. Commit the decision rule (before any code)

`src/test/20261110_reader_fix/DECISION_RULE.md` (next free sequence date; copy `.gitignore` from
`src/test/20261023_aspect_episode_spike/`), following `src/test/20261108_new_method_quick_checks/DECISION_RULE.md` and
`ADDENDUM_3_N6C.md`. It must fix:

- the candidate list (Steps 1 to 3) and, for Step 2, the four-partition told mapping used by the pick-accuracy
  diagnostic (which partition counts as correct for style), before any pick accuracy is computed;
- the **matched counterpart** of every configuration: the same fused score with T replaced by T_cf, and B extended by
  every new condition-free ingredient (for Step 2, the new partition's averaged head term; for a learned or soft reader,
  its probabilities averaged over the two conditions; for a gated score, the gate applied to T_cf);
- a **development bar** before any test seed is built: R@1 over max(B, the matched counterpart) of at least **0.5 with a
  lower bound above 0**, and gain lower bound above 0. N6's current reader on B (+0.19 over B, +0.14 over its
  counterpart) is the reference to beat, not a candidate. Every pick so far roughly halved on fresh seeds (gains 0.52 to 0.26
  and 0.26 to 0.15), so a smaller development margin is unlikely to survive;
- the single configuration carried forward (largest R@1 margin over its counterpart among those that clear the bar), the
  test (Section 5) and what follows each outcome.

### Step 1. A calibrated reader (cheapest, highest expected value)

1. **Standardised Δ (minutes).** Divide each partition's Δ_h by its spread and pick the arg-max. Estimate the spreads
   without labels on the seed-42 selection episodes themselves (both conditions pooled; this reads no label), or on bank
   episodes built from rows outside the heads' training draw. Do not estimate them on bank episodes whose rows the heads
   were trained on (those posteriors are in-sample and sharper).
2. **A learned reader (an hour or two).** A multinomial logistic regression over a few per-episode features (for each
   partition: mean support agreement S_h, mean contrast agreement C_h, their per-pair spread, and the share of support
   pairs whose image and caption arg-max clusters coincide), trained on E2's bank `bank_AIC.npz` to predict which
   partition the supports share. Labels are the bank's own pseudo-aspects, so no ArtELingo label is read. Two
   requirements:
   - **Cross-fitted heads.** The current heads were trained on 60,000 scorer-train rows (D0's draw), about a third of the
     rows the bank uses. Refit heads on half of the scorer-train paintings and build the reader's training features only
     from bank episodes on the other half (or swap halves and average), so the reader learns from out-of-sample
     posteriors as it will meet them on selection rows.
   - **Domain-shift check.** Bank supports share a partition cluster exactly; real supports share an ArtELingo value that
     the partitions only partly track (AMI 0.20 to 0.40). Published evidence against the dropped meta-conditioner N4 (a
     pseudo-task adapter that did not help real tasks) points at this gap; E3 never tested it, because method A barely
     fitted the bank. Report the reader's accuracy on held-out bank episodes beside its seed-42 pick accuracy; kill it if the
     first is high and the second does not move.
   Score with the arg-max partition or the expected term Σ_h P(h)·s_h; enforce the swap structure (condition b's
   probabilities computed from the swapped inputs).

**Kill:** neither variant raises the seed-42 pick accuracy by at least 10 points (current 52.4%, ceiling 83.3%) or the R@1
margin over the counterpart to at least 0.5.

Reuse: `aspect_deltas`, `probe_dots`, `inferred_scores` in `src/eval/aspect_quick_checks.py`; `fit_heads`,
`partition_labels`, `load_posteriors`, `n6_terms` in `src/test/20261108_new_method_quick_checks/run_n6.py`; current head
posteriors `src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz` (SHA-256 2ad75026…; selection rows only).
The bank's rows are local scorer-train indices: map them as `src/test/20261105_method_repair_diagnostics/common.py` does.

### Step 2. A partition that separates style from genre (representation)

Style × genre cannot be read from partitions in which both aspects live in the image clusters (told gain 0 there; E2 AMI
of the image partition 0.397 with genre, 0.318 with style). Add a fourth partition trained without ArtELingo labels and
aimed at style, **readable by both modalities**:

- **First kill (cheap): caption readability.** A partition the caption head cannot predict cannot show agreement in
  cross-modal pairs. Fit its caption head first; drop the partition if its held-out caption-head accuracy is near chance.
  This rules out a pure residual-of-caption image partition (caption-unpredictable by construction).
- **Candidates:** clusters of caption-predictable image directions with the genre-heavy component of the image partition
  removed (for example, project out the directions that predict the existing image clusters, then k-means what the
  caption head can still read); or colour and technique statistics of the image if captions describe them.
- **No label-guided choice.** Do not choose among partition variants by AMI with the style or genre labels. If AMI is
  computed (as E2 did, as a description), compute it once for the variant already chosen, and disclose it.

Rerun Step 1's reader with four partitions. **Look at:** style × genre pick accuracy (now 17.1% under the style condition)
and style × genre gain. **Kill:** style × genre gain stays below 0.5.

### Step 3. Confidence gating (only if Steps 1 and 2 leave many wrong picks)

s = z(B) + λ·g_e·z(T), with g_e in [0, 1] from the reader's probability margin. The per-episode z-score removes any
per-episode scale, so the gate must multiply after z-scoring. Choose λ and the threshold by A′'s min-margin cross-fit;
the matched counterpart applies the same gate to T_cf. A gate reduces the either cost of wrong picks; it cannot remove
the cost of correct reading (Section 3).

### Step 4. The fused configuration

Final score: z(B') + λ·z(T_reader) (B' = B plus any new condition-free ingredient), min-margin cross-fit against B'
(`crossfit_nested(B', B', T, parity)`, as `src/test/20261109_fix_diagnostics/diagnose_fixes.py` does), with the matched
counterpart from `diagnose_counterparts.py`.

### Step 5. Contingency if nothing clears the development bar

- An MLLM as a three-way partition reader (show the 4 + 4 pairs, ask which of three described partitions the supports
  share; a variant of candidate N5, with its privileged-reference caveat).
- A learned fusion: a small ranker over [z(cos), z(T_N1u), the head terms, the reader's probabilities] trained on the
  bank, judged against the same matched counterpart.
- If nothing passes, the evidence still supports an analysis paper (branch 3): the task, the either/gain arithmetic, the
  matched-control lesson, D0, N6's reading, the told-partition margin. Report to the user.

## 5. The test (write it in the decision rule before building the seeds)

- Build seeds 49, 50, 51 with `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>` (about 200 s
  each on CPU; also scores cosine and RCA). Check that the 9 new episode SHA-256s (3 pairs × 3 seeds) differ from all
  earlier ones.
- Score the one carried configuration with its cross-fit rerun on each seed's own halves (patterns:
  `src/test/20261108_new_method_quick_checks/run_n6c.py`, `test_seeds.py`).
- **GO** = pooled over the three seeds (one cluster per painting across seeds), R@1 and condition gain both have 95%
  lower bounds above 0 against **each of** cosine, RCA, B (or B') and the matched counterpart. All of these except RCA
  have gain 0, so gain against them is one number; do not count it several times. Report each seed alone.

## 6. Pitfalls already paid for

- A control that drops more than the condition fakes a pass (N1 on 3 October; caught by the final review).
- The pick rule mean(R@1, gain) rewards trading R@1 for gain; use A′'s min-margin rule.
- Per-episode z-scoring removes per-episode scale (Step 3).
- Δ under condition b is exactly −Δ under a; a condition-antisymmetric term flips B's top two in a k = 2 cascade.
- Ties count as misses; zero-weight episodes give constant rows (`flat_share` in the runners).
- B's cross-fit picks are tuned on the same parity halves a fusion reuses (a small second-order leak).
- B uses centering on the episode's own example items. A reviewer probe found centering on 8 random items gives 16.85
  and on the global mean 17.14, against 17.95. Part of B's lift may come from the value-disjoint episode construction.
  Before the paper, rerun B with a fixed reference set and report both.
- Exploratory looks on seed 42 are not evidence; only the fresh-seed test is. Seed 42 has been reused many times.

## 7. Code, data and environment

| What | Where |
|---|---|
| Episodes, metrics, bootstrap | `src/eval/aspect_episodes.py`, `src/eval/aspect_metrics.py` |
| Nested score, cross-fits, controls, readers, decision functions | `src/eval/aspect_nested.py`, `src/eval/aspect_quick_checks.py` (tests `src/test/test_aspect_quick_checks.py`, 28) |
| Seed-42 context, codes, probes, heads | `src/test/20261108_new_method_quick_checks/run_checks.py` (`EvalContext` via `run_gonogo`, `model_inputs`, `fit_probes`), `run_n6.py` (`fit_heads`, `partition_labels`, `load_posteriors`, `n6_terms`) |
| Base B, fusion and counterpart patterns | `src/test/20261109_fix_diagnostics/diagnose_fixes.py`, `diagnose_counterparts.py` |
| Partitions and banks | `src/test/20261031_pseudo_partitions/results/` (`partitions.npz` SHA-256 cfd57dbb…, `bank_AIC.npz`) |
| A3 checkpoint | `src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt` (SHA-256 dadfef1b…) |
| Seed ledger | `docs/superpowers/episode_seed_ledger.md` (42 development; 43, 45, 47, 48 spent; 44, 46 MLLM probes) |

- **Python:** `/root/miniconda3/envs/CoSiR/bin/python`; tests `... -m pytest <file> -q`; prefix CPU work with
  `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. Never install into the env.
- **GPU:** the local RTX 3090 is shared with other projects; all steps above run on CPU. DAS6 node404 is available to
  this project (cluster-run skill; `watch`, `logs`, `pull` need `--node node404`), for example for the MLLM reader.
- **Git:** main; stage by explicit path; `bin/` and `docs/paper/` stay untracked; push only when the user asks or
  `cluster sync` needs it.
- **Process the user uses:** superpowers writing-plans plus subagent-driven development for implementation; ARS for
  reviews and literature; a whole-branch final review on the most capable model that re-derives every load-bearing
  number, then one fix wave and a scoped re-review (`~/.claude/rules/final-review.md`).
- **Reports:** `docs/reports/auto/v2/<sequence date>_<topic>.md` (next free sequence date 2026-11-10), one row in
  `docs/reports/reports_sum.md`, then `python scripts/check_reports_sum.py`.

## 8. Open items (not blocking)

1. Reports not yet written: the 8B MLLM probe (`2026-11-06`, numbers in its log), the candidates write-up
   (`2026-11-07`), the fix diagnostics (`2026-11-09`; the log has the numbers).
2. The ARS paper intake draft (`docs/paper/`, untracked) is paused.
3. Privileged references (CRL and Qwen3-VL-Embedding told the aspect name), needed for the K3 claim in any branch, are not
   run.


---

# Appendix E. Stage report of 2026-10-04, sections 2 and 14 (verbatim)

## 2. Task, data and protocol

### 2.1 The aspect episode

![Aspect episode](../assets/2026-10-04_aspect_conditioned_similarity_methods/task_schematic.png)

*Figure 1. One aspect episode. The query has values on aspects A and B. Each of the four support pairs (teal) is an
image of one painting with a caption of another that agree on a value of A; the four pairs show four different values.
The four contrast pairs (orange) do the same for B. Candidate p_A shares the query's value of A, p_B its value of B, eleven negatives (grey) share neither; all 13 candidates differ from the query on the third aspect. Swapping
supports and contrasts must move p_B to the top.*

An *aspect episode* (spec §5.1) consists of:

- a **query** taken from one *anchor* row of a selection painting (a *row* is one image of a painting with one
  viewer's caption): the anchor's image when captions are ranked (image to caption, i2t) and its caption when images
  are ranked (caption to image, t2i), so every episode is scored in both directions;
- **4 support pairs**, each an image of one painting with a caption of another painting (*cross-item*), the two sharing
  a value of the conditioned aspect A and differing on B; the four pairs show four different values, never the query's own (*value-disjoint*);
- **4 contrast pairs** built the same way for a second aspect B;
- **13 candidates** in the other modality: p_A shares the query's value of A, p_B its value of B, 11 negatives share
  neither; all 13 candidates differ from the query on the third aspect. All 30 rows of an episode come from 30 different paintings.

Under condition A the target is p_A; swapping supports and contrasts (condition B) makes p_B the target. The aspect is
never named. On ArtELingo the aspects are **emotion** (8 values, the catch-all "something else" removed), **style**
(23 values with at least 30 selection paintings) and **genre** (10 values, from WikiArt via ArtGAN), giving three aspect
pairs: emotion × style, emotion × genre and style × genre. We draw 4,096 episodes per pair, 12,288 per *episode seed*.

Every *episode* in this report is an evaluation episode of this form. One episode holds both conditions (the same
query, candidates and eight pairs, with the roles of supports and contrasts exchanged) and both directions, so it
yields four rankings, and its per-episode metrics average over them (`per_anchor` in `src/eval/aspect_metrics.py`).
Anchors are drawn with replacement, so the 12,288 seed-42 episodes fall on 4,602 anchor paintings, which is why the
bootstrap resamples paintings. The pseudo-aspect episodes that trained method A (Section 5.1) are a separate set built
from scorer-train rows, with k-means clusters in place of labels; no result in this report is scored on them.

The construction controls the example pairs only partly (`_pairs` and `build_aspect_episodes` in
`src/eval/aspect_episodes.py`). Every one of the 16 example rows comes from a painting that has no row with the query's
value of A or of B, so the example items are drawn from the same "shares neither value" population as the 11
negatives (Section 11.3 returns to this). The third-aspect control applies to the 13 candidates only. The pairs are
not constrained on the third aspect, so an emotion pair in an emotion × style episode may also share a genre, and
anything correlated with the shared value (artist, period, palette) is free; how often pairs share the third aspect
has not been counted. The four support pairs show four different values, and what they have in common is that each
pair agrees within itself on A. This is why the readers of Sections 5 to 14 compare agreement within pairs; a prototype
of the supports would average four different values.

### 2.2 Data

ArtELingo (English): 308,723 image and caption rows over 61,402 paintings. Every item is represented by frozen CLIP
ViT-B/32 image or caption features; no backbone is fine-tuned.

The aspects are labelled at different levels (`artelingo_aspect_labels` in `src/data/artelingo_splits.py`). A row is
one viewer's caption with that viewer's emotion, and a painting has about five rows (308,723 / 61,402 ≈ 5.0), which can
carry different emotions. Style and genre belong to the painting and are shared by all its rows (genre is looked up by
painting name). For emotion the episode builder therefore mixes two strengths of rule. Membership is per row: two rows
share emotion v when both are labelled v. The image half of an emotion pair is then "a painting that at least one viewer
felt v about", labelled through a caption the episode does not show, while the caption half states the emotion of the
viewer who wrote it; and two rows that differ on emotion can still share it through other viewers of the same paintings.
Exclusion is per painting: p_B, the negatives and the example rows come from paintings that no viewer labelled with
the query's emotion (in style × genre episodes, where emotion is the third aspect, so do all 13 candidates), which
keeps false negatives out. How mixed the paintings' emotions are has not been measured
(spec E12 planned an annotator-agreement count; it was not run).

Rows are split by painting:

*Table 2. Splits (spec §2.3).*

| Split | Rows | Paintings | Use |
|---|---:|---:|---|
| scorer-train | 183,694 | 36,518 | training any model or probe |
| selection | 32,413 | 6,451 | all development and test episodes of this stage |
| val | 30,872 | | unused |
| held | 61,744 | 12,281 | reserved for the final paper test; not read for aspect episodes |

Episodes are drawn from selection rows only, so every development or test number below is on paintings no model was
trained on. Different episode seeds reuse the same 6,451 selection paintings: a fresh seed is a fresh draw of episodes,
not of paintings.

### 2.3 Metrics

For each ranking (two directions, image to caption and caption to image, times two conditions):

- **R@1**: the target ranks strictly first; ties count as misses. Chance is 1/13 = 7.69%.
- **Other-aspect rate**: the other aspect's candidate ranks first.
- **Condition gain** = R@1 − other-aspect rate. It is exactly 0 for any scorer that ignores the condition, whatever its
  R@1.
- **Either rate** = R@1 + other-aspect rate: how often *some* aspect-sharing candidate ranks first.

Each ranking ends one of three ways, and R@1 and the other-aspect rate are the shares of rankings that end the first
and the second way:

| What ranks strictly first | Adds to R@1 | Adds to the other-aspect rate |
|---|---|---|
| the target (p_A under condition A, p_B under B) | 1 | 0 |
| the other aspect's candidate (p_B under A, p_A under B) | 0 | 1 |
| a negative, or a tie at the top | 0 | 0 |

The code stores R@1 and the other-aspect rate; either rate and gain are derived from them. Gain counts first places
only, as a net difference. The pairwise *swap* statistic of spec §5.1 (p_A above p_B under A and p_B above p_A under B)
is a different metric: a ranking in which the target beats the other aspect's candidate while a negative ranks first
adds nothing to gain.

From the last two, **R@1 = (either + gain) / 2**. This identity organises the whole stage: a scorer can raise R@1 by
finding aspect-sharing candidates more often or by choosing the conditioned one more often, and a scorer that reads the
condition must not lose more of the first than it gains in the second.

The identity itself is algebra, ((R + O) + (R − O)) / 2 = R; its use is that it splits R@1 into two parts that scorers
move separately. Equivalently, R@1 = either × q, where q is the share of aspect-sharing first places that go to the
target. A scorer that ignores the condition ranks identically under A and B, so each of its aspect-sharing first places
is right under exactly one condition: q = 0.5 and the gain is 0 (cosine: either 25.92, R@1 12.96). Reading the condition
raises q (N6's reader alone: 16.51 / 28.61 = 58%; label probes told the aspect: 30.66 / 40.27 = 76%), and the either rate
is the base that q multiplies. The identity also holds for differences, ΔR@1 = (Δeither + Δgain) / 2, so a fused score
clears its control by 0.5 R@1 only if its gain exceeds its either loss by 1 point.

### 2.4 Uncertainty, seeds and pre-registration

Intervals resample anchor paintings (clusters), so an episode's dependence on its painting is respected. Development
used episode seed 42; tests used fresh seeds (43 for method A; 45, 47 and 48 for N1). A ledger records every use of
every seed. Each decision rule was committed to git before the numbers it governs were read; two exceptions are disclosed in Section 12 (Addendum 1 was committed two minutes after the N1 test had been computed but before its outputs were read; N6c's comparison with its declared control (cosine plus A3's centered term, R@1 17.94; +0.55) was already known from an exploratory run), and each stage ended
with a whole-branch review on the most capable model that re-derived every load-bearing number from the stored
per-anchor arrays with independent code.

### 2.5 Baselines and controls

- **Backbone only (cosine)**: CLIP cosine of query and candidate; condition gain 0 by construction.
- **Raw metric from pairs** (E1, Section 4): nine ways to turn example pairs into a metric (KISSME, RCA, Xing and CVS from the literature; the others are generic or our own). The best on seed
  42, **RCA**, is the *GO bar*.
- **Condition-free control**: the method's own score with the condition removed. A **matched** control removes only the
  condition and keeps every other ingredient of the score (same terms, same weight budget). Section 9.2 shows why the
  word "matched" matters.
- **Best condition-free score (B)**: the strongest condition-free score measured in the stage (R@1 18.34 on seed 42;
  defined in Section 14). B is a control built from our own ingredients, not a published baseline. For a new
  configuration the GO comparators are cosine, RCA, B (extended to B′ by any new condition-free ingredient) and the
  configuration's matched counterpart, which can sit above B (Section 11.3).
- **Label-probe reference** (diagnostic, never a method): logistic probes trained on the evaluation labels of 60,000
  scorer-train rows; an item is represented by its class posteriors.

*Sources: spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` §2 to §6 and §10;
[E0 evaluation setup](../auto/v2/2026-10-29_aspect_eval_setup.md); episode ledger `docs/superpowers/episode_seed_ledger.md`;
code `src/eval/aspect_episodes.py`, `src/eval/aspect_metrics.py`, `src/data/artelingo_splits.py`; episode and painting
counts from `src/test/20261109_fix_diagnostics/results/diagnose_fixes.txt`.*


## 14. What a passing method has to do, and a first diagnosis

The numbers point to one requirement: add selection without losing aspect-finding. A method must add more condition gain than it loses either rate against the strongest condition-free score B (R@1 18.34, either 36.68): the told partition does (gain 4.56 for an either loss of 1.57); N6's reader does not by a reliable margin (0.79 for 0.41).

To find which part blocks this, we ran exploratory diagnostics on seed 42 after the stage (`src/test/20261109_fix_diagnostics/`; they decide nothing). Each variant was added on top of B (N6c's matched control: cosine, A3's centered uniform term and the averaged heads, max-R@1 cross-fit; R@1 18.34, either 36.68) with A′'s min-margin cross-fit against B. Each reader also got a condition-free counterpart T_cf, the mean of its two condition scores, fused on B with a max-R@1 cross-fit (the condition-free counterpart of the min-margin rule); the difference between the fused reader and its counterpart is the margin from reading the condition (the counterpart's gain is 0, so the margin's gain equals the fused gain).

*Table 10. Exploratory, seed 42: what N6's reader needs to clear the condition-free bar. "Fused" is the variant fused on B; "counterpart" is T_cf fused on B; all columns in pp.*

| Variant added on top of B | Fused minus B: R@1 | gain | either | Counterpart minus B: R@1 | either | Fused minus counterpart: R@1 | either |
|---|---|---|---|---|---|---|---|
| N6 reader (trained without ArtELingo labels, hard pick) | +0.19 [0.01, 0.38] | 0.79 [0.52, 1.05] | −0.41 [−0.70, −0.12] | +0.05 [−0.05, 0.15] | +0.10 [−0.09, 0.29] | **+0.14 [−0.04, 0.32]** | −0.51 [−0.76, −0.25] |
| contrastive reader (picked partition minus contrast-like partition) | +0.07 [−0.08, 0.22] | 0.53 | −0.39 | not computed | not computed | not computed | not computed |
| same heads, partition told (emotion to affect, style and genre to image) | +1.50 [1.22, 1.79] | 4.56 [4.21, 4.93] | −1.57 [−2.03, −1.08] | +0.35 [0.16, 0.55] | +0.70 [0.32, 1.09] | **+1.14 [0.90, 1.41]** | −2.27 [−2.66, −1.88] |
| label-probe reference (diagnostic, not a bound) | +12.28 [11.81, 12.75] | 20.55 [19.99, 21.09] | +4.00 [3.35, 4.68] | +5.62 [5.27, 5.97] | +11.24 [10.53, 11.95] | +6.66 [6.31, 7.00] | −7.23 [−7.70, −6.76] |

The heads are good enough; the reader's choice of partition is not. Told the right partition, the same heads trained without ArtELingo labels clear B by 1.50 R@1 against +0.19 for N6's reader fused the same way. Part of that comes from the condition-free side: the told term with the condition removed already adds 0.35 [0.16, 0.55] on its own. The margin from reading the condition is therefore +1.14 [0.90, 1.41] R@1 for the told partition against +0.14 [−0.04, 0.32] for N6's label-free reader. The told term pays 2.27 either for 4.56 gain, so its gain outruns its either cost; it is not free. The label-free reader picks the told partition in 52% of rankings and is right under both conditions in only 28% of episodes. In those episodes the fusion gains +1.16 [0.80, 1.54] R@1 and 2.21 gain with no detectable either loss (+0.12 [−0.43, 0.66]); in the others it loses 0.61 [0.29, 0.94] either. The wrong picks are concentrated where the partitions are ambiguous: under the style condition of style × genre the reader picks the image partition in only 17% of rankings, and even the told mapping cannot separate style from genre, because both live in the image clusters (Table 4). A contrastive reader and a top-k cascade with N6's reader did not help (cascades lost 0.55 to 2.59 R@1).

Read as oracles, the three rows of Table 10 that have a matched counterpart form layers. With nothing perfect, N6's
reader adds +0.14 R@1 over its counterpart; a perfect reader on the current label-free heads (the told partition) adds
+1.14; a perfect reader on a representation in the evaluation taxonomy (label probes told the aspect) adds +6.66. The
told row is a reader oracle for these three partitions only: its mapping was chosen with the labels (from AMI), and its
gain on style × genre is exactly 0. What an oracle picks is a block, one partition's 64-class posterior, and not a
single factor, because value-disjoint episodes need every value of an aspect to live on shared factors (spec §6). Method
A's 32 learned factors carry no aspect identity that could be told, so no reader oracle exists for them; L3 (Section 6)
is a different diagnostic, a better representation read by the same rule. We score oracles as R@1 margins over the
matched counterpart and not by gain: the told partition minus the other partition reaches a gain of 7.03 on its own
while its R@1 falls to 11.31.

Caveats on this analysis:

- For each partition h, the reader's statistic Δ_h (mean within-pair agreement over the support pairs minus over the contrast pairs) under condition b is exactly the negative of its value under condition a, and the told mapping sends style and genre to the image partition, so no style × genre episode can have both picks right. The 28% both-correct episodes come from emotion × style and emotion × genre only (1,576 and 1,883 of 3,459 episodes, none from style × genre), and the subset is selected after the fact.
- On those same 3,459 episodes the told-partition fusion, whose picks are right everywhere, lost 3.01 [2.09, 3.93] either, so the small either change of N6's reader there holds only at the small weight the cross-fit gave it.
- B's cross-fit picks were tuned on the same parity halves the fusion reuses (a small second-order leak), and seed 42 was reused for every analysis.
- N6's reader on B nominally clears B (+0.19 [0.01, 0.38]) in this exploratory look, but the lower bound sits at 0 under another bootstrap stream; it is not a pass.
- The pick accuracy of any reader that takes the argmax of Δ_h and uses the told mapping is capped at 83.3% pooled (100% on emotion × style and emotion × genre, 50% on style × genre, because of the antisymmetry), and both picks can be right in at most 66.7% of episodes; the label-free reader reached 52.4% pick accuracy and 28.1% both-correct.

The handoff (`docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md`) turns this into a plan: a reader that
chooses the aspect correctly (calibrated on label-free pseudo-aspect episodes), a partition that separates style from
genre, a matched control for each, and a test on fresh seeds 49 to 51.



---

# Appendix F. Two project lessons recorded by the user (verbatim, without file metadata)

## F.1 Matched-control lesson


On 2026-10-04, N1-nested-A3 passed its declared control on seed 42 (R@1 +0.24 [0.09, 0.39]) because the control
z(cos) + σ·z(T_u) used the uncentered uniform term, while N1 also centred the query on the episode's 8 example items.
The final whole-branch review found it: against the matched control (same nested family, N1's term replaced by its
condition-free centered version) N1 lost 1.15 R@1. The same check later stopped N6c (gate vs C2, +0.15 [−0.06, 0.38]).

**Why:** R@1 = (either + gain) / 2; any condition-free change that raises the either rate (centering, an extra head term)
inflates R@1 against a control that lacks it, so the "pass" measures the extra ingredient, not reading the condition.

**How to apply:** for every new conditioned scorer, define its control as the identical score with only the condition
removed (uniform weights in place of the reader, same terms and weight budget, max-R@1 cross-fit) and put it in the
pre-registered pass/GO comparators; also report the strongest condition-free score measured so far (18.34 on seed 42 as
of 2026-10-04). Related: [[v2-publication-plan-pending]], [[final-review-catches-real-issues]].

## F.2 Seed-handling preference


On 2026-10-04, after I proposed guarding a "single seed-45 look" and confirming picks on extra seeds, the user said:
"I think you put too much effort on seeds. We run on single seeds and test on multiple, no need to emphasize too much
on which seed might be lucky/bad."

**Why:** the user considers testing on several fresh episode seeds sufficient protection; elaborate single-look rules,
winner's-curse gates and seed bookkeeping in plans and messages cost attention without changing decisions.

**How to apply:** in CoSiR plans, develop/pick on one episode seed (42) and evaluate the pick on multiple fresh seeds
(report each and pooled) against the pre-stated comparators. Keep the seed ledger as a plain record, but don't make
seed reuse or "spending" a test seed a headline concern in plans, briefings or decision tables. Pre-registering the
decision rule before testing still stands. Related: [[v2-publication-plan-pending]].
