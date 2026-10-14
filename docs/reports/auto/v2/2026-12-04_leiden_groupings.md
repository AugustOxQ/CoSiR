# Leiden groupings side check: Leiden at 64 groups ties k-means, and the attention-H1 communities alone carry no condition signal

> 2026-10-14 · Spec: none (side check outside the research loop; the design is the handoff,
> [2026-10-14-leiden-groupings-handoff.md](../../../superpowers/handoffs/2026-10-14-leiden-groupings-handoff.md), and
> the decisions in the run log) · Decision rule: the screen's pass bar (below) · Run log:
> [20261204_leiden_groupings_log.md](../../../../src/test/20261204_leiden_groupings/20261204_leiden_groupings_log.md)

## Results brief

**What we tried.** Two cheap screens of other groupings, on the told-oracle screen that once picked AFF's affect
grouping (round 3). (1′) Leiden communities on CLIP's image and caption graphs, at the resolution that gives 64 groups,
the count of the k-means groupings they replace. (2) The 19 communities of attention-H1, the buddy line's learned
student, as the only grouping, as in the buddy vs PercepT experiment. The user asked for both on 2026-10-14.

**Outcome against the bar.** Neither passes, so no full AFF rebuild runs. The bar (agent default): the screen's reader
margin at least the reference's, +0.35 R@1 points. Margins over the matched control, 12,288 seed-42 development episodes:

| Groupings | Told margin | Reader margin | Reader minus reference |
|---|---|---|---|
| reference: affect, image and caption k-means (64 each) | +1.64 [+1.37, +1.92] | +0.35 [+0.15, +0.57] | — |
| (1′) affect, image and caption Leiden (64 each) | +1.70 [+1.41, +2.01] | +0.22 [+0.06, +0.38] | −0.14 [−0.36, +0.09] |
| (2) attention-H1 alone (19 groups) | −0.46 [−0.63, −0.28] | −0.46 [−0.63, −0.28] | −0.81 [−1.09, −0.53] |

**What it does not show.** The screen is not AFF: a rule reader, no practice bank, no gate, no τ. Seed 42
development episodes only, one Leiden seed. H1 was mapped from a student that saw the held paintings (features only),
so its labels can never feed a held read; H1 was tried only as the only grouping (user).

**Options for next.**
- Stop both checks here. No cost.
- Run the full AFF rebuild on Leiden at 64 groups anyway (about 1 h plus a 1–2 h adapter). It tells whether AFF's
  learned reader uses Leiden differently from the screen's rule reader.
- Retrain H1 on the scorer-train paintings only (port plus GPU, hours). It would matter only if the mapped labels
  undersold H1.

**Your reading.** <left for the user>

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** At a matched group count, Leiden content groupings are as good as k-means when the grouping is told
(+0.06, within the noise). The reader gets a little less from them (−0.14, within the noise). So the loss of the
resolution-1.0 run (17 and 30 groups) came mostly from the group count. There is no reason to switch. H1 alone
carries no condition signal that the steering can use: the cross-fit gave its conditioned term zero weight in both
folds. Fairly sure on both, for the screen.

**Our recommendation.** Stop both checks. Add one line to the model card's groupings section: Leiden at 64 groups
ties k-means. Our view.

</details>

## What ran

- **Groupings** (`lg_screen.py group`, CPU, 12.5 min). The same k = 20 kNN union graphs as round 3's resolution-1.0
  run: image at painting level (36,518 nodes), caption at row level (183,694 nodes). Leiden RBConfiguration, seed 42,
  groups under 200 rows merged. Resolution search for 64 groups: r = 1.0, doubling, then bisection.
  - image: r = 5.047, 64 groups, 12 resolutions tried.
  - caption: r = 2.375, 64 groups, 7 resolutions tried.
  - H1: `train_community_post` of the buddy-line snapshot, by painting name. All 36,518 scorer-train paintings are in
    the snapshot (61,402 paintings). 19 groups, the smallest 1,321 rows, the largest 11.9% of the rows.
- **Heads** (5 min). Round 3's `fit_one_head` (60,000-row draw, logistic regression on frozen CLIP features) per new
  grouping.
- **Screen** (7 min). Round 3's `evaluate_general` with three arms: the reference (round 3's A0), A3r (Leiden at 64)
  and H1 (one grouping, every aspect told to it). Built-in checks, all passed: the reference reproduces round 3's
  stated A0 margins; the image graph at r = 1.0 reproduces round 3's image grouping (SHA-256); in H1 the reader and
  told terms are equal.

The **told margin** uses the grouping that matches the episode's aspect by the labels (a privileged ceiling): emotion
to affect, style and genre to the image grouping (to H1 in the H1 arm). The **reader margin** uses the screen's rule
reader, which picks one grouping per condition without labels.

## Results

R@1 points over each arm's matched control; the bar margin is over the strongest of B′ and the matched control.

| Arm | Told | Reader | Bar margin | Reader picks the told grouping |
|---|---|---|---|---|
| reference (k-means 64) | +1.64 [+1.37, +1.92] | +0.35 [+0.15, +0.57] | +0.31 [+0.10, +0.53] (B′) | 54.7% |
| Leiden at 64 | +1.70 [+1.41, +2.01] | +0.22 [+0.06, +0.38] | +0.22 [+0.06, +0.38] (matched control) | 53.8% |
| Leiden at r = 1.0 (17 and 30 groups; 2026-11-16) | +1.39 [+1.09, +1.69] | +0.04 [−0.20, +0.27] | — | — |
| H1 alone (19) | −0.46 [−0.63, −0.28] | −0.46 [−0.63, −0.28] | −0.46 [−0.63, −0.28] (matched control) | 100% (one grouping) |

Paired against the reference: Leiden at 64, told +0.06 [−0.19, +0.31], reader −0.14 [−0.36, +0.09]; H1, told −2.10
[−2.43, −1.78], reader −0.81 [−1.09, −0.53]. Leiden at r = 1.0 (2026-11-16): reader −0.32 [−0.57, −0.06].

Head accuracy on 10,000 held-out scorer-train rows (image head / caption head; majority group share in brackets):
image Leiden 87.3 / 27.8 (4.7), caption Leiden 28.9 / 82.1 (6.1), H1 43.7 / 35.4 (12.0).

## Analysis

**Group count, not the community method, explains the earlier Leiden loss.** From 17 and 30 groups to 64, the told
margin rose from +1.39 to +1.70 and the reader margin from +0.04 to +0.22. At 64 the told margin ties k-means. The
reader's pick share is about the same (53.8% against 54.7%). The reader's point is lower than with k-means, but the
paired interval includes zero.

**H1 adds nothing that depends on the condition.** The nested cross-fit picked weight 0 for H1's conditioned term in
both folds, so H1's fused score equals B on every episode. Its condition-free version got small positive weights and
gained +0.51, +0.48 and +0.38 over B on the three aspect pairs. The negative margin is that gain of the matched
control. Two likely causes (our reading): H1's labels are per painting, while emotion differs between captions of one
painting; and one grouping that mixes content and affect cannot follow a change of aspect between conditions.

## Deviations

- The caption graph used RBConfiguration with a resolution. The 2026-11-16 run used `detect_communities` (Modularity).
  At r = 1.0 the two gave 30 groups each after merging.
- The resolution search is simpler than round 8's rule M3 (doubling and bisection instead of a 30-point grid). No
  stability seeds 43 and 44.
- The pass bar (reader margin at least the reference's) is an agent default, set before the run.

## Files and storage

- `src/test/20261204_leiden_groupings/`: `lg_screen.py`, the run log and `results/lg_eval.txt` (committed);
  `results/lg_*.json`, `lg_*.npz` and the stage logs `run_*.log` (44 MB, untracked; kept as the run's outputs).
- Nothing of 1 GB or more; no ledger row.
