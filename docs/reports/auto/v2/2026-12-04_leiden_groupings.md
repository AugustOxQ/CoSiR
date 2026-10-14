# Leiden ties k-means on the screen and bar margin; AFF on Leiden is 0.56 lower on its own R@1; H1 fails; kept as records

> 2026-10-14 · Spec: none (side check outside the research loop; the design is the handoff,
> [2026-10-14-leiden-groupings-handoff.md](../../../superpowers/handoffs/2026-10-14-leiden-groupings-handoff.md), and
> the decisions in the run log) · Decision rule: the screen's pass bar (below) · Run log:
> [20261204_leiden_groupings_log.md](../../../../src/test/20261204_leiden_groupings/20261204_leiden_groupings_log.md)

## Results brief

**What we tried.** We screened Leiden at 64 and H1 alone, retrained H1 on scorer-train paintings, and rebuilt AFF with Leiden image and caption groupings. Affect stayed unchanged, already Leiden.

**Outcome against the bar.** Leiden at 64 ties k-means on the screen within the noise, but misses its pass bar. AFF is 0.20 lower on the bar margin, within the noise. Its own R@1 is 0.56 lower, with an interval below zero. R@1 points on 12,288 seed-42 development episodes; brackets give 95% intervals:

| AFF result | Leiden at 64 | k-means | Paired Leiden minus k-means |
|---|---|---|---|
| Bar margin | +0.50 [+0.30, +0.71] | +0.70 [+0.46, +0.94] | −0.20 [−0.44, +0.05] |
| Own R@1 | 18.58 | 19.14 | −0.56 [−0.80, −0.30] |

Retrained H1 over its matched control: −0.27 [−0.48, −0.07].

Its reader minus the k-means reference: −0.63 [−0.94, −0.32]. It fails; no rebuild, by the cheap rule. The user keeps all results as records.

**What it does not show.** Seed-42 development only, the seed every setting was picked on. CLIP only, no held rows, one Leiden seed, including the rebuild, and one group count (64). Screen rows use a rule reader, no practice bank, no gate, no τ; the rebuild is the AFF comparison. Old H1 saw held paintings as features; its labels cannot feed a held read. Retrained H1 used scorer-train paintings only.

**Options for next.** Keep records at no run cost, or reopen Leiden with more seeds and group counts at further compute cost. A held comparison needs a new decision. The user keeps Leiden as a record and a minor open direction; it was added to the parked ideas. Status: kept as a record (a parked idea).

**Your reading.** First screens (user, 2026-10-14 01:55, before seeing ours): "If leiden ties with k-means we just go with leiden, because this makes the design coherent". H1 was expected to be bad "because this is too different from the trained one". The user requested both follow-ups.

Rebuild (user, 2026-10-14, before seeing ours): "I don't think the way we do grouping is that important, given big issue exposed just now". Keep Leiden as a record that it "can also do the job", a minor open direction, and H1's one-grouping failure as a record. The "big issue" is the k-shot entry's question: are aspect labels needed in training to separate style from genre?

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The rebuild ties k-means on the bar margin within the noise. Leiden lowers B and B′(A0) too, so AFF's own R@1 falls more than its margin. Leiden is a coherent alternative: 0.20 lower on the bar margin, within the noise; 0.56 lower on AFF's own R@1. The two rebuild readings agree. The first-screen readings agreed on the tie but differed on the choice; we advised stopping, the user chose the follow-ups.

**Certainty.** Sure of the R@1 drop on this seed: its interval excludes zero. The margin difference is within the noise. One seed, development data only.

**Our recommendation.** Keep k-means and retain both checks as records. H1 alone still fails after retraining. Its painting-level labels may miss caption-level emotion; one grouping may fail to follow a change of aspect.

</details>

## What ran

The first screens missed the pass bar; the user then requested both follow-ups.
The screen pass bar was a reader margin at least the reference's +0.35 R@1 points (agent default).

- **Groupings** (`lg_screen.py group`, CPU, 12.5 min). The same k = 20 kNN union graphs as round 3's resolution-1.0
  run: image at painting level (36,518 nodes), caption at row level (183,694 nodes). Leiden RBConfiguration, seed 42,
  groups under 200 rows merged. Resolution search for 64 groups: r = 1.0, doubling, then bisection.
  - image: r = 5.047, 64 groups, 12 resolutions tried.
  - caption: r = 2.375, 64 groups, 7 resolutions tried.
  - H1: `train_community_post` of the buddy-line snapshot, by painting name. All 36,518 scorer-train paintings are in
    the snapshot (61,402 paintings). 19 groups, the smallest 1,321 rows, the largest 11.9% of the rows.
- **Heads** (5 min). Round 3's `fit_one_head` (60,000-row draw, logistic regression on frozen CLIP features) per new
  grouping.
- **Screen** (7 min). Round 3's `evaluate_general` with three arms: k-means reference, Leiden at 64
  and H1 (one grouping, every aspect told to it). Built-in checks, all passed: the reference reproduces round 3's
  stated reference margins; the image graph at r = 1.0 reproduces round 3's image grouping (SHA-256); in H1 the reader and
  told terms are equal.

- **H1 retrained.** Scorer-train paintings only; 20 communities, AMI 0.49 with the old mapped labels.
  Training took 22 s on GPU. The screen again gave the conditioned term cross-fit weight 0 in both folds.
  No rebuild, by the cheap rule (screen first: agent default under the user's "cheap is best").
- **AFF rebuilt on Leiden.** The user requested the rebuild despite the screen result.
  Image and caption k-means were replaced by Leiden at 64 everywhere, B and B′(A0) included (agent default).
  The affect grouping stayed unchanged, already Leiden in AFF. Only image and caption groupings changed.
  Picks and thresholds were refit; CLIP's content graph was reused.
  Run: 2026-10-14 02:07 to 2026-10-14 02:54. The AFF GPU stage took 749 s and used 3.93 GiB.
  Thresholds τ_0..3: 0.00005, 0.142, 0.342, 0.611.

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
| H1 alone (19; one grouping, so told and reader are the same) | −0.46 [−0.63, −0.28] | −0.46 [−0.63, −0.28] | −0.46 [−0.63, −0.28] (matched control) | 100% (one grouping) |

Retrained H1 alone (20 communities): told and reader −0.27 [−0.48, −0.07] over its matched control.
Its reader is −0.63 [−0.94, −0.32] against the reference; it fails the same screen bar.

### AFF rebuild

Leiden is 0.20 lower on the bar margin, within the noise; AFF's own R@1 is 0.56 lower, with its interval below zero.
The bar margin is over the stronger of B′(A0) and the matched control; here B′(A0) is stronger.
The same development episodes and 95% interval format apply.

| Metric (R@1 points) | Leiden at 64 | k-means | Paired Leiden minus k-means |
|---|---|---|---|
| Bar margin / over B′(A0) | +0.50 [+0.30, +0.71] | +0.70 [+0.46, +0.94] | −0.20 [−0.44, +0.05] |
| AFF's own R@1 | 18.58 | 19.14 | −0.56 [−0.80, −0.30] |
| Over its matched control | +0.53 [+0.34, +0.72] | +0.74 [+0.52, +0.96] | −0.21 (point only, no interval) |

The k-means matched-control margin comes from the k-shot check, which exactly reproduces round 6's seed-42 scores.
Source: `src/test/20261203_kshot_supports/results/kshot_seed42.json`, `k → 4 → differences → AFF_minus_CF_r1`.
The matched-control paired point is the difference of the two margins; no paired interval is available.
The k-means reference reproduces AFF's stated development margin, +0.70 [+0.46, +0.94].
Source: `results/rebuild/lg_result.json`; summary: the [follow-up handoff](../../../superpowers/handoffs/2026-10-14-leiden-groupings-2-handoff.md), section 3.

### First-screen comparisons

Paired differences from the k-means reference, R@1 points with 95% intervals:

| Screen arm | Told minus reference | Reader minus reference |
|---|---|---|
| Leiden at 64 | +0.06 [−0.19, +0.31] | −0.14 [−0.36, +0.09] |
| Old H1 alone | −2.10 [−2.43, −1.78] | −0.81 [−1.09, −0.53] |
| Leiden at r = 1.0 (2026-11-16) | — | −0.32 [−0.57, −0.06] |

Head accuracy on 10,000 held-out scorer-train rows:

| Head grouping | Image head accuracy (%) | Caption head accuracy (%) | Majority group share (%) |
|---|---|---|---|
| Image Leiden | 87.3 | 27.8 | 4.7 |
| Caption Leiden | 28.9 | 82.1 | 6.1 |
| Old H1 | 43.7 | 35.4 | 12.0 |

## Analysis

**The results are consistent with group count, rather than the method, causing the earlier Leiden loss.** This explanation is not tested.
There are two settings and one seed. The earlier caption run used Modularity; this run used RBConfiguration.

| Leiden setting | Image / caption groups | Told margin (R@1 points) | Reader margin (R@1 points) |
|---|---|---|---|
| Earlier, r = 1.0 | 17 / 30 | +1.39 | +0.04 |
| Matched count | 64 / 64 | +1.70 | +0.22 |

At 64 the told margin ties k-means. Reader pick shares are close: Leiden 53.8%, k-means 54.7%.
The reader point is lower with Leiden, but its paired interval includes zero.

**H1 adds nothing that depends on the condition.** The nested cross-fit picked weight 0 for H1's conditioned term in
both folds, so H1's fused score equals B on every episode. Its condition-free version got small positive weights and
gained +0.51, +0.48 and +0.38 over B on the three aspect pairs. The negative margin is that gain of the matched
control.

**The lower comparison scorer explains the smaller margin difference.** Leiden's B′(A0) is 18.08 R@1 points.
We computed it as AFF's own R@1 minus its margin over B′(A0), and checked it against the saved comparison score.
K-means' B′(A0) is 18.44, the round 6 seed-42 value.
Replacing the groupings lowers B and B′(A0) too. AFF's own R@1 is 0.56 lower, with its interval below zero.
Leiden is 0.20 lower on the bar margin, within the noise.

**Retraining does not rescue H1 alone.** The retrained communities differ from the mapped ones (AMI 0.49),
but the cross-fit still picks weight 0 in both folds. This supports the one-grouping failure as a record.
It does not test H1 alongside AFF's other groupings.

## Decision

AFF's fixed recipe does not change: k-means stays in r9 and the paper, with no Leiden held read.
The user said to "keep this just as a record".
We interpreted this as no recipe change and stated that interpretation to the user.

The user keeps Leiden as a record and a minor open direction; it was added to the parked ideas.
Leiden in AFF is kept as a record (a parked idea). H1's one-grouping failure is recorded too.
The user requested both follow-ups; replacing the groupings in B and B′(A0) was an agent default.
The H1 screen-first rule was an agent default under the user's "cheap is best".
Source: [direction log, Leiden side check](../../../superpowers/direction_log.md).

## Deviations

- The caption graph used RBConfiguration with a resolution. The 2026-11-16 run used `detect_communities` (Modularity).
  At r = 1.0 the two gave 30 groups each after merging.
- The resolution search is simpler than round 8's rule M3 (doubling and bisection instead of a 30-point grid). No
  stability seeds 43 and 44.
- The pass bar (reader margin at least the reference's) is an agent default, set before the run.
- H1 retraining used plateau recall on scorer-train, rather than held-out rows; it ran all 200 epochs.
- H1 affect inputs were stored per-row GoEmotions probabilities, mean-pooled over scorer-train rows.
  They were not re-extracted. Text and image inputs per painting came from scorer-train rows only.
  These are the deviations listed in `results/h1r.json`.
- The rebuild score step's summary crashed because `point_ci` returns a dict.
  After the fix, the result was recomputed from saved arrays with `--step result`.

## Files and storage

The kept outputs reproduce the result table without the rebuild intermediates.

- `src/test/20261204_leiden_groupings/`: `lg_screen.py`, `lg_h1r.py`, `lg_rebuild.py` and the run log.
- `results/`: 52 MB. Main arrays: `lg_heads.npz` 42 MB, `lg_group.npz` 7 MB, `h1r_labels.npz` 3 MB.
  It also keeps `h1r.json`, logs, `lg_eval.txt` (includes retrained H1) and the first screen, `lg_eval_v1.txt`.
- `results/rebuild/`: 1.2 MB. Kept: `A3_seed42.pt`, `lg_seed42_arrays.npz` and the JSON records,
  including `lg_result.json`, `rc_tau.json` and `build_record.json`.
  `lg_rebuild.py --step result` recomputes the table from the kept outputs.
- The user approved deletion of rebuild intermediates on 2026-10-14: heads, banks, posteriors, graph and partitions, about 400 MB.
  The builder's check runs `_check` and `_check2` (37 MB) were also removed.
- The frozen r9 worktree `/project/CoSiR-lg-r9` (82 MB) is scheduled for removal after this fix round.
  It is recreatable at `b84db1e`.
- Nothing of 1 GB or more. No ledger row for the first screen.
