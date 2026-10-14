# K-shot example pairs: AFF keeps its margin with 3 pairs, loses most of it with 2, and falls below the bar with 1

> 2026-10-14 · Spec: none (small loop; the design is the job brief, [CODEX_BRIEF.md](../../../../src/test/20261203_kshot_supports/CODEX_BRIEF.md), and its k = 1 follow-up, [CODEX_BRIEF_k1.md](../../../../src/test/20261203_kshot_supports/CODEX_BRIEF_k1.md)) · Decision rule: none (descriptive) · Run log: [20261203_kshot_supports_log.md](../../../../src/test/20261203_kshot_supports/20261203_kshot_supports_log.md)

## Results brief

**What we tried.** We gave AFF and its comparators fewer example pairs per episode: k = 1, 2, 3 or 4 support pairs and
the same number of contrast pairs, instead of 4 and 4. Every scorer stayed frozen at its seed-42 settings. The query
and the 13 candidates did not change. The user asked for this small k-shot check on 2026-10-14, and added k = 1 after
the first read.

**Outcome against the bar.** There is no bar; the check decides nothing. R@1 points, 12,288 seed-42 development
episodes, 95% intervals over 4,602 anchor paintings:

| Example pairs | AFF − B′(A0), R@1 | AFF − matched control, R@1 | AFF's condition gain |
|---|---|---|---|
| 4 (today) | +0.70 [+0.46, +0.94] | +0.74 [+0.52, +0.96] | +3.11 [+2.78, +3.46] |
| 3 | +0.63 [+0.40, +0.87] | +0.71 [+0.48, +0.92] | +3.10 [+2.77, +3.43] |
| 2 | +0.19 [−0.06, +0.44] | +0.36 [+0.12, +0.59] | +2.58 [+2.24, +2.92] |
| 1 | −1.73 [−2.05, −1.41] | −1.25 [−1.55, −0.96] | +1.99 [+1.66, +2.31] |

- From 4 to 3 pairs, AFF's R@1 changes by −0.10 [−0.26, +0.07] and its condition gain by −0.01 [−0.26, +0.24].
- From 4 to 2 pairs, AFF's R@1 falls by 0.56 [0.35, 0.77] and its condition gain by 0.53 [0.21, 0.85].
- From 4 to 1 pair, AFF's R@1 falls by 2.68 [2.37, 2.98] and its condition gain by 1.12 [0.74, 1.52].
- B′(A0) changes little: −0.04 [−0.18, +0.09] at 2 pairs, −0.25 [−0.42, −0.09] at 1 pair.
- The lead over B′(A0) comes from the two emotion pairs. On style × genre AFF is below B′(A0) at every k.

**What it does not show.** Seed 42 is the seed on which every frozen setting was picked, so the levels carry tuning
optimism. The reader, the gate thresholds and every weight were fitted for 4 pairs and kept frozen. So a drop mixes two
effects: less evidence, and a reader and gate tuned for 4 pairs. At 1 pair the reader's two spread features have no
value and were set to carry no information (see "What ran"). The table shows AFF used as it is with fewer pairs; it is
not AFF re-fitted for k pairs. One seed, CLIP only, no held rows. We did not test more than 4 pairs.

**Options for next.** The user chose to stop after k = 1 (2026-10-14). Not run:
1. More pairs (5 to 7) on new episodes with 7 pairs. About 20 to 30 minutes. It shows whether AFF gains past 4 pairs.
2. Re-fit the reader and re-pick the gate and weights at 1 or 2 pairs. About two hours; a diagnostic only. It separates
   the two effects above.

**Your reading.** *(to be recorded)*

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** AFF as it is needs at least 3 example pairs. With 3 it keeps almost all of its margin. With 2 its
margin over B′(A0) falls to an interval that includes zero; about two thirds of that drop (0.38 of 0.56, no interval)
is in the part that reads the condition. With 1 pair AFF is clearly worse than B′(A0) and than its own matched
control. Its gate then opens on both sides (62% of condition-b values at τ_0, against 30% at 4 pairs). So the reader is
confident and often wrong. It still reads the condition (gain +1.99), but the wrong-side steering costs more R@1 than
the right-side steering adds. We are fairly sure of the shape; the levels are development numbers.

**Our recommendation.** Stop, as the user chose. Add one line to the model card's limits: AFF needs at least 3 example
pairs, and with 1 pair it is worse than the condition-free bar. That is our view.

</details>

## What ran

- **Stages and types:** one exploratory stage (small loop): the round 6 scoring path, re-run with the first k support
  and the first k contrast pairs of each episode, k = 1, 2, 3, 4. The values of k are nested and paired.
- **The rule for k = 1** (agent default, 2026-10-14): the reader's features include the sample spread (ddof 1) of the
  support agreements and of the contrast agreements, per grouping; these 6 of the 18 features have no value with one
  pair. Each half-reader gets its own scaler means for them, so they scale to exactly 0 and add nothing to its logits.
  The other features (S, C, Δ, and the share of pairs whose top groups coincide, 0 or 1) stay as defined. A spread of
  0 was the other option; it was rejected because it tells the reader the pairs agree perfectly.
- **Data and seeds:** the seed-42 development episodes (12,288, 4,096 per aspect pair), checked against round 6's
  per-pair SHA-256s. No held rows or held annotations read.
- **Scorers:** cosine, RCA, B, B′(A0), AFF (cells 39 / 119) and its matched control (cells 149 / 10), all at the
  round 6 seed-42 picks (pick from tune half h scores parity 1 − h). Readers, heads, posteriors, the factor
  checkpoint and the thresholds τ_0 to τ_3 frozen; nothing re-fitted or re-picked.
- **Code:** `src/test/20261203_kshot_supports/` (`ks_core.py`, `run_ks.py`), branch r9-scope. Built and run by Codex
  (gpt-6.1-sol, high), CPU only; the final run, all four k, took 42.9 s.

## Results

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| C1 | k = 4 reproduces round 6's seed-42 frozen scores | AFF 19.14, B 18.34, B′(A0) 18.44; matched control, RCA, cosine also equal | recorded round 6 values | exact | pass |
| C2 | cosine at every k | 12.96 at k = 1 to 4 (raw score arrays identical) | itself | identical | pass |
| C3 | example arrays at k = 1, 2, 3 | k support and k contrast pairs everywhere; factor centring on 2k items | k | exact | pass |
| C4 | the k = 1 rule | the 6 spread columns scale to exactly 0 for each half-reader and condition; at k ≥ 2 the reader's features and probabilities are unchanged | the rule | exact | pass |
| C5 | k = 2, 3, 4 after the k = 1 change | every number and interval equal to the first committed run | first run | exact | pass |

**Verdict under the rule:** none (descriptive).

Pooled R@1 / condition gain (R@1 points):

| Scorer | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|
| cosine | 12.96 / 0.00 | 12.96 / 0.00 | 12.96 / 0.00 | 12.96 / 0.00 |
| RCA | 13.29 / −0.01 | 13.35 / 0.03 | 13.40 / 0.13 | 13.38 / 0.10 |
| B | 17.86 / 0.00 | 18.12 / 0.00 | 18.29 / 0.00 | 18.34 / 0.00 |
| B′(A0) | 18.18 / 0.00 | 18.39 / 0.00 | 18.41 / 0.00 | 18.44 / 0.00 |
| AFF | 16.46 / 1.99 | 18.58 / 2.58 | 19.04 / 3.10 | 19.14 / 3.11 |
| matched control | 17.71 / 0.00 | 18.22 / 0.00 | 18.33 / 0.00 | 18.40 / 0.00 |

AFF − B′(A0) per aspect pair (R@1 points, points only):

| Aspect pair | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|
| emotion × style | +0.93 | +0.82 | +1.09 | +0.98 |
| emotion × genre | −1.78 | +0.84 | +1.46 | +1.45 |
| style × genre | −4.32 | −1.10 | −0.65 | −0.33 |

## Analysis

**About 0.4 of the 0.56 drop at 2 pairs is in AFF's conditioned part.** B and B′(A0) use the example items (the factor
term is centred on them) but not the condition. From 4 to 2 pairs they move by −0.22 and −0.04. AFF moves by −0.56
and its matched control by −0.18. So AFF's margin over its matched control, the part that reads the condition, falls
from +0.74 to +0.36. The rest of the drop comes from AFF's base, B. This split is a difference of point values and has
no interval.

**With fewer pairs the gate is less one-sided; at 1 pair it opens on both sides.** AFF's gate should open mostly in
condition a (supports show aspect A). Gate-open share, all three pairs pooled:

| Example pairs | τ_0, condition a | τ_0, condition b | τ_2, condition a | τ_2, condition b |
|---|---|---|---|---|
| 4 | 80.9% | 29.5% | 59.6% | 11.1% |
| 3 | 76.3% | 30.0% | 58.1% | 13.7% |
| 2 | 72.3% | 33.0% | 56.2% | 18.4% |
| 1 | 83.6% | 61.9% | 80.0% | 57.1% |

Our reading: with 2 pairs, the reader's features (means and spreads over 2 values) are noisier and lie outside the
range the reader was trained on (practice episodes with 4 pairs). Its affect picks then fall more often on the wrong
side, and the thresholds, set for 4 pairs, do not adapt. With 1 pair, S, C and Δ come from a single agreement each and
are more extreme. The gate opens only when the reader picks affect with a large enough top-two margin, so the reader
now does that in most rankings of both conditions. The check cannot separate less evidence from a reader that no longer fits.

**Where the loss sits.** At 2 pairs, emotion × genre carries most of it (+1.45 → +0.84 over B′(A0)), and style × genre
falls further below B′(A0) (−0.33 → −1.10). At 1 pair, emotion × genre and style × genre fall well below B′(A0),
while emotion × style stays at about +0.9. These are points without intervals.

## Deviations

- **The design brief described the example pairs wrongly.** The brief (and `GLOSSARY.md`, "aspect episode") say the
  supports share the query's value of aspect A. In the code (`src/eval/aspect_episodes.py`), each pair is an image and
  a caption from two paintings that share one value of aspect A. The pairs show k different values, never the
  query's own. The episodes used are the real ones, so no number is affected. Their order within an episode is a
  random draw, so the first k pairs are a random subset. The glossary line was corrected on 2026-10-14.
- **k = 1 was added after the first read** (user, 2026-10-14), by a resume of the same Codex session. The rule for the
  spread features is an agent default (see "What ran").
- **RCA's PCA basis was rebuilt** from the frozen 60,000-row scorer-train draw, since no basis was saved. C1 shows it
  reproduces k = 4 exactly; the same basis is used at every k. RCA's covariance divides by the pair count and adds the
  identity, so it works at k = 1 unchanged.
- **Review:** an independent Sonnet review of the k = 2, 3 path found no number to change. It overwrote the dropped
  pairs with other rows' data and got bit-identical scores. It also ran the original reader-feature code on the cut
  episodes and got the same features. Its one point of meaning (the frozen reader and gate mix two effects) is in the
  results brief. The k = 1 change was checked by the controller: the replacement reader function matches round 1's
  `half_reader_probs` except for the rule, the patch reaches the reader call used by the scorer, and k = 2, 3, 4 are
  unchanged (C5).
- **One restart:** the first run stopped after 13.8 s on a wrong module name in the C1 code. Codex fixed it and re-ran
  the whole job; the raw log keeps every attempt.

## Files and storage

- **Outputs:** `src/test/20261203_kshot_supports/results/kshot_seed42.json` (all numbers, intervals and gate shares),
  `results/ks_run.log`, the run log, and the two job briefs (`CODEX_BRIEF.md`, `CODEX_BRIEF_k1.md`).
- **Storage:** the folder is under 100 KB. Nothing left on disk over 100 MB; no ledger rows.
