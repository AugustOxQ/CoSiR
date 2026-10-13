# K-shot example pairs: AFF keeps its margin with 3 pairs and loses most of it with 2

> 2026-10-14 · Spec: none (small loop; the design is the job brief, [CODEX_BRIEF.md](../../../../src/test/20261203_kshot_supports/CODEX_BRIEF.md)) · Decision rule: none (descriptive) · Run log: [20261203_kshot_supports_log.md](../../../../src/test/20261203_kshot_supports/20261203_kshot_supports_log.md)

## Results brief

**What we tried.** We gave AFF and its comparators fewer example pairs per episode: k = 2, 3 or 4 support pairs and the
same number of contrast pairs, instead of 4 and 4. Every scorer stayed frozen at its seed-42 settings. The query and the
13 candidates did not change. The user asked for this small k-shot check on 2026-10-14.

**Outcome against the bar.** There is no bar; the check decides nothing. R@1 points, 12,288 seed-42 development
episodes, 95% intervals over 4,602 anchor paintings:

| Example pairs | AFF − B′(A0), R@1 | AFF − matched control, R@1 | AFF's condition gain |
|---|---|---|---|
| 4 (today) | +0.70 [+0.46, +0.94] | +0.74 [+0.52, +0.96] | +3.11 [+2.78, +3.46] |
| 3 | +0.63 [+0.40, +0.87] | +0.71 [+0.48, +0.92] | +3.10 [+2.77, +3.43] |
| 2 | +0.19 [−0.06, +0.44] | +0.36 [+0.12, +0.59] | +2.58 [+2.24, +2.92] |

- From 4 to 3 pairs, AFF's R@1 changes by −0.10 [−0.26, +0.07] and its condition gain by −0.01 [−0.26, +0.24].
- From 4 to 2 pairs, AFF's R@1 falls by 0.56 [0.35, 0.77] and its condition gain by 0.53 [0.21, 0.85].
- B′(A0) hardly changes from 4 to 2 pairs: −0.04 [−0.18, +0.09].
- The lead over B′(A0) comes from the two emotion pairs. On style × genre AFF is below B′(A0) at every k (points:
  −0.33, −0.65, −1.10 at k = 4, 3, 2).

**What it does not show.** Seed 42 is the seed on which every frozen setting was picked, so the levels carry tuning
optimism. The reader, the gate thresholds and every weight were fitted for 4 pairs and kept frozen. So the drop at 2
pairs mixes two effects: less evidence, and a reader and gate tuned for 4 pairs. The table shows AFF used as it is with
fewer pairs; it is not AFF re-fitted for k pairs. One seed, CLIP only, no held rows. We did not test more than 4 pairs.

**Options for next.**
1. Stop here. Add one line to the model card's limits. Cost: none.
2. More pairs (5 to 7). This needs new episodes with 7 pairs, nested as here. Emotion has 8 values and the query's
   own value is left out, so 7 is the maximum without a repeated value. Cost: a Codex job of about an hour and a few
   CPU minutes. It shows whether AFF gains anything past 4 pairs.
3. Re-fit the reader and re-pick the gate and weights at 2 pairs, on the same seed-42 episodes. Cost: a Codex job of
   about two hours. It separates the two effects above. It re-tunes AFF's parts, so it is a diagnostic only.

**Your reading.** *(to be recorded)*

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** AFF needs at least 3 example pairs. With 3 it keeps almost all of its margin; with 2 its margin over
B′(A0) falls to an interval that includes zero. About two thirds of AFF's drop (0.38 of 0.56) is in the part that
reads the condition: AFF's margin over its matched control halves, while B′(A0) stays flat. This split has no
interval. At 2 pairs the gate also opens less on one side (see the analysis), which fits a noisier
reader, or a gate whose thresholds no longer fit; this check cannot tell the two apart. We are fairly sure of the shape (flat from 4 to 3, a drop at 2); the levels are development numbers.

**Our recommendation.** Option 1. The curve is flat from 4 to 3 pairs, so a gain past 4 pairs is not likely. That is our
view; option 2 is the check for it.

</details>

## What ran

- **Stages and types:** one exploratory stage (small loop): the round 6 scoring path, re-run with the first k support
  and the first k contrast pairs of each episode, k = 2, 3, 4. The values of k are nested and paired. k = 1 is left
  out: the reader's features include sample spreads over the pairs, which need at least 2.
- **Data and seeds:** the seed-42 development episodes (12,288, 4,096 per aspect pair), checked against round 6's
  per-pair SHA-256s. No held rows or held annotations read.
- **Scorers:** cosine, RCA, B, B′(A0), AFF (cells 39 / 119) and its matched control (cells 149 / 10), all at the
  round 6 seed-42 picks (pick from tune half h scores parity 1 − h). Readers, heads, posteriors, the factor
  checkpoint and the thresholds τ_0 to τ_3 frozen; nothing re-fitted or re-picked.
- **Code:** `src/test/20261203_kshot_supports/` (`ks_core.py`, `run_ks.py`), branch r9-scope. Built and run by Codex
  (gpt-6.1-sol, high), CPU only, 32.5 s.

## Results

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| C1 | k = 4 reproduces round 6's seed-42 frozen scores | AFF 19.14, B 18.34, B′(A0) 18.44; matched control, RCA, cosine also equal | recorded round 6 values | exact | pass |
| C2 | cosine at every k | 12.96 at k = 2, 3, 4 (raw score arrays identical) | itself | identical | pass |
| C3 | example arrays at k = 2, 3 | k support and k contrast pairs everywhere; factor centring on 2k items | k | exact | pass |

**Verdict under the rule:** none (descriptive).

Pooled R@1 / condition gain (R@1 points):

| Scorer | k = 2 | k = 3 | k = 4 |
|---|---|---|---|
| cosine | 12.96 / 0.00 | 12.96 / 0.00 | 12.96 / 0.00 |
| RCA | 13.35 / 0.03 | 13.40 / 0.13 | 13.38 / 0.10 |
| B | 18.12 / 0.00 | 18.29 / 0.00 | 18.34 / 0.00 |
| B′(A0) | 18.39 / 0.00 | 18.41 / 0.00 | 18.44 / 0.00 |
| AFF | 18.58 / 2.58 | 19.04 / 3.10 | 19.14 / 3.11 |
| matched control | 18.22 / 0.00 | 18.33 / 0.00 | 18.40 / 0.00 |

AFF − B′(A0) per aspect pair (R@1 points, points only):

| Aspect pair | k = 2 | k = 3 | k = 4 |
|---|---|---|---|
| emotion × style | +0.82 | +1.09 | +0.98 |
| emotion × genre | +0.84 | +1.46 | +1.45 |
| style × genre | −1.10 | −0.65 | −0.33 |

## Analysis

**About 0.4 of the 0.56 drop at 2 pairs is in AFF's conditioned part.** B and B′(A0) use the example items (the factor
term is centred on them) but not the condition. From 4 to 2 pairs they move by −0.22 and −0.04. AFF moves by −0.56
and its matched control by −0.18. So AFF's margin over its matched control, the part that reads the condition, falls
from +0.74 to +0.36. The rest of the drop comes from AFF's base, B. This split is a difference of point values and has
no interval.

**At 2 pairs the gate is less one-sided.** AFF's gate should open mostly in condition a (supports show aspect A).
Gate-open share, all three pairs pooled:

| Example pairs | τ_0, condition a | τ_0, condition b | τ_2, condition a | τ_2, condition b |
|---|---|---|---|---|
| 4 | 80.9% | 29.5% | 59.6% | 11.1% |
| 3 | 76.3% | 30.0% | 58.1% | 13.7% |
| 2 | 72.3% | 33.0% | 56.2% | 18.4% |

Our reading: with 2 pairs, the reader's features (means and spreads over 2 values) are noisier and lie outside the
range the reader was trained on (practice episodes with 4 pairs). Its affect picks then fall more often on the wrong
side, and the thresholds, set for 4 pairs, do not adapt. This check cannot separate the two causes.

**Where the loss sits.** Emotion × genre carries most of it (+1.45 → +0.84 over B′(A0)), and style × genre falls
further below B′(A0) (−0.33 → −1.10). Emotion × style is about flat. These are points without intervals.

## Deviations

- **The design brief described the example pairs wrongly.** The brief (and `GLOSSARY.md`, "aspect episode") say the
  supports share the query's value of aspect A. In the code (`src/eval/aspect_episodes.py`), each pair is an image and
  a caption from two paintings that share one value of aspect A. The pairs show k different values, never the
  query's own. The episodes used are the real ones, so no number is affected. Their order within an episode is a
  random draw, so the first k pairs are a random subset.
- **RCA's PCA basis was rebuilt** from the frozen 60,000-row scorer-train draw, since no basis was saved. C1 shows it
  reproduces k = 4 exactly; the same basis is used at every k.
- **Review:** an independent Sonnet review of the k < 4 path found no number to change. It overwrote the dropped pairs
  with other rows' data and got bit-identical scores. It also ran the original reader-feature code on the cut
  episodes and got the same features. Its one point of meaning (the frozen reader and gate mix two effects) is in the
  results brief.
- **One restart:** the first run stopped after 13.8 s on a wrong module name in the C1 code. Codex fixed it and re-ran
  the whole job; the raw log keeps both attempts.

## Files and storage

- **Outputs:** `src/test/20261203_kshot_supports/results/kshot_seed42.json` (all numbers, intervals and gate shares),
  `results/ks_run.log`, the run log, and `CODEX_BRIEF.md` (the job brief).
- **Storage:** the folder is 68 KB. Nothing left on disk over 100 MB; no ledger rows.
