# On paintings never used for tuning, affect steering passes its test, but the gain over the plain scorers is small and only pooled

> 10 October 2026. Full report: [ArtELingo held test of AFF](../reports/auto/v2/2026-11-25_artelingo_held_test.md)
> (its file name carries CoSiR's sequence-number date).

## Summary

- We ran the paper's final test of affect steering once, on held paintings that nothing was fitted, picked or tuned on.
- It passed all seven pre-registered checks, so the verdict is GO.
- Pooled over three aspect pairs it beats the condition-free scorers B and B′(A0) and its own matched control by about
  0.6 R@1 points, and plain cosine by 6.19.
- A margin over the strongest condition-free scorer (B′(A1), which adds a style grouping) and over the same reader
  without the gate (R1) is not established.
- Per aspect pair it wins two and loses style × genre, which design L, the planned grouping redesign, targets.
- The test covers one dataset (ArtELingo) and one backbone (CLIP ViT-B/32).
- The claim is small and pooled; the paper go/no-go on Fri 2026-10-16 decides whether to write the paper around it.

## How we got here

CoSiR ranks captions for a painting (or paintings for a caption) under an aspect that is never named. In each
**episode** (one ranking task) a few example pairs share one aspect, such as emotion, style or genre, and contrasting
pairs share another. The method has to work out which aspect the examples show (the **condition**) and rank
accordingly.

Scores are **R@1**: the share of rankings with the right candidate first, in percentage points. The
**condition gain** is R@1 with the right condition minus R@1 for the same target with the examples swapped; a
**condition-free** scorer, which ignores the examples, has a gain of 0.

A **reader** guesses the aspect from the examples. Affect steering (AFF), chosen in round 3 on development data, adds
the reader's score only when the reader picks the emotion-like grouping. Rounds 4 and 5 read about 10 more variants on
development data, and AFF stayed the method to test. A claim for the paper has to be tested on paintings it has never
seen, before the paper go/no-go on Fri 2026-10-16. This round is that test.

## 1. The held read

**Problem.** AFF was found among about 50 variants on development episodes. Any number from there is flattered by that
search, so the paper needs one clean look.

**Idea.** One **held read**: a single scoring of ArtELingo's held paintings (12,281 of them, kept apart from everything
used to fit, pick or tune), with every weight frozen on development data and the decision rule committed before the
read. We drew 12,288 episodes on each of three new seeds, 36,864 in all.

AFF had to beat five scorers (cosine, RCA, two condition-free scorers and its own matched control), and its condition
gain had to exceed theirs. That makes seven checks, tested together with a Holm correction, which keeps the error rate
across all seven at the usual level.

**Did it work.** Yes, with a narrow claim. All seven checks pass, so the verdict is GO. The claim is pooled over the
three aspect pairs only. The two secondary checks, run after a GO and never changing it, are inconclusive:

- against B′(A1), the strongest condition-free scorer (it adds the style grouping), AFF is ahead by +0.14 [-0.03, +0.30];
- against R1, the same reader without the gate, it is ahead by +0.02 [-0.06, +0.11].

So the test does not show that the affect gate matters.

The three seeds draw on nearly the same paintings, so their
agreement is not replication. The pooled interval resamples only the anchor paintings (the painting each episode is
built around), not the candidate and target paintings that recur across episodes. A check that also resamples those
widens it, and every lower bound stays above zero. For the paper, the report advises citing the wider interval, with
the target paintings resampled (against B′(A0): +0.60 [+0.40, +0.79]).

**Key evidence.** The margin over B, B′(A0) and the control is modest next to the margin over cosine and RCA. The
condition gain, which is 0 for any scorer that ignores the examples, is +3.06 [+2.87, +3.25] for AFF.

![AFF is far above cosine and RCA, and about 0.6 points above B, B′(A0) and its matched control](assets/2026-10-10_artelingo_held_test/fig1_pooled_r1.png)

*Figure 1. R@1 on the held paintings, pooled over three seeds and three aspect pairs. B′(A1) and R1 are not drawn.
The margins and their 95% intervals are in the text above.*

> Full report: Results brief, Results (the seven checks and the two secondary ones) and Disclosures.

## 2. Comparators added to make the claim fair

**Problem.** A reviewer will ask whether a language model or a fine-tuned CLIP does the same job without our reader.

**Idea.** We added external comparators, reported but never gating the verdict. DTS describes each aspect in words with
a language model and scores the words. DTS-N is told the true aspect name, a privileged ceiling variant. FT-LP, FT-LB
and FT-LoRA are fine-tuned CLIP models. MLLM is a Qwen3-VL-8B reranker, run on one of the three seeds (52) only.

**Did it work.** On R@1 and condition gain none comes near AFF. One result changes how the paper must talk. On **swap
success** (both targets reorder when the condition swaps) the told-name DTS-N and the MLLM are above AFF, and R1 is
slightly above it (0.17 points).

**Key evidence.** Pooled R@1 with AFF at 19.44, and swap success:

| Comparator | R@1 | AFF minus it |
|---|---|---|
| DTS | 13.06 | +6.38 [+6.15, +6.59] |
| DTS-N | 12.15 | +7.29 [+7.03, +7.53] |
| fine-tuned CLIP (three variants) | 15.32 to 15.80 | +3.64 to +4.11 |
| MLLM (seed 52) | 14.50 | +4.94 [+4.56, +5.33] |

| Swap success | % | AFF minus it |
|---|---|---|
| AFF | 11.50 | |
| R1 | 11.67 | -0.17 [-0.33, -0.01] |
| DTS-N | 16.43 | -4.93 [-5.29, -4.57] |
| MLLM (seed 52) | 13.08 | -1.61 [-2.19, -1.01] |

The MLLM's margins are taken on seed 52's episodes only, so they are not the difference of the pooled values.

> Full report: Analysis, "External comparators" and "Swap success".

## 3. The per-pair picture

**Problem.** A pooled number can hide a pair where the method does harm. Round 3 found one: style × genre.

**Idea.** The rule requires the per-pair result beside the pooled claim. We split the margin over B′(A0) by aspect
pair, 12,288 episodes each, after the verdict, so it decides nothing.

**Did it work.** AFF wins two pairs and loses one. The loss on style × genre is of similar size on new paintings as in
round 3 (-0.43 here, -0.58 there), so it is unlikely to be an accident of one set of paintings. That pair is what
design L targets. The claim stays pooled: no per-pair claim is licensed.

**Key evidence.** The same pattern, two wins and a loss, holds against B, the matched control and B′(A1); the gap is
widest against B′(A1), which AFF trails by 1.24 points on style × genre (the full report's per-pair table).

![AFF wins two aspect pairs and loses style × genre, as in round 3](assets/2026-10-10_artelingo_held_test/fig2_per_pair.png)

*Figure 2. AFF minus B′(A0), R@1 points with 95% intervals, per aspect pair. The grey point is round 3's value.*

> Full report: Analysis, "Per pair, the pooled margin is two wins and a loss".

<details>
<summary>Our advice (open after giving your reading)</summary>

This is our view; you decide.

The pooled claim holds: the margins over B, B′(A0) and the matched control are about +0.6 on each seed (+0.58 to
+0.66), and the intervals stay above zero when candidate or target paintings are resampled too. The seeds share
paintings, so that agreement is not independent replication.

The size is modest, it comes from emotion × style and
emotion × genre, and style × genre loses. We are fairly sure of the pooled sign over B, B′(A0) and the control. We are
not sure of any margin over B′(A1) or R1, and less sure the margin would survive another dataset.

We recommend design L as the next loop, because the style × genre loss is the clearest weakness and the go/no-go needs
to know whether it is fixable. In parallel, draft the ArtELingo-centred paper from the full report, with the per-pair
table beside the pooled claim.

</details>

## Next

**Where things stand.** The read is done and the verdict is GO. The full report has been through its final review,
a fix wave and a re-review of the fixes. Disclosures about how AFF was found, the DTS tuning and
the style × genre loss travel with every AFF number.

**Your decisions:**

1. **Your reading of the results** (decision C), at the r6 read chat, before you open our advice above.
2. **The fix in the after-verdict analysis.** That analysis (the descriptive pass) first stopped because the
   language model gave the same phrase two different answers on different seeds. We re-ran the phrase answers of
   seeds 53 and 54 so that each phrase has one answer, then finished the pass.

   This changed 632 of 5,429 and 750 of
   5,640 answers, so the DTS rows carry some noise their intervals do not measure; no conclusion depends on it. It
   touches only the DTS rows of the analysis, never the verdict, which was written before any DTS input was read.

   The rule does not cover this, so the call is yours: ratify the fix or not.
3. **What next, before the paper go/no-go on Fri 2026-10-16.** The full report's options:
   - (a) Design L, already chosen as the next loop (you, 2026-10-09): its own spec, a few days of CPU work plus
     cluster jobs. A design-L read on the held paintings would be a new held read, disclosed as designed after this
     one.
   - (b) Write the ArtELingo-centred paper now around this claim, with the style × genre result shown beside it; no
     compute.
   - (c) A second dataset or backbone, which needs new data preparation and a new held read.

## Glossary

| Term | Meaning | Full report's name |
|---|---|---|
| episode | one ranking task: 13 candidates, example pairs and contrasting pairs | episode |
| aspect pair | the two aspects an episode sets against each other | emotion × style, emotion × genre, style × genre |
| held rows | ArtELingo paintings reserved for the final test, never used to fit, pick or tune | held rows |
| held read | one pre-registered scoring of the held rows, ledgered | held read |
| R@1 | share of rankings with the right candidate first, in percentage points | R@1 |
| AFF | affect steering: the reader's score added only when it picks the emotion-like grouping; current best | AFF |
| R1 | the same reader, with no gate on which grouping it picks | R1 |
| cosine | plain CLIP similarity between query and candidate | COS |
| RCA | relevant component analysis: a classic baseline that re-weights CLIP features using the episode's example pairs, fused with cosine; it uses the condition | RCA |
| B | the best condition-free scorer before the reader | B |
| B′(A0) | B rebuilt on the reader's three groupings | B′(A0) |
| B′(A1) | B′ with the style grouping added: the strongest condition-free scorer | B′(A1) |
| condition-free scorer | a score that ignores which aspect the examples show | condition-free comparator |
| matched control | a scorer's own score with only the condition removed | matched control |
| condition | which aspect an episode's examples show | condition a / b |
| condition gain | R@1 under the right condition minus R@1 for the same target with the examples swapped; 0 for condition-free scorers | condition gain |
| anchor painting | the painting an episode is built around; the pooled intervals resample these | anchor paintings |
| swap success | share of episodes where both targets reorder when the condition swaps | swap success |
| Holm | a correction for testing several checks at once, so the error rate holds across all seven | Holm |
| DTS, DTS-N | describe-then-score with a language model; DTS-N is told the true aspect name | DTS, DTS-N |
| FT-LP, FT-LB, FT-LoRA | three fine-tuned CLIP comparators | FT-LP / FT-LB / FT-LoRA |
| MLLM | a Qwen3-VL-8B reranker comparator | MLLM |
| design L | the planned grouping redesign aimed at style × genre | design L |
| claim test | a test that licenses only the text of its pre-registered claim | claim test |
