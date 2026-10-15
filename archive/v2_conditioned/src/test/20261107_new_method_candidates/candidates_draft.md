# New-method candidates after E3, A′ and the 8B probe (controller's draft, hypotheses only)

Written 2026-10-04 by the controller from the project's own evidence. **Every candidate below is an untested
hypothesis and my own reasoning, not a literature finding.** An ARS deep-research pass checks each one against the
literature next (closest prior work, stronger published versions, what a reviewer would cite).

## 1. What any new method has to overcome (measured facts)

The task: given 4 support pairs that agree on aspect A (each pair shows one value of A, four different values, never
the anchor's) and 4 contrast pairs that agree on aspect B, rank 13 candidates so the one sharing the anchor's value of
A comes first. GO needs a reliable gain over cosine, over RCA **and over the method's own condition-free control** on
both R@1 and condition gain.

| Evidence | Number | What it rules in or out |
|---|---|---|
| Label-probe reference (aspect-episode spike; told the aspect, scores by dot product of label posteriors) | pooled 23.09 R@1 vs CLIP 11.13 | Value-level representations that transfer across modalities exist; the task is learnable when the aspect is known |
| Privileged names (spike; told the label names) | +1.51 [0.94, 2.09] | Naming the aspect to CLIP helps little |
| Factor uniform term (E3, seed 43) | either rate 33.4 vs cosine 27.1, gain 0 | Unconditioned factor similarity finds "some shared aspect" well |
| Agreement-weighted term, A3 (E3) | term-only gain 0.97, either rate 21.4 | Selection exists but costs aspect-finding |
| Nested score (A′ pilot, seed 42) | R@1 16.52 vs control 16.55, gain −0.01 | In every λ_u ≥ 1 row, weight on the term buys gain only by losing either rate as fast or faster |
| Label-trained factors (A′ H3) | term-only gain 1.85 vs A3 0.99; loss within 2% of constant | Labels help the term a little; this objective/score barely fits even with true aspects |
| Qwen3-VL-8B in context (seed 46) | R@1 +1.07 [0.17, 1.93], gain +0.21 [−0.51, 0.94] | Even a strong MLLM finds aspect-sharing candidates without selecting the demonstrated aspect |

Two readings follow (mine). First, **the condition must be read from within-pair agreement across pairs**: aspect A is
what is constant inside each support pair while varying across the four pairs. Second, **the representation must hold
each aspect's values as patterns over a shared block** (the 2026-10-03 mechanism note), otherwise value-disjoint
conditions cannot transfer to the anchor's value.

## 2. Candidates

### N1. Centered (covariance) agreement rule instead of co-activation
- **Idea.** The current rule weights factor l by mean over S of a_I,l·a_T,l minus the same over C: an *uncentered*
  second moment, dominated by how active a factor is overall. Aspect A should instead show up as **cross-modal
  covariation across the support pairs**: pairs high on a factor in the image are high on it in the caption, and pairs
  low are low. Weight by w_l = ReLU(cov_S(a_I,l, a_T,l) − cov_C(a_I,l, a_T,l)) (or a correlation), and score with
  centered codes, Σ_l w_l (q_l − μ_l)(c_l − μ_l), with μ the episode's candidate mean, so the score rewards matching the
  query's *deviation* on aspect factors rather than raw activation.
- **Why it might work.** It targets the likely cause of displacement: the uncentered term rewards candidates that are
  generally active on the selected factors. Centering also asks the representation question directly: do values of an
  aspect vary along shared factors?
- **Cheapest test.** Training-free: score the existing checkpoints (A3, LT, L3, C0, SE) on seed-42 development
  episodes; minutes of CPU. Kill criterion: term-only gain no better than the uncentered rule, or either rate still
  below cosine.
- **Risk.** With 4 pairs a covariance is noisy (but summed over 32 factors and 4 pairs × 2 conditions). Close to RCA and
  KISSME in spirit; those failed on raw CLIP in E1, so the claim would rest on the learned basis (K7).

### N2. Find-then-select cascade
- **Idea.** Use the condition only where it is needed: rank by the condition-free score (cosine plus uniform factor
  term), keep the top k, and reorder only those k by the conditioned term.
- **Why it might work.** The pilot showed displacement comes from the conditioned term promoting negatives. Inside a
  short list that already holds p_A and p_B, the term only has to choose between them.
- **Cheapest test.** Training-free on existing checkpoints, seed 42; k ∈ {2, 3, 5} pre-registered. Diagnostic first:
  how often are both p_A and p_B in the top k?
- **Risk.** The gain is capped by how often both aspect candidates reach the short list and by the term's weak
  selection (about 1 to 2 points term-only), so a GO needs both to be large. Novelty is low (retrieve-then-rerank).

### N3. Concept-group basis (aspect blocks by construction)
- **Idea.** Replace the learned factors with an interpretable concept basis: CLIP similarity of each item to a large
  open vocabulary of attribute phrases (SpLiCE-style sparse concept codes), with the phrases grouped into **aspect
  groups** (for example colour, mood, technique, subject). The condition picks the group whose concepts show within-pair
  agreement across the support pairs more than across the contrast pairs (the N1 rule applied per group), and the score
  compares query and candidate inside that group.
- **Why it might work.** The mechanism note says transfer needs aspect blocks of shared factors; a grouped vocabulary
  has them by construction, and the anchor's value is a different concept in the same group, so value-disjoint
  conditions can transfer.
- **Cheapest test.** A vocabulary and grouping built without ArtELingo labels (an LLM-generated attribute bank for
  describing artworks and images, grouped by the LLM; frozen before any episode is scored), CLIP features only, seed-42
  development episodes; hours.
- **Risk.** The grouping injects human or LLM knowledge of aspect types, close to "hand-matched partitions" (disclosed,
  as now). Group quality bounds everything. Overlaps K7's planned SpLiCE comparator and concept-bottleneck work.

### N4. Amortized conditioner (learned, meta-trained)
- **Idea.** A small set encoder (a transformer over the 4 support and 4 contrast pairs) outputs an episode-specific
  projection or gating of CLIP features; score = similarity of the projected query and candidate. Trained episodically
  on pseudo-aspect episodes (and GoEmotions distant supervision), like E3 but with a learned condition reader instead of
  the fixed agreement rule.
- **Why it might work.** The fixed rule is a weak estimator (H3: even label training barely moved its loss); a learned
  reader can learn what "agree within pairs, vary across pairs" looks like.
- **Cheapest test.** One architecture, one training run per bank (about an hour of GPU), seed-42 development.
- **Risk.** Leaves "training-free rule" (C2's framing changes); R-pseudo risk grows (it can learn a selector among the
  trained partitions, the K8 concern); E3 showed pseudo-partitions are weak proxies. Closest work: few-shot set-to-set
  adaptation and in-context embedders; novelty would rest on the cross-modal, value-disjoint setting.

### N5. Infer-the-name, then embed with the name
- **Idea.** Ask the MLLM a generative question it may find easier than 13-way scoring: "what do the example pairs share
  that the counter-examples do not?" Feed the generated aspect phrase as the instruction of an instruction-conditioned
  embedder (Qwen3-VL-Embedding) and rank by that embedding.
- **Why it might work.** The 8B model already beats cosine on finding aspect-sharing candidates; the failure is
  selection inside a long, 30-image prompt. Naming separates inferring the aspect from matching.
- **Cheapest test.** 8B naming on a few hundred seed-42 conditions, read for accuracy against the true aspect (a
  diagnostic using labels on development rows only); then the embedder with the inferred name.
- **Risk.** The privileged-names reference (+1.5) suggests that even correct names help an embedder little on
  ArtELingo. Compute is heavy. Overlaps the K3 "examples versus names" analysis rather than C2.

### N6. Cross-modal partition heads
- **Idea.** Train one cross-modal classifier head per pseudo-partition (predict each row's affect, image and caption
  cluster from either modality); represent each item by its predicted cluster posteriors per head; the condition picks
  the head on which support pairs agree within pairs; score by posterior agreement on that head. This mirrors the
  label-probe reference (23.09) with pseudo-labels in place of labels.
- **Why it might work.** It is the label-probe reference's own mechanism, which is known to work with true labels; each
  head is an aspect block by construction.
- **Cheapest test.** Linear heads on frozen CLIP, scorer-train rows; minutes; seed-42 development.
- **Risk.** Partition-to-aspect agreement is moderate (AMI 0.20 to 0.40), so transfer is partial; the heads are
  hand-matched to evaluation aspects (disclosed), and K8 already failed for a held-out aspect.

## 3. My suggested order (to revise after the literature check)

1. **N1 and N2 first**, together: training-free, minutes on existing checkpoints, and they test the two specific
   failure readings (an uncentered estimator; displacement).
2. **N6 and N3 next** (hours): they build aspect blocks by construction, the requirement the mechanism note names.
3. **N4** only if the cheap ones show the condition can be read at all; **N5** mainly as a K3 analysis.

Any candidate that goes forward gets its own pre-registration: development on seed 42 (already looked at many times,
disclosed), the single GO test on seed 45, the same three comparators.

## 4. Timeline honesty

The abstract deadline is Nov 10. A GO on ArtELingo alone would not carry branch 1's full claim table (K2 on three
datasets, K4 on a second backbone, K7 against SAE and SpLiCE bases); a narrower method paper, or a method section inside
the analysis paper, is the realistic ceiling for the time left.
