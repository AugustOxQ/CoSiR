# Brainstorm: validity of the trained-vs-frozen retrieval cross-reference

## Verdict

The concern is valid, but it does not invalidate Experiment 12's measured quantity. Matched `trained - frozen` rank change has an excellent causal contrast: same deterministic buddy-init, seed, backbone, data, and combiner training; the only intended difference is whether the condition table receives post-init gradients. Thus it cleanly asks, **which samples are most affected by this particular training intervention?** `|delta_rank|` is valid for that descriptive/mechanistic question even if the intervention is harmful on average.

It is not, however, a neutral proxy for whether bridge/false-transitivity matters to *useful retrieval learning*. Experiment 11.1 says that this intervention worsens held-out oracle i2t by 4.67 R1 in every seed; 11.3 reproduces the deficit rather than explaining it. A bridge association could therefore identify susceptibility to an as-yet-unexplained, globally counterproductive update rule, not a graph-induced performance failure. Conversely, the current null only says bridge labels do not explain heterogeneity in this one harmful/interpretable-unknown intervention's in-sample, own-condition rank movement. It cannot support the stronger claim that false transitivity is behaviorally inert under an appropriate training procedure. The existing report should say this more narrowly.

## Cheapest appropriateness check

First inspect the already logged per-epoch held-out metrics for all 3 trained and frozen runs (and the 3 `pred_coupled` runs): plot/pair `test_oracle/i2t_R1` versus epoch, with the trained-minus-frozen gap. If trained initially matches or exceeds frozen then steadily loses ground as drift accumulates, that is strong evidence for late-training degradation/overfitting rather than a fixed cost of allowing adaptation. If the gap appears immediately and remains flat, it instead points to an early optimization/dynamics issue. This costs no training and directly tests whether “final trained checkpoint” is a sensible outcome endpoint. It remains observational, so it diagnoses timing, not root cause.

## Ranked next experiments

1. **Reuse trajectory audit (recommended; no training, cheap).** Extract all epochs' held-out oracle i2t/t2i and drift for trained, frozen, and pred-coupled; test whether rank degradation tracks epoch/drift within seeds. This establishes whether post-init training is temporally well-behaved before treating its final checkpoint as an outcome reference.

2. **Replicate the per-sample bridge cross-reference across existing trained-like arms/seeds (no training, moderate analysis).** Dump the same 3,000-query rank/drift data for all three 11.1 trained seeds and all three 11.3 pred-coupled seeds, each paired with its matched frozen seed. Estimate label effects per run and meta-analyze them; preserve signed improvement and deterioration, not only `|delta_rank|`. Consistency would make it an intervention-specific result; recipe/seed dependence would expose the original single-pair null as underidentified.

3. **Early-versus-late checkpoint bridge test (no training if checkpoints exist; moderate).** Repeat #2 at early/mid/final saved checkpoints. A bridge effect emerging only during the period where trained loses held-out i2t implicates the later update dynamics; an effect present from the start but quality-neutral weakens that story.

4. **Short early-stop sweep (new training, moderate).** Run trained conditions with 0/10/25/50/100 trainable-condition epochs, then freeze, across 3 seeds. It directly locates whether any post-init window improves retrieval, but should follow the free trajectory audit rather than precede it.
