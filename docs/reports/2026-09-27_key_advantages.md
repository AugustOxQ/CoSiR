# Key advantages of buddy-graph topic formation over PercepT and simpler alternatives

Generated 2026-09-27. Companion to
[the comprehensive analysis](2026-09-27_comprehensive_analysis.md) and
[the full investigation](2026-09-26_artelingo_buddy_vs_percept_stage1_report.md).
Every claim below cites the specific report it comes from — this is a
synthesis, not a new experiment.

## 1. Versus PercepT (the main baseline)

**2026-09-28 REVERSAL — read this before anything else in this section.**
Every Stage 2 margin below (0.6334, 0.8461, 0.8534) carried an explicit
caveat that only buddy's mapper had been hyperparameter-tuned; PercepT's
was left at its original untuned `lr=1e-3, epochs=100`. That gap has now
been closed: the identical LR/epoch sweep applied to PercepT's fixed
Stage 2 mapper reaches **0.9226** (4-seed mean, std 0.0001, at
`lr=1e-2, epochs=400`) — beating buddy's tuned 0.8534 by -0.0692. Checked
directly for the obvious artifact (tiny held-out classes inflating AUC
under heavier training): held-out per-topic positive counts range 36-1097,
no degenerate classes, same single-label target convention as every other
number here. **"Buddy's Stage 2 classifier beats PercepT's, best-effort
vs. best-effort" is no longer a supportable claim.** This does not affect
Stage 1 (§2 below, AMI/silhouette/occupancy) — only the Stage 2
patch-image classifier comparison. Full account:
[`2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`](2026-09-26_artelingo_buddy_vs_percept_stage1_report.md)
§6g,
[`percept_mapper_symmetric_sweep_pilot_report.md`](../../src/test/20260927_deep_stage_analysis/percept_mapper_symmetric_sweep_pilot_report.md).
The rest of this subsection is preserved as originally written, below,
since it documents real, still-valid buddy-side improvements — only the
final comparison against PercepT changes.

**End-to-end (Stage 1 + Stage 2), buddy wins clearly, after correction and
improvement — SUPERSEDED, see reversal above.** Buddy's original, unmodified image-only Stage 2 mapper
reaches macro AUC **0.5978**. Against PercepT's originally cited 0.5690
this looked like a +0.0288 margin, but an independent review found
PercepT's implementation had two real bugs (backwards center-pruning
direction, reconstruction loss ~2,816x too weak), verified against the
actual paper; fixed, PercepT's Stage 2 macro AUC is **0.5925**, shrinking
buddy's unmodified margin to a near-tie (+0.0053). A follow-up deep
analysis then found buddy's macro AUC was itself being held down by its
three smallest, sparsest topics; merging them (K 19→16) and reweighting
the Stage 2 loss by inverse topic frequency — 4-seed validated, std
0.0026 — raised buddy's Stage 2 macro AUC to 0.6334, a +0.0409 margin.
**A follow-up mapper LR/epoch sweep then found buddy's Stage 2 mapper had
never been tuned** (`lr=1e-3, epochs=100` throughout, PercepT's own
original defaults, untouched); sweeping to `lr=1e-2, epochs=400` raises
buddy's Stage 2 macro AUC again, to **0.8461** (4-seed mean, std 0.0001)
— a +0.2536 margin over PercepT's corrected 0.5925, and higher than
even PercepT's own best-ever tuned Stage 2 number (0.8290, under a
different, richer target convention). **A follow-up richer-multi-label-
target experiment then raised it once more, to 0.8534** (4-seed mean,
std 0.0001, +0.2609 over PercepT) — but only after catching a real flaw
in the first attempt: that pilot's own negative verdict had compared
each cutoff's AUC against that cutoff's own (different, harder-to-compare)
multi-label held-out targets rather than the baseline's single-label
targets, the same apples-to-oranges error already flagged in PercepT's
own numbers above. An independent review caught this; re-scoring
properly reversed the verdict from negative to a robust, 4-seed-validated
gain. This also closed most of a top-1-accuracy gap the deep analysis
found (buddy 0.173→0.385 mean top-1 accuracy across the rounds of fixes).
**Explicit framing caveat**: only buddy's mapper was hyperparameter-tuned
this hard; PercepT's corrected mapper was not retuned, matching this
investigation's own stated priority (a reliable PercepT number, not
necessarily PercepT's best possible one) — state this caveat whenever
citing the +0.2609 margin. Sources:
[`buddy_stage2_pilot_report.md`](../../src/test/20260923_artelingo_buddy_analysis/buddy_stage2_pilot_report.md) §"Direct comparison with PercepT Stage 2";
[`percept_stage2_fixed_pilot_report.md`](../../src/test/20260922_percept_topic_pipeline/percept_stage2_fixed_pilot_report.md);
[`deep_stage_analysis_report.md`](../../src/test/20260927_deep_stage_analysis/deep_stage_analysis_report.md);
[`candidate1_variant_c_stress_report.md`](../../src/test/20260927_deep_stage_analysis/candidate1_variant_c_stress_report.md);
[`candidate2_mapper_sweep_pilot_report.md`](../../src/test/20260927_deep_stage_analysis/candidate2_mapper_sweep_pilot_report.md);
[`candidate4_fixed_stress_report.md`](../../src/test/20260927_deep_stage_analysis/candidate4_fixed_stress_report.md);
[`2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`](2026-09-26_artelingo_buddy_vs_percept_stage1_report.md) §6f.

**Robustness the PercepT replication does not have.** Buddy's plain
baseline clears the predeclared AMI Pareto bar (emotion > 0.1236 AND
genre > 0.1954) in **3/4 seeds** — the first time this was ever tested,
tonight. PercepT's own faithful (paper-recipe) replication was tested at
one seed and **missed the bar outright** (emotion AMI 0.1092), and was
independently flagged **collapsed** by this project's own predeclared
occupancy rule (50/67 held-out centers under 1% occupancy). The only
PercepT variant that clears the bar at all (the "standing balance-hack")
does so only by adding an admitted, ad hoc `LAMBDA_BALANCE=1000`
regularizer with no principled derivation, invented specifically to stop
collapse. Buddy needs no such hack. Sources:
[`attention_h1_baseline_seed_stress_pilot_report.md`](../../src/test/20260923_artelingo_buddy_analysis/attention_h1_baseline_seed_stress_pilot_report.md),
[`percept_stage1_faithful_recipe_pilot_report.md`](../../src/test/20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_pilot_report.md).

**Occupancy/balance.** Buddy's 19 held-out communities: 0 empty, 3/19
below 1%, median 523 of 9,365 points. PercepT's 67 surviving centers: 21
empty, 50/67 below 1%, median 13 points. This is not a sampling artifact
— the matched-protocol audit confirmed it side by side under identical
sampling. Source:
[`buddy_percept_matched_silhouette_audit_pilot_report.md`](../../src/test/20260923_artelingo_buddy_analysis/buddy_percept_matched_silhouette_audit_pilot_report.md).

**Downstream utility is not meaningfully behind, despite the silhouette
gap.** PercepT's silhouette lead is real (0.4973 vs. 0.0416, matched
sampling) — **though this number predates the §6b bug fixes and is likely
overstated: the same fixed pruning direction applied to the faithful-recipe
Stage 1 config dropped its silhouette from 0.5120 to 0.2224. The matched-
audit's own 0.4973 has not yet been re-measured under the fix and should
be treated as an upper bound, not a settled number.** But when tested on
what actually matters for a topic system
(does it help predict real human judgments from images alone), buddy is
statistically tied on emotion (AMI 0.0231 vs. 0.0221) and only modestly
behind on genre (0.2625 vs. 0.2973) — both far below what a raw-feature
control achieves for either system, meaning neither system's topic
bottleneck is close to lossless. The geometric gap that looked large in
isolation does not translate into a comparably large practical gap.
Source:
[`buddy_percept_downstream_probe_pilot_report.md`](../../src/test/20260923_artelingo_buddy_analysis/buddy_percept_downstream_probe_pilot_report.md).

**A materially simpler, less fragile training recipe.** PercepT's own
paper-faithful recipe requires: a two-phase schedule (100-epoch
reconstruction pretrain, then up to 500-epoch joint DEC), latent Gaussian
noise injection (σ=0.1) during pretraining, cosine-annealed learning rates
in both phases, and a fixed reconstruction/KL loss balance (`λ_R=1`) that
this investigation's own candidate-2 pilot showed is **not** portable to
a different input width without a full loss-scale audit (a 433× ratio
shift when the input changed from 2,816-D to 32-D). Even followed exactly,
it still collapses. Buddy's Stage 1 recipe is a single-phase, fixed-LR,
two-loss InfoNCE training with no noise, no phase separation, and no
reconstruction term to balance — and it already clears the Pareto bar in
3/4 seeds without modification. Source:
[`percept_stage1_faithful_recipe_pilot_report.md`](../../src/test/20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_pilot_report.md),
[`percept_on_buddy_embedding_pilot_report.md`](../../src/test/20260923_artelingo_buddy_analysis/percept_on_buddy_embedding_pilot_report.md).

**Six independent attempts to give buddy PercepT's own clustering
mechanism all failed** — Euclidean DEC, vMF-corrected DEC, a decoupled
cluster head (un-detached and detached), PercepT's own autoencoder fed
buddy's embedding, and a properly reconstruction-anchored undercomplete
cluster head. Read charitably, this says PercepT's specific mechanism
(an unconstrained, high-dimensional, reconstruction-anchored latent) is
not a drop-in upgrade for buddy's own geometry — not that buddy's own,
much simpler mechanism (InfoNCE + Leiden) is deficient on its own terms,
where it already works. Source:
[the investigation](2026-09-26_artelingo_buddy_vs_percept_stage1_report.md) §4, §6.

## 2. Versus simpler buddy-side alternatives (internal ablations)

Attention-h1 is not the only architecture tried for buddy's own Stage 1 —
an internal architecture sweep tested a scalar-gate linear-head fusion
("Linear"), two MLP-projection variants, and a 4-head attention variant,
all under otherwise identical data/loss/training conditions.
**Attention-h1 is the first and only configuration in that entire sweep
to clear both held-out AMI thresholds simultaneously** (emotion 0.1249,
genre 0.2404); MLP-64/MLP-128 and Attention-h4 all missed on at least one
axis, and MLP variants are independently flagged collapsed or "merely a
compromise." This is presented honestly, not cherry-picked: the "Linear"
variant scores a *higher* held-out genre AMI (0.2901) and a higher summed
AMI (0.3996 vs. Attention-h1's 0.3653) — Attention-h1 is preferred because
it is the first to jointly clear the predeclared bar this whole
investigation uses as its primary success criterion, not because it
strictly dominates every simpler alternative on every number. Source:
[`learned_student_arch_sweep_pilot_report.md`](../../src/test/20260923_artelingo_buddy_analysis/learned_student_arch_sweep_pilot_report.md).

Against even simpler, single-view or naive-fusion baselines from earlier
in this project's history (content-only CLIP features, GoEmotions-affect
features alone, and a naive late-fusion union) — content-only reaches
0.0593/0.4384, affect-only 0.1180/0.0396, late-fusion-union 0.1236/0.1394
— **none of these clears both AMI thresholds simultaneously either**;
Attention-h1's learned attention fusion of the two views is what closes
that gap, not the buddy graph alone. Source:
[`learned_student_stage2_pilot_report.md`](../../src/test/20260923_artelingo_buddy_analysis/learned_student_stage2_pilot_report.md)
§"Final comparison" (a September 22 reference table; read as historical
context, not a re-run of tonight's exact pipeline).

## 3. Honest ledger — where buddy does *not* win

- **Silhouette.** PercepT's held-out silhouette is higher under every
  protocol tested, matched or not (0.0416–0.0466 for buddy's best plain
  variants vs. 0.4973–0.5120 for PercepT, pre-bug-fix). **Update: this gap
  is smaller than reported.** Two real implementation bugs were found and
  fixed (backwards center-pruning direction, reconstruction loss ~2,816x
  too weak; see the master report §6b). Under the fix, the faithful-recipe
  silhouette dropped from 0.5120 to 0.2224 — still a real PercepT lead
  over buddy's ~0.04, but roughly half the previously reported margin. The
  matched-audit's own 0.4973 has not yet been re-measured under the fix.
  Part of the original gap reflected PercepT's own occupancy collapse
  (§1), which the fix also partially — not fully — corrects; six attempts
  to close the remaining gap via a different mechanism all failed and that
  conclusion is unaffected by this correction.
- **Genre AMI in isolation**, both against PercepT's own downstream-probe
  result (0.2625 vs. 0.2973) and against the internal "Linear" architecture
  variant (0.2404 vs. 0.2901) — buddy's Attention-h1 is not the single
  best-scoring configuration on genre alone in either comparison; it wins
  by the predeclared joint-bar criterion and by the end-to-end Stage 2
  deliverable, not by strictly dominating every individual metric. Under
  the bug fix, PercepT's faithful-recipe genre AMI also rose (0.3288 →
  0.3764), widening this particular gap further.
- **n=4 seeds is not a strong statistical base.** The bootstrap confidence
  intervals in the comprehensive analysis are wide; treat every seed-level
  comparison as indicative, not as a rigorous significance claim.
- **The Stage 2 end-to-end margin dipped to a near-tie after bug-fixing
  PercepT, then was restored and widened three times by fixing real
  weaknesses in buddy's own Stage 2 setup** (0.5978 → corrected-baseline
  near-tie of +0.0053 → 0.6334/+0.0409 after minimum-occupancy handling →
  0.8461/+0.2536 after tuning the mapper's LR/epochs →
  **0.8534/+0.2609** after richer multi-label targets, 4-seed validated
  at each step) — see §1 and the master report §6c/§6d/§6f. The margin
  is real, but it took four rounds of correction (one to PercepT, three
  to buddy) to get an honest number, including catching and reversing a
  flawed comparison in the last round — a reason for continued caution
  about any single reported figure in this investigation. The final
  +0.2609 margin also carried an explicit asymmetry caveat: only buddy's
  mapper was hyperparameter-tuned. **2026-09-28: that caveat turned out to
  be load-bearing** — closing it (§1 reversal, master report §6g) flips
  the Stage 2 comparison to PercepT 0.9226 vs. buddy 0.8534, a -0.0692
  margin. The caution was warranted; the fifth round of correction
  reversed the sign, not just the magnitude.

## Bottom line

**Stage 2 (patch-image → frozen-topic classifier): PercepT wins, once
both mappers are tuned the same way** (0.9226 vs. 0.8534, -0.0692 — master
report §6g). This reverses four rounds of buddy-favoring Stage 2 corrections,
all of which turned out to hold only because of an acknowledged-but-unclosed
tuning asymmetry. **Stage 1 (topic formation, the investigation's actual
subject) is unaffected by this reversal**: buddy's advantage there is not
"wins on every metric" — it is a simpler, non-fragile recipe that needs no
invented stabilizing hack, Pareto-bar clearance the PercepT replication
cannot match at any tested seed (pre- or post-bug-fix), and healthier topic
occupancy where PercepT's still collapses (though less severely post-fix).
The one place PercepT leads on Stage 1 (silhouette) is real but roughly
half the previously reported size, and does not translate into a
comparable downstream advantage; six dedicated attempts to close it
directly all failed for well-diagnosed reasons (one re-examined post-fix
in master report §6b addendum, conclusion unchanged), independent of the
Stage 2 correction above. A seventh, more speculative attempt at Stage 1
(upweighting the affect loss term directly, master report §6h) was also a
clean negative.

**2026-09-30 follow-up (master report §6i).** A ~2,267-trial joint
Stage 1 + Stage 2 sweep of buddy, then a 4-seed stress test of its top
10 finalists, found one configuration (`m8x7ifx4`) that clears the
Stage 1 Pareto bar on 4/4 seeds. The attention-h1 baseline cleared it on
3/4. That configuration reaches Stage 2 macro AUC 0.9355 ± 0.0046 at
13–16 topics. This does **not** change the Stage 2 bottom line above.
The number comes from a re-implemented pipeline, uses a different topic
count and target construction, and had far more tuning budget than
PercepT got, so a matched head-to-head is still needed before any Stage 2
claim changes. It also shows that, within buddy, emotion AMI and Stage 2
AUC trade off against each other (r = −0.85 across finalists). Emotion
remains buddy's binding constraint.
