# Condition-Space Steerability Audit — Experiment 17.1

**Date:** 2026-09-15
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, Experiment 17.1
**Plan:** `docs/superpowers/plans/2026-09-15-condition-space-audit.md`

## What was tested

Two training-free checks, per the spec's Phase A: (a) whether a CLIP-text-anchor
semantic direction (built from contrastive prompt pairs per axis, StyleCLIP/ActAdd-style)
separates RedCaps subreddit or Impressions `caption_type` proxy labels in raw CLIP
image/text feature space, beyond a matched random-prompt control (20 controls,
seed=42); (b) whether the same proxy labels are already linearly decodable from
already-trained buddy-init condition vectors (any completed asymmetric,
`combine_side="img"`, buddy-init checkpoint — discovered via
`discover_checkpoints.py`), beyond a matched random-relabeling control
(Hewitt & Liang selectivity, 20 shuffles).

Axes tested: RedCaps `warmth` (companion-animal vs. curiosity-framed subreddits)
and `register` (aesthetic-photography vs. casual-snapshot subreddits); Impressions
`aesthetic_vs_description` and `impression_vs_caption` (both `caption_type` contrasts
— no valence proxy exists for Impressions, a genuine data-availability gap, not an
oversight; see `src/test/20260915_condition_space_audit/axis_definitions.py` for
exact pole definitions and prompts).

Decision rule (per axis/modality): **positive** if the real direction's folded AUC
(`max(auc, 1-auc)`) clears the control mean by `z ≥ 2` (this project's existing
significance convention) *and* the folded AUC itself is `≥ 0.60`; **partial** if
`z ≥ 2` but AUC `< 0.60`; **null** otherwise. Overall gate: positive if any
axis/modality is positive, partial if none are positive but some are partial,
null if all are null.

## Results — Phase A (raw CLIP)

```
dataset        axis                         modality verdict   auc_folded        z
redcaps_150k   warmth                       img    positive       0.965     2.53
impressions    impression_vs_caption        txt    null           0.684     1.23
impressions    aesthetic_vs_description     img    null           0.500     0.61
redcaps_150k   register                     img    null           0.589     0.09
impressions    impression_vs_caption        img    null           0.501    -0.50
redcaps_150k   register                     txt    null           0.547    -0.75
redcaps_150k   warmth                       txt    null           0.545    -0.90
impressions    aesthetic_vs_description     txt    null           0.526    -1.21

GATE VERDICT: positive — strongest axis: warmth (redcaps_150k, img, z=2.53)
```

Only one of the eight (dataset, axis, modality) cells clears the bar:
`redcaps_150k / warmth / img` (real AUC 0.9647, folded AUC 0.9647, control mean
0.6877, control std 0.1095, z=2.53). The margin is large and unambiguous — the real
direction separates companion-animal/cute subreddits (cats, rarepuppers,
blackcats, dogpictures, pitbulls, guineapigs, eyebleach) from curiosity-framed
subreddits (mildlyinteresting, interestingasfuck, natureisfuckinglit) far outside
the spread of 20 random-prompt controls. All seven other cells are null, including
`warmth` on the **text** modality (z=-0.90) — the effect is image-modality-specific,
not a property of the axis's text prompts leaking into text-side CLIP embeddings.
`register` is null on both modalities for RedCaps, and both Impressions axes are
null on both modalities (`impression_vs_caption/txt` has a folded AUC of 0.68 but
z=1.23 < 2, so it does not clear the significance bar despite the numerically
decent AUC).

## Results — Phase B (existing checkpoints)

`checkpoint_probe_results.json` covers 21 already-trained buddy-init checkpoints
across two datasets:

- **redcaps_150k** — 15 checkpoints probed. `warmth`: 15/15 positive. `register`:
  15/15 positive. No cells skipped (all splits had sufficient samples).
- **impressions** — 6 checkpoints probed. `aesthetic_vs_description`: 6/6 positive.
  `impression_vs_caption`: 0/6 positive (all 6 null). No cells skipped.

**Caveat on the redcaps_150k "15/15" figure:** several of the 15 "independent"
redcaps_150k checkpoints have byte-for-byte identical `embeddings.npy` files. This
was verified directly against the raw JSON: e.g. the checkpoints at
`20260825_115853`, `20260825_170212`, `20260825_171558`, and `20260825_172950`
(all under `res/CoSiR_condition_freeze_ablation/redcaps_150k/`) report the exact
same `z` (14774.534509622463) and `selectivity` (0.24449929913314505) to 15+
significant digits on the `warmth` axis, which is not possible for genuinely
independent training runs on stochastic data. This is consistent with
duplicate/resumed run directories rather than 15 truly independent replications.
**The 15/15 and 6/6 counts above should be read as raw checkpoint counts, not as
15 (or 6) independent data points** — the true number of independent positive
observations is smaller, though this plan does not attempt to determine exactly
how many of the 15 (or 6) are independent (out of scope here; a job for whoever
next needs a rigorously-counted N).

With that caveat, the qualitative pattern is still informative for 17.2's
initialization design: existing buddy-init condition vectors on redcaps_150k
already carry decodable `warmth` and `register` signal (unsurprising for `warmth`
given the raw-CLIP result above, more surprising for `register`, which was null at
the raw-CLIP stage but shows up post-training — suggesting the trained embeddings
may pick up structure the frozen CLIP backbone did not expose directly). Impressions
condition vectors carry `aesthetic_vs_description` signal but not
`impression_vs_caption` signal, consistent with the raw-CLIP result where neither
Impressions axis passed the primary gate.

## Verdict and effect on Experiment 17.2

**GATE VERDICT: positive**, from Phase A (the primary gate per the spec) —
`redcaps_150k / warmth / img` is the one cell that clears both the `z ≥ 2` and
`AUC ≥ 0.60` thresholds (z=2.53, folded AUC=0.9647), and per the decision rule any
single positive cell makes the overall gate positive.

**Winning axis: `warmth`**, on RedCaps, image modality specifically (not text).
`warmth` is the companion-animal/cute-subreddit vs. curiosity-framed-subreddit
contrast, used here as a proxy for positive emotional valence.

Per the spec's stated routing for a positive gate: **Experiment 17.2 should be
scoped to the winning axis** — i.e., 17.2's condition-space steerability work
should target the RedCaps `warmth` axis on the image side, where this audit found
a real, pre-existing, trainable-free signal in CLIP's own feature space. The
Phase B result reinforces this: buddy-init condition vectors already carry
decodable `warmth` signal post-training (with the duplicate-checkpoint caveat
above), so 17.2 does not need to bootstrap this signal from nothing — it needs to
determine whether/how the buddy-init geometry can be steered *along* this axis in
a controlled way. `register` should not be a primary 17.2 target: it was null at
the raw-CLIP stage (Phase A), so its Phase B positivity is more likely an
artifact of training dynamics than evidence of an exploitable pre-existing
direction. Impressions should not be a primary 17.2 target on either axis tested
here, given both axes were null at the primary (Phase A) gate.

## Caveats

- All four axes are proxy labels, not ground truth for "emotional tone" or
  "political framing" specifically — a positive result here means the proxy axis
  is present, not that the originally-envisioned target concept is (see the
  spec's §8 risk row on this).
- No verified published benchmark exists for emotion/political-framing detection
  on Reddit *image* content specifically (per this project's literature research,
  2026-09-15) — this audit's own result is the first evidence either way for this
  domain.
- Impressions has no valence-adjacent proxy label at all in this pass; both of its
  axes are registerial/framing contrasts within `caption_type`, not emotional-tone
  contrasts — stated as a data-availability constraint, not routed around.
- **Phase B's "15/15" and "6/6" positive counts overstate independent replication.**
  Several of the 15 redcaps_150k checkpoints (and possibly some of the 6
  Impressions checkpoints) have identical or near-identical embeddings, most likely
  from duplicate or resumed training-run directories rather than independent
  training runs. Treat Phase B as qualitative/supplementary corroboration, not as
  15 (or 6) independent statistical trials — this plan did not attempt to
  determine the true independent N, since Phase B is explicitly supplementary
  to Phase A (the primary gate) per the spec, and root-causing the duplication is
  out of scope here.
- **The z-score formula (`z = (real_auc_folded - control_mean) / (control_std +
  1e-8)`) has no minimum-control-std floor.** When `control_std` is very small
  (as happens in several Phase B cells, e.g. `control_std` on the order of
  `1e-16`–`1e-4`), z can blow up to extreme, not-meaningfully-interpretable
  magnitudes (Phase B's raw z values reach into the tens of millions in places)
  purely from a near-zero denominator, not from a large real effect. This is a
  latent design limitation of the significance convention used throughout this
  audit. It is verified **not** to have caused a false positive in Phase A, the
  primary gate: the one positive cell (`redcaps_150k/warmth/img`) has a healthy
  `control_std` of 0.1095 and a large, unambiguous margin between the real AUC
  (0.9647) and the control mean (0.6877), so its z=2.53 reflects a real effect,
  not a denominator artifact. Anyone treating the z-scores here (especially
  Phase B's) as a fully robust significance test rather than a heuristic screen
  should be aware of this.
