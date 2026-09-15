# Condition-Space Steerability Audit — Experiment 17.1

**Date:** 2026-09-15 (original), **revised 2026-09-15** after a whole-branch final review
found 1 Critical + 7 Important issues in the original pass (see `.superpowers/sdd/2026-09-15-condition-space-audit/final-review-fix-report.md`
for the full fix wave). This revision replaces the original results/verdict with numbers
from the corrected pipeline — nothing here is a re-derivation from memory, every number
below comes from a real re-run of the scripts in `src/test/20260915_condition_space_audit/`.
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, Experiment 17.1
**Plan:** `docs/superpowers/plans/2026-09-15-condition-space-audit.md`

## What changed in this revision, and why

The original pass reported RedCaps `warmth` (image modality) as a "valence axis" —
z=2.53 vs. a *random-prompt* control. The final review found this conclusion
unsupported: `warmth`'s pole_a (cats, rarepuppers, dogpictures, ...) and pole_b
(mildlyinteresting, interestingasfuck, ...) are visually distinct subreddits —
photos of animals vs. photos of generic scenes/objects — so a random-prompt
control (built from semantically unrelated words like "table"/"cloud") cannot
distinguish "this direction detects emotional valence" from "this direction
detects animal-vs-not, which happens to correlate with these particular
subreddits' emotional framing." A **content-matched control** was needed and
was missing. This revision adds one (Fix 1), plus six other fixes: Impressions'
img-modality cells are structurally degenerate and are now marked `n/a` rather
than "null" (Fix 2); Phase B's selectivity floor (Fix 3); an empirical control
rank report (Fix 6); paraphrase-matched (not single-prompt) controls (Fix 7);
one positive-control sanity cell on the text modality (Fix 8); and a genuine
raw-CLIP probe baseline that Experiment 17.2's success criterion needs but the
original pass never computed (Fix 4/5).

## What was tested

Two training-free checks, per the spec's Phase A: (a) whether a CLIP-text-anchor
semantic direction (built from contrastive prompt pairs per axis, StyleCLIP/ActAdd-style)
separates RedCaps subreddit or Impressions `caption_type` proxy labels in raw CLIP
image/text feature space, beyond a matched random-prompt control (20 controls,
seed=42, each control direction now built from 3 paraphrase templates per word,
matching the real axes' paraphrase count — Fix 7) **and**, for RedCaps axes only,
beyond a **content-matched control** (a direction built from generic,
affect-free descriptions of what each pole's subreddits actually depict — Fix 1);
(b) whether the same proxy labels are already linearly decodable from
already-trained buddy-init condition vectors (any completed asymmetric,
`combine_side="img"`, buddy-init checkpoint — discovered via
`discover_checkpoints.py`), beyond a matched random-relabeling control
(Hewitt & Liang selectivity, 20 shuffles), now additionally gated on a minimum
selectivity floor of 0.10 (real accuracy minus control mean — Fix 3), not z alone.

Axes tested: RedCaps `warmth` (companion-animal vs. curiosity-framed subreddits)
and `register` (aesthetic-photography vs. casual-snapshot subreddits); Impressions
`aesthetic_vs_description` and `impression_vs_caption` (both `caption_type` contrasts
— no valence proxy exists for Impressions, a genuine data-availability gap, not an
oversight; see `src/test/20260915_condition_space_audit/axis_definitions.py` for
exact pole definitions and prompts). Impressions has no content-matched control
because, unlike RedCaps, its two axes don't contrast different *content* — they
contrast different *caption styles of the same underlying photos* (see Fix 2 below).

Also added this revision: a **pipeline sanity-check cell** (Fix 8) — RedCaps
subreddit `cats` vs. `carporn` on the **text** modality, using obviously-separable
content prompts ("a caption about a cat" vs. "a caption about a car"). This is
**not a formal experimental axis** — it exists purely to confirm the text-modality
pipeline can detect *some* real signal when it's actually present, contextualizing
why `warmth`/txt (a genuine, non-content-obvious axis) comes back null.

Decision rule (per axis/modality, unchanged from the original pass): **positive**
if the real direction's folded AUC (`max(auc, 1-auc)`) clears the random-prompt
control mean by `z ≥ 2` *and* the folded AUC itself is `≥ 0.60`; **partial** if
`z ≥ 2` but AUC `< 0.60`; **null** otherwise. Overall gate: positive if any
axis/modality is positive, partial if none are positive but some are partial,
null if all are null. **This statistical gate is unchanged** — the content-matched
control (Fix 1) is an *additional* diagnostic reported alongside it, not a
replacement, per the fix-wave's own instructions. New this revision: a third
possible per-cell status, **`n/a_same_images`** (Fix 2), for Impressions
img-modality cells where pole_a/pole_b are structurally the same photo set —
these are excluded from the gate's positive/partial/null tally entirely (they
were never a valid test to begin with, not a null result).

## Results — Phase A (raw CLIP)

Family size: **8 (dataset, axis, modality) cells were run, of which 6 are
informative** — the other 2 (both Impressions img-modality cells) are
structurally degenerate (Fix 2, see below) and excluded from the gate.
With only 20 controls per cell, z should be read as **a rough screen, not a
precise p-value** — see Fix 6's control-rank column below for the same signal
in a more interpretable form (how many of the 20 actual control directions the
real direction beat).

```
dataset        axis                         modality verdict          auc_folded    z      control_rank
redcaps_150k   warmth                       img      positive            0.965    2.51   20 of 20 controls exceeded
impressions    impression_vs_caption        txt      null                0.684    1.25   18 of 20 controls exceeded
impressions    aesthetic_vs_description     img      n/a_same_images     0.500    0.54   16 of 20 controls exceeded
redcaps_150k   register                     img      null                0.589    0.12   12 of 20 controls exceeded
impressions    impression_vs_caption        img      n/a_same_images     0.501   -0.40    8 of 20 controls exceeded
redcaps_150k   register                     txt      null                0.547   -0.71    6 of 20 controls exceeded
redcaps_150k   warmth                       txt      null                0.545   -0.95    5 of 20 controls exceeded
impressions    aesthetic_vs_description     txt      null                0.526   -1.23    3 of 20 controls exceeded

6 informative cells (2 excluded as n/a_same_images).
GATE VERDICT (statistical gate, unchanged): positive — strongest axis: warmth (redcaps_150k, img, z=2.51)

Pipeline sanity check (cats vs. carporn, txt, NOT a formal axis): verdict=positive, auc_folded=0.994, z=2.43
```

Only one of the six informative cells clears the random-prompt-control bar:
`redcaps_150k / warmth / img` (real AUC 0.9647, folded AUC 0.9647, control mean
0.6904, control std 0.1094, z=2.51, beating 20 of 20 random-prompt controls
outright). All other informative cells are null, including `warmth` on the
**text** modality (z=-0.95) — the sanity-check cell confirms the text-modality
pipeline is capable of detecting real signal when it's actually there (cats vs.
carporn captions, z=2.43), so `warmth/txt`'s null is a genuine null, not a
pipeline blind spot. `register` is null on both modalities for RedCaps, and
both remaining (non-degenerate) Impressions cells are null.

### Content-matched control (Fix 1) — the critical new result

For RedCaps `warmth` and `register`, a second control direction was built from
generic, affect-free descriptions of what each pole's subreddits actually
depict (e.g. "a photo of an animal" vs. "a photo of an object or scene" for
`warmth`) — this isolates whether the real axis captures anything **beyond raw
content category**.

| axis | modality | real (folded) | content control (folded) | beats content control? |
|---|---|---|---|---|
| warmth | img | **0.9647** | **0.9058** | **True** — margin 0.059 |
| warmth | txt | 0.5449 | 0.8318 | False |
| register | img | 0.5886 | 0.8573 | False |
| register | txt | 0.5472 | 0.5950 | False |

**This is the critical number.** The one cell that clears the statistical gate,
`redcaps_150k / warmth / img`, *does* beat its content-matched control
(0.9647 vs. 0.9058) — so the axis is not *purely* a content detector; there is
a small, real residual beyond raw animal-vs-non-animal content. But the content
control alone already achieves AUC 0.9058 — **the overwhelming majority of the
separability *above chance* (0.4058 of 0.4647 folded-AUC-above-0.5, i.e. 87.3%)
is explained by raw visual content (companion-animal photos vs. generic-scene
photos), not by anything specific to emotional valence.** (The raw AUC ratio,
0.9058/0.9647, is not the right quantity here — see below.) The margin
attributable to something beyond content is only 0.059 AUC points, against a
margin of 0.274 AUC points over the random-prompt control. Compare this to
`register`, where the content control (0.857) is *higher* than the real axis
itself (0.589) — meaning for `register`, raw content alone over-explains what
little separability exists; the "register" framing adds nothing measurable.

**Caveat on the 0.059 residual itself:** this content control is a single,
hand-written prompt set ("a photo of an animal" vs. "a photo of an object or
scene"), not a distribution over alternative content-framings the way the
random-prompt control is (20 independent draws, std=0.109). A different but
equally reasonable content control could plausibly shrink or grow this margin.
The 0.059 gap should be read as "not zero, on this one content control," not
as a precisely bounded effect size.

**Conclusion: `warmth`/img survives as more than pure content, but not as
established evidence of an emotional-valence axis.** The data supports
"predominantly a content detector (companion-animal vs. non-animal-scene),
with a small, real, currently-uninterpreted residual" — not "a valence axis."
Nothing in this audit's design (content-matched controls included) can
attribute that residual to emotional valence specifically, as opposed to other
factors correlated with these particular subreddits (framing distance,
composition, color palette, etc.). Confirming a valence interpretation would
require a genuinely valence-controlled test (e.g. happy vs. sad photos *within*
the same content category), which this audit does not attempt and is out of
scope here.

### Impressions img-modality cells are structurally degenerate (Fix 2)

Impressions' `aesthetic_vs_description` and `impression_vs_caption` axes
contrast different **caption styles of the same underlying photos** — each
photo (identified by the `ImgId` field, distinct from the per-caption
`image_id`) has multiple captions of different `caption_type`s. Checking the
Jaccard overlap of pole_a vs. pole_b `ImgId` sets for the img modality found
>90% overlap for both axes (`aesthetic_vs_description`: overlap-driven
`control_std` of 0.00013; `impression_vs_caption`: `control_std` of 0.0011 —
both near-zero, exactly the z-blowup failure mode flagged as a caveat in the
original report), confirming the img-modality test there compares
near-identical image features to themselves. Both cells are now reported as
`n/a_same_images` rather than `null` and excluded from the gate tally — this
was never a valid test of the axis, and reporting it as a clean "null" would
have been misleading in the other direction. RedCaps has no such degeneracy
(different subreddits are genuinely different photos; no `ImgId` field exists
on RedCaps records, so the check is a no-op there).

## Results — Phase B (existing checkpoints) — now with an effect-size floor (Fix 3)

`checkpoint_probe_results.json` covers the same 21 already-trained buddy-init
checkpoints across two datasets, re-run with `SELECTIVITY_FLOOR = 0.10` added
to the verdict rule (`z ≥ 2 AND selectivity ≥ 0.10`, selectivity =
real accuracy − control-mean accuracy), not z alone:

- **redcaps_150k** — 15 checkpoints probed. `warmth`: **15/15 positive**
  (selectivity 0.2445–0.2466, real accuracy ≈0.958–0.960). `register`:
  **15/15 positive** (selectivity 0.3004–0.3072, real accuracy ≈0.903–0.909).
  Unchanged from the original pass — these effects are large enough that the
  new floor doesn't touch them.
- **impressions** — 6 checkpoints probed. `aesthetic_vs_description`:
  **0/6 positive (was reported 6/6 in the original pass)** — every one of the
  6 checkpoints had z ≥ 2 (up to z≈7.0) but selectivity only 0.017–0.057, well
  below the 0.10 floor; these were exactly the near-zero-`control_std` z-blowup
  cases the original report's own caveat warned about, now correctly excluded.
  `impression_vs_caption`: **0/6 positive** (unchanged — was already null in
  the original pass; selectivity ≈ −0.002 to 0.0001, indistinguishable from
  chance).

**Caveat on the redcaps_150k "15/15" figure (unchanged from original):**
several of the 15 "independent" redcaps_150k checkpoints have byte-for-byte
identical `embeddings.npy` files (e.g. `20260825_115853`, `20260825_170212`,
`20260825_171558`, `20260825_172950` report identical selectivity/z to 15+
significant digits), consistent with duplicate/resumed run directories rather
than independent replications. Treat the "15/15" and "6/6"-style counts as raw
checkpoint counts, not independent data points.

## Raw-CLIP supervised-probe baseline (Fix 4) — the number 17.2 actually needs

Experiment 17.2's stated success criterion ("control-task-gated probe
selectivity on the *trained* subspace must exceed 17.1's raw-CLIP baseline
selectivity to count as 'the training added something'") requires a
raw-CLIP baseline computed with the **same probe method** used on trained
checkpoints (Hewitt & Liang selectivity via a supervised logistic-regression
probe) — not the weaker text-anchor direction method used in Phase A above.
This number was never computed in the original pass. `raw_clip_probe_baseline.py`
closes the gap: it reuses `raw_clip_audit.load_features` (same cached raw CLIP
img features as Phase A) and `checkpoint_probe.probe_selectivity` (same
function, same hyperparameters, used on trained checkpoints in Phase B) fed raw
512-D CLIP image features instead of a trained 16-D condition vector:

```json
{
  "warmth":   {"n": 23824, "real_acc": 0.9790, "control_mean": 0.7134, "selectivity": 0.2656, "z": 3344.5, "verdict": "positive"},
  "register": {"n": 14837, "real_acc": 0.9332, "control_mean": 0.5938, "selectivity": 0.3394, "z":  228.9, "verdict": "positive"}
}
```

(Full output: `src/test/20260915_condition_space_audit/raw_clip_probe_baseline.json`.)

**Both axes are already strongly decodable from raw, untrained CLIP image
features via a supervised probe** — `register`'s raw-CLIP baseline selectivity
(0.3394) is in fact the *highest* of any axis measured anywhere in this audit,
higher even than `warmth`'s (0.2656).

## Fix 5 — correcting the unsupported "post-training" claim

The original report claimed: *"`register`... was null at the raw-CLIP stage but
shows up post-training — suggesting the trained embeddings may pick up
structure the frozen CLIP backbone did not expose directly."* **This claim does
not survive the real numbers and must be withdrawn.**

What actually happened: `register` was null in Phase A's raw-CLIP **text-anchor
direction** test (a weak, unsupervised, difference-of-means construction) but
is strongly positive in a raw-CLIP **supervised logistic-regression probe**
(Fix 4, selectivity 0.3394 — the highest of any axis tested). These are two
different methods with very different statistical power; comparing Phase A's
null (text-anchor) to Phase B's positive (supervised probe on trained vectors)
was an apples-to-oranges comparison, not evidence that training added
anything. The correct, like-for-like comparison is trained-checkpoint
selectivity (Phase B, supervised probe) vs. raw-CLIP baseline selectivity
(Fix 4, same supervised probe):

| axis | raw-CLIP baseline selectivity | trained-checkpoint selectivity (range, n=15) |
|---|---|---|
| warmth | 0.2656 | 0.2445 – 0.2466 |
| register | 0.3394 | 0.3004 – 0.3072 |

**For both axes, every one of the 15 trained checkpoints has selectivity
below the raw-CLIP baseline, not above it.** Training did not add decodable
signal beyond what frozen CLIP already exposes for either axis — if anything,
training very slightly *reduced* linear decodability relative to raw CLIP
(plausibly because retrieval-objective fine-tuning reshapes the embedding
space for the retrieval task, not because it destroys the underlying content
signal, which remains almost entirely intact). Per Experiment 17.2's own
stated success criterion, **the "training added something" bar is not met for
either axis by the currently-trained checkpoints** — both `warmth` and
`register` are already fully explained by frozen CLIP.

**Important confound this comparison does not control for: dimensionality.**
The raw-CLIP baseline probes 512-D features; the trained-checkpoint probe
targets a 16-D condition vector — a ~32× difference in probe capacity, which
by itself can produce part or all of an accuracy gap this size (`warmth`: real
accuracy 0.979 raw vs. 0.958–0.960 trained; `register`: 0.933 raw vs.
0.903–0.909 trained — gaps of 0.02–0.03 in raw accuracy terms). Some of the
`register` selectivity gap (~0.0085 of it) also comes from a *shift in the
control side* (raw control_mean 0.594 vs. trained 0.602), which the
selectivity metric folds in without distinguishing. As instantiated, this
comparison — and therefore Experiment 17.2's stated success criterion, which
inherits it — is not a fair, capacity-matched test of "did training add
anything": a 16-D subspace is close to structurally unable to clear a 512-D
baseline regardless of what training does. A fair comparison for 17.2 would
use a dimension-matched raw-CLIP baseline (e.g. a 16-D PCA or random
projection of the raw features). The directional finding (trained
checkpoints don't clearly exceed raw CLIP) is still worth noting, but it
should not be read as clean evidence that training adds nothing — that
claim is confounded by capacity, not yet established.

## Verdict and effect on Experiment 17.2 (revised)

**GATE VERDICT: still formally positive**, per the unchanged statistical
decision rule — `redcaps_150k / warmth / img` is the one informative cell that
clears both the `z ≥ 2` and `AUC ≥ 0.60` thresholds (z=2.51, folded
AUC=0.9647), and it also beats its content-matched control (Fix 1: 0.9647 vs.
0.9058). Per the decision rule, any single positive cell makes the overall
gate positive, and that is still technically true here.

**But the substantive conclusion changes materially from the original pass.**
The original framing — "`warmth` is a valence axis, image-modality-specific,
with a real, pre-existing, trainable-free signal" — is **not supported** by
the content-matched control. The corrected framing:

- `warmth`/img is **predominantly a content detector** (companion-animal vs.
  non-animal-scene photos): 0.9058 of its 0.9647 folded AUC is explained by
  content alone.
- There is a small, real residual beyond content (0.059 AUC points) that
  current evidence cannot attribute to emotional valence specifically — that
  would require a valence-controlled test this audit does not run.
- `register` has essentially no signal beyond content or beyond raw-CLIP
  baseline expectations, on any modality.
- Neither axis shows trained checkpoints *exceeding* a raw-CLIP supervised
  probe (Fix 4/5) — but that comparison is dimension-confounded (512-D raw
  vs. 16-D trained probe capacity), so "training adds nothing beyond frozen
  CLIP" is a plausible reading, not an established one; a fair test needs a
  dimension-matched raw-CLIP baseline, not yet run.

**If Experiment 17.2 is pursued targeting `warmth`**, it should be scoped
honestly as targeting **an animal/companion-content-detection axis with an
unresolved small residual**, not as targeting an "emotional valence" axis —
the paper-facing framing from the original report ("proxy for positive
emotional valence") should not be carried forward without a genuinely
content-controlled follow-up test (out of scope for this fix wave). Given that
(a) the content-explained portion dominates the effect, (b) `register` adds
nothing beyond content or beyond raw CLIP, and (c) trained checkpoints do not
clearly exceed a raw-CLIP supervised probe on either axis — though that
comparison is dimension-confounded (512-D vs. 16-D) and not yet a fair test,
so this point is suggestive, not established — **17.2's original routing
("scope to `warmth`, image, as a validated pre-existing signal worth steering")
is on materially weaker footing than the original report suggested** — a
defensible alternative reading is that this is now closer to a **content
axis, not a valence axis**, and 17.2 should treat it as an open question
whether steering "companion-animal-ness" is the paper's intended
interpretability claim, rather than treating 17.1 as having validated a
semantic/emotional axis outright. `register` should not be a 17.2 target on
any of the grounds considered here (raw-CLIP text-anchor null, content-control
null). Impressions remains unsuitable for either axis (both non-degenerate
cells null; both img cells structurally invalid).

## Caveats

- All four axes are proxy labels, not ground truth for "emotional tone" or
  "political framing" specifically — a positive result here means the proxy
  axis is present, not that the originally-envisioned target concept is (see
  the spec's §8 risk row on this). This revision sharpens that caveat for
  `warmth` specifically: the proxy axis itself is now shown to be
  predominantly a content proxy, not even cleanly a valence proxy.
- No verified published benchmark exists for emotion/political-framing detection
  on Reddit *image* content specifically (per this project's literature research,
  2026-09-15) — this audit's own result is the first evidence either way for this
  domain.
- Impressions has no valence-adjacent proxy label at all in this pass; both of its
  axes are registerial/framing contrasts within `caption_type`, not emotional-tone
  contrasts — stated as a data-availability constraint, not routed around. Its
  img-modality cells for both axes are additionally structurally invalid
  (Fix 2) due to near-total pole_a/pole_b photo-set overlap.
- **Family size and statistical caution:** only 6 of 8 (dataset, axis,
  modality) cells are informative after excluding Impressions' degenerate
  img cells (Fix 2). With only 20 control directions per cell, `z` should be
  read as **a rough screen, not a precise p-value** — the `control_rank`
  field (Fix 6, e.g. "20 of 20 controls exceeded") is reported alongside `z`
  for every cell as a more directly interpretable signal of the same
  evidence.
- **Phase B's positive counts overstate independent replication.** Several of
  the 15 redcaps_150k checkpoints have identical or near-identical embeddings,
  most likely from duplicate or resumed training-run directories rather than
  independent training runs. Treat Phase B as qualitative/supplementary
  corroboration, not as 15 (or 6) independent statistical trials.
- **The z-score formula has no minimum-effect-size floor** (a related but
  distinct issue from a near-zero-`control_std` denominator artifact), and
  this revision found it caused a real false-positive-by-z in the original
  pass: Impressions' `aesthetic_vs_description` Phase B cells had z up to
  ≈7.0 with `control_std` around 0.008 (not near-machine-epsilon — a
  legitimately small but non-degenerate control spread) while actual
  selectivity (0.017–0.057) never cleared a reasonable effect-size bar. This
  is a small-effect-size case, not a denominator-collapse case (that failure
  mode — `control_std` near 1e-16 — did occur elsewhere in the original
  Phase B run without flipping any verdict; see the individual cell caveat
  above). Fix 3's `SELECTIVITY_FLOOR = 0.10` catches both this class of case
  and the denominator-collapse class directly (flipping `aesthetic_vs_description`
  from "6/6 positive" to "0/6 positive"), and Fix 2's Jaccard check catches
  the analogous structural-degeneracy failure mode in Phase A. The one
  positive cell that survives everywhere in this audit —
  `redcaps_150k/warmth/img` — has a healthy `control_std` of 0.109 in Phase A
  and clears the effect-size floor by a wide margin, but every other z-based
  claim in this audit's history should be read with both failure modes in mind.
