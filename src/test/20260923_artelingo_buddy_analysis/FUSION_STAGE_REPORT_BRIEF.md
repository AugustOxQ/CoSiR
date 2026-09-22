# Brief: fusion-mechanism stage report

Write `docs/reports/2026-09-22_artelingo_fusion_mechanism_investigation.md`.
This is a documentation task; do not write or run any experiment code.

## Audience and purpose

Same convention as the earlier `docs/reports/2026-09-22_artelingo_affect_investigation_stage_report.md`
(read it first for structure/tone/terminology-definition style — match it
closely, including defining every metric/term in plain language before or
at first use, so the report is self-contained for a reader unfamiliar with
this specific sub-investigation). That report closed with an open question:
could a fusion mechanism recover both content (genre) and affect (emotion)
structure at once, rather than trading one for the other? This report
covers everything done since to answer that question, in chronological
order, ending at the current honest stopping point.

Read every source file listed below in full before writing — do not rely on
this brief's summaries alone for exact numbers; verify each number against
its source report.

## Source materials, in chronological order

1. `affect_pilot_report.md`, `single_modality_pilot_report.md` — established
   the content-only baseline (emotion AMI=0.0593, genre AMI=0.4384) and the
   GoEmotions-only affect ceiling (emotion AMI=0.1180, genre AMI=0.0396),
   and the early (feature concatenation) fusion sweep's best point (weight=4.0:
   emotion AMI=0.1160, genre AMI=0.0867) — all already covered in the prior
   stage report; use these only as fixed reference numbers, do not re-explain
   their pilots in depth.
2. `late_fusion_pilot_report.md` — late (edge/graph-level) union and
   intersection fusion.
3. `fusion_brainstorm_codex_findings.md` — first two-way brainstorm (mine +
   independent Codex pass) on fusion mechanisms, given the late-union result.
4. `hierarchical_refinement_pilot_report.md` — content-first, affect-within-
   content hierarchical refinement, with its two matched-granularity
   controls (random split, content re-split).
5. `joint_fusion_brainstorm_codex_findings.md` — second two-way brainstorm on
   learned and classical joint-representation mechanisms (cross-attention/
   contrastive dual-teacher, SNF, co-regularized spectral clustering, CCA),
   prompted by the user's own reframe that content and affect might be
   fundamentally orthogonal axes that no single-graph fusion can reconcile.
6. `cca_audit_pilot_report.md` — the go/no-go linear CCA + conditional-
   residual audit both brainstorms converged on as the necessary next step
   before building anything heavier.
7. `learned_student_stage1_pilot_report.md` — the learned two-teacher
   student, Stage 1 (linear projection heads + scalar gate), licensed by the
   CCA audit's positive result.
8. `learned_student_stage2_pilot_report.md` — Stage 2 (small MLP projection
   heads), a controlled capacity-only comparison against Stage 1.

## Structure

### I. Why this investigation started

One paragraph: the prior stage report established that off-the-shelf affect
signal and a properly-converged DEC clustering method each gave real but
modest, non-Pareto gains over content-only, and closed with the open
question of whether a smarter FUSION MECHANISM (not just a better encoder or
a better single-signal clustering method) could do what neither achieved
alone: recover emotion structure while keeping most of content's genre
advantage. This investigation tests that question directly.

### II. Glossary (self-contained, plain language, define before or at first use)

In addition to terms already defined in the prior stage report (repeat the
essential ones briefly for self-containedness: AMI, V-measure, buddy graph,
mutual-kNN, Leiden, K, alpha), define NEW terms introduced in this
investigation:
- Late (edge/graph-level) fusion vs. early (feature) fusion
- Union graph / graph intersection
- Hierarchical/nested clustering, parent/child community
- Matched-granularity control (why AMI is not monotone under refinement —
  explain concretely: splitting a cluster into smaller pieces can raise or
  lower chance-corrected AMI even with zero new information, so any
  refinement result needs a same-size-split control to attribute a change to
  real signal rather than to the split itself)
- Similarity Network Fusion (SNF), co-regularized multi-view spectral
  clustering, in one sentence each (what they are, not implementation
  detail) — for context on why they were assessed and set aside
- Linear CCA (Canonical Correlation Analysis), canonical component, held-out
  canonical correlation, permutation null
- Conditional/partial residual (regressing one view on another, then testing
  the leftover)
- Two-teacher contrastive student, InfoNCE loss, in-batch negatives,
  positive pair, projection head, shared embedding space, gate/gating
  network
- Effective rank / representation collapse, gradient share (what "isolating
  one loss term's gradient contribution" means and why it matters for
  detecting one teacher dominating another)
- Pareto improvement / Pareto bar (as used throughout this investigation:
  beating the best known emotion AMI AND the best known genre AMI
  simultaneously, not just one or the other)

### III. The five things tried, each with: what was tested, exact numbers,
### and the honest takeaway (not just the numeric result)

1. **Late fusion — union and intersection.** Numbers from
   `late_fusion_pilot_report.md`. Explain the mechanism plainly (each view
   builds its own graph independently, THEN the edge sets are combined,
   vs. early fusion's feature-space blending). State the result: union
   (emotion AMI=0.1236, genre AMI=0.1394) beat early fusion on both axes at
   once but still traded off hard against content-only; intersection was
   degenerate (98.96% of nodes isolated before repair). Include the
   follow-up community-count diagnostic (28 content communities collapsed
   to 21 under union, smallest community grew from 6 to 324 members) as the
   mechanistic explanation for WHY union costs genre AMI: it's not a smooth
   dilution, it's Leiden outright merging specific communities once affect
   edges bridge them.

2. **Hierarchical refinement.** Numbers from
   `hierarchical_refinement_pilot_report.md`. Explain the mechanism (content
   partition fixed as a hard parent constraint, affect only splits WITHIN
   each parent, with predeclared minimum-size and cross-seed-stability
   guards before a split is allowed to stand). State the result: real,
   control-verified affect signal (hierarchical emotion AMI=0.1072 vs.
   Control A's random-matched-split emotion AMI=0.0362 — a 0.0709 margin,
   far past the predeclared 0.02 bar) — but genre AMI (0.1954) still fell
   well short of the 80%-retention floor (0.3507), and was barely better
   than Control A's own genre AMI (0.2014), meaning most of the genre cost
   traced to partition granularity itself (28 parents fragmenting into
   ~400+ final labels), not to affect specifically.

3. **The joint-fusion brainstorm's classical candidates (SNF,
   co-regularized spectral clustering).** Summarize from
   `joint_fusion_brainstorm_codex_findings.md`: both were assessed and
   deprioritized on mechanistic grounds (their premise is REINFORCING,
   locally-agreeing cross-view structure; the measured near-empty mutual-
   edge intersection between content and affect graphs argues against that
   premise) rather than being run as pilots. State plainly that this was a
   reasoned, evidence-based deprioritization, not an oversight — name the
   specific falsifiable predictions Codex made for each (e.g. SNF predicted
   to produce a smoothed union rather than a newly discovered shared
   backbone) so a future reader can see the reasoning, not just the
   conclusion.

4. **The CCA audit.** Numbers from `cca_audit_pilot_report.md`. This is the
   pivot point of the whole investigation — explain why. State the result
   precisely: top held-out canonical correlation = 0.7285, far above both
   the predeclared 0.15 bar and its own permutation-null 95th percentile
   (0.0699) — real, substantial, held-out-replicated linear correlation.
   Explain the important reconciliation with the "near-orthogonal" framing
   from earlier: mutual-kNN is a strict LOCAL criterion (top-20, both
   directions), CCA asks a softer GLOBAL question — the two are close to
   orthogonal locally but share real structure globally. State the two
   caveats plainly: held-out edge-retrieval recall from the joint-CCA-space
   graph was real but modest (content 7.18%, affect 7.83%, vs. ~0.1% random
   floor — a ~65-70x lift over chance, but far from strong local structure);
   and the conditional-residual result was a caution, not a confirmation
   (residual-affect-only emotion AMI=0.0914, BELOW the raw affect ceiling of
   0.1180 and below the 80%-retention bar of 0.0944) — meaning the
   emotion-useful part of affect substantially overlaps with the part
   correlated with content, not sitting in an orthogonal leftover as
   hierarchical refinement's result alone might have suggested.

5. **Learned two-teacher student, Stage 1 and Stage 2.** Numbers from both
   pilot reports. Explain the architecture progression (linear heads, then
   a controlled MLP-head capacity increase) and the full result table
   (reproduce exactly, train and held-out, both stages). State the Stage 1
   headline honestly: the only method in this investigation to clear both
   predeclared Pareto targets on train (emotion AMI=0.1284, genre
   AMI=0.2799), with a held-out result that held up reasonably (genre AMI
   actually improved slightly to 0.2901; emotion AMI dropped to 0.1095,
   missing the bar by about 11%, a materially gentler generalization gap
   than the supervised BERT ceiling pilot's drop in the prior investigation).
   State the important nuance about its own predeclared collapse
   determination: the mechanical rule labeled it "Collapsed" because
   content recall ended below its own true epoch-0 (pre-training) baseline,
   but the gradient-share and gate-saturation diagnostics both stayed
   healthy throughout (gradient share converged near 50/50, gate saturation
   was 0% at every checkpoint) — this is better characterized as a genuine,
   non-degenerate two-teacher trade-off than a collapse, a real limitation
   in the predeclared rule (it cannot distinguish healthy rebalancing from
   one-sided starvation using recall direction alone) rather than a defect
   in the trained model. State the Stage 2 result as a clean, informative
   negative: strictly more capacity (small MLP heads vs. linear heads,
   everything else held identical) made BOTH AMI axes WORSE on BOTH splits,
   while remaining clearly non-collapsed by all four criteria — direct
   evidence against "the two teachers are capacity-limited," which is
   exactly the condition under which both brainstorms said further
   architecture (cross-attention, deeper heads) would not be expected to
   help. State plainly that the investigation stopped escalating
   architecture at this point per that predeclared, evidence-based rule —
   not due to running out of ideas, but because the evidence specifically
   argued against the next escalation being useful.

### IV. Cross-method summary table

One consolidated table, every method from this investigation plus the
carried-forward references, train-split numbers (all methods in this
investigation report train-split AMI as their primary number; note this
explicitly as a caveat and point to Stage 1/2's held-out numbers as the only
held-out data points available):

| method | emotion AMI | genre AMI |
|---|---:|---:|
| Content-only | 0.0593 | 0.4384 |
| GoEmotions-only (affect ceiling) | 0.1180 | 0.0396 |
| Early fusion (best point) | 0.1160 | 0.0867 |
| Late fusion — union | 0.1236 | 0.1394 |
| Late fusion — intersection | 0.0510 | 0.2399 |
| Hierarchical refinement | 0.1072 | 0.1954 |
| Learned student, Stage 1 (linear) | 0.1284 | 0.2799 |
| Learned student, Stage 2 (MLP) | 0.1230 | 0.2319 |

Point out explicitly: Stage 1 is the Pareto-best point across every method
tried (highest genre AMI of any method with a non-trivial emotion AMI, and
the highest emotion AMI of any method with non-trivial genre AMI, among
those that aren't single-signal-only references) — the empirical frontier
of this whole investigation.

### V. What this means, and what's still open

- The honest headline: no method found a clean escape from the content/
  affect trade-off — every method still gives up real genre structure to
  gain emotion structure — but the learned two-teacher student (Stage 1)
  found the best available point on that trade-off by a clear margin, and
  did so via the simplest architecture tried in the learned-fusion family,
  with more capacity (Stage 2) making things worse, not better.
- Explicitly flag as open, NOT tasked, for a future session: (a) Stage 1's
  held-out emotion AMI (0.1095) narrowly misses the Pareto bar
  (0.1236) — whether a small, principled adjustment (e.g. loss-weight
  tuning between the two teachers, informed by the now-available gradient-
  share diagnostic, rather than another architecture change) could close
  this gap was not tested; (b) content-anchored re-ranking (affect breaks
  ties only within a narrow content-similarity band), the third-ranked idea
  from the first brainstorm, was never tried; (c) how any of this reconnects
  to Experiment 18's own RedCaps retrieval-vs-interpretability trade-off
  remains untested — everything in this investigation was run on ArtELingo
  only, and RedCaps has no emotion labels to build a comparable affect
  teacher from, so any reconnection would need to use a label-free proxy for
  the affect side, not GoEmotions directly.

### VI. Process note

One short paragraph: this investigation was conducted as a series of
brainstorm-then-verify cycles (a two-way brainstorm between the user, this
session, and an independently-dispatched Codex pass, followed by a locked,
fact-specific implementation brief, independent code review of every script
before execution, and GPU execution kept with this session rather than
delegated) — consistent with the practice established in the earlier
six-pilot affect investigation.

## Verification requirement

Before finishing, re-read every number you wrote against its source report
one more time. This investigation's own established standard (from the
prior stage report) is that every number must be independently checked, not
transcribed once and trusted.
