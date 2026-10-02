# CoSiR v2 CVPR plan: ARS peer review and the revision it triggered

**Report date:** 2026-10-27 (sequence date in this folder; the review ran on 2026-10-02 and 2026-10-03).
**What was reviewed:** the CVPR publication plan
([spec](../../../superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md), revision 1). Five evidence
reports were appended to it: the [literature review](2026-10-21_cvpr_literature_review.md), the
[support-baseline spike](2026-10-22_support_baseline_spike.md), the [aspect-episode spike](2026-10-23_aspect_episode_spike.md),
the [novelty check](2026-10-24_aspect_task_novelty_check.md) and the [backbone check](2026-10-25_backbone_check.md).
The [citation check](2026-10-28_citation_check.md) ran beside the review.
**Full record:** `src/test/20261027_ars_plan_review/`. It holds the reviewer configuration, the blind Phase 1 scoring
plans, the Phase 2 cards, the editorial decision, the 58-item revision roadmap and the provenance artifact.

## Verdict

**Major Revision, with two repairable blocks. No seat found a fatal flaw.**
- Following the contract's stage note, the panel judged the plan as a pre-results plan: missing results did not
  count against it, and design flaws that leave a claim untestable did.
- **D1 (methodology rigor) was at block** for three reasons:
  - the go/no-go rule and K2 could pass with a gain that ignores the condition;
  - the bootstrap resampled anchors although items recur across episodes;
  - "matches" in K3 had no equivalence margin.
- **D3 (argumentative coherence) was at block** through the devil's advocate's CRITICAL, which the synthesis
  validated. Every evaluation aspect had a training partition built to mirror it, so the design could not tell
  "infers the aspect from examples" from "selects among aspects it was trained on".
- The four other dimensions were at warn.

The user approved every proposed change on 2026-10-03, and the spec is now revision 2. Its §14 maps each must-fix
item to its change.

## How the review was run

We used the ARS `academic-paper-reviewer` skill (v3.22.2) in full mode under its v3.6.2 sprint contract
(`reviewer/reviewer_full/v2`).
- **Panel.** A field analyst configured four seats, which the user confirmed:
  - a CVPR area chair for vision and language (journal fit);
  - an ML evaluation methodologist;
  - a conditional-similarity and composed-retrieval researcher (domain);
  - a cognitive scientist of similarity who also annotates emotion in art (perspective).

  The fixed devil's advocate was the fifth seat.
- **Plan-type note.** At the user's request the contract carried this stage note: "Missing results are not a block
  by themselves; design flaws that leave a claim untestable are, and the appended evidence (including negative
  results) counts."
- **Two phases per seat.**
  - Phase 1 committed a scoring plan blind to the plan's content: contract and title metadata only.
  - Phase 2, in a fresh context, scored the plan against that plan's own triggers.
- **Checks.**
  - All ten seat outputs passed `check_phase_conformance.py`, rerun by the controller.
  - The synthesis passed `check_panel_synthesis.py`.
- **Provenance** (`review-panel-provenance/1.0`, replay-validated): roles separated, fresh contexts, blind to peer
  outputs. All seats used one model family, so correlated errors are possible and the letter discloses this. The
  journal-fit seat also saw the other seats' configuration cards, though not their reviews.

## What the panel found, and what changed

| Finding (seats) | Change in revision 2 |
|---|---|
| A condition-blind scorer can pass GO and K2. Its R@1 equals its other-aspect rate, as CLIP's does (11.13 on both), and it could reach about 50% (EIC, methodology) | **Condition gain** (R@1 minus the other-aspect rate, exactly 0 for any condition-blind scorer) is co-primary. A uniform-weight control is added, and GO needs both metrics |
| No held-out aspect (DA CRITICAL, validated; domain and perspective corroborate) | a leave-one-aspect-out test (genre held out) and generic CUB partitions; claim K8, with a rule that narrows C2 if it fails |
| An anchor-level bootstrap with heavy item reuse (methodology) | a bootstrap clustered by painting, species or reference image, with a two-way sensitivity check |
| K3 runs against the evidence (privileged names 13.31 vs 10.34 on emotion) and its subjective/objective typing came after results (EIC, domain, perspective, methodology) | K3 is outcome-neutral, with typing fixed in advance (viewer-response vs curated or physical labels) and a ±1.0 R@1 equivalence margin |
| C3's "belongs to the data" cannot separate modality from how ArtEmis was collected (all four seats) | C3 narrowed. E12 adds a painting-majority emotion reference, annotator agreement, and SemArt's neutral catalogue text as a protocol contrast |
| K5 compared a focus-attribute result with four-task averages (domain, DA) | the bar is now the focus-attribute column (17.9 to 21.1). K5 is a transfer diagnostic unless the text protocol lands |
| In-context baselines that use the same examples were missing (EIC, domain) | an in-context MLLM reranker is required. The per-episode probe and Tip-Adapter cache are restored, and an SAE basis joins K7 |
| The NO-GO fallback was not a CVPR paper (EIC) | three pre-declared branches on Oct 9 (method paper, benchmark paper if the MLLM works, analysis paper otherwise), each with a claim table |
| "No labels" understated the GoEmotions overlap (6 of 8 emotions) and the hand-matched partitions (domain, perspective) | C2's wording is changed. The disclosure, a supervision ablation and an emotion split by GoEmotions coverage are added |
| CUB's held-out species were read by the backbone check (methodology) | disclosed in the spec, the held ledger and the backbone report |

Smaller items were also handled:
- GeneCIS is read only once (E14);
- a pre-abstract review (E19) and a literature completion step (E18) are added before Nov 7;
- the cut order is explicit;
- "ceiling" is renamed "label-probe reference".

The roadmap's should-fix and consider items stay in `revision_roadmap.md` for the implementation plans.

## Errors found in our own reports

The review and the citation check found six factual errors in the appended reports, and we corrected all of them:
- the backbone report's colour-symmetry bound (2.3 to 2.6 points);
- its ArtELingo modality gaps (21 to 35, now 22 to 46 points);
- its GPU time (62 to 88 minutes);
- a mislabelled painting count;
- citation check E1: ArtGAN's genre class file exists;
- citation check E2: Contextual Visual Similarity uses VGGNet features with an unconstrained diagonal weight.

## Caveats

- **One model family.** The panel and the synthesis ran on one model family, so their errors may be correlated.
  Persona separation is not independence.
- **No outside eyes yet.** The review judged a plan, not results, and no human reviewer took part.
- **Cut-off search.** The literature-dependent findings inherit the incomplete search of the two literature reports.
  E18 completes it before the abstract is fixed.
