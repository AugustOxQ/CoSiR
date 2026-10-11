# CoSiR direction log

One entry per loop of the research loop (`/project/claude-config/docs/research_loop.md`): what we read, what the user decided, why,
and where the files are. Newest first. The wrap-up step after reading results adds the next entry. Details live in the
linked reports and handoffs, not here.

New entries follow `~/.claude/templates/direction_log_entry.md`, which adds your reading of the results (decision C)
and who made each decision. Entries before 2026-10-08 were moved from project memory on 2026-10-08 and lack those two
fields; their verbatim notes are in `direction_log_raw_2026-10.md`. Older history (before the CVPR plan) is in the
weekly and stage reports.

## Now (2026-10-11)

AFF (round 1's learned reader, with its gate open only when it picks the affect grouping) is the current best: GO on
fresh seeds 49 to 51 (round 3) and on ArtELingo's held paintings (round 6: +0.60 R@1 over B′(A0), condition gain
+3.06; no margin shown over B′(A1) or over R1; style × genre loses). The user aims beyond the ArtELingo paper, at a
bigger method paper, with the ArtELingo paper as the fallback. Next: design L, starting from the style source (a told
ceiling from the style labels, a screen of style sources, then the grouping redesign), in the `r7 decide` chat.
Go/no-go on Fri 2026-10-16; CVPR abstract 2026-11-10.

## Parked ideas

Candidates proposed since 2026-10-02 but neither chosen nor tried; step 3 reads this list first. Tried-and-killed ideas
stay in the entries below.

- A wider screen of new style encoders (ALADIN, DINOv2, SigLIP 2) for design L (not chosen at round 7's idea choice, 2026-10-11, for the user's generality principle; [proposed in](../../src/test/20261126_design_l_style_source/step3_candidates.md), idea 3)
- X-Cluster criteria (a multimodal LLM proposes criteria such as "style") placed under our reader, the comparator reviewers will ask for (paper stage; [proposed in](../reports/literature/2026-11-28_design_l_novelty.md), §1)
- COCO-4c (5,000 COCO images labelled for activity, location, mood and time of day, with COCO captions) as the second dataset for a generality test (never tried; [proposed in](../reports/literature/2026-11-28_design_l_novelty.md), §6)
- A learned source that cuts across content: a contrastive head on frozen CLIP whose negatives come from the query's own content group (never tried; [proposed in](../reports/literature/2026-11-28_design_l_novelty.md), §1)
- Fair clustering (FCMI) with the content grouping as the protected attribute, as a grouping builder (never tried; [proposed in](../reports/literature/2026-11-28_design_l_novelty.md), §1)
- Dropping the CLIP caption grouping, keeping the affect, image and CSD groupings (deferred by the user after a spike, 2026-10-06; [proposed in](handoffs/2026-10-06-round3-affect-gate-handoff.md))
- Caption-to-caption agreement to detect the emotion side, with retrained readers for idea 3 (never tried; option (c) in [whats-next](handoffs/2026-10-08-whats-next-handoff.md))
- Separate gate weights for image and caption queries (never tried, ranked low; [proposed in](../reports/auto/v2/2026-11-20_r1_levers_brainstorm.md), §4 idea 5)
- Practice-bank detectors with richer negatives or a "none" class (never tried, judged low value; [proposed in](../reports/auto/v2/2026-11-20_r1_levers_brainstorm.md), §4 idea 9)
- A grouped concept basis (N3) and "infer the aspect name, then embed" (N5) (never tried; their privileged references were never run; [proposed in](../../src/test/20261107_new_method_candidates/synthesis.md), §4.2)
- Other datasets and paper framings, CUB and SemArt (deferred by the user, 2026-10-07; [proposed in](../reports/stage/2026-10-07_cvpr_readiness.md), §3)

## 2026-10-11 · Round 6 held test GO; design L next, from the style source

- **Read:** GO under the rule: all seven checks pass with Holm, pooled over 36,864 held episodes (seeds 52 to 54).
  AFF 19.44 R@1: +0.60 [+0.46, +0.73] over B′(A0), +0.63 over B and over its matched control, +6.19 over cosine;
  condition gain +3.06 [+2.87, +3.25]. Not established: a margin over B′(A1) (+0.14 [−0.03, +0.30]) or over R1
  (+0.02 [−0.06, +0.11]). Style × genre loses (−0.43 [−0.65, −0.21] against B′(A0)).
- **Meaning (C):** "we may work beyond the ArtELingo paper, toward a bigger method paper; if the method fails, we drop
  back to the ArtELingo paper" (user, after seeing ours; the user's first reading, before ours, was that the GO already
  makes an ArtELingo-only paper). Ours: the pooled claim holds, but it is thin for reviewers (small margin, no margin
  over B′(A1), the affect gate not shown to matter against R1, style × genre loses); design L first, the ArtELingo paper
  drafted in parallel. The two agree that the GO carries the ArtELingo paper's claim; the user then set that paper as
  the fallback and aims higher.
- **Decided (A):** design L next, in its own loop (`r7 decide`; type and loop size set there), starting from the style
  source. The source question is the user's own ("can we solve style × genre without touching the grouping source?");
  the order, a told ceiling from ArtELingo's style labels, then a label-free screen of style sources by their match to
  style against genre, then the grouping redesign on the best source, is our proposal, accepted by the user.
- **Also decided (user):** the descriptive-pass fix of round 6 is ratified (DTS rows only); the storage summary
  approved in full (disk ledger, 2026-10-11).
- **Why:** style × genre is the clearest weakness and the only route past B′(A1) (−1.24 there); round 2's no-caption
  spike showed a told style × genre gain of only +0.134 with CSD, so the limit is the style signal in the grouping's
  source, not the reader.
- **Links:** report `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md`; user-read
  `docs/user_read/2026-10-10_artelingo_held_test.md`; handoffs `docs/superpowers/handoffs/2026-10-10-r6-read-handoff.md`
  and `docs/superpowers/handoffs/2026-10-11-r7-decide-handoff.md`.

## 2026-10-07 · Idea 3 killed at development; next step open

- **Read:** placing captions by their own GoEmotions scores placed them far better (86% against 36%), but neither
  candidate cleared the +0.5 bar (G-T +0.356, G-TF +0.427 over B′(A0)) and both lost to AFF on seed 42. The weak
  image head caps every image-caption agreement.
- **Meaning (C):** "a normal idea test, and it's going a bit narrow": an ordinary failed idea, and work on the affect
  score is narrowing (2026-10-09; user, after seeing ours, since the briefing showed our advice openly). Ours: the image
  head caps every agreement, so stop work on the affect score. Both readings stop the affect line.
- **Decided (A):** the ArtELingo held-split paper test with AFF frozen, now; type claim test, full loop. Design L
  follows in its own loop and chat. The go/no-go moves to Fri 2026-10-16 (plan revision 4, §16): an ArtELingo-centred
  paper first, the full paper when time and results allow (user, after seeing the recommendation; the go/no-go move and
  the paper target are the user's own).
- **Idea (B):** AFF frozen as in round 3, read on the full ArtELingo held split; the grill (2026-10-09) settled the
  rest: round 3's seven pass checks under Holm, 4,096 episodes per pair on seeds 52 to 54, B′(A1) and R1 as secondary
  checks, a new describe-then-score comparator reported beside (the build stops before the read if it beats AFF on
  seed 42), a pooled claim on new paintings, constitution C5 amended to one held read per final method and backbone
  (user; details in the handoff `docs/superpowers/handoffs/2026-10-09-r6-decide-2-handoff.md`).
- **Why:** the held test is required by every paper option with a method and is cheap (a day, mostly safeguards); the
  user sees hope in affect steering and wants more, through design L. Three DAS6 nodes (nine GPUs) are now available.
- **Links:** report `docs/reports/auto/v2/2026-11-23_idea3_goemotions.md`; user-read
  `docs/user_read/2026-10-07_idea3_goemotions.md`; handoff `docs/superpowers/handoffs/2026-10-08-whats-next-handoff.md`
  (new Herdr tab `whats-next`).

## 2026-10-07 · A CLIP fine-tuning comparator

- **Read:** the CVPR readiness memo (§2): a reviewer will ask whether a lightly fine-tuned CLIP closes the gap.
- **Decided (user):** build a label-free contrastive fine-tune of CLIP as a reported baseline; how it enters the paper
  is on hold.
- **Outcome:** about 15.0 R@1 pooled over seeds 49 to 51, against B′(A0) 18.29 and AFF 18.88: a reported baseline, not
  a floor.
- **Links:** spec `docs/superpowers/specs/2026-10-07-clip-lightweight-ft-design.md`; report
  `docs/reports/auto/v2/2026-11-24_clip_lightweight_ft.md`; memo `docs/reports/stage/2026-10-07_cvpr_readiness.md`.

## 2026-10-07 · Round 4 killed; idea 3 next

- **Read:** none of the three vetoes on AFF's gate beat AFF on seed 42; they also shut emotion-side steering that paid.
- **Decided (user):** try idea 3 (GoEmotions placement of captions) before the held-split paper test, to keep both held
  reads for the paper.
- **Links:** report `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md`; user-read
  `docs/user_read/2026-10-07_round4_vetoes.md`; handoff `docs/superpowers/handoffs/2026-10-07-idea3-goemotions-handoff.md`;
  spec `docs/superpowers/specs/2026-10-07-idea3-goemotions-design.md`.

## 2026-10-07 · Round 3 GO; improve the method before the paper test (option B)

- **Read:** AFF beat every comparator on fresh seeds 49 to 51 (bar margin over B′(A0) +0.59 [+0.46, +0.73]; over R1
  +0.20). On style × genre it fell below B′(A0), and a random-share control matched it: the gain comes from steering
  mostly one side.
- **Decided (user):** round 3's GO stands and AFF, frozen as tested, is the current best. Option B: improve the method
  first with the brainstorm's remaining ideas (2: CSD as detector evidence; 3: GoEmotions placement; 4: style × genre
  abstention). The held-split paper test is deferred, not dropped.
- **Links:** report `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md`; user-read
  `docs/user_read/2026-10-06_reader_fix_affect_steering.md`; handoff
  `docs/superpowers/handoffs/2026-10-07-method-improvements-handoff.md`; round 4 spec
  `docs/superpowers/specs/2026-10-07-round4-aff-vetoes-design.md`.

## 2026-10-06 · Round 3: AFF straight to fresh seeds

- **Read:** the brainstorm on R1's levers found R1's margin only in emotion conditions; its top idea, AFF (one-sided
  affect steering), reached +0.70 over B′ on seed 42 after about 50 variants. The no-caption spike was weaker
  (+0.256 against +0.444).
- **Decided (user):** pre-register AFF as round 3 and test it straight on seeds 49 to 51, with R1 beside it.
- **Links:** brainstorm `docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md`; handoff
  `docs/superpowers/handoffs/2026-10-06-round3-affect-gate-handoff.md`; spec
  `docs/superpowers/specs/2026-10-06-round3-affect-gate-design.md`.

## 2026-10-06 · Round 2 killed; keep improving R1

- **Read:** no round 2 reader cleared the bar (R1 +0.472, R2 +0.116, R3 +0.077). Style × genre is negative for R1
  (−0.598), and CSD leaks into genre.
- **Decided (user):** not design L now; keep improving on R1, with a no-caption spike and a brainstorm on R1's levers.
  Dropping the CLIP caption grouping is parked.
- **Links:** report `docs/reports/auto/v2/2026-11-19_reader_fix_round2.md`.

## 2026-10-06 · Reader fix with CSD killed; round 2

- **Read:** no candidate of the reader fix with CSD cleared the bar (best, R-c, +0.444 [+0.216, +0.674]).
- **Decided (user):** more reader improvements now (three readers on A0, a top-k restriction, round 1's rule), design L
  later.
- **Links:** report `docs/reports/auto/v2/2026-11-18_reader_fix_csd.md`; spec
  `docs/superpowers/specs/2026-10-06-reader-fix-round2-design.md`; handoff
  `docs/superpowers/handoffs/2026-10-06-reader-fix-round2-handoff.md`.

## 2026-10-06 · Reader fix with CSD in the set

- **Read:** the CSD style grouping gave the best told margin yet (+2.23), but the label-free reader gained nothing
  (about 0): the reader is now the binding part.
- **Decided (user):** fix the reader with CSD in the set; an ARS methodology review came back Major Revision, and the
  user adopted all its fixes.
- **Links:** handoffs `docs/superpowers/handoffs/2026-10-06-reader-fix-with-csd-handoff.md` and
  `docs/superpowers/handoffs/2026-10-06-reader-fix-run-handoff.md`; step 1 `src/test/20261116_grouping_step1_style/`.

## 2026-10-05 · Redesign the grouping first

- **Read:** grouping the affect side as Leiden communities beat k-means at every matched group count (told margin
  +1.14 to +1.64; reader up to +0.50 on seed 42, exploratory).
- **Decided (user):** the grouping component is not well shaped (K = 64 never tested, the same K for every grouping,
  fixed, no interaction between groups), so redesign it first; bars, the reader and robustness runs are on hold.
- **Links:** report `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md`; research synthesis
  `src/test/20261114_grouping_research/synthesis.md`.

## 2026-10-04 · Fix the reader

- **Read:** in the quick checks, no configuration beat its matched condition-free control; N1's apparent pass came from
  a control that also dropped centering. With the told grouping the heads gain +1.14 over their counterpart, but the
  label-free reader only +0.14.
- **Decided (user):** fix the reader: a calibrated or learned reader, a style-targeted image grouping, confidence
  gating, matched controls, and a test on fresh seeds with the rule committed first.
- **Links:** report `docs/reports/auto/v2/2026-11-08_new_method_quick_checks.md`; stage report
  `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`; handoff
  `docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md`.

## 2026-10-03 · Try a new method before choosing a branch

- **Read:** the repair diagnostics pointed to branch 3 (H1 not promising, H3's ceiling too low), and the 8B multimodal
  probe did not select the conditioned aspect.
- **Decided (user):** no branch decision yet; try new methods first; paper writing paused.
- **Links:** repair plan `docs/superpowers/plans/2026-10-03-method-repair-diagnostics.md`; probe
  `src/test/20261106_mllm_probe_8b/PREREGISTRATION.md`.

## 2026-10-03 · E0 to E5 NO-GO; repair method A

- **Read:** the pre-registered E0 to E5 test gave NO-GO (the picked model A3 on seed 43: gain +0.26 [−0.04, 0.56];
  its uniform control beat it by 2.96 R@1), and its map pointed to branch 3.
- **Decided (user):** option 3, repair method A to turn the NO-GO into a GO.
- **Links:** report `docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md`; handoff
  `docs/superpowers/handoffs/2026-10-03-method-repair-handoff.md`.

## 2026-10-02 · CVPR becomes the main goal

- **Read:** affect factor learning was confirmed on held data (2026-10-01).
- **Decided (user):** target CVPR (abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23). The condition
  names an aspect, not a value; ArtELingo aspect episodes are the primary benchmark; method A trains aspect factors;
  distant supervision from PercepT's affect teacher is adopted.
- **Why:** the user's four requirements: strong results, a good storyline, clear baselines on a comparable benchmark,
  and a clear problem definition and method.
- **Links:** spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`; handoff
  `docs/superpowers/handoffs/2026-10-02-cosir-v2-cvpr-publication-handoff.md`.
