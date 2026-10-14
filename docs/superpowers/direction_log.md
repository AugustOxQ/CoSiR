# CoSiR direction log

One entry per loop of the research loop (`/project/claude-config/docs/research_loop.md`): what we read, what the user decided, why,
and where the files are. Newest first. The wrap-up step after reading results adds the next entry. Details live in the
linked reports and handoffs, not here.

New entries follow `~/.claude/templates/direction_log_entry.md`, which adds your reading of the results (decision C)
and who made each decision. Entries before 2026-10-08 were moved from project memory on 2026-10-08 and lack those two
fields; their verbatim notes are in `direction_log_raw_2026-10.md`. Older history (before the CVPR plan) is in the
weekly and stage reports.

## Now (2026-10-13)

AFF stays the current best (round 6 held GO: +0.60 R@1 over B′(A0); style × genre loses). r9's spec is approved
(2026-10-13 15:05, [spec](specs/2026-10-13-r9-aff-scope-design.md)): AFF as a fixed recipe on Qwen3-VL-Embedding-2B and SigLIP 2
(held reads H6 and H7, round 6's seven checks per backbone), then the ArtELingo-28 language check on all three
backbones (746 paintings, captions translated to English, a two-level test), and CLIP AFF probes on the CUB and SemArt
development splits. Stage 3 (the CUB, SemArt and Affection reads) gets its own decide chat about 2026-10-17; Affection
is dropped if no data comes by 2026-10-24. Next: the r9 build, after the user's container restart (day run); the paper
chat opens beside it. Results enter the paper body by 2026-11-06; CVPR abstract 2026-11-10, paper 2026-11-16.
Side check (2026-10-14, outside r9): AFF as it is needs at least 3 example pairs; with 1 pair it is below B′(A0)
([k-shot report](../reports/auto/v2/2026-12-03_kshot_supports.md)).

## Parked ideas

Candidates proposed since 2026-10-02 but neither chosen nor tried; step 3 reads this list first. Tried-and-killed ideas
stay in the entries below.

- More example pairs (k = 5 to 7, new episodes) and a re-fit of the reader and gate at 1 or 2 pairs (not chosen by the user, 2026-10-14; [proposed in](../reports/auto/v2/2026-12-03_kshot_supports.md))
- ~~Leiden image and caption groupings at a group count matched to k-means 64~~ taken up 2026-10-14 by the Leiden groupings side check (entry "2026-10-14 · Leiden groupings side check").
- The DINOv2 reader as AFF's fourth grouping (r8 screen: +0.285 [−0.012, +0.592] R@1 over AFF, emotion × style −0.38; its lead sits on style × genre, where it is level with its own matched control, −0.07 [−0.43, +0.30]) (deferred by the user, 2026-10-13, "maybe after the main finished if we have time"; [proposed in](../reports/auto/v2/2026-11-29_learned_grouping.md))
- X-Cluster criteria (a multimodal LLM proposes criteria such as "style") placed under our reader, the comparator reviewers will ask for (paper stage; [proposed in](../reports/literature/2026-11-28_design_l_novelty.md), §1)
- COCO-4c (5,000 COCO images labelled for activity, location, mood and time of day, with COCO captions) as the second dataset for a generality test (never tried; [proposed in](../reports/literature/2026-11-28_design_l_novelty.md), §6; r9 step 3: mood is close to affect, but COCO captions rarely state mood, [2026-12-01 note](../reports/literature/2026-12-01_affect_caption_datasets.md))
- FCMI weights 0.40 to 1.60 with a longer time cap (r8 scored only 0.05 to 0.20, all below AFF; not chosen at r9 step 2, 2026-10-13; [proposed in](../reports/auto/v2/2026-11-29_learned_grouping.md))
- Dropping the CLIP caption grouping, keeping the affect, image and CSD groupings (deferred by the user after a spike, 2026-10-06; [proposed in](handoffs/2026-10-06-round3-affect-gate-handoff.md))
- Caption-to-caption agreement to detect the emotion side, with retrained readers for idea 3 (never tried; option (c) in [whats-next](handoffs/2026-10-08-whats-next-handoff.md))
- Separate gate weights for image and caption queries (never tried, ranked low; [proposed in](../reports/auto/v2/2026-11-20_r1_levers_brainstorm.md), §4 idea 5)
- Practice-bank detectors with richer negatives or a "none" class (never tried, judged low value; [proposed in](../reports/auto/v2/2026-11-20_r1_levers_brainstorm.md), §4 idea 9)
- A grouped concept basis (N3) and "infer the aspect name, then embed" (N5) (never tried; their privileged references were never run; [proposed in](../../src/test/20261107_new_method_candidates/synthesis.md), §4.2)

## 2026-10-14 · Leiden groupings side check: Leiden at 64 ties k-means on the screen; rebuild AFF on Leiden, retrain H1

- **Read:** round 3's told-oracle screen, 12,288 seed-42 development episodes, R@1 margins over the matched control.
  Reference (k-means 64): told +1.64 [+1.37, +1.92], reader +0.35 [+0.15, +0.57]. Leiden at 64 (image r 5.05,
  caption r 2.38): told +1.70 [+1.41, +2.01], reader +0.22 [+0.06, +0.38], reader minus reference −0.14 [−0.36,
  +0.09]. Attention-H1 alone (old labels, mapped): −0.46 [−0.63, −0.28]; the cross-fit gave its conditioned term
  weight 0. Neither passed the screen's bar (agent default: reader margin at least the reference's).
- **Meaning (C):** "If leiden ties with k-means we just go with leiden, because this makes the design coherent"; the
  user expected H1 to be bad "because this is too different from the trained one" (user, after seeing the brief,
  before opening ours). Ours: no reason to switch, stop both. The two agree that Leiden ties; they differ on the
  choice: the user takes Leiden on a tie for coherence.
- **Decided (A):** rebuild AFF on Leiden at 64 (image and caption replaced everywhere, B included: agent default), and
  retrain H1 on the scorer-train paintings only, then its screen (user; screen first for H1: agent default under the
  user's "cheap is best"). Side experiment outside the loop, no loop protocol (user). Only the one-grouping H1 slot
  (user, after seeing the recommendation).
- **Open:** if the rebuild ties, does "go with Leiden" change AFF's fixed recipe (r9, the paper, a new held read), or
  only the model card? To ask with the rebuild's result.
- **Links:** [report](../reports/auto/v2/2026-12-04_leiden_groupings.md) ·
  [user-read](../user_read/2026-10-14_leiden_groupings.md) · [handoff in](handoffs/2026-10-14-leiden-groupings-handoff.md)

## 2026-10-14 · K-shot side check: AFF needs at least 3 example pairs; stop there

- **Read:** frozen AFF with the first k support and contrast pairs of the 12,288 seed-42 development episodes. AFF −
  B′(A0): +0.70 [+0.46, +0.94] at 4 pairs, +0.63 [+0.40, +0.87] at 3, +0.19 [−0.06, +0.44] at 2, −1.73 [−2.05, −1.41]
  at 1. Condition gain 3.11, 3.10, 2.58, 1.99. At 1 pair the gate opens in 62% of condition-b values (30% at 4).
- **Meaning (C):** "Emotion is so different from style and genre that it needs fewer examples; style × genre stays
  hard at any k. This supports 'the gain is mostly from emotion, from the GoEmotions encoders'" (user, 2026-10-14, after
  seeing ours; the user notes it may hold mistakes). Ours: AFF as it is needs at least 3 pairs. A follow-up split by the
  aspect the pairs show (`ks_by_condition.py`): AFF − B′(A0) is +2.88 [+2.40, +3.38] on emotion at 4 pairs and +4.41
  at 1; style about 0; genre −0.69 [−1.07, −0.31] at 4 and −8.44 [−9.11, −7.73] at 1. AFF only steers toward emotion,
  so the pairs mostly tell it when not to steer. The two agree that the gain is emotion's; they differ on "needs fewer
  examples": the emotion gain needs none, the other aspects need the pairs. The GoEmotions source is not tested here.
- **Decided (A):** no k = 5 to 7 (user, after seeing the options); add k = 1, then stop (user). Exploration, small loop,
  side chat outside r9. The k = 1 rule for the reader's spread features: agent default.
- **Why:** a cheap check of how many example pairs AFF needs, in the spirit of k-shot learning (user).
- **Also answered:** why Leiden for affect and k-means for the CLIP groupings. k-means 64 was inherited from stage (d);
  Leiden on CLIP features lost in the 2026-11-16 descriptive arm (17 and 30 groups). The matched-count check is parked.
- **Links:** [report](../reports/auto/v2/2026-12-03_kshot_supports.md) ·
  [user-read](../user_read/2026-10-14_kshot_supports.md) · [handoff in](handoffs/2026-10-14-aff-questions-handoff.md)

## 2026-10-13 · Round 8: no pick in either track, AFF stays best; next, AFF on more backbones and datasets, and the paper

- **Read:** gate (`results/gate_verdict.json`, 2026-10-13 11:20): no main pick and not promising; no screen pick. Rule
  §8: round 8 ends at the gate, AFF stays current best, no held read, no `HELD_RULE.md`. Main line: 20 scored scorers,
  SC3 from −0.466 to −0.802 R@1 points against AFF, every upper bound below 0; the best is extended AFF on FCMI weight
  0.20, −0.466 [−0.651, −0.288]. FCMI weights 0.40 to 1.60 were dropped by the 60-minute cap (D21), 1 to 3 epochs
  short. NBC triggered and did not pass. Screen (each encoder in CSD's place): the DINOv2 reader +0.285 [−0.012,
  +0.592], emotion × style Σ(4d) −63 against the bar −49, genre +19; the SigLIP 2 reader +0.059, style −35 (passes),
  genre −64 (fails). Reference: raw CSD under own placement, R1 −0.173. Both C13 re-derivations agree (0
  disagreements).
- **Amendment 1** (2026-10-13 01:13, made after a result): M4's view-check threshold 0.9999 → 0.999 and a full
  restart. Lowering: user; the value 0.999: **user, after seeing the recommendation**. The verdict does not depend on
  it: all six view-grouping scorers fail the gate.
- **Meaning (C):** "learned groupings on frozen CLIP failed with basically no hope"; DINOv2 is "below the line but seems
  worth a little digging (not too much)" (user, before seeing ours). The user first read FCMI as having no results
  because of the cap; after our note that weights 0.05 to 0.20 were scored and all lose to AFF, the user accepted it
  ("that solved my mis-read of FCMI results"). Ours: the learned groupings fail clearly; DINOv2 is a narrow miss, not a
  pass; the completed FCMI weights give no trend towards a pass; we recommended closing the round and writing the
  paper with AFF, with DINOv2 as a later question. The two agree on the verdict; the user wants a little DINOv2 work
  now, where we would park it.
- **Asked before A:** does AFF alone pass the CVPR line? Ours: enough to submit (held read passed; external baselines
  far behind), main-track acceptance doubtful; the first attack is scope (one dataset, one backbone), then the small
  margin, the mechanism (+0.02 over R1), style × genre and the GoEmotions overlap. The user: AFF "depends too much on
  the selected encoder and selected dataset. Reviewers will easily attack this point."
- **Decided (A)** (2026-10-13): stay with AFF, frozen; the grouping line closes. Work on AFF's scope for the CVPR
  paper:
  - Backbones: Qwen3-VL-Embedding-2B first, then SigLIP 2; each a claim test with its own held read (user, own pick).
  - Datasets: both routes, CUB and SemArt as scope runs (the task, the comparators and AFF; no gain expected, a small
    loss likely, as on style × genre), and a search for a second dataset with affective captions (user, own pick
    among our three routes, no recommendation shown). The user added ArtELingo-28 as a candidate.
  - Paper: a `claude-paper` chat with round-by-round ARS review and revision (user, own pick).
  - DINOv2 reader: parked, "maybe after the main finished if we have time" (**user, after seeing the
    recommendation**). FCMI with a longer cap: parked (agent default; the user did not take it up).
  - Type and loop size: open, set at the grill (agent default to propose: a claim test per backbone and dataset, full
    loop, with step 3 reduced to the dataset search).
- **Step 3:** literature `docs/reports/literature/2026-12-01_affect_caption_datasets.md`: Affection (85,007 real photos,
  ArtEmis-style explanations) is the only cross-domain fit, but its release has no second label (a COCO-category join
  is unverified); EmoSet has many aspects and no captions; ArtELingo-28 is the same domain (1,430 to 2,000 paintings;
  AFF needs translation or a multilingual emotion classifier there).
- **Idea (B), the datasets** (2026-10-13; user, own pick, no recommendation shown): both, Affection with a COCO-category
  join as the second aspect, after a feasibility check (the COCO share of its images, the leakage of object words into
  explanations, access by the request form, sent by the user), and ArtELingo-28 as a language check. CUB and SemArt
  stay as scope runs (decision A).
- **Why:** the paper's first weakness is scope; a frozen AFF on more backbones and datasets answers it without changing
  the method that the held read tested.
- **Grill (step 5)** (2026-10-13, `r9 decide 2`): the key answers are spec §4 D1 to D13, each with who decided. Order:
  backbones on ArtELingo, then ArtELingo-28 on all backbones, then CUB, SemArt and Affection (user). "Frozen" means
  the fixed recipe: every part above the encoder re-fitted, the same automatic seed-42 pick re-run (user); if the pick
  sets λ_a = 0 in both halves, the backbone's reads are not run (user). Bar: round 6's seven checks per backbone
  (user). ArtELingo-28: translated to English (user), 746 paintings (user), the two-level test (**user, after seeing
  the recommendation**; the idea was the user's). Paper chat opens now, cut-off 2026-11-06. Held reads stay local
  (user, after seeing the recommendation). Terms added to `GLOSSARY.md`.
- **Spec (step 6):** [2026-10-13-r9-aff-scope-design.md](specs/2026-10-13-r9-aff-scope-design.md), approved 2026-10-13 15:05; handoffs
  [r9 build](handoffs/2026-10-13-r9-build-handoff.md) and [paper start](handoffs/2026-10-13-paper-start-handoff.md).
- **Links:** report `docs/reports/auto/v2/2026-11-29_learned_grouping.md`; user-read
  `docs/user_read/2026-10-13_learned_grouping.md`; specs `docs/superpowers/specs/2026-10-12-r8-learned-grouping-design.md`
  and `docs/superpowers/specs/2026-10-12-r8-encoder-screen-design.md`; rule
  `src/test/20261129_r8_grouping/DECISION_RULE.md` @ fdaee39; run log
  `src/test/20261129_r8_grouping/20261129_r8_grouping_log.md`; handoffs
  `docs/superpowers/handoffs/2026-10-12-r8-build-handoff.md` and `docs/superpowers/handoffs/2026-10-13-r9-decide-handoff.md`.

## 2026-10-12 · Round 7 design L: ceiling CLEAR, gate n = 0, Friday NO-GO; next, a learned grouping across content

- **Read:** K1 CLEAR: told true styles gain +3.92 [+3.35, +4.49] R@1 on style × genre over the matched counterpart
  (Σ(4d) 643, bar 164; prior +0.9). Seed-42 gate n = 0 of 12 candidate scorers: S1 fails 12 of 12 (best +0.25 against
  the bar +0.5; AFF itself +0.33), S3 emotion × genre fails 12 of 12 (−0.93 to −1.61 against AFF), S2 passes 5 of 12.
  Design L ends; Friday verdict NO-GO by the rule; AFF stays the current best. "Promising" for R1 on V2 and V3 and
  extended AFF on V3 is weak: two references meet it too, and the random controls beat the candidates on margin and
  gain. AMI: raw CSD style 0.34, genre 0.33; the cuts kept both, the projections removed both.
- **Meaning (C):** "the ceiling is far better than our expectation, which means, if working towards the correct
  direction, we could gain a lot on style × genre"; genre and style are "entangled so close that they cannot be set
  apart from CSD, so CSD almost treats them equally as a single aspect", so "we might need to find another encoder
  that disentangles better"; "go back to the parked idea to gain more on the grouping", on which every other part of
  the project rests (user, before seeing ours). Ours: the verdict is sure, its cause is not; the label-free groupings
  reach about a tenth of the ceiling, and the flag does not show that the removal step helped; recommended the
  ArtELingo paper with AFF, and if design L returns, the told route with V3 first. The two agree on the verdict and
  that the grouping is the likely limit; the user names a cause we hold consistent but not shown, and wants grouping
  work now where we recommended the paper first.
- **Also decided (user, own pick, before seeing ours):** the L3 departure: keep the fallback, "this evidence stays in
  the record without the rank-63 re-run". Ours agreed. Any later reuse of L3 names the rank (groups − 1).
- **Decided (A), in part; step 2 continues in `r8 decide 2`:** grouping work, "which every other part of this
  project is based on" (user). Main line: a learned grouping head on frozen CLIP whose negatives come from the query's
  own content group, as the reader's fourth grouping in CSD's place (user: "I think this is a good one", before any
  recommendation of ours). In parallel: a screen of fixed encoders (ALADIN, DINOv2, SigLIP 2) on the round 7
  pipeline, run on the cluster, read as an engineering finding, not a method claim (user). X-Cluster criteria and
  told prompts are comparators (user). Still open: FCMI's role (builder, comparator or drop), the paper plan (beside,
  instead of, or after the ArtELingo paper), placement in scope, and the type and loop size (agent default: new
  idea, full loop).
- **Decided (A), completed in `r8 decide 2`** (2026-10-12; each pick given before our view):
  - FCMI is a builder beside (ii), with the same tuning budget fixed before any seed-42 result, and is reported as a
    comparator too (user: the cost difference is small, "we spend a little bit more to gain the whole picture"; our
    page had said comparator, and we came to agree).
  - Paper plan: the grouping work runs beside the ArtELingo paper with AFF. Hard cut-off Fri 2026-10-23: the pick's
    held-read verdict must exist by then; a later verdict stays out of the CVPR paper (user, own pick among three
    dates; ours agreed).
  - Placement: gallery images take the new grouping's own assignment; captions keep the CLIP text head; caption
    placement waits for a later loop (user: "probably image part can gain more"; ours agreed, since the grouping is
    the first limit). Paired-image reference, told route, seed 42: true styles on style × genre +3.92 to +10.87, raw
    CSD +0.13 to +0.59 (`src/test/20261129_r8_grouping/step3_caption_ref/`).
  - Tracks: the encoder screen (i) is its own track with its own spec, report and pick; (ii) and (iii) form the main
    line (user; ours had leaned to one pool for the cut-off).
  - Type and size (agent default, accepted by the user): main line, new idea, full loop; encoder screen, medium loop.
- **Step 3:** literature `docs/reports/literature/2026-11-30_r8_learned_grouping.md` (MCL is the closest prior: the
  mechanism is not ours to claim; default augmentation positives keep content, not style).
- **Idea (B) and the grill** (`r8 decide 2`, 2026-10-12; spec to follow):
  - Positives for (ii): both forms, views of the same painting (a flip and four corner crops, no colour change) and the
    nearest CLIP neighbour outside the query's content group (user, own pick: no prior knowledge settles it). A
    CSD-space probe runs only if the CLIP form fails, as an extra candidate (user).
  - Settings: six per builder, (ii) two forms × three strengths (share of own-group negatives 100, 67, 33%), FCMI six
    fairness weights; 12 groupings of 17 groups, R1 and extended AFF, 24 candidate scorers, one overnight run (user,
    own pick, "an overnight run"; ours had been four each, for a smaller selection). The grid is not extended in this
    loop.
  - Seed-42 gate: beats AFF pooled (lower bound above 0) and S3 holds; round 7's gate is reported as the weaker goal
    ("promising" for work after the cut-off) (user, own pick; ours agreed).
  - Held read (seeds 52 to 54, the same episodes as AFF's H5): AFF's seven checks plus "beats AFF" under Holm at a
    family-wise one-sided 0.0125 (halved for two tracks); Tier 1, all eight pass: replaces AFF; Tier 2, beats AFF but
    fails P6 or P7: AFF stays; Tier 3, the seven pass but not AFF: AFF stays; the grouping goes to the analysis in 2
    and 3 (**user, after seeing the recommendation**; the user had first put "beats AFF only" as replacing AFF).
  - Priority: main line first; the screen's pick takes the held read only if the main line fails and the screen
    passes (user, own pick; Holm across the two added by us, accepted).
  - Agent defaults accepted: 17 groups; comparators raw CSD, random groupings, NrKmeans and told prompts (privileged);
    X-Cluster at the paper stage; MCL cited, not run; a CLIP-head placement arm as description; groupings on the
    scorer-train rows; FCMI drops out if its smoke run fails at 64 groups; the screen (ALADIN, DINOv2, SigLIP 2; 17
    groups as raw CSD; 6 candidate scorers) on the local GPU; one build chat and one overnight run for both tracks; no
    plan page.
- **Why:** told true styles gain +3.92 while the label-free groupings reach +0.38; CSD agrees with style and genre
  about equally; a removal approach fights the true styles' own link to content (AMI 0.32), so the user wants a
  better grouping source rather than another cut of CSD.
- **Links:** report `docs/reports/auto/v2/2026-11-26_design_l_style_source.md`; user-read
  `docs/user_read/2026-10-12_design_l_style_source.md`; spec `docs/superpowers/specs/2026-10-12-r7-design-l-design.md`;
  handoffs `docs/superpowers/handoffs/2026-10-12-r8-decide-handoff.md` and
  `docs/superpowers/handoffs/2026-10-12-r8-decide-2-handoff.md`; FCMI explainer `docs/user_read/2026-10-12_fcmi_intro.html`.

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
