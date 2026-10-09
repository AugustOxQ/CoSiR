# CoSiR v2 for CVPR: where the method stands, what the paper can claim, and what to decide

> 7 October 2026 (Amsterdam). A decision memo written after reader-fix round 4, answering four questions from the
> user: (1) whether a CLIP fine-tuned on ArtELingo is needed as a comparison, (2) how close the current design is to
> the CVPR GO bar, (3) what the novelty and contributions are, and (4) how to analyse the results deeply and clearly.
> It decides nothing; the decisions in §6 are the user's. Facts come from the reports cited per section; judgements
> are marked as ours.
>
> **Update, 7 October (the user's decisions):** the framing (§3, §6 item 1) is on hold for now. The fine-tuned CLIP
> comparator (§2) is adopted in a lightweight form with three variants (a linear probe, the last block, LoRA) and no
> full fine-tuning; the experiments run on the DAS6 nodes 404, 405 and 411.

## Summary

- **The method we have** is AFF: a label-free reader that adds its grouping term to B, the strongest condition-free
  score of the project, only when it picks the affect grouping. On fresh ArtELingo episodes it beat every
  condition-free comparator, but by +0.59 R@1 over the strongest one (18.88 against 18.29), with the whole gain on the
  two emotion pairs and a loss on style × genre.
- **Against the CVPR plan's GO bar** (K2: beat backbone-only, the best metric-from-pairs and the matched control on
  three datasets' held splits, with multiplicity correction), we have one dataset's development rows. CUB and SemArt
  are not started for this method, and its mechanism depends on a non-visual aspect that those datasets lack.
- **Our view:** a method paper that meets K2 on three datasets by the deadline (abstract 10 November, paper 16
  November) is unlikely. An ArtELingo-centred paper (task and protocol, analysis, AFF as a transparent label-free
  method with a held test) is feasible; whether it is strong enough for the CVPR main track is doubtful, and a
  datasets-and-benchmarks track or a workshop is the natural fallback.
- **A fine-tuned CLIP** is not required as a headline competitor, but a cheap version (contrastive fine-tune on
  ArtELingo image–caption pairs, scored by cosine) should join the condition-free comparators before the held test,
  because it could raise the floor exactly where AFF gains. **Adopted by the user:** three lightweight variants (a
  linear probe, the last block, LoRA), no full fine-tuning (§2).
- **The strongest contribution** is the task with its evaluation protocol (matched controls, condition gain, swap
  test), then the analysis (modality asymmetry; what label-free condition reading can and cannot do). The method's
  novelty is modest and must be described as what it is: a side detector distantly supervised by GoEmotions.
- **Decisions for the user (§6):** the paper's framing (on hold since 7 October), which decides whether CUB and SemArt
  work starts; how B′(A1) enters the held test; the analysis items to fund. The fine-tuned CLIP comparator is decided
  (three lightweight variants).

## 1. Where we stand

The task (plan §3): rank candidates in the other modality for a query image or caption, when the wanted aspect is shown
only by 4 support pairs and 4 contrast pairs and never named. Each ArtELingo episode has 13 candidates; the target
shares one aspect with the query and a distractor shares the other aspect of the pair. Both conditions (supports and
contrasts swapped) and both directions are scored. **Condition gain** is R@1 minus the rate at which the other
aspect's candidate comes first; any condition-free scorer has gain 0.

*Table 1. AFF and its comparators on the fresh seeds 49 to 51 (36,864 ArtELingo episodes on selection rows, pooled;
R@1 in %, differences in points with 95% painting-bootstrap intervals; round 3).*

| Scorer | R@1 | AFF minus it | What it is |
|---|---|---|---|
| AFF | 18.88 | | the method |
| B′(A0) | 18.29 | +0.591 [+0.462, +0.729] | strongest condition-free comparator (B on AFF's own groupings) |
| matched counterpart | 18.08 | +0.796 [+0.670, +0.920] | AFF with only the condition removed |
| B | 18.07 | +0.806 [+0.686, +0.930] | best condition-free score of the project |
| RCA | 13.16 | +5.72 | best metric learned from the example pairs (E1) |
| cosine (CLIP ViT-B/32) | 13.04 | +5.84 | backbone only |

What qualifies it (rounds 3 and 4):

- **Per aspect pair** (bar margin against B′(A0)): emotion × style +0.810, emotion × genre +1.544, style × genre
  −0.580 [−0.793, −0.363]. The pooled result rests on the two emotion pairs.
- **The gain is one-sided steering.** A random gate with AFF's open share per condition matched AFF (differences
  −0.04 and +0.03). AFF's label-free part behaves like a visual-contrast rule: it steers when the visual groupings
  agree more on the contrasts than on the supports.
- **Cost:** about 0.52 points of either rate per point of condition gain.
- **A stronger condition-free scorer exists.** B′(A1), which adds the CSD style grouping, reached 18.805 on seed 42,
  and AFF was only +0.332 [+0.048, +0.625] above it there, and below it on style × genre (round 4 report §9).
- **Round 4** tried three vetoes on AFF's gate; none beat AFF on development data (kill). Idea 3 (placing captions in
  the affect grouping by their own GoEmotions probabilities) is the next measured step, in its own tab.

*Sources: `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md` (Summary, §7, §8); `2026-11-22_round4_aff_vetoes.md`
(Summary, §5, §9); the CVPR plan `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` §3.*

## 2. Question 1: do we need a CLIP fine-tuned on ArtELingo as a comparison?

**Answer: not as a headline competitor; yes as one more condition-free comparator, in its cheap form, before the
held test.**

- **Why it is not required.** The plan puts backbone fine-tuning out of scope (§9). Our claim is about reading the
  condition, measured against matched controls on the same backbone. A fine-tuned CLIP is one more condition-free
  scorer, and B already contains an ArtELingo-trained, label-free component (the method-A factor term).
- **Why a reviewer will still ask.** AFF's absolute R@1 is 18.88 against plain CLIP's 13.04. A contrastive fine-tune
  on ArtELingo's emotion-laden captions would most likely raise the condition-free floor most on emotion, which is
  exactly where all of AFF's gain sits. If such a model alone came close to AFF, the claim "beats every condition-free
  scorer" would weaken. The backbone check found that four frozen backbones barely move the weak-side probes (emotion
  from images 35.1 to 36.6), so backbone strength alone is not the limit; in-domain fine-tuning was never tested.

*Table 2. Two kinds of fine-tuned CLIP and what each would mean.*

| Variant | Supervision | Role in the paper | Cost |
|---|---|---|---|
| (a) contrastive fine-tune on scorer-train image–caption pairs | none from the evaluation (domain adaptation) | a fair condition-free comparator; optionally a backbone for AFF | training: a few GPU hours (ViT-B/32, about 184k pairs); cosine scoring on episodes: hours |
| (b) fine-tune with emotion, style or genre labels | evaluation labels | a supervised reference, like the existing label probes (tier 3); not something we must beat | similar training cost |
| AFF on top of (a) | as AFF | robustness row (like K4's second backbone) | days: groupings, heads, B's factor term and the reader's practice banks all depend on the backbone |

**Our recommendation:** train (a), score it by cosine on the development seeds, and add it to the held test's
comparator list (as a GO comparator if it lands near AFF, otherwise as a reported baseline). Running AFF on top of it
belongs with K4 and only if time allows.

**Decided by the user (7 October).** Variant (a), in three lightweight forms, with no full fine-tuning:

| Form | What is trained | Why it is in the set |
|---|---|---|
| Linear probe | a linear map on the frozen image and caption features | the cheapest adaptation; shows how much a re-weighting of CLIP's space alone buys |
| Last block | the last transformer block of each encoder | the standard partial fine-tune |
| LoRA | low-rank adapters in the encoders' attention | the standard parameter-efficient fine-tune |

Each is trained without evaluation labels on scorer-train image–caption pairs, chosen on held-out image–caption
retrieval (never on aspect episodes), and scored by cosine on the development episodes beside AFF and its
comparators. The runs use the DAS6 nodes 404, 405 and 411 (up to 9 GPUs). The design details are in the experiment's
spec.

**Outcome (7 October; report `docs/reports/auto/v2/2026-11-24_clip_lightweight_ft.md`, final-reviewed).** Each variant
was selected on val retrieval only. Pooled over the fresh seeds 49 to 51, the fine-tuned cosine reached R@1 15.00
(linear probe), 14.97 (last block) and 15.14 (LoRA), against 13.04 for plain CLIP, 18.29 for B′(A0) and 18.88 for AFF.
Fine-tuning added about 2 points over plain CLIP, mostly through the genre candidate, and stayed about 3.2 points below
B′(A0); AFF beat every variant by +3.7 to +3.9 with intervals far from 0. Our reading: a lightweight fine-tuned CLIP is
a reported baseline for the held test, not the strongest condition-free comparator; B′(A0) and B′(A1) remain the floors.

*Sources: CVPR plan §8, §9; `docs/reports/auto/v2/2026-10-25_backbone_check.md`.*

## 3. Question 2: how close are we to the CVPR GO bar?

Two different bars apply, and only the first is met.

- **The rounds' bar** (our pre-registered fresh-seed test on ArtELingo selection rows): met by AFF in round 3, all
  seven checks pooled and on each seed.
- **The paper's bar** (plan §4, K2, with §10's statistics): our method beats backbone-only, the best metric-from-pairs
  baseline and the uniform-weight (matched) control on **both** R@1 and condition gain, on the **held splits** of
  ArtELingo, CUB and SemArt, Holm-corrected per backbone; plus K3 (examples against names), K4 (a second backbone) and
  K7 (the learned basis; written for the earlier factor method).

*Table 3. The paper's requirements against what exists (status on 7 October).*

| Requirement | Status | Risk (our view) |
|---|---|---|
| ArtELingo held read (1 main + 1 reserve; 0 of 2 used) | not done; planned after idea 3 | low to moderate: fresh seeds held 84% of the development margin |
| CUB (50 unseen species) | not started for this method; no CUB result for any method of this line | **high**: AFF's gain comes from emotion, a non-visual aspect with a GoEmotions-built grouping; CUB's aspects (colour, bill shape, wing colour) are visual |
| SemArt | not started | **high**: type, school and timeframe are catalogue facts, mostly visual |
| K4, second backbone | not done | moderate; days of rebuild per backbone |
| Tier-1 baselines | done on ArtELingo development rows: cosine, RCA and eight other metric-from-pairs baselines, Tip-Adapter and the per-episode probe among them (none separates the aspects, gain within 0.4 of 0), B, B′ and the matched control | low |
| In-context MLLM given the same examples | Qwen3-VL-2B and 8B probes did not work (8B: R@1 +1.07 [0.17, 1.93] over cosine, gain +0.21 [−0.51, 0.94]) | favourable for us |
| K3, examples against names | open; an early spike favoured names on emotion (13.3 against 10.3) and privileged names gained +1.51 on the aspect spike | moderate: names may win on emotion |
| Effect size | +0.59 R@1 (about 3% relative) over the strongest condition-free scorer; +3.3 condition gain | reviewers will weigh the small R@1 margin |

**Timeline.** Abstract 10 November, paper 16 November, supplementary 23 November: about five weeks. Rounds 3 and 4
each took about 3.5 hours end to end on ArtELingo, so ArtELingo-only steps are cheap; a new dataset needs groupings,
heads, a condition-free B, practice banks and episodes, which is days to a week each before any result.

**Three framings and our view of each.**

| Framing | Claims | What it still needs | Our view |
|---|---|---|---|
| A. Method paper (plan branch 1) | C1, C2, C3; K2 on three datasets | CUB and SemArt for this method, K4, K3 | unlikely by the deadline; the mechanism may not transfer to visual-only aspects |
| B. ArtELingo-centred paper | C1 (task, protocol, benchmark release), C3 (analysis), AFF as a transparent label-free method with a held test on ArtELingo; CUB as an analysis dataset where the mechanism is expected not to apply | idea 3 (optional), the held test, the fine-tuned CLIP comparator, the analysis items of §5, writing | feasible; doubtful for the CVPR main track |
| C. Benchmark and analysis paper (plan branch 3) | C1, C3, K1, K3, with every method as a baseline | the release, the analysis, the baselines | lowest risk; a datasets-and-benchmarks track or a workshop |

*Sources: CVPR plan §4 (claims table, branches), §10, §11; `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`
Table 6 (MLLM probes); `docs/reports/auto/v2/2026-10-30_aspect_baselines.md` (E1); `2026-10-23_aspect_episode_spike.md`;
`docs/reports/literature/2026-10-21_cvpr_literature_review.md` (deadlines in its header); `docs/superpowers/held_ledger.md`.*

## 4. Question 3: what novelty and contributions do we have?

Ranked by how well they would survive review (our view).

1. **The task, benchmark and protocol (C1), the strongest.**
   - The literature review and the novelty check found no paper that defines example-conditioned, cross-item
     image–caption similarity with contrast pairs (to our knowledge), although every property exists separately
     (Contextual Visual Similarity, metric learning from pairs, MARS, GeneCIS).
   - The protocol is part of the contribution: value-disjoint cross-item examples, contrast pairs, both directions,
     the swap test, condition gain, and **matched controls**. Our own history shows why the last matters: a method
     passed a weaker control on seed 42 and lost 1.15 R@1 to its matched control (N1), and R@1 = (either rate +
     condition gain) / 2 lets condition-blind gains masquerade as conditioning.
   - The release (episodes, splits, label joins, evaluation code) makes it a benchmark.
2. **The analysis (C3), second.**
   - Aspects are carried unevenly by the two modalities: emotion by captions, style by images, across four backbones
     (weak-side probes move by about one point while strong-side probes move up to 11).
   - Value episodes can be solved without the query (K1: the query adds +0.57 R@1 over a prototype of the supports).
   - What label-free condition reading achieves: it reduces to choosing which side to steer, it pays only on the
     non-visual aspect, and it costs about half a point of either rate per point of gain.
   - The negative results (the factor method's NO-GO, the held-out-aspect failure, the reader fixes, the vetoes) are
     evidence that the task is hard for embedders, metric learning and in-context MLLMs alike.
3. **The method, modest.** Label-free pseudo-aspect groupings, a reader trained on practice episodes generated from
   those groupings (no evaluation labels), and a one-sided confidence gate fused with a condition-free score. Prior
   art narrows it: label-free condition discovery is taken (SCE-Net, DiscoverNet), and distant emotion supervision too
   (EmotionCLIP). It must be described plainly: a visual-contrast side detector, distantly supervised by a GoEmotions
   classifier whose categories name 6 of the 8 evaluation emotions.

**What reviewers will attack, and the answer we have:**

| Attack | Our answer now | Gap |
|---|---|---|
| "The gain is small" | +0.59 R@1 but +3.3 condition gain; the task is hard for every baseline including MLLMs | show the share of the oracle gap recovered (§5) |
| "It is GoEmotions in disguise" | the gain sits on emotion, where the grouping comes from GoEmotions | split emotion results by GoEmotions coverage (planned E12); a text-only GoEmotions reference |
| "It does not read the condition, it picks a side" | true, and we say so (random-share control) | frame the method as a side detector |
| "Only one dataset" | CUB and SemArt not done | framing decision (§6) |
| "A fine-tuned CLIP would do as well" | untested | §2's cheap comparator |

*Sources: `docs/reports/literature/2026-10-21_cvpr_literature_review.md` (verdict), `docs/reports/literature/2026-10-24_aspect_task_novelty_check.md`,
`2026-10-22_support_baseline_spike.md` (K1), `2026-10-25_backbone_check.md`, `2026-11-08_new_method_quick_checks.md` (N1);
CVPR plan §4 (C2's supervision statement); project memory note on the matched-control lesson.*

## 5. Question 4: how to analyse the results deeply and clearly

Part of this exists in rounds 3 and 4; the new items are marked **new**, with a rough cost (our estimate).

| Analysis | What it answers | Status, cost |
|---|---|---|
| **Decompose every R@1** into condition gain, other-aspect rate and either rate, with swap success | how a number moved, not only that it moved | exists; keep it in every table |
| **Per aspect pair × condition × direction** tables and heatmaps | where the gain lives | exists |
| **Per emotion value**, and split by GoEmotions coverage | which emotions gain; the "GoEmotions in disguise" attack | **new**; minutes from stored arrays |
| **A ladder of baselines and ceilings**: cosine → B → B′ → matched control → AFF → random-share control → told grouping → perfect emotion gate → label probes | how much of the reachable gap the method recovers (on seed 42, AFF's +0.700 is about 39% of the perfect-emotion-gate ceiling of +1.805, our arithmetic; that ceiling used the pure affect term, so the ratio is a rough guide) | mostly exists; **new** as one figure |
| **Mechanism**: gate open shares by side and pair, the visual-contrast diagnostic, redundancy with B | why it works where it works | exists (round 3 §8, round 4 §6) |
| **Flip analysis**: episodes that go from wrong to right and from right to wrong, and the target's rank change | what steering does to a ranking | **new**; minutes |
| **Number of example pairs** (1, 2, 4, 8) | how many examples the reader needs | **new**; needs episode rebuilds, hours |
| **Gallery of 101 candidates** instead of 13 | does it hold in a harder retrieval | planned for the supplement (E17); hours |
| **One corrupted example pair** | robustness to a wrong example | **new**; hours |
| **Second backbone; fine-tuned CLIP comparator** | is it CLIP-specific | §2, K4; hours to days |
| **Qualitative episodes** (6 to 8, images and captions; successes and the style × genre failure) | makes the mechanism visible | **new**; an hour |

**How to present it clearly** (our recommendation for the paper):

- One claim per figure, its point stated in the title; every number beside its baseline.
- A forest plot of every pre-registered comparison with intervals, as the paper's main result figure.
- A scatter of condition gain against either-rate cost for every method we tried, which shows the frontier the whole
  line moved along.
- A claim → evidence → caveat table, so that a reviewer finds each caveat (one dataset, development selection, the
  one-sided mechanism) next to the claim it limits.
- Consistent terms throughout; one glossary.

*Sources: round 3 report §7 to §9; round 4 report §6; CVPR plan §10, §11 (E12, E17); the brainstorm's ceilings
(`docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md` §2.3).*

## 6. Decisions for the user

1. **The paper's framing** (§3): A, B or C. **On hold** (the user, 7 October). Our view: target B, keep C as the
   fallback, and decide before the ArtELingo held read, because the framing decides which comparators and claims the
   held test must carry and whether CUB and SemArt work starts now.
2. **The fine-tuned CLIP comparator** (§2): **decided and done** (the user, 7 October): three lightweight variants
   (linear probe, last block, LoRA), no full fine-tuning. Outcome: about 2 points over plain CLIP, about 3.2 below
   B′(A0); a reported baseline, not a floor (§2).
3. **B′(A1) in the held test**: as a GO check, a reported comparator, or not at all (open since round 4).
4. **Which new analyses to fund** (§5): our short list is per-emotion with GoEmotions coverage, the ladder figure,
   the flip analysis and the qualitative episodes (all cheap), then the number of example pairs.
5. **Idea 3** continues in its own tab; its measured step decides whether a round is worth writing before the held
   test.

## Sources

- CVPR plan: `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` (§3 task, §4 claims and
  branches, §8 baselines, §9 scope, §10 held budget and statistics, §11 schedule).
- Rounds: `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md`, `2026-11-22_round4_aff_vetoes.md`; the brainstorm
  `2026-11-20_r1_levers_brainstorm.md`.
- Baselines and spikes: `2026-10-30_aspect_baselines.md` (E1), `2026-11-01_aspect_factor_gonogo.md` (E3),
  `2026-10-22_support_baseline_spike.md`, `2026-10-23_aspect_episode_spike.md`, `2026-10-25_backbone_check.md`,
  `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md` (MLLM probes, Table 6).
- Literature: `docs/reports/literature/2026-10-21_cvpr_literature_review.md`, `docs/reports/literature/2026-10-24_aspect_task_novelty_check.md`.
- Held budget: `docs/superpowers/held_ledger.md`.
