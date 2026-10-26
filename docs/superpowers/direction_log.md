# CoSiR direction log

One entry per loop of the research loop (`/project/claude-config/docs/research_loop.md`): what we read, what the user decided, why,
and where the files are. Newest first. The wrap-up step after reading results adds the next entry. Details live in the
linked reports and handoffs, not here.

New entries follow `~/.claude/templates/direction_log_entry.md`, which adds your reading of the results (decision C)
and who made each decision. Entries before 2026-10-08 were moved from project memory on 2026-10-08 and lack those two
fields; their verbatim notes are in `direction_log_raw_2026-10.md`. Older history (before the CVPR plan) is in the
weekly and stage reports.

The conditioned line's log (2026-09-28 to 2026-10-15, rounds 1 to 9, parked ideas) is archived: [direction log](../../archive/v2_conditioned/docs/superpowers/direction_log.md).

## Now (2026-10-26)

The new line builds a condition-free model over PercepT: one Qwen3-VL-Embedding instruction per aspect, buddy graphs
on each view, one shared trunk with emotion, style and genre heads, read once on the default image. The current recipe
(no own-caption terms, teacher share 1/5; report 12-24) passes the held emotion claim test on four image encoders,
each with Qwen graphs, teacher and captions at training (12,281 held ArtELingo paintings, 5 seeds; bar: PercepT core's
own reading on the Qwen image, 0.0874; a pass needs +0.0100 with the interval above zero): Qwen 0.1029, SigLIP 2
0.1024 (12-24), SigLIP 1 0.1036 and CLIP L/14 0.0996 (12-25, 2026-10-26). CLIP B/32 (0.0885) and CLIP B/16 (0.0945)
are inconclusive; DINOv2-large fails at sample scale. The claims are k-means only. On 29,232 development paintings
every one of the seven students lifts over its own PercepT arm (+0.0117 to +0.0288). A relational genre term at weight
8 reaches the instructed Qwen genre image's level on development data (genre AMI 0.5022 against the recipe's 0.4694 on
fresh seeds; picked among seven variants on one seed; no held test). The stage report of the whole new line (design,
ablations, analyses) is written (2026-10-26). Next (user, decide 23, 2026-10-26): close the genre gap first, in one full loop: a
sweep of the relational genre term (weights 16 and 32 on seed 0; relational alone and a sharper teacher temperature
at weight 8; teacher anchors instead of in-batch rows), weights 2 and 4 and the sweep's best on fresh seeds, then a
held genre claim test of the selected variant (decision rule before any code); the write-up of the encoder claims
comes after; aspect-named retrieval for the 11-10 gate still waits (decide 20). The grill and the spec run in the
r10 decide 24 chat. The CVPR abstract is due
2026-11-10.

## Parked ideas

- An aspect list proposed by an LLM from the captions (B3); no way yet to map it back to the three named aspects
  (deferred by the user (2026-10-18); [proposed in](handoffs/2026-10-18-r10-decide-6-handoff.md))
- PercepT under a fair protocol as the whole paper, candidate (c); the PercepT baseline stays (deferred by the user
  (2026-10-18); [proposed in](../reports/auto/percept/2026-12-06_new_line_candidates.md))
- Culture as the condition (ArtELingo-28: +0.052 Jensen-Shannon on 746 paintings) (never tried; deferred by the user
  (2026-10-18); [proposed in](handoffs/2026-10-18-r10-decide-6-handoff.md))
- Within-painting caption variation as a signal (probe 57.3% to 46.6% on the residual) (never tried; deferred by the
  user (2026-10-18); [proposed in](../reports/auto/percept/2026-12-06_new_line_candidates.md))
- A short stage 2 loop for style and genre: relational distillation alone (0.5, 2) and combined with direct at 0.5
  (never tried; not chosen by the user, 2026-10-20; [proposed in](../reports/auto/percept/2026-12-16_model_sheet.md);
  the genre half, relational alone, taken up in r10 decide 23, 2026-10-26)
- Edge-token probe: a linear probe per token ring of Qwen's hidden states, for the emotion view's border pull
  (never tried; parked by the user, 2026-10-25; [proposed in](../reports/auto/percept/2026-12-23_attention_maps.md))
- Aspect words as a caption signal: weight the caption terms by the emotion view's word attention; needs an
  extraction on the fit paintings; revisit after part 1 of decide 17's loop (never tried; parked by the user,
  2026-10-25; [proposed in](../reports/auto/percept/2026-12-23_attention_maps.md))
- Masking test: mask the most-attended image tokens against random ones, measure the embedding shift (never tried;
  parked by the user, 2026-10-25; [proposed in](../reports/auto/percept/2026-12-23_attention_maps.md))
- Class differences in the image share under the emotion instruction against each class's gain (never tried; parked
  by the user, 2026-10-25; [proposed in](../reports/auto/percept/2026-12-23_attention_maps.md))
- Align and contrast the three aspect heads of one image (shared part aligned, aspect parts pushed apart) (never tried;
  not chosen by the user, 2026-10-20: the aspects may not be separable; proposed in the r10 decide 12 chat)

## 2026-10-26 · Encoders and genre round read: CLIP L/14 and SigLIP 1 pass held emotion, CLIP B/16 inconclusive, DINOv2 fails; a relational genre term reaches the target on development data; a stage report of the new line first

- **Read:** stage i (12,281 held paintings, 5 seeds, full scale; bar: PercepT core's own reading on the Qwen image,
  0.0874; a pass needs +0.0100 with the interval above zero): CLIP L/14 0.0996, +0.0122 [+0.0068, +0.0136], GO at
  both scales; SigLIP 1 0.1036, +0.0162 [+0.0106, +0.0179], GO at both scales; CLIP B/16 0.0945, +0.0070 [+0.0016,
  +0.0085], inconclusive at both scales; DINOv2-large −0.0041 [−0.0094, −0.0030] at sample scale (0.0814 against
  0.0855), fails, no full-scale test. ArtELingo-28 (746): CLIP B/16 fails, the other three qualified. k-means only:
  the Leiden emotion intervals of CLIP L/14 and SigLIP 1 include zero. Every student needs Qwen3-VL-Embedding at
  training (graphs, teacher, captions). Stage ii (29,232 development paintings, sample scale, 5 seeds): all seven
  students lift over their own PercepT arm, +0.0117 (DINOv2) to +0.0288 (SigLIP 1), intervals above zero; the lift
  follows the encoder's style statistics (Spearman +0.93 purity, +0.96 silhouette; seven points) and not its frozen
  emotion AMI (+0.04); Qwen's lift is inflated by the recipe pick on the same paintings. Stage iii (Qwen, development,
  seed 0; target: the instructed Qwen genre image, 0.4989; recipe 0.4674): the relational term at weight 8 reaches
  genre 0.5018 with both guards met; fresh seeds 1 to 3: 0.5022 against the recipe's 0.4694, +0.0328 [+0.0235,
  +0.0442], emotion +0.0004, style −0.0010. A higher teacher weight lowers genre (x2 0.4594, x4 0.4478); the
  regression term leaves the head unchanged (0.4675; its adaptor reaches 0.4985). Picked among seven variants on one
  seed; only weight 8, the highest tried, on fresh seeds; no held genre test.
- **Meaning (C):** "I think this is partially good, CLIP L and SigLip1 did the job, and results seems positive and
  okey. However, the failed other several models (clip b 16&32, dinov2) possibly means our design is not really a
  plug&play module that fits any model backbones." (user, before ours). Ours: the encoder results fit our priors
  (CLIP L/14 and SigLIP 1 pass, CLIP B/16 inconclusive, DINOv2 fails); the genre result is the surprise (every
  relational weight gives at least +0.0261 against a prior of at most +0.0120), fairly sure on development data since
  the fresh seeds agree, less sure which weight is best. Where they differ: on held paintings CLIP B/16 and CLIP B/32
  are inconclusive (positive, below the +0.0100 size bar), not failed; DINOv2 is the one fail. On development
  paintings the design lifts every one of the seven encoders over PercepT on the same encoder, so the "plug and play"
  question splits: a lift over PercepT on the same encoder, yes for all seven (development data); a student above
  the strongest PercepT entry, PercepT core on the Qwen image (0.0874), only for the stronger encoders, whose own
  PercepT arms start near Qwen's (CLIP B/16 0.0728, CLIP B/32 0.0708, DINOv2 0.0722 against Qwen's 0.0844). The
  training side is not plug and play at all: every student needs Qwen's graphs, teacher and captions. The user's
  reading leaves out the genre result and the k-means-only and ArtELingo-28 limits.
- **Decided (A):** a stage report of the whole new line first (design, ablations, analyses; self-contained, with
  enough context for the user and for outside reviewers; the markdown stage report with its outside-view page and
  dev-log post): "a writing chapter for me to comprehensively understand the current progress ... a good stage report
  for me and also outside reviewers to check, so every detail must be clear and self-contained" (user). The next
  experiment loop (user, own pick, 2026-10-26 16:40, decide 23, after the stage report): option 2, then 1, "close
  the genre gap first". Options: 1, a held genre claim test of the
  weight-8 variant (a new rule, a held read); 2, more development work on the relational term (weights above 8, the
  other weights on fresh seeds); 3, pause genre and write the two new encoder claims. Our recommendation (in the
  report): 2, then 1. Aspect-named retrieval for the 11-10 gate still waits (decide 20).
- **Idea (B):** none this loop (a writing task; the stage report covers 2026-10-15 to 2026-10-26, the 2026-10-19
  stage report folded in as its first part: agent default, the user's "overview of the whole design").
- **Why:** the user wants an overview of the whole design, with its ablations and analyses, before the next
  direction; the abstract deadline (2026-11-10) leaves time for one or two more loops after it.
- **Links:** [report](../reports/auto/percept/2026-12-25_encoders_genre.md) ·
  [user-read report](../user_read/2026-10-26_encoders_genre.md) ·
  [spec](specs/2026-10-25-encoders-and-genre-design.md) ·
  [handoff](handoffs/2026-10-26-r10-decide-22-handoff.md)

## 2026-10-25 · Recipe and backbone round read: SigLIP 2 and Qwen pass held emotion, CLIP B/32 inconclusive; next step open

- **Read:** part 1 (29,232 development paintings, seed 0): no own-caption terms at teacher share 1/5 replaces D2,
  emotion +0.0061 [+0.0046, +0.0076], style +0.0240 [+0.0216, +0.0261], genre +0.0070 [+0.0039, +0.0102] (below one
  seed range, 0.0128); image endpoints only at 1/5 close (mean gain 1.55 against 1.60 seed ranges). Part 2 (12,281
  held paintings, 5 seeds, full scale; bar: PercepT core's Qwen arm, own reading, 0.0874): SigLIP 2 0.1024, +0.0150
  [+0.0096, +0.0163], GO at both scales; Qwen with the new recipe 0.1029, +0.0154 [+0.0100, +0.0169], GO at both
  scales; CLIP B/32 0.0885, +0.0011 [-0.0043, +0.0025], inconclusive at both scales. Each encoder's own PercepT arm:
  SigLIP 2 0.0793, CLIP B/32 0.0716. ArtELingo-28 (746): SigLIP 2 and both Qwen recipes qualified, CLIP B/32 fails.
  The claims are k-means only (full-scale Leiden emotion intervals include zero for SigLIP 2 and Qwen). The ceiling
  probe puts genre information in the default image (ridge genre AMI 0.5040 against the instructed image's 0.4989);
  every student stays below the instructed Qwen image on genre.
- **Meaning (C):** "I think for SigLIP2 and Qwen thats a good result, CLIP B/32 suprisingly beging bad." (user,
  before ours). Ours: the new recipe gains on emotion and style on one development seed, genre less; SigLIP 2 meets
  the framing note's "any backbone" condition; this supports the narrow claim "the emotion lead over PercepT core holds
  on two image encoders, Qwen and SigLIP 2, with Qwen3-VL-Embedding at training"; write it for the abstract, then test
  genre losses. Where they differ: on held paintings CLIP B/32 is inconclusive, not bad, and fails only on
  ArtELingo-28; the student lifts each encoder above its own PercepT arm by a similar amount (SigLIP 2 +0.0231, CLIP
  B/32 +0.0169, Qwen +0.0155; arithmetic from the report's tables, not checked) and CLIP's own PercepT arm starts
  lowest (0.0716); correction (decide 20 facts brief, read from the outputs): CLIP B/32's frozen emotion AMI (0.0642)
  is above SigLIP 2's (0.0569), whose student ends near the top, so the starting level alone does not explain it;
  we put the abstract claim first, the user's next steps leave it out; the user's reading leaves out the k-means-only,
  ArtELingo-28 and one-seed limits.
- **Decided (A):** the user's three proposals as one medium loop, shaped by our recommendation (user, after seeing
  the recommendation). Stage i, more backbones (claim test, held): CLIP B/16, CLIP L/14 at 224 px, SigLIP 1 So400m/14
  at 384 px and DINOv2-large under 12-24's rule with only the encoder list changed (models: user, after seeing the
  recommendation). Stage ii, the backbone analysis (confirmation, existing outputs plus stage i's): per encoder its
  frozen AMI, its own PercepT arm, the student's lift and the four-view statistics; does the lift follow the encoder's
  starting level; a method idea from it is tested on development paintings in a later loop. Stage iii, teacher-faithful
  heads or losses for genre (incremental, development, Qwen, on the new recipe; literature note 12-20, ideas 2 and 3).
  Medium loop: step 3 skipped, the options are clear and stage i reuses a reviewed rule (agent default). The abstract:
  not now (user). Aspect-named retrieval for the 11-10 gate: the next loop, "the retrieval is important" (user).
- **Idea (B):** the three stages above; grill in decide 20, 2026-10-25 16:40 to 17:25 (record in the
  [decide 21 handoff](handoffs/2026-10-25-r10-decide-21-handoff.md) §2). Stage ii looks at development paintings
  first: the sample-scale students and PercepT arms of all seven encoders scored on the 29,232, then the held table,
  which chooses nothing; a method idea from it goes to development first, to held only as a new claim test in a later
  loop (user; the what-if confirmed). Stage iii: the target is the instructed genre value 0.4989 ("2c would be the
  target, but we might not be able to reach it directly from this round") (user); a variant never wins if another
  aspect loses more than one seed range (user, after seeing the recommendation); weights of every variant within one
  genre seed range of the target are kept, and the closest method gets two more weights (user, after seeing the
  recommendation). Stage i: one verdict per encoder, the rule unchanged except the encoder list (user); how the paper
  reports the verdicts is open, "we might exclude models that cannot pass, too soon to determine the paper writing
  now" (user). Minor points m1 to m6 accepted (user, after seeing the recommendation); no plan page (user). Closing
  summary confirmed (user, 17:25).
- **Why:** more encoders test whether the student's lift is general and whether CLIP B/32's result comes from its low
  start; the genre gap to the instructed Qwen image is the open problem; all three run in minutes per training run.
- **Links:** [report](../reports/auto/percept/2026-12-24_recipe_backbones.md) ·
  [user-read report](../user_read/2026-10-25_recipe_backbones.md) ·
  [spec](specs/2026-10-25-recipe-and-backbones-design.md) ·
  [handoff](handoffs/2026-10-25-r10-decide-20-handoff.md) · next loop's
  [spec](specs/2026-10-25-encoders-and-genre-design.md) (approved 2026-10-25 18:35)

## 2026-10-21 · D2 ablation read: the instructed graphs carry the gain, the teacher and the trunk are close to redundant; next step open

- **Read:** image AMI on 5,000 development paintings, 3 seeds; effect = variant minus D2 (0.0962 / 0.3697 / 0.4540,
  emotion / style / genre). One default-view graph −0.0222 / −0.1681 / −0.0287 and no buddy terms −0.0214 / −0.1473 /
  −0.0449 (clear on all three); no teacher −0.0032 / −0.0106 / +0.0063 (not resolved / small / not resolved);
  default-view teacher and separate heads not resolved on all three; image endpoints only +0.0057 / +0.0248 / −0.0086
  (small / clear / not resolved; style 0.3945 against the instructed Qwen image's 0.3952); no own-caption terms
  +0.0055 / +0.0180 / +0.0063 (small / clear / not resolved). No instruction anywhere 0.0733 / 0.1848 / 0.4241, below
  PercepT core (0.0772 / 0.2286 / 0.4359). Instructed inputs leave genre at 0.4545, below their own input (0.4956).
  Without the caption terms the teacher's loss share rises from 1/5 to 1/3; the ablation does not separate the two.
- **Meaning (C):** "(1) Seems we have a nice ablation and find some key points and also some design redundency. (2)
  Even this is only on development data, this is already at certain level of clearance to guide the next-step design.
  (3) seems our model depends very much of the instructions? (4) the difference across four qwen features is within my
  expectation, instructing towards emotion will hurt style and genre." (user, before ours). Ours: the instructed
  graphs are the core and the instruction acts through them, not through the teacher; the teacher and the trunk are
  close to redundant; the caption-free gains are a lead; genre is the open problem, and its input is not the limit; an
  instruction changes clustering far more than probe accuracy. Where they differ: the user takes the development
  result as clear enough to guide the design; we agree for the large effects but would check the caption-free gains on
  other paintings first (picked on these paintings, confounded with the teacher share); the user's reading leaves out
  genre. The user's follow-ups: "without caption terms" keeps captions in the graphs (the emotion graph is built from
  instructed captions); why the own-caption term does not help is open (the teacher share, or its instance-level
  contrast on content; not checked); genre is included in the caption-free check (user asked; all three aspects are
  scored); a statistics sheet of the four views (norms, anisotropy, default-to-instructed cosine, per-label neighbour
  purity, silhouette, hubness) goes beside the next run (user, after seeing the recommendation).
- **Amendment A1:** confirmed (user, after seeing the recommendation): the no-content-correction row uses the plain
  instructed style image graph (R0), not the style caption graph (R0o).
- **Side task (user):** Qwen's attention maps under the four prompts, 5,000 development paintings × 3 input modalities,
  an HTML viewer and a summary; side chat "r10 attention maps"
  ([handoff](handoffs/2026-10-21-r10-attention-maps-handoff.md)). Done 2026-10-25
  ([report 12-23](../reports/auto/percept/2026-12-23_attention_maps.md)): the instructions move the image map (L1 0.68
  to 0.81 from the default, floor 0.09) and halve the image's share of the pooled token's attention (0.22 to 0.12);
  the emotion instruction moves caption attention to emotion words. Its decision C: "very interesting results,
  provide some potential basis for deeper analysis." (user, before ours); ours: the instruction effects are solid, raw
  attention is a weak picture of where the model looks. Four candidate ideas handed to this loop's decision A (edge
  tokens probe, aspect words as a caption signal, masking test, class differences; report section "Candidate ideas
  for decide 17").
- **Decided (A):** two parts in one loop (user, after seeing the recommendation). Part 1, on development paintings:
  the caption-free variants (image endpoints only, no own-caption terms) on development paintings that did not choose
  D2, beside D2 with the teacher at 1/3, all three aspects; the genre information-ceiling probe (literature note 12-20,
  idea 1); the four-view statistics sheet. Part 2: code for item 5 (other backbones and ArtELingo-28) with the
  training recipe as a setting; its runs follow part 1's recipe. Type confirmation and exploration (part 1),
  incremental (part 2); medium loop. Retrieval for the 11-10 gate: image queries with an aspect name only (user,
  after seeing the recommendation). The side task's four ideas are parked (user, after seeing the recommendation).
- **Idea (B):** grill (decides 17 and 18, 2026-10-25). Part 1 scores six recipes (D2, image endpoints only, no
  own-caption terms; teacher at 1/5 and 1/3; one seed) on the 29,232 train-side paintings outside the old sample
  (user). A fixed guarded rule picks part 2's recipe: at least "small" on one aspect, never more than one 12-22 seed
  range below D2, the largest mean gain wins, else D2 (user). Students on SigLIP 2, then CLIP B/32; DINOv2 only if
  image endpoints only wins (user, after seeing the recommendation); Qwen captions, graphs and teacher kept (user,
  after seeing the recommendation). Bar: the strongest PercepT core arm per aspect, a matched arm on the backbone's
  image included; the frozen backbone image reported (user, after seeing the recommendation). Held, trained on the
  13,400 first, then the full split unless NO-GO (user; the NO-GO rule after seeing the recommendation).
  ArtELingo-28 as a test set and a descriptive culture split (user). The brief is approved through the closing
  summary (user, 2026-10-25 03:02).
- **Why:** part 1 is cheap and settles the recipe that item 5 runs; building both together saves a loop before 11-10.
- **Links:** [report](../reports/auto/percept/2026-12-22_d2_ablation.md) ·
  [user-read report](../user_read/2026-10-21_d2_ablation.md) · [spec](specs/2026-10-21-d2-ablation-design.md) ·
  [handoff](handoffs/2026-10-21-r10-decide-17-handoff.md) ·
  [grill handoff](handoffs/2026-10-25-r10-decide-19-handoff.md)

## 2026-10-21 · Claim test read: D2 passes both stages (S1-GO, S2-GO); an ablation and analysis of the design before more datasets and backbones

- **Read:** emotion AMI on 12,281 held paintings, k-means k = 20, 5 seeds, against the strongest uncollapsed PercepT
  (core, its own top topic). Stage 1 (13,400 fit paintings): D2 0.1001 against 0.0855, d +0.0146 [+0.0093, +0.0157],
  S1-GO. Stage 2 (42,969 train paintings): D2 0.1008 against 0.0874, d +0.0133 [+0.0081, +0.0146], S2-GO; the
  independent check agrees exactly. All strict paper settings runs collapsed in both stages; the K = 40 versions score
  0.064 to 0.071. Diagnostics: Leiden margin −0.0005 [−0.0048, +0.0023] (stage 1 +0.0019); k-means at Leiden's count
  keeps the lead (+0.0190); style +0.139 and genre +0.021 over core; D2 below the instructed Qwen image on style and
  genre (0.369 / 0.444 against 0.403 / 0.506). From stage 1 to stage 2 (3.2× training data): emotion D2 +0.7%, core
  +2.3%, a difference inside the seed noise (about ±0.001 per change; not tested formally); genre D2 −1.2%, core −4.1%.
- **Meaning (C):** "This seems a good overall result. (1) D2 cleared uncollapsed PercepT, that's a good starting point.
  But need to establish to more backbones and datasets. (2) Leiden community seems not really helping? (3) paper
  settings PercepT seems worse than our PercepT core setting which fixed many things." (user, stage 2, before ours;
  that it covers stage 1 too: agent default, decide 15 got no answer). Ours: a pass with a smaller lead than stage 1;
  no lead under Leiden; style and genre below the instructed target. Where they differ: the user adds backbones; the
  user reads Leiden as a weaker readout, we read it as a dependence on the clustering method (Leiden lowers D2 0.1008
  to 0.0757, core only 0.0799 to 0.0762); the user's reading leaves out the style and genre gap.
- **Paper sentence (S2-GO, agent; no paper folder for the new line yet):** "On 12,281 held ArtELingo paintings, a model
  trained on the 42,969 train paintings and read from the default Qwen image gives emotion clusters (k-means, 20
  clusters) with AMI 0.101, 0.013 above the strongest uncollapsed PercepT reimplementation trained on the same
  paintings (0.087; paired 95% interval of the difference [0.008, 0.015], over paintings and 5 seeds), each with its
  own recipe on our feature stores." It covers no style or genre, other dataset or backbone, the instructed Qwen bar,
  retrieval, or PercepT with its own encoders (spec §6).
- **Decided (A):** five parts, in order (user): (1) a quick ablation of D2's building parts on the cluster, 3 seeds;
  (2) a deep analysis of it, and of the four Qwen image embeddings (default and three instructed): how they differ
  and correlate; (3) a list of ways to improve, with deep research if needed, the three cluster-level candidates
  among them; (4) our advantages and how to show them (any dataset through Qwen prompting, any backbone, to prove,
  lightweight, not sure); (5) code for other backbones and ArtELingo-28. Parts 3 to 5 go on locally while part 1
  runs. Type ablation and analysis, medium loop, no bar (user). The rule's S2-GO step (ArtELingo-28's facts brief and
  spec) becomes part 5, with backbones added (user).
- **Idea (B):** track 2's cluster-level loss waits for part 3's list (user: "the three contrastive ways could be
  within the list").
- **Why:** "we did not really understand our current design" (user).
- **Links:** [stage 1 report](../reports/auto/percept/2026-12-18_claim_test_stage1.md) ·
  [stage 2 report](../reports/auto/percept/2026-12-19_claim_test_stage2.md) ·
  [spec](specs/2026-10-20-claim-test-design.md) · [ablation spec](specs/2026-10-21-d2-ablation-design.md) · [build handoff](handoffs/2026-10-21-r10-build-14-handoff.md)

## 2026-10-20 · Stage 2 model sheet read: D2 above the instructed target on emotion only; claim test with D2 next, cluster-level losses as a second track

- **Read:** 13 settings, development data (5,000 evaluation paintings, k = 20). Best per aspect, 3 seeds: emotion D2
  0.0962 (target 0.0929, PercepT core 0.0772), style D5 0.3700 (target 0.3952), genre D5 0.4891 (target 0.4956). D2
  (trunk + direct distillation 0.5): 0.0962 / 0.3697 / 0.4540. Trunk T1 0.0930 / 0.3591 / 0.4603 against separate
  heads A1 0.0894 / 0.3539 / 0.4574. Distillation only, D1 (1 seed): 0.0825 / 0.2751 / 0.4366. Outcomes: (a) no, (b)
  yes on emotion only, (c) no, (d) no; the emotion line 0.0872 is reached by nine default-input rows. Probe ceiling
  0.121 / 0.527 / 0.657.
- **Meaning (C):** "A partly positive result. D2 is the first default-input row above the instructed-image target on
  emotion. Style and genre stay below the target, but all three stay above PercepT core, so the 10-29 line holds. The
  gap to the supervised probe (about 20 / 30 / 25%) shows room to improve. The design is open to challenge: the
  clustering step (Leiden, which finds its own count, against k-means) and cluster-level training losses. One cluster
  set for many aspects does not scale." (user, before ours). Ours: outcome (b); direct distillation helps emotion and
  style, relational helps style and genre, no row combines them; the graphs carry most of the result; the trunk costs
  nothing; with instructed inputs (I1) style and genre still miss, so training limits them. Where they differ: the
  user measures the gap against the probe, we against the target; the user's best values mix D2 and D5.
- **Found in the discussion** (agent, shown to the user): k-means at k = 20 entered the new line as an agent default
  (premise checks 2026-10-17, then the aspect graph spec §3.4) and was never a listed decision; September used Leiden.
  Under random clusterings AMI and ARI stay at 0 for any count, V and FMI do not (simulation). PercepT states one
  training set of about 1.2M pairs and no test split; our AMI is on unseen paintings. A Codex comparison of our DEC
  uses with the paper (spot-checked; `.scratch/r10-cluster-losses/dec_comparison.md`): no September run started DEC
  from a good partition, and PercepT core differs from the paper beyond the backbone (autoencoder one layer short, no
  denoising, a balance term of weight 1000, lr 1e-4 full batch, K = 40, our scoring).
- **Decided (A):** two tracks (user, after seeing the recommendation). Track 1: the claim test spec next; type claim
  test, full loop. Track 2: step 3 for cluster-level losses; type new idea, full loop. It covers (1) cluster-level
  contrast per head and (2) teacher clusters as anchors (user, after seeing the recommendation), and (3) DEC started
  from the good partition of our heads (user). The short style and genre loop (option 1) is not chosen.
- **Idea (B), claim test:** row D2 (user, after seeing the recommendation). A Leiden reading in all major tests beside
  k-means (user). PercepT scored both with its own assignment (top topic of 40) and at matched k = 20; the gate is
  judged against the stronger reading per aspect (user, after seeing the recommendation). A "paper settings" PercepT,
  where only the backbone differs from the paper, runs beside PercepT core, with a collapse rule (user, after seeing
  the recommendation; "it can collapse easily, but I tend to give a try"). The grill and the spec follow in the next
  decide chat.
- **Grill, claim test (r10 decide 13, closing summary confirmed by the user 2026-10-20 01:20):** two stages, both
  claim tests. Stage 1 (the 10-29 gate): D2, PercepT core and the two paper settings PercepT versions (strict and
  K = 40; arms P1, P4) retrain on the developed sample (13,400 fit paintings), 5 seeds, weights saved, read on the
  12,281 held paintings (user: "develop on same samples now, but be careful of the result claim"). Gate: d = D2
  emotion AMI minus the strongest uncollapsed PercepT at its stronger reading, k-means k = 20, seed means; GO: d ≥
  0.01 and the paired 95% interval above 0 (user, 6b); "inconclusive, margin met" (d ≥ 0.01, interval reaches 0):
  improve the margin on development data, test again (user); inconclusive: 0 < d < 0.01; NO-GO: d ≤ 0, we talk.
  Paper settings PercepT enters the bar only when not collapsed; both versions run (user, 5c; its arms and collapse
  thresholds: agent proposal in the option). Leiden (union 20-NN of the held outputs, resolution 1.0, 3 seeds; then
  September's vote onto training communities) is reported beside the verdict and decides nothing (user, 3a then 3b;
  7a after the [development Leiden reading](../../src/test/20261020_leiden_dev_reading/20261020_leiden_dev_reading_log.md):
  emotion lead +0.019 under k-means, +0.009 under Leiden). Stage 2: the full ArtELingo train side (English), its own
  rule, run only after a GO or inconclusive stage 1 (user, 4b and 8b); its result counts over stage 1's, and an
  uncollapsed paper settings PercepT that beats D2 there becomes the comparator to improve against (user, S1).
  ArtELingo-28 (translated) gets its own facts brief and spec after the gate (user, 4b). Minors m1 to m10 (5 seeds,
  the sheet's label rule, a stage 0 regression, held instructed images, arm per aspect fixed from development data,
  diagnostics, validation split for stage 2's early stopping, names, local GPU) accepted (user, after seeing the
  recommendation). The spec and its decision rule follow in r10 decide 14.
- **Spec and rule (r10 decide 14, 2026-10-20 01:45):** [claim test spec](specs/2026-10-20-claim-test-design.md)
  and [decision rule](../../src/test/20261020_claim_test/DECISION_RULE.md), reviewed by a fresh Opus `xhigh` reviewer.
  Approval: the user's instruction to go on overnight (01:27, "automate and hand me the results in the morning"); the
  user checks the brief in the morning (spec D29). Night mode on; node408 offered for PercepT training (spec D23).
- **Why:** emotion clears the line on development data (+0.019 over PercepT core), and the 10-29 gate needs held
  data. The user challenges the clustering step; Leiden and cluster-level losses test it.
- **Links:** [auto report](../reports/auto/percept/2026-12-16_model_sheet.md) ·
  [user-read](../user_read/2026-10-19_model_sheet.md) ·
  [stage 2 spec](specs/2026-10-19-stage2-model-sheet-design.md) ·
  [handoff](handoffs/2026-10-20-r10-decide-13-handoff.md)

## 2026-10-19 · PercepT core with K = 40 read: all five arms established; two bars per aspect (option b); stage 2 next

- **Read:** with a fixed K = 40, all five arms are established on 3 seeds (seed sd at most 0.007, dev AUC 0.85 to
  0.96); Stage 1 training bitwise equal to run 2. Image-side AMI, emotion / style / genre: bar 1, the best PercepT core
  arm, P3 0.077, P2 0.229, P3 0.436; bar 2, the image encoder clustered directly, Qwen 0.070, CLIP 0.300, Qwen 0.449;
  our best default-input network 0.081 / 0.348 / 0.458. Ours minus bar 1: +0.003, +0.120, +0.022; minus bar 2:
  +0.011, +0.048, +0.0097. Development data only.
- **Meaning (C):** "I think this is a good comparator" (user, before ours). Ours: the fixed count removes the seed
  lottery; PercepT core's topics add little on the image side (level with or below the encoder it reads); emotion is
  open. Found after the two-bar table (agent, shown to the user): against bar 2, genre is level too, so our network
  is clearly ahead of both bars only on style.
- **Decided (A):** comparator option b: per aspect, PercepT core with K = 40 (bar 1) and the image encoder clustered
  directly (bar 2) (user). Extra metrics: V, ARI and FMI beside AMI, in stage 2's shared scoring code, then filled
  into the two-bar table once; H and C dropped; SI, CHI and DBI set aside for now, "without DEC I don't think we gain
  a lot on these" (user, after seeing the recommendation: ours was ARI, H, C and V, without FMI). Decisions stay on
  AMI (agent default, shown to the user). Then an image-side ceiling check before stage 2 (option b; user, after
  seeing the recommendation): a linear probe with labels reaches 0.121 / 0.527 / 0.657 on default Qwen images
  (instructed: 0.130 / 0.569 / 0.658); our network closes 21%, 41% and 4% of the gap from raw images
  ([log](../../src/test/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md)). Next: stage 2, the model sheet of
  the aspect graph spec; type exploration, medium loop (user, after seeing the recommendation).
- **Idea (B):** B2 unchanged. Grill of stage 2 (four key questions, answered before ours unless noted):
  - Emotion at 2026-10-29: at least +0.01 over PercepT core, and clearly above the other baselines (user).
  - Inputs: default input carries the claim; the instructed input is read as a second result (user).
  - Rows are picked on image-side AMI now; retrieval's pick waits until stage 3 (user).
  - Target of stage 2: reach the instructed Qwen image clustered directly (0.093 / 0.395 / 0.496) with one pass on
    the default image; may take one or two loops (user). The 10-29 gate stays as above and the instructed bar is
    judged with retrieval on 2026-11-10 (user, after seeing the recommendation).
  - Minor m1 to m6 accepted as recommended (four bars per table incl. the probe ceiling as reference; V, ARI, FMI
    reported; instructed-view distillation and image-side picks as new rows; CLIP input rows dropped; one seed, three
    for the top rows; build after the restart) (user, after seeing the recommendation). No plan page (user).
- **Why:** a plain encoder matches or beats PercepT core per aspect, so bar 1 alone is weak. The reviewer's first
  attack is "just prompt Qwen": today the instructed image beats our network on all three aspects.
- **Links:** [run report](../reports/auto/percept/2026-12-15_percept_core_k40.md) ·
  [user-read with the two-bar table](../user_read/2026-10-19_percept_core_k40.md) ·
  [spec (A3)](specs/2026-10-18-percept-core-design.md) ·
  [handoff](handoffs/2026-10-19-r10-decide-10-handoff.md) ·
  [stage 2 spec](specs/2026-10-19-stage2-model-sheet-design.md) (approved by the user 2026-10-19 19:58, no changes)

## 2026-10-19 · PercepT core arms read: the topic count follows the seed; K fixed at 40 (A3), the five arms rerun

- **Read:** stage 0 (amended) passed; a repeated Stage 1 run was bitwise equal. Algorithm 1 (γ = 2) kept 1, 2, 3 or
  99 topics by seed: the largest norm gap sat at an end in 9 of 9 Stage 1 runs. Only P2 is established, image-side
  AMI 0.036 / 0.162 / 0.217 against our best network 0.081 / 0.348 / 0.458. With 99 topics: P3 (2 seeds) 0.072 /
  0.194 / 0.446 against Qwen default images 0.070 / 0.222 / 0.449; P2 seed 0 0.063 / 0.250 / 0.412 against CLIP
  images 0.065 / 0.300 / 0.412. P5 kept 1 topic on every seed.
- **Meaning (C):** first "we found a valid PercepT comparator" (user); after our explanation of the count, "this
  comparator still seems a bit unstable" (user, after seeing ours). Ours: P2 is valid under the checks but weak, since
  two of its seeds kept 2 and 3 topics; with 99 topics PercepT core adds nothing over its image encoder clustered
  directly.
- **Decided (A):** option b: fix the topic count at K = 40 (A3) and rerun the five arms; type baseline, medium loop,
  as an amendment of the r10 spec, no grill (the user waived it) (user, after seeing the recommendation). Which 40:
  the 40 of 100 centers with the most fit pairs (agent default, shown to the user). The stage 2 decision moves to the
  next decide chat, after the rerun (agent default).
- **Idea (B):** unchanged (B2).
- **Why:** γ alone cannot help (the end gap is 4 to 12 times the mean gap). Norm and pair share correlate positively
  (Spearman 0.40 to 0.44 in the three 99-topic runs), so a norm cut keeps little-used topics. September's
  faithful-recipe run without the balance term also kept 99 of 100 under the gap rule, so faithful PercepT alone is not
  expected to fix the count (not checked on today's setup).
- **Links:** [run report](../reports/auto/percept/2026-12-14_percept_core_arms.md) ·
  [user-read report](../user_read/2026-10-19_percept_core_arms.md) ·
  [amended spec (A3)](specs/2026-10-18-percept-core-design.md) ·
  [handoff](handoffs/2026-10-19-r10-build-6-handoff.md)

## 2026-10-19 · PercepT core run read: stage 0 fails on September noise, rows 1, 6, 10, 13 filled; stage 0 amended, PercepT core runs next

- **Read:** stage 0 failed (September held-out emotion / genre AMI 0.1244 and 0.1287 / 0.2617 and 0.2804, recorded
  0.1225 / 0.2274), so the five PercepT arms did not run. Cause: the September k-means start sums its centers in a
  thread order that changes per run (eight local runs, three outcomes; one thread repeats). Rows (target AMI, 3
  seeds): row 6, network on the default graph, 0.082 / 0.163 / 0.388, below row 2 (default embedding, 0.089 / 0.171
  / 0.393); rows 10 and 13, the other side's graph, emotion 0.086 / 0.089, style 0.154 / 0.156, genre 0.428 / 0.434;
  row 1 (CLIP + RoBERTa, k-means) 0.127 / 0.174 / 0.240, RoBERTa captions emotion 0.189, CLIP images style 0.300.
- **Meaning (C):** no reading given; the user chose option a and to run PercepT core (user). Ours: stage 0 failed on
  noise in the September fit, not on our code; our network gains nothing without the instruction in the graph; the
  September encoders are strong per side, so the PercepT core comparator could be strong on emotion.
- **Decided (A):** unchanged direction: PercepT core (type baseline, medium loop), then stage 2. Stage 0 amended to
  the checks that repeat exactly (A1, option a; user), and one-thread k-means in PercepT core's Stage 1 (A2; user,
  after seeing the recommendation). No container restart (user).
- **Idea (B):** unchanged (B2).
- **Why:** a 4-decimal re-fit cannot pass on any machine we checked; the exact checks still catch a silent change in
  reused code (C12), and one thread makes PercepT core's seeds repeat.
- **Links:** [run report](../reports/auto/percept/2026-12-13_percept_core.md) ·
  [user-read report](../user_read/2026-10-19_percept_core.md) ·
  [amended spec](specs/2026-10-18-percept-core-design.md) ·
  [load check](../../src/test/20261019_stage0_load_check/20261019_stage0_load_check_log.md) ·
  [handoff](handoffs/2026-10-19-r10-decide-8-handoff.md)

## 2026-10-18 · Aspect graph follow-up read; the stage 1 order holds on three seeds; PercepT core next, with an ablation table to fill

- **Read:** 31 CPU head runs. Target AMI (mean of both sides, k = 20, 5,000 paintings, 3 seeds): residual graphs stay
  ahead of the plain aspect graph (emotion 0.137 ± 0.002 vs 0.118 ± 0.002, style 0.260 ± 0.003 vs 0.213 ± 0.005);
  genre image graph alone 0.428 ± 0.003 vs the image-weighted mix 0.420 and the caption graph 0.402. Prompted inputs
  raise the residual heads (emotion 0.151, style 0.284) to within 0.003 of the prompted view on its stronger side,
  not above it. Ablation: the best default-input head vs the default view +0.049, +0.089, +0.035; vs the prompted
  view −0.018, −0.005, +0.003 (emotion, style, genre).
- **Meaning (C):** "I think this is a good sign, so our best graph steadily beat plain aspect graph" (user, before
  ours). Ours: the same, plus the gap on the prompted view's stronger side is mostly the input; development data
  only, and stage 1 picked these graphs on the same paintings. After the ablation table (user): "what we applied
  parts are useful, but the biggest gain is still test time instruction ... a semi-good sign"; unsure whether the
  test-time instruction can carry a contribution claim; verify later, or improve our designs for better numbers.
- **Decided (A):** unchanged: PercepT core in two versions next (type baseline, medium loop), then stage 2. First the
  user asked for an ablation table in the user-read report (one part changed at a time, placeholders for missing
  numbers), then the grill (user).
- **Idea (B):** unchanged (B2).
- **Why:** the table shows which part of the design carries the gain, and PercepT core fills its open rows.
- **Links:** [follow-up report](../reports/auto/percept/2026-12-12_aspect_graph_followup.md) ·
  [user-read with the ablation table](../user_read/2026-10-18_aspect_graph_followup.md) ·
  [run log](../../src/test/20261018_aspect_graph_sheet/20261018_aspect_graph_sheet_log.md)

## 2026-10-18 · Aspect graph sheet (stages 0 and 1) read; a stage 1 follow-up, then PercepT core in two versions, then stage 2

- **Read:** stage 0 matched the pilot exactly. 21 graph rows per aspect, 28 head runs on 16 of them. Own-label lift
  is best for the plain aspect graph (emotion 2.42, style 7.07); for genre, image-weighted side fusion 5.35 vs the
  caption graph 4.41 (image graph alone also 5.35). Head target AMI: residual graphs best for emotion (0.137 ± 0.002
  vs plain 0.120, one seed) and style (0.260 ± 0.003 vs 0.218); genre image-weighted 0.420 ± 0.002 vs 0.402 ± 0.005.
  The prompted view clustered directly (mean of both sides): emotion 0.155, style 0.265, genre 0.424. Heads pass it
  on style and genre captions (0.172 vs 0.135, 0.386 vs 0.353).
- **Meaning (C):** "the exploration results are interesting and show a lot of potential, so we first do (b) to make
  the pilot more robust then go (c) then (a)" (user, before ours). Ours: a real signal (the graph choice matters; a
  cross-modal gain on style and genre), more cautious about the gap to the prompted view.
- **Decided (A):** (b) a stage 1 follow-up: the missing seeds of the plain and purified graph heads, a head on the
  image-only genre graph, a head that reads the prompted views; type exploration, small loop, local CPU. Then (c)
  PercepT core in both versions, CLIP B/32 + RoBERTa-GoEmotions and the Qwen default embeddings (user, after seeing
  the recommendation "both"); type baseline, medium loop. Then decide (a), stage 2 (user). Local runs: DAS6 has no
  approved reservation (requests on node408 and node406 pending at 18:50). Storage of this run: decided at the end of
  the loop (user).
- **Idea (B):** unchanged (B2).
- **Why:** the residual graphs' gain rests on one seed for the plain-graph references, and no head reaches the
  prompted view yet; PercepT core sets what the trunk must beat.
- **Links:** [aspect graph sheet](../reports/auto/percept/2026-12-11_aspect_graph_sheet.md) ·
  [user-read](../user_read/2026-10-18_aspect_graph_sheet.md) ·
  [spec (draft)](specs/2026-10-18-aspect-graph-sweep-design.md) ·
  [follow-up handoff](handoffs/2026-10-18-r10-followup-1-handoff.md)

## 2026-10-18 · Premise checks and three pilots read; named aspects first (B2); an exploration sweep next

- **Read:** prompted aspects (Qwen3-VL-Embedding-2B, train side): caption emotion AMI 0.278 with the emotion
  instruction vs 0.122 default and 0.209 GoEmotions; image style 0.395 vs 0.222 default and 0.317 CSD; image emotion
  0.069 to 0.093. Masking 132 affect words keeps most of the caption lift (0.187 vs 0.069). Image-only students fail
  their bar (0.12): own-caption target 0.074 vs control 0.074; purified buddy pairs 0.094 vs control 0.070
  (purification +0.004). Pairs shared with the content graph are the most emotion-consistent (lift 2.55; with CLIP
  content 2.61 vs 2.42).
- **Meaning (C):** night pilots: "there are aspects that we should not expect an image-only student model can learn"
  (user). Purification pilot: "something but not conclusive; the teacher-student net failed on cross-modal emotion
  transfer; the Stage 1 finding matters: purification lowers content-likeness but does not raise emotion purity;
  maybe the views of one model share a lot" (user). Ours agreed and added that content and emotion co-occur in art.
- **Decided (A):** still a draft (the 2026-10-15 entry). The user's guidance on 2026-10-18: "we are still mostly
  doing exploration jobs … get a basic number sheet to help us decide later" and "firstly just go for testing
  different ideas" (user). Type exploration sweep, loop size to confirm (agent default).
- **Idea (B):** B2: named aspects now (emotion, style, genre heads on one shared trunk; each aspect space built on
  the side where it is visible; cross-modal mappers); learned aspects (b) after aspect-named retrieval beats prompted
  Qwen (user, after seeing the recommendation). The grill's answers, with who decided each: [decide 6 handoff](handoffs/2026-10-18-r10-decide-6-handoff.md), section 2.
- **Why:** the premise checks show one instruction per aspect works on the side where the aspect is visible; image-only
  emotion has a low ceiling in this data, so the design must not rest on it.
- **Links:** [prompted aspects](../reports/auto/percept/2026-12-08_prompted_aspect_views_check.md) ·
  [night pilots](../reports/auto/percept/2026-12-09_image_transfer_and_word_masking.md) ·
  [purification pilot](../reports/auto/percept/2026-12-10_purified_buddy_pilot.md) ·
  [literature](../reports/literature/2026-12-07_prompted_aspect_views.md) ·
  [decide 6 handoff](handoffs/2026-10-18-r10-decide-6-handoff.md)

## 2026-10-15 · Round 9 read; the supervisors stop the conditioned line, archive it and turn to a condition-free model

- **Read:** r9 (fixed AFF recipe): GO on Qwen3-VL-Embedding-2B, AFF − B′(A0) +0.63 [+0.49, +0.76] R@1, and SigLIP 2,
  +0.51 [+0.36, +0.66] (CLIP in round 6: +0.60). ArtELingo-28: full claim only on Qwen (+0.37 [+0.23, +0.51]); CLIP and
  SigLIP 2 inconclusive (+0.06, +0.15). AFF − B′(A1), the strongest condition-free scorer: −0.52 to +0.13 over the
  five reads.
- **Meaning (C):** the supervisors' decision at the weekly meeting (2026-10-14), relayed by the user: the line became
  too specific and too complicated to make a good CVPR paper; stop it and archive the CVPR plan (user). Our reading
  recommended stage 3; it agrees that B′(A1), condition-free, is level with AFF or ahead of it.
- **Decided (A):** draft, still to be discussed (user): a condition-free model built from the buddy model (redesigned,
  not a pull of its branch), as an improvement over PercepT; PercepT's task plus retrieval, with aspect-conditioned
  retrieval as a third candidate; a method that discovers aspects, label-free if possible, a looser use of labels if
  that is too hard. CVPR is soft: submit if something works by about 2026-10-29, else try until the abstract deadline
  2026-11-10, else no submission. Type new idea, full loop (agent default).
- **Also:** the r9 choices left open (the SigLIP 2 refit, H9's claim wording) are closed by the stop, not decided.
  Held data may be re-read; only tuning or sweeps on it are flagged (user; global rule `held-data.md`).
- **Why:** a narrower, harder task with a small margin over its own bar, and no margin over a condition-free scorer.
- **Links:** [r9 report](../../archive/v2_conditioned/docs/reports/auto/v2/2026-12-02_aff_scope.md) ·
  [archive index](../../archive/v2_conditioned/README.md) · paper plan: `docs/paper/README.md` (private repo) ·
  [r10 decide handoff](../../archive/v2_conditioned/docs/superpowers/handoffs/2026-10-14-r10-decide-handoff.md)

