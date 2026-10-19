# CoSiR direction log

One entry per loop of the research loop (`/project/claude-config/docs/research_loop.md`): what we read, what the user decided, why,
and where the files are. Newest first. The wrap-up step after reading results adds the next entry. Details live in the
linked reports and handoffs, not here.

New entries follow `~/.claude/templates/direction_log_entry.md`, which adds your reading of the results (decision C)
and who made each decision. Entries before 2026-10-08 were moved from project memory on 2026-10-08 and lack those two
fields; their verbatim notes are in `direction_log_raw_2026-10.md`. Older history (before the CVPR plan) is in the
weekly and stage reports.

The conditioned line's log (2026-09-28 to 2026-10-15, rounds 1 to 9, parked ideas) is archived: [direction log](../../archive/v2_conditioned/docs/superpowers/direction_log.md).

## Now (2026-10-20)

The new line builds a condition-free model over PercepT: one Qwen3-VL-Embedding instruction per aspect, buddy graphs
on each view, one shared trunk with emotion, style and genre heads, read once on the default image. The stage 2 sheet's
best row for the claim, D2 (trunk + direct distillation 0.5), reaches image-side AMI 0.096 / 0.370 / 0.454
(development data, 3 seeds) against PercepT core 0.077 / 0.229 / 0.436 and the instructed Qwen image 0.093 / 0.395 /
0.496: above the instructed target on emotion only. Next, two tracks (user): the claim test with D2, grilled on
2026-10-20 (two claim tests: the developed sample for the 10-29 gate, then the full ArtELingo train side; the bar is
the strongest uncollapsed PercepT, core or one of two paper settings versions; Leiden reported beside), its spec
and decision rule committed 2026-10-20, stage 1 building and running overnight; and decision B for cluster-level losses (contrast per head, teacher clusters as anchors, DEC started from a
good partition). Gates: 2026-10-29 held-out emotion AMI at least +0.01 above PercepT, with a paired interval above 0;
2026-11-10 the instructed Qwen bar and aspect-named retrieval.

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
  (never tried; not chosen by the user, 2026-10-20; [proposed in](../reports/auto/percept/2026-12-16_model_sheet.md))
- Align and contrast the three aspect heads of one image (shared part aligned, aspect parts pushed apart) (never tried;
  not chosen by the user, 2026-10-20: the aspects may not be separable; proposed in the r10 decide 12 chat)

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

