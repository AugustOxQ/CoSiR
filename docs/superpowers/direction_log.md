# CoSiR direction log

One entry per loop of the research loop (`/project/claude-config/docs/research_loop.md`): what we read, what the user decided, why,
and where the files are. Newest first. The wrap-up step after reading results adds the next entry. Details live in the
linked reports and handoffs, not here.

New entries follow `~/.claude/templates/direction_log_entry.md`, which adds your reading of the results (decision C)
and who made each decision. Entries before 2026-10-08 were moved from project memory on 2026-10-08 and lack those two
fields; their verbatim notes are in `direction_log_raw_2026-10.md`. Older history (before the CVPR plan) is in the
weekly and stage reports.

The conditioned line's log (2026-09-28 to 2026-10-15, rounds 1 to 9, parked ideas) is archived: [direction log](../../archive/v2_conditioned/docs/superpowers/direction_log.md).

## Now (2026-10-19)

The new line builds a condition-free model over PercepT: one Qwen3-VL-Embedding instruction per aspect, buddy graphs
on each view, one shared trunk with emotion, style and genre heads (decision B2). Best heads so far (target AMI, 3
seeds): residual graphs for emotion 0.137 and style 0.260, the image graph for genre 0.428; image side with default
input 0.081 / 0.348 / 0.458. PercepT core run 2 passed stage 0, but Algorithm 1 kept 1 to 3 or 99 topics by seed, so
its one established arm (P2, image side 0.036 / 0.162 / 0.217) is weak; with 99 topics PercepT core is about level
with its image encoder clustered directly. The user fixed the count at K = 40 (A3). Next: a short build, the rerun of
the five arms (about 20 GPU minutes), then the decision on stage 2. "Something works" by 2026-10-29: per-aspect
held-out AMI above PercepT core; by 2026-11-10: aspect-named retrieval above prompted Qwen.

## Parked ideas

- An aspect list proposed by an LLM from the captions (B3); no way yet to map it back to the three named aspects
  (deferred by the user (2026-10-18); [proposed in](handoffs/2026-10-18-r10-decide-6-handoff.md))
- PercepT under a fair protocol as the whole paper, candidate (c); the PercepT baseline stays (deferred by the user
  (2026-10-18); [proposed in](../reports/auto/percept/2026-12-06_new_line_candidates.md))
- Culture as the condition (ArtELingo-28: +0.052 Jensen-Shannon on 746 paintings) (never tried; deferred by the user
  (2026-10-18); [proposed in](handoffs/2026-10-18-r10-decide-6-handoff.md))
- Within-painting caption variation as a signal (probe 57.3% to 46.6% on the residual) (never tried; deferred by the
  user (2026-10-18); [proposed in](../reports/auto/percept/2026-12-06_new_line_candidates.md))

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

