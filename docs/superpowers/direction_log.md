# CoSiR direction log

One entry per loop of the research loop (`/project/claude-config/docs/research_loop.md`): what we read, what the user decided, why,
and where the files are. Newest first. The wrap-up step after reading results adds the next entry. Details live in the
linked reports and handoffs, not here.

New entries follow `~/.claude/templates/direction_log_entry.md`, which adds your reading of the results (decision C)
and who made each decision. Entries before 2026-10-08 were moved from project memory on 2026-10-08 and lack those two
fields; their verbatim notes are in `direction_log_raw_2026-10.md`. Older history (before the CVPR plan) is in the
weekly and stage reports.

The conditioned line's log (2026-09-28 to 2026-10-15, rounds 1 to 9, parked ideas) is archived: [direction log](../../archive/v2_conditioned/docs/superpowers/direction_log.md).

## Now (2026-10-18)

The new line builds a condition-free model over PercepT: one Qwen3-VL-Embedding instruction per aspect, buddy graphs
on each view, one shared trunk with emotion, style and genre heads (decision B2). The stage 1 follow-up (3 seeds)
confirms that residual graphs (aspect minus content similarity) train the best heads (target AMI emotion 0.137 vs
0.118 for the plain aspect graph, style 0.260 vs 0.213) and that the image graph alone is best for genre (0.428);
without an instruction at test time, the best head is 0.018 (emotion) and 0.005 (style) below the prompted view
clustered directly and 0.003 above it for genre. Next (user): PercepT core in two versions (CLIP B/32 + RoBERTa, and
Qwen), whose grill round 1 is open and whose numbers fill the follow-up's ablation table; then the decision on stage
2. "Something works" by 2026-10-29: per-aspect held-out AMI above PercepT core; by 2026-11-10: aspect-named retrieval
above prompted Qwen.

## Parked ideas

- An aspect list proposed by an LLM from the captions (B3); no way yet to map it back to the three named aspects
  (deferred by the user (2026-10-18); [proposed in](handoffs/2026-10-18-r10-decide-6-handoff.md))
- PercepT under a fair protocol as the whole paper, candidate (c); the PercepT baseline stays (deferred by the user
  (2026-10-18); [proposed in](../reports/auto/percept/2026-12-06_new_line_candidates.md))
- Culture as the condition (ArtELingo-28: +0.052 Jensen-Shannon on 746 paintings) (never tried; deferred by the user
  (2026-10-18); [proposed in](handoffs/2026-10-18-r10-decide-6-handoff.md))
- Within-painting caption variation as a signal (probe 57.3% to 46.6% on the residual) (never tried; deferred by the
  user (2026-10-18); [proposed in](../reports/auto/percept/2026-12-06_new_line_candidates.md))

## 2026-10-18 · Aspect graph follow-up read; the stage 1 order holds on three seeds; PercepT core next, with an ablation table to fill

- **Read:** 31 CPU head runs. Target AMI (mean of both sides, k = 20, 5,000 paintings, 3 seeds): residual graphs stay
  ahead of the plain aspect graph (emotion 0.137 ± 0.002 vs 0.118 ± 0.002, style 0.260 ± 0.003 vs 0.213 ± 0.005);
  genre image graph alone 0.428 ± 0.003 vs the image-weighted mix 0.420 and the caption graph 0.402. Prompted inputs
  raise the residual heads (emotion 0.151, style 0.284) to within 0.003 of the prompted view on its stronger side,
  not above it. Ablation: the best default-input head vs the default view +0.049, +0.089, +0.035; vs the prompted
  view −0.018, −0.005, +0.003 (emotion, style, genre).
- **Meaning (C):** "I think this is a good sign, so our best graph steadily beat plain aspect graph" (user, before
  ours). Ours: the same, plus the gap on the prompted view's stronger side is mostly the input; development data
  only, and stage 1 picked these graphs on the same paintings.
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

