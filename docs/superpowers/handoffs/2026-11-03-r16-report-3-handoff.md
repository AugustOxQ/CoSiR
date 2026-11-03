# Handoff: r16 report 3: take the user's reading of the experiment overview, then their pick of to-dos (decision A)

## Brief
- Done: the experiment overview (user-read report): what exists and what is missing for clustering and retrieval, with 23 to-dos labelled by task and costed; the user's answers to the open items recorded.
- State: the grid is full on four encoders and four test sets; the gaps are outside methods, NMI/ARI/ACC and a few checks.
- Next: the user reads the overview and picks which to-dos run and when (decision A).
- Blocked: waiting for the user; experiments stay paused.

> Written 2026-11-03 17:39 by tab "r16 report 2" (cut at the context meter, over 200k). Loop step reached: r16 after
> decision C; decision A open. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md).

## 1. Read in this order

1. The overview, [`docs/user_read/2026-11-03_experiment_overview.md`](../../user_read/2026-11-03_experiment_overview.md):
   its to-do table (items 1 to 23, task, origin, machine cost, build size, earliest deadline) and "Advice" are what
   the user will answer from. Do not re-derive it.
2. The direction log's r16 entry: the sub-bullets "Open items of the r16 report chats" record every answer of this
   chat.
3. Only when a row is questioned: the facts sheet
   [`src/test/20261103_experiment_overview/facts_inventory.md`](../../../src/test/20261103_experiment_overview/facts_inventory.md)
   (each row cites its report) and the round log beside it.

Skills: none.

## 2. Decided by the user, not to be reopened

All on 2026-11-03, recorded in the direction log's r16 entry:

- CRL against c* on the same encoder in the stage note as point values; a CPU bootstrap later with a bar set first.
- Paper framing: a "training images from the domain" column; CRL, CLAY, TPIPS as context rows.
- On the decision A list: text queries through c*'s caption side (small exploration first), CRL on MMArt, CLAY
  (`github.com/kaist-ami/CLAY`, training-free, no license file).
- TPIPS: a timing probe on the released 8B late fusion model (no 2B exists); then the comparator, given its time.
- The r11 D1 correction (c* has a caption side) as a line in the direction log only.
- No bigger sweep before the paper (r14's 39 knob rows serve as the ablation).
- Deeper heads (3-layer residual): a to-do for the supplementary, not the main paper. DEC on every head: a to-do.
- The overview's scope: holes in our grid plus what a reviewer would expect; costs in GPU, CPU and build size; the
  new line only.
- Earlier, still standing: experiments paused until the user says so; the stage note's form waits for the user's
  other discussions; deadlines 2026-11-10, 11-16, 11-23.

## 3. Where things stand

- The overview's advice (ours): before 11-10, items 1 (NMI, ARI, ACC), 2 (cluster count), 7 (CRL bootstrap), 8 (CRL
  on MMArt), 13 (TPIPS probe), one small loop, mostly CPU; before 11-16, items 3 (TEMI or TURTLE on the plain
  image), 14 (TPIPS run, DAS6 or a night), 15 (CLAY), 16 (c* on DINOv2, CLIP B/32, B/16), 19 (cost table), 20
  (qualitative figure), 9, 11; supplementary 4, 6, 10, 12, 17, 18; cite only 5 and 21.
- Largest review risks named: AMI as the only clustering metric; no outside clustering method except PercepT.
- Facts behind the advice: c* trains in 3.5 to 5.5 min per seed on the 3090; Qwen reads about 11 images per second per
  view; c* held embeddings are saved (r10 `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held`, r15
  exports); r14 X-c* exports not checked.
- Codex: weekly limit at 5% (resets 2026-11-04 08:27); five-hour 78% (resets 22:11). Codex jobs before the weekly
  reset go to Claude.

## 4. Open points for the next step

- Key: the user's reading of the overview, then which items run and in what order (decision A). Ask for their pick
  first, then show where it differs from the overview's advice.
- Key: item 22, the stage note's line that c* leaves the frozen encoder's own retrieval unchanged (asked twice; the
  user answered with other questions). The stage note's form waits for the user.
- Minor: a DAS6 node is re-reserved on 2026-11-04 (one node, 3 GPUs, until the deadline); TPIPS (item 14) fits there.
- Minor: the user deletes five merged origin branches (`.scratch/pending_deletions.md`); `routing-check` items wait for
  a tooling chat after 2026-11-09.

## 5. Code and pitfalls

- zsh: use `git commit -F - <<'EOF'`; `*.json` and `*.log` are gitignored (`git add -f` for records). The paper chat
  also commits here: commit only your own paths.
- Do not install `tpips` into the CoSiR env: it pins transformers and peft; make a separate env. The 8B weights are a
  16.5 GB file (`sywang/TPIPS-Embed-Qwen3VL-8B`), into `/data/SSD2/HF_home` through the default cache link.
- CLAY's repo ships CLIP B/16 and SigLIP base code and Stanford40 prompts only; aspect prompts for emotion, style and
  genre are ours to write (a design choice to show the user).

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of this chat.
- On disk over 1 GB: nothing new.
