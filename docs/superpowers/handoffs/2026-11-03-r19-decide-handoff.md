# Handoff: r19 decide: grill and spec for c* past our own picks (three more encoders, SemArt new artists, cost table)

## Brief
- Done: decision A (user): six loops r17 to r22; r17, r18, r19 decided tonight, in parallel.
- State: r19 tests whether c* holds outside the encoders and artists we picked, and times c* against the teacher.
- Next: a short grill (bars set before looking), a short spec, the user's approval, then the build.
- Blocked: the DAS6 node's three GPUs each hold 9.65 GB of a process that is not a cluster job (told to the user).

> Written 2026-11-03 18:35 by tab "r16 report 3". Loop step reached: after decision A, at step 4 (medium loop, step 3
> skipped: the options are clear). Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets
> "The user's reading of the experiment overview" and "Decided (A)".

## 1. Read in this order

1. The overview's section "The plan (decided 2026-11-03)", near the end of
   [`docs/user_read/2026-11-03_experiment_overview.md`](../../user_read/2026-11-03_experiment_overview.md): loop 3 is
   r19. Its to-do table gives items 6, 16 and 19 with costs; "Clustering: what is missing" says why.
2. The SemArt check: [`src/test/20261103_semart_check/20261103_semart_check_log.md`](../../../src/test/20261103_semart_check/20261103_semart_check_log.md).
3. The last claim specs, for the bar form: `docs/superpowers/specs/2026-11-02-backbones-sweep-design.md` (r14) and
   `2026-11-02-mmart-test-design.md` (r15).
4. [`src/test/20261103_experiment_overview/facts_inventory.md`](../../../src/test/20261103_experiment_overview/facts_inventory.md),
   sections A1, B and E, for the earlier encoder tests and costs.
5. Before the grill: `~/.claude/references/grill.md`. Before the spec: `~/.claude/rules/spec-templates.md`.

Skills: grilling, domain-modeling (as grill-with-docs); cluster-run for the DAS6 node.

## 2. Decided by the user, not to be reopened

- r19's items (2026-11-03 18:30): 16 (the current c* recipe, unchanged, on DINOv2, CLIP B/32 and CLIP B/16), 6
  (SemArt genre test on artists outside ArtELingo; moved from the supplementary, user after seeing the
  recommendation), 19 (cost table: c*'s one pass against the teacher's one pass per aspect).
- For item 16 the user said "we update that with the current model": no re-tuning per encoder.
- **Each loop's report states shortly how much it changes the current paper writing: related work, method,
  experiments** (user, 2026-11-03 18:33). Put it in the spec's report section; the draft is
  `docs/paper/cvpr2027_aspect_spaces/drafts/v1/`.

## 3. Where things stand

- Earlier test on these encoders (r10, the older R6-5 model, before c*): ArtELingo held emotion AMI, DINOv2 0.0814
  NO-GO (sample only), CLIP B/32 0.0885 and B/16 0.0945 inconclusive. Matched PercepT on ArtELingo held exists for
  all three. MMArt stores exist only for Qwen, SigLIP 2, SigLIP 1 and CLIP L/14.
- SemArt (check of 2026-11-03): 11,162 paintings by 2,717 artists without ArtELingo train paintings; labels "type"
  (10 classes, close to genre), school, period; no emotion or style. The 0.95 copy cut misses copies across the two
  sources (the Last Supper at 0.944). Qwen and SigLIP 2 stores exist; SigLIP 1 and CLIP L/14 need extraction; the
  Qwen genre teacher view about 35 GPU minutes (estimate, not checked).
- Costs: c* trains in 3.5 to 5.5 min per seed on the 3090, about 516 s on a DAS6 A6000.
- DAS6: node402, about 115 h left at 18:35; its three GPUs each hold 9,650 MiB of a process that is not a cluster job,
  so `launch` picks none. Check `cluster status --node node402` again; never kill it. `cluster.conf` names node401.

## 4. Open points for the next step

- Key: the bars for item 16, set before looking (r14's form: WIN over plain and matched PercepT; the teacher
  comparison), and what a loss means for the paper (a limit sentence).
- Key: SemArt's copy check across sources (the threshold or a second check), and which encoders and comparators run.
- Key: item 16 on MMArt too (needs three extractions and matched PercepT runs), or ArtELingo held only.
- Minor: the cost table's measures (images per second, GPU memory, passes per aspect).

## 5. Code and pitfalls

- A new round folder `src/test/20261103_<name>/` with r19's own module prefix. Earlier round folders stay frozen:
  import, never edit. Score with r17's scoring module, read-only, after r17 merges (tonight); r19's build starts
  after that merge. New feature stores go in new folders under `/data/SSD2/pre_extract/<dataset>/<encoder>/`.
- Local GPU: on weekdays from 10:00 to 19:00, short or light jobs only; long runs go to DAS6 or a night. r18 shares
  both the local GPU and DAS6: take the GPU lock, and the first job to ask gets a DAS6 GPU.
- Codex: weekly limit at 5% until 2026-11-04 08:27.
- Open r19's main chat at brief approval if `--new-loop` did not: `loop-next ~/.claude/templates/main_chat_start.md
  --label "r19 main" --main --new-workspace "CoSiR r19" --model sonnet --effort high --notify`.

## 6. State at handoff

- Running: nothing of r19. Beside it: the r17 and r18 decide chats.
- Uncommitted: nothing.
- On disk over 1 GB: nothing new.
