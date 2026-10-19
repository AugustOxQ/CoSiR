# Handoff: build the stage 2 model sheet (trunk, loss forms a b c, distillation, scorer with V ARI FMI), run stage 0, then open the run chat

## Brief
- Done: the user approved the stage 2 spec without changes (19:58); no container restart (user).
- State: target, the instructed Qwen image clustered directly, 0.093 / 0.395 / 0.496 (image-side AMI); our network
  0.081 / 0.348 / 0.458, with one pass on the default image.
- Next: tickets from the spec, build, final review, merge; stage 0 (CPU regression check) last; then the run chat.
- Blocked: nothing.

> Written 2026-10-19 20:00 by "r10 decide 11". Loop step reached: step 6 done (spec approved); the next chat does
> step 7 (build). Direction log entry:
> [2026-10-19 · PercepT core with K = 40 read](../direction_log.md) (the grill; the spec link is in its Links).

## 1. Read in this order

1. The spec [2026-10-19-stage2-model-sheet-design.md](../specs/2026-10-19-stage2-model-sheet-design.md), from §1.
   Every ticket cites its D IDs; there are no SC IDs (descriptive mode), so the coverage line lists D1 to D18 and the
   §3.2 rows (A1, T1 to T5, D1 to D5, I1).
2. The base spec [2026-10-18-aspect-graph-sweep-design.md](../specs/2026-10-18-aspect-graph-sweep-design.md) §3.3 and
   §3.4 (the head and scoring that stage 2 inherits).
3. Stage 1's exact definitions: `src/test/20261018_aspect_graph_sheet/ags_rows.md`, `ags_head_build.md` ("Definitions
   settled"), and the round log's "Agent defaults" item 6 (the head's loss).
4. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

- The spec, approved as written (2026-10-19 19:58, this chat): all of §4, D1 to D18, and the base spec's D3 to D5.
- No container restart before the build (user, 2026-10-19 19:58). GPU check after the 19:08 restart: driver 595.99.02,
  GPU free, no lock.

## 3. Where things stand

Image-side AMI, k = 20, 5,000 development paintings, emotion / style / genre:

| Row | AMI |
|---|---|
| PercepT core, best arm per aspect (P3, P2, P3) | 0.077 / 0.229 / 0.436 |
| Image encoder clustered directly (Qwen, CLIP, Qwen) | 0.070 / 0.300 / 0.449 |
| Instructed Qwen image clustered directly (the target) | 0.093 / 0.395 / 0.496 |
| Linear probe on default Qwen (reference ceiling) | 0.121 / 0.527 / 0.657 |
| Our network, stage 1 picks (Fres1, Fres05, R0o) | 0.081 / 0.348 / 0.458 |
| Image-side picks, 3 seeds (Gres, Fres05, R0o) | 0.091 / 0.348 / 0.458 |

Stored files the build needs:
- `src/test/20261018_aspect_graph_sheet/graphs/` (67 MB): the graph archives of all 63 rows (fit and dev pairs).
- `.../outputs_fu/` (1.2 GB): default-input arrays for R0 and R1 seeds 1 and 2, genre R0o seeds 0 to 2, and the
  prompted runs. Stage 1's default Fres1 and Fres05 arrays are deleted; stage 0 regenerates them (seeds 0 to 2), and
  the back-fill reads the regenerated arrays.
- `ags_head_results.json` and `ags_fu_results.json`: the recorded AMI values that stage 0 must match (6 decimals).
- `src/test/20261018_percept_core/outputs/stage2/` (11 MB): PercepT core outputs for the scorer check and back-fill.

## 4. Open points for the next step

No user question is open. Agent defaults the build sets inside the spec, each recorded in the round log:
- The epoch when the three pair lists differ in size (D12 fixes one batch per aspect per step; for example, one
  epoch = one pass over the largest list, the smaller lists cycled).
- Form c's L1 weight: check the value in Veit et al. (2017) and cite it (D15).
- The temperature of relational distillation (D16); the caption input of I1 (the aspect's instructed caption, as
  follow-up (c) did).
- The run chat's queue: one script per GPU stage, under `flock -o -w`; stage 2's seed picks read seed 0's scores.

## 5. Code and pitfalls

- Helpers: `itp_data` (`load_frozen`, `split`, `default_images`), `ags_trainimg.aspect_images(s, key)` (instructed
  images of all 19,889 paintings; training stores `emotion_focus_trainimg`, `style_word_trainimg`,
  `genre_word_trainimg`), `pbp_graphs.label_vectors`, `pav_score.partitions` and `ami`; `pc_score.ami_scores` asserts
  unit norm. `src/test/20261019_emotion_ceiling/ec_ceiling.py` shows the imports together.
- Round folder `src/test/20261019_model_sheet/`, module prefix `ms_` (free; `ec_` is taken).
- `ags_head.py` calls `torch.set_num_threads(8)` and asserts that CUDA is absent. Stage 0 keeps that path (CPU,
  `CUDA_VISIBLE_DEVICES=''`, 8 threads); the GPU trainer for the sheet must not inherit the assert.
- Stage 0's separate-heads mode must train each aspect alone with stage 1's random-number order, or it cannot match
  the stored array bitwise.
- Distillation targets come from fit and dev paintings only; the evaluation paintings' instructed images only score
  the target bar (spec §3.3).
- Outputs stay float16 (5,000, 2,048), finite and unit-norm, as in stage 1.
- If stage 0 fails: trace the cause, put it at the top of the handoff, send it to the phone, and do not open the run
  chat (spec §3.5: the run waits for the user).
- Delete nothing (unattended); list items in `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing of this chat. The tab "outside-view qwen-buddy" belongs to another task.
- Uncommitted: nothing (after this chat's commit).
- On disk over 1 GB: `src/test/20261018_aspect_graph_sheet/outputs_fu/` (1.2 GB; deferred until the back-fill).
