# Handoff: write the stage 2 spec (the model sheet) from the finished grill, get the user's approval, then open the build

## Brief
- Done: decision C (comparator: PercepT core K = 40 plus the encoders clustered directly), an image-side ceiling check, and the grill of stage 2 (four key questions, minor m1 to m6).
- State: the target is the instructed Qwen image clustered directly, 0.093 / 0.395 / 0.496, with one pass on the default image; our network reaches 0.081 / 0.348 / 0.458.
- Next: write the stage 2 spec and get the user's approval of its brief; then the build chat.
- Blocked: nothing. The user restarts the container before this chat opens.

> Written 2026-10-19 19:10 by "r10 decide 10". Loop step reached: step 5 done (grill confirmed by the user); next is
> step 6 (spec). Direction log entry:
> [2026-10-19 · PercepT core with K = 40 read](../direction_log.md) (it holds every decision of the grill).

## 1. Read in this order

1. The direction log's newest entry: decision C, decision A (stage 2, exploration, medium loop), the four key answers
   with who decided, and m1 to m6. The spec writes these down; it does not re-ask them.
2. The aspect graph spec, [2026-10-18-aspect-graph-sweep-design.md](../specs/2026-10-18-aspect-graph-sweep-design.md):
   the brief (stage 2 = model sheet), §3.4 ("Model sheet"), §3.5, §4 (D3 to D5). Its status is still "draft". Agent
   default to confirm in the spec's "Check these": a new spec for stage 2 (the target and the inputs changed), which
   names this spec as its base, rather than an amendment.
3. The grill of decide 6, [2026-10-18-r10-decide-6-handoff.md](2026-10-18-r10-decide-6-handoff.md) §2: loss forms
   a, b, c (a: independent heads with own buddy positives; b: plus cross-aspect negatives; c: one embedding with a
   mask per aspect, CSN form), each head with an image-caption positive term; the mapping (Q2).
4. The ceiling log, [20261019_emotion_ceiling_log.md](../../../src/test/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md):
   the probe numbers and three candidate levers.
5. `~/.claude/rules/spec-templates.md` and the constitution `docs/superpowers/constitution.md`.

Skills: none (the grill is done).

## 2. Decided by the user, not to be reopened

All on 2026-10-19, recorded in the direction log entry above:
- Comparator option b; the extra measures V, ARI and FMI beside AMI (H and C dropped; SI, CHI and DBI left out).
- Stage 2 as the next step (decision A), after the ceiling check.
- Emotion gate at 2026-10-29: at least +0.01 over PercepT core, and clearly above the other baselines.
- Default input carries the claim; the instructed input is a second result.
- Rows are picked on image-side AMI; retrieval's pick waits until stage 3.
- Target (l): reach the instructed Qwen image with one pass on the default input, in one or two loops; judged with
  retrieval on 2026-11-10.
- m1 to m6 as recommended; no plan page.

Flag in the spec's brief (decided after seeing our recommendation): the gate timing of l, m1 to m6, the extra
measures, decision A.

## 3. Where things stand

Image-side AMI, k = 20, 5,000 development paintings, emotion / style / genre:

| Row | Passes | AMI |
|---|---:|---|
| Qwen default image, clustered directly | 1 | 0.070 / 0.222 / 0.449 |
| CLIP image, clustered directly | 1 | 0.065 / 0.300 / 0.412 |
| PercepT core, best setting (P3, P2, P3) | 1 | 0.077 / 0.229 / 0.436 |
| Instructed Qwen image, clustered directly (the target) | 3 | 0.093 / 0.395 / 0.496 |
| Our network, default input (stage 1 picks; genre: image graph head) | 1 | 0.081 / 0.348 / 0.458 |
| Our network, instructed input (row 12) | 3 | 0.087 / 0.393 / 0.458 |
| Linear probe on default Qwen (labels; reference only) | 1 | 0.121 / 0.527 / 0.657 |

- With one pass, our network recovers about 47%, 73% and 20% of the instruction's gain over the default image.
- Per label, each of our heads is the best of the three on its own aspect (emotion head 0.081 against 0.066 and
  0.072).
- The best image-side emotion head so far is row 8 (purified graph R1), 0.089; stage 1 picked Fres1 (0.081) on the
  mean of both sides.

## 4. Open points for the spec

No key question is open. Agent defaults to set and show in "Check these":
- The form of instructed-view distillation (for example a cosine loss from the default-image head output to the
  instructed image embedding of its aspect), its weight, and whether it shares the trunk.
- Which graph rows feed the trunk under image-side picks (from `ags_fu_results.json`).
- Stage 2 scoring: AMI decides; V, ARI and FMI on the same k-means partitions; the four bars and the probe row in
  every table; the two-bar table of the user-read report filled once with V, ARI and FMI.
- Cost and timing: the build takes 1 to 2 days; runs end by 11:00 (`shared-resources.md`).
- The 2026-10-29 claim test on held data gets its own spec after stage 2.

## 5. Code and pitfalls

- Helpers: `itp_data` (`load_frozen`, `split`, `default_images`), `ags_trainimg.aspect_images(s, key)` (instructed
  images of all 19,889 paintings; train stores `emotion_focus_trainimg`, `style_word_trainimg`, `genre_word_trainimg`
  exist), `pbp_graphs.label_vectors`, `pav_score.partitions` and `ami`; `pc_score.ami_scores` asserts unit norm.
  `ec_ceiling.py` shows the imports together.
- A new round folder needs a new module prefix (`ec_` is taken by the ceiling check).
- The V, ARI and FMI back-fill reads `src/test/20261018_aspect_graph_sheet/outputs_fu/` (1.2 GB, kept for this) and
  `src/test/20261018_percept_core/outputs/stage2/`. Sweep 1's head arrays and `outputs/heads/*.npz` are deleted.
- `sept_stage0/` is deleted; pytest needs no `--ignore` for it until a stage 0 rerun exports it again.
- After the restart, check the GPU before the build (`container.md`, "The GPU check").

## 6. State at handoff

- Running: nothing. The GPU is free, with no lock.
- Uncommitted: nothing (after this chat's commit).
- On disk over 1 GB: `src/test/20261018_aspect_graph_sheet/outputs_fu/` (1.2 GB; deferred until the back-fill).
