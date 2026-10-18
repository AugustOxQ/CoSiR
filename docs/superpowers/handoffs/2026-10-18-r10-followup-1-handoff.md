# Handoff: r10 follow-up 1: design the stage 1 follow-up in chat, run it on local CPU, report it; then draft the short spec for PercepT core in two versions

## Brief
- Done: the stage 1 reports are checked and committed; the user's reading (C) and direction (A) are recorded; the approved deletes ran.
- State: residual graphs train the best heads (emotion 0.137 vs 0.120 plain, style 0.260 vs 0.218), but no head reaches the prompted view yet (emotion 0.155).
- Next: the stage 1 follow-up on CPU (small loop, design in chat), its short report, then the PercepT core spec.
- Blocked: nothing. DAS6 has no approved reservation, so runs are local.

> Written 2026-10-18 19:00 by `r10 decide 7` (cut past 200k context). Loop step reached: after decisions C and A
> for stage 1; the follow-up (b) is a small loop at its "design in chat" step. Direction log entry:
> [2026-10-18, aspect graph sheet](../direction_log.md).

## 1. Read in this order

1. The stage 1 auto report, [2026-12-11_aspect_graph_sheet](../../reports/auto/percept/2026-12-11_aspect_graph_sheet.md):
   results, analysis, deviations. The follow-up answers its open points.
2. The round folder `src/test/20261018_aspect_graph_sheet/`: `ags_rows.md`, `ags_head.py` (the head that the
   follow-up extends), `ags_queue.sh`, and the run log (agent defaults).
3. The spec (draft), [aspect graph sweep](../specs/2026-10-18-aspect-graph-sweep-design.md), §3.4 (the head).
4. For PercepT core (later in the chat): `GLOSSARY.md` (the PercepT rows), the
   [code audit](../../reports/auto/percept/2026-12-04_percept_code_audit.md), the September branch
   `experiment/percept_topic_pipeline` (worktree `/project/CoSiR-buddy_prototype_conditioning`), and
   `~/.claude/rules/spec-templates.md`.
5. For the rule task: `~/.claude/references/maintaining-rules.md`, `~/.claude/rules/habit-scope.md`, the project
   memory `feedback_exploration-number-sheet.md`.

Skills: none for the first step.

## 2. Decided by the user, not to be reopened

- Everything in the [decide 7 handoff](2026-10-18-r10-decide-7-handoff.md), section 2.
- Decision C on stage 1 (user, before ours), recorded in the direction log 2026-10-18.
- Decision A (user, 18:45): first (b), the stage 1 follow-up; then (c), PercepT core; then decide (a), stage 2.
  The stage 2 spec stays a draft until then.
- PercepT core runs in both versions (user, after seeing the recommendation "both"): CLIP B/32 + RoBERTa-GoEmotions
  (the September encoders) and the Qwen default embeddings (the same input as our heads).
- Runs are local: no DAS6 reservation is approved (node408 and node406 requests pending at 18:50; node401 none).
  Before (c), re-check with `cluster status --node node408` and `--node node406`.
- Storage of the stage 1 run (`outputs/` 1.1 GB, `graphs/` 67 MB): decided at the end of the loop (user). Keep both.
- No container restart (user, 18:55).

## 3. Where things stand

- Stage 1 numbers: the report's tables. The comparator for the follow-up is the prompted view clustered directly,
  mean of both sides: emotion 0.155, style 0.265, genre 0.424 (best heads 0.137, 0.260, 0.420).
- The follow-up, as sketched to the user (agent default; confirm in one short design message, then run):
  1. seeds 1 and 2 for the heads that have one seed and serve as references: plain and purified graphs (R0, R1) for
     emotion and style, R1 for genre (10 runs);
  2. a head on the image-only genre graph (R0o), 3 seeds;
  3. a head that reads the prompted embeddings (the aspect's image and caption views) instead of the default ones,
     on R0 and the best row per aspect (emotion Fres1, style Fres05, genre Fside025), 3 seeds (18 runs).
  About 31 runs at about 10 min each, 3 processes in parallel: about 1.5 h CPU, plus the small input-key change.
- All inputs exist: prompted training images for emotion_focus, style_word and genre_word (14,889 each), all
  captions, the 5,000 evaluation images.
- Codex has room (73% of the five-hour window at 18:55; reset 20:05).

## 4. Open points for the next step

1. **Key: the follow-up design**, one short message (small loop, no spec file). One point to flag: a head that reads
   the prompted embeddings needs the aspect instruction at test time, so it tests the gap to the reference, not the
   final condition-free design.
2. **After the run:** a short auto report (next sequence date, `2026-12-12_…`) and a half-page user-read report, with
   their checks; the user's reading first.
3. **Then (c):** a medium loop of type baseline. A short grill and a short spec for PercepT core in both versions,
   with its cost and end time (the 11:00 rule, `shared-resources.md`) and the GPU check.
4. **Do: the global rule** for the number-sheet guidance (user's yes, 18:05), after `maintaining-rules.md`. Proposed
   home: a new item in `rules/spec-templates.md`, "Writing a spec", and a line in `references/grill.md`; the project
   memory then points at the rule.
5. **Minor:** the run log of stage 1 has two slips (39 head runs is 28; 16.8% is 14.3%). The report's Deviations
   record them.

## 5. Code and pitfalls

- Keep the follow-up in the same round folder (`ags_` prefix), with its own queue script and a new output subfolder;
  never overwrite stage 1 outputs while their storage decision is open.
- `ags_head.py --score` caches per output name; new runs need new names.
- CPU only: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, three processes as in stage 1. No GPU lock needed.
- Held data: none. Everything stays on the frozen train-side sample.
- This chat is opened with `--notify`: it needs the user for the design and the reading.

## 6. State at handoff

- Running: nothing (no job, subagent, Codex job or GPU lock).
- Uncommitted: untracked files from before this chat (archive result folders, `sample.npz`, `queue_status.txt` in
  the pilot folders).
- On disk over 1 GB: the aspect-instruction store (6.5 GB, keep); `src/test/20261018_aspect_graph_sheet/outputs/`
  (1.1 GB, decided at the end of the loop).
