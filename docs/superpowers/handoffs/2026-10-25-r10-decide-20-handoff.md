# Handoff: read report 12-24 with the user (decision C), wrap up, then choose the next direction (decision A)

## Brief
- Done: the recipe and backbone round; report 12-24 and its user-read report, both checked and committed.
- State: SigLIP 2 and Qwen (new recipe) pass held emotion at both scales (+0.0150, +0.0154 against 0.0874); CLIP B/32 inconclusive.
- Next: show the user the results brief without our reading, record their reading, wrap up, then step 2.
- Blocked: nothing. Day mode (set by the user at 12:35).

> Written 2026-10-25 by "r10 run 15". Loop step reached: step 9 done; step 1 (decision C) next. Direction log entry:
> [2026-10-21 · D2 ablation read](../direction_log.md) (this round's entry is written at the wrap-up).

## 1. Read in this order

1. This handoff.
2. The results brief of [report 12-24](../../reports/auto/percept/2026-12-24_recipe_backbones.md): show it to the
   user without the folded "Our reading and recommendation", ask for their reading in a sentence or two, then open
   ours and say where the two differ (`research-loop.md`, "Decision C and the results brief").
3. The user-read report, `docs/user_read/2026-10-25_recipe_backbones.md`: the user's own briefing; its advice is
   folded the same way.
4. For the wrap-up: `~/.claude/templates/direction_log_entry.md`, `~/.claude/rules/storage.md` ("After every task").
5. For step 2: the brief's options, the framing note
   [12-21](../../reports/auto/percept/2026-12-21_advantages_framing.md) (item 7 and the "Any backbone" row), the
   literature note `docs/reports/literature/2026-12-20_ways_to_improve.md` (ideas 2 and 3), and the CVPR abstract
   date 2026-11-10.

Skills: none.

## 2. Decided by the user, not to be reopened

- The spec's decisions (§4 of [the spec](../specs/2026-10-25-recipe-and-backbones-design.md)); D8, D9, D11 and D13
  were taken after seeing our recommendation (flagged in the report's brief).
- Day mode (2026-10-25 12:35) and the stage 4 report queue run at noon instead of 19:00 (2026-10-25 12:06).
- The user saw the outcomes in chat at about 12:30 (numbers and outcome names only, none of our reading). Their
  reading has not been given yet.

## 3. Where things stand

All in [12-24](../../reports/auto/percept/2026-12-24_recipe_backbones.md); held reads in the held ledger as H13 to
H15.

- Part 1 (29,232 development paintings, seed 0): the recipe without own-caption terms at teacher share 1/5 (R6-5)
  replaces D2: emotion +0.0061, style +0.0240, genre +0.0070 (genre below one seed range); the runner-up R7-5 had a
  mean gain of 1.55 against 1.60.
- Part 2 (12,281 held paintings, 5 seeds, bar B1 own 0.0874 at full scale): SigLIP 2 GO at both scales (+0.0150
  [+0.0096, +0.0163]); Qwen with R6-5 GO (+0.0154 [+0.0100, +0.0169]); CLIP B/32 inconclusive (+0.0011).
- ArtELingo-28 (746): SigLIP 2 and both Qwen recipes qualified; CLIP B/32 fails.
- Diagnostics: the ceiling probe puts genre information in the default image; full-scale Leiden emotion intervals
  include zero for SigLIP 2 and Qwen (the claims are k-means only).

## 4. Open points for the next step

a. **Decision C (key):** the user's reading, recorded as "user" (or "user, after seeing ours" if given later).
b. **Wrap-up** (agent): the direction log entry and "Now"; the storage summary (below); `wrapup-check`, `disk-check`,
   `routing-check`; `/project/claude-config/sync.sh`.
c. **Decision A (key):** the brief's options: teacher-faithful heads or losses for genre (development data, minutes per
   run), or the narrow backbone claim for the CVPR abstract ("the emotion lead over PercepT core holds on two image
   encoders, Qwen and SigLIP 2, with Qwen3-VL-Embedding at training; CLIP B/32 is inconclusive").

## 5. Code and pitfalls

- Outputs are gitignored and live only in the main checkout, `src/test/20261025_recipe_backbones/outputs/`; change no
  round code (blocks record the code commit 08cab72).
- Report 12-24 is 17.2 KB, over the 15 KB guide, after the checks added caveats and context; full detail is in the run
  log and the outputs.
- Codex: four turns in one resumed session cost 28% of the five-hour window (each resume re-reads the whole session);
  a fresh session for the user-read report cost 3%. Ledger rows in `/project/claude-config/codex/jobs.md`.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat).
- Uncommitted: nothing after this commit.
- On disk over 1 GB: the round's `outputs/` (23 GB on the system disk, 188 GB free): results the reports rest on,
  for the storage summary; the build's scratch trees on `.scratch/pending_deletions.md` (rows from builds 16 to 20).
  This chat's scratchpad is 1.1 MB (Codex briefs and logs).
