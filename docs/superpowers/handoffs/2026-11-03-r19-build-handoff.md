# Handoff: build r19 (c* past our picks: three encoders, SemArt's new artists, the cost table) from its approved spec, then open the run chat

## Brief
- Done: r19 specced, rule-reviewed (pass after 25 fixes) and approved by the user at 20:44; no container restart.
- State: 11,159 SemArt paintings by new artists to score; node402 free (3 GPUs, about 113 h left); queues empty.
- Next: tickets and build per matt-chain; the ticket that imports r17's module waits for r17's merge; then open the run chat.
- Blocked: nothing.

> Written 2026-11-03 21:10 by tab "r19 spec". Loop step reached: after step 6 (spec approved), before step 7.
> Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets "The user's reading of the
> experiment overview" and "Decided (A)".

## 1. Read in this order

1. The spec, from §1 on: [2026-11-03-past-picks-design.md](../specs/2026-11-03-past-picks-design.md). Its decision
   rule is §6 with §3.4 and §3.5; it inherits r14 part a and r15 (its header links both specs). A1 (the shared queue)
   is the newest change; §3.4, §8 and §9 carry it.
2. `~/.claude/references/matt-chain.md` (the build chain), then `~/.claude/references/codex-jobs.md` before any Codex
   job.
3. The rule review, [`src/test/20261103_past_picks/pp_rule_review.md`](../../../src/test/20261103_past_picks/pp_rule_review.md):
   the paths it checked on disk and the evidence behind each fix (the SemArt author rule, the DINOv2 tolerance, the
   loader behind the 86.3 ms reference, r14's and r15's stage 1 records).
4. The facts brief, [`src/test/20261103_past_picks/pp_facts_brief.md`](../../../src/test/20261103_past_picks/pp_facts_brief.md)
   (stores, matched-arm outputs, costs, timings).
5. The SemArt check: [`src/test/20261103_semart_check/`](../../../src/test/20261103_semart_check/20261103_semart_check_log.md)
   (`sc_check.py` is the set's rule; `sc_nearcopy.parquet` the per-painting cosines and authors).
6. `~/.claude/rules/shared-resources.md`, sections "The queue for the local GPU and CPU", "Local GPU", "CPU and RAM"
   and "DAS6 node" (new on 2026-11-03 20:45).

Skills: `to-tickets`, `implement-spec` (per `matt-chain.md`; slash-only, so read their SKILL.md); `cluster-run` for
the node402 checks and launches.

## 2. Decided by the user, not to be reopened

- All of the spec's §4 (D1 to D15 from the grill; D16 to D25 agent defaults) and A1. Approved from the brief at 20:44.
- The run's end time (r19 spec chat, 21:05): the CPU scoring may run past 11:00 on 11-04 on its one slot, until about
  13:00 (user, option a; ours the same). GPU work still ends by 10:00 on a weekday.
- No container restart before this chat (user, 21:05). GPU check at 20:50: driver 595.99.02 in the module and the
  library, GPU idle, no compute process.

## 3. Where things stand

- **SemArt set:** recomputed in the rule review from `sc_nearcopy.parquet` with `sc_check.py`'s rule: 11,162 paintings
  by 2,717 artists (gate (vi), with `sc_check.json`'s `new_artist_types`); the md5 step drops 3 (three pairs), so
  11,159 are scored; 217 paintings at nearest cosine ≥ 0.93 (D14's row). The unknown-author rule is "starts with
  UNKNOWN" (20 author forms), not "equals UNKNOWN MASTER".
- **Stores that exist:** ArtELingo DINOv2, CLIP B/32, B/16 (all 61,402 paintings); SemArt CLIP B/32, Qwen default,
  SigLIP 2; the ArtELingo `genre_word_s2img` view covers all 6,152 validation paintings. **To extract** (spec §9):
  MMArt B/32, B/16, DINOv2 (16,782 each); Affection the same three (85,007 each); SemArt SigLIP 1, CLIP L/14, B/16,
  DINOv2 and Qwen `genre_word` (21,382 each).
- **Matched arms:** M3 (`src/test/20261025_recipe_backbones/outputs/part2/full/{held,matched}/M3`), M6 and the sample
  M5 (`/data/SSD2/cosir_r10/20261025_encoders_genre/outputs/part2/{full,sample}/{held,matched}/`): outputs and all
  15 mappers present.
- **Queues at 20:44:** `res-queue` GPU and CPU idle (4 CPU slots free); node402: all three A6000s free, no launch
  waiting, 113 h left (ends about 11-08 14:00). r17's build and r18's build 2 chats opened after that and may queue
  work on both.
- **Codex:** at 18:46 the five-hour window was at 85% and the weekly at 98%; check `codex-usage --room` before each job.
- **Timing:** the build takes about 5 to 6 h from about 21:15, so it ends about 02:00 to 03:00. The run then: Affection
  extraction first (so the photo models can train), the other extractions and the probe on the 3090 (about 2¾ h,
  ending about 06:00), node402 training beside it (about 1¼ h), CPU scoring and C13 after (3 to 4 h), so the run ends
  about 09:00 to 10:00, at the latest about 13:00 (§2). If the run cannot start the heavy extraction by about 07:00 on
  11-04, the extraction waits until 19:00 (weekday rule) and the run ends early on 11-05; the report still comes about
  11-06.

## 4. Open points for the next step

None for the user. The build decides code and tickets; anything that would change a number goes in the handoff to
the run chat.

## 5. Code and pitfalls

- Round folder `src/test/20261103_past_picks/` (it holds the facts brief and the review), module prefix `pp_`. Outputs
  under `/data/SSD2/cosir_r19/20261103_past_picks/outputs/`; new stores in new folders under
  `/data/SSD2/pre_extract/<dataset>/<encoder>/`.
- **r17's module:** decision A says r17's scoring module merges first and r19 imports it read-only. It feeds only
  §3.6's context rows, so build everything else now; the ticket that wires r17's metrics in waits for r17's merge into
  main (check `git log main` for it). If it has not merged by the final review, leave the rows to the run chat, which
  skips them and notes it if the module is still missing.
- **The queue (A1):** every local GPU step is `res-queue gpu --est <duration> --label r19:<step> -- <cmd>` (`--short`
  under 15 minutes); CPU-heavy steps `res-queue cpu --label r19:<step> --est <duration> -- <cmd>`; DAS6 launches
  `cluster launch --queue --est <duration> --label r19:<step> -- <cmd>`, all with `run_in_background: true`. Queue each
  step, not a whole script, and never put a script that takes `/tmp/gpu0.lock` itself inside `res-queue`. Subagents
  pass `--no-wait`.
- **The `cluster` on PATH is Graphviz's** (`/root/miniconda3/bin/cluster`: "Can't open queue"). Use
  `~/.claude/skills/cluster-run/cluster` (the skill's `$CLUSTER`).
- **Extractors:** `scripts/extract_features.py` allows `--image-only` (DINOv2) only for ArtELingo, and its SemArt list
  holds CLIP B/32, Qwen and SigLIP 2 only; the round extractors (`bs_extract`, `mp_extract`) have no CLIP B/32, B/16 or
  DINOv2 view. The build adds these paths, copied into `pp_`; gate (i) checks them. Earlier round folders stay frozen:
  import, never edit.
- **The probe:** the end-to-end form uses the MMArt prep's loader (one prefetch thread), so gate (viii) compares like
  with like; save per-batch timings; Qwen at batch 32 is the extractor's own batch.
- **Verdict code** gets unit tests and an Opus `high` ticket review (spec §9): the SemArt set, the export and
  matched-arm paths, the picks, the scorer, the probe's timing code and the outcome code with the voids table. The rest
  gets the controller's diff read and one end-to-end smoke run.
- **Keys:** joins by key, never by store position (C6). SemArt keys are `image_file`, read through each store's
  `index.json`; evaluation order is sha256 of the key string.
- **Frozen inputs live only in `/project/CoSiR`:** gitignored outputs (r14's and r15's records, the claim test's held
  paintings) are not in worktrees; use absolute paths.
- **Git:** `*.json` is gitignored (`git add -f` for check files); `git fetch .` fails here, so move main with
  `git merge-base --is-ancestor main HEAD && git update-ref refs/heads/main HEAD`; commit only your own paths
  (`git commit -- <paths>`): the r17 and r18 chats commit on this repo tonight too.
- **Unattended (night mode):** delete nothing; list items on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing of r19 (no subagents, no Codex job, no queued job).
- Uncommitted: nothing of r19's after this cut's commit (older untracked files in `archive/` and `src/test/` predate it).
- On disk over 1 GB: nothing new.
