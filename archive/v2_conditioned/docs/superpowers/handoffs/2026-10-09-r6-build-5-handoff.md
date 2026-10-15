# Handoff: build round 6's ticket 15, then the final whole-branch review, fix wave, merge and the run handoff (unattended)

> Written 2026-10-09 17:50 by the `r6 build 4` chat, cut at a clean point because its context passed 200k tokens.
> Loop step reached: step 7. Fourteen of fifteen tickets are merged on the integration branch.
> Direction log entry: [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. The ticket index `/project/CoSiR/.scratch/r6-held-test/issues/00-index.md` (local build tracker, outside git) and
   **ticket 15** (`15-smoke-chain.md`). Its four "Controller notes" blocks bind its implementer: the descriptive
   runner's smoke-SHA refusal, the `held_arrays` suffix round trip, `--after-crash` / `--reserve` smoke records with a
   top-level `"passed": true`, and `external_sources.json` plus the DTS stop stage in the smoke.
2. In `.scratch/r6-held-test/notes/`:
   - `implementer_brief.md` and `reviewer_brief.md`: update the brief's session line to your chat.
   - `contracts.md`: every "Amendment (controller ...)"; this chat added five (16:22 to 17:32, from tickets 13, 08 and
     14).
   - `deferred_nits.md`: for the final fix wave; ticket 13's departures from plan §10 are listed there for the final
     review to rate.
   - `run_handoff_items.md` (new): items the run handoff must carry, beyond build-2 handoff §5.
3. Still binding:
   - the build-4 handoff [2026-10-09-r6-build-4-handoff.md](2026-10-09-r6-build-4-handoff.md) §2, §4 (after the
     tickets) and §5;
   - the build-3 handoff [2026-10-09-r6-build-3-handoff.md](2026-10-09-r6-build-3-handoff.md) §5;
   - the build-2 handoff [2026-10-09-r6-build-2-handoff.md](2026-10-09-r6-build-2-handoff.md) §5 (the final review;
     what the run handoff must carry);
   - the first build handoff [2026-10-09-r6-build-handoff.md](2026-10-09-r6-build-handoff.md) §2 to §5.
4. `~/.claude/references/matt-chain.md` ("implement-spec, with our changes") and `~/.claude/references/loop-chats.md`
   ("Unattended chats").
5. The run log `src/test/20261125_artelingo_held_test/20261125_artelingo_held_test_log.md` on the integration branch,
   rows from 14:36 on.

## 2. Decided by the user, not to be reopened

- As the build-4 handoff §2: the spec's §4 and the rule (approved 2026-10-09 04:53); build and run unattended; the
  run chat closes by opening `r6 read` with `--notify`; **delete nothing** in the unattended build (items go to
  `/project/CoSiR/.scratch/pending_deletions.md`).

## 3. Where things stand

- Integration branch `r6-held-test`, worktree `/project/CoSiR-r6`, head 686e494, pushed.
- Merged in this chat, each reviewed by Opus, given one fix round, then checked by the controller:
  - 07: regression runner and seed-42 sensitivity input.
  - 13: descriptive pass, core.
  - 08: held runner and smoke mode.
  - 14: descriptive external rows.
- Re-derived by this chat's reviewers, independently of the code:
  - every rule §6.4 target from round 3's and round 4's stored arrays;
  - the nine seed-42 Holm counts and σ parts;
  - round 3's bar margin, per-pair margins, R1's checks, gate shares and pick accuracy;
  - `held_pass.json` and `sensitivity_held.json` for the real smoke (N 576) and a synthetic held run (N 36,864);
  - FT-LP on 16 new rows, the three checkpoint SHAs, and the MLLM un-permutation.
- In ticket 07's worktree the real seed-42 regression passed 131 of 131 items. Ticket 08 then changed
  `run_r6_held.py`, so that record is stale by design. The run chat reruns regression and sensitivity before the smoke
  (rule §6.7); held and smoke modes refuse otherwise.
- **Comparator clock (rule §7.7): started 2026-10-09 12:29; DTS must be built by 2026-10-10 12:29.** The run chat
  must first run rule §6 items 1 to 5, then DTS sanity, seed-42 tuning and the chosen setting on 12,288 episodes
  (about 1.1 h of verbaliser time on nine GPUs, plus syncs). So finish ticket 15, the final review, the fix wave and
  the re-review tonight; open the run chat by about 02:00 at the latest. If the review's fix wave would run past that,
  log it and tell the run chat to start rule §6 items 1 to 6 on the reviewed code, so the clock is kept.
- DAS6 preparation: unchanged from the build-4 handoff §3 (node401, node402, node408 ready; seed-42 images staged).

## 4. Next steps

1. **15** (Opus implementer, then an Opus reviewer), blocked by 08, 09, 14 (all merged).
2. The final whole-branch review on Opus (`final-review.md`: re-derive every load-bearing number, hard rule 6 with its
   four gap types), folding in `deferred_nits.md`. Then one fix wave and a scoped re-review.
3. Merge `r6-held-test` into main and push.
4. **Remove no worktree or branch**: list them in `pending_deletions.md`. Rows for tickets 07, 13, 08, 14 and this
   chat's scratchpad are already there; add 15's.
5. Fix CoSiR's `CLAUDE.md` test command on main: add `MKL_NUM_THREADS=8`.
6. Write the run handoff, with build-2 handoff §5's contents, build-4 handoff §5 and `run_handoff_items.md`. Then
   `loop-next <handoff> --label "r6 run" --unattended` (no `--notify`).

## 5. Code and pitfalls (new in this chat)

- **Fix rounds go to the same implementer.** `SendMessage` to a finished implementer's agent ID resumes it with its
  context. That is cheaper and better than a fresh subagent; the IDs do not carry over to a new chat.
- **Repeated notifications.** A background subagent that waits on its own background job sends repeated interim
  "completed" notifications. The real report arrives as a separate message; ignore the interim ones.
- **Contract decisions made in this chat** (all in `contracts.md`):
  - `held_arrays` file names follow the pass file's suffix; `seed_index` is the seed's position.
  - H5's episode cell holds the nine 64-hex hashes. GPU-job SHAs go in H5's purpose cell, never after the runner's
    SHA.
  - `smoke_record_crash1.json` serves code corrected after a crash, and `smoke_record_reserve.json` the reserve read.
  - Smoke records need `"passed": true`.
  - In real mode, the descriptive pass refuses unless `external_sources.json` has all four entries.
- Agent defaults stricter than the rule (logged for the run chat): a crashed `--fix 1` cannot be rerun, and any
  changed AB file stops before `held_started.json` with exit 1.
- Times: the build-3 chat's log rows ran ahead of the clock (corrected at 14:36). Take every time from
  `TZ=Europe/Amsterdam date`.

## 6. State at handoff

- Running: nothing. No subagent, no GPU job, no DAS6 job.
- Uncommitted: nothing; `.scratch/` is local by design.
- Storage (nothing deleted; for the next storage summary): `/project/CoSiR/.scratch/pending_deletions.md` gained
  four ticket worktrees (about 70 MB each; 07's and 08's `results/` hold stale seed-42 and smoke runs) and this
  chat's session scratchpad (875 MB, reviewers' copies and logs). Earlier rows are as in the build-4 handoff §6.
