# Handoff: r13 build 5: the final whole-branch review (brief ready), one fix wave, re-review, merge to main, then the run handoff

## Brief
- Done: build chat 4 merged tickets 12 and 13; all 13 tickets are built and merged. The end-to-end smoke passes.
- State: the smoke found two bugs, both fixed. The C13 re-derivation agrees with our pipeline on all 298 items.
- Next: the final Opus `xhigh` review (brief ready), one fix wave, a Sonnet `high` re-review, merge into main, push, then the run handoff.
- Blocked: nothing.

> Written 2026-11-01 17:35 by "r13 build 4". Loop step reached: step 7 (build; all 13 tickets merged, smoke passed;
> final review not started). Direction log entry: [2026-10-31](../direction_log.md). Run mode: night.

## 1. Read in this order

1. The final review brief, ready to send: `/project/CoSiR/.scratch/r13-photo-finetune/notes/final_review_brief.md`.
   Send it as written to a `worker` on `opus` at `xhigh` (`~/.claude/references/matt-chain.md`, "The final whole-branch
   review"). It points at the smoke folder and asks for a leakage check of one smoke number (§3 below).
2. `/project/CoSiR/.scratch/r13-photo-finetune/notes/fix_wave.md`: every open review note, per ticket. With the final
   review's findings it makes up the fix wave. New in build 4: the t12 merge line and the t13 section.
3. The round log `src/test/20261031_photo_finetune/20261031_photo_finetune_log.md` on the integration branch (smoke 1,
   its timings and its two fixes).
4. `~/.claude/references/matt-chain.md` (fix wave, re-review, merge), `~/.claude/references/codex-jobs.md` (fixes go to
   Codex when it has room), `~/.claude/references/loop-chats.md` ("How to cut", for the run handoff).
5. For the run handoff: build 1 handoff §3 to §5 (stores, timing), build 2 handoff §4 and build 3 handoff §4 (agent
   defaults), and fix_wave.md's t10 item 2, t11 paragraph and t13 section.

Skills: none (matt-chain.md drives the review and fix wave).

## 2. Decided by the user, not to be reopened

The spec's §4 (D1 to D14), approved 2026-11-01 15:33. Nothing new was asked in build chats 1 to 4.

## 3. Where things stand

Integration branch `r13-photo-finetune` at `500ef02`, pushed (worktree `/project/CoSiR/.claude/worktrees/r13-int`);
241 fast tests pass (`-m 'not slow'` on the round folder).

- **Ticket 12** (C13 re-derivation, `pf_rederive.py`): Codex fix round by resume, commit `c2790fc`: all 7 review items
  fixed, 15 of 15 targeted mutants now fail a test. Merged `acdf0ec`.
- **Ticket 13** (queue `pf_queue.sh`; Sonnet `medium` worker, controller diff read): `full | smoke <dir> [--smoke-n N]
  [--dry-run]`; GPU steps (two extractions, stages 0a, 1, 2) under `flock -o -w 3600 -E 75 /tmp/gpu0.lock`; stage 3,
  C13 and diag on CPU (pf_export infers on CPU). Also `pf_run.require_gates`: stages 1 to 3 refuse without an ok
  `gates.json`. Merged `d34eb6c`.
- **Smoke 1** (17:03 to 17:31; folder
  `/tmp/claude-0/-project-CoSiR/bdddaed0-e3a1-4db8-b151-872b891c175d/scratchpad/smoke1`, 650 MB; keep it for the final
  review). Two controller fixes: `0fb35fe` (the smoke runs stage 0a at the full draws; the SC replay gate matches r12's
  intervals within 1e-9 only there) and `85f54be` (diag computes the photo R0 mAP through the read-out `none+k0`, as the
  verdict rows and r12 do; a bare retrieval differed by 3e-9). Then: 10 of 10 gate items ok, C13 compare 298 items all
  equal within 1e-6, diag ok.
- **A smoke number to check, not a result:** after 2 epochs on 2,000 train photos, row 1's sweep score is 0.389 (above
  the teacher on validation AMI and mAP; frozen c* row 0: -18.6). The stage 3 photo row has test AMI 0.1648 and mAP
  0.3001 (teacher 0.1666 and 0.2945; c* 0.1384 and 0.2755). This can be fast imitation of the teacher image. The final
  review brief asks the reviewer to rule out leakage of validation or test data into the training loss.
- **Timing for the run chat** (smoke, with model load, not checked at full size): images 2,000 in 129 s (about 15.5
  per second; the spec's 19.4 per second was measured without load), so 72,255 images take about 65 to 78 minutes;
  texts 38,913 in 183 s (about 210 per second), so about 474,000 texts take about 35 to 40 minutes (the spec said about
  26, not checked). Stage 0a at the full draws: about 9 minutes. Diag at the smoke: 75 s.
- **Codex:** 5h 53% left at 17:31 (reset 20:36), weekly 48%. Ledger rows are up to date
  (`/project/claude-config/codex/jobs.md`).
- **GPU check at 17:31:** driver 595.99.02 answers, module and library match, GPU idle, no lock.

## 4. Open points for the next step

None for the user. Agent defaults to add to the run handoff, beside builds 1 to 3: the smoke runs stage 0a at the full
draws; the queue commits the sheet and the near-copy record in one commit after stage 1; a rerun skips sealed steps
and fails on a changed committed record; diag's R0 mAP goes through `none+k0`.

## 5. Code and pitfalls

- Build 3's §5, build 2's §5 and build 1's §5 still hold.
- The queue writes its own logs as `OUTPUTS/logs/queue_<step>.log` and timings as `OUTPUTS/queue_seconds.txt`;
  `pf_run.py` keeps `OUTPUTS/logs/<stage>.log` and `OUTPUTS/step_seconds.txt`.
- `gates.json` with any `"ok": false` is not a seal: the queue reruns all of stage 0a (about 9 minutes). To redo one
  part, run `pf_run.py 0a --parts <part>` (it recollects `gates.json` from all evidence), then rerun the queue.
- Full mode needs the main checkout `/project/CoSiR` on main with no tracked changes in the round folder, so the
  merge into main comes before the run.
- At the end of the build, list on `.scratch/pending_deletions.md`: the worktrees `.claude/worktrees/{r13-t04, r13-t08,
  r13-t12, agent-*}` (ten `agent-*` folders now, the t13 one included) and their ticket branches (r13-t01 to r13-t13,
  worktree-agent-a19d52358fe5356fc, worktree-agent-a0b6a80528eda077e, worktree-agent-a5854a3da887cbd6f), the smoke
  folder above (650 MB), and the scratchpads of build chats 2 to 4 (small).

## 6. State at handoff

- Running: nothing (no subagents, no Codex job, no GPU lock).
- Uncommitted: nothing of this chat's in the project (`.scratch/` is outside git by design).
- On disk over 1 GB: nothing new.
