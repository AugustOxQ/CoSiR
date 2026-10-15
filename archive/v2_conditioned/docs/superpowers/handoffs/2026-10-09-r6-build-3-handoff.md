# Handoff: build round 6 from ticket 06 onward (implement-spec), unattended, then cut to the run chat

> Written 2026-10-09 14:10 by the `r6 build 2` chat, cut at a clean point because its context passed 200k tokens.
> Loop step reached: step 7, seven of fifteen tickets merged on the integration branch. Direction log entry:
> [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md).

## 1. Read in this order

1. The ticket index `/project/CoSiR/.scratch/r6-held-test/issues/00-index.md` (local build tracker, outside git): the
   15 tickets, blocking edges, verdict-code flags, the coverage line and the wave plan. Tickets 06, 08 and 11 now end
   with "Controller notes" added from earlier reviews; they bind the implementers.
2. In `.scratch/r6-held-test/notes/`: `implementer_brief.md` (session line updated to this loop's build), the new
   `reviewer_brief.md` (what every ticket reviewer gets), `contracts.md` (read every "Amendment (controller ...)"
   paragraph: they settle the gaps the first tickets found) and `deferred_nits.md` (nits held for the final fix wave).
3. The build-2 handoff [2026-10-09-r6-build-2-handoff.md](2026-10-09-r6-build-2-handoff.md) §5 (DAS6 preparation,
   the final review, what the run handoff must carry) and the first build handoff
   [2026-10-09-r6-build-handoff.md](2026-10-09-r6-build-handoff.md) §2 to §5: both still binding.
4. `~/.claude/references/matt-chain.md` ("implement-spec, with our changes") and `~/.claude/references/loop-chats.md`
   ("Unattended chats").
5. The run log `src/test/20261125_artelingo_held_test/20261125_artelingo_held_test_log.md` on the integration branch
   (timeline up to this cut).

## 2. Decided by the user, not to be reopened

- As the first build handoff §2: the spec's §4 and the rule, approved 2026-10-09 04:53; build and run unattended; the
  run chat closes by opening `r6 read` with `--notify`.

## 3. Where things stand

- Integration branch `r6-held-test`, worktree `/project/CoSiR-r6`, pushed. Merged and reviewed: 01 (r6_common), 04
  (r6_stats), 09 (run_r6_apply_rule), 02 (r6_episodes), 03 (r6_heads, run_r6_refit), 05 (r6_context, r6_bundle) and
  10 (DTS GPU side: settings, inputs, verbaliser, listing, wrappers, sync). Head 7b0b6a9. Every ticket passed review
  after at most two fix rounds.
- Numbers already re-derived by reviewers (selection rows only): split sizes, value sets 8, 23, 10, every input SHA;
  the twelve stored episode hashes (seeds 42, 9001 to 9003); the stage-1a refit (8 arrays bit-identical, coefficient
  SHAs identical across processes); B, B0, B1 cross-fitted on seed 42 = 18.341064453125, 18.436686197916664,
  18.804931640625 (equal to round 4's `dev_seed42.json`); seed 49 out of sample bit-identical to round 3's cache.
- **Comparator clock (rule §7.7): started 2026-10-09 12:29** (6beb360, first DTS commit); 24 hours end
  2026-10-10 12:29. Pass this to the run chat.
- Decided by the controller (agent defaults, in the contracts and the log): `sensitivity_held.json` format; one
  agreement file per pass, never overwritten; smoke records carry module SHAs and the held runner and apply step
  refuse unless their bytes equal the smoked ones (ticket 15 adds that check); r6 code imports r6_common before `src`;
  GPU job inputs ship images under neutral hashed names (WikiArt paths begin with the style folder, rule §8.3 read
  strictly); `dts_settings.json` committed in F and copied byte for byte to `results/dts_settings.json`, force-added
  by the run chat before the first seed-42 call; no PM metric before the verdict, so `score_seed` gets a switch that
  leaves the nine PM scorers out (ticket 06 note).

## 4. Next steps (wave plan from here)

- Now unblocked: **06** (picks and scores; blocked by 05), **11** (DTS CPU side; 05, 10), **12** (reported GPU
  baselines, sonnet, controller-checked; 10). Then **07** (04, 06), **08** (07), **13** (07, 09), **14** (11, 12, 13),
  **15** (08, 09, 14) last. At most three implementers at once.
- **DAS6 preparation** after 10 and 12 are merged: build-2 handoff §5. Ticket 10's GPU smoke commands are in its
  final report, summarised here: build `episodes_seed9001.npz` (`build_seed(..., 9001, 64)` + `save_episodes`) to
  `F/results/smoke/gpu/`; `r6_gpu_inputs.py --episodes ... --out F/results/smoke/gpu/jobs/s9001_verbalise`; local:
  `R6_JOB_ROOT=... R6_IMAGE_DIR=F/results/gpu_images HF_HUB_CACHE_OVERRIDE=/data/SSD2/HF_home/hub R6_PYTHON=<py>
  R6_OUT=... flock -n -o -E 75 /tmp/gpu0.lock bash scripts/run_r6_verbalise.sh s9001_verbalise --wordings W1
  --check-only`; DAS6: `/usr/bin/python3 scripts/das6_sync_r6.py --node node401 --job-dir ... --images --run`, a
  "cluster run" commit, `cluster sync`, `cluster launch --node node401 -- bash scripts/run_r6_verbalise.sh
  s9001_verbalise --wordings W1 --check-only`. Read the listing check-only's batched-vs-single agreement line ("k of
  4"); it does not fail by itself.
- After all tickets: the final whole-branch review on Opus, folding in `deferred_nits.md`; one fix wave; a scoped
  re-review; merge into main, push, remove the ticket worktrees and branches; fix CoSiR's CLAUDE.md test command
  (add `MKL_NUM_THREADS=8`; r6_heads refuses without it). Then the run handoff and `loop-next <handoff> --label
  "r6 run"` (no `--notify`).

## 5. Pitfalls learned in this chat

- **Subagents must not wait on a permission prompt.** The first review of 01 sat about six hours on one. Every prompt
  now ends with "If a command is refused by the permission check, do the work another way or report it; do not wait."
  Keep that line.
- `git merge r6-held-test` inside a subagent's worktree is sometimes refused; `git reset --hard r6-held-test` (when
  the branch is already merged) or a cherry-pick works.
- Run r6 tests per file, with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. Heavy ones: `test_r6_bundle.py` about 5 min,
  `test_r6_context.py` about 3.5 min, `test_r6_heads.py` 30 s plus a 2-min refit when real.
- Reviewers run real-data re-derivations; give each the load-bearing number to re-derive and where the stored value
  is. Reviewer scratch files land in this session's scratchpad (12 MB).
- A reviewer replaced the container's `/dev/null` with a file for about a minute at 05:49 and restored it (checked).

## 6. State at handoff

- Running: nothing. No GPU job and nothing on DAS6 has been launched.
- Uncommitted: nothing; `.scratch/` is local by design.
- Ticket worktrees under `/project/CoSiR/.claude/worktrees/agent-*` (7, 475 MB in all): keep until the build's end,
  then remove with their branches (the skill's step 9). Nothing over 1 GB on disk; nothing deleted.
