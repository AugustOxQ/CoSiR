# Handoff: go on with the recipe sweep build (fix waves for 09 and 10, review 12, merge 13, finish 11, the GPU gates, the node checks, ticket 15, the final review), then cut to the run chat

## Brief
- Done: tickets 06, 07 and 08 merged after review and fixes (587 tests pass); real V3a gate passes (39 of 39, difference 0).
- State: 09 and 10 built and reviewed (no wrong number; test gaps and guards to fix); 12 and 13 built; 11 half done.
- Next: fixers for 09 and 10, an Opus review of 12, merge 13, a worker to finish 11; then the GPU gates and N1, N2.
- Blocked: nothing. Codex refusal is with the user (they asked another chat to check it).

> Written 2026-10-27 12:58 by "r10 build 9" (cut at the context meter's 200k note). Loop step reached: step 7, the
> build, 9 of 15 tickets merged (01 to 08, 14). Direction log entry: [2026-10-27](../direction_log.md).

## 1. Read in this order

1. `.scratch/r10-recipe-sweep/notes/build8_reviews.md` (outside git), from "# Build 9 additions" on: the facts of
   each hand-back and review this chat took, and the fix-wave decisions (agent defaults) for 09 and 10.
2. The reviews `notes/review_09.md` and `notes/review_10.md`; the briefs this chat used, as models:
   `notes/brief_07fix.txt` (a fix wave in an existing worktree), `notes/brief_10.txt`, `notes/brief_11.txt`.
3. The ticket index `.scratch/r10-recipe-sweep/issues/00-index.md`, then tickets 11, 12, 15.
4. The implementer brief `notes/implementer_brief.md` (commit lines say build 9 and this session: change them to the
   new session).
5. The spec (`docs/superpowers/specs/2026-10-27-recipe-sweep-design.md`) from §1, the binding rule
   `src/test/20261027_recipe_sweep/DECISION_RULE.md`, `~/.claude/references/matt-chain.md`; the build 9 handoff
   `2026-10-27-r10-build-9-handoff.md` §4 and §5 (still true).

Skills: `cluster-run` (N1 and N2 on node402).

## 2. Decided by the user, not to be reopened

- Decisions A, B, C, the grill and the brief: as in the build 7 handoff §2 (approved 2026-10-27 03:48). Night mode
  (`run-mode`: night at 12:57); GPU check passed at 12:57 (driver 595.99.02 on all three), no restart asked.

## 3. Where things stand

| Ticket | State | Where |
|---|---|---|
| 01 to 05, 14 | merged (builds 7, 8) | `r10-recipe-sweep` |
| 06 stage 3 | merged after fixes (8 missed mutations now caught); real V3a passes 39/39 at 12:10 | 28ab274 |
| 08 full trainer | merged after fixes (13 missed mutations now caught) | b6ad07b |
| 07 removals | merged after review and fixes (4 missed now caught) | e751870 (587 tests pass) |
| 09 stage 4 | built (61 tests); reviewed: 2 must-fix, 5 should-fix, 4 nits; scratch `check v4a` max difference 0 | `worktree-agent-a061e7003ab7805ac`, ac3f81c (on 28ab274) |
| 10 ablation | built (47 tests); reviewed: 3 must-fix (test gaps), 10 should-fix; dev and held smokes ok | `worktree-agent-a5ce812c9a6585551`, 4d78479 (on b6ad07b) |
| 11 held re-derivation | progress note: 41 tests; full self-check passes (75 items, 0 disagree, difference 0) | `worktree-agent-a9b6b08a3a20079fc`, b3d5f86 (on 28ab274) |
| 12 dev re-derivation | finished (42 tests); end-to-end on 10's smoke: 99 items, 0 differ; not reviewed | `/project/CoSiR-wt-rs12`, `r10-rs-12`, dc6f8f2 (merged e751870) |
| 13 analysis | built (not verdict code); smoke and real run on G ok; diff not read yet | `worktree-agent-a5a042609c007fe8d`, ff790bb (merged e751870) |
| 15 queues | not started | |

Suggested order (agent default; at most three implementers at once): in parallel, a Sonnet `high` fixer for 09 and
one for 10 (in their worktrees; each merges `r10-recipe-sweep` first), an Opus `high` review of 12, and a Sonnet
`high` worker to finish 11 (its brief: `notes/brief_11.txt` plus the "Ticket 11" notes; give it 09's and 10's keys
and example files, `/tmp/claude-0/r10-rs-09/outputs/stage4/` and `/tmp/claude-0/r10-rs-10/outputs/ablation/`). Read
13's diff and merge it. Merge 09 (with the `require_v4a` wiring), 10, 12, 11. Then the gates (below), `cluster
sync` and N1, N2; then 15; then the final review (Opus `xhigh`).

Gates, in order, each after the last merge that touches its hashed files:
1. V1a training: stale (07 and 08 changed `rs_knobs.py`, one of `rs_train.TRAINING_FILES`; stage 0's seed 4 refuses
   it). Move `outputs/gates/v1a_train.json` and `outputs/regression/v1a_s{0,1}/` aside (never delete), then
   `rs_train.py regress v1a --device cuda` under the GPU lock (about 15 minutes; a short check, allowed in the day).
   Never rewrite it after seed 4 trains.
2. V0a: `rs_stage0.py check v0a --part cpu`, then `--part gpu` (binds `rs_train`, `rs_sheet`).
3. V4a: `rs_stage4.py check v4a` (CPU, 2 to 3 min; run from the repo root until 09's must-fix 1 is in) and
   `rs_full.py regress v4a --device cuda` (GPU, about 2 min), after 09's merge and wiring.
4. Full-scale removal smokes in scratch: `G-no_direct`, and `G-no_content_correction` and `G-no_instructed_graphs`
   (they build `graphs/full/{uncorrected,default_view}__k20`; the k11_full rebuild took about 4 min CPU, so expect
   minutes). Commands in the 07 fixer's notes (`/tmp/claude-0/r10-fix-07/`) and `rs_full.py --help`.
5. `rs_vp.py self-check --draws 2000` (CPU, about 10 min, background; expect 0 differ).
6. `cluster sync` (a commit with "cluster run" in its subject) and N1, N2 on node402. 07 did not change
   `rs_graphs.py`, so the node's k11 gate stays valid; check it again if a fix wave touches it.

`gates/v1a_scorer.json` was written by `rs_sweep.py` before 06's one-line edit (`written` in the ignore list). No
guard reads its code hash; say so in the run notes (ticket 15) or rerun it after moving it aside.

## 4. Open points for the build (agent defaults, no user question)

a. Codex: refused by the auto-mode classifier in build 8; the user asked another chat to check the refusal (12:5x).
   Use Codex only if that chat reports it fixed; otherwise Claude workers, as now.
b. Ticket 11 and 12 independence (C13): their workers never read `rs_stage0`, `rs_stage4`, `rs_ablation`,
   `rs_sweep`, `rs_stage3` or their tests; give them example files and keys instead.
c. Ticket 12's open item: no example of an ok stage 3 configuration. Make one (a stage 3 test with a qualifying
   configuration, `--basetemp`) and have 12's reviewer or a fixer check `compare_stage3` on it.
d. One rule for F = G without a stage 3 file (decided, build 9): 08's `sweep_ended_without_candidates` (every machine
   with a stage 1 file ended in SWEEP-NONE or TOP-NONE); 09 and 10 drop `--machine` and call it.
e. The run notes (ticket 15) state: `shared_trunk`'s reading (build 8); a failing graph check stops only its own
   removal; the genre outcome reading in `build8_reviews.md` (T-LEVEL by the rule, the interval below zero).
f. Workers write files with the Write and Edit tools, not shell heredocs (one call was refused in this chat).

## 5. Code and pitfalls

- Agent worktrees start at `main`: each new implementer resets onto `r10-recipe-sweep`.
- `.claude/20261027_log.md` is tracked; merges give add/add or content conflicts there: keep both entries.
- `rs_train.run_config` now refuses a removal arm outside seeds 1 to 3 and calls `rs_full.require_stage3` (07's fix).
- `rs_stage4.code_sha256` crashes when run from inside `src/test/` (09's must-fix 1): run it from the repo root
  until fixed.
- Reviewers mutate a `git clone --shared` copy under `/tmp/claude-0/` (the classifier refuses in-place mutations
  with `git checkout` restores); git in such a clone needs a `safe.directory` entry in a temporary HOME.
- Gates bind code SHA-256s; rerun a gate after the last merge that touches its hashed files.

## 6. State at handoff

- Running: nothing (every subagent handed back; no GPU lock, cluster job or Codex job of this chat).
- Uncommitted: nothing after the cut commit (`.scratch/` notes, briefs and ticket status lines are outside git).
- On disk over 1 GB: `/tmp/claude-0/r10-rs-10/` (4.1 GB, ticket 10's random fixture; 12's end-to-end check used
  it); node402 `/local/wding/r10_rs/` (27.4 GiB). Smaller scratch of this chat: `r10-fix-06`, `-07`, `-08`,
  `r10-review-07`, `-09`, `-10`, `r10-rs-09`, `-11`, `-12`, `-13`, `r10-fixtures`, `r10-b9`. All on
  `.scratch/pending_deletions.md`.
