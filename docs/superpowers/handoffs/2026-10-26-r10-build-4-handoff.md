# Handoff: finish the relational genre build (review 09, merge 11 and smoke it, final review, fix wave, gate checks, merge, run handoff)

## Brief
- Done: tickets 04 (fixed), 05, 06 (two fix rounds) and 10 (fixed) reviewed and merged; 9 of 11 on the integration branch, 256 tests pass.
- State: 09 built by Codex (both smoke re-derivations agree, 0 differences); 11 built (queues; dry run clean).
- Next: Opus review of 09; diff read, merge and GPU smoke chain of 11; then the final review (three Opus reviewers).
- Blocked: nothing. No container restart (user, 20:01, for this build).

> Written 2026-10-26 21:31 by "r10 build 3" (cut at 200k context; no subagent or Codex job running). Loop step
> reached: step 7 (build), 9 of 11 tickets merged. Direction log entry:
> [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. [2026-10-26-r10-build-3-handoff.md](2026-10-26-r10-build-3-handoff.md) §4 c to e and §5 (gate checks, final steps,
   code-hash gates, pitfalls), and through it the build 2 and first build handoffs (routing, frozen inputs). They
   still hold; this file adds only the state.
3. `/project/CoSiR/.scratch/r10-relational-genre/notes/build_progress.md`: every hand-back, review and fix in order;
   the entries from 20:58 on are this chat's (11's departures and 09's open point are listed there).
4. The ticket index `.scratch/r10-relational-genre/issues/00-index.md`, tickets 09 and 11;
   `notes/implementer_brief.md`, `reviewer_brief.md` and `final_review_brief.md` (give their paths to every
   implementer and reviewer).
5. `~/.claude/references/matt-chain.md`, `codex-jobs.md` and `~/.claude/rules/agent-routing.md` before the first
   dispatch.

Skills: none beyond those `matt-chain.md` names.

## 2. Decided by the user, not to be reopened

- As the build handoffs: the spec's D1 to D20, M1 to M10, the brief approved at 19:57; no container restart for this
  build (user, 20:01).

## 3. Where things stand

Integration branch `r10-relational-genre` at b416d29 (worktree `/project/CoSiR-r10-rg`, pushed): tickets 01 to 08
and 10; 256 tests pass on the round folder.

| # | State | Branch, worktree | Next step |
|---|---|---|---|
| 01 to 08, 10 | done, merged | — | — |
| 09 | built b8ff55b (Codex; integration merged at 7effb61; 40 tests; `--phase a` and `--phase rung --arm GR-32` agree on the stage A smoke tree `/tmp/claude-0/rg_t05_smoke`, 0 differences) | r10-rg-t09, `/project/CoSiR-r10-t09` | Opus high review, then merge |
| 11 | built, head 15cea80 (merges the integration tip with 06 and 10; `bash -n` clean; `RG_DRY=1` chain prints 23 steps) | r10-rg-t11, `.claude/worktrees/agent-a13830f5ed5f838c6` | controller diff read, merge, GPU smoke chain |

Codex at 21:29: five-hour window 52% left (reset 01:12), weekly 88%. Ticket 09 used 22% (estimate 10%).

## 4. Open points for the build (agent work; nothing for the user)

a. **Review of 09** (Opus `high` worker, reviewer brief; hand-back `notes/handback_09.md`). Points to check beyond the
   brief: independence (no `rg_score`, `rg_report`, `eg_score3`, `rb_pick`, `rb_ami`); Codex's note that "historical
   climb run.json digests are treated as provenance because later diagnostics changed those files" (which files, and
   whether the check still binds the exports it re-derives); `--phase b` cannot run yet (no stage B outputs), so the
   reviewer checks it on synthetic files or a hand case. On FIX: one fix round by resume of the Codex session (thread
   `01a179ca-5036-7920-9050-aee4ece165db`) when `codex-usage --room` says room, else a Sonnet `high` worker.
b. **Ticket 11** (plumbing): read the diff; judge its six departures (build_progress.md, 21:13 entry), in particular
   whether rule §4.1 VA2 needs a rung check of GR-16 (the queue checks only the newest rung of each climb decision)
   and GQ-8 training after a failed EQ; check its `rg_v2` flags against 09's real CLI (`--phase rung|a|b`, `--arm`,
   `--smoke`, `--outputs`, `--workers`). Merge after 09. Then the GPU smoke chain from the round folder of the
   integration checkout: `RG_SMOKE=1 RG_OUTPUTS=/tmp/claude-0/<scratch>/rg bash rg_queue_gpu.sh` (the queue takes the
   GPU lock per step; smoke gates only note failures; stage C runs only on a smoke B-PICK; `rg_held check vc1` has no
   smoke mode, about 6 min full). Fix what the smoke shows (Sonnet `medium`).
c. **Final whole-branch review**: three Opus `xhigh` workers in parallel, one per stage, brief
   `notes/final_review_brief.md` (stage A: 01, 02, 03, 09 rung and a, the loss-scale part of 05, `rg_queue_a.sh`;
   stage B: 04, 05, 09 b, `rg_queue_b.sh`; stage C: 06, 07, 08, 10, `rg_queue_c.sh`). One fix wave, which also takes
   02's five minors and aligns every module's dirty flag with rg_held's; then the scoped re-review (Sonnet `high`).
d. **Gate checks in the real outputs, after the fix wave**, as build 3's handoff §4 c. First move the stale
   `regression/va1_scorer.json` (code 4a536ba) aside, never delete it; `check va1` and `check vb1` now refuse a stale
   file instead of keeping it.
e. Then merge into main, push, worktrees and ticket branches on `.scratch/pending_deletions.md`, the run chat's handoff
   (build handoff §4 e; queue times from `rg_queue.md`: stage A 82 min, worst 106; B 65 to 70; C 52; total about
   200 min, worst 230) and `loop-next <handoff> --label "r10 run" --unattended` after the GPU check.
f. For the user, in the run handoff and the report: the rule's line 48 says a batch is 1,024 pairs; the frozen trainer
   uses 512 pairs (1,024 rows), which VA1 needs. The code is right; the report must not repeat the rule's figure.

## 5. Code and pitfalls

- As build 3's handoff §5. Added: `rg_full`'s guard now also checks the rg and rb code on disk against the VC1 run and
  ties the VC1 run's weights to `vc1_train.json`, so any edit to `rg_loss`, `rg_variants`, `rg_full` or the rb code
  after `regress vc1` makes `held` refuse until VC1 and G are redone.
- **Mutation checks** in a scratch copy: link the sibling round folders of `src/test/` into the copy (`ln -s`), copy
  the round folder with `rsync -a --exclude=outputs`, `git init` it; the base run then fails only
  `test_rule_commit_ancestry_checks_git_for_real` (needs the real repo). The unattended hook refuses `rm` outside the
  session's own temp folder, even under `/tmp/claude-0/`.
- rg_held's `check vc1` recomputes a stale record (no blocking gap).
- Ticket 05's report code is lazy-imported, so it does not change the scorer's code hash.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat).
- Uncommitted: nothing of this build (this handoff is committed with this cut; `.scratch/` is outside git).
- On disk over 1 GB: unchanged. Scratch under `/tmp/claude-0/` (smoke and check trees: `rg06_smoke` 241 MB,
  `rv10/`, `rg_t05_smoke`, `rg_t07_*`, `rg_t08_smoke`, 45 `rg-vc2-test-*` and `rg-vc2-labels-*` folders, each under
  20 MB): delete at the build's end (smoke rule). Codex 09 left `.scratch/t09-tests/` and `RG/.t09-*.log` in its
  worktree (untracked).
