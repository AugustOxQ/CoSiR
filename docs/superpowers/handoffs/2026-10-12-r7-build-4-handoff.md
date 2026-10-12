# Handoff: r7 build 4: the controller's chain smoke on the merged branch, verdict-code tests, the final whole-branch review, one fix wave, merge into main, then cut to r7 run

## Brief
- Done: all 16 tickets merged into `r7-design-l` (d956740); the real stage 0 passed (G1 to G4, 04:42).
- State: no stage 1, 2 or 3 number computed.
- Next: the chain smoke on the merged branch, the verdict-code tests, then the final review (Opus `xhigh`).
- Blocked: nothing. Night mode.

> Written 2026-10-12 05:03 by the `r7 build 3` chat (cut past 200k context). Loop step reached: step 7, build; every
> ticket merged, the closing checks not yet run. Direction log: this round's entry is written at its wrap-up.

## 1. Read in this order

1. `.scratch/r7-design-l/notes/build_log.md`, from "`r7 build 3` chat starts": every review, fix round, merge and
   controller fix of this chat, with the departures each ticket made.
2. `.scratch/r7-design-l/issues/00-index.md`: shared interfaces; new notes "For ticket 12 (review of 15)" and "For
   ticket 16 (review of 08)". Every ticket's `Status:` line is current.
3. `~/.claude/references/matt-chain.md` (the final review and the fix wave) and `notes/reviewer_brief.md`.
4. The previous handoffs' process references: `2026-10-12-r7-build-2-handoff.md` §1 item 4.

## 2. Decided by the user, not to be reopened

- Everything of `2026-10-12-r7-build-handoff.md` §2: spec D1 to D27 and the rule, approved 2026-10-12 02:34; build
  and run unattended (night mode).

## 3. Where things stand

| Ticket | State |
|---|---|
| 01 to 07, 09, 10 | merged in builds 1 and 2 |
| 08 | merged fb34bd5 (Opus review MERGE AFTER FIXES; Codex fix round: L13 default-position test, cells and σ asserted) |
| 14 | merged 7573435 (Opus MERGE AFTER FIXES; fix: A0 from R3 D2's heads as L10 says, value tests, rule-smoke roots) |
| 15 | merged e27124d (Opus MERGE; 0 core-key differences against `r7_gate`/`r7_chain` on 76 synthetic cases) |
| 11 | merged e9bc051 (Claude worker; Codex launch refused); real stage 0 passed, record `results/stage0_regression.json` |
| 12 | merged f3e7958 (Codex, fix round: chain smoke on the real path); controller fixes 91c7862 (`pass` key), 2d67f43 (exact array comparison, rule §9) |
| 13 | merged d956740 (Claude worker); its trial chain smoke ran the whole chain clean in about 6 min |
| 16 | merged e2f6295 (Codex); checked on synthetic files only |

- Real outputs so far: `results/stage0_regression.json`, `heads/{post,head,xheads}_{affect,image,caption,R0}`,
  `readers/reader_R0.*` with its banks (839 MB, system disk). Token cache on SSD2 (1.1 GB, ledgered).
- Codex: 5h window 4% left (reset 08:26), weekly 85%. So the next Codex jobs wait for 08:26 or go to Claude.

## 4. Open points for the build (agent defaults; nothing for the user)

1. **Chain smoke** (controller): from the r7-int worktree's RF, `python run_r7.py --chain-smoke` in the background
   (about 6 min, 8 threads), into the real `results/smoke/chain/` (unset `R7_RESULTS`). Then `run_r7_descriptive.py
   <part> --chain-smoke` for ceiling, gate, test (16 was never run on chain-smoke files).
2. **Verdict-code tests** (controller, background): `test_r7_fusion.py`, `test_r7_bundle.py`, `test_r7_gate.py`,
   `test_r7_chain.py`, `test_r7_rederive_arrays.py`, `rederive/test_rd7_stats.py` with `--rootdir=.` on r7-design-l.
3. **Final whole-branch review**, Opus `xhigh` (`final-review.md`): re-derive load-bearing numbers from code and data.
   Points it must cover:
   - 12's change to `r7_heads.build_heads` and `r7_reader.build_reader`: they now keep and validate existing files
     (resumable writes). Is the fresh-build path unchanged? Is a partial-resume path wrong anywhere?
   - 13's smoke-only departures (forced picks at n = 0, x = 0 where smoke clusters are degenerate, FV without the
     zone in smoke): they must never touch a real stage.
   - 15's minor point: `rd7_stats --overlay` takes a key missing from a phase file silently from the pipeline's
     file; add a per-phase key-set assertion.
   - Finalize: integers and arrays exact, floats within 1e-9 (rule §9); the key sets 12 writes against 14's and 15's.
4. **Then:** one fix wave; a Sonnet `high` re-review of the fixes; merge `r7-design-l` into main and push; cut to
   `r7 run` with `loop-next` (the run chat starts with `prep`).

## 5. Code and pitfalls

- Pitfalls of `2026-10-12-r7-build-3-handoff.md` §5 still hold (tests in a worktree with `--rootdir=.`; Codex
  launches from `run/` with `cd` in a separate call; nothing deleted while unattended).
- Codex launches are refused at random by the auto-mode check (11's here, with 12's identical launch passing a second
  later). A refused job goes to a Claude worker.
- A Claude Edit on a file not read in this chat fails; check the diff before a commit (91c7862's message names a fix
  that only 2d67f43 holds).
- Rule smoke reads the frozen real objects from `results/` and writes under `results/smoke/rule/`; chain smoke reads
  and writes under `results/smoke/chain/` (both in 13 and in 14).
- The seed-ledger edit of §7.2 is a docs edit for the run chat; `build` only prints it and records it.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing in CoSiR's git (`.scratch/` is outside git). claude-config: `codex/jobs.md` rows of this build
  and the disk ledger row of the token cache, committed with this cut.
- Worktrees: `/project/CoSiR-wt/r7-{int,01..16}` (r7-int is `r7-design-l`); all ticket branches merged.
- On disk over 1 GB: trial outputs of this chat, about 7 GB in the session scratchpad and `/tmp/codex-r7-12b`, listed
  on `.scratch/pending_deletions.md`; the token cache (1.1 GB, keep); `res/cluster_jobs/` 3.1 GB (round 6, keep).
