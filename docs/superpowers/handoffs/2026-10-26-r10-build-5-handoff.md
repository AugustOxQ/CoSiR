# Handoff: finish the relational genre build (fix wave for stages A and B, re-review, smoke chain, gate checks, merge, run handoff)

## Brief
- Done: all 11 tickets merged (09 after one Codex fix round); final review of stages A, B, C (three Opus xhigh); stage C fixes merged.
- State: every re-derived number matches 12-25 and 12-24 exactly; stage A review found three bugs that would stop or bypass the real run.
- Next: fix wave for stages A and B, then the scoped re-review, a fresh smoke chain, the gate checks, merge into main, the run handoff.
- Blocked: nothing. No container restart (user, 20:01, for this build).

> Written 2026-10-26 22:26 by "r10 build 4" (cut at 200k context; no subagent, Codex job or GPU lock of this chat
> running). Loop step reached: step 7 (build), final review done, fix wave half done. Direction log entry:
> [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. [2026-10-26-r10-build-4-handoff.md](2026-10-26-r10-build-4-handoff.md) §4 d to f and §5, and through it build 3's
   §4 c and §5 (gate checks, code-hash gates, pitfalls). They still hold; this file adds only the state.
3. `/project/CoSiR/.scratch/r10-relational-genre/notes/build_progress.md`, the entries from 21:32 on (this chat's).
4. The three final reviews: `notes/final_review_A.md`, `final_review_B.md`, `final_review_C.md` (each has file:line,
   the failure and a suggested fix), and `notes/review_09.md`.
5. `notes/implementer_brief.md` (give its path to every fixer); `~/.claude/references/matt-chain.md`, `codex-jobs.md`
   and `~/.claude/rules/agent-routing.md` before the first dispatch.

Skills: none beyond those `matt-chain.md` names.

## 2. Decided by the user, not to be reopened

- As the build handoffs: the spec's D1 to D20, M1 to M10, the brief approved at 19:57; no container restart for this
  build (user, 20:01).

## 3. Where things stand

Integration branch `r10-relational-genre` at 27fbc0b (worktree `/project/CoSiR-r10-rg`, pushed): tickets 01 to 11
and the stage C part of the fix wave. Full suite on ee746a9: 309 passed, 1 failed (the independence test, below).

- **Final review, re-derived numbers (all exact, difference 0.0 or 6.9e-18):** stage A: R6-5 and GR-8 A per aspect,
  GR-8's genre and emotion intervals, T_g, the partitions (against 12-25's `score_all.json`); stage B: VB1's 18 export
  hashes and partitions, GR-8 genre Ā 0.5021725 and ē 0.0327903 [0.0234956, 0.0441763], emotion and style (against
  `score_fresh.json`); stage C: 9,949 genre rows, A_K4, A_T, A_B1, d_T and its interval, b_matched, G_half (against
  12-24's `reported_stage4.json`).
- **Smoke chain** (scratch merge of 09 and 11, 21:34 to 22:01; stage B rerun with the fixed 09 at 22:10): stage A
  clean in 12 min (A-SAME, both VA2 checks agree); stage B: VB1 holds, VB2 agrees (564 s at 20 draws), report 542 s, B ends clean (22:25), B-PICK v_held
  GR-8. Stage C was not smoked yet. Status files and logs kept in `notes/smoke_build4/`.
- **Fix wave, stage C part:** done by Codex (27fbc0b, merged): the held gates check the VC1 scorer's code and VB2
  (agree, complete, mode, and the hash of the current B verdict); `rg_full train` needs B-PICK naming the arm outside a
  smoke and records the verdict hash; exit 1 after a non-finite loss; VC3 rerun counts in the verdict and report;
  `rg_held report` needs VC2 agreeing on the current verdict; `rg_vc2`'s dirty flag as rg_held's; `rg_queue.md` VC1
  recovery and the held ledger row. 145 tests; 16 guard mutations killed. Not yet re-reviewed.
- Codex at 22:20: five-hour window 30% left (reset 01:12), weekly 85%. This chat used 13% (09 fix) and 9% (fix C).

## 4. Open points for the build (agent work; nothing for the user)

a. **Fix wave, stages A and B** (Codex when `codex-usage --room` says room, else Sonnet `high` workers; disjoint files
   may run in parallel, at most one Codex job). From the reviews:
   - A1 (bug): `rg_v2.choose_a` (rg_v2.py:84) writes `stage_b_new = B_NEW + [winner]` on A-NEW; `rg_score.run_a`
     (rg_score.py:602) writes `B_NEW` and the winner in `trains_fresh`. Every A-NEW stops queue A at `va2_a` and VB2.
     Match the scorer; add an A-NEW stand-in in the scorer's real format.
   - A2 (bug): the VA1 gate reads only `pass`. Add `read_va1` to `climb` and `a` (both files pass, not smoke,
     `va1_scorer.json`'s code hash equals `scorer_code_sha256()`); make `va1_scorer` a hard queue step; move the stale
     real `regression/va1_scorer.json` (code 4a536ba) aside, never delete.
   - A3 (bug): rg_v2's records omit `rerun_of`, which the scorer copies from run.json: a VA3 rerun stops the rung
     check and phase a. Carry it, compare as a non-empty string; build the stand-in with the scorer's own record shape.
   - B1 (should-fix): `report_ab` is a hard step (rg_queue_b.sh:35) and `rg_report.loss_scale` asserts
     `loss_scale.json`: a failed loss-scale diagnostic stops stage C. Softstep with a note; the report records a
     missing or failed loss-scale run.
   - Minors: A4 (climb.json cannot be rebuilt once GR-64+ exists: fix or document "fixed in place"), A5 (VA4 misses a
     retrain moved to `replaced/`), A6 (eq.json reused without a code check), A7 (code hashes of rg_train, rg_loss,
     rg_variants in va1.json and run.json, required equal by `rg_score a`), A8 (test for the VA1 pass line, mutation
     M13), B2 (the independence test in a subprocess; it fails in a whole-folder run), B3 (test for run_b's frozen-hash
     recheck, mutation S5), B4 (VB1 gate after a failed `check vb1`; recovery: move vb1.json aside after a scorer fix),
     B5 (`rerun_of` in reported_ab.md's fresh-seed table), B7 (rg_queue.md: stale rg_v2 flag note; rg_v2 always
     recomputes, ~18 min for b).
   - Also: ticket 02's five minors (build 4 handoff §4 c; the final review brief lists them) and the dirty flag of
     `rg_train` (it borrows `eg_train3.code_commit`, whole repo `-uno`: a tracked doc edit by the run chat makes a VA3
     rerun refuse), `rg_score` (rg_score.py:348) and `rg_v2` (rg_v2.py:371): all as rg_held's
     (`git status --porcelain -- '*.py'` in the round folder).
   - Not fixed, agent default (say so in the run handoff): C9 (the guard's code set omits the frozen ms_train and ct
     modules); B6 (GR-2 and GR-4 get no old-5,000 or Leiden reading: spec §3.6 lists them under "Stage A and B", but the
     rule's §4.2 reported list does not, and the rule governs).
b. **Scoped re-review** (Sonnet `high`) of every fix-wave change, 27fbc0b included, with the mutations stage C's
   reviewer could not run: `rg_held.py:139-140` (`lo_r > 0`, `lo_t > -x` to `>=`), `rg_full.py:124` and `:128` asserts
   deleted, `rg_vc2.py:36` flipped.
c. **Fresh smoke chain** from the integration checkout's round folder, new outputs under the next chat's own scratch
   folder (`RG_SMOKE=1 RG_OUTPUTS=<scratch>/rg bash rg_queue_gpu.sh`; about 40 min; it writes about 7.5 GB to the
   system disk, so check `df -h /` first and delete the outputs at the end, keeping status files and logs). Stage C
   runs this time if the smoke gives B-PICK (it did: GR-8).
d. **Gate checks in the real outputs**, after the fix wave and its re-review: build 3 handoff §4 c, A2's move aside
   first.
e. Then build 4 handoff §4 e and f: merge into main, push, pending deletions, the run handoff (times from
   `rg_queue.md`; add about 10 min for `rg_v2 --phase b`, 564 s even at 20 draws), `loop-next <handoff> --label
   "r10 run" --unattended` after the GPU check.

## 5. Code and pitfalls

- As build 3's and build 4's handoffs §5. Every edit to `rg_score`, `rg_loss`, `rg_variants`, `rg_full` or the rb
  code after a gate check means rerunning that check (code-hash gates).
- The smoke worktree trick works: a detached worktree of a scratch merge under the session's scratch folder runs the
  queues (rg_paths resolves the frozen folders from the main checkout).
- Codex resume: `codex e --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -c mcp_servers={} --json
  resume <thread> - < brief.txt > log 2>&1`. Thread of the fix C job: `01a17a0a-9ad6-7d72-8f12-5f4286c2cebb`
  (worktree `/project/CoSiR-r10-fixC`, branch r10-rg-fixC).
- Reviewers at Opus `xhigh` passed 200k context every time; tell them to send long output to files and to do the
  mutation step before the coverage table.

## 6. State at handoff

- Running: nothing of this chat (no subagent, Codex job, GPU lock or background job).
- Uncommitted: nothing of this build (this handoff is committed with this cut; `.scratch/` is outside git).
- On disk over 1 GB: `/tmp/claude-0/final_review_B/` (4.8 GB, stage B reviewer's scratch; on
  `.scratch/pending_deletions.md`). This chat's 7.5 GB smoke outputs were deleted (smoke rule); their status files and
  logs are in `notes/smoke_build4/`. The scratch merge worktree `smoke_wt` (107 MB, detached) is on the pending list (the hook refused `git worktree remove`).
- Worktrees and branches to list at the merge: `/project/CoSiR-r10-t09` (r10-rg-t09), `/project/CoSiR-r10-fixC`
  (r10-rg-fixC), `.claude/worktrees/agent-a13830f5ed5f838c6` (r10-rg-t11), plus build 3's.
