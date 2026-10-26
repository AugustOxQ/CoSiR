# Handoff: run the relational genre round (stages A, B and C in one queue chain), then verify and write the reports

## Brief
- Done: build complete: 11 tickets, final review, one fix wave, re-review PASS, stage C smoke fixed; merged into main at b8310b1.
- State: all six build gates pass in the real outputs (VA1 both halves, EQ, VB1, VC1 both halves); 430 round tests pass in main.
- Next: end estimate, then `rg_queue_gpu.sh` from the main checkout (A, B, C; about 3.5 h), then verify and report 12-27.
- Blocked: nothing.

> Written 2026-10-27 00:32 by "r10 build 6". Loop step reached: step 7 done (build merged), step 8 (run) next.
> Direction log entry: [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. `src/test/20261027_relational_genre/rg_queue.md` (below: RG/): queues, order, gates and stops, times, recovery,
   the held ledger row. It is the run's manual.
3. The rule `RG/DECISION_RULE.md` @ 2e9529a (it wins over the spec) and the spec
   `docs/superpowers/specs/2026-10-26-relational-genre-design.md` (§3.5 stages and gates, §3.6 diagnostics, §6
   success criteria, §9 deliverables).
4. The ticket index `.scratch/r10-relational-genre/issues/00-index.md` (output layout table) and the build's controller
   notes `.scratch/r10-relational-genre/notes/build_progress.md` (every check of the build, with times).
5. Before the reports: `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`,
   `~/.claude/templates/results_brief.md`; before the cut: `~/.claude/references/loop-chats.md`.

Skills: none (the run is local; `cluster-run` only if a step moves to DAS6, not planned).

## 2. Decided by the user, not to be reopened

- The spec's §4 rows D1 to D20 and the grill's M1 to M10, the brief approved at 19:57 on 2026-10-26. D18: the run on
  2026-10-27, light GPU work; reports by 2026-10-27 22:00 (spec §9).
- No container restart for the build (user, 20:01). No restart before the run: agent default, no answer in 30
  minutes (asked 00:05 in day mode; GPU check passed at 00:01 and again at 00:32).

Agent defaults of the build (not the user's; recorded here as the build handoffs asked):
- Not fixed: C9, `rg_full`'s guard does not hash the frozen `ms_train` and `ct` modules (frozen code); B6, GR-2 and
  GR-4 get no old-5,000 or Leiden reading (rule §4.2 governs over spec §3.6; the stage A and B report says so in one
  line).
- Re-review points left: no unit test of `rg_report` (plumbing; the smoke chain covers it); a stale `eq.json` is found
  only at `rg_score a` (it cannot happen while no code changes after the gate checks; recovery in `rg_queue.md`, 4a).
- The queue trains GQ-8 after EQ, pass or fail (rule §4.1: a failed EQ leaves GQ-8 unable to win, and `rg_score a`
  needs it trained); the queue passes no `--failed` (the trainer records VA3 failures in `run.json`); the loss-scale
  run and `report_ab` are soft steps (a failure is noted, stage C still runs).
- A smoke trains and infers G's five seeds at 2 epochs (build 6), so the stage C scorer, VC2 and report have run end
  to end at smoke scale.

## 3. Where things stand

- **Code:** main at b8310b1 (integration branch `r10-relational-genre` at c438f29 merged). Outputs: `RG/outputs` links
  to `/data/SSD2/cosir_r10/20261027_relational_genre/outputs` (gitignored). Rule commit 2e9529a.
- **Gates passed in the build** (files on the real outputs, code c438f29; each binds the code hash):

| Gate | File | Result |
|---|---|---|
| VA1, training half | `regression/va1.json` | pass: 3 runs (R6-5 s0, GR-8 s0, R6-5 s1) against 12-25's frozen runs (148 s, GPU) |
| VA1, scorer half | `regression/va1_scorer.json` | pass: 16 items (A of R6-5 and GR-8 per aspect, GR-8's effect and interval, T_g) |
| EQ | `regression/eq.json` | pass: 3 items (29 s, CPU) |
| VB1 | `stage_b/vb1.json` | pass: 15 items (Ā of GR-8 and R6-5, effect and interval per aspect; 356 s) |
| VC1, training half | `stage_c/regression/vc1_train.json` | pass: R6-5 at full scale, seed 0, bit-equal to 12-24's K4 per aspect (weights, best epoch, held `img`, 12,281 rows; 165 s, GPU) |
| VC1, scorer half | `stage_c/regression/vc1_scorer.json` | pass: 25 items, none failing (169 s) |

  All at code c438f29 (not dirty, not smoke), 2026-10-26 23:48 to 23:58; 83 `pass` fields, all true, read from the
  files. The code in main (b8310b1) equals c438f29 under `src/`, so the hashes hold.

- **Build checks:** 430 round tests pass on c438f29 and again in main (150 s); `RG_DRY=1` dry run of `rg_queue_gpu.sh` in main reaches every DONE line (23 steps). Smoke chain at 2 epochs, 5,000 pairs, 20 draws
  (build 5, b99f538, and build 6's queue C rerun, 6934fc3): stage A clean in 12 min (VA1 noted as designed, EQ,
  climb, VA2 rung and a agree, A-SAME with v*_A GR-8, loss scale); stage B clean in 22 min (VB1 holds, VB2 agrees,
  B-PICK GR-8, `report_ab` 502 s); stage C clean in 10 min (five seeds 96 s, held 17 s, `score_c` 40 s, VC2 agrees with
  max difference 6.9e-18, report 175 s). Status files and logs: `.scratch/r10-relational-genre/notes/smoke_build5/`
  and `smoke_build6/`.
- **Final review and fixes:** three Opus `xhigh` reviewers by stage, all FIX (small); one fix wave (Codex for stages A
  and C, a Sonnet `high` worker for stage B and the re-derivations); a scoped Sonnet `high` re-review: PASS, 83
  mutations, 7 minor survivors (test gaps; `notes/rereview.md`); one follow-up fix. Reviewers re-derived VB1's 18
  export hashes, the stage A values against 12-25's `score_all.json`, and the stage C held values at diff 0.
- **Real outputs** before the run: the gate files above and old partition caches in `stage_a/partitions/`. A stale
  `regression/va1_scorer.json` (code 4a536ba) is in `/data/SSD2/cosir_r10/20261027_relational_genre/moved_aside/`.

### Tickets (all merged; index in `.scratch/r10-relational-genre/issues/00-index.md`)

| # | Title | Verdict code |
|---|---|---|
| 01 | Genre loss: the three switches, the anchors loss, EQ | yes |
| 02 | Sample-scale trainer: arms, VA1 training half, EQ on real batches, loss-scale run | yes (wiring) |
| 03 | Stage A scorer: effects, guard, climb, v*_A, A outcomes, VA1 scorer half | yes |
| 04 | Stage B scorer: VB1, fresh-seed effects, C levels, v_held, B outcomes | yes |
| 05 | Stage A and B reported readings | no |
| 06 | Full-scale trainer and held inference with its guard, VC1 training half | yes (guard, wiring) |
| 07 | Held scorer: d_T, d_R, G_half, PC, G outcomes, VC1 scorer half | yes |
| 08 | Stage C reported readings | no |
| 09 | VA2 and VB2 re-derivation | yes |
| 10 | VC2 re-derivation | yes |
| 11 | Queues and run notes | no |

## 4. Open points for the run (agent work; nothing for the user)

a. **End estimate first** (`shared-resources.md`, "Overnight runs end by 11:00"). `rg_queue.md` "Expected times":
   stage A 82 min (worst 106, a climb to GR-256), B 65 to 70 plus about 10 min for `rg_v2 --phase b`, C 52; total
   about 210 min, worst 240. GPU about 1 h, CPU about 2.5 h: light (spec §9), so the weekday and 11:00 rules are not
   touched. A start near 00:30 ends about 04:00 (worst 04:30). Without B-PICK the chain ends after stage B. Record the
   estimate in the run log.
b. **Start the chain** from the main checkout `/project/CoSiR`, never a worktree: a background Bash with
   `bash src/test/20261027_relational_genre/rg_queue_gpu.sh > <scratchpad>/gpu.out 2>&1`; read the status files
   `RG/outputs/queue_<a|b|c|gpu>_status.txt`. Every GPU step takes `/tmp/gpu0.lock` with a 4-hour wait. Run the GPU
   check and `nvidia-smi` first (`shared-resources.md`).
c. **Do not commit in the main checkout while a queue trains**: an interrupted VA3 rerun compares the origin run's
   commit with HEAD (`rg_train.py`, `run_variant`) and refuses after any new commit. Edits to docs are fine (dirty
   flags read only the round's `.py` files); commit them after the chain ends. No code edit during the run without
   the recovery steps of `rg_queue.md` ("Recovery"): every gate binds the code hash.
d. **First real runs:** the climb above GR-32, `rg_v2 --phase b` at full scale, and the stage C scorer and VC2 on real
   G weights have run only at smoke scale. On a VA2, VB2 or VC2 disagreement no outcome is stated; trace it by the
   rule's text (`rg_queue.md`, Recovery 4).
e. **Copy every `VA4 note: interrupted run` line** of the `score_a` log into the report and the handoff.
f. **Then step 9:** verify (Claude's claim review of every number the report states, against the output files), the
   auto report 12-27 in `docs/reports/auto/percept/` opening with a results brief, the user-read report in
   `docs/user_read/`, the held ledger row after stage C (`rg_queue.md`, last paragraph of "Recovery"; format of H17
   and H18), the index row in `docs/reports/reports_sum.md`. Then open the next loop's decide chat (`loop-chats.md`).
g. **For the user, in the report:** the rule's line 48 says a batch is 1,024 pairs; the frozen trainer uses 512 pairs
   (1,024 rows), which VA1's bit-for-bit match needs. The code is right; the report must not repeat the rule's figure.

## 5. Code and pitfalls

- The Bash tool runs zsh (or fish): put multi-step jobs in a bash script under the scratchpad; zsh does not
  word-split `$VAR` commands.
- **Code-hash gates:** `va1.json`, `va1_scorer.json`, `eq.json`, `vb1.json`, `vc1_train.json` and `vc1_scorer.json`
  carry the code hashes; `climb`, `a` and `b` refuse runs whose `rg_code_sha256` differs from `va1.json`'s; `rg_full
  held` refuses G's weights unless they carry the VC1 run's code hashes. A stale file is refused, never kept: move it
  to `<outputs>/aside/` (`mv`, never delete) and rerun.
- `rg_v2` always recomputes on a restart (no skip; about 18 min at full scale for phase b, which redoes stage A).
- A climb with GR-64 or higher on disk keeps `stage_a/climb.json` in place (`rg_queue.md`, Recovery 4).
- Outside the round, the old test file `src/test/20261018_percept_core/test_pc_stage0.py` needs the gitignored
  `res/percept/percept_stage1_embedding_snapshot.npz`: it passes in the main checkout and fails in a worktree.
- pytest's temp folder is `/tmp/pytest-of-root`. Never pass a frozen folder (`rg_paths.EG_OUT`, `RB_OUT`, `CT_OUT`)
  as `--outputs`.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of build 6).
- Uncommitted: nothing in git; build notes in `.scratch/r10-relational-genre/` (outside git).
- On disk over 1 GB: the build 5 smoke outputs (about 8.5 GB) and `/tmp/claude-0/final_review_B/` (4.8 GB), both
  system disk, and 15 build worktrees (about 100 MB each), all on `.scratch/pending_deletions.md`.
