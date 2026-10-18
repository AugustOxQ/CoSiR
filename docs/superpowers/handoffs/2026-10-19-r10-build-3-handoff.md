# Handoff: trace the stage 0 failure and run the final whole-branch review, then fix wave, merge, and open the run chat

## Brief
- Done: all PercepT core tickets built, reviewed and merged on `r10-percept-core`; RoBERTa store written; stage 2 gate PASS (difference 0); end-to-end smoke PASS.
- State: stage 0 FAILS. The unchanged September re-fit gives emotion AMI 0.1238 (recorded 0.1225) and genre 0.2617 (0.2274); row 2 and the unit tests pass. As built, the run skips every PercepT arm.
- Next: trace stage 0 and run the final review in parallel; one fix wave; merge into main; open the run chat.
- Blocked: nothing.

> Written 2026-10-19 00:56 by "r10 build 2". Loop step reached: step 7 (build), all tickets merged; the final review
> is left. Direction log entry: [2026-10-18, aspect graph follow-up](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. `.scratch/r10-percept-core/final_review_brief.md`: the final reviewer's brief, ready to pass to an Opus `xhigh`
   worker (add the smoke folder of §3).
3. The ticket index `.scratch/r10-percept-core/issues/00-index.md` and the spec
   [2026-10-18-percept-core-design.md](../specs/2026-10-18-percept-core-design.md) from §1, as needed (§3.5 for the
   stage 0 gate).
4. `~/.claude/references/matt-chain.md` ("The final whole-branch review"), `~/.claude/references/loop-chats.md`
   ("Unattended chats", "How to cut"), `~/.claude/references/codex-jobs.md` before any Codex job.

Skills: diagnosing-bugs (the stage 0 trace).

## 2. Decided by the user, not to be reopened

- The spec's §4 (D1 to D15) and its approval, 2026-10-18 23:34. The stage 0 gate and its failure action (§3.5: "stop
  stage 1, trace the cause, note it for the user; stage 2 still runs") are part of it. No container restart (23:34).
- Run mode night since 2026-10-18 23:40 ("switch to overnight-run mode. I am going to sleep."): no questions; agent
  defaults, recorded in the handoff.
- Exploration framing: "as many trials as possible to enlarge the ablation table and then pick".

## 3. Where things stand

| # | Ticket | Review and checks |
|---|---|---|
| 01, 03, 06, 07 | merged in build 1 | see the build 2 handoff |
| 02 | RoBERTa extraction | controller diff read; GPU smoke max abs diff 0 against September; full store written (99,999 rows, 48 s); 8 random rows re-encoded to 6e-6 |
| 04 | Stage 1 runner | Opus `high`: merge (fused rows, PCA mean sha256, K table re-derived); three test gaps fixed by Codex resume, each mutant caught |
| 05 | Stage 2 runner | Opus `high`: merge; its own CPU retrain of P2 seed 0 gives dev AUC 0.8964857 against the GPU 0.8964858; fix round by Sonnet (4 tests, npz before sidecar, `dev_auc_note` instead of `reason` when trained) |
| 06b | Score review fixes | built by Sonnet; Opus `high`: fix first (spec §6(b): failed or missing unit-test line must block "established"); fixed; controller mutant check caught |
| 08 | Stage 0 and queues | controller diff read; end-to-end smoke: every step exit 0 |

- Integration branch `r10-percept-core` at `dfa2bd2`, pushed; 82 round tests pass (61 s). `main` unchanged.
- **Stage 0, real run** (08 worktree, 499 s; `/project/CoSiR-wt/r10-pc-08-queue/src/test/20261018_percept_core/outputs/stage0.json`,
  reports and logs in `sept_stage0/stage0_outputs/1792363233321524805/`):

| Check | Recorded | Re-fit now | Status |
|---|---|---|---|
| September held-out emotion AMI (both scripts) | 0.1225 | 0.1238 | FAIL (inside the script's own tolerance 0.002) |
| September held-out genre AMI (both scripts) | 0.2274 | 0.2617 | FAIL |
| Seed 42 Stage 2 macro AUC | 0.8296 | none | FAIL: the Stage 2 script stops after the re-fit check fails |
| Row 2, our scoring | 0.088556 / 0.170954 / 0.393076 | equal | PASS |
| Four-fix unit tests | pass | 21 passed | PASS |

  Checked, not the cause: the parser (it reads 0.1225, 0.2274 and 0.8296 from the recorded report); packages (torch
  2.11.0+cu130 since 2026-04-28, sklearn 1.6.1 and numpy 2.2.6 since 2026-08-16); inputs (CLIP features 2026-09-22,
  patch features 2026-09-23, one RoBERTa revision `d750483` from 2026-09-22). Both scripts give the same re-fit, so it
  is deterministic now.
- **Topic count (spec §8 case, not a bug):** full-size Stage 1 timing runs, seed 0: clip_roberta K = 99, qwen_qwen
  K = 1, the same at every γ (the largest gap sits in a tail at about 7 times the mean gap). The code discards the
  centers above the cut, as the audit reads the paper. Spec: report it, no rerun with another γ. A seed with K < 2
  gets no Stage 2; its sidecar carries `reason`, and the sheet reports it.
- **Timings:** Stage 1 60 to 80 s per run (GPU); Stage 2 3 to 9 s per run (P1 dev AUC 0.799, P2 0.896); stage 0
  about 8 min (GPU); RoBERTa 48 s (done); one head run about 260 s on 8 threads, so the 24 head rows on three
  processes take about 40 min. GPU queue about 25 min, CPU queue about 50 min: the run ends long before 11:00.
- **Done early, kept:** the RoBERTa store (`/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions/`; the GPU
  queue skips it) and the gate rerun (`outputs/heads/genre__R0__s0.npz`, `outputs/gate.json`: PASS, target AMI
  0.406099, difference 0; the CPU queue verifies and skips it).
- End-to-end smoke folder: `/tmp/claude-0/-project-CoSiR/2dd1c3e6-64e2-406c-99f2-bf0014118ad1/scratchpad/e2e_smoke/`
  (status, logs, `smoke_times.json`).

## 4. Open points for the next step

None for the user. Agent work:

1. **The stage 0 trace** (a hard bug; first attempt per `agent-routing.md`: Codex if `codex-usage --room bug` says
   room, else Sonnet `high`), in parallel with the final review. Leads, not checked:
   - the thread cap: `pc_stage0.py` runs September with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`; September ran with
     the default (32 cores); sklearn KMeans and torch CPU ops may depend on it;
   - the GPU driver update to 595.99.02 on 2026-10-17;
   - how the recorded "rebased fixed point" was made (the snapshot script's last change `4e4480b`; code map CM-1 to
     CM-3; the September report `res/percept/` in `/project/CoSiR-buddy_prototype_conditioning`).
   If a runner setting causes it, fix `pc_stage0.py` so September runs in its own environment, and rerun stage 0 (8
   min GPU, under the lock). If the cause is outside our control, the spec's gate holds: the run stops the PercepT
   arms and notes it for the user; the gate and its tolerance are not changed without the user.
2. **The final review** (Opus `xhigh` worker, the brief above), then one fix wave (Codex or Sonnet `high`) and a scoped
   re-review (Sonnet `high`). The controller's own findings, in the brief as F1 to F4:
   - F1: the three CPU workers rewrite `gate.json` at once through a fixed temp name (`gate.pending.json`); one
     worker can crash on the rename.
   - F2: the CPU queue's final `pc_score.py` can run before the GPU queue ends, so the sheet misses PercepT outputs.
   - F3: `pc_queue_gpu.sh` without `--foreground` returns at once (nohup); the run chat must use `--foreground` inside
     a background Bash.
3. **Merge** `r10-percept-core` into main locally and push; set ticket statuses; worktrees and branches stay on
   `.scratch/pending_deletions.md` (build 2's rows are there).
4. **The run chat's handoff** lists the tickets (the index table: title, blocked by, spec IDs, verdict code), the agent
   defaults (index items 1 to 6), the controller fixes of both builds, the stage 0 outcome, the K finding, the
   timings and the run order (GPU queue, then or beside it the CPU queue, with the sheet after both). The comparator's
   re-derivation (C13) is the run chat's verify step. Open it with
   `loop-next <handoff> --label "r10 run" --unattended`; night mode, so the GPU check only.

## 5. Code and pitfalls

- `.scratch/r10-percept-core/mkwt_base.sh NN slug` makes a ticket worktree from `$BASE` (default the integration tip),
  with the data links of `mkwt.sh`.
- Codex task files: `codex/common.txt` points temp files at build 1's scratchpad (it exists). Codex 5h: 33% left at
  00:45, reset 02:03; weekly about 86%. Ledger rows for build 2 are in `/project/claude-config/codex/jobs.md`.
- Mutant checks of round code need the sibling round folders: copy the round folder into a temp tree with links to
  `src/test/*`, `src/data`, `src/utils` (a bare copy fails on `import itp_data`).
- `pc_stage0.py` writes `sept_stage0/` in the round folder (gitignored) and `outputs/stage0.json`; a smoke status is
  `SMOKE`, never `PASS`.
- The other pitfalls of the build 2 handoff (§5) still hold: September modules only in fresh processes; CLIP image
  features not normalized; `outputs/` gitignored.
- A subagent once handed back only "placeholder" (the 06b fixer); its commit was real. Read the diff and run the tests
  before trusting any hand-back.

## 6. State at handoff

- Running: nothing (no subagents, no background jobs, no GPU lock).
- Uncommitted: nothing (`.scratch/` is outside git).
- On disk over 1 GB: unchanged (stage 1 `outputs/` 1.1 GB and `outputs_fu/` 1.2 GB, deferred). Build 2 added the
  RoBERTa store (342 MB, kept) and the rows on `.scratch/pending_deletions.md` (worktrees, the smoke folder, 456 MB).
