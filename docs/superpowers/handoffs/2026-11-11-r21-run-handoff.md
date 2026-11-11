# Handoff: r21 run: GPU check, then stages 0c, 1a, 1b and 2 of the DEC round, then verify and report

## Brief
- Done: r21 build: six tickets built, reviewed and merged into main; gates 0a and 0b pass with all diffs 0; the final review found no must-fix.
- State: DEC's gradient is 0.02 to 0.08 of c*'s at λ = 1 and 0.17 to 0.75 at λ = 10; the balance term is 6 to 45 times c*'s at λ = 1.
- Next: the GPU check, then stage 0c (`dec-off` seed 0) with `r21_queue_stage0c.sh`.
- Blocked: nothing. Long GPU stages run from 19:00 unless the GPU stays under 200 W (D17).

> Written 2026-11-11 09:23 by the r21 build 3 chat. Loop step reached: end of step 7; step 8 next. Direction log
> entry: [2026-11-11 · r21 decide](../direction_log.md).

## 0. Flags for the user (top of the run's results brief too)

1. **The DEC term at λ = 1 is small; at λ = 10 it acts.** q is almost flat on unit head outputs (mean max q 0.133 /
   0.061 / 0.138 against 1/k 0.111 / 0.037 / 0.100). But on the first DEC batch from saved c* seed 0, the DEC term's
   gradient norm over c*'s loss's is 0.017 / 0.075 / 0.023 at λ = 1 and 0.17 / 0.75 / 0.23 at λ = 10 (emotion, style,
   genre; cosine with c*'s gradient 0.05 / 0.31 / 0.16). The build 2 chat's idea of λ ∈ {100, 1000} is withdrawn:
   λ = 100 would make DEC 1.7 to 7.5 times c*'s gradient. The spec's grid stays (night mode).
2. **The balance term dominates when it is on.** PercepT's weight ratio (×1000) gives a gradient 6.3 / 45 / 27 times
   c*'s at λ = 1. A balance-on arm mostly tests the balance term.
3. **The S1 stop rule mixes DEC with longer training.** S1 keeps the last state after up to 50 DEC epochs with c*'s
   losses still on; saved c* ran 19 / 16 / 33 epochs. The `dec-off` row does not control for training length. The
   report gives the epochs run per arm beside c*'s.

Source: the final review's records,
`/tmp/claude-0/-project-CoSiR/9ad8ae03-6854-4c8f-8421-1435e89de566/scratchpad/final_review/` (`grad/grad_ratio.json`,
`flat_q.json`), summed up in the round log.

## 1. Read in this order

1. This handoff.
2. The spec, which governs: [2026-11-11-r21-dec-heads-design.md](../specs/2026-11-11-r21-dec-heads-design.md), §3.5
   (stages, gates, grid, selection), §3.6 (diagnostics), §6 (success criteria), D16 (Codex re-derives the stage 2
   table before the report), D17 (device and timing).
3. The round log and decision lines: `src/test/20261111_r21_dec_heads/20261111_r21_dec_heads_log.md`, `decisions.md`.
4. `~/.claude/references/run-planning.md` (before setting start times), `~/.claude/references/loop-chats.md`
   ("Unattended chats"), the constitution's "Decisions agents never make alone" (`docs/superpowers/constitution.md`).
5. For step 9: `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`,
   `~/.claude/templates/results_brief.md`.

Skills: none by name.

## 2. Decided by the user, not to be reopened

All on 2026-11-11, in the spec's §4 (D1 to D7) and the direction log: ablation rows beside the eight removals, c* stays
the paper's model; DEC inside c*'s training after a warm-up, target recomputed each epoch, c*'s losses kept; three
forms (plain, a pull to the start clusters, a pull to the teacher's clusters); silhouette measured; one joint row on all
heads; the ablation table's mAP; ArtELingo only; k = the class count. Run mode: night (no questions to the user).

## 3. Where things stand

**Tickets** (`.scratch/r21-dec-heads/issues/`; all merged into main):

| # | Title | Blocked by | Spec IDs | Verdict code |
|---|-------|------------|----------|--------------|
| 01 | DEC math: assignments, target, terms, inits, stop helper | none | §3.2, D3, D4, D9, D10, D12 | yes |
| 02 | Trainer: c*'s loop with DEC hooks, DEC off = frozen loop | 01 | §3.2, §3.5 (S1, S2, W), D3, D9, D10, D15 | yes |
| 03 | Runner: train CLI on c*'s chain, gate 0b, GPU smoke | 02 | §3.5 (0b, 0c), D17 | no (gate 0b script: yes) |
| 04 | Runner: held exports, teacher-cluster cache, queue scripts | 03 | §3.5 stages 1, 2; D17 | no |
| 05 | Scoring: validation scorer, gate 0a, stage 1 sheet and picks | none | §3.4, §3.5 (0a, 0c, selection), D11, SC1 | yes |
| 06 | Stage 2 change table and diagnostics | 05 | §3.4, §3.6, D5, D13, D14, SC2, SC3, SC4 | yes |

**Gates and checks:**
- Gate 0a PASS: c* and teacher AMI at the class count equal r17's file (0.102413 / 0.410926 / 0.510070; 0.088875 /
  0.406315 / 0.501466), c* mAP equals r20's for all five seeds; all diffs 0. The final review re-derived it by its own
  route: diffs 0. Record `/data/SSD2/cosir_r21/20261111_dec_heads/scores/gate0a.json`.
- Gate 0b PASS on the merged code: DEC off equals the frozen chain in 12 fixed steps and a 2-epoch smoke, all diffs 0.
  Three mutations each FAIL at their own check. Record
  `/data/SSD2/cosir_r21/20261111_dec_heads/scores/r21_gate0b_20261111_080857.json`.
- Stage 2 table regression on r10's removal arm `k11=40-no_buddy`: 39 cells equal r10's and r20's files (diff 0).
- GPU smoke (2 epochs, `dec-off`, plain S1, start S2, teacher S1, seed 0, CUDA): all `ok`; every DEC head starts
  and stops at epoch 2; histories finite; arm names and stop reasons agree; exports 6,152 × 2,048 float16. Time: DEC
  start (k-means at the class count) 18 to 49 s per head, each DEC epoch end 1.4 to 3.3 s per head; the smoke runs took
  8 s (`dec-off`) to 100 s. A full run's time is first measured at stage 0c. Log:
  `/tmp/claude-0/-project-CoSiR/9ad8ae03-6854-4c8f-8421-1435e89de566/scratchpad/gpu_smoke/smoke.log`.
- Held export on the GPU: `r21_run.py held` on saved c* seed 0 equals r10's saved held export exactly (diff 0, all
  three aspects).

## 4. The run, in order (no open points for the user)

All scripts are in `src/test/20261111_r21_dec_heads/` on main and run from `/project/CoSiR`, each in a background Bash.
Each step queues through `res-queue` (GPU training runs `--est 15m` each; CPU scoring steps) and the script stops at the
first failed step. `--dry-run` prints the commands.
1. **GPU check** (`container-ops.md`, "The GPU inside the container"). Read `run-planning.md`; set start times from the
   smoke's run time and D17: in the day only if a run stays under 200 W (`nvidia-smi --query-gpu=power.draw`), else
   from 19:00; aim to end by 11:00.
2. **Stage 0c:** `bash src/test/20261111_r21_dec_heads/r21_queue_stage0c.sh` (`dec-off` seed 0 on the GPU, its
   validation scores, the sheet, which prints gate 0c's status). Gate 0c: best epochs within 2 of 4, 1, 18; validation
   AMI within 0.004 of 0.1010, 0.4206, 0.5298. On FAIL: constitution row 4 (fix code bugs only, at most three tries,
   never loosen a tolerance).
3. **Stage 1a:** `r21_queue_stage1a.sh` (8 plain runs, then the sheet writes the 1a pick).
4. **Stage 1b:** `r21_queue_stage1b.sh` (the teacher cache first, then 5 new runs at 1a's λ, W and stop rule; the
   sheet writes one pick per form). Picks are frozen once written; never pass `--force` to `sheet`.
5. **Stage 2:** `r21_queue_stage2.sh` (`dec-off` and the three picks on seeds 0 to 4, their held exports on the GPU,
   then `r21_table.py table` (1 to 2 h of one CPU slot) and `diag`).
6. **Step 9:** D16's Codex re-derivation of the stage 2 table (the rules of §3.4 and the saved embeddings, never our
   numbers); then the auto report with its results brief and the user-read report; flags of section 0 at their top.
   Then open the next loop's decide chat (`loop-chats.md`, "The run chat opens the next loop").

## 5. Code and pitfalls

- The scorer's code hash changed in the fix wave, so the first `val` or `sheet` recomputes the cached c* seed 0
  validation scores (about a minute) and checks them against r17's `stage1_ranges.json`.
- Stage 2 queues one training job per arm and seed; the seed 0 runs exist from stages 0c and 1, and `train` skips them
  ("verified weights; skip").
- Held exports run on the device in the weights' record (CUDA). A CPU export of c* differs from r10's GPU export by one
  float16 step on about 0.14% of entries; never mix devices within the table.
- The stage 1 sheet now shows DEC's own cluster shares (last DEC epoch) beside the k-means shares: the collapse check.
- Gate 0c and the picks read validation exports only; the test set is read only by gate 0a and stage 2.
- Run test files on main in one process: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
  /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261111_r21_dec_heads` (in a worktree, one file per
  process: a known import clash).
- Outputs: `/data/SSD2/cosir_r21/20261111_dec_heads/` (`full/`, `teacher_clusters/`, `scores/`).

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of r21.
- On disk over 1 GB: nothing. Listed on `.scratch/pending_deletions.md`: the worktrees `/project/CoSiR-r21`,
  `-r21-t01` to `-t06`, `-r21-fix0b`, `-r21-fixwave` and their branches; this chat's scratchpad.
