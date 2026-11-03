# Handoff: r17 build 3: stage 3 smoke, final whole-branch review, fix wave, merge into main, run handoff

## Brief
- Done: all 11 tickets built, reviewed and merged into `r17-new-scores` (624b980); full replays pass.
- State: gates (i) a to d, (ii) and (vi) match stored values exactly (diff 0.0): 209 and 150 items.
- Next: smoke stage 3 and the runner's gate wrappers, then the Opus `xhigh` final review, fixes, merge, run handoff.
- Blocked: nothing. Codex has room from now (5h window resets 23:34). The run should start by about 01:00.

> Written 2026-11-03 23:19 by tab "r17 build 2" (context meter past 200k). Loop step reached: step 7 (build), every
> ticket merged, final review not yet run. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md),
> bullet "r17 spec approved".

## 1. Read in this order

1. The controller notes, `/project/CoSiR/.scratch/r17-new-scores/notes/controller_notes.md`, section "r17 build 2":
   each hand-back's and review's key facts, and every note kept for the report.
2. The ticket index, `/project/CoSiR/.scratch/r17-new-scores/issues/00-index.md`: tickets, shared interfaces, coverage
   line, controller decisions (binding).
3. The spec from §1 on: [2026-11-03-new-scores-design.md](../specs/2026-11-03-new-scores-design.md) (rule: §6 with
   §3.4 and §3.5; §3.7 is the frozen scorer interface; §9 the budget).
4. `~/.claude/references/matt-chain.md`, "The final whole-branch review" (and the merge line of `implement-spec`'s
   notes there), then `~/.claude/references/codex-jobs.md` before a Codex job.
5. The first build handoff, [2026-11-03-r17-build-handoff.md](2026-11-03-r17-build-handoff.md), §5 (pitfalls: queue,
   threads, git).

Skills: none needed for the first step.

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D20), approved 2026-11-03 20:45 from its brief; no container restart (user).

## 3. Where things stand

**Integration branch** `r17-new-scores` (worktree `/project/CoSiR/.claude/worktrees/r17-int`) at 624b980, pushed. It
holds all of `src/test/20261103_new_scores/`: `ns_paths`, `ns_metrics` (01), `ns_outcome` (02), `ns_data` (03),
`ns_photo` (04), `ns_mappers` (05), `ns_clu`, `ns_clu_replay` (06), `ns_clu_cells` (07), `ns_ret` (08), `ns_picks`
(09), `ns_run`, `ns_run_cpu.sh`, `ns_score` (10), `ns_diag` (11). Tickets 01 to 09 each had an Opus `high` review
(all "approve with notes"; every should-fix closed with a test whose sabotage is caught); 10 and 11 are plumbing
(controller's merge and CLI check, smoke below). Main has moved past the integration base (r19's commits).

**The ticket list** (for the record; title, blocked by, spec IDs, verdict code): the table in `00-index.md`. All 11
are `Status: done`.

**Full replays at full draws, all max abs diff 0.0** (files under `/project/CoSiR/.scratch/r17-new-scores/smoke/`):

| Gate | Replay | Items | Bar | File |
|---|---|---|---|---|
| (i) a | r14 SigLIP 2 ArtELingo SC1, SC6 | 30 | 1e-9 | `t06/a_r14.json` |
| (i) a | r15 CLIP L/14 MMArt SC1, SC4 | 30 | 1e-9 | `t06/a_r15_fixed.json` |
| (i) a | r10 stage 4 c* held partitions | 15 | exact | `t06/a_partitions.json` |
| (i) b | r14 SC3 (`b_matched`) | 18 | 1e-9 | `t06/b.json` |
| (i) c | Affection SC4, SC6; EmoSet SC7, SC9 | 10, 10 | 1e-9 | `t06/c_affection.json`, `c_emoset.json` |
| (i) d | EmoSet K = 8; r15 class count | 24, 72 | 1e-9 | `t06/d_emoset.json`, `d_class_count.json` |
| (ii) | scorer, CRL, PercepT | 105, 12, 33 | 1e-9 | `t08/stage0/gate_ii_*.json` |
| (iv) | 20 mappers, Qwen validation | 20 | 1e-5 | build 1, `t05_check` |
| (v) | untold join, four encoders | all seeds | 1e-3 | build 1 |
| (vi) | r11's read-out rule, 7 arms | 7 arms | 1e-9 | `t09/stage0/gate_vi.json` |

`a_r15` failed first on one cell (SC4 genre teacher, 1.3e-5): r17's replay normalised r15's rows a second time, which
flipped one row in k-means. r17's wiring slip, fixed in 86c3cc0; the re-run passes 30 of 30. The real cells were
never affected (`ns_clu_cells` feeds rows as `ns_data` returns them). The stage 0 record should say so.

**Smoke of the runner (ticket 10):** gates iii, v, iv, ii_crl, ii_percept, i_a_partitions, i_b and i_c ran through
`ns_run.py`; stage 1, stage 2 and `outcomes` ran end to end on CLIP L/14 plus EmoSet Qwen at smoke draws (76 cells,
412 listed). **Not yet through the runner:** gates i_a_r14, i_a_r15, i_d_emoset, i_d_class_count, ii_scorer, vi, and
all of stage 3. **Ticket 11's own smoke was partial:** sweep (emotion, K 5 and 9), figure and per-class ran; told-untold,
`r0` and `tables` ran on stub inputs; the retrieval half of `qwen-crl` stopped on a stub picks record (the
implementer's slip, not rerun). SC7, SC8 on other encoders and MMArt SC6 have not run on real data (ticket 08 tested
them on synthetic data; ticket 10's smoke ran ArtELingo CLIP L/14 retrieval).

**Costs for the run (CPU h, measured at load 30 to 60, so high):** stage 0 about 1.3; stage 1 ranges 1.2; picks 4
(untold 24 min per arm, 8 arms; matched 3 to 4 min, 7 arms; CRL 1 min); stage 2 clustering 5 (13 steps), retrieval
1.6 (12 min per step is a guess); stage 3 sweep about 7 for the eight painting items (ticket 11's estimate; ticket
10's plan says 25 min per sweep, a guess that is too low: use 45 to 60 min), told-untold 12 min per encoder and set,
per-class 1.5 min, `qwen-crl` 15 min. Without C13: about 14 h by ticket 10's count, more with ticket 11's sweep
estimate. C13 is the largest part (spec §9).

**Machine at 23:17:** run mode night; GPU check passes (driver 595.99.02), though r17 needs no GPU. Queue: GPU idle,
CPU 1 slot free (r19 gates hold three). Load 30. Codex: "room" (5h window resets 23:34, counted as fresh).

## 4. Open points for the next step

None for the user. Agent steps, in order:

1. **Smoke what the runner has not run** (Sonnet `medium` worker, or Codex plumbing job): stage 3 through `ns_run.py
   diag` with real smoke inputs (a real picks record from ticket 10's smoke, cells present, so `qwen-crl` retrieval,
   `told-untold`, `r0` and `tables` run on real data), and one gate wrapper of each untested kind at smoke draws
   (`gate i_a_r14`, `gate ii_scorer`), through `res-queue cpu --no-wait --short`. Fix `ns_run_cpu.sh`'s stage 3
   estimate. Also read ticket 11's diff (controller check for plumbing, 444 lines, not yet read in detail) and one
   real SC7 or SC8 smoke cell on another encoder and MMArt, if cheap.
2. **Final whole-branch review** (a `worker`, Opus `xhigh`): the brief names the spec, the index, `matt-chain.md`'s
   final-review section, hard rule 6 (`spec-templates.md`) and the constitution; it re-derives the load-bearing numbers.
   Then one fix wave (Codex if `codex-usage --room` says room, else Sonnet `high`) and a scoped re-review (Sonnet
   `high`). Give it the review notes kept in the controller notes.
3. **Merge:** main has moved, so merge main into `r17-new-scores` first, re-run the round's tests through
   `res-queue cpu` (`pytest -q src/test/20261103_new_scores`, about 2 minutes), then move main as the first build
   handoff §5 says, and push.
4. **Run handoff** (`2026-11-03-r17-run-handoff.md`) and open the run chat with `loop-next --label "r17 run"`
   (unattended). It holds the ticket list, the run plan (`ns_run_cpu.sh` stages in order, with `ns_run.py
   stage1-records` and the commit of `records/stage1_*.json` and `records/status.json` before stage 2), the end-time
   estimate from `res-queue status`, and what C13 must write (`records/c13.json` with `pass: true` and
   `outcomes_sha256` of `records/outcomes.json`). Plan stage 0 and stage 1 over several slots: the plan runs them in one,
   and each untold arm is its own step (`--est 30m`). If the estimated end passes 11:00, weigh it as
   `shared-resources.md` says and record why (CPU use beyond 11:00 is a strong preference, not a hard limit); §3.6's
   rule cuts the photo sets of the sweep first.
5. **Delete nothing:** worktrees and ticket branches go on `.scratch/pending_deletions.md` at the end (list in §6).

## 5. Code and pitfalls

- **Row forms:** clustering rows are never normalised again: `ns_data.cstar(unit=False)`, `plain`, `teacher` and
  `ns_photo` rows as returned. Retrieval: MMArt plain and teacher get `mt_ret.unit`. A replay feeds each round its own
  form: r15's K = 20 route feeds `mt_data.features` directly, while r15's class-count diagnostic (`mt_diag.py:220`)
  normalised frozen rows twice, so `replay_d_class_count` keeps `_unit32` and `replay_a_r15` must not have it.
- **`ns_clu.paired_cell`** opens its own fork pool: one plain process per queue step, never inside a pool worker. It
  refuses cluster ids >= k.
- **`ns_diag`** imports private names (`ns_clu_cells._cand_parts`, `_comp_parts`, `ns_ret._pick`, `_unit`): the frozen
  interface of §3.7 does not cover them. The teacher is swept only in the Qwen item and CRL only in the CLIP L/14
  item, so a set's figure needs those items.
- **Gate status:** at smoke draws a gate counts as ok when its values agree (it cannot `pass`); a real run needs
  `pass`. `outcomes` refuses unless `records/status.json` equals git HEAD. Stage 2 reads stage 1 only through
  `ns_paths.read_committed_record`.
- **Queue discipline:** implementers ran some smokes outside `res-queue` (ticket 11) and load reached 60: repeat the
  rule in every brief. Only ~8 threads per slot.
- **Notes kept for the report and gate records** (controller notes): SC7 and SC8 sentences open with "On <set>," (not
  in §6's template; needed to tell cells apart); `meets_bar` against the strict `bar_met`; the guard decided CRL's
  MMArt pick; r15's diagnostic plain rows differ in form from SC1's; 06 review notes 1, 2, 4.
- **Timestamps:** take them from `TZ=Europe/Amsterdam date`, never from memory.

## 6. State at handoff

- Running: nothing of this chat (no subagents, no queue job, no Codex job).
- Uncommitted: nothing of this chat's. The `.scratch` notes, tickets and smoke results are outside git by design.
- On disk over 1 GB: nothing new (`.scratch/r17-new-scores/smoke` is 62 MB).
- Scratch and worktrees for `.scratch/pending_deletions.md` at the end of the build: worktrees
  `.claude/worktrees/agent-*` of tickets 01 to 11 (branches `r17-t01-metrics`, `r17-t02-outcomes`,
  `r17-t03-painting-data`, `r17-t04-photo-data`, `r17-t05-mappers`, `r17-t06-clustering-core`,
  `r17-t07-clustering-cells`, `r17-t08-retrieval`, `r17-t09-picks`, `r17-t10-runner`, `r17-t11-diagnostics`), and the
  smoke folders in this session's scratchpad and `/tmp/claude-0/t08/`, `/tmp/claude-0/smk/`.
