# Handoff: r21 build 2: review tickets 02 and 05, build 03, 04, 06, final review, merge, then open the r21 run chat

## Brief
- Done: ticket 01 (DEC math) merged; 02 (trainer) and 05 (scoring) built, not yet reviewed; 05 reproduces the saved validation AMI exactly.
- State: DEC off matches c*'s frozen loop bit for bit on a synthetic cohort; q is almost flat on unit head outputs, so λ ∈ {1, 10} may be weak.
- Next: Opus reviews of 02 and 05, then tickets 03, 04, 06, the final review, merge, the run handoff.
- Blocked: nothing; Codex had no room for implementer jobs (Claude workers build).

> Written 2026-11-11 06:50 by the r21 build chat (cut at 200k tokens). Loop step reached: step 7, mid-build.
> Direction log entry: [2026-11-11 · r21 decide](../direction_log.md).

## 1. Read in this order

1. The first build handoff, sections 2 to 5 (decisions, code facts, pitfalls; still true):
   [2026-11-11-r21-build-handoff.md](2026-11-11-r21-build-handoff.md). The spec governs:
   [2026-11-11-r21-dec-heads-design.md](../specs/2026-11-11-r21-dec-heads-design.md).
2. The tickets: `.scratch/r21-dec-heads/issues/00-index.md` (shared interfaces, output tree, coverage line) and
   `01` to `06`.
3. The decision lines so far: `src/test/20261111_r21_dec_heads/decisions.md` on branch `r21-dec-heads`.
4. `~/.claude/references/matt-chain.md` and `~/.claude/references/loop-chats.md` ("Unattended chats").

Skills: none by name; the chain is in `matt-chain.md`.

## 2. Decided by the user, not to be reopened

As the first build handoff's section 2 (spec §4, D1 to D7). Run mode is night: no questions to the user.

## 3. Where things stand

| # | Ticket | State | Branch, worktree |
|---|--------|-------|------------------|
| 01 | DEC math (`r21_dec.py`) | done: Opus `high` review, fix round, merged | in `r21-dec-heads` |
| 02 | Trainer (`r21_train.py`) | built, 46 tests pass, merged the integration tip; **needs its Opus `high` review** | `r21/t02` d5130472, `/project/CoSiR-r21-t02` |
| 05 | Scoring (`r21_score.py`, `r21_arms.py`) | built, 47 tests pass, merged the integration tip; validation scorer reproduces `stage1_ranges.json` (c* seed 0: 0.100984 / 0.420558 / 0.529839, diff 0.0); partial gate 0a exact (emotion 5-seed AMI mean 0.1024127745 and teacher 0.088875, diff 0.0; emotion seed 0 mAP 0.263832, diff 0.0); **needs its Opus `high` review and the full gate 0a run** | `r21/t05` a282ae62, `/project/CoSiR-r21-t05` |
| 03 | Runner train + gate 0b | not started (blocked by 02) | create from `r21-dec-heads` after 02 merges |
| 04 | Runner held + queue scripts | not started (blocked by 03) | – |
| 06 | Stage 2 table + diagnostics | not started (blocked by 05) | – |

- Integration branch `r21-dec-heads` (worktree `/project/CoSiR-r21`), pushed to origin; it holds 01 and the
  decision lines.
- Ticket 02's interface for ticket 03: `dec_data[aspect]` has `'k'` (the class count) always; the teacher form adds
  `'feats'` (float32 [n_fit, 2048], unit rows, in `graphs[aspect]['fit_node_ids']` order) and `'centres'` (the cached
  teacher centres). History entries carry a `dec` dict; the last carries `stop_reason`. `W < max_epochs` is asserted,
  so a 2-epoch smoke allows only W = 1.
- Ticket 01 review finding kept as a flag (decision line, "could change a claim: yes"): Student-t (α = 1) on unit
  head outputs gives an almost flat q. On saved c* seed 0 validation exports: mean max q 0.133 / 0.061 / 0.138 against
  1/k 0.111 / 0.037 / 0.100; KL(P‖q) 0.006 / 0.025 / 0.015 (emotion, style, genre). The spec's grid stays (night mode;
  more λ values would be runs the spec does not list, constitution row 6). **Put this at the top of the run handoff**
  as an item for the user: a cheap amendment before stage 1 is λ ∈ {100, 1000}, which needs no code.
- Ticket 01's CUDA device test was skipped (GPU busy). Ticket 03's GPU smoke is the first CUDA run of the DEC code.
- **Two mismatches between 02 and 05, to fix in 05's review round:** (a) 05 counts "DEC epochs" as history entries
  that carry a `dec` dict, but 02 also writes `dec` on warm-up entries (`phase: 'warmup'`); count `phase == 'dec'`.
  (b) 05 reads the stop reason from `run__s<seed>.json` at `heads.<aspect>.stop_reason`; 02 puts `stop_reason` in the
  last history entry of each validation sidecar. Ticket 03 writes the run record: make it hold
  `heads.<aspect>.stop_reason` (and the DEC start epoch), copied from the sidecars.
- Ticket 03 imports `parse_arm` / `arm_name` from `r21_arms.py` (ticket 05) and re-exports them.
- 05's loader expects r21 validation sidecars to keep c*'s `config` fields (`mode`, `smoke`, `aspect`, `seed`,
  `graph_row`, `output_rows`, `names_sha256.dev`) and `output_sha256` as the file hash; 03's smoke checks this.
- Full gate 0a (about 10 min of CPU), from `/project/CoSiR-r21-t05` or the integration worktree after the merge:
  `res-queue cpu --label r21:gate0a -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261111_r21_dec_heads/r21_score.py gate0a` (writes
  `/data/SSD2/cosir_r21/20261111_dec_heads/scores/gate0a.json`). The real `sheet` also goes through `res-queue cpu`
  (about 50 s per arm).

## 4. Open points for the next step

None for the user. For the build: review 02 and 05 (verdict code: an Opus `high` worker each, findings to the
original implementer or a fresh fixer), merge, then 03 (Sonnet `high`: the chain is subtle and it carries gate 0b),
06 (Sonnet `high`), 04 (Sonnet `medium`), reviews of 03's gate 0b script and of 06, then the final whole-branch review
(Opus `xhigh`), one fix wave, a scoped re-review (Sonnet `high`), merge into main, push.

## 5. Code and pitfalls

- Codex: `codex-usage --room implementer` said no room at 07:00 (five-hour window resets 10:09); check again for
  03, 04 and 06.
- Gate 0b runs on CPU through `res-queue cpu`; gate 0a's full run (ticket 05) too. Record both in the round log
  `src/test/20261111_r21_dec_heads/20261111_r21_dec_heads_log.md`.
- Workers' scratch files in the first build chat's scratchpad
  (`/tmp/claude-0/-project-CoSiR/c6e18fab-59ed-4012-8270-0b34e3297dd2/scratchpad/`: `dt.py`, `mut.py`, `flat.py`,
  `diff.txt`, `t05smoke/`) can go at the wrap-up; nothing over 1 GB. The index's smoke folder points into that
  scratchpad; the next chat may use its own scratchpad instead (update the index line).
- `/data/SSD2/cosir_r21/20261111_dec_heads/scores/` holds `gate0a_partial.json` and `val/cstar__s0.json` (a valid
  cache) from ticket 05's checks.
- At the end of the build: worktrees `/project/CoSiR-r21-t01`, `-t02`, `-t05` and later ones, and their branches, go
  on `.scratch/pending_deletions.md`, not deleted (night mode).

## 6. State at handoff

- Running: nothing of this chat (the subagent watch was stopped).
- Uncommitted: nothing of r21 (ticket branches hold their commits locally; `r21-dec-heads` is pushed).
- On disk over 1 GB: nothing.
