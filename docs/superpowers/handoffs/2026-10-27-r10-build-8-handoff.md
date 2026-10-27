# Handoff: go on with the recipe sweep build (tickets 05 to 15, the gates, the final review), then cut to the run chat

## Brief
- Done: tickets 01 to 04 built, reviewed by Opus `high` workers, fixed and merged on `r10-recipe-sweep` (189 tests).
- State: gates pass: V1a (G equal to 12-27's G bit for bit, seeds 0 and 1; scorer 3 of 3 within 1e-6), k11 and ablation graph checks. No new number on P or H.
- Next: tickets 05 to 15 per the index; Codex has room again after 06:22.
- Blocked: nothing. Night mode: do not wait for the user. The build will likely end after 10:30.

> Written 2026-10-27 04:43 by "r10 build 7" (cut at the context meter's 200k note). Loop step reached: step 7, the
> build, tickets 01 to 04 of 15 merged. Direction log entry: [2026-10-27](../direction_log.md).

## 1. Read in this order

1. The ticket index `.scratch/r10-recipe-sweep/issues/00-index.md` (outside git): the ticket table, the coverage line and
   the shared interfaces; then the ticket files 05 to 15 beside it.
2. The implementer brief every implementer gets: `.scratch/r10-recipe-sweep/notes/implementer_brief.md`.
3. The spec from §1 (`docs/superpowers/specs/2026-10-27-recipe-sweep-design.md`) and the binding rule
   `src/test/20261027_recipe_sweep/DECISION_RULE.md` @ 35fcf68.
4. `~/.claude/references/matt-chain.md`, and `~/.claude/references/codex-jobs.md` before any Codex job.
5. The build 7 handoff `2026-10-27-r10-build-7-handoff.md` §3 to §5 (schedule, open points, pitfalls): still true.

Skills: `cluster-run` for ticket 14's node check and smoke on node402.

## 2. Decided by the user, not to be reopened

- Decisions A, B, C, the grill and the brief: as in the build 7 handoff §2 (approved 2026-10-27 03:48; no container
  restart; night mode since 03:47).

## 3. Where things stand

| Ticket | State |
|---|---|
| 01 paths, number sheet | merged (controller; 54 tests read the sheet from the spec at the rule commit) |
| 02 knob layer `rs_knobs`, trainer `rs_train` | merged after review (2 must-fix test gaps, 2 should-fix; all applied) |
| 03 graph variants `rs_graphs` | merged after review (1 must-fix: a gate could vouch for a store built by older code; fixed) |
| 04 stages 1 and 2 `rs_sweep` | merged after review (no must-fix; the reviewer's own re-derivation agreed on 6,300 random tables) |
| 05 to 15 | not started; frontier now: 05, 06, 07, 08, 11, 12, 13, 14 (14 needs 02, 03, 04: all merged) |

Gates in `outputs/gates/` (real outputs, clean code, 04:20 to 04:41):
- `v1a_train.json`: pass. G by `rs_train` equals `RG/outputs/train/GS-0.02/s{0,1}` bit for bit (exports, old-5000
  exports, best states).
- `v1a_scorer.json`: pass. G's seed-0 A on P equals 12-27's stage A within 1e-6: emotion 0.10548, style 0.39710,
  genre 0.50744.
- `k11_sample.json`: pass. The k = 20 rebuild equals the frozen sample graphs bit for bit.
- `ablation_graphs.json`: pass. `default_view` and `uncorrected` builds equal the D2 and aspect graph sheet graphs.
- Not run yet: `k11_full` (V4c; about 4 minutes CPU: `rs_graphs.py check k11-full`), V0a, V3a, V4a, N1, N2.
- A k11 = 10 smoke ran through the real graph store and the trainer (2 epochs, CPU, scratch): ok.

Gates bind code SHA-256s. A merge that changes a hashed file (the trainer's training files, `rs_graphs`'s modules,
the scorer's round files) makes the gate stale: rerun it after the last merge that touches those files.

## 4. Open points for the build (agent defaults, no user question)

a. Codex: no room before the 06:22 reset (49% of the five-hour window left; an implementer needs about 80%). After
   it, tickets 11 and 12 (the re-derivations) suit Codex best: their code must be independent of Claude's scorers.
b. The node route (ticket 14): the 10-20 claim test's route (inputs and the local env synced to an `INPUTS_REMOTE`
   folder, a bind-mounted fake root so the code runs at its local paths) is the default, not a new `DATA_MAP` line;
   it passed its node check and smoke on node408. Node402 has none of those inputs yet.
c. A missing `gates/k11_sample.json` makes stage 1 stop as "missing", not skip k11 (ticket 04's choice); the gate
   exists and passes, so this does not arise.

## 5. Code and pitfalls

- **Agent worktrees start at `main` (e83b622), not at the integration branch.** Each implementer resets onto
  `r10-recipe-sweep` first (the brief says so); check it in each hand-back.
- **The main checkout is on branch `r10-recipe-sweep`** (not `main`). Merge tickets there; merge into `main` only
  after the final review (matt-chain.md).
- Interfaces fixed by tickets 02 to 04 (also in the index): G₀′ is the folder `G-repeat`; V1a's replays are `G-v1a`
  under `outputs/regression/`; `run.json` carries `name`, `config`, `seed`, `status`, `cause` (None when ok),
  `failure_kind`, `rerun_of`, `machine`, `p_sha256`, `smoke`, `best_epoch`, `knobs`, `code_sha256`; a score's
  `source` is `12-27` (local G₀, the fallback) or the round's trainer (node402).
- `rs_graphs.sample_store(k, variant, *, s, outputs)` and `full_graphs` / `full_training_data` return what the frozen
  loaders return; `row_names(variant)` gives the rows the trainers must set for `uncorrected` and `default_view`
  (tickets 07 and 08). Graph checks build into `outputs/graphs/<scale>/check/` and refuse if it exists: to rerun,
  move it aside (never delete).
- `rs_knobs.patches` refuses non-G configurations at `scale='full'` (ticket 08 extends it) and any `remove` (ticket 07).
- Ticket 14: N3's scorer parity reads 12-27's seed-0 exports at `RG/outputs/train/GS-0.02/s0`, so the node inputs must
  hold them at that path.

## 6. State at handoff

- Running: nothing (every worker finished; no GPU lock, Codex job or cluster job of this chat).
- Uncommitted: nothing after the cut commit.
- On disk over 1 GB: `/tmp/claude-0/r10-rs-02/` (1.5 GB smoke scratch); `outputs/regression/` (about 1 GB, V1a's
  replays). Both on `.scratch/pending_deletions.md` with the other scratch and the three merged worktrees.
