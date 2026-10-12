# Handoff: r8 build 2: create the integration branch and build the 13 tickets with implement-spec

## Brief
- Done: the gate decision rule for both tracks, written, reviewed by a fresh Opus reviewer (1 blocker, 6 major, 11
  minor, all fixed), re-reviewed and committed; 13 tickets written.
- State: no code yet. Outside code and weights checked: FCMI, ALADIN (ViT weights only), DINOv2-L, SigLIP 2 all load.
- Next: branch `r8-grouping`, then implementers for tickets 01, 05 and 10 first.
- Blocked: nothing.

> Written 2026-10-12 20:35 by the `r8 build` chat (cut at 200k tokens of context). Loop step reached: step 7, the rule
> committed (D33), before the first ticket. Direction log entry: [2026-10-12 · Round 7 design L](../direction_log.md).

## 1. Read in this order

1. `.scratch/r8-grouping/issues/00-index.md`: the ticket table, blocking edges, the shared interfaces (the
   `r8_common` API, space, grouping, posterior and arrays files) and the coverage line (hard rule 6, checked).
2. `.scratch/r8-grouping/notes/ticket_rules.md`: the rules every implementer brief points to (lean build, git,
   no deletions, 4-minute waits).
3. The committed rule `src/test/20261129_r8_grouping/DECISION_RULE.md` (governs where it differs from the specs) and
   its review trail in `rule_check/` (`opus_rule_check.md`, `fixes.md`, `rereview.md`).
4. The specs only as the tickets cite them: A `docs/superpowers/specs/2026-10-12-r8-learned-grouping-design.md`, B
   `docs/superpowers/specs/2026-10-12-r8-encoder-screen-design.md`.
5. `~/.claude/references/matt-chain.md` (implement-spec with our changes) and `~/.claude/references/codex-jobs.md`.
6. Outside-code facts: `.scratch/r8-grouping/notes/probe.md`.

## 2. Decided by the user, not to be reopened

As in the r8 build handoff (`2026-10-12-r8-build-handoff.md` §2) and both specs' §4 rows marked "user".

## 3. Where things stand

- Rule committed on main before any code (C7): commit 9fb9d68, SHA-256 50eb65ad…d50e (415 lines). Header records the
  review times; reviewer files in `rule_check/`.
- Agent defaults of this chat, for the report: FCMI keeps its own faiss k-means on a CPU index (faiss-cpu 1.15.1
  installed into the CoSiR env); FCMI and the encoders run on the local GPU under the lock; ALADIN's only public
  weights are the ViT variant from StyleBabel (`aladinVIT_bamfg_64.47.pt`), used as B:ED5's README weights and flagged
  for the report; the rule's other agent defaults are marked in it.
- Ticket plan (agent default): tickets 03 to 08 and 10 to 12 may start beside 01, coding against the `r8_common` API in
  the index; at most three implementers at once. Suggested first wave: 01 (Claude `worker`, sonnet high; critical
  path), 05 and 10 (Codex `gpt-6.1-sol` high in controller-made worktrees, while `codex-usage` shows room; it showed
  100% of the five-hour window and 85% weekly at 20:00). Then 02 (stage 0, after 01), 06, 07, 08, 11 (C13
  re-derivation suits Codex: separate code), 03, 04, 09, 12, and 13 last.
- Verdict-code tickets (01, 02, 05, 06, 07, 08, 10, 11) get an Opus `high` reviewer before merge; the rest are checked
  by the controller reading the diff and by ticket 13's smoke run.
- Stage 0 (SC1) is CPU, about 10 minutes: run it in the build after ticket 02 merges (it reads only seed-42 dev data
  and R7's stored arrays). GPU smoke runs (views, FCMI, encoders) also in the build, under the lock; the full GPU and
  CPU stages belong to the run chat (ticket 13's queue scripts).
- After the build: the final whole-branch review (Opus `xhigh`), one fix wave, a scoped re-review, merge to main,
  push, then the run chat with `loop-next --unattended`. HELD_RULE.md and the held-read code come after the gate
  verdict (A:D33), not in this build.

## 4. Open points for the next step

- None for the user. Night mode (`run-mode`): no questions.

## 5. Code and pitfalls

- Painting index p is the `g_st` code (= `rb_halves.npz` `paintings_sorted`), not the painting-id string order; the
  review proved the two differ (rule M1). Every painting-ordered array uses p.
- `r8_common` hard-codes `MAIN = /project/CoSiR` as R7 does, so results land in the main checkout from any worktree.
- Codex worktrees: create them outside the repo (for example `/project/CoSiR-wt/r8-NN`, branch `r8-NN` from
  `r8-grouping`); the root `data`/`res` may be symlinks in worktrees (`.gitignore` l.158). Codex launches were refused
  intermittently by the auto-mode check in r7: take the ticket back to a Claude `worker` when that happens.
- FCMI's per-epoch faiss k-means on CPU (niter 300, nredo 10, at most 17,000 points of 17 values, 300 epochs) is
  unmeasured against D21's 60 minutes: read the time per epoch from ticket 06's GPU wiring smoke before stage 1
  (re-review L3).
- FCMI needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`; ALADIN needs `PYTHONPATH=/data/SSD2/third_party/pydeps`.
- Round 7's folder and `results/` are read only and write-once.
- Keep `subagent-watch --watch` running while subagents run (it exits when none is running; restart it).

## 6. State at handoff

- Running: nothing (the main chat "CoSiR main" in tab `w12:t2`, reopened 19:57).
- Uncommitted: nothing in git; `.scratch/r8-grouping/` (tickets, notes) is outside git by design.
- On disk over 1 GB: nothing new. New small items: `/data/SSD2/third_party/{FCMI,ALADIN,ALADIN_weights,pydeps}`
  (about 0.7 GB with the 320 MB zip listed in `.scratch/pending_deletions.md`), DINOv2-L in the HF cache (1.2 GB).
