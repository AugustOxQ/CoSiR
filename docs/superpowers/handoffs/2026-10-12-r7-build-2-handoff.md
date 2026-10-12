# Handoff: r7 build 2: review tickets 09 and 10, then build the rest of the r7 tickets, final review, then cut to r7 run

## Brief
- Done: 16 tickets written from the rule; 01 (foundation) merged; 09 and 10 (verdict statistics) built, tests first.
- State: integration branch `r7-design-l`; no stage 1, 2 or 3 number computed. Codex launches are sometimes refused by the auto-mode check.
- Next: Opus reviews of 09 and 10, merge, then the frontier 02, 03, 05, 07, 14, 15.
- Blocked: nothing. Night mode.

> Written 2026-10-12 03:37 by the `r7 build` chat (cut at 200k context). Loop step reached: step 7, build in progress.
> Direction log: this round's entry is written at its wrap-up.

## 1. Read in this order

1. `.scratch/r7-design-l/issues/00-index.md`: the ticket table, shared interfaces, verdict JSON keys, coverage line
   (checked; complete). The tickets are `issues/NN-*.md`; each `Status:` line is current.
2. `.scratch/r7-design-l/notes/build_log.md`: what happened so far, the routing of each ticket.
3. `.scratch/r7-design-l/notes/implementer_brief.md`: what every implementer and reviewer gets.
4. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md` (the process), then the rule
   `src/test/20261126_design_l_style_source/DECISION_RULE.md` as each ticket needs it. The code maps in `notes/`
   (`code_map_groupings.md`, `code_map_scoring.md`) are for implementers; read a section only when a hand-back needs it.

## 2. Decided by the user, not to be reopened

- Everything of the previous handoff's §2 (`2026-10-12-r7-build-handoff.md`): spec D1 to D27 and the rule, approved
  2026-10-12 02:34; build and run unattended (night mode).

## 3. Where things stand

| Ticket | State |
|---|---|
| 01 foundation (`r7_common`) | merged into `r7-design-l` (25889b3); 41 of 41 §10 SHA-256s match the rule |
| 09 seed-42 statistics (`r7_gate`) | built cd9cc25 on `r7t/09-gate-stats`, 80 tests; needs its Opus review, then merge |
| 10 fresh-seed statistics (`r7_chain`) | built 4536954 on `r7t/10-chain-stats` (Codex), 61 tests; needs its Opus review, then merge. Its F2/F3 "not tested" form keeps point and CI under `not_tested`; check it matches the index's test keys |
| 02 to 08, 11 to 16 | not started; worktrees not created |

- Routing so far: 01 and 09 on Claude workers (Codex launch refused), 10 on Codex. Rule SHA used by `r7_common`: the
  rule file at HEAD (204a0ac5…), the approved text after its commit refs were filled in (0fd066b).
- 09's JSON choices are recorded in the index (alpha and random-control flag keyed by family). Note for 09's review:
  it imports `aspect_metrics` by absolute path with its own SHA check, written before `r7_common` existed; swap to
  `r7_common.load` at merge if the reviewer agrees.

## 4. Open points for the build (agent defaults; nothing for the user)

- **Order.** Critical path: 07 → 08 → 11 (stage 0) and 05 → 06. Then 02, 03, 04, 12, 13, 14, 15, 16. At most three
  implementers at once. Verdict-code tickets (07, 08, 09, 10, 14, 15) get an Opus `high` reviewer before merge.
- **Codex first** (`codex-usage`: 5h window resets 07:40). Launch exactly as `codex-jobs.md` says, from
  `.scratch/r7-design-l/run/` (`mkbrief.sh NN slug` makes the worktree `/project/CoSiR-wt/r7-NN` and the brief
  `codex_NN.txt`). If the auto-mode check refuses a launch, give the same brief to a `worker` in that worktree (no
  retry of the same command). Add a ledger row per Codex job (`/project/claude-config/codex/jobs.md`).
- **Controller runs** (not subagents): the token extraction on the GPU after 02 merges (`--check-only` first; the
  cosine ≥ 0.9999 check is against a cache whose preprocessing is not recorded, so a failure is possible: then the
  generic family stops, as spec §8 says, and it goes at the top of the next handoff); stage 0 after 11; the chain
  smoke after 16; unit tests of verdict code; the final whole-branch review (Opus `xhigh`), one fix wave, a Sonnet
  re-review; merge into main and push.
- **Interfaces drift.** When an implementer settles something the index left open, record it in the index before the
  next dependent ticket starts (as done for 09).

## 5. Code and pitfalls

- Merges: in the integration worktree `/project/CoSiR-wt/r7-int` (branch `r7-design-l`), `git merge --no-ff`.
- Round folders' `results/` are gitignored; r7 code reads them and writes its own through `r7_common` paths under
  `/project/CoSiR` (works from any worktree).
- Key pitfalls from the code maps: local vs global rows; `K.thresholds` returns `(taus, n)`; R3 code is hard-wired to
  three groupings; `fusion_pass` defaults to 896 cells (`n_kappa=1`); `run_baselines.py --smoke` overwrites silently
  (reuse existing smoke episodes); R6 modules refuse seeds 55 to 57; `r6_stats.holm` hard-codes 40; ClustPy is not
  installed; the CLIP cache is per row, not unit norm.
- The build computes no stage 1, 2 or 3 number; G1 to G4 may run.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (ticket branches committed; `.scratch/` is outside git).
- Worktrees: `/project/CoSiR-wt/r7-int`, `r7-01`, `r7-09`, `r7-10` (listed on `.scratch/pending_deletions.md` at the
  end of the build, not removed).
- On disk over 1 GB: `res/cluster_jobs/` 3.1 GB (round 6, keep, ledgered).
