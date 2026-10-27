# Handoff: go on with the recipe sweep build (fix waves for 06 and 08, review 07, finish 12, tickets 09 to 13 and 15, the gates, the final review), then cut to the run chat

## Brief
- Done: tickets 05 and 14 merged (286 tests pass); node402 has the round's inputs (27.4 GiB, verified equal); the k11 full-scale gate passes; V4a passes in scratch.
- State: 06 and 08 built and reviewed (numbers correct; test gaps and two code fixes to apply); 07 built, not reviewed; 12 half done.
- Next: fixers for 06 and 08, a review of 07, a worker to finish 12, then 09, 10, 11, 13, 15.
- Blocked: nothing. The build will end in the afternoon, later than the 10:30 planned.

> Written 2026-10-27 11:50 by "r10 build 8" (cut at the context meter's 200k note). Loop step reached: step 7, the
> build, 6 of 15 tickets merged (01 to 05, 14). Direction log entry: [2026-10-27](../direction_log.md).

## 1. Read in this order

1. `.scratch/r10-recipe-sweep/notes/build8_reviews.md` (outside git): the review findings of 06 and 08, the facts of
   07's and 12's hand-backs, and the decisions build 8 took. The fix briefs are written from it.
2. The ticket index `.scratch/r10-recipe-sweep/issues/00-index.md` (new line: c* is the field `pick` of
   `stage3/verdict.json`), then the ticket files 09, 10, 11, 13, 15.
3. The implementer brief `.scratch/r10-recipe-sweep/notes/implementer_brief.md` (commit lines now say build 8 and this
   session; change them to the new session) and the Codex-style brief of 12, `notes/codex_brief_12.txt`.
4. The spec (`docs/superpowers/specs/2026-10-27-recipe-sweep-design.md`) from §1 and the binding rule
   `src/test/20261027_recipe_sweep/DECISION_RULE.md`; `~/.claude/references/matt-chain.md`.
5. The build 7 handoff `2026-10-27-r10-build-7-handoff.md` §4 and §5 (still true) and the build 8 handoff
   `2026-10-27-r10-build-8-handoff.md` §3 to §5.

Skills: `cluster-run` (N1 and N2 on node402).

## 2. Decided by the user, not to be reopened

- Decisions A, B, C, the grill and the brief: as in the build 7 handoff §2 (approved 2026-10-27 03:48; no container
  restart; night mode since 03:47, still on at 11:42).

## 3. Where things stand

| Ticket | State | Where |
|---|---|---|
| 01 to 04 | merged (build 7) | `r10-recipe-sweep` |
| 05 stage 0 | merged after review (4 must-fix applied, every missed mutation now caught; 97 tests) | f76708a |
| 06 stage 3 | built, reviewed: 4 must-fix test gaps, 4 should-fix; not merged | `worktree-agent-a7ba174338be3ab0a` |
| 07 removals | built on 08's branch (86 tests, smokes ok); not reviewed | `worktree-agent-a678b36c5dff99884` |
| 08 full trainer | built, reviewed: 2 code fixes, 6 test gaps, 5 should-fix; CPU smoke ok; V4a in scratch: pass | `worktree-agent-af73341f10227714d` |
| 12 dev re-derivation | progress note: code and 39 tests, point self-check 57/57 equal; open items in the notes | `/project/CoSiR-wt-rs12`, `r10-rs-12` |
| 14 node route | merged; inputs synced and verified; manifest written | 7fe1ec3, 3ad016c |
| 09, 10, 11, 13, 15 | not started (09 is unblocked now; 10 after 06; 15 last) | |

Gates in `outputs/gates/`: v1a_train, v1a_scorer, k11_sample, ablation_graphs (build 7) and k11_full (V4c, pass,
04:50). Not yet in the real outputs: v0a (CPU then GPU part, commands in `rs_stage0.py` and the notes), v3a (after
06's merge), v4a_train (after 08's merge; about 2 min GPU), v4a_scorer (ticket 09), n1, n2 (node), v0c, v0d (run
chat). The sample k11 stores `aspect__k{10,30,40}` exist (built 04:50) and are on the node.

Suggested order (agent default): now, in parallel, a Sonnet `high` fixer for 08 (in its worktree), a Sonnet `high`
fixer for 06 (in its worktree), an Opus `high` review of 07, and a Sonnet `high` implementer for 09 (on
`r10-recipe-sweep`). Merge 08, then rebase 07 by merging `r10-recipe-sweep` into it; merge 06; then 10, 11, 12's
finish, 13; then the real gates (v0a, v3a, v4a), `cluster sync` and N1, N2; then 15 and the final review (Opus
`xhigh`). At most three implementers at once (fixers count).

## 4. Open points for the build (agent defaults, no user question)

a. Codex: `codex e ...` was refused by the auto-mode classifier at 08:5x (reason "Create Unsafe Agents"), although
   the 5-hour window had full room. Per `codex-jobs.md` the job went to Claude; do not retry. Note it for the user in
   the final reply (the allow rule may need a look).
b. Ticket 11 independence (C13): give it the same rules as 12's brief (never read `rs_stage0.py`, `rs_stage4.py`,
   `rs_ablation.py` or their tests, nor other worktrees).
c. 07's `shared_trunk` reading is decided (notes): one cohort, one trunk per aspect. The run notes (ticket 15) state it.
d. The node: `cluster sync` needs a commit with "cluster run" in its subject (3ad016c has it; a later code commit for
   the node needs its own). N1: `cluster launch --node node402 -- bash scripts/run_rs_node.sh check`; N2: `... smoke`.
   The k11 gate on the node is stamped with `rs_graphs`' code hash: if 07's merge changes `rs_graphs.py`, rerun
   `rs_graphs.py check k11-sample` (move `outputs/graphs/sample/check/` aside first) and resync before N1.
e. V0a binds `rs_train.py` and `rs_sheet.py` among its modules (05's fix): run it after the last merge that touches
   them. Never rewrite `gates/v1a_train.json` after seed 4 trains (V0d compares mtimes).

## 5. Code and pitfalls

- Agent worktrees start at `main` (e83b622): each implementer resets onto `r10-recipe-sweep` (07 was based on 08's
  branch on purpose).
- `.claude/20261027_log.md` was added on `r10-recipe-sweep` (3ad016c) and on 08's branch: the merge of 08 gives an
  add/add conflict there; keep both entries. 07's edits still need their entry.
- The cluster CLI is `~/.claude/skills/cluster-run/cluster`; a plain `cluster` on PATH is an unrelated conda binary.
- Gates bind code SHA-256s; rerun a gate after the last merge that touches its hashed files.

## 6. State at handoff

- Running: nothing (every subagent handed back; no GPU lock, cluster job or Codex job of this chat).
- Uncommitted: nothing after the cut commit (the `.scratch/` notes and ticket status lines are outside git).
- On disk over 1 GB: `/tmp/claude-0/r10-rs-14/` (about 2 GB, ticket 14's smoke and traces); node402
  `/local/wding/r10_rs/` (27.4 GiB, the round's inputs; keep until the round ends). Smaller scratch: `/tmp/claude-0/r10-rs-05`,
  `-06`, `-07`, `-08`, `-08-v4a`, `-12`, `r10-review-05`, `-06`, `-08`, `r10-fix-05`. All on `.scratch/pending_deletions.md`.
