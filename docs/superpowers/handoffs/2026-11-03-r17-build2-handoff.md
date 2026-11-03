# Handoff: r17 build 2: review tickets 01, 02, 03, then build 04, 06 to 11, smoke, final review, merge, run handoff

## Brief
- Done: tickets 01 (metrics), 02 (outcome code), 03 (painting data), 05 (CPU mappers) built; 05 reviewed and merged.
- State: gates (iv) and (v) and the class counts pass on real data; new k-means equals the older routes exactly.
- Next: Opus `high` reviews of 01, 02, 03 and their merges; implementers for 04, 06, 08, then 09, 07, 10, 11.
- Blocked: nothing. Codex has no room until 23:34.

> Written 2026-11-03 21:10 by tab "r17 build" (context meter past 200k). Loop step reached: step 7 (build), tickets
> 01 to 03 and 05 built. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r17 spec
> approved".

## 1. Read in this order

1. The controller notes, `/project/CoSiR/.scratch/r17-new-scores/notes/controller_notes.md`: what ran, each
   hand-back's key facts, worktrees and branches.
2. The ticket index, `/project/CoSiR/.scratch/r17-new-scores/issues/00-index.md`: tickets, blocking edges, shared
   interfaces, coverage line, and the controller decisions (binding; several came from the code maps and hand-backs).
3. The spec from §1 on: [2026-11-03-new-scores-design.md](../specs/2026-11-03-new-scores-design.md) (rule: §6 with
   §3.4 and §3.5; §3.7 is the frozen scorer interface).
4. `~/.claude/references/matt-chain.md` (the chain), then `~/.claude/references/codex-jobs.md` before a Codex job.
5. The earlier build handoff, [2026-11-03-r17-build-handoff.md](2026-11-03-r17-build-handoff.md), §5 (pitfalls:
   queue, threads, git) and §3 (timing).

Skills: `to-tickets` and `implement-spec` are already applied (tickets written); read `implement-spec`'s SKILL.md
only for the merge and cleanup steps.

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D20), approved 2026-11-03 20:45 from its brief; no container restart (user).

## 3. Where things stand

- **Integration branch** `r17-new-scores` (worktree `/project/CoSiR/.claude/worktrees/r17-int`), at f4b5a0d:
  `ns_paths.py` (controller-owned constants) and ticket 05 (merged after an Opus `high` review, approve with notes;
  note 1 applied by the controller). Pushed.
- **Built, not yet reviewed** (local branches, worktrees under `/project/CoSiR/.claude/worktrees/`):
  - 01 `r17-t01-metrics` ea7da51 (worktree `agent-a6dabaef7aa4e67b8`): `ns_metrics`, 160 tests; partitions equal
    `ct_score`, `pav_score`, `mt_diag.kmeans_k` (8 threads) and `bs_emoset_score.partitions_k` (1 thread) on 2,000
    real rows; gate (iii) metrics pass.
  - 02 `r17-t02-outcomes` f7ac16e (worktree `agent-a1efeb6dccb05c327`): `ns_outcome`, 112 tests; 412 cells; one test
    per row of §3.5's gate table.
  - 03 `r17-t03-painting-data` 6513d2f (worktree `agent-aac4058318080124e`; holds 05's commits): `ns_data`, 38
    tests; gate (v) ok on four encoders; class counts equal §3.3; CRL equals r11's form.
- **Not started:** 04 (photo data), 06 (clustering core, after 01), 07 (clustering cells, after 03, 04, 06), 08
  (retrieval, after 03), 09 (picks, after 03, 05), 10 (runner), 11 (diagnostics).
- **Real files:** ticket 05's validation probabilities in `/project/CoSiR/.scratch/r17-new-scores/smoke/t05` and
  `smoke/t05_check` (20 npz each, small). The real run writes them again under the outputs folder.
- **Timing:** about 1 h ahead of the first estimate (build began 20:48). The run must start by about 01:00 for the CPU
  to be free before 11:00 (spec §9).
- **Machine at 21:10:** run mode night; `res-queue status`: GPU idle, CPU 4 slots free; GPU check passes (driver
  595.99.02), though r17 needs no GPU. Codex: 5h 37% left, resets 23:34; recheck then (the C13 code and the
  final-review fix wave may go to Codex).

## 4. Open points for the next step

None for the user. Agent choices to confirm in the reviews: 02 counts a WIN-bar teacher cell at LEVEL as meeting its
bar with the "level with or better than" sentence (spec §6, per cell); 03 was mutation-tested after writing rather
than strictly test-first (eight sabotages caught).

## 5. Code and pitfalls

- **Dispatch:** reviews with the common brief `notes/reviewer_brief.md` (Opus `high`); implementers with
  `notes/implementer_brief.md` (Sonnet `high`, `isolation: "worktree"`, at most three at once). Implementers of 07 and
  08 merge the reviewed 01 and 03 first (or the unreviewed branches, said in the brief).
- **Merges:** in the integration worktree; `git merge --no-edit <branch>`; push `r17-new-scores` after each merge.
  Move main only at the end: `git merge-base --is-ancestor main HEAD && git update-ref refs/heads/main HEAD`
  (main may have moved: other chats commit on it; then merge main into `r17-new-scores` first).
- **Runner notes for ticket 10** are in its ticket file (gate exceptions become `failed`; gate output shapes).
- **C13** (§3.4): separate code written in the run chat (Codex if room, else a Claude worker), as in r15.
- **Unattended:** delete nothing; worktrees and ticket branches go on `.scratch/pending_deletions.md` at the end.

## 6. State at handoff

- Running: nothing of this chat (no subagents, no queue job, no Codex job).
- Uncommitted: nothing of this chat's (the `.scratch` notes and tickets are outside git by design).
- On disk over 1 GB: nothing new.
