# Handoff: review ticket 08, build tickets 04, 10, 12, 13, then the final review and the run chat

## Brief
- Done: tickets 01, 02, 03, 05, 06, 07, 09, 11 built, reviewed (Opus high), fixed and merged (integration tip 3ae28fe).
- State: gates (iv) and (vii), r14 parts, replay r14 exactly (max difference 0.0; 414 and 920 items).
- Next: Opus high review of ticket 08 (built, not reviewed); then 04 and 10, then 12 and 13. Codex has room after 23:34.
- Blocked: nothing.

> Written 2026-11-03 23:05 by tab "r19 build 2" (cut at the 200k context meter). Loop step reached: step 7, build
> half done. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets "The user's reading of
> the experiment overview" and "Decided (A)".

## 1. Read in this order

1. The ticket index `/project/CoSiR/.scratch/r19-past-picks/issues/00-index.md` (ticket table, dispatch order,
   routing, shared interfaces, controller decisions; binding). Each ticket's `Status:` line says what is merged.
2. **`/project/CoSiR/.scratch/r19-past-picks/notes/controller_notes.md`**: binding notes from this chat's reviews for
   tickets 04, 09, 12 and every implementer (traps of ticket 01, gate (v) file names, the status schema `pp_outcome`
   reads, M5's node job and data map). Point every new implementer at it, as this chat did.
3. Ticket 08's hand-back: `notes/t08_handback.md` (give it to its reviewer).
4. The previous handoff, [2026-11-03-r19-build2-handoff.md](2026-11-03-r19-build2-handoff.md), §2 and §5 (decisions,
   pitfalls), and its pointers to `matt-chain.md` and `codex-jobs.md`.
5. The spec from §1 on: [2026-11-03-past-picks-design.md](../specs/2026-11-03-past-picks-design.md), only where a review
   needs it.

Skills: `cluster-run` (tickets 04's node smoke; read its SKILL.md before the first cluster step).

## 2. Decided by the user, not to be reopened

- All of the spec's §4 (D1 to D25) and A1; the run's end time (CPU scoring may run to about 13:00 on 11-04; GPU work
  ends by 10:00 on a weekday). See the build handoff §2.

## 3. Where things stand

**Tickets** (integration branch `r19-past-picks`, worktree `/project/CoSiR/.claude/worktrees/r19-int`, pushed):

| # | State | Review | Merge |
|---|---|---|---|
| 01 stores, register | merged | APPROVE with notes; MMArt train-split refusal and held test added | 9ebfe04 |
| 02 extract | merged | controller diff read; GPU smoke: all parity and store checks pass | 9ed85ab |
| 03 SemArt set | merged | APPROVE with notes; fixes applied | 865f114 |
| 05 matched | merged | APPROVE with notes; two test gaps closed | 00c8902 |
| 06 exports | merged | APPROVE with notes; three test gaps closed; 35 tests pass on the merged tip | 3ae28fe |
| 07 ArtELingo, Affection | merged | APPROVE with notes; reviewer fixed a gate (vii) type slip; route tests added | d6025e6 |
| 09 SemArt scorer | merged | CHANGES NEEDED: float16 rows into a float32 reader (blocker); fixed, tests added | 9331ff5 |
| 11 outcomes | merged | CHANGES NEEDED: gates (ii), (iii) and a failed c* run did not void SC15; fixed | e8688b2 |
| 08 MMArt | **built, not reviewed** | branch `r19-t08-mmart` (13572b5), worktree `agent-a120e9c030e32f147` | none |
| 04, 10, 12, 13 | not started | 04, 10 unblocked now; 12 after 01 to 11; 13 after 07 to 10 | |
| 14 | waits for r17's merge into main | | |

**Gate checks run in the build** (outputs in this chat's scratchpad, not the real OUT; the run chat runs stage 0):
- Gate (iv), r14 part (ticket 07, `pp_gate14.py iv`): passes; max difference 0.0 against r14's stored CLIP L/14 cells
  (clustering 192 items, retrieval 138, Affection 84; 24, 4 and 3 min under load).
- Gate (vii), r14 part (`pp_gate14.py vii`): passes; max difference 0.0 against r14's stage 1 records (SigLIP 1 561
  items, CLIP L/14 359; 21 and 25 min under load). The first SigLIP 1 job ran before the reviewer's fix and was stopped.
- Gate (iii) (ticket 06): both parts pass in the smoke and the reviewer's rerun (r10 max 6.1e-5, r14 0.0; bar 1e-3).
- Gate (v) (ticket 05): M3 and M6 pass (CPU mapper replay at most 4.2e-7, bar 1e-5); the sample M5 seed-0 rerun
  differs by 0.0 (ran on the local 3090, not node402: accepted, weights hash equal byte for byte).
- Gate (vi) (ticket 03): passes, 11,162 paintings by 2,717 artists; 11,159 after the md5 step.
- Gate (i) parity (ticket 02 smoke, 100 keys): every minimum cosine 0.9997 or higher (bar 0.999).
- Not run yet: gate (iv) SemArt part (`pp_semart_score.py gate-iv`, about 10 to 15 min), gate (iv) and (vii) r15
  parts (`pp_mmart.py replay-iv`, `replay-vii`; ticket 08).

**Resources at 23:00:** CPU load 50 to 65 (r17's build: three of four CPU slots); subagents often got exit 75 from
`res-queue cpu --no-wait` and ran their tests directly with 8 threads. GPU idle. Codex at 21:55: five-hour window 37%
(no room for an implementer), resets 23:34; weekly 90%. node402 not checked since 20:44.

**Timing:** 08's review, 04 and 10 together, then 12 and 13 together, then the final review and its fix wave: about
3 to 3½ h from 23:15, so the build ends about 02:30 to 03:00. The run chat then starts the Affection extraction
first; the heavy extraction must start by 07:00 (weekday rule at 10:00).

## 4. Open points for the next step

None for the user.

## 5. Code and pitfalls

- The previous handoffs' §5 hold (keys, git, queues, `cluster` on PATH is Graphviz's, frozen inputs only in
  `/project/CoSiR`, `git add -f` for JSON records, commit only your own paths).
- **Agent tool worktrees start on main**, and `git reset --hard` is refused; the implementer brief now says
  `git checkout -b r19-tNN-<slug> r19-past-picks`. For a fixer on a finished ticket, reuse the ticket's worktree; when
  gate jobs run from a worktree, give the fixer its own worktree branched from the ticket's branch (as for 07).
- **Reviews found real defects in 4 of 8 tickets** (11, 09 blockers; 07 a type slip; 05, 06 test gaps that let a
  wrong row order pass). Keep the Opus `high` review for 08, 10, 12's verdict parts; ask each reviewer for sabotages.
- **Gate runs are slow under the r17 load**: queue each gate part as its own `res-queue cpu --est 20m --label
  r19:<x>` job from a background Bash; the parts write separate files and a `merge` command joins them.
- **Ticket 12** must: read `gate_v_<enc>.json` per encoder; write `status.json` in `pp_outcome`'s strict schema; add
  an artist count of the 11,159 set to `pp_semart.record()` and pass the real counts to `decide`; assert
  `stores_checked` covers the new SemArt stores; run one encoder per part at a time (`write_cells` has no lock).
- **Ticket 04**: an `m5` mode in `scripts/run_pp_node.sh`, and M5's reads in the data map (controller notes).
- Commit trailers: some subagents used Sonnet co-author lines; harmless.

## 6. State at handoff

- Running: nothing of r19 (no subagent, no Codex job, no queued job). The gate files are in
  `/tmp/claude-0/-project-CoSiR/5dffc27d-c94f-4137-9d05-bfebeec91b0e/scratchpad/t07_gate_out/stage0/`.
- Uncommitted: nothing of r19's in git (`.scratch/r19-past-picks/` is outside git by design).
- On disk over 1 GB: this chat's scratchpad on the system disk, about 3.6 GB, of which the ticket 06 reviewer's
  `t06rev/` (1.9 GB) is on `.scratch/pending_deletions.md`.
