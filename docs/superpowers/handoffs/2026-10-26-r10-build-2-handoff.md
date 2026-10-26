# Handoff: dispatch the relational genre round's tickets (01, 03, 07 first), build, review, final review, merge, run handoff

## Brief
- Done: tickets 01 to 11 with the coverage line, checked; integration branch `r10-relational-genre` with the round foundation (abbf031), pushed.
- State: no ticket started. Codex at 20:05: five-hour window 100% left (reset 01:04), weekly 96%.
- Next: dispatch 01 (Codex high), 03 and 07 (Sonnet high workers), then the frontier as the index says.
- Blocked: nothing. No container restart (user, 20:01, for this build); GPU check passes at 20:10.

> Written 2026-10-26 20:10 by "r10 build" (cut at 200k context before any dispatch). Loop step reached: step 7 (build),
> tickets written. Direction log entry: [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The build handoff [2026-10-26-r10-build-handoff.md](2026-10-26-r10-build-handoff.md) §2 to §5 (decided items,
   frozen inputs, the build's open points a to e, code and pitfalls). It still holds; this file adds only the state.
3. The ticket index `/project/CoSiR/.scratch/r10-relational-genre/issues/00-index.md` (conventions, shared
   interfaces, output layout, ticket table, coverage) and the tickets `01` to `11` beside it.
4. `.scratch/r10-relational-genre/notes/build_progress.md` (design notes: agent defaults already in the tickets),
   `implementer_brief.md` and `reviewer_brief.md` (give their paths to every implementer and reviewer).
5. The spec from §1 and the rule only where a ticket review or a dispatch question needs them (the tickets cite the
   sections). `~/.claude/references/matt-chain.md`, `codex-jobs.md` and `~/.claude/rules/agent-routing.md` before the
   first dispatch.

Skills: none beyond those `matt-chain.md` names (`implement-spec` is followed from its SKILL.md; `tdd` is the
implementers').

## 2. Decided by the user, not to be reopened

- As the build handoff §2: the spec's D1 to D20, M1 to M10 and the brief approved at 19:57.
- No container restart before the build (user, 20:01, r10 decide 25).

## 3. Where things stand

- Integration branch `r10-relational-genre` at abbf031 (worktree `/project/CoSiR-r10-rg`, pushed): `rg_paths.py`,
  `rg_variants.py` (every arm's switches m, lam, tau_t, anchors; orders; T_G, RANGES, X, G_HALF), `conftest.py`,
  `.gitignore` entry for the outputs link. `RG/outputs` links to `/data/SSD2/cosir_r10/20261027_relational_genre/outputs`
  (empty) in both checkouts.
- Frontier: 01, 03, 07 (no blockers). Then 02 and 06 after 01; 04 after 03; 05 after 04; 08 after 07; 09 after 04;
  10 after 07; 11 after all.
- Routing (agent-routing.md; the dispatcher may change it with the room check): 01 Codex `gpt-6.1-sol` high; 09 and 10
  Codex high (re-derivations, independent of the Claude-written scorers); verdict tickets 02, 03, 04, 06, 07 Sonnet
  `high` workers (Codex when room and no other Codex job runs); 05, 08, 11 Sonnet `medium`. Ticket reviews of verdict
  code: Opus `high` workers. Final whole-branch review: Opus `xhigh`; fix wave; scoped re-review Sonnet `high`.
- Codex room: an implementer job needs 80% of the five-hour window by the estimate (real jobs used 3% to 13%); one
  Codex implementer at a time.

## 4. Open points for the build (agent work; nothing for the user)

- As the build handoff §4 a to e: the build per `matt-chain.md`, the GPU checks during the build (VA1, EQ, VB1, VC1
  and smokes under the lock: light), the run chat's handoff (ticket list, queue order, commands, end estimate) and
  `loop-next <handoff> --label "r10 run" --unattended` after the GPU check.

## 5. Code and pitfalls

- As the build handoff §5. Added by the tickets: GR-λ goes through rg's own copy of eg_train3's genre body (term
  logging), held bit for bit by ticket 01's test and VA1; GQ's anchors are fit-local positions and a dev call masks
  nothing (ticket 01); the held guard of ticket 06 needs a mutation check in its review.
- `.scratch/` is outside git: the tickets and notes live only on disk; the run handoff carries the ticket list.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat).
- Uncommitted: nothing of this build (this handoff is committed with this cut).
- On disk over 1 GB: unchanged.
