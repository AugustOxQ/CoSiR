# Handoff: decide round 7, design L (a grouping redesign so that style stops leaking into genre), from step 3, while round 6 runs

> Written 2026-10-09 05:10 by the `r6 decide 2` chat, for the user to start by hand. Loop step reached: decision A
> made (design L next, in its own loop and chat); type and loop size to confirm; step 3 next. Direction log entry:
> [2026-10-07 · Idea 3 killed at development; next step open](../direction_log.md) ("Decided (A)").

## 1. Read in this order

1. `docs/superpowers/direction_log.md`: "Now", "Parked ideas" (the first two are design L and its 30-minute
   label-free check) and the 2026-10-07 entry.
2. `GLOSSARY.md` (repo root) for the terms; `docs/superpowers/constitution.md` (version 2).
3. What design L answers, in the reports (read the cited parts only):
   - `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md` §7 (per pair: AFF loses on style × genre) and §8 (the
     random-share control: the gain comes from steering mostly one side);
   - `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md` around lines 418 to 434 (CSD's genre
     content, pair ratio 0.83; design L as non-redundancy with the image grouping);
   - `docs/reports/auto/v2/2026-11-18_reader_fix_csd.md` around lines 43, 141 and 565 to 593 (design L as a
     late-fusion refinement of the groupings, and why it was deferred);
   - `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md` (B′(A1), the CSD style grouping's floor).
4. `~/.claude/rules/research-loop.md` steps 2 to 6, `~/.claude/references/grill.md`, `~/.claude/rules/spec-templates.md`.

## 2. Decided by the user, not to be reopened

- Design L is next, in its own loop and chat (user, 2026-10-09 02:48: "Then in another chat we go with design L").
- Design L may get its own pre-registered held read later (constitution C5, amendment 2, 2026-10-09); the paper
  discloses that it was designed after AFF's held read.
- The user starts this chat by hand while round 6 builds and runs unattended (user, 2026-10-09 05:05).

## 3. Where things stand

- AFF is the current best: round 3 GO on fresh seeds 49 to 51, bar margin +0.591 [+0.462, +0.729] R@1 over B′(A0).
  Per pair (round 3): emotion × style +0.810, emotion × genre +1.544, style × genre −0.580 [−0.793, −0.363].
- On seed 42, B′(A1) (B′ plus the CSD style grouping) is 18.805 against B′(A0) 18.437, and AFF (19.137) is only
  +0.332 [+0.048, +0.625] above it: a style grouping helps the condition-free score, but no reader has yet used it to
  steer (rounds 1 and 4).
- Round 6, the held-split paper test of AFF, is building and running unattended (tabs `r6 build`, then `r6 run`;
  spec `docs/superpowers/specs/2026-10-09-r6-held-test-design.md`). Its held verdict is not known yet.
- The 30-minute label-free check of a de-genred style grouping was proposed but never run.

## 4. Open points for this chat

1. **(key, ask answer-first) Blind to round 6's held numbers?** Should this chat avoid reading round 6's held results
   (its results folder, its reports, the filled H5 ledger row) until design L's spec is approved, so that design L is
   designed without seeing them? It costs nothing now; the paper could then say design L was fixed before AFF's held
   numbers were seen.
2. **Step 2, short:** confirm the type (new idea) and loop size (full: step 3 includes a literature check).
3. **Step 3:** the parked ideas first (design L, its 30-minute check, dropping the CLIP caption grouping), then our own
   results, then the literature check on the `deep-research` agent (skim `docs/reports/literature/` first; say which
   model and depth and why). Two or three candidates; the user picks (step 4), then the grill (step 5) and the spec
   (step 6).

## 5. Code and pitfalls

- **Shared resources while round 6 runs:** r6 run uses the DAS6 nodes (node401, node402, node408) and CPU overnight.
  Design L's checks run on CPU (8 threads, `uptime` and `free -g` first) or the local GPU under its lock; do not take
  DAS6 GPUs while r6 holds them, and never touch r6's files (`src/test/20261125_artelingo_held_test/`).
- New round folder in `src/test/` with the next sequence date after `20261125` and module prefix `r7_` (round folders
  go on `sys.path`). Development on seed 42 only; selection seeds 52 and later are free in the seed ledger, but r6 uses
  52 to 54 on held rows, so pick fresh selection seeds from 55 to avoid confusion.
- No held row is read (C5). Sample IDs per C6. Every Python call with `PYTHONDONTWRITEBYTECODE=1`.
- **Chats record:** if this chat was started by hand rather than by `loop-next`, add its row (label `r7 decide`, this
  session's id) to `docs/superpowers/handoffs/chats.tsv` and commit it.
- At this chat's end (spec and rule approved), cut to `r7 build` with `loop-next` as usual.

## 6. State at handoff

- Running: the `r6 build` chat (it opens `r6 run`, which ends by opening an `r6 read` decide chat for round 6's
  results).
- Uncommitted: nothing.
- On disk over 1 GB: nothing from the decide chats.
