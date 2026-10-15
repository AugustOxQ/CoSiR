# Handoff: r8 build: turn the two approved round 8 specs into tickets and build both tracks in one build

## Brief
- Done: r8 decide finished steps 1 to 6: grouping work beside the AFF paper, the grill, and two approved specs.
- State: main line, (ii) learned head and FCMI in CSD's slot, 24 candidate scorers; encoder screen, 6. Nothing is
  built yet. The hard cut-off for a held verdict is Fri 2026-10-23.
- Next: `to-tickets` on both specs, then `implement-spec`, one build for both tracks, ready for one overnight run.
- Blocked: nothing.

> Written 2026-10-12 by the `r8 decide 2` chat. Loop step reached: step 6 done (both spec briefs approved), before step 7.
> Direction log entry: [2026-10-12 · Round 7 design L](../direction_log.md) ("Decided (A), completed" and "Idea (B)
> and the grill").

## 1. Read in this order

1. The main-line spec (approved) `docs/superpowers/specs/2026-10-12-r8-learned-grouping-design.md`, from §1 (the brief is the
   user's page).
2. The screen spec (approved) `docs/superpowers/specs/2026-10-12-r8-encoder-screen-design.md`; it cites the main spec for
   everything shared.
3. `~/.claude/references/matt-chain.md` before step 7 (verdict code, ticket size, the lean research build).
4. `src/test/20261129_r8_grouping/step3_facts.md`: the plug-in path for a new grouping, the content grouping file, the
   image cache, the held rules and round 7's timings.

## 2. Decided by the user, not to be reopened

All in the direction log entry above and the specs' briefs (2026-10-12). Among them: both positive forms for (ii);
six settings per builder; CLIP space for neighbours; gate "beats AFF" with round 7's gate as the weaker goal; held
read on seeds 52 to 54 with eight checks under Holm at 0.0125 and three tiers (decided after seeing the
recommendation); main line first, the screen as fallback; captions keep the CLIP text head; the grid is not extended.

## 3. Where things stand

- Step 3 facts are committed in `src/test/20261129_r8_grouping/` (paired-image reference, facts brief) and
  `docs/reports/literature/2026-11-30_r8_learned_grouping.md`.
- AFF on seed 42: 19.14 R@1 (B′(A1) 18.81). Round 7's gate read 0 of 12.

## 4. Open points for the next step

- None for the user: both briefs are approved (2026-10-12).
- The specs' "Check these" items carry agent defaults; the user may still answer them.
- The user chose no stop before the held read (spec A, D36).

## 5. Code and pitfalls

- New round modules take the prefix `r8_` in `src/test/20261129_r8_grouping/`. Round 7's `results/` is write-once.
- Grouping IDs are hard-coded in round 7's `r7_bundle.py` and `r7_reader.py`; the cross-fitted head inputs
  (`xheads`) have no path for a grouping's own image posterior. The specs say how to add one.
- FCMI's official code has an undocumented MLP path for feature vectors (`NetFCN`); 64 protected groups are untested.
  A failed smoke run drops FCMI (spec rule).
- Held rows stay untouched until the held rule is committed (C5); develop on seed 42.
- GPU work takes the shared lock (`flock -n -o -E 75 /tmp/gpu0.lock ...`); one queue script per GPU stage.
- Codex implementer launches were refused intermittently by the auto-mode check in r7 (an open routing item); take
  the ticket back to a Claude worker when that happens.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing.
- On disk over 1 GB: nothing new (round 7's `results/`, `res/cluster_jobs/` and the token cache are ledgered keeps).
