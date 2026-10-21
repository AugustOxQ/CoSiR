# Handoff: finish the D2 ablation build (one schema fix, ticket 06, GPU smoke, final review, merge), then open the run chat

## Brief
- Done: tickets 01 to 05 of the D2 ablation are merged on branch `r10-d2-ablation`; 108 tests pass.
- State: the scorer reproduces the model sheet's D2 and T1 image AMI exactly (fake run on copies of their outputs).
- Next: fix the analysis reader of `scores.json`, build ticket 06, GPU smoke, final review, merge; open "r10 run 12".
- Blocked: nothing. Night mode. Cut at 200k context.

> Written 2026-10-21 04:25 by "r10 build 14". Loop step reached: step 7 (build), five of six tickets merged.
> Direction log entry: [2026-10-21 · Claim test read](../direction_log.md).

## 1. Read in this order

1. This handoff, then the ticket index and ticket 06: `/project/CoSiR/.scratch/r10-d2-ablation/issues/00-index.md`,
   `06-stage0-queues-smoke.md` (gitignored, on disk; tickets 01 to 05 are marked done).
2. The spec, [2026-10-21-d2-ablation-design.md](../specs/2026-10-21-d2-ablation-design.md), from §1.
3. `~/.claude/references/matt-chain.md` (final review, merge), `~/.claude/references/codex-jobs.md`.
4. The previous handoff, [2026-10-21-r10-build-14-handoff.md](2026-10-21-r10-build-14-handoff.md), §2 (user
   decisions) and §4 c to e (the run queue, the run chat's side work, its next chat).

Skills: none (`implement-spec` and the final review per `matt-chain.md`, read from its SKILL.md if needed).

## 2. Decided by the user, not to be reopened

As build 14's handoff §2: the spec's rows R0 to R12, development data only, 3 seeds, local 3090, no bar, item 5 in
the morning decide chat, no restart, night mode.

## 3. Where things stand

| # | Ticket | Commit | Built by | Check |
|---|---|---|---|---|
| 01 | Graph store `ab_graphs.py` | cd6d0ba | Sonnet high worker | diff read; 7 tests; sd values equal the aspect store's to the last digit |
| 02 | Runner and switches `ab_paths.py`, `ab_rows.py`, `ab_train.py` | f5e31ad | Codex gpt-6.1-sol high | diff read; 64 tests; CPU smoke R0, R3, R5, R7 |
| 03 | Metric helpers `ab_metrics.py` | 24a033d | Sonnet high worker | diff read; 22 known-answer tests |
| 04 | Scorer `ab_score.py` | f411d33 | Sonnet high worker | fake run: R0 = D2 .096188 / .369665 / .454025, R2 = T1 .093008 / .359080 / .460308, exact |
| 05 | Analysis `ab_analysis.py` | ad8ff7d | Sonnet high worker | views reproduce the earlier view AMIs exactly; 10 tests |
| 06 | Stage 0 check, queues, smoke | not started | | |

- Graphs (01): `Dview` 62,754 fit / 8,248 dev edges; it shares 31% of fit edges with genre R0o, under 1% with
  emotion Gres and style Fres05. The 18 random graphs keep every degree; 0.1% to 1.4% of their edges are original.
- Runner (02): all-off rows (R0, R2 and the flag rows) call `ms_train.run` unpatched. R3, R5, R6, R7 run it under a
  scoped `mock.patch` of `ms_train.head_loss`; R3 also patches `ags_trainimg.aspect_images` to the default image.
  `ab_row.json` per row binds the switches and a code hash.
- Cost: the full scoring is 50 to 70 min at 8 workers (156 probe calls carry most of it); the full four-view part 4
  min on a quiet machine, 13 min at load 16. The load from other sessions was 9 to 18 at 04:00.
- Codex: the ticket 02 job used 6% of the five-hour window against an estimate of 40% (ledger row added).

## 4. Open points for the next step (agent work; nothing for the user)

a. **Schema fix first.** `ab_analysis.read_scores` and `score_entry` assume
   `scores['rows'][row][aspect]['img'][str(seed)]`. `ab_score.py` writes `scores['aspects'][a]['rows'][R]['img']`
   with `kmeans.ami.per_seed[i]`, `probe.balanced_accuracy.per_seed[i]`, `probe.per_class_recall[seed]`,
   `probe.classes`. Change only those two functions, plus a test on a file written by `ab_score`'s own writer.
b. **Ticket 06**, as written; also make the smoke root a CLI or env setting: `ab_train.SMOKE_ROOT` is hard-coded to
   build 14's scratchpad (`/tmp/claude-0/-project-CoSiR/0f66078d-adcf-4c6d-beea-edeb3524ab4a/scratchpad/t02_smoke`).
c. **GPU smoke** under `flock -o -w ... /tmp/gpu0.lock`: R0, R3, R5, R7, R9, R10 on seed 0 with `--smoke`, then the
   scorer (20 resamples) and both analysis parts.
d. **Final whole-branch review** (Opus `xhigh` worker, `matt-chain.md`), one fix wave, a scoped re-review (Sonnet
   `high`); merge into main, push; the integration worktree goes on `pending_deletions.md`.
e. **The run chat**: write its handoff (ticket list from the index, run commands, timing) and open
   `loop-next <handoff> --label "r10 run 12" --unattended`; carry build 14's §4 c to e.
f. **Choices for the final review to check:** R3's teacher is the student's own input (D17); R7's loss is the
   image-image term alone (not divided by 4) plus distillation; R4 uses `ms_train`'s graph-free 1,024-painting path
   (D13); one bootstrap draw list per aspect, seeds drawn for all 13 rows in table order; the probe uses 1 BLAS thread
   per worker (third decimal differs from 8 threads); the default Qwen style reference is .221676 from
   `20261019_model_sheet/outputs/backfill.json` (the sheet's style bar is CLIP).
g. **Timing:** the build ends near 06:00 at best; GPU stages about 30 min, CPU stage 2 about 1 to 1.5 h. Reports are
   due by 10:00 (spec §9) and the machine is free by 11:00.

## 5. Code and pitfalls

- Integration worktree `/project/CoSiR-wt/r10-d2-ablation` (branch `r10-d2-ablation`, pushed). The main checkout
  stays on main. Tests: `... -m pytest -q src/test/20261021_d2_ablation` in that worktree.
- Earlier rounds are frozen (the claim test hashes `ms_*.py`): the new code imports and patches them, never edits.
- Artifacts exist only in the main checkout. `ab_paths.MAIN_ROUND` is `/project/CoSiR/src/test/20261021_d2_ablation`:
  the real graph store and outputs go there after the merge, never into the worktree.
- `ab_graphs.py` refuses to replace an existing store without `--overwrite`.
- Unattended: delete nothing; `.scratch/pending_deletions.md` has this chat's worktrees and ticket branches.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or node job).
- Uncommitted: nothing after this commit.
- On disk over 1 GB: `src/test/20261020_claim_test/outputs/stage2/` (4.6 GB, keep, ledger row).
