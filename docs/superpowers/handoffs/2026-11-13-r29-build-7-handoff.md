# Handoff: r29 build 7: fix wave B, scoped re-review, gate (vii) ablation try 3, merge into main, open the run chat

## Brief
- Done: tickets 11 and 13 merged (gates vi and vii_photo pass); suite 703 passed; final review (2 parts): pass after fixes; fix wave A.
- State: integration 2ede24ee (pushed); fix wave A on r29/final-fix 47bb8e94 (pushed, not merged).
- Next: fix wave B (Table 8 for a prefixed K*, plus part 2's findings), one scoped re-review of A and B, gate (vii) ablation try 3, merge, run chat.
- Blocked: nothing.

> Written 2026-11-13 03:40 by the "r29 build 6" tab (cut at the context meter's 200k note). Loop step reached: step 7
> (build), final review done, fix waves in progress. Direction log entry: [2026-11-12 · r29 decide](../direction_log.md).

## 1. Read in this order

1. The controller's open items: `/project/CoSiR/.scratch/r29-siglip-only/notes/controller_todo.md`, section "r29
   build 6 progress" at its end.
2. The first build handoff, sections 2 to 5: [2026-11-12-r29-build-handoff.md](2026-11-12-r29-build-handoff.md); the
   build 6 handoff, sections 2 and 5: [2026-11-13-r29-build-6-handoff.md](2026-11-13-r29-build-6-handoff.md).
3. The final review: `notes/final_review.md` (part 1, PARTIAL) and `notes/final_review_part2.md` (part 2).
4. Fix wave A: `notes/final_fix_a_brief.md` (pattern for wave B) and `notes/final_fix_a_handback.md`.
5. The run handoff draft: [2026-11-13-r29-run-handoff.md](2026-11-13-r29-run-handoff.md) (placeholders in capitals).
6. `~/.claude/references/matt-chain.md`, `~/.claude/references/codex-jobs.md`, `~/.claude/references/loop-chats.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

As in the build 6 handoff, section 2 (spec §4 D1 to D25; daytime GPU runs allowed this round with
`R29_ALLOW_DAYTIME=1`; r29's run goes before r21's next GPU job). Run mode is night.

## 3. Where things stand

Integration branch `r29-siglip-only`, worktree `/project/CoSiR-r29`, at 2ede24ee (pushed): all 15 tickets merged.
Round suite on 2ede24ee: 703 passed (`notes/suite_build6.log`). Gate records: every stage 0 gate passes at try 1
except i_c (try 3) and vii_trainer (try 2); vii_ablation is stale on purpose (try 2; try 3 after the fixes).

| Item | State |
|---|---|
| 11 | merged b5eb4be0 (fix 2bcb0873: 44 tests); gate vi pass try 1 (41 min; 30 exports, 12 cells, 15 r28 values, 167 pick items, all diff 0), merged a73af55b |
| 13 | Opus review pass after one fix; fix d8bdec2e (48 tests; `cells` reads v_photo and vii_photo through `r29_ladder.gate_status`); gate vii_photo pass try 1 (39 units, diff 0); merged 2ede24ee |
| Final review part 1 | Opus xhigh, PARTIAL at 200k: pass after fixes; SC2 on K0, an image and a text cell re-derived exact; 10 of 10 guard mutations caught; 46 CLIs ok |
| Final review part 2 | Opus xhigh, complete: pass after fixes (`notes/final_review_part2.md`). SC7 mAP re-derived on K0's held exports against r28's `ablation_ref_replay.json`: emotion 0.2628536890872099, genre 0.5986201303452721, diff 0. Ticket 13's fix changed no number path. Must-fix: `r29_ablation.py:283-286` (one removal's crash loses all four SC7 rows; contradicts D21 and §3.5). 4 should-fixes, 3 minor |
| Fix wave A | 47bb8e94 on `r29/final-fix` (worktree `/project/CoSiR-r29-ffix`): part 1's must-fix (stage 2b skips a piece, not the stage) and 4 should-fixes; tests 45 / 57 / 43 pass; dry runs `notes/ffix_dryrun_*.txt`; controller read the script diff |

Next steps, in order:
1. Fix wave B (Sonnet high fixer, on `r29/final-fix` in `/project/CoSiR-r29-ffix`; brief in the pattern of
   `final_fix_a_brief.md`):
   - Table 8 for a prefixed K* (fix A's flag): `r29_time.py:33`, `:72` assert `cap_source == 'siglip1'` and read the
     plain store. Read K* from `records/prefix_pick.json` (`kstar`, `kstar_form`); read the caption rows from K*'s
     store (`C.caption_rows(kstar, ...)`), check the sidecar against it, and time the SigLIP 1 text pass on the texts
     in K*'s form (with the prefix when prefixed, built as ticket 02's prefixed store builds them). Plumbing, not
     verdict code; a dry or CPU check of the text building.
   - Part 2's must-fix (`notes/final_review_part2.md` item 1): `r29_ablation.py:283-286` scores only the complete
     removals and lists the others as inconclusive with their cause (D21, §3.5 "the others go on"); a journal with no
     `status` counts as not ok. Test. It must land before gate (vii) ablation try 3.
   - Part 2's should-fixes: `r29_ablation.py:412` (KeyError on `doc["cells"]` when the gate is parked);
     `r29_run_stage2.sh:106` (the ablation step runs a second time on failure, D21's same-seed rerun);
     `r29_ladder.py:551-555` (the numbers sheet shows K*'s ablation rows, as ticket 14 asks); `r29_ladder.py:645`,
     `:742` (diagnostic (d): add MMArt and photo if §3.6 asks for them, else a decision line). Minor, if cheap: the
     sheet on a fail shows SC2's status and why SC4 to SC7 did not run.
   - Drop the stale "TO CHECK AFTER TICKET 13 MERGES" comments in `r29_run_stage2.sh` (13's CLI matches).
   - Frozen: `r29_trainer.py`, `r29_model.py`, `r29_captions.py`. `r29_ablation.py` and `test_r29_ablation.py` change
     in this wave (the must-fix above) and freeze at gate (vii) ablation try 3.
2. One scoped re-review (Sonnet high) of A and B together: `git diff 2ede24ee..<B commit>`, the findings and the
   rulings; one hand mutation per must-fix.
3. Merge `r29/final-fix` into `r29-siglip-only`; run the round suite (`res-queue cpu`, about 3.5 min); gate (vii)
   ablation try 3 from `/project/CoSiR-r29/src/test/20261112_siglip_only`:
   `res-queue cpu --label r29:gate-vii -- env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python r29_ablation.py gate-vii`
   (about 2.3 min; commit the record with `git add -f`). After it those files are frozen.
4. Merge `r29-siglip-only` into main locally, push main (`/project/CoSiR` has other chats' uncommitted files: commit
   only r29's). Mark tickets done in `.scratch/r29-siglip-only/issues/`. Add the worktrees and ticket branches to
   `.scratch/pending_deletions.md`.
5. Fill the run handoff's placeholders (MAINMERGE, SUITECOUNT, VIIABL, WRITTEN, FIXLINES, FIXPITFALL) and commit it.
   Check whether r28's run chat has ended (first build handoff, section 5: then the run chat opens r29's main chat;
   the draft already says so). GPU check (`loop-chats.md`, "How to cut"); then
   `loop-next docs/superpowers/handoffs/2026-11-13-r29-run-handoff.md --label "r29 run" --unattended`; commit
   `chats.tsv` and `briefs.md`; push.

## 4. Open points for the next step

None for the user (night mode). Decision lines to add with wave B: Table 8 times K*'s own caption form; the SC7 crash
rule after the must-fix (it replaces the 00:41 line); diagnostic (d)'s scope. Fix wave A named no result-shaping
choice.

## 5. Code and pitfalls

- As in the build 6 handoff, section 5 (gate (vii) ablation try 3 freezes its files; gate status rule; smoke roots;
  `decisions.md` conflicts at every merge: keep both sides).
- `r29_run_stage1.sh` now has a part `t8` (Table 8), run after 1b; 1c no longer runs it. The run handoff draft must say
  so in its section 5 (FIXPITFALL).
- A gate record in state `fail` (tries < 3) is skipped by the scripts' `have` check; a rerun after a fix needs
  `R29_FORCE=1`. Put that in the run handoff too.
- Fix wave A's misnamed dry-run files (`notes/ffix_dryrun_stage1 c_.txt` and three siblings) are on
  `.scratch/pending_deletions.md`.
- Codex had no room all night until the 04:44 reset; check `codex-usage --room` before each job.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing in the worktrees. The run handoff draft is committed with placeholders.
- On disk over 1 GB: as the build 6 handoff, plus nothing new (`.scratch/pending_deletions.md` has the build 6 rows).
