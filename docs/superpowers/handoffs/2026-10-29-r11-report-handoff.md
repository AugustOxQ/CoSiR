# Handoff: r11 report: write step 9's auto report and user-read report from the finished run, then open r12 decide

## Brief
- Done: the r11 run. Stage 0b first failed (PercepT core read from the wrong mapper); fixed as amendment A2, rerun, all checks pass.
- State: stage 5 (each model with its read-out picked on validation): GO. Stage 1 (plain cosine): NO-GO, emotion below the instructed Qwen image. Switch test GO.
- Next: the auto report and the user-read report (step 9), then open the r12 decide chat.
- Blocked: nothing.

> Written 2026-10-29 23:10 by "r11 run". Loop step reached: step 8 done, step 9 next. The cut is for context size
> (the run chat passed 200k tokens). Direction log entry: [2026-10-29](../direction_log.md).

## 1. Read in this order

1. The run's tables: `/data/SSD2/cosir_r11/20261029_retrieval/outputs/summary.md` (regenerate with
   `python src/test/20261029_retrieval/rt_summary.py /data/SSD2/cosir_r11/20261029_retrieval/outputs`). Every number
   below comes from it; numbers in the report come from the verdict files it reads, never from this handoff.
2. The spec, `docs/superpowers/specs/2026-10-29-retrieval-design.md`: the brief (prior and bar), §3.1 (prior), §3.4 to
   §3.6, §6 (decision rule and licensed claims), §8 (risks, selection disclosure), §9 (deliverables), and amendments
   A1 and A2 (§4).
3. The round log, `src/test/20261029_retrieval/20261029_retrieval_log.md`: "Amendment 2" and "The full run".
4. The report rules and templates: `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`,
   `~/.claude/templates/auto_report_template.md`, `results_brief.md`, `~/.claude/references/user-read-report.md`,
   `ste-writing.md`. Style model: report 12-28 (`docs/reports/auto/percept/2026-12-28_recipe_sweep.md`) and its index
   row in `docs/reports/reports_sum.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

- The protocol, the bar, the baselines, the read-out optimization and the claim table (stage 5 carries the paper's
  claim; stage 1 is the untuned row): spec §4, D1 to D8 and D15, approved 2026-10-29 18:30.
- Whether the paper claims the switch test: the user decides after the report (D7). Whether CRL enters a bar: after
  its number (D6).

## 3. Where things stand

12,281 held paintings (genre labels on 9,949), 5 seeds for c* and PercepT core. mAP; differences are c* minus the
comparator, 95% joint bootstrap. Seed ranges r (c*'s validation spread, committed in `9829741` before any held
number): emotion 0.0014, style 0.0022, genre 0.0038; emotion nDCG 0.00031.

| Table | Emotion | Style | Genre | Verdict |
|---|---|---|---|---|
| Stage 1 against plain Qwen (SC1 to SC3) | +0.0217 WIN | +0.1309 WIN | +0.1136 WIN | |
| Stage 1 against instructed Qwen (SC4 to SC6) | −0.0023 [−0.0032, −0.0014] BELOW | +0.0019 WIN | +0.0037 WIN | NO-GO |
| Stage 5 against plain Qwen (SC10 to SC12) | +0.0208 WIN | +0.1311 WIN | +0.0790 WIN | |
| Stage 5 against instructed Qwen (SC13 to SC15) | +0.0007 [−0.0002, +0.0016] LEVEL | +0.0017 [−0.0001, +0.0036] LEVEL | +0.0027 [+0.0013, +0.0042] WIN | GO |

- Switch test (SC7 to SC9): GO; in each column c*'s own head beats its other two heads by 0.0103 to 0.2304.
- Stage 3 (graded emotion, nDCG): qualified. c* beats plain (+0.0052) and instructed (+0.0002 [+0.0001, +0.0004],
  WIN), while SC4 is BELOW.
- Stage 4 picks (validation): centering plus query expansion with k = 10 for every arm except CRL (no centering,
  k = 10). Every pick sits at the menu's largest k. No whitening read-out passed the guard for any arm.
- Descriptive arms (stage 5): CRL 0.2554 / 0.2816 / 0.5033, CLIP L/14 0.2479 / 0.2732 / 0.4981, SigLIP 2
  0.2443 / 0.2576 / 0.4968, PercepT core 0.2483 / 0.1523 / 0.5049; c* 0.2691 / 0.3235 / 0.6151.
- C13 (`rt_vh`, independent code) agrees on every value of every stage (tolerance 1e-6); stage 0b passes 54 of 54.

**The run chat's notes for "our reading"** (the report chat weighs them; not checked beyond the summary):
- The prior (§3.1) expected emotion to be the safest win against the instructed image and genre the riskiest. Retrieval
  shows the opposite order: genre wins at both read-outs, emotion is BELOW at plain cosine and only LEVEL after the
  read-out. "Retrieval follows clustering" does not hold here.
- Per-class AP (stage 1): c*'s emotion loss sits in the largest class, contentment (−0.0324, 4,244 of 12,281 queries);
  c* gains on the other eight classes (+0.0027 to +0.0340). mAP weighs every query equally, so the largest class weighs most.
- The graded nDCG prefers c*: c* follows the mix of caption emotions, and loses on the exact majority label.
- The held seed spread of c*'s emotion mAP (0.00171) is above the validation r (0.00141), so LEVEL was slightly strict
  for emotion (§8 risk); r is not changed after results.
- Every pick sits at k = 10, the menu's edge, so a larger k may help further; any new read-out must again be picked on
  validation (`held-data.md`).
- The plain Qwen image gains more from the read-out than c* (validation score +9.25 seed ranges against +5.43); the
  margins over plain stay large. CLIP and SigLIP 2 gain most (+12.40, +11.97).

## 4. Open points for the next step

None for the user in this chat (night mode). Decision C and the next direction (more read-out work, retraining, or
r12's datasets as planned) belong to the r12 decide chat.

## 5. Code and pitfalls

- The auto report goes to `docs/reports/auto/percept/2026-12-29_retrieval.md` (CoSiR's sequence-number date), with
  its results brief and one row in `docs/reports/reports_sum.md` (then `python scripts/check_reports_sum.py`). The
  user-read report goes to `docs/user_read/<real Amsterdam date>_retrieval.md`, no index row. Time-box:
  2026-10-31 23:00.
- Routing (`agent-routing.md`): the auto report draft goes to Codex (`gpt-6.1-sol` medium) when
  `codex-usage --room report` says room, then Claude's claim review; the user-read report gets a cold read and a
  fidelity check in parallel, one fix wave. The run chat found no Codex room for an implementer job at 22:10 (the
  five-hour window resets 22:38).
- Deviations to report: amendment A2 (stage 0b's failure, cause, fix, rerun; PercepT core's numbers are the
  full-scale mapper's; the failed record is kept in `<outputs>/stage0b/check_failed_A1_sample_scale.json`).
- Selection disclosure (§8): the validation paintings chose c*'s epochs and sit inside the set where c* was picked;
  only held numbers are results.
- Glossary: "seed range" in `GLOSSARY.md` is r10's clustering unit; this round's r is a retrieval seed range on
  validation. Define it in the report; a glossary change needs the user's yes.
- `.claude/` is gitignored; the day's change log is `/project/CoSiR/.claude/20261029_log.md`.

## 6. State at handoff

- Running: nothing of the r11 run chat. GPU check (night mode, 23:08): driver 595.99.02, the GPU idle, no lock.
- Uncommitted: nothing (main at the commit that adds this handoff, pushed).
- On disk over 1 GB: nothing new from the run (outputs 59 MB at `/data/SSD2/cosir_r11/20261029_retrieval/outputs`).
  The Affection store (1.8 GB) still needs its disk-ledger row at the wrap-up.
- Listed on `.scratch/pending_deletions.md`: the old `percept_val/` exports (4.5 MB) and the `r11-fix3` worktree and
  branch, besides the build's rows.
