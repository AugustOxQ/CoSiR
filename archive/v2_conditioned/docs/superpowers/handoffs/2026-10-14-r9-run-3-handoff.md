# Handoff: r9 run 3: finish and check the auto report and the user-read report, commit them, then open `r10 decide`

## Brief
- Done: all five held reads and their descriptive passes. H6 GO, H7 GO, H8 NO-GO (inconclusive), H9 full claim, H10 NO-GO (inconclusive).
- State: every re-derivation agrees (166 or 171 items, 0 differences). A descriptive-pass bug on ArtELingo-28 reads was fixed (10b5a7b); no verdict touched.
- Next: Codex fills the last report parts; claim review; commit; open `r10 decide`.
- Blocked: nothing.

> Written 2026-10-14 12:42 by `r9 run 2` (cut: context over 200k). Loop step reached: step 9 (verify and report),
> all reads and descriptive passes done. Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

F = `src/test/20261202_r9_scope`.

## 1. Read in this order

1. Sections 3 to 5 below.
2. The run notes at the end of `F/20261202_r9_scope_log.md` ("r9 run chat 2"): every step, time and number.
3. The report draft `.scratch/r9-run/report_draft/2026-12-02_aff_scope.md` (15.4 KB) and its source ledger
   `notes_for_controller.md` (every number with its file and key). Briefs of the three Codex parts:
   `.scratch/r9-run/report_brief.md`, `report_part2a.md`, `report_part2b.md`.
4. Report rules: `~/.claude/rules/report-kinds.md`, `report-writing.md`, `user-read-reports.md`,
   `~/.claude/references/user-read-report.md`, `~/.claude/templates/results_brief.md`,
   `~/.claude/references/ste-writing.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34) and the rule wording (SHA f9b07b34). Reads ran locally (D12).

## 3. Where things stand

- **Verdicts** (`F/results/held_verdict_<Hn>.json`; posted to the user as they landed, no reading of ours):
  - H6 Qwen, ArtELingo held: GO, P1 to P7 pass, every n_j = 0. H7 SigLIP 2: GO, every n_j = 0.
  - H8 CLIP, ArtELingo-28: NO-GO (inconclusive). Level 1: P4 fails (n 1109, n* 124; x = 0.213, x95 0.193),
    P6 passes (n 0). Level 2 not tested.
  - H9 Qwen, ArtELingo-28: full claim, all seven pass, every n_j = 0. Set-aside none; flagged Emakhuwa,
    IsiNdebele; near-duplicates 0 / 0 of 746.
  - H10 SigLIP 2, ArtELingo-28: NO-GO (inconclusive). P4 fails (n 197, n* 124; x = 0.263, x95 0.239), P6 passes
    (n 0). Level 2 not tested. No boundary case in any read.
- **Agreements (C13 phase 2):** H6 and H7 166 compared, H8 to H10 171; 0 differences each. H8 selection replay
  rerun: true, 136 compared (the seven quantity labels added), margins not compared
  (`F/rederive/rd9_out/rd9_H8_selection_agreement_rerun.json`).
- **Descriptive passes** (`F/results/descriptive_<Hn>.json`): H7 2817 s, H8 1264 s (rerun after the fix), H6 4427 s,
  H9 3862 s, H10 1783 s (machine work ended 12:41).
- **Agent defaults of this chat** (for the report's brief): fix fx-desc-a28 (the descriptive pass refused every real
  ArtELingo-28 read: started mode `a28` vs pass mode `held`; plus a seed-key mismatch for H9/H10; Codex, merged
  10b5a7b, 110 tests, 3 of 3 mutants, controller read the diff; `F/fx_desc_a28_handback.md`). H6 and H7 descriptive
  passes started before H9's verdict (they had all their paired reads H5 to H7). Running past 11:00, CPU only (user
  told at 08:56; no answer). The run chat 1 defaults are in the run 2 handoff, §3.
- **Provenance note:** the H6 and H7 descriptive JSONs record the runner SHA at write time (`run_r9_descriptive.py`
  :530), so they show the 10b5a7b bytes though they started on b70502b; the change is in ArtELingo-28-only branches.
- **Report draft:** all verdict cells, H6 to H8 descriptive cells filled; left: `[[TODO DESC H9]]`,
  `[[TODO DESC H10]]`, one storage `[[TODO DESC]]`, `[[TODO BRIEF]]`. Codex session for a resume:
  `01a13938-3c33-7130-a597-362deecaaa3b` (its context is large; a fresh job with the draft and the ledger is fine too).
- **Storage:** `.scratch/r9-run/storage_table.md` (43 rows, 34.2 GB, 21.1 GB recommended for deletion by the rows;
  system disk 196 GB free). This chat's items are on `.scratch/pending_deletions.md`. The `rd9_out` agreement JSONs
  are bound by the verdicts' SHA-256: they stay even if the bundles go.

## 4. Open points for the next step

1. **Finish the draft** (Codex, `gpt-6.1-sol` medium): the H9 and H10 descriptive cells (paired H8/H9/H10 reads now
   complete), the storage line (pointer to the pending list; the decide chat decides), the results brief (template
   order, plain English, our reading and recommendation folded; the agent defaults listed), the title line. Fix the
   glossary link (relative, not `/project/...`). Under about 15 KB.
2. **User-read report** `docs/user_read/2026-10-14_aff_scope.md` (Codex or Sonnet), its advice folded away.
3. **Checks:** claim review of the auto report (worker, `sonnet` high): re-derive each load-bearing number from the
   files in `notes_for_controller.md`. User-read report: cold read (`haiku`) and fidelity check (`sonnet`) in parallel.
4. **Commit:** the report to `docs/reports/auto/v2/2026-12-02_aff_scope.md`, its row in `docs/reports/reports_sum.md`
   (`python scripts/check_reports_sum.py`, `docs-check`); the held ledger's episode and report cells for H6 to H10 and
   the "CVPR budget status" line; push.
5. **Open the next loop:** the handoff for `r10 decide` (it reads the results brief first, decision C; then the
   wrap-up with the storage summary), then `loop-next <handoff> --label "r10 decide" --new-loop --notify`. Commit
   `chats.tsv` and `briefs.md`, push, stop.
6. For the report, from the build: B1 LOW-1, B1 LOW-4, A1 L4 (the user decides the "keeps the same sentence" wording
   under D9; H9's claim text uses it).

## 5. Code and pitfalls

- No more runs are planned. Never write over a result file; the descriptive and agreement files are exclusive.
- Descriptive JSONs are 1 to 4 MB: read keys with python, never print them whole.
- Codex: `codex e --dangerously-bypass-approvals-and-sandbox ...` first on the line (`codex-jobs.md`).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours (`.scratch/` is gitignored).
- On disk over 1 GB: `F/rederive/rd9_out/` (11 GB); see the storage table.
