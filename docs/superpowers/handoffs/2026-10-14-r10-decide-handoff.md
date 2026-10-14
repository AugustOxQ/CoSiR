# Handoff: r10 decide: show the r9 results brief and get the user's reading (decision C), wrap up r9, then choose the next direction

## Brief
- Done: r9 run closed. Both r9 reports are committed (auto and user-read); the held ledger rows H6 to H10 are filled.
- State: H6 Qwen and H7 SigLIP 2 GO; H9 Qwen full claim on ArtELingo-28; H8 CLIP and H10 SigLIP 2 inconclusive.
- Next: show the results brief without our reading; record the user's reading; then the wrap-up and decision A.
- Blocked: two choices wait for the user: the SigLIP 2 refit (keep or drop) and the wording of H9's claim.

> Written 2026-10-14 13:08 by `r9 run 3`. Loop step reached: r9 step 9 done; r10 starts at step 1 (decision C).
> Direction log entry: none yet for r9's results; the r10 decide chat writes it at the wrap-up (the r9 decide entry
> is [2026-10-13 · Round 8](../direction_log.md)).

F = `src/test/20261202_r9_scope`.

## 1. Read in this order

1. The auto report's results brief: [2026-12-02_aff_scope](../../reports/auto/v2/2026-12-02_aff_scope.md). Show the
   user the brief **without** the folded "Our reading and recommendation" block; ask for their reading in a sentence
   or two; then open ours and say where the two differ (`research-loop.md`, "Decision C").
2. The user-read report: [2026-10-14_aff_scope](../../user_read/2026-10-14_aff_scope.md) (its advice is folded the
   same way). Offer it to the user before step 1 if they want the longer read.
3. For the wrap-up: the storage table `.scratch/r9-run/storage_table.md` and `.scratch/pending_deletions.md`.
4. Only if needed: the report's appendix `F/r9_report_tables.md`, the round log `F/20261202_r9_scope_log.md`
   (sections "r9 run chat 1" to "r9 run chat 3"), the claim review `.scratch/r9-run/claim_review.md`.

## 2. Decided by the user, not to be reopened

- All of the r9 spec §4 (D1 to D34) and the rule wording (SHA f9b07b34, 2026-10-13).
- No more AFF tuning (user, paper chat). Stage 3 (CUB, SemArt, Affection reads) gets its own decide chat; Affection is
  dropped if no data comes by 2026-10-24 (spec §2).

## 3. Where things stand

- **Verdicts** (`F/results/held_verdict_<Hn>.json`; ledger rows H6 to H10 filled with the verdict SHA-256):
  - H6 Qwen3-VL-Embedding-2B, ArtELingo held: GO, all seven checks; AFF − B′(A0) +0.63 [+0.49, +0.76] R@1 points.
  - H7 SigLIP 2: GO, all seven; +0.51 [+0.36, +0.66]. Round 6 CLIP for comparison: +0.60.
  - H9 Qwen, ArtELingo-28: full claim; +0.37 [+0.23, +0.51].
  - H8 CLIP and H10 SigLIP 2, ArtELingo-28: NO-GO (inconclusive): level 1 P4 fails (+0.06, +0.15; detectable
    margins 0.213 and 0.263); P6 passes (+3.23, +3.51); level 2 not tested.
- **Descriptive (rule §8, no verdict):** AFF − B′(A1) is −0.52 to +0.13 over the five reads (interval below zero on
  H6, H8, H9); style × genre stays below B′(A0) in every read; on ArtELingo-28 the margin over B′(A0) is larger on
  the 248 held paintings than on the 498 new ones, on all three backbones.
- **Agent defaults of the run** (all in the report's Deviations): the SigLIP 2 refit (a2, night mode); the
  descriptive-pass fix fx-desc-a28 (no verdict touched); CPU work past 11:00 (user told 08:56); the H8 selection
  agreement rerun; copied replay steps; the smoke guard and inf-writing fixes. This chat (run 3): the report's last
  parts (Codex), the results brief (controller), the appendix `F/r9_report_tables.md`, the ledger cells.
- **Checks:** claim review of the auto report (Sonnet, high): PASS WITH NOTES, about 700 values re-derived, 1 MEDIUM
  and 6 LOW, all fixed (`.scratch/r9-run/claim_review.md`). User-read report: cold read (Haiku) and fidelity check
  (Sonnet, FAIL: 1 HIGH, B′(A0) called the strongest condition-free scorer; 3 MEDIUM), one fix wave, no re-check.
- **Commits:** reports, appendix, index row, ledger cells and round log in bcf80fb (pushed).

## 4. Open points for the next step

1. **Key, decision C:** the user's reading of the r9 results (record it in the direction log, "Meaning (C)").
2. **Key, the user's: SigLIP 2.** Keep the refit (a2, agent default, recommended in the brief) or take (b): report "the
   fixed recipe does not run on SigLIP 2" and drop H7 and H10 from the claims.
3. **Key, the user's: H9's claim wording** (final review A1 L4, spec D9): H9's claim text uses "keeps the same
   sentence"; the user decides whether that wording stands, including after a NO-GO on ArtELingo.
4. **Wrap-up:** commit what is left, the direction log entry and its "Now" paragraph, the storage summary (34.2 GB in
   43 items; 21.1 GB recommended for deletion; the 10.4 GB re-derivation bundles deferred until the report is
   committed, which it now is; the `rd9_out` agreement JSONs stay, bound by the verdicts' SHA-256), `wrapup-check`,
   `disk-check`, `sync.sh`.
5. **Decision A:** the brief's three options (stage 3 decide; round 9 into the paper; a descriptive look at B′(A1) and
   the ArtELingo-28 shortfall). The CUB and SemArt probe notes for stage 3: `F/results/probes/{cub,semart}/probe_note.md`
   (machine-written, not read yet).
6. Open from the side checks, for the user's weekly meeting: are aspect labels needed in training; the k-shot entry in
   the direction log (`2026-10-14-side-checks-close-handoff.md`).

## 5. Code and pitfalls

- No more r9 runs; never write over a result file. Descriptive JSONs are 1 to 4 MB: read keys with python.
- `F/results/` is gitignored: the verdict, pass and descriptive files live only on disk (the ledger and the report
  carry their SHA-256). Keep them in the storage summary.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of ours (`.scratch/` is gitignored).
- On disk over 1 GB: `F/rederive/rd9_out/` (10.5 GB); the worktrees under `/project/CoSiR-wt/` (see the storage table).
