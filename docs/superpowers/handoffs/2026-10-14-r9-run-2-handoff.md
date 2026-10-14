# Handoff: r9 run 2: launch the H6 read now, then its agreement and verdict, then H8, then H9 and H10, the descriptive passes and the reports

## Brief
- Done: SigLIP pick (a2, agent default); stage 0a, 0b, stage 2 on Qwen and SigLIP; all five pre-reads passed; H7 = GO.
- State: H7 (SigLIP 2, ArtELingo held): GO, all seven checks pass (every n = 0), re-derivation agrees 166/0.
- Next: H6 ledger row and read (about 37 min), then H8, then H9 and H10; the reports.
- Blocked: nothing. Machine work runs to about 12:30, CPU only (agent default; user told, no answer yet).

> Written 2026-10-14 09:00 by `r9 run` (cut: context over 300k). Loop step reached: step 8 (run), reads H7 done, H6
> launch next. Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

F = `src/test/20261202_r9_scope`.

## 1. Read in this order

1. Sections 3 to 5 below.
2. The run chat's notes, now at the end of `F/20261202_r9_scope_log.md` ("r9 run chat 1"): every step, time and
   number of this chat.
3. `F/DECISION_RULE.md` §4 to §8 (bars, pre-read, read, reported-after); `F/rederive/rd9_commands.md` ("Phase 2");
   the apply step's header (`F/run_r9_apply_rule.py` lines 1 to 30).
4. The two fix reviews, for the report: `F/final_review/rr_fx_smoke.md`, `F/final_review/rr_fx_inf.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34) and the 2026-10-13 rule wording (SHA f9b07b34).
- Reads run locally (rule §7.7, D12); the user asked about DAS6 at 07:5x and was told why not.
- Running past 11:00 is an agent default (user told at 08:56, no answer yet): CPU only; pause after H8 if the user asks.

## 3. Where things stand

- **Code:** main @ 2ac49c3, pushed. No round-folder code may change now: every pre-read and smoke record binds the
  module bytes. A change would need new pre-reads (`--fix 3` is the last free generation name).
- **Agent decisions taken in night mode (record each in the report's brief, "agent default"):**
  - SigLIP (a2): merged `r9-siglip-a2` (5bc5ad2). SigLIP caption k-means seed 42 gave a 1-row group, refit with seed
    43 (smallest 939); recorded in `partitions.npz.record.json` `kmeans_restarts`. (b) stays open to the user.
  - Stage 0b: the new step 07 outputs were byte-equal to the old replay's, so steps 08 to 11 were copied from the
    old replay instead of rerun (`results/replay/clip_b32_a2`). Stage 0b PASSED (case 1, all exact).
  - Fix fx-smoke (78aac8f; Codex, Opus review PASS WITH NOTES): the CLIP pre-read smoke runs H6's path on CLIP; the
    apply step's backbone guard now exempts smoke only.
  - Fix fx-inf (6a282d2; Codex, Opus review PASS WITH NOTES): Qwen's pinned `wang` λ on tune half 1 is infinite;
    the descriptive JSON writes it as "inf". Pre-read generations `--fix 1` to `3`, newest first (the old code
    allowed one rerun per read; rule §6 sets no limit).
  - H6's `--fix 2` pre-read was killed (exit 137, likely out of memory, four pre-reads at once); rerun as
    `--after-crash` (crash1). Run at most two Qwen- or SigLIP-size jobs at once.
- **Stage results:** stage 0a PASSED on fbd578a (131 items, 0 differences). Qwen stage 2 rerun byte-equal to the
  run before the merge (pins 57a9cc72); SigLIP stage 2 ready (pins 45a84e72), not switched off, gate share 0.496.
- **Phase 1 (C13):** agreements true: CLIP 169/0, Qwen 160/0, SigLIP 160/0 (`rederive/rd9_out/`).
- **Pre-reads, all passed, in `F/results/`:** H8 `preread_H8_fix2.json`, H6 `preread_H6_crash1.json`, H7
  `preread_H7_fix2.json`, H9 `preread_H9_fix2.json`, H10 `preread_H10_fix2.json`. The current canonical smoke
  record is `smoke_record_fix2.json` (F copy committed 68ce498).
- **H7 done:** ledger row (fc90917); read 2202 s; started copy committed (cc92bf0); phase 2 105 s, agreement true
  (166 compared, 0 differences); `held_verdict_H7.json` = **GO**, P1 to P7 all pass, every n_j = 0.
- **Seed-42 development numbers** (told to the user; selection rows; AFF minus comparator, R@1): Qwen P1 to P7 4.83,
  4.65, 0.86, 0.70, 0.94, 2.76, 2.78; SigLIP 6.31, 5.87, 0.85, 0.50, 0.80, 3.04, 3.05; CLIP (R6) 6.18, 5.76, 0.80,
  0.70, 0.74, 3.11, 3.01. B1 is level with AFF on Qwen (19.25 vs 19.11) and SigLIP (19.26 vs 19.24); reported only.

## 4. Open points for the next step

1. **H6 now:** `python .scratch/r9-run/ledger_row.py H6` (writes the row with the runner and pins SHAs), commit and
   push, then `run_r9_held.py --backbone qwen3vl_emb_2b --read H6` (about 37 min, CPU; `CUDA_VISIBLE_DEVICES=`,
   8 threads). Commit `F/held_started_H6.json`. Then, as for H7 (the exact chain is in the notes, 08:55): 
   `rd9_phase2.py --read H6`, `rd9_compare.py ... --reported-sensitivity results/sensitivity_H6.json --out
   rd9_out/rd9_H6_agreement.json`, `run_r9_apply_rule.py --read H6`.
2. **H8** after both H6 and H7 verdicts exist (rule §7.3): ledger row, `--backbone clip_b32 --read H8`, phase 2
   (`--language-flags` and `--near-duplicates` defaults), agreement, apply.
3. **H9 and H10** after H8's verdict (H8's episodes); they may run at the same time (two jobs fit in memory).
4. **Descriptive passes** (rule §8) after all verdicts: `run_r9_descriptive.py --read <Hn>` for each, so the paired
   comparisons (H5/H6/H7, H8/H9/H10) are complete.
5. Ledger: fill each row's episode and report cells at report time; update the "CVPR budget status" line.
6. Post each verdict to the user in one line as it lands (verdict and check flags, no reading of ours); the user
   reads the results brief first (decision C) in the next decide chat.
7. Reports: the auto report `docs/reports/auto/v2/2026-12-02_aff_scope.md` with a results brief, and its user-read
   report; the report notes of the build handoff (B1 LOW-1, B1 LOW-4, A1 L4; the H8 selection agreement rerun is
   expected to give 136 items with margins not compared). Then the storage summary, wrap-up, and `loop-next --new-loop`.

## 5. Code and pitfalls

- Never edit an `r9_*.py` or `run_r9_*.py` now (see section 3).
- Held reads take about 37 min each (not the spec's 10 min). Pre-reads take 10 to 45 min.
- Agreement and smoke files are exclusive writes: a redo uses `_fix1` names (held) or the next generation.
- The held runner and apply step print no metric; read numbers only after the verdict exists.
- `subagent-watch --watch` in a background Bash while subagents run.

## 6. State at handoff

- Running: nothing (no read, no pre-read, no subagent, no Codex job). GPU idle since 07:10.
- Uncommitted: nothing of ours (notes in the gitignored `.scratch/r9-run/`).
- On disk over 1 GB: `F/rederive/rd9_out/` (11 GB, the phase-1 bundles; keep until the report). Every leftover of
  this chat is on `.scratch/pending_deletions.md` (rows dated 2026-10-14 09:05).
