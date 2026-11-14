# Handoff: write r30's short auto report and half-page user-read report, then get the user's reading of r30

## Brief
- Done: r30 ran (2026-11-14 02:47 to 02:50, after r31 ended): one UMAP map fitted on c*, the valence check, and an islands check.
- State: long DEC's two emotion islands are the k = 2 groups (AMI 0.879) and split by subject (people against scenes), not valence; c* has the same split without the gap.
- Next: the short auto report (Codex draft, Claude claim review), the half-page user-read report and its two checks; then the user's reading.
- Blocked: nothing.

> Written 2026-11-14 02:53 by "r30 decide" (cut at the 200k context meter). Loop step reached: r30 step 8 done, step
> 9 next (small loop). Direction log entry: [2026-11-13 · r21 read](../direction_log.md) (r30's entry comes after the
> user's reading).

## 1. Read in this order

1. The round log, `src/test/20261113_r30_dec_followup/20261113_r30_dec_followup_log.md`: why, design, build, run,
   results with the numbers, and the add-on. It holds everything the reports need.
2. The result files in `/data/SSD2/cosir_r21/20261111_dec_heads/r30_followup/`: `valence.md` / `valence.json` (AMIs are
   means of k-means states 0 to 2; group tables use state 0), `islands.json`, the figures.
3. For context only: r21's auto report `docs/reports/auto/percept/2027-01-17_dec_heads.md` (results brief) and the
   direction log entry "2026-11-13 · r21 read" (the user's reading that started r30).
4. Before writing: `~/.claude/templates/auto_report_template.md`, `~/.claude/templates/results_brief.md`,
   `~/.claude/rules/report-writing.md`, `~/.claude/rules/user-read-reports.md`,
   `~/.claude/references/user-read-report.md`, `~/.claude/references/ste-writing.md`, and `codex-jobs.md` for the
   Codex jobs.

Skills: none.

## 2. Decided by the user, not to be reopened

- r21 is read and closed (option b): DEC stays out of the main recipe and is an ablation row (2026-11-13, direction log).
- r30 = one UMAP map fitted on c* with each arm projected into it (user's yes after seeing ours) and the user's idea:
  are long DEC's two emotion groups positive against negative emotions? Run after r31 ended (user, 2026-11-13).
- Storage of r21: decided and done (disk ledger rows of 2026-11-13).

## 3. Where things stand

- **Shared map:** the four test arms (λ = 0 control, three DEC forms) land almost where c* is, on all three aspects;
  long DEC on validation also keeps c*'s layout. Placement with UMAP transform can make a large change look smaller,
  which is why the islands check below works in the 2,048-d head space.
- **Valence check (k = 2 groups of the emotion head, validation):** valence AMI 0.0459 for long DEC against 0.0402
  for c*; genre AMI 0.3255 against 0.3136; style 0.0751 against 0.0622; the gap in mean positive caption share 0.137
  against 0.152. Test arms: all within 0.0388 to 0.0399 valence AMI (c* 0.0389).
- **Islands (agent add-on, exploration):** the two islands of r21's long-DEC emotion map equal the k = 2 groups (AMI
  0.879; 3,363 / 2,789 against 3,362 / 2,790). Group A: portrait 0.39, genre painting 0.22, religious painting 0.19;
  positive caption share 0.563. Group B: landscape 0.43, abstract 0.17, cityscape 0.15, still life 0.09; positive
  0.700. c*'s k = 2 groups have the same make-up, with no gap in its map.
- **Our reading (keep it folded in the brief):** the guess does not hold as stated. The main axis is subject
  (paintings of people against scenes and objects), already in c*; long DEC pulls the two halves apart. Valence follows
  the subject split only in part. This also fits r21: long DEC lowers emotion AMI because it sharpens a subject axis.
- **Options for the brief** (no recommendation there): a. nothing more; the DEC line stays closed. b. one sentence in
  the paper's ablation text: long DEC splits the emotion head by subject, which explains why DEC does not help emotion.
  c. check whether the subject axis comes from the instructed emotion embedding itself (k = 2 on the instructed
  Qwen3 emotion view and on plain Qwen3; saved features, a few CPU minutes). Our recommendation (folded): a, plus b if
  the paper's ablation text discusses DEC on emotion.

## 4. Open points for the next step

1. **Step 9, unattended:**
   - The auto report `docs/reports/auto/percept/2027-01-20_dec_emotion_split.md` (CoSiR's sequence date; check that
     2027-01-20 is still free in `docs/reports/reports_sum.md`), short (a small loop), under about 8 KB. Spec: none
     (design in chat, the round log's "Design" section); decision rule: none (descriptive). Draft by Codex
     (`gpt-6.1-sol` medium), then Claude's claim review. Figures are already in
     `docs/reports/assets/2027-01-20_dec_emotion_split/`. Add its index row, then run
     `python scripts/check_reports_sum.py` and `docs-check`.
   - The user-read report `docs/user_read/2026-11-14_dec_emotion_split.md`, half a page; key figure
     `docs/user_read/assets/2026-11-14_dec_emotion_split/islands_emotion.png` (already copied). Its advice folded.
     Two checks in parallel on Codex (cold read `gpt-6-luna` medium, fidelity `gpt-6.1-terra` medium), one fix wave.
   - Commit and push both, then send the user-read report to the phone (SendUserFile, proactive).
2. **Key, decision C on r30:** when the user is back, show the results brief without the folded block, ask for their
   reading in a sentence or two, then open ours and say where the two differ. Record it as a new direction log entry
   ("r30 read") and update "Now" (its r21 sentence names r30 as waiting for r31: say r30 is done).
3. **Minor:** the storage summary (below) and the chat end per `routines.md`.

## 5. Code and pitfalls

- Numbers come from `valence.json` and `islands.json`, never from the round log's tables.
- c* has no island gap: its "islands" numbers come from a 2-way k-means of a connected map (AMI 0.572 with its k = 2
  groups); do not call them islands in the reports.
- The figure `islands_emotion.png` uses r21's own per-panel UMAP fits (what the user saw), not the shared map.
- Terms: "long DEC" = DEC plain, λ = 10, 50 DEC epochs, validation, seed 0 (`dec-plain_l10_w5_S1`). "Positive caption
  share" = the share of a painting's captions with amusement, awe, contentment or excitement. The glossary calls the
  teacher "the instructed embedding".

## 6. State at handoff

- Running: nothing of r30.
- Uncommitted: nothing of r30.
- On disk: `/data/SSD2/cosir_r21/20261111_dec_heads/r30_followup/` 12 MB (keep: the reports cite it); the smoke of the
  r30 build, `/tmp/claude-0/-project-CoSiR/3342c361-e970-4b68-9d8d-39be9b6cd28b/scratchpad/r30/` 1.9 MB (can go); Codex
  listed `/tmp/r30_umap.log`, `/tmp/r30_valence.log`, `/tmp/r30_plot.log` (smoke logs, can go). Nothing over 1 GB.
