# Handoff: r28 results for the paper chat: every SigLIP 1 cell of Tables 2 to 8 is in the numbers sheet; two flags for the text

## Brief
- Done: r28 scored or trained every SigLIP 1 cell the paper tables need, Table 5 included (four removals, five seeds); reports written.
- State: the removals follow Qwen3: without the instructed graphs, style AMI -0.210 (Qwen3 -0.186); without the relational term, genre AMI -0.067 (-0.071).
- Next: the paper chat checks two flags in its text: the Table 3 text replay tolerance and the close calls.
- Blocked: Figures 3 and 4 wait for the user's picks (the review page is ready).

> Written 2026-11-12 23:40 by the "r28 run" tab. Loop step reached: r28 steps 8 and 9 done. No next loop chat is
> opened: the label r29 belongs to the SigLIP-only adapter, and r28's results go to the paper chat ("paper plan 4"
> tab, 21:13). Direction log entry: [r28 reported](../direction_log.md).

## 1. Read in this order

1. The auto report's results brief: [2027-01-16_siglip_main](../../reports/auto/percept/2027-01-16_siglip_main.md)
   (the user-read report: [2026-11-12_siglip_main](../../user_read/2026-11-12_siglip_main.md)).
2. The numbers sheet `src/test/20261112_siglip_main/paper_cells.md`: each SigLIP 1 cell beside the Qwen3 cell and its
   source. Check it with `r28_cells_check.py --values-only` (152 value cells; PASS at 23:30). Its "now" cells are the
   .tex of about 20:00, before the paper chat moved the tables to SigLIP 1.
3. The run log `src/test/20261112_siglip_main/20261112_siglip_main_log.md` (newest lines last) and `decisions.md`.

Skills: none.

## 2. Decided by the user, not to be reopened

- SigLIP 1 is the main result; the comparators and claims as in the r28 run handoff, section 2 (2026-11-12).
- c*'s settings stay for SigLIP 1 (k = 40, learning rate 1e-3), after r28's settings check (user, after seeing the
  recommendation, 2026-11-12 22:50; paper README row of that date).

## 3. Where things stand

- **Table 5 on SigLIP 1** (removal minus c*, AMI at the class counts and plain-cosine mAP, ArtELingo test, five seeds;
  `records/ablation_siglip1.json`). The paper chat put these into `ablation.tex` at 23:15; the 36 numbers there equal
  the records (review part 2).

  | Removal | AMI E / S / G | mAP E / S / G | Qwen3 AMI (paper) |
  |---|---|---|---|
  | Instructed graphs | -0.031 / -0.210 / -0.007 | -0.020 / -0.143 / -0.001 | -0.033 / -0.186 / +0.010 |
  | Content correction | -0.011 / -0.079 / -0.004 | -0.006 / -0.069 / 0.000 | -0.016 / -0.068 / 0.000 |
  | Graph-positive loss | -0.033 / -0.164 / -0.006 | -0.018 / -0.118 / 0.000 | -0.029 / -0.140 / +0.008 |
  | Relational term | +0.001 / +0.006 / -0.067 | 0.000 / -0.001 / -0.114 | +0.001 / -0.002 / -0.071 |

  Each emotion and style loss of the first three rows is below c* in 5 of 5 seeds, as is the relational genre loss.
  The small genre changes of the first three rows have mixed signs over the seeds (1 to 4 of 5 below c*).
- **Flag 1, the Table 3 text directions.** The caption-tower replay on validation was accepted at float16 resolution
  (99.5% of values equal, the rest one float16 step, max 6.1e-5), not the brief's 1e-5: the stored exports are
  float16. An agent may not loosen a tolerance alone (constitution row 4), so this is the user's to accept. An
  independent reviewer check confirmed the inference path.
- **Flag 2, close calls** (wording in the text):
  - SigLIP 1 over SigLIP 2 on the validation pick: 0.3863 against 0.3856, ahead in 3 of 5 seeds.
  - Ours over CLAY on SigLIP 1 in style: seeds of ours above CLAY, AMI 5 / 4 / 5, mAP 5 / 5 / 5.
- **Final reviews:** part 1 (20:33) and part 2 (23:29) both PASS; every re-derived number matched exactly.
- **Not in r28:** Table 8's image times were not re-measured (r19 values); Figures 3 and 4 candidates exist with a
  review page, not reviewed by an agent (the user picks).

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of r28 (other tabs' files: see `git status`).
- On disk over 1 GB: `/data/SSD2/cosir_r28/20261112_siglip_main/cstar` (8.2 GB, the trained models; keep, in the disk
  ledger). Removal candidates for the user: `.scratch/pending_deletions.md`, rows "r28 run" (the smoke folder, 834 MB;
  old pick tables; the merged worktree `/project/CoSiR-r28-trainer` and branch `r28-trainer`).
