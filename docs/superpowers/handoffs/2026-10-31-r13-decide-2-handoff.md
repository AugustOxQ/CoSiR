# Handoff: r13 decide 2: the rule review of the r13 spec, the user's approval from the brief, then the build chat

## Brief
- Done: decision C on report 12-31, the r12 wrap-up, decision A and the grill for r13; the r13 spec is a draft.
- State: r13 fine-tunes c* on Affection's 72,255 unlabeled train photos (9-row sweep on validation, then 5 seeds) and tests it against its teacher with r12's rule.
- Next: a fresh Opus xhigh worker reviews the spec's rule (C7); apply fixes; paste the brief for the user's approval.
- Blocked: nothing; PolyArt waits for the user's link.

> Written 2026-10-31 15:45 by "r13 decide" (cut at the context meter, about 210k). Loop step reached: r13 step 6, the
> spec drafted; the rule review and approval remain. Direction log entry:
> [2026-10-31](../direction_log.md) (Read, Meaning (C), Decided (A), Idea (B) with the grill's answers).

## 1. Read in this order

1. The [r13 spec](../specs/2026-10-31-photo-finetune-design.md), draft, committed at `f0c77c2`. Its §4 holds every
   grill answer with who decided; do not ask them again.
2. The direction log's 2026-10-31 entry (the user's reading, the decisions).
3. Only for the review brief: the [r12 spec](../specs/2026-10-30-affection-confirmation-design.md), which r13
   inherits (§3.3, §3.4, §6), and [report 12-31](../../reports/auto/percept/2026-12-31_affection_confirmation.md).

Skills: none.

## 2. Decided by the user, not to be reopened

- Decision C (2026-10-31): "above plain is good but lower than teacher is not quite okey". Decision A: report 12-31's
  option 1, now, before the 11-10 freeze; medium loop, step 3 skipped; the large sweep moves to r14.
- The grill (spec §4, D1 to D8): fine-tune c* (A3) with a from-scratch row; r12's bar (B1), LEVEL in the new model's
  own seed range; a 9-row sweep on Affection validation (C2) with the grid of §3.5; the pick by the weaker teacher gap
  in c*'s r12 seed ranges (D2); a failure is reported as it is, with no second sweep row on the test; the minor batch
  accepted; no plan page.
- Storage of r12: all five deletes done and recorded in the disk ledger (2026-10-31).

## 3. Where things stand

- The spec's agent defaults that the review should test hardest: D12 (train photos with cosine ≥ 0.98 to a
  validation or test photo leave training; Visual Genome holds COCO images), D9 (ties within 0.5 r), D13 (the fixed
  sweep read-out), D14 (stage 3 as a claim test), stage 0a's gates (i) to (iv).
- Facts brief for the grill (Codex, checked):
  `/tmp/claude-0/-project-CoSiR/38810575-7ec5-48ca-88f0-feaecce1dc60/scratchpad/facts_brief.md` (temporary; its main
  points are in the spec). c*'s emotion graph reads the instructed emotion view of the explanations, so the
  extraction covers the train explanations too (about 448,000), not only the images.
- r12's frozen c* on Affection validation: AMI 0.1427 to 0.1461 per seed against the teacher's 0.1719; mAP 0.2858
  (centering plus k = 10, cross-fit) against 0.3044 (`/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs/`
  `stage0b/ranges.json`, `stage1/`).
- DAS6: node402 free (3 GPUs, about 100 h left at 14:20); node401 and node408 do not answer. The spec runs locally.
- `routing-check` repeats only the items the user deferred to a tooling chat after 11-09.

## 4. Open points for the next step

- Key: the rule review (C7). A fresh Opus `xhigh` `worker` reviews §3.4, §3.5 and §6 with the rest of the spec as
  context; apply must-fix and should-fix items; record the reviewer line in the spec header; amend the brief where a
  fix changes it (hard rule 5).
- Key: the user's approval from the brief (paste the brief with the file link). Then set "Status: approved", commit
  and push.
- Before the build chat opens: the GPU check and the container-restart question (`loop-chats.md`, "How to cut",
  step 3). Then `loop-next <build handoff> --label "r13 build" --unattended`.
- The loop has a main chat from r12's run 2 (`loop-next` reopens it if it is gone).

## 5. Code and pitfalls

- New round folder `src/test/20261031_photo_finetune/`, module prefix `pf_` (spec §9). The auto report's sequence
  date is 2027-01-01.
- c*'s checkpoints: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/weights/k11=40/emotion__s{0..4}.pt`.
  Each holds both sides' trunk and the emotion head (`ms_train.TrunkHead`); the saved sidecar's `dev_size=1489` and
  "buddy+own-caption" strings are stale (the facts brief).
- The training code reads only ArtELingo (`ct2_data.split2`, `ct2_d2.load_training_data`); an Affection provider
  and an emotion-only cohort are new code. r12's extractor (`ac_extract.py`) marks every non-valtest key list as a
  smoke; train extraction needs that policy changed. The r11 and r12 protocol code checks c*'s weight hashes against
  `rt_data.CSTAR_ROOT`; the photo model needs its own export root.

## 6. State at handoff

- Running: nothing. GPU idle at 14:25, no lock.
- Uncommitted: nothing (this handoff, `chats.tsv` and `briefs.md` are committed after `loop-next`).
- On disk over 1 GB: nothing new.
