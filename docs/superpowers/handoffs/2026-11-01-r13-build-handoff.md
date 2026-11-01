# Handoff: build r13 (c* fine-tuned on Affection's train photos) from its approved spec, then open the run chat

## Brief
- Done: r13 decided, specced, rule-reviewed (Opus xhigh, all fixes applied) and approved from the brief; no container restart.
- State: frozen c* is below its teacher on Affection test (AMI −0.0282, mAP −0.0190); r13 fine-tunes it on 72,255 unlabeled train photos minus about 2,328 near-copies of held photos.
- Next: tickets, build and final review of the spec's code; then open the run chat.
- Blocked: nothing (PolyArt waits for the user's link; not in r13).

> Written 2026-11-01 15:40 by "r13 decide 2". Loop step reached: step 6 done (spec approved 15:33), step 7 next.
> Direction log entry: [2026-10-31](../direction_log.md) (Read, Meaning (C), Decided (A), Idea (B), approval).

## 1. Read in this order

1. The spec, from §1 on: [2026-10-31-photo-finetune-design.md](../specs/2026-10-31-photo-finetune-design.md). Its
   decision rule is §6 with §3.4 and §3.5; it inherits measurement, the read-out menu and its selection rule from the
   [r12 spec](../specs/2026-10-30-affection-confirmation-design.md) (§3.3, §3.4, §6).
2. `~/.claude/references/matt-chain.md` (the build chain), then `~/.claude/references/codex-jobs.md` before any Codex job.
3. The facts brief, `src/test/20261031_photo_finetune/pf_facts_brief.md` (Codex, 2026-10-31, checked in the decide
   chat). It names c*'s training code, the extractor and r12's protocol code with line numbers. The spec governs where
   they differ.
4. The rule review, `src/test/20261031_photo_finetune/pf_rule_review.md`: the near-copy evidence (M1) and the code
   pointers behind gates (iv) and (vi).
5. r12's round log, `src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md` (stage timings,
   the extractor's run).

Skills: `to-tickets`, `implement-spec` (per `matt-chain.md`).

## 2. Decided by the user, not to be reopened

All in the spec's §4 (D1 to D8) and the direction log, 2026-10-31. The spec was approved from its brief on
2026-11-01 at 15:33, with no container restart. The near-copy cut at 0.85 (D12, changed by the rule review) was in
the brief's "Check these" and is approved.

## 3. Where things stand

- Default Qwen store `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/`: all 85,007 images and 526,749
  explanations, train included. Instructed views (`qwen3vl_emb_2b_aspect_prompts/emotion_focus/`): validation and
  test images only (12,752). No Affection instructed text view exists; the train and validation explanations are new.
- r12's outputs: `/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs/` (`stage0b/ranges.json`,
  `stage1/scores.json`, exports, per-query outputs). c*'s graphs: r10's `outputs/graphs/{full,sample}/` under
  `/data/SSD2/cosir_r10/20261027_recipe_sweep/`.
- Run mode: night. GPU check at 2026-10-31 15:40: driver 595.99.02 answers, module and library match, idle, no lock.
- The run chat estimates the end time at its start: heavy GPU about 1.5 h, light GPU about 1.2 h, CPU about 2 h.
  Monday 11-02 is a weekday, so the heavy part runs before 10:00 or after 19:00 and the run ends by 11:00
  (`shared-resources.md`).

## 4. Open points for the next step

None for the user. The build decides code and tickets; anything that would change a number goes in the handoff to the
run chat, not to the user.

## 5. Code and pitfalls

- New round folder `src/test/20261031_photo_finetune/` (it holds the facts brief and the review), module prefix `pf_`.
  The auto report's sequence date is 2027-01-01.
- c*'s checkpoints: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/weights/k11=40/emotion__s{0..4}.pt`; the
  sidecar's `dev_size=1489` and "buddy+own-caption" strings are stale (facts brief).
- The training code reads only ArtELingo (`ct2_data.split2`, `ct2_d2.load_training_data`); an Affection provider and
  an emotion-only cohort are new code. `ac_extract.py` marks every non-valtest key list as a smoke; train and
  validation text extraction needs that policy changed. The r11 and r12 protocol code checks c*'s weight hashes
  against `rt_data.CSTAR_ROOT`; the photo model gets its own root (spec §3.4; only gate (iii) uses c*'s).
- Keys are `<source>/<image_name>`; 1,640 file names repeat across sources, so never join by name or by store
  position. The near-copies are mostly Emotional Machines ↔ FI (the same Flickr photo at two sizes).
- 4,422 of 447,680 train explanations are on "ignored" rows; they stay out of E, D and the caption inputs.
- Verdict code (the near-copy rule, the graph builder, the sweep score and pick with row 9, D9's ties and failed rows,
  the outcome code) gets unit tests and an Opus `high` ticket review; the rest gets the controller's diff read and one
  end-to-end smoke run (`agent-routing.md`, "Builds").
- Heavy GPU (train image and text extraction) belongs to the run chat. Light checks (gate (i)'s 200 images and texts)
  may run in the build, under the GPU lock.
- Unattended: delete nothing; list items on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of this chat's (older untracked files in `archive/` and `src/test/` predate it).
- On disk over 1 GB: nothing new.
