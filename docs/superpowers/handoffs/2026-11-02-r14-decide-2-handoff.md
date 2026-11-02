# Handoff: r14 decide 2: run the short grill on r14 (backbones, large sweep as analysis, EmoSet test), then write the spec

## Brief
- Done: r13 wrap-up (decision C recorded, 14.2 GB deleted, ledger, checks); decision A: r14 now, three parts on DAS6.
- State: r13 holds (photo model above its teacher on Affection, AMI +0.0172, mAP +0.0227); facts brief for r14 done.
- Next: the grill (step 5, short), at most three key questions per round, answer first; then the spec (step 6).
- Blocked: nothing.

> Written 2026-11-02 03:00 by "r14 decide". Loop step reached: after decision A and the step 3 side jobs (dataset
> check, facts brief); step 5 (the grill) next, then step 6. Direction log entry:
> [2026-11-02](../direction_log.md) (its "Idea (B)" line is written after the grill). Cut because the chat passed
> 200k tokens of context.

## 1. Read in this order

1. The direction log's "Now" and the 2026-11-02 entry: what the user decided this night, and who decided each part.
2. The facts brief, [bs_facts_brief.md](../../../src/test/20261102_backbones_sweep/bs_facts_brief.md) (Codex
   `gpt-6.1-sol` `medium`, checked by the controller: three spot checks held). It holds the backbone stores, the code
   paths fixed to Qwen, the r10 and r13 sweep grids, the r12/r13 protocol that EmoSet must copy, run costs and loop
   timing. Do not redo it.
3. The dataset note, [2027-01-02_photo_aspect_datasets.md](../../reports/literature/2027-01-02_photo_aspect_datasets.md):
   EmoSet facts and why FI is out.
4. `~/.claude/references/grill.md` (the grill protocol), then `spec-templates.md` for the spec.

Skills: `grilling`, `domain-modeling` (the grill runs as grill-with-docs).

## 2. Decided by the user, not to be reopened

- Decision C, the user's reading of r13 (2026-11-02, before seeing ours), in the direction log.
- One r14 loop now, before the 2026-11-10 freeze: type confirmation, medium loop, on DAS6 (user, after seeing the
  recommendation). Three parts:
  a. c* and the photo fine-tune on more backbones, with the frozen recipe;
  b. the large sweep, analysis only: c* and the photo model stay frozen, whatever it finds (user, after seeing the rerun
     cost: about 7 h of compute and a day of rework for r10 held to r13);
  c. the frozen r13 photo model tested on EmoSet-118K against the plain Qwen image and the teacher, with a WIN bar set
     before the run (user, after seeing the recommendation).
- A style or mood aspect on photos (Flickr Style, AVA) is parked: discuss after r14's main experiments if time
  remains, else after the abstract (user).
- DAS6: node402 (3 RTX A6000, 48 GB) is reserved until 2026-11-04 18:06; the user re-reserves on 11-04 and keeps one
  node with 3 GPUs until the main deadline (user).
- Storage of r13: all recommendations approved and done (user).
- Earlier, still standing: c* stays the paintings model; the licensed r13 wording is "level with or better than" the
  teacher; PolyArt waits for the user's link.

## 3. Where things stand

- r13 (report 2027-01-01, holds): photo model against the teacher on 8,501 Affection test photos, AMI 0.1838 against
  0.1666, +0.0172 [+0.0121, +0.0225]; mAP 0.3172 against 0.2945, +0.0227 [+0.0210, +0.0244].
- Backbones (facts brief Q1): ArtELingo stores exist for SigLIP 2, SigLIP 1 and CLIP L/14 (all splits). Affection has
  no SigLIP 1 or SigLIP 2 store and no CLIP L/14 train store; `extract_features.py` allows only CLIP B/32, Qwen and
  SigLIP 2 for `affection_local`. The earlier encoder tests (10-25, 10-26) used recipe R6-5, not c*: SigLIP 2, SigLIP 1
  and CLIP L/14 passed held emotion; no non-Qwen c* run exists.
- Code fixed to Qwen: the r10 sweep runner (`rg_full.ENCODER`), the r13 trainer (width 2048 assert,
  `pf_train.py:85`), r11's and r12's arms. The older generic student `rb_student.py` accepts `--encoder`.
- Run cost: full-scale c* 3.5 to 5.3 min per seed on the 3090; photo fine-tune about 2.3 min per run; the r10 sweep
  ran about 20 sample-scale runs per hour per A6000.
- EmoSet-118K: 118,102 photos, 8 Mikels classes (Affection's 9 minus "something else"), own train/val/test JSON;
  downloaded to `/data/SSD/emoset/EmoSet-118K.zip` (11.3 GB, complete: size equals the server's, 236,226 entries),
  not unpacked. Splits (read from the zip): train 94,481, validation 5,905, test 17,716; test classes from 1,661
  (disgust) to 3,014 (excitement). Image licence unclear (repo MIT for code).
- Loop timing: r12 took 28 h and r13 42 h from decide start to report. Abstract 2026-11-10, paper 2026-11-16.

## 4. Open points for the grill

Key (answer first, at most three per round):
- **Backbones and bars:** which encoders (the 10-27 list was SigLIP 2, SigLIP 1, CLIP L/14), on which tasks (clustering,
  retrieval, both) and data (ArtELingo three aspects; Affection photo fine-tune). The bar per backbone: against its own
  plain image, its own PercepT arm, and the Qwen teacher (a stronger, different backbone).
- **EmoSet protocol:** read-outs carried from Affection validation (fully frozen) or picked on EmoSet validation;
  cluster count (Affection used K=20; EmoSet has 8 classes); teacher wording (r12's artwork pick, or photo); a
  near-copy check against Affection train; which models enter (photo model; frozen c* and the backbone photo models as
  descriptive rows?).
- **Sweep scope and budget:** what it should show (sensitivity of c*'s knobs, structural ablations such as c* without
  the caption endpoints or the direct teacher, photo fine-tune sensitivity: neighbours, learning rate, epochs), how many
  runs fit on node402 before 11-04 18:06, seeds, and on which split it is read (no selection reads held data).
- **Timing:** what must be done before the abstract (11-10), what may finish by the paper (11-16).

Minor (one batch with recommended answers, to the spec's "Check these"): round folder and module prefix (`bs_`,
checked free), store extraction on the local GPU (short work), the EmoSet unpack and loader adapter, near-copy cut 0.85
as r13.

## 5. Code and pitfalls

- The paper and abstract use the licensed r13 wording, not "beats the teacher"; EmoSet's WIN bar is what can license
  "better than".
- FI is an Affection source, so it cannot be a second test set.
- Round folders go on `sys.path`: use the prefix `bs_` for new modules in `src/test/20261102_backbones_sweep/`.
- Overnight local runs end by 11:00; long heavy local GPU runs not on weekdays 10:00 to 19:00. DAS6 runs at any time.

## 6. State at handoff

- Running: nothing (the EmoSet download finished before the cut; no subagents, no Codex job, no GPU lock).
- Uncommitted: nothing of this chat's.
- On disk over 1 GB: `/data/SSD/emoset/EmoSet-118K.zip` (11.3 GB, keep: r14 needs it; the unpacked copy will follow,
  then the zip can go at r14's storage summary).
