# Handoff: r15 decide: show the r14 results brief, take the user's reading (decision C), wrap up r14, then decide the next direction

## Brief
- Done: the r14 run finished. Both claim tests hold (GO). The auto report and the user-read report are checked and committed; main is pushed.
- State: EmoSet, photo model against the teacher: AMI 0.4571 against 0.4390, mAP 0.4838 against 0.4519 (both WIN). Teacher cells: 18 WIN, 5 LEVEL, 1 BELOW.
- Next: r15 decide: step 1 (the user's reading), the r14 wrap-up, then step 2.
- Blocked: nothing.

> Written 2026-11-02 17:38 by "r14 run 5". Loop step reached: step 9 done (verify and report); step 1 of the next
> loop next (decision C). Direction log entry: [2026-11-02](../direction_log.md) (the r13 read, which holds r14's
> decisions A and B; r14 gets its own entry, with "Meaning (C)", at the wrap-up).

## 1. Read in this order

1. The results brief of the auto report,
   [2027-01-02_backbones_sweep.md](../../reports/auto/percept/2027-01-02_backbones_sweep.md) (top section). At step 1,
   show it to the user without the folded "Our reading and recommendation" block, ask for their reading in a sentence
   or two, then open ours and say where the two differ (`research-loop.md`, "Decision C").
2. The user-read report beside it, [2026-11-02_backbones_sweep.md](../../user_read/2026-11-02_backbones_sweep.md),
   for the user; its advice is folded the same way.
3. The direction log's "Now" and the 2026-11-02 entry, for the wrap-up and step 2.
4. The run log [bs_run_log.md](../../../src/test/20261102_backbones_sweep/bs_run_log.md), only if a question needs
   run detail.

Skills: none for step 1. Step 5, if step 2 picks a medium or full loop, uses `grilling` and `domain-modeling`.

## 2. Decided by the user, not to be reopened

- The r14 spec's §4 (D1 to D21), approved 2026-11-02 04:13 (`c1377ff`). The verdicts follow its §6, unchanged, with
  no amendments.
- MMArt (PolyArt): the same split as ArtELingo, new paintings as the test set, its own emotion label set, aspects and
  teacher instructions as ArtELingo's; extraction after 19:00 (user, 2026-11-02; MMArt round log).
- Deadlines: abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23.

## 3. Where things stand

- **Part a (three more encoders: SigLIP 2, SigLIP 1, CLIP L/14): holds.** On each encoder, c* is a WIN over the
  plain image in all six painting cells and over matched PercepT in clustering. Each photo model is a WIN over its
  plain image on Affection test.
- **Teacher cells:** 18 WIN, 5 LEVEL, 1 BELOW (24 cells). The BELOW cell is CLIP L/14 emotion retrieval on paintings,
  -0.0013 [-0.0023, -0.0003]; it licenses no claim. One LEVEL cell, CLIP L/14 genre AMI, is below the teacher's point
  value: -0.0052 [-0.0103, -0.0000].
- **Part c (EmoSet test, 17,348 photos, frozen Qwen photo model): holds.** Photo model against the teacher: AMI
  +0.0181 [+0.0121, +0.0242], mAP +0.0320 [+0.0295, +0.0347]. Against plain Qwen: AMI +0.0653 (+0.0390 with the 368
  near-copies kept). Per emotion, the photo model is below the teacher on amusement (AP 0.3203 against 0.3471;
  diagnostic, now in the full report's caveats).
- **Part b (analysis):** complete, no verdict, names no better setting.
- **Reports:** both checked. The user-read report had a cold read (Haiku) and a fidelity check (Sonnet high: about 55
  numbers, none wrong, 6 minor findings), one fix wave and a humanizer pass. Run 5 changed the full report in three
  places: a Terms line, the amusement caveat, and option 2 (PolyArt data is now prepared).
- **MMArt (PolyArt):** a side chat, "r14 mmart run", extracts its seven views tonight, 19:00 to about 21:00, on the
  local GPU. It will replace "PolyArt waits for the user's link" in the direction log's "Now" itself. Test set: 16,782
  WikiArt paintings not in ArtELingo, before the near-copy cut.
- **Merge:** `r14-backbones-sweep` and `main` both at the last commit of this chat, pushed.

## 4. Open points for the next step

- **Key, decision C:** the user's reading of r14, recorded in r14's direction log entry ("user" or "user, after
  seeing ours").
- **Wrap-up items** (`research-loop.md`, "Wrap-up"):
  a. the r14 direction log entry and the "Now" rewrite (check first whether the MMArt chat has already edited "Now");
  b. the storage summary (below);
  c. `wrapup-check` (it prints three routing items that the user deferred to a tooling chat after 2026-11-09),
     `disk-check` (silent at 17:35), then `sync.sh`.
- **Key, decision A (step 2):** the next direction. The brief's options:
  a. freeze and write (no compute);
  b. the PolyArt (MMArt) test: data ready, features tonight; then a test design and one scoring round;
  c. a fourth encoder that no held test picked: about 10 runs on DAS6 and about 1 h of CPU scoring.
  Our recommendation, folded in the report: freeze and write; use the part b sheets as the paper's ablation.
- **Storage summary:**
  - New this loop: `/data/SSD2/cosir_r14/` (13 GB in all; `outputs/` 12 GB, which the reports rest on: recommend
    keep). It needs a disk ledger row. The new EmoSet and Affection encoder stores are under 1 GB each (EmoSet 421 MB).
  - Every row of `.scratch/pending_deletions.md`: the r14 build worktrees and branches (about 116 MB each, 17
    worktrees), node402 results and job folders (4.6 GB), node402 store copies (11.7 GB, `cluster clean` needs the
    user's yes), `outputs/regression` (1.7 GB), the smoke and probe pulls, and older rows.

## 5. Code and pitfalls

- **Licensed wording:** a LEVEL cell licenses only "level with or better than the teacher". The EmoSet cells are WIN.
  The one BELOW cell stays visible in the paper (our advice).
- **Another chat in this repo:** "r14 mmart run" commits to `r14-backbones-sweep`; commit only your own paths
  (`git commit -- <paths>`). It holds the GPU from 19:00 to about 21:00.
- `*.json` and `.claude/` are gitignored (`git add -f` for records). `git fetch .` fails here (dubious ownership):
  move a ref with `git update-ref` after `merge-base --is-ancestor`.
- **Codex budget at 17:36:** 5h window 47% left (resets 19:51), weekly 25% (resets 11-04 08:27). The ledger is up to
  date (`22f09b9` in `claude-config`).

## 6. State at handoff

- Running: nothing of this chat (no background job, subagent, Codex job, GPU lock or cluster job). The side chat "r14
  mmart run" waits for 19:00.
- Uncommitted: nothing of this chat in the project; `claude-config` clean and pushed.
- On disk over 1 GB: `/data/SSD2/cosir_r14/` (13 GB, recommend keep `outputs/`); the rest on the pending-deletions
  list.
- GPU check at 17:36: driver 595.99.02 answers; the kernel module and the library match; the GPU is free. Run mode:
  night.
