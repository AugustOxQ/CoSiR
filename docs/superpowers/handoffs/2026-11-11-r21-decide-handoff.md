# Handoff: r21 decide, the DEC loss on every head (to-do item 4): a short grill and a short spec, then build and run

## Brief
- Done: the user moved r21 forward to now, with DEC only (item 4); deeper heads (item 18) stay waiting.
- State: r21 never started: no spec, branch, round folder or outputs. Old estimate: GPU about 1 h, CPU 1 to 2 h.
- Next: read the sources below, then a short grill (key questions in section 4) and a short spec.
- Blocked: nothing. The local GPU is held by r25, then r26, until about 10:00.

> Written 2026-11-11 05:43 by a DustDevilVis chat (the user asked there; the work is CoSiR's). Loop step reached:
> decision A made (direction and scope, below), before step 3. Direction log entry: none yet; the decide chat writes
> r21's entry. Loop size: the user expects small to medium; the rule's size for an ablation is medium (short grill,
> short spec; step 3 skipped when the options are clear).

## 1. Read in this order

1. This handoff, all of it.
2. `docs/user_read/2026-11-08_what_is_missing.md`: the "Waiting" row (item 4) and "What is missing" (the r21 lines).
3. `docs/user_read/2026-11-03_experiment_overview.md`: to-do row 4 (DEC on every head, plain and with a pull toward
   the start clusters, retrieval reported too) and the ablation section around "What is missing: the DEC loss".
4. `docs/superpowers/direction_log.md`: the 2026-11-03 entry (the to-do list, "DEC for now, its place open"; the six
   loops, r21 = items 4 and 18) and the 2026-11-08 and 2026-11-10 entries (r21 was set for after 11-16; now changed).
5. `docs/reports/literature/2026-12-17_cluster_level_losses.md` (deep literature check of 2026-10-20 on per-aspect
   cluster losses, "DEC from a good start"). It says our September evidence conflicts with the field's positive
   reports; read its summary before step 3, which may then be skipped.
6. The September DEC pilots of the old buddy line, for what went wrong then:
   `docs/reports/auto/percept/pilots/20260923_artelingo_buddy_analysis/dec_pilot_report.md` and
   `dec_pilot_v2_report.md` (not read by this chat).
7. Paper context: `docs/paper/cvpr2027_aspect_spaces/README.md`, the Decisions rows (the ablation table, Table 2 at
   each aspect's class count).

Skills: grilling, domain-modeling (the grill runs as grill-with-docs; read `~/.claude/references/grill.md` first).

## 2. Decided by the user, not to be reopened

All on 2026-11-11, about 05:40, in a DustDevilVis chat:
- **r21 runs now**, beside the paper writing, not after 11-16. This changes the 2026-11-08 and 2026-11-10 decisions
  (user's own).
- **Scope: DEC only**, to-do item 4: a DEC loss on every head, in its two planned forms (plain, and with a pull toward
  the start clusters), with retrieval reported too. Item 18, the deeper head, stays waiting (user, after seeing the
  recommendation).
- **The result could go into the main paper**, not only the supplementary; the user expects a small to medium loop
  that ends soon, while the paper is written (user's own; our recommendation was the supplementary).
- **Chats:** r21 has its own Herdr workspace, "CoSiR r21"; this decide chat is its first tab (user's own). r21 has no
  main chat yet: open it at brief approval (`loop-chats.md`, "The main chat").

## 3. Where things stand

- r21 has nothing on disk or in git: no branch, no `src/test/` folder, no `/data/SSD2/cosir_r21/`, no spec (checked
  2026-11-11 05:42).
- In the new line, no DEC run exists. The DEC pilots of September (old buddy line) exist as reports only.
- Old cost estimates (not measured): 11-03 overview, item 4 alone: GPU about 1 h, CPU 1 to 2 h, a medium build;
  11-08 plan, r21 with deeper heads: GPU about 2 h.
- Paper deadlines (each 23:59 AoE, 12:59 the next day in Amsterdam): paper 2026-11-16, supplementary 2026-11-23. For
  the main paper, the results must be read well before 11-16.
- Table 2 (main clustering) now reports AMI at each aspect's class count (k = 9 emotion, 27 style, 10 genre); c* at
  0.102 / 0.411 / 0.510 against the teacher's 0.089 / 0.406 / 0.501 (paper-experiments-2 handoff).
- The local GPU at 05:41: r25's extraction until about 06:36, then r26's extraction and training (expected to end
  before 10:00). A heavy local run over 1 h may not run on weekdays 10:00 to 19:00 (`shared-resources.md`); DAS6 is
  first for long runs when a node is reserved (not checked).

## 4. Open points for the next step

Key (asked answer-first, per the grill protocol):
1. **What DEC trains from:** a fine-tune of the frozen c* heads ("DEC from a good start"), or c*'s full recipe with
   DEC added from the start. This sets the build size and the GPU time.
2. **The start clusters and the pull:** which clusters (k-means on each head's output at the class count, or the
   teacher's clusters) and what the pull is (a term toward the initial assignments or centroids). The 11-03 to-do
   names the pull only.
3. **Claim or descriptive:** if the result may enter the main paper, the bar against c* (and the teacher) must be set
   before the run, with the cluster count of Table 2. The ablation table's form decides where the row goes.
4. **Seeds and splits:** develop and pick on validation with one seed, then test the pick on several fresh seeds
   (`held-data.md`); DEC's own knobs (cluster count, target-distribution update, weight) are picked on validation.
5. **When and where it runs:** today's daytime GPU limits (section 3); a run tonight from 19:00 that ends by 11:00, or
   DAS6.

Minor:
6. Which retrieval read-out (the one in the ablation table's retrieval column, r20's report 2027-01-13).
7. Datasets: ArtELingo only, as the other ablations, or also MMArt.

## 5. Code and pitfalls

- CoSiR main has other chats' uncommitted edits (`src/test/20261111_paper_baselines/*`,
  `docs/superpowers/constitution.md`): commit only r21's files. Several worktrees exist (`git worktree list`);
  r21's build gets its own branch per `matt-chain.md`.
- The repo is public: no keys or personal data in handoffs, specs or briefs.
- Python: `/root/miniconda3/envs/CoSiR/bin/python` with `PYTHONDONTWRITEBYTECODE=1`. All GPU and heavy CPU work goes
  through `res-queue` with `--label r21:<step>`.
- The user has a meeting at 11:00 (paper-experiments-2 handoff); ask nothing about that meeting.

## 6. State at handoff

- Running: nothing of this chat. Other CoSiR tabs: r25 paper baselines, r26 PercepT original, paper experiments 2.
- Uncommitted: nothing of this chat after this handoff's commit.
- On disk over 1 GB: nothing.
