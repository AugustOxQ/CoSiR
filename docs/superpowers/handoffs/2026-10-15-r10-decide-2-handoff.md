# Handoff: r10 decide 2: discuss the new line's direction (decision A) with the user, then step 3 (gain knowledge)

## Brief
- Done: the conditioned line is archived (tags, `archive/v2_conditioned/`, paper archived), disk cleaned (+67 GB), memory and CLAUDE.md point at the new line.
- State: new line drafted: a condition-free model built from the buddy model, over PercepT; soft CVPR.
- Next: finish decision A with the user, then step 3: the reuse inventory and the PercepT literature check.
- Blocked: nothing.

> Written 2026-10-15 21:02 by `r10 decide` (cut at 200k context). Loop step reached: after decision C, decision A in
> draft (step 2). Direction log entry: [2026-10-15](../direction_log.md).

## 1. Read in this order

1. The direction log: its "Now" and the 2026-10-15 entry ([direction_log.md](../direction_log.md)). Decision A as the
   user stated it is there.
2. The short paper plan: `docs/paper/README.md` (private repo), "Current plan".
3. For step 3, the PercepT and buddy reports that stay live:
   [stage 1](../../reports/stage/2026-09-23_artelingo_percept_stage1.md),
   [stage 2](../../reports/stage/2026-09-23_artelingo_percept_stage2.md),
   [weekly 09-30](../../reports/weekly/2026-09-30_percept_buddy_to_v2.md) (buddy matched PercepT on held-out AMI).
4. Only when asked about the old line: [archive index](../../../archive/v2_conditioned/README.md).

## 2. Decided by the user, not to be reopened

- Stop the conditioned line, archive it and the CVPR plan (supervisors, relayed 2026-10-15).
- Archive form: tag `archive/v2-conditioned` + `archive/v2_conditioned/` + `.ignore` + index; memory and CLAUDE.md read
  only the main line (user, 2026-10-15). The paper: `docs/paper/archive/cvpr2027_aspect_episodes/`, tag
  `archive/cvpr2027-aspect-episodes`.
- Held data may be re-read; flag only tuning, selection or sweeps on it (user; global rule `held-data.md`).
- The r9 choices (SigLIP 2 refit, H9 wording) are closed by the stop.
- The weekly-report guide has no worked example any more (user).

## 3. Where things stand

- Decision A (draft, user's words in the log): base (b), the buddy model, redesigned, not a pull of branch
  `experiment/percept_topic_pipeline`; tasks: PercepT's task plus retrieval, aspect-conditioned retrieval a third
  candidate (interesting but maybe too hard); goal: a method that discovers aspects, label-free best, looser use of
  labels allowed; CVPR soft: works by ~2026-10-29, submit; else try to 2026-11-10; else no submission.
- The user's answer to the side-check question "are aspect labels needed in training": label-free is the aim; labels
  may be used loosely if needed.
- Live tests: 359 pass after the move. `check_reports_sum.py`, `docs-check`, `claude-md-check`, `disk-check` silent.
- No literature note on PercepT exists in `docs/reports/literature/` yet: step 3's check is new ground (opus, deep).

## 4. Open points for the next step

1. **Key, decision A:** settle the direction with the user: what "improvement over PercepT" is measured on (PercepT's
   AMI and stage-2 numbers on our split, retrieval R@1, both), what "discover aspects" means as an output, which
   datasets and backbones, and what counts as "something works" by 2026-10-29. Then type and loop size (agent default
   so far: new idea, full loop).
2. **Step 3**, after A: (a) a reuse inventory: code (`src/data`, `src/utils`, `src/eval`, `src/model`; the buddy
   branch), caches under `/data/SSD2/pre_extract/`, the PercepT snapshot `res/percept/`; a first module-by-module code
   map is in `.scratch/archive/move_list.md`, section "src code map" (gitignored, on disk);
   (b) the literature check on PercepT (arXiv 2606.03345) and what an improvement must beat (deep-research agent).
3. **Minor, storage:** DAS6 `cluster du` was not run for the archive; run it once at the next wrap-up and show the
   user. `.scratch/pending_deletions.md` keeps two deferred rows (`archive/v2_conditioned/.scratch/r9-run/`).

## 5. Code and pitfalls

- `archive/` is hidden from default search by `.ignore`; search it only with the path named. A bare `pytest` at the
  root collects it: always pass `src/test`.
- Archived round code cannot run in place; restore through the tag (archive index).
- `scripts/extract_features.py` reads the ArtELingo-28 read set from `/data/SSD2/pre_extract/artelingo28/`.
- 7 untracked result files sit under `archive/` (r8 and Leiden results, a features link); they were untracked before.

## 6. State at handoff

- Running: nothing. The r10 main chat is live.
- Uncommitted: nothing of ours.
- On disk over 1 GB: `archive/v2_conditioned/` (9.6 GB, kept); feature caches on SSD2 (kept).
