# Handoff: run the prepared stage-report jobs (the Opus draft and four Codex figures), review, build the outside-view page and the dev-log post, then open r10 decide 23

## Brief
- Done: the "r10 stage report" chat read the direction log, the previous stage report and reports 12-16 to 12-25, and wrote the writer's brief (outline, sources, fixed facts) and four figure briefs; its context passed 200k, so it cut here.
- State: no report text yet; the briefs are in .scratch/stage_pages/2026-10-26_condition_free_line/ (the stem of the report is 2026-10-26_condition_free_line).
- Next: launch the Opus writer and the four Codex figure jobs in parallel, then the claim review, the index row, the page, the post; the user looks; then "r10 decide 23".
- Blocked: nothing.

> Written 2026-10-26 14:10 by "r10 stage report". Loop step reached: after decision A of decide 22 (a stage report before the
> next experiment); the next loop's direction is open until the user has read it. Direction log entry:
> [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. [The first stage-report handoff](2026-10-26-r10-stage-report-handoff.md): the task, the user's decisions (§2),
   where things stand (§3), the open points (§4) and the pitfalls (§5). Everything there still holds; this handoff
   adds only the prepared briefs and the step plan.
3. `.scratch/stage_pages/2026-10-26_condition_free_line/writer_brief.md` (28 KB): the stage report's outline, its
   sources per section, the fixed facts and the rules of the text. Read it whole: it is this chat's reading of the
   twenty reports, and the controller reviews the draft against it and the direction log.
4. `~/.claude/references/outside-view-report.md` ("Who builds it", "Build and deliver", "Publishing"),
   `~/.claude/references/codex-jobs.md` (the figure jobs: command, usage check, ledger row),
   `~/.claude/rules/agent-routing.md` (models and efforts).
5. The direction log's "Now" and its 2026-10-26 entry (the options for decide 23).

Skills: none (the html-plan plugin stays off; the page is built by path).

## 2. Decided by the user, not to be reopened

As in the [first handoff](2026-10-26-r10-stage-report-handoff.md) §2: the stage report first; scope 2026-10-15 to
2026-10-26 with the 2026-10-19 stage report folded in; the next loop open until the user has read it; nothing
published on the dev-log site without the user's word; no repo name, link, account or author name on the page or in
the post; held data reported as results.

## 3. Where things stand

The facts of the line are in the reports and in the writer's brief §3. This chat produced:

- `.scratch/stage_pages/2026-10-26_condition_free_line/writer_brief.md`: the brief for an Opus `medium` `worker`.
  Its output is `docs/reports/stage/2026-10-26_condition_free_line.md` (7,000 to 9,000 words). It links four new
  figures at `docs/reports/assets/2026-10-26_condition_free_line/` and reuses the 10-19 figure.
- `fig_timeline.full.txt`, `fig_structure.full.txt`, `fig_held.full.txt`, `fig_ablation.full.txt` (same folder):
  complete Codex task files (the common part prepended), one per figure: `timeline.png`, `model_structure.png`,
  `held_emotion_by_encoder.png`, `ablation_effects.png`, each with its `build_*.py`. Every number in them was
  copied from reports 12-22, 12-24 and 12-25; the controller checks each PNG against the task's numbers.
- No draft, no figure, no index row yet. Nothing is committed beyond this handoff.

## 4. The step plan (agent; in this order, steps a and b in parallel)

a. **The draft:** `Agent` with `subagent_type: worker`, `model: opus`, `effort: medium`; the prompt names the
   writer's brief by path and asks for the hand-back it describes. Tell the user: "an Opus `medium` worker drafts
   the stage report".
b. **The figures:** `codex-usage --room figure figure figure figure` (four jobs at once); then four background
   Bash calls, each starting with `codex e --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -c
   mcp_servers={} -C /project/CoSiR -m gpt-6.1-sol -c model_reasoning_effort=medium --json - < <task file> >
   <log> 2>&1` (the structure figure at `high`: a diagram), the log in the session scratchpad. When each returns:
   open the PNG (Read), check its values against the task file, one fix round by `resume` if needed, then a
   ledger row in `/project/claude-config/codex/jobs.md`. Without Codex room, a Sonnet `medium` worker builds the
   figure from the same task file (the figure row's Claude fallback).
c. **The controller's read of the draft:** against the writer's brief §3 and §4 and the direction log; fix the
   structure yourself only where a section is missing; wording goes to step d.
d. **The claim review** (a check of text Claude wrote, so Codex: `codex-usage --room review`; `gpt-6.1-sol`,
   medium; the 12-25 claim review in the ledger is the model, about 6% of the window): every number against its
   auto report at that report's precision, every claim within the licensed sentences of 12-19, 12-24 and 12-25, no
   internal code in prose, the user's readings quoted as the direction log records them. Findings to
   `.scratch/stage_pages/2026-10-26_condition_free_line/claim_review.md`. A Sonnet `high` worker applies the
   must-fix items (or a Codex fix wave by resume); the controller re-reads the changed passages. Fallback for the
   review: a Sonnet `high` worker.
e. **Index and checks:** one row in `docs/reports/reports_sum.md` under "stage" (the 10-19 row is the model, with
   the page and post links), `python scripts/check_reports_sum.py`, `docs-check`; commit and push the report,
   the figures and their scripts.
f. **The outside-view page**, as "Who builds it" in `outside-view-report.md`: an Opus `medium` worker drafts
   `.scratch/stage_pages/2026-10-26_condition_free_line.draft.md` (its brief names the reference, the example source
   `~/.claude/templates/outside_view_example.src.html`, the previous page's source
   `.scratch/stage_pages/2026-10-19_qwen_buddy_line.src.html` to copy Preliminaries from and add the held split, the
   claim-test bar, the matched arm and the seed range; the stage report; the direction-log entry; the page's
   structure figure is `model_structure.png`, already built, so no new figure job). Then the Haiku cold read, the
   Sonnet `high` check and fix (`<stem>.changes.md`), the Sonnet `medium` HTML build (`<stem>.src.html`), the
   controller's pack from a scratch copy with an `assets` symlink, the `data:image` URIs replaced by the relative
   paths, every pack warning fixed except the prose budget, the text read with
   `python ~/.claude/templates/outside_view_text.py`. The page goes to `docs/reports/stage/<stem>.html`.
g. **The dev-log post:** `docs/reports/stage/<stem>_devlog.md`, update #4, in the short three-line form the user
   asked for on 2026-10-19 (see the 10-19 post), the link as a placeholder (public repo); the longer template draft in
   scratch. Stage the page with `devlog-publish --site /project/cosir-devlog add` and `check`; `serve` for the
   preview; `deploy --yes` only on the user's word, then commit and push the site repo.
h. **The user looks:** serve `docs/reports/` by the Cursor route (`container.md`, "Showing the user an HTML
   page") and send the markdown report with `SendUserFile`; small edits by a Sonnet `high` worker. Then the task-end
   routine (`routines.md`: `wrapup-check`, the storage summary, nothing big expected), commit and push, the
   handoff for "r10 decide 23" (step 2 of the next loop: decision A after the user's reading; the options in the
   direction log's 2026-10-26 entry; `~/.claude/templates/handoff.md`), and
   `loop-next <handoff> --label "r10 decide 23" --notify`. The user may read the report first and come to decide 23
   later; that is fine.

The context meter: this chat passed 200k while reading the sources; the next one holds the draft and the page, so
it may cut once more at a clean point (after step e, or after step g), labelled "r10 stage report 3".

## 5. Code and pitfalls

As in the [first handoff](2026-10-26-r10-stage-report-handoff.md) §5, plus:

- **The stem** is fixed: `2026-10-26_condition_free_line` (report, page, post, assets folder, scratch files).
- **Figure numbers** come only from the task files (copied from the reports); a Codex job must not read other result
  files. If a PNG shows a value that differs from its task file, the job is wrong, not the report.
- **The writer takes numbers from the reports**, never from the brief's orientation list; its hand-back lists any
  number it could not find. The claim review checks all of them again.
- **Codex figure jobs run in the project folder** (`-C /project/CoSiR`) and write only under
  `docs/reports/assets/2026-10-26_condition_free_line/`; check afterwards that no `.ccg` records and no stray files
  were left (`git status --short`).
- **No GPU work** in this chat; Python is `/root/miniconda3/envs/CoSiR/bin/python` with `PYTHONDONTWRITEBYTECODE=1`.
- **Delete nothing;** `.scratch/pending_deletions.md` is empty.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, background job or GPU lock of this chat).
- Uncommitted: nothing after this handoff's commit (chats.tsv and briefs.md are committed right after `loop-next`).
- On disk over 1 GB: unchanged from the first handoff (the round's outputs on /data/SSD2, kept; the shared stores).
