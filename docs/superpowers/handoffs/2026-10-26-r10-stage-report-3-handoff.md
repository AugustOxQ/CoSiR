# Handoff: build the outside-view page and the dev-log post of the stage report, show the user both, then open r10 decide 23

## Brief
- Done: the stage report of the condition-free line is written, reviewed, indexed and pushed (commit 6d81850): an Opus draft, a trim, a Codex claim review (23 must-fix items, provenance and overclaim; no wrong number in the result tables) applied by a fix wave, the controller's re-read; four figures with scripts.
- State: 10,657 words; held emotion passes on four encoders (Qwen 0.1029 against 0.0874, SigLIP 1 0.1036, SigLIP 2 0.1024, CLIP L/14 0.0996). No page, no post yet; the index row lacks their links.
- Next: the outside-view page (its draft brief is ready), the dev-log post, the user's look, the task end; then "r10 decide 23".
- Blocked: nothing.

> Written 2026-10-26 15:02 by "r10 stage report 2". Loop step reached: after decision A of decide 22 (a stage report before
> the next experiment); the next loop's direction is open until the user has read the report. Direction log entry:
> [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The stage report, `docs/reports/stage/2026-10-26_condition_free_line.md` (10,657 words; the only source of facts
   for the page and the post). Read it once, whole; then let subagents carry it.
3. `~/.claude/references/outside-view-report.md` ("The page, top to bottom", "Words", "Figures and tables", "Who
   builds it", "Build and deliver", "Publishing").
4. `.scratch/stage_pages/2026-10-26_condition_free_line/page_draft_brief.md`: the brief for the page's Opus
   `medium` draft, written by this chat before the report was final. Check its facts against the report (the step
   numbers 12 to 18, the marks, the limits) before launching; edit what drifted.
5. The two earlier handoffs of this task, for the user's decisions and the pitfalls:
   [the first](2026-10-26-r10-stage-report-handoff.md) (§2, §5) and [the second](2026-10-26-r10-stage-report-2-handoff.md) (§5).
6. `~/.claude/rules/agent-routing.md` (models and efforts), `~/.claude/rules/container.md` ("Showing the user an
   HTML page"), `~/.claude/rules/routines.md` (the task end), `~/.claude/references/loop-chats.md` ("How to cut")
   and `~/.claude/templates/handoff.md` (for the decide 23 handoff).

Skills: none (the html-plan plugin stays off; the page is built by path).

## 2. Decided by the user, not to be reopened

As in the [first handoff](2026-10-26-r10-stage-report-handoff.md) §2: the stage report first; scope 2026-10-15 to
2026-10-26 with the 2026-10-19 report folded in; the next experiment open until the user has read the report; nothing
published on the dev-log site without the user's word; no repo name, link, account or author name on the page or in the
post; held data reported as results.

**Agent defaults of this chat, not yet seen by the user** (say them in one line each when the user looks):
- The report's length: 10,657 words, above the writer's brief's own 7,000 to 9,000. The tables alone hold about
  3,270 words; a second trim would have cut content the user asked for.
- The page covers steps 12 to 18 (2026-10-19 to 10-26); its Background says where the previous page (steps 1 to 11)
  ended. One table at the end of the page's Experiments (the held result per encoder), as the 10-19 page had. No Next
  section on the page, since the next experiment is not decided (the reference's rule).
- Step 11's probe values come from its round log (the step has no auto report); the header says so. The user's quotes
  keep square brackets for an internal code replaced by its plain name; the header says so.

## 3. Where things stand

- **The report** is committed (6d81850) with its index row in `docs/reports/reports_sum.md` (the row still lacks
  the page and post links: add them when the files exist, else `check_reports_sum.py` flags broken links).
- **The figures**, in `docs/reports/assets/2026-10-26_condition_free_line/`: `timeline.png`,
  `model_structure.png` (the page's structure figure, so no new figure job), `held_emotion_by_encoder.png`,
  `ablation_effects.png`, each with its `build_*.py` (each rebuilds bit for bit; checked). Built by four Codex
  figure jobs (one fix round), each checked by the controller against its task file.
- **The review trail**, in `.scratch/stage_pages/2026-10-26_condition_free_line/`: `writer_brief.md`,
  `claim_review_task.md`, `claim_review.md` (39 KB; its "Should fix" items were applied too; its "Not checked"
  list names what no agent verified: the figures' contents against the reports, checked by the controller against
  the task files only; literature facts, checked against the notes only; whether artists are shared across the split,
  which the report states as not measured). The claim review's line numbers refer to the draft before the fix wave.
- **The post:** the long template draft is at `.scratch/stage_pages/2026-10-26_condition_free_line_devlog.draft.md`
  (written before the fix wave; check its numbers against the report). The short three-line post, the form the user
  asked for on 2026-10-19, is not written.
- **Known weak spots** a reader may raise: after the fix wave the prose is terse in the comparators table and the
  synthesis, where many paragraphs open with "Our reading:". Correct, within the rules; a Sonnet `high` worker
  smooths what the user points at.
- **Codex:** six ledger rows added (claude-config, pushed). The five-hour window had 74% left at 2026-10-26 15:02.

## 4. The step plan (agent; in this order)

f. **The page**, as "Who builds it" in `outside-view-report.md`: an Opus `medium` `worker` on
   `page_draft_brief.md` writes `.scratch/stage_pages/2026-10-26_condition_free_line.draft.md`; a Haiku `worker`
   cold read (the draft only, as a 5-minute outside reader; passages it cannot follow; no verdict); a Sonnet `high`
   `worker` check and fix against the stage report (overclaims, numbers, the cold read's passages, by replacing
   text; `<stem>.changes.md`); a Sonnet `medium` `worker` HTML build to `<stem>.src.html`, copying the markup
   of `~/.claude/templates/outside_view_example.src.html` and of the previous source
   `.scratch/stage_pages/2026-10-19_qwen_buddy_line.src.html` (each experiment a `div.exp` with its `h3`; marks as
   `span.pill pos|neg|chk`; Background and Preliminaries `details.part`, Experiments `details.part open`; the
   figure as `<doc-shot src="../assets/2026-10-26_condition_free_line/model_structure.png" caption="...">`; the
   table a `doc-mock frame="none" w="600"`). Then the controller's pack: copy the source to
   `<scratch>/stage/<stem>.src.html`, symlink `<scratch>/assets` to `docs/reports/assets`, run
   `node ~/.claude/plugins/cache/claude-community/html-plan/1.0.0/skills/html-plan/runtime/pack.mjs <scratch>/stage/<stem>.src.html -o docs/reports/stage/<stem>.html --root /project/CoSiR`,
   replace the `data:image` URIs by the source's relative paths, fix every pack warning except the prose budget and
   the long-paragraph count, read the text with `python ~/.claude/templates/outside_view_text.py`. Add the page link
   to the index row; `check_reports_sum.py`; `docs-check`; commit and push.
g. **The post:** `docs/reports/stage/<stem>_devlog.md`, update #4 (19 to 26 October), in the short three-line form
   of the 10-19 post (`docs/reports/stage/2026-10-19_qwen_buddy_line_devlog.md`: a leading HTML comment, three
   lines, the link as a placeholder, the structure figure attached). Add the post link to the index row. Stage the page:
   `devlog-publish --site /project/cosir-devlog add --title "<the page's h1>" --outcome "<one sentence>" --figure assets/2026-10-26_condition_free_line/model_structure.png docs/reports/stage/<stem>.html`
   (check `add --help`: title, outcome, figure, optional date, page), then `check`, then `serve` for the
   preview. `deploy --yes` only on the user's word, then commit and push the site repo.
h. **The user looks:** serve `docs/reports/` by the Cursor route (`python3 -m http.server <port> --bind 127.0.0.1
   --directory docs/reports` in the background; open `stage/<stem>.html?v=1` through the newest
   `/tmp/vscode-ipc-*.sock` and `$BROWSER`; ask whether it opened) and send the markdown report with
   `SendUserFile`. Say the agent defaults of §2 in one line each. Small edits by a Sonnet `high` worker. Then the
   task-end routine (`routines.md`: `wrapup-check`; the storage summary, where nothing big is expected: this task
   left the figures, 1.3 MB, and scratch files under 1 MB; `.scratch/pending_deletions.md` is empty), commit and
   push, the handoff for "r10 decide 23" (step 2 of the next loop: decision A after the user's reading; the options
   in the direction log's 2026-10-26 entry: a held genre claim test of the weight-8 variant; more development work on
   the relational term; the write-up of the encoder claims; aspect-named retrieval for the 11-10 gate waits), and
   `loop-next <handoff> --label "r10 decide 23" --notify`. The user may read first and come to decide 23 later.

## 5. Code and pitfalls

- **The stem** is fixed: `2026-10-26_condition_free_line` (page, post, source, drafts).
- **Facts on the page are a subset of the stage report**: no new number or claim; at most two numbers per sentence;
  no "Our reading", "we think" or advice on the page; no process words (GO, gate, bar as a name).
- **The index check** flags a link to a file that does not exist: add the page and post links only once the files
  are there (this chat removed them for that reason).
- **The pack** cannot resolve `../assets` from `.scratch/stage_pages/`: pack from the scratch copy with the
  `assets` symlink. No `data:image` may remain in the committed page. A rebuilt page may show from the browser's
  cache: open it with `?v=<n>`.
- **Codex figure jobs** are done; no new Codex job is planned. If one is needed, `codex-usage --room <kind>` first,
  the command starts with `codex e --dangerously-bypass-approvals-and-sandbox`, logs in the session scratchpad, a
  ledger row after. The `.ccg` folder at the repo root predates this task and is gitignored.
- **No GPU work**; Python is `/root/miniconda3/envs/CoSiR/bin/python` with `PYTHONDONTWRITEBYTECODE=1`.
- **Delete nothing**; `.scratch/pending_deletions.md` is empty.
- **Context:** this chat passed 200k during the claim review. The next chat reads the report once and keeps the page
  work in subagents; if it passes 200k before step h, it cuts after step g as "r10 stage report 4".

## 6. State at handoff

- Running: nothing (no subagent, Codex job, background job or GPU lock of this chat).
- Uncommitted: nothing after this handoff's commit (`chats.tsv` and `briefs.md` are committed right after `loop-next`).
- On disk over 1 GB: unchanged from the first handoff (the round's outputs on /data/SSD2, kept; the shared stores).
