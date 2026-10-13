# Handoff: review the second stage page (2026-10-12 affect steering) with the user, by the new outside-page rules

## Brief
- Done: the dev-log start page (primer) and the first stage page (2026-10-04) are rebuilt for outside readers; the user's rules are collected in a draft.
- State: both pages are committed; the start page is built in the site repo, not deployed; nothing else is online.
- Next: open the 2026-10-12 page for the user, apply the draft's rules, then take the user's comments one by one.
- Blocked: nothing.

> Written 2026-10-13 20:58 by the outside-pages chat. Outside the research loop (a report and process topic).

## 1. Read in this order

1. `/project/claude-config/docs/drafts/2026-10-13_html_page_requirements.md`: every rule the user gave on 2026-10-13,
   the page order, and the open questions. This is the spec for the next page.
2. `~/.claude/references/stage-page.md`: the current process (who builds, pack, deliver). The draft overrides it where
   they differ.
3. The model page: `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.html`, source
   `.scratch/stage_pages/2026-10-04_aspect_conditioned_similarity_methods.src.html` (gitignored), checked draft beside
   it (`.draft.md`).
4. The page to do: `.scratch/stage_pages/2026-10-12_affect_steering_line.src.html`, report
   `docs/reports/stage/2026-10-12_affect_steering_line.md`.

## 2. Decided by the user, not to be reopened

All on 2026-10-13, recorded in the draft (item 1 above): the 5-minute outside reader; Background and Preliminaries
folded; no internal process words (GO, no-go, gate, pass test); Experiments in plain words, about three sentences each
(what, why, positive or negative with a short explanation); no figure or table by default, but one concise structure
figure (seconds to read) for each positive finding; no Limits, no "give your reading first", no "Our advice"; a
one- or two-sentence "Next" only when the user decided it (direction log "Decided (user)"); glossary at most 5 terms;
no "a X, not Y" phrasing; marks "Positive finding", "Negative finding", "Check"; one site style (html-plan look, 740 px
column, figures at column width); the dates line under the start page title. The rules apply to all stage pages;
Preliminaries is the exception (figures, tables, longer text allowed). The user reads user-read and weekly reports;
the HTML page becomes an outside report every one or two days.

## 3. Where things stand

- **Start page (primer):** `docs/reports/stage/2026-10-12_devlog_primer.md` and its figures
  (`docs/reports/assets/2026-10-12_devlog_primer/`) committed on `r9-scope` (4f03469). Built into the site repo
  `/project/cosir-devlog` (`site/start.html`, `site/style.css`, two new assets): uncommitted there, not deployed.
- **`devlog-publish`** (tools 151bdf6): site style of the stage pages, framed figures at column width linked full size,
  the italic line under the primer title becomes the dates line, `style.css` rewritten on every run. 43 tests pass.
- **First stage page (2026-10-04):** rebuilt and committed on `r9-scope` (6c32cd2). It has 11 experiments, one figure
  for the positive finding (`head_reader.png`, the heads themselves not drawn: open, see 4), and Next from the
  2026-10-04 decision. Not published.
- **Second stage page (2026-10-12):** unchanged; the 2026-10-12 version is online.
- **At the end (the user asked at the start of the chat):** fold the draft into the rules, then update the online site.
  Deploy only after the user names the pages to publish (`devlog-publish add`, `start`, `check`, `deploy --yes`).

## 4. Open points for the next step

- Key, at the rule update: the page's name ("outside-view report" or "outside report"); the length limit (about 1,800
  words of prose now); one primer for site and Slack, or a short Slack version; the stage-report cadence in
  `report-kinds.md`. Rename in one change across `stage-page.md`, `report-kinds.md`, `docs-layout.md` and the dev-log
  templates, following `~/.claude/references/maintaining-rules.md`.
- Minor: should box 2 of the head-reader figure show the heads? The user has not answered.

## 5. Code and pitfalls

- **Pack:** packing from `.scratch/stage_pages/` cannot resolve `../assets`. Pack a copy from a scratch folder
  `<dir>/stage/<stem>.src.html` with `<dir>/assets` a symlink to `/project/CoSiR/docs/reports/assets`:
  `node <html-plan>/runtime/pack.mjs <dir>/stage/<stem>.src.html -o docs/reports/stage/<stem>.html --root /project/CoSiR`.
  Then put each relative image link back: replace the `data:image` URIs in the order of the source's `<doc-shot src>`.
- **Comment buttons:** the runtime puts one on every `main h3`; wrap each experiment (h3 and paragraph) in
  `<div class="exp">` so it gets exactly one.
- **Shared checkout:** the r9 build commits on `r9-scope` in `/project/CoSiR`. Commit only your own paths with
  `git commit --only -- <paths>`; check `.git/index.lock` first. `*.json` is gitignored in this repo.
- **Figures:** Codex's five-hour window was nearly empty until 21:37; check `codex-usage` before a figure job.
- **Writing:** after each draft, a Haiku cold read as a 5-minute outside reader, then a Sonnet `high` check against the
  report, found real gaps on the 2026-10-04 page (an overclaim and terms defined only in folded parts).

## 6. State at handoff

- Running: two local servers of the old chat (port 8765 serves `docs/reports/`, 8766 serves `/project/cosir-devlog/site`);
  they end with that chat. Start your own if they are gone (`rules/container.md`).
- Uncommitted: `/project/cosir-devlog/site/` (start page, staged for the deploy); `.scratch/stage_pages/` (gitignored).
- On disk over 1 GB: nothing.
