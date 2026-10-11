# Handoff: the plain-English and stage-page topic is closed; nothing to open next

## Brief
- Done: plain English (our STE guide at 80%) for all text the user reads; stage reports now get an interactive HTML page on the html-plan runtime instead of a user-read report; the first page (10-04 stage report) is checked and committed.
- State: claude-config `ff2f221`, CoSiR `8edc90d`; a fresh session follows both rules.
- Next: nothing in this topic. The next real stage report is the first live use of the page rule.
- Blocked: nothing.

> Written 2026-10-11 16:15 by the process plain-English chat (session 2b387618). Not a loop step: a change to how we
> work, between rounds 6 and 7. Source: Karpathy's note on readable model output (four formats: ASD-STE100 writing,
> diagrams, HTML pages, videos) and the make-it-click skill, used only as a reference.

## 1. Read in this order

1. Section 2 (what the user decided).
2. `~/.claude/rules/plain-english.md`, then `~/.claude/references/stage-page.md`.
3. The example page: `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.html` (serve `docs/reports/`
   on a localhost port; Cursor forwards it).

## 2. Decided by the user, not to be reopened

- **Plain English** (user's own picks): our own guide in the style of ASD-STE100 at 80%, for chat messages, user-read
  reports, stage pages, results briefs and handoff briefs; not for paper drafts, code or full auto reports. The user
  asked for our own guide with field words, not a copy of make-it-click; 80% kept after the recommendation (the field
  words remove most of the friction), open to 60% if the text feels stiff.
- **Videos:** skipped. **Diagrams:** our rules already cover them.
- **HTML only for stage reports** (after the recommendation; weekly reports keep their slides). The page replaces the
  stage report's user-read report and is committed beside it (user's own picks). The user's reason: a simple
  user-read markdown gains little from HTML; HTML pays off on a rich source.
- **Opus builds the page** (user). The user likes html-plan's comment box and Respond sheet ("talk to Claude").
- **The page has Background and Preliminaries before the findings** (user, after reading version 2: "lacks a bit of
  context as usual").
- **Parked** (user): Codex builds a skeleton of the report and its figures, then an Opus worker builds the HTML. Try
  it on a real stage report when the user asks. Weekly pages: revisit after a few stage pages.

## 3. Where things stand

- Rules: `rules/plain-english.md` (always loaded), `references/ste-writing.md`, `references/stage-page.md`; the report
  rules, the stage template, `docs-layout.md`, `worker.md` and the user's loop doc point to them.
- Templates: `stage_page_example.src.html` (the example source) and `stage_page_text.py` (extracts the page's text
  for the checks, notes included).
- The trial took three builds and a fix wave: about 534k tokens on Opus for the builds, about 250k for the checks
  and fixes. One build is about 140k to 210k tokens.

## 4. Open points for the next step

- **Figures:** the trial page reused older figures that do not fit the current setting. The next page gets figures
  made for it, by a Codex figure job first (`stage-page.md`).
- **Checks matter:** the fidelity check found two overclaims among the five findings that the Opus builder wrote, and
  the check of the notes found two numbers the report does not state. Never skip them.

## 5. Code and pitfalls

- **`pack.mjs` always inlines images.** After the pack, put the relative `../assets/...` links back. The trial's
  helper did this for `doc-shot` elements; no `data:image` may remain.
- **Pack from a folder where `../assets` resolves** (the trial used `.scratch/stage_pages/work/` with an `assets`
  link beside it).
- **A wide table breaks:** `doc-mock` draws at `w` and scales down to the column. At most 4 columns at `w` 600 or less.
- **zsh reads a word that starts with `=` as a command** (`echo =====` fails); use `---` as a separator.
- **The map checker** reads `references/<name>.md` in any rule text as a pointer into our references.

## 6. State at handoff

- Running: the local server on port 8771 (serves `docs/reports/`); stop it when the user is done.
- Uncommitted: nothing after this handoff's commit.
- On disk over 1 GB: nothing. Small leftovers for the user's decision: `.scratch/stage_pages/` (3.9 MB, trial
  sources, copies and packed files) and this session's scratchpad (5.7 MB).
