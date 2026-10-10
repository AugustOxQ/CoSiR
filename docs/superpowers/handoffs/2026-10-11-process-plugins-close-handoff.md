# Handoff: the plugins topic is closed; nothing to open next (CoSiR main stays closed until the automation starts)

## Brief
- Done: plugins weighed (html-plan installed but off, used at the grill on the user's yes; planning-with-files skipped); routing-check drops version triggers, adds an effort-entry check; the setup is now 11 modules with a generated map.
- State: fresh chat 36,355 tokens on its first call, unchanged by tonight's changes; `design-map --check` silent (141 items).
- Next: nothing in this topic; the HTML report idea is parked.
- Blocked: nothing.

> Written 2026-10-11 01:10 by the process plugins chat (session 98a9e364). Not a loop step: changes to how we work, between rounds
> 6 and 7. Closed by the user.

## 1. Read in this order

1. Section 2 (what the user decided tonight).
2. `/project/claude-config/design/map.html` (or `modules.yaml` beside it): every part of the setup by module.

## 2. Decided by the user, not to be reopened

- **Version triggers out of `routing-check`** (2026-10-10): no rule pins a program version; Codex CLI and Claude Code
  updates no longer prompt a routing review (tools `2971e90`, claude-config `7911644`).
- **Stale models handled when they bite** (2026-10-10, after the recommendation): a Codex "model not found" error →
  the newest model of the same tier, finish the job, ledger note, open item (`codex-jobs.md`). Silent case:
  `routing-check` names a Claude model that ran in the last week with no `modelSettings` effort entry (Haiku
  excepted), the round 6 `claude-sonnet-5` key problem (tools `032af70`, claude-config `342f6b0`).
- **The free Codex model-list check stays** (it reads Codex's local cache).
- **html-plan** (marketplace `claude-community`, added 2026-10-10): installed but off. At the grill's last round of a
  medium or full loop, ask once whether the plan wants a page; on yes an Opus worker follows the skill by path, minor
  questions as forms, key answers as notes, at most one page per loop (`grill.md` rule 7; claude-config `2d32ad3`).
  The user's reasons: key grill questions stay interactive in chat; the page is worth its cost (about 140k tokens)
  only for plans hard to read as markdown.
- **planning-with-files skipped** (2026-10-11): its plan-on-disk idea is covered by specs, tickets, handoffs and the
  context meter; 666 tokens per main-chat call and a hook on every tool call in every chat and subagent.
- **The setup as modules** (2026-10-11): `claude-config/design/modules.yaml` (agents edit it) assigns every rule,
  reference, template, agent, skill, tool, hook and settings key to one of 11 modules; `design-map` writes
  `design/map.html` for the user; `design-map --check` runs at every wrap-up, silent unless they disagree;
  `maintaining-rules.md` step 2 updates the registry with each change (tools `57fa7c1`, claude-config `96f0d89`).
- **Parked: an HTML user-read report.** The user likes reading HTML, but the user-read report is short with simple
  figures, unlike html-plan's code-heavy design. Revisit after several real rounds, when the user-read report needs
  an update anyway; options then: a tool that renders our markdown into an interactive page (no model tokens), or a
  writing skill.

## 3. Where things stand

- Plugins on: humanizer, slack. Per chat: ARS (`claude-paper`). Installed but off: html-plan (and the older ones in
  `opt-in/unused/README.md`).
- Trial page from the repo at `c4775f8` (round 6 at build start): `.scratch/html_plan_trial/r6_plan.packed.html`
  (kept, gitignored).

## 4. Open points for the next step

- **Minor:** round 7's first grill is the first real use of `grill.md` rule 7; check that the question comes once,
  before the closing summary.

## 5. Code and pitfalls

- **`pkill -f "<pattern>"` kills the Bash tool's own shell** when the pattern appears in the command line (exit
  144); use `pkill -f "[h]ttp.server 8766"`.
- **HTML for the user from the container:** serve the folder on a localhost port and open it with `"$BROWSER" <url>`;
  Cursor (attached to the container) forwards the port to the user's browser. `/tmp` is not visible on the host.
- **A Codex tool brief that asks for a look-and-fix loop** may leave the preview step in the shipped tool
  (`design-map` wrote a PNG beside itself); check for files written beside the tool.
- **modelSettings keys are full model IDs**; the docs do not support alias keys such as `sonnet`.

## 6. State at handoff

- Running: nothing (the localhost servers are stopped).
- Uncommitted: nothing after this handoff's commit.
- On disk over 1 GB: nothing. Pending deletions: the 26 round-6 rows, left for CoSiR's next decide chat (the DAS6
  folders need the cluster CLI before the reservation ends).
