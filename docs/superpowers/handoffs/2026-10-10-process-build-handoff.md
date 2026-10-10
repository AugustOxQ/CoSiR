# Handoff: build the process changes decided on 2026-10-10 (model and effort per role, the subagent wait hook, the lean research build, a per-project main chat), check that each works, estimate the effect roughly; round 7 is the real test

> Written 2026-10-10 16:20 by the process discussion chat (session 9f4f0592). Not a loop step: a change to how we work,
> between rounds. No direction log entry. Unattended: every decision is below; nothing needs the user except the case
> in section 5.

## 1. Read in this order

1. Section 2 of this handoff: the decisions are this build's spec.
2. `~/.claude/references/maintaining-rules.md`: the procedure for every rule, reference and template edit (re-read
   first, change the owning file, check cross-references, changelog line, fresh-session test, sync).
3. The files to change: `~/.claude/rules/agent-routing.md`, `~/.claude/references/matt-chain.md`,
   `~/.claude/references/loop-chats.md`, `~/.claude/rules/research-loop.md` with the user's doc
   `/project/claude-config/docs/research_loop.md`, `~/.claude/references/codex-jobs.md` (consistency only),
   `~/.claude/templates/handoff.md`.
4. For the wait hook: `/project/tools/bin/unattended-guard` and `unattended-guard-install` (an existing `PreToolUse`
   hook on `Bash|ExitWorktree`), `/project/tools/bin/context-meter` (lines 44 to 62: how a hook knows it runs in a
   subagent, the `agent_id` field).
5. For the main chat: `/project/tools/bin/loop-next`, `/project/tools/bin/subagent-watch`, and the Herdr skill
   `~/.claude/skills/herdr/SKILL.md` (`agent get` states, `agent read`, `agent prompt`, waiting on state changes).
6. Only when a number is needed: `docs/reports/auto/process/2026-11-26_engine_switch_check.md`.

## 2. Decided by the user, not to be reopened

All on 2026-10-10 in the process discussion chat. "After the recommendation" means the user picked after seeing ours.

- **D1, subagent waits enforced** (user, after the recommendation; the user leaned the same way before it). A
  `PreToolUse` hook that acts only inside subagents (`agent_id` present) refuses waits over about 4 minutes: a `Bash`
  timeout over 240 s, `sleep` over 240 s, `until`/`while … sleep` polling loops, `flock -w` over 240 s, and the
  `Monitor` tool with a timeout over 240 s. The refusal message says what to do instead: run only the tests your
  change touches, list the long run in your hand-back, then stop. In every subagent, attended or unattended.
- **D2, the model standard** (user, own pick, refined after seeing Superpowers' rule). Superpowers' structure (least
  powerful model per role, the model named on every Agent call, the next fixer one tier up after a failed fix) with a
  raised floor. **Sonnet 5.5 is the main implementer and reviewer.** Opus only for the few important cases:
  verdict-code reviews, the final whole-branch review, design judgment, and the attempt after a failed Sonnet fix.
  Haiku only for work that decides nothing (cold reads, fully specified edits), as the 2026-10-09 rule says; never
  code with tests, never a review. Debugging starts on Sonnet and escalates. Codex first stays as it is.
- **D3, effort per call** (user, own pick). The controller names an effort on every subagent call from a menu:
  Haiku at its default; Sonnet `medium` or `high`; Opus `high` or `xhigh`. Sonnet is the most common choice (user).
  Agent defaults for the menu: Sonnet medium for plumbing tickets (runners, wrappers, file formats, sync); Sonnet
  high for verdict-code implementers, ticket reviews, re-reviews and the first debugging attempt; Opus high for
  escalations and verdict-code reviews; Opus xhigh for the final review and design judgment.
- **D4, the engine** (user, after the recommendation): keep Matt's chain for round 7 and compare again after it.
  Update the "Checked 2026-10-10" paragraph of `matt-chain.md` to record this.
- **D5, the lean research build** (user, after the recommendation; revisit after round 7). Unit tests only for
  verdict code; plumbing (runners, wrappers, GPU jobs, file formats) is proven by one end-to-end smoke run; code
  handles the cases our data has and asserts the rest (it crashes loudly, no graceful handling, no tests for inputs
  that cannot occur); guards only where constitution C5 requires them (refusing a second held read, the ledger row),
  and a mutation check only on those. Three safeguards: (1) when unsure whether code is verdict code, treat it as
  verdict code; (2) cheap asserts at every file boundary, on the reading side (column names, row counts, sample-ID
  alignment, C6); (3) round 7's quality check against round 6 (final-review findings by severity, any defect that
  reached the run); a silent wrong number from plumbing that reaches the final review sends that kind of code back to
  unit tests. The quality of code must not drop; we cut only what research does not use (user).
- **D6, build documents** (user agreed to our suggestions): code maps only when the code is new to us; a short
  "shared interfaces" section in the ticket index instead of a contracts file; tickets of about 1.5 KB that cite spec
  and rule sections instead of restating them (this limits text; the 200k cap still limits scope). The descriptive
  pass after a verdict stays a grill question per spec (decided earlier the same day).
- **D7, a per-project main chat** (the user's idea; our adjustments accepted after the recommendation).
  - One "CoSiR main" tab per project (agent default), on **Sonnet at high** (user, after the recommendation), cut by
    the context meter like any chat. It organizes and watches; it makes no decisions and does no heavy thinking.
  - Phase chats keep opening the next tab themselves with `loop-next`, so the chain never depends on the main chat.
  - Each handoff gets a brief block at its top: 3 to 5 lines, at most about 600 characters (what was done, the state
    or key number, what comes next, anything blocked, the handoff's path). `loop-next` sends that block to the main
    chat's pane (`herdr agent prompt`) and appends it to a round log file, from which a restarted main chat recovers.
  - By default the main chat reads no handoffs, reports or code; it opens one only when the user asks (user). It
    reads the briefs (user). It checks that each cut left a handoff and a `chats.tsv` row.
  - It watches the active phase chat's Herdr state in a background Bash; on `blocked` or a long silence without a
    brief it reads that chat's last lines and notifies the user. It never answers a prompt, stops a chat or edits
    files.
- **D8, settings** (user; already done in this chat, not yet synced): the stale `modelSettings` key `claude-sonnet-5`
  was removed; `claude-sonnet-5-5` is now `high` (an entry at `medium` had appeared at 15:25, not from this chat).
  Opus 5.5 stays `xhigh`.
- **D9** (agent default): the phase controllers' own effort stays as it is for round 7, one change at a time.
- **D10, testing** (user): no separate trial of each change. Check that each new setting and tool works and give a
  rough estimate of the effect; the real test is round 7.

## 3. Where things stand

- Round 6 against round 5: $706 against $341 (API-price weight), 30h31 against 12h09; implementers $293. 41 of 50
  subagent calls ran on Opus (rounds 4 and 5: 18 of 36 and 12 of 29). 52 cache rewrites in waiting subagents, about
  $80; 16 of them were resumes of a handed-back subagent (the fresh-fixer rule covers those), about 36 polling waits.
- Code written per loop (Python lines, measured in the round folders): round 4's AFF-vetoes loop 2,094 product and
  2,162 test; round 5 3,721 and 4,551; round 6 10,964 and 14,956. Round 6's descriptive pass was about 4,300 lines
  with tests, the held runner about 3,100, the smoke chain about 1,700.
- Build markdown per loop is about the same in both engines: round 4 about 164 KB (plan plus `.superpowers/sdd/` task
  briefs, reports and reviews), round 5 about 138 KB, round 6 about 184 KB (tickets 57, code maps 71, contracts 25,
  briefs 31). The handoff before this one compared round 6 with the plans alone. Documents cost little to read
  (about $0.40 per implementer); the cost is the code they ask for.
- Haiku in rounds 4 and 5: 3 and 3 subagents (user-read report checks; two small re-reviews in round 5), all on
  Haiku 4.5 through the `haiku` alias. Superpowers sets no effort for Claude subagents (only its Codex path does).
- The Agent tool takes an `effort` per call, used only when a rule asks for it; transcripts do not record effort.
- Herdr: `agent get` reports `working`, `idle`, `done`, `blocked` (an approval or question UI) or `unknown`;
  `agent prompt` sends text and Enter and refuses a blocked agent. `loop-next` starts a new chat on the user's default
  model unless `--model` is given, so a Sonnet main chat does not make phase chats Sonnet.
- Not ours to touch: the `r6 read` tab (session 5809c3c0) holds round 6's decision C and wrap-up, still open; it
  continues the research loop. The `.scratch/pending_deletions.md` rows belong to that wrap-up.

## 4. Open points for the next step

No user decisions are open. The work, in this order:

1. **Rules** (D2 to D6, D9): edit the owning files, then a cross-reference pass. The model and effort rules live in
   `agent-routing.md`; the build details in `matt-chain.md`. Changelog lines; one fresh-session test of the changed
   rules (`maintaining-rules.md` step 5).
2. **The wait hook** (D1): a small tool, so Codex first (`codex-usage` before, `codex-jobs.md`), Claude if Codex
   cannot. Its own script or an extension of `unattended-guard`, the builder's choice; tests on sample hook inputs,
   with and without `agent_id`; the install line in `settings.json`'s `PreToolUse` (matcher covering `Bash` and
   `Monitor`). Check it once in a real subagent: a refused long wait and an allowed short command.
3. **The main chat** (D7): the brief block in `~/.claude/templates/handoff.md`; `loop-next` sends the block and
   appends it to the round log; a watch command on Herdr states (a new tool or an extension of `subagent-watch`); the
   main chat's rule in `loop-chats.md` and a line in `research-loop.md`, the user's doc changed with it (the figure is
   redrawn only if it shows the chats); a dry run with two dummy chats that checks a brief reaches the main pane and
   the round log, and that a `blocked` state is noticed.
4. **Checks** (D10): which model the `haiku` alias runs now (start a tiny Haiku subagent and read the `model` field
   in its transcript); whether a per-call `effort` takes effect (if it cannot be observed, say so); a rough estimate
   of the combined effect on a round like round 6, from section 3's numbers, labelled as an estimate.
5. **Close:** run `/project/claude-config/sync.sh`, commit and push `claude-config` (it carries D8's settings change),
   `/project/tools` and CoSiR (`git.md`); point the project memory at the new handoff; `wrapup-check`. Then open
   "CoSiR main" with `loop-next --model sonnet --notify` on a short start file that names the open phase chats, and
   tell the user, in the closing brief, what changed, what was checked and the rough estimate.

## 5. Code and pitfalls

- Auto mode may refuse a session's edit of the `hooks` block in `settings.json`. Then do not work around it: put the
  exact lines at the top of the closing handoff and in the brief for the user to add.
- `~/.claude/rules/`, references and templates may be edited by another session: re-read right before each edit.
- `/project/tools` is a shared git repo; commit only your own files. The `unattended-guard` hook refuses deletions in
  this chat: list anything to delete on `.scratch/pending_deletions.md`.
- Keep command output short (`agent-routing.md`); briefs to subagents and Codex carry the wait and no-deletion lines.

## 6. State at handoff

- Running: nothing from this chat. The `r6 read` tab is open and belongs to the research loop.
- Uncommitted: `~/.claude/settings.json` (D8), synced to `claude-config` before this chat closes.
- On disk over 1 GB: nothing.
