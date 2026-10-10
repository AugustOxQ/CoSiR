# Handoff: finish the process build: fix the review findings on loop-next and loop-watch, dry-run the main chat, estimate, close and open "CoSiR main"

## Brief
- Done: rules for D2 to D6 and D9 edited and passed a fresh-session test; `wait-guard` built, installed, refused a long wait in a real subagent; `loop-next` brief delivery and `loop-watch` built (45 tests); main chat rule, start template, figure.
- State: `haiku` runs claude-haiku-5-5; effort is recorded in each subagent's `.meta.json`, and Sonnet wrote 3,636 output tokens at low against 5,039 at high on one task.
- Next: review fixes, dry run with dummy chats, the estimate, then close and open CoSiR main.
- Blocked: nothing.

> Written 2026-10-10 by the process build chat (session 98aeaa02), cut by the context meter at about 210k tokens.
> Not a loop step: a change to how we work, between rounds. No direction log entry. Unattended.

## 1. Read in this order

1. Sections 2 and 4 of this handoff.
2. `2026-10-10-process-build-handoff.md` (this folder), sections 2, 3 and 5: the decisions D1 to D10, the round 6
   numbers for the estimate, the pitfalls. Do not redo its section 4 items 1 and 2; they are done (below).
3. `~/.claude/references/loop-chats.md`, section "The main chat": what the dry run must show.
4. Only when fixing: the Codex brief `.scratch/process-build/codex_loop_main.md` and the review findings in section 4.

## 2. Decided by the user, not to be reopened

- D1 to D10 as in the previous handoff.
- **D11, the main chat's workspace** (user, 2026-10-10, in this chat): when the automation starts, `loop-next` creates
  a new Herdr workspace and opens the main chat there; every later phase chat opens its tab in that workspace, found
  by the main chat's session id, never by tab position ("which can be wrong if I close previous tabs").

## 3. Where things stand

- **Rules** (claude-config, committed and pushed at this cut): `agent-routing.md` (model and effort table, Haiku limits,
  escalation, `wait-guard` line, unit tests only for verdict code), `matt-chain.md` (models per role, debugging,
  ticket documents, the lean research build, D4 recorded), `final-review.md` (Opus at xhigh), `loop-chats.md` (main
  chat section, brief delivery, tab placement), `research-loop.md` (main chat line), templates `handoff.md` (brief
  block above section 1) and new `main_chat_start.md`, the user's doc `docs/research_loop.md` with its figure (main
  chat node, checked by eye), the changelog line in `loop_templates_changelog.md`, five Codex ledger rows.
- **Fresh-session test** (`claude -p --model haiku`): models and efforts per role, escalation, `sleep 600` refused,
  tab placement all right; it first missed that plumbing gets no unit tests, so `agent-routing.md` now says so, and
  the re-test passed. Output: `.scratch/process-build/rule_test.txt`.
- **`wait-guard`**: committed in `/project/tools` (`fb2b686`), registered in `settings.json` (`PreToolUse`, matcher
  `Bash|Monitor`; auto mode allowed the installer). Live check in a Haiku subagent: Bash timeout 600000 refused by the
  hook (logged in `~/.cache/wait-guard/denied.log`), `echo` allowed; `sleep 300` was refused by Claude Code's own
  sleep guard before the hook saw it.
- **`loop-next` and `loop-watch`**: written by Codex (thread `01a12636-3d1d-73e0-af04-cd031fb7e762`, gpt-6.1-sol
  medium), 45 tests pass (`python3 -m unittest -q tests/test_loop_next.py tests/test_loop_watch.py` in
  `/project/tools`), `loop-watch --once` read the real Herdr correctly. Reviewed (section 4, item 1) and committed with
  the README rows; live from the working tree (`~/.local/bin` links to it).
- **D10 checks**: the `haiku` alias runs `claude-haiku-5-5` (transcript `model` field). Effort is recorded in the
  subagent's `.meta.json` (`"effort": "low"`), so the previous handoff's "transcripts do not record effort" is out of
  date. One Sonnet pair on the same puzzle: 3,636 output tokens at low, 5,039 at high (one sample, weak evidence that
  effort takes effect; thinking text is not stored).
- Codex: 4 points of the 5h window used, weekly 80% left.

## 4. Open points for the next step

No user decisions are open. The work, in this order:

1. **Review findings** on `loop-next`/`loop-watch` (Sonnet high reviewer): the existing `loop-next` path is unchanged;
   no blocker or major. Fix these five in one Codex fix round by resume (`codex-jobs.md`; thread above; a fresh Sonnet
   fixer at medium if Codex fails), with a test each, then rerun the tests and commit only those files:
   - `loop-next` about line 205: the `tab rename` after `--new-workspace` uses `check=True`; a failure exits before
     `claude` starts and loses the chat. Use `check=False`.
   - `loop-watch`: a Herdr failure sleeps `--interval` before retrying even with `--once` (about 4 minutes to exit 6).
     No sleep in `--once`.
   - `loop-next` `extract_brief`: `## Brief` followed directly by a heading gives an empty brief with no warning; use
     the fallback line and warn.
   - `loop-watch`: a `main` row counts as its opener's successor, so the chat that opens the main chat (often a
     decide chat) stops being watched. Rows with loop `main` never make their opener finished.
   - `loop-watch`: every child of `claude` counts as a background job, including a foreground tool call's shell, so a
     `working` chat gets the 480-minute limit. When the status is `working`, use `--silence` whatever the count.
2. **Dry run** (handoff D7): a temp project with `docs/superpowers/handoffs/` in your scratchpad (`git init`), a dummy
   main chat opened with `loop-next <start> --label main --main --new-workspace "dry run" --model haiku --keep-label
   --cwd <temp>`, then a dummy phase chat opened with `--cwd <temp> --keep-label --model haiku` on a handoff that has
   a brief block. Check: the brief is in `<temp>/docs/superpowers/handoffs/briefs.md` and reached the dummy main pane
   (`herdr agent read`), the phase tab opened in the dry-run workspace, and `loop-watch --project <temp>` reports
   `BLOCKED` for a dummy chat started with `--mode default` whose handoff asks it to run a command that needs
   approval. Close the dummy tabs and the workspace afterwards (`herdr tab close`, `herdr workspace close`; not
   deletions); the temp folder stays in the scratchpad.
3. **The rough estimate** of the combined effect on a round like round 6, from the previous handoff's section 3
   (labelled as an estimate): the model shift (41 of 50 Opus calls to mostly Sonnet), the wait rewrites (about $80),
   the lean build (round 6's 15,000 test lines), the smaller build documents.
4. **Close:** `/project/claude-config/sync.sh`, commit and push `claude-config` (whatever changed since this cut), `/project/tools`, CoSiR (this handoff, `chats.tsv`, `briefs.md`); project memory points at
   this handoff's successor; `wrapup-check`. Then open the main chat:
   `loop-next ~/.claude/templates/main_chat_start.md --label main --main --new-workspace "CoSiR loop" --model sonnet --effort high --notify`
   and give the user the closing brief: what changed, what was checked, the estimate.

## 5. Code and pitfalls

- `loop-next` is live from `/project/tools`'s working tree: the `r6 read` chat (session 5809c3c0) may call it at any
  time to open round 7's chats. Keep the existing path working at every step; if a fix is risky, test it before
  saving over `bin/loop-next`.
- This chat and its successor are marked unattended: deletions are refused; list anything on
  `.scratch/pending_deletions.md`. `.scratch/process-build/` holds the Codex briefs.
- Auto mode refused a Codex call written as a compound command (`codex e … ; echo …`); the plain documented form
  (`codex-jobs.md`) passed.
- Keep command output short; briefs to subagents and Codex carry the wait and no-deletion lines.

## 6. State at handoff

- Running: nothing from this chat.
- Uncommitted: nothing (claude-config synced, committed and pushed at this cut; `/project/tools` committed).
- On disk over 1 GB: nothing.
