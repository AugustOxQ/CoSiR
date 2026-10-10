# Handoff: discuss with the user how to shrink the memory and settings that load into every chat and subagent

## Brief
- Done: process build closed and extended: run mode (day or night), phone pings from the main chat, Codex column in the routing table.
- State: about 59 KB of instruction text (global CLAUDE.md, 21 rule files, CoSiR's CLAUDE.md, memory index) loads into every chat; a subagent costs about 55k tokens to start.
- Next: measure the fixed start context, then discuss with the user what to trim.
- Blocked: nothing.

> Written 2026-10-10 18:23 by the process build 2 chat (session c4e7312f, resumed as e65c5296), cut by the context
> meter at about 207k tokens. Not a loop step: a change to how we work, between rounds 6 and 7. Attended: the user
> is here and decides.

## 1. Read in this order

1. Sections 2 to 4 of this handoff.
2. `2026-10-10-process-build-close-handoff.md` (this folder), sections 2 and 4: what the process build changed.
3. `~/.claude/opt-in/unused/README.md`: what was already switched off on 2026-10-09 to keep each chat's fixed
   context small, and how each item is restored.

## 2. Decided by the user, not to be reopened

- D1 to D11 of the two process build handoffs, and on 2026-10-10 18:07 to 18:22: the run mode and its trigger (the
  user's words to the main chat), the phone push on every chat close whether or not the user is at the terminal,
  the default after 30 minutes in day mode and the Codex column (both after the recommendation).
- The user's request for this chat (18:10): "first apply the update, then we discuss the final part which optimize
  the memory and setting that enters each agent after this update". The update is applied; this chat is that
  discussion.

## 3. Where things stand

- Every chat loads: the system prompt, the global `~/.claude/CLAUDE.md`, every file in `~/.claude/rules/` (21 files),
  the project's `.claude/CLAUDE.md`, the project memory index `MEMORY.md`, the skills list, the deferred-tools list,
  and SessionStart hook output (the ARS plugin prints about 40 lines of routing text at every start and resume).
  The four instruction sources together are about 59 KB (`wc -c`, 2026-10-10 18:22).
- A subagent costs about 55k tokens to start (measured 2026-10-09, `agent-routing.md`). Whether subagents load the
  rules and memory is not yet checked.
- The real cost of a loop is context size times calls (round 4: about 490k tokens of context per main-chat call over
  764 calls; `agent-routing.md`). A fixed start context of N tokens is paid, cached, on every call of every chat.
- CoSiR main is closed until the automation starts; the run mode of CoSiR is unset (night).

## 4. Open points for this chat

1. **Measure first, then discuss.** The fixed start context of a fresh chat and of a fresh subagent: the first usage
   record of a fresh `claude -p` session's transcript (cache writes plus reads), and of a subagent's transcript; then
   the share of each source (rules per file, CLAUDE.md files, memory, skills list, plugin hook text, MCP and deferred
   tool lists). Give the user one table, biggest first.
2. **Discuss with the user** what to trim, with a recommendation per item. Candidates to weigh: rules that only some
   chats need moving to references read on demand (as `loop-chats.md` already is); duplicated text between rules;
   plugin hook output (ARS) and plugins or skills no project uses; what subagents receive. The user decides each;
   rules change only through `~/.claude/references/maintaining-rules.md`, with a fresh-session test.
3. Then build what the user approves, sync and commit `claude-config`, and close with a handoff.

## 5. Code and pitfalls

- **The Bash tool's shell is zsh**, though the environment line says fish: use bash syntax (`T=…`, `&&`).
- **PushNotification may report "Not sent — this terminal is active" and still reach the phone** (tested 18:18).
- A folder Claude has never opened stops at the folder-trust question; dummy chats go in `.scratch/dryrun2`.
- Take every time from `TZ=Europe/Amsterdam date '+%F %H:%M'`; never write one from memory.
- Keep command output short; briefs to subagents and Codex carry the wait and no-deletion lines.

## 6. State at handoff

- Running: nothing from this chat.
- Uncommitted: nothing (CoSiR, `/project/tools`, `claude-config` committed and pushed).
- Pending deletions: the three dry-run rows on `.scratch/pending_deletions.md` (under 200 KB together).
- On disk over 1 GB: nothing.
