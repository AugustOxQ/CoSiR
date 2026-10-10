# Handoff: discuss with the user which new plugins are worth installing, by what each costs every chat

## Brief
- Done: fixed-context trim built (fresh chat 35.0k tokens, was 50.2k; `worker` subagents 5.2k, were 42.1k); CLAUDE.md read allowed for workers.
- State: unused plugins and skills are off; ARS only via `claude-paper`.
- Next: the user names the plugins to weigh; measure each one's fixed cost and recommend install, per-chat only, or skip.
- Blocked: nothing.

> Written 2026-10-10 23:27 by the process context chat (session 9986e9be), cut by the context meter (about 200k
> tokens). Not a loop step: a change to how we work, between rounds 6 and 7. Attended: the user decides each plugin.

## 1. Read in this order

1. Sections 2 to 5 of this handoff.
2. `2026-10-10-process-context-close-handoff.md` (this folder), sections 3 and 5: what was trimmed and the measured
   numbers.
3. `~/.claude/opt-in/unused/README.md`: what is on, what is off and why, and how each is restored.

## 2. Decided by the user, not to be reopened

- The five trim changes of 2026-10-10 18:28 (`worker` subagents, hidden skills, ARS via `claude-paper`, path-scoped
  report and spec rules, trimmed rule text), all after the recommendation.
- The CLAUDE.md read allow rule (23:26, the user's ok after the recommendation).
- Slack stays: its only use is reading the W&B alert channel (about 9 channel reads and 3 searches in all
  transcripts; no Slack skill ever used), and its skills cannot be hidden without dropping the plugin.

## 3. Where things stand

- A fresh CoSiR chat starts at 35,038 tokens; the skills list is about 2.3k of it. A `worker` subagent loads no skills
  and no rules, so a plugin's skills and hooks cost main chats (and `general-purpose` subagents), not workers. A
  plugin's PreToolUse hooks still run in every subagent.
- What a plugin adds to every chat: its skills' descriptions in the skills list, its agents' descriptions, its MCP
  tool names (deferred: names only, about 15 tokens each), any SessionStart hook text (ARS printed 2.3k), and the
  run time of its PreToolUse hooks.

## 4. Open points for this chat

1. **The user lists the plugins** to weigh. For each, before recommending: what it does, overlap with what we have
   (Matt's skills, ARS, our `/project/tools`, the cluster CLI), how often the work would use it, and its fixed cost.
2. **Measure, don't guess:** install into a probe (or load it for one session with `claude --plugin-dir <dir>`), run
   `claude -p "/context"` for the per-skill split and `claude -p "ok" --output-format json` for the real first-call
   total (`/context` in print mode undercounts Claude Code's own tools), from a dummy folder under `.scratch/dryrun2/`.
3. **Recommend one of three per plugin:** install for every chat; install but enabled only per chat (the
   `claude-paper` pattern: `claude --settings '{"enabledPlugins":{"<plugin>@<marketplace>":true}}'`); or skip. The user
   decides each; record it in `opt-in/unused/README.md` (or a new "on" list), sync and commit `claude-config`.

4. **Minor, from the last `wrapup-check` (23:28):** propose to the user (a) a chat cleanup of the 25 cleanable
   sessions (mostly the context chat's probes) and (b) the routing review for Codex CLI 0.161.0 to 0.162.1; our
   suggestion: no rule edit, then `routing-check --accept`.

## 5. Code and pitfalls

- **Plugin skills ignore `skillOverrides`**: a plugin's skills come and go with the plugin (tested 2026-10-10).
- **The auto-mode classifier refused a Bash read right after a `settings.json` permission edit** (reason
  "Self-Modification"); the Read tool worked. Edit settings only on the user's yes, and say so.
- **The Bash tool's shell is zsh**: `echo ====` fails (`=` expansion); quote it.
- Probe chats leave headless sessions; about 20 from the 2026-10-10 context chat are left for the next
  `chat-organize check` (they were under 30 minutes old at its cleanup).
- Take every time from `TZ=Europe/Amsterdam date '+%F %H:%M'`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (CoSiR, `claude-config`, `/project/tools` committed and pushed).
- Pending deletions: unchanged (the three dry-run rows of `.scratch/pending_deletions.md`).
- On disk over 1 GB: nothing.
