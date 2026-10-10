# Handoff: the fixed-context trim is built; round 7 continues from the r6 read chat when the user returns

## Brief
- Done: the fixed start context trimmed: lean `worker` subagent type, report and spec rules path-scoped, unused skills hidden, ARS only via `claude-paper`, routing rule trimmed.
- State: a fresh CoSiR chat starts at 35.0k tokens instead of 50.2k; a `worker` subagent at 5.2k instead of 42.1k (general-purpose).
- Next: round 7 from the `r6 read` chat (session 5809c3c0); watch the first build's workers for a missed rule.
- Blocked: nothing.

> Written 2026-10-10 18:43 by the process context chat (session 9986e9be). Not a loop step: a change to how we work,
> between rounds 6 and 7. Attended: the user chose all five changes, after the recommendation.

## 1. Read in this order

1. Sections 2 to 5 of this handoff.
2. `~/.claude/rules/agent-routing.md`, first bullet: Claude subagents are `worker` by default.
3. `~/.claude/opt-in/unused/README.md`, "Off since 2026-10-10": what is hidden and how to restore each item.

## 2. Decided by the user, not to be reopened

All five on 2026-10-10 18:28, after the recommendation (recorded in `loop_templates_changelog.md` 18:41):
1. A lean default subagent type, `worker`.
2. Unused skills hidden with `skillOverrides`.
3. ARS only in paper chats, started with `claude-paper`.
4. Report and spec rules load only when needed; `chat-housekeeping` and `project-layout` moved into references.
5. Rule text trimmed: `agent-routing.md` evidence moved out, `claude-code-native.md` shortened, `data-storage.md`
   merged into `storage.md`.

At the wrap-up (18:42): chat cleanup now, the Codex tool-job estimate recorded, the probe folder deleted.

## 3. Where things stand

Measured on the first API call of a fresh `claude -p` session in CoSiR:

| Start context | Before | After |
|---|---|---|
| Main chat | 50,150 | 35,038 |
| Subagent, `general-purpose` | 42,089 | unchanged (now used only when skills, web or MCP tools are needed) |
| Subagent, `worker` | (new) | 5,218 on its first call, about 6.5k after reading the project CLAUDE.md |
| Subagent, `Explore` | 17,172 | unchanged (it never loaded the rules) |

Where the main chat's cut came from (`/context` estimates): rules and CLAUDE.md files 21.9k to 16.7k, skills list
9.0k to 2.3k, ARS start text 2.3k to 0. Over the last 300 subagents, a Sonnet general-purpose subagent made about 20
calls and its fixed start context was about 63% of everything it read; that is the part the `worker` type removes.

What changed where (`claude-config` `df45e1b`, `tools` `2d698cf`):
- `~/.claude/agents/worker.md` (new): `omitClaudeMd: true`, tools Read, Edit, Write, Bash, Grep, Glob, default model
  sonnet, about 1k tokens of rules in its body.
- `agent-routing.md`, `matt-chain.md`, the user's `research_loop.md`: subagents are `worker`; build briefs name
  the spec, tickets and rules they check against.
- `references/routing-evidence.md` (new): the measurements that were in `agent-routing.md`.
- `report-kinds.md`, `report-writing.md`, `user-read-reports.md` (`paths:` `docs/reports/**`, `docs/user_read/**`,
  `docs/paper/**`) and `spec-templates.md` (`docs/superpowers/specs/**`, the constitution); pointer at the top of
  `docs-layout.md`.
- `settings.json`: `skillOverrides` (13 claude.ai skills off, 8 built-ins user-only, 6 name-only); ARS `false`.
- `claude-paper` (`/project/tools`): `claude --settings '{"enabledPlugins":{"academic-research-skills@...":true}}'`.

## 4. Open points for the next step

- **Minor:** in round 7's first build, check that the `worker` hand-backs respect the GPU lock, output and debug-folder
  rules, and that builds name `subagent_type: worker`. If a worker misses a rule, add its line to `worker.md`
  (`maintaining-rules.md` step 2).
- **Minor:** the Slack plugin's 13 skills (1.2k) stay: plugin skills ignore `skillOverrides`, and a bare Slack MCP
  would need a new OAuth login.
- **Minor, not proposed yet:** `storage.md` (1.6k) and `research-loop.md` (2.7k) still load into every chat; parts
  needed only at wrap-up could move to references in a later trim.

## 5. Code and pitfalls

- **Plugin skills ignore `skillOverrides`** (tested with Slack and ARS keys, prefixed and bare); only built-in and
  claude.ai skills obey it. A plugin's skills go only with the plugin.
- **A path rule delivered by a Write** arrives in the tool result. Haiku flagged a test "codeword" rule as unexpected
  and ignored it, but followed the real `report-writing.md` (and Opus took it plainly). The `docs-layout.md` pointer
  covers a report written before any file is read.
- **A subagent's Read of a CLAUDE.md file asks for permission** in the default permission mode (`claude -p` denied
  it); in auto mode, which loop chats inherit, it worked. A chat started in default mode would see a `worker` stop at
  that read.
- **`/context` in `claude -p` undercounts** Claude Code's own tool schemas; take totals from the first usage record
  (`--output-format json`, or the transcript).
- **The Bash tool's shell is zsh**: `echo ====` fails (`=` expansion); quote it.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (CoSiR, `claude-config`, `/project/tools` committed and pushed).
- Pending deletions: unchanged (the three dry-run rows); the probe folder `.scratch/dryrun2/ctxprobe` (60 KB) was
  deleted at the user's yes; ten old empty or headless chats were trashed (`chat-organize`, restorable 30 days).
- On disk over 1 GB: nothing.
