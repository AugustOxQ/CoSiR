# Handoff: process main-per-loop: one main chat and one Herdr workspace per loop, opened by the run chat

## Brief
- Done: the user decided one main chat per loop, each in its own new Herdr workspace with that loop's tabs.
- State: decided, not built. The rule still says one main chat per project; round 8 runs under the old setup.
- Next: change `loop-chats.md`, `research-loop.md`, the main chat's start file, `loop-next` and `loop-watch`; show the
  user the diff.
- Blocked: nothing.

> Written 2026-10-12 21:00 by the `r8 build` chat, after the user's question about the closed main chat. Not a CoSiR loop
> step: a process change to the global loop setup.

## 1. Read in this order

1. `~/.claude/references/maintaining-rules.md`, before any change to a rule, reference, template or tool of the setup.
2. `~/.claude/references/loop-chats.md`, section "The main chat" (opening, watching, restarting), and the "main chat"
   sentence of `~/.claude/rules/research-loop.md` ("One chat per phase", l.53 or near).
3. `~/.claude/templates/main_chat_start.md` (what a new main chat reads first).
4. `/project/tools/bin/loop-next` and `/project/tools/bin/loop-watch` (git repo `/project/tools`, private GitHub
   `tools`): how the main chat's workspace is found (its session id in `chats.tsv`) and which chats are watched.

## 2. Decided by the user, not to be reopened

User, 2026-10-12, after seeing the recommendation ("Yeah I think so that would be what I want. So round 8 build
without it and from the next one this rule applies"):

- **One main chat per loop.** Each loop starts with a new main chat in a new Herdr workspace that holds that loop's
  tabs (main, decide, build, run); the old main chat closes with its loop.
- **The boundary** is the moment the run chat opens the next loop's decide chat. In one step it creates the new
  workspace (for example "CoSiR r9"), opens the new main chat there, then opens the decide tab beside it. The old main
  chat still sees that last event and pings the phone. Then the user closes the old workspace as a whole.
- **When it applies:** round 8 runs without it. It applies from the next loop: the first use is `r8 run` opening
  `r9 decide`, if the change is built and pushed by then; otherwise the boundary after that.

Why (the user's question): the user closed the old main chat at 19:56 together with the finished `r8 decide 2` tab,
reading the setup as one main chat per task; the rule never said when a main chat ends. Nothing watches the main chat
itself, so the gap was found only when the user asked (the `r8 build` chat reopened it at 19:57, tab `w12:t2`).

## 3. Where things stand

The recommendation the user accepted, with its reasons: a main chat makes one call per event, about 5 to 8 per loop,
so a fresh one per loop never reaches the 200k cut; its state lives in files (`chats.tsv`, `briefs.md`, `run-mode`,
`~/.cache/loop-watch/`), so a new one loses nothing (the 19:57 one caught up in 11 seconds); one workspace per loop
makes closing a finished loop one action, with no tab to guess about. The run chat is the boundary because it is a
fixed point every loop; today a decide chat opens the main chat at brief approval and can forget it.

What needs to change (agent's list, accepted as the scope):
- `loop-chats.md` "The main chat": opening (by the run chat at the loop boundary, in a new workspace "<project>
  <loop>"), ending (when the next loop's main chat opens; the user closes the old workspace; the main chat never closes
  itself), "Restarting" (a main chat past its threshold inside a loop still cuts as now), and the decide chat's
  opening step at brief approval (kept only for the first loop of a project or when no live main chat exists).
- `research-loop.md`: the main-chat sentence ("in its own Herdr workspace, where every later tab opens") to say one
  per loop.
- `main_chat_start.md`: the new main chat learns its loop label and that it watches only that loop.
- `loop-next`: one option that, from the run chat, creates the new workspace, opens the new main chat
  (`main_chat_start.md`, Sonnet, `high`) and then the decide tab there with `--notify`; `chats.tsv` records the main
  chat with the loop's label so later tabs find the loop's workspace.
- `loop-watch`: watch only the current loop's chats.
- Optional, to propose to the user (not decided): `loop-next` warns, or reopens the main chat, when the loop's main
  chat in `chats.tsv` is not live in Herdr.

## 4. Open points for the next step

- Show the user the rule and reference diffs before committing them (maintaining-rules.md); the tool changes follow the
  decided scope. Ask about the optional liveness warning.
- Tell the `r8 build 2` chat nothing: it does not need the change. If the change is pushed before `r8 run` cuts, the
  run chat reads `loop-chats.md` before its cut and uses the new option; otherwise it cuts the old way.

## 5. Code and pitfalls

- `/project/tools` and `/project/claude-config` are git repos; commit and push each change; run
  `/project/claude-config/sync.sh` after editing rules or templates. Re-read a rule right before editing it: other
  sessions may edit the same files.
- Do not touch the live chats: "CoSiR main" (`w12:t2`), `r8 build 2` (`w12:t3`) and the old `r8 build` tab (`w12:t1`)
  stay as they are. Test `loop-next` changes without opening real loop tabs (a dry-run or a scratch project label).
- Project memory (`~/.claude/projects/-project-CoSiR/memory/`) names the workspace "CoSiR loop": update that pointer
  when the change lands.

## 6. State at handoff

- Running: the round 8 build (`r8 build 2`, unattended); the main chat "CoSiR main" in workspace "CoSiR loop".
- Uncommitted: nothing of this change.
- On disk over 1 GB: nothing.
