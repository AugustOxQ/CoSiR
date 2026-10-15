# Handoff: build the global rule `routines.md` and its checklist `upkeep.md`, then take the user's scattered ideas

## Brief
- Done: chat rule `occam-razor-chat.md` written and tested; routing updated (Opus medium, fit the task best, subagent chats kept).
- State: the user decided a new rule for chat end, task end and a regular check; nothing of it is built yet.
- Next: build `routines.md` and `upkeep.md`, trim `wrapup-check`; then the user brings scattered ideas to discuss.
- Blocked: nothing.

> Written 2026-10-15 23:47 by the chat-rule chat (CoSiR, outside a loop). Not research-loop work: no direction-log
> entry.

## 1. Read in this order

1. This handoff: the decisions are in section 2.
2. `~/.claude/references/maintaining-rules.md`: the procedure for a new rule (registry, map, changelog, test). Its
   step 5 now uses `--tools Read Grep Glob`; `--allowedTools` let a test session commit and push.
3. The files the new rule points to, without restating them: `~/.claude/rules/git.md` (commit and push before a
   chat ends), `~/.claude/references/loop-chats.md` ("How to cut", "Never cut"), `~/.claude/rules/storage.md`
   ("After every task"), `~/.claude/rules/research-loop.md` ("Wrap-up and handoff", step 6),
   `~/.claude/references/chat-cleanup.md`.
4. `/project/tools/bin/wrapup-check` (what it runs today).

## 2. Decided by the user, not to be reopened

All on 2026-10-15, in the chat-rule chat.

- **One new always-loaded rule, `routines.md`**, about 200 words, easy to maintain; it gets its module entry in
  `/project/claude-config/design/modules.yaml` (user; the names `routines.md` and `upkeep.md` are the agent's
  suggestion, accepted by the user). It points to the owning files and does not restate them.
- **Chat end** (user, after the recommendation):
  - handoff and next chat as now; commit and push in every chat, as `git.md` says;
  - the last reply lists every open question ("For you to decide") and what the user must not miss: a run still
    going, a failed push, a skipped step;
  - the last line, only when truly done (nothing open, nothing running, all pushed), in one of two forms:
    `[Chat done: nothing left here, you can close it.]` or
    `[Chat done: work goes on in tab "<name>"; you can close this one.]`
  - "nothing running" is checked, not assumed: no background shell, no subagent still listed, no GPU lock, no
    cluster job. The user: "many times the chat said it ended but there's still shell running or stuck subagent
    listed".
- **Task end** (any task, in or outside a loop; user, after the recommendation): run `wrapup-check` and the storage
  summary (`storage.md`); background jobs, the GPU lock, cluster jobs and worktrees the task started are finished or
  named in the last reply. Today `wrapup-check` is named only in the research loop's wrap-up, which is why tasks
  outside a loop skip it.
- **Project branch end:** left empty for now (user).
- **The regular check runs only when the user asks** (user's own pick): no hook, no due date printed in any chat.
  Its checklist is a new reference, `~/.claude/references/upkeep.md`:
  1. chats: `chat-organize` cleanup of near-empty chats and names for unnamed ones (moved here from every wrap-up);
  2. rule review: what changed in claude-config since the last check, the word count of the always-loaded rules,
     overlaps and conflicts; the user decides each change;
  3. storage: the disk ledger's "keep" and "deferred" rows, does each reason still hold;
  4. project memory: each memory still true;
  5. git: unpushed commits, old branches, leftover worktrees across the repos in `/project`;
  6. plugin updates: did an update write files back (the CCG block, `claude-code-native.md`);
  7. the monthly effort summary (`effort-summary`) when one is due (moved here from `wrapup-check`).
- **Chat cleanup leaves every wrap-up** (user): `wrapup-check` drops `chat-organize check` and the monthly
  effort reminder; `research-loop.md` step 6 and `chat-cleanup.md` follow.

## 3. Where things stand

- This chat's commits: claude-config `f25ecf6` to `fe1b36f` (the chat rule; routing: model and effort said at launch,
  paper and fact-finding rows, Opus `medium` for writing drafts, "fit the task best, quality first", the worker's
  "do what the brief asks, then stop"); tools `aa2eae6` (`chat-organize` keeps HEADLESS and SCRATCH sessions);
  CoSiR `f0351da` (reverted a glossary row that a test session pushed).
- The new rule's shape follows `occam-razor-chat.md`, which loads in every chat.

## 4. Open points for the next step

- None on the build. After it, the user wants to discuss "some scattered ideas"; take them one at a time, with the
  reply shape of `occam-razor-chat.md`.

## 5. Code and pitfalls

- Steps: write `routines.md` and `upkeep.md`; edit `wrapup-check` (drop `chat-organize check` and the monthly
  reminder; update its docstring) and its pointers (`research-loop.md` step 6, `chat-cleanup.md` "At every
  wrap-up", `docs/research_loop.md` if it names them); module entry in `modules.yaml`; `design-map`, then
  `design-map --check` silent; one changelog line (`loop_templates_changelog.md`, setup); a fresh-session test
  (`--tools Read Grep Glob`, Haiku and one Sonnet run) of a chat end with a background shell still running: the done
  line must not appear; `sync.sh`, commit and push both repos.
- `wrapup-check` is a small tool change, on the Codex list (`agent-routing.md`); read `codex-jobs.md` first.
- Re-read each rule right before editing it: other sessions edit the rules too. `~/.claude/agents/worker.md` is a
  link into claude-config again; edit the repo path.
- No inline who-and-date tags in rules; the changelog line records who asked.
- Tell the user each subagent's model and effort when it launches (`agent-routing.md`).

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing after this handoff's commit.
- On disk over 1 GB: nothing (a few KB of test replies in this chat's scratchpad).
