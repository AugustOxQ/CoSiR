# Handoff: discuss with the user how subagents wait and how their model and effort are chosen, then whether the build protocol is heavier than research needs

> Written 2026-10-10 by the engine-switch-check chat (session 3a9a33e8, cut at about 400k tokens). Not a loop step: a
> discussion about how we work, between rounds. The changes the user approved are already applied (below).

## 1. Read in this order

1. The user-read report `docs/user_read/2026-10-10_engine_switch_check.md`: rounds 4 and 5 (Superpowers) against
   round 6 (Matt's chain), cost, time, quality, and the proposed changes.
2. The full report `docs/reports/auto/process/2026-11-26_engine_switch_check.md` ("Analysis" and "Proposed changes")
   when you need a number.
3. The rules as they now stand: `~/.claude/rules/agent-routing.md` (model tiers, "No long waits inside subagents",
   Codex first), `~/.claude/references/matt-chain.md` (verdict code, ticket size, "Checking the switch"),
   `~/.claude/references/loop-chats.md` (unattended chats: the stuck-subagent watcher, the context meter in subagents).
4. For agenda item 2: Superpowers' model rules, `/root/.claude/plugins/cache/claude-plugins-official/superpowers/6.4.1/skills/subagent-driven-development/SKILL.md`
   lines 184 to 205 ("Model Selection").

## 2. Decided by the user, not to be reopened

All on 2026-10-10, in the engine-switch-check chat; the rules record them.
- Adopted: a watcher for stuck subagents in unattended chats; implementers kept under about 200k tokens of context
  (the context meter now tells a subagent to hand back); no subagent waits more than about 4 minutes on one command;
  model per ticket with verdict code narrowed; one queue script per GPU stage in run chats; keep the rule review,
  ticket reviews on verdict code, the final review with re-derivation, Sonnet drafts with an Opus claim review.
- Codex first, in the user's words: know Codex's five-hour and weekly limits, give each job the Codex model that fits
  it, give Codex jobs that fit the five-hour window, and use the subscription until its weekly limit reaches zero.
  The user added the allow rule for the Codex command to `~/.claude/settings.json` (2026-10-10).
- Dropped: a rule for the descriptive pass after the verdict (the paper needs those rows; a heavy pass is a grill
  question per spec).

## 3. Where things stand

- Round 6 cost about twice round 5 ($706 against $341 in API-price terms) and took 30h31 against 12h09; quality held.
  Implementers were $293 of the $706; five long ones took 27% of round 6's input tokens.
- **Model and effort, as found.** Superpowers' build skill makes the model a required field on every subagent call
  and asks for the least powerful model that can do each role; rounds 4 and 5 put 15 and 14 subagent calls on Sonnet.
  Matt's implement-spec says nothing about models; our matt-chain.md said Sonnet by default and Opus for verdict
  code, and the controller marked 14 of 15 tickets as verdict code, so 41 of round 6's 50 subagent calls ran on Opus.
  No round ever set a subagent's effort; Opus subagents most likely ran at the `xhigh` set in `settings.json`
  (`modelSettings`; the transcripts do not record effort). The `modelSettings` key for Sonnet is `claude-sonnet-5`,
  which does not match Sonnet 5.5.
- **Waiting, as found.** The new rule (no waits over about 4 minutes) is an instruction only; the watcher
  (`subagent-watch --watch`) catches a subagent silent for over 20 minutes. Between 5 and 20 minutes nothing stops a
  subagent that ignores the rule from paying for its expired cache.
- Tools added today in `/project/tools`: `subagent-watch` (new) and `context-meter` (subagent mode).

## 4. Open points for the next step

The user's agenda, in this order:
1. **Subagent waiting** (key): enforce the 4-minute limit or leave it as an instruction? An option: a hook that, in
   subagents only, refuses a Bash call with a timeout over about 4 minutes and polling loops (`until ... sleep`).
2. **Choosing model and effort** (key): the user thinks Superpowers' standard is the better one (least powerful model
   per role, the model always named explicitly, reviews scaled to the diff's risk). Whether to adopt it in
   matt-chain.md and agent-routing.md in place of the verdict-code split, and whether to set effort per role (an
   untested lever: lower effort means fewer, more consolidated tool calls). Also fix the stale Sonnet key.
3. **The build protocol's weight** (key; the user's concern): this is research, not software development, and most
   edge cases need no handling. Subagents ran over 400k tokens and over an hour; round 6's build documents (tickets,
   contracts, briefs) were about 178 KB against 10 to 20 KB plans in rounds 4 and 5. What can be cut.
4. Keep Matt's chain or switch back to Superpowers: still open (the user leans to Superpowers' model standard; the
   engine choice itself is undecided).

## 5. Code and pitfalls

- Rule edits follow `~/.claude/references/maintaining-rules.md` (re-read first, log, test in a fresh session, sync).
- Auto mode refuses a session's edit of its own permission settings; the user makes those.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (this handoff is committed with its chats.tsv row).
- On disk over 1 GB: nothing.
