# Handoff: write the global chat rule `occam-razor-chat.md` (make the chat easy to follow)

## Brief
- Done: setup trim committed and pushed; the user decided the new chat rule (goal, Occam's razor, names, reply shape).
- State: `plain-english.md` is 174 words; the rules loaded in every chat total 5,825 words.
- Next: rename `plain-english.md` to `occam-razor-chat.md`, write the rule, add names guidance, test, show the user.
- Blocked: nothing; three small points to confirm with the user (section 4).

> Written 2026-10-15 22:45 by the setup-trim chat (CoSiR, outside a loop). Not research-loop work: no direction-log
> entry.

## 1. Read in this order

1. This handoff: the user's decisions are in section 2.
2. `~/.claude/rules/plain-english.md`: the rule to widen and rename (its words part stays as it is).
3. `~/.claude/references/ste-writing.md`: the on-demand guide; it gets the names guidance and examples.
4. `~/.claude/references/maintaining-rules.md`: the procedure for changing a rule (registry, map, changelog, test).
5. `~/.claude/references/grill.md` and `~/.claude/agents/worker.md`: each gets a short names line.
6. `/project/CoSiR/archive/v2_conditioned/GLOSSARY.md`: examples of the invented names the user wants to stop
   (reader, AFF, B′(A0), DTS-CF, RCA, bar margin, held tiers, recipe replay).

## 2. Decided by the user, not to be reopened

All on 2026-10-15, in the setup-trim chat.

- **The goal:** "make the chat easy to follow" is the ultimate goal. The way to it is Occam's razor: when a thing can
  be done two ways, pick the simpler one. A guideline, not a detailed template (user).
- **One rule, not several:** widen `plain-english.md` and rename it `occam-razor-chat.md` (one rule: user, after the
  recommendation; the file name is the user's own pick).
- **Three parts:** words (STE at 80%, as now), names, the shape of a reply.
- **Scope:** the words part keeps its scope (chat, user-read reports, outside-view reports, results briefs, handoff
  briefs; not auto reports, specs, paper drafts, code). The names part covers everything the user reads, specs and
  auto reports included; code and paper drafts are out (user, after the recommendation).
- **Names** (user, after the recommendation: "That's a good one"):
  - use the field's standard term; a known thing gets no new name ("reader" is a classifier, "steering" is probing);
  - a new name only when the field has none and it will be used across many chats; it goes in `GLOSSARY.md` with
    its plain meaning; the grill proposes names, never adds one silently;
  - no new acronyms; field acronyms (CLIP, AMI, R@1) are fine; an approved name is written out;
  - no symbols for models or methods (B, B′(A0)); a short descriptive name instead; math notation stays in formulas
    and paper drafts;
  - spec IDs and code names stay, but the first time in chat say what they are ("the R@1 criterion (SC2)");
  - the glossary may list an old name beside the new one, so old files stay searchable.
- **Shape of a reply** (user's own):
  - the point first; a two-line summary at the top of a long reply with many parts;
  - then only what the user needs to judge it; the process only when it changes how far to trust the result;
  - at the bottom, "For you to decide": every item that needs the user's answer, numbered, each with the
    recommendation, and a reminder to read them before going on; left out when nothing needs an answer;
  - bullets for unordered lists, numbers for ordered options or lists (a preference, not a hard rule);
  - each enumerated list in one reply uses its own labels, so the user can answer by label: the first 1. 2. 3., the
    second a. b. c., the third i. ii. iii.;
  - a list has at most 6 items (except when showing results); a longer one is cut and continued under a new heading;
  - one reply has at most 2 tables and at most 3 lists (enumerated and bullet together).
- **Four more lines, all added** (user, after the recommendation): (a) a claim not checked says "not checked";
  (b) no internal labels (ticket numbers, run tags, subagent roles) without their meaning; (c) progress notes in one or
  two lines, what is done and what is next; (d) restate a point in a few words instead of "as above" or "item 3".

## 3. Where things stand

- The setup trim is done: `claude-config` commits `6ce7801` to `0aaef30`, `tools` `0e3e08b` (`routing-check` reads
  `codex-jobs.md` too), CoSiR `d558fb2`. `final-review.md` is now a section of `matt-chain.md`. Rules carry no inline
  who-and-date tags (`maintaining-rules.md`, step 4).
- `plain-english.md` belongs to the `documents` module in `/project/claude-config/design/modules.yaml`; its
  changelog is `report_templates_changelog.md`.

## 4. Open points for the next step

1. **The guard on the razor** (key; the user asked whether a simplicity goal lowers quality or hides options). The old
   chat answered: add one line, "the razor picks the simpler of two ways that do the job equally well; simpler never
   means fewer real options, fewer caveats or skipped checks." Show it in the draft and ask the user to confirm.
2. **Who approves a new name** (minor). Read from "That's a good one" as: the user approves each new name. Confirm in
   one line.
3. **How firm the counts are** (minor). Bullets vs numbers is a preference; the limits (6 items, 2 tables, 3 lists)
   were written as firm. Propose: firm for chat replies, with the stated exception for results.

## 5. Code and pitfalls

- Steps: `git mv` the rule in `/project/claude-config/claude/rules/`; write it (about 300 to 350 words, it loads in
  every chat); add a "Names" section with examples to `ste-writing.md`; a names line in `grill.md` (the glossary
  gate; the pinned `domain-modeling` skill is overridden there, not edited) and in `worker.md` (subagents write specs
  and reports too); grep `~/.claude`, `/project/tools` and CoSiR for `plain-english` and fix every pointer
  (`modules.yaml`, `ste-writing.md`, other rules); `design-map`, then `design-map --check` must be silent; one
  changelog line; a fresh-session test (`echo "<scenario>" | claude -p --model haiku --allowedTools Read Grep Glob`)
  that checks the reply shape and that an important option is not dropped; `sync.sh`, commit, push; show the user the
  final rule text.
- Re-read each file right before editing: other sessions edit the rules too.
- `design-map --check` reports any text such as `references/<name>.md` that names a missing file, even inside a
  longer path.
- No inline who-and-date tags in the rule; the changelog line records who asked.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing after this handoff's commit.
- On disk over 1 GB: nothing (the old chat left only a few KB of scratch files).
