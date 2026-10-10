# Handoff: the process build is closed; round 7 runs under the new rules (no chat starts on this file)

## Brief
- Done: process build closed: model and effort per role, `wait-guard`, lean research build, briefs and the CoSiR main chat (dry run passed: brief delivered, tab placed, BLOCKED caught).
- State: estimate for a round like round 6: about $440 to $520 instead of $706.
- Next: round 7 from the `r6 read` chat, under the new rules; after it, the D4 and D5 checks against round 6.
- Blocked: nothing.

> Written 2026-10-10 16:46 by the process build 2 chat (session c4e7312f). Not a loop step: a change to how we work,
> between rounds 6 and 7. No direction log entry. The record that project memory points at; no chat starts on it.

## 1. Read in this order

1. Section 4 (what comes next) and section 3 (the estimate), when judging round 7's cost.
2. The decisions D1 to D11: `2026-10-10-process-build-handoff.md` section 2 and `2026-10-10-process-build-2-handoff.md`
   section 2. Not to be reopened before round 7 ends.

## 2. What changed and how it was checked

| Change | Where | Checked |
|---|---|---|
| Model and effort named on every Agent call; Sonnet the main implementer and reviewer; Opus for verdict-code reviews, the final review, design judgment, escalations; Haiku never writes tested code or gives a verdict | `agent-routing.md`, `matt-chain.md`, `final-review.md` | fresh `claude -p` session answered the routing questions right (`.scratch/process-build/rule_test.txt`) |
| No waits over about 4 minutes in subagents | `wait-guard` hook (`/project/tools`, `fb2b686`), `settings.json` | refused a 600 s Bash timeout in a real Haiku subagent |
| Lean research build: unit tests only for verdict code, plumbing proven by one smoke run | `agent-routing.md`, `matt-chain.md` | same fresh-session test (passed after one rule fix) |
| Brief block on every handoff, sent to the main chat and logged in `briefs.md` | `handoff.md`, `loop-next` | dry run (below) |
| Per-project main chat in its own workspace; `loop-watch` reports NEW, BLOCKED, SILENT, GONE | `loop-chats.md`, `main_chat_start.md`, `loop-next`, `loop-watch` | 50 unit tests; review (5 findings) fixed by Codex and committed (`154d19a`); dry run |
| `loop-watch` reports each blocked or silent spell once across runs | `loop-watch` (`9d8188d`), `loop-chats.md` | 6 new tests (56 in all); found after CoSiR main opened: the main chat restarts the watch after every event, so a chat left blocked or silent would have re-alerted every few minutes |

**Dry run** (three Haiku dummy chats in `.scratch/dryrun2`, closed afterwards):
- The main chat opened in a new workspace (w13).
- The phase chat's brief landed in `briefs.md` and reached the main chat, which replied "got brief".
- The phase tab opened in the main chat's workspace.
- `loop-watch` exited with `BLOCKED blk run (pane w13:p3)`, code 3, for a `--mode default` chat waiting at a `touch` approval.

Two fixes came from the dry run (changelog line 2026-10-10 16:43):
- The rule now reads a needy chat with `herdr agent read <pane> --source visible`; the `recent` sources returned only blank lines for Claude's approval prompt.
- The main chat's start file says that a brief arrives as a pasted block.

## 3. The estimate (an estimate, not a measurement)

Round 6 as the base ($706 dollar weight, the engine switch check's role split and prices). Sonnet halves the price
of cache writes and output and leaves cache reads unchanged, so a role moved from Opus costs about 73% of before.

| Effect | Conservative | Optimistic | Assumption |
|---|---|---|---|
| Wait rewrites gone | −$60 | −$80 | of the measured $80; some waits come back as cheap fresh starts |
| Lean build, fewer ticket reviews | −$89 | −$149 | 20% to 35% fewer lines written and reviewed (round 6: 14,956 test lines of 25,920); reviews only on verdict-code tickets (×0.7 to ×0.5); final review −10% to −25% |
| Implementers on Sonnet | −$55 | −$43 | ×0.73 on what the lean build leaves |
| Escalations to Opus after failed Sonnet fixes | +$25 | +$10 | a few tickets |
| Smaller build documents | −$4 | −$7 | code maps only for new code; reading documents was about $0.40 per implementer |
| **Round total** | **$523 (−26%)** | **$437 (−38%)** | the main chats ($167) unchanged (D9); the new main chat adds a few dollars |

Not counted: Codex first takes implementer tickets while its weekly limit lasts (a further cut in Claude usage),
and the earlier changes of 2026-10-10 14:59 (subagent watchdog, the 200k ticket cap, one queue script per GPU stage),
which act on the same tokens. Build time should drop with the code written; we did not put a number on it.

## 4. What comes next

- **Round 7** continues from the `r6 read` chat (session 5809c3c0), which holds round 6's decision C and wrap-up. Its
  `loop-next` calls log each brief in `briefs.md`; with no live main chat, tabs open in the caller's workspace.
- **CoSiR main** was opened at 16:46 and closed at the user's request at 18:07 (the "CoSiR loop" workspace), until
  the user has read `r6 read` and this build. Reopen it with the command in `loop-chats.md`, "The main chat".
- **Next process changes** (user, 2026-10-10 18:07; not built, the user's answers pending; our recommendation beside
  each):
  1. A run mode on the automation, `day` or `night`. Day: the user checks often, so build and run chats may ask when
     unsure. Night: today's unattended behaviour, aim for results. Ours: a project-level mode the main chat sets on the
     user's word, read by each phase chat before it asks; in day mode a question carries a default.
  2. A phone ping from the main chat each time a phase chat closes ("<project>: <chat> closed", with the brief's line).
     Ours: Claude's push notification (reaches the phone when Remote Control is on), tested once; Slack as fallback.
  3. Answered: the model and effort rule applies to every chat that calls the Agent tool, not only the loop.
  4. Codex first when its five-hour window has room (already the rule since 2026-10-10), plus a Claude-to-Codex model
     mapping. Ours: a Codex column in the routing table of `agent-routing.md`.
- **After round 7** (D4, D5): compare the engine again; the quality check against round 6 (final-review findings by
  severity, any defect that reached the run); a silent wrong number from plumbing that reaches the final review sends
  that kind of code back to unit tests. Then check this estimate with `loop-cost --loop r7`.

## 5. Pitfalls found

- **The Bash tool's shell is zsh**, though the environment line says fish: `set T …` sets nothing. One slip put three
  dummy files under `/docs` on the container layer; they were moved to the scratchpad.
- **A folder Claude has never opened stops at the folder-trust question**, also under a trusted project when the folder
  has its own `.git`. `loop-next --cwd` into a brand-new folder therefore blocks until the user answers (`loop-watch`
  shows it as BLOCKED). Real project folders are already trusted.
- `herdr agent read --source recent…` misses Claude's approval prompt; use `--source visible`.
- **An unattended chat that ends without opening a next chat** (like this one) gets one alert from CoSiR main: SILENT
  after 60 minutes, or GONE if its tab is closed while the watch runs. Expected for this chat, possibly twice (the
  watch running now started before the fix); ignore it.

## 6. State at handoff

- Running: nothing from this chat (CoSiR main closed at 18:07); the `r6 read` tab, not ours.
- Uncommitted: nothing (CoSiR, `/project/tools` and `claude-config` committed and pushed).
- Pending deletions: three dry-run rows on `.scratch/pending_deletions.md` (`.scratch/dryrun`, `.scratch/dryrun2`,
  the dummy transcripts; under 200 KB together).
- On disk over 1 GB: nothing.
