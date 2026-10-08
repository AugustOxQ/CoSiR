# The research loop

> Written 2026-10-08. Applies to every project; this is the shared copy, for reading. Agents follow
> `~/.claude/rules/research-loop.md`, which says the same in their terms; when one changes, both change.

![The research loop: read results, wrap up and hand off, decide what to do (steps 2 to 5, repeated until the user
has decided), do it (steps 6 to 9), and read the new results.](assets/research_loop.png)

*The loop has two halves. Steps 2 to 5 decide what to do and repeat until every decision is the user's; steps 6 to 9
carry it out and end in new results. Between reading the results and the next discussion, the work is wrapped up so
that a new chat can start cleanly. Colours show who leads each step.*

## Your three decisions

You read the core of the work and make three decisions in every loop. Everything else is delegated, so that nothing
happens outside your thinking and you do not have to read code-level detail. A decision you only nodded through does
not count as yours: the grill and the results brief are built so that you decide first and see our view after.

| | Decision | When | What you read for it |
|---|---|---|---|
| A | The direction: what to do next, its experiment type and loop size | step 2 | the results brief, and the discussion |
| B | The idea, and the key answers of the grill | steps 4 and 5 | the knowledge note's two or three candidates |
| C | What the results mean | step 1 | the results brief; you give your reading before seeing ours |

Agents decide the rest (implementation, code, the details of a plan). Every choice an agent makes that shapes a result
is shown to you: in the spec's brief under "Check these", and in the report. Approving the spec's brief at step 6
checks that the spec says what you decided; it is not a new decision.

Every record of a decision says who made it: **user** (your own pick), **user, after seeing the recommendation**, or
**agent default**. A key decision you made only after seeing our recommendation is flagged in the spec's brief, so
you can come back to it.

## The steps

| # | Step | Led by | Tool or skill | What it leaves behind |
|---|---|---|---|---|
| 1 | Read results | you (decision C) | the results brief at the top of the report; a user-read report when asked for | your reading of the results |
| | Wrap up and hand off | agent | see below | the direction log entry, a handoff, and a fresh chat for step 2 |
| 2 | Discuss and decide next | together (decision A) | chat | what to do next, its experiment type (`~/.claude/templates/spec_experiment.md`) and loop size |
| 3 | Gain knowledge | agent | brainstorming for options from our own results; deep research for what the field knows | a short note with two or three candidate ideas |
| 4 | Choose the idea | you (decision B) | | the chosen idea and a rough procedure |
| 5 | Grill the idea | together | grill-with-docs, with the changes below | your answers, and terms for the project glossary |
| 6 | Spec | agent | Superpowers brainstorming with the spec templates | the spec; you approve its brief |
| 7 | Plan and build | agents | writing-plans, subagent-driven development | code, tests, task reviews |
| 8 | Run | main session | cluster-run, or the local GPU under its lock | run folders and logs |
| 9 | Verify and report | agents | independent re-derivation, the final whole-branch review, the report | the report with its results brief, and its index row |

## Loop sizes

The experiment type chosen at step 2 sets how many steps run, as it sets how long a spec is. The agent picks the size
from the type and asks you when it is unsure. The figure shows the full loop.

| Size | Types | Steps | Your decisions |
|---|---|---|---|
| Small | exploration that takes hours | 1, wrap-up, 2, a design in chat, 8, a short note | C, A |
| Medium | sweep, ablation, baseline, confirmation, incremental | every step, but step 3 is skipped when the options are already clear, the grill is short and the spec is short | C, A, B |
| Full | new idea, claim test | every step; step 3 includes a literature check: is it new, and what is the strongest comparator | C, A, B |

## The grill (step 5)

The grill makes sure you understand each important choice and that the choices that matter are yours. It runs in
medium and full loops, once per idea; the spec step afterwards does not ask again what the grill settled. A toy run on
2026-10-08 took 2 rounds, about 8 minutes and about 6,000 tokens.

- **Context first.** Each question opens with what the choice is about, why it matters now, and the facts it rests
  on, each number beside its comparator. Each option says what it leads to: cost, what we could claim, what it rules
  out. The agent finds the facts; you are never asked for something it could look up.
- **Key and minor questions.** A question is key if it changes the bar, the data touched, what we can claim, the cost,
  or anything hard to undo. Minor questions come in one batch with our recommended answers. At most three key
  questions per round.
- **You answer first on key questions.** You pick an option and give one line of why; then we show our view and where
  it differs. "Unsure, show me" is allowed, and recorded as such.
- **A scenario check after each key answer**: one concrete "what happens if" case. If it shows we meant different
  things, the question opens again.
- **A closing summary** of three to five lines that you confirm. About three rounds, then we ask whether to go on.
- **Outputs.** New terms go to the project's glossary, `CONTEXT.md` at the repo root, which briefs and reports link
  to. Decisions go to the spec's §4 and the direction log; the grill writes no separate decision files.

## The results brief and decision C

Every automatic report opens with a results brief (template: `~/.claude/templates/results_brief.md`), one screen in
plain words:

- **What we tried**, linking the spec.
- **Outcome against the bar**: the verdict and the few numbers that decided it, each beside its comparator.
- **What it does not show**: the data, the seeds, the claim ceiling of the experiment's type.
- **Options for next**: two or three, each with its cost and what it could tell us.
- **Your reading**, which you write.
- **Our reading and recommendation**, folded away until you have written yours.

At step 1 we show you the brief without our reading, ask for yours in a sentence or two, and only then open ours and
say where they differ. Your reading is the one recorded.

## Wrapping up and handing off

When the results have been read, the session that produced them wraps up before the next discussion starts in a new
chat. The same wrap-up applies whenever work moves to another session, for example a run handed to a new tab.

1. The report and everything it rests on are committed, and the reports index has its row.
2. Your reading (decision C) and the loop's entry go into the **direction log**, `docs/superpowers/direction_log.md`:
   one short entry per loop, newest first, under a "Now" paragraph that says where things stand. CoSiR's log holds
   the loops since 2026-10-02.
3. Storage is cleaned as `storage.md` asks, and anything over 1 GB left behind is listed.
4. A handoff is written in `docs/superpowers/handoffs/` (template: `~/.claude/templates/handoff.md`): read in this
   order, decided by you and not to be reopened, where things stand, open points, code and pitfalls, state at handoff.
5. A one-paragraph start prompt goes beside it, and a Herdr launcher when the next chat opens in a new tab.
6. Project memory gets a one-line pointer (current state, the direction log, the latest handoff), never the history.

## Rules and templates

- The loop itself: `~/.claude/rules/research-loop.md`; templates `results_brief.md`, `handoff.md` and
  `direction_log_entry.md` in `~/.claude/templates/`; changes logged in `loop_templates_changelog.md` there.
- Experiment types and specs: `~/.claude/rules/spec-templates.md` and the templates it names.
- The project's standing principles: `docs/superpowers/constitution.md`.
- Reviews and reports: `final-review.md`, `report-writing.md`, `reports-layout.md`, `user-read-reports.md` in
  `~/.claude/rules/`.
