# CoSiR constitution

> Trimmed on 2026-10-15 when the conditioned line was archived; the new line's spec rewrites it. The full v2 constitution: [archived](../../archive/v2_conditioned/docs/superpowers/constitution.md).
> Amended 2026-11-10 by the user: C7 applies to a claim ladder instead of a go bar; new section "Decisions agents never
> make alone".

This constitution outranks any single spec. When a spec conflicts with it, the spec changes, or the user amends the
constitution explicitly (the archived constitution's Amendments table has the history). A principle is never quietly reinterpreted to fit a spec. A spec that needs an
exception lists it in its §7 with the reason and the simpler option it rejected.

To change it, the user asks any agent ("amend the CoSiR constitution: …"). The agent edits the principle, notes the change under the title.
Specs written earlier are not rewritten.

## Evidence and comparators

- **C1 Strongest real comparator.** Every headline number MUST stand beside the strongest real alternative: the
  strongest condition-free score measured so far, a prior method replicated on the same split, or the current best.
  Never a lift over nothing or random. *Why:* a claim without a real baseline cannot be judged. *Source:*
  `report-writing.md`.

## Data and seeds

- **C6 Sample IDs.** Data joins MUST follow the "Sample ID consistency" section of the project's `.claude/CLAUDE.md`.
  *Why:* mismatched sample IDs are the project's most frequent and most dangerous bug. *Source:* `.claude/CLAUDE.md`.

## Pre-registration and statistics

- **C7 Decision rule before code.** A stage with a claim ladder MUST have a decision rule written from its spec,
  checked by a fresh reviewer on the most capable model, and committed before any code. Where the rule and the spec
  differ, the rule governs. *Why:* a ladder written after seeing numbers is not a ladder. *Source:* rounds 2 to 5.
- **C8 Prior and disclosure.** The rule MUST state the prior before any number exists, and the spec MUST disclose how
  many variants have been tried on the development seed. *Why:* development numbers after many variants are inflated,
  and the reader needs to know by how much to trust them. *Source:* rounds 3 to 5.

## Process and review

- **C12 Regression check first.** A pipeline reused from an earlier round MUST reproduce that round's recorded numbers
  exactly before any candidate number is computed. A difference stops the work and goes to the user with its cause
  traced. *Why:* a silent change in reused code invalidates every comparison with earlier rounds. *Source:* rounds 2
  to 5.
- **C13 Independent re-derivation.** Every number that decides a verdict MUST be re-derived by separate code before the
  rule is applied. *Why:* the final reviews kept finding wrong load-bearing numbers. *Source:* the final whole-branch review
  (`~/.claude/references/matt-chain.md`).

## Decisions agents never make alone

Kinds of decision only the user makes, and what a build or run chat does with each at night, when nobody answers
(`~/.claude/references/loop-chats.md`). "Flag" means one line at the top of the handoff and in the report's results
brief. Resources (the queue, deletions, the 11:00 end, DAS6) are in `loop-chats.md`.

| # | Kind of decision | What a build or run chat does at night |
|---|---|---|
| 1 | **Pre-run commitments**: the claim ladder's levels and sentences, margins, the priors and the surprise line | Apply the written ladder and report the level each result reaches. If a level looks wrong, report it and flag the doubt. Never edit a level. |
| 2 | **Held rows shaping any choice**: a pick, a tuning step, a threshold, a re-run with changed settings after a held number is seen | Stop that path. If the record of the development pick is missing, that piece is inconclusive, as the spec says. Never re-pick. |
| 3 | **Frozen things**: anything the spec or an earlier round froze (weights, the read-out menu, hit rules, test cuts, an earlier stage's picks and ranges) | Stop. Report any pressure to change one (for example a failed gate that a different frozen value would fix). |
| 4 | **A failed gate**: loosening a tolerance, changing a scoring definition so a replay passes | Fix code bugs only, at most three attempts per gate. Never loosen a tolerance. After three tries, or if the cause is in the rule, park that piece as inconclusive (the spec's §6), flag it, and go on with the rest. |
| 5 | **Changing which rows or texts are scored**: dropping rows, cleaning labels, a rule for truncated or odd texts, new rows | Take the spec's named default (if it names none: keep every row, cut texts at the encoder limit, as r22 did), flag it and its count at the top of the report. If a level cannot be computed, park that cell. Do not invent a rule. |
| 6 | **New experiments**: other encoders, retraining a frozen model, extra data sets or splits, any run the spec does not list | Park. One line under "Parked ideas" in the handoff. Do not run. The spec's diagnostics (§3.6) may run, after the levels are written. |
| 7 | **Statistics where the spec is silent**: sample sizes, extra draws, a correction for many cells, an interval unit, a per-seed pick | If the choice cannot change the level a result reaches, take a simple named default and write a decision line. If it can, stop that cell and park it. |
| 8 | **Wording of claims**: paper sentences, why a setting helps, any cause the numbers do not show, "better" outside a named cell | Write numbers and tables. Fill only the slots of the sentence the level reached allows. Label every reading "our reading". |
| 9 | **Words to people, and citations**: anything for the other authors or a supervisor, any new reference | Do not send. Do not add. Park as a line in the handoff for the user. |

## Global rules that apply

- The final whole-branch review (`~/.claude/references/matt-chain.md`): on the most capable model, one fix wave and a
  scoped re-review before a report is committed.
- `agent-routing.md`: subagents implement; the main session launches every real run.
- `shared-resources.md`: GPU work under the lock; CPU-heavy work at 8 threads or fewer.
- `storage.md`: feature caches in `/data/SSD2/pre_extract/`; run folders hold only their own
  outputs.
- `report-writing.md`, `docs-layout.md`: reports in `docs/reports/auto/<line>/`, one row in `reports_sum.md` each.
- `timestamps.md`: times in Amsterdam local time.
