# CoSiR constitution

> Trimmed on 2026-10-15 when the conditioned line was archived; the new line's spec rewrites it. The full v2 constitution: [archived](../../archive/v2_conditioned/docs/superpowers/constitution.md).

This constitution outranks any single spec. When a spec conflicts with it, the spec changes, or the user amends the
constitution explicitly (the archived constitution's Amendments table has the history). A principle is never quietly reinterpreted to fit a spec. A spec that needs an
exception lists it in its §7 with the reason and the simpler option it rejected.

To change it, the user asks any agent ("amend the CoSiR constitution: …"). The agent edits the principle, notes the change under the title.
Specs written earlier are not rewritten.

## Evidence and comparators

- **C1 Strongest real comparator.** Every headline number MUST stand beside the strongest real alternative: the
  strongest condition-free score measured so far, a prior method replicated on the same split, or the current best.
  Never a lift over nothing or random. *Why:* a claim without a real baseline cannot be judged. *Source:*
  `report-writing.md`; memory `feedback_report-writing-principles`.

## Data and seeds

- **C6 Sample IDs.** Data joins MUST follow the "Sample ID consistency" section of the project's `.claude/CLAUDE.md`.
  *Why:* mismatched sample IDs are the project's most frequent and most dangerous bug. *Source:* `.claude/CLAUDE.md`.

## Pre-registration and statistics

- **C7 Decision rule before code.** A stage with a go bar MUST have a decision rule written from its spec, checked by a
  fresh reviewer on the most capable model, and committed before any code. Where the rule and the spec differ, the rule
  governs. *Why:* a bar written after seeing numbers is not a bar. *Source:* rounds 2 to 5.
- **C8 Prior and disclosure.** The rule MUST state the prior before any number exists, and the spec MUST disclose how
  many variants have been tried on the development seed. *Why:* development numbers after many variants are inflated,
  and the reader needs to know by how much to trust them. *Source:* rounds 3 to 5.

## Process and review

- **C12 Regression check first.** A pipeline reused from an earlier round MUST reproduce that round's recorded numbers
  exactly before any candidate number is computed. A difference stops the work and goes to the user with its cause
  traced. *Why:* a silent change in reused code invalidates every comparison with earlier rounds. *Source:* rounds 2
  to 5.
- **C13 Independent re-derivation.** Every number that decides a verdict MUST be re-derived by separate code before the
  rule is applied. *Why:* the final reviews kept finding wrong load-bearing numbers. *Source:* `final-review.md`;
  memory `feedback_final-review-catches-real-issues`.

## Global rules that apply

- `final-review.md`: the whole-branch review on the most capable model, one fix wave and a scoped re-review before a
  report is committed.
- `agent-routing.md`: subagents implement; the main session launches every real run.
- `shared-resources.md`: GPU work under the lock; CPU-heavy work at 8 threads or fewer.
- `storage.md` and `data-storage.md`: feature caches in `/data/SSD2/pre_extract/`; run folders hold only their own
  outputs.
- `report-writing.md`, `reports-layout.md`: reports in `docs/reports/auto/<line>/`, one row in `reports_sum.md` each.
- `timestamps.md`: times in Amsterdam local time.
