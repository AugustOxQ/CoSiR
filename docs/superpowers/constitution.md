# CoSiR constitution

> Version 1 · Adopted 2026-10-08 by the user · Last amended 2026-10-08
> Scope: CoSiR v2 experiments on `main` (aspect episodes on ArtELingo and the paper tests)

This constitution outranks any single spec. When a spec conflicts with it, the spec changes, or the user amends the
constitution explicitly (a row below). A principle is never quietly reinterpreted to fit a spec. A spec that needs an
exception lists it in its §7 with the reason and the simpler option it rejected.

To change it, the user asks any agent ("amend the CoSiR constitution: …"). The agent edits the principle, adds a
row under Amendments and raises the version. Specs written earlier are not rewritten.

## Evidence and comparators

- **C1 Strongest real comparator.** Every headline number MUST stand beside the strongest real alternative: the
  strongest condition-free score measured so far, a prior method replicated on the same split, or the current best.
  Never a lift over nothing or random. *Why:* a claim without a real baseline cannot be judged. *Source:*
  `report-writing.md`; memory `feedback_report-writing-principles`.
- **C2 Matched control.** A conditioned scorer MUST be compared with its matched control: the identical score with only
  the condition removed (uniform weights in place of the reader, same terms and weight budget, max-R@1 cross-fit). The
  matched control is a pre-registered comparator. *Why:* R@1 = (either + gain) / 2, so any condition-free addition
  inflates R@1 against a control that lacks it; N1 faked a pass this way on 2026-10-04. *Source:* memory
  `v2-matched-control-lesson`.
- **C3 Beat the current best.** A candidate MUST beat the current best method in a paired check before it replaces it,
  and that check is one of its GO checks. *Why:* clearing the baseline bar alone can leave a candidate worse than what
  we already have. *Source:* rounds 4 and 5 specs (user decision 2026-10-07).

## Data and seeds

- **C4 One development seed, several fresh test seeds.** Development and picks MUST use one episode seed (42). A pick is
  tested on several fresh seeds, reported per seed and pooled, each hash-checked against every seed used before and
  recorded in `docs/superpowers/episode_seed_ledger.md`. No extra single-look or lucky-seed rules beyond this. Smoke
  seeds (9001 to 9003) are wiring checks, never results. *Why:* fresh seeds are enough protection; more ceremony costs
  attention without changing decisions. *Source:* memory `feedback_seed-handling-light`; the ledger.
- **C5 Held data.** Held rows and final test splits (ArtELingo held rows, CUB test, SemArt, GeneCIS) MUST be read only by
  a paper test. Every read is one row in `docs/superpowers/held_ledger.md`, within its budget (one main and one reserve
  read per dataset), and final scripts refuse a second read. *Why:* the paper claims rest on data no selection has
  seen. *Source:* the held ledger; CVPR plan spec §10.
- **C6 Sample IDs.** Data joins MUST follow the "Sample ID consistency" section of the project's `.claude/CLAUDE.md`.
  *Why:* mismatched sample IDs are the project's most frequent and most dangerous bug. *Source:* `.claude/CLAUDE.md`.

## Pre-registration and statistics

- **C7 Decision rule before code.** A stage with a go bar MUST have a decision rule written from its spec, checked by a
  fresh reviewer on the most capable model, and committed before any code. Where the rule and the spec differ, the rule
  governs. *Why:* a bar written after seeing numbers is not a bar. *Source:* rounds 2 to 5.
- **C8 Prior and disclosure.** The rule MUST state the prior before any number exists, and the spec MUST disclose how
  many variants have been tried on the development seed. *Why:* development numbers after many variants are inflated,
  and the reader needs to know by how much to trust them. *Source:* rounds 3 to 5.
- **C9 Standard statistics.** Unless a spec says otherwise: comparisons are paired per anchor; the bootstrap resamples
  anchor paintings as clusters (one cluster per painting across seeds), 5,000 resamples, seed 42, with cross-fit picks
  fixed before resampling; "lower bound above 0" means the 95% lower bound is strictly greater than 0; a GO needs every
  check to pass. *Why:* episodes that share a painting are not independent. *Source:* round 4 decision rule §2.
- **C10 Development bar.** Unless a spec says otherwise, the development bar is round 2's D12: a bar margin of at least
  +0.5 R@1 over the strongest condition-free comparator, its lower bound above 0, and the gain statistic's lower bound
  above 0. *Why:* one bar across rounds keeps rounds comparable. *Source:* round 2 decision rule, D12.
- **C11 Reading a failure.** A failed check with its point above 0 MUST be reported as "inconclusive at a detectable
  margin of x", and at or below 0 as "did not beat <comparator>". *Why:* a small sample is not evidence of no effect.
  *Source:* rounds 3 and 4.

## Process and review

- **C12 Regression check first.** A pipeline reused from an earlier round MUST reproduce that round's recorded numbers
  exactly before any candidate number is computed. A difference stops the work and goes to the user with its cause
  traced. *Why:* a silent change in reused code invalidates every comparison with earlier rounds. *Source:* rounds 2
  to 5.
- **C13 Independent re-derivation.** Every number that decides a verdict MUST be re-derived by separate code before the
  rule is applied. *Why:* the final reviews kept finding wrong load-bearing numbers. *Source:* `final-review.md`;
  memory `feedback_final-review-catches-real-issues`.

## Claims

- **C14 Claim scope.** A GO on fresh seeds of the selection rows MUST be worded as "beats <comparators> on new episodes
  from the same selection paintings": not transfer to new paintings, not a margin on each aspect pair. Paper claims come
  only from a paper test on held data (C5). *Why:* fresh episodes reuse the same paintings. *Source:* round 4 spec §4;
  the claim ceilings in `~/.claude/templates/spec_experiment.md`.

## Global rules that apply

- `final-review.md`: the whole-branch review on the most capable model, one fix wave and a scoped re-review before a
  report is committed.
- `agent-routing.md`: subagents implement; the main session launches every real run.
- `shared-resources.md`: GPU work under the lock; CPU-heavy work at 8 threads or fewer.
- `storage.md` and `data-storage.md`: feature caches in `/data/SSD2/pre_extract/`; run folders hold only their own
  outputs.
- `report-writing.md`, `reports-layout.md`: reports in `docs/reports/auto/v2/`, one row in `reports_sum.md` each.
- `timestamps.md`: times in Amsterdam local time.

## Amendments

| Version | Date | Change | Why | Decided by |
|---------|------|--------|-----|------------|
| 1 | 2026-10-08 | Drafted from memories, rules and the rounds 2 to 5 specs and rules | Specs restated these every round | user (adopted as drafted) |
