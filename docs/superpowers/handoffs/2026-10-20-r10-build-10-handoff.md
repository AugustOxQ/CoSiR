# Handoff: build stage 0 and stage 1 of the claim test (tickets, build, reviews, merge), unattended; then open the run chat

## Brief
- Done: the claim test spec and its decision rule, reviewed and committed; night mode on (user asleep).
- State: development emotion AMI, D2 0.096 against PercepT core 0.077 (k-means) and 0.082 (own top topic).
- Next: tickets and build of stage 0 and stage 1 (held inference, held scorer, paper settings PercepT, bootstrap).
- Blocked: nothing.

> Written 2026-10-20 01:45 by "r10 decide 14". Loop step reached: track 1 (claim test) after step 6, the spec and
> rule committed; step 7 (build) next. Track 2 (cluster-level losses) waits for decision B in the morning. Direction
> log entry: [2026-10-20 · Stage 2 model sheet read](../direction_log.md), bullet "Grill, claim test".

## 1. Read in this order

1. The spec, [2026-10-20-claim-test-design.md](../specs/2026-10-20-claim-test-design.md), from §1 onward: arms,
   data roles, measurement, stages, decisions D1 to D29.
2. The binding rule, `src/test/20261020_claim_test/DECISION_RULE.md` @ 9faf18a (committed before any code; a fresh
   Opus `xhigh` reviewer passed it after twelve fixes). Where it and the spec differ, the rule governs. It fixes the
   bootstrap (paintings and seeds), the collapse rule R-COL, and the checks V1 to V4.
3. The facts brief `.scratch/r10-claim-test/facts.md` (not in git): code paths per arm (§3, §5), the paper settings
   edit sites (§6), Leiden code (§7), the scorer's 5,000-row assumptions (§8), timings (§9).
4. `~/.claude/references/matt-chain.md` (the build chain) and `~/.claude/references/codex-jobs.md` (Codex room:
   5 h window 68% left at 01:30; implementer + review + re-derivation fit).

Skills: none named; the chain is `to-tickets`, `implement-spec`, then the final whole-branch review. Use the
`cluster-run` skill only if a ticket touches node408.

## 2. Decided by the user, not to be reopened

- Every grill answer: spec §4, D1 to D15 (direction log bullet "Grill, claim test").
- Night mode and the go-ahead: "Change the mode to overnight run now, so you can automate and hand me the results in
  the morning" (2026-10-20 01:27); spec D29 reads it as the approval of the brief, which the user checks in the
  morning.
- node408 on DAS6 (3 GPUs, reserved) may be used (user, 2026-10-20 01:27); spec D23 says when.

## 3. Where things stand

- Development data (5,000 evaluation paintings, 3 seeds), image-side emotion AMI: D2 0.0962; PercepT core P3 0.0772
  (k = 20) and 0.082 (own top topic); instructed-image target 0.0929. Bar from development data about 0.092.
- No weights exist for D2 or PercepT core: every arm retrains with weights saved (facts §3, §5).
- The held split (12,281 paintings) was never read by this line.

## 4. Open points for the next step

None for the user. Build scope (spec D28): stage 0 and stage 1 with their diagnostics. Stage 2 builds later, in its
own chat, only after the stage 1 verdict allows it.

Ticket pieces the spec implies (the build may split them otherwise; tickets cite D, V, R-COL and S1 IDs):
1. Held painting list, held labels (rule §3), held inference for D2 and the PercepT arms with saved weights.
2. Training entry points that save weights: D2 (from the sheet's trainer), PercepT core P2, P3 (seeds 0 to 4);
   stage 0 also runs core P1 and P4 seed 0 (saved `PC/outputs/stage2/P1__s0.npz`, `P4__s0.npz`).
3. PercepT paper settings (strict and K = 40, arms P1, P4) behind switches, with D21's DEC details; Algorithm 1 cut
   and R-COL as verdict code with unit tests.
4. Held instructed Qwen images, three instructions (12,281 paintings, about 150 MB, shared cache under
   `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/`, new held folders).
5. Held scorer (N from the vectors, rule §3) and the paired bootstrap; verdict file `verdict_stage1.json` written
   before any diagnostic. Verdict code: ticket reviews and unit tests.
6. Stage 0 (spec §3.5) as a script the run chat launches first; diagnostics (spec §3.6) and the Leiden 3a reading.

The C13 re-derivation (rule V2) is not built here: the run chat gives it to Codex with the rule and the data only.
Save every Stage 1 artifact that R-COL and V2 need (center norms, fit counts, kept masks, fit hard topics).

## 5. Code and pitfalls

- New round folder `src/test/20261020_claim_test/`, module prefix `ct_` (`sys.path` clash between round folders).
- The sheet's code asserts 5,000 evaluation rows and 19,889-row inputs (facts §4, §8); write held paths, do not
  loosen the sheet's checks in place.
- PercepT core's Stage 1 k-means start runs on one CPU thread (core spec A2); keep it.
- Partial Qwen extraction: the adapter's expected counts fail for a subset; use `expected=None` (facts §2).
- `docs/superpowers/held_ledger.md` still has an old header ("final scripts check this file and refuse to run");
  `held-data.md` overrides it: no script reads or checks the ledger. The run chat adds one plain row.
- GPU under the lock (`flock -n -o -E 75 /tmp/gpu0.lock ...`); CPU work at 8 threads.
- Unattended: delete nothing; list items in `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing (no GPU lock, background job, Codex job or subagent).
- Uncommitted: nothing after this cut's commit.
- On disk over 1 GB: `src/test/20261019_model_sheet/outputs/`, 3.4 GB, kept (disk ledger).

## 7. For the run chat (pass on in the build's handoff)

- Estimate the end time at the start; if the local run ends after 11:00, PercepT training moves to node408 (D23).
- After the stage 1 reports: open "r10 decide 15" (`--notify`) for the user's reading of stage 1 (decision C) and
  track 2's decision B, from [the cluster-level losses note](../../reports/literature/2026-12-17_cluster_level_losses.md)
  (ranked ideas: 1 teacher-partition anchor, 2 TEMI-style head, 3 comparator probe; the note does not support plain
  DEC from a good partition; key question, user answers first). If the rule allows stage 2 (S1-GO, S1-MM, S1-IN),
  also open "r10 build 11" (`--unattended`) for stage 2. Stage 2's local GPU work waits for the evening or goes to
  node408, unless the user says otherwise.
