# Handoff: r7 run: start the real chain with `prep`, then stages 1 to 3 in the rule's §9 order, then the reports

## Brief
- Done: the r7 build is merged into main (ddbb600). The final review found no critical defect; one fix wave, re-review PASS.
- State: no stage 1, 2 or 3 number computed. Stage 0 passed (G1 to G4).
- Next: `run_r7.py prep` on real data (about 4 CPU hours with the 14 readers), then ceiling, gate, test.
- Blocked: nothing. Night mode.

> Written 2026-10-12 06:24 by the `r7 build 4` chat. Loop step reached: step 7 done; step 8 (run) next. Direction log:
> this round's entry is written at its wrap-up.

## 1. Read in this order

1. `src/test/20261126_design_l_style_source/DECISION_RULE.md` §9 (order of work, re-derivation, rule smoke) and §7.2
   (the seed-ledger edit before the test seeds are built). The rule governs every step.
2. `run_r7.py`'s docstring and `chain_smoke()` (about line 866): the real order is the same, with the rule smoke
   (`run_r7.py smoke`) before `build`, and each re-derivation (`rederive/rd7_arrays.py`, `rederive/rd7_stats.py`) run as
   its own process before each `--finalize`.
3. The spec, `docs/superpowers/specs/2026-10-12-r7-design-l-design.md`, §3.5, §6, §8 and §9 (stages, success criteria,
   risks, deliverables, budget and time-box: report by Thursday 2026-10-15).
4. `.scratch/r7-design-l/notes/build_log.md` (build 4 entries) and `notes/final_review{,_b,_c}.md`: the report notes
   below come from them.

## 2. Decided by the user, not to be reopened

- Everything of `2026-10-12-r7-build-handoff.md` §2: spec D1 to D27 and the rule, approved 2026-10-12 02:34; build and
  run unattended (night mode).

## 3. Where things stand

The tickets (the build's record; details in `.scratch/r7-design-l/issues/`, outside git):

| # | Ticket | Blocked by | Rule and spec IDs | Verdict code |
|---|---|---|---|---|
| 01 | Foundation `r7_common` | none | §9 read-only, §10 (setup) | no |
| 02 | CLIP token statistics | 01 | L5; D12, D19 | no |
| 03 | Groupings V1 to V6, Z1 to Z6, T, R0 | 01 | L1 to L4, L6 to L9; D12 to D14, D17 to D19 | no |
| 04 | NrKmeans reference N | 03 | L6, L7; D27 | no |
| 05 | Standard and cross-fitted heads | 01 | L10, L11 | no |
| 06 | Reader path for A0+X | 05 | L11 | no |
| 07 | Per-seed bundle | 01 | L11, L14, §7.3 | yes |
| 08 | Scorers | 07 | L12, L13, K1, §7.3; D5, D8, D21 | yes |
| 09 | Seed-42 statistics | none | L15, L16, K1, S1 to S3, P1, §6; D5, D9, D23, D24 | yes |
| 10 | Fresh-seed statistics | none | L16, §7.1, §7.5 to §7.7; D10, D11 | yes |
| 11 | Stage 0 runner | 06, 08 | G1 to G4, §10; D22 | no |
| 12 | Runner: prep, ceiling, gate, finalize | 03, 04, 06, 08, 09 | L12, K1, S1 to S3, P1, §6, §9; D21 | no |
| 13 | Runner: sensitivity, rule smoke, build, test | 10, 12 | §7.1 to §7.7, §9 | no |
| 14 | Re-derivation of arrays | 01 | §9 | yes |
| 15 | Re-derivation of statistics | none | §9 | yes |
| 16 | Descriptive pass | 12, 13 | §5, §7.9; D13 | no |

- Checks on main's code (399c08d, merged as ddbb600): 353 verdict-code tests pass; the whole-chain smoke and the
  descriptive pass on its files ran clean twice (before and after the fix wave); the real stage 0 passed at 04:42.
- Final review (Opus xhigh, in three parts): 0 critical; 3 important, all fixed (finalize did not require the episode
  IDs; `rd7_stats --overlay` filled missing keys silently; no test of `run_r7.py`'s seed and held guards). Stage 0
  re-derived from frozen files: every number matches the record. Re-review (Sonnet high): PASS.
- Real outputs so far: `results/stage0_regression.json`, stage 0's heads (affect, image, caption, R0) and reader R0
  with its banks (839 MB). `prep` keeps them (resume, now checked against the stored heads and bank layout).

## 4. Open points for the run (agent defaults; nothing for the user)

- `prep` is the first full run of `r7_groupings` and `r7_nrkmeans` (only trial subsets and fixtures so far). If it
  fails, debug per `diagnosing-bugs` within the spec's budget; a failure that cannot be fixed is reported as a result.
- The seed-ledger edit (§7.2) is a docs edit the run chat makes; `build` prints it and records `ledger_edit_due`.
- **Notes for the auto report** (not defects; from the final review): m3 the orth step's `pinv` keeps a 64th tiny
  singular value (the rule's formula; state the recorded `pinv_rank`); m6 standard base heads of affect, image,
  caption are fitted but unused in scoring; m7 the token outlier filter dropped nothing; B2 stage 0's G3 record lacks
  reader R0's bank arrays (the reviewer compared all 10 with round 1's: equal); B5 stage 0 ran on e9bc051 (only resume
  logic changed since); Phase A re-derives V1 to V6 only, so the reference scorers and the random-control flag rest on
  the pipeline's arrays (the rule asks no more); three mutations of the new bank-layout check survive its test (a
  file-boundary assert, not verdict code).
- **Engine check** (`matt-chain.md`, safeguard 3): round 7's final review found 0 critical and 3 important findings;
  report whether any defect reaches the run.

## 5. Code and pitfalls

- Run from main's round folder with `R7_RESULTS` unset; real outputs go to `results/`, write-once (a fix writes
  `<name>_fix<n>`; the descriptive pass then refuses until the user decides, rule §8).
- Readers: `r7_reader --jobs 3` (8 threads each). Never compute a number out of the rule's order.
- Tests in a worktree need `--rootdir=.` (see `2026-10-12-r7-build-3-handoff.md` §5). Codex launches are refused at
  random by the auto-mode check; a refused job goes to a Claude worker. Codex 5h window resets 08:26.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing in CoSiR's git (`.scratch/` is outside git).
- On disk over 1 GB: `results/smoke/chain/` 1.2 GB and the session scratchpad's `smoke_fix/` 1.2 GB, with the earlier
  trial outputs (about 7 GB), all on `.scratch/pending_deletions.md`; the token cache on SSD2 (1.1 GB, keep);
  `res/cluster_jobs/` 3.1 GB (round 6, keep). Worktrees `/project/CoSiR-wt/r7-*` and the ticket branches are all merged
  and listed there too.
