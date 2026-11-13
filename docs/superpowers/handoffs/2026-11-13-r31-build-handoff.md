# Handoff: build r31: the A1 run flag and the SigLIP-only encoder pick (SC9) for r29's stage 2, then open the run chat

## Brief
- Done: r29 read (user: promising; it keeps the current model's pattern against the instructed Qwen3); r29 spec amended (A1, A2); r29 storage cleaned.
- State: the SigLIP-only model is an ablation or supplementary row, not the main model (user with co-authors). Stage 2 still runs.
- Next: build the A1 flag and the SC9 encoder pick, final review, merge; open "r31 run" so stage 2 starts at 19:00 or later.
- Blocked: nothing.

> Written 2026-11-13 15:45 by "r31 decide". Loop step reached: after decisions C and A and the spec amendment (step 6
> done); step 7 next. Direction log entry: [2026-11-13 · r29 read](../direction_log.md).

## 1. Read in this order

1. The r29 spec, `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md`, from §1: the Amendments table
   (§4: A1, A2), §3.5 row 2a, §6 (SC9 and its bullet). The amendment is the build's whole scope.
2. `~/.claude/references/matt-chain.md` (the build chain) and `~/.claude/references/codex-jobs.md` (Codex first).
3. The stage 2 code as it stands: `src/test/20261112_siglip_only/r29_run_stage2.sh`, `r29_run_common.sh`
   (`require_pass`), `r29_decide.py` (`groups`, `image_groups`, `k0_baseline`), `r29_ladder.py` (`sheet`). The r29
   build tracker `.scratch/r29-siglip-only/issues/` has the tickets that wrote them (10 to 15 are stage 2).
4. r28's encoder pick for the protocol and the replay target: `src/test/20261112_siglip_main/r28_valpick.py`,
   `records/valpick.json` (SigLIP 1 0.3863, SigLIP 2 0.3856, CLIP L/14 0.3756).

Skills: `to-tickets`, `implement-spec`, `tdd` (verdict code only), `writing-for-agents` (briefs).

## 2. Decided by the user, not to be reopened

- r29's reading (decision C, 2026-11-13): promising; judged against the instructed Qwen3 and plain SigLIP 1, not the
  current model (direction log).
- A1 (14:40, after seeing the recommendation): stage 2 runs although SC2 failed; `records/pass_rule.json` stays
  "fail" and is never edited. Place (15:25, the user with the co-authors): an ablation or supplementary row, not the
  main model ("the fewer numbers we change now the better").
- A2 (14:40, after seeing the recommendation): the encoder pick among the SigLIP-only models on validation (SC9).
  Its rule is r28's (agent default, written in §6).
- All of stage 2 runs (E2, E3, MMArt, photo model, ablation, order D24), on the local GPU from 19:00 (D7).
- r29 storage: decided and done (disk ledger rows of 2026-11-13).

## 3. Where things stand

- r29 stages 0 and 1 are done and reported (auto report 2027-01-18). K* = `siglip1_prefix` (prefixed captions).
  SC2 fail: text group 0.2549 against the bound 0.2573; image group 0.3882 against 0.3833. SC3 level 1.
- Stage 2 code exists and was reviewed in r29's build (dry-runs of parts a and b recorded in the tracker's notes).
  It refuses at `require_pass` because the pass rule failed. No stage 2 output exists yet.
- r29 outputs (5.8 GB, `/data/SSD2/cosir_r29/20261112_siglip_only/`): K*'s models and exports, which stage 2 reads.

## 4. What to build (tickets for `to-tickets`; IDs A1, A2, SC9)

1. **The A1 flag (plumbing, not verdict code).** A committed record `records/amendment_A1.json` (spec commit, date,
   the decision lines of A1 with who decided). `require_pass` accepts a "fail" only when the runner gets an explicit
   flag naming A1 and that record exists, is committed and is clean; otherwise it refuses as now. The dry-run prints
   which path it took. The ladder records and `paper_cells.md` carry one line: "stage 2 ran on the fail under A1".
2. **SC9 (verdict code: yes; `tdd`; its own ticket review).** An `encoder-pick` step: E2's and E3's five-seed image
   groups on validation by the same code path as stage 1b's `groups` (extend it from SigLIP 1 to SigLIP 2 and CLIP
   L/14); K*'s per-seed groups read from `records/groups_siglip1_prefix.json`, not recomputed; c* on Qwen3 as an
   information row from r28's `valpick.json`. The rule, margin, seeds ahead and close call as §6 SC9. Output
   `records/encoder_pick.json`. Regression check before its first new number: on r14's SigLIP 2 and CLIP L/14 c*
   validation exports (the ones r28's pick read), the new path reproduces r28's `valpick.json` encoder scores within
   1e-6 (as gate (i)(c)). If those exports are gone, say so in a decision line and use the closest replay available.
3. **Stage 2 wiring (plumbing).** Part a queues E2's and E3's `groups` (CPU, about 30 minutes each) after their
   validation exports, then `encoder-pick`. The run chat commits `encoder_pick.json` and has Codex re-derive it (C13)
   before part b, as the other picks. `r29_ladder.py sheet` gets an SC9 section.

Coverage: A1 → 1; A2 and SC9 → 2, 3. Integration branch `r31-stage2` from main; merge into main and push after the
final review (Opus `xhigh`), the fix wave and the re-review (Sonnet `high`).

## 5. Code and pitfalls

- Records are gitignored (`*.json`): commit them with `git add -f`.
- Never edit `records/pass_rule.json`; the flag reads A1's own record.
- The stage 2 runner refuses long GPU work on weekdays 10:00 to 19:00 (D7). The build's smoke runs stay short or on
  the CPU; GPU smoke goes through `res-queue gpu --short`.
- New code stays in the r29 round folder with the `r29_` prefix (it is r29's stage 2).
- Codex's untracked notes in `src/test/20261112_siglip_only/rederive/` stay out of git. The link
  `rederive/r29rd_c1_cache` was removed with its cache (38 GB, approved).

## 6. State at handoff, and the run chat

- Running: nothing (queue idle). GPU check passed at 15:30 (driver 595.99.02, library and kernel match).
- Uncommitted: nothing of this chat after its commit.
- On disk over 1 GB: r29 outputs 5.8 GB (keep, disk ledger).
- **For the run chat's handoff (written by this build chat):** stage 2 part a from 19:00 (estimate 6 to 9 hours with
  part b; plan the end before 11:00 Saturday, `run-planning.md`); commit the picks and SC9, Codex C13, then part b;
  the ladders; then the auto report and the user-read report (results brief; A1 flagged "after results"; K* written
  as an ablation or supplementary row). The spec's time box is Saturday 11-14 12:00; pieces not done by then are
  reported as not run (D22). Then it opens the next loop's decide chat.
- The user still runs one approved deletion: the two remote r29 branches (`.scratch/pending_deletions.md`).
