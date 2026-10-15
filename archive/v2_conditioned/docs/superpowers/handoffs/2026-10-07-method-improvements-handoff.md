# Handoff: improve the method on top of AFF (the brainstorm's remaining ideas), after round 3's GO

Written 2026-10-07 00:02 (Amsterdam) by the round3-affect-gate tab of the CoSiR Herdr workspace, for a fresh agent in a
new tab. The user drives the decisions; the ones already taken are in §2 and are not to be reopened. The user restarts
the container before the new tab is opened (see §8).

## 1. Read in this order

1. **The user-read briefing** `docs/user_read/2026-10-06_reader_fix_affect_steering.md`: rounds 1 to 3 in plain words,
   the two options the user chose between (A paper path, B method first), and the open decisions. Then round 1's
   briefing `docs/user_read/2026-10-06_reader_fix.md` only if you need more background.
2. **Round 3's full report** `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md` (final-reviewed), in particular the
   Summary, §3 (method), §7 (per aspect pair), §8 (mechanism: the random-share control), §10 (limitations) and §12
   (what follows). Round 3's rule `src/test/20261121_round3_affect_gate/DECISION_RULE.md` is the template for the next
   rule (pre-registered fresh-seed test, matched counterpart, integer cross-fits, re-derivation agreement, outcome
   table). Its run log `src/test/20261121_round3_affect_gate/20261121_round3_affect_gate_log.md` is the process that
   worked. Its final review `src/test/20261121_round3_affect_gate/final_review/final_review.md` holds S1 (the mechanism
   reading) and N11 (tests to add before the code is reused).
3. **The brainstorm** `docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md`: §3.2 (idea 2, CSD as detector
   evidence), §3.3 (idea 3, GoEmotions placement of captions), §3.4 (idea 4, style × genre abstention), §4 (other ideas),
   §5 (what not to do), §6 (disclosures). Its code is in `src/test/20261120_r1_levers_brainstorm/` (`bs_07_detector.py`,
   `bs_08_csd_evidence.py`, `bs_04_readers.py` with `CSDABST` and `IMGABST`, `bs_03_sxg.py`; harness `bs_lib.py`,
   unreviewed, exploratory). Its numbers were computed on R1, not on AFF.
4. **Round 2's rule** `src/test/20261118_reader_fix_round2/DECISION_RULE.md` for the development-bar machinery (D10 to
   D12, the carry rule) if the next round has a development step; **the no-caption spike**
   `src/test/20261119_no_caption_csd_spike/20261119_no_caption_csd_spike_log.md` (the A1 reader's P(affect), which idea
   2 can use as a detector).
5. Project memory: `project_v2-publication-plan-pending.md` (top entries), `v2-matched-control-lesson.md`,
   `feedback_seed-handling-light.md`, `feedback_final-review-catches-real-issues.md`; global rules in `~/.claude/rules/`
   (`agent-routing.md`, `final-review.md`, `shared-resources.md`, `storage.md`, `timestamps.md`,
   `debug-and-changelog.md`, `reports-layout.md`, `report-writing.md`, `user-read-reports.md`).

## 2. Decided by the user (2026-10-07), not to be reopened

- **Round 3's verdict GO stands.** AFF (round 1's learned reader R1 with its confidence gate opened only when the
  reader picks the affect grouping), frozen as tested, beat cosine, RCA, B, B′(A0) and its matched counterpart on the
  fresh seeds 49 to 51; the pre-registered secondary check (AFF minus R1) passed. AFF is the current best and the
  reference for any improvement.
- **The user chose option B of the briefing: improve the method first, following the brainstorm's remaining ideas,**
  before the held-split paper test. The held-split test with AFF frozen is deferred, not dropped.
- **Process as in rounds 2 and 3:** a short spec the user approves (brainstorming flow: propose, the user approves, one
  question at a time); the rule written from it; a fresh Opus check of the rule (not an ARS round); fixes; commit before
  any code; the rule sent to the user with SendUserFile (`proactive`); a short implementation plan and one question for
  the execution method (subagent-driven has been the norm); subagents implement; the main session launches every real
  run (CPU only); an independent agent re-derives every decision number with its own code; the rule applied; the
  whole-branch final review on the most capable model (one fix wave, a scoped re-review) **before** the report is
  committed; the full report; a user-read report when the user asks.
- **Commits:** allowed on `main` without asking, scoped: spec and rule before any code, then one commit per finished
  step. Stage by explicit path; never `git add -A`; never push; `bin/`, `docs/paper/` and `.DS_Store` stay untracked;
  `.claude/` is gitignored. End commit messages with your session's attribution lines.

## 3. Where things stand (facts; R@1 in percentage points, pooled over seeds 49 to 51, 36,864 episodes)

| Scorer | R@1 | AFF minus it [95% interval] |
|---|---|---|
| AFF | 18.88 | |
| B′(A0), strongest aspect-blind scorer | 18.29 | +0.591 [+0.462, +0.729] (the bar margin) |
| AFF's matched counterpart | 18.08 | +0.796 [+0.670, +0.920] |
| B | 18.07 | +0.806 [+0.686, +0.930] |
| R1 (round-1 R-c) | 18.68 | +0.202 [+0.093, +0.309] (pre-registered secondary) |
| cosine / RCA | 13.04 / 13.16 | +5.84 / +5.72 |

- **Per aspect pair** (bar margin against B′, descriptive): emotion × style +0.810, emotion × genre +1.544, **style ×
  genre −0.580 [−0.793, −0.363]**. The pooled claim rests on the two emotion pairs.
- **Either-rate cost:** −1.727 against the counterpart (about 0.52 points of either rate per point of condition gain).
- **Mechanism** (final review S1, our reading): AFF's label-free part behaves like a visual-contrast rule. When both
  visual groupings have Δ < 0, the reader picks affect in 93 to 96% of condition a while the affect Δ is about 0. It
  opens the gate on the style side of style × genre 77.6% of the time, where AFF loses.
- **Random-share control** (reads condition identity, so a control, never a method): R1's gates times a Bernoulli mask
  with AFF's per-condition open share (80.6% of condition a, 29.7% of b at τ_0) reached +0.633 and +0.562; AFF minus
  control −0.042 [−0.138, +0.048] and +0.029 [−0.076, +0.132]. The rule's §9 row "the control matches or beats AFF"
  applied: reported only. The gain is one-sided steering; the reader's choice of episodes within a condition added
  nothing measurable.
- **Seeds:** 42 is the development seed and has been read very many times (including about 50 brainstorm variants and
  ideas 2 and 4 on R1); 49, 50, 51 are spent (round 3); **52 and later are free**; 9001 to 9003 are smoke-only. Ledger:
  `docs/superpowers/episode_seed_ledger.md`.

## 4. The ideas on the table (for the spec; none decided)

| Idea | What it changes | Exploratory evidence (seed 42, on R1, decides nothing) | Notes |
|---|---|---|---|
| 2. CSD as detector evidence | the gate decides whether to steer from a detector that also sees the CSD agreements (bank-trained "supports share affect" classifier, or the A1 reader's P(affect)); CSD never enters a score | emotion-condition AUC 0.822 to 0.833 with CSD against 0.785 to 0.798 without; bar margins +0.633 to +0.722 | the comparator floor must be settled in the rule: B′(A0) = 18.437 or B′(A1) = 18.805 on seed 42 (the same fused scores give +0.26 to +0.35 against B′(A1)) |
| 3. GoEmotions placement of captions | the affect grouping's caption side placed by each caption's own GoEmotions probabilities instead of the CLIP caption head | the perfect-emotion-gate ceiling is +1.805; affect pair lift 2.71 on the groups against 1.15 through the heads; the caption head reaches 35.7% held-out accuracy | the only idea that raises the ceiling; needs GoEmotions on the selection captions (GPU minutes when the GPU is free, or tens of minutes on CPU); measure the detector AUC first; B′ and the counterpart rebuilt with the same placement; ask the user whether it counts as the parked grouping redesign |
| 4. Style × genre abstention | the gate times 1[v below a percentile], v = min(S, C) agreement of the image or CSD grouping (both sides visually coherent) | +0.566 (image, 75th percentile) on R1; s×g detectability AUC 0.61 to 0.65; a perfect abstention would add about +0.11 pooled | can cost the emotion pairs (CSD at the median: e×g +1.221 to +0.922) |

**Implications of round 3 for these ideas** (our view, for the spec):
- The random-share control matched AFF. A candidate that only sharpens the reader's choice within a condition (ideas 2,
  3) may show no gain over AFF unless it changes which side it steers or how much it pays for it. Consider putting AFF
  itself (paired) and the random-share control in the rule, at least as pre-registered secondary or descriptive
  comparators.
- Style × genre is the clearest weakness. On A0 the best these groupings allow there is no harm (brainstorm §2.4); a
  real style signal needs the grouping work (design L), which the user parked earlier.
- The brainstorm's numbers are on R1; every candidate must be re-measured on AFF.

## 5. Open points for the new spec (ask the user, one at a time; our proposals)

1. **Which ideas, and how many candidates in the round.** Proposal: one round with at most two or three candidates built
   on AFF (for example idea 4 alone, idea 2 alone, both combined), each with its matched counterpart, carried by a
   pre-registered development rule; idea 3 as a separate measured step (detector AUC first) because it needs GoEmotions
   inference.
2. **Development data.** Seed 42 has been read heavily, including these ideas. Options: (a) develop on seed 42 with
   round 2's development bar and test the carried candidate on fresh seeds 52 to 54 (the user's light seed handling:
   develop on one seed, test on several fresh); (b) a fresh development seed (52) and test seeds 53 to 55.
3. **The bar and the comparators.** Proposal: GO = round 3's seven checks for the new candidate plus a paired check
   against AFF (improvement over the current best); the random-share control descriptive or secondary. The user's call.
4. **The comparator floor when CSD enters the gate** (idea 2): B′(A0) or B′(A1).
5. **Idea 3's scope and GPU use:** whether GoEmotions placement counts as the parked grouping redesign; the local GPU
   is shared (check `nvidia-smi` and use `flock -n -o -E 75 /tmp/gpu0.lock`).

## 6. Code and pitfalls

| What | Where |
|---|---|
| Seed-parameterised pipeline (bundle reproduces round 1's exactly on seed 42; any seed through `EvalContext`) | `src/test/20261121_round3_affect_gate/r3_bundle.py` |
| Readers, R1 / AFF / random-share gates, the 224-cell families, frozen-cell scoring, open counts | `r3_fusion.py` |
| Pooled checks, sensitivity projection, pooled bar margin | `r3_stats.py` |
| Runners: seed-42 regression and sensitivity, seed build with hash checks, GO and descriptive passes, rule application, wiring smoke test | `run_r3_seed42.py`, `run_r3_build.py`, `run_r3_test.py`, `r3_apply_rule.py`, `test_r3_wiring.py` |
| Shared constants (rule SHA, inputs, frozen τ, regression targets) | `r3_common.py`: the next round needs its own copy with its own rule SHA |
| Independent re-derivation (phases 1 and 2) | `src/test/20261121_round3_affect_gate/rederive/rd3_*.py` |

- **Before reusing round 3's code** (final review N11): add a GO-phase assertion that `cl == groups[anchor]` and that
  AFF's family sees AFF's τ_0 open counts; a test of §5's order (no candidate number before the regression checks); a
  criterion test for ρ_ctrl. Deferred minors from the SDD ledger (`.superpowers/sdd/2026-10-06-round3-affect-gate/progress.md`):
  bind `--boundary-reported` to the current `go_pooled.json`; the wiring test's leak check misses one-decimal numbers.
- No module may share a name with a round-1, round-2 or round-3 module (`common`, `rc_core`, `rb_*`, `r2_*`, `r3_*`,
  `run_*`, `test_*` of those rounds): their folders end up on `sys.path`.
- z-score per ranking row first, then gate; the matched counterpart of anything gated is G_cf under the candidate's
  own gates, never ḡ·z(T_cf); cross-fit ties compare the integer sums ρ and γ.
- Smoke and dry runs never print or log a metric value; read their logs with grep for markers only. Round 1's
  `load_bundle` prints seed-42 values: wrap it (`r3_bundle.compare_with_round1` does).
- `run_baselines.py` rewrites `codes_provenance.json` on every build: build fresh seeds in one invocation of
  `run_r3_build.py` (or its successor); never open the build logs (they hold scorer tables).
- Re-derivation agreement: τ within 1e-15 absolute or 1e-9 relative; points and bounds within 1e-9 pp; a lower bound
  within 1e-12 of 0 goes to the user before the verdict.
- Times with `TZ=Europe/Amsterdam date '+%F %H:%M'`; do not estimate them.
- CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`, at most 3
  processes, `uptime` and `free -g` first, `run_in_background` for real runs.
- Commit the full report only after the whole-branch final review and its fix wave (round 3 committed it before; final
  review N6).

## 7. State at handoff

- Round 3's commits on `main`: 728f5d7 (spec), fab5ae1 (rule, SHA 2d311dbe…5925), 83b51f4, a4b1af9, f1d6d25, 137d46f,
  05fca40, 7ece674, d38f1f9 (code and re-derivation phase 1), 2f8c784 (seeds built), e48cd9c (verdict GO), 66d68a0,
  02215b8 (report), dc52401 (final review), 6bcab85 (fix wave), c2cc501 (closed), 282808a and 9ef36b4 (user-read
  briefing). Nothing pushed (82 commits ahead of `origin/main` at 00:02).
- No process of this line is running. Storage left behind, all gitignored and under 1 GB:
  `src/test/20261121_round3_affect_gate/rederive/out/` 255 MB, `final_review/out/` 61 MB, `results/` 54 MB.
- Suggested names for the next round: folder `src/test/20261122_<topic>/` (folder dates are sequence numbers, next
  free after 20261121), report `docs/reports/auto/v2/2026-11-22_<topic>.md`, spec
  `docs/superpowers/specs/2026-10-07-<topic>-design.md`.

## 8. Container restart

The user restarts the container before opening the new tab. `/project`, `/data` and the container's writable layer
(`/root`, `/tmp`) survive a restart (not a recreate). Herdr's workspace, tab and pane IDs may change: the launcher
`docs/superpowers/handoffs/2026-10-07-method-improvements-launch.sh` uses the caller's own workspace. Run it from a pane
in the CoSiR workspace. It opens a tab labelled `method-improvements`, starts Claude there as agent `improve`, and sends
the start prompt `docs/superpowers/handoffs/2026-10-07-method-improvements-start-prompt.md`. The round-3 session can be
resumed with `claude --resume c7a6becd-74e9-4bbe-85de-5d7c205c205f` if needed.
