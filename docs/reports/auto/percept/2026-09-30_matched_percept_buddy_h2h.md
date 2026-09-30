# Matched PercepT-vs-buddy head-to-head

**Status: IN PROGRESS (interim, 2026-09-30 morning).** The four
equal-budget sweeps are running on DAS6. §5 holds an interim leaderboard;
§6 (the result) will be filled in when selection and the test runs finish.

Design: [`2026-09-30-matched-percept-buddy-h2h-design.md`](../../../superpowers/specs/2026-09-30-matched-percept-buddy-h2h-design.md).
Plan: [`2026-09-30-matched-percept-buddy-h2h.md`](../../../superpowers/plans/2026-09-30-matched-percept-buddy-h2h.md).
Master report: [§6i–§6j](../../2026-09-26_artelingo_buddy_vs_percept_stage1_report.md).

## 1. Question

Earlier numbers could not be compared with one another:
- **§6g:** PercepT scored 0.9226 and buddy 0.8534.
- **§6i:** the buddy sweep reached 0.9355, but in a harness whose Stage 1
  differed from the pilots, with different topic counts and held-out
  labels, and after buddy-only tuning.

This experiment runs both systems through **one validated harness**, at
**matched topic counts** (K≈16 and K≈40). Both use the **same held-out
labels and evaluation**, and both get the **same tuning budget**. The
question is which system gives better Stage 2 macro AUC, and how the two
compare on Stage 1 AMI when it is measured both ways.

## 2. Protocol (user-approved)

- **Objective:** Stage 2 macro AUC on the held-out val half, averaged over
  2 seeds per trial (1001, 1002). There is no Stage 1 gate.
- **Reported alongside:** Stage 1 AMIs on two yardsticks, for every trial:
  - *transfer:* a k-NN vote into the system's own train topics;
  - *independent:* the pilots' own re-clustering, the method behind the
    §2/§3 Pareto bar.
- **Held-out labels:** a k-NN vote with k = 20, **fixed and never tuned**.
  For PercepT, its native DEC assignment is also scored, as a secondary AUC.
- **Split:** the 9,365 held-out paintings are split 50/50 into val (4,686)
  and test (4,679), stratified by majority emotion and by whether genre is
  known (digest `5f9e2827…`). All tuning and selection use val only.
  Test is used once, for the final runs.
- **Cells:** {buddy, PercepT} × {K=16, K=40}. PercepT sets K directly.
  Buddy reaches K±2 by bisection over Leiden resolution after merging
  small topics.
- **Budget:** one W&B Bayesian sweep per cell (`polysemic/CoSiR-h2h`), 300
  trials each, with the same Stage 2 search space in all four. Each system
  searches its own Stage 1 space. Buddy's includes both the pilot-faithful
  Stage 1 and the §6i harness Stage 1.
- **Selection:** for each cell, the top 5 by val objective are re-run at 4
  stress seeds (42, 7, 123, 2024) on val, and the winner is the one with
  the highest mean. The winner is then run at 5 fresh test seeds
  (11, 23, 57, 101, 211). Two references are also run on test, without
  tuning, to connect to the old numbers: the §6i winner `m8x7ifx4` and
  §6g's PercepT config.
- **Secondary analysis (controller ruling):**
  - the AUC-vs-emotion-AMI Pareto front for each cell;
  - an emotion-constrained selection per cell (best val AUC among trials
    above an emotion-AMI floor that both systems reach, fixed before test
    is touched), stressed and tested the same way.

  Reason: optimising Stage 2 AUC alone rewards topics that are easy to
  predict from images, whether or not they carry affect (see §5).

## 3. Validation: both systems are faithful to their pilots

Every arm runs on DAS6 GPUs. The pilot results depend on GPU type (§6j),
so nothing is compared across GPU types.

| check | what | result |
|---|---|---|
| **V2** buddy Stage 1 | port vs the unchanged snapshot pilot, seeds 42/7/123/2024, same node and slot | **bit-exact**: train and held-out embeddings identical (max abs diff 0.0); independent AMIs and community counts identical |
| **V3** PercepT Stage 1 | port at §6g's fixed-pilot settings, seed 42, scored with the pilots' own Stage 2 | K = 40; native AMI 0.1073 / 0.2770 vs 0.1094 / 0.2798; AUC 0.9258 vs 0.9226 (4-seed mean, lr 1e-2 / 400 epochs); 0.5803 vs 0.5843 untuned |
| **V1** shared Stage 2 | the harness Stage 2 on the pilots' saved topics | buddy §6f: 0.85341 vs 0.8534, bit-equal to the pilot's own training loop at seed 42. PercepT: 0.92264 vs 0.9226 at the tuned setting |

Two strict-rule differences in V1 were measured and accepted:
- **Buddy held-out labels:** 132 of 9,365 differ from the pilot's. All of
  them are ties in the k=20 vote. The pilot breaks a tie by the nearest
  neighbour; the harness picks the lowest topic id. The effect is +0.0002
  AUC, and the same rule is applied to both systems.
- **PercepT at the untuned setting:** seed 42 gives 0.5843 against the
  pilot snapshot's 0.5925. This gap is *consistent with* the pilot drawing
  its mapper's initial weights from an unseeded random stream, but that has
  not been tested. At this setting the seed-to-seed spread alone is
  0.5830–0.5906.

## 4. Confirmation checks (from the master report §6j)

Scored on the pilots' independent re-clustering, the §6i winner
`m8x7ifx4` clears the emotion bar on only 1 of 8 seeds (mean 0.1189). The
untuned pilot baseline scores higher on emotion on both yardsticks (0.1230
independent, 0.1341 gate). The winner's gains were in genre AMI and
Stage 2 AUC, not in emotion.

## 5. Sweep progress (interim, 07:28, about 52 of 300 trials per cell)

**Interim only.** This is 17% of the budget; the Bayesian search is still
exploring, and nothing here has been stress-tested or run on test. Throughput
is about 72 trials per hour on 9 GPUs, so the sweeps should end around 21:00.
Every number is **val** macro AUC, averaged over the 2 search seeds.

| cell | finished | K misses | best val AUC | top-5 mean | median | best AUC with ind. emotion AMI ≥ 0.10 | ≥ 0.11 | ≥ 0.12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| buddy K=16 | 53 | 3 | 0.9891 | 0.9852 | 0.9354 | 0.9567 (n=19) | 0.9567 (n=10) | 0.8306 (n=3) |
| PercepT K=16 | 53 | 0 | 0.9606 | 0.9529 | 0.9331 | 0.9477 (n=12) | 0.9314 (n=3) | none |
| buddy K=40 | 52 | 0 | 0.9923 | 0.9911 | 0.9610 | 0.9597 (n=21) | 0.9597 (n=17) | 0.8902 (n=7) |
| PercepT K=40 | 52 | 1 | 0.9661 | 0.9648 | 0.9508 | 0.9638 (n=17) | 0.9533 (n=4) | none |

**What the search is finding so far:**
- **Buddy's top trials all use the §6i harness Stage 1.** They sit at an
  independent emotion AMI of about 0.05 (about half the pilot's 0.12),
  genre AMI of about 0.40, and val AUC of about 0.99. These are topics that
  are easy to predict from images, with little affect structure left. This
  is the emotion/AUC trade-off from §6i, now at a larger scale.
- **PercepT's top trials** reach 0.96–0.97 at an emotion AMI of about
  0.09–0.10. So the tuned Stage 2 search (queries, epochs, targets) already
  lifts PercepT well above §6g's 0.9226 at K=40.
- **With an emotion floor, the two systems are close at 0.10.** Above 0.11
  buddy is ahead, and only buddy reaches 0.12 or more. This is the question
  the secondary, emotion-constrained selection will answer properly.

## 6. Result

_Pending: selection, stress and test runs._

## 7. Rulings and deviations

See the SDD ledger (preserved in git once the work is done) and the spec's
rulings R1–R7. The main ones:
- **R1:** each system keeps its native inputs.
- **R2:** held-out labels use k=20, fixed.
- **R3:** buddy K is controlled by bisection, ±2.
- **Budget:** 300 trials per cell, sized from measured trial times.
- **Merge options:** buddy's merge options at K=40 are {0.002, 0.005,
  0.01}, because 0.02 cannot reach 40 topics. At K=16 they are {0.005,
  0.01, 0.02}. 0.0 is excluded because it leaves isolated nodes as
  singleton topics.
- **Secondary analysis:** the emotion-constrained selection described in §2.
