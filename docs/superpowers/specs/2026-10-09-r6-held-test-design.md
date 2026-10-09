# Round 6: the ArtELingo held-split paper test of AFF: design

> Date: 2026-10-09 04:00 · Status: approved 2026-10-09 04:53 (the user, from the brief in chat); revised after the rule review from 04:33 and its re-review from 04:46 ·
> Kind: experiment
> Builds on: [round 3 report](../../reports/auto/v2/2026-11-21_round3_affect_gate.md) (AFF GO on fresh seeds 49 to 51),
> the grill's facts `src/test/20261125_artelingo_held_test/design/facts.md`, the
> [literature check](../../reports/literature/2026-11-25_held_claim_check.md) · Constitution:
> `docs/superpowers/constitution.md` (version 2)
> Decision rule: `src/test/20261125_artelingo_held_test/DECISION_RULE.md` @ <commit> (governs where it differs)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** The paper test of affect steering (AFF: round 1's learned reader, which adds its grouping score only
when it picks the affect grouping) on ArtELingo's held split: 12,281 paintings that nothing in AFF was fitted, picked
or tuned on. Round 3 passed on new episodes of the development paintings; a paper claim needs new paintings.

**Plan.**
1. *Baseline stage, development seed 42.* Build the describe-then-score comparator (an 8B vision-language model says
   what the 4 example pairs share that the 4 counter-examples lack; CRL, a published method, scores with that phrase
   on the same frozen CLIP) and tune it on seed 42. Refit the classifiers, store every scorer's seed-42 weight picks,
   and check that the code still reproduces round 3's seed-42 numbers.
2. *Claim test, held split.* One read: 4,096 episodes per aspect pair on each of seeds 52 to 54, every weight frozen
   from seed 42, the verdict by the decision rule.

The build stops before the read and waits for you if the new comparator beats AFF's R@1 on seed 42 (19.14) or cannot
be built and run there within a day; if the refit classifiers do not reproduce their stored outputs exactly; if the
code does not reproduce round 3's seed-42 numbers exactly; or if a safety check fails (a file, split or episode
check). Otherwise nothing waits for you until the reports.

**Compared against.** Pass checks (they decide GO): AFF's R@1 against cosine, RCA (the best metric learned from the
example pairs), the condition-free scorers B and B′(A0), and AFF's matched control; its condition gain against the
condition-free scorers and RCA. Secondary checks (tested only after a GO): the style-aware scorer B′(A1), the
strongest condition-free scorer so far (18.80 on seed 42 against AFF's 19.14), and the round 1 reader without the
affect gate. Reported beside: the new comparator with its true-aspect-name ceiling, the three fine-tuned CLIP
baselines, the other pair metrics and the in-context reranker (seed 52 only).

**Outcomes.** GO: all seven pass checks above 0 under the Holm correction; the paper may then say AFF beats them on
new paintings, pooled over the three aspect pairs. NO-GO: any check fails, read as "did not beat X" (point at or
below 0) or "inconclusive" (point above 0). Every outcome goes in the paper. We expect a GO, the margin over B′(A0)
shrinking to about +0.4 R@1 (round 3: +0.59), B′(A1) near a coin flip, style × genre a loss again (disclosed).

**Cost.** CPU: a few hours, three processes. DAS6: about 45 GPU-hours (the 8B model on seed 42 and the held
episodes, the reranker, held image features for two fine-tunes), 6 to 8 hours on the nine GPUs. Build and run
unattended from tonight; reports expected by Mon 2026-10-12; time-box Thu 2026-10-15, before the go/no-go on Fri 16.

**Check these.**
1. You chose to read the full held split now, after seeing our advice to split it in two; B′(A1) as a secondary check,
   its level (tested after a GO, Holm across the two), and the new comparator reported beside, with its stop, were
   also settled after seeing ours.
2. "Frozen from seed 42" means each scorer's two seed-42 cross-fit picks, each applied to the held episodes of the
   other parity. Round 3 checked this only for the cells of AFF, its matched control and the round 1 reader (+0.585
   against +0.591 cross-fitted), with the condition-free scorers re-picked; freezing their picks is new. Agent default.
3. The new comparator gets four prompt wordings, fixed in the rule, times two basis sizes, plus the weight grid. Its
   stop compares exact counts of correct rankings on seed 42 (AFF: 9,406 of 49,152), with no interval; "a day" is 24
   hours from its first commit. Agent default.
4. The extra held style (New_Realism) is excluded only as a target value; its paintings stay in the episodes in other
   roles, as its 17 development paintings did. Agent default.
5. In a NO-GO, checks that pass are reported but license no claim. Stage types (baseline, then claim test) and the
   cheap tier-1 pair metrics as reported rows. Agent defaults.

## 1. Context and goal

Asked: "Ok, I think we go with the held test now. Then in another chat we go with design L. So let's follow the loop
and start the task about held-out test on ArtElingo."

AFF passed round 3's seven checks on fresh seeds 49 to 51 (bar margin +0.591 [+0.462, +0.729] R@1 over B′(A0)), but
those seeds reuse the 6,451 selection paintings, so C14 allows no claim about new paintings. Rounds 4 and 5 did not
improve AFF. Plan revision 4 (§16) makes an ArtELingo-centred paper the first target; constitution C5 (amendment 2)
gives each final method and backbone one held read.

Goal: AFF's paper claim on new paintings, or an honest NO-GO, from one pre-registered read of the held split.
Question this step answers: does AFF, frozen exactly as in round 3, beat round 3's comparators on ArtELingo's held
paintings, pooled over the three aspect pairs?

## 2. Scope

In: the seed-42 baseline stage (the describe-then-score comparator, head refits, frozen picks, regression); the held
read of AFF on CLIP ViT-B/32 with every comparator, secondary check and reported baseline listed in §3.2; the held
ledger row; the auto report and the user-read report.

Out: design L (its own loop, next); the second backbone (its own later read, C5); CUB, SemArt and GeneCIS (plan §16,
after the go/no-go); K7's unsupervised bases and method A's three training seeds (plan §8, §10: they concern method
A, not frozen AFF); any change to AFF; the reserve read (only for a pre-registered fix after a final-review
finding, C5; rule §9, D26).

## 3. Design

### 3.1 Hypothesis and prior

AFF's gain comes from steering on the emotion side, which its affect heads read from CLIP features learned on
scorer-train rows; nothing in it depends on which paintings it is scored on. So the pooled gain should carry to new
paintings, smaller than on development. Prior (the rule states it in full): bar margin over B′(A0) about +0.4; GO
likely; AFF − B′(A1) about +0.15, near a coin flip; style × genre negative again. A NO-GO on cosine, RCA or the gain
checks would surprise us.

### 3.2 Arms and controls

| Arm | What it is | Differs from AFF by | Role |
|---|---|---|---|
| AFF | round 3's fused reader, seed-42 cells | — | candidate |
| Matched control | AFF's counterpart (round 3's D9), seed-42 cells | the condition removed | matched control, pass check |
| Cosine; RCA | backbone only; best metric learned from the 4+4 pairs (λ frozen) | no reader | comparators, pass checks |
| B; B′(A0) | the condition-free scorers of round 3 (D10, D11), seed-42 picks | no condition | comparators, pass checks |
| B′(A1) | B′ on (affect, image, caption, CSD style) | no condition, one more grouping | secondary check |
| R1 | round 1's reader without the affect gate, seed-42 cells | gate open on every pick | secondary check |
| Describe-then-score | §3.8; λ frozen from seed 42 | names the shared quality in words | reported beside |
| Its matched control; true-name ceiling | two-condition mean of its term; the true aspect name in place of the phrase | condition removed; privileged | reported only |
| Fine-tuned CLIP LP, LB, LoRA | round 2026-11-24's checkpoints, frozen | a fine-tuned backbone | reported |
| Other tier-1 pair metrics | diagonal, diagonal ReLU, bilinear, KISSME, Xing, Wang, probe, Tip-Adapter, value prototype (λ frozen) | no reader | reported |
| In-context reranker | the 8B probe's setting, frozen; seed 52 only | an MLLM reads the examples | reported |

Every arm uses the same episodes, features and frozen weight grid as in development (plan §8, equal tuning).

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---|---|---|
| Selection rows, seed 42 | development: picks, comparator tuning, regression (C12) | 1 |
| Selection rows, smoke seeds 9001 to 9003 | wiring only, never results | 1 |
| Scorer-train rows (same 60,000-row draw) | head refits | 1 |
| Held rows, seeds 52, 53, 54 (4,096 per pair each) | the claim test, one read (ledger row H5) | 2 |
| Selection seeds 43 to 51; held rows outside H5 | not touched | — |

Held rows: the 20% part of the grouped split (seed 42), rows with all three labels; the development value sets (8
emotions, 23 styles, 10 genres). New_Realism is excluded as a value only: its rows stay in the episode pool in other
roles, as its 17 selection paintings did in development (rule §5 items 2 and 3, D17). Every member of a held episode
is a held row.

### 3.4 Measurement

Metrics, episodes, directions and the cluster bootstrap as round 3's rule §2 and C9: R@1 and condition gain per
episode, pooled over pairs, conditions and directions; 5,000 resamples of anchor paintings, seed 42; the three seeds
concatenated, a painting one cluster. Pass checks: Holm across the seven at a family-wise 5% (two-sided equivalent),
decided from integer bootstrap counts (rule §3). Secondary checks: tested only after a GO, then Holm across the two
(D16). Three seeds and 4,096 per pair keep round 3's count (user); the held pool has 1.9 times the anchor paintings,
so intervals should be narrower than round 3's ±0.13.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 1a | refit check | heads refit on scorer-train | every stored selection posterior reproduced bit for bit (rule §6 item 2) | any difference: stop before the read |
| 1b | regression (C12) | the held runner in selection mode, seed 42, seed-42 picks, refit posteriors | round 3's seed-42 numbers, per-anchor arrays and episode hashes exactly; B′(A1) 18.804931640625 (rule §6 items 3, 4) | stop, trace, the user decides |
| 1c | baseline | describe-then-score on seed 42 | built within 24 hours, its true-name ceiling sane; hit count not above AFF's 9,406 of 49,152 (rule §7) | above, or not built in time: stop before the read |
| 1d | smoke, last | the same script bytes on smoke seeds, plus a wiring mutation | every assertion passes; the mutation fires; script hashes recorded (rule §6 item 7) | fix the wiring (not a stop) |
| 2 | claim test | held seeds 52 to 54, one read | the rule's verdict, written after the re-derivation agrees (rule §8) | reported either way |

Any failed safety assertion (a file hash, the split, sample IDs, value eligibility, the held mask, a condition-free
check) also stops that step before the read (rule §9). Steps 1a to 1d run in this order; a step whose code changes
after it passed is rerun before the smoke, and the comparator's stop is then evaluated again (rule §6 item 7).

### 3.6 Diagnostics

After the verdict file only (written once the re-derivation agrees), deciding nothing: per seed and per pair; the
bar margin (round 3's D12) and the margin over B′(A1); the two-way (anchor × candidate) bootstrap with the item-reuse
rate, and swap success (plan §10); gate-open shares, pick accuracy, redundancy (round 3's §7); R1's own seven checks;
the describe-then-score rows (fused, control, gain, AFF minus it, its ceiling, its phrases' most frequent wordings,
its parsing failures); the reported baselines.

### 3.7 Frozen vs rerun

Frozen from development: the readers, τ, the affect restriction, the groupings, the A3 checkpoint, every seed-42
pick (AFF, counterpart, R1: round 3's chosen cells; B, B′(A0), B′(A1): recomputed on seed 42 and stored before the
read; RCA and the other pair metrics: their seed-42 λ; describe-then-score: its setting and λ).
Refit on the same scorer-train rows (checked against their stored selection outputs): the heads, and RCA's PCA basis
and pair scaler. Recomputed on held: the heads' posteriors, the reader's features, probabilities, gates and every
score. No pick is made on held data (rule §11).

### 3.8 The describe-then-score comparator

Built from the literature check's §1; rule §7 fixes every setting. Qwen3-VL-8B-Instruct sees an episode's 4 support
pairs and 4 contrast pairs (images and captions) and answers, greedily and in a short phrase, what the supports share
that the contrasts lack (VisDiff-style); condition b swaps the sets. CRL (NeurIPS 2025) then has the same model list
the phrase's values, encodes them with CLIP B/32's text encoder into a basis, maps query and candidates to their
cosines with it and scores the cosine of those vectors. The score is fused with cosine on the external baselines' λ
grid; its matched control averages the two conditions' terms, as round 3's counterpart does.

On seed 42 the true-name ceiling (phrase = the true aspect's name) runs first: its condition gain must be above 0, or
the pipeline is debugged. Then 4 prompt wordings (written in the rule) × 2 basis sizes (8, 16) are compared on 1,024
episodes per pair by the baselines' criterion (mean of R@1 and gain). The chosen setting runs on all 12,288 seed-42
episodes, which give its λ picks and the stop's count. Unparseable answers fall back to cosine and are counted.

## 4. Decisions

All grill rows are from the grill of 2026-10-09 (02:50 to 03:57, Amsterdam); the handoff §2 lists them. Agent
defaults set while drafting carry the draft's start, 04:00; those changed after the rule review carry 04:37, or 04:47
after its re-review.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|---|---|---|---|---|---|
| D1 | Paper target | ArtELingo-centred paper first; go/no-go Fri 2026-10-16 | time; full paper if results allow | user | 2026-10-09 02:43 |
| D2 | What the held read covers | AFF now, on the full held split; later methods get their own reads (C5 v2) | ours: split the paintings in two; the user: reuse is common if nothing is tuned on it | user, after seeing the recommendation | 2026-10-09 03:48 |
| D3 | What is tested | AFF frozen exactly as in round 3 | the current best (direction log) | user | 2026-10-09 02:48 |
| D4 | Weight picks on held | frozen from seed 42 (plan §8) | re-picking on held would tune on it | user (plan §8, approved 2026-10-02) | 2026-10-02 |
| D5 | How seed-42 picks are applied | the pick of seed-42 tune half h scores held episodes of parity 1 − h | a new single pick on all of seed 42 was never tested; round 3 tested this scheme for the cells of AFF, its counterpart and R1 (+0.585); B's and B′'s picks were re-picked there, so freezing them is new | agent default | 2026-10-09 04:00 |
| D6 | Pass checks | round 3's seven; they contain K2's six | K2's six alone would drop B and B′(A0) (C1) | agent default (confirmed in the grill summary) | 2026-10-09 03:57 |
| D7 | Bar | each pass check above 0 under Holm across the seven; no minimum size; GO needs all seven | 95% each (ours the same as the user's) | user | 2026-10-09 03:54 |
| D8 | Episodes and seeds | 4,096 per pair per seed; seeds 52, 53, 54 | more episodes add little (user: "do not overcomplicate") | user | 2026-10-09 03:54 |
| D9 | B′(A1) | pre-registered secondary check, never decides GO; if it passes the paper may say "also beats it" | the user chose "reported beside", then agreed to this form | user, after seeing the recommendation | 2026-10-09 03:43 |
| D10 | Describe-then-score comparator | built now (§3.8); reported beside, with a true-name ceiling; not a pass check | a pass check could fail AFF on a comparator never tested | user, after seeing the recommendation | 2026-10-09 03:37 |
| D11 | Stop rule | if the comparator's seed-42 R@1 is above AFF's, the build stops before the read | reading anyway would spend the read on a plan already in doubt | user, after seeing the recommendation | 2026-10-09 03:43 |
| D12 | How the stop compares | exact hit counts on all seed-42 episodes, pooled (stop if the comparator's exceeds AFF's 9,406 of 49,152); not built within 24 hours of its first commit counts as a stop (rule §7) | a bound-based test would let a comparator that is ahead through | agent default | 2026-10-09 04:37 |
| D13 | Claim | pooled, "on new paintings"; per pair reported; the style × genre loss disclosed with design L as the route | a per-pair claim would fail on style × genre | user | 2026-10-09 03:54 |
| D14 | R1 | secondary check, as in round 3 | descriptive only would drop a pre-registered line | agent default | 2026-10-09 03:57 |
| D15 | Fine-tuned CLIP (LP, LB, LoRA) | reported; LB and LoRA held features on DAS6 | a pass check on a 15.0 baseline adds nothing | agent default | 2026-10-09 03:57 |
| D16 | Level of the secondary checks | tested only after a GO, then Holm across the two at a family-wise 5%; reported untested after a NO-GO | stricter than the grill's literal answer ("lower bound above 0", option a); (a) each at 95% alone (round 3's form) leaves two claims uncorrected; (c) Holm whatever the verdict leaves the nine claims uncontrolled together | user, after seeing the recommendation | 2026-10-09 04:20 |
| D17 | Style set | the 23 development styles; New_Realism excluded as a value only, its rows kept in the pool (rule §5 items 2, 3) | the protocol matches development, where its 17 paintings were in the pool but not eligible; removing its rows would also change the emotion × genre episodes | agent default | 2026-10-09 04:37 |
| D18 | Comparator tuning budget | 8 settings fixed in the rule (4 wordings × basis sizes 8 and 16) on 1,024 episodes per pair, then the baselines' λ grid on all of seed 42 | the external baselines got a λ grid only; the extra settings cover the new parts; fixing them now leaves no open build choice | agent default | 2026-10-09 04:37 |
| D19 | B′(A1)'s CSD heads | refit on scorer-train; round 4 or 5 code reused only as an `r6_` copy, with the "before reuse" items of round 4's final review §5 and N13 and round 5's final review §4 fixed in it (rule §6 item 1) | stored posteriors cover selection rows only; round 4's stored-posterior loader cannot run on held rows | agent default | 2026-10-09 04:37 |
| D20 | Refit check | every refit posterior on selection rows equals the stored one bit for bit (float32 as stored); any difference stops before the read (rule §6 item 2) | the user's stop as confirmed in the grill; a tolerance would loosen it | agent default | 2026-10-09 04:37 |
| D21 | Safeguards | ledger row H5 (committed before launch) and a refuse-twice runner; the smoke last, its script hashes kept, and any step whose code changed rerun before it; C12; the verdict file written only after the C13 re-derivation agrees, with at most one corrected held pass before it; GPU jobs on held rows get no labels or targets; final review (rule §6, §8) | H3's pattern | agent default | 2026-10-09 04:47 |
| D22 | Descriptive extras | two-way bootstrap, swap success (plan §10); the in-context reranker on seed 52 only | | agent default | 2026-10-09 03:57 |
| D23 | Extra reported rows | the other tier-1 pair metrics of plan §8 | cheap now; scoring them on held later would be another look | agent default | 2026-10-09 04:00 |
| D24 | Stage types | stage 1 baseline, stage 2 claim test | the first question's yes decides each (module) | agent default | 2026-10-09 04:00 |
| D25 | Compute | DAS6 (node401, node402, node408) for GPU work; local CPU for scoring | | agent default | 2026-10-09 03:57 |
| D26 | A bug found after the verdict | C5's reserve read: the same held episodes, frozen picks and rule, corrected code only, ledger row H5-R, its own single-use guard, originals kept and reported; only after the user has seen the cause (rule §9) | C5 allows one reserve read for a pre-registered fix after a final-review finding, never for a better number | agent default | 2026-10-09 04:47 |
| D27 | What a NO-GO licenses | checks that pass in a NO-GO are reported as passed and license no claim | the bar (D7) makes a GO the only route to a claim; Holm alone would allow stating them | agent default | 2026-10-09 04:37 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: the refit heads reproduce the stored posteriors bit for bit (same code, draw and library versions).
  Checked: not yet; stage 1a stops the build if they do not.
- Assumption: Qwen3-VL-8B-Instruct and CRL's code run on DAS6. Checked: the 8B model ran in the 2026-11-06 probe; CRL
  not yet.
- Assumption: every development value is eligible on held rows (≥ 30 held paintings). Checked: facts §B (labels only).
- Open questions: none (Q1, the secondary checks' level, answered by the user: D16).

## 6. Success criteria

**Decision mode.** Binding rule: `src/test/20261125_artelingo_held_test/DECISION_RULE.md` @ <commit>, committed
before any code; it governs where it differs from this spec.

- SC1 (stage 1b): round 3's seed-42 numbers, per-anchor arrays and episode hashes reproduced exactly (rule §6
  item 4).
- SC2 (stage 1a): every refit posterior reproduces the stored one bit for bit (rule §6 item 2).
- SC3 (stage 1c): the describe-then-score comparator established on seed 42 (built, ceiling sane, λ frozen), with its
  hit count at or below AFF's 9,406 of 49,152 rankings (R@1 19.136555989583336; rule §7).
- SC4 (stage 2): the seven pass checks, Holm across the seven (rule §3, §4).
- SC5 (stage 2): the secondary checks, B′(A1) and R1, tested only after a GO (rule §4).

- GO: SC1 to SC3 pass and all seven of SC4 pass.
- NO-GO, did not beat: a failed SC4 check with its point at or below 0.
- Inconclusive: every failed SC4 check has its point above 0; read at its detectable margin (C11). The read is spent
  either way; the user decides what follows. Checks that pass in a NO-GO are reported as passed and license no claim
  (D27).

**Licensed claim** (GO): "On paintings never used to fit, select or tune it, AFF beats cosine, RCA, the condition-free
scorers B and B′(A0) and its matched control on aspect R@1, and its condition gain exceeds theirs and RCA's, pooled
over three aspect pairs, with a Holm correction across the seven checks." Not licensed: a margin on each aspect pair,
other datasets or backbones, beating B′(A1) or R1 unless their secondary checks pass, anything about why AFF works.

## 7. Constitution check

| Principle | Pass or exception | Note |
|---|---|---|
| C1 strongest comparator | pass | B′(A1), the strongest condition-free score so far, stands beside every headline number |
| C2 matched control | pass | AFF's counterpart is a pass check; the new comparator has its own control |
| C3 beat the current best | n/a | AFF is the current best |
| C4 seeds | pass | development on 42; held seeds 52 to 54 hash-checked, in the held ledger |
| C5 held data | pass | one read of AFF on B/32, ledger row H5, refuse-twice runner, nothing picked on held; the reserve read only for a fix after a final-review finding (D26) |
| C6 sample IDs | pass | held rows by the extraction IDs; asserted |
| C7 rule before code | pass | the rule is reviewed and committed before any code |
| C8 prior and disclosure | pass | rule §1; §8 below |
| C9 statistics | exception | Holm across the seven replaces "each 95% lower bound above 0" (stricter) |
| C10 development bar | n/a | a claim test; no minimum size (D7) |
| C11 reading a failure | pass | rule §4 |
| C12 regression | pass | stage 1b |
| C13 re-derivation | pass | rule §8 |
| C14 claim scope | pass | "new paintings" comes from the held read, as C14 requires |

Exceptions: C9: Holm across the seven checks · the plan's §10 asks for a multiplicity correction on the paper's
primary comparisons, and the user chose it (D7) · each check at 95% alone was rejected as uncorrected across seven.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| A held-path bug that selection runs cannot show (row masks, eligibility, IDs) | unit tests on the real data shape; held-only assertions; C13 re-derivation; final review | before the verdict file: correct the code to match the rule and run one corrected held pass on the same episodes, after a new smoke, then repeat the re-derivation (rule §8.1, §9); after it: C5's reserve read, once (D26) |
| Refit heads differ from the stored ones, even in the last bit | stage 1a | stop before the read |
| Near-duplicate images: 24 held rows have a train image at CLIP cosine ≥ 0.99 (63 at ≥ 0.98) | known (facts §B) | disclosed |
| Held rows shaped earlier designs (H1 to H3, value episodes) | known | disclosed (plan §10) |
| The new comparator is built weakly, so "AFF beats it" means little | its true-name ceiling; the literature check's expectations | reported with the ceiling; never a pass check |
| DAS6 nodes fail or the reservation ends | cluster status | the CPU verdict does not need the GPU; GPU rows follow or are reported missing |

**Selection disclosure.** AFF was found among about 50 label-free variants on seed 42 (round 3 §6.11), and rounds 4
and 5 read about 10 more there; seed 42 is development data for every pick here. The new comparator is tuned with at
most 8 settings on seed 42. The held paintings were read three times with value episodes (H1 to H3) for earlier
methods. Design L will be designed after this read. Per-pair results are reported beside the pooled claim, with the
style × genre margin disclosed (round 3: −0.580 [−0.793, −0.363] over B′(A0)) and design L named as the route to it
(D13). All of this goes in the report and the paper; the held paintings protect this test because no pick sees them.

## 9. Deliverables and review

- At the end: code and results in `src/test/20261125_artelingo_held_test/` (modules `r6_*`); ledger row H5; the auto
  report `docs/reports/auto/v2/2026-11-25_artelingo_held_test.md` with its `reports_sum.md` row; the user-read report
  in `docs/user_read/`, dated the day it is written; `GLOSSARY.md` updated.
- Reviews: the rule checked by a fresh Opus reviewer before its commit (C7); ticket reviews for verdict code
  (metrics, splits, statistics, joins, the rule's logic); the C13 re-derivation; the final whole-branch review, one
  fix wave and a scoped re-review.
- Stop points: none between approval and the reports except the pre-read stops of §3.5 (steps 1a to 1c, and any
  failed safety assertion). · Time-box: if the read has not started by Thu 2026-10-15, no read starts (rule §9).
- Budget: CPU scoring at 8 threads, at most three processes; DAS6 about 45 GPU-hours (8B verbaliser on about 25,000
  seed-42 tuning and 25,000 seed-42 calls and 74,000 held calls; the reranker on 12,288 held episodes; LB and LoRA
  held image features); the local GPU only for small steps under its lock. Not counted: CRL's text-only value
  listing, one call per distinct phrase and basis size; if most phrases are distinct, it roughly doubles the number
  of held calls (about 74,000 more, each text only).
- Length: the body runs about 245 lines against a target of about 180 (claim test plus a baseline stage), because §4
  carries the grill's 15 decisions besides the agent defaults, and the rule review's fixes added stops and checks.
