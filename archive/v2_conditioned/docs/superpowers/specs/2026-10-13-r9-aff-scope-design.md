# Round 9, AFF's scope: two more backbones on ArtELingo, a language check on ArtELingo-28, and probes of CUB and SemArt: design

> Date: 2026-10-13 15:05 · Status: approved 2026-10-13 15:05 · Kind: experiment
> Builds on: [round 6 held test](../../reports/auto/v2/2026-11-25_artelingo_held_test.md) (AFF GO on CLIP), [round 8 report](../../reports/auto/v2/2026-11-29_learned_grouping.md) (no pick, AFF stays best), [direction log 2026-10-13](../direction_log.md), [literature note](../../../../../docs/reports/literature/2026-12-01_affect_caption_datasets.md) · Constitution: `docs/superpowers/constitution.md`
> Decision rule: `src/test/20261202_r9_scope/DECISION_RULE.md` @ a443902 (written first in the build, reviewed, committed before any scoring code)
> Length: body about 255 lines against a target of about 240; the grill settled 34 decisions, all kept in §4.

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** AFF passed its held test on one backbone (CLIP) and one dataset, and reviewers will say it depends
on both. This loop tests AFF, frozen as a recipe, on two more backbones (Qwen3-VL-Embedding-2B and SigLIP 2) on
ArtELingo, then on captions written in 27 other languages (ArtELingo-28), and probes CUB and SemArt for the next loop.

**Plan.**
1. Regression and recipe replay on CLIP: the reworked code must give back CLIP's numbers and picks exactly.
2. Preparation in parallel on the cluster: SigLIP 2 features, the English translation of ArtELingo-28, and its features
   for each backbone.
3. A claim test per new backbone on ArtELingo's held paintings. AFF is re-fitted by its fixed recipe: every part above
   the frozen encoder is fitted again, and its settings are picked again by the same automatic rule on seed 42. If
   that pick turns AFF's affect term off, the backbone's reads are not run.
4. A claim test per backbone, CLIP included, on ArtELingo-28, after step 3 and whatever its verdict.
5. An exploration: CLIP AFF on the development splits of CUB and SemArt, to prepare the next loop.

**Compared against.** On each backbone: cosine, RCA (the best pair-metric baseline), B and B′(A0) (the best scorers
without the condition), and AFF's matched control (the same score with the condition removed). On CLIP's held read
AFF beat all five, by +0.60 R@1 over B′(A0).

**Outcomes.**
- ArtELingo, per backbone: GO if all seven checks pass, the bar CLIP passed; otherwise NO-GO, and each failed check is
  read on its own.
- ArtELingo-28, per backbone, in two levels: the small claim if AFF beats B′(A0) on R@1 and its control on condition
  gain; the full claim if the other five checks then pass too.
- Every read is reported in one main table, pass or fail.
- We expect positive margins on both new backbones, likely below CLIP's. ArtELingo-28 may fail from its size alone
  (746 paintings; intervals up to about four times wider than CLIP's).

**Cost.** About 3 to 5 GPU-hours on the cluster (6 GPUs on node406 and node408) and small local GPU jobs; about 1 to
1.5 h of CPU per backbone re-fit. Build 10-14 to 10-15, run 10-15 to 10-16: you hear back about 2026-10-16. It is a day
run, so the build and run chats may ask you.

**Check these.**
1. The two-level test on ArtELingo-28 was your idea; its form was my recommendation, which you took.
2. Stage types (my default): claim tests for steps 3 and 4, an exploration for step 5, one full loop.
3. Feature extraction and translation start before the decision rule is committed: they read no label and score
   nothing (my default).
4. The translator (NLLB-200) may handle some of the 27 languages poorly. Per-language numbers are reported, and the
   claim says "machine-translated".
5. The describe-then-score, fine-tuned CLIP and reranker comparators stay CLIP-only (after my recommendation).

## 1. Context and goal

Asked: "AFF \"depends too much on the selected encoder and selected dataset. Reviewers will easily attack this
point.\"" (decision A, 2026-10-13); then "I think we go first with backbones on Artelingo, then extend all backbones to
artelingo-28, then go for CUB, SemArt and Affection" (grill, Q1).

AFF (glossary) is the current best: its held read H5 on CLIP passed all seven checks (+0.60 [+0.46, +0.73] R@1 over
B′(A0)). Round 8 closed the grouping line. Decision A keeps AFF frozen and widens its scope for the CVPR paper
(abstract 2026-11-10, paper 2026-11-16). This spec covers the first two of three stages; the third (CUB, SemArt,
Affection reads) gets its own decide chat.

Goal: test whether AFF's held result holds on two more backbones and on captions from other languages.
Question this step answers: does AFF, re-fitted by its fixed recipe, pass round 6's bar on Qwen3-VL-Embedding-2B and
SigLIP 2, and does its gain hold on ArtELingo-28's translated captions on all three backbones?

## 2. Scope

In: the AFF chain made generic over backbone and dataset; the fixed recipe on Qwen3-VL-Embedding-2B and SigLIP 2
so400m/14-384; held reads H6 and H7 on ArtELingo; the ArtELingo-28 language check on CLIP, Qwen and SigLIP 2 (reads H8
to H10); development probes of CLIP AFF on CUB and SemArt; SigLIP 2 features of SemArt (for stage 3).

Out: the CUB, SemArt and Affection reads (stage 3, own decide chat about 2026-10-17; Affection dropped if no data by
2026-10-24, then future work or the appendix); native-language ArtELingo-28 text (not chosen at Q5); the
describe-then-score, fine-tuned CLIP and reranker comparators on the new backbones; PE-Core; the DINOv2 reader and FCMI
(parked); any change to AFF's method; the paper itself (its own `claude-paper` chat, opened after this spec).

## 3. Design

### 3.1 Hypothesis and prior

AFF steers the score towards the affect grouping when the reader says the examples share affect. Nothing in that
mechanism is tied to CLIP: the affect grouping comes from GoEmotions on English captions, and the heads, reader and
fusion are re-fitted on any backbone. So the claim is that AFF's gain over the strongest condition-free scorer holds on
other frozen encoders and on captions written by other annotator pools.

Prior (stated before any number; the rule restates it, C8): a positive margin over B′(A0) on both new backbones,
likely +0.2 to +0.6 R@1, since a stronger encoder lifts B as well. A GO on each new backbone at about even odds. On
ArtELingo-28 the small claim at about even odds and the full claim less likely, from power. A surprise: AFF below B′(A0)
on a backbone where the gate opens as often as on CLIP, or the recipe switching AFF off.

### 3.2 Arms and controls

Every arm is built per backbone from that backbone's features by the fixed recipe (D3); the condition-free arms get the
same picks from the same seed-42 rules as on CLIP.

| Arm | What it is | Role |
|---|---|---|
| AFF | R1 with its gated term only on an affect pick, fused on z(B), cells by the min-margin cross-fit pick (R3 rule D5 to D8) | candidate |
| CF | AFF's matched control: the same terms and gates, the condition removed (R3 rule D9) | matched control (C2) |
| COS | cosine on the backbone's features | comparator |
| RCA | relevant component analysis fused with cosine, λ by `crossfit_lambda` | comparator, best pair metric (C1) |
| B, B′(A0) | the condition-free scorers (R3 rule D10, D11) | comparators, strongest condition-free |
| B′(A1), R1, PM | B′ with the CSD grouping; R1; the other pair metrics of `run_baselines.py` | reported, no check |
| DTS, fine-tuned CLIP, reranker | round 6's external comparators | CLIP only (H5 numbers), not run here (D27) |

### 3.3 Data roles

| Data or seeds | Role | Used in stage |
|---|---|---|
| ArtELingo scorer-train rows | fitting every part of the recipe | 2 |
| ArtELingo selection rows, seed 42 | development: each backbone's picks | 0, 2 |
| ArtELingo held rows (12,281 paintings), seeds 52 to 54 | held: one read per new backbone (H6 Qwen, H7 SigLIP 2) | 3 |
| ArtELingo-28 read set: 746 paintings (248 of our held + 498 new), non-English rows translated to English, seeds 55 to 57 | held: one read per backbone (H8 CLIP, H9 Qwen, H10 SigLIP 2) | 4 |
| ArtELingo-28's 912 train-side paintings and its English rows | not touched | – |
| CUB: 120 training species fit, 30 development species score, seed 42 | development | 5 |
| SemArt: official train fit, official val score, seed 42 | development | 5 |
| CUB test (50 unseen species), SemArt test, Affection | not touched | – |
| Smoke seeds 9001 to 9003 | wiring only | 0 to 4 |

The split is CLIP's for every backbone: its leakage groups are passed in and the sizes asserted (D15).

### 3.4 Measurement

Standard protocol (C9; round 6 rule §3) with these settings. Episodes: 4,096 per aspect pair per seed, three pairs
(emotion × style, emotion × genre, style × genre), values with at least 30 paintings, pooled over the three seeds;
clusters are anchor paintings; one bootstrap of 5,000 resamples, seed 42, the same indices for every check.

- **ArtELingo reads (H6, H7):** P1 to P7 as in round 6 (R@1 against COS, RCA, B, B′(A0), CF; the gain statistic
  against CF; condition gain against RCA), Holm across the seven at one-sided α = 0.025, one family per backbone
  (round 6's integer form). The same episodes as H5, so backbones compare pair by pair (D16).
- **ArtELingo-28 reads (H8 to H10), two levels:** level 1 is P4 and P6, Holm over two at α = 0.025; level 2, only
  if both level-1 checks pass, is P1, P2, P3, P5 and P7, Holm over five at α = 0.025. Serial gatekeeping keeps the
  family-wise error at α. One family per backbone.
- **Reported, no check:** per pair and per direction; per language (27); the 498-painting new-only subset; gate share;
  the reader's affect-pick rate; B′(A1), R1 and PM; AFF on one backbone against AFF on another (same episodes).
- **Detectable margin** per check and read, as in round 6 §8.2, for C11's wording.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0a | regression | the generic chain on CLIP | round 6's seed-42 regression (131 items) reproduced exactly (C12) | stop, trace, ask the user (day mode) |
| 0b | recipe replay | the fixed recipe on CLIP from scratch | every CPU-made part bit-equal to CLIP's pinned files; A3 and what follows within the rule's tolerance, traced to GPU non-determinism; the seed-42 picks equal CLIP's (cells 39 and 119, τ) | stop, trace; the recipe is not used until it replays |
| 1 | preparation | SigLIP 2 on all of ArtELingo; NLLB translation; ArtELingo-28 and SemArt features | counts, finite values, sample-ID joins (C6), a coverage check per language | fix and rerun; a language the translator lacks is flagged and reported |
| 2 | recipe per backbone | Qwen and SigLIP 2 features | files pinned by SHA-256; the switched-off check: λ_a = 0 in both halves' cells stops that backbone's reads (D4) | "switched off" is a result: reported, no read |
| 3 | claim test | H6, H7 | §6 | NO-GO is reported (C11) |
| 4 | claim test | H8 to H10 | §6 | as above |
| 5 | exploration | CUB, SemArt development | none: a reading | – |

Order: 0a and 0b before anything that uses the chain; 1 starts at once; 2 after 0b; 3 after 2; 4 after 3 has run
for every backbone (D9); 5 in parallel with 2 to 4. Before each read: round 6's pre-read checks (rule §6: smoke chain,
pins, the ledger row committed and pushed first, a runner that refuses a second run). Reads run locally and load the
pinned files; re-fits may run on the cluster (D12).

### 3.6 Diagnostics

After the verdicts, decides nothing: the gate share and the affect-pick rate per backbone (does the reader behave as on
CLIP: 54.5% gate share at τ_0 on held); per-pair margins (does style × genre lose again); per-language margins on
ArtELingo-28 (are low-resource languages the weak ones); the 498-painting subset against the 248 held ones (does the
earlier reading matter). Probes (stage 5): per dataset the usable aspect values and pairs, episode counts, COS, RCA, B
and AFF on seed 42, the gate share; they feed stage 3's decide chat.

### 3.7 Frozen vs rerun

- **Frozen, carried exactly:** AFF's method and every setting of its recipe (R3 rule D1 to D15): the 224-cell grid
  (τ at percentiles 0, 25, 50, 75 of the reader's seed-42 margins; λ_u in 0 to 16; λ_a in 0 to 16), the pick rules,
  head settings, k-means k = 64, reader features, A3's chosen settings (D18); the affect partition and the affect
  k-means grouping (built from GoEmotions on English captions, so the same for every backbone); the split; the
  episodes; CLIP's whole AFF (for H8).
- **Re-fitted per backbone by the recipe:** the image and caption groupings, all heads, A3, the practice banks and the
  reader, τ, every seed-42 pick (AFF, CF, B, B′, RCA, PM).
- **Nothing is picked on held rows or on ArtELingo-28.**

## 4. Decisions

Decided in the grill on 2026-10-13 between 13:05 and 15:00 unless marked.

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|---|---|---|---|---|---|
| D1 | Order of the work | backbones on ArtELingo first, then ArtELingo-28 on all backbones, then CUB, SemArt and Affection | confirm on the known dataset, then extend (one loop for all; two loops) | user | 13:20 |
| D2 | How the work runs | preparation in parallel: cluster jobs and dataset probes inside one build, reads in order | the user's idea; probes in a separate chat would edit the same chain at once | user, after seeing the recommendation | 14:05 |
| D3 | "Frozen" on a new backbone | the fixed recipe: re-fit every part above the encoder, re-run the same automatic seed-42 pick, for AFF and every baseline | carrying CLIP's knob values would not tell method from settings | user | 14:20 |
| D4 | If the seed-42 pick sets λ_a = 0 in both halves | stop: no held read; reported as "switched off" | a certain NO-GO would spend a read for nothing | user | 15:00 |
| D5 | Bar on ArtELingo per backbone | round 6's seven checks, Holm per backbone | a joint family is stricter for no gain; a two-check bar differs from CLIP's | user | 13:20 |
| D6 | ArtELingo-28 languages | translate to English, all three backbones | native text drops CLIP and mixes encoder and AFF; PercepT released no translations | user | 14:20 |
| D7 | ArtELingo-28 paintings | 746: our 248 held plus 498 new | all 1,658 include 912 fitted ones; 498 alone is too thin | user | 14:35 |
| D8 | ArtELingo-28 bar | the two-level test (§3.4) | seven checks may fail from power alone; two checks alone give up the full claim | user, after seeing the recommendation | 14:50 |
| D9 | ArtELingo-28 after a NO-GO on ArtELingo | runs anyway, on every backbone | the language check stands on its own | user | 14:20 |
| D10 | Affection | dropped if no data by 2026-10-24; later, future work or the appendix | leaves time for a full claim test before the cut-off | user | 14:50 |
| D11 | Paper chat and cut-off | opens after this spec; a result enters the paper body if its report exists by 2026-11-06 | the stable parts can be written now; the date leaves ten days | user (opening), user, after seeing the recommendation (date) | 14:20 |
| D12 | Where reads run | locally, from pinned files; the cluster for re-fits, sweeps, extraction, translation, probes | the run-once guard and bit-exactness are local; a read saves 10 min | user, after seeing the recommendation | 15:00 |
| D13 | Paper table | every backbone in one main table, pass or fail | C5: no read is dropped | user, after seeing the recommendation | 14:35 |
| D14 | Time cap | none that drops anything; past 3× its estimate a stage pings the main chat | AFF has no variants a cap could drop | user, after seeing the recommendation | 14:50 |
| D15 | Split on new backbones | CLIP's leakage groups and split, sizes asserted | other features would hash other groups | user, after seeing the recommendation | 14:50 |
| D16 | Seeds for H6, H7 | 52 to 54, the H5 episodes | pairs backbones on the same episodes; the rows are the same either way | user, after seeing the recommendation | 14:50 |
| D17 | Ledger | C5 amendment 2: one read per method, backbone and split; a backbone column; rows H6 to H10 | the ledger counted per dataset | user, after seeing the recommendation | 14:50 |
| D18 | A3 on a new backbone | re-trained with its chosen settings, not re-picked from the 8-run grid | the grid pick was a method choice | user, after seeing the recommendation | 14:50 |
| D19 | SigLIP 2 features | so400m/14-384, one fresh run of images and captions; round 8's image-only features not reused; added to `scripts/extract_features.py` | no equivalence check exists for round 8's | user, after seeing the recommendation | 14:50 |
| D20 | Translator | NLLB-200 3.3B for all 27 languages, coverage checked first | GemmaX2-28 supports only 12 of the 27 | user, after seeing the recommendation | 14:50 |
| D21 | ArtELingo-28 text | English rows out (88% copies of our captions); 27 languages pooled; per-language descriptive | – | user, after seeing the recommendation | 14:50 |
| D22 | ArtELingo-28 licence | checked and recorded before the paper uses it; the read does not wait | – | user, after seeing the recommendation | 14:50 |
| D23 | Probes | development splits only; no claim | test splits stay for stage 3 | user, after seeing the recommendation | 14:50 |
| D24 | Cluster data | CUB and SemArt in `DATA_MAP`; a read-only check of features and weights on the node before launch | – | user, after seeing the recommendation | 14:50 |
| D25 | Wording | a run stopped by the rule is "dropped under the rule", not "failed" | – | user, after seeing the recommendation | 14:50 |
| D26 | Old branch | delete the remote `origin/r8-grouping` | as for round 7's | user, after seeing the recommendation | 14:50 |
| D27 | Comparators on new backbones | the seven-check comparators plus B′(A1), R1 and PM; DTS, fine-tuned CLIP and the reranker stay CLIP-only | those three are CLIP-built and costly to port | user, after seeing the recommendation | 15:00 |
| D28 | Stage types and loop size | claim tests (stages 3, 4), exploration (stage 5), one full loop | each read is a paper claim | agent default | 15:05 |
| D29 | Seeds for ArtELingo-28 | 55, 56, 57 (free) | nothing was picked there; fresh seeds keep the ledger plain | agent default | 15:05 |
| D30 | A switched-off backbone on ArtELingo-28 | its ArtELingo-28 read is not run either | AFF equals B there too | agent default | 15:05 |
| D31 | Preparation before the rule | extraction and translation may start before the rule is committed | they read no label and score nothing; C7 is about scoring code | agent default | 15:05 |
| D32 | Recipe replay | stage 0b on CLIP, A3 tolerance fixed in the rule | shows the recipe is the one AFF went through | agent default | 15:05 |
| D33 | Cluster use | waves on node406 and node408 (3 GPUs each, reserved until 2026-10-18 14:40): SigLIP 2 and translation first, then ArtELingo-28 features, then the two re-fits; node work ends by 10-16 and nodes are cleaned before release | the work is small; the cluster saves wall time and keeps the local GPU free | user, after seeing the recommendation | 14:55 |
| D34 | Run mode | day: build and run chats may ask, with a 30-minute default | the user is at the desk | user | 15:00 |

**Amendments**: none.

## 5. Assumptions and open questions

- Assumption: Qwen's `artelingo_full` store joins `load_artelingo()` rows by `sample_id`. Checked: 308,723 unique ids
  (facts brief §2); the build asserts the join.
- Assumption: the recipe on CLIP replays CLIP's files, apart from A3's GPU non-determinism. Checked: not yet; stage 0b.
- Assumption: NLLB-200 covers the 27 languages well enough to keep affect words. Checked: not yet; the coverage check,
  and per-language numbers.
- Assumption: the 498 new ArtELingo-28 paintings carry style and genre labels like the others. Checked: the CSV has
  both for every row; the build asserts it.
- No open questions: the grill settled all.

## 6. Success criteria

**Decision mode.** Binding rule: `src/test/20261202_r9_scope/DECISION_RULE.md` @ a443902, committed before any
scoring code; it governs where it differs from this spec.

| ID | What is measured | Compared against | Bar | On which data |
|---|---|---|---|---|
| SC1 | P1 to P7 on Qwen | COS, RCA, B, B′(A0), CF | all seven pass, Holm, one-sided α 0.025 | ArtELingo held, seeds 52 to 54 (H6) |
| SC2 | P1 to P7 on SigLIP 2 | as SC1 | as SC1 | as SC1 (H7) |
| SC3 | level 1 (P4, P6), then level 2 (P1, P2, P3, P5, P7), per backbone: CLIP, Qwen, SigLIP 2 | as SC1 | the two-level test (§3.4) | ArtELingo-28 read set, seeds 55 to 57 (H8 to H10) |
| SC4 | probe reading per dataset | – | none (exploration) | CUB and SemArt development |

- **GO (SC1, SC2), per backbone:** the pre-read checks pass and all seven checks pass.
- **SC3, per backbone:** "full claim" if both levels pass; "small claim" if level 1 passes and level 2 does not;
  NO-GO if level 1 fails.
- **NO-GO:** each failed check read on its own (C11): "inconclusive at a detectable margin of x" or "did not beat".
- **Switched off:** stage 2's stop (D4, D30); reported as a result, no read made.
- **Licensed claims.** SC1 or SC2 GO: "on ArtELingo's held paintings, AFF re-fitted by its fixed recipe on frozen
  Qwen (or SigLIP 2) features beats cosine, RCA, B, B′(A0) and its matched control on R@1, with a positive condition gain over
  its control and RCA, pooled over three aspect pairs." Not: each pair, other datasets, a margin size. SC3 full:
  "with captions from 27 other languages, machine-translated to English, AFF trained on English ArtELingo keeps
  the same sentence on that backbone"; small: "beats B′(A0) on R@1 with a positive condition gain over its matched
  control". Not: native-language text, any single language. SC4: a direction only.

## 7. Constitution check

| Principle | Pass or exception | Note |
|---|---|---|
| C1 strongest comparator | pass | B′(A0) and RCA per backbone; B′(A1) reported |
| C2 matched control | pass | CF per backbone |
| C3 beat the current best | not applicable | AFF is not replaced; the reads test its scope |
| C4 seeds | exception | H6 and H7 reuse H5's seeds 52 to 54 (D16) |
| C5 held data | pass | one read per method, backbone and split; ledger rows first; every read reported |
| C6 sample IDs | pass | joins by `sample_id` and painting name, asserted |
| C7 rule before code | pass, with D31 | extraction and translation precede the rule; no scoring code does |
| C8 prior and disclosure | pass | §3.1; §8 |
| C9 statistics | exception | ArtELingo-28: a two-level test instead of "every check must pass" (D8) |
| C10 development bar | not applicable | no development decision; the stop of D4 is not a bar |
| C11, C12, C13 | pass | stage 0a; re-derivation of every verdict number before the rule is applied |
| C14 claim scope | pass | held reads, claims in §6 |

Exceptions:
- **C4:** the H6 and H7 episodes equal H5's · needed to compare backbones on the same episodes · the held rows are the
  same with fresh seeds, so fresh seeds would add no protection. The seeds are still hash-checked against every
  development and earlier test seed except H5's own episodes, which they equal by design.
- **C9:** a small read set (746 paintings) would fail seven checks from power alone · the two-level test keeps the
  family-wise error at α · seven checks or two checks alone were the rejected options (D8).

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| The generic chain changes a CLIP number | stage 0a | stop and trace (C12) |
| The recipe is not the procedure AFF went through | stage 0b | stop; fix the recipe before any new backbone |
| A misjoined row (Qwen ids, ArtELingo-28 paintings, translated rows) | join asserts; spot checks of 20 rows per store | fix before stage 2 |
| Poor translation in some languages | coverage check; per-language numbers | report it; the claim says "machine-translated" |
| ArtELingo-28 too small to pass | the detectable margin | C11 wording; the two-level test |
| 248 ArtELingo-28 paintings were read in H1 to H5 | known | disclosed; the 498-painting subset reported |
| Cluster and local numbers differ | reads load pinned files locally; C13 re-derives locally | trace any difference before the verdict |
| The reservation ends (2026-10-18 14:40) | the build's timeline | node work by 10-16; else the rest runs locally |

**Selection disclosure.** AFF was found among about 50 variants on CLIP's seed 42 (R3 rule). On each new backbone only
the automatic pick runs on seed 42 (224 cells, plus the condition-free and baseline picks); no human choice. ArtELingo's
held rows were read five times (H1 to H5); this loop adds H6 and H7, a read per method and backbone (C5). Protection:
held rows, nothing picked on them, a frozen method.

## 9. Deliverables and review

- **At the end:** the round folder `src/test/20261202_r9_scope/` (prefix `r9_`), its `DECISION_RULE.md` and log; the
  generic chain; the SigLIP 2 and ArtELingo-28 stores under `/data/SSD2/pre_extract/`; held-ledger rows H6 to H10; the
  auto report `docs/reports/auto/v2/2026-12-02_aff_scope.md`, its user-read report, and the probe note for stage 3.
- **Reviews:** the rule checked by a fresh Opus reviewer before commit (C7); verdict-code ticket reviews; C13
  re-derivation; the final whole-branch review on Opus at xhigh, one fix wave, a scoped re-review.
- **Stop points:** none between approval and the reports. Day mode: a chat may ask, with its default after 30 min.
- **Budget:** cluster about 3 to 5 GPU-hours [estimate: SigLIP 2 0.5 to 1 h, translation 1 to 3 h, ArtELingo-28
  features 0.5 h, two A3 runs 0.4 h]; local GPU under the lock for GoEmotions and A3 in the probes; CPU about 1 to 1.5 h
  per backbone re-fit and about 10 min per read. Wall time: build 2026-10-14 to 10-15, run 10-15 to 10-16.
  Time-box: node work by 2026-10-16, all reads by 2026-10-18.
