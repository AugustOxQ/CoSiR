# Rule review: r14 spec (c* on three encoders, the photo model on EmoSet, the analysis sweep)

> Reviewed 2026-11-02, finished 04:01, by a fresh Opus subagent (constitution C7). Spec:
> `docs/superpowers/specs/2026-11-02-backbones-sweep-design.md` @ 387d0e0 (unchanged since). Read: `spec_core.md`,
> `spec_experiment.md`, `held-data.md`, the constitution, the glossary, the r10, r11, r12, r13 and 10-25 specs, the
> facts brief, reports 12-24, 12-25, 12-28, 12-29 and 2027-01-01, the stored outputs named below (sidecars and
> listings only; no feature array loaded), and the EmoSet split files inside the zip (entry counts and format).
> This file is the only edit.

**Verdict: pass after fixes.** There are 2 must-fix items: EmoSet's key order carries the class, and the failure
paths contradict each other and have no outcome. There are also 13 should-fix items and 13 nits. No planned step
tunes, selects or stops on held data. One earlier selection on held data (the encoder set) needs a note in §7.

## Answers to the six questions

### Q1. Hard rules and parts table

| Hard rule | Status | Note |
|---|---|---|
| 1 Brief and nine sections | pass | all present; §5 "Open questions: none"; Amendments "none" |
| 2 At most three clarifications, none left | pass | none raised or open |
| 3 Comparator and numeric bar, fixed before results | pass in form | WIN: lower 95% bound > 0. LEVEL: lower bound > −r, with r fixed on validation before test. Strong comparators present (teacher, matched arm, strongest PercepT entry beside). Two bars are not yet pinned to one computation (M1, S1) |
| 4 Who decided | pass | every row names it; D1 ("user, after seeing the recommendation") is missing from "Check these" (N3) |
| 5 Amendments everywhere | n/a | none yet |
| 6 Tickets cite IDs, coverage line | n/a | applies at `to-tickets`; §9 commits to it |
| 7 Body near target, reason if over | pass | over target, with a one-line reason; the stated count is off (N4) |

Parts table:

| Stage type | Required parts | Status |
|---|---|---|
| Claim test (parts a, c) | 3.1 to 3.5, 3.7, §6 licensed claim, §8 disclosure, §9 budget (3.6, selection rule when they apply) | all present. The part c sentence is inaccurate (S10). §7 lacks the held-data note (S11) |
| Ablation (part b) | 3.2 to 3.5, 3.7, §6 licensed claim, §9 budget | §6 licensed claim **missing** (S9); the rest present |
| Stages 1 and 2 | typed "measurement", which is not in the type table | they pick read-outs and wordings, so they are sweeps (S8). The sweep parts (selection rule, disclosure) are already present |

### Q2. held-data.md

Pass for every planned step:
- Stopping uses ArtELingo validation (X-c*) and Affection validation (X photo models).
- Seed ranges, read-out picks and wording picks use the validation splits (ArtELingo, Affection, EmoSet).
- B1 and B3 read validation only. B2 reads held and selects nothing. An ablation may touch the data of the result it
  explains.
- The matched arm's "larger reading" is chosen on held. That makes the bar stricter (r10 and 10-25 practice). It
  shapes no method.
- Not recorded in §7: the three encoders were chosen by an earlier held emotion test on the same 12,281 held
  paintings (10-25, 10-26). §8 and the licensed sentence disclose it, but §7 says "pass" and "Exceptions: none" (S11).
  Query expansion reads unlabelled test embeddings at test time. r11 and r12 note this in §7; r14 does not (S11).

### Q3. Can each SC be computed, and is each quantity defined once?

| SC | Inputs: exist or produced | Defined once? |
|---|---|---|
| SC1 | X-c* held exports (new); X ArtELingo stores exist (61,402 rows each, facts brief Q1a); labels | yes |
| SC2 | X-c* picks (stage 1). Frozen bars: r11 stage 5 per-query outputs exist for CLIP L/14 and SigLIP 2 (`cosir_r11/.../stage5/perquery/clip__*`, `siglip2__*`; both picks center+k10). SigLIP 1 is new | SigLIP 1's pick range is unclear (S4) |
| SC3 | Full-scale matched arms with seeds 0 to 4: M2 (10-25 `full/held/M2`, 10 files) and M7 and M8 (10-26 `full/held/M*/P3__s*.npz`, 12,281 × 40). Own and 20-cluster partitions are in `partitions/full/M*` | statistic is ambiguous (S1); scale not named (S2) |
| SC4, SC5 | X Affection stores: to be extracted (CLIP L/14 valtest exists) | the X frozen image's pick is ambiguous (S4) |
| SC6 | Teacher outputs exist: 10-26 `partitions/full/T`, r11 `instructed__*`, r12 `teacher_artwork` per-query and partitions | yes; the LEVEL unit is new for clustering (N8) |
| SC7 to SC10 | EmoSet zip present (11.3 GB). Checked: `val.json` 5,905 and `test.json` 17,716 entries, 8 classes. r13's final checkpoints exist (`runs/final/s*/emotion.pt`) | evaluation order (M1); r_E and the order of picks (S3) |
| B2 | All eight removals have weights (30 files each), held exports (5 seeds × 3 heads) and validation exports (`full/train/<arm>/R6-5/`, 6,152 rows, `output_sha256`). All nine full-scale arms ran on host `cbd437f8ce92`, the 3090 (`run__s*.json`) | the check is keyed on the weights, not the exports (S5); "applied unchanged" (S4) |
| B1, B3 | new runs on node402 | what they keep (N7) |

### Q4. Verdict logic

The cell outcomes map to exactly one verdict each:
- Paintings: 9 WINs give holds. All 6 SC1 and SC2 cells WIN, with 1 or more SC3 cells not WIN, gives qualified. Any
  SC1 or SC2 cell not WIN gives fails.
- Photos: both SC4 and SC5 WIN give holds; anything else gives fails.
- Part c: 4 WINs give holds. SC7 and SC8 WIN, with {SC9, SC10} = {WIN, LEVEL}, gives qualified. Anything else gives
  fails.

D3, D4, the brief and §6 agree on these patterns. The gaps:
- Missing runs and failed gates: the rules contradict each other, and "inconclusive" is never named (M2).
- Part c has no GO/NO-GO mapping (fixed in M2).
- The brief's "qualified" for part c leaves out the WINs over the plain image (N1).

### Q5. The six points

- **a. SC3:** the data are there. The statistic needs one definition: the larger reading is taken again in each
  draw, and the seeds of the two arms are drawn independently, as 10-26 did. r10's shared seed draw does not fit here
  (S1). Name the full scale (S2).
- **b. Part a pattern:** complete and disjoint for cell outcomes (Q4). There are 24 SC6 cells (8 per encoder), each
  its own claim, with no correction for multiplicity. This is disclosed (§3.4 line 172), and BELOW keeps its row.
  Gaps: a run failure or failed gate gives no named outcome (M2); the qualified sentence has no form for zero SC3
  WINs (N6).
- **c. r_E:** the right unit. LEVEL uses the candidate's own range (§3.4 line 172), as r13 did (r13 spec line 153).
  r12 used c*'s range because c* was r12's candidate. SC7 and SC8 are WIN bars, so r_E does not enter them. For every
  arm's read-out score, r_E only scales the score, so it changes a pick only through the 0.001 tie band. Its two parts
  and the order of the picks need defining (S3).
- **d. B2:** the same-machine claim holds: all r10 full-scale runs used one host, the 3090. The retrain branch keys
  on the wrong artefact, has no budget, and conflicts with the stage 0 table (S5).
- **e. Near-copy cut:** label-free, with r13's method. The reference set is ambiguous (S7). The keys carry the class
  (M1). Add an uncut diagnostic (N11).
- **f. Budget:** feasible. The node402 estimate is optimistic but has wide margin before 2026-11-04 18:06. The
  report's behaviour if B1 overruns is not stated (S13). Disk and local GPU numbers are nearly right (N9).

### Q6. Placeholders, contradictions, scope, numbers

- **Placeholders:** none, except the intended slots in the licensed sentences (<X>, <n>, <aspects>, <task>).
- **Contradictions:**
  - the stage 1 gate against the rerun rule (M2);
  - the stage 0 table against the rerun and retrain branches (S5);
  - D9 against D15 (N7);
  - the brief's list of replays against 0(vi) (N2).
- **Scope creep:** none. The descriptive EmoSet rows are small (D10).
- **Ambiguous:** M1, S1, S3, S4, S7.
- **Numbers that disagree with their sources:** S12, N4, N9.

## MUST-FIX

**M1. The EmoSet key order carries the class.** §3.3 lines 142 to 143, with §3.4 line 164.
- What is wrong:
  - Each EmoSet split entry is `[label, "image/<class>/<class>_<n>.jpg", "annotation/..."]`.
  - r12's protocol, which §3.4 inherits, uses the Python-sorted key list as the evaluation order (r12 spec lines 121
    to 124). That order fixes the cross-fit halves (parity of position), the bootstrap rows and the tie order of the
    gallery ("ties keep gallery row order", r11 spec line 117).
  - Sorted EmoSet keys group the photos by class. The cross-fit halves then follow the classes, and ties on the
    held test set are broken by class. This contradicts "labels … read only when scoring" (line 143).
  - "In extraction order" (line 142) may also differ from the sorted order, so the rule does not fix one order.
- Evidence (checked): in `val.json`, every image path has its class as its folder (all 5,905 entries). The split
  file's own order is shuffled: 5,174 label runs, the longest 4. The sorted order is not.
- Fix: "EmoSet key: the image path string of the split file (for example `image/awe/awe_08973.jpg`). No code reads
  the class from a key. Labels come only from the entry's first field, at scoring. Evaluation order (bootstrap rows,
  gallery order and ties, cross-fit halves by parity) is the keys sorted by the sha256 of the key string, in place of
  r12's sorted keys." Using the split file's row order is also acceptable; pick one. Add a unit test that the order
  does not group classes.

**M2. The failure paths contradict each other and have no outcome.** §3.5 line 185, §3.4 lines 151 to 152, §8 line
330, §6 lines 283 to 289.
- What is wrong:
  - The stage 1 gate needs "30 runs trained", else "stop". So a second failure on one encoder stops all of part a.
  - §3.4 stops only "that encoder's part", and §8 says "the others go on".
  - The claim-test outcomes are GO / NO-GO / inconclusive (type table), but the spec never names "inconclusive".
  - The spec does not say whether a failed X photo run removes that encoder's paintings verdict.
  - Part c has no GO/NO-GO mapping.
- Fix, in three places:
  - Stage 1 gate: "for each encoder still in part a: its five X-c* runs and five X photo runs trained; its seed
    ranges and read-out picks written and committed before any held number of that encoder".
  - Add to §6: "A paintings verdict needs X-c*'s five runs. A photos verdict needs X's five photo runs. If an
    encoder's runs or its stage 0 gate fail, that encoder's claim and its SC6 cells are inconclusive: no claim, and
    the row is reported. Part c: holds is GO, qualified is GO for the qualified claim, fails is NO-GO. A failed stage
    0 or stage 2 gate makes part c inconclusive."
  - Brief Outcomes: add "If a run fails twice, that encoder gets no verdict (inconclusive); the others go on."

## SHOULD-FIX

**S1. The SC3 statistic has two inherited readings.** §3.4 lines 154 to 156; §6 line 274.
- Problem:
  - r10's joint bootstrap draws one set of seeds, shared by both arms (r10 `DECISION_RULE.md` lines 128 to 129).
  - 10-26's matched margin draws the seeds of each arm independently and takes the larger reading again in each
    draw (10-25 spec line 151; 10-25 `rb_score.py` lines 28 to 45, one seed draw per arm).
  - §3.4 states r10's form, then says "as 10-26's matched margin".
  - "Larger reading" can also be read as fixed once, from the point values.
  - X-c* and the matched arm are independent trainings, so pairing seed s with seed s has no basis.
- Fix: "SC3: d = X-c*'s 20-cluster AMI, mean over seeds 0 to 4, minus the larger of the matched arm's two readings
  (own topics; 20 clusters, three states), each a mean over its seeds 0 to 4. The bootstrap is 10-26's `b_matched`:
  2,000 draws with `default_rng(0)`; each draw resamples the rows once, then draws seeds independently for X-c* and
  for the matched arm; the larger reading is taken again in each draw." Then 0(vii) replays 10-26's `b_matched` d and
  intervals (S6).

**S2. The matched arm's source and scale are not named.** §3.2 line 110; §3.5 lines 206 to 208.
- Problem:
  - 10-25 and 10-26 each stored a sample-scale (stage 3, 13,400 training paintings) and a full-scale (stage 4)
    matched arm.
  - The decide-3 handoff (line 37) points at `reported_stage3.md`, which is sample scale (its line 1: "stage 3
    (sample scale)"). X-c* trains at full scale.
  - 0(vii) cites only 10-26, but SigLIP 2's arm (M2) is 10-25's.
- Fix: "X matched arm: the full-scale (stage 4) PercepT core P3 arms, seeds 0 to 4, with their stored held outputs
  and partitions: M2 (SigLIP 2, 10-25); M8 (SigLIP 1) and M7 (CLIP L/14) (10-26)." In 0(vii), write "10-25's and
  10-26's `reported_stage4` entries".

**S3. r_E and the order of the EmoSet picks.** §3.4 lines 164 to 168; §6 line 291.
- Problem:
  - r_E has two parts but one name.
  - "AMI and mAP at R0" does not say K = 20, three states, or the photo set after the near-copy cut.
  - r12's retrieval wording rule compares the wordings "by the cross-fit validation mAP at each wording's own
    read-out pick" (r12 spec lines 253 to 255). So each wording's read-out pick comes before the wording pick, but
    §3.4 puts the wording first.
- Fix: "r_E,clu: maximum minus minimum, over the photo model's five final runs, of EmoSet validation emotion AMI (20
  clusters, mean of three states, all validation photos left after the near-copy cut). r_E,mAP: the same for emotion
  mAP at R0, with those photos as queries and gallery. Order of the picks:
  1. r_E.
  2. Each arm's read-out (photo model, plain Qwen, each teacher wording) by r12's rule, scored as
     (mAP − mAP(R0)) / r_E,mAP, with seeded arms averaged over seeds.
  3. The wording per task by r12's rule: clustering by validation AMI; retrieval by cross-fit mAP at each wording's
     own pick. The photo wording is picked only if better by more than 0.001 r_E.

  SC9's LEVEL uses r_E,clu; SC10's uses r_E,mAP."

**S4. The read-out picks of the bar arms are not fully defined.** §3.4 lines 157 to 163; §6 lines 290 to 291.
- Problem, in four parts:
  - (a) SigLIP 1's frozen image picks "by the same rule", but r11's rule uses "c*'s r_a for every arm" in its score
    and guard (r11 spec lines 237 to 238). The spec does not say which r_a: Qwen c*'s (as for the reused CLIP L/14
    and SigLIP 2 picks) or SigLIP 1-c*'s. The guard can change the pick.
  - (b) On Affection, CLIP L/14's frozen image already has r12's pick (center+k10, r12 `stage1/picks.json`).
    §3.4 says the picks use the X photo model's ranges, while §3.7 says reused picks stay frozen.
  - (c) B2's "c*'s r11 pick applied unchanged" can be read as applying c*'s fitted centering vector to the ablation
    outputs.
  - (d) The spec does not say whether the descriptive EmoSet rows use a picked read-out or R0 only.
- Fix:
  - (a) "SigLIP 1's frozen image picks by r11's stage 4 rule with r11's r_a, as for the reused picks." Or choose
    SigLIP 1-c*'s ranges, but say which.
  - (b) "On Affection, CLIP L/14's frozen image keeps r12's pick. The SigLIP 2 and SigLIP 1 frozen images pick by
    r12's rule with their X photo model's r'_mAP."
  - (c) "B2: R0, and the read-out form centering + k = 10, fit on each run's own validation exports."
  - (d) "Descriptive EmoSet rows: R0, and the read-out picked by r12's rule with r_E."

**S5. B2's checks and the failure branches.** §3.2 lines 120 to 122; §3.5 lines 184 and 206 to 208.
- Problem:
  - B2 scores from the exports, so it does not need the weights. Yet the branch checks the weights' hashes and then
    retrains.
  - If an export fails but the weights are good, CPU inference is enough.
  - A retrain also needs c* at seeds 0 to 4 on node402; B1's node402 c* has seeds 1 to 3 only. The budget does not
    include this.
  - The stage table says a failed stage 0 gate stops the part. But (vii) reruns a failed matched arm, and §3.2
    retrains a failed B2 run.
  - The matched-arm rerun has no machine, no seeds and no check.
- Evidence:
  - Validation exports: `full/train/<arm>/R6-5/`, 30 files per removal, each with `output_sha256`.
  - Held exports: 5 seeds × 3 heads per removal.
  - Weight sidecars hold `weights_sha256` and `output_sha256`.
  - All nine full-scale arms ran on host `cbd437f8ce92`.
- Fix:
  - "0(vii), B2: each export's sha256 matches its sidecar. All eight removals' held Δ reproduce report 12-28's held
    table within 1e-9 (r10 `ablation/held.json`)."
  - "If an export fails, export it again from its weights on CPU (weights hash checked). Only if that fails: retrain
    the removal and c* on node402, seeds 0 to 4, after B1; that row then uses the node402 c*."
  - "If a matched arm fails its replay, rerun it on node402 with 10-25's Stage 2 settings, seeds 0 to 4, after part
    a's training, and replay its regression first. Any other failed stage 0 gate stops its part."
  - Add both branches to §9 as contingencies.

**S6. The regression check (C12) misses several paths that feed a verdict.** §3.5 lines 193 to 210.
- Problem: these paths have no replay:
  - (a) the generalized export path (r13 had an inference gate, its 0a(iii));
  - (b) the near-copy code (r13 removed 2,328 of 72,255);
  - (c) the wording-pick code (r12 picked the artwork wording in both tasks);
  - (d) the bar paths: the frozen-image clustering (F2-r20 0.0569, F7-r20 0.0659, F8-r20 0.0672), r11's CLIP L/14
    and SigLIP 2 stage 5 mAPs, and the matched margin's d and interval;
  - (e) the B2 Δ of seven removals (see S5).
- Fix: add to stage 0:
  - "(ix) The generalized export path, given c*'s and r13's checkpoints, reproduces their stored ArtELingo validation
    and held exports and their Affection validation and test exports within 1e-3, all five seeds."
  - "(x) The near-copy code reproduces r13's removed set (2,328 keys) exactly."
  - "(xi) The wording rule reproduces r12's wording picks and their scores within 1e-9."
  - "(xii) The scoring path reproduces, within 1e-9: the F2, F7 and F8 20-cluster AMIs; the `b_matched` d and
    intervals of K2, K7 and K8 (style, genre; 10-25 and 10-26 `reported_stage4`); and r11's stage 5 CLIP L/14 and
    SigLIP 2 mAPs."

**S7. The near-copy reference set has two readings.** §3.3 lines 143 to 145.
- Problem: "any Affection train photo" can mean all 72,255 photos, or the 69,927 the photo model trained on.
- Fix: "Cosine against all 72,255 Affection train photos (the larger set; it includes r13's 2,328 removed photos)."
  Or 69,927, but say which. Add: "The cut set is the evaluation set of every EmoSet arm."

**S8. Two stage types are not in the type table.** §3.5 lines 185 to 186; D11 line 244.
- Problem: "measurement" is not a type. Stages 1 and 2 choose read-outs and wordings, so they are sweeps by the type
  questions. r12 typed the same stage "sweep" (r12 spec line 155).
- Fix: type stages 1 and 2 as "sweep (development only)" and update D11. Stage 5 can stay "descriptive" or move
  into §3.6.

**S9. Part b has no licensed claim.** §6 lines 293 to 308.
- Problem: the parts table requires §6 Licensed claim for an ablation. §6 has only the "Not licensed" line.
- Fix: add "Licensed (part b): 'On <data>, removing <part> from c* (or setting <knob> to <value>) changes <aspect>
  AMI (or mAP) by <Δ> [<interval>] against c* on the same machine (<seeds>).' This is an attribution for c*'s and
  the photo model's results, never a better setting."

**S10. The part c licensed sentence is wrong.** §6 line 303.
- Problem: "the photo model, trained only on Affection" is false. The photo model is c*'s emotion pathway, trained
  on ArtELingo paintings, then fine-tuned on Affection's unlabelled train photos (glossary, "photo model").
- Fix: "the photo model (c*, trained on ArtELingo paintings, then fine-tuned on Affection's unlabeled train photos;
  frozen; never trained on EmoSet)".

**S11. The §7 held-data row leaves out the earlier selection.** §7 lines 320 and 322.
- Problem: an earlier held emotion test (10-25, 10-26), on the same 12,281 held paintings, chose the three encoders.
  §8 and the licensed sentence disclose this, but §7 says "pass" and "Exceptions: none".
- Fix: "pass, with one disclosed earlier selection: the three encoders passed a held emotion test (10-25, 10-26) on
  these held paintings, and the licensed sentence names it (D2, user). Query expansion reads unlabelled test
  embeddings at test time, with k fixed on validation, as in r11 and r12."

**S12. Numbers in §3.1 disagree with their sources.** §3.1 lines 89 to 94; brief line 42.
- Problem: "+0.021 to +0.036 over the frozen image" and "only SigLIP 1 and CLIP L/14 reached the teacher on style"
  are 12-25's figures for its four students (CLIP B/16, CLIP L/14, SigLIP 1, DINOv2). The full-scale figures for the
  three r14 encoders are:

  | Encoder | Emotion over its frozen image | Emotion over the teacher | Style over the teacher |
  |---|---|---|---|
  | SigLIP 2 | +0.0455 | +0.0103 [+0.0073, +0.0135] | +0.0278 [+0.0224, +0.0327] |
  | CLIP L/14 | +0.0337 | +0.0075 [+0.0044, +0.0106] | above the teacher |
  | SigLIP 1 | +0.0364 | +0.0116 [+0.0082, +0.0150] | above the teacher |

- Evidence: 10-25 `outputs/part2/reported_stage4.md` lines 11, 12 and 18; 10-26 `outputs/part2/reported_stage4.md`
  lines 55, 56, 62, 99, 100 and 106.
- Fix: write "+0.034 to +0.046; under R6-5 all three reached the teacher on emotion and on style, none on genre".
  Then restate the teacher priors on this evidence (emotion for CLIP L/14, style for SigLIP 2); the probabilities stay
  the author's. Align the brief's expectation line.

**S13. The budget and the time-box.** §9 lines 346 to 352; §5 line 257.
- Problem:
  - The node402 time rests on an unchecked full-scale time on the A6000 (§5).
  - r10's node figure (3 min per run) was at sample scale (facts brief Q2d).
  - r10's held ablation took 252 GPU-min against 80 estimated (report 12-28 line 253).
  - Some B1 values make runs longer: batch 256, patience 30, learning rate 0.0003.
  - B1 runs last. The spec does not say what the report does if B1 is not done when node402 ends (2026-11-04 18:06).
    The report is due at 23:00 that day.
- Our check:
  - node402: 12.4 GPU-h on 3 GPUs is about 4.1 h of wall time. Two to three times that still fits before 11-04
    18:06, if node work starts by the morning of 11-03.
  - Local GPU: 2¼ h matches the handoff's rates (SigLIP 2 75, SigLIP 1 51, CLIP L/14 114, Qwen 20 images/s). That
    gives about 57 min for Affection and 75 min for EmoSet.
  - Disk: /data/SSD has 2.4 TB free and /data/SSD2 1.8 TB free.
- Fix: "Time the first X-c* runs on node402. Rows not finished by 2026-11-04 18:06 are listed as not run. The report
  does not wait for a new reservation."

## NITS

- **N1.** Brief lines 40 to 41: write "qualified: WINs over the plain image in both tasks; over the teacher, a WIN in
  one task and LEVEL in the other".
- **N2.** Brief line 25 says the stage 0 replays are "the r11, r12 and r13 scores". 0(vi) replays report 12-28
  (r10), r11 and r13. Align the two.
- **N3.** Brief "Check these": add D1, which the user decided after seeing the recommendation (hard rule 4).
- **N4.** Line 354: the body has about 256 non-blank lines (297 with blank lines), not "about 230".
- **N5.** The names B1, B2 and B3 for part b clash with B1 to B6, the PercepT entries in reports 12-24, 12-25 and
  12-28 ("B1-own 0.0874"). Use "the knob rows", "the ablation rows" and "the photo rows" (or (i) to (iii)).
- **N6.** §6 lines 297 to 298: the qualified sentence "better in <aspects>" needs a form for zero SC3 WINs, such as
  "…; against PercepT on the <X> image it is better in no aspect".
- **N7.** D9 says "scoring local"; D15 says B1 and B3 keep "metrics, per-item scores and logs only". State that B1
  and B3 score on node402. Each run keeps its per-painting partitions (three states per head) and its per-query AP.
  The paired bootstraps and the C13 spot check need them.
- **N8.** SC6's clustering LEVEL on ArtELingo uses X-c*'s own five-seed validation range. Report 12-28's genre LEVEL
  used the glossary's 0.0128. Say so in §3.4, and report both units beside the cells.
- **N9.** §9 disk: part a's outputs come to about 9 to 10 GB, not about 8 GB:
  - 15 X-c* runs × about 0.38 GB;
  - 15 X photo models × about 0.18 GB, with their EmoSet exports;
  - EmoSet exports of the photo model and frozen c*, about 1 GB.

  The local GPU budget leaves out the 0(iv) and 0(v) training (about 8 min). Add to §5 an unchecked assumption:
  large EmoSet images may slow Qwen's extraction (dynamic resolution).
- **N10.** §4, "When" column: use "2026-11-02 03:40" and similar, not "grill 2" or "03:40" (template).
- **N11.** §8: add a row: "r_E is very small or very large, so LEVEL is too strict or too loose | r_E beside the held
  spread | reported; r_E not changed after results". In §3.6, after the verdict, add SC7 to SC10 on the uncut EmoSet
  test, and the removed photos per class.
- **N12.** §3.1 line 96: "EmoSet shares no source with Affection" has no source. Not checked.
- **N13.** §3.2 line 125 (B3): say that epoch 0 is the start point, so "2 epochs" means the checkpoint after two
  training epochs (r13 spec line 129).

## Not checked, and left out

- Not checked: the node402 reservation itself (the `cluster` on PATH is a graph-clustering tool; the 18:06 end comes
  from the decide-2 and decide-3 handoffs); A6000 timing; EmoSet's sources, licence and image sizes; the contents of
  the r12 and r13 per-query files (names only); M2's emotion entries in 10-25 `verdict_stage4.json`; whether the
  weight hashes match their files.
- Left out: no feature array loaded; no code run beyond listings, sidecar reads and the EmoSet split-file counts. One
  review pass.
- I would remove: the scratch listing of the EmoSet zip,
  `/tmp/claude-0/-project-CoSiR/3b37edee-531f-453c-b86c-a4cc6113de3a/scratchpad/emoset_list.txt`. Nothing in the
  project.
