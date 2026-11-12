# r29 decision rule review (C7)

> 2026-11-12 21:51 · Reviewer: fresh Opus worker, read-only · Spec: `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md`
> (untracked, last written 21:40, sha256 6613784dc35a…) · Scope: §6 with §3.4 and §3.5, the brief against the body,
> hard rules 1 to 7, C1, C6, C7, C8, C12, C13, held data.

**Verdict: pass after fixes.** All the rule's numbers match their records exactly (table at the end). The pass rule
(D3, D4) and the caption-form check (D5) can be applied with no judgment once the bounds are unrounded (S1). Three
must-fix items block approval:
- M1: stage 0 (i) cannot pass on the local GPU, because r14's SigLIP 1 weights were trained on node402.
- M2: the LEVEL unit, the read-out pick rule and the comparator source are not defined for each cell type.
- M3: stage 0 does not cover the test scorers, the MMArt path, the photo path or the new stores (C12).

## Must-fix

**M1. §3.5, line 159, gate (i): the bit-for-bit target is on another machine.**
- *What is wrong:* all five r14 SigLIP 1 seeds were trained on node402, the DAS6 A6000 (`run__s{0..4}.json`:
  `"host": "node402"`, commit 5f28b2b). D7 puts this loop on the local RTX 3090. r14's own gate 0(viii) measured the
  same recipe on node402 against the 3090: seed 1 AMI changed by −0.0017 / +0.0008 / +0.0024
  (`bs_run_log.md`, 07:38). r10's V4a passed because both runs were on this machine: `gates/v4a_train.json` takes its
  reference from `src/test/20261027_relational_genre/outputs/`, trained locally.
- *Effect:* gate (i) fails, and no code bug causes the failure. Three tries cannot fix it. "The rest waits" then
  stops the whole loop.
- *Fix (replace (i)):* "(i) Trainer and export. (a) On the local GPU, r14's trainer (`bs_cstar.py`, sha256 05fccaa4…,
  unchanged since r14) and the new trainer with the Qwen3 caption input each train SigLIP 1 seed 0. Their weights are
  equal bit for bit, as in r10's V4a. (b) r28's code-level checks pass for the new trainer at caption width 2048:
  model state at start, loss and gradients on a fixed batch equal r14's (`torch.equal`, as
  `test_r28_trainer.py::test_siglip_ref_config_model_loss_are_exactly_r14`). A test shows that width 1,152 changes only
  the caption trunk's first layer and the checkpoint metadata. (c) Given r14's SigLIP 1 seed 0 weights, the new export
  code reproduces r14's validation and test exports bit for bit. It also reproduces r28's `valpick` seed 0 cells
  within 1e-6. A local run is not compared bit for bit with r14's node402 weights. The local minus node402 change in
  the six validation cells is recorded, with no gate (diagnostic, §3.6)." If a DAS6 node is reserved, (a) can instead
  run there against r14's weights.

**M2. §3.4, lines 140 to 144: the LEVEL unit, the pick rule and the comparator are undefined for most cells.**
- *What is wrong:*
  1. "r = K*'s own seed range on validation for that cell" is wrong for SC4 and SC6, where the candidate is E2, E3 or
     F1.
  2. The inherited protocols disagree on the unit. r14 §3.4 takes ranges of AMI at 20 clusters and of mAP at R0. r17
     §3.4, the source of r28's ArtELingo and MMArt AMI cells, takes ranges at the class counts. r23 §3.4 takes text
     ranges at the picked read-out.
  3. The MMArt, Affection and EmoSet validation sets used for the ranges are not named.
  4. "From r28's menus" has no image-mAP case: r28 picked no image read-out for ours. K0's image-to-image pick and
     per-query files come from r14 by r11's rule (`r28_text.py`, `R14_PERQUERY`). For EmoSet, r28 used r14's committed
     r_E,mAP = 0.00695 for every arm (`decisions.md`, 19:44), not each arm's own range.
  5. Q1 is given as "r17 image cells", but r17 has no instructed-Qwen3 mAP cell. Its teacher cells are SC3 and SC4,
     for AMI, NMI, ARI and accuracy only. Q1's ArtELingo image mAP is in r11's `stage5/perquery/instructed__*`.
- *Effect:* under constitution row 7, a night chat must park every cell where this choice can change the outcome.
  That covers the Q1 LEVEL cells of all of SC3 to SC6.
- *Fix:* add a table to §3.4 and a §4 row (agent default). The sources below were checked, except where marked
  "name". Check and complete the marked ones before approval.

| Cells | Read-out pick (validation only) | LEVEL unit r (unrounded) | P1 / plain per-unit values | Q1 per-unit values | Interval |
|---|---|---|---|---|---|
| ArtELingo AMI, class counts | none (k-means as r17) | candidate's own five-seed range of AMI at the class count, ArtELingo validation (r17 §3.4) | r17 SC1 cells, rebuilt from the plain store | r17 SC3 teacher cells, rebuilt from the instructed views | r17: 2,000 draws per cell, rows then seeds |
| ArtELingo image mAP | r11's rule, in units of the candidate's own validation mAP ranges at R0 (as r14 for X-c*) | candidate's own range of mAP at R0 (r14 §3.4) | r17 SC7 | r11 `stage5/perquery/instructed__<a>__*` at r11's pick | r11: 10,000 draws over queries and seeds |
| ArtELingo text, three directions | r23's rule (highest five-seed mean; ties r0, center, center_k10) | candidate's own range at its picked read-out (r23 §3.4) | r23 `scores/art_held/caption/siglip_so400m` at r23's pick | r23 `.../teacher` at r23's pick | r23: 10,000 draws, painting unit, shared seed draw |
| MMArt AMI 9/20/10, image mAP | r15's rule on MMArt validation (5,769 paintings), as r17 | candidate's own range on MMArt validation (r17) | r17 SC1 / SC7 mmart plain | r17 SC3 mmart teacher (AMI); r15 `stage1/picks_teacher.json` (mAP) | r17 |
| Affection AMI, mAP | r12's rule with F1's own five-seed range of validation mAP at R0 (as r28 for c*) | F1's own range on Affection validation | r20 plain cells (name the per-unit file) | r17 SC4 teacher (AMI); mAP: name | r12 |
| EmoSet AMI, mAP | r14 part c step 2; choose and state: F1's own r_E, or r14's committed 0.00695 as r28 used | F1's own r_E on EmoSet validation (r14 part c step 1, recomputed for F1) | r28 `photos.json` cell c | r14 part c teacher at its wording pick (name) | r14 / r12 |
| E2, E3 (Table 6) | rows 1 and 2 | E2's or E3's own range | r17 SC1 / SC7 for siglip2, clip_l14 | as rows 1 and 2 | as rows 1 and 2 |

**M3. §3.5, line 159, and §7, line 246: stage 0 does not cover every path that makes a verdict number (C12).**
- *What is wrong:* stage 0 replays only the trainer, the validation image scorer, the validation text scorer and two
  store checks. These paths are new or changed and produce SC3 to SC7 numbers, but nothing replays them:
  - (a) the test-cell scorers and the WIN, LEVEL and BELOW code. r28 recorded no text intervals. Its image cells are
    r17's records, which r28 did not rescore.
  - (b) the MMArt export and scoring. `r28_mmart.py` only reads r17 and r18 records. The facts brief §7 says r15's
    `mt_export` must be adapted.
  - (c) the photo trainer with a 1,152-wide caption input, and its Affection and EmoSet scoring.
  - (d) the ablation trainer at caption width 1,152.
  - (e) the new stores: the Affection SigLIP 1 text store for F1, the prefixed SigLIP 2 and CLIP L/14 stores if K2 wins,
    and the joins of the SigLIP 2 and CLIP L/14 text stores (C6). Gate (iii) covers SigLIP 1 only.
- *Fix:* add these gates. Each one runs in the build, before the first candidate number of the stage it serves. Each
  one blocks only that stage (see S4).
  - "(v) Test scorers: given K0's stored test exports and the comparators' frozen per-unit values, the new outcome
    code reproduces r17's SigLIP 1 SC1, SC3 and SC7 cells (value, d, lo, hi, outcome) within 1e-9. On the Qwen3 c*
    files, it reproduces r23's nine ArtELingo bar cells (value, interval, outcome) within 1e-9."
  - "(vi) MMArt: the adapted export code, given r14's SigLIP 1 weights, reproduces r15's stored MMArt exports. The new
    scorer then reproduces r17's MMArt SigLIP 1 cstar cells (`r28/mmart.json`) within 1e-9."
  - "(vii) Photo: the photo trainer with the Qwen3 caption input passes r28-style code-level equality with r14's
    `bs_photo` (state, loss and gradients on a fixed batch). Given r14's photo weights, the scorer reproduces r28's
    `photos.json` cell a and r20's Affection photo cells within 1e-9. The ablation scorer reproduces r28's
    `ablation_ref_replay.json`."
  - "(viii) New stores: one row per key, finite values, unit norm within 1e-3, keys join by ID (C6), truncation count
    recorded."
- Update §7's C12 note to "stage 0 (i) to (viii)".

## Should-fix

**S0. §3.5, line 163, and §3.3, lines 125 to 128: the stage 2 claim tests have no development step.**
- *What is wrong:* E2, E3, F1 and the ablation rows are trained, early-stopped, picked and ranged on validation
  inside a stage typed "claim tests". By the type table, a claim test reads test data only. §3.3 gives ArtELingo
  validation to stages 0, 1a and 1b only. It does not cover K*'s own picks and ranges, which 1c needs, or those of E2
  and E3. Line 127 ("its validation picks (r28)") reads as if K* takes r28's MMArt picks, but §3.7 says they are
  rerun.
- *Fix:*
  - Split stage 2 into "2a, sweep (development only): train E2, E3, F1 and A1 to A4; picks and seed ranges on
    validation, written and committed before any test number of that piece" and "2b, claim tests and ablation".
  - In §3.3, set ArtELingo validation's stage column to "0, 1a, 1b, and the picks and ranges of 1c and 2a".
  - Replace line 127 with "MMArt test rows (r15's 16,384 paintings): test. MMArt validation (5,769 ArtELingo validation
    paintings, r17): development for K*'s MMArt picks and ranges (2a)".
  - Add "Affection validation, EmoSet validation: development for F1's early stopping, picks and ranges (2a)".

**S1. §6 line 219 and §3.4 lines 137 to 139: use unrounded bounds.**
- *What is wrong:* "image ≥ 0.38629 − 0.00298" gives 0.38331. The record values give 0.3833128. A K* group mean in
  that 2.9e-6 band passes one reading and fails the other. C13's 1e-6 tolerance is smaller than the band.
- *Fix (SC2's level 1 cell):* "image ≥ m_I − r_I and text ≥ m_T − r_T, with m and r the unrounded record values (m_I =
  0.3862917081, r_I = 0.0029788602, so the bound is 0.3833128479; m_T = 0.2593512884, r_T = 0.0020107136, so the bound
  is 0.2573405749)". Level 2: text ≥ m_T. Level 3: image > 0.3878696916 and text > 0.2606346548. Write these into
  `pass_rule.json`.

**S2. §3.5 (ii): the tolerance is for a different quantity.**
- *What is wrong:* r28's float16 tolerance (`decisions.md`, 19:58: at least 99.5% of values equal, the rest one
  float16 step, maximum 6.1e-5) applies to tower embeddings. It does not apply to mAP values.
- *Fix:* "(ii) The new text scorer on K0 gives every per-seed validation mAP at r0, center and center_k10 within 1e-6
  of `text.json` `ours_siglip1/picks/<a>/<d>/per_seed`, and the same nine picks. The tower embeddings match at r28's
  float16 rule." Add a §4 row for the gate tolerances (1e-6, 1e-9, cosine 0.999). By `spec_experiment.md`, gate
  tolerances are agent defaults with a decision line.

**S3. §3.5 (iv) and §8 line 256: the prefixed store needs a check that only the prefix changed, and the store hash
needs a gate.**
- *Fix:* "(iv) … and, with an empty prefix, the extractor reproduces the first 100 rows of the existing
  `siglip_so400m/txt.npy` (cosine ≥ 0.999, as r14 0(i)). The manifest records the exact stored prefix after the
  extractor's lowercasing ('represent the caption: '). The trainer and the text scorer each record the caption store's
  index hash, and the hashes are equal." Without this item, §8 row 1 points to a check that does not exist.

**S4. §3.5 line 159: "the rest waits" does not say what each gate blocks.**
- *What is wrong:* constitution row 4 says to park the failed piece and go on with the rest.
- *Fix:* "If it fails: fix code bugs only, at most three tries. A gate that still fails parks what it serves: (i) and
  (iii) all training; (ii) SC1 and SC2; (iv) K2 only, so K1 becomes K* and this is flagged in `prefix_pick.json`;
  (v) to (viii) their stage 2 piece. The rest goes on."

**S5. §3.4, no line: no rule for a training crash.**
- *What is wrong:* r28 had a transient CUDA error at 90 °C (log, 20:33) and reran once by r14's rule. This spec
  states no rule, so 1b or a stage 2 piece goes inconclusive at the first crash.
- *Fix:* "A run with a non-finite loss or a crash reruns once with the same seed and code (r14 §3.4). A second failure
  makes that arm's stage inconclusive." Add a §4 row.

**S6. §6 line 234: no rule for counting inconclusive cells.**
- *What is wrong:* r14 required all cells for a verdict. This ladder counts "≥ 12 of 15", and the spec does not say
  how a parked cell counts.
- *Fix:* "An inconclusive cell counts as neither WIN nor LEVEL. A level is reached when its count holds on the decided
  cells. If every inconclusive cell becoming a WIN would raise the level, the report gives 'level L, inconclusive
  above'." Add a §4 row.

**S7. §3.2 line 112, D5: P1's caption form if K2 wins (C1).**
- *What is wrong:* P1's text rows are r23's unprefixed side row. If the prefix helps SigLIP 1's text, P1 with the
  prefix is the stronger comparator.
- *Fix (agent default, §4 row):* "If K2 is picked, P1's text rows use the caption form with the higher P1 text group
  on validation (r0; an exact tie goes to plain). Its read-outs follow r23's rule on that form." The prefixed store
  exists (D15), so this costs scoring only. The alternative is to keep P1 unprefixed and give the reason in §4.

**S8. §3.4 lines 152 to 153 and §3.5: the order of commits and re-derivations is not stated.**
- *What is wrong:* the 1a pick sets what 1b trains. C13 must re-derive it before 1b starts, not "before any
  verdict".
- *Fix:* "Codex re-derives SC1 before stage 1b trains, and SC2 before 1c or stage 2 reads any test row. The SC1 and
  SC2 records, and every read-out pick and seed range, are committed before any test number exists." Add "the seed
  ranges and read-out picks" to the C13 list, as r14 did.

**S9. §3.1 lines 99 to 101: no priors for the test ladders (C8).**
- *What is wrong:* the priors for the test levels are in the brief only ("level 2 at 0.6 and level 3 at 0.3", line
  43–44). The body governs.
- *Fix:* add to §3.1: "SC3: level 1 <p>, level 2 0.6, level 3 0.3. SC4 to SC6: <p> each. Surprise: <line>." Also state
  the pass prior as a joint value. Image 0.8 and text 0.4 give 0.32 if independent, while the spec says 0.4. Either
  write "0.4, assuming the image group holds whenever the text group does" or change it to 0.32.

**S10. §8 lines 262 to 264: the selection disclosure is wrong and incomplete (C8).**
- *What is wrong:* `valpick.json` `pick/scores` holds three candidates. SigLIP 1 was picked over two others; Qwen3 was
  an information row. The disclosure also leaves out r28's settings check on the same development seed.
- *Fix:* "…SigLIP 1 was picked over two other encoders (SigLIP 2, CLIP L/14) by 0.0007 (r28, a close call). r28 also
  tried four settings rows on SigLIP 1 seed 0 validation (k11 = 10, 20, 80; learning rate 0.0005), none adopted. This
  loop adds two caption forms (seed 0) and one pass decision."

**S11. §3.2 line 118: "Everything else is c*'s" is wrong for F1.**
- *Fix:* "F1: r13's final photo settings and r14's photo protocol. Seed s starts from K*'s seed-s emotion checkpoint,
  with training seed 100 + s (r14 `photo/siglip1/s0/run.json`: `cstar_seed` 0, `train_seed` 100). Only the caption
  input changes (D16)."

**S12. §6 lines 227 to 229: the SC2 sentences use the LEVEL word with no interval.**
- *What is wrong:* "level with" is the LEVEL outcome word (glossary since r23 D14), and LEVEL needs a paired interval.
  SC2 has no interval.
- *Fix:*
  - Level 1: "On validation, the SigLIP-only model's image and text group means are within one seed range of the
    current SigLIP 1 model's."
  - Level 2: "…and its text group mean is at or above the current model's."
  - Level 3: "On validation, both group means are above the current model's best seed."

**S13. §3.6, lines 169 to 171: two cheap diagnostics are missing.**
- *What is wrong:* the paper reader will ask what the switch costs on test, against K0 (C1: "the current best").
  Also, the pass rule compares K* trained on the 3090 with K0 trained on node402.
- *Fix:* add "(d) per test cell, K* minus K0 with its paired interval, deciding nothing" and "(e) the local minus
  node402 seed 0 change from gate (i)(a), beside the pass rule".

**S14. §4: verdict-shaping choices with no row (spec-templates.md, step 7).** Add rows for:
- the pass tolerance: the range of K0's per-seed group means, unrounded, with ≥ and >. Another reading, the mean of
  the per-cell ranges, gives 0.0078 for images and 0.0054 for text, 2.6 times wider. The definition must be written.
- the gate tolerances (S2);
- r, the pick rule and the comparator sources per cell type (M2);
- P1's caption form (S7);
- the crash rule (S5);
- inconclusive-cell counting (S6);
- stage 2's order (line 165).

**S15. §6 line 235: test numbers and the model pick.**
- *Fix:* add "K*'s place (main model or extra row) is set by SC2 alone. If it changes after any test number exists,
  the report says the choice was made after reading test data (constitution row 2)." D9 lets the paper chat place
  rows. This line keeps that placement from becoming a choice made on test data.

**S16. Header line 3 and §4 "When" column: the times are later than the file.**
- *What is wrong:* the file was last written at 21:40:22 and this review started at 21:51. The header says 22:10, D3
  says 21:50, D4 to D9 say 21:58 and D10 to D16 say 22:10.
- *Fix:* replace them with the real grill and writing times (`timestamps.md`). The "When" column is the record that
  the rule came before the results.

## Nits

- N1. D3, line 185: "Decided by" takes one of three values (hard rule 4). Write "user" and move "own pick, option c with
  the encoder extension" to Why.
- N2. Brief, line 39: "level 2 adds a text group at or above 0.2594". The body says ≥ 0.25935. Write "at or above the
  current model's text group (0.2594, rounded)".
- N3. §6 lines 231 to 232: replace "in most cells" with the count, "in N of 15 cells" (and of 6, of 4).
- N4. §6: add the one-sentence licensed claims that the parts table requires for SC1 (sweep), SC7 (ablation) and SC8
  (descriptive). Example for SC8: "On one RTX 3090, the SigLIP 1 text pass takes x ms per caption against 3.616 ms for
  the Qwen3 pass (r28 protocol)."
- N5. §3.2 line 115: the arm IDs A1 to A4 clash with the amendment IDs (A1, …) of §4. Rename them, for example to R1 to
  R4.
- N6. §9 line 279: F1's time is known: r14's SigLIP 1 photo run took 187 s per seed on node402 (`run.json`
  `seconds`). If K2 wins, the budget also needs two prefixed extractions (SigLIP 2, CLIP L/14) and a truncation count
  at CLIP's 77-token limit.
- N7. §3.5 row 1a: if one arm fails, the other becomes K* with no comparison. Say so, and record it in
  `prefix_pick.json`.
- N8. Brief: it runs to 54 lines against the template's "about 40". D6 (settings unchanged although k6 led on seed 0;
  user, after seeing the recommendation) is a key decision that "Check these" does not list (hard rule 4). Merge items
  2 and 4 to make room.
- N9. §9 lines 269 to 270: one blank line too many. Each stage 1 GPU job stays under 1 hour on weekdays (the 12 runs as
  separate jobs, not one queue job of about 72 minutes).

## Checks with no finding

- Brief and body agree on the stages, the comparators, the ladder counts (12 of 15, 5 of 6, 3 of 4), the cost and the
  stage types. The differences are only the rounding in N2 and the priors in S9.
- Held data: the prefix pick, the pass, the read-out picks, the seed ranges and stage 2's trigger all read validation
  only, once S0, S8 and S15 make the order explicit.
- Licensed sentences: none goes above its type's ceiling. SC1 and SC2 say "on validation". SC3 to SC6 stay within
  their cells and sets.
- Comparator data exist for a paired bootstrap in every SC3 cell: per-query files with `q_pid` for P1 and Q1 in every
  text direction (r23, 61,744 caption queries), r11 per-query `instructed` files for image mAP, and r17 cells for AMI.
  The MMArt and photo Q1 mAP per-unit files were not opened; M2 asks the spec to name them.
- §1 and §3.1 figures match `paper_cells.md`: text to image 0.264 / 0.179 / 0.489 against 0.268 / 0.157 / 0.443 and
  0.211 / 0.152 / 0.419; text to text emotion 0.206 and 0.253; 3.616 ms. D6's +0.0007 inside 0.0030 matches
  `settings_check.json` and r28's log.

## Recomputed numbers

Recomputed from r28's records with the CoSiR environment: image cells from `records/valpick.json`, `encoders/siglip1/cells/{ami,map}/{emotion,style,genre}/per_seed`, mean of six per seed; text cells from `records/text.json`, `ours_siglip1/picks/<aspect>/<t2i|i2t|t2t>/per_seed/r0`, mean of nine per seed.

| Group | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | Min | Max | Range | Spec |
|---|---|---|---|---|---|---|---|---|---|---|
| Image | 0.3848908314 | 0.3857966619 | 0.3859741878 | 0.3869271680 | 0.3878696916 | 0.3862917081 | 0.3848908314 | 0.3878696916 | 0.0029788602 | 0.38629; 0.38489 to 0.38787; 0.00298: match |
| Text | 0.2595700735 | 0.2587331093 | 0.2606346548 | 0.2591946633 | 0.2586239413 | 0.2593512884 | 0.2586239413 | 0.2606346548 | 0.0020107136 | 0.25935; 0.25862 to 0.26063; 0.00201: match |

| Bound | Unrounded | From the rounded spec numbers | Difference |
|---|---|---|---|
| SC2 level 1, image | 0.3833128479 | 0.38331 | 2.85e-6 |
| SC2 level 1, text | 0.2573405749 | 0.25734 | 5.7e-7 |
| SC2 level 2, text | 0.2593512884 | 0.25935 | 1.3e-6 |
| SC2 level 3, image / text | 0.3878696916 / 0.2606346548 | 0.38787 / 0.26063 | 3e-7 / 4.7e-6 |

For reference (not in the rule):
- the image mean equals `encoders/siglip1/score` (0.38629170813) exactly;
- the mean of the per-cell seed ranges is 0.00779 for images and 0.00542 for text (r0);
- K0's per-cell validation mAP ranges at R0, the image-mAP LEVEL units under r14, are 0.00216 / 0.00468 / 0.00410;
- r17's class-count AMI unit for emotion is 0.0037954, equal to `valpick`'s `ami/emotion` range.
