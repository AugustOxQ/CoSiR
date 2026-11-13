# SigLIP-only adapter: the text group fails the validation pass rule; on test it beats plain SigLIP 1 in 13 of 15 cells (level 1)

> 2027-01-18 (sequence date; written 2026-11-13) · Run: 2026-11-13, 04:05 to 06:34 · Spec: [r29 design](../../../superpowers/specs/2026-11-12-siglip-only-adapter-design.md) · Decision rule: spec §6 with §3.4 and §3.5 @ 7806b241 · [Run log](../../../../src/test/20261112_siglip_only/20261112_siglip_only_log.md)

## Results brief

**What we tried.** We trained the aspect adapter of c* on SigLIP 1 alone ([spec](../../../superpowers/specs/2026-11-12-siglip-only-adapter-design.md)).
The caption tower reads the captions' own SigLIP 1 text embedding, so a text needs no Qwen3 pass at test time. The
graphs, the content correction and the alignment targets stay on Qwen3. We compared the new model with the current
SigLIP 1 model on validation, then scored the paper cells on ArtELingo test.

**The level reached.**

- **Caption form** (validation, seed 0): the prefix "Represent the caption: " wins, 0.25528 against 0.25527. This is a
  close call (0.0000167).
- **Pass rule** (validation, five seeds): level 0, fail. The image group holds and the text group falls below its
  bound. So the new model is an extra row, and stage 2 did not run.

| Five-seed mean on validation | New model | Current SigLIP 1 model | Pass bound |
|---|---|---|---|
| Image group (AMI and mAP, 3 aspects) | 0.3882 | 0.3863 | 0.3833 |
| Text group (3 directions × 3 aspects) | 0.2549 | 0.2594 | 0.2573 |

- **Paper cells** (ArtELingo test, 15 cells): level 1, "With no Qwen3 pass at test time, training adds over the plain
  embedding in 13 of 15 cells."
  - Against plain SigLIP 1: 13 WIN. Style image mAP is BELOW, 0.287 against 0.296; the current model's cell is BELOW
    too. Genre text to image is inconclusive: the independent re-derivation differs by 1.02e-6, over the 1e-6 tolerance.
  - Against the instructed Qwen3: 12 WIN. The three emotion text cells are BELOW, each interval wholly below zero
    (text to text 0.327 against 0.377).
- **Text time:** the SigLIP 1 text pass takes 1.138 ms per caption against 3.616 ms for the Qwen3 pass.

**What it does not show.** ArtELingo only: MMArt, the photo model, the other encoders and the ablation did not run. The
LEVEL unit is one seed range. The new model trained on the local GPU and the current model on DAS6. At seed 0 that
machine change moves the image group by +0.0025 and the text group by -0.0001. Intervals are per cell, with no
correction for the 30 comparisons. Emotion text is the cost of the switch: on test, emotion text to text is 0.019
below the current model.

**Options for next.**

a. Keep the current model as the main model. Put the SigLIP-only model in the paper as an extra row (Tables 2, 3 and
   8). Cost: no runs; the paper chat places the cells. It gives the light-model sentence with its emotion text cost.
b. Run part of stage 2 for the extra row (MMArt, the photo model, the ablation). Cost: about 6 GPU-hours. It tells if
   the extra row holds outside ArtELingo.
c. A new loop to close the emotion text gap with no Qwen3 pass at test time, for example a caption-tower term that
   also pulls toward the Qwen3 caption embedding. Cost: a medium loop, about 1 to 2 GPU-hours on validation. It tells
   if the loss comes from the SigLIP 1 text input or from the training.

**Your reading.** Left for the user.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The image side holds and the text side does not. The text fail looks real: all five seeds fall below
the bound. At seed 0 the machine change moved the text group by -0.0001, against a gap of -0.0045; other seeds were
not measured on both machines. The loss sits in emotion (validation text to text -0.0245). Training still adds much
over plain SigLIP 1 (test emotion text to text +0.121). The one BELOW over plain SigLIP 1 depends on the read-out:
with the new model's own style head the cell is a clear WIN. We are fairly sure of this on ArtELingo; other data did
not run.

**Choices that could change a verdict.**

- The inconclusive genre text-to-image cell: both sides find a WIN (d +0.066). As a WIN, 14 of 15 over plain SigLIP
  1: the same level 1 (recomputed).
- Image mAP against plain SigLIP 1 reads the joined three-head vector (the comparator protocol, spec D18, as the
  current model's cell in r17). Under the new model's own aspect head all three cells are WIN (style +0.0588 [+0.0565,
  +0.0611]): level 1, inconclusive above. With the genre cell also a WIN it becomes level 3 (15 of 15; Qwen3 12 of 15).
  The joined vector is the like-for-like form for one plain embedding.
- The close caption-form pick (spec D5): the plain form has seed 0 only, so the pass rule cannot be recomputed under
  it. Its seed 0 text group (0.2553) is below the text bound (0.2573) too.

The decision lines marked yes are build guards, the caption-input design (D2), the LEVEL unit and stage 2 settings
that did not run; none changes a level here ("Agent choices checked").

**Our recommendation.** Option a: the pass rule makes the new model an extra row, and the paper is due 11-16. In the
paper chat, decide how Table 2 shows image mAP against plain SigLIP 1 (joined vector or own head). Park option c for
after 11-16. Option b only if the paper needs the extra row on MMArt or photos.

</details>

## What ran

Stages 0 and 1 ran as specified; stage 2 did not run, because the pass rule failed (D3).

- **Stages and types:** stage 0, regression gates (all pass, build chats). Stage 1a, sweep: caption form, plain (K1)
  against prefixed (K2), validation seed 0. Stage 1b, incremental: the pick (K*) on five seeds against the current
  model (K0), validation. Stage 1c, claim test: 15 ArtELingo test cells (runs on a pass and on a fail, D9). Table 8,
  descriptive timing. Stage 2 (E2, E3, MMArt, photo model F1, ablation R1 to R4): not run.
- **Data and seeds:** validation 6,152 paintings and 30,872 captions for every pick, range and the pass rule; test
  12,281 paintings and 61,744 captions (genre 9,949 paintings); seeds 0 to 4.
- **Code:** built at 007fb786 (15 tickets); run commits 31507322 to d36e7047. [Round folder](../../../../src/test/20261112_siglip_only/),
  local RTX 3090. K0 is r14's c* on SigLIP 1 (node402, frozen). Codex re-derived SC1, SC2 and every pick, range and
  cell of 1c (C13).

## Results

| ID | Measured | Number | Compared against | Levels 1 / 2 / 3 | Level reached |
|---|---|---|---|---|---|
| SC1 | text group, validation, seed 0 | prefixed 0.2552827 | plain 0.2552660 | pick the higher | prefixed (K2) by 0.0000167 |
| SC2 | image and text group, five-seed means, validation | image 0.3882, text 0.2549 | K0 0.3863, 0.2594 | image ≥ 0.3833 and text ≥ 0.2573 / L1 and text ≥ 0.2594 / image > 0.3879 and text > 0.2606 | 0 (fail on text) |
| SC3 | 15 cells, ArtELingo test | P1: 13 WIN, 1 BELOW, 1 inconclusive; Q1: 12 WIN, 3 BELOW | plain SigLIP 1 (P1); instructed Qwen3 (Q1) | P1 WIN ≥ 12 / P1 WIN in 15 / L2 and Q1 WIN or LEVEL ≥ 12 | 1 |
| SC4 to SC7 | E2, E3, MMArt, photo model, ablation | not run | – | – | not run (D3) |
| SC8 | text pass, ms per caption | SigLIP 1 (prefixed) 1.138; caption towers 0.017 | Qwen3 pass 3.616 (r28) | measured | descriptive |

Per-seed text groups of K*: 0.2539 to 0.2558, all below the bound 0.2573. Per-seed image groups: 0.3851 to 0.3905.

**SC3 cells** (five-seed means; d is the paired difference with its 95% interval; source
[`paper_cells.md`](../../../../src/test/20261112_siglip_only/paper_cells.md)). Image mAP against P1 reads K*'s joined
three-head vector (0.261 / 0.287 / 0.549); the value column shows the aspect head. Intervals are per cell, with no
multiplicity correction (spec §3.4).

| Cell | K0 | K* | d against P1 | P1 | d against Q1 | Q1 |
|---|---|---|---|---|---|---|
| emotion AMI | 0.099 | 0.102 | +0.0406 [+0.0363, +0.0446] | WIN | +0.0129 [+0.0093, +0.0165] | WIN |
| emotion mAP | 0.269 | 0.270 | +0.0157 [+0.0151, +0.0164] | WIN | +0.0012 [+0.0001, +0.0022] | WIN |
| emotion t2i | 0.264 | 0.262 | +0.0508 [+0.0497, +0.0518] | WIN | -0.0060 [-0.0070, -0.0051] | BELOW |
| emotion i2t | 0.286 | 0.281 | +0.0603 [+0.0583, +0.0624] | WIN | -0.0229 [-0.0253, -0.0207] | BELOW |
| emotion t2t | 0.346 | 0.327 | +0.1211 [+0.1193, +0.1230] | WIN | -0.0493 [-0.0509, -0.0477] | BELOW |
| style AMI | 0.441 | 0.446 | +0.0644 [+0.0594, +0.0692] | WIN | +0.0401 [+0.0356, +0.0441] | WIN |
| style mAP | 0.355 | 0.355 | -0.0090 [-0.0107, -0.0074] | BELOW | +0.0333 [+0.0315, +0.0353] | WIN |
| style t2i | 0.179 | 0.175 | +0.0237 [+0.0222, +0.0253] | WIN | +0.0189 [+0.0172, +0.0205] | WIN |
| style i2t | 0.141 | 0.137 | +0.0295 [+0.0284, +0.0305] | WIN | +0.0127 [+0.0117, +0.0135] | WIN |
| style t2t | 0.109 | 0.107 | +0.0147 [+0.0139, +0.0156] | WIN | +0.0089 [+0.0081, +0.0097] | WIN |
| genre AMI | 0.528 | 0.517 | +0.1155 [+0.1046, +0.1261] | WIN | +0.0157 [+0.0061, +0.0255] | WIN |
| genre mAP | 0.623 | 0.623 | +0.0457 [+0.0431, +0.0482] | WIN | +0.0105 [+0.0086, +0.0122] | WIN |
| genre t2i | 0.489 | 0.485 | +0.0662 [+0.0641, +0.0683] | inconclusive (C13) | +0.0417 [+0.0393, +0.0442] | WIN |
| genre i2t | 0.407 | 0.404 | +0.0502 [+0.0483, +0.0521] | WIN | +0.0077 [+0.0062, +0.0092] | WIN |
| genre t2t | 0.316 | 0.317 | +0.0645 [+0.0622, +0.0667] | WIN | +0.0184 [+0.0162, +0.0206] | WIN |

**Level reached under the rule:** SC1 picks the prefixed form. SC2 is level 0, so K* is an extra row (D3). SC3 is
level 1: "With no Qwen3 pass at test time, training adds over the plain embedding in 13 of 15 cells." Level 2 fails
on style image mAP, so level 3 does not hold although Q1 has 12 of 15. SC8: "On one RTX 3090, the SigLIP 1 text pass
takes 1.138 ms per caption against 3.616 ms for the Qwen3 pass."

## Agent choices checked

No decision line can move SC1 or SC2. The top three choices (two lines and one spec protocol) were recomputed for SC3.

| Rank | Choice (decisions.md line) | Main alternative, recomputed | Level under it |
|---|---|---|---|
| 1 | Image mAP against P1 reads the joined three-head vector (spec D18, an agent default of the decide chat; not a run line) | K*'s own aspect head: emotion +0.0244, style +0.0588 [+0.0565, +0.0611], genre +0.1194, all WIN (`alt_told_vs_p1.json`, exploration) | level 1, inconclusive above; level 3 with the C13 cell as a WIN |
| 2 | The C13 cell is inconclusive, no side chosen (06:20, maybe) | accept the WIN of both sides: P1 WIN 14 of 15 (`outcome_before_c13` in the text cell record) | level 1 |
| 3 | LEVEL unit r = the candidate's own validation range (23:04, yes) | K0's range: style mAP 0.0047 against K*'s 0.0038; the cell's lower bound -0.0107 stays below -r. Every other image cell is WIN, where r does not count | level 1 |
| 4 | P1's form from `prefix_pick.json` (00:05, yes) | prefixed P1 has validation scores only (0.1794 against plain 0.1801); not recomputed on test. The smallest P1 text lower bound is +0.0139 | not checked on test |
| 5 | Codex's outcome field not used (06:20, no) | its own 1e-9 replay flag would mark 27 cells inconclusive; every replay is within the spec's 1e-6 | – |

The other lines marked yes or maybe do not touch a stage 1 number:

- The caption-input swap (23:08, yes) is the spec's design D2, decided by the user.
- The guards against stale or wrong inputs (23:29, 00:05 cells gate, 00:29 and 01:10 gate checks) refuse a run; every
  gate passed, so the alternative gives the same numbers. Gate (i)(a) passed bit-equal, so its fallback rules (00:29)
  did not apply.
- The inconclusive rule (00:27, yes): with the one P1 cell as a WIN, 14 of 15; no "inconclusive above".
- The F1, E2 and E3, MMArt, photo, ablation and held-export lines belong to stage 2, which did not run.

## Analysis

**The pass rule failed on text emotion; the image side held.** Validation changes K* minus K0 (diagnostic (a)):

| Validation, K* minus K0 | Emotion | Style | Genre |
|---|---|---|---|
| AMI | -0.0005 | +0.0042 | +0.0080 |
| mAP | +0.0002 | -0.0009 | +0.0004 |
| text to image | -0.0032 | -0.0050 | +0.0016 |
| image to text | -0.0051 | -0.0032 | -0.0010 |
| text to text | -0.0245 | -0.0022 | +0.0020 |

- **Machine (e):** local minus node402 at seed 0 (old caption input): image mean +0.0025 (genre AMI +0.0184), text mean
  -0.0001. Our reading: K*'s image group above K0's best seed may come from the machine. At seed 0 the machine moved
  the text group far less than the five-seed gap (-0.0045); the other seeds were not measured on both machines.
- **Caption tower over its input (c):** text to text, read-out r0, K* minus plain SigLIP 1: emotion +0.119, style
  +0.018, genre +0.057. The tower adds most where SigLIP 1's text is weakest.
- **Best epochs (b):** genre as K0's in every seed (14 to 18); style 1 to 3 for both. Emotion differs: K* 6 to 18
  against K0 10 to 18 (seed 0: 6 against 12; seed 2: 6 against 17).
- **Test, K* minus K0 (d):** text 8 of 9 cells lower (emotion t2t -0.0191 [-0.0213, -0.0169]; genre t2t +0.0011
  [-0.0025, +0.0044]). Image AMI: emotion +0.0033, style +0.0057, genre -0.0107 [-0.0209, -0.0013]. Image mAP within
  ±0.001 (style and genre intervals cover 0).

Our reading: SigLIP 1's text encoder carries less emotion than Qwen3's (plain text to text emotion mAP 0.206 against
0.253, r28). The caption tower recovers most of that gap, not all of it. K0 trails Q1 in the same three cells, by less:

| Emotion text, test mAP | P1 | K* | K0 | Q1 |
|---|---|---|---|---|
| text to image | 0.211 | 0.262 | 0.264 | 0.268 |
| image to text | 0.220 | 0.281 | 0.286 | 0.304 |
| text to text | 0.206 | 0.327 | 0.346 | 0.377 |

**K0's same cell is BELOW too.** K0's cell in r17 SC7 reads the same joined vector: 0.2877 against 0.2962, d -0.0085
[-0.0101, -0.0069]. The spec's prior (§3.1) assumed the current models win every cell over the plain embedding, and
named "a BELOW against plain SigLIP 1 in any cell" as a surprise. The surprise came. It depends on the read-out:
under K*'s own style head the cell is a WIN (+0.0588).

**Priors (§3.1) against outcomes:** pass 0.4, fail; prefix wins 0.4, won; SC3 level 1 0.8, level 1; level 2 0.6, not
reached.

## Deviations

One cell is inconclusive by C13; one analysis is exploration; nothing else differed from the spec in stage 1.

- **C13, one number over tolerance.** Genre text to image against P1, upper interval end: ours 0.0682642562, Codex
  0.0682652749 (1.02e-6). Cause (`c1_cause_check.json`): Codex's interval code on our per-query APs gives our
  interval within 6e-17. The two forward passes differ in about 1,150 of 61,744 queries per seed (near-tied captions
  change rank). No side chosen, tolerance not widened; the cell is inconclusive (§3.4).
- **Codex's outcome field** was not used: it flags replays over 1e-9, stricter than the spec's 1e-6. Its outcomes,
  recomputed from its lower bounds and r by §3.4, equal ours in all 30 cells.
- **Exploration:** the image mAP read-out against P1 with K*'s own aspect head (`r29_alt_told_vs_p1.py`). It decides
  nothing.
- **Amendment not used:** the user allowed daytime GPU for stage 2 (amends D7 for r29, 2026-11-13 02:19). Stage 2 did
  not run.
- **Build gates:** (i)(c) and (vii) ablation passed at try 3 of 3, (vii) trainer and (v) text at try 2 (build chats;
  decision lines). Diagnostic (d) covers SC3 only, as SC4 did not run.
- **No crash and no rerun** in stage 1.

## Claim review

Claude, 2026-11-13 06:35, from the record files (not from this report's tables):

| Brief value | Recomputed | File |
|---|---|---|
| Pass rule: image 0.3882, text 0.2549 | means of the five per-seed groups: 0.3882157, 0.2548566 | `pass_rule.json`, `per_seed` |
| Bounds 0.3833 and 0.2573; K0 means 0.3863 and 0.2594 | from K0's per-seed groups: m − r = 0.3833128 and 0.2573406; m 0.3862917 and 0.2593513 | `pass_rule.json`, `k0` |
| SC3: P1 13 WIN, 1 BELOW, 1 inconclusive; Q1 12 WIN, 3 BELOW | counted from the 30 cells; each outcome recomputed from lo and r by §3.4; the one mismatch is the C13 cell (WIN before C13) | `image_cells_…json`, `text_cells_…json` |
| Style image mAP against P1: 0.287 against 0.296 | 0.28718 against 0.29622, d -0.00904 [-0.01065, -0.00739] | `image_cells_…json` |
| Emotion t2t against Q1: 0.327 against 0.377 | 0.32718 against 0.37651, d -0.04933 | `text_cells_…json` |
| Text pass 1.138 ms per caption | mean of three repeats 1.13842 (1.1320, 1.1382, 1.1451) | `time_text.json` |

**Decision lines.** Every line marked "yes" is in the folded block's last paragraph or in "Agent choices checked"
with the reason it cannot change a level.

**Independent check.** Codex (gpt-6.1-terra, medium, a fresh session) checked every number and level against the
records: 3 findings, all fixed. (1) The best epochs of emotion differ from K0's (the run log said "as K0's"; now
corrected there too). (2) "The text fail does not come from the machine" was too strong: the machine change was
measured at seed 0 only. (3) "Inherited" implied a cause: now "K0's same cell is BELOW too". The user-read report's
fidelity check added one caveat here: intervals are per cell, with no multiplicity correction.

## Files and storage

- **Round folder:** [code, records, logs](../../../../src/test/20261112_siglip_only/): `records/` (gates, picks,
  `pass_rule.json`, cell files, `SC3_siglip1_prefix.json`, `c1_compare.json`, `c1_cause_check.json`, diagnostics,
  `time_text.json`), [`paper_cells.md`](../../../../src/test/20261112_siglip_only/paper_cells.md),
  [`decisions.md`](../../../../src/test/20261112_siglip_only/decisions.md), `rederive/` (Codex's code and outputs).
- **Untracked:** Codex's working notes `rederive/findings.md`, `progress.md`, `task_plan.md`,
  `r29rd_c1_replay_text_impl.txt`; they can stay out of git.
- **Outputs:** `/data/SSD2/cosir_r29/20261112_siglip_only/`, 52 GB (`du -sh`, 2026-11-13 06:33): models, exports,
  scores, logs.
- **Storage:** nothing deleted (night mode). On `.scratch/pending_deletions.md`: Codex's cache `rd_c1_cache` (38 GB),
  `gate_vi/try1/` (1.5 GB), ticket 12's test scratch (4.6 GB, system disk), the build's test and smoke folders and
  worktrees. The next wrap-up decides them with the user.
