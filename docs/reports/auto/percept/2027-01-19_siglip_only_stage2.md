# SigLIP-only stage 2 reaches level 3 on both encoders, MMArt and photos

> 2027-01-19 (sequence date, written 2026-11-14) · Run: part a, 2026-11-13 19:00 to 2026-11-14 00:22 · part b, 00:23 to 01:13 · scoring/ladders through 01:48 · Spec: [design](../../../superpowers/specs/2026-11-12-siglip-only-adapter-design.md) · Decision rule: spec §6 with §3.4 and §3.5, amended by A1 and A2 @ fa032409 (record, spec d018f926) · [Run log](../../../../src/test/20261112_siglip_only/20261112_siglip_only_log.md)

## Results brief

**What we tried.** We tested the SigLIP-only adapter on SigLIP 2 and CLIP L/14, MMArt paintings outside ArtELingo, and Affection and EmoSet photos.
We removed four training parts one at a time. We also picked among the three encoders on validation.
Stage 1 failed its text validation rule (SC2). After seeing those results, the user allowed stage 2 anyway (amendment A1). The fail keeps the model an extra ablation or supplementary row.

**The level reached.** Each test has its own verdict.

|Claim and scope|Level|WIN over plain embedding|Against instructed Qwen3|
|---|---|---|---|
|Other encoders (SC4), SigLIP 2|3|6 of 6|5 WIN, emotion retrieval LEVEL|
|Other encoders (SC4), CLIP L/14|3|6 of 6|5 WIN, emotion retrieval LEVEL|
|Separate paintings (SC5), MMArt|3|6 of 6|5 WIN, emotion retrieval LEVEL|
|Photo model (SC6)|3|4 of 4|4 WIN|

Each painting row allows us to write: “Training adds over the plain embedding in every cell.” Against the instructed Qwen3, emotion retrieval is “level with” and the other five cells are “above”.
The photo row allows us to write: “The photo model is above the instructed Qwen3 in every cell.”
The ablation (SC7) has no claim level: it attributes changes, with intervals. Style AMI falls most without instructed graphs (-0.215) or graph-positive loss (-0.169). Only removing the relational term lowers genre clearly: AMI -0.060, mAP -0.114.
Encoder pick (SC9): “Among the SigLIP-only models, SigLIP 1 did best on the validation split (0.3882 against 0.3838 and 0.3762).”
SigLIP 1 leads SigLIP 2 by 0.0045 in all five seeds. This is a close call because its seed range is 0.0055.

**What it does not show.** These image tests use fixed settings and five seeds. They do not test the new encoders' text retrieval.
CLIP L/14 emotion retrieval is lower than instructed Qwen3: [-0.0027, -0.0004], within the allowed margin r = 0.0041.
LEVEL is not equality: the interval's lower end only exceeds minus r, the new model's own validation seed range. (EmoSet retrieval uses the current photo model's range instead; every EmoSet cell is a WIN, so it does not matter.)
The current same-encoder models are descriptive comparisons only. There is no correction for many comparisons.

**Options for next.**

a. Write the supplementary rows. Cost: paper work, no runs. Records the image claims beside the text limit.

b. r30, the planned check from saved outputs: a shared map of the emotion groups. Cost: bounded CPU work. Describes what those groups separate.

c. Caption-form and prefix searches per encoder, after the deadline. Cost: validation training and scoring. Could locate the text loss.

**Your reading.** Left for the user.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The image claims extend to these encoders and datasets, with a clear margin over the plain embedding. We are fairly sure of it: C13 re-derived every number. The earlier text fail keeps the model an extra row.

**Choices that could change a verdict.**

- The ablation (SC7) exists only through the gate (vii) recheck. Without it, the stale gate record blocks scoring and SC7 is inconclusive.
- If SigLIP 2's void start-up failure counts as a second crash, SigLIP 2 has no model: no encoder pick (SC9), and neither SC4 claim runs.
- With independent seed draws, the small genre mAP gain without content correction crosses zero (recomputed). SC7 has no level.

The other "yes" lines are guards and gates that all passed, or rules the spec names. None moves a level (Claim review).

**Our recommendation.** Write the decided extra rows before the paper deadline. Then do the saved-output emotion-group check (r30). Defer new training.

</details>

## What ran

All pieces completed on the local RTX 3090. The caption tower reads its encoder's text embedding, so texts need no Qwen3 pass at test time.
Qwen3 still supplies graphs and targets during training. The current model uses SigLIP 1 for images and Qwen3 for captions.
The paper is due 2026-11-16.

- **Types:** stage 2a: development sweep, training and validation picks. Stage 2b: claim tests and ablation. The caption timing test (SC8) ran in stage 1 and is not repeated here.
- **Seeds:** five, 0 to 4. Photos fine-tune matching emotion checkpoints on Affection.

|Data|Validation|Test|
|---|---|---|
|ArtELingo|6,152|12,281, genre queries 9,949|
|MMArt|5,769 ArtELingo validation paintings that carry MMArt labels (r17)|16,384 outside ArtELingo|
|Affection|4,251|8,501|
|EmoSet|5,770|17,348|

MMArt labeled emotion/style/genre counts: 16,218/16,384/13,420.

- **Code:** build 1427d43e, start-up fix e2641622, independent precision fix 3d081848, recheck 10205d96. Picks and ranges were committed before test reads.
- **Independent re-derivation (C13):** `s2_compare.json`: all pieces pass at 1e-6. Largest validation/test differences: 1.6e-8/2.8e-14. No inconclusive cells.

## Results

Every stage 2 claim test reached its top level. Stage 1's text fail (SC2) still decides the paper placement: an extra row.

|ID|Measured|Number|Compared against|Levels 1/2/3|Reached|
|---|---|---|---|---|---|
|SC4, SigLIP 2|image cells|6 P WIN, 6 Q WIN/LEVEL|plain encoder (P), instructed Qwen3 (Q)|≥5 P WIN/all 6/L2 and ≥5 Q WIN/LEVEL|3|
|SC4, CLIP L/14|image cells|6 P WIN, 6 Q WIN/LEVEL|same|same|3|
|SC5, MMArt|image cells|6 P WIN, 6 Q WIN/LEVEL|plain SigLIP 1, Q|same|3|
|SC6, photos|emotion cells|4 P WIN, 4 Q WIN|plain SigLIP 1, Q|≥3 P WIN/all 4/L2 and ≥3 Q WIN/LEVEL|3|
|SC7|removals|24 changes|full model|attribution with intervals/none/none|no level|
|SC9|validation image mean|0.3882/0.3838/0.3762|SigLIP 1/SigLIP 2/CLIP L/14|The highest validation mean wins. An exact tie goes to SigLIP 1. No ladder applies.|SigLIP 1, close|

The sentences allowed by the results are in the brief. The validation encoder pick (SC9) only selects among SigLIP-only models. It does not replace the current model.

Cells: five-seed means, P = plain, Q = instructed Qwen3, ordered P / Q. W = WIN, L = LEVEL. Shared values appear once.
AMI: clustering agreement. mAP: mean average precision. Image mAP uses the joined vector against P, aspect head against Q.
Photos use the emotion head throughout. WIN requires the interval's lower end above zero.
Otherwise, LEVEL requires that end above minus r, normally the candidate's own validation seed range. The remaining outcomes are BELOW.
EmoSet mAP uses r14's committed Qwen3 photo-model range instead of the new photo model's range (spec D19).
This exception does not affect the outcomes because every EmoSet cell is WIN.
Paired percentile 95% intervals: 2,000 row/seed draws for AMI, 10,000 query/seed draws for mAP.

### Other encoders (SC4)

Both reach level 3. Current columns: `paper_cells.md`, checked from diagnostic (d), descriptive only.
E2 (SigLIP 2) and E3 (CLIP L/14) use the caption form picked for SigLIP 1, the prefixed caption. Neither had its own caption-form pick. Sources: `image_cells_{siglip2,clipl14}_prefix.json`, SC4 ladders.

**SigLIP 2**

|Cell|Current (descriptive)|New P/Q|Comparator P/Q|d [lo, hi], P/Q|r|Outcome P/Q|
|---|---|---|---|---|---|---|
|emotion AMI|0.098|0.101|0.054/0.089|+0.0472 [+0.0429, +0.0512]/+0.0121 [+0.0085, +0.0156]|0.0111|W/W|
|style AMI|0.439|0.439|0.323/0.406|+0.1153 [+0.1095, +0.1194]/+0.0324 [+0.0277, +0.0361]|0.0187|W/W|
|genre AMI|0.527|0.527|0.355/0.501|+0.1717 [+0.1620, +0.1816]/+0.0251 [+0.0168, +0.0337]|0.0181|W/W|
|emotion mAP|0.269|0.261/0.269|0.244/0.268|+0.0169 [+0.0162, +0.0176]/+0.0011 [-0.0001, +0.0021]|0.0026|W/L|
|style mAP|0.348|0.283/0.348|0.258/0.322|+0.0254 [+0.0239, +0.0269]/+0.0260 [+0.0244, +0.0276]|0.0036|W/W|
|genre mAP|0.623|0.549/0.622|0.497/0.612|+0.0521 [+0.0496, +0.0547]/+0.0096 [+0.0079, +0.0113]|0.0024|W/W|

**CLIP L/14**

|Cell|Current (descriptive)|New P/Q|Comparator P/Q|d [lo, hi], P/Q|r|Outcome P/Q|
|---|---|---|---|---|---|---|
|emotion AMI|0.098|0.096|0.063/0.089|+0.0330 [+0.0288, +0.0374]/+0.0067 [+0.0027, +0.0109]|0.0089|W/W|
|style AMI|0.422|0.422|0.363/0.406|+0.0593 [+0.0543, +0.0641]/+0.0157 [+0.0109, +0.0203]|0.0112|W/W|
|genre AMI|0.516|0.516|0.397/0.501|+0.1188 [+0.1099, +0.1277]/+0.0141 [+0.0059, +0.0220]|0.0262|W/W|
|emotion mAP|0.267|0.260/0.267|0.248/0.268|+0.0117 [+0.0111, +0.0124]/-0.0016 [-0.0027, -0.0004]|0.0041|W/L|
|style mAP|0.335|0.276/0.336|0.273/0.322|+0.0025 [+0.0013, +0.0037]/+0.0143 [+0.0127, +0.0160]|0.0029|W/W|
|genre mAP|0.618|0.544/0.618|0.498/0.612|+0.0463 [+0.0440, +0.0485]/+0.0051 [+0.0033, +0.0069]|0.0012|W/W|

### MMArt (SC5)

The model reaches level 3 on MMArt. Emotion retrieval is level with instructed Qwen3, and the other cells are above it.
Source: `mmart_cells_siglip1_prefix.json`, SC5 ladder.

|Cell|New P/Q|Comparator P/Q|d [lo, hi], P/Q|r|Outcome P/Q|
|---|---|---|---|---|---|
|emotion AMI|0.055|0.035/0.049|+0.0201 [+0.0174, +0.0229]/+0.0055 [+0.0032, +0.0079]|0.0061|W/W|
|style AMI|0.458|0.391/0.416|+0.0664 [+0.0613, +0.0712]/+0.0420 [+0.0372, +0.0470]|0.0127|W/W|
|genre AMI|0.535|0.410/0.526|+0.1250 [+0.1167, +0.1327]/+0.0090 [+0.0015, +0.0154]|0.0212|W/W|
|emotion mAP|0.197/0.202|0.191/0.202|+0.0059 [+0.0056, +0.0062]/+0.0001 [-0.0004, +0.0005]|0.0004|W/L|
|style mAP|0.307/0.383|0.303/0.348|+0.0036 [+0.0020, +0.0051]/+0.0351 [+0.0333, +0.0370]|0.0039|W/W|
|genre mAP|0.538/0.628|0.493/0.617|+0.0446 [+0.0429, +0.0465]/+0.0113 [+0.0098, +0.0127]|0.0038|W/W|

### Photos (SC6)

The photo model reaches level 3 and is above instructed Qwen3 in every cell.
Source: `photo_cells_siglip1_prefix.json`, SC6 ladder.

|Cell|New P/Q|Comparator P/Q|d [lo, hi], P/Q|r|Outcome P/Q|
|---|---|---|---|---|---|
|affection AMI|0.183|0.099/0.155|+0.0843 [+0.0786, +0.0901]/+0.0277 [+0.0216, +0.0332]|0.0076|W/W|
|affection mAP|0.325|0.255/0.295|+0.0705 [+0.0673, +0.0733]/+0.0305 [+0.0275, +0.0331]|0.0058|W/W|
|emoset AMI|0.432|0.267/0.394|+0.1652 [+0.1530, +0.1796]/+0.0382 [+0.0265, +0.0528]|0.0309|W/W|
|emoset mAP|0.487|0.364/0.452|+0.1234 [+0.1193, +0.1275]/+0.0354 [+0.0313, +0.0393]|0.0069|W/W|

### Ablation (SC7)

The removals give attribution with intervals, without a claim level. The table below gives every change from the full model.
The licensed sentence is: “On ArtELingo test, removing `<part>` changes AMI by x and mAP by y (interval), as strong as the result it explains.”
No r or outcome labels apply. mAP uses plain cosine. Blank part cells repeat the part above.

|Removed part|Aspect|AMI change [interval]|mAP change [interval]|
|---|---|---|---|
|instructed graphs|emotion|-0.0357 [-0.0406, -0.0308]|-0.0199 [-0.0214, -0.0184]|
||style|-0.2147 [-0.2181, -0.2069]|-0.1416 [-0.1445, -0.1385]|
||genre|+0.0042 [-0.0080, +0.0155]|-0.0010 [-0.0026, +0.0003]|
|content correction|emotion|-0.0128 [-0.0160, -0.0094]|-0.0057 [-0.0066, -0.0043]|
||style|-0.0876 [-0.0913, -0.0822]|-0.0674 [-0.0693, -0.0655]|
||genre|+0.0108 [-0.0003, +0.0240]|+0.0003 [+0.0001, +0.0006]|
|graph-positive loss (buddy)|emotion|-0.0359 [-0.0398, -0.0319]|-0.0180 [-0.0189, -0.0171]|
||style|-0.1695 [-0.1736, -0.1620]|-0.1167 [-0.1193, -0.1141]|
||genre|+0.0049 [-0.0022, +0.0125]|-0.0007 [-0.0018, +0.0004]|
|relational term|emotion|+0.0001 [-0.0027, +0.0031]|-0.0001 [-0.0004, +0.0003]|
||style|-0.0016 [-0.0059, +0.0030]|-0.0004 [-0.0017, +0.0013]|
||genre|-0.0597 [-0.0753, -0.0469]|-0.1144 [-0.1190, -0.1104]|

### Encoder pick (SC9)

SigLIP 1 wins the validation pick by a margin smaller than its seed range: five-seed means 0.3882 (SigLIP 1), 0.3838 (SigLIP 2)
and 0.3762 (CLIP L/14); margin 0.0045; SigLIP 1's seed range 0.0055. Source: `encoder_pick.json`, checked against
`image_groups_*.json` and stage 1's groups.
Qwen3 is information only. SigLIP 1 leads at every matched seed.

## Agent choices checked

Ranked [decisions](../../../../src/test/20261112_siglip_only/decisions.md): certification, then interval pairing.

|Rank|Choice|Alternative checked and effect|
|---|---|---|
|1|Gate (vii) recheck, 2026-11-14 01:23|Without the recheck, the scorer refuses the try-3 gate record, whose code hashes predate the A1 edits, and the gate cannot run a fourth try: SC7 inconclusive. The recheck ran the gate's own checks (trainer test, frozen K0 replay) on the current code and passed at try 1.|
|2|Void crash, 2026-11-13 19:08|If the void start-up counts as a second crash, SigLIP 2 has no trained model. The encoder pick (SC9) does not run. Part b runs neither encoder's test cells, so both SC4 claims are not run. Saved scores cannot override that rule.|
|3|Shared seeds, 2026-11-13 00:41|Recomputed genre mAP without content correction, independent seeds: +0.0003 [-0.0011, +0.0017], shared: +0.0003 [+0.0001, +0.0006]. Positive attribution disappears, no level.|
|4|Float64 fix, 2026-11-13 19:24|With Codex's float32 cast kept, one MMArt replay entry misses its 1e-9 target (1.03e-9) and the six MMArt mAP cells would park. Without the cast (r11's float64 convention), the full replay passes; tolerance unchanged.|
|5|Ranges/picks, 2026-11-12 23:04, 2026-11-13 02:50, 02:30|The choice uses each candidate's validation range and the frozen read-out picks. Using the AMI range instead for MMArt emotion retrieval still gives LEVEL (recomputed). The claim level stays the same. Every EmoSet cell is WIN, so r does not affect its outcome. Different read-outs need new paired scores, which were not recomputed. Under the current model's own range instead (CLIP L/14 emotion mAP 0.0017, r28 `valpick.json`), CLIP L/14's emotion retrieval against Q would be BELOW; SC4 stays level 3 with 5 Q WIN, and its sentence would say 5 of 6 cells. SigLIP 2's cell stays LEVEL (range 0.0015); MMArt not checked (recomputed in the claim review).|
|6|Caption form, 2026-11-13 01:58, photo means, 00:10|The choice reuses SigLIP 1's caption form and includes only eligible photo explanations. Alternatives pick each encoder's caption form separately or include ignored explanations. These require new models. Saved scores cannot reconstruct them, so their effects on claims were not recomputed.|

Rank 3 began as Codex's exploration. Claude recomputed it in the claim review from the cached per-query APs of this round's exports
(`rederive/r29rd_s2abl_cache/`, five seeds; script and output in `claim_review/rank3_seed_pairing.*`): query draws shared, seed draws
shared or independent, 10,000 draws, numpy seed 0.

Other yes/maybe lines: input normalisation and caption-only swap define the model. Graph swaps need retraining.
Guards passed, and the trainer fallback was unused. Test exports have no same-device rule (decision line 2026-11-12 23:41); here every model trained and exported on the same local RTX 3090.
Parking, photo variants and partial-arm rules are inactive: gates pass, removals complete.
No encoder truncation. Text-form/tolerance choices concern stage 1.

[Handoff §3](../../../superpowers/handoffs/2026-11-13-r31-run-2-handoff.md#3-where-things-stand), Codex choices checked:
matched seeds ahead recompute to all seeds, tie band preserves the pick, runner-up range also gives a close call.
MMArt uses the specified joined vector against plain. Aspect-head alternative needs different scores, not recomputed.
EmoSet's inherited unrounded unit does not affect WIN cells. Ablation pairing is rank 3.
All P cells WIN and five Q cells WIN per painting claim, so changing only LEVEL units cannot move these levels.

## Analysis

**Our reading:** image changes from current models are small. Diagnostic (d), spec §3.6: every interval crosses zero.
SigLIP 2 spans -0.0007 to +0.0027, CLIP L/14 -0.0026 to +0.0011.
The new models trained on the local GPU. The current models trained on DAS6 node402.
In r14, the machine change alone moved single cells by up to 0.0024, as recorded in the spec's machine decision (D25).
These differences therefore show neither equality nor a cost of the caption-input switch.

Graph-positive loss pulls neighbours together. Content correction reduces shared graph content.
Their removal, or uninstructed graphs, lowers emotion/style scores. The relational term matches Qwen3's image-similarity pattern.
Its removal lowers genre scores, with negative intervals. Other genre AMI intervals cross zero. Interactions remain untested.
CLIP L/14 emotion retrieval is lower than instructed Qwen3: its interval [-0.0027, -0.0004] lies wholly below zero.
It is LEVEL only within the allowed margin r = 0.0041. SigLIP 2 and MMArt emotion retrieval are also “level with”, never “above”.
Photos are above instructed Qwen3 in all four cells. These tests concern image emotion, not text.

## Deviations

Amendments/fixes below. Other deviations: none. Seed sensitivity is exploration.

- **A1, A2:** after stage 1 results, the user allowed stage 2 on the fail and added validation encoder selection. Fail and placement stand. Record names spec d018f926, containing both.
- **Start-up:** both SigLIP 2 attempts failed the unset `CUBLAS_WORKSPACE_CONFIG` assertion before training. Encoder/photo entry points now set `:4096:8` when absent. False crash retained in `records/void/`, treated as wiring rather than a second training crash.
- **MMArt replay:** Codex's query-expansion float32 downcast was removed under the earlier float64 convention. Full replay passed, tolerance unchanged.
- **SC7:** scoring refused stale gate hashes after amendment edits. Separate recheck repeated trainer/frozen replay checks, preserving the original record, before scoring.
- **Truncation, as specified:** 4,045 of 526,749 photo explanations cut at 64 tokens and kept. Both encoder stores: zero truncated captions.
- **C13 caveats:** genre mAP row count differs from Codex's query count as information only, `n_queries` agrees. Codex's reference lacks earlier AMI seed targets, means pass. Stage 2 comparison passes.

## Claim review

Claude (chat "r31 run 5"), 2026-11-14 02:10 to 02:40, from the record files, not from this report's tables:

| Brief value | Recomputed | File |
|---|---|---|
| SC9: 0.3882 / 0.3838 / 0.3762; margin 0.0045; 5 of 5 seeds; seed range 0.0055 | means of the per-seed groups 0.38822 / 0.38376 / 0.37618 (SigLIP 2's also from its six cells); margin 0.00445; 5; 0.00546 | `groups_siglip1_prefix.json`, `image_groups_{siglip2,clipl14}_prefix.json` |
| Level table: P all WIN; Q 5 WIN + 1 LEVEL (SC4 twice, SC5), 4 WIN (SC6) | each outcome re-applied from lo and r (WIN lo > 0, LEVEL lo > −r): same counts, and every stored outcome agrees | `image_cells_*`, `mmart_cells_*`, `photo_cells_*` |
| CLIP L/14 emotion mAP against Q: [-0.0027, -0.0004], r 0.0041 | d -0.0016 [-0.0027, -0.0004], r 0.0041 | `image_cells_clipl14_prefix.json` |
| SC7: style AMI -0.215, -0.169; genre AMI -0.060, mAP -0.114 | from the per-seed arm values: -0.2147, -0.1695, -0.0597, -0.1144 | `ablation.json` (`arms`) |
| Folded block, third line: crosses zero with independent seeds | change +0.00031; shared [+0.00008, +0.00055], independent [-0.00114, +0.00170] | per-query APs, `claim_review/rank3_seed_pairing.*` |

**Decision lines.** 40 lines are marked yes or maybe. The three that can change a claim are in the folded block. The
rest cannot move a level here: the guards and gates all passed (no arm out, no parked or inconclusive cell), and the
LEVEL units are the spec's own. The one unit alternative that changes a cell (the current model's range: CLIP L/14's
emotion retrieval BELOW) keeps SC4 at level 3; it is in "Agent choices checked", row 5.

**Independent check.** C13 re-derived every number of the brief (`s2_compare.json`). The user-read report had a
fidelity check and a cold read (Codex, fresh sessions), then one fix wave.

## Files and storage

The output folder grew from 5.8 GB before stage 2 to 23 GB.
[Round folder](../../../../src/test/20261112_siglip_only/) holds records, the numbers sheet, decisions, code, logs and re-derivations.
Storage figures come from the [run handoff](../../../superpowers/handoffs/2026-11-14-r31-run-5-handoff.md).

| Location | Size |
|---|---|
| `/data/SSD2/cosir_r29/20261112_siglip_only/` | 23 GB |
| Its `ablation/` folder | 10 GB |
| Its `cstar/` folder | 7.7 GB |
| Its `mmart/` folder | 1.5 GB |
| `/data/SSD2/pre_extract/affection_local/siglip_so400m_txt/` | 1.2 GB |
| `/data/SSD2/pre_extract/artelingo_full/siglip2_so400m_capprefix/` | 0.7 GB |
| `/data/SSD2/pre_extract/artelingo_full/clip_l14_capprefix/` | 0.5 GB |
| Codex's untracked files in `rederive/` | about 114 MB |

The untracked re-derivation caches, probe scripts and notes stay out of git. Nothing was deleted, and no removal is proposed here.
The next decide chat's storage summary decides what to keep or remove.
