# SigLIP-only stage 2 reaches level 3 on both encoders, MMArt and photos

> 2027-01-19 (sequence date, written 2026-11-14) · Run: part a, 2026-11-13 19:00 to 2026-11-14 00:22 · part b, 00:23 to 01:13 · scoring/ladders through 01:48 · Spec: [design](../../../superpowers/specs/2026-11-12-siglip-only-adapter-design.md) · Decision rule: spec §6 with §3.4 and §3.5, amended by A1 and A2 @ fa032409 (record, spec d018f926) · [Run log](../../../../src/test/20261112_siglip_only/20261112_siglip_only_log.md)

## Results brief

**What we tried.** Stage 2 tested encoders, datasets and removals. Captions use their encoder's text embedding, graphs and targets still use Qwen3.
The user approved A1 **after seeing stage 1 results**. The fail stands.

**The level reached.** Verdicts:

|Claim and scope|Level|WIN over plain embedding|Against instructed Qwen3|
|---|---|---|---|
|SC4, SigLIP 2, ArtELingo images|3|6 of 6|5 WIN, 1 LEVEL|
|SC4, CLIP L/14, ArtELingo images|3|6 of 6|5 WIN, 1 LEVEL|
|SC5, MMArt images|3|6 of 6|5 WIN, 1 LEVEL|
|SC6, photo model|3|4 of 4|4 WIN|

Each painting row licenses: “With no Qwen3 pass at test time, training adds over the plain embedding in every cell and it is level with or above the instructed Qwen3 in 6 of 6 cells.”
Photos license: “With no Qwen3 pass at test time, training adds over the plain embedding in every cell and it is level with or above the instructed Qwen3 in 4 of 4 cells.”
For paintings, emotion retrieval is LEVEL:

|Emotion mAP|New|Instructed Qwen3|Difference [interval]|LEVEL unit|
|---|---|---|---|---|
|SigLIP 2|0.269|0.268|+0.0011 [-0.0001, +0.0021]|0.0026|
|CLIP L/14|0.267|0.268|-0.0016 [-0.0027, -0.0004]|0.0041|
|MMArt|0.202|0.202|+0.0001 [-0.0004, +0.0005]|0.0004|

Ablation (SC7), no level. Licensed sentence:
“On ArtELingo test, removing &lt,part&gt, changes AMI by x and mAP by y (interval), as strong as the result it explains.”
Changes below are relative to the full model. Blank part cells repeat the part above.

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

Encoder pick (SC9): “Among the SigLIP-only models, SigLIP 1 did best on the validation split (0.3882 against 0.3838 and 0.3762).”

Against SigLIP 2 and CLIP L/14, respectively. Margin: 0.0045, ahead in 5 of 5 seeds.
Close call: the margin is below its seed range, 0.0055.

**What it does not show.** SC2's text fail stands, the model stays an extra row.
Scope: image tests on these datasets, encoders and five seeds. New encoder text retrieval was not tested.
LEVEL uses a validation seed range, with an inherited EmoSet exception. It does not prove equality.
No multiplicity correction. Current models are descriptive only.

**Options for next.**

a. Write supplementary rows. Cost: paper work, no runs. Records the image claims beside the text limit.
b. Do the planned saved-output emotion-group check. Cost: bounded CPU analysis with a shared embedding map. Describes what those groups separate.
c. Revisit captions and prefix picks after the deadline. Cost: validation training/scoring. Could locate the text loss.

**Your reading.** Left for the user.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Image claims extend, the text fail stands. Graphs support style, the relational term supports genre.

**Choices that could change a verdict.**

- Without the gate (vii) recheck, stale hashes leave the ablation uncertified.
- Count the void start-up as a second crash: SigLIP 2 and encoder pick inconclusive, runner blocks CLIP test cells.
- Independent seed draws make the small genre mAP gain without content correction cross zero (recomputed). No level applies.

Other choices preserve levels or need earlier data/retraining.

**Our recommendation.** Write the decided extra rows before the paper deadline. Then do the saved-output emotion check (r30). Defer new training.
Paper chat still chooses joined-vector or aspect-head presentation.

</details>

## What ran

Completed locally, RTX 3090.

- **Types:** stage 2a: development sweep, training and validation picks. Stage 2b: claim tests and ablation. Timing belongs to stage 1.
- **Seeds:** five, 0 to 4. Photos fine-tune matching emotion checkpoints on Affection.

|Data|Validation|Test|
|---|---|---|
|ArtELingo|6,152|12,281, genre queries 9,949|
|MMArt|5,769 labeled ArtELingo paintings|16,384 outside ArtELingo|
|Affection|4,251|8,501|
|EmoSet|5,770|17,348|

MMArt labeled emotion/style/genre counts: 16,218/16,384/13,420.

- **Code:** build 1427d43e, start-up fix e2641622, independent precision fix 3d081848, recheck 10205d96. Picks and ranges were committed before test reads.
- **C13:** `s2_compare.json`: all pieces pass at 1e-6. Largest validation/test differences: 1.6e-8/2.8e-14. No inconclusive cells.

## Results

Top levels hold. The validation fail sets placement.

|ID|Measured|Number|Compared against|Levels 1/2/3|Reached|
|---|---|---|---|---|---|
|SC4, SigLIP 2|image cells|6 P WIN, 6 Q WIN/LEVEL|plain encoder (P), instructed Qwen3 (Q)|≥5 P WIN/all 6/L2 and ≥5 Q WIN/LEVEL|3|
|SC4, CLIP L/14|image cells|6 P WIN, 6 Q WIN/LEVEL|same|same|3|
|SC5, MMArt|image cells|6 P WIN, 6 Q WIN/LEVEL|plain SigLIP 1, Q|same|3|
|SC6, photos|emotion cells|4 P WIN, 4 Q WIN|plain SigLIP 1, Q|≥3 P WIN/all 4/L2 and ≥3 Q WIN/LEVEL|3|
|SC7|removals|24 changes|full model|attribution with intervals/none/none|no level|
|SC9|validation image mean|0.3882/0.3838/0.3762|SigLIP 1/SigLIP 2/CLIP L/14|highest, exact tie to SigLIP 1/none/none|SigLIP 1, close|

Sentences are in the brief. SC9 decides no replacement.

Cells: five-seed means, P = plain, Q = instructed Qwen3, ordered P / Q. W = WIN, L = LEVEL. Shared values appear once.
AMI: clustering agreement. mAP: mean average precision. Image mAP uses the joined vector against P, aspect head against Q.
Photos use the emotion head throughout. WIN: lo > 0. Otherwise LEVEL: lo > −r. Otherwise BELOW.
Paired percentile 95% intervals: 2,000 row/seed draws for AMI, 10,000 query/seed draws for mAP.

### Other encoders (SC4)

Both reach level 3. Current columns: `paper_cells.md`, checked from diagnostic (d), descriptive only.
Both use SigLIP 1's chosen prefix, no own check. Sources: `image_cells_{siglip2,clipl14}_prefix.json`, SC4 ladders.

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

Source: `photo_cells_siglip1_prefix.json`, SC6 ladder.

|Cell|New P/Q|Comparator P/Q|d [lo, hi], P/Q|r|Outcome P/Q|
|---|---|---|---|---|---|
|affection AMI|0.183|0.099/0.155|+0.0843 [+0.0786, +0.0901]/+0.0277 [+0.0216, +0.0332]|0.0076|W/W|
|affection mAP|0.325|0.255/0.295|+0.0705 [+0.0673, +0.0733]/+0.0305 [+0.0275, +0.0331]|0.0058|W/W|
|emoset AMI|0.432|0.267/0.394|+0.1652 [+0.1530, +0.1796]/+0.0382 [+0.0265, +0.0528]|0.0309|W/W|
|emoset mAP|0.487|0.364/0.452|+0.1234 [+0.1193, +0.1275]/+0.0354 [+0.0313, +0.0393]|0.0069|W/W|

### Ablation (SC7)

Removals/intervals: brief, `ablation.json`. No r or outcome labels.
mAP uses plain cosine.

### Encoder pick (SC9)

Pick/margin: brief, `encoder_pick.json`, checked against `image_groups_*.json` and stage 1 groups.
Qwen3 is information only. SigLIP 1 leads at every matched seed.

## Agent choices checked

Ranked [decisions](../../../../src/test/20261112_siglip_only/decisions.md): certification, then interval pairing.

|Rank|Choice|Alternative checked and effect|
|---|---|---|
|1|Gate (vii) recheck, 2026-11-14 01:23|Alternative: retain stale try-3 hashes, SC7 inconclusive. Scores cannot certify code. Separate try-1 recheck passes trainer and frozen replay checks.|
|2|Void crash, 2026-11-13 19:08|Keep `void/crash_siglip2_prefix_20261113_1906.json`: two failures, out. SigLIP 2 SC4 and SC9 inconclusive, runner blocks CLIP. Scores cannot override admission.|
|3|Shared seeds, 2026-11-13 00:41|Recomputed genre mAP without content correction, independent seeds: +0.0003 [-0.0011, +0.0017], shared: +0.0003 [+0.0001, +0.0006]. Positive attribution disappears, no level.|
|4|Float64 fix, 2026-11-13 19:24|Float32 leaves MMArt replay uncertified. Float64 replay passes, tolerance unchanged.|
|5|Ranges/picks, 2026-11-12 23:04, 2026-11-13 02:50, 02:30|AMI r instead: MMArt emotion still LEVEL, same level (recomputed). EmoSet WIN ignores r. Other read-outs need new paired cells.|
|6|Caption form, 2026-11-13 01:58, photo means, 00:10|Own prefix picks or ignored explanations require untrained alternatives. Claims could move, saved scores cannot reconstruct them.|

Rank 3 is exploration from genre `ap` arrays in `rederive/r29rd_s2abl_cache/`. Sidecars select `/cosir_r29/`, all seeds.
Queries pair, seeds draw independently, with recorded draw count and RNG seed. No files written.

Other yes/maybe lines: input normalisation and caption-only swap define the model. Graph swaps need retraining.
Guards passed, trainer fallback unused. Same-device exports unchecked.
Parking, photo variants and partial-arm rules are inactive: gates pass, removals complete.
No encoder truncation. Text-form/tolerance choices concern stage 1.

[Handoff §3](../../../superpowers/handoffs/2026-11-13-r31-run-2-handoff.md#3-where-things-stand), Codex choices checked:
matched seeds ahead recompute to all seeds, tie band preserves the pick, runner-up range also gives a close call.
MMArt uses the specified joined vector against plain. Aspect-head alternative needs different scores, not recomputed.
EmoSet's inherited unrounded unit does not affect WIN cells. Ablation pairing is rank 3.
All P cells WIN and five Q cells WIN per painting claim, so changing only LEVEL units cannot move these levels.

## Analysis

**Our reading:** image changes from current models are small. Diagnostic (d), spec §3.6: every interval crosses zero.
SigLIP 2 spans -0.0007 to +0.0027, CLIP L/14 -0.0026 to +0.0011. Machines differ, so this proves neither equality nor a switch cost.

Graph-positive loss pulls neighbours together. Content correction reduces shared graph content.
Their removal, or uninstructed graphs, lowers emotion/style scores. The relational term matches Qwen3's image-similarity pattern.
Its removal lowers genre scores, with negative intervals. Other genre AMI intervals cross zero. Interactions remain untested.
CLIP emotion retrieval is LEVEL despite a negative interval. Photos test image emotion, not text.

## Deviations

Amendments/fixes below. Other deviations: none. Seed sensitivity is exploration.

- **A1, A2:** after stage 1 results, the user allowed stage 2 on the fail and added validation encoder selection. Fail and placement stand. Record names spec d018f926, containing both.
- **Start-up:** both SigLIP 2 attempts failed the unset `CUBLAS_WORKSPACE_CONFIG` assertion before training. Encoder/photo entry points now set `:4096:8` when absent. False crash retained in `records/void/`, treated as wiring rather than a second training crash.
- **MMArt replay:** Codex's query-expansion float32 downcast was removed under the earlier float64 convention. Full replay passed, tolerance unchanged.
- **SC7:** scoring refused stale gate hashes after amendment edits. Separate recheck repeated trainer/frozen replay checks, preserving the original record, before scoring.
- **Truncation, as specified:** 4,045 of 526,749 photo explanations cut at 64 tokens and kept. Both encoder stores: zero truncated captions.
- **C13 caveats:** genre mAP row count differs from Codex's query count as information only, `n_queries` agrees. Codex's reference lacks earlier AMI seed targets, means pass. Stage 2 comparison passes.

## Claim review

Pending (Claude, before commit).

## Files and storage

Outputs: `/data/SSD2/cosir_r29/20261112_siglip_only/`.
[Round folder](../../../../src/test/20261112_siglip_only/): records, numbers sheet, decisions, code, logs and re-derivations.
Storage: see the wrap-up, disk ledger rows dated 2026-11-14. Nothing deleted.
No removals proposed. Defer storage to the wrap-up ledger.
