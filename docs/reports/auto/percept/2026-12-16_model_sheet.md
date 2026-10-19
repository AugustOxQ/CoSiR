# Stage 2 model sheet: emotion reaches the line; style and genre remain below the instructed-image target

> 2026-10-19 (Amsterdam). The file name uses CoSiR's sequence date.
> Spec: [stage 2 model sheet](../../../superpowers/specs/2026-10-19-stage2-model-sheet-design.md).
> Decision rule: none (descriptive). Run log: [round log](../../../../src/test/20261019_model_sheet/20261019_model_sheet_log.md).

## Results brief

**What we tried.** One number sheet of 13 settings
([spec](../../../superpowers/specs/2026-10-19-stage2-model-sheet-design.md)): a shared first layer (the trunk) with
one head per aspect, two other loss forms, graph picks, and distillation toward the instructed Qwen image. The target
is the instructed image clustered directly, which needs three image passes; our model reads the default image once.

**Outcome against the bar.** Default-input rows reach the emotion line: PercepT core 0.0772 + 0.0100 = **0.0872**.
Qualifying rows: separate heads [A1], trunk [T1], negatives [T4], masks [T5], graph-plus-distillation [D2–D5, D17].
Headline best-per-aspect values use three-seed means; one-seed results are labelled.

**Outcomes (a) to (d).** AMI measures cluster agreement with labels; higher is better.
Values below are image-side AMI, in emotion / style / genre order.

|Outcome|Measurements beside their comparators|Reached?|
|---|---|---|
|(a) One default-input row reaches every target|Even the best per aspect: 0.0962 / 0.3700 / 0.4891; target: 0.0929 / 0.3952 / 0.4956|No|
|(b) Target reached on some aspects only|Emotion: direct distillation, weight 0.5 [D2], 0.0962 vs 0.0929; style and genre remain below target as above|Yes|
|(c) Distillation alone is level with the best|Alone [D1], one seed: 0.0825 / 0.2751 / 0.4366; best per aspect, three seeds: 0.0962 / 0.3700 / 0.4891|No|
|(d) Trunk below separate heads everywhere|Trunk [T1]: 0.0930 / 0.3591 / 0.4603; separate heads [A1]: 0.0894 / 0.3539 / 0.4574; both three seeds|No|

**What it does not show.** Development results only; no held data were read. Evaluation paintings also chose rows.
Six rows have one seed (T3, T4, D1, D4, D17, I1). T1's emotion tie band (0.0010) is narrower than most rows' own
spread (0.0019 to 0.0046). Distillation alone was run with the direct pull only. Retrieval and the claim test remain.

**Options for next.**
1. A second short stage 2 loop for style and genre: relational distillation alone, at weights 0.5 and 2, and with
   direct distillation at 0.5. Cost: a small build, stage 0 again, about 30 minutes of GPU. It tells whether the genre
   gap (0.0065) or the style gap (0.0251) closes.
2. The claim test spec now (held data, the 10-29 emotion gate), with D2 or T1; the style and genre target waits for
   11-10. Cost: one decide chat, then a build and a run. It tells whether the development emotion margin (+0.016 to
   +0.019 over PercepT core) holds on held data.
3. Both: the claim test spec in the next decide chat, and the short loop in the same night as its build. Cost: the
   sum; the claim test row may change after the loop.

**Your reading.** Left for the user.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Outcome (b): the target holds on emotion only. The parts help different aspects: direct
distillation at 0.5 helps emotion and style, relational distillation helps style and genre, and no row combines
them. Direct distillation alone is far below the graph rows; that control drops the graph and caption terms
together, so together they carry most of the result. The trunk costs
nothing. With instructed inputs the model still misses style and genre (one seed), so the training, not only the
input, limits those two. We are fairly sure of each part's direction (three seeds for the main rows). The sizes are
optimistic: the same 5,000 paintings chose the rows.

**Our recommendation.** Option 1, then option 2, inside the 10-23 time-box. Genre is 0.0065 short and relational
distillation is the closest part, so one cheap loop can tell whether a combined pull reaches it; we expect the style
gap to stay. For the claim test, D2 is our first choice: the best emotion row (+0.019 over PercepT core) and level with
the best on style. Its genre (0.4540) is level with the stage 1 network (our view).

</details>

## What ran

Stage 0 passed its regression check; stages 1 to 3 were explorations.

|Stage|Type|Rows and seeds|Local time|
|---|---|---|---|
|0|Regression check, CPU|Stage 1 separate-head reproduction and stored PercepT outputs|PASS at 21:26|
|1|Exploration|A1, T1–T5, D1–D5; seed 0|21:59–22:12|
|2|Exploration|A1, T1, T2, T5, D2, D3, D5; seeds 1 and 2; D17 seed 0|22:12–22:30|
|3|Exploration|I1, seed 0|22:30–22:33|

- **Data:** frozen train-side sample: fit 13,400, dev 1,489, evaluation 5,000 paintings; genre uses its 4,053 labelled paintings.
- **Scoring:** k-means at k = 20, seeds 0 to 2, on image and caption outputs; AMI decides picks.
- **Code and device:** run chat “r10 run 4”; main at `03946d2`, build merged at `e066c92`; no code changed during the run.
  Training used the local RTX 3090 under the GPU lock; scoring used CPU.
- **Regression evidence:** genre arrays match bitwise; 36/36 AMI checks, 7/7 best epochs, and 45/45 PercepT checks match.
  `outputs/stage0.json` records trainer and scorer versions and no stale outputs.
- **Queues:** every recorded step exits 0. Status files and the seed manifest give 27 row-and-seed runs:
  11 initial + 14 extra-seed + 1 rerun + 1 instructed-input run. Separate heads comprise three aspect fits per row seed.

## Results

The emotion line is reached; the full instructed-image target is not reached.

|ID|Measured|Number|Compared against|Bar|Outcome|
|---|---|---|---|---|---|
|§6|Default-input emotion AMI|D2 0.0962, three seeds; qualifying rows listed in brief|PercepT core 0.0772|+0.0100 = 0.0872|reached|
|(a)|All targets, same default-input row|Best per aspect: 0.0962 / 0.3700 / 0.4891|Instructed image|0.0929 / 0.3952 / 0.4956|not reached|
|(b)|Some targets only|Emotion D2 0.0962; style D5 0.3700; genre D5 0.4891|Instructed image|0.0929 / 0.3952 / 0.4956|reached|
|(c)|Distillation alone level with best|D1 0.0825 / 0.2751 / 0.4366, one seed|Best per aspect, three seeds|level = within T1's tie band of 0.0962 / 0.3700 / 0.4891|not reached|
|(d)|Trunk below separate heads everywhere|T1 0.0930 / 0.3591 / 0.4603|A1, both three seeds|0.0894 / 0.3539 / 0.4574|not reached|

**Verdict under the rule: none (descriptive).**

Rows rank by their own-label image AMI. AMI is mean ± population sd over three model seeds, or seed-0 AMI marked “1 seed”.
Other columns show means; full spreads: [number sheet](../../../../src/test/20261019_model_sheet/ms_tables_final.md) (a committed copy of `outputs/ms_tables.md`).
Bar spreads use model seeds for learned rows, k-means seeds for single vectors. Probe probabilities were not stored.
The image-encoder bar uses Qwen for emotion and genre, CLIP for style.
Caption AMI uses the own label; cosine uses the instructed image. Epochs list seeds in order. † marks rows within T1's population-sd tie band; form a stays on a tie.
All rows use default inputs except I1, which uses instructed inputs.

### Emotion

T1 tie band: ±0.0010 AMI.

|Row|Image AMI|V|ARI|FMI|Caption AMI|Cosine|Best epochs|
|---|---:|---:|---:|---:|---:|---:|---|
|PercepT core|0.0772 ± 0.0012|0.083|0.024|0.134|—|—|—|
|Image encoder|0.0696 ± 0.0020|0.076|0.027|0.135|—|—|—|
|Target|0.0929 ± 0.0017|0.099|0.040|0.149|—|—|—|
|Linear probe|0.1215 ± 0.0014|—|—|—|—|—|—|
|Stage 1 network|0.0805 ± 0.0033|0.087|0.028|0.135|—|—|—|
|D2|0.0962 ± 0.0020|0.102|0.038|0.146|0.1828|0.478|3/4/4|
|T4|0.0959 (1 seed)|0.102|0.038|0.146|0.1686|0.004|3|
|D3|0.0957 ± 0.0026|0.102|0.038|0.146|0.1904|0.441|6/6/6|
|T5|0.0955 ± 0.0019|0.101|0.037|0.145|0.1711|-0.007|3/3/2|
|I1|0.0953 (1 seed)|0.101|0.049|0.161|0.1791|0.002|6|
|T1 †|0.0930 ± 0.0010|0.099|0.039|0.149|0.1727|-0.005|3/3/2|
|D17|0.0919 (1 seed)|0.098|0.040|0.150|0.1757|-0.007|3|
|D4|0.0895 (1 seed)|0.095|0.036|0.144|0.1924|0.428|6|
|A1|0.0894 ± 0.0024|0.095|0.035|0.143|0.1711|-0.002|3/4/2|
|D5|0.0884 ± 0.0046|0.094|0.035|0.144|0.1708|-0.011|3/3/3|
|T3|0.0856 (1 seed)|0.092|0.031|0.139|0.1608|0.008|3|
|D1|0.0825 (1 seed)|0.088|0.030|0.137|untrained|0.442|84|
|T2|0.0778 ± 0.0004|0.084|0.029|0.137|0.1900|-0.004|5/4/4|

### Style

T1 tie band: ±0.0072 AMI.

|Row|Image AMI|V|ARI|FMI|Caption AMI|Cosine|Best epochs|
|---|---:|---:|---:|---:|---:|---:|---|
|PercepT core|0.2286 ± 0.0049|0.243|0.084|0.150|—|—|—|
|Image encoder|0.3000 ± 0.0027|0.313|0.120|0.180|—|—|—|
|Target|0.3952 ± 0.0028|0.406|0.196|0.253|—|—|—|
|Linear probe|0.5275 ± 0.0082|—|—|—|—|—|—|
|Stage 1 network|0.3485 ± 0.0070|0.360|0.156|0.214|—|—|—|
|I1|0.3835 (1 seed)|0.395|0.184|0.242|0.1786|-0.015|8|
|D5|0.3700 ± 0.0081|0.381|0.177|0.235|0.1735|-0.007|2/4/2|
|D2|0.3697 ± 0.0034|0.381|0.173|0.231|0.1728|0.483|4/4/4|
|D17|0.3692 (1 seed)|0.381|0.166|0.225|0.1725|-0.014|3|
|T1 †|0.3591 ± 0.0072|0.371|0.162|0.220|0.1705|0.001|4/4/2|
|T4 †|0.3556 (1 seed)|0.367|0.164|0.222|0.1731|-0.007|4|
|A1 †|0.3539 ± 0.0068|0.365|0.161|0.220|0.1746|-0.006|3/2/2|
|D3 †|0.3538 ± 0.0071|0.365|0.161|0.219|0.1704|0.480|4/5/5|
|T5|0.3501 ± 0.0024|0.362|0.156|0.215|0.1728|-0.003|3/3/3|
|T2|0.3480 ± 0.0094|0.360|0.155|0.213|0.1709|-0.001|2/3/2|
|D4|0.3385 (1 seed)|0.350|0.147|0.205|0.1734|0.466|4|
|T3|0.3059 (1 seed)|0.318|0.120|0.179|0.1667|-0.006|2|
|D1|0.2751 (1 seed)|0.288|0.104|0.164|untrained|0.449|102|

### Genre

T1 tie band: ±0.0045 AMI.

|Row|Image AMI|V|ARI|FMI|Caption AMI|Cosine|Best epochs|
|---|---:|---:|---:|---:|---:|---:|---|
|PercepT core|0.4359 ± 0.0037|0.441|0.250|0.346|—|—|—|
|Image encoder|0.4486 ± 0.0035|0.453|0.265|0.362|—|—|—|
|Target|0.4956 ± 0.0036|0.500|0.340|0.433|—|—|—|
|Linear probe|0.6566 ± 0.0010|—|—|—|—|—|—|
|Stage 1 network|0.4583 ± 0.0050|0.463|0.266|0.364|—|—|—|
|D5|0.4891 ± 0.0002|0.493|0.303|0.400|0.3959|0.001|1/1/1|
|I1|0.4852 (1 seed)|0.490|0.293|0.390|0.3913|0.000|3|
|D17|0.4843 (1 seed)|0.489|0.302|0.399|0.3938|-0.007|1|
|T2|0.4699 ± 0.0046|0.474|0.272|0.371|0.4018|-0.002|1/1/1|
|T4|0.4674 (1 seed)|0.472|0.261|0.360|0.4047|-0.009|1|
|T3|0.4657 (1 seed)|0.470|0.275|0.373|0.3896|-0.003|1|
|T5 †|0.4630 ± 0.0023|0.468|0.264|0.362|0.3962|-0.004|1/1/1|
|T1 †|0.4603 ± 0.0045|0.465|0.265|0.363|0.3963|-0.002|1/1/1|
|A1 †|0.4574 ± 0.0078|0.462|0.265|0.364|0.3966|-0.006|1/1/1|
|D2|0.4540 ± 0.0053|0.459|0.246|0.344|0.4013|0.450|2/2/2|
|D3|0.4420 ± 0.0030|0.447|0.242|0.339|0.4025|0.455|2/2/2|
|D1|0.4366 (1 seed)|0.441|0.234|0.332|untrained|0.463|160|
|D4|0.4322 (1 seed)|0.437|0.221|0.319|0.4021|0.468|3|

## Analysis

**Our reading:** the trunk helps emotion; direct distillation at 0.5 helps emotion and style; relational distillation
helps style and genre. Stronger direct-distillation weights lose genre AMI.

Cells compare each row against the trunk [T1]: AMI difference, then ↑ above, ≈ within, or ↓ below its tie band.
Bands describe seed spread; they are not significance tests.

|Change (row)|Seeds|Emotion Δ|Style Δ|Genre Δ|
|---|---:|---:|---:|---:|
|Separate heads (A1)|3|-0.0036 ↓|-0.0052 ≈|-0.0029 ≈|
|Stage 1 picks (T2)|3|-0.0152 ↓|-0.0111 ↓|+0.0096 ↑|
|Second picks (T3)|1|-0.0074 ↓|-0.0531 ↓|+0.0054 ↑|
|Other-aspect negatives, form b (T4)|1|+0.0029 ↑|-0.0035 ≈|+0.0070 ↑|
|Learned masks, form c (T5)|3|+0.0025 ↑|-0.0090 ↓|+0.0027 ≈|
|Graphs and caption terms removed (D1)|1|-0.0105 ↓|-0.0839 ↓|-0.0237 ↓|
|Direct distillation, weight 0.5 (D2)|3|+0.0032 ↑|+0.0106 ↑|-0.0063 ↓|
|Direct distillation, weight 1 (D3)|3|+0.0027 ↑|-0.0052 ≈|-0.0183 ↓|
|Direct distillation, weight 2 (D4)|1|-0.0036 ↓|-0.0206 ↓|-0.0281 ↓|
|Relational distillation, weight 1 (D5)|3|-0.0047 ↓|+0.0109 ↑|+0.0288 ↑|
|Relational distillation on form b (D17)|1|-0.0011 ↓|+0.0101 ↑|+0.0239 ↑|

**Our reading:** the trunk beats separate heads on emotion; style and genre tie.
Form b raises emotion and genre on one seed; form c raises emotion but lowers style.
Both alternative graph picks lose emotion and style. Stage 1 picks raise genre despite its unchanged graph.
**Our reading:** T2 changes only the emotion graph, so its genre gain (+0.0096) and style loss (−0.0111) pass through
the shared layer; in the trunk, one aspect's graph moves the other heads.

The separate-head emotion gain comes with Gres replacing Fres1: 0.0894 against stage 1's 0.0805, both three-seed means.
The trunk on stage 1 picks [T2] falls to 0.0778. **Our reading:** graph choice matters in this comparison.

Distillation alone removes graphs and caption terms together.
**Our reading:** its deficit supports the combined objective; the graph effect remains mixed with caption terms.
Relational and low-weight direct distillation tie on style; relational distillation leads on genre.
Only the direct pull was run alone; relational distillation alone, the best part for genre, is untested.

### Diagnostics

Direct-distillation cosines lie from 0.428 to 0.483; other rows lie near zero, from −0.015 to 0.008.
**Our reading:** coordinate alignment alone does not yield the best clustering. Relational distillation matches similarities, compatible with low coordinate cosine.

Each head's image AMI on the other two labels (means; full values in the sheet copy linked above):
emotion heads score 0.1266 to 0.1723 on style and 0.3118 to 0.4229 on genre; style heads 0.0611 to 0.0773 on
emotion and 0.3235 to 0.4137 on genre; genre heads 0.0685 to 0.0778 on emotion and 0.2189 to 0.2507 on style.

**Our reading:** emotion heads retain strong genre signal. Relational distillation [D5]: genre 0.4229 versus own-label emotion 0.0884, both three-seed means.
This limits claims of aspect separation. AMI scales differ by label, so compare each value with the default Qwen
image's genre AMI (0.4486): the emotion heads keep most of the input's genre structure (0.3118 to 0.4229).

Form c's masks overlap substantially across all three seeds:

|Mask pair|Cosine range|Shared active dimensions, range|
|---|---:|---:|
|emotion / style|0.605–0.635|1,478–1,541|
|emotion / genre|0.629–0.634|1,473–1,539|
|genre / style|0.637–0.648|1,542–1,578|

**Our reading:** masks share many dimensions. They start as N(0.9, 0.7) through a ReLU, so about 90% of the 2,048
dimensions are active at the start; two random masks would then share about 1,660 (our estimate, not measured). The
trained masks share somewhat fewer, so they moved little toward an aspect split. Graph rows peak early (tables above).
Distillation alone [D1] peaks at epochs 84 / 102 / 160 for emotion / style / genre, one seed.
**Our reading:** longer training does not close the gap.

### Instructed inputs and the final rerun

Instructed inputs raise emotion and style against their base, but miss style and genre targets. Both rows have one seed.

|Aspect|Base D17|Instructed I1|I1 − base|Target|
|---|---:|---:|---:|---:|
|Emotion|0.0919|0.0953|+0.0034|0.0929|
|Style|0.3692|0.3835|+0.0143|0.3952|
|Genre|0.4843|0.4852|+0.0009|0.4956|

**Our reading:** instructed inputs leave target gaps; the form-b rerun [D17] gives no consistent gain over form a [D5].
The input explains about half of D17's style gap (+0.0143 of 0.0259) and almost none of its genre gap (+0.0009 of
0.0114), so the training objective limits style and genre too (one seed).

## Deviations

None from the spec; agent defaults specify the rerun and one-seed final rows.

- Form b [T4], on one seed, triggered the rerun: emotion 0.0959 exceeds T1 mean plus sd, 0.0940;
  genre 0.4674 exceeds 0.4648. Relational distillation [D5] was the best D row on seed 0.
- The rerun [D17] is relational distillation on form b with the trunk; it has one seed (agent defaults 12, 13, 16).
- The instructed row [I1] uses that base with instructed inputs, one seed (agent defaults 13, 17).
  Seed-0 mean AMI across aspects chose the base: D17 0.3151 vs D5 0.3140, computed from the stored seed-0 values.

**Selection disclosure (§8):** the same 5,000 evaluation paintings chose the extra-seed rows, the conditional rerun, and the instructed row's base.
This sheet adds 27 row-and-seed training runs to the earlier evaluation reads.
Those reads include the premise check, three pilots, stage 1 (63 graphs and 59 head runs), PercepT core, and the ceiling check.
Development numbers are optimistic; a top row picked from 12 is biased upward.

Distillation may learn artists seen during training because the split groups by painting.
This sheet does not measure that risk; the held split also shares artists.
No held data were read.

## Files and storage

- **Round:** [src/test/20261019_model_sheet/](../../../../src/test/20261019_model_sheet/).
- **Outputs:** `outputs/` is gitignored, **3.4G** (`du -sh`); it holds each head's float16 output arrays and config
  (no checkpoints), `stage0.json`, `ms_results.json`, `ms_tables.md`, `seed_rows.json`, `d17.json`, and `i1.json`.
- **User-read report:** [2026-10-19_model_sheet](../../../user_read/2026-10-19_model_sheet.md).
- **Status:** `queue_stage1_status.txt`, `queue_stage2_status.txt`, `queue_stage3_status.txt`; all recorded steps exit 0.
- **Back-fill:** [ms_backfill_k40.md](../../../../src/test/20261019_model_sheet/ms_backfill_k40.md) adds V, ARI and FMI;
  already folded into both K = 40 reports.

- **Storage:** nothing deleted. `outputs/` (3.4 GB, system disk) is listed in the run handoff for the decide chat's
  storage summary (recommendation: keep while this sheet is in use); see the disk ledger after that decision.
