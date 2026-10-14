# Round 9 report: detail tables

[Back to the report](../../../docs/reports/auto/v2/2026-12-02_aff_scope.md).

Descriptive tables follow rule §8: none changes a verdict; an interval excluding zero is not a test.

## Stage timings

Sources: `20261202_r9_scope_log.md`, “r9 run chat 1” and “r9 run chat 2”, stage/read timing entries; `results/stage0b.json:runtime_s`; `results/held_pass_H{6,7,8,9,10}.json:runtime_s`.

|Completed work|Recorded timing / gate|
|---|---|
|Stage 0b|step 07: 455 s; compare record: 390.2 s; case 1, CPU parts and picks exact|
|Qwen stage 2|04:38–06:01; 5,062 s; ready, not switched off|
|SigLIP stage 2|ended 07:10; 4,114 s; ready, not switched off|
|H7 read / phase 2|08:19–08:55, 2,195 s / 105 s; verdict at 08:57|
|H6 read|08:59–09:41; 2,477 s|
|H8 read|09:43–09:53; 593 s|
|H9 / H10 reads (together)|09:54–10:56 / 09:54–10:30; 3,708 s / 2,169 s|

## Agreements

Sources: `rederive/rd9_out/rd9_{backbone}_phase1_agreement.json` and `rd9_{read}_agreement.json`; keys `agreement`, `compared`, `failed`.

**Agreements:** `rederive/rd9_out/rd9_{backbone}_phase1_agreement.json` and `rd9_{read}_agreement.json`: `agreement/compared/failed`.

|Phase|Backbone / read|Agreement|Compared|Failed|
|---|---|---|---|---|
|1|clip_b32|true|169|0|
|1|qwen3vl_emb_2b|true|160|0|
|1|siglip2_so400m|true|160|0|
|2|H6|true|166|0|
|2|H7|true|166|0|
|2|H8|true|171|0|
|2|H9|true|171|0|
|2|H10|true|171|0|

## Selection rows

Sources: `pins_{clip,qwen3vl_emb_2b,siglip2_so400m}.json`, `seed42.checks`; `results/stage0b.json:item3.L1.result.points.*.want`. R@1 points.

**Selection rows, seed 42; no verdict.** Check differences, R@1 points. Sources: pins `seed42.checks`; CLIP `stage0b.json`, `item3.L1.result.points.*.want`.

|Backbone|P1|P2|P3|P4|P5|P6|P7|
|---|---|---|---|---|---|---|---|
|clip|+6.17|+5.76|+0.80|+0.70|+0.74|+3.11|+3.01|
|qwen3vl_emb_2b|+4.83|+4.65|+0.86|+0.70|+0.94|+2.76|+2.78|
|siglip2_so400m|+6.31|+5.87|+0.85|+0.50|+0.80|+3.04|+3.05|

## Development

Sources: `pins_{clip,qwen3vl_emb_2b,siglip2_so400m}.json:picks.cells.gate_share`; Qwen/SigLIP `seed42.mean_r1.{AFF,B1}`; CLIP `results/stage0b.json:item3.L1.result.r1.{AFF,B1}.want`.

Development (pins/replay):

|Backbone|Gate fraction|AFF/B′(A1) R@1|
|---|---|---|
|CLIP|0.4505|19.14/18.80|
|Qwen|0.6134|19.11/19.25|
|SigLIP|0.4958|19.24/19.26|

## Full pair and direction points

Sources: `results/descriptive_H{6,7,8,9,10}.json`; `rows.per_pair.<pair>.{B,B0,aff_cf}.aff_minus.{r1,gain}.point`; `rows.per_direction.{i2t,t2i}.per_pair.<pair>.{B0,aff_cf}.aff_minus.{r1,gain}.point`. R@1 points.

**Pair/direction points:** B/B′(A0)/CF margins, CF gain; then i2t and t2i B′(A0) margin/CF gain. Intervals: `rows.per_pair`, `rows.per_direction.{i2t,t2i}.per_pair`; earlier a/b skeleton corrected.

|Pair|H6|H7|H8|H9|H10|
|---|---|---|---|---|---|
|emotion × style|+1.42/+1.22/+1.08/+1.28; +0.86/+1.05; +1.59/+1.51|+1.28/+0.97/+0.95/+1.64; +0.44/+1.43; +1.50/+1.86|+0.37/+0.12/+0.05/-0.03; -0.13/+0.02; +0.37/-0.09|+1.01/+0.87/+0.60/+0.31; +0.55/+0.10; +1.19/+0.52|+0.77/+0.58/+0.34/+0.43; +0.26/+0.26; +0.89/+0.59|
|emotion × genre|+1.73/+1.52/+1.76/+5.46; +0.94/+5.00; +2.10/+5.93|+1.85/+1.42/+1.85/+6.43; +1.14/+6.25; +1.70/+6.61|+0.95/+0.56/+1.03/+5.14; +0.44/+4.96; +0.68/+5.32|+1.19/+0.97/+1.21/+4.59; +0.55/+3.90; +1.40/+5.28|+1.26/+0.68/+1.39/+5.84; +0.46/+5.50; +0.89/+6.18|
|style × genre|-0.71/-0.87/-0.51/+1.63; -0.98/+1.57; -0.76/+1.69|-0.57/-0.86/-0.30/+1.97; -1.09/+1.85; -0.62/+2.10|-0.35/-0.49/+0.16/+4.60; -0.84/+4.16; -0.15/+5.03|-0.54/-0.73/-0.17/+3.40; -0.67/+2.95; -0.79/+3.84|-0.22/-0.80/+0.10/+4.28; -0.74/+4.09; -0.86/+4.48|

## Per-seed points

Sources: `results/descriptive_H{6,7,8,9,10}.json`; `rows.per_seed.<seed>.pooled.{cosine,rca,B,B0,aff_cf}.aff_minus.{r1,gain}.point`. R@1 points.

**Seeds:** P1–P7 points; `rows.per_seed.<seed>.pooled.<arm>.aff_minus.{r1,gain}`.

|Seed|H6|H7|H8|H9|H10|
|---|---|---|---|---|---|
|52|+4.95/+4.64/+0.83/+0.70/+0.83/+2.88/+2.81|+6.62/+5.81/+0.71/+0.40/+0.69/+3.09/+2.91|—|—|—|
|53|+4.71/+4.59/+0.80/+0.57/+0.77/+2.67/+2.66|+6.43/+5.91/+0.92/+0.57/+0.93/+3.33/+3.29|—|—|—|
|54|+4.80/+4.60/+0.80/+0.61/+0.74/+2.83/+2.85|+6.73/+6.09/+0.94/+0.56/+0.88/+3.63/+3.61|—|—|—|
|55|—|—|+5.42/+5.20/+0.36/+0.09/+0.43/+3.37/+3.19|+3.87/+3.99/+0.52/+0.34/+0.60/+2.77/+2.98|+6.58/+5.96/+0.55/+0.04/+0.54/+3.50/+3.43|
|56|—|—|+5.46/+5.43/+0.36/+0.10/+0.47/+3.00/+3.06|+4.16/+4.23/+0.55/+0.37/+0.51/+2.52/+2.57|+6.51/+5.90/+0.61/+0.16/+0.68/+3.49/+3.50|
|57|—|—|+5.47/+5.29/+0.26/+0.00/+0.34/+3.33/+3.16|+4.23/+4.14/+0.59/+0.41/+0.53/+3.00/+2.98|+6.55/+5.81/+0.66/+0.25/+0.61/+3.56/+3.27|

## Scorers and diagnostics

Sources: `results/descriptive_H{6,7,8,9,10}.json`; `rows.pooled.<scorer>.r1.point`, `rows.pooled.<scorer>.aff_minus.r1.{point,ci95}`, `gate_share.chosen_cells.overall`, `reader_affect_pick_rate.overall.affect`. R@1 points; gate/pick shares are percentages.

**Scorers (untested):** AFF minus it [95%], `rows.pooled`. PM vectors: AFF minus diag, diag_relu, bilinear, kissme, xing, wang, probe, tip, value_prototype (points); scorer R@1 in JSON.

|Scorer / diagnostic|H6|H7|H8|H9|H10|
|---|---|---|---|---|---|
|B′(A1)|19.65; -0.33 [-0.52, -0.16]|19.55; +0.13 [-0.05, +0.31]|18.32; -0.37 [-0.58, -0.16]|18.74; -0.52 [-0.76, -0.30]|18.73; -0.16 [-0.41, +0.07]|
|R1|19.37; -0.06 [-0.14, +0.02]|19.78; -0.10 [-0.21, +0.00]|17.99; -0.03 [-0.12, +0.06]|18.34; -0.12 [-0.19, -0.05]|18.68; -0.12 [-0.22, -0.01]|
|R1 counterpart|18.58; +0.74 [+0.61, +0.86]|18.98; +0.70 [+0.56, +0.84]|17.67; +0.29 [+0.14, +0.44]|17.79; +0.43 [+0.30, +0.55]|18.11; +0.46 [+0.31, +0.61]|
|PM|+4.95/+5.25/+4.82/+4.82/+4.85/+4.96/+4.91/+4.97/+4.85|+6.59/+6.59/+6.59/+6.67/+6.78/+6.61/+6.62/+6.59/+6.73|+5.49/+5.62/+5.55/+5.51/+5.51/+5.61/+5.75/+5.49/+5.45|+4.14/+4.39/+4.09/+4.09/+4.12/+4.14/+4.14/+4.17/+4.07|+6.55/+6.55/+6.55/+6.61/+6.70/+6.57/+6.70/+6.55/+6.67|
|Chosen gate / affect pick|59.89% / 64.12%|48.41% / 52.81%|53.42% / 63.28%|68.35% / 72.12%|59.06% / 63.50%|

## Anchor-emotion distributions

Sources: `results/descriptive_H{8,9,10}.json`; `artelingo28.anchor_emotion_distribution.{this_read,H5}.pooled.shares`. Emotion fractions in order: amusement, anger, awe, contentment, disgust, excitement, fear, sadness. Per-pair distributions: `artelingo28.anchor_emotion_distribution.{this_read,H5}.per_pair`.

|Distribution|H8|H9|H10|
|---|---|---|---|
|Anchor emotion (this read; H5)|0.105/0.022/0.115/0.314/0.056/0.135/0.101/0.152; 0.106/0.015/0.182/0.332/0.055/0.088/0.093/0.129|0.105/0.022/0.115/0.314/0.056/0.135/0.101/0.152; 0.106/0.015/0.182/0.332/0.055/0.088/0.093/0.129|0.105/0.022/0.115/0.314/0.056/0.135/0.101/0.152; 0.106/0.015/0.182/0.332/0.055/0.088/0.093/0.129|

## Execution provenance

Source: `20261202_r9_scope_log.md`, “r9 run chat 1” and “r9 run chat 2”.

The build used `71843c1`, the restart `5bc5ad2`, smoke/inf fixes `78aac8f`/`6a282d2`, certification `fbd578a`, and handoff `2ac49c3`. Reads used local pinned CPUs; stage-2 A3 used the GPU, free at 07:10. The round log’s “r9 run chat 1” and “r9 run chat 2” sections record these steps.

## Language support and translation coverage

Source: `results/descriptive_H8.json`; `artelingo28.per_language.*.{episodes,median_rows_per_painting,values_on_fewer_than_30_paintings,translation_coverage}`.

H8 gain over CF ranges +0.40 to +4.59; median rows/painting ranges 1–6. Support: `artelingo28.per_language.<language>.{episodes,median_rows_per_painting,values_on_fewer_than_30_paintings}`. Eligibility is pooled; language values can have under 30 paintings.

Coverage includes 167,230 translated rows: 0 empty, 160 equal to source, 151 at the token cap and 1 still in source script. Emakhuwa and IsiNdebele have proxy flags. Length/script: `artelingo28.per_language.*.translation_coverage`.

## Test definitions and provenance

Sources: `DECISION_RULE.md`; `results/held_pass_H{6,7,8,9,10}.json:checks.P*.{point,ci95,n}`; `results/held_verdict_H{6,7,8,9,10}.json:checks.P*.{n_star,passes,reading}`.

Condition gain is correct minus swapped condition for the same target ([glossary](../../../GLOSSARY.md)); CF is the matched control. Bootstrap used 5,000 anchor resamples, seed 42, α = 0.025 one-sided. n_j counts means ≤ 0. H7 had 36,864 episodes and 9,519 anchors.

Sources are pass `checks.P*.{point,ci95,n}` and verdict `{n_star,passes,reading}`. No boundary or secondary tests ran; files retain Holm/sensitivity detail.

## File inventory

Source: `20261202_r9_scope_log.md`, run-chat sections.

F = `src/test/20261202_r9_scope/`: pins/stage/pre-read/smoke records; `results/{held_pass,held_verdict,sensitivity,held_started}_Hn.json`; `rederive/rd9_out/rd9_Hn_agreement.json`.

## Episode design

Sources: `20261202_r9_scope_log.md`; `results/held_pass_H{6,7,8,9,10}.json:n_episodes`; descriptive files `seeds` and `rows.per_pair`.

- **Data/seeds:** ArtELingo scorer-train fits used selection seed 42. H6/H7 reused H5 episodes, seeds 52–54. H8–H10 shared episodes, seeds 55–57, streams `1000·s+ℓ`. Each had 36,864 episodes, 4,096 per pair/seed; smoke used 9001–9003.

The stages completed were 0a regression, 0b recipe replay, 1 preparation, 2 recipes, pre-reads, 3 H6/H7 checks, 4 H8–H10 tests and descriptive passes.

## Deviations detail

The full list behind the report's Deviations section.

The rule wording is unchanged. Defaults and disclosures follow.

- **Agent default, a2:** SigLIP restarted below 100 rows: caption seed 42 smallest 1 → seed 43 smallest 939 (`partitions.npz.record.json:kmeans_restarts`). CLIP/Qwen were byte-equal; option (b) is open.
- **Agent default:** Replay steps 08–11 were copied after byte-equal step 07; case 1 was exact. Qwen regeneration was byte-equal (528/528).
- **Agent default, fx-smoke:** Only smoke bypasses the backbone guard; CLIP follows H6. Regression/pre-reads reran. Review was PASS WITH NOTES.
- **Agent default, fx-inf:** Qwen Wang half-1 λ is infinite. JSON writes `"inf"`/`"nan"`; scoring is unchanged. Review was PASS WITH NOTES.
- **Agent default:** Pre-read generations ran through 3, newest first; §6 has no limit. H8 published the second fix alone after scratch checks.
- **Agent default:** H6’s second fix was killed (137), likely OOM with four jobs. `--after-crash` passed alone; later runs used at most two. The stale-regression SigLIP stop reran.
- **User / agent default:** Reads ran locally, about 37 min/read versus the spec’s 10 min. CPU work passed 11:00; the user was told at 08:56, with no answer at handoff. GPU work ended at 07:10.
- **User, B1 LOW-1:** Near-duplicate counts were 0/0/72 over 49,121 non-held paintings at cosine ≥ 0.99/0.98/0.92; narrow counts over 912 were 0/0/3 (`final_review/fr_b1_neardup_wide.json`). Two new paintings duplicate train-side artwork below 0.98; none was excluded. `fr_b1.md` and controller notes record the pairs.
- **User, B1 LOW-4:** CSV policy keeps differing styles for six paintings (`results/a28_style_differs.json`; round log/controller notes).
- **User pending, A1 L4 / D9:** “keeps the same sentence” remains open, including after ArtELingo NO-GO.
- **Agent default:** H8 selection agreement reran true: 136 compared, 0 failed. Seven labels were added to the historical 129 items; margins were not compared (`rederive/rd9_out/rd9_H8_selection_agreement_rerun.json`).
- **Agent default, fx-desc-a28:** ArtELingo-28 was refused by a started/pass mode mismatch and H9/H10 int/string seed keys. `10b5a7b` passed 110 tests and 3/3 mutants; verdicts were untouched (`fx_desc_a28_handback.md`).
- **Agent default, provenance:** H6/H7 started at 10:58/10:33 on the code of `b70502b` (that commit, 10:57, added only a started-file copy); write-time SHAs show `10b5a7b`. Changes affect only ArtELingo-28 (run notes 11:14). All five descriptive files were read.
