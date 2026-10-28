# 40 graph neighbours give held clusters above the instructed Qwen image on emotion and style; genre is below by less than one seed range (0.0128)

> 2026-10-28 (sequence name 12-28) · Spec: [design](../../../superpowers/specs/2026-10-27-recipe-sweep-design.md) · Decision rule: [binding](../../../../src/test/20261027_recipe_sweep/DECISION_RULE.md) @ 506c40c (first commit 35fcf68) · Run log: [round log](../../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md) · Previous round: [12-27](2026-12-27_relational_genre.md)

## Results brief

**What we tried.** We tested the held emotion claim of G, the previous recipe with a sharper genre teacher row ([spec](../../../superpowers/specs/2026-10-27-recipe-sweep-design.md)).
Then we swept G's settings, confirmed three candidates and tested the pick, c*, once on held paintings.
c* changes only the graph neighbours, from 20 to 40. Last, we removed each of c*'s eight parts in turn.

**Outcome against the bar.** c* replaces G under the rule.
On 12,281 held paintings (genre labels on 9,949) and 5 seeds, c* beats the instructed Qwen image on emotion and style.
Genre is below the instructed image by less than one seed range (0.0128), with its whole interval below zero.

| Aspect / comparator T | c* AMI | Comparator AMI | Difference [95%] |
|---|---|---|---|
| emotion | 0.1066 | 0.0921 | +0.0145 [+0.0113, +0.0177] |
| style | 0.4066 | 0.4026 | +0.0041 [+0.0004, +0.0081] |
| genre | 0.5001 | 0.5062 | -0.0060 [-0.0103, -0.0015] |
| emotion / PercepT core | 0.1066 | 0.0874 | +0.0192 [+0.0137, +0.0207] |

Key: T: the instructed Qwen image of each aspect; PercepT core: the project baseline.

- The style win is narrow: its lower bound is about 0.05 seed ranges above zero.
- G passed the emotion claim against PercepT core at both scales: full +0.0164 [+0.0107, +0.0182], sample +0.0188 [+0.0137, +0.0200].
- Held ablation: the relational term carries genre (without it, genre AMI changes by -0.0440 [-0.0512, -0.0372]). The instructed graphs, the buddy terms and the content correction carry emotion and style.
- Without the direct teacher or the caption endpoints, held emotion or style rises.

**What it does not show.** The sweep's seed-0 gains are inflated, because we picked among about 60 configurations.
Seeds 1 to 3 are new to these runs, but rounds 12-25 and 12-27 also picked on them.
The held split and T were read in earlier rounds, and G's known held emotion AMI (0.1039) made stage 0 close to known.
The claims cover k-means, the Qwen image encoder and ArtELingo only. The ablation decides nothing, and there is no retrieval result yet.

**Options for next.**

1. Drop the caption endpoints from c*. On development paintings, c* without them gains +0.315 [+0.121, +0.534] seed ranges over c*. Its held numbers already exist as an ablation arm, so a held test would be close to known; it costs under an hour.
2. Test the recipe with SigLIP 2, SigLIP 1 and CLIP L/14. It costs about half a day of GPU, and node402 is idle until about 2026-10-30 23:30. It shows whether the recipe transfers to other image encoders.
3. Draft the CVPR abstract (due 2026-11-10) from the supported claims. It costs writing time and fixes what the paper can claim.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** We trust the emotion result most: c* beats T by about three seed ranges and PercepT core by +0.0192.
The style win passes the rule but is narrow, and c*'s held style is level with G's (+0.0009, interval across zero).
Genre is within the agreed margin of T, not a win.
The ablation is clear for the large parts: each of the instructed graphs, the buddy terms and the content correction is worth many seed ranges on style.
The direct teacher and the caption endpoints look unneeded, since development and held paintings agree on the direction.
Our priors fit: stage 0 GO at 0.85 (sample) and 0.9 (full), a pick at 0.5; given a pick, a held win at 0.9 for emotion, 0.4 for style and 0.15 for genre.

**Our recommendation.** Take option 1 first, because it is small and may become the recipe for the other tests.
Then run option 2 on node402 before the reservation ends. Draft the abstract (option 3) in parallel, since the emotion claim holds with either recipe.

</details>

## What ran

Development chose c* before held testing.

| Data | Paintings / role | Seeds |
|---|---|---|
| Sample scale | 13,400 fit; 1,489 stopping | stage 0: 0 to 4; sweeps: 0 |
| Development P | 29,232; genre labels on 23,682 | confirmation: 1 to 3 |
| Full scale | 42,969 train; 6,152 validation | 0 to 4 |
| Held | 12,281; genre labels on 9,949 | 0 to 4 |

G reuses runs except sample seed 4 and sample held inference; ArtELingo-28 is untouched.

| Stage / type | Time, 2026-10-27 | Machine |
|---|---|---|
| 0 / claim test, then G analysis | 17:08 to 17:20 | local RTX 3090 |
| 1 / sweep | 17:08 to 18:29 | node402 |
| 2 / sweep | 18:30 to 19:39 | node402 |
| 3 / confirmation with pick | 19:39 to 20:13 | local |
| 4 / claim test, then c* analysis | 20:14 to 21:09 | local |
| Ablation / development, then held | development 21:11 to 22:50; held from 22:50 | local |

- Code: eb1d04b, launched a87cb1d; node402: three RTX A6000s; local: 8 CPU threads.
- Measurement: image heads, k-means with 20 clusters and three states; paired 95% intervals, 2,000 painting/seed draws.
- Stage 3 shares draws across aspects. Diagnostics choose nothing.

## Results

All completed checks pass.

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| V0b | check | 53; 0 differences | outputs | ≤1e-6 | pass |
| G-ES | sample margin | +0.0188 | PercepT core | ≥0.0100; lower >0 | G-ES-GO |
| G-EF | full margin | +0.0164 | PercepT core | same | G-EF-GO |
| E0 | sentence | true | both scales | full GO; sample not NO | pass |
| GRID | knob gains | +0.924/+0.371/+0.288/+0.164 | node G | top four | GRID-4 |
| TOP | pool/pick | 21/3 | guarded runs | mean gain | TOP |
| V1c | check | 41 then 59 runs; 0 differences | outputs | equal; ≤1e-6 | pass |
| N3 | repeat | bit-for-bit equal | node G | half range | pass |
| PICK | mean gain | +0.683 | local G | aspect lower >−r; mean lower >0 | PICK (k11=40) |
| V3c | check | 213; 0 differences | outputs | ≤1e-6 | pass |
| T-WIN / emotion | held margin | +0.0145 | T | lower >0 | T-WIN |
| T-WIN / style | held margin | +0.0041 | T | lower >0 | T-WIN |
| T-LEVEL / genre | held margin | -0.0060 | T | lower >−r | T-LEVEL |
| PE | emotion margin | +0.0192 | B1-own | ≥0.0100; lower >0 | PE-GO |
| V4b | check | 66; 0 differences | outputs | ≤1e-6 | pass |

Key: r: seed range, emotion 0.0046/style 0.0082/genre 0.0128; B1-own: PercepT core P3, own clustering.

**Verdict under the rule:** c* replaces G. Held outcomes trigger no next step automatically.

### Stage 1: node402 knob curves

Graph neighbours 40 leads. Guard: no aspect loses more than half a seed range against node G.

The four largest knob gains, in seed ranges over node G, formed the stage 2 grid:

- graph neighbours 40: +0.924; teacher row temperature 0.035: +0.371;
- style relational weight 8: +0.288; trunk width 512: +0.164.

The fifth was a lower direct teacher weight (0.0625: +0.158). Patience changed the mean gain by at most 0.001. All 13 curves, with guard failures: [run log, stage 1](../../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md#stage-1-single-knob-sweep-on-node402-2026-10-27).

Node parity is recorded, not a gate; no decision compares node with local (rule exception).

- Node G minus local seed 0: emotion +0.0003; style +0.0020; genre -0.0077, about 0.6 genre seed ranges. Scorer parity: zero.

### Stage 2: node402 top runs

The graph change leads after 18 new runs.

| Rank / run | Emotion AMI | Style AMI | Genre AMI | Mean gain |
|---|---|---|---|---|
| 1 / k11=40 | 0.1107 | 0.4090 | 0.5060 | +0.924 |
| 2 / k2=2+k9=512+k11=40 | 0.1103 | 0.4031 | 0.5144 | +0.874 |
| 3 / k4=0.035+k11=40 | 0.1111 | 0.4018 | 0.5093 | +0.751 |

All three pass. Teacher row 0.035 alone: +0.371; neighbours 40 alone: +0.924; together: +0.751.

### Stage 3: local confirmation

Graph neighbours 40 wins.

| Run | Emotion mean / gain [95%] | Style mean / gain [95%] | Genre mean / gain [95%] | Mean gain [95%] | Qualifies |
|---|---|---|---|---|---|
| k11=40 | 0.1089 / +0.0026 [+0.0011, +0.0045] | 0.4072 / +0.0141 [+0.0121, +0.0160] | 0.5019 / -0.0031 [-0.0090, +0.0044] | +0.683 [+0.452, +0.959] | yes |
| k2=2+k9=512+k11=40 | 0.1085 / +0.0023 [+0.0009, +0.0036] | 0.4020 / +0.0089 [+0.0035, +0.0135] | 0.4990 / -0.0060 [-0.0098, -0.0012] | +0.369 [+0.001, +0.724] | yes |
| k4=0.035+k11=40 | 0.1082 / +0.0020 | 0.3991 / +0.0060 | 0.5006 / -0.0045 | +0.272 [-0.187, +0.714] | no: mean lower <0 |

G means (emotion/style/genre): 0.1062/0.3931/0.5050. c*'s genre is not distinguishable from G (interval crosses zero).
Lead: 0.314 seed ranges; half-width: 0.253. The mean-gain intervals overlap.
The rule only reports this; the pick is the run with the largest mean gain.

### Stage 4: held, full scale

Emotion/style intervals are above zero; genre's is below zero.

| Aspect | c* / T AMI | Difference [95%] | d/r | Rule outcome | Leiden AMI |
|---|---|---|---|---|---|
| emotion | 0.1066 / 0.0921 | +0.0145 [+0.0113, +0.0177] | 3.155 | T-WIN | not computed |
| style | 0.4066 / 0.4026 | +0.0041 [+0.0004, +0.0081] | 0.497 | T-WIN | not computed |
| genre | 0.5001 / 0.5062 | -0.0060 [-0.0103, -0.0015] | -0.472 | T-LEVEL | 0.5082 |

Key: d/r: AMI difference divided by seed range.

Leiden genre gains +0.0627 [+0.0529, +0.0736] over PercepT B4-r20 (0.4454), its paper-settings P1 entry.
Spec §3.4 says Leiden beside stage 4; rule §3.4 narrows it to genre, as in 12-27. The rule governs.

Reported comparisons:

| Aspect | G AMI | c* minus G [95%] | Best PercepT entry / AMI | c* minus PercepT [95%] |
|---|---|---|---|---|
| emotion | 0.1039 | +0.0027 [+0.0001, +0.0054] | B1-own / 0.0874 | +0.0192 [+0.0137, +0.0207] |
| style | 0.4057 | +0.0009 [-0.0044, +0.0065] | B6-own / 0.2297 | +0.1769 [+0.1663, +0.1809] |
| genre | 0.5001 | 0.0000 [-0.0028, +0.0030] | B5-r20 / 0.4269 | +0.0732 [+0.0622, +0.0796] |

Key: B5-r20: PercepT paper P4, reclustered at 20; B6-own: core P2, own clustering.

c* and G round to genre 0.5001; held outputs differ in every head.
Held style minus G: +0.0009 [-0.0044, +0.0065], crossing zero.

### Ablation

On development paintings, removing the buddy terms, the instructed graphs or the content correction from c* lowers emotion and style; removing the relational term lowers genre.
The ablation decides nothing.

**Development.** Each removal ran at seeds 1 to 3 on P, against c*'s stage 3 runs (emotion/style/genre 0.1089/0.4072/0.5019), with 2,000 joint draws per removal.
All eight removals ran; the separate re-derivation agrees on all 442 items.

| Removal | Emotion ē [95%] | Style ē [95%] | Genre ē [95%] | Mean gain [95%] |
|---|---|---|---|---|
| relational term | -0.0009 [-0.0035, +0.0013] | -0.0008 [-0.0050, +0.0052] | -0.0297 [-0.0339, -0.0258] | -0.869 [-1.150, -0.574] |
| sharper teacher row | +0.0002 [-0.0015, +0.0018] | -0.0063 [-0.0134, -0.0006] | +0.0003 [-0.0082, +0.0073] | -0.233 [-0.355, -0.111] |
| direct teacher | +0.0022 [+0.0005, +0.0037] | -0.0007 [-0.0026, +0.0014] | +0.0011 [-0.0051, +0.0057] | +0.165 [-0.002, +0.327] |
| buddy terms | -0.0292 [-0.0319, -0.0264] | -0.1555 [-0.1591, -0.1506] | -0.0001 [-0.0020, +0.0018] | -8.441 [-8.733, -8.067] |
| caption endpoints | 0.0000 [-0.0016, +0.0017] | +0.0042 [+0.0013, +0.0072] | +0.0056 [-0.0018, +0.0155] | +0.315 [+0.121, +0.534] |
| content correction | -0.0103 [-0.0131, -0.0074] | -0.0678 [-0.0737, -0.0599] | +0.0006 [-0.0039, +0.0067] | -3.491 [-3.896, -3.097] |
| instructed graphs | -0.0346 [-0.0381, -0.0315] | -0.1994 [-0.2029, -0.1948] | +0.0015 [-0.0002, +0.0032] | -10.573 [-10.853, -10.258] |
| shared trunk | 0.0000 [-0.0025, +0.0023] | -0.0015 [-0.0058, +0.0019] | +0.0048 [+0.0029, +0.0067] | +0.062 [-0.143, +0.276] |

Key: ē: removal minus c*, mean AMI over seeds 1 to 3; mean gain: ē divided by the seed range, averaged over aspects.

- On P, removing the instructed graphs from c* changes style AMI by -0.1994 [-0.2029, -0.1948], the largest change.
- Three removals raise one aspect with the interval above zero: caption endpoints (style), shared trunk (genre) and direct teacher (emotion).

**Held.** Each removal ran at full scale, seeds 0 to 4, on the 12,281 held paintings (genre labels on 9,949), against c*'s stage 4 runs (emotion/style/genre 0.1066/0.4066/0.5001), with 2,000 draws per aspect.
All eight removals ran; the separate re-derivation agrees on all 384 items.

| Removal | Emotion AMI / Δ [95%] | Style AMI / Δ [95%] | Genre AMI / Δ [95%] |
|---|---|---|---|
| relational term | 0.1075 / +0.0009 [-0.0013, +0.0032] | 0.4066 / 0.0000 [-0.0054, +0.0048] | 0.4561 / -0.0440 [-0.0512, -0.0372] |
| sharper teacher row | 0.1082 / +0.0016 [-0.0015, +0.0046] | 0.4075 / +0.0008 [-0.0030, +0.0048] | 0.4963 / -0.0038 [-0.0074, -0.0002] |
| direct teacher | 0.1101 / +0.0035 [+0.0008, +0.0056] | 0.4108 / +0.0041 [+0.0016, +0.0067] | 0.4970 / -0.0031 [-0.0080, +0.0014] |
| buddy terms | 0.0788 / -0.0277 [-0.0305, -0.0250] | 0.2554 / -0.1512 [-0.1547, -0.1451] | 0.4986 / -0.0015 [-0.0059, +0.0027] |
| caption endpoints | 0.1075 / +0.0009 [-0.0024, +0.0044] | 0.4138 / +0.0072 [+0.0015, +0.0121] | 0.4967 / -0.0034 [-0.0119, +0.0045] |
| content correction | 0.0976 / -0.0090 [-0.0119, -0.0062] | 0.3315 / -0.0752 [-0.0814, -0.0679] | 0.5073 / +0.0072 [+0.0030, +0.0113] |
| instructed graphs | 0.0739 / -0.0327 [-0.0361, -0.0292] | 0.2163 / -0.1903 [-0.1959, -0.1822] | 0.5055 / +0.0054 [-0.0014, +0.0132] |
| shared trunk | 0.1065 / -0.0001 [-0.0025, +0.0027] | 0.4090 / +0.0023 [-0.0022, +0.0069] | 0.5000 / -0.0001 [-0.0057, +0.0059] |

Key: AMI: the removal's held AMI; Δ: removal minus c*.

- On held paintings, removing the relational term from c* changes genre AMI by -0.0440 [-0.0512, -0.0372]. No other removal changes genre by more than 0.0072.
- Removing the instructed graphs changes style AMI by -0.1903 [-0.1959, -0.1822] and emotion AMI by -0.0327 [-0.0361, -0.0292]. The buddy terms and the content correction act in the same direction.
- Removing the direct teacher raises emotion and style; removing the caption endpoints raises style. All three intervals are above zero.
- On development paintings, the same removals raise the same aspect with the interval above zero: the direct teacher on emotion, the caption endpoints on style.
- Removing the content correction raises held genre by +0.0072 [+0.0030, +0.0113]; on development paintings the change is +0.0006, with the interval across zero.
- Removing the shared trunk changes no aspect by more than 0.0023, with every interval across zero.

## Analysis

Emotion-head AMI on genre falls from 0.4048 to 0.3889. Each label set favours its named head.

| Head | Emotion labels, G → c* | Style labels, G → c* | Genre labels, G → c* |
|---|---|---|---|
| emotion | 0.1039 → 0.1066 | 0.1647 → 0.1641 | 0.4048 → 0.3889 |
| style | 0.0622 → 0.0610 | 0.4057 → 0.4066 | 0.2983 → 0.2903 |
| genre | 0.0695 → 0.0701 | 0.2668 → 0.2720 | 0.5001 → 0.5001 |

The heads agree slightly less with each other in c* than in G (emotion-genre head pair 0.4234 → 0.4066; head-pair table in the run log).

Class F1 shifts both ways; teacher overlap rises for style/genre and falls for emotion.
See [F1 and geometry/overlap tables](../../../../src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md#diagnostic-detail-completed-after-the-relevant-verdicts).

**Our reading.** Broader positives could explain the development style gain; this and gallery interpretation are **not checked**.

## Deviations

The stage 2 watch timed out, and the ablation took longer than estimated.

- User exception, 15:31: node start 2026-10-27 17:08; end-by 2026-10-28 01:08.
- At 19:27, stage 2's watch hit its 30-minute silence limit. Verification kept running; the chat used `--stall-minutes 120` and finished the ordered tail by hand. No rerun or changed result.
- Metadata: stale scorer hash, unread by guards (queue note e); fixed `v4b: pending` and printed `None` precede the passing held check. Commit fields track documentation; hashes bind code.
- Other stages: none beyond the node parity exception.

The ablation took longer than estimated: development 63 GPU minutes against 17, held 252 against 80 (per-removal times in the run log).
The chain ended at 03:53, before 11:00 and outside weekday working hours, but after the planned end (23:00) and report time (03:00).

## Files and storage

Outputs: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/`.

- Files: verdicts/checks, `sweep/node402/`, `scores/node402/`, `gates/`, `stage4/{reported.json,leiden/}`, `analysis/{G,k11=40}/analysis.json`, statuses, `logs/`; ledger H20/H21.
- `node402/train/`: 1.9 GB; keep reference, shortlist, confirmation, held and ablation runs.
- Would remove: regression weights `v1a_s0`, `v1a_s1`, `v4a_run`, unkept node runs and scratch after review. Nothing removed.

Storage: decided at the wrap-up; see `.scratch/pending_deletions.md` and the disk ledger `/project/claude-config/storage/disk_ledger.md`.
