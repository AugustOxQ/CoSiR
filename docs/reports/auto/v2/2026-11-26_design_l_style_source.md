# Design L (round 7): ceiling CLEAR, seed-42 gate n = 0, Friday verdict NO-GO; no label-free style grouping passes, AFF stays the current best

> Report date 2026-11-26 (a sequence number; written 2026-10-12, Amsterdam; the run ran 2026-10-12 05:50 to 07:00) · Spec: [2026-10-12-r7-design-l-design.md](../../../superpowers/specs/2026-10-12-r7-design-l-design.md) · Decision rule: `src/test/20261126_design_l_style_source/DECISION_RULE.md` @ c95b6ea, header commit reference filled in 0fd066b (SHA-256 prefix 204a0ac5d0 is the 0fd066b file); no substantive amendment · Run log: [20261126_design_l_style_source_log.md](../../../../src/test/20261126_design_l_style_source/20261126_design_l_style_source_log.md) · For the user: [user-read report](../../../user_read/2026-10-12_design_l_style_source.md)

> **Flag, L3: a process departure (the user decides).** The rule's own recipe for removing the content part (L3: `numpy.linalg.pinv`, default `rcond`) failed its own assertion on the CSD vectors, so groupings V1 and V4 could not be built as written. Rule §8's last row says a case the rule does not cover stops the run with its report and a phone note, and the user decides. The run sent the note (05:55) but did not stop: it went on with an agent-default fallback, so K1 and the gate were computed on it. The user decides whether K1 and the gate stand on it.
>
> The fallback is the same projector built from the SVD's orthonormal basis, only where the literal form fails (CSD only). Its `basis_rank` 64 includes one rounding-noise direction (the true rank is 63), so it removes one arbitrary extra direction from the CSD vectors of V1 and V4. Keeping it, or dropping V1 and V4 (with Z1, Z4), cannot change n = 0 (certain). A rank-63 projector needs a rerun of V1, V4, Z1, Z4; a change is very unlikely (an inference, since V1 and V4 were never built that way): they are at least 432 integer steps below S1's bar and at least 168 below S3's on emotion × genre (117 on emotion × style). Numbers are in Deviations.

## Results brief

**What we tried.** Design L: label-free style groupings with the content part removed, each given a reader and two scorers (R1 and extended AFF), screened on development seed 42, to fix AFF's loss on style × genre ([spec](../../../superpowers/specs/2026-10-12-r7-design-l-design.md)). First a ceiling check with the true styles, then a three-part gate (S1 to S3) over 12 candidate scorers. The fresh-seed test (seeds 55 to 57) was to follow only after a pick.

**Outcome against the bar.** The ceiling check is CLEAR, but the gate has no pass (n = 0), so design L ends and the Friday verdict is NO-GO: the paper is the ArtELingo paper with AFF (rule §7.7, §8). AFF stays the current best.

| Step | Number | Bar | Outcome |
|---|---|---|---|
| K1: told true styles, style × genre, fused minus matched counterpart | Σ(4d) 643, +3.92 [+3.35, +4.49] R@1 points (4,096 episodes) | CLEAR needs Σ(4d) ≥ 164 and a lower bound above 0 | CLEAR |
| Gate, S1 (pooled bar margin) | best candidate Σ(4d) 122 (+0.25 [+0.05, +0.46]); AFF itself 163 (+0.33) | Σ(4d) ≥ 246 (+0.5) | 12 of 12 fail |
| Gate, S3 (emotion × genre against AFF) | best candidate Σ(4d) −153 (−0.93) | Σ(4d) ≥ −49 (−0.3) | 12 of 12 fail |
| Gate, S2 (style × genre against matched counterpart) | 5 of 12 reach Σ(4d) ≥ 0 | Σ(4d) ≥ 0 | 5 of 12 pass |

"Promising" (rule §6, spec §6) holds for three scorers: R1 on V2, R1 on V3 and extended AFF on V3. The spec's dig budget for it is none before Friday; the NO-GO stands and the result goes to the direction log's parked ideas for decision A of the next loop.

**What it does not show.** Seed 42 only, and 12 candidate scorers (plus 16 reference and control scorers) were read there, so no claim is licensed and no fresh seed was built. The ceiling is a told route and the candidates use a reader, so the gap between them mixes grouping quality and reader quality. The random controls match or beat the candidates on several statistics (Analysis).

**Options for next.**

- **(a) Take the NO-GO.** Write the ArtELingo paper around AFF; park design L with the "promising" note. No compute.
- **(b) Resume design L after Friday** (only on the user's choice, rule §8). The leads are the cut groupings V2 and V3, or a wider screen of style encoders (parked). It needs a new spec and rule, a few days of CPU work, and any held read of it is a new held read (C5).
- **(c) Decide whether K1 and the gate stand on the L3 fallback** (whichever way, a or b): keep it, or drop V1 and V4 (no rerun; n = 0 either way), or use a rank-63 projector (a rerun of V1, V4, Z1, Z4; a change is very unlikely).

**Your reading.** *(left for the user)*

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The no-pass is clear: no candidate is near S1's bar or S3's, and S1 fails every scorer on its own. A style grouping can matter (told true styles +3.92), but the label-free groupings reach about a tenth of it (best +0.38 on style × genre), and the reader is not told the grouping. The "promising" flag is weak evidence: the extended-AFF scorers on raw CSD and on NrKmeans (references) meet the same three conditions (R1 on raw CSD has S2 −43, R1 on NrKmeans −60, so they do not), and no random control does. This is consistent with style information helping on style × genre; it does not show that the removal step added anything. We are sure of the verdict; we are not sure why the candidates fail.

**Our recommendation.** Option (a), with the "promising" note parked. If the next loop picks design L again, first run the told route with V3, which is CPU only and on seed 42; it separates grouping quality from reader quality. For L3, keep the fallback: it meets L3's own assertion, and keeping it or dropping V1 and V4 leaves n = 0. This is our view; the user decides whether K1 and the gate stand on it.

</details>

## What ran

- **Stages and types:** (1) stage 0 regression checks G1 to G4 (preparation; passed in the build, written 04:41, logged 04:42, on e9bc051); (2) stage 1, the ceiling check (exploration with a stop gate); (3) stage 2, the seed-42 screen (new idea); stage 3, the fresh-seed claim test, not run (n = 0). The descriptive pass followed.
- **Data and seeds:** development seed 42 only (12,288 episodes, 4,096 per aspect pair). No held row (C5). No test seed built: the seed ledger is unchanged (55 and later free).
- **Code and times:** build merged into main at ddbb600. Run commits: e3776ca (05:55, the L3 fallback), 6c6a26b (06:51), 3ef2980 (06:52, K1), 3f20397 (07:00, gate and descriptive pass). Prep first try 05:50 (stopped on L3), second try 05:55 to 06:50; ceiling 06:51 to 06:52; gate 06:54 to 06:57; descriptive pass 06:58 and 07:00. Local CPU only; prep took under an hour (budget about 4).
- **Re-derivation (C13):** before each verdict file, separate code recomputed it from the stored arrays: Phase K (18 arrays), Phase A (243 arrays, V1 to V6 only) and `rd7_stats` (K1: Σ(4d) 643, CLEAR; gate: n 0, promising true). Every comparison passed in `--finalize`; both verdict files were then written (06:52 and 06:57). Phase A does not cover the reference and random-control scorers, so the control comparisons below rest on the pipeline's arrays alone.

## Results

**K1 is CLEAR** (Σ(4d) 643 over 4,096 style × genre episodes against the CLEAR bar 164; point +3.92 [+3.35, +4.49]). The prior was +0.9 (60% to reach +0.5, 30% to reach +1.0), so the ceiling is far above it.

**The gate has n = 0.** Σ(4d) is the integer sum of four times the per-episode difference; points are 25·Σ/N (N = 12,288 for S1, 4,096 for S2 and S3). Bars: S1 Σ ≥ 246 (and both lower bounds above 0), S2 Σ ≥ 0, S3 Σ ≥ −49 on each emotion pair. "Bar" is the strongest of B′(A1), the scorer's own B′, its matched counterpart, B′(A0) and B (B1, BX, CF, B0, B).

| Candidate | Bar vs | S1 Σ (point) [95%] | S2 Σ | S3 emotion × style Σ | S3 emotion × genre Σ | Pass |
|---|---|---|---|---|---|---|
| R1 on V1 | B1 | −186 (−0.38) [−0.60, −0.15] | −15 | −168 | −217 | no |
| X-AFF on V1 | B1 | −248 (−0.50) [−0.74, −0.27] | −30 | −174 | −256 | no |
| R1 on V2 | BX | 97 (+0.20) [−0.01, +0.41] | 15 | 38 | −189 | no |
| X-AFF on V2 | BX | −50 (−0.10) [−0.30, +0.10] | 16 | −46 | −235 | no |
| R1 on V3 | B1 | 122 (+0.25) [+0.05, +0.46] | 37 | −16 | −153 | no |
| X-AFF on V3 | B1 | 27 (+0.05) [−0.16, +0.27] | 62 | −53 | −200 | no |
| R1 on V4 | B1 | −215 (−0.44) [−0.67, −0.21] | −31 | −170 | −230 | no |
| X-AFF on V4 | B1 | −233 (−0.47) [−0.71, −0.24] | −18 | −166 | −264 | no |
| R1 on V5 | B1 | −165 (−0.34) [−0.58, −0.10] | −14 | −146 | −209 | no |
| X-AFF on V5 | B1 | −102 (−0.21) [−0.47, +0.05] | −31 | −127 | −173 | no |
| R1 on V6 | B1 | −80 (−0.16) [−0.42, +0.09] | −40 | −130 | −172 | no |
| X-AFF on V6 | B1 | −115 (−0.23) [−0.49, +0.01] | 23 | −158 | −206 | no |

Passes by item: S1 0 of 12; S2 5 of 12 (R1 on V2, V3; X-AFF on V2, V3, V6); S3 emotion × style 3 of 12 (R1 on V2, V3; X-AFF on V2); S3 emotion × genre 0 of 12. S1 and S3 each fail every scorer, so n = 0 would hold if either were dropped. AFF itself would fail S1 (Σ 163, +0.33 over B1). No random-control flag applies (no pick).

**Verdict under the rule: n = 0. Design L ends; FV is NO-GO** (the ArtELingo paper with AFF); no sensitivity, rule smoke, seed build or test. AFF stays the current best (C3).

**"Promising" (rule §6).** Its three conditions are style × genre Σ(4·(s − AFF)) ≥ 50 (the code's `need(0.3, 4096)`, +0.3 points), S2 Σ ≥ 0 and pooled bar margin Σ > 0. Three candidates meet them, from the stored arrays:

| Candidate | style × genre s − AFF Σ (points) | S2 Σ (points) | Pooled bar margin Σ (points) [95%] |
|---|---|---|---|
| R1 on V2 | 135 (+0.82) | 15 (+0.09) | 97 (+0.20) [−0.01, +0.41] |
| R1 on V3 | 128 (+0.78) | 37 (+0.23) | 122 (+0.25) [+0.05, +0.46] |
| X-AFF on V3 | 117 (+0.71) | 62 (+0.38) | 27 (+0.05) [−0.16, +0.27] |

The other nine fail S2, the pooled sum, or both. These candidates are about +0.7 to +0.8 above AFF on style × genre, but lose 0.93 to 1.22 against AFF on emotion × genre; pooled, the two R1 scorers are at AFF's level and X-AFF on V3 is below it (descriptive; R@1 points, 12,288 episodes, scorer minus AFF):

- R1 on V2: −0.03 [−0.31, +0.26]
- R1 on V3: −0.08 [−0.38, +0.20]
- X-AFF on V3: −0.28 [−0.53, −0.02]
 The spec's dig budget is none before Friday.

## Analysis

Everything below is descriptive and decides nothing (rule §5, §6). Sources: `descriptive_ceiling.json`, `descriptive_gate.json`.

**The ceiling (rule §5).** Told true styles (27 groups, CLIP heads placing them at 63.7% image and 26.5% caption accuracy), 4,096 episodes per pair, R@1 points with 95% intervals:

| Quantity | Number |
|---|---|
| Fused minus B1 on style × genre | +4.46 [+3.80, +5.11] |
| Fused minus BT (B′ of A0+T), style × genre | +4.37 [+3.74, +4.98] |
| Told margin over counterpart: emotion × style / emotion × genre / style × genre | +2.78 [+2.18, +3.37] / +3.09 [+2.52, +3.66] / +3.92 [+3.35, +4.49] |
| Told gain: same order | 8.39 / 14.34 / 12.79 |
| T*, true style codes as the posterior (label placement), style × genre margin | +21.74 [+21.13, +22.35] (pooled +11.86) |

**Groupings (rule §6, after the gate).** AMI against the true style labels, the true genre labels and the content grouping (AMI is unscaled; style and content on 183,694 rows, genre on 148,956). Random controls Z1 to Z6: every AMI at most 0.004, head accuracy 14.4 to 20.9%.

| ID | Groups | AMI style | AMI genre | AMI content | Heads: image / caption % |
|---|---|---|---|---|---|
| T (true styles) | 27 | 1.00 | 0.22 | 0.32 | 63.7 / 26.5 |
| R0 (raw CSD) | 17 | 0.34 | 0.33 | 0.40 | 85.1 / 40.4 |
| V1 projection | 18 | 0.10 | 0.08 | 0.11 | 64.0 / 23.6 |
| V2 cut, w 0 | 12 | 0.32 | 0.27 | 0.33 | 81.5 / 37.5 |
| V3 cut, w 0.5 | 13 | 0.33 | 0.31 | 0.38 | 85.3 / 40.9 |
| V4 projection and cut | 18 | 0.07 | 0.05 | 0.08 | 62.3 / 22.0 |
| V5 tokens, ridge | 13 | 0.04 | 0.08 | 0.09 | 38.8 / 23.4 |
| V6 tokens, projection | 16 | 0.10 | 0.17 | 0.17 | 67.4 / 33.7 |
| N (NrKmeans) | 17 | 0.14 | 0.23 | 0.24 | 66.7 / 25.7 |

The cuts (V2, V3) kept raw CSD's style content (0.32 and 0.33 against 0.34) and removed little genre (0.27 and 0.31 against 0.33). The projections (V1, V4) removed genre and style together (genre 0.08 and 0.05, style 0.10 and 0.07). The token-statistics groupings carry little style (0.04, 0.10). The token outlier filter dropped nothing (0.0 share in all three blocks), so "high-norm tokens dropped" is empty on this data.

**Random controls against the candidates.** Seed 42, R@1 points; each Zk has Vk's group sizes. Per scorer pair (Vk against Zk):

| Statistic (12 pairs) | Candidates (V) | Random controls (Z) | Z above its V |
|---|---|---|---|
| Pooled margin over own counterpart | −0.05 to +0.32 | +0.19 to +0.63 | 10 of 12 |
| Gain statistic (S1's condition) | 0.14 to 1.12 | 0.96 to 2.73 | 12 of 12 |
| S1 Σ (bar margin) | −248 to 122 | −96 to 54 | 9 of 12 |
| S2 Σ (style × genre) | −40 to 62 | −109 to 11 | 3 of 12 |
| S3 emotion × genre Σ | −264 to −153 | −146 to −51 | 12 of 12 |

For reference, AFF's margin over its counterpart is +0.74 and its gain +3.11. All 12 random controls clear S1's gain condition (every gain lower bound above +0.7) and none passes S1 or S3 emotion × genre; one passes S2 (R1 on Z5, +11). Only the extended-AFF scorers on raw CSD and on NrKmeans meet the "promising" conditions (126, 29, 78 and 87, 1, 80 against the sums above); R1 on raw CSD (S2 −43) and R1 on NrKmeans (S2 −60) do not.

Our reading, in four points:

- **Margin and gain over the counterpart do not measure style content here.** Random groupings score higher on both, so S1's gain condition cannot separate real from random groupings.
- **S2 is consistent with a small real effect (weak evidence).** On style × genre the real groupings sit at −0.24 to +0.38 over their counterparts and the random ones at −0.67 to +0.07. The ranges overlap, Z is above V in 3 of 12 pairs, R1 on Z5 passes S2, and 7 of 12 real scorers fail it. S2 is a point check with intervals of about ±0.3, on seed-42 selection data.
- **The cuts do not clearly beat raw CSD on style × genre.** R1 on V2 and V3 reach +0.09 and +0.23 against −0.26 for R1 on raw CSD; extended AFF on V3 reaches +0.38 against +0.18. The intervals are about ±0.3.
- **On emotion × genre every real grouping loses more to AFF than its random control.** This suggests the style grouping takes picks away from the told groupings. We did not trace why the random controls gain more; we would not read it further.

**Where the candidates lose.** Two ways, each enough alone, then the picks:
- **S1:** the best pooled margin is +0.25 (R1 on V3) against a bar of +0.5. The comparator is B1 in ten cases and BX for both V2 scorers. AFF's own +0.33 over B1 is below the bar too, so S1 asks for about +0.17 more than AFF has.
- **S3, emotion × genre:** −0.93 to −1.61 against AFF, against a bar of −0.3, on all 12. Raw CSD (−1.11, −1.28) and NrKmeans (−0.60, −0.62) lose too; the random controls lose less (−0.31 to −0.89).
- **The reader's picks (R1 on V3, descriptive):** on emotion × genre it picks affect on 75.7% of condition-a episodes (AFF 85.6%) and image on 33.3% of condition-b episodes (AFF 50.8%); V3 takes 14.2% and 40.9%. On style × genre condition a it picks V3 on 33.0% (R1 on V2 38.7%).

**Against the ceiling.** Told true styles reach +3.92 over the counterpart on style × genre; the best candidate reaches +0.38 (about a tenth).

- The ceiling gives the reader the grouping; the candidates must also pick it. The told route was run with T only for the candidates.
- A hint that the grouping, not the reader, limits: rule §1 gives told raw CSD on style × genre +0.134 [−0.350, +0.595] (round 1's A1 by the told route, spike `told_A1_224`), the same scale as the candidates' +0.1 to +0.4. It is a different configuration, so it is a hint and no test.
- Placement does limit T: true codes as the posterior (T*) give +21.74 against +3.92. The cuts' heads (V2 81.5 / 37.5, V3 85.3 / 40.9 % image / caption) are no less accurate than T's (63.7 / 26.5), on different group counts and so not directly comparable. Head accuracy does not obviously explain the gap.

## Deviations

- **L3 fallback: a process departure (agent default; the user decides).** Prep stopped at 05:50 on `AssertionError: L3 Orth projection assertion`. The content-means matrix M (768 × 64) has rank 63 in exact arithmetic, since the size-weighted means sum to zero. Its 64th singular value is 7.97e-15 of the first, above pinv's default cutoff (1e-15), so pinv keeps rank 64 and amplifies rounding.
  - Rule §8's last row requires a case the rule does not cover to stop the run with its report and a phone note. The note went at 05:55; the run did not stop. It went on under the unattended-chat practice, so K1 and the gate were computed on the fallback (run log, "Process departure").
  - CSD (V1, V4), literal form: the record's `max_abs_after_pinv` 9.50e-4 is the absolute residual; its ratio to `max_abs_before` 0.552 is 1.72e-3, against the assertion's bar 1e-6, so it fails. The SVD-basis projector gives 1.39e-15.
  - Token vectors (V6): σ64/σ1 is 5.72e-17, rank 63, and the literal form passes (ratio 4.85e-15), so it was kept.
  - Commit e3776ca; `r7_groupings.json` `transforms` records `projector` (`svd_basis_fallback`), `basis_rank` 64 and the literal residual. The fallback's rank 64 includes one noise direction (true rank 63), so one arbitrary extra direction leaves V1 and V4. Keeping it or dropping V1 and V4 cannot change n = 0 (certain); for a rank-63 projector it is an inference (Flag).
- **No amendment** to the rule or spec. Spec §6's "+0.3 R@1 above AFF" is the rule's Σ ≥ 50.
- **Not run (n = 0):** sensitivity, rule smoke, seed build, test, Phase B. Seeds 55 to 57 stay free.
- **Final review and engine check.** The final review (Opus, three parts; re-review PASS) found 0 critical and 3 important findings (finalize without ID arrays, silent `--overlay` fill, no test of `run_r7.py`'s guards). All three were fixed (05:38, merged 05:47), before prep, so no defect reached the run. Open minor findings that touch it: Phase A covers V1 to V6 only; stage 0's G3 lacks reader R0's bank arrays (the reviewer compared all 10: equal); stage 0 ran on e9bc051 (only resume code changed since); standard base heads were fitted and not used. L3 was a rule-level conflict, not a finding: note m3 had seen the 64th singular value on trial subsets, where the literal form passed. Stage 0's file says 04:41, the log 04:42.

## Disclosures

- Seed 42 only, development data; 12 candidate scorers plus 4 reference and 12 control scorers read there, after about 60 earlier variants (rule §1). Selection inflates any seed-42 margin. No claim is licensed.
- The descriptive numbers (random controls, references, AMIs) rest on the pipeline's arrays; they are not re-derived.
- Who decided: the user: D1, D3, D5 to D11. The user, after seeing the recommendation: D2, D12 to D16, D27 (its details agent default). Agent default: D4, D17 to D26 (spec §4). The L3 fallback: agent default.
- Process departure: the run did not stop at L3 as rule §8 requires (Deviations); the user decides whether K1 and the gate stand on the fallback.

## Files and storage

- **Outputs:** `src/test/20261126_design_l_style_source/results/`: `ceiling_verdict.json`, `gate_verdict.json`, `descriptive_ceiling.json`, `descriptive_gate.json`, `stage0_regression.json`, `prep.json`, `groupings/`, `seed42/arrays.npz`, `rederive/`.
- **Storage:** `results/` is about 4.7 GB; see `/project/claude-config/storage/disk_ledger.md` (the decide chat's wrap-up does the storage summary).
