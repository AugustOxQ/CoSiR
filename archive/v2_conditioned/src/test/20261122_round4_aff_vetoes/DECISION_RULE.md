# Decision rule: reader fix, round 4 (vetoes on AFF's gate, developed on seed 42, fresh-seed test), committed before any code

**Written** 2026-10-07 from 01:08 (Amsterdam), from the approved spec
`docs/superpowers/specs/2026-10-07-round4-aff-vetoes-design.md` (commit 7f0a00c, SHA-256
d1584d8ccb3f5d7a2862351aed5a15236aa423c1f95969dada18d57c7a457d5b), before any implementation script of this folder
exists. The only numbers it states are earlier rounds' and the brainstorm's, which it reuses, and two seed-42 constants
computed from stored seed-42 arrays while it was written (v₇₅ of D5 and the mean R@1 of B′(A1) of D4).

**Precedence.** Where this file differs from the spec, from round 3's rule
(`src/test/20261121_round3_affect_gate/DECISION_RULE.md`, SHA-256
2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925), from round 2's rule or from round 1's rule, **this
file governs**. Round 3's rule is part of this file by reference: every section of it that this file cites, with the
text of that section unchanged unless this file says otherwise (§2, D1 to D12, D15 with its text, §4, §5 items 1 to 3,
§6.1, §6.2, §6.11, §7 item 7, §9 and §10). In particular, D15's build records `results/build_seed{s}.json` are written
for seeds 52 to 54 and asserted by every later step. Round 3's D1 sentence "The csd grouping and configuration A1 are
not used in this round" does not apply. The earlier rules and the spec are otherwise cited for provenance only.

**Checked** before its commit by a fresh Opus reviewer (spec §7; not an ARS round; report `rule_check/opus_rule_check.md`).
It found 1 blocking (round 3's seed guard refuses seeds 52 to 54), 7 should-fix and 14 nit findings, all applied before
the commit. It verified every SHA-256 of D11 and of round 3's D15, and reproduced through the float32 path, at full
precision, v₇₅ and its count, B′(A1), the A1 reader's stored probabilities and picks, R1, AFF and `IMGABST_q75`, without
computing any number of V4, V2 or V24.

**Status.** Three candidates built on AFF (V4, V2, V24; §3 D6) in one development family on seed 42 (§5), after
regression checks of the code on seed 42. At most one candidate is carried; it alone is tested on the fresh episode
seeds 52, 53 and 54 (§6), the only confirmatory step. AFF, frozen as tested in round 3, runs beside it and enters one
GO check (the paired check against AFF).

**Authorisation.** The user's decisions of 2026-10-07 (handoff
`docs/superpowers/handoffs/2026-10-07-method-improvements-handoff.md` §2, and the spec's open points settled one at a
time in chat between 00:20 and 01:00, both design sections approved, the spec approved and committed at 01:02
(7f0a00c)): round 3's GO
stands and AFF is the current best; option B, improve the method with the brainstorm's remaining ideas before the
held-split test; candidates = idea 4, idea 2 and their combination, each on AFF with its own matched counterpart, idea 3
later as its own step; development on seed 42 with round 2's bar and a carry rule, test on seeds 52 to 54; a candidate
is carried only if it beats AFF on seed 42 (paired point above 0) and the paired check against AFF is a GO check; a
candidate that reads CSD has B′(A1) as its floor, an A0-only candidate B′(A0), and B′(A1) is reported beside AFF. At 01:14 the
user asked that the rule check's findings be applied with the controller's recommended fixes and the plan be run
without further questions (subagent-driven); a situation this file reserves for the user stops the work at that point.
The user allowed this file, and the work under it, to be committed to `main` without asking. After its commit this file
changes only with the user's approval.

**Dates.** Folder and report dates (`20261122`, `2026-11-22`; earlier rounds' `20261117`, `20261118`, `20261120`,
`20261121`) are sequence numbers, not calendar dates. Calendar times in this file are Amsterdam local time.

**Prior** (written before any number of this round). We expect V2 and V24 to fail the development bar against B′(A1)
(they need a seed-42 fused R@1 of about 19.31, about +0.17 above AFF's 19.137, by vetoes alone), and V4 to be the
likeliest candidate carried, with a seed-42 gain over AFF of at most about +0.1 (on R1 the same abstention moved the
in-sample margin, best fused minus best counterpart cell, by +0.045, and the in-sample best fused cell by +0.026). On the fresh seeds we expect the carried candidate's gain over AFF to be about
+0.05, below its detectable margin, so the likeliest outcome is a NO-GO read as "inconclusive at x" on the AFF check,
with the other checks passing. A kill at the carry would not surprise us; a GO would.

## 1. Glossary

Round 3's glossary (its §1) applies; the terms below are new or changed.

| Term | Meaning in this file |
|---|---|
| AFF | round 3's candidate, frozen as tested: R1 with the gate opened only on affect picks, g_t^c = 1[m^c ≥ τ_t]·1[π^c = affect] (round 3's D6), its 224 cells and per-seed cross-fits (round 3's D8) |
| A1 | the configuration (affect, image, caption, csd), in this order; the order breaks every arg-max tie (the first grouping wins) |
| csd | the CSD style grouping of step 1 (`step1_group_style.npz`) with its image and caption heads (`step1_heads_style.npz`, keys `style_csd__img`, `style_csd__txt`) |
| A1 reader, P_A1^c, π_A1^c | round 1's two A1 half-readers; their mean probability over A1's groupings under condition c; its arg max (D3) |
| v, v₇₅ | the abstention signal min(S_image, C_image) of an episode and its frozen seed-42 75th percentile (D5) |
| V4, V2, V24 | the three candidates: AFF's gate times the abstention factor, times the A1-pick factor, or times both (D6) |
| floor | B′(A0) for V4, B′(A1) for V2 and V24 (D8) |
| AFF check | the GO check "the candidate's fused R@1 minus AFF's fused R@1, paired per anchor" (§6.5) |
| Δ_k | the integer seed-42 paired difference of candidate k against AFF (D9) |
| carried candidate | the candidate the carry rule of §5 item 8 selects; the only one run on the test seeds |
| test seeds | episode seeds 52, 53 and 54 (§6) |
| smoke seeds | episode seeds 9001, 9002 and 9003, built only with `--smoke` (64 episodes per pair) for the wiring smoke test (§10); never results |

## 2. Data, metrics and intervals

As round 3's rule §2, with these changes:

- **Test-seed episodes:** `src/test/20261030_aspect_baselines/results/episodes_seed{52,53,54}.npz`, built once each by
  §6.2, 12,288 episodes per seed on selection rows, loaded through
  `src/test/20261101_aspect_factor_gonogo/run_gonogo.py::EvalContext(seed, False)`.
- **Pooled over the test seeds:** the per-anchor values of seeds 52, 53 and 54 are concatenated in that order (36,864
  episodes) and the clusters are the anchor paintings (`groups[anchor]`), so a painting that anchors episodes on several
  seeds is one cluster.
- **Metrics, intervals and precision** as round 3's §2: R@1, other-aspect rate, condition gain, either rate, per
  episode averaged over the four rankings, in percentage points; 95% percentile intervals of the anchor-painting
  bootstrap (`src.eval.aspect_metrics.cluster_bootstrap`, 5,000 resamples, seed 42, chunk 250), cross-fit picks fixed
  before resampling; "a lower bound above 0" means strictly greater than 0; every threshold applies to the
  full-precision value, and the integer comparisons of D9 and §5 item 8 are exact.
- Held rows are not read. Nothing in this file uses the GPU.

## 3. Definitions (the only definitions; every item below refers to them)

**D1. Inherited from round 3.** Round 3's D2 (standard heads), D3 (agreement and Δ), D4 (grouping score), D5 (the A0
reader, term T^c, pick π^c, margin m^c), D6 (τ_0..τ_3 and the R1 and AFF gates), D8 (the 224-cell family, z-scoring
before the gate, the integer min-margin cross-fit with the nested control σ*), D9 (matched counterpart G_cf and its
max-R@1 cross-fit), D10 (B), D11 (B′(A0)) and D12 (margins, gain statistic, bar comparator per scope) apply unchanged,
with "AFF" in D8 and D9 replaced by each candidate where this file says so. Round 3's D1 groupings (affect, image,
caption) and D7 (affect frozen by name) apply; round 3's D1 sentence that excludes csd and A1 does not (D2 below adds
them). Round 3's D13 is replaced by D10 below; round 3's D14 (told mapping) is
not used. Every candidate's term is AFF's: T^c = Σ_h P^c(h)·s_h on A0 from round 1's two A0 half-readers.

**D2. CSD posteriors and A1.** The csd grouping is step 1's CSD style grouping
(`src/test/20261116_grouping_step1_style/results/step1_group_style.npz`). Its posteriors are step 1's heads, as round
1's `common.load_bundle` builds them: z = `numpy.load(step1_heads_style.npz)`, asserted `z["selection"] ==
ctx.selection`; post["csd"] = {"img": `run_step1.full_post(z, "style_csd__img", ctx)`, "txt":
`run_step1.full_post(z, "style_csd__txt", ctx)`} (`run_step1` = `src/test/20261116_grouping_step1_style/run_step1.py`);
finite on selection rows (asserted). Heads and posteriors do not depend on the episode seed. A1 = (affect, image,
caption, csd).

**D3. A1 features, the A1 reader and its pick.** The 24 A1 features of both conditions are round 1's
`rb_eval.seed42_features(SimpleNamespace(ctx=ctx, post=post), A1)` with post holding the four groupings of A1 (D2 and
round 3's D2): per grouping, in A1 order, S, C, Δ, the two sample standard deviations (ddof 1) and the support match
share, as round 3's D5 lists them for A0. The first 18 columns are the A0 features of round 3's D5 (asserted equal,
exactly). The A1 reader is round 1's two A1 half-readers (`src/test/20261117_reader_fix_csd/results/rb_reader_A1.pkl`,
loaded with `rb_build.load_readers("A1", False)`; each a `StandardScaler` and a multinomial logistic regression trained
on the A1 practice bank of one painting half), frozen. P_A1^c(h) = the mean over the two half-readers of
`model_j.predict_proba(scaler_j.transform(x))` on condition c's 24 features (round 1's `rb_eval.half_reader_probs` and
`rb_features.average_probs`, as for A0). **π_A1^c** = arg max_h P_A1^c(h), ties to the first grouping in A1 order
(`numpy.argmax`); index 0 is affect. Only π_A1 is used; the A1 reader's term, margin and thresholds are not.

**D4. B′(A1).** `crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post, ctx.pooled, A1), ctx.parity)[0]`,
with T_N1u as in round 3's D10 and post holding A1's four groupings (D2), as step 1's `run_step1.evaluate_general`
computes T_6u for a configuration and round 1's `common.load_bundle` builds `Bp["A1"]`. It depends on the seed only,
never on a reader. Seed 42 (round 1's bundle, through the brainstorm's cache `pBpA1__r1`): mean R@1 18.804931640625.

**D5. The abstention signal v and its threshold v₇₅.** For an episode, v = min(S_image^a, C_image^a), the support and
contrast agreements of the image grouping under condition a (A0 feature columns 6 and 7 of condition a, float64). Since
condition b swaps supports and contrasts, min(S_image^b, C_image^b) = v exactly (asserted on every seed). **v₇₅** =
`numpy.percentile(v, 75)` (linear interpolation) over seed 42's 12,288 episodes = **0.021043562795966864**, frozen for
every seed (as τ is). Seed 42 check: recomputed exactly, with 1[v < v₇₅] = 1 on 9,216 of the 12,288 episodes. The
abstention factor is a_v = 1[v < v₇₅], the same for both conditions and both directions of an episode. It is the
brainstorm's `IMGABST_q75` factor (`bs_04_readers.py`), which computed the same percentile on the same seed-42 values.

**D6. The candidates' gates.** For τ index t ∈ {0, 1, 2, 3}, per episode and condition, the same for both directions,
with g_AFF,t^c AFF's gate (round 3's D6):

| Candidate | Gate g_t^c | Reads CSD | Floor |
|---|---|---|---|
| **V4** | g_AFF,t^c · a_v | no | B′(A0) |
| **V2** | g_AFF,t^c · 1[π_A1^c = affect] | yes | B′(A1) |
| **V24** | g_AFF,t^c · 1[π_A1^c = affect] · a_v | yes | B′(A1) |

Every gate (AFF's, R1's and each candidate's, at every τ index) is a float32 array of 0s and 1s per condition, as
round 3's `r3_fusion.gates_aff` returns it, so that the gated term stays on D8's float32 path (asserted). Every
candidate's gate is 0 wherever AFF's is 0 (asserted). With both factors set to 1 the gate is AFF's exactly
(asserted on seed 42: the regression path of §5 item 2). The gates read only the reader's own outputs, the A1 reader's
pick and the image agreements of the episode, and treat the two conditions identically, so every candidate is
label-free.

**D7. Family, counterpart and cross-fits per candidate.** Round 3's D8 and D9 with the candidate's gates of D6: the 224
cells (4 τ indices × 7 λ_u × 8 λ_a, cell number (t·7 + u)·8 + a), z(B) and z(T^c) per ranking row before any gate, the
gated term g_t^c·z(T^c), the fused score z(B) + λ_u·z(B) + λ_a·(g_t^c·z(T^c)) in float32, the nested control σ*, the
fused reader's integer min-margin cross-fit, and the matched counterpart G_cf,t = (g_t^a·z(T^a) + g_t^b·z(T^b)) / 2
under the candidate's own gates with its integer max-R@1 cross-fit; ties to the lowest cell number. Round 3's
`r3_fusion.run_family(bundle, T, gates)` computes exactly this for any gates. AFF itself is round 3's family with AFF's
gates.

**D8. Comparators, the bar comparator and margins.**
- V4's condition-free comparators: B, B′(A0) and its matched counterpart. V2's and V24's: B, B′(A1), B′(A0) and the
  matched counterpart. External baselines for all: cosine and RCA (`per_anchor_seed{s}.npz`, as round 3's D12).
- *Bar comparator* = whichever of the candidate's condition-free comparators has the largest mean R@1 over the
  episodes considered (full precision); ties to the earliest in the order B′(A1) (V2, V24 only), B′(A0), counterpart,
  B. For V4 this is round 3's rule (order B′(A0), counterpart, B). Chosen once per scope (seed 42; each test seed; the
  pooled test seeds), as round 3's D12. Including B′(A0) in the set for V2 and V24 is a refinement of the spec's "the
  strongest of the three" (B, floor, counterpart): it can only raise the bar, and it changes nothing on seed 42, where
  B′(A1) (18.804931640625) is above B′(A0) (18.436686197916664).
- *Margin*, *bar margin* and *gain statistic* as round 3's D12 for the candidate.

**D9. Paired difference against AFF.** Per episode, d_k = R@1 of candidate k's fused reader minus R@1 of AFF's fused
reader (each a multiple of 0.25 in fraction units, per round 3's D8 item 5). On seed 42, **Δ_k** = Σ over the 12,288
episodes of 4·d_k, an integer, summed from integer per-episode values (round 2's `r2_fusion.as_int4` of each per-anchor
R@1, which also asserts that every value is a multiple of 0.25), never from float means; its point in percentage points
is 100·Δ_k / (4·12,288), with its 95% interval from the anchor-painting bootstrap (§2). "Beats AFF on seed 42" means
Δ_k > 0 (integer comparison).

**D10. Development bar** (round 2's D12 with D8's bar comparator), on seed 42: (1) bar margin point at least +0.5; (2)
bar margin lower bound above 0; (3) gain statistic lower bound above 0.

**D11. Inputs.** Read only; never written or modified. Scripts assert every SHA-256 before using the file; a mismatch
stops the run. Every input of round 3's D15 table applies, with its SHA-256, and round 1's `common.verify_inputs()` runs
on every seed (it already covers `step1_heads_style.npz` and `step1_group_style.npz`). In addition:

| File (under `src/test/`) | SHA-256 | Used for |
|---|---|---|
| `20261121_round3_affect_gate/DECISION_RULE.md` | 2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925 | round 3's rule (part of this file by reference) |
| `20261121_round3_affect_gate/r3_common.py` | b1a60b1fd801bf9147d0bd58ae6da806d12fbcc982796135454f31dce3836f86 | round 3's constants and input asserts (imported) |
| `20261121_round3_affect_gate/r3_bundle.py` | bef50cbbef17f6c40d53c705a060967950d0690168c4b046b1cc3f0dd9fa0e9d | seed-parameterised A0 bundle (imported) |
| `20261121_round3_affect_gate/r3_fusion.py` | ce51a819157842035fdfde488f042c25fa4764ed1b1c363728d5e3837e314637 | reader, gates, 224-cell family (imported) |
| `20261121_round3_affect_gate/r3_stats.py` | 846a4f5b3175302280c20fa415cfff4b0652522d06aa4fc46290f1a1db5de54d | pooled checks, sensitivity (imported) |
| `20261117_reader_fix_csd/results/rb_reader_A1.pkl` | 4e1e4e23c20333b839aa1f92942891bdef51955317640d1014db5d097b7060f6 | the two A1 half-readers with their scalers |
| `20261117_reader_fix_csd/results/rb_reader_A1.json` | 41ec3bd1ef99523430ce6aacb2576c7928446a93b65fb9ded2b3ae6a6089d649 | the pickle's record |
| `20261117_reader_fix_csd/results/rb_reader_A1.npz` | f0297b92276cb0eda69f3717545c74a766203517950622f2966978b209c488cd | checked by `rb_build.load_readers` |
| `20261116_grouping_step1_style/run_step1.py` | f4bea509a6ca4fbb48e60823b4add1ba89570b626416127aaf57db66d63018c0 | `full_post` (imported) |
| `20261116_grouping_step1_style/results/step1_heads_style.npz` | 898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b | CSD posteriors (D2) |
| `20261118_reader_fix_round2/results/cand_R1_A1.npz` | 7494f8ef3f17db4ec1ef4cc5e5948b5fbd0c793beaf700a3a8464b296d01e075 | the A1 reader's seed-42 probabilities and picks (§5 item 4) |
| `20261118_reader_fix_round2/results/cand_R1_A1.json` | 4d13edaf7f3ea9f25782685c2f7daa9ebafeec6a5595d8de233cda376a4b9eb2 | its record |
| `20261120_r1_levers_brainstorm/results/bs_08_csd_evidence.json` | 7786d2aeaca8cc6ab67797b654a5b5ef3c4fd4df197f3b21eb8253e46c136a4b | provenance of idea 2's seed-42 numbers (not a target) |
| `20261120_r1_levers_brainstorm/bs_08_csd_evidence.py` | c4001e27c54afc15e8fa1fe51b84f59434770cb240731495ee872ef12f48813b | provenance (not imported) |
| `20261120_r1_levers_brainstorm/bs_07_detector.py` | 1b287bcc725270af9b3b0a0b41b9021d9ed3ea844ef404df52ca4f4e9e47088a | provenance of §6.11 (not imported) |
| `20261120_r1_levers_brainstorm/results/bs_07_detector.json` | e9d52462a2c128b39cb5a2fe7b78b425e5ceac982b091c334d99d5d951a485e1 | provenance of §6.11 |
| `20261120_r1_levers_brainstorm/bs_11_visual_side.py` | cb27cd636bae5f92ba016602148ad2508ed8a50867fe4c6b3b0c7c7885704ad2 | provenance of §6.11 (not imported) |
| `20261120_r1_levers_brainstorm/results/bs_11_visual_side.json` | a1b51be306750af10a2634488fdf775ce096588e36ac143c61194630282dc6c7 | provenance of §6.11 |
| `20261118_reader_fix_round2/run_r2_fusion.py` | fe6fb9ff870c3942bef98327ff23894782522e516fc100cb1635aeec6e7e5b65 | provenance of `cand_R1_A1.npz` (`round1_probs`; not imported) |
| `20261030_aspect_baselines/results/baselines_seed49.json` | a7e9fbce85f2d2724a0d07ab3df6d4c0b5d83e314d2efd02aac278f9be1c21b3 | earlier episode SHA-256s (§6.2) |
| `20261030_aspect_baselines/results/baselines_seed50.json` | b0567f8c2ee4782780097d8c601e0e043a9512a99a421f8463b2aba199cdc8fa | earlier episode SHA-256s (§6.2) |
| `20261030_aspect_baselines/results/baselines_seed51.json` | 0d1c9c87446e99e4bed73cd2b105eeb6ef62630e0d7c3200e3dffea31299f42d | earlier episode SHA-256s (§6.2) |

`bs_04_readers.json` (round 3's D15: 42bf6f20…691e) holds the `IMGABST_q75` targets of §5 item 3, and
`codes_provenance.json` must keep round 3's D15 SHA-256 (8e6a517b…bfaf) before and after each build. The pickles were
written with scikit-learn 1.6.1 and numpy 2.2.6; the run uses the same versions.

## 4. The seed-parameterised pipeline

Round 3's §4 pipeline, reused by import, extended. One code path, a function of the episode seed s (and of `smoke` for
the wiring smoke test, §10), used unchanged on seed 42 (§5) and on the test seeds (§6):

1. **Bundle:** round 3's bundle (`r3_bundle.build_bundle(s, smoke)`: episodes, parity halves, anchor paintings, pair
   index, cosine, B, B′(A0), the A0 posteriors, s_h on A0, the 18 A0 features, the A0 half-readers), then the A1
   extension: post["csd"] (D2), the 24 A1 features (D3), the A1 half-readers, B′(A1) scores and per-anchor arrays (D4),
   and v (D5). The A1 extension never changes a field of round 3's bundle (asserted by comparing them before and after).
   The smoke flag is never passed to `model_inputs`, `load_readers` or the head files. Round 3's seed guard
   (`r3_bundle._check_seed`, which reads `r3_common.TEST_SEEDS` when it is called) admits only 42, 49 to 51 and the
   smoke seeds. Before its first bundle call, this round's code sets `r3_common.TEST_SEEDS = (52, 53, 54)` in its own
   process. No file of round 3 changes, and no other round-3 function that reads `TEST_SEEDS` or `EARLIER_SEEDS`
   (`run_r3_build.py`'s seed checks, `r3_apply_rule.py`) is called. A unit test shows that the guard then admits 42, 52,
   53 and 54 and refuses 49 and 55.
2. **External baselines:** as round 3's §4 item 2.
3. **Readers:** P^c, T^c, m^c, π^c from the A0 half-readers (round 3's D5); P_A1^c and π_A1^c from the A1 half-readers
   (D3).
4. **Gates:** AFF's gates (round 3's D6) and each candidate's gates (D6) at τ_0..τ_3; on seed 42 also R1's gates and
   R1's gates times a_v (§5 item 3). On a test seed, only AFF's gates and the carried candidate's gates are computed
   (and the A1 reader of item 3 only if the carried candidate reads CSD).
5. **Families:** for AFF and for each candidate needed by the step (§5, §6.4): the 224 cells, the gated terms, G_cf,
   the integer statistics, σ*, the fused reader's min-margin cross-fit and the counterpart's max-R@1 cross-fit, and the
   assembled per-anchor arrays (D7).
6. **Records:** per seed and scorer: the chosen cells of both cross-fits on both tune halves (cell number, τ index and
   value, λ_u, λ_a), σ*, the per-anchor arrays (`r1`, `gain`, `other`, `swap`, `strict`) of each fused reader and
   counterpart computed, and B, B′(A0), B′(A1), cosine and RCA per anchor. What may be computed before the test verdict
   is limited by §6.4.

## 5. Seed 42: the regression checks, then the development step

Items 1 to 5 are the regression checks. They run first, in this order. No *candidate result* of V4, V2 or V24 is
computed, written or printed before items 1 to 5 pass. A candidate result is a score, per-anchor array, chosen cell,
R@1, margin, gain, either rate, bar margin or Δ_k. Item 5 computes the candidates' gates and their τ_0 open counts,
which are gate statistics and not candidate results, once items 1 to 4 have passed (a test enforces this order, §10). If any check fails, no further number is written,
and the work stops and goes to the user with the cause traced.

**Item 1. Bundle.** The A0 part of the seed-42 bundle passes round 3's §5 item 1 exactly (round 3's
`r3_bundle.compare_with_round1`, the D7 redundancy check included). The A1 extension equals round 1's
`common.load_bundle()` exactly: post["csd"] (image and caption sides), B′(A1) (scores `Bp["A1"]` and per-anchor arrays
`pBp["A1"]`), and the 24 A1 features of both conditions equal `rb_eval.seed42_features(<round 1's bundle>, A1)`; the
first 18 columns equal the A0 features. B′(A1)'s mean R@1 equals 18.804931640625 exactly.

**Item 2. R1 = round-1 R-c and AFF = round 3's targets,** exactly as round 3's rule §5 items 2 and 3 state them (R1's
T^c, margins, picks, τ, chosen cells 116, 119 / 58, 123, σ* 0 / 0, per-anchor arrays and bar margin; AFF's fused R@1
19.136555989583336, counterpart 18.39599609375, bar margin 0.6998697916666667 [0.4598852740816973,
0.9371680126852968] against B′(A0), gain statistic 3.110758463541667 [2.780005709854805, 3.4559584315470384], chosen
cells 39, 119 / 149, 10, AFF minus R1 and the open counts 9,941 / 3,627 at τ_0), computed through this round's code
path: the candidate gate function with both factors set to 1 must return AFF's gates exactly, and AFF's family run from
those gates.

**Item 3. The abstention path = the brainstorm's `IMGABST_q75`** (`bs_04_readers.json`, key `results.IMGABST_q75`):
R1's gates times a_v (D5), computed by the same gate-factor function that builds V4's gate, with R1's gates in place of
AFF's, through the 224 cells, R1's term, both cross-fits and R1's comparators, at full precision (the rule check
reproduced every number below through the float32 path, with no float32 against float64 difference in any of the 224 ×
12,288 integer statistics):
- v₇₅ = 0.021043562795966864 recomputed exactly; a_v = 1 on 9,216 episodes; v^b = v^a exactly;
- fused R@1 19.059244791666664; counterpart R@1 18.49365234375; bar comparator the counterpart;
- bar margin 0.5655924479166667 [0.3448683992591827, 0.79821625538382]; gain statistic 2.878824869791667
  [2.554983173204304, 3.2105685950938248]; either change against the counterpart −1.7476399739583333;
- per-pair bar margins 0.677490234375 (emotion × style), 1.28173828125 (emotion × genre), −0.262451171875 (style ×
  genre);
- chosen cells: fused 117 on tune half 0 (τ_2, λ_u 0, λ_a 4) and 119 on tune half 1 (τ_2, 0, 16); counterpart 58
  (τ_1, 0, 0.5) and 67 (τ_1, 0.5, 1); σ* = 0 on both halves.

The brainstorm scored in float64 and this pipeline in float32 (round 3's D8). A difference is traced to its cause before
anything else runs (a near-tie whose order differs between float32 and float64 would be such a cause) and goes to the
user, who decides before anything else runs.

**Item 4. The A1 reader = round 2's stored R1/A1 reader** (`cand_R1_A1.npz`): P_A1^c equal `probs__a` and `probs__b`
exactly in value and π_A1^c equal `pick__a` and `pick__b` (stored as int8) exactly. The stored probabilities are the
half-reader mean of D3 (round 2's `run_r2_fusion.round1_probs`; confirmed by the rule check, maximum absolute difference
0.0). Seed 42 has no exact arg-max tie in P_A1 under either condition, so the tie rule of D3 is tested on constructed
ties (§10).

**Item 5. Gate algebra on seed 42:** for every τ index and condition, each candidate's gate is 0 wherever AFF's is 0;
V24's gate equals V4's gate times V2's gate exactly; V2's gate equals AFF's gate times 1[`pick__c` = 0], with `pick__c`
from `cand_R1_A1.npz` per condition (an independent source of π_A1); V4's gate equals AFF's gate times 1[v <
0.021043562795966864], with v from the A0 features of round 3's bundle; the open counts of each gate at τ_0 are written
as integers (gate statistics, not candidate results).

**Item 6. Development numbers** (only after items 1 to 5 pass). For each of V4, V2 and V24 on seed 42: its family (D7);
fused and counterpart R@1; the chosen cells and σ*; the bar comparator (D8); the bar margin, the margin against the
counterpart and the gain statistic with their intervals; the either change against the counterpart; Δ_k (D9) with its
point and interval; the clauses of D10; per-pair bar margins (descriptive). Written to `results/dev_seed42.json` with
this file's SHA-256 and the Amsterdam time.

**Item 7. Development bar:** D10, per candidate.

**Item 8. Carry.** Let E be the set of candidates that clear the development bar and have Δ_k > 0. Let M be the
largest Δ_k in E. Every member of E with M − Δ_k ≤ 24 is tied (24 = the integer part of 0.05 percentage points ×
4·12,288 / 100 = 24.576). The carried candidate is the tied member that comes first in the order V4, V2, V24. The carry
is written to `results/carry.json` with the Δ_k, the D10 clauses, E, M and the tied set.

**Item 9. Kill.** If E is empty, no test seed is built, the seed-42 results go to the user, and AFF stays the current
best; the user decides what follows.

No other variant is computed on seed 42.

## 6. The test (the only confirmatory step)

1. **Sensitivity, after the carry and before the seeds are built** (round 3's §6.1 formula, unchanged). For each GO
   check of item 5, take the carried candidate's seed-42 per-episode difference (its fused R@1 minus the comparator's;
   its fused gain minus 0 for the gain statistic; its fused gain minus RCA's gain; its fused R@1 minus AFF's fused R@1)
   and project the pooled standard error over three seeds; write SE, 1.96·SE, the **detectable margin x = 2.80·SE** and
   the seed-42 bootstrap half-width to the log and to `results/sensitivity.json`. Used only to read a failed check
   (item 7); it never stops the round.
2. **Build.** Seeds 52, 53 and 54 (free in `docs/superpowers/episode_seed_ledger.md`), once each, in one invocation of
   this round's build runner, each with `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`,
   without `--overwrite`. The 9 new per-pair episode SHA-256s must differ from each other and from every per-pair
   SHA-256 in `baselines_seed{42,43,45,47,48,49,50,51}.json`; on a match the test stops and goes to the user.
   `codes_provenance.json` is checked before and after each build as round 3's §6.2 says. Each build's console output
   goes to `results/build_seed{s}.log`, which is not opened before the verdict. A crashed build is handled as round 3's
   §6.2 says. After the hash check the seed ledger gets its rows: seeds 52, 53, 54 (test, round 4), 55 and later free.
3. **Frozen from seed 42:** the A0 and A1 half-readers with their scalers, τ_0..τ_3, v₇₅, the affect restriction, the
   224-cell family (λ grid, cell order, tie rules), the heads and posteriors, the recipes of B, B′(A0) and B′(A1), the
   method-A checkpoint. **Rerun on each test seed's own parity halves:** B, B′(A0), B′(A1), the readers' probabilities on
   that seed's features, v, the gates (that seed's margins, picks and v against the frozen τ and v₇₅), σ*, and for the
   carried candidate and for AFF the fused reader's min-margin cross-fit, and for the carried candidate the
   counterpart's max-R@1 cross-fit. Per-seed cross-fitting is part of each method's definition, as in round 3.
4. **Order of computation.** On the test seeds, until the verdict is written to `results/test_verdict.json` (with this
   file's SHA-256 and the Amsterdam time), only these are computed and written: the per-seed bundles and reader arrays
   of §4 items 1 to 4 (cached for later use; a bundle holds B′(A1)'s per-anchor arrays, which enter no comparison
   before the verdict unless B′(A1) is a GO comparator); the per-anchor arrays of the carried candidate's fused reader
   and counterpart, of AFF's fused reader, of B, B′(A0), B′(A1) (only if the carried candidate reads CSD), cosine and
   RCA; the chosen cells and σ* needed to assemble them; and the pooled checks of item 5. AFF's counterpart is not
   cross-fitted, assembled or written (its per-cell statistics may be computed in the same pass and are not read).
   The candidates that were not carried are never computed on a test seed. Per-seed and per-pair numbers, bar margins,
   gate shares, AFF's own checks, B′(A1) beside AFF, the frozen-cell line and the random-share control come only after
   the verdict (§7). The GO pass asserts, before it computes any check (round 3's final review N11, M31, M11): (a) each
   seed's clusters equal `groups[anchor]` of that seed's episodes; (b) before each family call, the gates passed in
   equal, at every τ index and in both conditions, the gates recomputed from that method's D6 definition (round 3's D6
   for AFF) and the seed's margins, picks, A1 picks and v; (c) the build records' file and episode-hash checks of §6.2
   are re-run; (d) AFF's counterpart is not cross-fitted. The open counts these assertions use are held in memory and
   are not written or printed before the verdict.
5. **GO** if, pooled over the three seeds (§2), every one of these checks has a 95% lower bound above 0:
   - R@1, the carried candidate's fused reader minus each of cosine, RCA, B, B′(A0), B′(A1) (only if the candidate
     reads CSD) and its matched counterpart;
   - the gain statistic (D8), which is also the gain difference against cosine, B, B′(A0) and B′(A1), counted once;
   - condition gain, the candidate's fused reader minus RCA;
   - **the AFF check:** R@1, the candidate's fused reader minus AFF's fused reader, paired per anchor.

   Eight checks for V4, nine for V2 or V24. Points, intervals and pass or fail of every check are written to
   `results/test_verdict.json` with the verdict.
6. *(No secondary check in this round.)*
7. **NO-GO** if any check fails, including a partial pass. Each failed check is read on its own: if its pooled point is
   above 0, it is **inconclusive at a detectable margin of x** (that check's x from item 1, with the realised pooled
   half-width beside it), not evidence that the candidate fails; if its point is at or below 0, it is "<candidate> did
   not beat <name> on fresh episodes", where <name> is cosine, RCA, B, B′(A0), B′(A1), the matched counterpart or AFF
   for the R@1 checks, "the condition-free comparators on condition gain" for the gain statistic, and "RCA on condition
   gain" for the condition-gain check. If the AFF check is the only failed check, the NO-GO also reads "the candidate
   works, but no improvement over AFF was shown"; in every NO-GO AFF stays the current best. For the AFF check, x may
   understate the margin needed: it is projected from one seed-42 pair of cross-fits, and when the two methods' cells
   agree on seed 42 but differ on a test seed the realised variance is larger (in round 3 the analogous paired check,
   AFF minus R1, realised a half-width 18% above its projection, the other checks within 5%).
8. **Per-seed and per-pair results** are reported after the verdict and never change it; per-pair results are not
   tested.
9. **Frozen-cell line** (descriptive, after the verdict): on each test seed, the carried candidate's fused reader and
   counterpart scored with the cells seed 42's cross-fits chose for it (the cell chosen on seed-42 tune half h scores
   the test seed's episodes of parity 1 − h), and AFF's with its seed-42 cells (fused 39 and 119, counterpart 149 and
   10), with the comparators of item 5.
10. **Claim licensed.** A GO shows that the carried candidate (AFF with its veto factor or factors) beats AFF and each
    comparator of item 5, pooled over the three aspect pairs, on new episodes drawn from the same 6,451 selection
    paintings, and so replaces AFF as the current best for the held-split paper test. It does not show transfer to new
    paintings (the held split stays reserved) or a margin on each aspect pair. For V2 and V24 the method reads CSD
    agreements in its gate (never in its score) and the claim includes B′(A1). For the paper: the groupings were built
    without evaluation labels; AFF and the candidates were developed on seed 42 (item 11).
11. **Multiplicity disclosure** (reported with every number of the carried candidate): AFF was found among about 50
    label-free variants read on seed 42 (round 3's §6.11); ideas 2 and 4 were explored on seed 42 on R1 (the
    brainstorm's `bs_04_readers.py`: four abstention variants; `bs_07_detector.py` and `bs_08_csd_evidence.py`: nine
    label-free detectors and two declared oracles; `bs_05_aff.py`: four variants of the A1 reader's own gate, "A1
    affect only" the nearest earlier relative of V2; `bs_11_visual_side.py`: `AFF_and_VIS`, a visual veto on AFF's
    affect pick with the term z(s_affect), +0.553 against B′(A0), the nearest earlier relative of V4); V4's signal and
    percentile are the best of the four abstention variants; V2's AND form was not tried there; the development family has three candidates and the carry takes the largest paired gain over AFF. The
    development numbers are therefore inflated; the fresh seeds 52 to 54 are the protection, and the frozen-cell line
    accompanies the test.

## 7. After the verdict (descriptive; decides nothing)

Computed only after `results/test_verdict.json` is written, from the cached per-seed arrays where possible:

1. Per-seed and per-pair results for the carried candidate (the measures of §6.5 per seed, and per pair pooled over
   seeds), its bar margin (D8, the bar comparator chosen per scope), its margin, gain and either change against its
   counterpart, and its chosen cells per seed; the frozen-cell line (§6.9).
2. **AFF's own seven checks** on the test seeds (round 3's §6.5 list; its counterpart cross-fit run now), with its bar
   margin, per seed and per pair, labelled descriptive; and **B′(A1) beside AFF**: B′(A1)'s pooled mean R@1 and AFF's
   fused R@1 minus B′(A1), paired, pooled, with its interval.
3. Gate open shares at each τ index, overall, per condition and per pair (label-free), for the carried candidate and
   AFF, per seed and pooled; and the **veto closure share**: among the (episode, condition) values where AFF's gate is
   open, the share the candidate's gate closes, per pair and condition, at each τ index, pooled; if V24 is carried,
   also the share each of its two factors closes alone.
4. **Random-share control** at the carried candidate's per-condition τ_0 open shares, as round 3's §7 item 7 defines
   it (R1's gates times keep^c, `numpy.random.default_rng(100·s + r)`, draws r ∈ {0, 1}, condition a first, share_c the
   carried candidate's τ_0 open count in condition c over the seed's episode count, float64; the 224 cells, its own
   cross-fit and its own counterpart). It reads which condition is a, so it is a mechanism control, never a method.
   Reported per draw, pooled: its bar margin against the same condition-free scorer as the carried candidate's pooled
   bar comparator (or against the control's own counterpart when that comparator is the candidate's counterpart), and
   the candidate minus the control (fused R@1, paired per anchor).
5. Nothing else is computed on the test seeds. Idea 3 is not part of this round.

## 8. Order of work, re-derivation and timeline

- **Order:** (1) implementation with unit tests, including the tests of §10 that round 3's final review asked for
  before reuse; (2) the seed-42 regression checks (§5 items 1 to 5); (3) the development step (§5 items 6 to 9); (4)
  if a candidate is carried, the sensitivity projection (§6.1; seed-42 arrays only); (5) the independent
  re-derivation, phase 1, which must agree before a kill is reported or step (6) starts; (6) the end-to-end wiring
  smoke test on seeds 9001 to 9003 (§10); (7) the three seeds built (§6.2); (8) the GO pass (§6.4); (9) the
  re-derivation, phase 2; (10) the rule applied, the verdict written; (11) the descriptive pass (§7); (12) the
  whole-branch final review, one fix wave and a scoped re-review; (13) the report.
- **Independent re-derivation.** An agent that has not written or read the implementation re-derives, with its own
  code, in two phases. Phase 1 may start once this file is committed. It computes its own §5 items 1 to 5 first, and
  it computes no candidate result (§5) before those match this file's targets. It then re-derives every candidate's
  development numbers of §5 item 6, the D10 clauses, the Δ_k and the carry of item 8, and the detectable margins x of
  §6.1 for the carried candidate; it compares with the implementation's files only after its own results are written
  and hashed. Phase 2 (after the GO pass, before
  the rule is applied): the hash check of §6.2; per test seed B, B′(A0), B′(A1) where it is a GO comparator, v, the
  gates, σ*, the chosen cells of the carried candidate's two cross-fits and of AFF's fused cross-fit; every GO check of
  §6.5 (points, bounds, pass or fail). The re-derivation may import the data loaders and frozen components that round
  3's rule §8 lists, plus `run_step1.full_post`, `rb_build.load_readers("A1", False)` and round 1's
  `common.load_bundle` (seed 42 only). It writes its own code for the features, P, T, margins, picks, v, the gates,
  G_cf, the 224 cells' integer statistics, σ*, both cross-fits, the assembly, the per-anchor metrics, the comparisons,
  the checks, Δ_k, the carry and x. It writes only to `rederive/` of this folder.
- **Agreement** means: every discrete quantity identical (picks, gates, chosen cells, σ*, the bar comparator, Δ_k, the
  carry, each check's pass or fail); τ and v₇₅ equal to their stated values within 1e-15 absolute or 1e-9 relative;
  every margin, gain statistic, check point and bound within 1e-9 percentage points; every per-anchor array exactly. A
  difference beyond these is traced to its cause before the step that depends on it; the computation that follows this
  file's text settles it, and the user is told.
- **Boundaries.** A check whose lower bound lies within 1e-12 of 0, a development-bar clause within 1e-12 of its
  threshold, or a Δ_k at 0 or a tie gap at exactly 24, is reported to the user with both values before the step that
  depends on it is recorded (the stated inequalities still decide).
- **Time-box:** ends Wednesday 14 October. If the test cannot be completed by then, no partial verdict is issued; the
  user is told and decides.

| Day (Amsterdam) | Work |
|---|---|
| Wed 7 Oct | This rule written, checked by a fresh Opus reviewer (findings applied), committed and sent to the user; the plan; implementation by subagents; the regression checks; the development step and the carry; phase 1; if carried: sensitivity, smoke test, seeds 52 to 54 built, the GO pass, phase 2, the verdict (target) |
| Thu 8 Oct | Anything left; the descriptive pass (§7); the whole-branch final review and its fix wave; the report |
| by Wed 14 Oct | Time-box ends |

## 9. Outcome-to-action table

| Outcome | Action |
|---|---|
| A script check fails (a SHA-256 of this rule, round 3's rule or a D11 input, round 1's or round 3's bundle check, §5 items 1 to 5, an episode-hash match, a change of `codes_provenance.json`, a condition-free, gate-algebra or alignment assertion) | Stop that step and report to the user; nothing is improvised |
| §5 item 3 or item 4 differs from the stored numbers | Traced to its cause; the user decides before anything else runs |
| The wiring mutation of §10 does not fire an assertion | Stop; no test seed is built; the user is told |
| No candidate clears the development bar with Δ_k > 0 (§5 item 9) | **Kill:** no test seed is built; the seed-42 results go to the user; AFF stays the current best |
| A candidate is carried | Sensitivity, smoke test, build, GO pass (§8 order) |
| On the test seeds, before the verdict | Only the quantities of §6.4 |
| All checks of §6.5 pass | **GO** within the claim of §6.10, with the disclosure of §6.11: the carried candidate replaces AFF as the current best |
| Any check of §6.5 fails, including a partial pass | **NO-GO**, read by §6.7; AFF stays the current best; the user decides what follows |
| AFF's own checks (§7 item 2) fail on the test seeds, or the random-share control matches or beats the candidate (the control's pooled bar-margin point is at or above the candidate's, or the candidate-minus-control interval contains 0, for either draw), or B′(A1) is above AFF (B′(A1)'s pooled mean R@1 at or above AFF's fused R@1) | Reported only; no second verdict; the user decides what follows |
| The re-derivation or the final review disagrees with a reported number, the carry or the verdict beyond the agreement of §8 | Traced to its cause; the computation that follows this file's text settles it; the corrected result goes to the user with the cause; the rule is not changed |
| A run crashes, or a bug is found before the rule is applied | As round 3's §9: correcting code to match this file is not a change of the rule; a crashed run is repeated after its partial outputs are deleted; a written results file is never overwritten (the corrected run writes beside it with the suffix `_fix<n>`, the log records the cause and the numbers before and after); a completed seed is never rebuilt |
| Any situation this file does not cover | The user decides; this file is not changed after its commit without the user |

## 10. Process

- Scripts of this folder assert this file's SHA-256, round 3's rule's SHA-256 and the SHA-256 of every D11 input they
  read, refuse to overwrite non-smoke results, and write only to this folder's `results/` (gitignored by this folder's
  `.gitignore`, a copy of round 3's), except that `run_baselines.py` (§6.2) writes its own outputs for seeds 52, 53 and
  54 in `src/test/20261030_aspect_baselines/results/` and rewrites `codes_provenance.json` there, the smoke builds write
  to that folder's `results/smoke/`, and the seed ledger gets its rows.
- Module names of this folder start with `r4_` (or `run_r4_`, `test_r4_`), so that none shares a name with a module of
  rounds 1 to 3, whose folders are on `sys.path`. The folders of rounds 1 to 3, step 1 and the brainstorm are read
  only: their modules are imported by path and not modified, and their functions that write files are not called.
- **Tests added before round 3's code is reused** (round 3's final review N11 and its mutation survivors, and the
  deferred minors of its SDD ledger): the GO-pass assertions (a) to (d) of §6.4, each shown to fire under a mutation;
  (e) the descriptive pass checks that its cached arrays reproduce the GO pass's (M27); a test that the seed-42 runner
  writes and prints no candidate result (§5) before §5 items 1 to 5 pass (M20); a criterion test for ρ_ctrl (the
  nested control's σ* and ρ_ctrl against a hand-computed example; M06); tie tests for π and π_A1 with constructed exact
  ties (M28); a bar-comparator test with D8's four-way order in which the per-seed, pooled and per-pair choices differ
  (T2-M4); tests of the bundle guards (head identity, finiteness, parity, selection; T1-1) and of the seed guard of §4
  item 1; the boundary-reported path and the phase-2 agreement record each bound to the SHA-256 of the current
  `go_pooled.json` (T3-1, T3-10); and a leak check in the wiring smoke test that catches any decimal number,
  one-decimal numbers included.
- **Smoke runs** write to `results/smoke/`, may be overwritten, are not results, and never print or log a metric value
  of any scorer on any seed: the console and the log show only shapes, counts, file names and the pass or fail of each
  assertion. Their files are opened only by the assertions and are deleted when the smoke test has passed. They never
  use the test seeds. The **end-to-end wiring smoke test** runs after §6.1: the GO pass, the rule application (reading
  `results/sensitivity.json`) and the descriptive pass on the smoke seeds 9001, 9002 and 9003 (built with
  `run_baselines.py --smoke --episodes-seed <s>`), with the carried candidate. It must complete with every assertion
  passing before any test seed is built, and it runs three mutations of the wiring that must each fire an assertion:
  the candidate's gated term passed where the counterpart expects G_cf; AFF's fused arrays swapped for the candidate's
  in the AFF check; and the candidate's gate passed without its veto factor.
- CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1`, at most 3 processes,
  `uptime` and `free -g` checked first. The main session launches every real run (`run_in_background`); subagents
  implement.
- Records: the log `20261122_round4_aff_vetoes_log.md` in this folder (times from `TZ=Europe/Amsterdam date`); the
  report `docs/reports/auto/v2/2026-11-22_round4_aff_vetoes.md`, committed only after the whole-branch final review and
  its fix wave, with a row in `docs/reports/reports_sum.md` and `scripts/check_reports_sum.py` run; `.claude/<yyyymmdd>_log.md`
  for any edit to an existing source file; the seed ledger rows. Commits go to `main` without a further request
  (authorisation above), scoped by explicit path, never pushed.
