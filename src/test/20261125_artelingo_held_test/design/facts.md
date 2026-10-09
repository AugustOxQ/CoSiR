# Facts for the grill: ArtELingo held-split paper test of AFF (frozen as in round 3)

> Gathered 2026-10-09 02:59 (Amsterdam) by a read-only agent. No held row was scored, no held episode built, no held
> feature put into a model; §B's counts use labels and IDs only. Paths are under `/project/CoSiR/`. "Rule" =
> `src/test/20261121_round3_affect_gate/DECISION_RULE.md`; "R3" = `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md`;
> "plan" = `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md`.

## A. AFF as frozen

| Component | Artifact | Fitted on | On held |
|---|---|---|---|
| Groupings: affect (41 Leiden groups on GoEmotions caption probs), image and caption (k-means 64 on CLIP) | `partition_L` (told-oracle npz); `20261031_pseudo_partitions/results/partitions.npz` (rule:95-104) | scorer-train | applied via heads; no GoEmotions on held captions |
| Heads: logistic regressions on CLIP B/32, 60,000-row scorer-train draw | affect refit every run (rule:106-111); image, caption and B's affect-km: **only selection posteriors stored** (`n6_posteriors.npz`, `run_n6.py:81-92`), heads not saved | scorer-train | **refit** (same draw), applied to held; check = refit reproduces the stored selection posteriors |
| Two A0 half-readers + scalers | `20261117_reader_fix_csd/results/rb_reader_A0.pkl` (sklearn 1.6.1 enforced) | practice banks of scorer-train halves (rule:56) | applied |
| τ_0..τ_3; affect restriction | `rc_tau.json` (seed-42 margin percentiles, rule:131); D7 | selection seed 42 | applied |
| Method-A term in B, B′ | `A3_seed42.pt` | scorer-train only (`run_gonogo.py:11-12`) | held rows encoded (CPU) |
| RCA and pair metrics | PCA basis, PairScaler (`run_baselines.py:126-132`) | scorer-train | applied |
| **Cell and weight picks**: AFF (224 cells, min-margin), counterpart (max R@1), σ*, B and B′ (56 cells, `aspect_quick_checks.py:297-327`), RCA λ (`aspect_scorers.py:93-97`) | none stored as frozen | **re-picked on each test seed's parity halves using episode targets** (rule:371-376) | the only use of evaluated-episode labels |

- Nothing learned would be fitted on held rows; the cross-fit picks are the exception. **Plan §8 (:446): for final
  reads, fusion weights are chosen on development and frozen.**
- Frozen-cell evidence (R3 Table 11): AFF with seed 42's cells (fused 39, 119; counterpart 149, 10) +0.585 [+0.455,
  +0.719] over B′(A0) vs +0.591 cross-fitted, seven checks passing; R1 +0.495 frozen vs +0.389.
- B's and B′'s seed-42 picks are **not recorded** (scores only: `crossfit_condition_free(...)[0]`,
  `20261117_reader_fix_csd/common.py:578`). RCA's seed-42 λ: 0.5 on both halves (`baselines_seed42.json`).
- **Entry points:** `run_r3_seed42.py` (regression checks, sensitivity), `run_r3_build.py` (build, hash check),
  `run_r3_test.py` (GO, descriptive), `r3_apply_rule.py`; libraries `r3_bundle`, `r3_fusion`, `r3_stats`, `r3_common`.
  `r3_fusion` and `r3_stats` are pure functions on a bundle. `r3_bundle.build_bundle` uses `run_gonogo.EvalContext`,
  which masks to selection rows and asserts it (`run_gonogo.py:207-251`); its seed guard admits 42, 49-51 and smoke
  (`r3_bundle.py:133-134`); `run_baselines.py` is selection-only (:109-123, :141). A held runner needs a held context,
  held episodes and baselines, and the refit heads.

## B. The held split

Held = the 20% part of `grouped_split(leakage_groups(...), seed 42)` (`src/data/splits.py:45-60`,
`src/data/artelingo_splits.py:10-34`); IDs are recomputed (H3 asserted equality with
`src/test/20261013_stage_d_selection/cache/prepare.npz`). 24 held rows have a train image at CLIP cosine ≥ 0.99, 63 at
≥ 0.98 (`splits.py:66-71`).

| (all three labels known; value eligible at ≥ 30 paintings) | Selection | Held |
|---|---|---|
| Rows / paintings | 32,413 / 6,451 | 61,744 / 12,281 |
| Rows with all three labels (paintings) | 23,580 (5,253) | 44,385 (9,947) |
| Eligible emotion / style / genre values | 8 / 23 / 10 | 8 / **24** / 10 |
| Anchor paintings: emo×style / emo×genre / style×genre | 5,204 / 5,253 / 5,204 | 9,893 / 9,947 / 9,893 |

- Extra held style: New_Realism (37 held paintings, 17 selection). Smallest eligible style: 37 held, 30 selection.
- Round 3: 4,096 per pair per seed × 3 seeds = 36,864 episodes on 5,195 anchor paintings (seed 42: 4,602), near the
  selection maximum. Held has about 1.9× the anchor paintings. All three pairs are feasible (`aspect_episodes.py:81-99`).
- Features exist for held rows: CLIP B/32 (all 308,723 rows), Qwen3-VL-Embedding-2B (308,723 captions, 61,402 images),
  CSD (61,402 images), under `/data/SSD2/pre_extract/artelingo*/`.
- **H1 to H3** (held ledger:8-10): **value episodes** (one label), Candidate A factor models (G3, SE, C0), seeds 42 and
  43; H3 touched all 12,281 held paintings. Not counted in the aspect budget (ledger:13); disclosed (plan:501-504).

## C. Round 3's decision rule

GO = all seven pooled checks with 95% lower bound > 0 (rule:386-389); **no minimum size** (the +0.5 bar, C10/D13, was
a seed-42 record only). Bootstrap over anchor paintings, 5,000, seed 42, picks fixed, seeds concatenated (rule §2).
Prior +0.3. Claim (rule:406-410): beats each comparator, pooled, on new episodes of the same 6,451 selection
paintings; no transfer, no per-pair margin.

| AFF's fused reader minus … | Pooled 49-51 | x | Seed 42 |
|---|---|---|---|
| cosine, R@1 | +5.840 [+5.611, +6.069] | 0.324 | +6.175 |
| RCA, R@1 | +5.722 [+5.495, +5.946] | 0.320 | +5.756 |
| B, R@1 | +0.806 [+0.686, +0.930] | 0.180 | +0.795 |
| B′(A0), R@1 (bar margin) | +0.591 [+0.462, +0.729] | 0.193 | +0.700 |
| matched counterpart, R@1 | +0.796 [+0.670, +0.920] | 0.181 | +0.741 |
| gain statistic (= gain vs cosine, B, B′, counterpart) | +3.319 [+3.130, +3.518] | 0.280 | +3.111 |
| RCA, condition gain | +3.286 [+3.076, +3.494] | 0.311 | +3.007 |
| secondary: R1, R@1 | +0.202 [+0.093, +0.309] | 0.131 | +0.218 |

*R3 Table 4 (:330-343). Per pair (R3 Table 8): +0.810, +1.544, **−0.580 [−0.793, −0.363]** (style × genre).*
Unchanged for held: frozen pieces, bootstrap, C11 reading, process. Need a decision: picks (§A), episodes and seeds,
B′(A1), K2's Holm, the claim wording (new paintings), the reserve, which seed-42 numbers the C12 check reproduces.

## D. The paper's bar (K2, §10) mapped to AFF

K2 (plan:228): beat backbone-only, the best raw metric-from-pairs baseline and the uniform-weight control on **both**
R@1 and condition gain, on ArtELingo, CUB and SemArt, under §10's multiplicity; per-pair reported, not claimed.

| K2 comparator | AFF's object | R@1 | Condition gain |
|---|---|---|---|
| backbone only | cosine | +5.840 | +3.319 (cosine gain 0) |
| best metric-from-pairs | RCA (seed-42 GO bar, `baselines_seed42.json` go_bar_ranking) | +5.722 | +3.286 |
| uniform-weight control | C2 matched control = D9 counterpart (rule:173-179) | +0.796 | +3.319 |

- Round 3's seven contain K2's six; they add B and B′(A0). C2's wording ("uniform weights in place of the reader",
  constitution:19-23) fits B′(A0) more literally than D9 (two-condition mean of AFF's own gated term).
- AFF's condition gain: 3.593, 3.267, 3.097 per seed; 3.111 on seed 42; either-rate cost 1.727 (R3 §10).
- **§10 (plan:485-540):** Holm over four comparisons (incl. K7) × three datasets, per backbone; K3, K8 separate.
  Also: two-way (anchor × candidate) bootstrap with item-reuse rate; swap success; a power simulation for a 2.0 gain
  difference at Holm α; headline over 3 training seeds (AFF has one A3 checkpoint and one reader pair).
- **Budget:** main read = one pre-registered run covering all models and baselines on both backbones, weights frozen
  (:497-498); reserve = only a pre-registered fix after a final-review finding (:499-500). ArtELingo 0 of 2 used.

## E. Comparators

R@1 / gain; pooled 49-51 from `baselines_seed{49,50,51}.json` and the reports.

| Comparator | On held without fitting on held? | Pooled 49-51 | Seed 42 |
|---|---|---|---|
| AFF; R1 | yes if picks frozen | 18.880 / 3.319; 18.677 / 2.521 | 19.137 / 3.111; 18.919 (our arithmetic) / 2.667 |
| counterpart; B′(A0); B | heads refit, picks frozen or re-picked | 18.083; 18.288; 18.073 (gain 0) | 18.396; 18.437; 18.341 |
| B′(A1) (adds CSD style grouping) | CSD posteriors stored for selection only (`r4_bundle.py:151-203`) | **never measured** | 18.805; AFF − it +0.332 [+0.048, +0.625] |
| cosine; RCA; Tip-Adapter | yes; λ frozen or re-picked | 13.040 / 0; 13.158 / +0.033; 12.874 / −0.041 | 12.962; 13.381 / +0.104; 12.917 |
| other pair metrics (diag, KISSME, Xing, Wang, probe, prototype) | as RCA | 12.840 to 13.034 | 12.793 to 13.064 |
| CLIP fine-tune LP / LB / LoRA | checkpoints in `res/cluster_jobs/2026100707*/code/outputs/clipft/`; held never extracted; LP on cached features (CPU); LB, LoRA need held images on GPU | 15.00 / 14.97 / 15.14 (gain 0) | 15.19 / 15.02 / 15.21 |
| MLLM in context (Qwen3-VL 8B) | GPU | not run | +1.07 [0.17, 1.93] over cosine (seed 46) |

## F. Reuse hazards

- Round 5 final review §4 (`src/test/20261123_idea3_goemotions/final_review/final_review.md:280-307`): "before reuse"
  items sit in `r5_*` code (`chosen_cells` fused/counterpart untested, `sharper_term` sign, `load_ext` not checked
  against `r4_shas`, an untested key-set check, an unanchored regex). None in `r3_*`; relevant only if `r4_`/`r5_`
  code (e.g. B′(A1)) is reused. §5: no seed-52+ files; ledgers unchanged since round 3.
- Round 5 lapses (report §7, §8.4): `.pyc` in read-only folders (no `PYTHONDONTWRITEBYTECODE=1`), a mutation run in
  place, a shape bug missed because tests used a wrong synthetic shape, an estimated log time.
- **Refuse-twice:** no script parses `held_ledger.md`. H3's `src/test/20261019_affect_factor_learning_held/run_held.py`
  refuses a rerun when its results or started file exists (`--after-crash` recorded), requires `power.json` first and a
  passing selection-row smoke of the same script bytes (:16-18, :413-470), and asserts the split and held-only rows
  (:245-265). Reusable as a pattern.

## G. Seeds

- Ledger (`episode_seed_ledger.md:8-16`): 42 development; 43 to 51 spent; **52 and later free**; 9001-9003 smoke.
  It covers selection rows; held reads go to the held ledger.
- Round 3's hash check (`run_r3_build.py:46, 124`): new per-pair `episodes_sha256` distinct from each other and from
  seeds 42, 43, 45, 47, 48. Held episodes use other rows, so no collision; H1-H3 hashes are of another builder.
- Anchors are drawn with replacement (`aspect_episodes.py:103`); any seed works on held. H1/H2 used 42, H3 43.

## H. Open design questions (facts and options only)

1. **Picks on held** (KEY: data touched, claim). (a) Re-pick on held parity halves as round 3 (held targets choose
   every scorer's picks). (b) Freeze all from seed 42, per plan §8 (B/B′ picks must be recomputed first). (c) One as
   verdict, the other reported.
2. **Pass checks** (KEY). (a) Round 3's seven. (b) K2's three on both metrics (drops B, B′(A0); C1 still wants the
   strongest condition-free score beside the headline). (c) Both (the seven include K2's six).
3. **B′(A1)** (KEY). Never on fresh episodes; seed-42 lead +0.332; needs CSD heads on held and `r4_bundle`. (a) Pass
   check (could fail the test). (b) Reported beside. (c) Left out.
4. **Bar size, gain** (KEY). (a) Lower bounds > 0 only. (b) A minimum point (e.g. C10's +0.5). (c) Gain co-primary
   against every comparator (K2; round 3 already has it).
5. **Multiplicity** (KEY). (a) All at 95%, all must pass. (b) Holm per §10 on ArtELingo's K2 tests: with 6 tests
   the strictest level is 0.05/6, about 0.18 half-width at round 3's bar-margin SE (our arithmetic, from 0.134). (c) Bonferroni.
6. **Episodes, seeds** (KEY: cost, irreversible). Per-pair count (4,096 × 3, more for 1.9× paintings, or by §10's
   power simulation); one or three seeds; 52+ or reuse 42.
7. **What the main read covers** (KEY, irreversible). AFF on Qwen (K4) and design L do not exist; reading AFF on B/32
   now leaves them no ArtELingo held read unless the user amends C5 and §10. (a) Read now. (b) Wait. (c) Amend first.
8. **Reserve** (KEY). (a) §10 as written, same episodes. (b) Pre-registered triggers, maybe new episodes. (c) For
   K4 or design L (needs an amendment).
9. **Claim and style × genre** (KEY). (a) Pooled claim, per-pair reported, loss disclosed. (b) A pre-registered
   per-pair descriptive or non-inferiority line. (c) A per-pair claim. "New paintings" wording goes beyond C14.
10. **R1** (MINOR): secondary check, descriptive, or out. **Fine-tuned CLIP** (MINOR, cost): reported (LP on CPU, or all
    three on GPU), pass check, or out. **Style set** (MINOR): 24 held styles or the 23 of selection.
11. **Extras and safeguards** (MINOR, agent default): two-way bootstrap, swap success, sensitivity projection on the
    held anchor distribution, the prior (C8); ledger row and refuse-twice guard; selection smoke with the same bytes;
    C12 regression, head-refit check, C13 re-derivation; disclosures (H1-H3, about 50 variants on seed 42,
    near-duplicates).
