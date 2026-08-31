# Closed-triangle bridges: does the buddy-init embedding discriminate real edges from false-transitivity artifacts?

**Date:** 2026-08-31 · **Dataset:** RedCaps, 150,000 rows (same operating point as Experiments 9–12) · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `scripts/analyze_polysemy_bridges.py` (`extract_hub_pairs`, `closed_triangle_membership`, `count_hub_pairs`, `_build_typed_graph`, `--counts-only`/`--n-hub-sample`), `src/conditional_buddy/buddy_graph.py` (`hub_neighbor_pairs`)
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 14
**Motivated by:** `docs/reports/2026-08-26_polysemy_bridge_diagnostic.md` (Experiment 12/C10) — the B–C bridge pull is real but only very weakly explained by shared-neighbor structure ("false transitivity"), and every pair measured there was, by construction, never directly connected — no positive control existed to check whether the embedding treats a genuine edge any differently from an indirect artifact.
**Implementation plan:** `docs/superpowers/plans/2026-08-31-closed-triangle-bridge-diagnostic.md`, executed via `subagent-driven-development` (Tasks 1–5, all reviewed clean; Task 6 caught and corrected a construct-validity bug in its own control group during the final whole-branch review — see Caveats)

---

## TL;DR

**The buddy-init embedding does discriminate a genuine edge from a false-transitivity artifact, by a real and seed-stable ~31%.** Closed-triangle pairs (two of a hub node's text-only neighbors that are *also* directly connected by a real image-only edge) pull together with mean **+3.18** (embedding-distance units, pooled across 3 sampling seeds); pairs with **no edge of any kind** between them pull **+2.42**. This is the corrected, load-bearing number in this report. **An earlier version of this analysis compared closed-triangle pairs against an "open" group defined only as "not an img_only edge" — the final whole-branch review caught that this control group was ~52% contaminated by pairs that ARE connected via some other edge type (almost certainly `txt_only`, since both endpoints are already text-only neighbors of the same hub) — narrowing the apparent gap to a "discriminates, but modestly" ~1.2× ratio. Correcting the open group to genuinely-unconnected pairs widens the gap to ~1.31×, strengthening rather than weakening the "discriminates" verdict.** Two secondary findings, now correctly caveated: hub-node pairs pull harder than Experiment 12's plain single-neighbor bridge pairs, but the two comparisons differ in edge-type composition as well as hub degree, so "hub-ness amplifies pull" is a plausible but not isolated reading; and `in_closed_triangle`/`is_hub` node membership correlates with `delta_rank` more strongly than C10's broad `is_polysemic` label, but the labels have very different population base rates (98.7% vs. 19–82%), so the magnitude comparison is not apples-to-apples.

## Method

Task 0 (incidence count, `--counts-only`, zero embedding/template loading): rebuilt the RedCaps-150k buddy graph (K=30, α=0.5, same template as every prior experiment in this plan) and counted **110,405 hub nodes** (≥2 text-only neighbors) and **8,940,974 hub-neighbor pairs**, of which **70,544 are closed triangles** (also a real `img_only` edge). 70,544 closed-triangle candidates clears this plan's 30-sample floor by three orders of magnitude — **no escalation to RedCaps-300k was needed.**

Primary run: `scripts/analyze_polysemy_bridges.py`, 3 independent sampling seeds (0, 1, 2 — this experiment's only randomness is which pairs/baselines get drawn from the deterministic, already-built buddy graph and its one already-completed init template; there is no model training in this experiment, so "seed" here means resampling, not retraining), each drawing up to 5,000 pairs per group, computing pull against a degree-decile-matched baseline exactly as Experiment 12's B/C statistic does (`baseline_dist − pair_dist`).

**Construct-validity correction (found during the final whole-branch review, before this report was finalized):** the codebase's `hub_neighbor_pairs()` labels a pair "closed" iff it is connected by an `img_only` edge specifically, and "open" otherwise — which is a correct and useful graph-topology label in its own right, but is *not* the same thing as "no edge of any kind," since two nodes that are both text-only neighbors of the same hub are plausible candidates for a direct `txt_only` edge between themselves too. A supplementary check confirmed this: of the raw population's 8,870,430 "open" (not-`img_only`) pairs, **4,585,902 (51.7%) are actually connected by some other edge type**, leaving only 4,284,528 (48.3%) genuinely unconnected. All pull comparisons in this report's Results section use the corrected, genuinely-unconnected definition (checking pair membership against the union graph `E`'s full edge-key set, not just its `img_only` subset) as the "open" group; the original, contaminated comparison is reported alongside for transparency, not as the headline result.

Retrieval cross-reference reused the same 6 already-on-disk per-sample `.npz` dumps Experiment 12.3 produced (3 seeds `trained`-vs-`frozen` from 11.1, 3 seeds `pred_coupled`-vs-`frozen` from 11.3) — this part of the computation is fully deterministic (built from the raw, unsampled hub-pair/label structure, not the capped per-seed sample), confirmed byte-identical across all 3 runs, `n_joined=3,000` per run throughout.

## Results

**Closed-triangle vs. genuinely-unconnected pull, pooled across 3 sampling seeds (corrected primary comparison):**

| Statistic | Closed triangle | Genuinely unconnected | Contrast (closed − open) |
|---|---|---|---|
| Per-seed mean pull | 3.1815, 3.1796, 3.1704 | 2.4332, 2.4355, 2.4057 | 0.7483, 0.7441, 0.7647 |
| Pooled mean (± SEM across seeds) | **+3.1772 ± 0.0034** | **+2.4248 ± 0.0096** | **+0.7524 ± 0.0063** |
| mean/SEM (across seeds) | +927.0 | +253.3 | **+119.7** |
| frac_pulled_closer (single seed, n=5000) | 0.998–0.999 | 0.961–0.965 | — |
| Relative ratio (closed / genuinely-open) | | | **1.31×** |

**For transparency, the original (contaminated) comparison** — closed vs. "not-`img_only`," including the 51.7%-contaminated pairs:

| Statistic | Closed triangle | "Open" (contaminated) | Contrast |
|---|---|---|---|
| Pooled mean (± SEM) | +3.1804 ± 0.0070 | +2.6047 ± 0.0114 | +0.5757 ± 0.0179 (mean/SEM=+32.1) |
| Relative ratio | | | 1.20–1.24× |

The corrected comparison's closed-triangle numbers are consistent with the original (both ≈+3.18, as expected — the closed group's definition didn't change). The "open" side dropped from +2.60 to +2.42 once contaminated pairs were excluded, **widening** the closed/open gap from ~1.22× to ~1.31× — i.e. the original analysis, if anything, *understated* how much the embedding discriminates a real edge from a genuine non-edge.

**Population-level degree check (closed vs. genuinely-open groups, full 8.94M-pair population, not just the sampled subset):**

| | Hub `deg_txt_only` | C-endpoint E-degree |
|---|---|---|
| Closed group | mean 19.23, median 20 | mean 34.28, median 35 |
| Genuinely-open group | mean 20.88, median 22 | mean 31.22, median 31 |

The two groups differ by ~8–10% on both degree measures, in opposite directions (open-group hubs are somewhat higher-degree; closed-group C-endpoints are somewhat higher-degree) — a real but modest confound, not a dramatic one, and not obviously biased toward inflating the closed/open pull gap in either direction given the two measures pull opposite ways.

For context, Experiment 12's original single-neighbor bridge pull (the classic B–C statistic, one img_only + one txt_only edge, reproduced here unchanged as a consistency check across all 3 runs): mean **+1.97 to +2.00**, mean/SEM +102 to +106 — matching the originally reported +1.98/+102.1 almost exactly.

**Secondary — hub vs. plain-bridge pull (descriptive, #1):** hub pairs (even the genuinely-unconnected ones, +2.42) pull harder than Experiment 12's plain single-neighbor bridge pairs (+1.98). **Caveat:** this comparison varies both hub degree *and* edge-type composition simultaneously (Experiment 12's B/C pair is one img_only + one txt_only edge; this experiment's hub pair is two txt_only edges) — it is not a clean isolated test of "does hub-ness alone amplify pull," only evidence that the two populations differ, in the amplifying rather than diluting direction.

**Secondary — retrieval cross-reference (descriptive, #3), pooled across the same 6 runs C10/Experiment 12.3 used (n_joined=3,000 per run):**

| Flag | Population true-rate | corr vs. `delta_rank` (signed) | corr vs. `\|delta_rank\|` |
|---|---|---|---|
| `is_polysemic` (C10's original label) | 2,962/3,000 = 98.7% | rho=−0.025, mean/SEM=−9.9 | rho=+0.017, mean/SEM=+8.3 |
| `is_hub` (≥2 text-only neighbors) | 2,222/3,000 = 74.1% | rho=+0.096, mean/SEM=+14.9 | rho=−0.099, mean/SEM=−145.0 |
| `in_closed_triangle` | 570/3,000 = 19.0% | rho=+0.107, mean/SEM=+30.8 | rho=−0.273, mean/SEM=−375.7 |
| `in_open_hub_pair` | 2,475/3,000 = 82.5% | rho=+0.108, mean/SEM=+15.9 | rho=−0.101, mean/SEM=−80.9 |

**Caveat:** `is_polysemic`'s 98.7% true-rate means its correlation rests on only 38 minority (negative) cases — a much more attenuated statistic than the other three, which sit at more balanced base rates. The magnitude comparison between `in_closed_triangle` (rho=+0.107) and `is_polysemic` (rho=−0.025) is therefore not a clean apples-to-apples comparison; the former isn't necessarily "4× stronger" in any mechanistic sense, only larger under very different population balance. Separately, `is_hub` (74.1%) and `in_open_hub_pair` (82.5%) are near-collinear populations (most hubs' sampled pairs are open, since open pairs vastly outnumber closed ones) and produce nearly identical rhos (+0.096 vs. +0.108) — these are not two independent pieces of evidence.

## Interpretation

1. **Primary criterion result: "Discriminates," and more clearly than first measured.** The corrected +0.75 gap (closed +3.18 vs. genuinely-unconnected +2.42) is real, stable across 3 independent sampling seeds, clears this project's `mean/SEM ≥ 2` bar by two orders of magnitude, and — per this experiment's own instruction to judge on magnitude rather than significance alone — represents a genuine ~31% relative difference, wider than the ~22% first measured against a contaminated control group. The embedding is not blind to the difference between a genuine edge and a genuine non-edge. This still qualifies rather than reverses C10/C12's false-transitivity caution — the embedding pulls a truly-unconnected artifact pair together at roughly three-quarters the strength of a real edge, not a small fraction of it — but it discriminates more clearly than the first pass suggested.
2. **The control-group contamination itself is worth naming as a methodological lesson, not just a footnote.** Defining "open" as "not the specific edge type under test" rather than "no edge at all" is an easy mistake in any graph-diagnostic experiment with multiple edge types, and it happened here despite passing individual task review five times — because no single task's reviewer was positioned to question the experiment's own construct validity, only its code's fidelity to the plan. It took the final whole-branch review, explicitly checking the report's own conclusions against the underlying data, to catch it.
3. **Hub-ness plausibly amplifies pull, but the evidence for that specific mechanism (vs. edge-type composition) is not isolated here.** Both closed and genuinely-open hub-pairs pull harder than Experiment 12's plain single-neighbor bridge pairs — but that comparison changes two things at once (hub degree AND edge-type composition: hub pairs are two txt_only-type endpoints, Experiment 12's classic pair is one img_only + one txt_only). A future experiment holding edge-type composition fixed while varying only hub degree would be needed to isolate the mechanism.
4. **The retrieval-rank cross-reference's `in_closed_triangle` correlation is a real, more concentrated signal than C10's broad label — but its size shouldn't be read as "4× stronger" in a mechanistic sense**, given the very different population base rates involved. It remains a legitimate lead for a future, more targeted diagnostic (is this driven by the same `img_only_only`/`combine_side` mechanism C10 isolated, or something specific to hub/closed-triangle structure?), reported descriptively per this experiment's own scope — not part of the pre-registered primary criterion, no causal claim made.
5. **What this does not establish:** why hub-ness amplifies pull (isolated from edge-type composition), and what mechanism produces the closed-triangle/hub retrieval correlation, remain open questions this experiment surfaces but does not answer. The mirror configuration (an img-only hub with a closed triangle via a txt_only edge, rather than the txt-only hub/img_only-closure tested here) was also not examined — given C10's headline was precisely that its subgroup effect flips modality under `combine_side="txt"`, this asymmetry is a natural next check.

**Bottom line for the paper:** C10/C12's "false transitivity" framing narrows further, in the same direction Experiment 13's symmetric-conditioning result (C11, on the `experiment/two_side_conditioning` branch) narrowed C9 — not overturned, but qualified: the buddy-init embedding does discriminate a genuine edge from a genuine non-edge by a real, seed-stable ~31%, though it still pulls a genuinely-unconnected artifact pair together at roughly three-quarters the strength of a real one. Report this experiment's corrected numbers, not the original ~22%/1.2× figures, if citing this result.

## Caveats

- **This report was revised after its first version incorrectly used a contaminated "open" control group** (see Method) — the final whole-branch review caught this before the result was cited elsewhere; the corrected numbers throughout this document supersede the original ones, which are shown only for transparency in the Results table.
- **The +31% gap is still a magnitude judgment call, not a bright line** — a reader could reasonably read "pulls together at 3/4 the strength of a real edge" as either a meaningful discrimination or a modest one. This report states the number precisely (1.31×) rather than forcing a binary verdict, per the pre-registered three-way criteria's own spirit.
- **The closed/genuinely-open groups are not perfectly degree-matched** (see Results) — the ~8-10% degree differences run in opposite directions on the two measures checked, so this is unlikely to be the dominant driver of the ~31% gap, but it was not controlled for by design.
- **The retrieval cross-reference's flags are not mutually exclusive** with each other or with C10's `label_nodes()` categories (`is_hub`/`in_closed_triangle`/`in_open_hub_pair` are additive boolean flags, deliberately not folded into the existing 4-way partition — see the plan's self-review notes for why). `is_hub` and `in_open_hub_pair` are additionally near-collinear populations (see Results), not independent evidence.
- **The bridge-subgroup relocation question from C10** (does the `img_only_only`/`txt_only_only` concentration effect interact with hub/closed-triangle structure specifically?) **was not investigated here.**
- **n=3 sampling seeds**, matching this plan's standard bar — this is resampling-seed variation on a fixed, already-built graph and embedding, not training-seed variation; it establishes the pull estimates are stable under resampling, not that they'd be stable under a differently-constructed buddy graph.
- **The modality-mirrored configuration was not tested** (see Interpretation #5).
- **The CLI's printed summary omits the `is_hub`/`in_closed_triangle`/`in_open_hub_pair` correlations** (a minor gap in the implementation plan's print-block spec, not a computation bug) — all numbers in this report were read directly from the `--out` JSON and from supplementary one-off scripts (see Reproduction), where they are correctly present.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR

# Task 0: cheap incidence count (no template/embedding loading)
python scripts/analyze_polysemy_bridges.py --counts-only \
  --storage-dir /data/SSD2/pre_extract/redcaps_150k/features

# Original (contaminated-open) full protocol, one sampling seed (repeat with --seed 1, --seed 2)
python scripts/analyze_polysemy_bridges.py \
  --storage-dir /data/SSD2/pre_extract/redcaps_150k/features \
  --template-dir res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings \
  --device cuda --seed 0 \
  --per-sample-npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_161846_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_163307_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_164733_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_100355_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_102258_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/20260826_103723_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz \
  --out docs/reports/assets/2026-08-31_closed_triangle_bridge_diagnostic_seed0.json

# Corrected (genuinely-unconnected-open) comparison and the population degree/contamination check:
# a standalone script, not yet promoted into analyze_polysemy_bridges.py itself (a follow-up
# task, not committed here) -- reuses _build_typed_graph/hub_neighbor_pairs/degree_deciles/
# sample_baselines/embedded_l2_distance/paired_pull_summary unchanged, redefining only which
# pairs count as the "open" comparison group (checked against typed["keys"] in full, not just
# typed["img_only"]). See this report's git history / session record for the exact script.
```
