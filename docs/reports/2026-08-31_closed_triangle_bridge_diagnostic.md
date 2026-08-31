# Closed-triangle bridges: does the buddy-init embedding discriminate real edges from false-transitivity artifacts?

**Date:** 2026-08-31 · **Dataset:** RedCaps, 150,000 rows (same operating point as Experiments 9–12) · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `scripts/analyze_polysemy_bridges.py` (`extract_hub_pairs`, `closed_triangle_membership`, `count_hub_pairs`, `_build_typed_graph`, `--counts-only`/`--n-hub-sample`), `src/conditional_buddy/buddy_graph.py` (`hub_neighbor_pairs`)
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 14
**Motivated by:** `docs/reports/2026-08-26_polysemy_bridge_diagnostic.md` (Experiment 12/C10) — the B–C bridge pull is real but only very weakly explained by shared-neighbor structure ("false transitivity"), and every pair measured there was, by construction, never directly connected — no positive control existed to check whether the embedding treats a genuine edge any differently from an indirect artifact.
**Implementation plan:** `docs/superpowers/plans/2026-08-31-closed-triangle-bridge-diagnostic.md`, executed via `subagent-driven-development` (Tasks 1–5, all reviewed clean, zero open findings)

---

## TL;DR

**The buddy-init embedding does discriminate a real edge from a false-transitivity artifact — but only modestly, and the gap is dwarfed by how much it pulls artifacts together in the first place.** Closed-triangle pairs (two of a hub node's text-only neighbors that are *also* directly connected by a real image-only edge) are pulled together with mean pull **+3.18** (embedding-distance units); open hub-pairs (same structure, but with no direct edge — Experiment 12's B/C case) pull **+2.60**. The **+0.58 gap is real and tight across 3 sampling seeds** (per-seed contrast 0.55–0.61, mean/SEM=+32.1), a stable ~22% relative difference — the "Discriminates" branch of this experiment's pre-registered criteria, but a modest one: the embedding pulls a false-transitivity artifact together at ~80% of the strength of a genuine edge, not a fraction of it. Two secondary findings sharpen this further: **hub-node pairs pull substantially harder than plain single-neighbor bridge pairs** (+2.60 vs. Experiment 12's own +1.98, a ~30% increase — the opposite of a dilution effect), and **hub/closed-triangle node membership correlates with retrieval-rank change more strongly, and in the opposite direction, than C10's broad `is_polysemic` label** (`in_closed_triangle` rho=+0.107 vs. `is_polysemic` rho=−0.025) — both purely descriptive, not part of the pre-registered primary criterion.

## Method

Task 0 (incidence count, `--counts-only`, zero embedding/template loading): rebuilt the RedCaps-150k buddy graph (K=30, α=0.5, same template as every prior experiment in this plan) and counted **110,405 hub nodes** (≥2 text-only neighbors) and **8,940,974 hub-neighbor pairs**, of which **70,544 are closed triangles** (also a real `img_only` edge) and 8,870,430 are open (no direct edge). 70,544 closed-triangle candidates clears this plan's 30-sample floor by three orders of magnitude — **no escalation to RedCaps-300k was needed.**

Primary run: `scripts/analyze_polysemy_bridges.py`, 3 independent sampling seeds (0, 1, 2 — this experiment's only randomness is which pairs/baselines get drawn from the deterministic, already-built buddy graph and its one already-completed init template; there is no model training in this experiment, so "seed" here means resampling, not retraining), each drawing up to 5,000 closed-triangle pairs and 5,000 open hub-pairs independently (`extract_hub_pairs`, capped per group so the far-more-common open pairs can't crowd out the rare closed ones), computing pull against a degree-decile-matched baseline exactly as Experiment 12's B/C statistic does (`baseline_dist − pair_dist`). Retrieval cross-reference reused the same 6 already-on-disk per-sample `.npz` dumps Experiment 12.3 produced (3 seeds `trained`-vs-`frozen` from 11.1, 3 seeds `pred_coupled`-vs-`frozen` from 11.3) — this part of the computation is fully deterministic (built from the raw, unsampled hub-pair/label structure, not the capped per-seed sample), confirmed byte-identical across all 3 runs.

## Results

**Closed-triangle vs. open-hub-pair pull, pooled across 3 sampling seeds:**

| Statistic | Closed triangle | Open hub-pair | Contrast (closed − open) |
|---|---|---|---|
| Per-seed mean pull | 3.1716, 3.1943, 3.1753 | 2.6253, 2.5861, 2.6027 | 0.5463, 0.6082, 0.5726 |
| Pooled mean (± SEM across seeds) | **+3.1804 ± 0.0070** | **+2.6047 ± 0.0114** | **+0.5757 ± 0.0179** |
| mean/SEM (across seeds) | +452.0 | +229.2 | **+32.1** |
| frac_pulled_closer (single seed, n=5000) | 0.998–0.999 | 0.973–0.976 | — |
| Relative ratio (closed / open) | | | **1.20–1.24×** |

For context, Experiment 12's original single-neighbor bridge pull (the classic B–C statistic, reproduced here unchanged as a consistency check across all 3 runs): mean **+1.97 to +2.00**, mean/SEM +102 to +106 — matching the originally reported +1.98/+102.1 almost exactly.

**Secondary — hub vs. plain-bridge pull (descriptive, #1):** open hub-pairs (+2.60) pull noticeably *harder* than plain single-text-neighbor bridge pairs (+1.98) — a ~30% increase, not the dilution a naive "more neighbors spread the smoothing thinner" hypothesis would predict.

**Secondary — retrieval cross-reference (descriptive, #3), pooled across the same 6 runs C10/Experiment 12.3 used:**

| Flag | corr vs. `delta_rank` (signed) | corr vs. `\|delta_rank\|` |
|---|---|---|
| `is_polysemic` (C10's original label) | rho=−0.025, mean/SEM=−9.9 | rho=+0.017, mean/SEM=+8.3 |
| `is_hub` (≥2 text-only neighbors) | rho=+0.096, mean/SEM=+14.9 | rho=−0.099, mean/SEM=−145.0 |
| `in_closed_triangle` | rho=+0.107, mean/SEM=+30.8 | rho=−0.273, mean/SEM=−375.7 |
| `in_open_hub_pair` | rho=+0.108, mean/SEM=+15.9 | rho=−0.101, mean/SEM=−80.9 |

## Interpretation

1. **Primary criterion result: "Discriminates," but modestly, not decisively.** The +0.58 gap (closed > open) is real, stable across 3 independent sampling seeds (relative spread of the contrast itself is only ~5%), and clears this project's `mean/SEM ≥ 2` bar by more than an order of magnitude — but per this experiment's own instruction to judge on magnitude rather than significance alone, a ~22% relative difference is a real qualifier on C10/C12's false-transitivity finding, not a reversal of it. The embedding is not blind to the difference between a genuine edge and an artifact bridge, but it also doesn't come close to treating them as categorically different: an artifact bridge is pulled together at roughly four-fifths the strength of a real edge. This softens, rather than resolves, the "false transitivity" concern — the embedding partially discriminates, so caution about over-reading bridge-induced closeness as semantic relatedness (C12's original point) still stands, just somewhat less starkly than if closed and open pull had come out indistinguishable.
2. **Hub-ness itself amplifies pull, it doesn't dilute it.** Both closed (+3.18) and open (+2.60) hub-pairs pull harder than Experiment 12's plain single-neighbor bridge pairs (+1.98). A node with more text-only neighbors doesn't spread the spectral-smoothing effect thinner across each one — if anything, the opposite. This is consistent with the same "broad, largely content-independent" smoothing account C12 already gave for the original bridge pull: a higher-degree node in the union graph sits in a denser, more strongly-connected neighborhood, and the spectral embedding's global smoothing appears to respond to that density rather than to any single specific edge's provenance.
3. **The retrieval-rank cross-reference tells a genuinely different, more specific story than C10's broad label.** `is_polysemic` (any bridge/one-sided node) barely correlates with `delta_rank` (rho=−0.025) — consistent with C10's own reading that the effect there is concentrated narrowly in `img_only_only` nodes, not spread broadly. But `is_hub`, and especially `in_closed_triangle` (rho=+0.107, more than 4× `is_polysemic`'s magnitude), correlate more strongly — and with the **opposite sign**. This is worth flagging for a future, more targeted diagnostic (is this driven by the same `img_only_only`/`combine_side` mechanism C10 isolated, or something specific to hub/closed-triangle structure?) but is reported here descriptively, per this experiment's own scope — it was not part of the pre-registered primary criterion and no causal claim is made.
4. **What this does not establish:** why hub-ness amplifies pull, and why closed-triangle/hub membership correlates with `delta_rank` more strongly (and oppositely-signed) than the generic bridge label, are both open mechanistic questions this experiment surfaces but does not answer.

**Bottom line for the paper:** C10/C12's "false transitivity" framing narrows further, in the same direction Experiment 13's symmetric-conditioning result (C11, on the `experiment/two_side_conditioning` branch) narrowed C9 — not overturned, but qualified: the buddy-init embedding does weight a genuine edge somewhat more than an indirect artifact (~20-24% more pull), it just doesn't weight it *categorically* more, and hub structure specifically (not just generic bridge-ness) is where the retrieval-side signal concentrates most strongly.

## Caveats

- **The +22% gap is a magnitude judgment call, not a bright line.** A reader could reasonably read this as "discriminates" or as "barely discriminates" — this report states the number precisely (1.20–1.24×) rather than forcing a binary verdict, per the pre-registered three-way criteria's own spirit.
- **The retrieval cross-reference's flags are not mutually exclusive** with each other or with C10's `label_nodes()` categories (`is_hub`/`in_closed_triangle`/`in_open_hub_pair` are additive boolean flags, deliberately not folded into the existing 4-way partition — see the plan's self-review notes for why). A node can be simultaneously `bridge`, `is_hub`, and `in_closed_triangle`; these correlations are not disjoint population comparisons.
- **The bridge-subgroup relocation question from C10 (does the `img_only_only`/`txt_only_only` concentration effect interact with hub/closed-triangle structure specifically?) was not investigated here** — this experiment's retrieval cross-reference reports the new flags' own correlations, not a joint breakdown against C10's existing labels.
- **n=3 sampling seeds**, matching this plan's standard bar — note again this is resampling-seed variation on a fixed, already-built graph and embedding, not training-seed variation; it establishes the pull estimates are stable under resampling, not that they'd be stable under a differently-constructed buddy graph.
- **Only the CLI's printed summary omits the new `is_hub`/`in_closed_triangle`/`in_open_hub_pair` correlations** (a minor gap in the implementation plan's print-block spec, not a computation bug) — all numbers in this report were read directly from the `--out` JSON, where they are correctly present.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR

# Task 0: cheap incidence count (no template/embedding loading)
python scripts/analyze_polysemy_bridges.py --counts-only \
  --storage-dir /data/SSD2/pre_extract/redcaps_150k/features

# Full protocol, one sampling seed (repeat with --seed 1, --seed 2 for the 3-seed pool)
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
```
