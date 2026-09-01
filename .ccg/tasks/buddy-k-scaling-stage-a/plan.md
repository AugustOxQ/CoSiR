# 实施计划: Experiment 16.1 Stage A — buddy K graph-diagnostic sweep

## 需求
Implement the Stage A diagnostic sweep from `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 16.1: for RedCaps N∈{150k,300k,500k} × K∈{10,20,30,50,75,100} (18 cells), compute strict-buddy stats, union-buddy stats + edge-type composition, and C1a's subreddit-lift proxy — no training. Derive a candidate K(N) scaling rule and a Stage-B short-list.

## 方案
Two Codex analyzer passes (this session, background) converged on a corrected design:

1. **Dataset-variant correction**: use `redcaps_150k`, `redcaps_300k_diverse`, `redcaps_500k_diverse` feature stores with their matching annotation JSONs — not the plain 300k/500k stores (15/28-of-350 subreddit coverage, unusable for lift comparison). Already folded into the spec doc.
2. **Exact API contract confirmed**: `mutual_knn(features, K, device, ...)`, `union_graph(A_img, A_txt)`, `ensure_min_degree(E, img, txt, device) -> (E_fixed, {"num_isolated": int})`, `classify_edges(A_img, A_txt, E, N) -> {"keys", "img_only", "txt_only", "both", "repair"}` (boolean masks, not integer codes), `subreddit_lift(data, e, top_k=None) -> {"obs_same_frac", "exp_same_frac", "overall_lift", "n_subreddits", "top_enriched"}`, `FeatureManager.load_all_to_ram(["img_features","txt_features"])`.
3. **K(N) inversion procedure**: isotonic regression over each N's 6 measured `(K, strict_avg_degree)` points (enforces monotonicity, exact mutual-kNN should be nondecreasing in K), then monotone PCHIP interpolation, then bisection to find the smallest K meeting the 150k/K=30 target degree. Clamp and flag if outside `[10,100]` — do not extrapolate.
4. **Output schema**: one row per (N,K) cell, columns per Codex analyzer #2's schema (dataset_variant, n_samples, k, strict_edge_count/avg_degree/zero_degree_fraction/median/p90, union edge/degree/composition counts+fractions pre- and post-repair, subreddit lift fields, run metadata). Integrity checks: `union_both_edge_count == strict_edge_count`; the four union composition fractions sum to 1.
5. **Stage-B short-list rule**: deterministic — anchor K=30, integer-rounded invariant-matched prediction (merged into anchor if within 1), plus one bracketing grid point on the opposite side of the prediction (log-distance nearest). Pseudocode from analyzer #2 to be implemented as-is.

## 执行模式
**Codex (external model), as a subagent** — per the user's explicit instruction this session ("using codex through ccg skill to write code as subagent"). Claude reviews the diff and runs verification; Claude does not write the implementation code directly. This also satisfies this project's own convention of not letting Codex hold long GPU jobs — Stage A is training-free and GPU-light (36 `mutual_knn` calls, tens of seconds each at worst), so Codex may run it itself as part of verification, with Claude spot-checking the final artifacts afterward.

## 步骤
1. `src/test/20260901_buddy_k_scaling/buddy_k_sweep.py` — new sweep driver: `SCALES` config (feature store + annotation path per N, using the corrected `_diverse` stores for 300k/500k), `load_scale()`, `strict_stats()`, `union_stats()` (calls `classify_edges`, derives composition fractions), `quality_stats()` (calls `subreddit_lift` with `top_k=None`), `run_cell()`, `derive_k_prediction()` (isotonic + PCHIP + bisection per analyzer #2's pseudocode), `select_shortlist()` (per analyzer #2's pseudocode), `main()` writing `stage_a_cells.csv`/`.json` + a short printed summary.
2. Run the script end-to-end (`conda activate CoSiR`) and confirm all 18 cells populate without error, integrity checks pass.
3. `.planning/2026-09-01-buddy-k-scaling-ablation/findings.md` — append the 18-row results summary, the derived K(N) rule, and the Stage-B short-list per size.
4. `.planning/2026-09-01-buddy-k-scaling-ablation/progress.md` — log the session's actions.
5. `.planning/2026-09-01-buddy-k-scaling-ablation/task_plan.md` — check off Phase 3 items, update Current Status/Next Step to point at Phase 4 (Stage B, gated on user review of 16.1's results).
6. `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` — append a "**Result (2026-09-01, ...)**" paragraph to 16.1 summarizing the outcome, matching this doc's existing convention (see 15.1/15.2's Result paragraphs).

## 影响范围
- 新增: `src/test/20260901_buddy_k_scaling/buddy_k_sweep.py`, `src/test/20260901_buddy_k_scaling/stage_a_cells.csv`, `src/test/20260901_buddy_k_scaling/stage_a_cells.json`
- 修改: `.planning/2026-09-01-buddy-k-scaling-ablation/{findings,progress,task_plan}.md`, `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
- 测试: no new pytest file required (this is a one-shot diagnostic script per project convention, same as `dim_hparam_study.py`); correctness is verified by the script's own integrity assertions (composition fractions sum to 1, `union_both_edge_count == strict_edge_count`) plus a successful full run.
