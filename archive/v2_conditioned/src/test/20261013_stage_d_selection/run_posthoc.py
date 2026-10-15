"""Stage (d) post-hoc diagnostics (final fix wave): SELECTION rows only, plus scorer-train rows for one check.

Everything here is **post-hoc, on selection rows, and informed no pre-registered decision**. No model is
trained and no held row is read. The only held-side numbers are re-printed from Task 7's stored JSON
(``src/test/20261014_stage_d_final/results/final_results.json``) and combined arithmetically.

Run from the repository root with the CoSiR environment:

    python src/test/20261013_stage_d_selection/run_posthoc.py --run       # compute, write results/posthoc_*.json
    python src/test/20261013_stage_d_selection/run_posthoc.py --tables    # reprint the tables from the JSON

What it computes (review finding in brackets):
1. [C1] The human swap test built on selection rows (``build_human_swap_episodes(..., seed=42)``, 1,024
   episodes) for the naive rule at beta in {0.3, 0.2, 0.1, G3's learned beta, 0.02}, for G3 at its learned
   beta, and for G3's interface with beta reset to 0.3, with paired bootstrap differences.
2. [I2] For G1-G5 on the selection label episodes: the condition-use gain Delta against the pre-registered
   naive (beta 0.3), against naive at the run's own learned beta, and the run's interface at beta 0.3,
   per direction and per label type. Delta is a difference of per-episode quantities, so
   Delta(run, naive@0.3) = Delta(naive@beta_run, naive@0.3) + Delta(run, naive@beta_run) exactly.
3. [C2] Four ceilings on the selection label episodes: the per-episode ceiling, its random-target null
   (a random distractor declared the positive), the cross-validated label oracle (one weight vector per
   target label), and the label oracle's random-target null, all at beta 0.3; plus the label oracle at G3's
   learned beta, compared with naive at that beta.
4. [I3] Mechanism check on 1,024 mined scorer-train episodes per condition source: how often CLIP alone,
   and the naive rule as beta goes 0.3 -> 0, rank a positive above a hard negative vs a random negative.
5. [I1, M2] From the stored held numbers only: the held standard errors, the power of criterion 1 if the
   held effect equalled the selection estimates, and the effect of dropping same-label wrong conditions.

Row-scope guards: the selection analyses NaN-mask CLIP features and codes outside the selection rows; the
mechanism check masks everything outside scorer-train rows. The stored Task 6 ranks of naive and G1-G5
are recomputed and compared as a reproduction check.
"""

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import multivariate_normal, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_FINAL_PATH = ROOT / "src/test/20261014_stage_d_final/run_final.py"
_spec = importlib.util.spec_from_file_location("run_final", _FINAL_PATH)
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)
sel = fin.sel                                           # Task 6's run_selection.py, imported once

from src.data.artelingo import load_artelingo  # noqa: E402
from src.eval.condition_eval import (build_human_swap_episodes, ceiling_ranks, condition_use_gain,  # noqa: E402
                                     human_swap_success, label_oracle_ranks, label_ranks, paired_bootstrap,
                                     swap_success_difference, wrong_condition)
from src.eval.label_episodes import label_episodes_sha256, standard_label_episodes  # noqa: E402
from src.model.condition_interface import ConditionalScorer, ResidualConditionInterface  # noqa: E402
from src.model.conditioning import conditional_score, pair_codes  # noqa: E402
from src.train.condition_episodes import mine_condition_episodes, pair_feature_units  # noqa: E402
from src.train.train_scorer import load_scorer_checkpoint  # noqa: E402

SEED = sel.SEED
LABELS, SCOPES, DIRECTIONS, RUNS = sel.LABELS, sel.SCOPES, sel.DIRECTIONS, sel.RUNS
BETA_FIXED = sel.BETA_FIXED
NAIVE_BETAS = (0.3, 0.2, 0.1, 0.02)                    # plus each run's learned beta
MECHANISM_BETAS = (0.3, 0.2, 0.1, 0.05, 0.02, 0.0)
N_SWAP, N_MINED = 1024, 1024
SOURCES = ("clip_cluster", "factor_combo", "community")
NUM_POSITIVE, NUM_HARD = 4, 6                           # mine_condition_episodes defaults
Z95 = float(norm.ppf(0.975))
LABEL = "post-hoc, selection rows, informed no pre-registered decision"
POSTHOC_JSON = sel.RESULTS / "posthoc_results.json"
POSTHOC_NPZ = sel.RESULTS / "posthoc_ranks.npz"
FINAL_JSON = fin.FINAL_JSON
log = sel.log


# ----------------------------------------------------------------------------- models

def naive_at(beta: float, scale: torch.Tensor) -> ConditionalScorer:
    """The naive rule (a fresh interface; its last layer is zero) at a given CLIP weight beta."""
    return ConditionalScorer(ResidualConditionInterface(scale), beta_init=beta).eval()


def at_beta(scorer: ConditionalScorer, beta: float) -> ConditionalScorer:
    """The same trained interface with beta reset (in place) to ``beta``."""
    with torch.no_grad():
        scorer.beta_raw.fill_(math.log(math.expm1(beta)))
    return scorer


def beta_of(scorer) -> float:
    return scorer.beta.detach().item()


def load_run(name: str) -> ConditionalScorer:
    scorer, config = load_scorer_checkpoint(sel.CKPT / f"{name}.pt", device="cpu")
    if dataclasses.asdict(config) != dataclasses.asdict(sel.run_config(name)):
        raise AssertionError(f"{name}: checkpoint config differs from the plan's run config")
    return scorer


def build_models(scale: torch.Tensor) -> tuple[dict, dict]:
    """Naive at the fixed betas and at every run's beta; G1-G5; G1-G5 with beta reset to 0.3."""
    runs = {name: load_run(name) for name in RUNS}
    betas = {name: beta_of(scorer) for name, scorer in runs.items()}
    models = {f"naive@{b:g}": naive_at(b, scale) for b in NAIVE_BETAS}
    for name in RUNS:
        models[f"naive@{name}beta"] = naive_at(betas[name], scale)
        models[name] = runs[name]
        models[f"{name}@0.3"] = at_beta(load_run(name), BETA_FIXED)
    return models, betas


# ----------------------------------------------------------------------------- helpers

def pct_block(block: dict) -> dict:
    return {"point": 100 * block["point"], "ci95": [100 * x for x in block["ci95"]]}


def gain_points(gain: dict) -> dict:
    return {k: pct_block(gain[k]) for k in (*DIRECTIONS, "mean")}


def gain_all_scopes(ranks: dict, a: str, b: str) -> dict:
    """Delta(a vs b) in R@1 points for every scope."""
    return {scope: gain_points(condition_use_gain(ranks[a]["right"][scope], ranks[a]["wrong"][scope],
                                                  ranks[b]["right"][scope], ranks[b]["wrong"][scope]))
            for scope in SCOPES}


def r1_diff(ranks: dict, a: str, b: str, scope: str = "pooled") -> dict:
    hit = lambda r: (np.asarray(r) <= 1).astype(float)  # noqa: E731
    return {d: pct_block(paired_bootstrap(hit(ranks[a]["right"][scope][d]) - hit(ranks[b]["right"][scope][d])))
            for d in DIRECTIONS}


def recall_points(per_label: dict) -> dict:
    pooled = sel.pool(per_label)
    return {scope: {d: 100 * float(np.mean(np.asarray((pooled if scope == "pooled" else per_label[scope])[d]) <= 1))
                    for d in DIRECTIONS} for scope in SCOPES}


def pairwise_accuracy(scores: torch.Tensor, a: slice, b: slice) -> float:
    """Share of (episode, column-in-a, column-in-b) triples with score_a > score_b; ties count one half."""
    sa, sb = scores[:, a][:, :, None], scores[:, b][:, None, :]
    return float(((sa > sb).float() + 0.5 * (sa == sb).float()).mean())


# ----------------------------------------------------------------------------- analyses

def selection_label_ranks(models: dict, data, cache, img, txt, img_codes, txt_codes):
    episodes, wrong, meta = {}, {}, {}
    in_sel = np.zeros(len(cache["groups"]), dtype=bool)
    in_sel[cache["selection"]] = True
    for label in LABELS:
        eps = standard_label_episodes(data, cache["groups"], cache["selection"], label, sel.N_SELECTION_EPISODES,
                                      seed=SEED)
        sha = label_episodes_sha256(eps)
        if sha != fin.SELECTION_SHA256[label]:
            raise AssertionError(f"{label} selection episodes differ from Task 6's: {sha}")
        if not in_sel[np.column_stack([eps.anchor, eps.positive, eps.supports, eps.contrasts, eps.distractors])].all():
            raise AssertionError(f"{label} episodes use rows outside the selection set")
        episodes[label], wrong[label] = eps, wrong_condition(eps, seed=SEED)
        meta[label] = {"n": int(len(eps.anchor)), "sha256": sha}
    ranks = {}
    for name, scorer in models.items():
        right = {label: label_ranks(scorer, img, txt, img_codes, txt_codes, episodes[label]) for label in LABELS}
        wr = {label: label_ranks(scorer, img, txt, img_codes, txt_codes, wrong[label]) for label in LABELS}
        ranks[name] = {"right": {**right, "pooled": sel.pool(right)}, "wrong": {**wr, "pooled": sel.pool(wr)}}
    return episodes, ranks, meta


def reproduction_check(ranks: dict) -> dict:
    """Recomputed ranks of naive (beta 0.3) and G1-G5 vs Task 6's stored ranks: identical-rank share."""
    stored = np.load(sel.RANKS_NPZ)
    out = {}
    for ours, theirs in (("naive@0.3", "naive"), *((r, r) for r in RUNS)):
        shares = [float(np.mean(np.asarray(ranks[ours][kind][scope][d]) == stored[f"{theirs}__{kind}__{scope}__{d}"]))
                  for kind in ("right", "wrong") for scope in SCOPES for d in DIRECTIONS]
        out[ours] = min(shares)
    if min(out.values()) < 0.999:
        raise AssertionError(f"recomputed ranks differ from Task 6's stored ranks: {out}")
    task6 = json.loads(sel.RESULTS_JSON.read_text())["models"]
    gains_equal = {name: all(
        gain_points(condition_use_gain(ranks[name]["right"][s], ranks[name]["wrong"][s], ranks["naive@0.3"]["right"][s],
                                       ranks["naive@0.3"]["wrong"][s])) == gain_points(task6[name]["gain"][s])
        for s in SCOPES) for name in RUNS}
    return {"min_identical_rank_share": out, "gain_equals_task6": gains_equal}


def decomposition(ranks: dict, betas: dict) -> dict:
    out = {}
    for name in RUNS:
        total = gain_all_scopes(ranks, name, "naive@0.3")
        drop = gain_all_scopes(ranks, f"naive@{name}beta", "naive@0.3")
        beyond = gain_all_scopes(ranks, name, f"naive@{name}beta")
        pooled = total["pooled"]["mean"]["point"]
        out[name] = {"beta": betas[name], "total_vs_naive0.3": total, "beta_drop": drop,
                     "beyond_beta_drop": beyond, "interface_at_0.3": gain_all_scopes(ranks, f"{name}@0.3", "naive@0.3"),
                     "beta_drop_share_of_total_pooled_mean": drop["pooled"]["mean"]["point"] / pooled if pooled else None,
                     "additivity_error_pooled_mean": abs(pooled - drop["pooled"]["mean"]["point"]
                                                         - beyond["pooled"]["mean"]["point"]),
                     "r1_minus_naive_at_own_beta": r1_diff(ranks, name, f"naive@{name}beta"),
                     "r1_minus_naive0.3": r1_diff(ranks, name, "naive@0.3")}
    return out


def swap_analysis(models: dict, data, cache, img, txt, img_codes, txt_codes) -> dict:
    rows = cache["selection"]
    eps = build_human_swap_episodes(data, cache["groups"], rows, N_SWAP, seed=SEED)
    checks = fin.swap_episode_checks(data, cache["groups"], rows, eps)
    names = [n for n in models if n.startswith("naive@") or n in RUNS or n.endswith("@0.3")]
    success, halves = {}, {}
    for name in names:
        success[name] = human_swap_success(models[name], img, txt, img_codes, txt_codes, eps)
        comp = fin.swap_components(models[name], img, txt, img_codes, txt_codes, eps)
        if not all(np.array_equal(comp[d]["both"], success[name][d]) for d in DIRECTIONS):
            raise AssertionError(f"{name}: swap components disagree with human_swap_success")
        halves[name] = {d: {k: 100 * float(np.mean(v)) for k, v in comp[d].items()} for d in DIRECTIONS}

    def diff(a, b):
        out = swap_success_difference(success[a], success[b])
        return {k: pct_block(out[k]) for k in (*DIRECTIONS, "pooled")}

    pairs = {f"{n}_minus_naive@0.3": (n, "naive@0.3") for n in names if n != "naive@0.3"}
    pairs.update({f"{r}_minus_naive@{r}beta": (r, f"naive@{r}beta") for r in RUNS})
    smoke = json.loads(fin.SMOKE_JSON.read_text()) if fin.SMOKE_JSON.exists() else None
    smoke_match = None
    if smoke is not None:
        rates = smoke["criterion2"]["success_rates"]
        smoke_match = {"sha256_equal": smoke["meta"]["swap_episodes"]["sha256"] == checks["sha256"],
                       "naive_rates_equal": all(rates["naive"][d] == float(np.mean(success["naive@0.3"][d]))
                                                for d in DIRECTIONS),
                       "G3_rates_equal": all(rates["G3_seed42"][d] == float(np.mean(success["G3"][d]))
                                             for d in DIRECTIONS)}
    return {"episodes": checks, "equals_task7_discarded_smoke_run": smoke_match,
            "success": {n: {d: 100 * float(np.mean(v[d])) for d in DIRECTIONS} for n, v in success.items()},
            "halves": halves, "differences": {k: diff(a, b) for k, (a, b) in pairs.items()}}


def ceiling_analysis(ranks: dict, episodes: dict, betas: dict, img, txt, img_codes, txt_codes) -> tuple[dict, dict]:
    """The four ceilings at beta 0.3, plus the label oracle at G3's learned beta (compared with naive there)."""
    rng = np.random.default_rng(SEED)
    targets = {label: rng.integers(1, episodes[label].distractors.shape[1] + 1, len(episodes[label].anchor))
               for label in LABELS}
    timings, per = {}, {}
    for kind in ("ceiling", "ceiling_null", "label_oracle", "label_oracle_null", "label_oracle@G3beta"):
        t0 = perf_counter()
        per[kind] = {}
        beta = betas["G3"] if kind.endswith("@G3beta") else BETA_FIXED
        for label in LABELS:
            target = targets[label] if kind.endswith("_null") else 0
            if kind.startswith("ceiling"):
                per[kind][label] = ceiling_ranks(img, txt, img_codes, txt_codes, episodes[label], beta=beta,
                                                 target_column=target)
            else:
                per[kind][label] = label_oracle_ranks(img, txt, img_codes, txt_codes, episodes[label],
                                                      beta=beta, folds=2, steps=200, lr=0.1, seed=SEED,
                                                      target_column=target)
        timings[kind] = perf_counter() - t0
        log(f"{kind}: {timings[kind]:.1f} s")
    stored = np.load(sel.RANKS_NPZ)
    reproduces = min(float(np.mean(per["ceiling"][label][d] == stored[f"ceiling__right__{label}__{d}"]))
                     for label in LABELS for d in DIRECTIONS)
    oracle = {k: {"right": {**per[k], "pooled": sel.pool(per[k])}} for k in ("label_oracle", "label_oracle@G3beta")}
    out = {"recall_r1": {"naive@0.3": ranks_recall(ranks["naive@0.3"]["right"]),
                         **{k: recall_points(v) for k, v in per.items()},
                         "naive@G3beta": ranks_recall(ranks["naive@G3beta"]["right"])},
           "chance_r1": 100 / 13,
           "label_oracle_minus_naive_r1": {scope: r1_diff({"o": oracle["label_oracle"], "n": ranks["naive@0.3"]},
                                                          "o", "n", scope) for scope in SCOPES},
           "label_oracle_at_G3beta_minus_naive_at_G3beta_r1": {
               scope: r1_diff({"o": oracle["label_oracle@G3beta"], "n": ranks["naive@G3beta"]}, "o", "n", scope)
               for scope in SCOPES},
           "g3_beta": betas["G3"],
           "ceiling_reproduces_task6_identical_rank_share": reproduces,
           "seconds": timings, "null_targets": "one random distractor per episode, rng(42).integers(1, 13)",
           "label_oracle_settings": {"folds": 2, "steps": 200, "lr": 0.1, "seed": SEED, "beta": BETA_FIXED}}
    return out, per


def ranks_recall(block: dict) -> dict:
    return {scope: {d: 100 * float(np.mean(np.asarray(block[scope][d]) <= 1)) for d in DIRECTIONS} for scope in SCOPES}


def mechanism_check(data, cache, scale: torch.Tensor, g3: ConditionalScorer) -> dict:
    """Scorer-train rows only: positive-over-hard vs positive-over-random accuracy on mined training episodes."""
    st = cache["scorer_train"]
    img, txt, ic, tc = sel.scoped_inputs(data, cache, st)
    units = pair_feature_units(img, txt)
    allowed = np.zeros(len(cache["groups"]), dtype=bool)
    allowed[st] = True
    naive = naive_at(BETA_FIXED, scale)
    pos, hard, rnd = slice(0, NUM_POSITIVE), slice(NUM_POSITIVE, NUM_POSITIVE + NUM_HARD), slice(NUM_POSITIVE + NUM_HARD, None)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    out = {}
    for name in SOURCES:
        source = sel.build_source(name, cache, 0.5 * (ic + tc))
        ep = mine_condition_episodes(source, units, cache["groups"], N_MINED, np.random.default_rng(SEED), 4,
                                     hard_pool=2048)
        rows = np.column_stack([ep.anchor[:, None], ep.supports, ep.contrasts, ep.candidates])
        if not allowed[rows].all():
            raise AssertionError(f"{name}: a mined row lies outside scorer-train")
        if ep.candidates.shape[1] != 16 or not ep.positive_mask[:, pos].all() or ep.positive_mask[:, NUM_POSITIVE:].any():
            raise AssertionError(f"{name}: unexpected candidate layout")
        with torch.no_grad():
            support = pair_codes(t(ic[ep.supports]), t(tc[ep.supports]))
            contrast = pair_codes(t(ic[ep.contrasts]), t(tc[ep.contrasts]))
            w = {"naive": naive.weights(support, contrast), "G3": g3.weights(support, contrast)}
            block = {"clip_only": {}, "naive": {f"{b:g}": {} for b in MECHANISM_BETAS}, "G3_learned": {}}
            for d in DIRECTIONS:
                qf, cf, qc, cc = (img, txt, ic, tc) if d == "i2t" else (txt, img, tc, ic)
                args = (t(qf[ep.anchor]), t(cf[ep.candidates]), t(qc[ep.anchor]), t(cc[ep.candidates]))
                zero = torch.zeros_like(w["naive"])
                scored = {("clip_only",): conditional_score(*args, zero, 1.0),
                          ("G3_learned",): conditional_score(*args, w["G3"], beta_of(g3))}
                for b in MECHANISM_BETAS:
                    scored[("naive", f"{b:g}")] = conditional_score(*args, w["naive"], b)
                for key, s in scored.items():
                    if not torch.isfinite(s).all():
                        raise AssertionError(f"{name}: non-finite scores ({key})")
                    target = block[key[0]] if len(key) == 1 else block[key[0]][key[1]]
                    target[d] = {"pos_over_hard": pairwise_accuracy(s, pos, hard),
                                 "pos_over_random": pairwise_accuracy(s, pos, rnd)}
        out[name] = block
        log(f"mechanism {name}: CLIP-only {block['clip_only']}")
    return {"episodes_per_source": N_MINED, "g3_beta": beta_of(g3), "sources": out,
            "note": "pairwise accuracy; ties count one half; candidates = 4 positives, 6 hard, 6 random negatives"}


def held_power_and_rescale(selection_gain: dict, corr: float) -> dict:
    """From stored numbers only: held SEs (from the stored 95% CIs), power, and the same-label rescale (M2)."""
    held = json.loads(FINAL_JSON.read_text())
    c1 = held["criterion1"]["by_seed"]["G3_seed42"]
    se = {d: 100 * (c1[d]["ci95"][1] - c1[d]["ci95"][0]) / (2 * Z95) for d in DIRECTIONS}
    point = {d: 100 * c1[d]["point"] for d in DIRECTIONS}
    ci_held = {d: [100 * x for x in c1[d]["ci95"]] for d in DIRECTIONS}
    mu = {d: 100 * selection_gain[d]["point"] for d in DIRECTIONS}
    shift = {d: mu[d] / se[d] - Z95 for d in DIRECTIONS}                   # P(LB > 0) = Phi(shift)
    each = {d: float(norm.cdf(shift[d])) for d in DIRECTIONS}
    joint_corr = float(multivariate_normal(mean=[0, 0], cov=[[1, corr], [corr, 1]]).cdf([shift["i2t"], shift["t2i"]]))
    eps = held["meta"]["episodes"]
    n = {label: eps[label]["n"] for label in LABELS}
    f = sum(eps[label]["wrong_same_label_fraction"] * n[label] for label in LABELS) / sum(n.values())
    k_point, k_se = 1 / (1 - f), 1 / math.sqrt(1 - f)
    rescaled = {d: {"point": point[d] * k_point, "se": se[d] * k_se,
                    "lower_bound": point[d] * k_point - Z95 * se[d] * k_se,
                    "z_before": point[d] / se[d], "z_after": point[d] * k_point / (se[d] * k_se)}
                for d in DIRECTIONS}
    return {"source": str(FINAL_JSON.relative_to(ROOT)), "note": "normal approximation; stored held numbers only",
            "held_point": point, "held_ci95": ci_held, "held_se": se, "selection_point": mu,
            "selection_point_inside_held_ci": {d: ci_held[d][0] <= mu[d] <= ci_held[d][1] for d in DIRECTIONS},
            "power_each_direction": each, "power_both_independent": each["i2t"] * each["t2i"],
            "power_both_with_selection_correlation": joint_corr, "selection_i2t_t2i_correlation": corr,
            "same_label": {"pooled_fraction": f, "point_factor": k_point, "se_factor": k_se,
                           "z_factor": k_point / k_se, "rescaled": rescaled,
                           "assumption": "same-label episodes contribute 0 to Delta; others keep their variance"}}


def selection_direction_correlation(ranks: dict) -> float:
    """Correlation of G3's per-episode Delta contributions (vs naive 0.3) between i2t and t2i, selection rows."""
    hit = lambda r: (np.asarray(r) <= 1).astype(float)  # noqa: E731
    g, n = ranks["G3"], ranks["naive@0.3"]
    per = {d: (hit(g["right"]["pooled"][d]) - hit(g["wrong"]["pooled"][d]))
           - (hit(n["right"]["pooled"][d]) - hit(n["wrong"]["pooled"][d])) for d in DIRECTIONS}
    return float(np.corrcoef(per["i2t"], per["t2i"])[0, 1])


# ----------------------------------------------------------------------------- run

def run() -> dict:
    started = perf_counter()
    torch.manual_seed(SEED)
    cache, _ = sel.load_prepared()
    data = load_artelingo()
    scale = torch.as_tensor(cache["factor_scale"])
    img, txt, img_codes, txt_codes = sel.scoped_inputs(data, cache, cache["selection"])   # selection rows only
    models, betas = build_models(scale)
    log(f"Learned betas: {betas}")

    episodes, ranks, meta_eps = selection_label_ranks(models, data, cache, img, txt, img_codes, txt_codes)
    repro = reproduction_check(ranks)
    log(f"Reproduction of Task 6 ranks: {repro}")
    results = {"label": LABEL, "betas": betas,
               "recall_r1": {name: ranks_recall(block["right"]) for name, block in ranks.items()},
               "recall_r1_wrong": {name: ranks_recall(block["wrong"]) for name, block in ranks.items()},
               "naive_beta_sweep": {name: gain_all_scopes(ranks, name, "naive@0.3")
                                    for name in models if name.startswith("naive@") and name != "naive@0.3"},
               "decomposition": decomposition(ranks, betas)}
    log("Decomposition done")
    results["swap"] = swap_analysis(models, data, cache, img, txt, img_codes, txt_codes)
    log(f"Swap test on selection rows: {results['swap']['success']['naive@0.3']} naive, "
        f"{results['swap']['success']['G3']} G3; smoke-run match {results['swap']['equals_task7_discarded_smoke_run']}")
    results["ceilings"], ceiling_ranks_out = ceiling_analysis(ranks, episodes, betas, img, txt, img_codes, txt_codes)
    log(f"Ceilings R@1 pooled: { {k: v['pooled'] for k, v in results['ceilings']['recall_r1'].items()} }")
    results["mechanism"] = mechanism_check(data, cache, scale, load_run("G3"))
    corr = selection_direction_correlation(ranks)
    task6_g3 = json.loads(sel.RESULTS_JSON.read_text())["models"]["G3"]["gain"]["pooled"]
    results["held_power"] = held_power_and_rescale(task6_g3, corr)
    results["meta"] = {"episodes": meta_eps, "reproduction": repro, "torch_threads": torch.get_num_threads(),
                       "versions": {"torch": torch.__version__, "numpy": np.__version__},
                       "rows": {"selection_analyses": "selection rows only (everything else NaN-masked)",
                                "mechanism_check": "scorer-train rows only", "held": "not read; stored JSON only"},
                       "seconds": perf_counter() - started}
    POSTHOC_JSON.write_text(json.dumps(results, indent=2))
    np.savez(POSTHOC_NPZ, **{f"{kind}__{label}__{d}": np.asarray(v[label][d])
                             for kind, v in ceiling_ranks_out.items() for label in LABELS for d in DIRECTIONS})
    log(f"Post-hoc run in {results['meta']['seconds']:.1f} s -> {POSTHOC_JSON}")
    return results


# ----------------------------------------------------------------------------- tables

def ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['point']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def tables(path: Path = POSTHOC_JSON) -> None:
    res = json.loads(path.read_text())
    out = [f"All numbers: {res['label']}.\n"]
    b = res["betas"]
    out.append("Learned beta: " + ", ".join(f"{k} {v:.4f}" for k, v in b.items()) + "\n")

    out.append("### C1: human swap test on selection rows (1,024 episodes, seed 42), success %\n")
    sw = res["swap"]
    out.append(f"Episodes: {sw['episodes']}; equal to Task 7's discarded smoke run: {sw['equals_task7_discarded_smoke_run']}\n")
    out.append("| Scorer | beta | Success i2t | Success t2i | Diff vs naive@0.3, pooled | Diff i2t | Diff t2i |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    order = ["naive@0.3", "naive@0.2", "naive@0.1", "naive@G3beta", "naive@0.02", "G3", "G3@0.3"]
    for name in order:
        s = sw["success"][name]
        beta = {"naive@G3beta": b["G3"], "G3": b["G3"], "G3@0.3": 0.3}.get(name, None)
        beta = beta if beta is not None else float(name.split("@")[1])
        if name == "naive@0.3":
            out.append(f"| {name} | {beta:.4f} | {s['i2t']:.2f} | {s['t2i']:.2f} | — | — | — |")
            continue
        dd = sw["differences"][f"{name}_minus_naive@0.3"]
        out.append(f"| {name} | {beta:.4f} | {s['i2t']:.2f} | {s['t2i']:.2f} | {ci(dd['pooled'])} | {ci(dd['i2t'])} | "
                   f"{ci(dd['t2i'])} |")
    out.append("")
    out.append("| Paired difference | Pooled | i2t | t2i |")
    out.append("|---|---:|---:|---:|")
    for r in RUNS:
        dd = sw["differences"][f"{r}_minus_naive@{r}beta"]
        out.append(f"| {r} − naive at {r}'s beta ({b[r]:.4f}) | {ci(dd['pooled'])} | {ci(dd['i2t'])} | {ci(dd['t2i'])} |")
    for r in RUNS:
        dd = sw["differences"][f"{r}@0.3_minus_naive@0.3"]
        out.append(f"| {r} interface at beta 0.3 − naive@0.3 | {ci(dd['pooled'])} | {ci(dd['i2t'])} | {ci(dd['t2i'])} |")
    out.append("")
    out.append("Swap halves (%, i2t emotion / style / both; t2i emotion / style / both):\n")
    for name in order:
        h = sw["halves"][name]
        out.append(f"- {name}: i2t {h['i2t']['emo_ok']:.2f} / {h['i2t']['style_ok']:.2f} / {h['i2t']['both']:.2f}; "
                   f"t2i {h['t2i']['emo_ok']:.2f} / {h['t2i']['style_ok']:.2f} / {h['t2i']['both']:.2f}")
    out.append("")

    out.append("### I2: condition-use gain Delta decomposed (selection label episodes, pooled, R@1 points)\n")
    out.append("| Run | beta | Delta vs naive@0.3 (pre-registered) | beta drop: naive@own beta − naive@0.3 | "
               "Beyond the beta drop: run − naive@own beta | Interface at beta 0.3 − naive@0.3 | beta-drop share |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    dec = res["decomposition"]
    for r in RUNS:
        x = dec[r]
        out.append(f"| {r} | {x['beta']:.4f} | {ci(x['total_vs_naive0.3']['pooled']['mean'])} | "
                   f"{ci(x['beta_drop']['pooled']['mean'])} | {ci(x['beyond_beta_drop']['pooled']['mean'])} | "
                   f"{ci(x['interface_at_0.3']['pooled']['mean'])} | {100 * x['beta_drop_share_of_total_pooled_mean']:.0f}% |")
    out.append("")
    out.append("Beyond the beta drop (run − naive at the run's own beta), per direction and label type:\n")
    out.append("| Run | pooled i2t | pooled t2i | emotion mean | art style mean |")
    out.append("|---|---:|---:|---:|---:|")
    for r in RUNS:
        x = dec[r]["beyond_beta_drop"]
        out.append(f"| {r} | {ci(x['pooled']['i2t'])} | {ci(x['pooled']['t2i'])} | {ci(x['emotion']['mean'])} | "
                   f"{ci(x['art_style']['mean'])} |")
    out.append("")
    out.append("Interface at beta 0.3 − naive@0.3, per direction and label type:\n")
    out.append("| Run | pooled i2t | pooled t2i | emotion mean | art style mean |")
    out.append("|---|---:|---:|---:|---:|")
    for r in RUNS:
        x = dec[r]["interface_at_0.3"]
        out.append(f"| {r} | {ci(x['pooled']['i2t'])} | {ci(x['pooled']['t2i'])} | {ci(x['emotion']['mean'])} | "
                   f"{ci(x['art_style']['mean'])} |")
    out.append("")
    out.append("beta drop alone (naive at the run's beta − naive@0.3), per direction and label type:\n")
    out.append("| Run | pooled i2t | pooled t2i | emotion mean | art style mean |")
    out.append("|---|---:|---:|---:|---:|")
    for r in RUNS:
        x = dec[r]["beta_drop"]
        out.append(f"| {r} | {ci(x['pooled']['i2t'])} | {ci(x['pooled']['t2i'])} | {ci(x['emotion']['mean'])} | "
                   f"{ci(x['art_style']['mean'])} |")
    out.append("")
    out.append("Plain R@1, run − naive at own beta (pooled): " + "; ".join(
        f"{r} {ci(dec[r]['r1_minus_naive_at_own_beta']['i2t'])} / {ci(dec[r]['r1_minus_naive_at_own_beta']['t2i'])}"
        for r in RUNS) + "\n")
    out.append("Naive beta sweep, Delta vs naive@0.3 (pooled mean; i2t; t2i): " + "; ".join(
        f"{k} {ci(v['pooled']['mean'])}; {ci(v['pooled']['i2t'])}; {ci(v['pooled']['t2i'])}"
        for k, v in res["naive_beta_sweep"].items()) + "\n")

    out.append("### C2: ceilings on the selection label episodes (R@1 %, beta 0.3)\n")
    c = res["ceilings"]
    out.append("| Scorer | pooled i2t | pooled t2i | emotion i2t | emotion t2i | art style i2t | art style t2i |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, v in c["recall_r1"].items():
        out.append(f"| {k} | " + " | ".join(f"{v[s][d]:.2f}" for s in SCOPES for d in DIRECTIONS) + " |")
    out.append(f"\nChance {c['chance_r1']:.2f}%. Rows without a beta suffix use beta 0.3; @G3beta rows use "
               f"{c['g3_beta']:.4f}.\n")
    for key, what in (("label_oracle_minus_naive_r1", "Label oracle − naive, beta 0.3"),
                      ("label_oracle_at_G3beta_minus_naive_at_G3beta_r1", "Label oracle − naive, both at G3's beta")):
        out.append(f"- {what} (paired R@1, i2t / t2i): " + "; ".join(
            f"{s} {ci(c[key][s]['i2t'])} / {ci(c[key][s]['t2i'])}" for s in SCOPES))
    out.append(f"Ceiling reproduces Task 6's stored ceiling ranks: identical-rank share "
               f"{c['ceiling_reproduces_task6_identical_rank_share']:.4f}. Seconds: {c['seconds']}\n")

    out.append("### I3: mechanism check on mined scorer-train episodes (pairwise accuracy, %)\n")
    m = res["mechanism"]
    out.append("| Source | Scorer | pos>hard i2t | pos>hard t2i | pos>random i2t | pos>random t2i |")
    out.append("|---|---|---:|---:|---:|---:|")
    for src, block in m["sources"].items():
        rows = [("CLIP only", block["clip_only"])] + [(f"naive beta {k}", v) for k, v in block["naive"].items()] \
            + [(f"G3 (beta {m['g3_beta']:.4f})", block["G3_learned"])]
        for name, v in rows:
            out.append(f"| {src} | {name} | {100 * v['i2t']['pos_over_hard']:.1f} | {100 * v['t2i']['pos_over_hard']:.1f} | "
                       f"{100 * v['i2t']['pos_over_random']:.1f} | {100 * v['t2i']['pos_over_random']:.1f} |")
    out.append("")

    out.append("### I1 / M2: from the stored held numbers (normal approximation)\n")
    hp = res["held_power"]
    for d in DIRECTIONS:
        out.append(f"- {d}: held {hp['held_point'][d]:+.2f} {hp['held_ci95'][d]}, SE {hp['held_se'][d]:.3f}; selection "
                   f"point {hp['selection_point'][d]:+.2f} inside held CI: {hp['selection_point_inside_held_ci'][d]}; "
                   f"P(LB > 0 | effect = selection point) {hp['power_each_direction'][d]:.3f}")
    out.append(f"- P(both LB > 0): {hp['power_both_independent']:.3f} (independent), "
               f"{hp['power_both_with_selection_correlation']:.3f} (selection i2t/t2i correlation "
               f"{hp['selection_i2t_t2i_correlation']:.3f})")
    sl = hp["same_label"]
    out.append(f"- Same-label rescale: pooled fraction {sl['pooled_fraction']:.4f}; point x{sl['point_factor']:.3f}, SE "
               f"x{sl['se_factor']:.3f}, z x{sl['z_factor']:.3f}; lower bounds " + ", ".join(
                   f"{d} {sl['rescaled'][d]['lower_bound']:+.2f}" for d in DIRECTIONS))
    out.append(f"\nReproduction of Task 6: {res['meta']['reproduction']}; torch threads {res['meta']['torch_threads']}; "
               f"{res['meta']['seconds']:.0f} s")
    print("\n".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    tables()


if __name__ == "__main__":
    main()
