"""CoSiR v2 factor-learning grid: post-hoc diagnostics on selection rows (final-review fix wave, F6).

Every number this script writes is a POST-HOC diagnostic on selection rows. It informed no pre-registered
decision: the spec §6 rule had already stopped at "no cell qualifies" before any of it ran. Nothing is
retrained. The only fitted models are diagnostic multinomial logistic probes, fit on scorer-train rows and
scored on selection rows; human labels are used only to fit or score these diagnostics, as in the headroom
probe. Val and held rows are never read: CLIP features are NaN outside scorer-train + selection rows right after
loading, and label arrays are blanked outside those rows (both asserted).

Run from the repository root with the CoSiR environment:

    python src/test/20261016_factor_learning_grid/run_posthoc.py --run      # -> results/posthoc_results.json
    python src/test/20261016_factor_learning_grid/run_posthoc.py --tables   # reprint every table from the JSON

Analyses (numbering follows the fix-wave findings list, F6):

  1. Per-modality linear probes on each model's 32-d codes (R3, C0, A, S, AS): emotion from caption codes, art
     style from image codes, and the cross terms. Standardized codes, multinomial logistic regression (C 1.0),
     fit on scorer-train rows, top-1 accuracy on selection rows with the fit-majority baseline; differences to C0
     with a painting-cluster bootstrap (selection paintings resampled).
  2. Within-painting caption-residual emotion probe: caption code minus the mean caption code of the same
     painting's rows in the same part (scorer-train for the fit, selection for the score), paintings with at
     least 2 rows; raw CLIP captions as the reference (the headroom probe's 47.0%). Also the within-painting
     share of caption-code variance.
  3. From the stored results (selection_results.json / selection_ranks.npz): the emotion guard's power, the
     per-direction emotion table on the beta grid, and the balance-matched S vs C0 comparison (linear
     interpolation on the stored beta grid, plus a direct re-scoring of the same episodes at the matched beta).
  4. AMI of each model's argmax pair-code factor with the CLIP image clusters (stage (d) cache ``clip_image``,
     which labels scorer-train rows only, so scorer-train rows), and with art style and emotion (selection rows).
  5. Per-target S - C0 at beta 0.3 and beta 0 from the stored ranks (per emotion, per art style, with counts).

Plus, for the report's attribution fixes: the graph-term confound behind cell A (same-painting pairs in the
scorer-train content graph, and their share of the graph term's positive pairs with and without the shared
painting sampler, on the first training batches, replayed exactly); ties at beta 0 (the naive rule's R@1 with
random tie-breaking, next to the project's tie-aware convention, where a tie is a miss); same-beta oracle minus
naive columns; and a condition-loss history summary.
"""

import argparse
import dataclasses
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, triu
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_GRID_PATH = HERE / "run_grid.py"
_gspec = importlib.util.spec_from_file_location("run_grid", _GRID_PATH)
grid = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(grid)
sel, probe = grid.sel, grid.probe

from src.data.artelingo import load_artelingo  # noqa: E402
from src.eval.condition_eval import paired_bootstrap  # noqa: E402
from src.eval.label_episodes import label_episode_weights  # noqa: E402
from src.model.conditioning import conditional_score  # noqa: E402
from src.train.train_factors import GroupRows  # noqa: E402

SEED = 42
MODELS, LABELS, DIRECTIONS, SCOPES = grid.MODELS, grid.LABELS, grid.DIRECTIONS, grid.SCOPES
BETAS, key = grid.BETAS, grid.key
PROBE_SETTINGS = {"C": 1.0, "max_iter": 1000}
PROBE_TASKS = (("txt", "emotion"), ("img", "art_style"), ("txt", "art_style"), ("img", "emotion"))
PROBE_JOBS = 12
CLUSTER_BOOT = 2000                              # painting-cluster bootstrap resamples (row-level probe diffs)
GUARD, HALF_WIDTH_Z = grid.GUARD_POINTS, 1.959964
SAMPLER_BATCHES = 50                             # training steps 1..50 replayed for the graph-term shares
POSTHOC_JSON = grid.RESULTS / "posthoc_results.json"
LABEL = ("POST-HOC diagnostic on selection rows (probes fit on scorer-train rows); informed no pre-registered "
         "decision; val and held rows never read")
log = sel.log


# ----------------------------------------------------------------------------- loading and row scope

def load():
    data = load_artelingo()
    cache, prep, meta, graph = grid.load_grid()
    st, sl = cache["scorer_train"], cache["selection"]
    n_rows = len(cache["groups"])
    if np.intersect1d(cache["groups"][st], cache["groups"][sl]).size:
        raise AssertionError("a painting spans scorer-train and selection")
    keep = np.concatenate([st, sl])
    data = dataclasses.replace(data, img_features=sel.masked(data.img_features, keep),
                               txt_features=sel.masked(data.txt_features, keep))
    allowed = grid._row_mask(n_rows, keep)
    for name, arr in (("img features", data.img_features), ("txt features", data.txt_features)):
        probe.assert_row_scope(name, arr, allowed)
    labels = {"emotion": np.where(allowed, np.asarray(data.emotions), ""),
              "art_style": np.where(allowed, np.asarray(data.art_styles), "")}
    return data, cache, prep, meta, graph, labels, allowed


def label_rows(labels: dict, label: str, rows: np.ndarray) -> np.ndarray:
    y = labels[label][rows]
    if (y == "").any():
        raise AssertionError(f"{label}: label read outside scorer-train + selection rows")
    return y


# ----------------------------------------------------------------------------- probes (analyses 1 and 2)

def _fit_probe(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray) -> dict:
    t0 = perf_counter()
    mu, sd = x_fit.mean(axis=0), x_fit.std(axis=0)
    sd = np.maximum(sd, 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(**PROBE_SETTINGS, random_state=SEED).fit((x_fit - mu) / sd, y_fit)
    n_iter = int(np.max(model.n_iter_))
    return {"pred": model.predict((x_eval - mu) / sd), "n_iter": n_iter,
            "converged": n_iter < PROBE_SETTINGS["max_iter"], "seconds": perf_counter() - t0}


def majority(y_fit: np.ndarray, y_eval: np.ndarray) -> tuple[str, float]:
    values, counts = np.unique(y_fit, return_counts=True)
    top = values[counts.argmax()]
    return str(top), float(np.mean(y_eval == top))


def cluster_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray, clusters: np.ndarray) -> dict:
    """Accuracy difference a - b (points) with a 95% CI from resampling clusters (paintings) with replacement."""
    _, inv = np.unique(clusters, return_inverse=True)
    diff = np.bincount(inv, weights=correct_a.astype(np.float64) - correct_b.astype(np.float64))
    count = np.bincount(inv).astype(np.float64)
    idx = np.random.default_rng(SEED).integers(0, len(count), (CLUSTER_BOOT, len(count)))
    boots = diff[idx].sum(axis=1) / count[idx].sum(axis=1)
    return {"point": 100 * float(diff.sum() / count.sum()),
            "ci95": [100 * float(np.percentile(boots, 2.5)), 100 * float(np.percentile(boots, 97.5))]}


def within_painting_residual(x: np.ndarray, rows: np.ndarray, groups: np.ndarray):
    """x[rows] minus the mean of the same painting's rows (within ``rows``); keep = painting has >= 2 rows."""
    g = groups[rows]
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    xr = x[rows].astype(np.float64)
    means = np.zeros((cnt.size, x.shape[1]))
    np.add.at(means, inv, xr)
    means /= cnt[:, None]
    keep = cnt[inv] >= 2
    return xr - means[inv], keep


def probe_analyses(codes: dict, data, cache: dict, labels: dict) -> tuple[dict, dict]:
    st, sl, groups = cache["scorer_train"], cache["selection"], cache["groups"]
    y = {lab: (label_rows(labels, lab, st), label_rows(labels, lab, sl)) for lab in LABELS}
    jobs, meta = [], []
    for m in MODELS:
        ic, tc = codes[m]
        side = {"img": ic, "txt": tc}
        for modality, lab in PROBE_TASKS:
            jobs.append((side[modality][st].astype(np.float64), y[lab][0], side[modality][sl].astype(np.float64)))
            meta.append(("code", m, modality, lab))
    residual_keep = {}
    for m in (*MODELS, "clip512"):
        x = data.txt_features if m == "clip512" else codes[m][1]
        r_fit, k_fit = within_painting_residual(x, st, groups)
        r_sel, k_sel = within_painting_residual(x, sl, groups)
        xf = x[st].astype(np.float64)
        share = float((r_fit[k_fit] ** 2).sum() / ((xf[k_fit] - xf[k_fit].mean(axis=0)) ** 2).sum())
        residual_keep[m] = (k_fit, k_sel, share)
        jobs.append((r_fit[k_fit], y["emotion"][0][k_fit], r_sel[k_sel]))
        meta.append(("residual", m, "txt", "emotion"))
    log(f"Probes: {len(jobs)} logistic fits on {PROBE_JOBS} workers")
    t0 = perf_counter()
    fitted = Parallel(n_jobs=PROBE_JOBS, backend="loky")(delayed(_fit_probe)(*j) for j in jobs)
    log(f"Probes done in {perf_counter() - t0:.1f} s")

    code_probes, residual_probes, correct = {}, {}, {}
    sel_groups = groups[sl]
    for (kind, m, modality, lab), out in zip(meta, fitted):
        if kind == "code":
            truth = y[lab][1]
            ok = out["pred"] == truth
            top, base = majority(y[lab][0], truth)
            correct[(kind, m, modality, lab)] = (ok, sel_groups)
            code_probes.setdefault(m, {})[f"{modality}->{lab}"] = {
                "accuracy": 100 * float(ok.mean()), "majority": 100 * base, "majority_class": top,
                "n_fit": int(len(st)), "n_eval": int(len(sl)), "n_iter": out["n_iter"],
                "converged": out["converged"], "seconds": out["seconds"]}
        else:
            k_fit, k_sel, share = residual_keep[m]
            truth = y["emotion"][1][k_sel]
            ok = out["pred"] == truth
            top, base = majority(y["emotion"][0][k_fit], truth)
            correct[(kind, m, modality, lab)] = (ok, sel_groups[k_sel])
            residual_probes[m] = {"accuracy": 100 * float(ok.mean()), "majority": 100 * base, "majority_class": top,
                                  "n_fit": int(k_fit.sum()), "n_eval": int(k_sel.sum()),
                                  "within_painting_variance_share": share, "n_iter": out["n_iter"],
                                  "converged": out["converged"], "seconds": out["seconds"]}
    for m in MODELS:
        if m == "C0":
            continue
        for modality, lab in PROBE_TASKS:
            a, ga = correct[("code", m, modality, lab)]
            b, _ = correct[("code", "C0", modality, lab)]
            code_probes[m][f"{modality}->{lab}"]["minus_C0"] = cluster_bootstrap_diff(a, b, ga)
        a, ga = correct[("residual", m, "txt", "emotion")]
        b, _ = correct[("residual", "C0", "txt", "emotion")]
        residual_probes[m]["minus_C0"] = cluster_bootstrap_diff(a, b, ga)
    return code_probes, residual_probes


# ----------------------------------------------------------------------------- analysis 4: AMI

def ami_analysis(codes: dict, cache: dict, labels: dict) -> dict:
    st, sl = cache["scorer_train"], cache["selection"]
    clip_image = cache["clip_image"]
    if (clip_image[st] < 0).any():
        raise AssertionError("clip_image must label every scorer-train row")
    out = {}
    for m in MODELS:
        ic, tc = codes[m]
        pair = 0.5 * (ic.astype(np.float64) + tc.astype(np.float64))
        arg = np.where(pair.max(axis=1) > 0, pair.argmax(axis=1), -1)          # all-zero pair code: own group
        res = {"clip_image@scorer_train": float(adjusted_mutual_info_score(clip_image[st], arg[st]))}
        for part, rows in (("selection", sl), ("scorer_train", st)):
            for lab in LABELS:
                res[f"{lab}@{part}"] = float(adjusted_mutual_info_score(label_rows(labels, lab, rows), arg[rows]))
        res["factors_used@selection"] = int(len(np.unique(arg[sl][arg[sl] >= 0])))
        res["all_zero_share@selection"] = float(np.mean(arg[sl] < 0))
        out[m] = res
        log(f"AMI {m}: " + ", ".join(f"{k} {v:.4f}" for k, v in res.items() if isinstance(v, float)))
    out["reference_clip_image"] = {
        lab: float(adjusted_mutual_info_score(label_rows(labels, lab, st), clip_image[st])) for lab in LABELS}
    return out


# ----------------------------------------------------------------------------- analysis 3: stored results

def guard_power(res: dict) -> dict:
    """Normal approximation from the stored D_emotion CIs: SE = (lower-bound distance) / 1.96."""
    out = {"cells": {}}
    for c in ("A", "S", "AS"):
        b = res["d_emotion"][c]
        out["cells"][c] = {"point": b["point"], "ci95": b["ci95"],
                           "half_width": 0.5 * (b["ci95"][1] - b["ci95"][0]),
                           "lower_distance": b["point"] - b["ci95"][0]}
    lower = np.mean([v["lower_distance"] for v in out["cells"].values()])
    se = lower / HALF_WIDTH_Z
    threshold = GUARD + lower                     # the point estimate a cell needs for its lower bound > -1.0
    out.update({"mean_lower_distance": float(lower), "se": float(se), "pass_threshold_point": float(threshold),
                "pass_probability": {f"{mu:+.1f}": float(1 - norm.cdf((threshold - mu) / se))
                                     for mu in (0.5, 0.0, -0.5, -1.0)}})
    scaled = {}
    for n in (2048, 4096, 8192):                  # SE_n = SE * sqrt(2048 / n), the spec §7 scaling
        se_n = se * np.sqrt(2048 / n)
        thr = GUARD + HALF_WIDTH_Z * se_n
        scaled[str(n)] = {"half_width": float(HALF_WIDTH_Z * se_n),
                          "pass_probability_true_0": float(1 - norm.cdf(thr / se_n)),
                          "pass_probability_true_-0.5": float(1 - norm.cdf((thr + 0.5) / se_n))}
    out["by_emotion_episodes"] = scaled
    return out


def direction_table(res: dict) -> dict:
    """Per-direction emotion R@1 on the beta grid (naive) and the oracle at beta 0, from the stored JSON."""
    out = {"naive": {}, "oracle_0": {}, "S_minus_C0": {}, "S_oracle0_minus_C0_oracle0": {}}
    for m in MODELS:
        out["naive"][m] = {f"{b:g}": {d: res["r1"][key("naive", m, b)]["emotion"][d] for d in DIRECTIONS}
                           for b in BETAS}
        out["oracle_0"][m] = {d: res["r1"][key("oracle", m, 0.0)]["emotion"][d] for d in DIRECTIONS}
    out["S_minus_C0"] = {b: {d: res["vs_c0_beta_grid"]["S"][b]["emotion"][d] for d in (*DIRECTIONS, "mean")}
                         for b in res["vs_c0_beta_grid"]["S"]}
    out["S_oracle0_minus_C0_oracle0"] = {d: res["oracle_minus_c0_oracle"]["S"]["emotion"][d]
                                         for d in (*DIRECTIONS, "mean")}
    return out


def interpolate(res: dict, model: str, beta: float, scope: str) -> float:
    """Naive R@1 (mean of directions) of ``model`` at ``beta``, linear in beta between the stored grid points."""
    grid_b = np.array(BETAS)
    vals = np.array([res["r1"][key("naive", model, b)][scope]["mean"] for b in BETAS])
    return float(np.interp(beta, grid_b, vals))


def stored_ranks(npz, scorer: str, model: str, beta: float) -> dict:
    prefix = key(scorer, model, beta).replace("|", "__")
    return {label: {d: npz[f"{prefix}__{label}__{d}"] for d in DIRECTIONS} for label in LABELS}


def balance_matched(res: dict, npz, codes: dict, data, cache: dict, episodes: dict) -> dict:
    spread = res["term_spread_beta0.3"]
    ratio = {m: spread[m]["factor_to_clip"] for m in ("C0", "S")}
    beta_c0 = 0.3 * spread["C0"]["factor_spread"] / spread["S"]["factor_spread"]   # C0 leans on CLIP like S at 0.3
    beta_s = 0.3 * spread["S"]["factor_spread"] / spread["C0"]["factor_spread"]    # S leans on CLIP like C0 at 0.3
    out = {"spread_ratio_beta0.3": ratio, "clip_reliance_S_over_C0": ratio["C0"] / ratio["S"],
           "beta_C0_matched": beta_c0, "beta_S_matched": beta_s, "interpolated": {}, "rescored": {}}
    for scope in SCOPES:
        out["interpolated"][scope] = {
            "S@0.3 - C0@matched": interpolate(res, "S", 0.3, scope) - interpolate(res, "C0", beta_c0, scope),
            "S@matched - C0@0.3": interpolate(res, "S", beta_s, scope) - interpolate(res, "C0", 0.3, scope),
            "S@0.3 - C0@0.3": interpolate(res, "S", 0.3, scope) - interpolate(res, "C0", 0.3, scope)}
    # direct re-scoring of the same episodes at the matched beta (same naive weights; beta does not change them)
    sl = cache["selection"]
    img, txt = (sel.masked(x, sl) for x in (data.img_features, data.txt_features))
    ranks = {}
    for m, beta in (("C0", beta_c0), ("S", beta_s), ("C0", 0.3), ("S", 0.3)):
        ic, tc = (sel.masked(x, sl) for x in codes[m])
        weights = {label: label_episode_weights(ic, tc, episodes[label]) for label in LABELS}
        ranks[(m, beta)], _ = probe.fixed_weight_ranks(img, txt, ic, tc, episodes, weights, beta)
    for m in ("C0", "S"):                          # the re-scoring path reproduces the stored beta-0.3 ranks
        share = probe.identical_share(ranks[(m, 0.3)], stored_ranks(npz, "naive", m, 0.3))
        if share < grid.IDENTICAL_SHARE_MIN:
            raise AssertionError(f"{m}: re-scored beta-0.3 ranks differ from the stored ones ({share:.4f})")
    s03, c03 = stored_ranks(npz, "naive", "S", 0.3), stored_ranks(npz, "naive", "C0", 0.3)
    out["rescored"] = {"S@0.3 - C0@matched": probe.r1_diff(s03, ranks[("C0", beta_c0)]),
                       "S@matched - C0@0.3": probe.r1_diff(ranks[("S", beta_s)], c03),
                       "C0@matched": probe.r1_points(ranks[("C0", beta_c0)]),
                       "S@matched": probe.r1_points(ranks[("S", beta_s)])}
    d03 = res["vs_c0"]["S"]
    out["share_of_D_explained"] = {
        variant: 1 - out["rescored"][variant]["pooled"]["mean"]["point"] / d03["pooled"]["mean"]["point"]
        for variant in ("S@0.3 - C0@matched", "S@matched - C0@0.3")}
    return out


def oracle_columns(res: dict, npz) -> dict:
    """Same-beta oracle minus naive (the report's stored column mixes oracle beta 0 with naive beta 0.3)."""
    out = {"same_beta_0.3": {}, "same_beta_0": {}, "mixed_oracle0_minus_naive0.3": {}, "A_minus_C0_oracle": {}}
    for m in MODELS:
        for name, ob, nb in (("same_beta_0.3", 0.3, 0.3), ("same_beta_0", 0.0, 0.0),
                             ("mixed_oracle0_minus_naive0.3", 0.0, 0.3)):
            out[name][m] = probe.r1_diff(stored_ranks(npz, "oracle", m, ob), stored_ranks(npz, "naive", m, nb))
    for b in (0.3, 0.0):
        out["A_minus_C0_oracle"][f"{b:g}"] = probe.r1_diff(stored_ranks(npz, "oracle", "A", b),
                                                         stored_ranks(npz, "oracle", "C0", b))
    return out


# ----------------------------------------------------------------------------- ties at beta 0

def tie_analysis(npz, codes: dict, data, cache: dict, episodes: dict) -> dict:
    """Naive R@1 at beta 0 under the tie-aware convention (a tie is a miss) and with random tie-breaking."""
    sl = cache["selection"]
    img, txt = (sel.masked(x, sl) for x in (data.img_features, data.txt_features))
    out = {}
    for m in MODELS:
        ic, tc = (sel.masked(x, sl) for x in codes[m])
        stored = stored_ranks(npz, "naive", m, 0.0)
        tie_aware, random_break, tied_top = {}, {}, {}
        for label in LABELS:
            eps = episodes[label]
            w = label_episode_weights(ic, tc, eps)
            cands = np.concatenate([eps.positive[:, None], eps.distractors], axis=1)
            tie_aware[label], random_break[label] = {}, {}
            for d, qf, cf, qc, cc in (("i2t", img, txt, ic, tc), ("t2i", txt, img, tc, ic)):
                t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)       # noqa: E731
                s = conditional_score(t(qf[eps.anchor]), t(cf[cands]), t(qc[eps.anchor]), t(cc[cands]), w, 0.0)
                pos, others = s[:, :1], s[:, 1:]
                greater = (others > pos).sum(dim=1).numpy()
                equal = (others == pos).sum(dim=1).numpy()
                rank = 1.0 + greater + 0.5 * equal
                if not np.array_equal(rank, np.asarray(stored[label][d])):
                    raise AssertionError(f"{m} {label} {d}: recomputed beta-0 ranks differ from the stored ranks")
                tie_aware[label][d] = rank
                random_break[label][d] = np.where(greater == 0, 1.0 / (1.0 + equal), 0.0)
                tied_top[f"{label}|{d}"] = int(((greater == 0) & (equal > 0)).sum())
        expected = {}
        for scope in SCOPES:
            per = (random_break if scope == "pooled" else {scope: random_break[scope]})
            vals = {d: 100 * float(np.concatenate([per[lab][d] for lab in per]).mean()) for d in DIRECTIONS}
            expected[scope] = {**vals, "mean": 0.5 * (vals["i2t"] + vals["t2i"])}
        out[m] = {"tie_aware": probe.r1_points(tie_aware), "random_tie_break": expected,
                  "positive_tied_at_top": tied_top}
    return out


# ----------------------------------------------------------------------------- analysis 5: per target

def per_target(npz, episodes: dict) -> dict:
    out = {}
    for label in LABELS:
        targets = episodes[label].labels
        out[label] = {}
        for beta in (0.3, 0.0):
            for scorer in ("naive", "oracle"):
                if scorer == "oracle" and beta != 0.0:
                    continue
                s = stored_ranks(npz, scorer, "S", beta)[label]
                c = stored_ranks(npz, scorer, "C0", beta)[label]
                hit = lambda r: {d: probe.hits(r[d]) for d in DIRECTIONS}      # noqa: E731
                hs, hc = hit(s), hit(c)
                rows = {}
                for t in np.unique(targets):
                    mask = targets == t
                    diff = {d: hs[d][mask] - hc[d][mask] for d in DIRECTIONS}
                    rows[str(t)] = {
                        "n": int(mask.sum()),
                        "C0": 100 * float(0.5 * (hc["i2t"][mask] + hc["t2i"][mask]).mean()),
                        "S": 100 * float(0.5 * (hs["i2t"][mask] + hs["t2i"][mask]).mean()),
                        "diff": probe.pct_block(paired_bootstrap(0.5 * (diff["i2t"] + diff["t2i"]))),
                        **{f"diff_{d}": 100 * float(diff[d].mean()) for d in DIRECTIONS}}
                total = sum(v["n"] * v["diff"]["point"] for v in rows.values())
                for v in rows.values():
                    v["share_of_total"] = (v["n"] * v["diff"]["point"] / total) if total else float("nan")
                out[label][f"{scorer}@{beta:g}"] = rows
    return out


# ----------------------------------------------------------------------------- graph-term confound (F2)

def graph_confound(graph: csr_matrix, prep: dict) -> dict:
    groups = prep["local_groups"]
    n = len(groups)
    member = csr_matrix((np.ones(n), (groups, np.arange(n))))
    same = (member.T @ member).tocsr()
    same.setdiag(0)
    same.eliminate_zeros()
    same_upper = triu(same, k=1).tocsr()
    adjacency = (graph > 0).astype(np.int8).tocsr()
    covered = same_upper.multiply(adjacency).tocsr()
    covered.eliminate_zeros()
    upper = triu(graph, k=1).tocoo()
    upper.eliminate_zeros()
    edges = np.column_stack((upper.row, upper.col)).astype(np.int64)
    same_edges = int((groups[edges[:, 0]] == groups[edges[:, 1]]).sum())
    expand = GroupRows(groups)
    shares = {"edge_sampled": [], "painting_expanded": []}
    pairs = {"edge_sampled": [], "painting_expanded": []}
    config = grid.cell_config("C0", SEED)
    for epoch in range(1, SAMPLER_BATCHES + 1):                  # replay train_factors' batch sampler exactly
        rng = np.random.default_rng(config.seed + epoch)
        sampled = edges[rng.choice(len(edges), size=config.batch_size, replace=len(edges) < config.batch_size)]
        nodes = np.unique(sampled.reshape(-1))
        for name, ids in (("edge_sampled", nodes), ("painting_expanded", expand.expand(nodes))):
            sub = triu(graph[ids][:, ids], k=1).tocoo()
            sub.eliminate_zeros()
            same_pos = groups[ids][sub.row] == groups[ids][sub.col]
            shares[name].append(float(same_pos.mean()))
            pairs[name].append(int(len(same_pos)))
    return {"scorer_train_rows": n, "same_painting_pairs": int(same_upper.nnz),
            "same_painting_pairs_in_graph": int(covered.nnz), "graph_edges": int(len(edges)),
            "graph_edges_same_painting": same_edges,
            "sampler_batches": SAMPLER_BATCHES,
            "positive_pair_same_painting_share": {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                                      "max": float(np.max(v))} for k, v in shares.items()},
            "positive_pairs_per_batch": {k: float(np.mean(v)) for k, v in pairs.items()}}


# ----------------------------------------------------------------------------- training history (F5c)

def history_summary() -> dict:
    out = {}
    for c in ("S", "AS"):
        h = grid.history_record(c)["history"]
        cl = np.asarray(h["condition_loss"])
        out[c] = {"steps_logged": len(cl), "step_1": float(cl[0]), "mean_steps_50_500": float(cl[1:11].mean()),
                  "mean_last_10_logged": float(cl[-10:].mean()), "last_10_steps": h["step"][-10:],
                  "final_single_batch": float(cl[-1]), "tau_final": float(h["tau"][-1])}
    s = np.asarray(grid.history_record("S")["history"]["condition_loss"])
    a = np.asarray(grid.history_record("AS")["history"]["condition_loss"])
    out["AS_above_S_every_logged_step_after_first"] = bool((a[1:] > s[1:]).all())
    return out


# ----------------------------------------------------------------------------- run

def run() -> dict:
    started = perf_counter()
    torch.manual_seed(SEED)
    data, cache, prep, meta, graph, labels, allowed = load()
    res = json.loads(grid.SELECTION_JSON.read_text())
    npz = dict(np.load(grid.SELECTION_NPZ))
    episodes, _, eps_meta = grid.selection_episodes(data, cache)
    codes = {}
    for m in MODELS:
        ic, tc = grid.model_codes(m, data, cache)
        for side, arr in (("img", ic), ("txt", tc)):
            probe.assert_row_scope(f"{m} {side} codes", arr, allowed)
        codes[m] = (ic, tc)
    log(f"Codes loaded ({perf_counter() - started:.1f} s)")
    out = {"label": LABEL, "episodes": eps_meta,
           "settings": {"probe": PROBE_SETTINGS, "probe_standardize": "fit-row mean / std",
                        "cluster_bootstrap": {"n_boot": CLUSTER_BOOT, "seed": SEED, "unit": "selection painting"},
                        "episode_bootstrap": {"n_boot": 5000, "seed": SEED, "unit": "episode (paired)"},
                        "guard_power": "normal approximation; SE = mean over A, S, AS of (point - lower bound) / 1.96",
                        "interpolation": "linear in beta between the two neighbouring stored grid points",
                        "sampler_batches": SAMPLER_BATCHES}}
    t0 = perf_counter()
    out["graph_confound"] = graph_confound(graph, prep)
    log(f"Graph confound: {json.dumps(out['graph_confound'])} ({perf_counter() - t0:.1f} s)")
    out["guard_power"] = guard_power(res)
    out["emotion_by_direction"] = direction_table(res)
    t0 = perf_counter()
    out["balance_matched"] = balance_matched(res, npz, codes, data, cache, episodes)
    out["oracle_columns"] = oracle_columns(res, npz)
    out["ties_beta0"] = tie_analysis(npz, codes, data, cache, episodes)
    out["per_target"] = per_target(npz, episodes)
    out["history"] = history_summary()
    log(f"Stored-result analyses in {perf_counter() - t0:.1f} s")
    t0 = perf_counter()
    out["ami"] = ami_analysis(codes, cache, labels)
    log(f"AMI in {perf_counter() - t0:.1f} s")
    out["code_probes"], out["residual_probes"] = probe_analyses(codes, data, cache, labels)
    out["seconds"] = perf_counter() - started
    POSTHOC_JSON.write_text(json.dumps(grid._jsonable(out), indent=2))
    log(f"Post-hoc diagnostics in {out['seconds']:.1f} s -> {POSTHOC_JSON}")
    return out


# ----------------------------------------------------------------------------- tables

def _ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['point']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def tables(path: Path = POSTHOC_JSON) -> None:
    r = json.loads(path.read_text())
    out = [f"All numbers: {r['label']}.\n"]

    g = r["graph_confound"]
    sh = g["positive_pair_same_painting_share"]
    out.append("### Graph-term confound (scorer-train graph; training batches replayed)\n")
    out.append(f"- same-painting row pairs: {g['same_painting_pairs']:,}; of them graph edges: "
               f"{g['same_painting_pairs_in_graph']:,}; graph edges {g['graph_edges']:,}, same-painting "
               f"{g['graph_edges_same_painting']:,} ({100 * g['graph_edges_same_painting'] / g['graph_edges']:.1f}%)")
    out.append(f"- same-painting share of the graph term's positive pairs over {g['sampler_batches']} batches: "
               f"edge-sampled {100 * sh['edge_sampled']['mean']:.1f}% (range {100 * sh['edge_sampled']['min']:.1f}"
               f"-{100 * sh['edge_sampled']['max']:.1f}), painting-expanded {100 * sh['painting_expanded']['mean']:.1f}% "
               f"(range {100 * sh['painting_expanded']['min']:.1f}-{100 * sh['painting_expanded']['max']:.1f}); "
               f"positive pairs per batch {g['positive_pairs_per_batch']['edge_sampled']:.0f} vs "
               f"{g['positive_pairs_per_batch']['painting_expanded']:.0f}\n")

    gp = r["guard_power"]
    out.append("### Emotion guard power (normal approximation from the stored CIs)\n")
    out.append("| Cell | D_emotion | half-width | point - lower bound |")
    out.append("|---|---:|---:|---:|")
    for c, v in gp["cells"].items():
        out.append(f"| {c} | {v['point']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | {v['half_width']:.2f} | "
                   f"{v['lower_distance']:.2f} |")
    out.append(f"\nSE {gp['se']:.3f}; a cell passes when its point estimate exceeds {gp['pass_threshold_point']:+.3f}. "
               "P(pass | true emotion effect): " + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in
                                                            gp["pass_probability"].items()))
    out.append("By emotion episodes per label: " + "; ".join(
        f"n {n}: half-width {v['half_width']:.2f}, P(pass | 0) {100 * v['pass_probability_true_0']:.1f}%, "
        f"P(pass | -0.5) {100 * v['pass_probability_true_-0.5']:.1f}%" for n, v in gp["by_emotion_episodes"].items())
        + "\n")

    e = r["emotion_by_direction"]
    out.append("### Emotion naive R@1 (%) per direction on the beta grid\n")
    out.append("| Model | " + " | ".join(f"beta {b} i2t / t2i" for b in e["naive"]["C0"]) + " | oracle 0 i2t / t2i |")
    out.append("|---|" + "---|" * (len(BETAS) + 1))
    for m in MODELS:
        out.append(f"| {m} | " + " | ".join(f"{v['i2t']:.2f} / {v['t2i']:.2f}" for v in e["naive"][m].values())
                   + f" | {e['oracle_0'][m]['i2t']:.2f} / {e['oracle_0'][m]['t2i']:.2f} |")
    out.append("\n| S - C0, emotion | i2t | t2i | mean |")
    out.append("|---|---:|---:|---:|")
    for b, v in e["S_minus_C0"].items():
        out.append(f"| naive beta {b} | {_ci(v['i2t'])} | {_ci(v['t2i'])} | {_ci(v['mean'])} |")
    v = e["S_oracle0_minus_C0_oracle0"]
    out.append(f"| oracle beta 0 | {_ci(v['i2t'])} | {_ci(v['t2i'])} | {_ci(v['mean'])} |\n")

    bm = r["balance_matched"]
    out.append("### Balance-matched S vs C0\n")
    out.append(f"Spread ratio at beta 0.3: C0 {bm['spread_ratio_beta0.3']['C0']:.3f}, S {bm['spread_ratio_beta0.3']['S']:.3f} "
               f"(C0 / S = {bm['clip_reliance_S_over_C0']:.3f}); matched beta: C0 {bm['beta_C0_matched']:.4f}, "
               f"S {bm['beta_S_matched']:.4f}\n")
    out.append("| Comparison | pooled | emotion | art style |")
    out.append("|---|---:|---:|---:|")
    for variant in ("S@0.3 - C0@0.3", "S@0.3 - C0@matched", "S@matched - C0@0.3"):
        i = bm["interpolated"]
        out.append(f"| interpolated: {variant} | {i['pooled'][variant]:+.2f} | {i['emotion'][variant]:+.2f} | "
                   f"{i['art_style'][variant]:+.2f} |")
    for variant in ("S@0.3 - C0@matched", "S@matched - C0@0.3"):
        v = bm["rescored"][variant]
        out.append(f"| re-scored: {variant} | {_ci(v['pooled']['mean'])} | {_ci(v['emotion']['mean'])} | "
                   f"{_ci(v['art_style']['mean'])} |")
    out.append("\nShare of D (+1.44) explained by the scale difference: " + ", ".join(
        f"{k}: {100 * v:.1f}%" for k, v in bm["share_of_D_explained"].items()) + "\n")

    oc = r["oracle_columns"]
    out.append("### Oracle minus own naive (paired, R@1 points)\n")
    out.append("| Model | mixed: oracle 0 - naive 0.3 (pooled) | same beta 0.3 (pooled) | same beta 0 (pooled) | "
               "mixed (emotion) | same 0.3 (emotion) | same 0 (emotion) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        out.append(f"| {m} | " + " | ".join(_ci(oc[k][m][s]["mean"]) for s in ("pooled", "emotion")
                                            for k in ("mixed_oracle0_minus_naive0.3", "same_beta_0.3", "same_beta_0"))
                   + " |")
    out.append("\nA - C0 oracle: " + "; ".join(f"beta {b}: pooled {_ci(v['pooled']['mean'])}, emotion "
                                             f"{_ci(v['emotion']['mean'])}, style {_ci(v['art_style']['mean'])}"
                                             for b, v in oc["A_minus_C0_oracle"].items()) + "\n")

    out.append("### Ties at beta 0: naive R@1 (%), mean of directions\n")
    out.append("| Model | tie-aware pooled | random tie-break pooled | tie-aware emotion | random emotion | "
               "tie-aware style | random style | episodes with the positive tied at the top (emo i2t/t2i, style i2t/t2i) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for m in MODELS:
        t = r["ties_beta0"][m]
        tt = t["positive_tied_at_top"]
        out.append(f"| {m} | " + " | ".join(f"{t['tie_aware'][s]['mean']:.2f} | {t['random_tie_break'][s]['mean']:.2f}"
                                            for s in SCOPES)
                   + f" | {tt['emotion|i2t']} / {tt['emotion|t2i']}, {tt['art_style|i2t']} / {tt['art_style|t2i']} |")
    out.append("")

    out.append("### Per-target S - C0 (mean of directions; paired bootstrap per target)\n")
    for label in LABELS:
        pt = r["per_target"][label]
        cols = list(pt)
        out.append(f"**{label}**\n")
        out.append("| Target | n | " + " | ".join(f"{c}: C0 / S / S - C0 [CI] / i2t, t2i" for c in cols) + " |")
        out.append("|---|---:|" + "---|" * len(cols))
        order = sorted(pt[cols[0]], key=lambda t: pt["naive@0"][t]["diff"]["point"])
        for t in order:
            cells = [f"{pt[c][t]['C0']:.1f} / {pt[c][t]['S']:.1f} / {_ci(pt[c][t]['diff'])} / "
                     f"{pt[c][t]['diff_i2t']:+.1f}, {pt[c][t]['diff_t2i']:+.1f}" for c in cols]
            out.append(f"| {t} | {pt[cols[0]][t]['n']} | " + " | ".join(cells) + " |")
        out.append("")

    a = r["ami"]
    out.append("### AMI of the argmax pair-code factor\n")
    out.append("| Model | CLIP image clusters (scorer-train) | art style (selection) | emotion (selection) | "
               "art style (scorer-train) | emotion (scorer-train) | factors used | all-zero rows |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        v = a[m]
        out.append(f"| {m} | {v['clip_image@scorer_train']:.4f} | {v['art_style@selection']:.4f} | "
                   f"{v['emotion@selection']:.4f} | {v['art_style@scorer_train']:.4f} | {v['emotion@scorer_train']:.4f} | "
                   f"{v['factors_used@selection']} | {100 * v['all_zero_share@selection']:.2f}% |")
    ref = a["reference_clip_image"]
    out.append(f"\nReference: CLIP image clusters vs art style {ref['art_style']:.4f}, vs emotion {ref['emotion']:.4f} "
               "(scorer-train).\n")

    out.append("### Linear probes on 32-d codes: top-1 accuracy (%) on selection rows\n")
    tasks = [f"{m}->{lab}" for m, lab in PROBE_TASKS]
    out.append("| Model | " + " | ".join(tasks) + " |")
    out.append("|---|" + "---:|" * len(tasks))
    cp = r["code_probes"]
    for m in MODELS:
        out.append(f"| {m} | " + " | ".join(f"{cp[m][t]['accuracy']:.2f}" + (
            f" ({_ci(cp[m][t]['minus_C0'])})" if "minus_C0" in cp[m][t] else "") for t in tasks) + " |")
    out.append("| majority | " + " | ".join(f"{cp['C0'][t]['majority']:.2f}" for t in tasks) + " |")
    conv = all(cp[m][t]["converged"] for m in MODELS for t in tasks)
    out.append(f"\nAll probes converged: {conv}.\n")

    rp = r["residual_probes"]
    out.append("### Within-painting caption-residual emotion probe (%)\n")
    out.append("| Code | accuracy | minus C0 | majority | within-painting variance share | fit / eval rows | converged |")
    out.append("|---|---:|---:|---:|---:|---|---|")
    for m in (*MODELS, "clip512"):
        v = rp[m]
        diff = _ci(v["minus_C0"]) if "minus_C0" in v else ""
        out.append(f"| {m} | {v['accuracy']:.2f} | {diff} | {v['majority']:.2f} | "
                   f"{v['within_painting_variance_share']:.3f} | {v['n_fit']:,} / {v['n_eval']:,} | {v['converged']} |")
    out.append("")

    h = r["history"]
    out.append("### Condition loss history\n")
    for c in ("S", "AS"):
        v = h[c]
        out.append(f"- {c}: step 1 {v['step_1']:.3f}; mean of logged steps 50-500 {v['mean_steps_50_500']:.3f}; mean "
                   f"of the last 10 logged steps ({v['last_10_steps'][0]}-{v['last_10_steps'][-1]}) "
                   f"{v['mean_last_10_logged']:.3f}; final single batch {v['final_single_batch']:.3f}; final tau "
                   f"{v['tau_final']:.4f}")
    out.append(f"- AS above S at every logged step after the first: {h['AS_above_S_every_logged_step_after_first']}")
    out.append(f"\nRun time {r['seconds']:.0f} s.")
    print("\n".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
        tables()
    elif args.tables:
        tables()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
