"""Probe recipe, part 2 (ticket 27): seed-42 episodes, heads, and the frozen-pick scoring on a CUB or SemArt corpus.
Exploration: no verdict, no bar, no unit tests. Library for `run_r9_probe.py` (stages episodes, heads, score).

Plan (progress note of part 1). No `build_bundle_r6`, `fit_heads_r9`, `RowContext` (they are wired to ArtELingo's 308,723
rows and to pinned files); the recipe's own functions are called on a light bundle:

  episodes  seed 42, `n` per pair, on the selection rows and on the scoring rows. A copy of `r9_episodes.build_seed`'s loop
            (its aspects, pairs and row count are ArtELingo's) with the pairs and values of ticket 26's plan.json: a
            plan value that is not eligible on a pool (30 distinct units with all three labels) is dropped and recorded.
            The third aspect of a pair is the remaining one, as in R.PAIRS. If a pair cannot reach 4,096 episodes on a
            pool the builder raises; n is then halved (2,048 ... 256) for every pair and pool, and recorded.
  heads     the four heads (partition L, affect-km, image, caption) on scorer-train rows (`fit_one_head_r6`), posteriors
            on selection rows and scoring rows (`H.predict`, one call per row set), saved compact.
  score     on the selection episodes: tau (reader margins), RCA lambda, B and B0 picks (cross-fitted), AFF's 224-cell
            picks (`family_picks`); then `r9_score.score_seed` on the scoring episodes with those frozen picks
            (include_pm False, secondary False). A3 term from the probe's A3 checkpoint (`--no-a3` smoke: a fixed random
            projection stands in for A3's codes; limit folders only, never the real output).
"""
import json
import pickle
import time
from types import SimpleNamespace

import r9_probe_recipe as P
from r9_probe_recipe import log, write_json

R = P.R
import numpy as np  # noqa: E402

import r9_bundle as RB  # noqa: E402
import r9_context as X  # noqa: E402
import r9_episodes as E  # noqa: E402
import r9_heads as H  # noqa: E402
import r9_picks as PK  # noqa: E402
import r9_pins as PINS  # noqa: E402
import r9_recipe_cells as RCE  # noqa: E402
import r9_recipe_reader as RR  # noqa: E402
import r9_score as S  # noqa: E402
from src.eval.aspect_episodes import (NUM_PAIRS, PaintingValueIndex, concat_episodes, eligible_values,  # noqa: E402
                                      episodes_sha256, validate_aspect_episodes, AspectEpisodes)
from src.eval.aspect_metrics import cluster_bootstrap, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import _require_condition_free, centered_term, uniform_probe_scores  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores, crossfit_lambda  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free  # noqa: E402
from src.eval.pair_metric_baselines import fit_pca_basis, rca_term  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

C, rbe = R.C, R.rbe
SEED = P.SEED
N_TRY = (4096, 2048, 1024, 512, 256)
# Episode-builder settings per corpus (choices the plan did not fix). ArtELingo's: third aspect on, 30-unit floor.
# CUB: only 22% of images carry all three attributes and the builder needs more than 4 eligible values per aspect, so the
# pools with the third aspect hold 3 to 4 eligible values (selection and scoring alike): third aspect off (negatives and
# examples then lack only the pair's two aspects) and a floor of 10 distinct images per value.
CONFIG = {"semart": {"use_third": True, "min_units": 30}, "cub": {"use_third": False, "min_units": 10}}
POOLS = ("selection", "scoring")
HEAD_NAMES = ("affect", "affect_km", "image", "caption")     # partition L, affect-km, image, caption
FAKE_A3_DIM = 32


def pool_rows(rows, pool):
    return np.sort(rows.selection if pool == "selection" else rows.scoring_rows)


# ---------------------------------------------------------------- episodes

def pair_plan(rows):
    """[(a, b, third)], names, {aspect: plan value codes}."""
    plan = rows.plan
    aspects = list(plan["aspects"])
    pairs = [tuple(p) for p in plan["pairs"]]
    assert set(aspects) == set(rows.labels) and len(aspects) == 3
    out = []
    for a, b in pairs:
        (third,) = [x for x in aspects if x not in (a, b)]
        out.append((a, b, third))
    values = {a: sorted(int(v["code"]) for v in plan["aspects"][a]["values"]) for a in aspects}
    return out, [f"{a}__{b}" for a, b, _ in out], values


def build_pool(rows, pool, n):
    """The three pairs' episodes (n each) on the pool's rows, validated, member rows inside the pool.
    -> (list of AspectEpisodes, record per pair). Raises RuntimeError when a pair cannot reach n."""
    pairs, names, plan_values = pair_plan(rows)
    pr = pool_rows(rows, pool)
    in_rows = np.zeros(rows.n_rows, dtype=bool)
    in_rows[pr] = True
    labels = {a: np.asarray(v, dtype=np.int64) for a, v in rows.labels.items()}
    index = PaintingValueIndex(labels, rows.groups)
    eps, recs = [], {}
    for (a, b, third), name in zip(pairs, names):
        cfg = CONFIG[rows.name]
        known = (labels[a][pr] >= 0) & (labels[b][pr] >= 0)
        if cfg["use_third"]:
            known &= labels[third][pr] >= 0
        pool_pair = pr[known]
        ok = {x: eligible_values(labels[x], rows.groups, pool_pair, cfg["min_units"]) for x in (a, b)}
        kept = {x: sorted(set(ok[x]) & set(plan_values[x])) for x in (a, b)}
        dropped = {x: sorted(set(plan_values[x]) - set(ok[x])) for x in (a, b)}
        rec = {"pool_rows": int(len(pool_pair)), "pool_units": int(len(np.unique(rows.groups[pool_pair]))),
               "values_kept": kept, "values_dropped": dropped, "third": third if cfg["use_third"] else None,
               "min_units": cfg["min_units"]}
        if len(kept[a]) <= NUM_PAIRS or len(kept[b]) <= NUM_PAIRS:
            raise RuntimeError(f"{pool} {name}: {len(kept[a])} and {len(kept[b])} plan values are eligible, "
                               f"need more than {NUM_PAIRS}")
        use3 = third if cfg["use_third"] else None
        ep = E.build_aspect_episodes_r6(labels, rows.groups, pr, a, b, n, SEED, third=use3,
                                        min_paintings=cfg["min_units"], index=index, values=(kept[a], kept[b]))
        assert len(ep.anchor) == n
        validate_aspect_episodes(ep, labels, rows.groups, index, third=use3)
        every = E.members(ep)
        assert in_rows[every].all(), f"{pool} {name}: member rows outside the pool"
        rec.update(n_episodes=int(n), distinct_anchor_rows=int(len(np.unique(ep.anchor))),
                   distinct_anchor_units=int(len(np.unique(rows.groups[ep.anchor]))), sha256=episodes_sha256(ep))
        eps.append(ep)
        recs[name] = rec
    return eps, recs


def stage_episodes(rows, out):
    path = {p: out / f"episodes_{p}.npz" for p in POOLS}
    assert not any(v.exists() for v in path.values()), "episodes files exist"
    _, names, _ = pair_plan(rows)
    tried = []
    for n in N_TRY:
        try:
            built = {p: build_pool(rows, p, n) for p in POOLS}
            break
        except RuntimeError as e:
            tried.append({"n_per_pair": n, "failed": str(e)})
            log(f"n={n} per pair cannot be reached: {e}")
    else:
        return P.fail(out, "episodes", f"no n of {N_TRY} per pair is reachable: {tried}")
    for p in POOLS:
        eps, _ = built[p]
        pooled = concat_episodes(eps)
        np.savez(path[p], n_per_pair=np.int64(n), pair_names=np.array(names),
                 pair_index=np.repeat(np.arange(len(eps)), n),
                 **{f: np.ascontiguousarray(getattr(pooled, f), dtype=np.int64) for f in E.FIELDS},
                 **{f"sha__{nm}": np.array(built[p][1][nm]["sha256"]) for nm in names})
    write_json(out / "episodes_record.json", {
        "seed": SEED, "n_per_pair": int(n), "planned_n_per_pair": N_TRY[0], "reached_planned": n == N_TRY[0],
        "fallbacks_tried": tried, "pairs": names, "config": CONFIG[rows.name], "pools": {p: built[p][1] for p in POOLS}, "counts": rows.counts})
    log(f"episodes: {n} per pair on {POOLS}")
    return 0


def load_episodes(out, pool, names):
    with np.load(out / f"episodes_{pool}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    assert [str(x) for x in d["pair_names"]] == names
    n = int(d["n_per_pair"])
    eps = []
    for i, nm in enumerate(names):
        a, b = nm.split("__")
        ep = AspectEpisodes(a, b, *(np.ascontiguousarray(d[f][i * n:(i + 1) * n]) for f in E.FIELDS))
        assert episodes_sha256(ep) == str(d[f"sha__{nm}"]), f"{pool} {nm}: hash differs"
        eps.append(ep)
    assert np.array_equal(d["pair_index"], np.repeat(np.arange(len(names)), n))
    return eps, n


# ---------------------------------------------------------------- heads

def stage_heads(rows, out):
    path = out / "probe_post.npz"
    assert not path.exists(), "probe_post.npz exists"
    H.require_threads()
    st = rows.scorer_train
    aff = P.load_affect(out, len(st))
    with np.load(out / "partitions.npz") as z:
        labs = {"affect": aff["partition_L"], "affect_km": aff["affect_km"],
                "image": z["image"].astype(np.int64), "caption": z["caption"].astype(np.int64)}
    n_rows = min(H.HEAD_ROWS, int(0.8 * len(st)))
    heads, prov = {}, {}
    for g in HEAD_NAMES:
        t = time.time()
        lab = RR.rto_labels(labs[g], st, rows.n_rows)
        clfs, prov[g] = H.fit_one_head_r6(rows.data, lab, st, n_rows)
        assert prov[g]["n_classes"] == int(labs[g].max()) + 1, f"{g}: the draw misses a class"
        heads[g] = clfs
        log(f"head {g}: {prov[g]['n_classes']} classes, held-out acc {prov[g]['heldout_accuracy']} [{time.time() - t:.0f}s]")
    arrays = {}
    for p in POOLS:
        pr = pool_rows(rows, p)
        post = H.predict(heads, rows.data, pr)
        arrays[f"rows__{p}"] = pr
        for g in HEAD_NAMES:
            for m in H.MODALITIES:
                arrays[f"{p}__{g}__{m}"] = post[g][m][pr]
    np.savez(path, **arrays)
    write_json(out / "heads_record.json", {"head_rows": n_rows, "heads": prov, "coef_sha256": H.coef_sha256(heads),
                                           "counts": rows.counts})
    return 0


def load_post(out, rows, pool):
    """{head: {"img", "txt": (n_rows_all, n_cls) float32}}, NaN outside the pool."""
    with np.load(out / "probe_post.npz") as z:
        pr = z[f"rows__{pool}"]
        assert np.array_equal(pr, pool_rows(rows, pool)), "probe_post.npz rows differ from the split"
        post = {}
        for g in HEAD_NAMES:
            post[g] = {}
            for m in H.MODALITIES:
                a = z[f"{pool}__{g}__{m}"]
                full = np.full((rows.n_rows, a.shape[1]), np.nan, dtype=np.float32)
                full[pr] = a
                post[g][m] = full
    return post


# ---------------------------------------------------------------- score

def a3_codes(rows, out, pool, fake):
    """(img codes, txt codes) as full-size arrays, NaN outside the pool."""
    pr = pool_rows(rows, pool)
    if fake:
        proj = np.random.default_rng(0).standard_normal((rows.data.img_features.shape[1], FAKE_A3_DIM)).astype(np.float32)
        ic, tc = rows.data.img_features[pr] @ proj, rows.data.txt_features[pr] @ proj
    else:
        model, _ = load_factor_checkpoint(out / "A3_seed42.pt", device="cpu")
        ic, tc = encode_rows(model, rows.data.img_features, rows.data.txt_features, rows=pr, device="cpu")
    full = []
    for codes in (ic, tc):
        f = np.full((rows.n_rows, codes.shape[1]), np.nan, dtype=np.float32)
        f[pr] = codes
        assert np.isfinite(f[pr]).all()
        full.append(f)
    return full


def pool_bundle(rows, out, pool, eps, basis, fake_a3, aspects_post):
    """The light bundle of one pool (the fields score_seed and the picks read) and the pool's raw pieces."""
    t0 = time.time()
    pooled = concat_episodes(eps)
    n = len(pooled.anchor)
    inp = EvalInputs(rows.data.img_features, rows.data.txt_features)
    cos = cosine_scores(inp, pooled)
    ic, tc = a3_codes(rows, out, pool, fake_a3)
    inp.img_codes, inp.txt_codes = ic, tc
    t_n1u = centered_term(inp, pooled, uniform=True)
    del ic, tc
    inp.img_codes = inp.txt_codes = None
    post = aspects_post
    pr = pool_rows(rows, pool)
    for g in HEAD_NAMES:
        for m in H.MODALITIES:
            assert np.isfinite(post[g][m][pr]).all(), f"{g}/{m}: non-finite posteriors on the pool"
    t6u_B = uniform_probe_scores({part: post[RB.E2_FROM[part]] for part in RB.E2_PARTS}, pooled, RB.E2_PARTS)
    a0 = {h: post[h] for h in RB.A0}
    t6u_B0 = uniform_probe_scores(a0, pooled, RB.A0)
    stack = C.grouping_stack(a0, pooled, RB.A0)
    F, checks = rbe.seed42_features(SimpleNamespace(ctx=SimpleNamespace(pooled=pooled), post=a0), RB.A0)
    rca = rca_term(inp, pooled, basis)
    b = SimpleNamespace(n=n, cl=rows.groups[pooled.anchor], parity=np.arange(n) % 2,
                        pair_index=np.repeat(np.arange(len(eps)), len(eps[0].anchor)), anchor=pooled.anchor,
                        cos=cos, t_n1u=t_n1u, t6u_B=t6u_B, t6u_B0=t6u_B0, stack=stack, F=F, pm_terms={"rca": rca},
                        mode="selection", seed=SEED, smoke=False, pooled=pooled, checks=dict(checks))
    for k in ("cos", "t_n1u", "t6u_B", "t6u_B0"):
        RB.check_scores(getattr(b, k), n, k, np.float32)
        _require_condition_free(getattr(b, k), k)
    RB.check_scores(rca, n, "rca")
    RB.check_stack_f(b)
    log(f"{pool} bundle: {n} episodes [{time.time() - t0:.0f}s]")
    return b


def mean_r1(scores):
    return 100 * float(np.mean(per_anchor(scores)["r1"]))


def summarize(res, bundle, names):
    """R@1 (x100), condition gain (x100) overall and by pair, for every scorer in res."""
    pi = np.asarray(bundle.pair_index)
    out = {}
    for k in S.PRIMARY_SCORERS:
        r1, gain = np.asarray(res[k]["r1"]), np.asarray(res[k]["gain"])
        out[k] = {"r1": 100 * float(r1.mean()), "gain": 100 * float(gain.mean()),
                  "r1_by_pair": {nm: 100 * float(r1[pi == i].mean()) for i, nm in enumerate(names)}}
    return out


def diffs(res, bundle):
    """Cluster bootstrap (anchor unit) of paired R@1 differences, x100; 2000 resamples (a direction only)."""
    cl = np.asarray(bundle.cl)
    out = {}
    for x, y in (("aff_fused", "B"), ("aff_fused", "rca"), ("aff_fused", "cosine"), ("B", "cosine"), ("rca", "cosine"),
                 ("aff_cf", "B")):
        d = np.asarray(res[x]["r1"]) - np.asarray(res[y]["r1"])
        r = cluster_bootstrap(d, cl, n_boot=2000)
        out[f"{x}-{y}"] = {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]], "n_clusters": r["n_clusters"]}
    return out


def stage_score(rows, out, no_a3=False):
    dest = out / f"probe_{rows.name}.json"
    assert not dest.exists(), f"{dest} exists"
    if no_a3:
        assert rows.limit, "--no-a3 is a smoke switch: limit folders only, never the real output"
    else:
        assert (out / "A3_seed42.pt").exists(), "A3_seed42.pt is missing (stage a3, GPU)"
    H.require_threads()
    st = rows.scorer_train
    _, names, _ = pair_plan(rows)
    eps = {}
    for p in POOLS:
        eps[p], n = load_episodes(out, p, names)
    with open(out / "rb_reader_A0.pkl", "rb") as f:
        readers = pickle.load(f)
    RB.check_readers(readers)
    fit = np.random.default_rng(RB.FIT_SEED).choice(st, min(RB.FIT_ROWS, len(st)), replace=False)
    fi, ft = (np.asarray(x[fit], np.float32) for x in (rows.data.img_features, rows.data.txt_features))
    fi, ft = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (fi, ft))
    basis = fit_pca_basis(fi, ft)
    bundles = {p: pool_bundle(rows, out, p, eps[p], basis, no_a3, load_post(out, rows, p)) for p in POOLS}

    # ---- picks on the selection episodes
    sel = bundles["selection"]
    parity = sel.parity
    t0 = time.time()
    base_B, pick_B = crossfit_condition_free(sel.cos, sel.t_n1u, sel.t6u_B, parity)
    base_B0, pick_B0 = crossfit_condition_free(sel.cos, sel.t_n1u, sel.t6u_B0, parity)
    rca_scores, lam = crossfit_lambda(sel.cos, sel.pm_terms["rca"], parity)
    picks = {"B": PK.as_nested_pick(pick_B, "B"), "B0": PK.as_nested_pick(pick_B0, "B0")}
    lambdas = {"rca": PK.as_lambda_pick(lam, "rca")}
    log(f"B {picks['B']} B0 {picks['B0']} rca lambda {lambdas['rca']} [{time.time() - t0:.0f}s]")
    margins, _ = RR.reader_margins(readers, sel.F)
    taus = [float(t) for t in RR.thresholds(margins, n_expected=2 * sel.n)[0]]
    rd = R.RF.reader(sel, readers)
    RCE.check_taus(rd["m"], taus, sel.n)
    g_aff = R.RF.gates_aff(rd["m"], rd["pick"], taus)
    fam = RCE.family_picks(base_B, rd["T"], g_aff, parity)
    cells = {"aff": {"fused": tuple(int(fam["fpick"][h]) for h in (0, 1)),
                     "cf": tuple(int(fam["cpick"][h]) for h in (0, 1))}}
    log(f"AFF cells {cells['aff']} [{time.time() - t0:.0f}s]")
    gates_sel = {c: np.stack([g_aff[t][c] for t in range(S.N_TAU)]) for c in C.CONDITIONS}
    share_sel = PINS.gate_share(gates_sel, parity, cells["aff"]["fused"])
    selection_r1 = {"cosine": mean_r1(sel.cos), "rca": mean_r1(rca_scores), "B": mean_r1(base_B),
                    "B0": mean_r1(base_B0), "aff_fused": 100 * float(np.mean(fam["fused"]["r1"]))}

    # ---- the frozen scoring
    sc = bundles["scoring"]
    patched = {"key": ["selection", SEED], "before": X.ADMITTED.get(("selection", SEED)), "after": int(n)}
    X.ADMITTED[("selection", SEED)] = int(n)       # probe process only; recorded in the output
    res = S.score_seed(sc, picks, lambdas, readers, include_pm=False, secondary=False, cells=cells, taus=taus)
    metrics = summarize(res, sc, names)
    gates_sc = res["gates"]["aff"]
    share_sc = PINS.gate_share(gates_sc, sc.parity, cells["aff"]["fused"])
    rec = {
        "dataset": rows.name, "seed": SEED, "backbone": "clip_b32", "limit": rows.limit,
        "fake_a3": bool(no_a3), "n_per_pair": int(n), "episode_counts_per_pair": {
            p: {nm: bundles[p].n // len(names) for nm in names} for p in POOLS},
        "scoring": metrics,                      # COS, RCA, B, B0, AFF (fused) and AFF-CF: R@1 x100, gain x100
        "paired_differences_r1_x100": diffs(res, sc),
        "labels": {"COS": "cosine", "RCA": "rca", "B": "B", "B0": "B0", "AFF": "aff_fused", "AFF_CF": "aff_cf"},
        "gate_share_scoring": share_sc, "gate_share_selection": share_sel,
        "selection_crossfit_r1": selection_r1,
        "picks": {"B": picks["B"], "B0": picks["B0"], "rca_lambda": lambdas["rca"], "aff_cells": cells["aff"],
                  "sigma_star": [float(fam["sigma"][h]) for h in (0, 1)], "taus": taus,
                  "aff_cells_text": {str(c): S.F2.cell_values(c) for h in cells["aff"].values() for c in h}},
        "episodes": json.loads((out / "episodes_record.json").read_text()),
        "admitted_patch": patched,
        "heads": json.loads((out / "heads_record.json").read_text())["heads"],
        "counts": rows.counts}
    write_json(dest, rec)
    write_note(rows, out, rec)
    log(f"wrote {dest}")
    return 0


def write_note(rows, out, rec):
    m = rec["scoring"]
    row = lambda k: f"{m[k]['r1']:.2f} (gain {m[k]['gain']:.2f})"    # noqa: E731
    d = rec["paired_differences_r1_x100"]
    ci = lambda k: f"{d[k]['point']:+.2f} [{d[k]['ci95'][0]:+.2f}, {d[k]['ci95'][1]:+.2f}]"    # noqa: E731
    gs = rec["gate_share_scoring"]["gate_share"]
    ep = rec["episodes"]
    dropped = {p: {nm: {a: v for a, v in r["values_dropped"].items() if v} for nm, r in ep["pools"][p].items()
                   if any(r["values_dropped"].values())} for p in POOLS}
    dropped = {p: v for p, v in dropped.items() if v}
    lines = [f"# Probe note: {rows.name}, CLIP, seed 42" + (" (SMOKE: fake A3, limit run)" if rec["fake_a3"] else ""), "",
             "Exploration for stage 3's decide chat. A direction only (SC4): no bar, no verdict.", "",
             f"Scoring split: {rec['counts']['scoring_units']} units; {rec['n_per_pair']} episodes per pair on the "
             f"selection and on the scoring rows (planned {ep['planned_n_per_pair']}). Picks come from the selection "
             f"episodes only.", "",
             "R@1 x100 on the scoring episodes (condition gain in brackets):", "",
             f"- COS {row('cosine')}", f"- RCA {row('rca')}", f"- B {row('B')}", f"- B0 {row('B0')}",
             f"- AFF {row('aff_fused')}", f"- AFF counterpart {row('aff_cf')}", "",
             "Paired differences in R@1 x100 (95% interval, resampling anchor units):", "",
             f"- AFF - B {ci('aff_fused-B')}", f"- AFF - RCA {ci('aff_fused-rca')}", f"- AFF - COS {ci('aff_fused-cosine')}",
             f"- B - COS {ci('B-cosine')}", f"- RCA - COS {ci('rca-cosine')}", "",
             f"AFF gate share on the scoring episodes: {gs:.3f} (selection: {rec['gate_share_selection']['gate_share']:.3f}). "
             f"AFF cells {rec['picks']['aff_cells']}.", "",
             "Distinct anchor units per pair (scoring): " + ", ".join(
                 f"{nm} {r['distinct_anchor_units']}" for nm, r in ep["pools"]["scoring"].items()) + ".",
             f"Plan values dropped as not eligible on a pool (value codes): {dropped if dropped else 'none'}.",
             f"Episode builder settings: {ep['config']} (third aspect in the pool, floor of distinct units per value); "
             f"selection share of the fit rows {P.SELECTION_FRACTION[rows.name]}.", ""]
    (out / "probe_note.md").write_text("\n".join(lines))
