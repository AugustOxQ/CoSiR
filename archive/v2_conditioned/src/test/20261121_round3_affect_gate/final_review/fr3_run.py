"""Final review: from my own per-seed inputs (fr3_build), run my own families for AFF, R1 and the random-share
control, the frozen-cell line, the GO checks, the secondary check and every descriptive number the report uses; then
compare with the implementation's stored files. Writes out/fr3_run.json and out/fr3_arrays_seed{s}.npz.

    ... python fr3_run.py
"""
import json
import time

import numpy as np

import fr3_lib as L

RES = L.R3DIR / "results"
AB = L.ROOT / "src/test/20261030_aspect_baselines/results"
SEEDS = (49, 50, 51)
METRICS = ("r1", "gain", "other", "swap", "strict")
AFF42 = {"fused": {0: 39, 1: 119}, "cf": {0: 149, 1: 10}}
R142 = {"fused": {0: 116, 1: 119}, "cf": {0: 58, 1: 123}}

CMP = []          # (name, mine, theirs, agree)


def record(name, mine, theirs, tol=0.0):
    if isinstance(mine, np.ndarray) or isinstance(theirs, np.ndarray):
        ok = bool(np.array_equal(np.asarray(mine), np.asarray(theirs)))
        CMP.append({"name": name, "agree": ok, "kind": "array"})
    elif isinstance(mine, (float, int)) and isinstance(theirs, (float, int)) and not isinstance(mine, bool):
        ok = abs(float(mine) - float(theirs)) <= tol
        CMP.append({"name": name, "agree": ok, "mine": mine, "theirs": theirs, "diff": float(mine) - float(theirs)})
    else:
        ok = mine == theirs
        CMP.append({"name": name, "agree": bool(ok), "mine": str(mine), "theirs": str(theirs)})
    if not ok:
        print(f"  DISAGREE {name}: mine {mine if not isinstance(mine, np.ndarray) else 'array'} theirs "
              f"{theirs if not isinstance(theirs, np.ndarray) else 'array'}", flush=True)
    return ok


def cmp_ci(name, mine, theirs):
    record(name + ".point", mine["point"], theirs["point"])
    record(name + ".lo", mine["ci95"][0], theirs["ci95"][0])
    record(name + ".hi", mine["ci95"][1], theirs["ci95"][1])


def loadseed(s):
    z = np.load(L.OUT / f"fr3_seed{s}.npz")
    g = {k: z[k] for k in z.files}
    sc = lambda k: {c: {d: g[f"{k}__{c}__{d}"] for d in L.DIRS} for c in L.CONDS}  # noqa: E731
    return {"cl": g["cl"], "parity": g["parity"], "pair_index": g["pair_index"], "anchor": g["anchor"],
            "B": sc("B"), "Bp": sc("Bp"), "cos": sc("cos"), "T": sc("T"),
            "P": {c: g[f"P__{c}"] for c in L.CONDS}, "m": {c: g[f"m__{c}"] for c in L.CONDS},
            "pick": {c: g[f"pick__{c}"] for c in L.CONDS}, "F": {c: g[f"F__{c}"] for c in L.CONDS},
            "stack": {d: g[f"stack__{d}"] for d in L.DIRS}, "meta": json.loads(str(g["meta"]))}


def ext(s, S):
    with np.load(AB / f"per_anchor_seed{s}.npz") as z:
        assert np.array_equal(z["anchor_group"], S["cl"]) and np.array_equal(z["pair_index"], S["pair_index"])
        mc = L.metrics(S["cos"])
        assert all(np.array_equal(mc[m], z[f"cosine__{m}"]) for m in METRICS), "cosine per-anchor differs"
        return {k: {m: np.asarray(z[f"{k}__{m}"], np.float64) for m in METRICS} for k in ("cosine", "rca")}


def family_run(S, glist, with_cf=True):
    fam = L.Family(S["B"], S["T"], glist, S["parity"], with_cf=with_cf)
    fp, fcrit = fam.pick_fused()
    out = {"fam": fam, "fpick": fp, "fcrit": fcrit, "sigma": fam.ctrl, "fused": fam.assemble(fp)}
    if with_cf:
        cp, ccrit = fam.pick_cf()
        out.update(cpick=cp, ccrit=ccrit, cf=fam.assemble(cp, cf=True))
    return out


def frozen(fam, cells):
    return fam.assemble(cells["fused"]), fam.assemble(cells["cf"], cf=True)


def main():
    t0 = time.time()
    R = {"per_seed": {}}
    data = []
    for s in SEEDS:
        S = loadseed(s)
        E = ext(s, S)
        S["pB"], S["pBp"] = L.metrics(S["B"]), L.metrics(S["Bp"])
        gA = L.gates(S["m"], S["pick"], "AFF")
        gR = L.gates(S["m"], S["pick"], "R1")
        aff = family_run(S, gA)
        r1 = family_run(S, gR)
        share = {c: int(np.count_nonzero(gA[0][c])) / len(gA[0][c]) for c in L.CONDS}
        rnd = {r: family_run(S, L.random_gates(gR, share, 100 * s + r, len(S["cl"]))) for r in (0, 1)}
        fzA = frozen(aff["fam"], AFF42)
        fzR = frozen(r1["fam"], R142)
        red = L.redundancy(S["stack"], S["B"])
        D = {"seed": s, "cl": S["cl"], "pair_index": S["pair_index"], "parity": S["parity"], "cosine": E["cosine"],
             "rca": E["rca"], "B": S["pB"], "Bp": S["pBp"], "aff": aff["fused"], "aff_cf": aff["cf"],
             "r1": r1["fused"], "r1_cf": r1["cf"], "fzA": fzA[0], "fzA_cf": fzA[1], "fzR": fzR[0], "fzR_cf": fzR[1],
             "rnd0": rnd[0]["fused"], "rnd0_cf": rnd[0]["cf"], "rnd1": rnd[1]["fused"], "rnd1_cf": rnd[1]["cf"],
             "gA": gA, "gR": gR, "pick": S["pick"], "m": S["m"], "share": share}
        data.append(D)
        R["per_seed"][str(s)] = {
            "cells": {"AFF": {"fused": aff["fpick"], "cf": aff["cpick"], "fcrit": aff["fcrit"], "ccrit": aff["ccrit"]},
                      "R1": {"fused": r1["fpick"], "cf": r1["cpick"], "fcrit": r1["fcrit"], "ccrit": r1["ccrit"]},
                      "rnd0": {"fused": rnd[0]["fpick"], "cf": rnd[0]["cpick"]},
                      "rnd1": {"fused": rnd[1]["fpick"], "cf": rnd[1]["cpick"]}},
            "sigma": {h: aff["sigma"][h] for h in (0, 1)}, "sigma_R1": {h: r1["sigma"][h] for h in (0, 1)},
            "share": share, "redundancy": red, "meta": S["meta"],
            "B_r1": 100 * S["pB"]["r1"].mean(), "Bp_r1": 100 * S["pBp"]["r1"].mean(),
            "aff_r1": 100 * aff["fused"]["r1"].mean(), "aff_cf_r1": 100 * aff["cf"]["r1"].mean()}
        # ---- compare with the implementation's GO arrays (go_seed{s}.npz) and caches
        with np.load(RES / f"go_seed{s}.npz") as z:
            for who, pa in (("aff_fused", aff["fused"]), ("aff_cf", aff["cf"]), ("r1_fused", r1["fused"]),
                            ("B", S["pB"]), ("Bp", S["pBp"]), ("cosine", E["cosine"]), ("rca", E["rca"])):
                for m in METRICS:
                    record(f"seed{s}.{who}.{m}", pa[m], z[f"{who}__{m}"])
            record(f"seed{s}.cl", S["cl"], z["cl"])
            record(f"seed{s}.pair_index", S["pair_index"], z["pair_index"])
            record(f"seed{s}.aff_fused_cells", [aff["fpick"][0], aff["fpick"][1]], z["aff_fused_cells"].tolist())
            record(f"seed{s}.aff_cf_cells", [aff["cpick"][0], aff["cpick"][1]], z["aff_cf_cells"].tolist())
            record(f"seed{s}.r1_fused_cells", [r1["fpick"][0], r1["fpick"][1]], z["r1_fused_cells"].tolist())
            record(f"seed{s}.sigma", [aff["sigma"][0][0], aff["sigma"][1][0]], z["sigma"].tolist())
        with np.load(RES / f"cache_reader_seed{s}.npz") as z:
            for c in L.CONDS:
                record(f"seed{s}.P.{c}", S["P"][c], z[f"P__{c}"])
                record(f"seed{s}.m.{c}", S["m"][c], z[f"m__{c}"])
                record(f"seed{s}.pick.{c}", S["pick"][c], z[f"pick__{c}"])
                for d in L.DIRS:
                    record(f"seed{s}.T.{c}.{d}", S["T"][c][d], z[f"T__{c}__{d}"])
        with np.load(RES / f"cache_seed{s}.npz") as z:
            keys = set(z.files)
            for k in sorted(keys):
                pass
            R["per_seed"][str(s)]["cache_keys"] = sorted(keys)[:80]
        print(f"seed {s} done [{time.time() - t0:.0f}s]", flush=True)

    cl_all = np.concatenate([D["cl"] for D in data])
    pi_all = np.concatenate([D["pair_index"] for D in data])
    C = lambda k: L.cat([D[k] for D in data])  # noqa: E731

    def scope(fused, cf, second, mask=None, ds=data):
        Sx = {"fused": L.cat([D[fused] for D in ds]), "cf": L.cat([D[cf] for D in ds]),
              "second": L.cat([D[second] for D in ds]) if second else None,
              "cosine": L.cat([D["cosine"] for D in ds]), "rca": L.cat([D["rca"] for D in ds]),
              "B": L.cat([D["B"] for D in ds]), "Bp": L.cat([D["Bp"] for D in ds])}
        cl = np.concatenate([D["cl"] for D in ds])
        if mask is not None:
            Sx = {k: (L.sub(v, mask) if v is not None else None) for k, v in Sx.items()}
            cl = cl[mask]
        return Sx, cl

    # ---- GO checks and secondary, pooled
    Sx, cl = scope("aff", "aff_cf", "r1")
    go = L.seven(Sx, cl)
    R["go"] = go
    R["verdict"] = "GO" if go["go"] else "NO-GO"
    R["n_clusters_pooled"] = int(len(np.unique(cl_all)))
    stored = json.loads((RES / "go_pooled.json").read_text())
    verdict = json.loads((RES / "test_verdict.json").read_text())
    for k in stored["checks"]:
        cmp_ci(f"pooled.{k}", go[k], stored["checks"][k])
        cmp_ci(f"verdict.{k}", go[k], verdict["checks"][k])
        record(f"pooled.{k}.pass", go[k]["pass"], stored["checks"][k]["pass"])
    cmp_ci("pooled.secondary", go["secondary"], stored["secondary"])
    cmp_ci("verdict.secondary", go["secondary"], verdict["secondary"])
    record("verdict", R["verdict"], verdict["verdict"])
    record("n_clusters_pooled", R["n_clusters_pooled"], stored["n_clusters_pooled"])

    # ---- descriptive (§7), compared with descriptive.json
    desc = json.loads((RES / "descriptive.json").read_text())
    i1, i2, i3 = desc["item1_per_seed_and_per_pair"], desc["item2_bar_margin_cells_frozen"], desc["item3_R1_checks"]
    R["per_seed_checks"] = {}
    for j, D in enumerate(data):
        x = L.seven(*scope("aff", "aff_cf", "r1", ds=[D]))
        R["per_seed_checks"][str(D["seed"])] = x
        for k in list(stored["checks"]) + ["secondary"]:
            cmp_ci(f"desc.seed{D['seed']}.{k}", x[k], i1["per_seed"][str(D["seed"])]["checks"][k]
                   if k != "secondary" else i1["per_seed"][str(D["seed"])]["secondary"])
    R["per_pair"] = {}
    for i, p in enumerate(L.PAIRS):
        x = L.seven(*scope("aff", "aff_cf", "r1", mask=pi_all == i))
        R["per_pair"][p] = x
        for k in list(stored["checks"]) + ["secondary"]:
            cmp_ci(f"desc.pair.{p}.{k}", x[k], i1["per_pair_pooled"][p]["checks"][k]
                   if k != "secondary" else i1["per_pair_pooled"][p]["secondary"])

    def bar(fused, cf, ds=data):
        f, c = L.cat([D[fused] for D in ds]), L.cat([D[cf] for D in ds])
        Bp, B = L.cat([D["Bp"] for D in ds]), L.cat([D["B"] for D in ds])
        clx = np.concatenate([D["cl"] for D in ds])
        pix = np.concatenate([D["pair_index"] for D in ds])
        name, comp, means = L.bar_comparator(Bp, c, B)
        v = f["r1"] - comp["r1"]
        return v, {"comparator": name, "means_pp": [100 * m for m in means], "r1": L.ci(v, clx),
                   "per_pair": {p: L.ci(v[pix == i], clx[pix == i]) for i, p in enumerate(L.PAIRS)},
                   "fused_mean": 100 * f["r1"].mean()}

    vA, bA = bar("aff", "aff_cf")
    vR, bR = bar("r1", "r1_cf")
    R["bar"] = {"AFF": bA, "R1": bR, "AFF_minus_R1": L.ci(vA - vR, cl_all)}
    st = i2["AFF_bar_margin_pooled"]
    record("bar.AFF.comparator", bA["comparator"], st["comparator"])
    cmp_ci("bar.AFF", bA["r1"], st["r1"])
    for p in L.PAIRS:
        cmp_ci(f"bar.AFF.pair.{p}", bA["per_pair"][p], st["per_pair_r1"][p])
    cmp_ci("bar.AFF_minus_R1", R["bar"]["AFF_minus_R1"], i2["AFF_minus_R1_bar_margin_pooled"])
    record("bar.R1.comparator", bR["comparator"], i3["bar_margin_pooled"]["comparator"])
    cmp_ci("bar.R1", bR["r1"], i3["bar_margin_pooled"]["r1"])
    for p in L.PAIRS:
        cmp_ci(f"bar.R1.pair.{p}", bR["per_pair"][p], i3["bar_margin_pooled"]["per_pair_r1"][p])
    R["bar_per_seed"] = {}
    for D in data:
        va, ba = bar("aff", "aff_cf", ds=[D])
        vr, br = bar("r1", "r1_cf", ds=[D])
        ss = str(D["seed"])
        R["bar_per_seed"][ss] = {"AFF": ba, "R1": br, "AFF_minus_R1": L.ci(va - vr, D["cl"]),
                                 "AFF_vs_cf": {"r1": L.ci(D["aff"]["r1"] - D["aff_cf"]["r1"], D["cl"]),
                                               "gain": L.ci(D["aff"]["gain"] - D["aff_cf"]["gain"], D["cl"]),
                                               "either": L.ci(L.either(D["aff"]) - L.either(D["aff_cf"]), D["cl"])},
                                 "net_rankings_vs_Bp": float(np.sum(4 * (D["aff"]["r1"] - D["Bp"]["r1"])))}
        cmp_ci(f"bar.seed{ss}.AFF", ba["r1"], i2["per_seed"][ss]["AFF"]["r1"])
        cmp_ci(f"bar.seed{ss}.R1", br["r1"], i2["per_seed"][ss]["R1"]["r1"])
        record(f"bar.seed{ss}.AFF.comparator", ba["comparator"], i2["per_seed"][ss]["AFF"]["comparator"])
        record(f"bar.seed{ss}.R1.comparator", br["comparator"], i2["per_seed"][ss]["R1"]["comparator"])
        cmp_ci(f"bar.seed{ss}.AFF_minus_R1", R["bar_per_seed"][ss]["AFF_minus_R1"],
               i2["per_seed"][ss]["AFF_minus_R1_bar_margin"])
        for k in ("r1", "gain", "either"):
            cmp_ci(f"seed{ss}.AFF_vs_cf.{k}", R["bar_per_seed"][ss]["AFF_vs_cf"][k], i2["per_seed"][ss]["AFF_vs_counterpart"][k])
    fa, fc = C("aff"), C("aff_cf")
    R["AFF_vs_cf"] = {"r1": L.ci(fa["r1"] - fc["r1"], cl_all), "gain": L.ci(fa["gain"] - fc["gain"], cl_all),
                      "either": L.ci(L.either(fa) - L.either(fc), cl_all)}
    for k in ("r1", "gain", "either"):
        cmp_ci(f"AFF_vs_cf.{k}", R["AFF_vs_cf"][k], i2["AFF_vs_counterpart_pooled"][k])
    rf_, rc_ = C("r1"), C("r1_cf")
    R["R1_vs_cf"] = {"r1": L.ci(rf_["r1"] - rc_["r1"], cl_all), "gain": L.ci(rf_["gain"] - rc_["gain"], cl_all),
                     "either": L.ci(L.either(rf_) - L.either(rc_), cl_all)}

    # R1's own seven checks
    x = L.seven(*scope("r1", "r1_cf", None))
    R["R1_seven"] = x
    for k in stored["checks"]:
        cmp_ci(f"R1.{k}", x[k], i3["checks"][k])
    R["R1_seven_per_seed"] = {str(D["seed"]): L.seven(*scope("r1", "r1_cf", None, ds=[D])) for D in data}
    for ss, xx in R["R1_seven_per_seed"].items():
        for k in stored["checks"]:
            cmp_ci(f"R1.seed{ss}.{k}", xx[k], i3["per_seed"][ss]["checks"][k])

    # frozen-cell line
    fz = i2["frozen_cell_line"]
    xA = L.seven(*scope("fzA", "fzA_cf", "fzR"))
    xR = L.seven(*scope("fzR", "fzR_cf", None))
    _, bfA = bar("fzA", "fzA_cf")
    _, bfR = bar("fzR", "fzR_cf")
    R["frozen"] = {"AFF": xA, "R1": xR, "AFF_bar": bfA, "R1_bar": bfR,
                   "per_seed": {str(D["seed"]): {"AFF": bar("fzA", "fzA_cf", ds=[D])[1]["r1"],
                                                 "R1": bar("fzR", "fzR_cf", ds=[D])[1]["r1"],
                                                 "AFF_minus_R1": L.ci(D["fzA"]["r1"] - D["fzR"]["r1"], D["cl"])}
                                for D in data}}
    for k in list(stored["checks"]) + ["secondary"]:
        cmp_ci(f"frozen.AFF.{k}", xA[k], fz["AFF_pooled"]["checks"][k] if k != "secondary" else fz["AFF_pooled"]["secondary"])
    for k in stored["checks"]:
        cmp_ci(f"frozen.R1.{k}", xR[k], fz["R1_pooled"]["checks"][k])
    cmp_ci("frozen.AFF.bar", bfA["r1"], fz["AFF_pooled"]["bar"]["r1"])
    cmp_ci("frozen.R1.bar", bfR["r1"], fz["R1_pooled"]["bar"]["r1"])
    for p in L.PAIRS:
        cmp_ci(f"frozen.AFF.bar.pair.{p}", bfA["per_pair"][p], fz["AFF_pooled"]["bar"]["per_pair_r1"][p])

    # gate shares (item 4)
    def shares(key):
        out = {}
        for t in range(4):
            ga = np.concatenate([D[key][t]["a"] for D in data])
            gb = np.concatenate([D[key][t]["b"] for D in data])
            E = len(ga)
            row = {"a": 100 * np.count_nonzero(ga) / E, "b": 100 * np.count_nonzero(gb) / E,
                   "overall": 100 * (np.count_nonzero(ga) + np.count_nonzero(gb)) / (2 * E),
                   "count_a": int(np.count_nonzero(ga)), "count_b": int(np.count_nonzero(gb)), "per_pair": {}}
            for i, p in enumerate(L.PAIRS):
                mk = pi_all == i
                row["per_pair"][p] = {"a": 100 * np.count_nonzero(ga[mk]) / mk.sum(),
                                      "b": 100 * np.count_nonzero(gb[mk]) / mk.sum()}
            out[f"tau_{t}"] = row
        return out

    R["gates"] = {"AFF": shares("gA"), "R1": shares("gR")}
    for who in ("AFF", "R1"):
        for t in range(4):
            g = desc["item4_gate_open_shares"][who]["pooled"][f"tau_{t}"]
            for k in ("a", "b", "overall"):
                record(f"gates.{who}.tau{t}.{k}", R["gates"][who][f"tau_{t}"][k], g[k], tol=1e-12)
    # pick accuracy (D14): told emotion->affect(0), style->image(1), genre->image(1)
    aspects = {0: ("emotion", "style"), 1: ("emotion", "genre"), 2: ("style", "genre")}
    told = {"emotion": 0, "style": 1, "genre": 1}
    pa = np.concatenate([D["pick"]["a"] for D in data])
    pb = np.concatenate([D["pick"]["b"] for D in data])
    ta = np.array([told[aspects[i][0]] for i in pi_all])
    tb = np.array([told[aspects[i][1]] for i in pi_all])
    ca, cb = pa == ta, pb == tb
    R["pick"] = {"accuracy": L.ci(0.5 * (ca.astype(float) + cb.astype(float)), cl_all),
                 "both": 100 * float((ca & cb).mean()),
                 "per_pair": {p: {"a": 100 * ca[pi_all == i].mean(), "b": 100 * cb[pi_all == i].mean()}
                              for i, p in enumerate(L.PAIRS)},
                 "shares": {c: {h: 100 * float((pk == j).mean()) for j, h in enumerate(L.A0)}
                            for c, pk in (("a", pa), ("b", pb))},
                 "affect_share_per_pair": {p: {"a": 100 * (pa[pi_all == i] == 0).mean(),
                                               "b": 100 * (pb[pi_all == i] == 0).mean()}
                                           for i, p in enumerate(L.PAIRS)}}
    cmp_ci("pick.accuracy", R["pick"]["accuracy"], desc["item5_pick_accuracy"]["pick_accuracy"]["correct_share"])
    # redundancy (item 6)
    for D in data:
        for h in L.A0:
            for d in L.DIRS:
                record(f"red.seed{D['seed']}.{h}.{d}", R["per_seed"][str(D["seed"])]["redundancy"][h][d],
                       desc["item6_redundancy"][str(D["seed"])]["redundancy"][h][d], tol=1e-9)
    # random-share control (item 7)
    R["random"] = {}
    for r in (0, 1):
        vX, bX = bar(f"rnd{r}", f"rnd{r}_cf")
        st7 = desc["item7_random_share_control"]["draws"][f"r{r}"]
        R["random"][f"r{r}"] = {"bar": bX, "AFF_minus_fused": L.ci(C("aff")["r1"] - C(f"rnd{r}")["r1"], cl_all),
                                "AFF_minus_bar": L.ci(vA - vX, cl_all),
                                "gain_statistic": L.ci(C(f"rnd{r}")["gain"] - C(f"rnd{r}_cf")["gain"], cl_all),
                                "per_pair_AFF_minus": {p: float(bA["per_pair"][p]["point"] - bX["per_pair"][p]["point"])
                                                       for p in L.PAIRS}}
        record(f"random.r{r}.comparator", bX["comparator"], st7["bar_margin_pooled"]["comparator"])
        cmp_ci(f"random.r{r}.bar", bX["r1"], st7["bar_margin_pooled"]["r1"])
        cmp_ci(f"random.r{r}.AFF_minus_fused", R["random"][f"r{r}"]["AFF_minus_fused"], st7["AFF_minus_control_fused_r1"])
        cmp_ci(f"random.r{r}.AFF_minus_bar", R["random"][f"r{r}"]["AFF_minus_bar"], st7["AFF_minus_control_bar_margin"])
    for D in data:
        for c in L.CONDS:
            record(f"random.share.seed{D['seed']}.{c}", D["share"][c],
                   desc["item7_random_share_control"]["shares"][str(D["seed"])][c])
    # chosen cells against descriptive.json
    for D in data:
        ss = str(D["seed"])
        dc = i2["chosen_cells"][ss]
        mine = R["per_seed"][ss]["cells"]
        for who, key in (("AFF", "AFF"), ("R1", "R1"), ("rnd0", "random_r0"), ("rnd1", "random_r1")):
            record(f"cells.seed{ss}.{who}.fused", [mine[who]["fused"][0], mine[who]["fused"][1]],
                   [dc[key]["fused"]["0"]["cell"], dc[key]["fused"]["1"]["cell"]])
            record(f"cells.seed{ss}.{who}.cf", [mine[who]["cf"][0], mine[who]["cf"][1]],
                   [dc[key]["counterpart"]["0"]["cell"], dc[key]["counterpart"]["1"]["cell"]])

    # ---- report diagnostics
    rep = {}
    rep["pooled_means"] = {k: 100 * float(C(k)["r1"].mean()) for k in
                           ("aff", "aff_cf", "r1", "r1_cf", "B", "Bp", "cosine", "rca", "fzA", "fzR")}
    rep["either_cost_ratio_AFF"] = -R["AFF_vs_cf"]["either"]["point"] / R["AFF_vs_cf"]["gain"]["point"]
    rep["either_cost_ratio_R1"] = -R["R1_vs_cf"]["either"]["point"] / R["R1_vs_cf"]["gain"]["point"]
    rep["R1_either_vs_cf"] = R["R1_vs_cf"]["either"]
    rep["bar_kept_share_AFF"] = bA["r1"]["point"] / 0.6998697916666667
    # Table 9 decomposition per pair vs B'
    rep["table9"] = {}
    for who, key in (("AFF", "aff"), ("R1", "r1")):
        f, bp = C(key), C("Bp")
        for i, p in enumerate(L.PAIRS):
            mk = pi_all == i
            rep["table9"][f"{who}.{p}"] = {
                "r1": L.ci((f["r1"] - bp["r1"])[mk], cl_all[mk]), "gain": L.ci((f["gain"] - bp["gain"])[mk], cl_all[mk]),
                "other": L.ci((f["other"] - bp["other"])[mk], cl_all[mk]),
                "either": L.ci((L.either(f) - L.either(bp))[mk], cl_all[mk])}
    # tau_0 closures of R1's gate
    rep["r1_tau0_closed"] = {c: int(sum(int((D["m"][c] < L.TAUS[0]).sum()) for D in data)) for c in L.CONDS}
    # pooled realised half widths
    rep["realised_half_width"] = {k: 0.5 * (go[k]["ci95"][1] - go[k]["ci95"][0])
                                  for k in list(stored["checks"]) + ["secondary"]}
    # AFF tau_0 open share per pair and condition
    rep["aff_tau0_per_pair"] = R["gates"]["AFF"]["tau_0"]["per_pair"]
    # per-pair R1 differences: AFF - R1 per pair
    R["report"] = rep
    np.savez_compressed(L.OUT / "fr3_perepisode.npz", **{
        f"s{D['seed']}__{k}__{m}": np.asarray(D[k][m]) for D in data
        for k in ("aff", "aff_cf", "r1", "r1_cf", "fzA", "fzA_cf", "fzR", "fzR_cf", "rnd0", "rnd0_cf", "rnd1",
                  "rnd1_cf", "B", "Bp", "cosine", "rca") for m in METRICS})
    R["comparisons"] = {"n": len(CMP), "n_agree": sum(c["agree"] for c in CMP),
                        "disagree": [c for c in CMP if not c["agree"]]}
    R["runtime_s"] = time.time() - t0
    (L.OUT / "fr3_run.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                                   else (int(o) if isinstance(o, np.integer) else str(o))))
    (L.OUT / "fr3_run_comparisons.json").write_text(json.dumps(CMP, indent=0, default=str))
    print(f"comparisons {R['comparisons']['n']}, agree {R['comparisons']['n_agree']}; verdict {R['verdict']} "
          f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
