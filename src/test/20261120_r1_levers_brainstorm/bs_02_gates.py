"""Exploratory, seed 42, decides nothing. Where does R1's margin come from per episode, and do label-free gates that
look at BOTH conditions (the contrast between P^a and P^b) concentrate the weight where it buys gain?

  1. Per-episode margin (fused minus counterpart, R1's chosen cells, k_top 13), split into gain and either, binned by
     label-free reader statistics: total variation TV(P^a, P^b), same pick under both conditions, gate pattern at tau_2.
     Also per aspect pair x gate pattern (pair labels used only to describe).
  2. Gate variants in R1's tied family (fused = (1+lu) zB + la g^c zT^c, counterpart with G_cf of the same gates):
       sym_min : g^a = g^b = 1[min(m^a, m^b) >= tau]
       sym_max : g^a = g^b = 1[max(m^a, m^b) >= tau]
       tv      : g^a = g^b = 1[TV(P^a, P^b) >= tau]
       diff    : g = 1[pick^a != pick^b] (one cell set, plus tau_0 = always open)
       soft_m  : g^c = m^c (continuous weight, no threshold) and soft_tv: g = TV
     thresholds = 0/25/50/75th percentiles of the statistic over seed 42 (as R1's).
Writes results/bs_02_gates.json.
"""
import json
import time

import numpy as np

import bs_lib as L


def tied_eval(data, MDs, label):
    fam = L.Family(data, MDs, tied=True).stats()
    fp, cp = fam.crossfit()
    pn, pc = fam.assemble(fp, cp)
    r, _ = L.evaluate(data, pn, pc, label)
    r["fused_cells"] = [fam.cells[fp[h]] for h in (0, 1)]
    r["cf_cells"] = [fam.cf_cells[cp[h]] for h in (0, 1)]
    r["in_sample"] = fam.in_sample()
    print(L.fmt(r), r["fused_cells"], r["cf_cells"], flush=True)
    return r, pn, pc


def gates_sym(stat, pcts=(0, 25, 50, 75)):
    taus = [float(x) for x in np.percentile(stat, pcts)]
    return [{c: (stat >= t).astype(np.float32) for c in L.CONDITIONS} for t in taus], taus


def binned(data, pn, pc, key, groups):
    out = {}
    v_r1 = np.asarray(pn["r1"], np.float64) - np.asarray(pc["r1"], np.float64)
    v_g = np.asarray(pn["gain"], np.float64) - np.asarray(pc["gain"], np.float64)
    v_e = L.C.either(pn) - L.C.either(pc)
    for name, mask in groups.items():
        n = int(mask.sum())
        out[name] = {"share": 100 * n / data.E, "margin_r1": 100 * float(v_r1[mask].mean()),
                     "gain": 100 * float(v_g[mask].mean()), "either": 100 * float(v_e[mask].mean()),
                     "contrib_to_pooled_margin": 100 * float(v_r1[mask].sum()) / data.E}
    print(f"-- {key}")
    for name, v in out.items():
        print(f"   {name:<28s} share {v['share']:5.1f}%  margin {v['margin_r1']:+.3f}  gain {v['gain']:+.3f}  "
              f"either {v['either']:+.3f}  contrib {v['contrib_to_pooled_margin']:+.3f}")
    return out


def main():
    t0 = time.time()
    data = L.Data()
    P = data.P("R1")
    MDs, taus, zT, g, m = L.standard_MDs(data, P)
    out = {"note": "exploratory, seed 42, decides nothing"}
    r1, pn, pc = tied_eval(data, MDs, "R1 (tied, tau on own margins)")
    out["R1"] = r1

    # ---- 1. per-episode margin by label-free statistics
    tv = 0.5 * np.abs(P["a"] - P["b"]).sum(axis=1)
    pa_, pb_ = P["a"].argmax(1), P["b"].argmax(1)
    same = pa_ == pb_
    q = np.percentile(tv, [25, 50, 75])
    tvbin = np.digitize(tv, q)
    ga, gb = g[2]["a"] > 0, g[2]["b"] > 0
    out["bins"] = {}
    out["bins"]["tv_quartile"] = binned(data, pn, pc, "TV(P^a, P^b) quartile",
                                        {f"TV q{i + 1}": tvbin == i for i in range(4)})
    out["bins"]["same_pick"] = binned(data, pn, pc, "same pick under both conditions",
                                      {"same pick": same, "different picks": ~same})
    out["bins"]["gate_pattern_tau2"] = binned(data, pn, pc, "gate pattern at tau_2 (a, b)",
                                              {"open/open": ga & gb, "open/closed": ga & ~gb,
                                               "closed/open": ~ga & gb, "closed/closed": ~ga & ~gb})
    groups = {}
    for i, p in enumerate(L.PAIRS):
        for nm, mk in (("both open", ga & gb), ("one open", ga ^ gb), ("both closed", ~ga & ~gb)):
            groups[f"{p} {nm}"] = (data.pi == i) & mk
    out["bins"]["pair_x_gate"] = binned(data, pn, pc, "pair x gate pattern (pair = description only)", groups)
    groups = {}
    for i, p in enumerate(L.PAIRS):
        groups[f"{p} same"] = (data.pi == i) & same
        groups[f"{p} diff"] = (data.pi == i) & ~same
    out["bins"]["pair_x_same"] = binned(data, pn, pc, "pair x same pick", groups)
    # pick pattern per pair (a-pick, b-pick)
    names = L.A0
    pat = {}
    for i, p in enumerate(L.PAIRS):
        mk = data.pi == i
        cnt = {}
        for x in range(3):
            for y in range(3):
                s = mk & (pa_ == x) & (pb_ == y)
                cnt[f"{names[x]}/{names[y]}"] = round(100 * s.sum() / mk.sum(), 1)
        pat[p] = cnt
    out["pick_patterns_per_pair"] = pat
    print(json.dumps(pat))

    # ---- 2. gate variants (tied family)
    out["variants"] = {}

    def run_variant(name, gl):
        mds = [L.MD(zT, gg) for gg in gl]
        r, _, _ = tied_eval(data, mds, name)
        out["variants"][name] = r

    m_min = np.minimum(m["a"], m["b"])
    m_max = np.maximum(m["a"], m["b"])
    gl, _ = gates_sym(m_min)
    run_variant("sym_min: both gates on min margin", gl)
    gl, _ = gates_sym(m_max)
    run_variant("sym_max: both gates on max margin", gl)
    gl, _ = gates_sym(tv)
    run_variant("tv: both gates on TV(P^a,P^b)", gl)
    # product: per-condition margin gate at R1's tau AND a TV gate at its median
    tv_med = float(np.median(tv))
    gl = [{c: (g[t][c] * (tv >= tv_med)).astype(np.float32) for c in L.CONDITIONS} for t in range(4)]
    run_variant("own-margin gate x TV >= median", gl)
    gl = [{c: np.ones(data.E, np.float32) for c in L.CONDITIONS},
          {c: (~same).astype(np.float32) for c in L.CONDITIONS}]
    run_variant("diff picks: open only if pick^a != pick^b", gl)
    gl = [{c: g[2][c] * (~same) for c in L.CONDITIONS}, {c: g[1][c] * (~same) for c in L.CONDITIONS},
          g[2], g[1]]
    run_variant("own gate tau_1/tau_2 x (pick^a != pick^b)", [{c: x[c].astype(np.float32) for c in x} for x in gl])
    gl = [{c: m[c].astype(np.float32) for c in L.CONDITIONS}]
    run_variant("soft: weight = own margin m^c", gl)
    gl = [{c: tv.astype(np.float32) for c in L.CONDITIONS}]
    run_variant("soft: weight = TV(P^a,P^b)", gl)
    out["runtime_s"] = round(time.time() - t0)
    (L.HERE / "results" / "bs_02_gates.json").write_text(json.dumps(L.C.jsonable(out), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
