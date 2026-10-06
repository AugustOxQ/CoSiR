"""Exploratory, seed 42, decides nothing. Follow-up of the affect-only gate (AFF in bs_04_readers.py).

  1. Label-free redundancy of each grouping score with B: mean per-row Pearson correlation of z(s_h) and z(B) over
     the 13 candidates, per direction (no labels).
  2. The family version: the set H of groupings whose picks may open the gate is part of the cell and chosen by the
     cross-fit (7 non-empty subsets x 224 cells), with the counterpart choosing among the same cells.
  3. AFF and AFF_one-side (open only if pick^c = affect and P^c(affect) > P^c'(affect)) against R1: paired
     differences, per pair x condition R@1 / other / either, in-sample curves at tau_0 and tau_2 (lu = 0).
  4. The same gates on A1 (round 1's A1 reader, stored as R1/A1's probabilities) with B'(A1) as comparator floor:
     H = all (R1/A1 tied), {affect}, {affect, csd}, {csd}.
Writes results/bs_05_aff.json.
"""
import itertools
import json
import time

import numpy as np

import bs_lib as L
from bs_03_sxg import assembled_scores, first


def row_corr(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x * x).sum(1) * (y * y).sum(1))
    ok = den > 0
    return float(np.mean((x * y).sum(1)[ok] / den[ok]))


def gate_sets(P, g, H, one_side=False):
    pk = {c: P[c].argmax(1) for c in L.CONDITIONS}
    allowed = np.zeros(P["a"].shape[1], bool)
    allowed[list(H)] = True
    out = []
    for t in range(len(g)):
        gg = {}
        for c, o in (("a", "b"), ("b", "a")):
            x = g[t][c] * allowed[pk[c]]
            if one_side:
                x = x * (P[c][:, 0] > P[o][:, 0])
            gg[c] = x.astype(np.float32)
        out.append(gg)
    return out


def run_tied(data, MDs, label, comparator_bp=None):
    fam = L.Family(data, MDs, tied=True).stats()
    fp, cp = fam.crossfit()
    pn, pc = fam.assemble(fp, cp)
    if comparator_bp is not None:
        saved = data.pBp
        data.pBp = comparator_bp
    r, bar_v = L.evaluate(data, pn, pc, label)
    if comparator_bp is not None:
        data.pBp = saved
    r["fused_cells"] = [fam.cells[fp[h]] for h in (0, 1)]
    r["cf_cells"] = [fam.cf_cells[cp[h]] for h in (0, 1)]
    r["in_sample"] = fam.in_sample()
    print(L.fmt(r), r["fused_cells"], r["cf_cells"], flush=True)
    return r, fam, fp, cp, pn, pc, bar_v


def main():
    t0 = time.time()
    data = L.Data()
    out = {"note": "exploratory, seed 42, decides nothing"}
    zB = {d: data.zB["a"][d].numpy().astype(np.float64) for d in L.DIRECTIONS}
    names4 = ("affect", "image", "caption", "csd")

    # ---- 1. redundancy with B
    red = {}
    for h, nm in enumerate(names4):
        red[nm] = {d: row_corr(L.zrows(data.stack4[d][:, h]).numpy().astype(np.float64), zB[d]) for d in L.DIRECTIONS}
    out["corr_zs_h_with_zB"] = red
    print("row correlation of z(s_h) with z(B):", {k: {d: round(v, 3) for d, v in x.items()} for k, x in red.items()})

    # ---- 2/3. A0
    P1 = data.P("R1")
    MDs, taus, zT, g, m = L.standard_MDs(data, P1)
    rR1 = run_tied(data, MDs, "R1")
    MDa = [L.MD(zT, gg) for gg in gate_sets(P1, g, [0])]
    rA = run_tied(data, MDa, "AFF: open on affect picks")
    MDo = [L.MD(zT, gg) for gg in gate_sets(P1, g, [0], one_side=True)]
    rO = run_tied(data, MDo, "AFF_one-side: affect pick and P^c(aff) > P^c'(aff)")
    out["A0"] = {"R1": rR1[0], "AFF": rA[0], "AFF_one_side": rO[0]}
    for nm, rr in (("AFF", rA), ("AFF_one_side", rO)):
        out["A0"][f"{nm}_minus_R1"] = {
            "fused_r1": L.C.point_ci(np.asarray(rr[4]["r1"], float) - np.asarray(rR1[4]["r1"], float), data.cl),
            "bar_margin": L.C.point_ci(rr[6] - rR1[6], data.cl),
            "fused_minus_R1_counterpart": L.C.point_ci(np.asarray(rr[4]["r1"], float) - np.asarray(rR1[5]["r1"], float), data.cl)}
        print("  ", nm, "minus R1:", {k: (round(v["point"], 3), [round(x, 3) for x in v["ci95"]])
                                     for k, v in out["A0"][f"{nm}_minus_R1"].items()})
    # per pair x condition for AFF vs R1
    tab = {}
    for nm, rr in (("R1", rR1), ("AFF", rA)):
        S = assembled_scores(rr[1], rr[2], rr[3])
        for c, col, oth in (("a", 0, 1), ("b", 1, 0)):
            for who in ("fused", "cf"):
                hit = np.mean([first(S[who][c][d]) == col for d in L.DIRECTIONS], axis=0)
                other = np.mean([first(S[who][c][d]) == oth for d in L.DIRECTIONS], axis=0)
                for i, p in enumerate(L.PAIRS):
                    mk = data.pi == i
                    tab.setdefault(nm, {}).setdefault(p, {}).setdefault(c, {})[who] = {
                        "r1": 100 * hit[mk].mean(), "other": 100 * other[mk].mean(), "either": 100 * (hit[mk] + other[mk]).mean()}
    out["A0"]["per_pair_condition"] = tab
    for p in L.PAIRS:
        for c in L.CONDITIONS:
            a, b = tab["R1"][p][c], tab["AFF"][p][c]
            print(f"   {p} {c}: R1 fused/cf R@1 {a['fused']['r1']:.2f}/{a['cf']['r1']:.2f} either {a['fused']['either']:.2f}/"
                  f"{a['cf']['either']:.2f} | AFF fused/cf R@1 {b['fused']['r1']:.2f}/{b['cf']['r1']:.2f} either "
                  f"{b['fused']['either']:.2f}/{b['cf']['either']:.2f}")
    # in-sample curves
    curves = {}
    for nm, rr in (("R1", rR1), ("AFF", rA)):
        fam = rr[1]
        for t in (0, 2):
            for la in L.NESTED_A:
                i = fam.cells.index((t, 0.0, la, la))
                j = fam.cf_cells.index((t, 0.0, la))
                curves.setdefault(nm, []).append({"tau": t, "la": la,
                                                  "r1": 100 * fam.fri[i].sum(dtype=np.int64) / 4 / data.E,
                                                  "gain": 100 * fam.fgi[i].sum(dtype=np.int64) / 4 / data.E,
                                                  "cf_r1": 100 * fam.cri[j].sum(dtype=np.int64) / 4 / data.E})
    out["A0"]["curves_lu0"] = curves
    for nm in curves:
        print("  curve", nm, " ".join(f"t{c['tau']}/{c['la']}:{c['r1']:.2f}({c['cf_r1']:.2f})" for c in curves[nm]))

    # family version: the subset H is part of the cell
    subsets = [s for k in (1, 2, 3) for s in itertools.combinations(range(3), k)]
    big_MDs, labels = [], []
    for H in subsets:
        for t, gg in enumerate(gate_sets(P1, g, H)):
            big_MDs.append(L.MD(zT, gg))
            labels.append((H, t))
    fam = L.Family(data, big_MDs, tied=True).stats()
    fp, cp = fam.crossfit()
    pn, pc = fam.assemble(fp, cp)
    r, _ = L.evaluate(data, pn, pc, "AFF family: H chosen by the cross-fit")
    r["fused_cells"] = [(labels[fam.cells[fp[h]][0]], fam.cells[fp[h]][1:]) for h in (0, 1)]
    r["cf_cells"] = [(labels[fam.cf_cells[cp[h]][0]], fam.cf_cells[cp[h]][1:]) for h in (0, 1)]
    print(L.fmt(r), r["fused_cells"], r["cf_cells"], flush=True)
    out["A0"]["family_H_crossfit"] = r

    # ---- 4. A1
    P1A1 = data.P("R1A1")
    T = L.term(data.stack4, P1A1)
    zT1 = L.zterm(T)
    m1 = L.margins_of(P1A1)
    taus1 = L.thresholds(m1)
    g1 = L.gates_from(m1, taus1)
    out["A1"] = {}
    for H, nm in (((0, 1, 2, 3), "R1/A1 (all)"), ((0,), "A1 affect only"), ((0, 3), "A1 affect + csd"),
                  ((3,), "A1 csd only")):
        MDs1 = [L.MD(zT1, gg) for gg in gate_sets(P1A1, g1, H)]
        rr = run_tied(data, MDs1, nm, comparator_bp=data.pBp1)
        out["A1"][nm] = rr[0]
    out["runtime_s"] = round(time.time() - t0)
    (L.HERE / "results" / "bs_05_aff.json").write_text(json.dumps(L.C.jsonable(out), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
