"""Exploratory, seed 42, decides nothing. Idea: decouple the condition weight from the condition-free weight.

R1's fused score at a cell equals its counterpart at the SAME cell plus (la / 2) * (g^c z(T^c) - g^c' z(T^c')), so the
condition part and the condition-free part share one weight. This script
  0. reproduces R1 (k_top 13 only, = round-1 R-c) with the generic harness (sanity),
  1. decomposes R1's margin at its chosen cells into (fused - counterpart at the same cell) and
     (counterpart at the same cell - counterpart at its own chosen cell),
  2. runs the decoupled family (t, lu, lm, ld), 1,792 fused cells, with the unchanged 224-cell counterpart,
  3. runs the anchored family: fused = the counterpart's own chosen cell on the tune half + ld * D_t' (32 cells),
  4. prints in-sample R@1 / gain along ld at tau_2, lu = 0 for lm in {0, 0.5, 1}.
Writes results/bs_01_decouple.json in this folder.
"""
import json
import time

import numpy as np

import bs_lib as L

OUT = L.HERE / "results" / "bs_01_decouple.json"


def anchored(data, MDs, cf_family, cpick, lds=L.NESTED_A[1:]):
    """Per tune half: base = the counterpart's chosen cell; fused cells (t', ld); min-margin on the tune half."""
    par = data.parity
    zB64 = cf_family.zB64
    fused = {c: {d: np.empty((data.E, 13)) for d in L.DIRECTIONS} for c in L.CONDITIONS}
    chosen = {}
    for half in (0, 1):
        tune, ap = par == half, par != half
        t0, lu0, lm0 = cf_family.cf_cells[cpick[half]]
        M0, _ = MDs[t0]
        best, best_cell = None, None
        cand = [(t, ld) for t in range(len(MDs)) for ld in lds]
        for (t, ld) in cand:
            _, D = MDs[t]
            s = {c: {d: (1 + lu0) * zB64[d] + lm0 * M0[c][d] + ld * D[c][d] for d in L.DIRECTIONS} for c in L.CONDITIONS}
            r, g = L.ints(s)
            crit = min(int(r[tune].sum(dtype=np.int64)) - data.ctrl[half][1], int(g[tune].sum(dtype=np.int64)))
            if best is None or crit > best:
                best, best_cell, best_s = crit, (t, ld), s
        chosen[half] = {"base_cf_cell": [t0, lu0, lm0], "t_prime": best_cell[0], "ld": best_cell[1], "crit": best}
        for c in L.CONDITIONS:
            for d in L.DIRECTIONS:
                fused[c][d][ap] = best_s[c][d][ap]
    return L.per_anchor(fused), chosen


def main():
    t0 = time.time()
    data = L.Data()
    P = data.P("R1")
    MDs, taus, zT, g, m = L.standard_MDs(data, P)
    out = {"note": "exploratory, seed 42, decides nothing", "taus": taus}

    # 0. tied family = R1 / round-1 R-c
    tied = L.Family(data, MDs, tied=True).stats()
    fp, cp = tied.crossfit()
    pn, pc = tied.assemble(fp, cp)
    r, _ = L.evaluate(data, pn, pc, "R1 tied (= round-1 R-c)")
    print(L.fmt(r), "cells", [tied.cells[fp[h]] for h in (0, 1)], [tied.cf_cells[cp[h]] for h in (0, 1)], flush=True)
    out["tied"] = {**r, "fused_cells": [tied.cells[fp[h]] for h in (0, 1)],
                   "cf_cells": [tied.cf_cells[cp[h]] for h in (0, 1)], "in_sample": tied.in_sample()}

    # 1. decomposition at the chosen cells: counterpart at the fused reader's own cell
    par = data.parity
    same = {c: {d: np.empty((data.E, 13)) for d in L.DIRECTIONS} for c in L.CONDITIONS}
    for half in (0, 1):
        ap = par != half
        t, lu, lm, ld = tied.cells[fp[half]]
        M, D = MDs[t]
        s = L.score(tied.zB64, M, D, lu, lm, 0.0)
        for c in L.CONDITIONS:
            for d in L.DIRECTIONS:
                same[c][d][ap] = s[c][d][ap]
    psame = L.per_anchor(same)
    dec = {"fused_minus_cf_same": C_diff(data, pn, psame), "cf_same_minus_cf_own": C_diff(data, psame, pc)}
    print("decomposition:", json.dumps({k: {m: round(v[m]["point"], 3) for m in v} for k, v in dec.items()}), flush=True)
    out["decomposition"] = dec

    # 2. decoupled family
    t1 = time.time()
    fam = L.Family(data, MDs).stats()
    fp2, cp2 = fam.crossfit()
    pn2, pc2 = fam.assemble(fp2, cp2)
    r2, _ = L.evaluate(data, pn2, pc2, "decoupled (t, lu, lm, ld)")
    print(L.fmt(r2), "cells", [fam.cells[fp2[h]] for h in (0, 1)], [fam.cf_cells[cp2[h]] for h in (0, 1)],
          f"[{time.time() - t1:.0f}s]", flush=True)
    out["decoupled"] = {**r2, "fused_cells": [fam.cells[fp2[h]] for h in (0, 1)],
                        "cf_cells": [fam.cf_cells[cp2[h]] for h in (0, 1)], "in_sample": fam.in_sample()}
    out["decoupled_minus_tied_bar"] = C_point(data, pn2, pn, pc2, pc)

    # 3. anchored family
    pn3, ch = anchored(data, MDs, fam, cp2)
    r3, _ = L.evaluate(data, pn3, pc2, "anchored: cf cell + ld * D_t'")
    print(L.fmt(r3), "chosen", ch, flush=True)
    out["anchored"] = {**r3, "chosen": ch}

    # 4. in-sample profile at tau_2, lu = 0
    prof = []
    for lm in (0.0, 0.5, 1.0, 2.0):
        for ld in L.NESTED_A:
            i = fam.cells.index((2, 0.0, lm, ld))
            prof.append({"lm": lm, "ld": ld, "r1": 100 * fam.fri[i].sum(dtype=np.int64) / 4 / data.E,
                         "gain": 100 * fam.fgi[i].sum(dtype=np.int64) / 4 / data.E})
    out["profile_tau2_lu0"] = prof
    for p in prof:
        print(f"  lm {p['lm']:<4} ld {p['ld']:<5} R@1 {p['r1']:.3f} gain {p['gain']:+.3f}")
    out["runtime_s"] = round(time.time() - t0)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(L.C.jsonable(out), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


def C_diff(data, pa, pb):
    return L.C.diff3(pa, pb, data.cl)


def C_point(data, pn_new, pn_old, pc_new, pc_old):
    v_new = np.asarray(pn_new["r1"], np.float64) - np.asarray(pc_new["r1"], np.float64)
    v_old = np.asarray(pn_old["r1"], np.float64) - np.asarray(pc_old["r1"], np.float64)
    return {"margin_diff": L.C.point_ci(v_new - v_old, data.cl),
            "fused_diff": L.C.point_ci(np.asarray(pn_new["r1"], np.float64) - np.asarray(pn_old["r1"], np.float64),
                                       data.cl)}


if __name__ == "__main__":
    main()
