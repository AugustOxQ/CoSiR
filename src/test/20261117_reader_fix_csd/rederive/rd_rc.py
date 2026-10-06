"""Phase 2c: R-c re-derivation (rule 4.3) on the parent Rb_expected_A0, with our own code.

tau_0..tau_3 = numpy.percentile(linear) of the parent's top-two P margins over the 24,576 (episode, condition)
values; gates g^c = 1[m^c >= tau]; fused score z(B) + lam_u z(B) + lam_a (g^c z(T^c)) in aspect_nested._combine's
order (float32 torch, zscore_rows per ranking row, gate after z-scoring), 224 cells (tau outer, lam_u, lam_a), the
min(R@1 - R@1 of B, gain) cross-fit; counterpart G_cf = (g^a z(T^a) + g^b z(T^b))/2 fused as z(B) + lam_u z(B) +
lam_a G_cf with the max-R@1 cross-fit; bar margin, gain statistic, clauses, tau_0 sanity, gate shares, float32 vs
float64 margin gate flips. Two runs: 'mine' (our parent T and margins from out/rd_rb.npz) and 'stored_parent' (the
stored parent's T and margins, isolating the R-c machinery). Writes out/rd_rc.{json,npz}.
"""
import json
import time

import numpy as np
import torch

import rd_core as K
from rd_core import CONDITIONS, DIRECTIONS, METRICS, per_anchor
from src.eval.aspect_nested import nested_cells, nested_scores
from src.model.aspect_rule import zscore_rows

PARENT, CFG = "Rb_expected_A0", "A0"
PCTS = (0, 25, 50, 75)


def z(x):
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32))


def taus_of(m_a, m_b):
    allm = np.concatenate([np.asarray(m_a), np.asarray(m_b)])
    return [float(v) for v in np.percentile(allm, PCTS)]


def gates_of(m, taus):
    return [{c: (np.asarray(m[c]) >= t) for c in CONDITIONS} for t in taus]


def rc_scores(zB, zT, gate, lam_u, lam_a):
    out = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        g = torch.as_tensor(gate[c].astype(np.float32))[:, None]
        for d in DIRECTIONS:
            s = zB[c][d]
            if lam_u > 0:
                s = s + lam_u * zB[c][d]
            if lam_a > 0:
                s = s + lam_a * (g * zT[c][d])
            out[c][d] = s.numpy().astype(np.float32)
    return out


def gcf_term(zT, gate):
    m = {}
    for d in DIRECTIONS:
        ga = torch.as_tensor(gate["a"].astype(np.float32))[:, None]
        gb = torch.as_tensor(gate["b"].astype(np.float32))[:, None]
        m[d] = (ga * zT["a"][d] + gb * zT["b"][d]) / 2
    return {c: {d: m[d] for d in DIRECTIONS} for c in CONDITIONS}


def cf_scores(zB, G, lam_u, lam_a):
    out = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for d in DIRECTIONS:
            s = zB[c][d]
            if lam_u > 0:
                s = s + lam_u * zB[c][d]
            if lam_a > 0:
                s = s + lam_a * G[c][d]
            out[c][d] = s.numpy().astype(np.float32)
    return out


def crossfit(per_cell, parity, criterion):
    """per_cell: list of per-anchor dicts; criterion(pa, tune) -> float; first max wins; returns arrays and picks."""
    out = {m: np.empty(len(parity)) for m in METRICS}
    picks = {}
    for half in (0, 1):
        tune, apply = parity == half, parity != half
        vals = [criterion(pa, tune) for pa in per_cell]
        best = 0
        for i in range(1, len(vals)):
            if vals[i] > vals[best]:
                best = i
        picks[half] = best
        for m in METRICS:
            out[m][apply] = np.asarray(per_cell[best][m])[apply]
    return out, picks


def run(P, T, margins, label):
    t0 = time.time()
    ctx, parity = P.ctx, P.ctx.parity
    cells56 = nested_cells()
    cells = [(ti, u, a) for ti in range(4) for (u, a) in cells56]
    taus = taus_of(margins["a"], margins["b"])
    gates = gates_of(margins, taus)
    zB = {c: {d: z(P.B[c][d]) for d in DIRECTIONS} for c in CONDITIONS}
    zT = {c: {d: z(T[c][d]) for d in DIRECTIONS} for c in CONDITIONS}
    pB_r1 = np.asarray(P.pB["r1"], np.float64)
    fused_pa, cf_pa = [], []
    for ti, u, a in cells:
        fused_pa.append(per_anchor(rc_scores(zB, zT, gates[ti], u, a)))
    G = [gcf_term(zT, gates[ti]) for ti in range(4)]
    for ti, u, a in cells:
        cf_pa.append(per_anchor(cf_scores(zB, G[ti], u, a)))

    def crit_fused(pa, tune):
        r1 = float(np.asarray(pa["r1"])[tune].mean())
        gain = float(np.asarray(pa["gain"])[tune].mean())
        return min(r1 - float(pB_r1[tune].mean()), gain)

    def crit_cf(pa, tune):
        return float(np.asarray(pa["r1"])[tune].mean())

    pn, fpk = crossfit(fused_pa, parity, crit_fused)
    pc, cpk = crossfit(cf_pa, parity, crit_cf)
    picks = {c: np.asarray(margins["pick"][c]) for c in CONDITIONS}
    ev, bar_v = K.eval_arrays(pn, pc, picks, CFG, P)
    ev["crossfit"] = {
        "fused": {h: {"tau_index": cells[i][0], "tau": taus[cells[i][0]], "lambda_u": cells[i][1], "lambda_a": cells[i][2]}
                  for h, i in fpk.items()},
        "counterpart": {h: {"tau_index": cells[i][0], "tau": taus[cells[i][0]], "lambda_u": cells[i][1],
                            "lambda_a": cells[i][2]} for h, i in cpk.items()},
        "B_r1_on_tune_half": {h: float(pB_r1[parity == h].mean()) for h in (0, 1)}}
    ev["taus"] = taus
    ev["gate_open_share"] = {f"tau_{i}": {"overall": 100 * float(np.mean(np.concatenate([g["a"], g["b"]]))),
                                          "a": 100 * float(np.mean(g["a"])), "b": 100 * float(np.mean(g["b"]))}
                             for i, g in enumerate(gates)}
    # tau_0 sanity: gates all open; fused scores equal the parent's nested scores cell by cell
    tau0_open = all(bool(gates[0][c].all()) for c in CONDITIONS)
    eq_cells, maxdiff_cf = True, 0.0
    cfT = K.cf_term(T)
    for (u, a) in cells56:
        mine = rc_scores(zB, zT, gates[0], u, a)
        par = nested_scores(P.B, P.B, T, u, a)
        eq_cells &= all(np.array_equal(mine[c][d], par[c][d]) for c in CONDITIONS for d in DIRECTIONS)
        mcf = cf_scores(zB, G[0], u, a)
        pcf = nested_scores(P.B, P.B, cfT, u, a)
        maxdiff_cf = max(maxdiff_cf, max(float(np.abs(mcf[c][d].astype(np.float64) - pcf[c][d]).max())
                                         for c in CONDITIONS for d in DIRECTIONS))
    pn0, _ = crossfit(fused_pa[:56], parity, crit_fused)
    pc0, _ = crossfit(cf_pa[:56], parity, crit_cf)
    ev["tau0_sanity"] = {"tau0_gate_always_open": tau0_open, "fused_score_arrays_equal_parent_cells": bool(eq_cells),
                         "counterpart_max_abs_score_diff_vs_parent_cf_cells": maxdiff_cf,
                         "tau0_restricted_fused_r1_mean": 100 * float(np.mean(pn0["r1"])),
                         "tau0_restricted_counterpart_r1_mean": 100 * float(np.mean(pc0["r1"]))}
    ev["G_cf_condition_free"] = all(torch.equal(Gt["a"][d], Gt["b"][d]) for Gt in G for d in DIRECTIONS)
    ev["runtime_s"] = round(time.time() - t0, 1)
    print(f"{label}: taus {taus}; bar {ev['bar']['r1']} ({ev['bar']['comparator']}); gain {ev['gain_statistic']}; "
          f"clears {ev['clears_bar']}; cells {ev['crossfit']}", flush=True)
    return ev, pn, pc, bar_v, gates, taus, (pn0, pc0)


def main():
    t0 = time.time()
    P = K.prepare_seed42()
    mz64 = np.load(K.OUT / "rd_rb.npz")
    mz = np.load(K.OUT / "rd_rb_ad.npz")
    sz = np.load(K.RES / f"cand_{PARENT}.npz")
    result, npz = {"checks": P.checks}, {}
    for m in METRICS:
        npz[f"B__{m}"] = np.asarray(P.pB[m])
        npz[f"{CFG}__Bprime__{m}"] = np.asarray(P.pBp[CFG][m])
    variants = {"mine_f64feat": ({c: {d: mz64[f"{PARENT}__T__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS},
                                 {"a": mz64[f"{PARENT}__margin__a"], "b": mz64[f"{PARENT}__margin__b"],
                                  "pick": {c: mz64[f"{PARENT}__pick__{c}"] for c in CONDITIONS}}),
                "mine": ({c: {d: mz[f"{PARENT}__T__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS},
                         {"a": mz[f"{PARENT}__margin__a"], "b": mz[f"{PARENT}__margin__b"],
                          "pick": {c: mz[f"{PARENT}__pick__{c}"] for c in CONDITIONS}}),
                "stored_parent": ({c: {d: sz[f"T__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS},
                                  {"a": sz["margin__a"], "b": sz["margin__b"],
                                   "pick": {c: sz[f"pick__{c}"] for c in CONDITIONS}})}
    gates_by = {}
    for label, (T, margins) in variants.items():
        ev, pn, pc, bar_v, gates, taus, _ = run(P, T, margins, label)
        name = f"Rc_{label}"
        result[name] = ev
        gates_by[label] = (gates, taus)
        for m in METRICS:
            npz[f"{name}__fused__{m}"] = np.asarray(pn[m])
            npz[f"{name}__cf__{m}"] = np.asarray(pc[m])
        npz[f"{name}__bar_v"] = bar_v
        for c in CONDITIONS:
            npz[f"{name}__pick__{c}"] = np.asarray(margins["pick"][c]).astype(np.int8)
            npz[f"{name}__margin__{c}"] = np.asarray(margins[c])
            npz[f"{name}__gate__{c}"] = np.stack([g[c] for g in gates]).astype(np.float32)
            for d in DIRECTIONS:
                npz[f"{name}__T__{c}__{d}"] = np.asarray(T[c][d])
        npz[f"{name}__taus"] = np.asarray(taus)
    # gate flips: our float64 margins vs the stored margins, and float64 vs float32 margins (from our P)
    gm, tm = gates_by["mine"]
    gs, ts = gates_by["stored_parent"]
    result["gate_flips_mine_vs_stored_margins"] = {f"tau_{i}": {c: int((gm[i][c] != gs[i][c]).sum()) for c in CONDITIONS}
                                                   for i in range(4)}
    result["tau_mine_minus_stored"] = [a - b for a, b in zip(tm, ts)]
    g64, t64 = gates_by["mine_f64feat"]
    result["gate_flips_f64feat_vs_mine"] = {f"tau_{i}": {c: int((g64[i][c] != gm[i][c]).sum()) for c in CONDITIONS}
                                            for i in range(4)}
    result["tau_f64feat_minus_mine"] = [a - b for a, b in zip(t64, tm)]
    Pr = {c: mz[f"{CFG}__P__{c}"] for c in CONDITIONS}
    m32 = {c: (lambda s: (s[:, 0] - s[:, 1]))(-np.sort(-Pr[c].astype(np.float32), axis=1)) for c in CONDITIONS}
    t32 = taus_of(m32["a"], m32["b"])
    g32 = gates_of(m32, t32)
    result["gate_flips_f64_vs_f32_margins"] = {f"tau_{i}": {c: int((gm[i][c] != g32[i][c]).sum()) for c in CONDITIONS}
                                               for i in range(4)}
    result["taus_f32_margins"] = t32
    # distance of the nearest margin to each tau (how fragile the gates are)
    allm = np.concatenate([variants["mine"][1]["a"], variants["mine"][1]["b"]])
    result["nearest_margin_distance_to_tau"] = [float(np.min(np.abs(allm - t))) for t in tm]
    result["runtime_s"] = round(time.time() - t0, 1)
    K.save_json(K.OUT / "rd_rc.json", result)
    np.savez_compressed(K.OUT / "rd_rc.npz", **npz)
    print(json.dumps({k: result[k] for k in ("gate_flips_mine_vs_stored_margins", "tau_mine_minus_stored",
                                             "gate_flips_f64feat_vs_mine", "tau_f64feat_minus_mine",
                                             "gate_flips_f64_vs_f32_margins", "nearest_margin_distance_to_tau")}),
          flush=True)


if __name__ == "__main__":
    main()
