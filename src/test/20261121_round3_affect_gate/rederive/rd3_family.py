"""The 224-cell fusion family (rule D6, D8, D9), our own code: gates, z-scores before the gate, gated terms, G_cf, the
integer statistics rho and gamma per cell and episode, the nested control's sigma*, the min-margin cross-fit of the
fused reader, the max-R@1 cross-fit of the counterpart, the assembled per-anchor arrays. Adapted from round 2's own
re-derivation (rd2_family.py, copied, not imported), without the top-k restriction."""
import numpy as np

import rd3_core as K
from rd3_core import COND, DIRS, METRICS


def gates_r1(margins, taus):
    """D6: g_t^c = 1[m^c >= tau_t] (float64 compare)."""
    return [{c: np.asarray(margins[c], np.float64) >= np.float64(t) for c in COND} for t in taus]


def gates_aff(margins, picks, taus):
    """D6: g_t^c = 1[m^c >= tau_t] * 1[pi^c = affect (index 0)]."""
    return [{c: (np.asarray(margins[c], np.float64) >= np.float64(t)) & (np.asarray(picks[c]) == 0) for c in COND}
            for t in taus]


def control(zB, parity):
    """D8 item 5: sigma* = the smallest control sum with the largest rho((1 + sigma) z(B)) per tune half."""
    tune = {h: parity == h for h in (0, 1)}
    ctrl_rho = np.zeros((len(K.CONTROL_SUMS), 2), np.int64)
    for i, s in enumerate(K.CONTROL_SUMS):
        sc = {c: {d: K.combine(zB[c][d], None, float(s), 0.0) for d in DIRS} for c in COND}
        hit = K.counts(sc)[0]
        ctrl_rho[i] = [hit[tune[0]].sum(), hit[tune[1]].sum()]
    idx = {h: int(np.argmax(ctrl_rho[:, h])) for h in (0, 1)}     # first maximum = smallest sum (ascending list)
    return idx, {h: int(ctrl_rho[idx[h], h]) for h in (0, 1)}, ctrl_rho


def run_family(B, T, gates, taus, parity):
    parity = np.asarray(parity)
    E = len(parity)
    tune = {h: parity == h for h in (0, 1)}
    for d in DIRS:
        if not np.array_equal(B["a"][d], B["b"][d]):
            raise AssertionError("B differs between conditions")
    zB = {c: {d: K.z(B[c][d]) for d in DIRS} for c in COND}
    zT = {c: {d: K.z(T[c][d]) for d in DIRS} for c in COND}                     # z before any gate (D8 item 1)
    gz = [{c: {d: gates[t][c].astype(np.float32)[:, None] * zT[c][d] for d in DIRS} for c in COND}
          for t in range(N_T(taus))]
    G = []
    for t in range(N_T(taus)):
        g = {d: (0.5 * (gz[t]["a"][d].astype(np.float64) + gz[t]["b"][d].astype(np.float64))).astype(np.float32)
             for d in DIRS}
        G.append({c: {d: g[d] for d in DIRS} for c in COND})
        if not all(np.array_equal(G[t]["a"][d], G[t]["b"][d]) for d in DIRS):
            raise AssertionError("G_cf differs between conditions")
    sigma_idx, rho_ctrl, ctrl_rho = control(zB, parity)

    f_cnt = np.zeros((K.N_CELLS, 4, E), np.int8)       # hits, others, swap2, strict2
    c_cnt = np.zeros((K.N_CELLS, 4, E), np.int8)
    cf_condfree = True
    for cid in range(K.N_CELLS):
        t, u, a = K.cell_decode(cid)
        lu, la = K.NU[u], K.NA[a]
        fs = {c: {d: K.combine(zB[c][d], gz[t][c][d], lu, la) for d in DIRS} for c in COND}
        cs = {c: {d: K.combine(zB[c][d], G[t][c][d], lu, la) for d in DIRS} for c in COND}
        cf_condfree &= all(np.array_equal(cs["a"][d], cs["b"][d]) for d in DIRS)
        f_cnt[cid] = np.stack(K.counts(fs))
        c_cnt[cid] = np.stack(K.counts(cs))
    if not cf_condfree:
        raise AssertionError("a counterpart cell is not condition-free")
    hit_f, oth_f = f_cnt[:, 0].astype(np.int64), f_cnt[:, 1].astype(np.int64)
    hit_c = c_cnt[:, 0].astype(np.int64)
    rho_f = np.stack([hit_f[:, tune[h]].sum(1) for h in (0, 1)], axis=1)
    gam_f = np.stack([(hit_f - oth_f)[:, tune[h]].sum(1) for h in (0, 1)], axis=1)
    rho_c = np.stack([hit_c[:, tune[h]].sum(1) for h in (0, 1)], axis=1)
    crit_f = np.stack([np.minimum(rho_f[:, h] - rho_ctrl[h], gam_f[:, h]) for h in (0, 1)], axis=1)
    pick_f = {h: int(np.argmax(crit_f[:, h])) for h in (0, 1)}       # first maximum = lowest cell number
    pick_c = {h: int(np.argmax(rho_c[:, h])) for h in (0, 1)}

    def assemble(cnt, pick):
        out = {m: np.empty(E, np.float64) for m in METRICS}
        for h in (0, 1):
            app = parity != h
            met = K.metrics_from_counts(*(cnt[pick[h], i].astype(np.int64) for i in range(4)))
            for m in METRICS:
                out[m][app] = met[m][app]
        return out

    pn, pc = assemble(f_cnt, pick_f), assemble(c_cnt, pick_c)

    # literal check: per-anchor metrics of the literally assembled scores of the chosen cells
    def assembled_scores(pick, kind):
        out = {c: {d: np.empty((E, 13), np.float32) for d in DIRS} for c in COND}
        for h in (0, 1):
            app = parity != h
            t, u, a = K.cell_decode(pick[h])
            term = gz[t] if kind == "f" else G[t]
            for c in COND:
                for d in DIRS:
                    out[c][d][app] = K.combine(zB[c][d], term[c][d], K.NU[u], K.NA[a])[app]
        return out

    lit_f, lit_c = K.metrics(assembled_scores(pick_f, "f")), K.metrics(assembled_scores(pick_c, "c"))
    literal = {"fused": all(np.array_equal(lit_f[m], pn[m]) for m in METRICS),
               "counterpart": all(np.array_equal(lit_c[m], pc[m]) for m in METRICS)}
    if not all(literal.values()):
        raise AssertionError(f"count-based arrays differ from the literally assembled scores: {literal}")
    nt = {h: int(tune[h].sum()) for h in (0, 1)}
    rec = {"control": {str(h): {"sigma_star": float(K.CONTROL_SUMS[sigma_idx[h]]), "rho_ctrl": rho_ctrl[h],
                                "n_tune": nt[h], "r1_on_tune_half": rho_ctrl[h] / (4.0 * nt[h])} for h in (0, 1)},
           "fused_cells": {str(h): {**K.cell_desc(pick_f[h], taus), "criterion": int(crit_f[pick_f[h], h]),
                                    "rho": int(rho_f[pick_f[h], h]), "gamma": int(gam_f[pick_f[h], h]),
                                    "n_cells_at_max": int((crit_f[:, h] == crit_f[pick_f[h], h]).sum()),
                                    "scores_parity": 1 - h} for h in (0, 1)},
           "counterpart_cells": {str(h): {**K.cell_desc(pick_c[h], taus), "rho": int(rho_c[pick_c[h], h]),
                                          "n_cells_at_max": int((rho_c[:, h] == rho_c[pick_c[h], h]).sum()),
                                          "scores_parity": 1 - h} for h in (0, 1)},
           "literal_assembly_equal": literal, "counterpart_condition_free_all_cells": bool(cf_condfree),
           "G_cf_identical_both_conditions": True, "taus": [float(t) for t in taus]}
    extra = {"rho_f": rho_f, "gam_f": gam_f, "rho_c": rho_c, "crit_f": crit_f, "ctrl_rho": ctrl_rho,
             "f_cnt": f_cnt, "c_cnt": c_cnt}
    return rec, pn, pc, extra


def N_T(taus):
    if len(taus) != K.N_TAU:
        raise AssertionError("four thresholds expected")
    return K.N_TAU
