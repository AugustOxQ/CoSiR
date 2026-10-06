"""The round-2 fusion family (rule §4.5, §4.6) on seed 42, our own code: weighted term, margins, thresholds, gates,
z-scores, top-k sets from B, restriction, the cells, the integer cross-fits (min-margin for the fused reader, max-R@1 for
the counterpart, the control's sigma*), assembly of the per-anchor arrays. Which k_top values may be scored is decided by
the caller (rd2_core.allowed_ktops(): k_top = 13 only until the controller authorises phase 2)."""
import numpy as np

import rd2_core as K
from rd2_core import COND, DIRS, METRICS
from src.eval.aspect_metrics import per_anchor


def reader_parts(cache, P, config):
    """T^c, picks, margins of a reader with probabilities P = {'a': (E,H), 'b': (E,H)}."""
    idx = K.stack_idx(cache, config)
    T = K.weighted_term(cache["stack"], P, idx)
    pm = {c: K.picks_margins(P[c]) for c in COND}
    return T, {c: pm[c][0] for c in COND}, {c: pm[c][1] for c in COND}


def run_family(cache, T, margins, taus, ktops, keep_cell_counts=False):
    """All cells with k_top in `ktops` (a subset of KTOPS, in KTOPS order). Returns the cross-fit record, the assembled
    per-anchor arrays of the fused reader (pn) and its counterpart (pc), and checks."""
    for k in ktops:
        if k not in K.allowed_ktops():
            raise SystemExit(f"k_top = {k} is not authorised in this phase")
    parity = np.asarray(cache["parity"])
    E = len(parity)
    tune = {h: parity == h for h in (0, 1)}
    B = cache["B"]
    zB = {c: {d: K.z(B[c][d]) for d in DIRS} for c in COND}
    zT = {c: {d: K.z(T[c][d]) for d in DIRS} for c in COND}
    gates = [{c: (np.asarray(margins[c], np.float64) >= np.float64(t)) for c in COND} for t in taus]
    gz = [{c: {d: gates[t][c].astype(np.float32)[:, None] * zT[c][d] for d in DIRS} for c in COND}
          for t in range(len(taus))]
    G = []
    for t in range(len(taus)):
        g = {d: (0.5 * (gz[t]["a"][d].astype(np.float64) + gz[t]["b"][d].astype(np.float64))).astype(np.float32)
             for d in DIRS}
        G.append({c: {d: g[d] for d in DIRS} for c in COND})
    # top-k positions from the stored B (float32); B is condition-free, so the sets are the same under both conditions
    for d in DIRS:
        if not np.array_equal(B["a"][d], B["b"][d]):
            raise AssertionError("B differs between conditions")
    pos = {d: K.topk_positions(B["a"][d])[1] for d in DIRS}

    # control: sigma* = smallest control sum with the largest rho((1 + sigma) z(B)), unrestricted
    ctrl_rho = np.zeros((len(K.CONTROL_SUMS), 2), np.int64)
    for i, s in enumerate(K.CONTROL_SUMS):
        sc = {c: {d: K.combine(zB[c][d], None, float(s), 0.0) for d in DIRS} for c in COND}
        hit = K.counts(sc)[0]
        ctrl_rho[i] = [hit[tune[0]].sum(), hit[tune[1]].sum()]
    sigma_idx = {h: int(np.argmax(ctrl_rho[:, h])) for h in (0, 1)}
    rho_ctrl = {h: int(ctrl_rho[sigma_idx[h], h]) for h in (0, 1)}

    cells = [K.cell_id(K.KTOPS.index(k), t, u, a) for k in ktops for t in range(4) for u in range(len(K.NU))
             for a in range(len(K.NA))]
    n = len(cells)
    if cells != sorted(cells):
        raise AssertionError("cells must be scored in ascending cell number (ties go to the lowest)")
    f_cnt = np.zeros((n, 4, E), np.int8)       # hit, oth, swap2, strict2
    c_cnt = np.zeros((n, 4, E), np.int8)
    cf_condfree = True
    for j, cid in enumerate(cells):
        kappa, t, u, a = K.cell_decode(cid)
        k = K.KTOPS[kappa]
        lu, la = K.NU[u], K.NA[a]
        fs = {c: {d: K.restrict(K.combine(zB[c][d], gz[t][c][d], lu, la), pos[d], k) for d in DIRS} for c in COND}
        cs = {c: {d: K.restrict(K.combine(zB[c][d], G[t][c][d], lu, la), pos[d], k) for d in DIRS} for c in COND}
        cf_condfree &= all(np.array_equal(cs["a"][d], cs["b"][d]) for d in DIRS)
        f_cnt[j] = np.stack(K.counts(fs))
        c_cnt[j] = np.stack(K.counts(cs))
    if not cf_condfree:
        raise AssertionError("a restricted counterpart is not condition-free")
    hit_f, oth_f = f_cnt[:, 0].astype(np.int64), f_cnt[:, 1].astype(np.int64)
    hit_c = c_cnt[:, 0].astype(np.int64)
    rho_f = np.stack([hit_f[:, tune[h]].sum(1) for h in (0, 1)], axis=1)
    gam_f = np.stack([(hit_f - oth_f)[:, tune[h]].sum(1) for h in (0, 1)], axis=1)
    rho_c = np.stack([hit_c[:, tune[h]].sum(1) for h in (0, 1)], axis=1)
    crit_f = np.stack([np.minimum(rho_f[:, h] - rho_ctrl[h], gam_f[:, h]) for h in (0, 1)], axis=1)
    pick_f = {h: int(np.argmax(crit_f[:, h])) for h in (0, 1)}       # first maximum = lowest cell number
    pick_c = {h: int(np.argmax(rho_c[:, h])) for h in (0, 1)}
    # the same choice through round 1's mean criterion (floating point), for comparison only
    nt = {h: int(tune[h].sum()) for h in (0, 1)}
    critf_float = np.stack([np.minimum(rho_f[:, h] / (4.0 * nt[h]) - rho_ctrl[h] / (4.0 * nt[h]),
                                       gam_f[:, h] / (4.0 * nt[h])) for h in (0, 1)], axis=1)
    pick_f_float = {h: int(np.argmax(critf_float[:, h])) for h in (0, 1)}

    def assemble(cnt, pick):
        out = {m: np.empty(E, np.float64) for m in METRICS}
        for h in (0, 1):
            app = parity != h
            met = K.metrics_from_counts(*(cnt[pick[h], i].astype(np.int64) for i in range(4)))
            for m in METRICS:
                out[m][app] = met[m][app]
        return out

    pn, pc = assemble(f_cnt, pick_f), assemble(c_cnt, pick_c)

    # literal check: assemble the scores of the chosen cells and score them with the base library's per_anchor
    def scores_of(cid, kind):
        kappa, t, u, a = K.cell_decode(cid)
        term = gz[t] if kind == "f" else G[t]
        return {c: {d: K.restrict(K.combine(zB[c][d], term[c][d], K.NU[u], K.NA[a]), pos[d], K.KTOPS[kappa])
                    for d in DIRS} for c in COND}

    def assemble_scores(pick, kind):
        out = {c: {d: np.empty((E, 13), np.float64) for d in DIRS} for c in COND}
        for h in (0, 1):
            app = parity != h
            s = scores_of(cells[pick[h]], kind)
            for c in COND:
                for d in DIRS:
                    out[c][d][app] = np.asarray(s[c][d], np.float64)[app]
        return out

    lit_f, lit_c = per_anchor(assemble_scores(pick_f, "f")), per_anchor(assemble_scores(pick_c, "c"))
    literal_equal = {"fused": all(np.array_equal(lit_f[m], pn[m]) for m in METRICS),
                     "counterpart": all(np.array_equal(lit_c[m], pc[m]) for m in METRICS)}
    if not all(literal_equal.values()):
        raise AssertionError(f"count-based per-anchor arrays differ from per_anchor on assembled scores: {literal_equal}")

    rec = {"ktops_scored": list(ktops), "n_cells": n,
           "control": {str(h): {"sigma": float(K.CONTROL_SUMS[sigma_idx[h]]), "rho_ctrl": rho_ctrl[h],
                                "n_tune": nt[h], "r1_on_tune_half": rho_ctrl[h] / (4.0 * nt[h])} for h in (0, 1)},
           "fused_cells": {str(h): {**K.cell_desc(cells[pick_f[h]], taus), "criterion": int(crit_f[pick_f[h], h]),
                                    "rho": int(rho_f[pick_f[h], h]), "gamma": int(gam_f[pick_f[h], h]),
                                    "n_cells_at_max": int((crit_f[:, h] == crit_f[pick_f[h], h]).sum())}
                           for h in (0, 1)},
           "counterpart_cells": {str(h): {**K.cell_desc(cells[pick_c[h]], taus), "rho": int(rho_c[pick_c[h], h]),
                                          "n_cells_at_max": int((rho_c[:, h] == rho_c[pick_c[h], h]).sum())}
                                 for h in (0, 1)},
           "fused_cells_by_float_mean_criterion": {str(h): int(cells[pick_f_float[h]]) for h in (0, 1)},
           "literal_per_anchor_equal": literal_equal, "counterpart_condition_free_all_cells": bool(cf_condfree),
           "taus": [float(t) for t in taus],
           "gate_open_share": {f"tau_{t}": {"overall": 100 * float(np.mean(np.concatenate([gates[t]["a"], gates[t]["b"]]))),
                                            "a": 100 * float(np.mean(gates[t]["a"])),
                                            "b": 100 * float(np.mean(gates[t]["b"])),
                                            "per_pair": {p: 100 * float(np.mean(np.concatenate(
                                                [gates[t]["a"][cache["pair_index"] == i],
                                                 gates[t]["b"][cache["pair_index"] == i]])))
                                                for i, p in enumerate(K.PAIR_NAMES)}}
                               for t in range(len(taus))}}
    extra = {"cells": np.asarray(cells), "rho_f": rho_f, "gam_f": gam_f, "rho_c": rho_c, "crit_f": crit_f,
             "ctrl_rho": ctrl_rho, "gates": gates}
    if keep_cell_counts:
        extra["f_cnt"], extra["c_cnt"] = f_cnt, c_cnt
    return rec, pn, pc, extra
