"""Report diagnostics for docs/reports/auto/v2/2026-11-19_reader_fix_round2.md (reader-fix round 2).

Descriptive numbers computed for the report; none enters the decision rule (src/test/20261118_reader_fix_round2/
DECISION_RULE.md) and none changes its verdict. Reads only: the round-2 results files (cand_R{1,2,3}_A0 and cand_R1_A1, json and npz),
round 1's round-1 R-c arrays (D14), round 1's cand_Rb_expected_A1.npz and the seed-42 bundle built by round 1's
common.load_bundle (about 72 s, CPU).
Writes only diagnostics.json next to this script.

For each reader R1, R2, R3 it rebuilds the 896-cell statistics from the stored probabilities with round 2's own code
(r2_fusion, imported read-only), asserts that the stored terms, picks, margins, thresholds, chosen cells and per-anchor
arrays are reproduced exactly, and then computes:
  1. the cross-fit restricted to the k_top = 13 cells (cells 0 to 223) for the fused reader and the counterpart: what
     each reader scores without the top-k restriction (for R1 this must be round-1 R-c exactly), and the paired effect
     of allowing k_top < 13;
  2. for every chosen cell: its tune-half criterion and its R@1 and gain on the other half, beside its k_top = 13 twin;
  3. full-seed (in-sample, not cross-fitted) R@1 and gain along the term weight at lambda_u = 0, tau_2, k_top = 13;
  4. each reader evaluated at round-1 R-c's cells (fused 116/119, counterpart 58/123): the reader change with the
     fusion settings held fixed;
  5. label-free and told-mapping reader statistics: mean top probability, mean total variation between P^a and P^b,
     share of episodes with the same pick under both conditions, the per-row correlation of z(T^a) and z(T^b), the
     probability on the told grouping, told contrast on the two emotion pairs, pick accuracy with the gate open;
  6. at each reader's chosen cells, the B rank of the fused reader's first place and of the target, and what
     restricting the same scores to k_top = 5, 3, 2 would have flipped (rankings that turn from miss to hit and back);
  7. the A1 ablation reader (R1/A1) against round 1's R-b expected on A1: fused arrays and counterpart change.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python docs/reports/assets/2026-11-19_reader_fix_round2/diagnostics.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
R2DIR = ROOT / "src/test/20261118_reader_fix_round2"
RES = R2DIR / "results"
OUT = Path(__file__).resolve().parent / "diagnostics.json"
sys.path.insert(0, str(R2DIR))

import r2_common as R  # noqa: E402  (also puts round 1's folder and the repo root on sys.path)
import r2_fusion as F  # noqa: E402
import run_r2_fusion as RF  # noqa: E402

C, K = R.C, R.K
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_nested import NESTED_A, _zdict  # noqa: E402

READERS = ("R1", "R2", "R3")
PAIRS = C.POOLED_ORDER
RC_FUSED, RC_CF = {0: 116, 1: 119}, {0: 58, 1: 123}


def pp(x):
    return 100.0 * float(x)


def load_cand(name):
    rec = json.loads((RES / f"cand_{name}.json").read_text())
    assert rec["provenance"]["rule_sha256"] == R.RULE_SHA
    npz = RES / f"cand_{name}.npz"
    assert R.sha_file(npz) == rec["npz_sha256"], f"{npz} differs from its json"
    return rec, np.load(npz)


def summary(bundle, pn, pc):
    """Bar margin (with the rule's comparator), margin, gain statistic, means: the quantities of D10 and D11."""
    ctx, cl = bundle.ctx, bundle.cl
    bar_v, bar = C.bar_info(pn, pc, bundle.pBp["A0"], bundle.pB, cl, ctx.pair_index)
    d = C.diff3(pn, pc, cl)
    return bar_v, {"fused_r1": pp(np.mean(pn["r1"])), "counterpart_r1": pp(np.mean(pc["r1"])),
                   "comparator": bar["comparator"], "bar_margin": bar["r1"], "margin": d["r1"],
                   "gain_statistic": d["gain"], "either_vs_counterpart": d["either"]}


def half_stats(fri, fgi, cell, parity, half):
    """Criterion pieces on the tune half and R@1 / gain (pp) of the cell on the other half."""
    tune, app = parity == half, parity != half
    return {"cell": int(cell), "setting": F.cell_values(int(cell)),
            "rho_tune": int(fri[cell][tune].sum(dtype=np.int64)),
            "gamma_tune": None if fgi is None else int(fgi[cell][tune].sum(dtype=np.int64)),
            "r1_other_half": 100.0 * fri[cell][app].sum(dtype=np.int64) / 4.0 / app.sum(),
            "gain_other_half": None if fgi is None else 100.0 * fgi[cell][app].sum(dtype=np.int64) / 4.0 / app.sum()}


def row_corr(x, y):
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    ok = den > 0
    return float(np.mean((x * y).sum(axis=1)[ok] / den[ok])), int((~ok).sum())


def reader_stats(bundle, P, picks, margins, taus, zT):
    ctx = bundle.ctx
    pi = np.asarray(ctx.pair_index)
    told = C.told_index(C.CONFIGS["A0"], C.TOLD["A0"], pi)
    rows = np.arange(len(pi))
    out = {}
    top = np.concatenate([P["a"].max(axis=1), P["b"].max(axis=1)])
    out["top_probability_mean"] = float(top.mean())
    tv = 0.5 * np.abs(P["a"] - P["b"]).sum(axis=1)
    out["tv_a_b_mean"] = float(tv.mean())
    out["tv_a_b_per_pair"] = {p: float(tv[pi == i].mean()) for i, p in enumerate(PAIRS)}
    same = np.asarray(picks["a"]) == np.asarray(picks["b"])
    out["same_pick_share"] = pp(same.mean())
    out["same_pick_share_per_pair"] = {p: pp(same[pi == i].mean()) for i, p in enumerate(PAIRS)}
    za = {d: zT["a"][d].numpy().astype(np.float64) for d in DIRECTIONS}
    zb = {d: zT["b"][d].numpy().astype(np.float64) for d in DIRECTIONS}
    out["corr_zTa_zTb"] = {d: row_corr(za[d], zb[d])[0] for d in DIRECTIONS}
    out["corr_zTa_zTb_constant_rows"] = {d: row_corr(za[d], zb[d])[1] for d in DIRECTIONS}
    # probability on the told grouping, per condition and pair
    pt = {c: P[c][rows, told[c]] for c in CONDITIONS}
    out["p_told_mean"] = {c: float(pt[c].mean()) for c in CONDITIONS}
    out["p_told_per_pair"] = {p: {c: float(pt[c][pi == i].mean()) for c in CONDITIONS} for i, p in enumerate(PAIRS)}
    # told contrast on the emotion pairs: how much more the told grouping of each condition gets under its own condition
    emo = np.isin(pi, [0, 1])
    ga, gb = told["a"], told["b"]
    con = 0.5 * ((P["a"][rows, ga] - P["b"][rows, ga]) + (P["b"][rows, gb] - P["a"][rows, gb]))
    out["told_contrast_emotion_pairs"] = {p: float(con[pi == i].mean()) for i, p in enumerate(PAIRS[:2])}
    out["told_contrast_emotion_pairs_pooled"] = float(con[emo].mean())
    both = (np.asarray(picks["a"]) == ga) & (np.asarray(picks["b"]) == gb)
    out["both_correct_share_per_pair"] = {p: pp(both[pi == i].mean()) for i, p in enumerate(PAIRS)}
    # pick accuracy with the gate open / closed at each tau (diagnostic only)
    corr = {c: np.asarray(picks[c]) == told[c] for c in CONDITIONS}
    acc = {}
    for t, tau in enumerate(taus):
        op = np.concatenate([np.asarray(margins[c]) >= tau for c in CONDITIONS])
        cc = np.concatenate([corr[c] for c in CONDITIONS])
        acc[f"tau_{t}"] = {"open_share": pp(op.mean()), "acc_open": pp(cc[op].mean()),
                           "acc_closed": None if op.all() else pp(cc[~op].mean())}
    out["pick_accuracy_by_gate"] = acc
    out["pick_accuracy"] = pp(np.mean(np.concatenate([corr[c] for c in CONDITIONS])))
    return out


def rank_analysis(bundle, info, fused_scores):
    """At the assembled fused score: B rank of the first-placed candidate and of the target, per ranking row, and the
    flips that restricting the same scores to B's top k would cause (miss -> hit, hit -> miss)."""
    tot, out = 0, {}
    first_pos_all, tpos_all, hit_all = [], [], []
    flips = {k: {"miss_to_hit": 0, "hit_to_miss": 0} for k in (5, 3, 2)}
    for c, col in (("a", 0), ("b", 1)):
        for d in DIRECTIONS:
            S = np.asarray(fused_scores[c][d], np.float64)
            pos = info[d]["pos"]
            first = S.argmax(axis=1)
            mx = S.max(axis=1, keepdims=True)
            unique = (S == mx).sum(axis=1) == 1
            hit = unique & (first == col)
            first_pos_all.append(pos[np.arange(len(S)), first])
            tpos_all.append(pos[:, col])
            hit_all.append(hit)
            for k in flips:
                Sr = F.restrict(S, pos, k)
                fr = Sr.argmax(axis=1)
                mr = Sr.max(axis=1, keepdims=True)
                hr = ((Sr == mr).sum(axis=1) == 1) & (fr == col)
                flips[k]["miss_to_hit"] += int((hr & ~hit).sum())
                flips[k]["hit_to_miss"] += int((hit & ~hr).sum())
            tot += len(S)
    fp, tp, h = np.concatenate(first_pos_all), np.concatenate(tpos_all), np.concatenate(hit_all)
    out["n_rankings"] = tot
    out["hits"] = int(h.sum())
    for k in (5, 3, 2):
        outside = fp >= k
        out[f"first_outside_top{k}_share"] = pp(outside.mean())
        out[f"first_outside_top{k}_hit_share"] = pp(h[outside].mean()) if outside.any() else None
        out[f"target_outside_top{k}_share"] = pp((tp >= k).mean())
        out[f"restrict_top{k}"] = {**flips[k], "net_rankings": flips[k]["miss_to_hit"] - flips[k]["hit_to_miss"],
                                   "net_r1_pp": 100.0 * (flips[k]["miss_to_hit"] - flips[k]["hit_to_miss"]) / tot}
    out["first_B_rank_hist"] = np.bincount(fp, minlength=13).tolist()
    out["target_B_rank_hist"] = np.bincount(tp, minlength=13).tolist()
    return out


def main():
    t0 = time.time()
    R.assert_rule()
    R.assert_inputs(["results/cand_Rc_Rb_expected_A0.npz"])
    rc = np.load(R.r1_path("results/cand_Rc_Rb_expected_A0.npz"))
    bundle = C.load_bundle(smoke=False)
    ctx, cl = bundle.ctx, bundle.cl
    parity = np.asarray(ctx.parity)
    zB = _zdict(bundle.B)
    info = F.rank_info(bundle.B)
    ctrl = F.control_choice(zB, parity)
    res = {"note": "diagnostics for the report; descriptive, decide nothing", "readers": {}}
    curve_cells = [F.cell_number(0, 2, 0, a) for a in range(len(NESTED_A))]
    for name in READERS:
        rec, z = load_cand(f"{name}_A0")
        P = {c: np.asarray(z[f"probs__{c}"]) for c in CONDITIONS}
        T, picks, margins = RF.terms_from_probs(bundle, "A0", P)
        for c in CONDITIONS:
            assert np.array_equal(np.asarray(picks[c]), z[f"pick__{c}"].astype(np.int64))
            assert np.array_equal(np.asarray(margins[c]), z[f"margin__{c}"])
            for d in DIRECTIONS:
                assert np.array_equal(np.asarray(T[c][d], np.float32), z[f"T__{c}__{d}"])
        taus, _ = K.thresholds(margins)
        assert taus == rec["taus"] == list(z["taus"])
        zT = _zdict(T)
        g = K.gates(margins, taus)
        gated = {t: K.gated_terms(zT, g[t]) for t in g}
        G = {t: K.g_cf(gated[t]) for t in g}
        fri, fgi, cri = F.cell_statistics(zB, info, gated, G)
        fp = F.select_fused(fri, fgi, ctrl, parity)
        cp = F.select_cf(cri, parity)
        assert [fp[0], fp[1]] == z["fused_cells"].tolist() and [cp[0], cp[1]] == z["cf_cells"].tolist()
        fused = F.assemble(zB, info, gated, fp, parity)
        cfs = F.assemble(zB, info, G, cp, parity)
        pn, pc = per_anchor(fused), per_anchor(cfs)
        for m in METRICS:
            assert np.array_equal(np.asarray(pn[m]), z[f"fused__{m}"]) and np.array_equal(np.asarray(pc[m]), z[f"cf__{m}"])
        bar_v, s896 = summary(bundle, pn, pc)
        assert np.array_equal(bar_v, z["bar_v"]) and s896["bar_margin"]["point"] == rec["bar"]["r1"]["point"]

        # 1. k_top = 13 only (cells 0 to 223)
        a13 = range(F.CELLS_PER_KAPPA)
        fp13, cp13 = F.select_fused(fri, fgi, ctrl, parity, a13), F.select_cf(cri, parity, a13)
        pn13 = per_anchor(F.assemble(zB, info, gated, fp13, parity))
        pc13 = per_anchor(F.assemble(zB, info, G, cp13, parity))
        bar13, s13 = summary(bundle, pn13, pc13)
        if name == "R1":
            assert np.array_equal(np.asarray(pn13["r1"]), rc["fused__r1"]) and np.array_equal(np.asarray(pc13["r1"]), rc["cf__r1"])
            assert np.array_equal(bar13, rc["bar_v"])
        topk = {"fused_r1": C.point_ci(np.asarray(pn["r1"], np.float64) - np.asarray(pn13["r1"], np.float64), cl),
                "counterpart_r1": C.point_ci(np.asarray(pc["r1"], np.float64) - np.asarray(pc13["r1"], np.float64), cl),
                "bar_margin": C.point_ci(bar_v - bar13, cl)}

        # 2. every chosen cell beside its k_top = 13 twin and the 224-cell pick
        cells = {}
        for h in (0, 1):
            cells[f"fused_tune{h}"] = {"chosen": half_stats(fri, fgi, fp[h], parity, h),
                                       "twin_k13": half_stats(fri, fgi, fp[h] % F.CELLS_PER_KAPPA, parity, h),
                                       "pick_k13_only": half_stats(fri, fgi, fp13[h], parity, h),
                                       "rho_ctrl": int(ctrl[h][1])}
            cells[f"cf_tune{h}"] = {"chosen": half_stats(cri, None, cp[h], parity, h),
                                    "twin_k13": half_stats(cri, None, cp[h] % F.CELLS_PER_KAPPA, parity, h),
                                    "pick_k13_only": half_stats(cri, None, cp13[h], parity, h)}

        # 3. full-seed curve along lambda_a at lambda_u = 0, tau_2, k_top = 13 (in-sample)
        curve = [{"lambda_a": float(NESTED_A[a]), "cell": int(i), "r1": 100.0 * fri[i].mean() / 4.0,
                  "gain": 100.0 * fgi[i].mean() / 4.0, "cf_r1": 100.0 * cri[i].mean() / 4.0}
                 for a, i in enumerate(curve_cells)]
        best = {"fused_best_r1_cell_any": int(np.argmax(fri.sum(axis=1, dtype=np.int64))),
                "fused_best_r1_any": 100.0 * fri.sum(axis=1, dtype=np.int64).max() / 4.0 / fri.shape[1],
                "fused_best_r1_k13": 100.0 * fri[:224].sum(axis=1, dtype=np.int64).max() / 4.0 / fri.shape[1],
                "cf_best_r1_any": 100.0 * cri.sum(axis=1, dtype=np.int64).max() / 4.0 / cri.shape[1],
                "cf_best_r1_k13": 100.0 * cri[:224].sum(axis=1, dtype=np.int64).max() / 4.0 / cri.shape[1]}

        # 4. at round-1 R-c's cells
        pnr = per_anchor(F.assemble(zB, info, gated, RC_FUSED, parity))
        pcr = per_anchor(F.assemble(zB, info, G, RC_CF, parity))
        _, srcells = summary(bundle, pnr, pcr)

        # 5. reader statistics; 6. rank analysis at the reader's own chosen cells
        rs = reader_stats(bundle, P, picks, margins, taus, zT)
        ra = rank_analysis(bundle, info, fused)
        res["readers"][name] = {"check": "stored terms, picks, margins, taus, cells and per-anchor arrays reproduced exactly",
                                "taus": taus, "chosen": {"fused": fp, "counterpart": cp},
                                "own_896": s896, "k13_only": {"cells": {"fused": fp13, "counterpart": cp13}, **s13},
                                "topk_effect_896_minus_k13": topk, "cells": cells, "curve_tau2_lu0_k13": curve,
                                "in_sample_best": best, "at_round1_Rc_cells": srcells, "reader": rs, "ranks": ra}
        C.log(f"{name} done")
    # 7. A1 ablation reader (R1/A1) against round 1's R-b expected on A1 (D14 inputs: none; round 1's results read only)
    rec1, z1 = load_cand("R1_A1")
    rb1 = np.load(R.R1 / "results/cand_Rb_expected_A1.npz")
    same = {m: bool(np.array_equal(z1[f"fused__{m}"], rb1[f"fused__{m}"])) for m in METRICS}
    res["a1_vs_round1_Rb_expected_A1"] = {
        "fused_arrays_equal": same,
        "counterpart_r1_diff": C.point_ci(z1["cf__r1"].astype(np.float64) - rb1["cf__r1"].astype(np.float64), cl),
        "bar_v_diff": C.point_ci(z1["bar_v"].astype(np.float64) - rb1["bar_v"].astype(np.float64), cl),
        "cells": {"fused": z1["fused_cells"].tolist(), "counterpart": z1["cf_cells"].tolist()}}
    res["runtime_s"] = round(time.time() - t0, 1)
    res["provenance"] = {"written": R.now_ams(), "rule_sha256": R.RULE_SHA, "git_head": C.git_head()}
    OUT.write_text(json.dumps(C.jsonable(res), indent=1))
    print(f"wrote {OUT} [{res['runtime_s']}s]")


if __name__ == "__main__":
    main()
