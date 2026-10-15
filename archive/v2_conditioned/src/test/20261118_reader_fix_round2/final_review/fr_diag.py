"""Final review of reader-fix round 2: re-derive the report's own diagnostics (sections 4.3 to 5.4 of
docs/reports/auto/v2/2026-11-19_reader_fix_round2.md) from this review's own per-cell statistics (out/fr_fam_*.npz,
written by fr_cands.py) and the cache, with fr_core. Writes out/fr_diag.json.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/final_review/fr_diag.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import fr_core as Q  # noqa: E402
from src.eval.aspect_nested import _zdict  # noqa: E402

R1RES = ROOT / "src/test/20261117_reader_fix_csd/results"
RES = HERE.parent / "results"
NAMES = {"R1": "R1_A0", "R2": "R2_A0", "R3": "R3_A0_mine"}
PAIRS3 = ["emotion__style", "emotion__genre", "style__genre"]


def fam(name):
    z = np.load(Q.OUT / f"fr_fam_{name}.npz")
    return {k: z[k] for k in z.files}


def gated_terms(f, taus):
    T = {c: {d: f[f"T__{c}__{d}"] for d in Q.DIRS} for c in Q.CONDS}
    zT = _zdict(T)
    gated, G = {}, {}
    for t, tau in enumerate(taus):
        g = {c: torch.as_tensor((f[f"margin__{c}"] >= tau).astype(np.float32)) for c in Q.CONDS}
        gated[t] = {c: {d: g[c][:, None] * zT[c][d] for d in Q.DIRS} for c in Q.CONDS}
        avg = {d: (0.5 * (gated[t]["a"][d].numpy().astype(np.float64) + gated[t]["b"][d].numpy().astype(np.float64)))
               .astype(np.float32) for d in Q.DIRS}
        G[t] = {c: {d: torch.as_tensor(avg[d].copy()) for d in Q.DIRS} for c in Q.CONDS}
    return zT, gated, G


def main():
    cache = Q.load_cache()
    cl, parity, pi = cache["cl"], cache["parity"], cache["pair_index"]
    B = Q.B_of(cache)
    pos = Q.positions(B)
    zB = _zdict(B)
    ctrl = Q.control(cache, zB)
    out = {"ctrl": {h: list(ctrl[h]) for h in (0, 1)}}
    F = {r: fam(n) for r, n in NAMES.items()}
    rc = np.load(R1RES / "cand_Rc_Rb_expected_A0.npz")
    rcj = json.loads((R1RES / "cand_Rc_Rb_expected_A0.json").read_text())
    out["round1_Rc_either"] = rcj["margin"]["either"]
    # item 2 paired differences
    vec = {r: {"fused": F[r]["f_r1"], "margin": F[r]["f_r1"] - F[r]["c_r1"], "bar": F[r]["bar_v"]} for r in F}
    vec["Rc"] = {"fused": rc["fused__r1"].astype(np.float64), "margin": rc["fused__r1"] - rc["cf__r1"],
                 "bar": rc["bar_v"].astype(np.float64)}
    out["R1_fused_equals_Rc_every_episode"] = bool(np.array_equal(F["R1"]["f_r1"], rc["fused__r1"]))
    out["R1_fused_gain_equals_Rc"] = bool(np.array_equal(F["R1"]["f_gain"], rc["fused__gain"]))
    for x, y in (("R1", "Rc"), ("R2", "R1"), ("R3", "R1")):
        out[f"{x}_minus_{y}"] = {k: Q.ci(vec[x][k] - vec[y][k], cl) for k in ("fused", "margin", "bar")}
    # counterpart move of R1 on tune half 1: cells 571 and 123
    cri = F["R1"]["cri"]
    tune1, other1 = parity == 1, parity != 1
    out["R1_cf_tune1"] = {c: {"rho_tune": int(cri[c][tune1].astype(np.int64).sum()),
                              "r1_other": 100 * cri[c][other1].astype(np.int64).sum() / 4 / other1.sum()}
                          for c in (571, 123)}
    out["R1_cf_r1_round1_vs_now"] = [100 * float(np.mean(rc["cf__r1"])), 100 * float(np.mean(F["R1"]["c_r1"]))]
    # number of tied maxima per pick
    ties = {}
    for r in F:
        for h in (0, 1):
            tune = parity == h
            rho = F[r]["fri"][:, tune].astype(np.int64).sum(1)
            gam = F[r]["fgi"][:, tune].astype(np.int64).sum(1)
            crit = np.minimum(rho - ctrl[h][1], gam)
            crho = F[r]["cri"][:, tune].astype(np.int64).sum(1)
            ties[f"{r}_fused_h{h}"] = np.flatnonzero(crit == crit.max()).tolist()
            ties[f"{r}_cf_h{h}"] = np.flatnonzero(crho == crho.max()).tolist()
    out["ties_at_max"] = ties
    # Table 6: restricted picks beside the best k_top 13 cell of the same tune half
    t6 = {}
    for r, who, h, cell in (("R1", "cf", 1, 571), ("R2", "fused", 1, 615), ("R2", "cf", 0, 278), ("R3", "fused", 1, 574)):
        tune, oth = parity == h, parity != h
        f = F[r]

        def tune_val(c):
            if who == "cf":
                return int(f["cri"][c][tune].astype(np.int64).sum())
            rho = int(f["fri"][c][tune].astype(np.int64).sum())
            gam = int(f["fgi"][c][tune].astype(np.int64).sum())
            return min(rho - ctrl[h][1], gam)

        M = f["cri"] if who == "cf" else f["fri"]
        vals13 = [tune_val(c) for c in range(224)]
        best13 = int(np.argmax(vals13))
        t6[f"{r}_{who}_tune{h}"] = {
            "chosen": [cell, tune_val(cell), 100 * M[cell][oth].astype(np.int64).sum() / 4 / oth.sum()],
            "best_k13": [best13, vals13[best13], 100 * M[best13][oth].astype(np.int64).sum() / 4 / oth.sum()],
            "twin_k13": [cell % 224, tune_val(cell % 224), 100 * M[cell % 224][oth].astype(np.int64).sum() / 4 / oth.sum()]}
    out["table6"] = t6
    # k13-only cross-fits and the top-k effect (from fr_cands results)
    cands = json.loads((Q.OUT / "fr_cands_R1_A0_R2_A0_R3_A0_stored_R3_A0_mine_R1_A1.json").read_text())["results"]
    out["k13_only"] = {r: cands[NAMES[r]]["k13"] for r in F}
    out["topk_effect_bar"] = {r: cands[NAMES[r]]["topk_effect_bar"] for r in F}
    # Table 7 and curves: each reader at round-1 R-c's cells, own thresholds
    t7, curve, best = {}, {}, {}
    for r in F:
        f = F[r]
        fp, cp = {0: 116, 1: 119}, {0: 58, 1: 123}
        fr1, fg = Q.from_cells(f["fri"], fp, parity), Q.from_cells(f["fgi"], fp, parity)
        cr1 = Q.from_cells(f["cri"], cp, parity)
        # other rate: recompute from assembled scores
        taus = f["taus"].tolist()
        zT, gated, G = gated_terms(f, taus)
        famlite = {"zB": zB, "pos": pos}
        pn = Q.per_episode(Q.assembled_scores(famlite, gated, fp, parity))
        pc = Q.per_episode(Q.assembled_scores(famlite, G, cp, parity))
        assert np.array_equal(pn["r1"], fr1) and np.array_equal(pc["r1"], cr1)
        D = Q.decision(cache, pn["r1"], pn["gain"], pc["r1"], pc["gain"])
        either = 100 * float(np.mean((pn["r1"] + pn["other"]) - (pc["r1"] + pc["other"])))
        t7[r] = {"fused": D["fused"], "cf": D["cf"], "margin": D["margin"]["point"], "bar": D["bar"],
                 "comparator": D["comparator"], "gain": D["gain_stat"]["point"], "either": either,
                 "either_per_gain": -either / D["gain_stat"]["point"]}
        cells = [Q.cellno(0, 2, 0, a) for a in range(8)]
        curve[r] = [[Q.NA[a], 100 * f["fri"][c].mean() / 4, 100 * f["fgi"][c].mean() / 4, 100 * f["cri"][c].mean() / 4]
                    for a, c in enumerate(cells)]
        tot = f["fri"].astype(np.int64).sum(1)
        best[r] = {"cell": int(np.argmax(tot)), "r1": 100 * tot.max() / 4 / f["fri"].shape[1],
                   "best_k13": 100 * tot[:224].max() / 4 / f["fri"].shape[1],
                   "best_restricted": 100 * tot[224:].max() / 4 / f["fri"].shape[1],
                   "cf_best_any": 100 * f["cri"].astype(np.int64).sum(1).max() / 4 / f["fri"].shape[1],
                   "cf_best_k13": 100 * f["cri"][:224].astype(np.int64).sum(1).max() / 4 / f["fri"].shape[1]}
        # reader statistics
        P = {c: f[f"P__{c}"] for c in Q.CONDS}
        pk = {c: f[f"pick__{c}"] for c in Q.CONDS}
        told = {"a": np.array([[0, 0, 1][i] for i in pi]), "b": np.array([[1, 1, 1][i] for i in pi])}
        tv = 0.5 * np.abs(P["a"] - P["b"]).sum(1)
        emo = pi < 2
        rows = np.arange(len(pi))
        con = 0.5 * ((P["a"][rows, told["a"]] - P["b"][rows, told["a"]]) + (P["b"][rows, told["b"]] - P["a"][rows, told["b"]]))
        corr = {}
        for d in Q.DIRS:
            a = zT["a"][d].numpy().astype(np.float64)
            b = zT["b"][d].numpy().astype(np.float64)
            a = a - a.mean(1, keepdims=True)
            b = b - b.mean(1, keepdims=True)
            den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
            ok = den > 0
            corr[d] = float(np.mean((a * b).sum(1)[ok] / den[ok]))
        tau2 = taus[2]
        op = np.concatenate([f["margin__a"] >= tau2, f["margin__b"] >= tau2])
        cc = np.concatenate([pk["a"] == told["a"], pk["b"] == told["b"]])
        shares = {c: [100 * float(np.mean(pk[c] == j)) for j in range(3)] for c in Q.CONDS}
        shares["all"] = [100 * float(np.mean(np.concatenate([pk["a"], pk["b"]]) == j)) for j in range(3)]
        t7[r]["reader"] = {"top_mean": float(np.mean(np.concatenate([P["a"].max(1), P["b"].max(1)]))),
                           "tv_mean": float(tv.mean()), "tv_emotion": float(tv[emo].mean()),
                           "told_contrast_emotion": float(con[emo].mean()), "corr_zTa_zTb": corr,
                           "acc_gate_open_tau2": 100 * float(cc[op].mean()),
                           "acc_gate_closed_tau2": 100 * float(cc[~op].mean()),
                           "pick_share": shares}
    out["table7"], out["curve"], out["in_sample_best"] = t7, curve, best
    # rank analysis at R1's assembled fused scores
    f = F["R1"]
    zT, gated, G = gated_terms(f, f["taus"].tolist())
    S = Q.assembled_scores({"zB": zB, "pos": pos}, gated, {0: 116, 1: 119}, parity)
    tot, hits, fo3, hit_fo3, tgt_o3 = 0, 0, 0, 0, 0
    flips = {k: [0, 0, 0] for k in (5, 3, 2)}
    all13 = np.ones((len(parity), 13), bool)
    for c, col in (("a", 0), ("b", 1)):
        for d in Q.DIRS:
            s = S[c][d]
            hit = Q.first(s, col, all13).astype(bool)
            fpos = pos[d][np.arange(len(s)), s.argmax(1)]
            tot += len(s)
            hits += int(hit.sum())
            fo3 += int((fpos >= 3).sum())
            hit_fo3 += int((hit & (fpos >= 3)).sum())
            tgt_o3 += int((pos[d][:, col] >= 3).sum())
            for k in flips:
                hk = Q.first(s, col, pos[d] < k).astype(bool)
                flips[k][0] += int((hk & ~hit).sum())
                flips[k][1] += int((hit & ~hk).sum())
                flips[k][2] += int((hk & ~hit & (fpos >= 3)).sum()) if k == 3 else 0
    out["ranks_R1"] = {"n": tot, "hit_share": 100 * hits / tot, "first_outside_top3": fo3,
                       "first_outside_top3_share": 100 * fo3 / tot, "hit_share_first_outside_top3": 100 * hit_fo3 / fo3,
                       "target_outside_top3_share": 100 * tgt_o3 / tot,
                       "flips": {k: {"miss_to_hit": v[0], "hit_to_miss": v[1], "net": v[0] - v[1],
                                     "net_pp": 100 * (v[0] - v[1]) / tot,
                                     "miss_to_hit_within_first_outside_top3": v[2]} for k, v in flips.items()}}
    # A1: R1/A1 against round 1's R-b expected on A1, and A1 - A0
    a1 = fam("R1_A1")
    rb1 = np.load(R1RES / "cand_Rb_expected_A1.npz")
    out["A1"] = {"fused_equal_round1_Rb_expected_A1": {m: bool(np.array_equal(a1[f"f_{m}"], rb1[f"fused__{m}"]))
                                                       for m in ("r1", "gain", "other")},
                 "cf_minus_round1_cf": Q.ci(a1["c_r1"] - rb1["cf__r1"], cl),
                 "round1_A1_bar": json.loads((R1RES / "cand_Rb_expected_A1.json").read_text())["bar"]["r1"],
                 "A1_minus_A0_fused": Q.ci(a1["f_r1"] - F["R1"]["f_r1"], cl),
                 "A1_minus_A0_bar": Q.ci(a1["bar_v"] - F["R1"]["bar_v"], cl),
                 "cf_A1_minus_cf_A0_mean": 100 * float(np.mean(a1["c_r1"]) - np.mean(F["R1"]["c_r1"]))}
    # R2 shift report summaries
    c2 = cands["R2_A0"]
    names = [f"{h}__{x}" for h in ("affect", "image", "caption")
             for x in ("S", "C", "Delta", "sd_support", "sd_contrast", "argmax_match")]
    out["R2_shift"] = {j: {n: (round(c2["shift"][j]["offset"][i], 3), round(c2["shift"][j]["ratio"][i], 3))
                           for i, n in enumerate(names)} for j in ("0", "1")}
    out["R2_top_prob"], out["R2_picks_differ"] = c2["top_prob"], c2["picks_differ_from_R1"]
    # ranking counts behind the bar margin
    out["ranking_counts"] = {r: float(np.sum(F[r]["bar_v"]) * 4) for r in F}
    out["ranking_counts"]["Rc"] = float(np.sum(rc["bar_v"]) * 4)
    (Q.OUT / "fr_diag.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:20000])


if __name__ == "__main__":
    main()
