"""Final review of reader-fix round 2: rebuild every candidate from probabilities (R1 and R2 from the reader pickles and
this review's own seed-42 features; R3 from this review's own retrained half-readers when fr_r3.py has run, and from
the stored R3 pickle), through T, gates, the 896 cells, the integer cross-fits, the assembly and the bootstrap, with
this review's own code (fr_core). Compares each number with the stored results (read only) and writes
out/fr_cands.json and out/fr_fam_<name>.npz (per-cell integer statistics for fr_diag.py).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/final_review/fr_cands.py [names...]
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import fr_core as Q  # noqa: E402

R2DIR = HERE.parent
RES = R2DIR / "results"
R1RES = ROOT / "src/test/20261117_reader_fix_csd/results"
PARTS = {"A0": ("affect", "image", "caption"), "A1": ("affect", "image", "caption", "csd")}
TOLD = {"A0": {"emotion": "affect", "style": "image", "genre": "image"},
        "A1": {"emotion": "affect", "style": "csd", "genre": "image"}}
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
PNAMES = ["emotion__style", "emotion__genre", "style__genre"]
ROWS = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def row(name, reported, mine, exact=True, tol=0.0):
    if isinstance(reported, (list, tuple)) and not isinstance(reported, str):
        ok = len(reported) == len(mine) and all(
            (a == b) if exact else abs(a - b) <= tol for a, b in zip(reported, mine))
    elif isinstance(reported, (bool, str, int)) or reported is None:
        ok = reported == mine
    else:
        ok = (reported == mine) if exact else abs(reported - mine) <= tol
    ROWS.append({"quantity": name, "reported": reported, "rederived": mine, "agree": bool(ok)})
    if not ok:
        print(f"  DISAGREE {name}: stored {reported!r} vs mine {mine!r}", flush=True)
    return ok


def probs(halves, F, stats=None):
    out = {}
    for c in ("a", "b"):
        ps = []
        for j, h in enumerate(halves):
            X = np.asarray(F[c], np.float64)
            if stats is None:
                Xs = h["scaler"].transform(X)
            else:
                m, s = stats[j]
                Xs = (X - m) / s
            ps.append(h["model"].predict_proba(Xs))
        out[c] = (ps[0] + ps[1]) / 2.0
    return out


def em(P, H, tol=1e-10, cap=10_000):
    pt = np.full(H, 1.0 / H)
    pi = pt.copy()
    s = 0
    while True:
        w = P * pi / pt                      # rule 4.3 b, left to right: P(h) pi(h) / pi_train(h)
        Pp = w / w.sum(axis=1, keepdims=True)
        new = Pp.mean(axis=0)
        if np.max(np.abs(new - pi)) < tol:
            return new, s + 1, False
        if s + 1 == cap:
            return new, s + 1, True
        pi = new
        s += 1


def adapt(P, pi_hat, H):
    w = P * pi_hat / np.full(H, 1.0 / H)
    return w / w.sum(axis=1, keepdims=True)


def told_idx(cfg, pair_index):
    parts = PARTS[cfg]
    return {c: np.array([parts.index(TOLD[cfg][PAIRS[i][j]]) for i in pair_index]) for j, c in enumerate(("a", "b"))}


def run(name, cfg, P, cache, stored_name=None, rc_regression=False):
    t0 = time.time()
    stored_name = stored_name or name
    rec = json.loads((RES / f"cand_{stored_name}.json").read_text())
    z = np.load(RES / f"cand_{stored_name}.npz")
    parity, cl, pi = cache["parity"], cache["cl"], cache["pair_index"]
    parts = PARTS[cfg]
    for c in ("a", "b"):
        row(f"{name}: probs__{c} bit-identical", True, bool(np.array_equal(P[c], z[f"probs__{c}"])))
    T, picks, margins = Q.terms(P, cache, parts)
    for c in ("a", "b"):
        for d in Q.DIRS:
            row(f"{name}: T__{c}__{d} bit-identical", True, bool(np.array_equal(T[c][d], z[f"T__{c}__{d}"])))
        row(f"{name}: pick__{c} identical", True, bool(np.array_equal(picks[c], z[f"pick__{c}"].astype(np.int64))))
        row(f"{name}: margin__{c} bit-identical", True, bool(np.array_equal(margins[c], z[f"margin__{c}"])))
    taus = Q.taus_of(margins)
    stored_tau = json.loads((RES / f"tau_{stored_name}.json").read_text())["taus"]
    row(f"{name}: tau_0..3", stored_tau, taus)
    fam = Q.family(cache, T, margins, taus)
    ctrl = Q.control(cache, fam["zB"])
    cc = rec["crossfit"]["cells"]
    for h in (0, 1):
        row(f"{name}: control sigma* half {h}", cc["control"][str(h)]["sigma"], ctrl[h][0])
        row(f"{name}: rho_ctrl half {h}", cc["control"][str(h)]["rho_ctrl_int"], ctrl[h][1])
    fp, cp, crit = Q.pick(fam, ctrl, parity)
    for h in (0, 1):
        row(f"{name}: fused cell, tune half {h}", cc["fused"][str(h)]["cell"], fp[h])
        row(f"{name}: counterpart cell, tune half {h}", cc["counterpart"][str(h)]["cell"], cp[h])
        ic = cc["integer_criteria"]
        row(f"{name}: fused rho/gamma tune half {h}", [ic["fused"][str(h)]["rho"], ic["fused"][str(h)]["gamma"]],
            [crit[h]["fused_rho"], crit[h]["fused_gamma"]])
        row(f"{name}: counterpart rho tune half {h}", ic["counterpart"][str(h)]["rho"], crit[h]["cf_rho"])
    # per-episode arrays, two routes: from the cell matrices and from the assembled restricted scores
    f_r1, f_gain = Q.from_cells(fam["fri"], fp, parity), Q.from_cells(fam["fgi"], fp, parity)
    c_r1, c_gain = Q.from_cells(fam["cri"], cp, parity), np.zeros(len(parity))
    pn = Q.per_episode(Q.assembled_scores(fam, fam["gated"], fp, parity))
    pc = Q.per_episode(Q.assembled_scores(fam, fam["G"], cp, parity))
    row(f"{name}: fused r1 from cells == from assembled scores", True, bool(np.array_equal(f_r1, pn["r1"])))
    row(f"{name}: cf r1 from cells == from assembled scores", True, bool(np.array_equal(c_r1, pc["r1"])))
    for m in ("r1", "gain", "other", "swap", "strict"):
        row(f"{name}: fused__{m} per-anchor array", True, bool(np.array_equal(pn[m], z[f"fused__{m}"])))
        row(f"{name}: cf__{m} per-anchor array", True, bool(np.array_equal(pc[m], z[f"cf__{m}"])))
    D = Q.decision(cache, pn["r1"], pn["gain"], pc["r1"], pc["gain"], cfg)
    row(f"{name}: bar_v array", True, bool(np.array_equal(D["bar_v"], z["bar_v"])))
    row(f"{name}: fused R@1", rec["r1_means"]["fused"], D["fused"])
    row(f"{name}: counterpart R@1", rec["r1_means"]["counterpart"], D["cf"])
    row(f"{name}: bar comparator", rec["bar"]["comparator"], D["comparator"])
    row(f"{name}: bar margin point", rec["bar"]["r1"]["point"], D["bar"]["point"])
    row(f"{name}: bar margin ci95", rec["bar"]["r1"]["ci95"], D["bar"]["ci95"])
    row(f"{name}: gain statistic point", rec["gain_statistic"]["point"], D["gain_stat"]["point"])
    row(f"{name}: gain statistic ci95", rec["gain_statistic"]["ci95"], D["gain_stat"]["ci95"])
    row(f"{name}: margin point", rec["margin"]["r1"]["point"], D["margin"]["point"])
    row(f"{name}: margin ci95", rec["margin"]["r1"]["ci95"], D["margin"]["ci95"])
    cls = rec["clears_bar"]
    row(f"{name}: D12 clauses", [cls["clause1_bar_point_at_least_0.5"], cls["clause2_bar_lower_above_0"],
                                 cls["clause3_gain_statistic_lower_above_0"]], [bool(x) for x in D["clauses"]])
    # further measures of item 2
    ef, ec = pn["r1"] + pn["other"], pc["r1"] + pc["other"]
    eith = Q.ci(ef - ec, cl)
    row(f"{name}: either (fused - cf)", [rec["margin"]["either"]["point"]] + rec["margin"]["either"]["ci95"],
        [eith["point"]] + eith["ci95"])
    fvb = Q.ci(pn["r1"] - cache["pB__r1"], cl)
    row(f"{name}: fused - B R@1", [rec["fused_vs_B"]["r1"]["point"]] + rec["fused_vs_B"]["r1"]["ci95"],
        [fvb["point"]] + fvb["ci95"])
    fvbp = Q.ci(pn["r1"] - cache[f"pBp_{cfg}__r1"], cl)
    row(f"{name}: fused - B' R@1", [rec["fused_vs_Bprime"]["r1"]["point"]] + rec["fused_vs_Bprime"]["r1"]["ci95"],
        [fvbp["point"]] + fvbp["ci95"])
    for i, p in enumerate(PNAMES):
        m = pi == i
        bp = Q.ci(D["bar_v"][m], cl[m])
        sr = rec["bar"]["per_pair_r1"][p]
        row(f"{name}: per-pair bar margin {p}", [sr["point"]] + sr["ci95"], [bp["point"]] + bp["ci95"])
        comp = {"B_prime": cache[f"pBp_{cfg}__gain"], "counterpart": pc["gain"], "B": cache["pB__gain"]}[D["comparator"]]
        gp = Q.ci((pn["gain"] - comp)[m], cl[m])
        row(f"{name}: per-pair gain {p}", rec["bar"]["per_pair_gain"][p]["point"], gp["point"])
    # pick accuracy (D13) and gate open shares
    ti = told_idx(cfg, pi)
    per_ep = 0.5 * ((picks["a"] == ti["a"]).astype(float) + (picks["b"] == ti["b"]).astype(float))
    pa = Q.ci(per_ep, cl)
    sp = rec["pick_accuracy"]["correct_share"]
    row(f"{name}: pick accuracy", [sp["point"]] + sp["ci95"], [pa["point"]] + pa["ci95"])
    ppc = {p: {c: 100 * float(np.mean(picks[c][pi == i] == ti[c][pi == i])) for c in ("a", "b")}
           for i, p in enumerate(PNAMES)}
    row(f"{name}: pick accuracy per pair and condition",
        [rec["pick_accuracy"]["per_pair_condition"][p][c] for p in PNAMES for c in ("a", "b")],
        [ppc[p][c] for p in PNAMES for c in ("a", "b")])
    for t, tau in enumerate(taus):
        ga, gb = margins["a"] >= tau, margins["b"] >= tau
        sh = rec["gate_open_share"][f"tau_{t}"]
        row(f"{name}: gate open share tau_{t} (overall, a, b)", [sh["overall"], sh["a"], sh["b"]],
            [100 * float(np.mean(np.concatenate([ga, gb]))), 100 * float(ga.mean()), 100 * float(gb.mean())],
            exact=False, tol=1e-4)      # stored shares are float32 means
    if cfg == "A0":
        av = rec["vs_step1_argmax"]["bar_margin_r1"]
        vs = Q.ci(D["bar_v"] - cache["argmaxA0_bar_v"], cl)
        row(f"{name}: bar margin minus step-1 arg-max reader's", [av["point"]] + av["ci95"], [vs["point"]] + vs["ci95"])
    # k_top = 13 only (diagnostic)
    fp13, cp13, _ = Q.pick(fam, ctrl, parity, range(224))
    f13, c13 = Q.from_cells(fam["fri"], fp13, parity), Q.from_cells(fam["cri"], cp13, parity)
    g13 = Q.from_cells(fam["fgi"], fp13, parity)
    D13 = Q.decision(cache, f13, g13, c13, np.zeros(len(parity)), cfg)
    out = {"taus": taus, "ctrl": {h: list(ctrl[h]) for h in (0, 1)}, "fused_cells": fp, "cf_cells": cp, "crit": crit,
           "bar": D["bar"], "gain_stat": D["gain_stat"], "margin": D["margin"], "comparator": D["comparator"],
           "comparator_means": D["comparator_means"], "fused": D["fused"], "cf": D["cf"], "clauses": D["clauses"],
           "either": eith, "fused_vs_B": fvb, "pick_accuracy": pa, "pick_acc_pair_cond": ppc,
           "k13": {"fused_cells": fp13, "cf_cells": cp13, "bar": D13["bar"], "comparator": D13["comparator"],
                   "fused": D13["fused"], "cf": D13["cf"], "gain_stat": D13["gain_stat"]},
           "topk_effect_bar": Q.ci(D["bar_v"] - D13["bar_v"], cl)}
    np.savez_compressed(Q.OUT / f"fr_fam_{name}.npz", fri=fam["fri"], fgi=fam["fgi"], cri=fam["cri"],
                        f_r1=pn["r1"], f_gain=pn["gain"], f_other=pn["other"], c_r1=pc["r1"], c_gain=pc["gain"],
                        c_other=pc["other"], bar_v=D["bar_v"], taus=np.array(taus),
                        **{f"P__{c}": P[c] for c in ("a", "b")}, **{f"pick__{c}": picks[c] for c in ("a", "b")},
                        **{f"margin__{c}": margins[c] for c in ("a", "b")},
                        **{f"T__{c}__{d}": T[c][d] for c in ("a", "b") for d in Q.DIRS})
    if rc_regression:
        out["regression"] = regression(fam, ctrl, cache, T, picks, margins, taus)
    print(f"{name}: bar {D['bar']['point']:+.4f} [{D['bar']['ci95'][0]:+.3f}, {D['bar']['ci95'][1]:+.3f}] "
          f"({D['comparator']}), gain {D['gain_stat']['point']:+.3f} [{D['gain_stat']['ci95'][0]:+.3f}, "
          f"{D['gain_stat']['ci95'][1]:+.3f}], clauses {D['clauses']}, cells {fp} / {cp} [{time.time() - t0:.0f}s]",
          flush=True)
    return out


def regression(fam, ctrl, cache, T, picks, margins, taus):
    """Rule 4.7: R1 restricted to cells 0 to 223 reproduces round-1 R-c."""
    rc = np.load(R1RES / "cand_Rc_Rb_expected_A0.npz")
    rcj = json.loads((R1RES / "cand_Rc_Rb_expected_A0.json").read_text())
    rct = json.loads((R1RES / "rc_tau.json").read_text())["taus"]
    parity = cache["parity"]
    out = {}
    for c in ("a", "b"):
        for d in Q.DIRS:
            row(f"regression: T__{c}__{d}", True, bool(np.array_equal(T[c][d], rc[f"T__{c}__{d}"])))
        row(f"regression: pick__{c}", True, bool(np.array_equal(picks[c], rc[f"pick__{c}"].astype(np.int64))))
        row(f"regression: margin__{c}", True, bool(np.array_equal(margins[c], rc[f"margin__{c}"])))
    row("regression: taus == rc_tau.json", rct, taus)
    fp, cp, _ = Q.pick(fam, ctrl, parity, range(224))
    row("regression: fused cells 116/119", [116, 119], [fp[0], fp[1]])
    row("regression: cf cells 58/123", [58, 123], [cp[0], cp[1]])
    row("regression: sigma* 0/0", [0.0, 0.0], [ctrl[0][0], ctrl[1][0]])
    pn = Q.per_episode(Q.assembled_scores(fam, fam["gated"], fp, parity))
    pc = Q.per_episode(Q.assembled_scores(fam, fam["G"], cp, parity))
    for m in ("r1", "gain", "other", "swap", "strict"):
        row(f"regression: fused__{m}", True, bool(np.array_equal(pn[m], rc[f"fused__{m}"])))
        row(f"regression: cf__{m}", True, bool(np.array_equal(pc[m], rc[f"cf__{m}"])))
    D = Q.decision(cache, pn["r1"], pn["gain"], pc["r1"], pc["gain"], "A0")
    row("regression: bar_v", True, bool(np.array_equal(D["bar_v"], rc["bar_v"])))
    row("regression: bar margin", [rcj["bar"]["r1"]["point"]] + rcj["bar"]["r1"]["ci95"],
        [D["bar"]["point"]] + D["bar"]["ci95"])
    row("regression: bar margin == rule literal", [0.4435221354166667, 0.21646171563312194, 0.6735669710776852],
        [D["bar"]["point"]] + D["bar"]["ci95"])
    row("regression: gain statistic == rule literal", [2.667236328125, 2.325087836946873, 3.012361650695922],
        [D["gain_stat"]["point"]] + D["gain_stat"]["ci95"])
    out["fused_cells"], out["cf_cells"] = fp, cp
    return out


def main():
    names = sys.argv[1:] or ["R1_A0", "R2_A0", "R3_A0_stored", "R1_A1"]
    cache = Q.load_cache()
    res = {}
    F = {cfg: {c: cache[f"F_{cfg}__{c}"] for c in ("a", "b")} for cfg in ("A0", "A1")}
    pk = {}
    for cfg in ("A0", "A1"):
        with open(R1RES / f"rb_reader_{cfg}.pkl", "rb") as f:
            pk[cfg] = pickle.load(f)
    for cfg, want in (("A0", "6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c"),
                      ("A1", "4e1e4e23c20333b839aa1f92942891bdef51955317640d1014db5d097b7060f6")):
        assert sha(R1RES / f"rb_reader_{cfg}.pkl") == want
    if "R1_A0" in names:
        P1 = probs(pk["A0"]["halves"], F["A0"])
        res["R1_A0"] = run("R1_A0", "A0", P1, cache, rc_regression=True)
        # R1 top probability (round 1: 0.694)
        res["R1_A0"]["top_prob_mean"] = float(np.mean(np.concatenate([P1["a"].max(1), P1["b"].max(1)])))
    if "R2_A0" in names:
        halves = pk["A0"]["halves"]
        X = np.vstack([F["A0"]["a"], F["A0"]["b"]]).astype(np.float64)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        rec2 = json.loads((RES / "probs_R2_A0.json").read_text())
        row("R2: mu42 (18)", rec2["mu42"], mu.tolist())
        row("R2: sigma42 (18)", rec2["sigma42"], sd.tolist())
        row("R2: no sigma42 entry exactly 0", True, bool((sd != 0).all()))
        # code check: own scaler statistics, no EM, reproduces R1
        P1 = probs(halves, F["A0"])
        Pown = probs(halves, F["A0"], [(h["scaler"].mean_, h["scaler"].scale_) for h in halves])
        dmax = max(float(np.max(np.abs(P1[c] - Pown[c]))) for c in ("a", "b"))
        same = all(np.array_equal(P1[c].argmax(1), Pown[c].argmax(1)) for c in ("a", "b"))
        row("R2: code check max |diff| <= 1e-12 and identical picks", True, bool(dmax <= 1e-12 and same))
        Pr = probs(halves, F["A0"], [(mu, sd), (mu, sd)])
        pi_hat, n_iter, capped = em(np.vstack([Pr["a"], Pr["b"]]), 3)
        row("R2: pi_hat", rec2["pi_hat"], pi_hat.tolist())
        row("R2: EM iterations", rec2["em"]["n_iter"], n_iter)
        row("R2: EM cap reached", rec2["em"]["cap_reached"], capped)
        P2 = {c: adapt(Pr[c], pi_hat, 3) for c in ("a", "b")}
        res["R2_A0"] = run("R2_A0", "A0", P2, cache)
        top = lambda P: float(np.mean(np.concatenate([P["a"].max(1), P["b"].max(1)])))  # noqa: E731
        res["R2_A0"]["top_prob"] = {"R1": top(P1), "R2_before_EM": top(Pr), "R2": top(P2)}
        p1 = np.concatenate([P1[c].argmax(1) for c in ("a", "b")])
        res["R2_A0"]["picks_differ_from_R1"] = {
            "before_EM": 100 * float(np.mean(np.concatenate([Pr[c].argmax(1) for c in ("a", "b")]) != p1)),
            "after_EM": 100 * float(np.mean(np.concatenate([P2[c].argmax(1) for c in ("a", "b")]) != p1))}
        sh = {}
        for j, h in enumerate(halves):
            m, s = h["scaler"].mean_, h["scaler"].scale_
            sh[j] = {"offset": ((mu - m) / s).tolist(), "ratio": (sd / s).tolist()}
        res["R2_A0"]["shift"] = sh
        res["R2_A0"]["mu42"], res["R2_A0"]["sigma42"], res["R2_A0"]["pi_hat"] = mu.tolist(), sd.tolist(), pi_hat.tolist()
    if "R3_A0_stored" in names:
        with open(RES / "r3_reader_A0.pkl", "rb") as f:
            p3 = pickle.load(f)
        P3 = probs(p3["halves"], F["A0"])
        res["R3_A0_stored"] = run("R3_A0_stored", "A0", P3, cache, stored_name="R3_A0")
    if "R3_A0_mine" in names:
        z3 = np.load(Q.OUT / "fr_r3_probs.npz")
        P3 = {c: z3[f"P__{c}"] for c in ("a", "b")}
        res["R3_A0_mine"] = run("R3_A0_mine", "A0", P3, cache, stored_name="R3_A0")
    if "R1_A1" in names:
        P1a = probs(pk["A1"]["halves"], F["A1"])
        res["R1_A1"] = run("R1_A1", "A1", P1a, cache)
    tag = "_".join(names)
    (Q.OUT / f"fr_cands_{tag}.json").write_text(json.dumps({"results": res, "rows": ROWS}, indent=1, default=float))
    n_bad = sum(not r["agree"] for r in ROWS)
    print(f"{len(ROWS)} comparisons, {n_bad} disagreements", flush=True)


if __name__ == "__main__":
    main()
