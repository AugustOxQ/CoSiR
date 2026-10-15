"""Spike: R1's reader on A1c = (affect, image, csd) vs A0 and A1 with one fusion family (224 cells; 896 for completeness),
plus the told ceiling through the same fusion. CPU only; exploratory; writes only results/.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261119_no_caption_csd_spike/sp_run.py
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261118_reader_fix_round2"))
import r2_common as R  # noqa: E402
import run_r2_fusion as RF  # noqa: E402

C, K, rb, rbe, rf = R.C, R.K, R.rb, R.rbe, R.rf
from src.eval.aspect_metrics import CONDITIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores  # noqa: E402

RES = HERE / "results"
A1C = ("affect", "image", "csd")
RC_BAR = 0.4435221354166667


def bprime(bundle, parts):
    """B' exactly as common.load_bundle builds it: uniform_probe_scores over the configuration's groupings (step-1 names),
    then crossfit_condition_free(ctx.cos, t_n1u, t6u, parity)."""
    ctx = bundle.ctx
    sp = {C.STEP1_NAME[h]: bundle.post[h] for h in parts}
    t6u = uniform_probe_scores(sp, ctx.pooled, tuple(C.STEP1_NAME[h] for h in parts))
    Bp, _ = crossfit_condition_free(ctx.cos, bundle.t_n1u, t6u, ctx.parity)
    return Bp


def same_scores(x, y):
    return all(np.array_equal(np.asarray(x[c][d]), np.asarray(y[c][d])) for c in CONDITIONS for d in x[c])


def onehot_probs(bundle, config):
    parts = C.CONFIGS[config]
    ti = C.told_index(parts, C.TOLD[config], bundle.ctx.pair_index)
    return {c: np.eye(len(parts))[ti[c]] for c in CONDITIONS}


def a1c_probs(bundle):
    pk = pickle.load(open(RES / "sp_reader_A1c.pkl", "rb"))
    F42, feat_checks = rbe.seed42_features(bundle, A1C)
    per_half = {c: rbe.half_reader_probs(pk, F42[c], 3) for c in CONDITIONS}
    P = {c: rf.average_probs(per_half[c]) for c in CONDITIONS}
    assert all(np.allclose(P[c].sum(axis=1), 1.0, rtol=0, atol=1e-12) for c in CONDITIONS)
    rec = json.loads((RES / "sp_reader_A1c.json").read_text())
    return P, {"source": "sp_reader_A1c.pkl", "feature_checks": feat_checks,
               "half_readers": {j: {"chosen_C": rec["halves"][j]["chosen_C"],
                                    "oof_bank_accuracy": rec["halves"][j]["oof_bank_accuracy"]} for j in ("0", "1")}}


def pair_means(bundle, config, pn, pc):
    ctx = bundle.ctx
    pBp, pB = bundle.pBp[config], bundle.pB
    out = {}
    for i, p in enumerate(C.POOLED_ORDER):
        m = ctx.pair_index == i
        out[p] = {k: 100 * float(np.mean(np.asarray(v["r1"])[m])) for k, v in
                  (("fused", pn), ("counterpart", pc), ("B", pB), ("B_prime", pBp))}
    out["pooled"] = {k: 100 * float(np.mean(np.asarray(v["r1"]))) for k, v in
                     (("fused", pn), ("counterpart", pc), ("B", pB), ("B_prime", pBp))}
    return out


def run_one(bundle, config, kind, P, src, n_kappa, store):
    name = f"{kind}_{config}_{224 if n_kappa == 1 else 896}"
    ctx = bundle.ctx
    t0 = time.time()
    T, picks, margins = RF.terms_from_probs(bundle, config, P)
    taus, n_vals = K.thresholds(margins)
    res = RF.fusion_pass(bundle, T, margins, taus, n_kappa)
    pn, pc = per_anchor(res["fused"]), per_anchor(res["cfs"])
    ci = RF.cell_info(res, taus)
    summary, arr = C.evaluate_fused(bundle, config, pn, pc, picks, name, crossfit={"cells": ci})
    summary.update(kind=kind, n_cells=res["n_cells"], taus=taus, n_margins=n_vals, gate_open_share=RF.open_shares(res["g"], ctx),
                   probs_source=src, pair_r1_means=pair_means(bundle, config, pn, pc),
                   pair_gain_means={p: {k: 100 * float(np.mean(np.asarray(v["gain"])[ctx.pair_index == i]))
                                        for k, v in (("fused", pn), ("counterpart", pc))}
                                    for i, p in enumerate(C.POOLED_ORDER)},
                   pair_either_means={p: {k: 100 * float(np.mean(C.either(v)[ctx.pair_index == i]))
                                          for k, v in (("fused", pn), ("counterpart", pc))}
                                      for i, p in enumerate(C.POOLED_ORDER)})
    store[name] = {"vec": arr["vec"], "bar_v": arr["bar_v"], "config": config}
    (RES / f"sp_cand_{name}.json").write_text(json.dumps(C.jsonable(summary), indent=1))
    np.savez_compressed(RES / f"sp_cand_{name}.npz", fused_r1=arr["vec"]["fused_r1"], bar_v=arr["bar_v"],
                        margin_r1=arr["vec"]["margin_r1"], anchor_group=ctx.anchor_group, pair_index=ctx.pair_index)
    C.log(f"{name}: margin {summary['margin']['r1']['point']:+.4f}, bar {summary['bar']['r1']['point']:+.4f} "
          f"vs {summary['bar']['comparator']} [{time.time()-t0:.0f}s]")
    return summary


def main():
    R.assert_rule()
    R.assert_inputs(RF.CODE_INPUTS + tuple(RF.reader_files("A0")) + tuple(RF.reader_files("A1")))
    t0 = time.time()
    bundle = C.load_bundle(smoke=False)
    ctx, cl = bundle.ctx, bundle.cl
    C.CONFIGS["A1c"] = A1C                                   # runtime registration only; no file is touched
    C.TOLD["A1c"] = {"emotion": "affect", "style": "csd", "genre": "image"}
    checks = {}
    # ---- B' reproduction
    for a in ("A0", "A1"):
        Bp = bprime(bundle, C.CONFIGS[a])
        pBp = per_anchor(Bp)
        checks[f"Bprime_{a}_scores_equal_stored"] = bool(same_scores(Bp, bundle.Bp[a]))
        checks[f"Bprime_{a}_per_anchor_equal_stored"] = bool(all(np.array_equal(np.asarray(pBp[m]), np.asarray(bundle.pBp[a][m])) for m in METRICS))
        s1 = np.load(C.INPUT_FILES["step1_eval_style"])
        checks[f"Bprime_{a}_equals_step1_eval_style_npz"] = bool(all(np.array_equal(np.asarray(pBp[m]), s1[f"{a}__Bprime__{m}"]) for m in METRICS))
    Bp = bprime(bundle, A1C)
    bundle.Bp["A1c"], bundle.pBp["A1c"] = Bp, per_anchor(Bp)
    C.log(f"B' checks: {checks}")
    if not all(checks.values()):
        raise SystemExit("B' reproduction failed")
    pB, pBp = bundle.pB, bundle.pBp
    bprime_means = {a: 100 * float(np.mean(pBp[a]["r1"])) for a in ("A0", "A1", "A1c")}
    C.log(f"B R@1 {100*np.mean(pB['r1']):.3f}; B' means {bprime_means}")

    store, summaries = {}, {}
    P_cache = {}
    for config in ("A0", "A1"):
        P_cache[config] = RF.round1_probs(bundle, config, False)
    P_cache["A1c"] = a1c_probs(bundle)
    for config in ("A0", "A1", "A1c"):
        P, src = P_cache[config]
        summaries[f"R1_{config}_224"] = run_one(bundle, config, "R1", P, src, 1, store)
        if config == "A0":
            b = summaries["R1_A0_224"]["bar"]["r1"]["point"]
            checks["regression_R1_A0_224_bar_equals_0.4435221354166667_exactly"] = bool(b == RC_BAR)
            checks["regression_R1_A0_224_bar_ci_equals_round1"] = bool(summaries["R1_A0_224"]["bar"]["r1"]["ci95"] == RF.RC_BAR[1])
            C.log(f"REGRESSION bar point {b!r}: {b == RC_BAR}")
            if b != RC_BAR:
                print(json.dumps(checks, indent=1))
                raise SystemExit("regression failed")
    for config in ("A0", "A1", "A1c"):
        summaries[f"told_{config}_224"] = run_one(bundle, config, "told", onehot_probs(bundle, config),
                                                  {"source": "one-hot on the told grouping (evaluation-label oracle)"}, 1, store)
    # 896-cell version
    for config in ("A1c", "A0", "A1"):
        P, src = P_cache[config]
        summaries[f"R1_{config}_896"] = run_one(bundle, config, "R1", P, src, 4, store)
    # round-2 stored 896 values for A0 and A1 (cross-check of my 896 code path)
    for a in ("A0", "A1"):
        s = json.loads((R.HERE / f"results/cand_R1_{a}.json").read_text())
        mine = summaries[f"R1_{a}_896"]
        checks[f"R1_{a}_896_bar_equals_round2"] = bool(mine["bar"]["r1"]["point"] == s["bar"]["r1"]["point"])
        checks[f"R1_{a}_896_margin_equals_round2"] = bool(mine["margin"]["r1"]["point"] == s["margin"]["r1"]["point"])

    # ---- paired differences
    diffs = {}
    for fam in (224, 896):
        for other in ("A0", "A1"):
            x, y = store[f"R1_A1c_{fam}"], store[f"R1_{other}_{fam}"]
            d = {}
            for metric, key in (("fused_r1", None), ("bar_margin_r1", "bar_v"), ("margin_r1", None)):
                v = (x["vec"]["fused_r1"] - y["vec"]["fused_r1"]) if metric == "fused_r1" else \
                    (x["bar_v"] - y["bar_v"]) if metric == "bar_margin_r1" else (x["vec"]["margin_r1"] - y["vec"]["margin_r1"])
                d[metric] = {"pooled": C.point_ci(v, cl),
                             **{p: C.point_ci(v[ctx.pair_index == i], cl[ctx.pair_index == i]) for i, p in enumerate(C.POOLED_ORDER)}}
            diffs[f"A1c_minus_{other}_{fam}"] = d
    out = {"checks": checks, "diffs": diffs, "bprime_r1_means": bprime_means, "B_r1_mean": 100 * float(np.mean(pB["r1"])),
           "summaries": {k: {"r1_means": v["r1_means"], "margin": v["margin"], "bar": v["bar"], "gain_statistic": v["gain_statistic"],
                             "per_pair": v["per_pair"], "pick_accuracy": v["pick_accuracy"], "pick_share": v["pick_share"],
                             "gate_open_share": v["gate_open_share"], "pair_r1_means": v["pair_r1_means"],
                             "pair_gain_means": v["pair_gain_means"], "pair_either_means": v["pair_either_means"],
                             "cells": v["crossfit"]["cells"], "taus": v["taus"], "clears_bar": v["clears_bar"]}
                         for k, v in summaries.items()},
           "runtime_s": round(time.time() - t0, 1), "time_amsterdam": R.now_ams()}
    (RES / "sp_results.json").write_text(json.dumps(C.jsonable(out), indent=1))
    C.log(f"done [{time.time()-t0:.0f}s]; checks {checks}")


if __name__ == "__main__":
    main()
