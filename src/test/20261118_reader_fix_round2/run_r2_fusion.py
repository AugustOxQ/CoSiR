"""Round 2's fusion stream (DECISION_RULE.md sections 4.5 to 4.7): one reader's seed-42 probabilities -> gated, top-k
restricted 896-cell fusion family, its matched counterpart and exact integer cross-fits, on one configuration.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/run_r2_fusion.py \
        --reader {R1,R2,R3} --config {A0,A1} [--smoke]
    ... --regression        R1, A0, cells 0 to 223 only: rule 4.7 against round-1 R-c -> results/regression_check.json
    ... --regression-dry    the same computation on the real data, written to results/smoke/regression_check_dry.json

R1's probabilities are computed here exactly as round 1's rb_eval.stage_config does (round 1's two half-readers). R2 and
R3 read results/probs_<R>_<config>.{npz,json} written by the reader stream (keys P__a, P__b). Writes
results/tau_<R>_<config>.json before any score, then results/cand_<R>_<config>.{json,npz,txt}. Non-smoke outputs are
never overwritten. A non-smoke candidate run refuses to start unless results/regression_check.json says passed: true.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_common as R  # noqa: E402
import r2_fusion as F  # noqa: E402

C, K, rb, rbe, rf = R.C, R.K, R.rb, R.rbe, R.rf
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_nested import _zdict  # noqa: E402

CODE_INPUTS = ("common.py", "rc_core.py", "rb_build.py", "rb_eval.py", "rb_features.py")
RC_FILES = ("results/cand_Rc_Rb_expected_A0.npz", "results/cand_Rc_Rb_expected_A0.json", "results/rc_tau.json")
# rule 4.7 literals (round-1 R-c)
RC_TAUS = [3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211]
RC_BAR = (0.4435221354166667, [0.21646171563312194, 0.6735669710776852])
RC_GAIN = (2.667236328125, [2.325087836946873, 3.012361650695922])
RC_FUSED_CELLS, RC_CF_CELLS = {0: 116, 1: 119}, {0: 58, 1: 123}


def reader_files(config):
    return [f"results/rb_reader_{config}.{e}" for e in ("pkl", "json", "npz")]


# ---------------------------------------------------------------- reader probabilities

def probs_paths(reader, config, smoke):
    d = R.res_dir(smoke)
    return d / f"probs_{reader}_{config}.npz", d / f"probs_{reader}_{config}.json"


def load_stream_probs(reader, config, smoke, n_episodes, n_groupings):
    """R2 / R3 probabilities from the reader stream: json record and npz SHA-256 checked, keys P__a, P__b."""
    pz, pj = probs_paths(reader, config, smoke)
    if not pj.exists():
        raise SystemExit(f"{pj} is missing: the reader stream has not written {reader} on {config}")
    rec = json.loads(pj.read_text())
    if rec.get("provenance", {}).get("rule_sha256") != R.RULE_SHA:
        raise SystemExit(f"{pj} was written under another rule")
    if rec.get("r3_is_r1"):
        raise SystemExit(f"{reader} on {config} is R1 (k* = 4; rule 4.4 h): it is not evaluated a second time")
    if bool(rec.get("smoke")) != bool(smoke) or rec.get("reader") != reader or rec.get("config") != config:
        raise SystemExit(f"{pj}: other reader, configuration or smoke flag")
    if rec.get("groupings") != list(C.CONFIGS[config]):
        raise SystemExit(f"{pj}: groupings {rec.get('groupings')} differ from the configuration's")
    if rec.get("n_episodes") != n_episodes:
        raise SystemExit(f"{pj}: {rec.get('n_episodes')} episodes, this bundle has {n_episodes}")
    if not pz.exists() or R.sha_file(pz) != rec["npz_sha256"]:
        raise SystemExit(f"{pz} is missing or differs from its json (SHA-256)")
    z = np.load(pz)
    P = {c: np.asarray(z[f"P__{c}"]) for c in CONDITIONS}
    for c in CONDITIONS:
        if P[c].dtype != np.float64 or P[c].shape != (n_episodes, n_groupings):
            raise SystemExit(f"{pz}: P__{c} has dtype {P[c].dtype} and shape {P[c].shape}")
        if not np.isfinite(P[c]).all() or not np.allclose(P[c].sum(axis=1), 1.0, rtol=0, atol=1e-9):
            raise SystemExit(f"{pz}: P__{c} is not a probability array")
    return P, {"source": "reader stream", "probs_json": pj.name, "probs_npz_sha256": rec["npz_sha256"],
               "reader_record": {k: rec[k] for k in rec if k not in ("provenance",)}}


def round1_probs(bundle, config, smoke):
    """R1 exactly as rb_eval.stage_config: the mean of round 1's two half-readers on the seed-42 features."""
    parts = C.CONFIGS[config]
    pk, rrec, _ = rb.load_readers(config, smoke)
    if pk["feature_names"] != rf.feature_names(parts):
        raise AssertionError("the readers were trained on another feature layout")
    F42, feat_checks = rbe.seed42_features(bundle, parts)
    per_half = {c: rbe.half_reader_probs(pk, F42[c], len(parts)) for c in CONDITIONS}
    P = {c: rf.average_probs(per_half[c]) for c in CONDITIONS}
    if not all(np.allclose(P[c].sum(axis=1), 1.0, rtol=0, atol=1e-12) for c in CONDITIONS):
        raise AssertionError("averaged probabilities do not sum to 1")
    return P, {"source": "round 1's half-readers (rb_reader_%s.pkl)" % config, "feature_checks": feat_checks,
               "half_readers": {j: {"chosen_C": rrec["halves"][j]["chosen_C"],
                                    "oof_bank_accuracy": rrec["halves"][j]["oof_accuracy_at_chosen_C"]}
                                for j in ("0", "1")}}


def terms_from_probs(bundle, config, P):
    """T^c = sum_h P^c(h) s_h, picks (arg max, ties to the first grouping), top-two margins."""
    parts = C.CONFIGS[config]
    pm = {c: rf.picks_and_margins(P[c]) for c in CONDITIONS}
    picks = {c: pm[c][0] for c in CONDITIONS}
    margins = {c: pm[c][1] for c in CONDITIONS}
    if not all(np.array_equal(margins[c], C.top_two_margin(P[c])) for c in CONDITIONS):
        raise AssertionError("top-two margin differs from common.top_two_margin")
    stack = C.grouping_stack(bundle.post, bundle.ctx.pooled, parts)
    return C.expected_term(stack, P), picks, margins


# ---------------------------------------------------------------- the fusion pass

def fusion_pass(bundle, T, margins, taus, n_kappa=len(F.KTOPS)):
    """Gates, restricted cells, integer cross-fits and assembled scores of one reader (n_kappa = 1: cells 0 to 223)."""
    ctx = bundle.ctx
    zB, zT = _zdict(bundle.B), _zdict(T)
    info = F.rank_info(bundle.B)
    g = K.gates(margins, taus)
    gated = {t: K.gated_terms(zT, g[t]) for t in g}
    G = {t: K.g_cf(gated[t]) for t in g}
    t0 = time.time()
    fri, fgi, cri = F.cell_statistics(zB, info, gated, G, n_kappa)
    C.log(f"{len(fri)} cells scored [{time.time() - t0:.0f}s]")
    ctrl = F.control_choice(zB, ctx.parity)
    fpick = F.select_fused(fri, fgi, ctrl, ctx.parity)
    cpick = F.select_cf(cri, ctx.parity)
    fused = F.assemble(zB, info, gated, fpick, ctx.parity)
    cfs = F.assemble(zB, info, G, cpick, ctx.parity)
    C._assert_finite(fused, "fused"), C._assert_finite(cfs, "counterpart")
    return {"g": g, "ctrl": ctrl, "fpick": fpick, "cpick": cpick, "fused": fused, "cfs": cfs,
            "details": F.pick_details(fri, fgi, cri, ctrl, ctx.parity, fpick, cpick), "n_cells": len(fri)}


def open_shares(g, ctx):
    return {f"tau_{k}": {"overall": 100 * float(np.mean(np.concatenate([g[k]["a"], g[k]["b"]]))),
                         "a": 100 * float(g[k]["a"].mean()), "b": 100 * float(g[k]["b"].mean()),
                         "per_pair": {p: 100 * float(np.mean(np.concatenate([g[k]["a"][ctx.pair_index == i],
                                                                             g[k]["b"][ctx.pair_index == i]])))
                                      for i, p in enumerate(C.POOLED_ORDER)}} for k in g}


def cell_info(res, taus):
    return {"fused": {str(h): F.describe_cell(i, taus) for h, i in res["fpick"].items()},
            "counterpart": {str(h): F.describe_cell(i, taus) for h, i in res["cpick"].items()},
            "control": {str(h): {"sigma": s, "rho_ctrl_int": r} for h, (s, r) in res["ctrl"].items()},
            "integer_criteria": res["details"],
            "numbering": "((kappa * 4 + t) * 7 + u) * 8 + a, k_top order " + str(list(F.KTOPS))}


# ---------------------------------------------------------------- output

def check_inputs(reader, config, smoke):
    names = list(CODE_INPUTS)
    if reader == "R1" and not smoke:
        names += reader_files(config)
    return R.assert_inputs(names)


def regression_gate(smoke):
    p = R.RES / "regression_check.json"
    if smoke:
        return
    if not p.exists():
        raise SystemExit(f"{p} is missing: run --regression first (no candidate number before it passes; rule 4.7)")
    rec = json.loads(p.read_text())
    if rec.get("passed") is not True or rec.get("provenance", {}).get("rule_sha256") != R.RULE_SHA:
        raise SystemExit(f"{p}: the regression check did not pass; stop and report to the user (rule section 8)")


def run_candidate(args):
    reader, config, smoke = args.reader, args.config, args.smoke
    name = f"{reader}_{config}"
    R.assert_rule()
    inputs = check_inputs(reader, config, smoke)
    out = R.res_dir(smoke)
    p = {"tau": out / f"tau_{name}.json", **{e: out / f"cand_{name}.{e}" for e in ("json", "npz", "txt")}}
    R.refuse_existing(p.values(), smoke)
    if reader == "R3":                                       # refuse early when R3 is R1
        pj = probs_paths("R3", config, smoke)[1]
        if pj.exists() and json.loads(pj.read_text()).get("r3_is_r1"):
            raise SystemExit(f"R3 on {config} is R1 (k* = 4; rule 4.4 h): not evaluated a second time")
    regression_gate(smoke)
    t_start = time.time()
    bundle = C.load_bundle(smoke=smoke)
    ctx, cl = bundle.ctx, bundle.cl
    H = len(C.CONFIGS[config])
    if reader == "R1":
        P, src = round1_probs(bundle, config, smoke)
    else:
        P, src = load_stream_probs(reader, config, smoke, ctx.n, H)
    T, picks, margins = terms_from_probs(bundle, config, P)

    # ---- thresholds, written before any score
    taus, n_vals = K.thresholds(margins)
    if not smoke and n_vals != 24_576:
        raise AssertionError(f"expected 24,576 (episode, condition) margins, got {n_vals}")
    R.write_json_once(p["tau"], {"reader": reader, "config": config, "smoke": bool(smoke), "taus": taus,
                                 "percentiles": list(K.PCTS), "n_values": n_vals,
                                 "method": "numpy.percentile, linear, of the reader's own seed-42 top-two margins "
                                           "(condition a's values then condition b's)",
                                 "probs_source": src.get("probs_npz_sha256", src["source"])}, smoke)
    if json.loads(p["tau"].read_text())["taus"] != taus:
        raise AssertionError("tau did not round-trip at full precision")
    C.log(f"tau_0..3 written: {taus}")

    res = fusion_pass(bundle, T, margins, taus)
    pn, pc = per_anchor(res["fused"]), per_anchor(res["cfs"])
    shares = open_shares(res["g"], ctx)
    ci = cell_info(res, taus)
    summary, arr = C.evaluate_fused(bundle, config, pn, pc, picks, name, crossfit={"cells": ci})
    summary.update(reader=reader, smoke=bool(smoke), taus=taus, gate_open_share=shares, n_cells=res["n_cells"],
                   probs_source=src, inputs_sha256={**inputs, **bundle.input_sha256},
                   code_sha256={f: R.sha_file(Path(__file__).parent / f) for f in ("r2_fusion.py", "run_r2_fusion.py")})
    save_candidate(p, summary, arr, T, picks, margins, P, res, taus, ctx, smoke)
    C.log(f"done [{time.time() - t_start:.0f}s]")


def save_candidate(p, summary, arr, T, picks, margins, P, res, taus, ctx, smoke):
    npz = {}
    for part in ("fused", "cf"):
        for m in METRICS:
            npz[f"{part}__{m}"] = np.asarray(arr[part][m])
    npz["bar_v"] = np.asarray(arr["bar_v"])
    for c in CONDITIONS:
        for d in DIRECTIONS:
            npz[f"T__{c}__{d}"] = np.asarray(T[c][d], dtype=np.float32)
        npz[f"pick__{c}"] = np.asarray(picks[c]).astype(np.int8)
        npz[f"margin__{c}"] = np.asarray(margins[c], dtype=np.float64)
        npz[f"probs__{c}"] = np.asarray(P[c], dtype=np.float64)
        npz[f"gate__{c}"] = np.stack([res["g"][t][c] for t in sorted(res["g"])])
    npz["taus"] = np.array(taus)
    npz["anchor_group"], npz["pair_index"], npz["parity"] = (np.asarray(ctx.anchor_group), np.asarray(ctx.pair_index),
                                                             np.asarray(ctx.parity))
    npz["fused_cells"] = np.array([res["fpick"][0], res["fpick"][1]])
    npz["cf_cells"] = np.array([res["cpick"][0], res["cpick"][1]])
    npz["ctrl_sigma"] = np.array([res["ctrl"][0][0], res["ctrl"][1][0]])
    np.savez_compressed(p["npz"], **npz)
    rec = R.write_json_once(p["json"], {**summary, "npz_sha256": R.sha_file(p["npz"])}, smoke)
    rec = json.loads(p["json"].read_text())
    txt = C.summary_text(rec) + "\n" + cells_text(rec)
    p["txt"].write_text(txt + "\n")
    print(txt, flush=True)


def cells_text(rec):
    cf = rec["crossfit"]["cells"]
    L = []
    for who in ("fused", "counterpart"):
        L.append(f"  {who} cells: " + "; ".join(
            f"half {h}: cell {v['cell']} (k_top {v['k_top']}, tau_{v['tau_index']}={v['tau']:.4f}, "
            f"lambda_u {v['lambda_u']}, lambda_a {v['lambda_a']})" for h, v in cf[who].items()))
    L.append("  control sigma*: " + ", ".join(f"half {h} {v['sigma']}" for h, v in cf["control"].items()))
    L.append("  gate open share (%): " + ", ".join(f"{k} {v['overall']:.1f}" for k, v in rec["gate_open_share"].items()))
    return "\n".join(L)


# ---------------------------------------------------------------- regression check (rule 4.7)

def run_regression(args):
    """R1 on A0, cells 0 to 223 only. No number of cells 224 to 895 is computed, written or printed."""
    if args.reader != "R1" or args.config != "A0" or args.smoke:
        raise SystemExit("--regression is R1, A0, real data only")
    dry = args.regression_dry
    R.assert_rule()
    inputs = check_inputs("R1", "A0", False)
    inputs.update(R.assert_inputs(RC_FILES))
    dest = (R.res_dir(True) / "regression_check_dry.json") if dry else (R.RES / "regression_check.json")
    R.refuse_existing([dest], dry)
    t0 = time.time()
    bundle = C.load_bundle(smoke=False)
    ctx = bundle.ctx
    P, src = round1_probs(bundle, "A0", False)
    T, picks, margins = terms_from_probs(bundle, "A0", P)
    taus, n_vals = K.thresholds(margins)
    r1dir = R.R1 / "results"
    z = np.load(r1dir / "cand_Rc_Rb_expected_A0.npz")
    stored = json.loads((r1dir / "cand_Rc_Rb_expected_A0.json").read_text())
    stored_tau = json.loads((r1dir / "rc_tau.json").read_text())["taus"]
    res = fusion_pass(bundle, T, margins, taus, n_kappa=1)
    pn, pc = per_anchor(res["fused"]), per_anchor(res["cfs"])
    ci = cell_info(res, taus)
    summary, arr = C.evaluate_fused(bundle, "A0", pn, pc, None, "R1_regression", crossfit={"cells": ci})

    checks = []

    def add(name, ok, **detail):
        checks.append({"name": name, "ok": bool(ok), **detail})

    for c in CONDITIONS:
        for d in DIRECTIONS:
            add(f"T__{c}__{d} equals round 1's", np.array_equal(T[c][d], z[f"T__{c}__{d}"]),
                max_abs_diff=float(np.max(np.abs(T[c][d].astype(np.float64) - z[f"T__{c}__{d}"]))))
        add(f"margin__{c} equals round 1's", np.array_equal(margins[c], z[f"margin__{c}"]))
        add(f"pick__{c} equals round 1's", np.array_equal(picks[c].astype(np.int64), z[f"pick__{c}"].astype(np.int64)))
    add("tau_0..3 equal rc_tau.json exactly", taus == stored_tau, mine=taus, stored=stored_tau)
    add("tau_0..3 equal the rule's literals", taus == RC_TAUS, mine=taus, literals=RC_TAUS)
    add("n margins = 24,576", n_vals == 24_576)
    add("fused cells are 116 and 119", res["fpick"] == RC_FUSED_CELLS, mine=res["fpick"])
    add("counterpart cells are 58 and 123", res["cpick"] == RC_CF_CELLS, mine=res["cpick"])
    add("control sigma* is 0 on both halves", all(res["ctrl"][h][0] == 0.0 for h in (0, 1)),
        mine={h: res["ctrl"][h][0] for h in (0, 1)})
    sc = stored["crossfit"]["cells"]
    for who, picks_ in (("fused", res["fpick"]), ("counterpart", res["cpick"])):
        for h in (0, 1):
            mine = F.describe_cell(picks_[h], taus)
            s = sc[who][str(h)]
            add(f"{who} half {h}: (tau index, lambda_u, lambda_a) equal round 1's",
                (mine["tau_index"], mine["lambda_u"], mine["lambda_a"]) == (s["tau_index"], s["lambda_u"], s["lambda_a"]),
                mine=[mine["tau_index"], mine["lambda_u"], mine["lambda_a"]],
                stored=[s["tau_index"], s["lambda_u"], s["lambda_a"]])
    for part, pa in (("fused", pn), ("cf", pc)):
        for m in METRICS:
            add(f"{part}__{m} per-anchor array equals round 1's", np.array_equal(np.asarray(pa[m]), z[f"{part}__{m}"]),
                max_abs_diff=float(np.max(np.abs(np.asarray(pa[m]) - z[f"{part}__{m}"]))))
    add("bar_v equals round 1's", np.array_equal(arr["bar_v"], z["bar_v"]))
    bar, gs = summary["bar"]["r1"], summary["gain_statistic"]
    add("bar comparator is the counterpart", summary["bar"]["comparator"] == "counterpart",
        mine=summary["bar"]["comparator"])
    add("bar margin equals round 1's stored value, exactly",
        bar["point"] == stored["bar"]["r1"]["point"] and bar["ci95"] == stored["bar"]["r1"]["ci95"],
        mine=bar, stored=stored["bar"]["r1"])
    add("bar margin equals the rule's literals (0.4435221354166667 [0.21646171563312194, 0.6735669710776852])",
        bar["point"] == RC_BAR[0] and bar["ci95"] == RC_BAR[1], mine=bar)
    add("gain statistic equals round 1's stored value, exactly",
        gs["point"] == stored["gain_statistic"]["point"] and gs["ci95"] == stored["gain_statistic"]["ci95"],
        mine={"point": gs["point"], "ci95": gs["ci95"]}, stored={k: stored["gain_statistic"][k] for k in ("point", "ci95")})
    add("gain statistic equals the rule's literals (2.667236328125 [2.325087836946873, 3.012361650695922])",
        gs["point"] == RC_GAIN[0] and gs["ci95"] == RC_GAIN[1], mine={"point": gs["point"], "ci95": gs["ci95"]})
    passed = all(c["ok"] for c in checks)
    rec = {"passed": bool(passed), "dry_run": bool(dry), "n_comparisons": len(checks),
           "failed": [c["name"] for c in checks if not c["ok"]], "comparisons": checks, "cells_computed": "0 to 223 only",
           "inputs_sha256": inputs, "runtime_s": round(time.time() - t0, 1)}
    R.write_json_once(dest, rec, dry)
    print(f"regression check ({'dry run, ' if dry else ''}cells 0 to 223): {sum(c['ok'] for c in checks)}/{len(checks)} "
          f"comparisons exact; passed = {passed}")
    for c in checks:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['name']}")
    if not passed:
        raise SystemExit("the regression check failed: no candidate number may be written; report to the user")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", choices=("R1", "R2", "R3"), required=True)
    ap.add_argument("--config", choices=("A0", "A1"), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--regression-dry", action="store_true")
    args = ap.parse_args()
    if args.regression_dry:
        args.regression = True
    if args.regression:
        run_regression(args)
    else:
        run_candidate(args)


if __name__ == "__main__":
    main()
