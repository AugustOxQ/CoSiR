"""R-c, the confidence gate (DECISION_RULE.md section 4.3), on a parent candidate chosen by the main session.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261117_reader_fix_csd/run_rc.py --parent Ra_A1 [--smoke]

Reads results/cand_<parent>.{json,npz} (T, picks, top-two margins). Writes results/rc_tau.json (tau_0..tau_3, before
any R-c score) and results/cand_Rc_<parent>.{json,txt,npz}. Non-smoke outputs are never overwritten.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import rc_core as K  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS  # noqa: E402
from src.eval.aspect_nested import _combine, _zdict, nested_cells, nested_scores  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    C.assert_rule()
    name = f"Rc_{args.parent}"
    out = C.res_dir(args.smoke)
    if not args.smoke:
        if (out / "rc_tau.json").exists():
            raise SystemExit(f"{out / 'rc_tau.json'} exists; refusing to overwrite")
        if any(p.exists() for p in C._paths(name, False)[1].values()):
            raise SystemExit(f"candidate {name} exists; refusing to overwrite")
    prec, pz = C.load_candidate(args.parent, args.smoke)
    if bool(prec["smoke"]) != bool(args.smoke):
        raise SystemExit("the parent and this run must both be smoke or both real")
    config = prec["config"]
    bundle = C.load_bundle(smoke=args.smoke)
    ctx, cl = bundle.ctx, bundle.cl
    T = {c: {d: np.asarray(pz[f"T__{c}__{d}"]) for d in DIRECTIONS} for c in CONDITIONS}
    picks = {c: np.asarray(pz[f"pick__{c}"]).astype(np.int64) for c in CONDITIONS}
    margins = {c: np.asarray(pz[f"margin__{c}"]) for c in CONDITIONS}
    if len(margins["a"]) != ctx.n:
        raise SystemExit("the parent's episodes differ from this bundle's")

    # ---- thresholds, written before any R-c score
    taus, n_vals = K.thresholds(margins)
    if not args.smoke and n_vals != 24_576:
        raise AssertionError(f"expected 24,576 (episode, condition) margins, got {n_vals}")
    C.write_json_once(out / "rc_tau.json", {"parent": args.parent, "config": config, "taus": taus,
                                            "percentiles": list(K.PCTS), "n_values": n_vals,
                                            "method": "numpy.percentile, linear, of the parent's top-two margins",
                                            "parent_npz_sha256": prec["provenance"]["npz_sha256"]}, args.smoke)
    reread = json.loads((out / "rc_tau.json").read_text())["taus"]
    if reread != taus:
        raise AssertionError("tau did not round-trip at full precision")
    C.log(f"tau_0..3 written: {taus}" if not args.smoke else "tau written (smoke)")

    # ---- scores
    t0 = time.time()
    zB, zT = _zdict(bundle.B), _zdict(T)
    g = K.gates(margins, taus)
    gated = {k: K.gated_terms(zT, g[k]) for k in g}
    G = {k: K.g_cf(gated[k]) for k in g}
    cells = K.rc_cells()
    if len(cells) != 224:
        raise AssertionError("expected 224 cells")
    fr1, fg, cr1 = K.cell_statistics(zB, gated, G, cells)
    C.log(f"{len(cells)} cells scored [{time.time() - t0:.0f}s]")
    ctrl = K.control_choice(zB, ctx.parity)
    fpick = K.select_fused(fr1, fg, ctrl, ctx.parity, range(len(cells)))
    cpick = K.select_cf(cr1, ctx.parity, range(len(cells)))
    fused = K.assemble(zB, gated, cells, fpick, ctx.parity)
    cfs = K.assemble(zB, G, cells, cpick, ctx.parity)
    C._assert_finite(fused, name), C._assert_finite(cfs, name)
    from src.eval.aspect_metrics import per_anchor
    pn, pc = per_anchor(fused), per_anchor(cfs)

    # ---- tau_0 sanity report (decides nothing)
    n56 = len(nested_cells())
    sanity = {"fused_score_arrays_equal_parent_cells": True, "cells_checked": n56, "counterpart_max_abs_score_diff": 0.0}
    for i, (k, u, a) in enumerate(cells[:n56]):
        mine = _combine(zB, zB, gated[k], u, a)
        par = _combine(zB, zB, zT, u, a)
        if not all(np.array_equal(mine[c][d], par[c][d]) for c in CONDITIONS for d in DIRECTIONS):
            sanity["fused_score_arrays_equal_parent_cells"] = False
        pc_scores = nested_scores(bundle.B, bundle.B, C.cf_version(T), u, a)      # parent's counterpart cell
        mine_cf = _combine(zB, zB, G[k], u, a)
        diffs = [np.max(np.abs(mine_cf[c][d].astype(np.float64) - pc_scores[c][d].astype(np.float64)))
                 for c in CONDITIONS for d in DIRECTIONS]
        sanity["counterpart_max_abs_score_diff"] = max(sanity["counterpart_max_abs_score_diff"], float(max(diffs)))
    allowed0 = range(n56)
    f0, c0 = K.select_fused(fr1, fg, ctrl, ctx.parity, allowed0), K.select_cf(cr1, ctx.parity, allowed0)
    v_f, v_g, v_c = np.empty(ctx.n), np.empty(ctx.n), np.empty(ctx.n)
    for half in (0, 1):
        ap_ = ctx.parity != half
        v_f[ap_], v_g[ap_], v_c[ap_] = fr1[f0[half]][ap_], fg[f0[half]][ap_], cr1[c0[half]][ap_]
    sanity["tau0_restricted_fused_equals_parent_fused_per_anchor"] = bool(
        np.array_equal(v_f, pz["fused__r1"]) and np.array_equal(v_g, pz["fused__gain"]))
    sanity["tau0_restricted_counterpart_r1_equals_parent_counterpart"] = bool(np.array_equal(v_c, pz["cf__r1"]))
    sanity["tau0_restricted_counterpart_r1_mean"] = 100 * float(v_c.mean())
    sanity["parent_counterpart_r1_mean"] = 100 * float(np.mean(pz["cf__r1"]))
    sanity["tau0_restricted_counterpart_per_anchor_max_abs_r1_diff"] = float(np.max(np.abs(v_c - pz["cf__r1"])))
    sanity["note"] = ("the counterpart differs from the parent's only by z-scoring before (R-c) rather than after "
                      "(parent) the two-condition average")

    # ---- summary through the common code
    open_share = {f"tau_{k}": {"overall": 100 * float(np.mean(np.concatenate([g[k]["a"], g[k]["b"]]))),
                               "a": 100 * float(g[k]["a"].mean()), "b": 100 * float(g[k]["b"].mean()),
                               "per_pair": {p: 100 * float(np.mean(np.concatenate([g[k]["a"][ctx.pair_index == i],
                                                                                   g[k]["b"][ctx.pair_index == i]])))
                                            for i, p in enumerate(C.POOLED_ORDER)}} for k in g}
    cell_info = {"fused": {str(h): {"tau_index": cells[i][0], "tau": taus[cells[i][0]], "lambda_u": cells[i][1],
                                    "lambda_a": cells[i][2]} for h, i in fpick.items()},
                 "counterpart": {str(h): {"tau_index": cells[i][0], "tau": taus[cells[i][0]], "lambda_u": cells[i][1],
                                          "lambda_a": cells[i][2]} for h, i in cpick.items()},
                 "control": {str(h): {"sigma": s, "r1_on_tune_half": r} for h, (s, r) in ctrl.items()}}
    summary, arr = C.evaluate_fused(bundle, config, pn, pc, picks, name,
                                    crossfit={"cells": cell_info})
    summary.update(smoke=bool(args.smoke), parent=args.parent, taus=taus, gate_open_share=open_share,
                   tau0_sanity=sanity)
    extra = {"gate_a": np.stack([g[k]["a"] for k in sorted(g)]), "gate_b": np.stack([g[k]["b"] for k in sorted(g)]),
             "taus": np.array(taus)}
    C.save_candidate(name, summary, arr, T, picks, margins, extra=extra, smoke=args.smoke)
    print("tau0 sanity:", json.dumps(sanity, indent=1))
    print("gate-open share per tau:", json.dumps({k: round(v["overall"], 3) for k, v in open_share.items()}))
    C.log(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
