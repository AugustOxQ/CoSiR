"""Phase 2a re-derivation: R-b on A1, A0, AR from the stored half-readers (loaded as data) applied to seed-42
features computed here (rule 4.2 item 4) from the standard heads; picks, both scorings, fused reader, counterpart,
comparators, bar margin, gain statistic, clauses, pick accuracy, AR picks to rand. Writes out/rd_rb.{json,npz}."""
import sys
import time

import numpy as np

import rd_core as K
from rd_core import CONDITIONS, DIRECTIONS, METRICS


def main():
    t0 = time.time()
    P = K.prepare_seed42()
    ep = P.ep
    stack_all = K.score_stack(P.post, ep, K.GROUPINGS)
    result = {"checks": P.checks, "candidates": {}, "readers": {}}
    npz = {}
    arrays = {}
    for cfg in ("A1", "A0", "AR"):
        g = K.CONFIGS[cfg]
        idx = [K.GROUPINGS.index(h) for h in g]
        X = {c: K.rb_features(P.post, ep, g, c) for c in CONDITIONS}
        X32 = {c: K.rb_features_f32(P.post, ep, g, c) for c in CONDITIONS}
        if AD:                       # primary features in aspect_deltas' precision; the float64 version as the variant
            X, X32 = X32, X
        r = K.load_readers(cfg)
        Pr, Pr32, halves = {}, {}, {}
        for c in CONDITIONS:
            Pr[c], halves[c] = K.reader_proba(r, X[c])
            Pr32[c], _ = K.reader_proba(r, X32[c])
        picks = {c: K.first_argmax(Pr[c]) for c in CONDITIONS}
        picks32 = {c: K.first_argmax(Pr32[c]) for c in CONDITIONS}
        margins = {c: K.top_two_margin(Pr[c]) for c in CONDITIONS}
        stack = {d: stack_all[d][:, idx] for d in DIRECTIONS}
        result["readers"][cfg] = {
            "C": [float(h["model"].C) for h in r["halves"]],
            "features_f64_vs_f32_max_abs": float(max(np.abs(X[c] - X32[c]).max() for c in CONDITIONS)),
            "proba_f64_vs_f32_max_abs": float(max(np.abs(Pr[c] - Pr32[c]).max() for c in CONDITIONS)),
            "picks_f64_vs_f32_disagree": {c: int((picks[c] != picks32[c]).sum()) for c in CONDITIONS},
            "half_reader_pick_agreement": {c: 100 * float(np.mean(halves[c][0].argmax(1) == halves[c][1].argmax(1)))
                                           for c in CONDITIONS},
            "delta_b_equals_minus_delta_a": bool(np.array_equal(X["b"][:, 2::6], -X["a"][:, 2::6]))}
        for c in CONDITIONS:
            npz[f"{cfg}__X__{c}"] = X[c]
            npz[f"{cfg}__P__{c}"] = Pr[c]
            npz[f"{cfg}__P32feat__{c}"] = Pr32[c]
        for scoring in ("argmax", "expected"):
            t = time.time()
            T = K.hard_term(stack, picks) if scoring == "argmax" else K.expected_term(stack, Pr)
            ev, pn, pc, bar_v = K.full_eval(T, picks, cfg, P)
            name = f"Rb_{scoring}_{cfg}"
            ev["runtime_s"] = round(time.time() - t, 1)
            result["candidates"][name] = ev
            arrays[name] = (pn, pc, bar_v)
            for m in METRICS:
                npz[f"{name}__fused__{m}"] = np.asarray(pn[m])
                npz[f"{name}__cf__{m}"] = np.asarray(pc[m])
            npz[f"{name}__bar_v"] = bar_v
            for c in CONDITIONS:
                npz[f"{name}__pick__{c}"] = picks[c].astype(np.int8)
                npz[f"{name}__margin__{c}"] = margins[c]
                for d in DIRECTIONS:
                    npz[f"{name}__T__{c}__{d}"] = T[c][d]
            print(f"[{time.time() - t0:.0f}s] {name}: bar {ev['bar']['r1']} ({ev['bar']['comparator']}), gain "
                  f"{ev['gain_statistic']}, clears {ev['clears_bar']}, pick {ev['pick_accuracy']['correct_share']}",
                  flush=True)
        for m in METRICS:
            npz[f"{cfg}__Bprime__{m}"] = np.asarray(P.pBp[cfg][m])
    for m in METRICS:
        npz[f"B__{m}"] = np.asarray(P.pB[m])
    cl = P.cl
    result["A1_minus_A0"] = {}
    for scoring in ("argmax", "expected"):
        pn1, pc1, b1 = arrays[f"Rb_{scoring}_A1"]
        pn0, pc0, b0 = arrays[f"Rb_{scoring}_A0"]
        result["A1_minus_A0"][scoring] = {"fused_r1": K.point_ci(pn1["r1"] - pn0["r1"], cl),
                                          "bar_r1": K.point_ci(b1 - b0, cl)}
    result["runtime_s"] = round(time.time() - t0, 1)
    tag = "rd_rb_ad" if AD else "rd_rb"
    result["features"] = "aspect_deltas precision (float32 S, C, Delta)" if AD else "float64 agreements"
    K.save_json(K.OUT / f"{tag}.json", result)
    np.savez_compressed(K.OUT / f"{tag}.npz", **npz)
    print(f"done in {result['runtime_s']}s; readers {result['readers']}", flush=True)


AD = len(sys.argv) > 1 and sys.argv[1] == "ad"

if __name__ == "__main__":
    main()
