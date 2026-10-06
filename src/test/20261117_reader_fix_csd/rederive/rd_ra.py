"""Phase 1 re-derivation: B, B', sigma_h, R-a on A1, A0, AR (picks, reader term, fused reader, counterpart, bar
comparator, bar margin, gain statistic, clauses, pick accuracy) and the AR check's pick shares. Writes
rederive/out/rd_ra.json and rd_ra.npz. Our own code (rd_core); see its docstring for what is reused.

Run (from /project/CoSiR):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261117_reader_fix_csd/rederive/rd_ra.py
"""
import time

import numpy as np

import rd_core as K
from rd_core import CONDITIONS, DIRECTIONS, METRICS, per_anchor


def main():
    t0 = time.time()
    K.check_inputs()
    S = K.seed42_setup()
    ctx, ep, cl = S.ctx, S.ctx.pooled, S.ctx.anchor_group
    z1 = np.load(K.STEP1_EVAL)
    checks = {}
    checks["partition_L_equals_step1_npz"] = bool(np.array_equal(S.partition_L, z1["partition_L"]))
    checks["anchor_group_pair_index_equal_step1"] = bool(np.array_equal(z1["anchor_group"], cl)
                                                         and np.array_equal(z1["pair_index"], ctx.pair_index))

    post, e2, prov = K.standard_posteriors(ctx, S.scorer_train, S.partition_L)
    checks.update({k: v for k, v in prov.items() if isinstance(v, bool)})
    print(f"[{time.time() - t0:.0f}s] posteriors ready; affect head {prov['affect_head']}", flush=True)

    t_n1u = K.t_n1u_term(ctx, S.scorer_train)
    B, B_picks = K.rebuild_B(ctx, t_n1u, e2)
    pB = per_anchor(B)
    checks["B_equals_step1_B"] = all(np.array_equal(np.asarray(pB[m]), z1[f"B__{m}"]) for m in METRICS)
    checks["B_scores_equal_setup_B"] = all(np.array_equal(B[c][d], S.B[c][d]) for c in CONDITIONS for d in DIRECTIONS)
    pBp, Bp_picks = {}, {}
    for cfg in K.CONFIGS:
        Bp, Bp_picks[cfg] = K.rebuild_Bprime(ctx, t_n1u, post, cfg)
        pBp[cfg] = per_anchor(Bp)
        checks[f"Bprime_{cfg}_equals_step1"] = all(np.array_equal(np.asarray(pBp[cfg][m]), z1[f"{cfg}__Bprime__{m}"])
                                                   for m in METRICS)
    print(f"[{time.time() - t0:.0f}s] B and B' rebuilt: {checks}", flush=True)

    sig64 = K.sigma_h(post, ep)
    sig32 = K.sigma_h(post, ep, f32=True)
    d32, d64 = K.deltas(post, ep, K.GROUPINGS)
    checks["delta_b_is_minus_delta_a_f32"] = bool(np.array_equal(d32["b"], -d32["a"]))
    stack_all = K.score_stack(post, ep, K.GROUPINGS)
    print(f"[{time.time() - t0:.0f}s] sigma64 {sig64}", flush=True)

    result = {"checks": checks, "sigma": sig64, "sigma_f32_agreements": sig32, "B_picks": B_picks,
              "Bprime_picks": Bp_picks, "affect_head": prov["affect_head"], "candidates": {}}
    npz = {f"B__{m}": np.asarray(pB[m]) for m in METRICS}
    npz.update({f"sigma__{h}": np.float64(v) for h, v in sig64.items()})
    arrays = {}
    for cfg in ("A1", "A0", "AR"):
        t = time.time()
        g = K.CONFIGS[cfg]
        idx = [K.GROUPINGS.index(h) for h in g]
        sv = np.array([sig64[h] for h in g], dtype=np.float64)
        scaled = {c: d32[c][:, idx].astype(np.float64) / sv for c in CONDITIONS}
        scaled64 = {c: d64[c][:, idx] / sv for c in CONDITIONS}
        picks = {c: K.first_argmax(scaled[c]) for c in CONDITIONS}
        picks64 = {c: K.first_argmax(scaled64[c]) for c in CONDITIONS}
        margins = {c: K.top_two_margin(scaled[c]) for c in CONDITIONS}
        argmax_step1 = {c: K.first_argmax(d32[c][:, idx]) for c in CONDITIONS}
        stack = {d: stack_all[d][:, idx] for d in DIRECTIONS}
        T = K.hard_term(stack, picks)
        nested, control, Tp, cf, cfp = K.fuse(B, T, ctx.parity)
        pn, pc = per_anchor(nested), per_anchor(cf)
        ev, bar_v = K.evaluate(pn, pc, pB, pBp[cfg], ctx)
        ev["crossfit"] = {"T_picks": Tp, "cf_picks": cfp,
                          "control_ranks_as_B": all(np.array_equal(np.asarray(per_anchor(control)[m]), np.asarray(pB[m]))
                                                    for m in METRICS)}
        ev["pick_accuracy"] = K.pick_accuracy(picks, cfg, ctx)
        ev["pick_share"] = K.pick_shares(picks, cfg, ctx)
        ev["picks_count"] = {c: np.bincount(picks[c], minlength=len(g)).tolist() for c in CONDITIONS}
        ev["picks_f32_vs_f64_delta_disagree"] = {c: int((picks[c] != picks64[c]).sum()) for c in CONDITIONS}
        ev["step1_argmax_picks_reproduced"] = all(np.array_equal(argmax_step1[c], z1[f"{cfg}__reader_pick__{c}"])
                                                  for c in CONDITIONS)
        # step-1 arg-max reader (stored arrays) as the reference: margin and bar margin (D10 comparator)
        s1n = {m: z1[f"{cfg}__reader__fused__{m}"] for m in METRICS}
        s1c = {m: z1[f"{cfg}__reader__cf__{m}"] for m in METRICS}
        s1_name, s1_comp, _ = K.bar_comparator(pBp[cfg], s1c, pB)
        s1_bar_v = s1n["r1"] - s1_comp["r1"]
        ev["vs_step1_argmax"] = {
            "step1_bar_comparator": s1_name,
            "step1_bar_r1": K.point_ci(s1_bar_v, cl),
            "step1_margin_r1": K.point_ci(s1n["r1"] - s1c["r1"], cl),
            "step1_fused_r1": 100 * float(np.mean(s1n["r1"])),
            "margin_r1": K.point_ci((pn["r1"] - pc["r1"]) - (s1n["r1"] - s1c["r1"]), cl),
            "bar_margin_r1": K.point_ci(bar_v - s1_bar_v, cl),
            "fused_r1": K.point_ci(pn["r1"] - s1n["r1"], cl)}
        if cfg == "AR":
            ev["ar_check_step1_argmax_shares"] = K.pick_shares({c: z1[f"AR__reader_pick__{c}"].astype(int)
                                                                for c in CONDITIONS}, "AR", ctx)
        ev["runtime_s"] = round(time.time() - t, 1)
        result["candidates"][f"Ra_{cfg}"] = ev
        arrays[cfg] = (pn, pc, bar_v)
        for m in METRICS:
            npz[f"{cfg}__fused__{m}"] = np.asarray(pn[m])
            npz[f"{cfg}__cf__{m}"] = np.asarray(pc[m])
            npz[f"{cfg}__Bprime__{m}"] = np.asarray(pBp[cfg][m])
        npz[f"{cfg}__bar_v"] = bar_v
        for c in CONDITIONS:
            npz[f"{cfg}__pick__{c}"] = picks[c].astype(np.int8)
            npz[f"{cfg}__margin__{c}"] = margins[c]
            for d in DIRECTIONS:
                npz[f"{cfg}__T__{c}__{d}"] = T[c][d]
        print(f"[{time.time() - t0:.0f}s] Ra_{cfg}: bar {ev['bar']['r1']} ({ev['bar']['comparator']}), "
              f"gain {ev['gain_statistic']}, clears {ev['clears_bar']}, pick {ev['pick_accuracy']['correct_share']}",
              flush=True)

    pn1, pc1, b1 = arrays["A1"]
    pn0, pc0, b0 = arrays["A0"]
    result["A1_minus_A0_under_Ra"] = {"fused_r1": K.point_ci(pn1["r1"] - pn0["r1"], cl),
                                      "bar_r1": K.point_ci(b1 - b0, cl),
                                      "margin_r1": K.point_ci((pn1["r1"] - pc1["r1"]) - (pn0["r1"] - pc0["r1"]), cl)}
    result["runtime_s"] = round(time.time() - t0, 1)
    K.OUT.mkdir(exist_ok=True)
    K.save_json(K.OUT / "rd_ra.json", result)
    np.savez_compressed(K.OUT / "rd_ra.npz", **npz)
    print(f"done in {result['runtime_s']}s; checks {checks}", flush=True)


if __name__ == "__main__":
    main()
