"""Regression check of rule §4.7 with our own code: R1 (round 1's A0 half-readers, frozen) on the k_top = 13 cells only
(cells 0 to 223), both cross-fits restricted to them, must reproduce round-1 R-c exactly: T, margins, picks, tau_0..3,
gates, chosen cells (fused 116 / 119, counterpart 58 / 123, sigma* = 0 on both halves), the per-anchor arrays
fused__*, cf__*, bar_v, the bar margin and the gain statistic with their intervals at full precision.
Never scores a cell with k_top < 13. Writes out/rd2_regress.{json,npz}."""
import json
import time

import numpy as np

import rd2_core as K
import rd2_family as FAM
from rd2_core import COND, DIRS, METRICS, RC

STORED = "results/cand_Rc_Rb_expected_A0"
RULE_TAUS = [3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211]
RULE_BAR = (0.4435221354166667, 0.21646171563312194, 0.6735669710776852)
RULE_GAIN = (2.667236328125, 2.325087836946873, 3.012361650695922)
RULE_FUSED_R1, RULE_CF_R1 = 18.918863932291664, 18.475341796875


def r1_probs(cache, config):
    """R1: P^c = mean over the two half-readers of model_j.predict_proba(scaler_j.transform(x))."""
    pk = K.load_reader_pickle(config)
    halves = {c: [K.half_probs_scaled(h["model"], h["scaler"].transform(cache["F"][config][c])) for h in pk["halves"]]
              for c in COND}
    P = {c: K.mean_two(*halves[c]) for c in COND}
    for c in COND:
        if not np.allclose(P[c].sum(1), 1.0, rtol=0, atol=1e-12):
            raise AssertionError("R1 probabilities do not sum to 1")
    return P, halves, pk


def main():
    t0 = time.time()
    RC.assert_rule()
    RC.assert_inputs([f"{STORED}.npz", f"{STORED}.json", "results/rc_tau.json", "results/rb_reader_A0.pkl",
                      "results/rb_reader_A0.json"])
    if K.allowed_ktops() != (13,):
        K.log("note: phase-2 marker present; the regression check still scores k_top = 13 only")
    cache = K.load_cache()
    sz = np.load(RC.r1_path(f"{STORED}.npz"))
    sj = json.loads(RC.r1_path(f"{STORED}.json").read_text())
    st = json.loads(RC.r1_path("results/rc_tau.json").read_text())
    P, halves, pk = r1_probs(cache, "A0")
    T, picks, margins = FAM.reader_parts(cache, P, "A0")
    taus = K.thresholds(margins["a"], margins["b"])
    chk = {}
    chk["T_equal_stored"] = {f"{c}__{d}": bool(np.array_equal(T[c][d], sz[f"T__{c}__{d}"])) for c in COND for d in DIRS}
    chk["T_max_abs_diff"] = float(max(np.abs(T[c][d].astype(np.float64) - sz[f"T__{c}__{d}"]).max()
                                      for c in COND for d in DIRS))
    chk["margin_equal_stored"] = {c: bool(np.array_equal(margins[c], sz[f"margin__{c}"])) for c in COND}
    chk["pick_equal_stored"] = {c: bool(np.array_equal(picks[c], sz[f"pick__{c}"].astype(np.int64))) for c in COND}
    chk["taus"] = taus
    chk["taus_equal_rc_tau_json"] = [a == b for a, b in zip(taus, st["taus"])]
    chk["taus_equal_rule_text"] = [a == b for a, b in zip(taus, RULE_TAUS)]
    chk["taus_equal_stored_npz"] = bool(np.array_equal(np.asarray(taus), sz["extra__taus"]))
    K.log(f"T/margins/picks/taus: {chk['T_equal_stored']} {chk['margin_equal_stored']} {chk['pick_equal_stored']} "
          f"{chk['taus_equal_rc_tau_json']}", t0)

    rec, pn, pc, extra = FAM.run_family(cache, T, margins, taus, ktops=(13,))
    K.log(f"224 cells scored: fused {rec['fused_cells']}, cf {rec['counterpart_cells']}, ctrl {rec['control']}", t0)
    chk["gates_equal_stored"] = {c: bool(np.array_equal(np.stack([g[c] for g in extra["gates"]]).astype(np.float32),
                                                        sz[f"extra__gate_{c}"])) for c in COND}
    chk["fused_cells"] = {h: rec["fused_cells"][h]["cell"] for h in ("0", "1")}
    chk["counterpart_cells"] = {h: rec["counterpart_cells"][h]["cell"] for h in ("0", "1")}
    chk["sigma_star"] = {h: rec["control"][h]["sigma"] for h in ("0", "1")}
    chk["cells_equal_round1"] = (chk["fused_cells"] == {"0": 116, "1": 119}
                                 and chk["counterpart_cells"] == {"0": 58, "1": 123}
                                 and chk["sigma_star"] == {"0": 0.0, "1": 0.0})
    # the decoded settings of round 1's JSON
    s1 = sj["crossfit"]["cells"]
    chk["cell_settings_equal_round1_json"] = all(
        rec[f"{kind}_cells"][h][k] == s1[name][h][k]
        for kind, name in (("fused", "fused"), ("counterpart", "counterpart")) for h in ("0", "1")
        for k in ("tau_index", "tau", "lambda_u", "lambda_a"))
    chk["control_r1_on_tune_half_equal_round1_json"] = all(
        rec["control"][h]["r1_on_tune_half"] == s1["control"][h]["r1_on_tune_half"] for h in ("0", "1"))
    chk["float_mean_criterion_cells"] = rec["fused_cells_by_float_mean_criterion"]
    chk["per_anchor_equal_stored"] = {**{f"fused__{m}": bool(np.array_equal(pn[m], sz[f"fused__{m}"])) for m in METRICS},
                                      **{f"cf__{m}": bool(np.array_equal(pc[m], sz[f"cf__{m}"])) for m in METRICS}}
    ev, bar_v, gain_v = K.evaluate(pn, pc, cache["pB"], cache["pBp"]["A0"], cache["cl"], cache["pair_index"])
    chk["bar_v_equal_stored"] = bool(np.array_equal(bar_v, sz["bar_v"]))
    chk["bar_comparator"] = ev["bar_comparator"]
    chk["bar_margin"] = ev["bar_margin"]
    chk["gain_statistic"] = ev["gain_statistic"]
    chk["bar_margin_equal_rule"] = (ev["bar_margin"]["point"] == RULE_BAR[0]
                                    and ev["bar_margin"]["ci95"] == [RULE_BAR[1], RULE_BAR[2]])
    chk["gain_statistic_equal_rule"] = (ev["gain_statistic"]["point"] == RULE_GAIN[0]
                                        and ev["gain_statistic"]["ci95"] == [RULE_GAIN[1], RULE_GAIN[2]])
    chk["bar_equal_round1_json"] = (ev["bar_margin"] == sj["bar"]["r1"] and ev["bar_comparator"] == sj["bar"]["comparator"])
    chk["gain_equal_round1_json"] = (ev["gain_statistic"]["point"] == sj["gain_statistic"]["point"]
                                     and ev["gain_statistic"]["ci95"] == sj["gain_statistic"]["ci95"])
    chk["r1_means"] = ev["r1_means"]
    chk["fused_cf_r1_equal_rule"] = (ev["r1_means"]["fused"] == RULE_FUSED_R1
                                     and ev["r1_means"]["counterpart"] == RULE_CF_R1)
    chk["gate_open_share_equal_round1_json"] = all(
        rec["gate_open_share"][f"tau_{t}"][k] == sj["gate_open_share"][f"tau_{t}"][k]
        for t in range(4) for k in ("overall", "a", "b"))
    passed = (all(chk["T_equal_stored"].values()) and all(chk["margin_equal_stored"].values())
              and all(chk["pick_equal_stored"].values()) and all(chk["taus_equal_rc_tau_json"])
              and all(chk["taus_equal_rule_text"]) and all(chk["gates_equal_stored"].values())
              and chk["cells_equal_round1"] and chk["cell_settings_equal_round1_json"]
              and all(chk["per_anchor_equal_stored"].values()) and chk["bar_v_equal_stored"]
              and chk["bar_margin_equal_rule"] and chk["gain_statistic_equal_rule"] and chk["bar_comparator"] == "counterpart"
              and chk["fused_cf_r1_equal_rule"])
    chk["PASSED"] = bool(passed)
    out = {"what": "rule §4.7 regression check (R1 on cells 0..223 vs round-1 R-c), our own code", "checks": chk,
           "family_record_224": rec, "evaluation_224": ev, "R1_halves_C": [float(h["model"].C) for h in pk["halves"]],
           "runtime_s": round(time.time() - t0, 1), "provenance": K.provenance()}
    K.save_json(K.OUT / "rd2_regress.json", out)
    npz = {f"R1__P__{c}": P[c] for c in COND}
    for c in COND:
        npz[f"R1__P_half0__{c}"], npz[f"R1__P_half1__{c}"] = halves[c]
        npz[f"R1__margin__{c}"] = margins[c]
        npz[f"R1__pick__{c}"] = picks[c]
        for d in DIRS:
            npz[f"R1__T__{c}__{d}"] = T[c][d]
    for m in METRICS:
        npz[f"reg224__fused__{m}"] = pn[m]
        npz[f"reg224__cf__{m}"] = pc[m]
    npz["reg224__bar_v"] = bar_v
    npz["reg224__gain_v"] = gain_v
    npz["R1__taus"] = np.asarray(taus)
    np.savez(K.OUT / "rd2_regress.npz", **npz)
    K.log(f"REGRESSION CHECK {'PASSED' if passed else 'FAILED'}: bar {ev['bar_margin']} ({ev['bar_comparator']}), "
          f"gain {ev['gain_statistic']}", t0)
    if not passed:
        print(json.dumps(K.jsonable(chk), indent=1))


if __name__ == "__main__":
    main()
