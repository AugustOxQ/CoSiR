"""Round 5 re-derivation, stage B (seed 42): the GoEmotions spot check, then the GE head and every development
number of the round with the GE placement, the carry and, for a carried candidate, the detectable margins x.

Runs only after rd5_stageA.json recorded every stage-A target as matched; the GE placement is refused (Guard) until
the stage-A record and the spot check have passed in this process. Writes rederive/results/rd5_stageB.json and
rd5_stageB_arrays.npz (refuses to overwrite) and prints their SHA-256s. It reads none of the implementation's files.

Run (from /project/CoSiR):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261123_idea3_goemotions/rederive/rd5_stageB.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd5_paths as paths  # noqa: E402
import rd5_core as core  # noqa: E402
import rd5_stats as stats  # noqa: E402
import rd5_goemotions as ge_mod  # noqa: E402
from rd5_bundle import build, pa_equal, sha_pa  # noqa: E402
from rd5_candidates import Guard, candidates, comparators, extend  # noqa: E402
from rd5_placement import DRAW_SHA, check_mapping, ge_head, ge_input  # noqa: E402
from rd5_stageA import PAIRS, TARGET, _js, cells_of, pci, stored_pa  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main():
    out_json = paths.RESULTS / "rd5_stageB.json"
    out_npz = paths.RESULTS / "rd5_stageB_arrays.npz"
    for p in (out_json, out_npz):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    fixes = sorted(paths.RESULTS.glob("rd5_stageA_fix*.json"), key=lambda q: int(q.stem.split("fix")[1]))
    a_path = fixes[-1] if fixes else paths.RESULTS / "rd5_stageA.json"        # the latest corrected stage-A record
    stage_a = json.loads(a_path.read_text())
    if not (stage_a["passed"] and not stage_a["failed"]):
        raise SystemExit("stage A did not pass; no GE-placement result is computed")
    started = paths.now_ams()
    inputs = paths.assert_inputs()
    A = paths.allowed()
    s4 = np.load(paths.P["seed42_arrays"])
    b = build(42, A, log)

    # ---- the A0 path in this process (AFF re-verified against the stored arrays before any GE number)
    rd = core.reader(b.F, b.stack, b.halves)
    taus = core.taus_from_margins(rd["a"]["m"], rd["b"]["m"])
    if [float(x) for x in taus] != TARGET["taus"]:
        raise SystemExit("tau differs from rc_tau.json")
    T = {c: rd[c]["T"] for c in core.CONDITIONS}
    g_aff = {c: core.gates_aff(rd[c]["m"], rd[c]["pi"], taus) for c in core.CONDITIONS}
    f_aff = core.run_family(b.B, T, g_aff, b.parity, A["zscore_rows"])
    if not (cells_of(f_aff) == (TARGET["aff"]["cells_fused"], TARGET["aff"]["cells_cf"], TARGET["aff"]["sigma"])
            and pa_equal(f_aff.pa_fused, stored_pa(s4, "aff_fused")) and pa_equal(f_aff.pa_cf, stored_pa(s4, "aff_cf"))):
        raise SystemExit("AFF is not reproduced in this process")
    log("AFF reproduced in this process")

    # ---- the GoEmotions file and the CPU spot check
    splits = A["artelingo_splits"](b.ctx.data)
    ge = ge_mod.load_ge_file(b.ctx, splits)
    spot = ge_mod.spot_check(b.ctx, ge, log)
    log(f"spot check passed={spot['passed']}")
    if not spot["passed"]:
        raise SystemExit("the CPU spot check failed; no GE-placement result is computed")
    Guard.released = True

    # ---- the GE head (D4) and Q_GE
    if paths.sha_file(paths.P["affect_npz"]) != paths.SHA["affect_npz"]:
        raise SystemExit("affect_prepare.npz SHA-256 differs")
    aff_probs = np.load(paths.P["affect_npz"])["affect_probs"]
    rec_aff = json.loads(paths.P["affect_json"].read_text())
    if paths.sha_array(aff_probs) != rec_aff["affect_probs_sha256"]:
        raise SystemExit("affect_probs bytes differ from affect_prepare.json")
    st, rows = b.scorer_train, ge["rows"]
    X = ge_input(len(b.ctx.groups), st, aff_probs, rows, ge["probs"])
    log("GE head fit")
    gh = ge_head(X, b.lab, st, rows)
    if gh["head"] is None:
        raise SystemExit(f"GE head reached its cap after the fallback: {gh}")
    head = gh["head"]
    if not (head["draw_sha"] == DRAW_SHA and check_mapping(st, head["draw"])
            and np.isfinite(X[head["draw"]]).all() and np.isfinite(X[head["check"]]).all()):
        raise SystemExit("GE head draw, mapping or finiteness check failed")
    Q = head["post"]
    counts = np.bincount(b.lab[head["check"]])
    placement_rec = {"accuracy": head["accuracy"], "classes": head["classes"].tolist(), "n_iter_first": gh["n_iter_first"],
                     "fallback_used": gh["fallback_used"], "n_iter_fallback": gh["n_iter_fallback"],
                     "check_majority_share": 100 * float(counts.max() / counts.sum()), "uniform": 100.0 / 41,
                     "clip_txt": 35.72, "clip_img": 9.81, "draw_sha256": head["draw_sha"],
                     "Q_GE_sel_sha256": paths.sha_array(Q[rows]), "Q_GE_full_sha256": paths.sha_array(Q)}
    log("GE head done")

    # ---- the GE extension, the positive check and the candidates
    ext = extend(b, Q, "ge", A)
    pi_img = b.post["affect"]["img"]
    ind = {"i2t": np.einsum("nc,nkc->nk", pi_img[b.ep.anchor], Q[b.ep.candidates]),
           "t2i": np.einsum("nc,nkc->nk", Q[b.ep.anchor], pi_img[b.ep.candidates])}
    positive = (all(np.array_equal(ext.stack[d][:, 0], ind[d]) for d in core.DIRECTIONS)
                and any(not np.array_equal(ext.stack[d][:, 0], b.stack[d][:, 0]) for d in core.DIRECTIONS)
                and any(not np.array_equal(ext.F[c][:, :6], b.F[c][:, :6]) for c in core.CONDITIONS)
                and not np.array_equal(Q[b.ctx.selection], b.post["affect"]["txt"][b.ctx.selection]))
    if not positive:
        raise SystemExit("D5 positive check failed")
    cand = candidates(b, ext, rd, taus, A)
    if not all(np.array_equal(cand["G-T"]["gates"][c], g_aff[c]) for c in core.CONDITIONS):
        raise SystemExit("G-T's gates differ from AFF's")
    bp1 = stored_pa(s4, "Bp1")
    records, arrays = {}, {"Q_GE_sel": Q[rows], "rows": rows}
    for name, v in cand.items():
        fam = v["family"]
        comps = [("B'_G" if k == "B'_Q" else k, pa) for k, pa in comparators(ext, b, fam)]
        rec = stats.dev_record(fam.pa_fused, fam.pa_cf, comps, f_aff.pa_fused, bp1, b.cl, b.pair_index,
                               A["cluster_bootstrap"], PAIRS)
        rec["taus"] = [float(x) for x in v["taus"]]
        rec["open_tau0"] = v["open_tau0"]
        rec["picks"] = fam.summary(v["taus"])
        rec["per_anchor_sha256"] = {"fused": sha_pa(fam.pa_fused), "cf": sha_pa(fam.pa_cf)}
        rec["gates_sha256"] = {c: paths.sha_array(v["gates"][c]) for c in core.CONDITIONS}
        records[name] = rec
        key = name.replace("-", "")
        for c in core.CONDITIONS:
            arrays[f"{key}_gate__{c}"] = v["gates"][c]
            arrays[f"{key}_P__{c}"] = v["reader"][c]["P"]
            arrays[f"{key}_m__{c}"] = v["reader"][c]["m"]
            arrays[f"{key}_pi__{c}"] = v["reader"][c]["pi"]
        for m in core.METRICS:
            arrays[f"{key}_fused__{m}"] = fam.pa_fused[m]
            arrays[f"{key}_cf__{m}"] = fam.pa_cf[m]
        arrays[f"{key}_taus"] = v["taus"]
        log(f"{name}: D10 {rec['D10']['clears']}, delta {rec['delta_vs_aff']['delta_int']}")
    bpg = {"Bprime_G_r1": stats.mean_pp(ext.pB["r1"]),
           "Bprime_G_minus_Bprime_A0": stats.point_ci(ext.pB["r1"] - b.pBp0["r1"], b.cl, A["cluster_bootstrap"])}
    for m in core.METRICS:
        arrays[f"BpG__{m}"] = ext.pB[m]
    car = stats.carry(records)
    log(f"carry: {car['decision']} {car.get('carried')}")

    # ---- detectable margins x for a carried candidate (round 3 §6.1)
    sens = None
    if car["carried"]:
        fam = cand[car["carried"]]["family"]
        f = fam.pa_fused
        diffs = {"r1_vs_cosine": f["r1"] - b.cos_pa["r1"], "r1_vs_rca": f["r1"] - b.rca_pa["r1"],
                 "r1_vs_B": f["r1"] - b.pB["r1"], "r1_vs_Bprime_A0": f["r1"] - b.pBp0["r1"],
                 "r1_vs_Bprime_G": f["r1"] - ext.pB["r1"], "r1_vs_counterpart": f["r1"] - fam.pa_cf["r1"],
                 "gain_statistic": f["gain"] - 0.0, "gain_vs_rca": f["gain"] - b.rca_pa["gain"],
                 "r1_vs_AFF": f["r1"] - f_aff.pa_fused["r1"]}
        sens = {}
        for k, dv in diffs.items():
            s = stats.sensitivity(dv, b.cl)
            pc = stats.point_ci(dv, b.cl, A["cluster_bootstrap"])
            sens[k] = {"SE_pp": 100 * s["SE"], "half_width_1.96SE_pp": 100 * s["half_width_1.96SE"],
                       "x_pp": 100 * s["x_2.80SE"], "seed42_bootstrap_half_width_pp": (pc["ci95"][1] - pc["ci95"][0]) / 2,
                       "seed42_point_pp": pc["point"], "detail": s}

    result = {"stage": "B", "seed": 42, "started": started, "finished": paths.now_ams(), "runtime_s": time.time() - T0,
              "stageA_sha256": paths.sha_file(a_path), "inputs_sha256": inputs,
              "ge_file": {"sha256": ge["sha256"], "probs_sha256": ge["probs_sha256"], "checks": ge["checks"]},
              "spot_check": spot, "placement": placement_rec, "positive_check": True,
              "aff_reference": {"open_tau0": [int(g_aff[c][0].sum()) for c in core.CONDITIONS],
                                "fused_r1": stats.mean_pp(f_aff.pa_fused["r1"])},
              "records": records, "Bprime_G": bpg, "carry": car, "sensitivity": sens,
              "comparator_order": ["B'_G", "B'(A0)", "counterpart", "B"]}
    out_json.write_text(json.dumps(_js(result), indent=1))
    np.savez(out_npz, **arrays)
    print(f"WROTE {out_json.relative_to(paths.ROOT)} sha256 {paths.sha_file(out_json)}")
    print(f"WROTE {out_npz.relative_to(paths.ROOT)} sha256 {paths.sha_file(out_npz)}")


if __name__ == "__main__":
    main()
