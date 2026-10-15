"""Round 5 re-derivation, stage A (seed 42): own code against the stated targets, before any GoEmotions number.

  §5 item 1: R1 = round-1 R-c and AFF = round 3's targets at full precision (plus the stored seed-42 arrays);
  §5 item 3: own placement code on the CLIP features = fit_one_head's posteriors (same process), 35.72 / 9.81;
  §5 item 4 (tau and B' parts, and the candidate path): the CLIP placement through the G-T / G-TF code gives AFF.
No GoEmotions model call, no GE head on real data, no number with a GoEmotions placement.

Run (from /project/CoSiR):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261123_idea3_goemotions/rederive/rd5_stageA.py [--round1-bundle]
-> rederive/results/rd5_stageA.json (refuses to overwrite)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd5_paths as paths  # noqa: E402
import rd5_core as core  # noqa: E402
import rd5_stats as stats  # noqa: E402
from rd5_bundle import build, pa_equal, scores_equal, sha_pa  # noqa: E402
from rd5_candidates import COMPARATOR_ORDER, candidates, comparators, extend  # noqa: E402
from rd5_placement import DRAW_SHA, check_mapping, placement, unit  # noqa: E402

PAIRS = ("emotion__style", "emotion__genre", "style__genre")
T0 = time.time()

TARGET = {
    "taus": [3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211],
    "r1": {"cells_fused": [116, 119], "cells_cf": [58, 123], "sigma": [0.0, 0.0],
           "bar_margin": [0.4435221354166667, 0.21646171563312194, 0.6735669710776852],
           "gain_statistic": [2.667236328125, 2.325087836946873, 3.012361650695922]},
    "aff": {"fused_r1": 19.136555989583336, "cf_r1": 18.39599609375, "bar_comparator": "B'(A0)",
            "bar_margin": [0.6998697916666667, 0.4598852740816973, 0.9371680126852968],
            "margin_vs_cf": [0.7405598958333333, 0.5196896694963071, 0.9598857494832738],
            "gain_statistic": [3.110758463541667, 2.780005709854805, 3.4559584315470384],
            "either_change_vs_cf": -1.629638671875,
            "per_pair_bar_margin": [0.9765625, 1.45263671875, -0.32958984375],
            "cells_fused": [39, 119], "cells_cf": [149, 10], "sigma": [0.0, 0.0],
            "minus_r1_fused": [0.21769205729166666, 0.06425880757348419, 0.3709597330984391],
            "minus_r1_bar": [0.25634765625, 0.04280778303598444, 0.46195041633015954],
            "open_tau0": [9941, 3627],
            "minus_Bp_A1": [0.33162434895833337, 0.048231414333532084, 0.6246158772581268],
            "Bp_A1_r1": 18.804931640625},
    "heads": {"txt": 35.72, "img": 9.81},
}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def pci(x):
    return [x["point"], x["ci95"][0], x["ci95"][1]]


class Checks:
    def __init__(self):
        self.items = {}

    def add(self, name, ok, **info):
        self.items[name] = {"ok": bool(ok), **{k: _js(v) for k, v in info.items()}}
        log(f"{'PASS' if ok else 'FAIL'} {name}")

    def all_ok(self):
        return all(v["ok"] for v in self.items.values())

    def failed(self):
        return [k for k, v in self.items.items() if not v["ok"]]


def _js(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {str(k): _js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_js(x) for x in v]
    return v


def cells_of(fam):
    return ([fam.picks[0]["fused_cell"], fam.picks[1]["fused_cell"]],
            [fam.picks[0]["cf_cell"], fam.picks[1]["cf_cell"]],
            [fam.picks[0]["sigma"], fam.picks[1]["sigma"]])


def stored_pa(z, prefix):
    return {m: z[f"{prefix}__{m}"] for m in core.METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1-bundle", action="store_true", help="also compare with round 1's common.load_bundle()")
    ap.add_argument("--fix", type=int, default=0, help="write rd5_stageA_fix<n>.json (an earlier file is kept)")
    args = ap.parse_args()
    out_path = paths.RESULTS / ("rd5_stageA.json" if args.fix == 0 else f"rd5_stageA_fix{args.fix}.json")
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; refusing to overwrite")
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    started = paths.now_ams()
    inputs = paths.assert_inputs()
    mods = paths.module_shas()
    log("input SHA-256s asserted")
    A = paths.allowed()
    ck = Checks()
    s4 = np.load(paths.P["seed42_arrays"])
    rcz = np.load(paths.P["cand_rc"])
    rc_tau = json.loads(paths.P["rc_tau"].read_text())["taus"]

    # ------------------------------------------------ bundle
    b = build(42, A, log)
    ck.add("bundle.head_identity_told_oracle", all(b.head_checks.values()), detail=b.head_checks)
    ck.add("bundle.cl_pair_parity_eq_stored", np.array_equal(b.cl, s4["cl"]) and np.array_equal(b.pair_index, s4["pair_index"])
           and np.array_equal(b.parity, s4["parity"]))
    ck.add("bundle.B_per_anchor_eq_stored", pa_equal(b.pB, stored_pa(s4, "B")), B_r1=stats.mean_pp(b.pB["r1"]))
    ck.add("bundle.Bp0_per_anchor_eq_stored", pa_equal(b.pBp0, stored_pa(s4, "Bp0")), Bp0_r1=stats.mean_pp(b.pBp0["r1"]))
    ck.add("bundle.cosine_rca_eq_stored", pa_equal(b.cos_pa, stored_pa(s4, "cosine")) and pa_equal(b.rca_pa, stored_pa(s4, "rca")))
    ck.add("bundle.B_r1_round3_D10", stats.mean_pp(b.pB["r1"]) == 18.341064453125, value=stats.mean_pp(b.pB["r1"]))
    ck.add("bundle.Bp0_r1_round3_D11", stats.mean_pp(b.pBp0["r1"]) == 18.436686197916664, value=stats.mean_pp(b.pBp0["r1"]))
    ck.add("bundle.v_eq_stored", np.array_equal(np.minimum(b.F["a"][:, 6], b.F["a"][:, 7]), s4["v"]))

    # ------------------------------------------------ reader (R1's and AFF's; the same A0 half-readers)
    rd = core.reader(b.F, b.stack, b.halves)
    for c in core.CONDITIONS:
        ck.add(f"reader.P_{c}_eq_stored", np.array_equal(rd[c]["P"], s4[f"P__{c}"]))
        ck.add(f"reader.margin_{c}_eq_stored", np.array_equal(rd[c]["m"], rcz[f"margin__{c}"])
               and np.array_equal(rd[c]["m"], s4[f"margin__{c}"]))
        ck.add(f"reader.pick_{c}_eq_stored", np.array_equal(rd[c]["pi"].astype(np.int8), rcz[f"pick__{c}"])
               and np.array_equal(rd[c]["pi"], s4[f"pick__{c}"]))
        for d in core.DIRECTIONS:
            ck.add(f"reader.T_{c}_{d}_eq_stored", np.array_equal(rd[c]["T"][d], rcz[f"T__{c}__{d}"]))
    taus = core.taus_from_margins(rd["a"]["m"], rd["b"]["m"])
    ck.add("tau_eq_rc_tau_json", all(float(x) == y for x, y in zip(taus, rc_tau)) and list(rc_tau) == TARGET["taus"]
           and np.array_equal(taus, s4["taus"]) and np.array_equal(taus, rcz["extra__taus"]), taus=taus)
    T = {c: rd[c]["T"] for c in core.CONDITIONS}

    # ------------------------------------------------ R1 = round-1 R-c
    g_r1 = {c: core.gates_r1(rd[c]["m"], taus) for c in core.CONDITIONS}
    ck.add("r1.gates_eq_stored", all(np.array_equal(g_r1[c], s4[f"r1_gate__{c}"]) and np.array_equal(g_r1[c], rcz[f"extra__gate_{c}"])
                                     for c in core.CONDITIONS))
    log("R1 family")
    f_r1 = core.run_family(b.B, T, g_r1, b.parity, A["zscore_rows"])
    cf_, cc_, sg_ = cells_of(f_r1)
    ck.add("r1.cells_sigma", cf_ == TARGET["r1"]["cells_fused"] and cc_ == TARGET["r1"]["cells_cf"] and sg_ == TARGET["r1"]["sigma"]
           and cf_ == s4["r1_fused_cells"].tolist() and cc_ == s4["r1_cf_cells"].tolist() and sg_ == s4["r1_sigma"].tolist(),
           fused=cf_, cf=cc_, sigma=sg_, picks=f_r1.summary(taus))
    ck.add("r1.per_anchor_eq_stored", pa_equal(f_r1.pa_fused, stored_pa(rcz, "fused")) and pa_equal(f_r1.pa_cf, stored_pa(rcz, "cf"))
           and pa_equal(f_r1.pa_fused, stored_pa(s4, "r1_fused")) and pa_equal(f_r1.pa_cf, stored_pa(s4, "r1_cf")))
    r3_order = [("B'(A0)", b.pBp0), ("counterpart", f_r1.pa_cf), ("B", b.pB)]
    rec_r1 = stats.dev_record(f_r1.pa_fused, f_r1.pa_cf, r3_order, None, None, b.cl, b.pair_index, A["cluster_bootstrap"], PAIRS)
    bar_pa = dict(r3_order)[rec_r1["bar_comparator"]]
    ck.add("r1.bar_v_eq_stored", np.array_equal(f_r1.pa_fused["r1"] - bar_pa["r1"], rcz["bar_v"]),
           bar=rec_r1["bar_comparator"], meaning="bar_v = per-anchor fused R@1 minus the bar comparator's R@1")
    ck.add("r1.bar_margin_gain_statistic", rec_r1["bar_comparator"] == "counterpart"
           and pci(rec_r1["bar_margin"]) == TARGET["r1"]["bar_margin"] and pci(rec_r1["gain_statistic"]) == TARGET["r1"]["gain_statistic"],
           bar_margin=pci(rec_r1["bar_margin"]), gain_statistic=pci(rec_r1["gain_statistic"]))

    # ------------------------------------------------ AFF = round 3's targets
    g_aff = {c: core.gates_aff(rd[c]["m"], rd[c]["pi"], taus) for c in core.CONDITIONS}
    ck.add("aff.gates_eq_stored", all(np.array_equal(g_aff[c], s4[f"aff_gate__{c}"]) for c in core.CONDITIONS))
    open0 = [int(g_aff[c][0].sum()) for c in core.CONDITIONS]
    ck.add("aff.open_tau0", open0 == TARGET["aff"]["open_tau0"], open=open0)
    log("AFF family")
    f_aff = core.run_family(b.B, T, g_aff, b.parity, A["zscore_rows"])
    cf_, cc_, sg_ = cells_of(f_aff)
    ck.add("aff.cells_sigma", cf_ == TARGET["aff"]["cells_fused"] and cc_ == TARGET["aff"]["cells_cf"] and sg_ == TARGET["aff"]["sigma"]
           and cf_ == s4["aff_fused_cells"].tolist() and cc_ == s4["aff_cf_cells"].tolist() and sg_ == s4["aff_sigma"].tolist(),
           fused=cf_, cf=cc_, sigma=sg_, picks=f_aff.summary(taus))
    ck.add("aff.per_anchor_eq_stored", pa_equal(f_aff.pa_fused, stored_pa(s4, "aff_fused")) and pa_equal(f_aff.pa_cf, stored_pa(s4, "aff_cf")))
    bp1 = stored_pa(s4, "Bp1")
    r3_order = [("B'(A0)", b.pBp0), ("counterpart", f_aff.pa_cf), ("B", b.pB)]
    rec_aff = stats.dev_record(f_aff.pa_fused, f_aff.pa_cf, r3_order, None, bp1, b.cl, b.pair_index, A["cluster_bootstrap"], PAIRS)
    ta = TARGET["aff"]
    got = {"fused_r1": rec_aff["fused_r1"], "cf_r1": rec_aff["cf_r1"], "bar_comparator": rec_aff["bar_comparator"],
           "bar_margin": pci(rec_aff["bar_margin"]), "margin_vs_cf": pci(rec_aff["margin_vs_cf"]),
           "gain_statistic": pci(rec_aff["gain_statistic"]), "either_change_vs_cf": rec_aff["either_change_vs_cf"],
           "per_pair_bar_margin": [rec_aff["per_pair_bar_margin"][p] for p in PAIRS],
           "minus_Bp_A1": pci(rec_aff["beside_Bp_A1"]["minus_Bp_A1"]), "Bp_A1_r1": rec_aff["beside_Bp_A1"]["Bp_A1_r1"]}
    for k, v in got.items():
        ck.add(f"aff.{k}", v == ta[k], value=v, target=ta[k])
    m_r1 = stats.point_ci(f_aff.pa_fused["r1"] - f_r1.pa_fused["r1"], b.cl, A["cluster_bootstrap"])
    bar_r1 = dict([("B'(A0)", b.pBp0), ("counterpart", f_r1.pa_cf), ("B", b.pB)])[rec_r1["bar_comparator"]]
    m_bar = stats.point_ci((f_aff.pa_fused["r1"] - b.pBp0["r1"]) - (f_r1.pa_fused["r1"] - bar_r1["r1"]), b.cl, A["cluster_bootstrap"])
    ck.add("aff.minus_r1_fused", pci(m_r1) == ta["minus_r1_fused"], value=pci(m_r1))
    ck.add("aff.minus_r1_bar", pci(m_bar) == ta["minus_r1_bar"], value=pci(m_bar))
    ck.add("aff.D10_all_true", rec_aff["D10"]["clears"], D10=rec_aff["D10"])

    # ------------------------------------------------ item 3: placement function = the CLIP heads
    st, sel = b.scorer_train, b.ctx.selection
    item3 = {}
    for m, Ff in (("txt", b.ctx.data.txt_features), ("img", b.ctx.data.img_features)):
        log(f"item 3: placement function on CLIP {m} features")
        r = placement(Ff, unit, b.lab, st, sel, 300)
        same = (r["post"].dtype == np.float32 and r["post"].shape == (len(b.ctx.groups), 41)
                and np.array_equal(r["post"], b.heads[m], equal_nan=True))
        ck.add(f"item3.{m}_posterior_eq_fit_one_head", same)
        ck.add(f"item3.{m}_accuracy", round(r["accuracy"], 2) == TARGET["heads"][m]
               and r["accuracy"] == b.head_prov["heldout_accuracy"][m], accuracy=r["accuracy"])
        ck.add(f"item3.{m}_classes", np.array_equal(r["classes"], np.arange(41)))
        ck.add(f"item3.{m}_draw_sha", r["draw_sha"] == DRAW_SHA == b.head_prov["draw_rows_sha256"])
        ck.add(f"item3.{m}_mapping", check_mapping(st, r["draw"]))
        item3[m] = {"n_iter": r["n_iter"], "accuracy": r["accuracy"]}
        del r
    ck.add("item3.fit_one_head_record", b.head_prov["n_classes"] == 41 and b.head_prov["uniform"] == 2.4390243902439024
           and round(b.head_prov["check_majority_share"], 2) == 6.41, record=b.head_prov)

    # ------------------------------------------------ item 4: the CLIP placement through the candidate path
    log("item 4: CLIP placement through the extension and the candidates")
    ext = extend(b, b.post["affect"]["txt"], "clip", A)
    ck.add("item4.stack_Q_eq_stack", all(np.array_equal(ext.stack[d], b.stack[d]) for d in core.DIRECTIONS))
    ck.add("item4.F_Q_eq_F", all(np.array_equal(ext.F[c], b.F[c]) for c in core.CONDITIONS))
    ck.add("item4.Bprime_Q_scores_eq_Bp0", scores_equal(ext.B, b.Bp0))
    ck.add("item4.Bprime_Q_per_anchor_eq_Bp0", pa_equal(ext.pB, b.pBp0) and pa_equal(ext.pB, stored_pa(s4, "Bp0")))
    cand = candidates(b, ext, rd, taus, A)
    item4 = {}
    for name, v in cand.items():
        rdv = v["reader"]
        ck.add(f"item4.{name}.reader_eq_aff", all(np.array_equal(rdv[c]["P"], rd[c]["P"]) and np.array_equal(rdv[c]["m"], rd[c]["m"])
                                                  and np.array_equal(rdv[c]["pi"], rd[c]["pi"])
                                                  and all(np.array_equal(rdv[c]["T"][d], rd[c]["T"][d]) for d in core.DIRECTIONS)
                                                  for c in core.CONDITIONS))
        ck.add(f"item4.{name}.tau_eq", all(float(x) == y for x, y in zip(v["taus"], TARGET["taus"])), taus=v["taus"])
        ck.add(f"item4.{name}.gates_eq_aff", all(np.array_equal(v["gates"][c], g_aff[c]) for c in core.CONDITIONS))
        fam = v["family"]
        cf_, cc_, sg_ = cells_of(fam)
        ck.add(f"item4.{name}.cells_sigma", cf_ == ta["cells_fused"] and cc_ == ta["cells_cf"] and sg_ == ta["sigma"],
               fused=cf_, cf=cc_, sigma=sg_)
        ck.add(f"item4.{name}.per_anchor_eq_aff", pa_equal(fam.pa_fused, stored_pa(s4, "aff_fused"))
               and pa_equal(fam.pa_cf, stored_pa(s4, "aff_cf")) and pa_equal(fam.pa_fused, f_aff.pa_fused))
        rec = stats.dev_record(fam.pa_fused, fam.pa_cf, comparators(ext, b, fam), f_aff.pa_fused, bp1, b.cl, b.pair_index,
                               A["cluster_bootstrap"], PAIRS)
        same = (rec["bar_comparator"] == "B'_Q" and rec["fused_r1"] == ta["fused_r1"] and rec["cf_r1"] == ta["cf_r1"]
                and pci(rec["bar_margin"]) == ta["bar_margin"] and pci(rec["margin_vs_cf"]) == ta["margin_vs_cf"]
                and pci(rec["gain_statistic"]) == ta["gain_statistic"] and rec["either_change_vs_cf"] == ta["either_change_vs_cf"]
                and [rec["per_pair_bar_margin"][p] for p in PAIRS] == ta["per_pair_bar_margin"]
                and pci(rec["beside_Bp_A1"]["minus_Bp_A1"]) == ta["minus_Bp_A1"])
        ck.add(f"item4.{name}.record_eq_aff", same, bar_comparator=rec["bar_comparator"], fused_r1=rec["fused_r1"])
        ck.add(f"item4.{name}.D10_all_true", rec["D10"]["clears"])
        ck.add(f"item4.{name}.delta_zero", rec["delta_vs_aff"]["delta_int"] == 0, delta=rec["delta_vs_aff"])
        ck.add(f"item4.{name}.open_tau0", [v["open_tau0"][c] for c in core.CONDITIONS] == ta["open_tau0"])
        item4[name] = {"bar_comparator": rec["bar_comparator"], "comparator_order": list(COMPARATOR_ORDER),
                       "comparator_means": rec["comparator_means"], "delta_int": rec["delta_vs_aff"]["delta_int"],
                       "per_anchor_sha256": {"fused": sha_pa(fam.pa_fused), "cf": sha_pa(fam.pa_cf)}}
    ck.add("item4.bundle_post_unchanged", b.post["affect"]["txt"] is b.heads["txt"])

    # ------------------------------------------------ optional: round 1's bundle
    r1b = None
    if args.round1_bundle:
        log("round 1's common.load_bundle() (its console output is round 1's own)")
        rb = paths.load_bundle_round1()
        r1b = {"B": scores_equal(rb.B, b.B), "Bp_A0": scores_equal(rb.Bp["A0"], b.Bp0),
               "t_n1u": scores_equal(rb.t_n1u, b.t_n1u),
               "post": all(np.array_equal(rb.post[h][m], b.post[h][m], equal_nan=True) for h in core.A0 for m in ("img", "txt")),
               "cl_pair_parity": np.array_equal(rb.cl, b.cl) and np.array_equal(rb.pair_index, b.pair_index)
               and np.array_equal(rb.parity, b.parity),
               "pBp_A1_eq_stored_Bp1": pa_equal(rb.pBp["A1"], bp1)}
        ck.add("round1_bundle.equal", all(r1b.values()), detail=r1b)
        del rb

    # ------------------------------------------------ write
    result = {"stage": "A", "seed": 42, "started": started, "finished": paths.now_ams(), "runtime_s": time.time() - T0,
              "passed": ck.all_ok(), "failed": ck.failed(), "checks": ck.items, "inputs_sha256": inputs,
              "module_sha256": mods, "taus": _js(taus), "item3": item3, "item4": item4,
              "aff_picks": f_aff.summary(taus), "r1_picks": f_r1.summary(taus),
              "versions": versions(), "round1_bundle": r1b}
    out_path.write_text(json.dumps(_js(result), indent=1))
    log(f"wrote {out_path.relative_to(paths.ROOT)} sha256 {paths.sha_file(out_path)}; passed={ck.all_ok()} "
        f"failed={ck.failed()}")


def versions():
    import sklearn
    import torch
    return {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
            "python": sys.version.split()[0]}


if __name__ == "__main__":
    main()
