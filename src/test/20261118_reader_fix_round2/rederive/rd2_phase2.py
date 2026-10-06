"""Phase 2 of the round-2 re-derivation: every candidate number of rule §5 / §7 on seed 42, our own code.

Refuses to run until out/PHASE2_AUTHORISED exists (created only after the controller's message that the main run's
regression check has passed). For R1, R2, R3 on A0 (probabilities from our own phase-1 files): T, margins, picks, the
896 cells (k_top 13, 5, 3, 2 x tau_0..3 x 56 weight cells), top-k sets from B, restriction, gates, the control's
sigma*, the integer min-margin cross-fit of the fused reader and the max-rho cross-fit of the matched counterpart, the
per-anchor arrays, B, B'(A0), the bar comparator, bar margin, gain statistic and their intervals, D12's clauses, per
aspect pair; R1 minus round-1 R-c; R2 - R1 and R3 - R1; pick accuracy and gate shares (diagnostics); the rule's
carry (§5 item 4). Optional --a1 READER: the A1 ablation of §4.8 for that reader (descriptive).
Writes out/rd2_phase2.{json,npz} (and out/rd2_phase2_A1.{json,npz})."""
import json
import sys
import time

import numpy as np

import rd2_core as K
import rd2_family as FAM
from rd2_core import COND, DIRS, METRICS, RC

READERS = ("R1", "R2", "R3")


def reader_probs(config):
    """{name: P} from our phase-1 files; R3 = R1 when k* = 4."""
    zr = np.load(K.OUT / "rd2_regress.npz") if config == "A0" else None
    z2 = np.load(K.OUT / f"rd2_r2_{config}.npz")
    j3 = json.loads((K.OUT / f"rd2_r3_{config}.json").read_text())
    P = {"R1": {c: (zr[f"R1__P__{c}"] if zr is not None else z2[f"R1__P__{c}"]) for c in COND},
         "R2": {c: z2[f"P__{c}"] for c in COND}}
    if j3["k_star"] == 4:
        P["R3"] = None
    else:
        z3 = np.load(K.OUT / f"rd2_r3_{config}.npz")
        P["R3"] = {c: z3[f"P__{c}"] for c in COND}
    taus_phase1 = {"R2": [float(x) for x in z2["taus"]],
                   "R3": (None if P["R3"] is None else [float(x) for x in np.load(K.OUT / f"rd2_r3_{config}.npz")["taus"]])}
    if config == "A0":
        taus_phase1["R1"] = [float(x) for x in zr["R1__taus"]]
    return P, taus_phase1, j3["k_star"]


def pick_accuracy(picks, config, cache):
    parts = K.CONFIGS[config]
    pi = cache["pair_index"]
    told = {c: np.array([parts.index(K.TOLD[config][K.PAIRS[i][j]]) for i in pi]) for j, c in enumerate(COND)}
    corr = {c: np.asarray(picks[c]) == told[c] for c in COND}
    v = 0.5 * (corr["a"].astype(np.float64) + corr["b"].astype(np.float64))
    return {"correct_share": K.point_ci(v, cache["cl"]),
            "per_pair_condition": {p: {c: 100 * float(corr[c][pi == i].mean()) for c in COND}
                                   for i, p in enumerate(K.PAIR_NAMES)},
            "per_pair": {p: K.point_ci(v[pi == i], cache["cl"][pi == i]) for i, p in enumerate(K.PAIR_NAMES)},
            "chance": 100.0 / len(parts)}


def evaluate_reader(cache, P, config, taus_expected=None):
    T, picks, margins = FAM.reader_parts(cache, P, config)
    taus = K.thresholds(margins["a"], margins["b"])
    if taus_expected is not None and taus != taus_expected:
        raise AssertionError(f"tau differs from phase 1: {taus} vs {taus_expected}")
    rec, pn, pc, extra = FAM.run_family(cache, T, margins, taus, ktops=K.KTOPS)
    ev, bar_v, gain_v = K.evaluate(pn, pc, cache["pB"], cache["pBp"][config], cache["cl"], cache["pair_index"])
    ev["family"] = rec
    ev["pick_accuracy"] = pick_accuracy(picks, config, cache)
    return ev, pn, pc, bar_v, gain_v, picks, margins, T, extra


def main(a1_reader=None):
    t0 = time.time()
    RC.assert_rule()
    if K.allowed_ktops() != K.KTOPS:
        raise SystemExit("phase 2 is not authorised (out/PHASE2_AUTHORISED missing)")
    cache = K.load_cache()
    cl = cache["cl"]
    if a1_reader:
        return main_a1(a1_reader, cache, t0)
    RC.assert_inputs(["results/cand_Rc_Rb_expected_A0.npz"])
    rc = np.load(RC.r1_path("results/cand_Rc_Rb_expected_A0.npz"))
    P, taus1, kstar = reader_probs("A0")
    out = {"k_star": kstar, "candidates": {}}
    arr = {}
    npz = {}
    for name in READERS:
        if P[name] is None:
            out["candidates"][name] = "R3 is R1 (k* = 4); not evaluated a second time"
            continue
        ev, pn, pc, bar_v, gain_v, picks, margins, T, extra = evaluate_reader(cache, P[name], "A0", taus1.get(name))
        out["candidates"][name] = ev
        arr[name] = (pn, pc, bar_v)
        for m in METRICS:
            npz[f"{name}__fused__{m}"], npz[f"{name}__cf__{m}"] = pn[m], pc[m]
        npz[f"{name}__bar_v"], npz[f"{name}__gain_v"] = bar_v, gain_v
        for k in ("rho_f", "gam_f", "rho_c", "crit_f", "ctrl_rho"):
            npz[f"{name}__{k}"] = extra[k]
        for c in COND:
            npz[f"{name}__P__{c}"], npz[f"{name}__pick__{c}"], npz[f"{name}__margin__{c}"] = P[name][c], picks[c], margins[c]
        K.log(f"{name}: bar {ev['bar_margin']} ({ev['bar_comparator']}), gain {ev['gain_statistic']}, "
              f"clauses {ev['clauses']}, fused cells {[ev['family']['fused_cells'][h]['cell'] for h in ('0', '1')]}, "
              f"cf cells {[ev['family']['counterpart_cells'][h]['cell'] for h in ('0', '1')]}", t0)
    # R1 minus round-1 R-c; R2 - R1, R3 - R1
    pn1, pc1, b1 = arr["R1"]
    out["R1_minus_round1_Rc"] = {
        "fused_r1": K.point_ci(pn1["r1"] - rc["fused__r1"], cl),
        "margin_r1": K.point_ci((pn1["r1"] - pc1["r1"]) - (rc["fused__r1"] - rc["cf__r1"]), cl),
        "bar_margin_r1": K.point_ci(b1 - rc["bar_v"], cl)}
    for name in ("R2", "R3"):
        if name not in arr:
            continue
        pn, pc, b = arr[name]
        out[f"{name}_minus_R1"] = {"fused_r1": K.point_ci(pn["r1"] - pn1["r1"], cl),
                                   "margin_r1": K.point_ci((pn["r1"] - pc["r1"]) - (pn1["r1"] - pc1["r1"]), cl),
                                   "bar_margin_r1": K.point_ci(b - b1, cl)}
    # rule application (D12, §5 item 4)
    evs = {n: out["candidates"][n] for n in READERS if n in arr}
    E = [n for n in READERS if n in evs and evs[n]["clauses"]["clears_bar"]]
    carried = None
    if E:
        M = max(evs[n]["bar_margin"]["point"] for n in E)
        tied = [n for n in E if evs[n]["bar_margin"]["point"] >= M - 0.05]
        carried = tied[0]
    best = None
    for n in READERS:
        if n in evs and (best is None or evs[n]["bar_margin"]["point"] > evs[best]["bar_margin"]["point"]):
            best = n
    out["rule"] = {"clear_the_bar": E, "carried": carried, "best_development_candidate": best,
                   "summary": {n: {"bar_margin": evs[n]["bar_margin"], "comparator": evs[n]["bar_comparator"],
                                   "gain_statistic": evs[n]["gain_statistic"], "clauses": evs[n]["clauses"]}
                               for n in evs}}
    out["runtime_s"] = round(time.time() - t0, 1)
    out["provenance"] = K.provenance()
    K.save_json(K.OUT / "rd2_phase2.json", out)
    np.savez(K.OUT / "rd2_phase2.npz", **npz)
    K.log(f"rule: {out['rule']['clear_the_bar']} clear; carried {carried}; best {best}", t0)


def main_a1(reader, cache, t0):
    """§4.8: the named reader built on A1 (descriptive), with B'(A1) and its own counterpart; A1 - A0 paired."""
    P1, taus1, kstar1 = reader_probs("A1")
    if reader == "R3" and P1["R3"] is None:
        reader_used = "R1"
    else:
        reader_used = reader
    ev, pn, pc, bar_v, gain_v, picks, margins, T, extra = evaluate_reader(cache, P1[reader_used], "A1",
                                                                          taus1.get(reader_used))
    z0 = np.load(K.OUT / "rd2_phase2.npz")
    a0_r1 = z0[f"{reader}__fused__r1"]
    a0_bar = z0[f"{reader}__bar_v"]
    out = {"reader": reader, "reader_used_on_A1": reader_used, "k_star_A1": kstar1, "evaluation": ev,
           "A1_minus_A0": {"fused_r1": K.point_ci(pn["r1"] - a0_r1, cache["cl"]),
                           "bar_margin_r1": K.point_ci(bar_v - a0_bar, cache["cl"])},
           "runtime_s": round(time.time() - t0, 1), "provenance": K.provenance()}
    K.save_json(K.OUT / "rd2_phase2_A1.json", out)
    np.savez(K.OUT / "rd2_phase2_A1.npz", **{f"fused__{m}": pn[m] for m in METRICS}, **{f"cf__{m}": pc[m] for m in METRICS},
             bar_v=bar_v, gain_v=gain_v)
    K.log(f"A1 {reader}: bar {ev['bar_margin']} ({ev['bar_comparator']}), A1-A0 {out['A1_minus_A0']}", t0)


if __name__ == "__main__":
    a1 = sys.argv[sys.argv.index("--a1") + 1] if "--a1" in sys.argv else None
    main(a1)
