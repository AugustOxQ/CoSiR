"""Round 4 rule check (read only, seed 42 only, CPU). Recomputes the rule's seed-42 constants and regression targets:
v, v75 (D5), the A0/A1 feature identity (D3), B'(A1) (D4, through uniform_probe_scores with post keyed "csd"), the A1
reader's probabilities and picks against round 2's cand_R1_A1.npz (§5 item 4), R1, AFF and R1 x a_v (IMGABST_q75)
through round 3's float32 path (r3_fusion.run_family) against the rule's targets (§5 items 2 and 3), and a float32 vs
float64 cell-by-cell comparison for R1 x a_v. Computes NO number of the candidates V4, V2 or V24.

Writes only rule_check/rc4_seed42_check.json. Run from /project/CoSiR:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261122_round4_aff_vetoes/rule_check/rc4_seed42_check.py
"""
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parents[1]
R3DIR = T / "20261121_round3_affect_gate"
BS = T / "20261120_r1_levers_brainstorm"
R2DIR = T / "20261118_reader_fix_round2"
R1DIR = T / "20261117_reader_fix_csd"
sys.path.insert(0, str(R3DIR))
import r3_common as R3  # noqa: E402
import r3_bundle as RB  # noqa: E402
import r3_fusion as FU  # noqa: E402

C, K, rb, rbe, rf, F = R3.C, R3.K, R3.rb, R3.rbe, R3.rf, R3.F
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_nested import NESTED_A, NESTED_U, _zdict  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores  # noqa: E402

A1 = ("affect", "image", "caption", "csd")
V75_RULE = 0.021043562795966864
BPA1_RULE = 18.804931640625
out = {}
t0 = time.time()


def log(msg):
    print(f"[rc4 {time.time() - t0:5.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 0. seed guard of round 3's bundle (no data read)
guard = {}
for s, sm in ((52, False), (53, False), (54, False), (9001, True), (42, False)):
    try:
        RB._check_seed(s, sm)
        guard[f"{s}{'_smoke' if sm else ''}"] = "allowed"
    except ValueError as e:
        guard[f"{s}{'_smoke' if sm else ''}"] = f"ValueError: {e}"
out["r3_bundle_check_seed"] = guard
log(f"r3_bundle._check_seed: {guard}")

# ---------------------------------------------------------------- 1. cache-only checks
z = np.load(BS / "cache" / "bs_cache.npz")
fa, fb = z["feat_A1__a"], z["feat_A1__b"]
v_a = np.minimum(fa[:, 6], fa[:, 7])
v_b = np.minimum(fb[:, 6], fb[:, 7])
v75 = float(np.percentile(v_a, 75))
out["D5_cache"] = {"v_b_equals_v_a": bool(np.array_equal(v_a, v_b)), "v75": v75, "v75_repr": repr(v75),
                   "v75_equals_rule_literal": v75 == V75_RULE, "n_below": int((v_a < v75).sum()),
                   "n_equal": int((v_a == v75).sum()), "n": int(len(v_a)), "v_dtype": str(v_a.dtype),
                   "S_img_b_equals_C_img_a": bool(np.array_equal(fb[:, 6], fa[:, 7])),
                   "C_img_b_equals_S_img_a": bool(np.array_equal(fb[:, 7], fa[:, 6]))}
srt = np.sort(v_a)
out["D5_cache"]["neighbours"] = [float(srt[9215]), float(srt[9216])]
out["D5_cache"]["gap_to_neighbours"] = [v75 - float(srt[9215]), float(srt[9216]) - v75]
log(f"D5 (cache): {out['D5_cache']}")
out["D4_cache_BpA1_mean_r1"] = 100 * float(np.mean(z["pBpA1__r1"]))
out["D4_cache_equals_rule"] = out["D4_cache_BpA1_mean_r1"] == BPA1_RULE
log(f"B'(A1) mean R@1 (cache) {out['D4_cache_BpA1_mean_r1']!r} == rule: {out['D4_cache_equals_rule']}")

# ---------------------------------------------------------------- 2. A1 reader against cand_R1_A1.npz (§5 item 4)
pkA1, recA1, npzA1 = rb.load_readers("A1", False)
if npzA1 is not None:
    npzA1.close()
out["A1_reader"] = {"feature_names_match_A1": pkA1["feature_names"] == rf.feature_names(A1),
                    "groupings": list(pkA1["groupings"]), "sklearn": pkA1["sklearn_version"]}
cand = np.load(R2DIR / "results/cand_R1_A1.npz")
PA1 = {c: rf.average_probs(rbe.half_reader_probs(pkA1, z[f"feat_A1__{c}"], 4)) for c in CONDITIONS}
pick = {c: rf.picks_and_margins(PA1[c])[0] for c in CONDITIONS}
for c in CONDITIONS:
    P = PA1[c]
    top = P.max(axis=1, keepdims=True)
    n_tied = int(((P == top).sum(axis=1) > 1).sum())
    out["A1_reader"][c] = {
        "probs_equal_stored": bool(np.array_equal(P, cand[f"probs__{c}"])),
        "probs_max_abs_diff": float(np.max(np.abs(P - cand[f"probs__{c}"]))),
        "pick_equal_stored": bool(np.array_equal(pick[c], cand[f"pick__{c}"].astype(np.int64))),
        "stored_pick_dtype": str(cand[f"pick__{c}"].dtype),
        "stored_pick_is_argmax_of_stored_probs": bool(np.array_equal(cand[f"probs__{c}"].argmax(1),
                                                                     cand[f"pick__{c}"].astype(np.int64))),
        "exact_argmax_ties": n_tied,
        "cache_P_R1A1_equals_stored": bool(np.array_equal(z[f"P_R1A1__{c}"], cand[f"probs__{c}"]))}
out["A1_reader"]["anchor_parity_pair_equal_cache"] = all(
    bool(np.array_equal(cand[k], z[k])) for k in ("anchor_group", "pair_index", "parity"))
log(f"A1 reader vs cand_R1_A1: {out['A1_reader']}")

# ---------------------------------------------------------------- 3. round 3's bundle on seed 42 + the A1 extension
b = RB.build_bundle(42, False)
ctx, ep = b.ctx, b.ctx.pooled
before = {k: (getattr(b, k)) for k in ("F", "stack", "Bp", "pBp", "B", "pB")}


def _load_step1():
    spec = importlib.util.spec_from_file_location("run_step1", T / "20261116_grouping_step1_style/run_step1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_step1"] = mod
    spec.loader.exec_module(mod)
    return mod


with contextlib.redirect_stdout(io.StringIO()):
    rs1 = _load_step1()
zh = np.load(T / "20261116_grouping_step1_style/results/step1_heads_style.npz")
assert np.array_equal(zh["selection"], ctx.selection)
csd = {"img": rs1.full_post(zh, "style_csd__img", ctx), "txt": rs1.full_post(zh, "style_csd__txt", ctx)}
assert np.isfinite(csd["img"][ctx.selection]).all() and np.isfinite(csd["txt"][ctx.selection]).all()
post4 = {**b.post, "csd": csd}
F1, fchk = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post4), A1)
d3 = {"seed42_features_checks": fchk}
for c in CONDITIONS:
    d3[c] = {"A1_features_equal_cache": bool(np.array_equal(F1[c], z[f"feat_A1__{c}"])),
             "first18_equal_A0_features": bool(np.array_equal(F1[c][:, :18], b.F[c])),
             "shape": list(F1[c].shape), "dtype": str(F1[c].dtype)}
v_bundle = np.minimum(F1["a"][:, 6], F1["a"][:, 7])
d3["v_from_bundle_equals_cache_v"] = bool(np.array_equal(v_bundle, v_a))
d3["v_from_A0_F_equals"] = bool(np.array_equal(np.minimum(b.F["a"][:, 6], b.F["a"][:, 7]), v_a))
stack4 = C.grouping_stack(post4, ep, A1)
d3["stack4_equal_cache"] = all(bool(np.array_equal(stack4[d], z[f"stack4__{d}"])) for d in DIRECTIONS)
out["D3"] = d3
log(f"D3: {d3}")

t6u_csd = uniform_probe_scores(post4, ep, A1)
post_s1 = {"affect": b.post["affect"], "image": b.post["image"], "caption": b.post["caption"], "style_csd": csd}
t6u_s1 = uniform_probe_scores(post_s1, ep, ("affect", "image", "caption", "style_csd"))
d4 = {"t6u_csd_equals_t6u_style_csd": all(bool(np.array_equal(t6u_csd[c][d], t6u_s1[c][d]))
                                         for c in CONDITIONS for d in DIRECTIONS),
      "t6u_dtype": str(np.asarray(t6u_csd["a"]["i2t"]).dtype)}
BpA1, _ = crossfit_condition_free(ctx.cos, b.t_n1u, t6u_csd, ctx.parity)
pBpA1 = per_anchor(BpA1)
d4["per_anchor_equal_cache"] = {m: bool(np.array_equal(np.asarray(pBpA1[m]), z[f"pBpA1__{m}"])) for m in METRICS}
d4["mean_r1"] = 100 * float(np.mean(pBpA1["r1"]))
d4["mean_r1_equals_rule"] = d4["mean_r1"] == BPA1_RULE
d4["BpA1_mean_r1_minus_BpA0"] = d4["mean_r1"] - 100 * float(np.mean(b.pBp["r1"]))
out["D4"] = d4
log(f"D4: {d4}")
after_same = all(getattr(b, k) is before[k] for k in before)
out["A1_extension_left_bundle_fields_untouched_identity"] = after_same

# ---------------------------------------------------------------- 4. R1, AFF and R1 x a_v through r3_fusion (float32)
taus = R3.assert_taus()
rd = FU.reader(b)
rc_ = np.load(R1DIR / "results/cand_Rc_Rb_expected_A0.npz")
out["R1_reader_equal_stored"] = {
    "T": all(np.array_equal(rd["T"][c][d], rc_[f"T__{c}__{d}"]) for c in CONDITIONS for d in DIRECTIONS),
    "m": all(np.array_equal(rd["m"][c], rc_[f"margin__{c}"]) for c in CONDITIONS),
    "pick": all(np.array_equal(rd["pick"][c], rc_[f"pick__{c}"].astype(np.int64)) for c in CONDITIONS)}
g_r1 = FU.gates_r1(rd["m"], taus)
g_aff = FU.gates_aff(rd["m"], rd["pick"], taus)
a_v = (v_bundle < V75_RULE).astype(np.float32)
g_img = [{c: (g[c] * a_v).astype(np.float32) for c in CONDITIONS} for g in g_r1]
one = np.ones(b.n, np.float32)
g_aff_x1 = [{c: (g[c] * one * one).astype(np.float32) for c in CONDITIONS} for g in g_aff]
out["factor_one_returns_AFF_gates"] = all(np.array_equal(g_aff_x1[t][c], g_aff[t][c]) for t in range(4)
                                          for c in CONDITIONS)
cl, pi = b.cl, b.pair_index


def evaluate(label, gates):
    fam = FU.run_family(b, rd["T"], gates)
    pn, pc = fam["fused"], fam["cf"]
    bar_v, bar = C.bar_info(pn, pc, b.pBp, b.pB, cl, pi)
    d = C.diff3(pn, pc, cl)
    res = {"fused_r1": 100 * float(np.mean(pn["r1"])), "cf_r1": 100 * float(np.mean(pc["r1"])),
           "comparator": bar["comparator"], "bar": bar["r1"], "margin": d["r1"], "gain": d["gain"],
           "either": d["either"], "per_pair_bar": {p: v["point"] for p, v in bar["per_pair_r1"].items()},
           "fused_cells": [int(fam["fpick"][h]) for h in (0, 1)], "cf_cells": [int(fam["cpick"][h]) for h in (0, 1)],
           "cells_described": {"fused": [list(F.cell_values(fam["fpick"][h])) for h in (0, 1)],
                               "cf": [list(F.cell_values(fam["cpick"][h])) for h in (0, 1)]},
           "sigma": [fam["sigma"][h] for h in (0, 1)], "details": fam["details"],
           "open_tau0": {c: FU.open_count(gates[0], c) for c in CONDITIONS}}
    log(f"{label}: fused {res['fused_r1']!r} cf {res['cf_r1']!r} {res['comparator']} cells {res['fused_cells']} / "
        f"{res['cf_cells']} sigma {res['sigma']}")
    return res, (pn, pc, bar_v)


out["R1"], arrR1 = evaluate("R1", g_r1)
out["AFF"], arrAFF = evaluate("AFF", g_aff)
out["IMGABST_q75"], arrIMG = evaluate("R1 x a_v", g_img)
out["AFF_minus_R1"] = {"fused_r1": C.point_ci(np.asarray(arrAFF[0]["r1"], float) - np.asarray(arrR1[0]["r1"], float), cl),
                       "bar": C.point_ci(arrAFF[2] - arrR1[2], cl)}

# ---------------------------------------------------------------- 5. compare with the rule's / brainstorm's targets
b4 = json.loads((BS / "results/bs_04_readers.json").read_text())["results"]["IMGABST_q75"]
I = out["IMGABST_q75"]
cmp = {"fused_r1": I["fused_r1"] == b4["fused_r1"] == 19.059244791666664,
       "cf_r1": I["cf_r1"] == b4["cf_r1"] == 18.49365234375,
       "comparator": I["comparator"] == b4["comparator"] == "counterpart",
       "bar": I["bar"]["point"] == b4["bar"]["point"] and I["bar"]["ci95"] == b4["bar"]["ci95"]
       and b4["bar"]["ci95"] == [0.3448683992591827, 0.79821625538382] and b4["bar"]["point"] == 0.5655924479166667,
       "gain": I["gain"]["point"] == b4["gain"]["point"] == 2.878824869791667
       and I["gain"]["ci95"] == b4["gain"]["ci95"] == [2.554983173204304, 3.2105685950938248],
       "either": I["either"]["point"] == b4["either"]["point"] == -1.7476399739583333,
       "per_pair_bar": I["per_pair_bar"] == b4["per_pair_bar"] == {"emotion__style": 0.677490234375,
                                                                     "emotion__genre": 1.28173828125,
                                                                     "style__genre": -0.262451171875},
       "fused_cells": I["fused_cells"] == [117, 119], "cf_cells": I["cf_cells"] == [58, 67],
       "brainstorm_fused_cells_to_numbers": [(int(c[0]) * 7 + NESTED_U.index(c[1])) * 8 + NESTED_A.index(c[2])
                                            for c in b4["fused_cells"]],
       "brainstorm_cf_cells_to_numbers": [(int(c[0]) * 7 + NESTED_U.index(c[1])) * 8 + NESTED_A.index(c[2])
                                         for c in b4["cf_cells"]],
       "sigma": I["sigma"] == [0.0, 0.0]}
out["IMGABST_vs_rule_and_brainstorm"] = cmp
A = out["AFF"]
cmpA = {"fused_r1": A["fused_r1"] == 19.136555989583336, "cf_r1": A["cf_r1"] == 18.39599609375,
        "comparator": A["comparator"] == "B_prime",
        "bar": A["bar"]["point"] == 0.6998697916666667 and A["bar"]["ci95"] == [0.4598852740816973, 0.9371680126852968],
        "gain": A["gain"]["point"] == 3.110758463541667 and A["gain"]["ci95"] == [2.780005709854805, 3.4559584315470384],
        "cells": A["fused_cells"] == [39, 119] and A["cf_cells"] == [149, 10], "sigma": A["sigma"] == [0.0, 0.0],
        "open_tau0": A["open_tau0"] == {"a": 9941, "b": 3627},
        "aff_minus_r1": out["AFF_minus_R1"]["fused_r1"]["point"] == 0.21769205729166666}
out["AFF_vs_rule"] = cmpA
R = out["R1"]
cmpR = {"cells": R["fused_cells"] == [116, 119] and R["cf_cells"] == [58, 123], "comparator": R["comparator"] == "counterpart",
        "bar": R["bar"]["point"] == 0.4435221354166667 and R["bar"]["ci95"] == [0.21646171563312194, 0.6735669710776852],
        "gain": R["gain"]["point"] == 2.667236328125 and R["gain"]["ci95"] == [2.325087836946873, 3.012361650695922],
        "arrays": all(bool(np.array_equal(np.asarray(arrR1[0][m]), rc_[f"fused__{m}"])) and
                      bool(np.array_equal(np.asarray(arrR1[1][m]), rc_[f"cf__{m}"])) for m in METRICS)
        and bool(np.array_equal(arrR1[2], rc_["bar_v"]))}
out["R1_vs_rule"] = cmpR
log(f"IMGABST vs targets: {cmp}")
log(f"AFF vs targets: {cmpA}")
log(f"R1 vs targets: {cmpR}")

# ---------------------------------------------------------------- 6. float32 vs float64 cell statistics for R1 x a_v
zB, zT = _zdict(b.B), _zdict(rd["T"])
info = F.rank_info(b.B)
gated = {t: K.gated_terms(zT, g_img[t]) for t in range(4)}
G = {t: K.g_cf(gated[t]) for t in range(4)}
fri, fgi, cri = F.cell_statistics(zB, info, gated, G, 1)
ctrl = F.control_choice(zB, b.parity)
par = np.asarray(b.parity)
E = b.n
zB64 = {d: zB["a"][d].numpy().astype(np.float64) for d in DIRECTIONS}
f64 = [np.zeros((224, E), np.int8) for _ in range(3)]
for t in range(4):
    ga = np.asarray(g_img[t]["a"], np.float64)[:, None]
    gb = np.asarray(g_img[t]["b"], np.float64)[:, None]
    M, D = {}, {}
    for d in DIRECTIONS:
        xa = ga * zT["a"][d].numpy().astype(np.float64)
        xb = gb * zT["b"][d].numpy().astype(np.float64)
        M[d], D[d] = 0.5 * (xa + xb), 0.5 * (xa - xb)
    for u, lu in enumerate(NESTED_U):
        for a, la in enumerate(NESTED_A):
            i = (t * 7 + u) * 8 + a
            sf, sc = {}, {}
            for c, sgn in (("a", 1.0), ("b", -1.0)):
                sf[c], sc[c] = {}, {}
                for d in DIRECTIONS:
                    s = (1.0 + lu) * zB64[d]
                    sc[c][d] = s + la * M[d] if la else s
                    if la:
                        s = s + la * M[d]
                        s = s + la * (sgn * D[d])
                    sf[c][d] = s
            pa = per_anchor(sf)
            f64[0][i], f64[1][i] = F.as_int4(pa["r1"]), F.as_int4(pa["gain"])
            f64[2][i] = F.as_int4(per_anchor(sc)["r1"])
diff = {}
for lab, a32, a64 in zip(("fused_r1", "fused_gain", "cf_r1"), (fri, fgi, cri), f64):
    bad = a32 != a64
    diff[lab] = {"cells_with_any_difference": int(bad.any(1).sum()), "entries_differing": int(bad.sum())}
ties = {}
for h in (0, 1):
    tune = par == h
    crit, _, _ = F.fused_criterion(fri, fgi, tune, ctrl[h][1], np.arange(224))
    u = np.sort(np.unique(crit))
    ties[f"fused_h{h}"] = {"max": int(crit.max()), "tied": np.flatnonzero(crit == crit.max()).tolist(),
                           "second": int(u[-2])}
    rr = cri[:, tune].sum(1, dtype=np.int64)
    u = np.sort(np.unique(rr))
    ties[f"cf_h{h}"] = {"max": int(rr.max()), "tied": np.flatnonzero(rr == rr.max()).tolist(), "second": int(u[-2])}
out["IMGABST_float32_vs_float64"] = diff
out["IMGABST_tie_sets"] = ties
out["IMGABST_control"] = {str(h): list(ctrl[h]) for h in (0, 1)}
log(f"float32 vs float64: {diff}; ties {ties}")

(HERE / "rc4_seed42_check.json").write_text(json.dumps(C.jsonable(out), indent=1))
log("written rc4_seed42_check.json")
