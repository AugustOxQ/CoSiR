"""Reads only the seed-42 per-anchor npz files and result jsons listed below; writes figure_data.json."""
import sys, json, os, numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/project/CoSiR")
from src.eval.aspect_metrics import summarize, compare, cluster_bootstrap, _points

P = "/project/CoSiR/src/test/"
E1 = P + "20261030_aspect_baselines/results/per_anchor_seed42.npz"
E3 = P + "20261101_aspect_factor_gonogo/results/per_anchor_select_seed42.npz"
AP = P + "20261105_method_repair_diagnostics/results/per_anchor_pilot_seed42.npz"
Q = P + "20261108_new_method_quick_checks/results/"
CK, AD, N6, N6C = Q + "per_anchor_checks_seed42.npz", Q + "per_anchor_addendum1.npz", Q + "per_anchor_n6_seed42.npz", Q + "per_anchor_n6c_gate.npz"
ADJ = Q + "addendum1.json"
MLLM8 = "/project/CoSiR/src/test/20261106_mllm_probe_8b/20261106_mllm_probe_8b_log.md"
GONOGO = "/project/CoSiR/docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md"
METS = ["r1", "gain", "other", "swap", "strict"]
_cache = {}
def load(f):
    if f not in _cache: _cache[f] = np.load(f, allow_pickle=True)
    return _cache[f]
ref = load(E1)["anchor_group"]
def clusters(f):
    d = load(f)
    g = d["anchor_group"] if "anchor_group" in d.files else ref
    assert len(g) == len(ref) == 12288
    return g
def arrs(f, prefix):
    d = load(f)
    return {m: np.asarray(d[f"{prefix}__{m}"], float) for m in METS}
def per_pair(a):
    pi = load(CK)["pair_index"]
    return {int(p): {m: 100 * float(a[m][pi == p].mean()) for m in ("r1", "gain", "other")} for p in (0, 1, 2)}
def point(label, family, f, prefix, seed=42, note="", kind="aware", show=False, hollow=False, pf=None):
    a = arrs(f, prefix); c = clusters(f)
    s = summarize(a, c)
    eb = _points(cluster_bootstrap(a["r1"] + a["other"], c))
    r1, g, ei = s["r1"]["point"], s["gain"]["point"], eb["point"]
    assert abs((ei + g) / 2 - r1) < 1e-6, (label, ei, g, r1)
    return dict(label=label, family=family, kind=kind, show=show, hollow=hollow, r1=r1, r1_ci=s["r1"]["ci95"], gain=g,
                gain_ci=s["gain"]["ci95"], either=ei, either_ci=eb["ci95"], source_file=f.replace("/project/CoSiR/", ""),
                source_key=prefix + "__{r1,other,gain}", seed=seed, note=note, per_pair=per_pair(a))
def diff(label, family, fa, pa, fb, pb, ctrl, seed=42, note=""):
    a, b, c = arrs(fa, pa), arrs(fb, pb), clusters(fa)
    r, g = compare(a, b, c, "r1"), compare(a, b, c, "gain")
    return dict(label=label, family=family, control=ctrl, d_r1=r["point"], d_r1_ci=r["ci95"], d_gain=g["point"], d_gain_ci=g["ci95"],
                source_file=fa.replace("/project/CoSiR/", ""), source_key=f"{pa} minus {pb}", seed=seed, note=note)

pts = []
add = pts.append
add(point("cosine", "baseline", E1, "cosine", note="CLIP cosine", show=True))
for k in ["diag", "diag_relu", "bilinear", "kissme", "rca", "xing", "wang", "probe", "tip"]:
    add(point(k, "baseline", E1, k, show=k in ("rca", "wang"), note="rca = GO bar" if k == "rca" else "E1 pair-metric baseline"))
add(point("SE_uniform", "baseline", E1, "SE_uniform", note="E1 SE with uniform weights"))
CKJ = Q + "checks_seed42.json"
_ck = json.load(open(CKJ))["factors"]
for k in ("C0", "SE"):  # term-only (agreement term scored alone), not E1's fused values
    t = _ck[k]["term_only"]["agree"]; sm = t["summary"]
    add(dict(label=k, family="factor", kind="aware", show=False, hollow=False, frontier=False, r1=sm["r1"]["point"], r1_ci=sm["r1"]["ci95"],
             gain=sm["gain"]["point"], gain_ci=sm["gain"]["ci95"], either=t["either"]["point"], either_ci=t["either"]["ci95"],
             source_file=CKJ.replace("/project/CoSiR/", ""), source_key=f"factors.{k}.term_only.agree", seed=42,
             note="agreement term scored alone (gain chart only); replaces E1's fused values", per_pair=None))
add(point("A3 (E3 score)", "factor", E3, "A3", note="picked run under E3 selection score, cross-fitted", show=True))
add(point("A3 agree term", "factor", CK, "A3__term_agree", note="agreement term scored alone", show=True))
add(point("A3 N1 term", "factor", CK, "A3__term_N1", note="centered rule term scored alone"))
add(point("A' nested on A3", "factor", AP, "A3__nested", note="nested score", show=True))
add(point("A' control (A3)", "factor", AP, "A3__control", kind="control", note="condition-free control of the nested score"))
add(point("N1 nested on A3", "factor", CK, "A3__nested_N1", note="declared control = A3__control of the quick checks", show=True))
add(point("A3 N1 uniform term", "free", CK, "A3__term_N1_uniform", kind="control", note="condition-free centered score"))
add(point("A3 matched control", "free", AD, "dev__A3__matched", kind="control", note="matched condition-free control, A3", show=True))
add(point("C2 (N6c matched)", "free", N6C, "matched", kind="control", note="matched control of N6c", show=True))
add(point("D0 told", "label", CK, "d0_told", note="label probe with the true labels (diagnostic)", show=True, hollow=True))
add(point("D0 hard", "label", CK, "d0_hard", note="label probe, hard inferred labels (diagnostic)", hollow=True))
add(point("D0 soft", "label", CK, "d0_soft", note="label probe, soft inferred labels (diagnostic)", hollow=True))
add(point("L3 agree term", "label", CK, "L3__term_agree", note="label-trained factors, agreement term (diagnostic)", hollow=True))
add(point("L3 N1 term", "label", CK, "L3__term_N1", note="label-trained factors, N1 term (diagnostic)", hollow=True))
add(point("N6 T6 term", "partition", N6, "term_T6", note="partition head T6 scored alone", show=True))
add(point("N6 T6soft term", "partition", N6, "term_T6soft"))
add(point("N6 T6u term", "partition", N6, "term_T6u", note="T6 with uniform weights"))
add(point("N6 nested", "partition", N6, "nested", note="nested score of the partition heads", show=True))
add(point("N6 control", "partition", N6, "control", kind="control", note="condition-free control of N6 nested"))
add(point("N6c", "partition", N6C, "config", note="N6c gate configuration", show=True))
add(point("N6c C1 control", "partition", N6C, "control", kind="control", note="declared control of N6c"))
# MLLM probes (different episodes)
add(dict(label="Qwen3-VL-8B", family="mllm", kind="aware", show=True, hollow=True, r1=14.21, r1_ci=None, gain=0.21, gain_ci=[-0.51, 0.94],
         either=28.21, either_ci=None, source_file=MLLM8.replace("/project/CoSiR/", ""), source_key="verdict table, lines 55-57", seed=46,
         note="seed 46, 600 episodes per pair, different episodes; cosine on the same episodes: R@1 13.14, either 26.28", per_pair=None))
add(dict(label="Qwen3-VL-2B v2", family="mllm", kind="aware", show=False, hollow=True, r1=13.50, r1_ci=None, gain=-0.53, gain_ci=[-1.68, 0.66],
         either=None, either_ci=None, source_file=GONOGO.replace("/project/CoSiR/", ""), source_key="early MLLM probe paragraph (line 61)", seed=44,
         note="seed 44; cosine R@1 13.28 on the same episodes; absolute either not reported, so absent from the frontier", per_pair=None))

margins = []
a_ = margins.append
a_(diff("A' nested A3", "factor", AP, "A3__nested", AP, "A3__control", "declared control", note="seed 42"))
a_(diff("N1 nested A3", "factor", CK, "A3__nested_N1", CK, "A3__control", "declared control", note="seed 42"))
# matched control lives in another file; same episodes and order
def diff_x(label, family, fa, pa, fb, pb, ctrl, note):
    a, b, c = arrs(fa, pa), arrs(fb, pb), clusters(fa)
    r, g = compare(a, b, c, "r1"), compare(a, b, c, "gain")
    return dict(label=label, family=family, control=ctrl, d_r1=r["point"], d_r1_ci=r["ci95"], d_gain=g["point"], d_gain_ci=g["ci95"],
                source_file=fa.replace("/project/CoSiR/", "") + " vs " + fb.replace("/project/CoSiR/", ""), source_key=f"{pa} minus {pb}", seed=42, note=note)
a_(diff_x("N1 nested A3", "factor", CK, "A3__nested_N1", AD, "dev__A3__matched", "matched control", "seed 42"))
adj = json.load(open(ADJ))["test"]["pooled"]["vs"]
for ck, name in (("control", "declared control"), ("matched", "matched control")):
    v = adj[ck]
    margins.append(dict(label="N1 nested A3, fresh seeds", family="factor", control=name, d_r1=v["r1"]["point"], d_r1_ci=v["r1"]["ci95"],
                        d_gain=v["gain"]["point"], d_gain_ci=v["gain"]["ci95"], source_file=ADJ.replace("/project/CoSiR/", ""),
                        source_key=f"test.pooled.vs.{ck}", seed="45,47,48 pooled", note="pooled 36,864 episodes, 5,203 clusters"))
a_(diff("N6 nested", "partition", N6, "nested", N6, "control", "declared control", note="seed 42"))
a_(diff("N6c", "partition", N6C, "config", N6C, "control", "declared control (C1)", note="seed 42"))
a_(diff("N6c", "partition", N6C, "config", N6C, "matched", "matched control (C2)", note="seed 42"))

# per-pair heat map: reuse the points' per_pair, in the requested order
heat_rows = ["A3 agree term", "A3 N1 term", "D0 hard", "D0 told", "N6 T6 term", "N6 nested", "N6c"]
byl = {p["label"]: p for p in pts}
heat = [dict(label=l, pairs=["emotion x style", "emotion x genre", "style x genre"], gain=[byl[l]["per_pair"][str(i) if False else i]["gain"] for i in range(3)],
             source_file=byl[l]["source_file"], source_key=byl[l]["source_key"], seed=42, note="gain per pair_index 0,1,2") for l in heat_rows]
json.dump(dict(points=pts, margins=margins, heat=heat), open(os.path.join(OUT, "figure_data.json"), "w"), indent=1)
for p in pts: print(f'{p["label"]:22s} r1 {p["r1"]:.2f} either {p["either"] if p["either"] is None else round(p["either"],2)} gain {p["gain"]:.2f}')
for m in margins: print(m["label"], m["control"], round(m["d_r1"], 2), [round(x, 2) for x in m["d_r1_ci"]], round(m["d_gain"], 2))
for h in heat: print(h["label"], [round(x, 2) for x in h["gain"]])
