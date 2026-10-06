"""Final review: the seed-42 regression targets (rule §5 items 1 to 4, D7, D13) and the sensitivity projection (§6.1)
from my own seed-42 inputs (fr3_build) and my own family code; plus the bundle-cache check of the test seeds (my
bundles against results/cache_seed{s}.npz). Writes out/fr3_seed42.json.
"""
import json

import numpy as np

import fr3_lib as L

TEST = L.ROOT / "src/test"
R1RES = TEST / "20261117_reader_fix_csd/results"
RES = L.R3DIR / "results"
METRICS = ("r1", "gain", "other", "swap", "strict")
CMP = []


def rec(name, mine, theirs, tol=0.0):
    if isinstance(mine, np.ndarray) or isinstance(theirs, np.ndarray):
        ok = bool(np.array_equal(np.asarray(mine), np.asarray(theirs)))
    elif isinstance(mine, (float, int)) and isinstance(theirs, (float, int)) and not isinstance(mine, bool):
        ok = abs(float(mine) - float(theirs)) <= tol
    else:
        ok = mine == theirs
    CMP.append({"name": name, "agree": bool(ok), "mine": None if isinstance(mine, np.ndarray) else mine,
                "theirs": None if isinstance(theirs, np.ndarray) else theirs})
    if not ok:
        print(f"  DISAGREE {name}: {CMP[-1]['mine']} vs {CMP[-1]['theirs']}", flush=True)


def load(s):
    z = np.load(L.OUT / f"fr3_seed{s}.npz")
    g = {k: z[k] for k in z.files}
    sc = lambda k: {c: {d: g[f"{k}__{c}__{d}"] for d in L.DIRS} for c in L.CONDS}  # noqa: E731
    return g, sc


def main():
    out = {}
    # ---------------- bundle cache check for the test seeds (before any use of the caches)
    for s in (49, 50, 51):
        g, _ = load(s)
        with np.load(RES / f"cache_seed{s}.npz") as z:
            for k in ("cl", "parity", "pair_index", "anchor"):
                rec(f"cache{s}.{k}", g[k], z[k])
            for k in ("cos", "B", "Bp"):
                for c in L.CONDS:
                    for d in L.DIRS:
                        rec(f"cache{s}.{k}.{c}.{d}", g[f"{k}__{c}__{d}"], z[f"{k}__{c}__{d}"])
            for d in L.DIRS:
                rec(f"cache{s}.stack.{d}", g[f"stack__{d}"], z[f"stack__{d}"])
            for c in L.CONDS:
                rec(f"cache{s}.F.{c}", g[f"F__{c}"], z[f"F__{c}"])
            meta = json.loads(str(z["meta"]))
        mine_meta = json.loads(str(g["meta"]))
        rec(f"cache{s}.episodes_sha256", mine_meta["episodes_sha256"], meta.get("episodes_sha256"))
        bj = json.loads((RES / f"build_seed{s}.json").read_text())
        out[f"build_seed{s}_keys"] = sorted(bj)
    # ---------------- seed 42
    g, sc = load(42)
    B, Bp, T, stack = sc("B"), sc("Bp"), sc("T"), {d: g[f"stack__{d}"] for d in L.DIRS}
    m = {c: g[f"m__{c}"] for c in L.CONDS}
    pick = {c: g[f"pick__{c}"] for c in L.CONDS}
    cl, pi, parity = g["cl"], g["pair_index"], g["parity"]
    pB, pBp = L.metrics(B), L.metrics(Bp)
    out["B_r1"], out["Bp_r1"] = 100 * pB["r1"].mean(), 100 * pBp["r1"].mean()
    rec("D10.B_r1", out["B_r1"], 18.341064453125, tol=1e-9)
    rec("D11.Bp_r1", out["Bp_r1"], 18.436686197916664, tol=1e-9)
    z1 = np.load(R1RES / "cand_Rc_Rb_expected_A0.npz")
    for c in L.CONDS:
        for d in L.DIRS:
            rec(f"item2.T.{c}.{d}", T[c][d], z1[f"T__{c}__{d}"])
        rec(f"item2.margin.{c}", m[c], z1[f"margin__{c}"])
        rec(f"item2.pick.{c}", pick[c], z1[f"pick__{c}"].astype(np.int64))
    taus = np.percentile(np.concatenate([m["a"], m["b"]]), [0, 25, 50, 75])
    rec("item2.taus", tuple(float(t) for t in taus), L.TAUS)
    tj = json.loads((R1RES / "rc_tau.json").read_text())
    rec("item2.rc_tau.json", tuple(tj["taus"]), L.TAUS)
    gR = L.gates(m, pick, "R1")
    gA = L.gates(m, pick, "AFF")
    for t in range(4):
        rec(f"item2.gate_a.t{t}", gR[t]["a"], z1["extra__gate_a"][t])
        rec(f"item2.gate_b.t{t}", gR[t]["b"], z1["extra__gate_b"][t])
    # R1 family on seed 42
    famR = L.Family(B, T, gR, parity)
    fpR, critR = famR.pick_fused()
    cpR, ccritR = famR.pick_cf()
    rec("item2.fused_cells", (fpR[0], fpR[1]), (116, 119))
    rec("item2.cf_cells", (cpR[0], cpR[1]), (58, 123))
    rec("item2.sigma", (famR.ctrl[0][0], famR.ctrl[1][0]), (0.0, 0.0))
    fR, cR = famR.assemble(fpR), famR.assemble(cpR, cf=True)
    for mm in METRICS:
        rec(f"item2.fused.{mm}", fR[mm], z1[f"fused__{mm}"])
        rec(f"item2.cf.{mm}", cR[mm], z1[f"cf__{mm}"])
    name, comp, means = L.bar_comparator(pBp, cR, pB)
    rec("item2.comparator", name, "counterpart")
    rec("item2.bar_v", fR["r1"] - comp["r1"], z1["bar_v"])
    b2 = L.ci(fR["r1"] - comp["r1"], cl)
    g2 = L.ci(fR["gain"] - cR["gain"], cl)
    for k, v, t in (("bar", b2, (0.4435221354166667, 0.21646171563312194, 0.6735669710776852)),
                    ("gain_statistic", g2, (2.667236328125, 2.325087836946873, 3.012361650695922))):
        rec(f"item2.{k}.point", v["point"], t[0])
        rec(f"item2.{k}.lo", v["ci95"][0], t[1])
        rec(f"item2.{k}.hi", v["ci95"][1], t[2])
    out["R1_seed42"] = {"fused_r1": 100 * fR["r1"].mean(), "cf_r1": 100 * cR["r1"].mean(), "bar": b2,
                        "gain_statistic": g2, "minus_Bp": L.ci(fR["r1"] - pBp["r1"], cl),
                        "minus_B": L.ci(fR["r1"] - pB["r1"], cl), "crit": critR, "ccrit": ccritR}
    # AFF family on seed 42 (item 3)
    famA = L.Family(B, T, gA, parity)
    fpA, critA = famA.pick_fused()
    cpA, ccritA = famA.pick_cf()
    rec("item3.fused_cells", (fpA[0], fpA[1]), (39, 119))
    rec("item3.cf_cells", (cpA[0], cpA[1]), (149, 10))
    rec("item3.sigma", (famA.ctrl[0][0], famA.ctrl[1][0]), (0.0, 0.0))
    fA, cA = famA.assemble(fpA), famA.assemble(cpA, cf=True)
    rec("item3.fused_r1", 100 * fA["r1"].mean(), 19.136555989583336, tol=1e-9)
    rec("item3.cf_r1", 100 * cA["r1"].mean(), 18.39599609375, tol=1e-9)
    nameA, compA, _ = L.bar_comparator(pBp, cA, pB)
    rec("item3.comparator", nameA, "B_prime")
    vA = fA["r1"] - compA["r1"]
    bar = L.ci(vA, cl)
    marg = L.ci(fA["r1"] - cA["r1"], cl)
    gst = L.ci(fA["gain"] - cA["gain"], cl)
    eit = 100 * float(np.mean(L.either(fA) - L.either(cA)))
    for k, v, t in (("bar", bar, (0.6998697916666667, 0.4598852740816973, 0.9371680126852968)),
                    ("margin", marg, (0.7405598958333333, 0.5196896694963071, 0.9598857494832738)),
                    ("gain_statistic", gst, (3.110758463541667, 2.780005709854805, 3.4559584315470384))):
        rec(f"item3.{k}.point", v["point"], t[0])
        rec(f"item3.{k}.lo", v["ci95"][0], t[1])
        rec(f"item3.{k}.hi", v["ci95"][1], t[2])
    rec("item3.either", eit, -1.629638671875, tol=1e-12)
    for i, (p, t) in enumerate(zip(L.PAIRS, (0.9765625, 1.45263671875, -0.32958984375))):
        rec(f"item3.per_pair_bar.{p}", 100 * float(vA[pi == i].mean()), t, tol=1e-12)
    d_f = L.ci(fA["r1"] - fR["r1"], cl)
    vR = fR["r1"] - comp["r1"]
    d_b = L.ci(vA - vR, cl)
    for k, v, t in (("aff_minus_r1_fused", d_f, (0.21769205729166666, 0.06425880757348419, 0.3709597330984391)),
                    ("aff_minus_r1_bar", d_b, (0.25634765625, 0.04280778303598444, 0.46195041633015954))):
        rec(f"item3.{k}.point", v["point"], t[0])
        rec(f"item3.{k}.lo", v["ci95"][0], t[1])
        rec(f"item3.{k}.hi", v["ci95"][1], t[2])
    rec("item3.open_a", int(np.count_nonzero(gA[0]["a"])), 9941)
    rec("item3.open_b", int(np.count_nonzero(gA[0]["b"])), 3627)
    out["AFF_seed42"] = {"bar": bar, "margin": marg, "gain_statistic": gst, "either": eit,
                         "fused_r1": 100 * fA["r1"].mean(), "cf_r1": 100 * cA["r1"].mean(),
                         "crit": critA, "ccrit": ccritA, "minus_B": L.ci(fA["r1"] - pB["r1"], cl),
                         "either_cost_ratio": -eit / gst["point"]}
    # item 4: D13
    out["D13"] = {"bar_ge_0.5": bool(bar["point"] >= 0.5), "bar_lo_gt_0": bool(bar["ci95"][0] > 0),
                  "gain_lo_gt_0": bool(gst["ci95"][0] > 0)}
    rec("item4.D13", all(out["D13"].values()), True)
    # D7
    red = L.redundancy(stack, B)
    tgt = {"affect": (0.35348060377541385, 0.3828024789253903), "image": (0.7145397990123284, 0.7090878258485419),
           "caption": (0.6182295729609555, 0.665152773464146)}
    for h in L.A0:
        for j, d in enumerate(L.DIRS):
            rec(f"D7.{h}.{d}", red[h][d], tgt[h][j])
    out["D7"] = red
    # sensitivity (§6.1) from AFF's seed-42 per-episode differences
    with np.load(TEST / "20261030_aspect_baselines/results/per_anchor_seed42.npz") as z:
        cos = {mm: z[f"cosine__{mm}"] for mm in METRICS}
        rca = {mm: z[f"rca__{mm}"] for mm in METRICS}
    mc = L.metrics(sc("cos"))
    rec("item1.cosine_per_anchor", np.stack([mc[mm] for mm in METRICS]), np.stack([cos[mm] for mm in METRICS]))
    diffs = {"r1_vs_cosine": fA["r1"] - cos["r1"], "r1_vs_rca": fA["r1"] - rca["r1"], "r1_vs_B": fA["r1"] - pB["r1"],
             "r1_vs_Bprime": fA["r1"] - pBp["r1"], "r1_vs_counterpart": fA["r1"] - cA["r1"],
             "gain_statistic": fA["gain"] - cA["gain"], "gain_vs_rca": fA["gain"] - rca["gain"],
             "secondary": fA["r1"] - fR["r1"]}
    sens = {k: L.sensitivity(100 * v, cl) for k, v in diffs.items()}
    stored = json.loads((RES / "sensitivity.json").read_text())["checks"]
    for k in sens:
        for f in ("SE", "x", "half_width", "seed42_half_width"):
            rec(f"sens.{k}.{f}", sens[k][f], stored[k][f], tol=1e-9)
    out["sensitivity"] = sens
    out["seed42_points"] = {k: L.ci(v, cl)["point"] for k, v in diffs.items()}
    # stored regression_check.json headline
    rc = json.loads((RES / "regression_check.json").read_text())
    out["regression_check_keys"] = sorted(rc)[:40]
    out["comparisons"] = {"n": len(CMP), "n_agree": sum(c["agree"] for c in CMP),
                          "disagree": [c for c in CMP if not c["agree"]]}
    (L.OUT / "fr3_seed42.json").write_text(json.dumps(out, indent=1, default=str))
    (L.OUT / "fr3_seed42_comparisons.json").write_text(json.dumps(CMP, indent=0, default=str))
    print(f"seed42 + cache: {out['comparisons']['n']} comparisons, {out['comparisons']['n_agree']} agree", flush=True)


if __name__ == "__main__":
    main()
