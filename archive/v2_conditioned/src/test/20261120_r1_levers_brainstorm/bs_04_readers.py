"""Exploratory, seed 42, decides nothing. Label-free transforms of R1's probabilities, each in R1's tied 224-cell family
with thresholds from its own margins and its own matched counterpart:

  J_eps   : joint (exclusive) decoding of the two conditions. Q(h_a, h_b) ∝ P^a(h_a) P^b(h_b) w(h_a, h_b) with
            w = 1 for h_a != h_b and eps for h_a == h_b; P'^a, P'^b = the marginals of Q. eps = 1 is R1.
            Symmetric under swapping a and b.
  ENS     : R1 and R3 combined: arithmetic mean, and geometric mean (renormalised).
  TEMP    : temperature on R1, P^(1/T) renormalised, T in {0.5, 2}.
  PGATE   : per-grouping thresholds: tau_t,h = the t-th percentile of the margins among values whose pick is h.
  AFF     : affect-only gate: g^c = 1[pick^c = affect and m^c >= tau] (term unchanged).
  CSDABST : R1's gate times an abstention on min(S_csd, C_csd) (symmetric; both conditions) at its q-th percentile.
Also reports pick accuracy (told mapping, diagnostic) and the same-pick share for each transform.
Writes results/bs_04_readers.json.
"""
import json
import time

import numpy as np

import bs_lib as L


def joint(P, eps):
    W = np.ones((3, 3))
    np.fill_diagonal(W, eps)
    Q = P["a"][:, :, None] * P["b"][:, None, :] * W[None]
    Q /= Q.sum(axis=(1, 2), keepdims=True)
    return {"a": Q.sum(axis=2), "b": Q.sum(axis=1)}


def norm(x):
    return x / x.sum(axis=1, keepdims=True)


def describe(data, P):
    pk = {c: P[c].argmax(1) for c in L.CONDITIONS}
    acc = 100 * np.mean(np.concatenate([pk[c] == data.told[c] for c in L.CONDITIONS]))
    same = 100 * np.mean(pk["a"] == pk["b"])
    per = {p: {c: round(100 * float(np.mean(pk[c][data.pi == i] == data.told[c][data.pi == i])), 1) for c in L.CONDITIONS}
           for i, p in enumerate(L.PAIRS)}
    tv = float(np.mean(0.5 * np.abs(P["a"] - P["b"]).sum(1)))
    return {"pick_accuracy": acc, "same_pick": same, "per_pair_condition": per, "tv_mean": tv,
            "top_prob": float(np.mean(np.concatenate([P[c].max(1) for c in L.CONDITIONS])))}


def run(data, P, label, gate_fn=None, store=None):
    MDs, taus, zT, g, m = L.standard_MDs(data, P)
    if gate_fn is not None:
        gl = gate_fn(P, m, g)
        MDs = [L.MD(zT, gg) for gg in gl]
    fam = L.Family(data, MDs, tied=True).stats()
    fp, cp = fam.crossfit()
    pn, pc = fam.assemble(fp, cp)
    r, bar_v = L.evaluate(data, pn, pc, label)
    r["fused_cells"] = [fam.cells[fp[h]] for h in (0, 1)]
    r["cf_cells"] = [fam.cf_cells[cp[h]] for h in (0, 1)]
    r["in_sample"] = fam.in_sample()
    r["reader"] = describe(data, P)
    print(L.fmt(r), r["fused_cells"], r["cf_cells"], "| acc %.1f same %.1f tv %.3f" % (
        r["reader"]["pick_accuracy"], r["reader"]["same_pick"], r["reader"]["tv_mean"]), flush=True)
    if store is not None:
        store[label] = (pn, pc, bar_v)
    return r


def main():
    t0 = time.time()
    data = L.Data()
    out = {"note": "exploratory, seed 42, decides nothing"}
    P1, P3 = data.P("R1"), data.P("R3")
    arrays = {}
    res = {}
    res["R1"] = run(data, P1, "R1", store=arrays)
    for eps in (0.0, 0.1, 0.3, 0.6):
        res[f"J_eps{eps}"] = run(data, joint(P1, eps), f"J_eps{eps}: joint exclusive decoding", store=arrays)
    res["J_eps0_R3"] = run(data, joint(P3, 0.0), "J_eps0 on R3", store=arrays)
    res["ENS_arith"] = run(data, {c: 0.5 * (P1[c] + P3[c]) for c in L.CONDITIONS}, "ENS arithmetic R1+R3")
    res["ENS_geo"] = run(data, {c: norm(np.sqrt(P1[c] * P3[c])) for c in L.CONDITIONS}, "ENS geometric R1+R3")
    for T in (0.5, 2.0):
        res[f"TEMP_{T}"] = run(data, {c: norm(P1[c] ** (1.0 / T)) for c in L.CONDITIONS}, f"TEMP T={T}")

    def pgate(P, m, g):
        pk = {c: P[c].argmax(1) for c in L.CONDITIONS}
        allm = np.concatenate([m["a"], m["b"]])
        allp = np.concatenate([pk["a"], pk["b"]])
        gl = []
        for pct in (0, 25, 50, 75):
            th = np.array([np.percentile(allm[allp == h], pct) if np.any(allp == h) else 0.0 for h in range(3)])
            gl.append({c: (m[c] >= th[pk[c]]).astype(np.float32) for c in L.CONDITIONS})
        return gl
    res["PGATE"] = run(data, P1, "PGATE per-grouping thresholds", pgate)

    def affonly(P, m, g):
        pk = {c: P[c].argmax(1) for c in L.CONDITIONS}
        return [{c: (g[t][c] * (pk[c] == 0)).astype(np.float32) for c in L.CONDITIONS} for t in range(4)]
    res["AFF"] = run(data, P1, "AFF: gate open only on affect picks", affonly)

    fa, fb = data.feat["a"], data.feat["b"]
    csdmin = np.minimum(fa[:, 18], fa[:, 19])                       # symmetric: min(S_csd, C_csd)
    for q in (50, 75, 90):
        thr = np.percentile(csdmin, q)
        keep = (csdmin < thr).astype(np.float32)

        def csdab(P, m, g, keep=keep):
            return [{c: (g[t][c] * keep).astype(np.float32) for c in L.CONDITIONS} for t in range(4)]
        res[f"CSDABST_q{q}"] = run(data, P1, f"CSDABST: abstain if min(S_csd,C_csd) >= p{q}", csdab)
    imgmin = np.minimum(fa[:, 6], fa[:, 7])
    for q in (75,):
        thr = np.percentile(imgmin, q)
        keep = (imgmin < thr).astype(np.float32)

        def imgab(P, m, g, keep=keep):
            return [{c: (g[t][c] * keep).astype(np.float32) for c in L.CONDITIONS} for t in range(4)]
        res[f"IMGABST_q{q}"] = run(data, P1, f"IMGABST: abstain if min(S_img,C_img) >= p{q}", imgab)
    # paired differences against R1 for the joint decoders
    pn1, pc1, bv1 = arrays["R1"]
    out["paired_vs_R1"] = {}
    for k, (pn, pc, bv) in arrays.items():
        if k == "R1":
            continue
        out["paired_vs_R1"][k] = {"fused_r1": L.C.point_ci(np.asarray(pn["r1"], float) - np.asarray(pn1["r1"], float), data.cl),
                                  "bar_margin": L.C.point_ci(bv - bv1, data.cl)}
        print("  paired vs R1:", k, {kk: round(v["point"], 3) for kk, v in out["paired_vs_R1"][k].items()},
              {kk: [round(x, 3) for x in v["ci95"]] for kk, v in out["paired_vs_R1"][k].items()})
    out["results"] = res
    out["runtime_s"] = round(time.time() - t0)
    (L.HERE / "results" / "bs_04_readers.json").write_text(json.dumps(L.C.jsonable(out), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
