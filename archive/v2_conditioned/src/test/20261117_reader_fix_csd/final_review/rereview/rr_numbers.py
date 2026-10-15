"""Scoped re-review: re-derive every number the fix wave added, from the stored per-anchor arrays, with
src.eval.aspect_metrics.cluster_bootstrap (5,000 resamples, seed 42, painting clusters). Independent of common.py."""
import json
import numpy as np
from src.eval.aspect_metrics import cluster_bootstrap

R = "/project/CoSiR/src/test/20261117_reader_fix_csd/results/"
S = np.load("/project/CoSiR/src/test/20261116_grouping_step1_style/results/step1_eval_style.npz")
cl, pi = S["anchor_group"], S["pair_index"]
PAIRS = ["emotion × style", "emotion × genre", "style × genre"]
f64 = lambda x: np.asarray(x, np.float64)


def ci(v, m=None):
    v = f64(v)
    if m is not None:
        v, c = v[m], cl[m]
    else:
        c = cl
    r = cluster_bootstrap(v, c)
    return f"{100*r['point']:+.3f} [{100*r['ci95'][0]:+.3f}, {100*r['ci95'][1]:+.3f}]"


def cand(name):
    z = np.load(R + f"cand_{name}.npz")
    if "extra__anchor_group" in z.files:
        assert np.array_equal(z["extra__anchor_group"], cl) and np.array_equal(z["extra__pair_index"], pi)
    return z


def step1(arm):
    g = lambda k: S[f"{arm}__{k}"]
    return {"fused__r1": g("reader__fused__r1"), "fused__gain": g("reader__fused__gain"), "fused__other": g("reader__fused__other"),
            "cf__r1": g("reader__cf__r1"), "cf__gain": g("reader__cf__gain"), "cf__other": g("reader__cf__other"),
            "Bp__r1": g("Bprime__r1")}


def bar_v(z, arm):
    """fused minus the largest-mean of B', counterpart, B (ties to the earlier)."""
    comps = [("B_prime", S[f"{arm}__Bprime__r1"]), ("counterpart", z["cf__r1"]), ("B", S["B__r1"])]
    best = 0
    for i in (1, 2):
        if f64(comps[i][1]).mean() > f64(comps[best][1]).mean():
            best = i
    return comps[best][0], f64(z["fused__r1"]) - f64(comps[best][1])


print("n anchors", len(cl), "clusters", len(np.unique(cl)))
print("\n# Finding 1: per pair, gain / either against the counterpart")
for name in ("Rc_Rb_expected_A0", "Rb_expected_A0", "Rb_argmax_A1"):
    z = cand(name)
    g = f64(z["fused__gain"]) - f64(z["cf__gain"])
    e = (f64(z["fused__r1"]) + f64(z["fused__other"])) - (f64(z["cf__r1"]) + f64(z["cf__other"]))
    row = [f"{100*g[pi == i].mean():+.3f} / {100*e[pi == i].mean():+.3f}" for i in range(3)]
    print(f"{name:20s}", " | ".join(row), f"   pooled gain {100*g.mean():+.3f} either {100*e.mean():+.3f}")

print("\n# Finding 4: Table 4, A1 minus A0, paired per anchor")
readers = {"arg-max (step 1)": (step1("A1"), step1("A0")),
           "R-a": (cand("Ra_A1"), cand("Ra_A0")),
           "R-b arg-max": (cand("Rb_argmax_A1"), cand("Rb_argmax_A0")),
           "R-b expected": (cand("Rb_expected_A1"), cand("Rb_expected_A0"))}
for rn, (z1, z0) in readers.items():
    n1, b1 = bar_v(z1, "A1")
    n0, b0 = bar_v(z0, "A0")
    if "bar_v" in getattr(z1, "files", []):
        assert np.allclose(b1, z1["bar_v"], atol=0) or np.max(np.abs(b1 - z1["bar_v"])) < 1e-12, rn
        assert np.max(np.abs(b0 - z0["bar_v"])) < 1e-12, rn
    fused = f64(z1["fused__r1"]) - f64(z0["fused__r1"])
    marg = (f64(z1["fused__r1"]) - f64(z1["cf__r1"])) - (f64(z0["fused__r1"]) - f64(z0["cf__r1"]))
    bp = 100 * (f64(S["A1__Bprime__r1"]).mean() - f64(S["A0__Bprime__r1"]).mean())
    cp = 100 * (f64(z1["cf__r1"]).mean() - f64(z0["cf__r1"]).mean())
    print(f"{rn:18s} fused {ci(fused)} | bar {ci(b1 - b0)} ({n1}/{n0}) | margin {ci(marg)} | B' {bp:+.3f} | cf {cp:+.4f}")

print("\n# Finding 6: against B alone and B' alone")
for name, arm in (("Rc_Rb_expected_A0", "A0"), ("Rb_argmax_A1", "A1"), ("Rb_expected_A1", "A1"),
                  ("Rb_expected_A0", "A0"), ("Rb_argmax_A0", "A0"), ("Ra_A1", "A1"), ("Ra_A0", "A0")):
    z = cand(name)
    print(f"{name:20s} vs B {ci(f64(z['fused__r1']) - f64(S['B__r1']))} | vs B' {ci(f64(z['fused__r1']) - f64(S[f'{arm}__Bprime__r1']))}"
          f" | comparator {bar_v(z, arm)[0]}")
