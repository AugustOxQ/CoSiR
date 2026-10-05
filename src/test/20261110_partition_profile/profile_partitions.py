"""Descriptive per-cluster and pair-level profile of the E2 label-free partitions (affect, image, caption).

Read-only, CPU-only, deterministic. Uses evaluation labels on scorer-train rows (checks 1 to 3) and selection rows
(check 3b) for description only; nothing here chooses anything.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.data.artelingo import load_artelingo
from src.data.artelingo_splits import artelingo_splits, artelingo_aspect_labels, encode_labels, EMOTION_CATCH_ALL
from src.data.wikiart_genre import GENRE_NAMES

T0 = time.time()
HERE = Path(__file__).parent
OUT = HERE / "results"
PARTS = ("affect", "image", "caption")
ASPECTS = ("emotion", "style", "genre")
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
N_SAMPLE = 2_000_000
Z_MAX = 4.0
LINES = []
CHECKS = []   # (name, exact, sampled, z)


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def mix(arrs):
    """Combine integer arrays into one cell code per row."""
    code = np.zeros(len(arrs[0]), dtype=np.int64)
    for a in arrs:
        u, inv = np.unique(a, return_inverse=True)
        code = code * len(u) + inv
    return code


def pairs2(arrs, n):
    """Sum over cells of C(n_cell, 2): unordered pairs agreeing on all of arrs."""
    if not arrs:
        return n * (n - 1) / 2
    cnt = np.bincount(np.unique(mix(arrs), return_inverse=True)[1]).astype(np.float64)
    return float((cnt * (cnt - 1) / 2).sum())


def dpairs(arrs, g):
    """Unordered pairs on DIFFERENT paintings agreeing on arrs (exact)."""
    n = len(g)
    return pairs2(arrs, n) - pairs2(list(arrs) + [g], n)


def dots2(arrs, Pi, Pt, g):
    """(sum of dot(Pi[i], Pt[j]), count) over ORDERED pairs on different paintings agreeing on arrs."""
    def one(keys):
        if keys:
            inv = np.unique(mix(keys), return_inverse=True)[1]
        else:
            inv = np.zeros(len(g), dtype=np.int64)
        k = inv.max() + 1
        si = np.stack([np.bincount(inv, weights=Pi[:, c], minlength=k) for c in range(Pi.shape[1])], 1)
        st = np.stack([np.bincount(inv, weights=Pt[:, c], minlength=k) for c in range(Pt.shape[1])], 1)
        cnt = np.bincount(inv, minlength=k).astype(np.float64)
        return float((si * st).sum()), float((cnt ** 2).sum())
    a, b = one(list(arrs)), one(list(arrs) + [g])
    return a[0] - b[0], a[1] - b[1]


def sample_pairs(g, n, seed=0):
    rng = np.random.default_rng(seed)
    I, J = [], []
    got = 0
    while got < n:
        i = rng.integers(0, len(g), n)
        j = rng.integers(0, len(g), n)
        ok = g[i] != g[j]
        I.append(i[ok]); J.append(j[ok]); got += ok.sum()
    return np.concatenate(I)[:n], np.concatenate(J)[:n]


def binom_check(name, exact, mask, hits):
    """Exact probability vs sampled indicator ``hits`` over sampled pairs where ``mask``."""
    n = int(mask.sum())
    p = float(hits[mask].mean()) if n else float("nan")
    se = np.sqrt(max(exact * (1 - exact), 1e-12) / max(n, 1))
    z = (p - exact) / se
    CHECKS.append((name, exact, p, float(z)))
    return p


def entropy_eff(counts):
    p = counts[counts > 0] / counts.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def cover_k(counts, frac):
    s = np.sort(counts)[::-1]
    c = np.cumsum(s) / s.sum()
    return int(np.searchsorted(c, frac - 1e-12) + 1)


def frag(counts):
    return {"k50": cover_k(counts, .5), "k80": cover_k(counts, .8), "effK": entropy_eff(counts)}


# ---------------------------------------------------------------- load and align
data = load_artelingo()
splits = artelingo_splits(data)
st = np.asarray(splits.scorer_train)
lab_all = artelingo_aspect_labels(data)
z = np.load("src/test/20261031_pseudo_partitions/results/partitions.npz")
local = np.unique(splits.groups[st], return_inverse=True)[1]
assert np.array_equal(local, z["local_groups"]), "partitions not aligned with scorer_train"
say("ALIGNMENT OK: local_groups == unique(groups[scorer_train]) inverse; n scorer_train = %d" % len(st))
cl = {h: z[h].astype(np.int64) for h in PARTS}
for h in PARTS:
    assert len(cl[h]) == len(st)
    say("  partition %-8s clusters: %d (ids %d..%d)" % (h, len(np.unique(cl[h])), cl[h].min(), cl[h].max()))
lab = {a: lab_all[a][st] for a in ASPECTS}
names = {"emotion": encode_labels(data.emotions, exclude=(EMOTION_CATCH_ALL,))[1],
         "style": encode_labels(data.art_styles)[1], "genre": list(GENRE_NAMES)}
for a in ASPECTS:
    assert lab[a].max() == len(names[a]) - 1, a
    say("  aspect %-8s labelled scorer-train rows: %d of %d, %d values, missing code -1: %d" %
        (a, (lab[a] >= 0).sum(), len(st), len(names[a]), (lab[a] < 0).sum()))

zs = np.load("src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz")
sel = zs["selection"]
assert len(sel) == 32413 and np.array_equal(np.sort(sel), np.sort(splits.selection)), "selection mismatch"
assert not (set(sel.tolist()) & set(st.tolist()))
post = {h: {"img": zs[f"{h}__img"].astype(np.float64), "txt": zs[f"{h}__txt"].astype(np.float64)} for h in PARTS}
assert all(np.isfinite(post[h][m]).all() for h in PARTS for m in ("img", "txt"))
say("  selection rows: %d, posterior row sums within 1e-3 of 1: %s" %
    (len(sel), all(np.abs(post[h][m].sum(1) - 1).max() < 1e-3 for h in PARTS for m in ("img", "txt"))))
labS = {a: lab_all[a][sel] for a in ASPECTS}
gS = splits.groups[sel]
gT = local

R = {"names": names}

# ---------------------------------------------------------------- checks 1 and 2
say("\n=== CHECK 1 and 2: cluster x value tables and fragmentation (scorer-train rows) ===")
R["check1"], R["check2"] = {}, {}
for h in PARTS:
    for a in ASPECTS:
        m = lab[a] >= 0
        c, x = cl[h][m], lab[a][m]
        V = len(names[a])
        T = np.zeros((64, V), dtype=np.int64)
        np.add.at(T, (c, x), 1)
        base = T.sum(0) / T.sum()
        size = T.sum(1)
        dom = T.argmax(1)
        pur = np.where(size > 0, T.max(1) / np.maximum(size, 1), 0.0)
        lift = pur / base[dom]
        cov = T[np.arange(64), dom] / T.sum(0)[dom]
        R["check1"][f"{h}|{a}"] = {
            "values": names[a], "base_rate": base.tolist(), "table": T.tolist(), "size": size.tolist(),
            "dominant": [names[a][d] for d in dom], "purity": pur.tolist(), "lift": lift.tolist(),
            "coverage_of_dominant": cov.tolist(), "mean_purity_size_weighted": float((pur * size).sum() / size.sum()),
            "mean_lift_size_weighted": float((lift * size).sum() / size.sum())}
        bl = frag(size.astype(np.float64))
        per = {names[a][v]: frag(T[:, v].astype(np.float64)) for v in range(V)}
        med = {k: float(np.median([per[n][k] for n in per])) for k in ("k50", "k80", "effK")}
        R["check2"][f"{h}|{a}"] = {"baseline_cluster_size": bl, "per_value": per, "median_over_values": med}
        say("%-8s x %-8s size-weighted purity %.3f, lift %.2f | median over values k50 %.1f k80 %.1f effK %.1f"
            " | baseline k50 %d k80 %d effK %.1f" % (h, a, R["check1"][f"{h}|{a}"]["mean_purity_size_weighted"],
            R["check1"][f"{h}|{a}"]["mean_lift_size_weighted"], med["k50"], med["k80"], med["effK"],
            bl["k50"], bl["k80"], bl["effK"]))

say("\n--- affect x emotion, full 64 x 8 cluster table (size, dominant, purity, lift, coverage of dominant) ---")
t = R["check1"]["affect|emotion"]
say("emotion base rates: " + ", ".join("%s %.3f" % (n, b) for n, b in zip(names["emotion"], t["base_rate"])))
say("cluster  size  dominant         purity  lift  coverage  | counts per emotion " + "/".join(names["emotion"]))
for k in range(64):
    say("%5d %6d  %-15s %6.3f %5.2f %8.3f  | %s" % (k, t["size"][k], t["dominant"][k], t["purity"][k], t["lift"][k],
        t["coverage_of_dominant"][k], " ".join("%5d" % v for v in t["table"][k])))

say("\n--- affect x emotion, per emotion ---")
say("best-purity cluster among clusters where the emotion is dominant (size, purity, lift, coverage) | largest-coverage"
    " cluster (id, purity of that cluster, coverage) | k50 k80 effK  (baseline of all labelled rows: k50 %d k80 %d effK %.1f)"
    % tuple(R["check2"]["affect|emotion"]["baseline_cluster_size"][k] for k in ("k50", "k80", "effK")))
T = np.array(t["table"]); base = np.array(t["base_rate"]); size = T.sum(1)
R["affect_emotion_per_value"] = {}
for v, n in enumerate(names["emotion"]):
    doms = [k for k in range(64) if T[k].argmax() == v]
    if doms:
        kb = max(doms, key=lambda k: T[k, v] / size[k])
        best = (kb, int(size[kb]), T[kb, v] / size[kb], T[kb, v] / size[kb] / base[v], T[kb, v] / T[:, v].sum())
        bs = "cl %2d size %5d purity %.3f lift %.2f cov %.3f" % best
    else:
        best = None
        bs = "never dominant in any cluster"
    kc = int(T[:, v].argmax())
    f = R["check2"]["affect|emotion"]["per_value"][n]
    say("%-12s base %.3f | %s | cl %2d pur %.3f cov %.3f | k50 %d k80 %d effK %.1f" %
        (n, base[v], bs, kc, T[kc, v] / size[kc], T[kc, v] / T[:, v].sum(), f["k50"], f["k80"], f["effK"]))
    R["affect_emotion_per_value"][n] = {"base": float(base[v]), "best_purity_dominant_cluster": best and list(map(float, best)),
        "largest_coverage_cluster": [kc, float(T[kc, v] / size[kc]), float(T[kc, v] / T[:, v].sum())], **f}

for hp, ap in (("image", "style"), ("image", "genre")):
    say("\n--- %s x %s per value (k50 k80 effK; baseline k50 %d k80 %d effK %.1f) ---" % (hp, ap,
        *[R["check2"][f"{hp}|{ap}"]["baseline_cluster_size"][k] for k in ("k50", "k80", "effK")]))
    for n, f in R["check2"][f"{hp}|{ap}"]["per_value"].items():
        say("%-26s base %.3f  k50 %2d k80 %2d effK %5.1f" % (n, R["check1"][f"{hp}|{ap}"]["base_rate"][names[ap].index(n)],
            f["k50"], f["k80"], f["effK"]))

# ---------------------------------------------------------------- check 3
say("\n=== CHECK 3: pair co-membership, hard partitions (scorer-train, different-painting pairs) ===")
R["check3"] = {"lift": {}, "contrast": {}}
sI, sJ = sample_pairs(gT, N_SAMPLE)
for h in PARTS:
    for a in ASPECTS:
        m = lab[a] >= 0
        c, x, g = cl[h][m], lab[a][m], gT[m]
        tot = dpairs([], g)
        same_x = dpairs([x], g)
        diff_x = tot - same_x
        sc, sxc = dpairs([c], g), dpairs([x, c], g)
        p_same, p_diff, base_ = sxc / same_x, (sc - sxc) / diff_x, sc / tot
        R["check3"]["lift"][f"{h}|{a}"] = {"p_same": p_same, "p_diff": p_diff, "base": base_, "lift": p_same / p_diff,
                                          "n_pairs": tot, "frac_pairs_same_value": same_x / tot}
        # sample check on rows labelled for a
        full_c, full_x = cl[h], lab[a]
        ok = (full_x[sI] >= 0) & (full_x[sJ] >= 0)
        sx = full_x[sI] == full_x[sJ]
        sc_ = full_c[sI] == full_c[sJ]
        binom_check(f"c3 {h}|{a} p_same", p_same, ok & sx, sc_)
        binom_check(f"c3 {h}|{a} p_diff", p_diff, ok & ~sx, sc_)
        binom_check(f"c3 {h}|{a} base", base_, ok, sc_)
    for A, B in PAIRS:
        m = (lab[A] >= 0) & (lab[B] >= 0)
        c, xa, xb, g = cl[h][m], lab[A][m], lab[B][m], gT[m]
        nA = dpairs([xa], g) - dpairs([xa, xb], g)
        nB = dpairs([xb], g) - dpairs([xa, xb], g)
        sAB = (dpairs([xa, c], g) - dpairs([xa, xb, c], g)) / nA
        sBA = (dpairs([xb, c], g) - dpairs([xa, xb, c], g)) / nB
        R["check3"]["contrast"][f"{h}|{A}x{B}"] = {"s_AB": sAB, "s_BA": sBA, "diff": sAB - sBA, "ratio": sAB / sBA,
                                                   "base": dpairs([c], g) / dpairs([], g)}
        ok = (lab[A][sI] >= 0) & (lab[B][sI] >= 0) & (lab[A][sJ] >= 0) & (lab[B][sJ] >= 0)
        sa, sb = lab[A][sI] == lab[A][sJ], lab[B][sI] == lab[B][sJ]
        sc_ = cl[h][sI] == cl[h][sJ]
        binom_check(f"c3 {h}|{A}x{B} s_AB", sAB, ok & sa & ~sb, sc_)
        binom_check(f"c3 {h}|{A}x{B} s_BA", sBA, ok & sb & ~sa, sc_)

say("lift = P(same cluster | same value) / P(same cluster | different value), with the random-pair base rate P(same cluster)")
say("%-8s %-8s %9s %9s %9s %7s %12s" % ("part", "aspect", "p_same", "p_diff", "base", "lift", "P(same val)"))
for h in PARTS:
    for a in ASPECTS:
        r = R["check3"]["lift"][f"{h}|{a}"]
        say("%-8s %-8s %9.5f %9.5f %9.5f %7.3f %12.4f" % (h, a, r["p_same"], r["p_diff"], r["base"], r["lift"], r["frac_pairs_same_value"]))
say("\nepisode contrasts: s_AB = P(same cluster | same A, diff B), s_BA = P(same cluster | same B, diff A)")
say("%-8s %-15s %9s %9s %10s %7s %9s" % ("part", "pair", "s_AB", "s_BA", "diff", "ratio", "base"))
for h in PARTS:
    for A, B in PAIRS:
        r = R["check3"]["contrast"][f"{h}|{A}x{B}"]
        say("%-8s %-15s %9.5f %9.5f %+10.5f %7.3f %9.5f" % (h, A + "x" + B, r["s_AB"], r["s_BA"], r["diff"], r["ratio"], r["base"]))

# ---------------------------------------------------------------- check 3b
say("\n=== CHECK 3b: through the heads, cross-modal (image head on row i, caption head on row j), selection rows ===")
R["check3b"] = {"by_aspect": {}, "contrast": {}}
bI, bJ = sample_pairs(gS, N_SAMPLE)


def sample_mean_check(name, exact, mask, vals):
    v = vals[mask]
    n = len(v)
    mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))
    CHECKS.append((name, exact, mean, (mean - exact) / se))


for h in PARTS:
    Pi, Pt = post[h]["img"], post[h]["txt"]
    agree = np.einsum("nc,nc->n", Pi[bI], Pt[bJ])
    for a in ASPECTS:
        m = labS[a] >= 0
        x, g, pi_, pt_ = labS[a][m], gS[m], Pi[m], Pt[m]
        s_all, n_all = dots2([], pi_, pt_, g)
        s_same, n_same = dots2([x], pi_, pt_, g)
        mean_all, mean_same, mean_diff = s_all / n_all, s_same / n_same, (s_all - s_same) / (n_all - n_same)
        R["check3b"]["by_aspect"][f"{h}|{a}"] = {"mean_same": mean_same, "mean_diff": mean_diff, "mean_all": mean_all,
                                                "ratio_same_over_diff": mean_same / mean_diff, "n_labelled_rows": int(m.sum())}
        ok = (labS[a][bI] >= 0) & (labS[a][bJ] >= 0)
        sx = labS[a][bI] == labS[a][bJ]
        sample_mean_check(f"c3b {h}|{a} same", mean_same, ok & sx, agree)
        sample_mean_check(f"c3b {h}|{a} diff", mean_diff, ok & ~sx, agree)
        sample_mean_check(f"c3b {h}|{a} all", mean_all, ok, agree)
    for A, B in PAIRS:
        m = (labS[A] >= 0) & (labS[B] >= 0)
        xa, xb, g, pi_, pt_ = labS[A][m], labS[B][m], gS[m], Pi[m], Pt[m]
        sab, nab = dots2([xa, xb], pi_, pt_, g)
        sa, na = dots2([xa], pi_, pt_, g)
        sb, nb = dots2([xb], pi_, pt_, g)
        mAB, mBA = (sa - sab) / (na - nab), (sb - sab) / (nb - nab)
        R["check3b"]["contrast"][f"{h}|{A}x{B}"] = {"mean_sameA_diffB": mAB, "mean_sameB_diffA": mBA, "diff": mAB - mBA,
                                                   "ratio": mAB / mBA}
        ok = (labS[A][bI] >= 0) & (labS[B][bI] >= 0) & (labS[A][bJ] >= 0) & (labS[B][bJ] >= 0)
        qa, qb = labS[A][bI] == labS[A][bJ], labS[B][bI] == labS[B][bJ]
        sample_mean_check(f"c3b {h}|{A}x{B} sameA_diffB", mAB, ok & qa & ~qb, agree)
        sample_mean_check(f"c3b {h}|{A}x{B} sameB_diffA", mBA, ok & qb & ~qa, agree)

say("agreement = dot(image-head posterior of row i, caption-head posterior of row j); uniform-random 64-class baseline = 1/64 = %.5f" % (1 / 64))
say("%-8s %-8s %9s %9s %9s %7s" % ("part", "aspect", "same", "diff", "overall", "ratio"))
for h in PARTS:
    for a in ASPECTS:
        r = R["check3b"]["by_aspect"][f"{h}|{a}"]
        say("%-8s %-8s %9.5f %9.5f %9.5f %7.3f" % (h, a, r["mean_same"], r["mean_diff"], r["mean_all"], r["ratio_same_over_diff"]))
say("\nepisode contrasts through the heads: mean agreement (sameA,diffB) minus (sameB,diffA)")
say("%-8s %-15s %9s %9s %10s %7s" % ("part", "pair", "sameA/dB", "sameB/dA", "diff", "ratio"))
for h in PARTS:
    for A, B in PAIRS:
        r = R["check3b"]["contrast"][f"{h}|{A}x{B}"]
        say("%-8s %-15s %9.5f %9.5f %+10.5f %7.3f" % (h, A + "x" + B, r["mean_sameA_diffB"], r["mean_sameB_diffA"], r["diff"], r["ratio"]))

# ---------------------------------------------------------------- cross-check summary
zs_ = np.array([c[3] for c in CHECKS])
bad = [c for c in CHECKS if abs(c[3]) > Z_MAX]
say("\n=== EXACT vs SAMPLED (%d pairs, seed 0): %d comparisons, max |z| = %.2f, mean |z| = %.2f, beyond %.0f SE: %d ===" %
    (N_SAMPLE, len(CHECKS), np.abs(zs_).max(), np.abs(zs_).mean(), Z_MAX, len(bad)))
for c in bad:
    say("  FAIL %s exact %.6f sampled %.6f z %.2f" % c)
R["crosscheck"] = {"n_comparisons": len(CHECKS), "max_abs_z": float(np.abs(zs_).max()), "n_beyond": len(bad),
                   "rows": [{"name": c[0], "exact": c[1], "sampled": c[2], "z": c[3]} for c in CHECKS]}
say("runtime %.1f s" % (time.time() - T0))
R["runtime_s"] = time.time() - T0
(OUT / "profile.json").write_text(json.dumps(R, indent=1, default=float))
(OUT / "profile.txt").write_text("\n".join(LINES) + "\n")
assert not bad, "exact and sampled disagree beyond 4 SE"
