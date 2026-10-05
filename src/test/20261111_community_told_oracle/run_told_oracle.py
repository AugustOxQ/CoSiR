"""EXPLORATORY (decides nothing; seed 42 development episodes only). PLAN.md in this folder fixes the arms, the
measurements and the reading (SHA-256 asserted below). Only the affect partition changes between arms:

  R0  E2's affect partition (MiniBatchKMeans, 64 clusters on the 28 GoEmotions probabilities); stored N6 posteriors
  L   Leiden communities (src.model.communities.detect_communities, defaults) on the same probabilities, communities
      under 200 rows merged into the community (of at least 200 rows) with the nearest centroid (Euclidean, 28-d)
  K   MiniBatchKMeans on the same probabilities with E2's settings and k = L's community count after merging

For L and K only the affect heads are refit (copy of run_n6.fit_heads for one partition, same draw and check rows);
image and caption posteriors and B (= stored C2) stay as in R0. For each arm: group sizes, the partition profile's pair
statistics for affect x emotion (on groups and through the heads), held-out affect head accuracy, the told term
(emotion -> affect, style and genre -> image) and N6's reader term fused on B against their matched condition-free
counterparts (diagnose_counterparts.py procedure), the reader's pick accuracy, the paired per-anchor difference of the
told margin against R0's, and (information only) B' = B rebuilt with the arm's averaged-heads term.
Reuses the reviewed diagnostics code by import; modifies nothing outside this folder. CPU only. Reads no other seed.

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261111_community_told_oracle/run_told_oracle.py
    -> results/told_oracle.json, results/told_oracle.txt, results/per_anchor_told_oracle.npz (refuses to overwrite)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
FIX = HERE.parent / "20261109_fix_diagnostics"
sys.path.insert(0, str(FIX))
import diagnose_counterparts as dc  # noqa: E402  (imports diagnose_fixes, run_checks, run_n6, run_n6c, repo root)

df, rc, rg, n6, n6c = dc.df, dc.rc, dc.rg, dc.n6, dc.n6c

from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, METRICS, compare, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free  # noqa: E402
from src.model.communities import detect_communities  # noqa: E402

ROOT = rc.ROOT
PLAN = HERE / "PLAN.md"
PLAN_SHA = "3b8f90bf22a462600d042f73a2a3b881b0952ec601a7f65a52c2d8f7474027a9"
AFFECT_DIR = ROOT / "src/test/20261018_affect_factor_learning/cache"
AFFECT_NPZ, AFFECT_JSON = AFFECT_DIR / "affect_prepare.npz", AFFECT_DIR / "affect_prepare.json"
STORED_CF = FIX / "results" / "diagnose_counterparts.json"
STORED_FIXES = FIX / "results" / "diagnose_fixes.json"
STORED_N6 = df.RES_QC / "n6_seed42.json"
PROFILE = HERE.parent / "20261110_partition_profile" / "results" / "profile.json"
OUT = HERE / "results"
OUT_JSON, OUT_TXT, OUT_NPZ = OUT / "told_oracle.json", OUT / "told_oracle.txt", OUT / "per_anchor_told_oracle.npz"
MIN_ROWS = 200                                     # PLAN.md arm L: communities under 200 rows are merged
ARMS = ("R0", "L", "K")
CONTRASTS = (("emotion", "style"), ("emotion", "genre"))
PROMISING_TOLD, READER_TARGET = 2.0, 0.5           # PLAN.md "How the numbers will be read"


# ---------------------------------------------------------------- partitions

def merge_small(labels, X, min_rows):
    """Communities under min_rows rows -> the community of at least min_rows rows whose centroid (mean of X over its
    rows, float64) is nearest in Euclidean distance; one pass from the unmerged centroids; labels re-compacted."""
    labels = np.asarray(labels, dtype=np.int64)
    k = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=k)
    X = np.asarray(X, dtype=np.float64)
    cent = np.stack([np.bincount(labels, weights=X[:, j], minlength=k) for j in range(X.shape[1])], axis=1)
    cent /= sizes[:, None]
    big, small = np.flatnonzero(sizes >= min_rows), np.flatnonzero(sizes < min_rows)
    if len(big) == 0:
        raise SystemExit("no community has at least 200 rows: merging as written in PLAN.md is impossible")
    target = np.arange(k)
    moves = []
    for c in small:
        dist = np.linalg.norm(cent[big] - cent[c], axis=1)
        target[c] = big[int(np.argmin(dist))]
        moves.append({"community": int(c), "rows": int(sizes[c]), "into": int(target[c]),
                      "distance": float(dist.min())})
    merged = np.unique(target[labels], return_inverse=True)[1].astype(np.int64)
    return merged, moves


def group_stats(local):
    sizes = np.bincount(local)
    p = sizes[sizes > 0] / sizes.sum()
    return {"n_groups": int((sizes > 0).sum()), "min": int(sizes.min()), "median": float(np.median(sizes)),
            "max": int(sizes.max()), "n_under_200": int((sizes < MIN_ROWS).sum()),
            "effective_n": float(np.exp(-(p * np.log(p)).sum())), "largest_share": float(sizes.max() / sizes.sum()),
            "sizes_desc": sorted(sizes.tolist(), reverse=True)}


# ---------------------------------------------------------------- pair statistics (copied from
# src/test/20261110_partition_profile/profile_partitions.py, which runs on import; checks 3 and 3b only)

def mix(arrs):
    code = np.zeros(len(arrs[0]), dtype=np.int64)
    for a in arrs:
        u, inv = np.unique(a, return_inverse=True)
        code = code * len(u) + inv
    return code


def pairs2(arrs, n):
    if not arrs:
        return n * (n - 1) / 2
    cnt = np.bincount(np.unique(mix(arrs), return_inverse=True)[1]).astype(np.float64)
    return float((cnt * (cnt - 1) / 2).sum())


def dpairs(arrs, g):
    n = len(g)
    return pairs2(arrs, n) - pairs2(list(arrs) + [g], n)


def dots2(arrs, Pi, Pt, g):
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


def pair_stats_groups(cl, lab, g_all):
    """Check 3 for one affect partition (scorer-train local rows): affect x emotion lift and the e x s, e x g contrasts."""
    m = lab["emotion"] >= 0
    c, x, g = cl[m], lab["emotion"][m], g_all[m]
    tot, same_x = dpairs([], g), dpairs([x], g)
    sc, sxc = dpairs([c], g), dpairs([x, c], g)
    p_same, p_diff = sxc / same_x, (sc - sxc) / (tot - same_x)
    out = {"lift": {"p_same": p_same, "p_diff": p_diff, "base": sc / tot, "lift": p_same / p_diff}, "contrast": {}}
    for A, B in CONTRASTS:
        m = (lab[A] >= 0) & (lab[B] >= 0)
        c, xa, xb, g = cl[m], lab[A][m], lab[B][m], g_all[m]
        nA = dpairs([xa], g) - dpairs([xa, xb], g)
        nB = dpairs([xb], g) - dpairs([xa, xb], g)
        sAB = (dpairs([xa, c], g) - dpairs([xa, xb, c], g)) / nA
        sBA = (dpairs([xb, c], g) - dpairs([xa, xb, c], g)) / nB
        out["contrast"][f"{A}x{B}"] = {"s_AB": sAB, "s_BA": sBA, "diff": sAB - sBA, "ratio": sAB / sBA,
                                       "base": dpairs([c], g) / dpairs([], g)}
    return out


def pair_stats_heads(Pi, Pt, labS, gS):
    """Check 3b for one affect head pair (selection rows): mean dot(image head row i, caption head row j) over
    different-painting pairs, same vs different emotion, and the e x s, e x g contrasts."""
    m = labS["emotion"] >= 0
    x, g, pi_, pt_ = labS["emotion"][m], gS[m], Pi[m], Pt[m]
    s_all, n_all = dots2([], pi_, pt_, g)
    s_same, n_same = dots2([x], pi_, pt_, g)
    mean_same, mean_diff = s_same / n_same, (s_all - s_same) / (n_all - n_same)
    out = {"by_aspect": {"mean_same": mean_same, "mean_diff": mean_diff, "mean_all": s_all / n_all,
                         "ratio_same_over_diff": mean_same / mean_diff, "uniform_baseline": 1.0 / Pi.shape[1]},
           "contrast": {}}
    for A, B in CONTRASTS:
        m = (labS[A] >= 0) & (labS[B] >= 0)
        xa, xb, g, pi_, pt_ = labS[A][m], labS[B][m], gS[m], Pi[m], Pt[m]
        sab, nab = dots2([xa, xb], pi_, pt_, g)
        sa, na = dots2([xa], pi_, pt_, g)
        sb, nb = dots2([xb], pi_, pt_, g)
        mAB, mBA = (sa - sab) / (na - nab), (sb - sab) / (nb - nab)
        out["contrast"][f"{A}x{B}"] = {"mean_sameA_diffB": mAB, "mean_sameB_diffA": mBA, "diff": mAB - mBA,
                                       "ratio": mAB / mBA}
    return out


# ---------------------------------------------------------------- heads

def fit_one_head(ctx, lab, scorer_train, n_rows):
    """run_n6.fit_heads restricted to ONE partition (the original loops over E2's three partitions, read from E2's
    file): same row draw (rng PROBE_SEED), same 10,000 check rows (rng 1), same LogisticRegression(C=1, max_iter=300)
    per modality on unit-normalised CLIP features. Nothing here assumes 64 classes."""
    draw = np.random.default_rng(rc.PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(1).choice(rest, min(n6.CHECK_ROWS, len(rest)), replace=False)
    feats = {"img": ctx.data.img_features, "txt": ctx.data.txt_features}
    sel = ctx.selection
    clfs = {m: LogisticRegression(C=1.0, max_iter=300).fit(rc.unit(F[draw]), lab[draw]) for m, F in feats.items()}
    if not np.array_equal(clfs["img"].classes_, clfs["txt"].classes_):
        raise AssertionError("image and caption heads have different classes")
    post = {}
    for m, F in feats.items():
        full = np.full((len(ctx.groups), len(clfs[m].classes_)), np.nan, dtype=np.float32)
        full[sel] = clfs[m].predict_proba(rc.unit(F[sel]))
        if not (np.isfinite(full[sel]).all() and np.isnan(full[~ctx.in_sel]).all()):
            raise AssertionError(f"{m}: posteriors must be finite on selection rows and NaN elsewhere")
        post[m] = full
    counts = np.bincount(lab[check])
    prov = {"n_classes": int(len(clfs["img"].classes_)), "draw_rows_sha256": rg.sha_array(np.sort(draw)),
            "heldout_accuracy": {m: 100 * float(clfs[m].score(rc.unit(F[check]), lab[check])) for m, F in feats.items()},
            "check_majority_share": 100 * float(counts.max() / counts.sum()), "uniform": 100.0 / len(clfs["img"].classes_)}
    return post, prov


def global_labels(local, scorer_train, n_rows_all):
    lab = np.full(n_rows_all, -1, dtype=np.int64)
    lab[scorer_train] = local
    return lab


# ---------------------------------------------------------------- one arm on B

def per_pair_margin(pn, pc, pB, ctx):
    cl, out = ctx.anchor_group, {}
    for i, name in enumerate(rg.POOLED_ORDER):
        mask = ctx.pair_index == i
        a, b, base = df.sub(pn, mask), df.sub(pc, mask), df.sub(pB, mask)
        out[name] = {"margin": {"r1": compare(a, b, cl[mask], "r1"), "gain": compare(a, b, cl[mask], "gain"),
                                "either": rc.point_ci(df.either(a) - df.either(b), cl[mask])},
                     "fusedT_vs_B_r1": compare(a, base, cl[mask], "r1"),
                     "fusedTcf_vs_B_r1": compare(b, base, cl[mask], "r1")}
    return out


def evaluate_arm(post, ctx, B, pB, part_idx):
    """diagnose_counterparts.py section 2 for T6oracle ('told') and T6 ('reader') on B, per pair, plus the reader's
    pick accuracy (diagnose_fixes.py step 4)."""
    ep, cl = ctx.pooled, ctx.anchor_group
    t6, _, t6u, info, _ = n6.n6_terms(post, ep)
    st6 = df.stacked(post, ep, df.PARTS)
    terms = {"told": df.as_f32({c: df.pick(st6, part_idx[c]) for c in CONDITIONS}), "reader": df.as_f32(t6)}
    out, arrays = {}, {}
    for name, s in terms.items():
        rc.assert_finite_scores(s, name)
        nested, ctrl, picks = crossfit_nested(B, B, s, ctx.parity)
        cf, cf_picks = crossfit_condition_free(B, B, dc.cf_version(s), ctx.parity)
        pn, pc = per_anchor(nested), per_anchor(cf)
        out[name] = {"fusedT_vs_B": df.diff(pn, pB, cl), "fusedTcf_vs_B": df.diff(pc, pB, cl),
                     "fusedT_vs_fusedTcf": df.diff(pn, pc, cl), "per_pair": per_pair_margin(pn, pc, pB, ctx),
                     "T_picks": picks, "cf_picks": cf_picks,
                     "control_ranks_as_B": df.same_metrics(per_anchor(ctrl), pB)}
        arrays[name] = (pn, pc)
        rc.log(f"  {name}: margin R@1 {out[name]['fusedT_vs_fusedTcf']['r1']['point']:.2f}")
    correct = {c: info[c]["weights"].argmax(axis=1) == part_idx[c] for c in CONDITIONS}
    both = correct["a"] & correct["b"]
    out["pick"] = {
        "correct_share": rc.point_ci(0.5 * (correct["a"].astype(float) + correct["b"].astype(float)), cl),
        "per_pair_condition": {name: {c: 100 * float(correct[c][ctx.pair_index == i].mean()) for c in CONDITIONS}
                               for i, name in enumerate(rg.POOLED_ORDER)},
        "both_correct_share": 100 * float(both.mean()),
        "picked_partition": {name: {c: {h: 100 * float(np.mean(info[c]["weights"][ctx.pair_index == i].argmax(axis=1) == j))
                                        for j, h in enumerate(df.PARTS)} for c in CONDITIONS}
                             for i, name in enumerate(rg.POOLED_ORDER)}}
    return out, arrays, t6u


def margin_arrays(arrays, name, metric):
    pn, pc = arrays[name]
    if metric == "either":
        return df.either(pn) - df.either(pc)
    return np.asarray(pn[metric]) - np.asarray(pc[metric])


def roundtrip(x):
    return json.loads(json.dumps(x))


# ---------------------------------------------------------------- text

def c(x, w=6):
    return f"{x['point']:+{w}.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"


def text(r):
    L = [f"EXPLORATORY told-partition oracle with community and coarser affect partitions (PLAN.md sha "
         f"{r['plan_sha256'][:12]}); decides nothing; seed 42 dev episodes only (n={r['n_episodes']}, "
         f"{r['n_clusters']} paintings); 95% CIs resample paintings (5,000). Margin = fused T minus fused T_cf on B.",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in r["checks"].items()),
         f"R0 fresh run_n6.fit_heads refit: max |posterior - stored| {r['R0_fresh']['max_abs_diff_vs_stored']}; "
         f"told margin R@1 {c(r['R0_fresh']['told_margin']['r1'])}; reader margin R@1 "
         f"{c(r['R0_fresh']['reader_margin']['r1'])}; held-out acc " + "; ".join(
             f"{h} {v['heldout_accuracy']['img']:.2f}/{v['heldout_accuracy']['txt']:.2f}"
             for h, v in r["R0_fresh"]["heads"].items() if isinstance(v, dict) and "heldout_accuracy" in v),
         f"B (= stored C2): R@1 {c(r['B']['r1'])} gain {c(r['B']['gain'])} either {c(r['B']['either'])}",
         "", "1. Affect partitions (scorer-train rows, 183,694)",
         f"  {'arm':4s} {'groups':>6s} {'min':>6s} {'median':>8s} {'max':>6s} {'<200':>5s} {'effN':>6s} {'largest%':>8s}"]
    for a in ARMS:
        g = r["arms"][a]["groups"]
        L.append(f"  {a:4s} {g['n_groups']:6d} {g['min']:6d} {g['median']:8.0f} {g['max']:6d} {g['n_under_200']:5d} "
                 f"{g['effective_n']:6.1f} {100 * g['largest_share']:8.2f}")
    lr = r["arms"]["L"]["leiden"]
    L.append(f"  L raw Leiden: {lr['raw']['n_groups']} communities (min {lr['raw']['min']}, max {lr['raw']['max']}, "
             f"{lr['raw']['n_under_200']} under 200 rows holding {lr['rows_merged']} rows); after merging "
             f"{r['arms']['L']['groups']['n_groups']}; kNN+Leiden {lr['runtime_s']:.1f}s, merge {lr['merge_runtime_s']:.2f}s")
    L.append(f"  K MiniBatchKMeans {r['arms']['K']['kmeans']['settings']} in {r['arms']['K']['kmeans']['runtime_s']:.1f}s")
    L += ["", "2. Pair statistics, affect partition x emotion (groups: scorer-train; heads: image head row i vs caption "
              "head row j, selection rows; different-painting pairs)",
          f"  {'arm':4s} | groups lift (p_same/p_diff) | e×s s_AB/s_BA ratio | e×g ratio | heads ratio same/diff "
          f"(1/k) | e×s heads ratio | e×g heads ratio"]
    for a in ARMS:
        p = r["arms"][a]["pairs"]
        g, h = p["groups"], p["heads"]
        L.append(f"  {a:4s} | {g['lift']['lift']:5.3f} ({g['lift']['p_same']:.4f}/{g['lift']['p_diff']:.4f}) | "
                 f"{g['contrast']['emotionxstyle']['ratio']:5.3f} | {g['contrast']['emotionxgenre']['ratio']:5.3f} | "
                 f"{h['by_aspect']['ratio_same_over_diff']:5.3f} ({h['by_aspect']['uniform_baseline']:.4f}) | "
                 f"{h['contrast']['emotionxstyle']['ratio']:5.3f} | {h['contrast']['emotionxgenre']['ratio']:5.3f}")
    L += ["", "3. Held-out affect head accuracy (10,000 scorer-train check rows): img / txt (majority-class share; 1/k)"]
    for a in ARMS:
        hd = r["arms"][a]["head"]
        L.append(f"  {a:4s} {hd['heldout_accuracy']['img']:5.1f} / {hd['heldout_accuracy']['txt']:5.1f} "
                 f"(majority {hd['check_majority_share']:.1f}; 1/k {hd['uniform']:.2f}; classes {hd['n_classes']})")
    L += ["", "4. Told term (emotion -> affect, style and genre -> image) fused on B: R@1 | gain | either (pp)"]
    for a in ARMS:
        t = r["arms"][a]["eval"]["told"]
        L.append(f"  {a:4s} fused T - B {c(t['fusedT_vs_B']['r1'])} | fused T_cf - B {c(t['fusedTcf_vs_B']['r1'])} | "
                 f"margin R@1 {c(t['fusedT_vs_fusedTcf']['r1'])} gain {c(t['fusedT_vs_fusedTcf']['gain'])} "
                 f"either {c(t['fusedT_vs_fusedTcf']['either'])}")
        L.append("       per pair margin R@1: " + " | ".join(
            f"{df.PAIR_SHORT[p]} {c(v['margin']['r1'])}" for p, v in t["per_pair"].items()))
        if a != "R0":
            L.append(f"       paired told margin vs R0: R@1 {c(r['arms'][a]['vs_R0']['told']['r1'])} "
                     f"gain {c(r['arms'][a]['vs_R0']['told']['gain'])} either {c(r['arms'][a]['vs_R0']['told']['either'])}")
    L += ["", "5. N6 reader term (hard argmax Delta over affect, image, caption) fused on B"]
    for a in ARMS:
        t, pk = r["arms"][a]["eval"]["reader"], r["arms"][a]["eval"]["pick"]
        L.append(f"  {a:4s} margin R@1 {c(t['fusedT_vs_fusedTcf']['r1'])} gain {c(t['fusedT_vs_fusedTcf']['gain'])} "
                 f"either {c(t['fusedT_vs_fusedTcf']['either'])} | pick accuracy {pk['correct_share']['point']:.1f} "
                 f"[{pk['correct_share']['ci95'][0]:.1f}, {pk['correct_share']['ci95'][1]:.1f}], both correct "
                 f"{pk['both_correct_share']:.1f}%")
        L.append("       per pair margin R@1: " + " | ".join(
            f"{df.PAIR_SHORT[p]} {c(v['margin']['r1'])}" for p, v in t["per_pair"].items())
            + " | pick a/b: " + "; ".join(f"{df.PAIR_SHORT[p]} {v['a']:.1f}/{v['b']:.1f}"
                                         for p, v in pk["per_pair_condition"].items()))
        if a != "R0":
            L.append(f"       paired reader margin vs R0: R@1 {c(r['arms'][a]['vs_R0']['reader']['r1'])}")
    L += ["", "6. Information only: B' = B rebuilt with the arm's averaged-heads term (crossfit_condition_free)"]
    for a in ("L", "K"):
        bp = r["arms"][a]["B_prime"]
        L.append(f"  {a:4s} B' R@1 {c(bp['describe']['r1'])} either {c(bp['describe']['either'])}; "
                 f"B' - B R@1 {c(bp['vs_B']['r1'])}; picks {bp['picks']}")
    L += ["", "7. Reading (PLAN.md, applied literally)"]
    for a in ("L", "K"):
        rd = r["reading"][a]
        L.append(f"  {a}: {rd['label']} | told margin {rd['told_margin']:+.2f} (>= {PROMISING_TOLD}: "
                 f"{rd['told_margin_at_least_2']}); paired vs R0 lower bound {rd['paired_lower']:+.2f} (> 0: "
                 f"{rd['paired_lower'] > 0}); reader margin {rd['reader_margin']:+.2f} vs R0 {rd['reader_margin_R0']:+.2f} "
                 f"(moves toward +0.5: {rd['reader_moves_toward_target']})")
    lk = r["reading"]["L_vs_K"]
    L.append(f"  L vs K: each point inside the other's 95% interval: {lk['within_each_other']} -> {lk['reading']}; "
             f"paired L - K told margin R@1 {c(lk['paired_L_minus_K'])}")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    if any(p.exists() for p in (OUT_JSON, OUT_TXT, OUT_NPZ)):
        raise SystemExit(f"results exist in {OUT}; refusing to overwrite")
    if rg.sha_file(PLAN) != PLAN_SHA:
        raise SystemExit("PLAN.md differs from the dispatched version (SHA-256)")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    checks, arrays_out = {}, {}

    # ---------------------------------------------------------- context, inputs, B = C2 (diagnose_counterparts.py)
    ctx = rg.EvalContext(df.SEED, False)
    ep, cl = ctx.pooled, ctx.anchor_group
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    inp, _, a3_prov = rc.model_inputs(ctx, "A3", scorer_train, False)
    t_n1u = centered_term(inp, ep, uniform=True)
    if rg.sha_file(df.POSTERIORS) != n6c.POSTERIORS_SHA:
        raise AssertionError("stored posteriors: SHA-256 differs from ADDENDUM_3_N6C.md")
    post_stored = n6.load_posteriors(df.POSTERIORS, ctx)
    _, _, t6u_stored, _, _ = n6.n6_terms(post_stored, ep)
    B, picks_B = crossfit_condition_free(ctx.cos, t_n1u, t6u_stored, ctx.parity)
    pB = per_anchor(B)
    gate = np.load(df.GATE)
    if not (np.array_equal(gate["anchor_group"], cl) and np.array_equal(gate["pair_index"], ctx.pair_index)
            and all(np.array_equal(pB[m], gate[f"matched__{m}"]) for m in METRICS)):
        raise SystemExit("per_anchor(B) differs from the stored C2 (matched__*) arrays: stopping")
    checks["B_equals_stored_C2"] = True
    rc.log(f"B reproduces C2 exactly ({ctx.n} episodes)")

    aspect_of = {"a": np.array([rg.PAIRS[i][0] for i in ctx.pair_index]),
                 "b": np.array([rg.PAIRS[i][1] for i in ctx.pair_index])}
    part_idx = {cnd: np.array([df.PARTS.index(df.TOLD_PART[a]) for a in aspect_of[cnd]]) for cnd in CONDITIONS}

    # ---------------------------------------------------------- labels and partitions inputs
    data = ctx.data
    lab_all = artelingo_aspect_labels(data)
    lab_st = {a: lab_all[a][scorer_train] for a in ("emotion", "style", "genre")}
    lab_sel = {a: lab_all[a][ctx.selection] for a in ("emotion", "style", "genre")}
    g_st = np.unique(ctx.groups[scorer_train], return_inverse=True)[1]
    g_sel = ctx.groups[ctx.selection]
    affect_rec = json.loads(AFFECT_JSON.read_text())
    if rg.sha_file(AFFECT_NPZ) != affect_rec["affect_npz_sha256"]:
        raise AssertionError("affect_prepare.npz: SHA-256 differs from affect_prepare.json")
    zA = np.load(AFFECT_NPZ)
    probs, affect_local = zA["affect_probs"], zA["affect_local"].astype(np.int64)
    if rg.sha_array(probs) != affect_rec["affect_probs_sha256"] or probs.shape != (len(scorer_train), 28):
        raise AssertionError("affect_probs: SHA-256 or shape differs from affect_prepare.json")
    e2_affect = np.load(n6.PARTITIONS)["affect"]
    if rg.sha_file(n6.PARTITIONS) != n6.PARTITIONS_SHA or not np.array_equal(e2_affect, affect_local):
        raise AssertionError("affect_local differs from E2's affect partition")
    n6.partition_labels(ctx.groups, scorer_train)          # asserts E2's local_groups align with scorer_train
    checks["affect_local_equals_E2_affect"] = True
    km_settings = dict(affect_rec["kmeans_settings"])
    km64 = MiniBatchKMeans(**km_settings).fit_predict(probs).astype(np.int64)
    checks["E2_kmeans_settings_reproduce_E2_affect"] = bool(np.array_equal(km64, affect_local))
    rc.log(f"E2 k-means settings reproduce E2's affect partition: {checks['E2_kmeans_settings_reproduce_E2_affect']}")

    # ---------------------------------------------------------- R0: stored posteriors
    result = {"exploratory": "decides nothing; seed 42 development episodes only; PLAN.md applied as written",
              "plan_sha256": PLAN_SHA, "n_episodes": int(ctx.n), "n_clusters": int(len(np.unique(cl))),
              "provenance": {"script_sha256": rg.sha_file(Path(__file__)), "a3": a3_prov,
                             "posteriors_sha256": rg.sha_file(df.POSTERIORS), "c2_arrays_sha256": rg.sha_file(df.GATE),
                             "affect_npz_sha256": affect_rec["affect_npz_sha256"], "episodes_sha256": ctx.shas,
                             "diagnose_counterparts_sha256": rg.sha_file(Path(dc.__file__)),
                             "diagnose_fixes_sha256": rg.sha_file(Path(df.__file__)),
                             "run_n6_sha256": rg.sha_file(Path(n6.__file__))},
              "B": df.describe(pB, cl), "B_picks": picks_B, "arms": {}}
    rc.log("R0 (stored posteriors) on B")
    ev_r0, arr_r0, _ = evaluate_arm(post_stored, ctx, B, pB, part_idx)
    stored_cf = json.loads(STORED_CF.read_text())["matched"]
    stored_fix = json.loads(STORED_FIXES.read_text())["reader"]
    keys = ("fusedT_vs_B", "fusedTcf_vs_B", "fusedT_vs_fusedTcf", "T_picks", "cf_picks")
    ok_told = all(roundtrip(ev_r0["told"][k]) == stored_cf["T6oracle"][k] for k in keys)
    ok_reader = all(roundtrip(ev_r0["reader"][k]) == stored_cf["T6"][k] for k in keys)
    ok_pick = (roundtrip(ev_r0["pick"]["correct_share"]) == stored_fix["correct_share"]
               and ev_r0["pick"]["both_correct_share"] == stored_fix["both_correct_share"])
    checks["R0_stored_told_equals_diagnose_counterparts"] = ok_told
    checks["R0_stored_reader_equals_diagnose_counterparts"] = ok_reader
    checks["R0_stored_pick_equals_diagnose_fixes"] = ok_pick
    if not (ok_told and ok_reader and ok_pick):
        print(json.dumps({"checks": checks, "told": roundtrip(ev_r0["told"]["fusedT_vs_fusedTcf"]),
                          "reader": roundtrip(ev_r0["reader"]["fusedT_vs_fusedTcf"]),
                          "pick": roundtrip(ev_r0["pick"]["correct_share"])}, indent=1))
        raise SystemExit("R0 does not reproduce the stored told/reader margins: stopping (PLAN.md)")
    rc.log("R0 reproduces the stored told and reader margins exactly")

    # ---------------------------------------------------------- R0: fresh fit_heads refit (original function)
    rc.log("R0 fresh refit with run_n6.fit_heads (all three partitions)")
    post_fresh, heads_fresh = n6.fit_heads(ctx, n6.partition_labels(ctx.groups, scorer_train), scorer_train,
                                           n6.HEAD_ROWS)
    sel = ctx.selection
    refit_diff = {h: {m: float(np.abs(post_fresh[h][m][sel].astype(np.float64)
                                      - post_stored[h][m][sel].astype(np.float64)).max())
                      for m in ("img", "txt")} for h in df.PARTS}
    refit_equal = {h: all(np.array_equal(post_fresh[h][m][sel], post_stored[h][m][sel]) for m in ("img", "txt"))
                   for h in df.PARTS}
    stored_heads = json.loads(STORED_N6.read_text())["heads"]
    checks["fresh_refit_bit_identical"] = refit_equal
    checks["fresh_refit_heldout_equals_stored"] = all(
        heads_fresh[h]["heldout_accuracy"] == stored_heads[h]["heldout_accuracy"] for h in df.PARTS)
    rc.log(f"fresh refit bit-identical: {refit_equal}; max |diff| {refit_diff}")
    rc.log("R0 (fresh posteriors) on B")
    ev_r0f, _, _ = evaluate_arm(post_fresh, ctx, B, pB, part_idx)
    checks["R0_fresh_told_equals_stored"] = all(roundtrip(ev_r0f["told"][k]) == stored_cf["T6oracle"][k] for k in keys)
    checks["R0_fresh_reader_equals_stored"] = all(roundtrip(ev_r0f["reader"][k]) == stored_cf["T6"][k] for k in keys)
    fresh_block = {"max_abs_diff_vs_stored": refit_diff, "bit_identical": refit_equal, "heads": heads_fresh,
                   "told_margin": ev_r0f["told"]["fusedT_vs_fusedTcf"],
                   "reader_margin": ev_r0f["reader"]["fusedT_vs_fusedTcf"],
                   "pick_correct_share": ev_r0f["pick"]["correct_share"]}
    rc.log(f"R0 fresh: told margin {ev_r0f['told']['fusedT_vs_fusedTcf']['r1']['point']:.4f}, reader margin "
           f"{ev_r0f['reader']['fusedT_vs_fusedTcf']['r1']['point']:.4f}")

    # the one-partition copy of fit_heads must equal the original on R0's affect labels
    lab_r0 = global_labels(affect_local, scorer_train, len(ctx.groups))
    post_copy, prov_copy = fit_one_head(ctx, lab_r0, scorer_train, n6.HEAD_ROWS)
    copy_diff = max(float(np.abs(post_copy[m][sel].astype(np.float64)
                                 - post_fresh["affect"][m][sel].astype(np.float64)).max()) for m in ("img", "txt"))
    checks["fit_one_head_equals_fit_heads_affect"] = bool(
        all(np.array_equal(post_copy[m], post_fresh["affect"][m], equal_nan=True) for m in ("img", "txt"))
        and prov_copy["heldout_accuracy"] == heads_fresh["affect"]["heldout_accuracy"])
    checks["fit_one_head_max_abs_diff"] = copy_diff
    if copy_diff > 1e-4:
        raise AssertionError("the one-partition copy of fit_heads differs from run_n6.fit_heads on R0's affect labels")

    # ---------------------------------------------------------- pair statistics: copy must reproduce the profile (R0)
    prof = json.loads(PROFILE.read_text())
    ps_r0 = {"groups": pair_stats_groups(affect_local, lab_st, g_st),
             "heads": pair_stats_heads(post_stored["affect"]["img"][sel].astype(np.float64),
                                       post_stored["affect"]["txt"][sel].astype(np.float64), lab_sel, g_sel)}
    ref = [(ps_r0["groups"]["lift"][k], prof["check3"]["lift"]["affect|emotion"][k]) for k in ("p_same", "p_diff", "lift")]
    ref += [(ps_r0["groups"]["contrast"][f"{A}x{B_}"][k], prof["check3"]["contrast"][f"affect|{A}x{B_}"][k])
            for A, B_ in CONTRASTS for k in ("s_AB", "s_BA", "ratio")]
    ref += [(ps_r0["heads"]["by_aspect"][k], prof["check3b"]["by_aspect"]["affect|emotion"][k])
            for k in ("mean_same", "mean_diff", "ratio_same_over_diff")]
    ref += [(ps_r0["heads"]["contrast"][f"{A}x{B_}"][k], prof["check3b"]["contrast"][f"affect|{A}x{B_}"][k])
            for A, B_ in CONTRASTS for k in ("mean_sameA_diffB", "mean_sameB_diffA", "ratio")]
    checks["pair_stats_copy_reproduces_profile_R0"] = bool(all(np.isclose(a, b, rtol=1e-9, atol=0) for a, b in ref))
    if not checks["pair_stats_copy_reproduces_profile_R0"]:
        raise AssertionError("copied pair statistics do not reproduce profile.json for R0")

    result["R0_fresh"] = fresh_block
    result["arms"]["R0"] = {"groups": group_stats(affect_local), "pairs": ps_r0,
                            "head": {**heads_fresh["affect"], "check_majority_share": prov_copy["check_majority_share"],
                                     "uniform": prov_copy["uniform"], "note": "fresh refit (equals stored)"
                                     if refit_equal["affect"] else "fresh refit; stored posteriors used for R0's terms"},
                            "eval": ev_r0}
    for name in ("told", "reader"):
        for m in ("r1", "gain", "other"):
            arrays_out[f"R0__{name}__fused__{m}"] = np.asarray(arr_r0[name][0][m])
            arrays_out[f"R0__{name}__cf__{m}"] = np.asarray(arr_r0[name][1][m])
    del post_copy, post_fresh

    # ---------------------------------------------------------- arm L: Leiden communities
    rc.log("arm L: kNN union graph + Leiden (detect_communities defaults)")
    t1 = time.time()
    raw = detect_communities(probs)
    leiden_s = time.time() - t1
    t1 = time.time()
    lab_L, moves = merge_small(raw, probs, MIN_ROWS)
    merge_s = time.time() - t1
    k_L = int(lab_L.max()) + 1
    rc.log(f"arm L: {raw.max() + 1} raw communities -> {k_L} after merging ({leiden_s:.0f}s)")

    # ---------------------------------------------------------- arm K: k-means with k = k_L, E2's settings
    t1 = time.time()
    km_K = {**km_settings, "n_clusters": k_L}
    lab_K = MiniBatchKMeans(**km_K).fit_predict(probs).astype(np.int64)
    kmeans_s = time.time() - t1
    rc.log(f"arm K: MiniBatchKMeans {km_K} ({kmeans_s:.1f}s)")
    arrays_out.update(partition_L_raw=raw, partition_L=lab_L, partition_K=lab_K, partition_R0=affect_local)

    # ---------------------------------------------------------- arms L and K: refit affect heads, measure
    for arm, local in (("L", lab_L), ("K", lab_K)):
        rc.log(f"arm {arm}: affect heads")
        post_aff, prov = fit_one_head(ctx, global_labels(local, scorer_train, len(ctx.groups)), scorer_train,
                                      n6.HEAD_ROWS)
        post = {"affect": post_aff, "image": post_stored["image"], "caption": post_stored["caption"]}
        pairs = {"groups": pair_stats_groups(local, lab_st, g_st),
                 "heads": pair_stats_heads(post_aff["img"][sel].astype(np.float64),
                                           post_aff["txt"][sel].astype(np.float64), lab_sel, g_sel)}
        rc.log(f"arm {arm}: told and reader on B")
        ev, arr, t6u_arm = evaluate_arm(post, ctx, B, pB, part_idx)
        vs = {}
        for name in ("told", "reader"):
            vs[name] = {m: rc.point_ci(margin_arrays(arr, name, m) - margin_arrays(arr_r0, name, m), cl)
                        for m in ("r1", "gain", "either")}
            for m in ("r1", "gain", "other"):
                arrays_out[f"{arm}__{name}__fused__{m}"] = np.asarray(arr[name][0][m])
                arrays_out[f"{arm}__{name}__cf__{m}"] = np.asarray(arr[name][1][m])
        Bp, picks_Bp = crossfit_condition_free(ctx.cos, t_n1u, t6u_arm, ctx.parity)
        pBp = per_anchor(Bp)
        entry = {"groups": group_stats(local), "pairs": pairs, "head": prov, "eval": ev, "vs_R0": vs,
                 "B_prime": {"describe": df.describe(pBp, cl), "vs_B": df.diff(pBp, pB, cl), "picks": picks_Bp}}
        if arm == "L":
            entry["leiden"] = {"raw": group_stats(raw), "moves": moves, "rows_merged": int(sum(m["rows"] for m in moves)),
                               "runtime_s": leiden_s, "merge_runtime_s": merge_s,
                               "settings": "detect_communities(probs) defaults: k=20 kNN union graph (Euclidean), "
                                           "ModularityVertexPartition, seed 42"}
        else:
            entry["kmeans"] = {"settings": km_K, "runtime_s": kmeans_s}
        result["arms"][arm] = entry
        arrays_out[f"{arm}__t_told_margin_r1_minus_R0"] = margin_arrays(arr, "told", "r1") - margin_arrays(arr_r0, "told", "r1")

    # ---------------------------------------------------------- reading (PLAN.md, literally)
    r0_reader = ev_r0["reader"]["fusedT_vs_fusedTcf"]["r1"]["point"]
    reading = {}
    for arm in ("L", "K"):
        told = result["arms"][arm]["eval"]["told"]["fusedT_vs_fusedTcf"]["r1"]["point"]
        rdr = result["arms"][arm]["eval"]["reader"]["fusedT_vs_fusedTcf"]["r1"]["point"]
        lower = result["arms"][arm]["vs_R0"]["told"]["r1"]["ci95"][0]
        toward = bool(rdr > r0_reader)
        promising = bool(told >= PROMISING_TOLD or (lower > 0 and toward))
        not_better = bool(lower <= 0)
        label = ("promising and not better (both rules fire)" if promising and not_better else
                 "promising" if promising else "not better" if not_better else "in between")
        reading[arm] = {"label": label, "told_margin": told, "told_margin_at_least_2": bool(told >= PROMISING_TOLD),
                        "paired_lower": lower, "reader_margin": rdr, "reader_margin_R0": r0_reader,
                        "reader_moves_toward_target": toward}
    mL = result["arms"]["L"]["eval"]["told"]["fusedT_vs_fusedTcf"]["r1"]
    mK = result["arms"]["K"]["eval"]["told"]["fusedT_vs_fusedTcf"]["r1"]
    within = bool(mK["ci95"][0] <= mL["point"] <= mK["ci95"][1] and mL["ci95"][0] <= mK["point"] <= mL["ci95"][1])
    pa_L = arrays_out["L__t_told_margin_r1_minus_R0"] + margin_arrays(arr_r0, "told", "r1")
    pa_K = arrays_out["K__t_told_margin_r1_minus_R0"] + margin_arrays(arr_r0, "told", "r1")
    reading["L_vs_K"] = {"within_each_other": within,
                         "reading": "the change comes from granularity, not from Leiden" if within
                         else "L and K differ beyond each other's intervals",
                         "paired_L_minus_K": rc.point_ci(pa_L - pa_K, cl)}
    result["reading"] = reading
    result["checks"] = checks
    result["runtime_s"] = round(time.time() - t0, 1)

    arrays_out.update(anchor_group=cl, pair_index=ctx.pair_index,
                      **{f"B__{m}": np.asarray(pB[m]) for m in ("r1", "gain", "other")})
    result = roundtrip(result)
    rg.assert_finite_tree(result)
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    np.savez_compressed(OUT_NPZ, **arrays_out)
    txt = text(result)
    OUT_TXT.write_text(txt + "\n")
    print(txt)
    rc.log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
