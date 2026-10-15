"""EXPLORATORY (decides nothing). ADDENDUM_0a.md in this folder (SHA-256 asserted) fixes: the share F = (H - 1) / (R_u - 1)
of the painting-level ceiling that the image head reaches, on scorer-train rows of paintings unseen by the head's
60,000-row fitting draw, for R0 and L. Reuses run_checks (stage-a code, partitions) and run_told_oracle (head recipe).

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261115_grouping_step0_checks/run_addendum_0a.py [--smoke]
    -> results/step0a_addendum.{json,txt}; --smoke -> results/smoke/ (overwritable, 3,000-row heads).
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
import importlib.util  # noqa: E402

# The old quick-checks module is also called run_checks, so import the told oracle first (it loads that one), then load
# this folder's run_checks.py under another name.
sys.path.insert(0, str(HERE.parent / "20261111_community_told_oracle"))
import run_told_oracle  # noqa: E402,F401
_spec = importlib.util.spec_from_file_location("step0_run_checks", HERE / "run_checks.py")
rk = importlib.util.module_from_spec(_spec)
sys.modules["step0_run_checks"] = rk
_spec.loader.exec_module(rk)

rto, rc, rg, n6, df = rk.rto, rk.rc, rk.rg, rk.n6, rk.df
from src.data.artelingo_splits import artelingo_splits  # noqa: E402

ADDENDUM = HERE / "ADDENDUM_0a.md"
ADDENDUM_SHA = "56ba5b5f27e399f99c8e1722168d67d46123b6f20f7d7eae9208bc18cadc110d"
N_PERM, N_BOOT, BOOT_SEED, BRUTE_PAINTINGS = 20, 1000, 0, 300
NO_ROOM, ROOM = 0.75, 0.40
STORED_ACC = {"R0": ("n6", None), "L": ("told", None)}


def head_image(ctx, lab, scorer_train, n_rows):
    """Local copy of run_told_oracle.fit_one_head's image branch that returns the classifier and the draw."""
    draw = np.random.default_rng(rc.PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(1).choice(rest, min(n6.CHECK_ROWS, len(rest)), replace=False)
    F = ctx.data.img_features
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(C=1.0, max_iter=300).fit(rc.unit(F[draw]), lab[draw])
    acc = 100 * float(clf.score(rc.unit(F[check]), lab[check]))
    return clf, draw, acc, int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))


def paint_stats(P, lab, g, K, n_paint):
    """Per-painting sums: n_p, S_p = sum_i p_i (K), c_p = group counts (K), A_p = S_p.c_p - sum_i p_i(g_i)."""
    n = np.bincount(g, minlength=n_paint).astype(np.float64)
    S = np.stack([np.bincount(g, weights=P[:, k], minlength=n_paint) for k in range(K)], 1)
    c = np.zeros((n_paint, K))
    np.add.at(c, (g, lab), 1.0)
    own = P[np.arange(len(lab)), lab]
    A = (S * c).sum(1) - np.bincount(g, weights=own, minlength=n_paint)
    return {"n": n, "S": S, "c": c, "A": A, "Sc": (S * c).sum(1)}


def measures(st, w=None):
    """H, R_u from per-painting sums; w = painting copy counts (bootstrap), copies distinct paintings."""
    n, S, c, A, Sc = st["n"], st["S"], st["c"], st["A"], st["Sc"]
    w = np.ones(len(n)) if w is None else w
    N = (w * n).sum()
    within = (w * n * (n - 1)).sum()                     # ordered same-painting pairs
    num = (w * A).sum() / within
    C = (w[:, None] * c).sum(0)
    Sw = (w[:, None] * S).sum(0)
    den = (Sw @ C - (w * Sc).sum()) / (N * N - (w * n * n).sum())
    H = num / den
    same_in = (w * (c * (c - 1) / 2).sum(1)).sum()       # unordered, same painting and group
    p_same = same_in / (within / 2)
    tot_same = (C * (C - 1) / 2).sum()
    p_diff = (tot_same - same_in) / (N * (N - 1) / 2 - within / 2)
    return float(H), float(p_same / p_diff), float(p_same), float(p_diff)


def brute(P, lab, g):
    """Loop-free-of-formula brute force on a small subset: ordered pairs with M[i, j] = p_i(g_j)."""
    M = P[:, lab]
    same_p = g[:, None] == g[None, :]
    eye = np.eye(len(g), dtype=bool)
    H = M[same_p & ~eye].mean() / M[~same_p].mean()
    bs, bd = rk.pairs_bruteforce(lab, g)
    return float(H), float(bs / bd)


def ci(v):
    v = np.asarray(v)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def reading(F):
    return "No room" if F >= NO_ROOM else "Room" if F <= ROOM else "Limited room"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    if rg.sha_file(ADDENDUM) != ADDENDUM_SHA:
        raise SystemExit("ADDENDUM_0a.md differs from the dispatched version (SHA-256)")
    d = HERE / "results" / ("smoke" if smoke else "")
    paths = {"json": d / "step0a_addendum.json", "txt": d / "step0a_addendum.txt"}
    if not smoke and any(p.exists() for p in paths.values()):
        raise SystemExit(f"{d} already holds step0a_addendum results; refusing to overwrite")
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_perm, n_boot = (2, 50) if smoke else (N_PERM, N_BOOT)
    rows = n6.SMOKE_HEAD_ROWS if smoke else n6.HEAD_ROWS
    inputs = [n6.PARTITIONS, rk.TOLD_NPZ, rk.TOLD_JSON, rk.STORED_N6, df.POSTERIORS]
    prov = {"addendum_sha256": ADDENDUM_SHA, "script_sha256": rg.sha_file(Path(__file__)),
            "run_checks_sha256": rg.sha_file(Path(rk.__file__)), "run_told_oracle_sha256": rg.sha_file(Path(rto.__file__)),
            "inputs_sha256": {rc.rel(p): rg.sha_file(p) for p in inputs}, "started": rk.now_ams()}

    ctx = rg.EvalContext(df.SEED, False)
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    groups = np.asarray(ctx.groups)
    g_all = np.unique(groups[scorer_train], return_inverse=True)[1].astype(np.int64)
    e2 = rk.e2_partitions(groups, scorer_train)
    z = np.load(rk.TOLD_NPZ)
    local = {"R0": e2["affect"], "L": np.asarray(z["partition_L"], np.int64)}
    if not np.array_equal(np.asarray(z["partition_R0"], np.int64), local["R0"]):
        raise AssertionError("told oracle's partition_R0 differs from E2's affect partition")
    checks = {"E2_partitions_sha_and_painting_alignment": True, "told_oracle_partition_R0_equals_E2_affect": True}
    stored_acc = {"R0": json.loads(rk.STORED_N6.read_text())["heads"]["affect"]["heldout_accuracy"]["img"],
                  "L": json.loads(rk.TOLD_JSON.read_text())["arms"]["L"]["head"]["heldout_accuracy"]["img"]}
    checks["stored_accuracies_equal_ADDENDUM"] = all(round(stored_acc[k], 2) == rk.A_HEAD_PLAN[k] for k in stored_acc)
    if not checks["stored_accuracies_equal_ADDENDUM"]:
        raise AssertionError(f"stored accuracies {stored_acc} differ from the addendum's 13.47 / 9.81")
    post_stored = n6.load_posteriors(df.POSTERIORS, ctx)
    sel = ctx.selection

    # identical image features within a painting (all scorer-train rows)
    Fimg = ctx.data.img_features[scorer_train]
    first = np.zeros(int(g_all.max()) + 1, dtype=np.int64)
    first[g_all[::-1]] = np.arange(len(g_all))[::-1]
    dev = np.abs(Fimg - Fimg[first[g_all]]).max(axis=1)
    bad_p = np.zeros(len(first), bool)
    np.logical_or.at(bad_p, g_all, dev > 0)
    identical = {"paintings_total": int(len(first)), "paintings_with_differing_image_feature": int(bad_p.sum()),
                 "max_abs_deviation": float(dev.max())}
    del Fimg

    draw_ids = None
    out = {}
    for arm in ("R0", "L"):
        t = time.time()
        lab_l = local[arm]
        lab_glob = rto.global_labels(lab_l, scorer_train, len(groups))
        clf, draw, acc, nwarn = head_image(ctx, lab_glob, scorer_train, rows)
        if not smoke:
            if acc != stored_acc[arm]:
                raise SystemExit(f"{arm}: refit image held-out accuracy {acc!r} differs from stored {stored_acc[arm]!r}")
        eq_post = None
        if arm == "R0" and not smoke:
            mine = clf.predict_proba(rc.unit(ctx.data.img_features[sel])).astype(np.float32)
            eq_post = bool(np.array_equal(mine, post_stored["affect"]["img"][sel]))
            if not eq_post:
                raise SystemExit("R0: refit selection posteriors differ from stored n6_posteriors affect__img")
        K = len(clf.classes_)
        if not np.array_equal(clf.classes_, np.arange(K)):
            raise AssertionError(f"{arm}: classes are not 0..K-1")
        seen_paint = np.unique(g_all[np.isin(scorer_train, draw)])
        unseen_row = ~np.isin(g_all, seen_paint)
        n_unseen_paint = int(len(np.unique(g_all[unseen_row])))
        gu = np.unique(g_all[unseen_row], return_inverse=True)[1].astype(np.int64)
        Pu = clf.predict_proba(rc.unit(ctx.data.img_features[scorer_train[unseen_row]])).astype(np.float64)
        labu = lab_l[unseen_row]
        st = paint_stats(Pu, labu, gu, K, n_unseen_paint)
        H, R_u, p_same, p_diff = measures(st)
        pc = rk.pair_counts(labu, gu)
        assert np.isclose(pc["p_same"], p_same, rtol=1e-12, atol=0) and np.isclose(pc["p_diff"], p_diff, rtol=1e-12, atol=0)
        # brute force on the first BRUTE_PAINTINGS unseen paintings
        sub = gu < BRUTE_PAINTINGS
        bH, bR = brute(Pu[sub], labu[sub], gu[sub])
        sub_st = paint_stats(Pu[sub], labu[sub], gu[sub], K, BRUTE_PAINTINGS)
        fH, fR, _, _ = measures(sub_st)
        brute_ok = bool(np.isclose(bH, fH, rtol=1e-12, atol=0) and np.isclose(bR, fR, rtol=1e-12, atol=0))
        if not brute_ok:
            raise AssertionError(f"{arm}: per-painting-sum H/R_u differ from brute force: {(bH, fH, bR, fR)}")
        # random control
        ctrl = []
        for s in range(n_perm):
            perm = np.random.default_rng(s).permutation(lab_l)[unseen_row]
            h, r, _, _ = measures(paint_stats(Pu, perm, gu, K, n_unseen_paint))
            ctrl.append((h, r))
        ctrl = np.asarray(ctrl)
        # painting bootstrap
        rng = np.random.default_rng(BOOT_SEED)
        bh, br = [], []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, n_unseen_paint, n_unseen_paint), minlength=n_unseen_paint).astype(np.float64)
            h, r, _, _ = measures(st, w)
            bh.append(h)
            br.append(r)
        bh, br = np.asarray(bh), np.asarray(br)
        bF = (bh - 1) / (br - 1)
        F = (H - 1) / (R_u - 1)
        out[arm] = {
            "n_groups": int(K), "unseen_paintings": n_unseen_paint, "unseen_rows": int(unseen_row.sum()),
            "seen_paintings_in_scorer_train": int(len(seen_paint)), "scorer_train_paintings": int(g_all.max()) + 1,
            "scorer_train_rows": int(len(g_all)), "draw_rows": int(len(draw)),
            "head": {"heldout_accuracy_img": acc, "stored_heldout_accuracy_img": stored_acc[arm],
                     "equals_stored": bool(acc == stored_acc[arm]), "selection_posteriors_bit_equal_stored": eq_post,
                     "convergence_warnings": nwarn},
            "R_u": R_u, "H": H, "F": F, "p_same": p_same, "p_diff": p_diff,
            "H_ci95": ci(bh), "R_u_ci95": ci(br), "F_ci95": ci(bF), "bootstrap": {"n": n_boot, "seed": BOOT_SEED},
            "control": {"seeds": list(range(n_perm)), "H": rk.summary(ctrl[:, 0]), "R_u": rk.summary(ctrl[:, 1]),
                        "H_per_seed": ctrl[:, 0].tolist(), "R_u_per_seed": ctrl[:, 1].tolist()},
            "bruteforce_300_paintings": {"rows": int(sub.sum()), "H_brute": bH, "H_formula": fH, "R_u_brute": bR,
                                         "R_u_formula": fR, "equal_rtol_1e-12": brute_ok},
            "pair_counts_equal_run_checks": True, "reading": reading(F),
            "reading_rule": f"No room: F >= {NO_ROOM}; Room: F <= {ROOM}; Limited room otherwise"}
        rc.log(f"{arm}: unseen {n_unseen_paint} paintings / {int(unseen_row.sum())} rows, R_u {R_u:.4f} H {H:.4f} F {F:.3f} "
               f"CI {ci(bF)} control H {ctrl[:, 0].mean():.4f} R_u {ctrl[:, 1].mean():.4f} [{time.time() - t:.0f}s]")
    checks["refit_image_heldout_equals_stored"] = "skipped (smoke)" if smoke else True
    checks["R0_selection_posteriors_bit_equal_stored"] = "skipped (smoke)" if smoke else True
    checks["H_R_u_vs_bruteforce_300_paintings"] = True
    checks["pair_counts_equal_run_checks_pair_counts"] = True
    prov["finished"] = rk.now_ams()
    result = {"exploratory": "decides nothing; seed 42 development rows only; ADDENDUM_0a.md applied as written",
              "smoke": smoke, "provenance": prov, "checks": checks, "identical_image_feature_within_painting": identical,
              "groupings": out, "runtime_s": round(time.time() - t0, 1)}
    result = rto.roundtrip(result)
    rg.assert_finite_tree(result)
    paths["json"].write_text(json.dumps(result, indent=1, ensure_ascii=False))
    L = [f"EXPLORATORY addendum 0a, share of the painting-level ceiling reached by the image head (addendum sha "
         f"{ADDENDUM_SHA[:12]}{', SMOKE' if smoke else ''})",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in checks.items()),
         f"Image feature identical within painting: {identical}", ""]
    for arm, m in out.items():
        c = m["control"]
        L += [f"{arm}: {m['n_groups']} groups; unseen paintings {m['unseen_paintings']} of {m['scorer_train_paintings']} "
              f"(rows {m['unseen_rows']} of {m['scorer_train_rows']}); image head held-out {m['head']['heldout_accuracy_img']:.2f}%",
              f"  R_u {m['R_u']:.4f} {m['R_u_ci95']}  H {m['H']:.4f} {m['H_ci95']}  F {m['F']:.3f} {m['F_ci95']}",
              f"  control (n={len(c['seeds'])}): H {c['H']['mean']:.4f} [{c['H']['min']:.4f}, {c['H']['max']:.4f}]; "
              f"R_u {c['R_u']['mean']:.4f} [{c['R_u']['min']:.4f}, {c['R_u']['max']:.4f}]",
              f"  reading: {m['reading']} ({m['reading_rule']})"]
    L.append(f"runtime {result['runtime_s']:.0f}s")
    txt = "\n".join(L)
    paths["txt"].write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
