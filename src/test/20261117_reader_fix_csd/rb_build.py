"""R-b, the learned reader (DECISION_RULE.md §4.2, items 1 to 6): painting halves, cross-fitted heads, pseudo-aspect
banks and the two half-readers. Every setting is fixed by the rule; this script chooses none. Evaluation on seed 42 is
rb_eval.py. CPU only; reads no held row and no evaluation label.

Stages (each asserts the rule's SHA-256 through common.assert_rule and refuses to overwrite non-smoke outputs):
  halves                       item 1  -> results/rb_halves.npz/.json
  heads --grouping G           item 2  -> results/rb_heads_<G>.npz/.json   (G in affect, image, caption, csd, rand)
  bank --config X [--half J]   item 3  -> results/rb_bank_<X>_half<J>.npz/.json   (X in A1, A0, AR)
  train --config X             items 4 to 6 -> results/rb_reader_<X>.pkl/.json/.npz

Index spaces. "Local row" = position in the ascending scorer-train row list (183,694 rows); the grouping label arrays,
the halves, the banks and the cross-fitted posteriors all live in it. Heads are fitted on global rows (the feature
arrays are global); local and global orders agree because scorer_train is strictly ascending (asserted).

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261117_reader_fix_csd/rb_build.py <stage> [...] [--smoke]
--smoke: heads on 3,000 rows, banks of 256 episodes per block, outputs in results/smoke/ (overwritable). Smoke numbers
are not results.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402  (puts the repo root on sys.path)
import rb_features as rf  # noqa: E402

ROOT = C.ROOT
T = ROOT / "src/test"

# ---------------------------------------------------------------- constants written in the rule (§4.2)
CONFIGS = C.CONFIGS
GROUPINGS = C.GROUPINGS
N_ST, N_PAINTINGS, HALF_PAINTINGS, HALF_SEED = 183_694, 36_518, 18_259, 0             # item 1
HEAD_ROWS, SMOKE_HEAD_ROWS, CHECK_ROWS = 60_000, 3_000, 10_000                         # item 2
DRAW_SEED, CHECK_SEED_OWN, CHECK_SEED_OTHER = 0, 1, 2                                  # item 2
HEAD_C, HEAD_MAX_ITER = 1.0, 300                                                       # item 2 (fit_one_head)
BANK_N, SMOKE_BANK_N, BANK_SEEDS, MIN_PAINTINGS = 16_384, 256, (11_700, 11_800), 30    # item 3
READER_MAX_ITER = 2000                                                                 # item 6
VALIDATE_N = 1000                                                                      # as build_banks.py
BLOCKS = {                                                                             # item 3, as written
    "A1": [("affect", "caption"), ("affect", "csd"), ("affect", "image"), ("caption", "csd"), ("caption", "image"),
           ("csd", "image")],
    "AR": [("affect", "caption"), ("affect", "image"), ("affect", "rand"), ("caption", "image"), ("caption", "rand"),
           ("image", "rand")],
    "A0": [("affect", "caption"), ("affect", "image"), ("caption", "image")],
}
THIRD = {"A0": {("affect", "caption"): "image", ("affect", "image"): "caption", ("caption", "image"): "affect"},
         "A1": {}, "AR": {}}
BANK_SIZE = {"A1": 98_304, "AR": 98_304, "A0": 49_152}                                 # item 3, per half
N_GROUPS = {"affect": 41, "image": 64, "caption": 64, "csd": 17, "rand": 17}           # D1
STANDARD_ACC = {"affect": (9.81, 35.72), "image": (92.7, 22.9), "caption": (21.6, 89.7), # item 2: image / caption
                "csd": (85.10, 40.41), "rand": (15.64, 13.20)}
SOURCES = {   # D1: file, SHA-256, key (local scorer-train arrays)
    "affect": (T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz",
               "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366", "partition_L"),
    "image": (T / "20261031_pseudo_partitions/results/partitions.npz",
              "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa", "image"),
    "caption": (T / "20261031_pseudo_partitions/results/partitions.npz",
                "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa", "caption"),
    "csd": (T / "20261116_grouping_step1_style/results/step1_group_style.npz",
            "b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2", "style_csd"),
    "rand": (T / "20261116_grouping_step1_style/results/step1_group_style.npz",
             "b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2", "style_rand"),
}
EPISODE_FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")

if rf.C_GRID != (0.01, 0.1, 1.0, 10.0, 100.0) or (rf.N_FOLDS, rf.FOLD_SEED) != (5, 0):
    raise AssertionError("rb_features' C grid or folds differ from the rule (§4.2 item 6)")


# ---------------------------------------------------------------- small helpers

def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def paths(stem, smoke, exts=("npz", "json", "txt")):
    d = C.res_dir(smoke)
    return {e: d / f"{stem}.{e}" for e in exts}


def start(p, smoke):
    """Rule check first, then refuse to overwrite non-smoke outputs."""
    C.assert_rule()
    if not smoke and any(q.exists() for q in p.values()):
        raise SystemExit(f"outputs exist ({', '.join(rel(q) for q in p.values() if q.exists())}); refusing to overwrite")
    next(iter(p.values())).parent.mkdir(parents=True, exist_ok=True)


def header(stage, smoke, inputs=()):
    return {"stage": stage, "smoke": bool(smoke), "exploratory": "development work on scorer-train rows; decides nothing "
            "by itself (DECISION_RULE.md §4.2)", "rule_sha256": C.RULE_SHA,
            "provenance": {"script_sha256": C.sha_file(Path(__file__)), "rb_features_sha256": C.sha_file(Path(rf.__file__)),
                           "common_sha256": C.sha_file(Path(C.__file__)), "git_head": C.git_head(),
                           "inputs_sha256": {rel(q): C.sha_file(q) for q in inputs}, "argv": sys.argv,
                           "pid": os.getpid(), "started_amsterdam": C.now_ams(),
                           "threads": {k: os.environ.get(k) for k in
                                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}}


def finish(p, rec, text_fn, npz=None, pkl=None):
    """npz / pkl first (their SHA-256 go into the JSON), then the JSON (marks the stage done), then the text."""
    if npz is not None:
        np.savez(p["npz"], **npz)
        rec["provenance"]["npz_sha256"] = C.sha_file(p["npz"])
    if pkl is not None:
        with open(p["pkl"], "wb") as f:
            pickle.dump(pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
        rec["provenance"]["pkl_sha256"] = C.sha_file(p["pkl"])
    rec["provenance"]["finished_amsterdam"] = C.now_ams()
    rec = C.jsonable(rec)
    C.assert_finite_tree(rec)
    p["json"].write_text(json.dumps(rec, indent=1, allow_nan=False))
    txt = text_fn(rec)
    p["txt"].write_text(txt + "\n")
    print(txt, flush=True)


def load_json_npz(p, smoke, what):
    if not p["json"].exists():
        raise SystemExit(f"{rel(p['json'])} is missing: run {what}{' --smoke' if smoke else ''} first")
    rec = json.loads(p["json"].read_text())
    if rec["rule_sha256"] != C.RULE_SHA or rec["smoke"] != bool(smoke):
        raise SystemExit(f"{rel(p['json'])}: written under another rule or smoke flag")
    for ext in ("npz", "pkl"):
        if ext in p and f"{ext}_sha256" in rec["provenance"]:
            if C.sha_file(p[ext]) != rec["provenance"][f"{ext}_sha256"]:
                raise SystemExit(f"{rel(p[ext])}: SHA-256 differs from its JSON")
    z = np.load(p["npz"]) if "npz" in p else None
    return rec, z


def splits():
    """(data, splits, scorer_train, groups): artelingo rows and the grouped splits; scorer_train strictly ascending."""
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_splits
    data = load_artelingo()
    sp = artelingo_splits(data)
    st, groups = np.asarray(sp.scorer_train, dtype=np.int64), np.asarray(sp.groups)
    if len(st) != N_ST or not (np.diff(st) > 0).all():
        raise AssertionError("scorer_train must hold 183,694 strictly ascending rows")
    return data, sp, st, groups


# ---------------------------------------------------------------- halves (item 1)

def halves_paths(smoke):
    return paths("rb_halves", smoke)


def stage_halves(args):
    p = halves_paths(args.smoke)
    start(p, args.smoke)
    t0 = time.time()
    _, _, st, groups = splits()
    paint = groups[st]                                       # painting id (leakage group) of each local row
    paintings, g_st = np.unique(paint, return_inverse=True)  # sorted painting ids; local painting code
    if len(paintings) != N_PAINTINGS or 2 * HALF_PAINTINGS != N_PAINTINGS:
        raise AssertionError(f"expected {N_PAINTINGS} scorer-train paintings, got {len(paintings)}")
    perm = np.random.default_rng(HALF_SEED).permutation(paintings)
    order = np.random.default_rng(HALF_SEED).permutation(len(paintings))
    if not np.array_equal(perm, paintings[order]):
        raise AssertionError("permutation of the painting ids differs from the permutation of their positions")
    half_p = {0: perm[:HALF_PAINTINGS], 1: perm[HALF_PAINTINGS:]}
    half_of_code = np.zeros(N_PAINTINGS, dtype=np.int8)
    half_of_code[order[HALF_PAINTINGS:]] = 1
    half_of_row = half_of_code[g_st]
    local = {k: np.flatnonzero(half_of_row == k).astype(np.int64) for k in (0, 1)}
    for k in (0, 1):
        if not np.array_equal(np.unique(paint[local[k]]), np.sort(half_p[k])):
            raise AssertionError(f"half {k}: rows do not follow their paintings")
    if len(local[0]) + len(local[1]) != N_ST or np.intersect1d(local[0], local[1]).size:
        raise AssertionError("the halves must partition the scorer-train rows")
    e2 = np.load(SOURCES["image"][0])
    if C.sha_file(SOURCES["image"][0]) != SOURCES["image"][1] or not np.array_equal(e2["local_groups"], g_st):
        raise AssertionError("local painting codes differ from E2's partitions.npz local_groups")
    rec = header("halves", args.smoke, [SOURCES["image"][0]])
    rec.update(
        rule_text="item 1: the 36,518 scorer-train paintings (sorted painting ids) permuted with "
                  "numpy.random.default_rng(0).permutation; half 0 = first 18,259, half 1 = the rest; rows follow paintings",
        n_scorer_train_rows=N_ST, n_paintings=N_PAINTINGS,
        paintings_per_half={str(k): int(len(half_p[k])) for k in (0, 1)},
        rows_per_half={str(k): int(len(local[k])) for k in (0, 1)},
        sha256={**{f"paintings_half{k}_permuted_order": sha_array(half_p[k].astype(np.int64)) for k in (0, 1)},
                **{f"paintings_half{k}_sorted": sha_array(np.sort(half_p[k]).astype(np.int64)) for k in (0, 1)},
                **{f"local_rows_half{k}": sha_array(local[k]) for k in (0, 1)}},
        painting_id="artelingo_splits().groups (leakage groups) of the scorer-train rows",
        runtime_s=round(time.time() - t0, 1))
    npz = {"scorer_train": st, "painting_of_local_row": paint.astype(np.int64), "g_st": g_st.astype(np.int64),
           "paintings_sorted": paintings.astype(np.int64), "paintings_half0": half_p[0].astype(np.int64),
           "paintings_half1": half_p[1].astype(np.int64), "half_of_local_row": half_of_row,
           "local_rows_half0": local[0], "local_rows_half1": local[1],
           "global_rows_half0": st[local[0]], "global_rows_half1": st[local[1]]}
    finish(p, rec, text_halves, npz)


def text_halves(r):
    return (f"R-b halves{' [SMOKE]' if r['smoke'] else ''}: {r['n_paintings']} paintings -> "
            f"{r['paintings_per_half']['0']} / {r['paintings_per_half']['1']}; rows {r['rows_per_half']['0']} / "
            f"{r['rows_per_half']['1']} (of {r['n_scorer_train_rows']}); painting-list SHA-256 (permuted order) "
            f"{r['sha256']['paintings_half0_permuted_order'][:16]} / {r['sha256']['paintings_half1_permuted_order'][:16]}")


def load_halves(smoke):
    rec, z = load_json_npz(halves_paths(smoke), smoke, "rb_build.py halves")
    h = {k: z[k] for k in z.files}
    h["sha256"] = rec["provenance"]["npz_sha256"]
    return h


# ---------------------------------------------------------------- grouping labels (D1)

def load_groupings(names, st, g_st):
    """{name: local label array (183,694,)} from the files D1 names (SHA-256 asserted), each checked to be a local
    scorer-train array whose index is the position in the ascending scorer-train row list."""
    if len(st) != N_ST or not (np.diff(st) > 0).all():
        raise AssertionError("scorer_train must be strictly ascending")
    out, cache = {}, {}
    for name in names:
        path, sha, key = SOURCES[name]
        if path not in cache:
            if C.sha_file(path) != sha:
                raise SystemExit(f"{rel(path)}: SHA-256 differs from DECISION_RULE.md D1")
            cache[path] = np.load(path)
        z = cache[path]
        lab = np.asarray(z[key], dtype=np.int64)
        if name in ("image", "caption"):
            ok = np.array_equal(z["local_groups"], g_st)
        elif name in ("csd", "rand"):
            ok = np.array_equal(z["scorer_train"], st) and np.array_equal(z["g_st"], g_st)
        else:   # affect: the told-oracle file's R0 partition is E2's affect partition, row for row
            e2 = np.load(SOURCES["image"][0])
            ok = (C.sha_file(SOURCES["image"][0]) == SOURCES["image"][1] and np.array_equal(e2["local_groups"], g_st)
                  and np.array_equal(np.asarray(z["partition_R0"], dtype=np.int64), e2["affect"]))
        if not ok:
            raise AssertionError(f"{name}: label array is not aligned with the ascending scorer-train rows")
        counts = np.bincount(lab) if lab.min() >= 0 else None
        if lab.shape != (N_ST,) or counts is None or len(counts) != N_GROUPS[name] or (counts == 0).any():
            raise AssertionError(f"{name}: expected {N_GROUPS[name]} non-empty groups 0..{N_GROUPS[name] - 1} on "
                                 f"{N_ST} local rows")
        if name in ("csd", "rand"):
            first = np.unique(g_st, return_index=True)[1]
            if not np.array_equal(lab[first][g_st], lab):
                raise AssertionError(f"{name}: labels are not constant within a painting")
        out[name] = lab
    return out


# ---------------------------------------------------------------- heads (item 2)

def heads_paths(g, smoke):
    return paths(f"rb_heads_{g}", smoke)


def fit_head_pair(F, lab, pool, n_rows, predict_rows, other_rows=None, check_rows=CHECK_ROWS):
    """fit_one_head's recipe on an arbitrary row pool: draw = default_rng(0).choice(pool, n_rows, replace=False);
    LogisticRegression(C=1, max_iter=300) per modality on unit-normalised CLIP features; held-out accuracy on 10,000
    pool rows outside the draw (default_rng(1)) and, when other_rows is given, on 10,000 of them (default_rng(2)).
    Returns ({m: posteriors on predict_rows (float32)}, provenance, classes)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    unit = _rc().unit
    draw = np.random.default_rng(DRAW_SEED).choice(pool, n_rows, replace=False)
    rest = np.setdiff1d(pool, draw)
    check_own = np.random.default_rng(CHECK_SEED_OWN).choice(rest, min(check_rows, len(rest)), replace=False)
    check_other = (np.random.default_rng(CHECK_SEED_OTHER).choice(other_rows, min(check_rows, len(other_rows)),
                                                                  replace=False) if other_rows is not None else None)
    post, prov, classes = {}, {"draw_rows_sha256": sha_array(np.sort(draw)), "n_draw": int(n_rows),
                               "heldout_accuracy_own_half": {}, "accuracy_other_half": {}, "convergence_warnings": {},
                               "fit_s": {}}, None
    for m in ("img", "txt"):
        t = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LogisticRegression(C=HEAD_C, max_iter=HEAD_MAX_ITER).fit(unit(F[m][draw]), lab[draw])
        prov["convergence_warnings"][m] = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
        if classes is None:
            classes = np.asarray(clf.classes_)
        elif not np.array_equal(classes, clf.classes_):
            raise AssertionError("image and caption heads have different classes")
        post[m] = clf.predict_proba(unit(F[m][predict_rows])).astype(np.float32)
        prov["heldout_accuracy_own_half"][m] = 100 * float(clf.score(unit(F[m][check_own]), lab[check_own]))
        if check_other is not None:
            prov["accuracy_other_half"][m] = 100 * float(clf.score(unit(F[m][check_other]), lab[check_other]))
        prov["fit_s"][m] = round(time.time() - t, 1)
    prov["n_classes"] = int(len(classes))
    prov["check_rows_own_sha256"] = sha_array(np.sort(check_own))
    if check_other is not None:
        prov["check_rows_other_sha256"] = sha_array(np.sort(check_other))
    return post, prov, classes


_RC = None


def _rc():
    """The quick-checks module (unit, PROBE_SEED) through run_told_oracle's import chain."""
    global _RC
    if _RC is None:
        sys.path.insert(0, str(T / "20261111_community_told_oracle"))
        import run_told_oracle as rto
        if rto.rc.PROBE_SEED != DRAW_SEED or rto.n6.CHECK_ROWS != CHECK_ROWS or rto.n6.HEAD_ROWS != HEAD_ROWS:
            raise AssertionError("fit_one_head's draw seed, check rows or head rows differ from the rule's")
        _RC = SimpleNamespace(rto=rto, unit=rto.rc.unit)
    return _RC


def stage_heads(args):
    g = args.grouping
    p = heads_paths(g, args.smoke)
    start(p, args.smoke)
    t0 = time.time()
    H = load_halves(args.smoke)
    data, sp, st, groups = splits()
    if not np.array_equal(st, H["scorer_train"]):
        raise AssertionError("scorer_train differs from the halves stage's")
    local = load_groupings([g], st, H["g_st"])[g]
    rto = _rc().rto
    lab = rto.global_labels(local, st, len(groups))
    F = {"img": data.img_features, "txt": data.txt_features}
    n_rows = SMOKE_HEAD_ROWS if args.smoke else HEAD_ROWS
    checks = {}

    # the recipe copy must equal run_told_oracle.fit_one_head on its own pool (all scorer-train rows), 3,000-row draw
    sel = np.asarray(sp.selection)
    in_sel = np.zeros(len(groups), dtype=bool)
    in_sel[sel] = True
    ctx_like = SimpleNamespace(data=data, selection=sel, groups=groups, in_sel=in_sel)
    ref_post, ref_prov = rto.fit_one_head(ctx_like, lab, st, SMOKE_HEAD_ROWS)
    mine, mprov, _ = fit_head_pair(F, lab, st, SMOKE_HEAD_ROWS, sel)
    checks["recipe_copy_equals_fit_one_head_(3000-row draw on scorer_train)"] = bool(
        all(np.array_equal(mine[m], ref_post[m][sel]) for m in ("img", "txt"))
        and mprov["draw_rows_sha256"] == ref_prov["draw_rows_sha256"]
        and mprov["heldout_accuracy_own_half"] == ref_prov["heldout_accuracy"])
    if not all(checks.values()):
        raise SystemExit("the head recipe copy differs from run_told_oracle.fit_one_head: stopping")
    del ref_post, mine
    C.log(f"{g}: recipe copy equals fit_one_head")

    G = N_GROUPS[g]
    cf = {m: np.full((N_ST, G), np.nan, dtype=np.float32) for m in ("img", "txt")}
    filled_by = np.full(N_ST, -1, dtype=np.int8)
    heads, classes_of = {}, {}
    for k in (0, 1):
        own_g, other_g = H[f"global_rows_half{k}"], H[f"global_rows_half{1 - k}"]
        other_l = H[f"local_rows_half{1 - k}"]
        post, prov, classes = fit_head_pair(F, lab, own_g, n_rows, other_g, other_rows=other_g)
        missing = sorted(set(range(G)) - set(classes.tolist()))
        prov.update(n_groups=G, all_groups_in_draw=not missing, missing_groups=missing,
                    standard_heldout_accuracy={"img": STANDARD_ACC[g][0], "txt": STANDARD_ACC[g][1]},
                    posteriors_on="all rows of half %d" % (1 - k))
        if missing and not args.smoke:
            raise SystemExit(f"{g}, half {k}: groups {missing} are missing from the 60,000-row draw; stopping "
                             "(DECISION_RULE.md §4.2 item 2: the step goes to the user)")
        for m in ("img", "txt"):
            full = np.zeros((len(other_l), G), dtype=np.float32)
            full[:, classes] = post[m]
            cf[m][other_l] = full
        filled_by[other_l] = k
        heads[str(k)], classes_of[k] = prov, classes
        C.log(f"{g} half {k}: img own/other {prov['heldout_accuracy_own_half']['img']:.2f}/"
              f"{prov['accuracy_other_half']['img']:.2f}, txt {prov['heldout_accuracy_own_half']['txt']:.2f}/"
              f"{prov['accuracy_other_half']['txt']:.2f} (standard {STANDARD_ACC[g]}) "
              f"[{time.time() - t0:.0f}s]")
    if not (np.array_equal(filled_by, 1 - H["half_of_local_row"]) and all(np.isfinite(cf[m]).all() for m in cf)):
        raise AssertionError("every local row must carry finite posteriors from the other half's heads")
    checks["row_of_half_j_scored_by_heads_of_half_1-j"] = True
    rec = header("heads", args.smoke, [SOURCES[g][0], halves_paths(args.smoke)["npz"]])
    rec.update(grouping=g, n_groups=G, head_rows=n_rows, halves_npz_sha256=H["sha256"], label_sha256=sha_array(local),
               checks=checks, heads=heads, runtime_s=round(time.time() - t0, 1),
               layout="img/txt: (183,694 local rows, n_groups) float32; row r carries the posterior of the head "
                      "trained on the half that does NOT contain r (filled_by_half)")
    npz = {"img": cf["img"], "txt": cf["txt"], "filled_by_half": filled_by,
           "classes_half0": classes_of[0], "classes_half1": classes_of[1]}
    finish(p, rec, text_heads, npz)


def text_heads(r):
    L = [f"R-b cross-fitted heads, grouping {r['grouping']} ({r['n_groups']} groups){' [SMOKE]' if r['smoke'] else ''}; "
         f"draw {r['head_rows']} rows per half; accuracy (%) on 10,000 own-half rows outside the draw (rng 1) and on "
         f"10,000 other-half rows (rng 2); standard heads' held-out accuracy beside",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in r["checks"].items()),
         f"  {'half':4s} {'classes':>7s} {'img own':>8s} {'img oth':>8s} {'img std':>8s} | {'txt own':>8s} "
         f"{'txt oth':>8s} {'txt std':>8s} | conv img/txt | fit s img/txt"]
    for k, h in r["heads"].items():
        a, b, s = h["heldout_accuracy_own_half"], h["accuracy_other_half"], h["standard_heldout_accuracy"]
        L.append(f"  {k:4s} {h['n_classes']:7d} {a['img']:8.2f} {b['img']:8.2f} {s['img']:8.2f} | {a['txt']:8.2f} "
                 f"{b['txt']:8.2f} {s['txt']:8.2f} | {h['convergence_warnings']['img']}/{h['convergence_warnings']['txt']}"
                 f" | {h['fit_s']['img']:.0f}/{h['fit_s']['txt']:.0f}"
                 + (f"  MISSING {h['missing_groups']}" if h["missing_groups"] else ""))
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


def load_heads(g, smoke, halves_sha):
    rec, z = load_json_npz(heads_paths(g, smoke), smoke, f"rb_build.py heads --grouping {g}")
    if rec["halves_npz_sha256"] != halves_sha:
        raise SystemExit(f"rb_heads_{g}: built on other halves")
    return {"img": z["img"], "txt": z["txt"], "filled_by_half": z["filled_by_half"]}, rec


# ---------------------------------------------------------------- banks (item 3)

def bank_paths(config, j, smoke):
    return paths(f"rb_bank_{config}_half{j}", smoke)


def subset_episodes(ep, a, b, idx):
    from src.eval.aspect_episodes import AspectEpisodes
    return AspectEpisodes(a, b, *(getattr(ep, f.name)[idx] for f in fields(ep)[2:]))


def stage_bank(args):
    from src.eval.aspect_episodes import PaintingValueIndex, eligible_values, validate_aspect_episodes
    from src.train.pseudo_partitions import build_episode_bank
    config = args.config
    halves = (0, 1) if args.half is None else (args.half,)
    ps = {j: bank_paths(config, j, args.smoke) for j in halves}
    C.assert_rule()
    for j in halves:
        start(ps[j], args.smoke)
    H = load_halves(args.smoke)
    parts = CONFIGS[config]
    labels = load_groupings(parts, H["scorer_train"], H["g_st"])
    partitions = {k: labels[k] for k in parts}                 # exactly the keys of the configuration
    paint = H["painting_of_local_row"]                         # groups = painting id of each local row
    blocks = rf.bank_blocks(partitions)
    if blocks != BLOCKS[config]:
        raise AssertionError(f"{config}: builder blocks {blocks} differ from the rule's {BLOCKS[config]}")
    n = SMOKE_BANK_N if args.smoke else BANK_N
    index = PaintingValueIndex(partitions, paint)
    for j in halves:
        t0 = time.time()
        rows = H[f"local_rows_half{j}"]
        if not (np.diff(rows) > 0).all():
            raise AssertionError("half rows must be ascending")
        try:
            bank = build_episode_bank(partitions, paint, rows, n_per_pair=n, seed=BANK_SEEDS[j],
                                      min_paintings=MIN_PAINTINGS)
        except RuntimeError as e:
            raise SystemExit(f"{config} half {j}: a block cannot be filled ({e}); stopping (DECISION_RULE.md §4.2 "
                             "item 3 and §8: the step goes to the user)")
        build_s = time.time() - t0
        total = len(blocks) * n
        if len(bank.anchor) != total or (not args.smoke and total != BANK_SIZE[config]):
            raise AssertionError(f"{config} half {j}: {len(bank.anchor)} episodes, expected {total}")
        in_half = np.zeros(N_ST, dtype=bool)
        in_half[rows] = True
        if not in_half[bank.rows()].all():
            raise AssertionError(f"{config} half {j}: a bank row lies outside half {j}")
        # block i = episodes i*n .. (i+1)*n - 1: every episode's pairs share its block's groupings (all episodes)
        layout, checked = [], {}
        rng = np.random.default_rng(42)
        sample = np.sort(rng.choice(total, size=min(VALIDATE_N, total), replace=False))
        for i, (a, b) in enumerate(blocks):
            sl = slice(i * n, (i + 1) * n)
            la, lb = partitions[a], partitions[b]
            ok = bool((la[bank.pairs_a_img[sl]] == la[bank.pairs_a_txt[sl]]).all()
                      and (lb[bank.pairs_b_img[sl]] == lb[bank.pairs_b_txt[sl]]).all()
                      and (la[bank.pairs_a_img[sl]] != la[bank.anchor[sl]][:, None]).all()
                      and (la[bank.candidates[sl, 0]] == la[bank.anchor[sl]]).all()
                      and (lb[bank.candidates[sl, 1]] == lb[bank.anchor[sl]]).all())
            if not ok:
                raise AssertionError(f"{config} half {j} block {i} ({a}, {b}): episodes do not follow the block layout")
            idx = sample[(sample >= sl.start) & (sample < sl.stop)]
            validate_aspect_episodes(subset_episodes(bank, a, b, idx), partitions, paint, index,
                                     third=THIRD[config].get((a, b)))
            checked[f"{a}__{b}"] = int(len(idx))
            layout.append({"block": i, "first": a, "second": b, "third_controlled": THIRD[config].get((a, b)),
                           "seed": BANK_SEEDS[j] + i, "episodes": [i * n, (i + 1) * n - 1]})
        rec = header("bank", args.smoke, [SOURCES[k][0] for k in parts] + [halves_paths(args.smoke)["npz"]])
        rec.update(config=config, half=j, groupings=list(parts), keys_sorted=sorted(partitions), n_per_pair=n,
                   seed=BANK_SEEDS[j], min_paintings=MIN_PAINTINGS, n_episodes=int(total), blocks=layout,
                   validated_per_block=checked, validation="passed (validate_aspect_episodes on a 1,000-episode "
                   "sample, default_rng(42), split by block; block layout checked on every episode)",
                   halves_npz_sha256=H["sha256"],
                   label_sha256={k: sha_array(partitions[k]) for k in parts},
                   eligible_values={k: len(eligible_values(partitions[k], paint, rows, MIN_PAINTINGS)) for k in parts},
                   rows_used=int(len(np.unique(bank.rows()))), build_s=round(build_s, 1),
                   runtime_s=round(time.time() - t0, 1))
        npz = {f: getattr(bank, f) for f in EPISODE_FIELDS}
        npz.update(block_pairs=np.array([f"{a}__{b}" for a, b in blocks]), block_size=np.array(n),
                   seed=np.array(BANK_SEEDS[j]), half=np.array(j))
        finish(ps[j], rec, text_bank, npz)


def text_bank(r):
    return "\n".join([f"R-b bank {r['config']} half {r['half']}{' [SMOKE]' if r['smoke'] else ''}: {r['n_episodes']} "
                      f"episodes ({r['n_per_pair']} per block, seed {r['seed']}, min_paintings {r['min_paintings']}); "
                      f"keys {r['keys_sorted']}; rows used {r['rows_used']}; eligible values {r['eligible_values']}; "
                      f"build {r['build_s']:.0f}s"]
                     + [f"  block {b['block']} ({b['first']}, {b['second']}; third {b['third_controlled']}) seed "
                        f"{b['seed']} episodes {b['episodes'][0]}..{b['episodes'][1]}; validated "
                        f"{r['validated_per_block'][b['first'] + '__' + b['second']]}" for b in r["blocks"]])


def load_bank(config, j, smoke, halves_sha):
    from src.eval.aspect_episodes import AspectEpisodes
    rec, z = load_json_npz(bank_paths(config, j, smoke), smoke, f"rb_build.py bank --config {config}")
    if rec["halves_npz_sha256"] != halves_sha or rec["half"] != j or rec["config"] != config:
        raise SystemExit(f"rb_bank_{config}_half{j}: built on other halves, half or configuration")
    ep = AspectEpisodes("mixed", "mixed", *(z[f].astype(np.int64) for f in EPISODE_FIELDS))
    blocks = [tuple(s.split("__")) for s in z["block_pairs"].tolist()]
    if blocks != BLOCKS[config]:
        raise AssertionError(f"rb_bank_{config}_half{j}: block order differs from the rule")
    return ep, blocks, int(z["block_size"]), rec


# ---------------------------------------------------------------- readers (items 4 to 6)

def reader_paths(config, smoke):
    return paths(f"rb_reader_{config}", smoke, exts=("pkl", "npz", "json", "txt"))


def bank_features(config, smoke):
    """Per half j: features of both conditions of half j's bank from the cross-fitted posteriors (rows of half j carry
    the heads of half 1 - j), labels, episode index; and the inputs' records."""
    H = load_halves(smoke)
    parts = CONFIGS[config]
    post, head_recs = {}, {}
    for h in parts:
        hp, hrec = load_heads(h, smoke, H["sha256"])
        if not np.array_equal(hp["filled_by_half"], 1 - H["half_of_local_row"]):
            raise AssertionError(f"rb_heads_{h}: rows are not scored by the other half's heads")
        post[h], head_recs[h] = {"img": hp["img"], "txt": hp["txt"]}, hrec
    out, bank_recs = {}, {}
    for j in (0, 1):
        ep, blocks, n, brec = load_bank(config, j, smoke, H["sha256"])
        if not (H["half_of_local_row"][ep.rows()] == j).all():
            raise AssertionError(f"bank half {j}: a row outside half {j}, its posteriors would not be cross-fitted")
        F = rf.both_conditions(post, parts, ep)
        ya, yb = rf.bank_labels(blocks, n, parts)
        X, y, epi = rf.stack_conditions(F["a"], F["b"], ya, yb)
        if not np.isfinite(X).all():
            raise AssertionError(f"bank half {j}: non-finite features")
        out[j] = {"X": X, "y": y, "episode": epi, "n_episodes": len(ep.anchor)}
        bank_recs[j] = brec
    return out, H, head_recs, bank_recs


def fit_half_reader(X, y, epi, n_episodes, n_classes):
    """Item 6 for one half: scaler on all of the half's bank, C by 5-fold CV over episodes (log loss, ties to the
    smaller C), refit at the chosen C. Returns (scaler, model, record, out-of-fold probabilities at the chosen C)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    fold = rf.episode_folds(n_episodes)[epi]
    labels = np.arange(n_classes)
    table, oof_all = [], []
    for C_ in rf.C_GRID:
        t = time.time()
        oof = np.full((len(y), n_classes), np.nan)
        losses, warn = [], []
        for f in range(rf.N_FOLDS):
            tr, te = fold != f, fold == f
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                m = LogisticRegression(C=C_, solver="lbfgs", max_iter=READER_MAX_ITER).fit(Xs[tr], y[tr])
            warn.append(int(sum(issubclass(w.category, ConvergenceWarning) for w in caught)))
            if not np.array_equal(m.classes_, labels):
                raise AssertionError("a training fold lacks a class")
            oof[te] = m.predict_proba(Xs[te])
            losses.append(float(log_loss(y[te], oof[te], labels=labels)))
        if not np.isfinite(oof).all():
            raise AssertionError("an example got no out-of-fold probability")
        table.append({"C": C_, "mean_log_loss": float(np.mean(losses)), "fold_log_loss": losses,
                      "convergence_warnings": warn, "oof_accuracy": 100 * float(np.mean(oof.argmax(axis=1) == y)),
                      "fit_s": round(time.time() - t, 1)})
        oof_all.append(oof)
    best = rf.choose_c([r["mean_log_loss"] for r in table])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(C=rf.C_GRID[best], solver="lbfgs", max_iter=READER_MAX_ITER).fit(Xs, y)
    refit_warn = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
    # multinomial (sklearn >= 1.5 always uses the multinomial loss with lbfgs for 3+ classes): softmax of the scores
    dfun = model.decision_function(Xs[:1000])
    soft = np.exp(dfun - dfun.max(axis=1, keepdims=True))
    soft /= soft.sum(axis=1, keepdims=True)
    if not np.allclose(soft, model.predict_proba(Xs[:1000]), rtol=0, atol=1e-10):
        raise AssertionError("the refit reader is not multinomial (predict_proba != softmax of decision_function)")
    oof = oof_all[best]
    rec = {"cv_table": table, "chosen_C": rf.C_GRID[best], "chosen_index": int(best),
           "refit_convergence_warnings": refit_warn, "oof_accuracy_at_chosen_C": 100 * float(np.mean(oof.argmax(1) == y)),
           "n_examples": int(len(y)), "n_episodes": int(n_episodes), "n_iter_refit": np.asarray(model.n_iter_).tolist(),
           "class_counts": np.bincount(y, minlength=n_classes).tolist(),
           "fold_sizes_episodes": np.bincount(rf.episode_folds(n_episodes), minlength=rf.N_FOLDS).tolist(),
           "multinomial_checked": True}
    return scaler, model, rec, oof


def stage_train(args):
    import sklearn
    config = args.config
    p = reader_paths(config, args.smoke)
    start(p, args.smoke)
    t0 = time.time()
    parts = CONFIGS[config]
    feats, H, head_recs, bank_recs = bank_features(config, args.smoke)
    halves, rec_h, npz = [], {}, {}
    for j in (0, 1):
        d = feats[j]
        cnt = np.bincount(d["y"], minlength=len(parts))
        if (cnt != cnt[0]).any():
            raise AssertionError(f"half {j}: classes are not balanced ({cnt})")
        scaler, model, rec, oof = fit_half_reader(d["X"], d["y"], d["episode"], d["n_episodes"], len(parts))
        halves.append({"scaler": scaler, "model": model, "C": rec["chosen_C"]})
        rec_h[str(j)] = rec
        npz.update({f"half{j}__X": d["X"], f"half{j}__y": d["y"], f"half{j}__episode": d["episode"],
                    f"half{j}__oof_proba": oof, f"half{j}__fold": rf.episode_folds(d["n_episodes"])})
        C.log(f"{config} half {j}: chosen C {rec['chosen_C']} [{time.time() - t0:.0f}s]")
    rec = header("train", args.smoke, [heads_paths(h, args.smoke)["npz"] for h in parts]
                 + [bank_paths(config, j, args.smoke)["npz"] for j in (0, 1)] + [halves_paths(args.smoke)["npz"]])
    rec.update(config=config, groupings=list(parts), feature_names=rf.feature_names(parts),
               classes={str(i): h for i, h in enumerate(parts)}, halves_npz_sha256=H["sha256"],
               heads_npz_sha256={h: head_recs[h]["provenance"]["npz_sha256"] for h in parts},
               banks_npz_sha256={str(j): bank_recs[j]["provenance"]["npz_sha256"] for j in (0, 1)},
               model_spec={"scaler": "StandardScaler fitted once per half on all of that half's bank features (both "
                                     "conditions)",
                           "reader": f"LogisticRegression(solver='lbfgs', max_iter={READER_MAX_ITER}, C from grid, "
                                     "no class weights; multinomial loss)",
                           "C_grid": list(rf.C_GRID), "cv": "KFold(5, shuffle=True, random_state=0) over episode "
                           "indices; both conditions of an episode share a fold; criterion mean of log_loss over the "
                           "5 held-out folds; ties to the smaller C; refit on the whole half's bank"},
               halves=rec_h, sklearn_version=sklearn.__version__, numpy_version=np.__version__,
               runtime_s=round(time.time() - t0, 1))
    pkl = {"config": config, "groupings": list(parts), "feature_names": rf.feature_names(parts), "halves": halves,
           "sklearn_version": sklearn.__version__, "rule_sha256": C.RULE_SHA, "smoke": bool(args.smoke)}
    finish(p, rec, text_train, npz, pkl)


def text_train(r):
    L = [f"R-b half-readers, configuration {r['config']} ({', '.join(r['groupings'])})"
         f"{' [SMOKE]' if r['smoke'] else ''}; {len(r['feature_names'])} features; multinomial logistic regression, "
         "5-fold CV over episodes, mean held-out log loss"]
    for j, h in r["halves"].items():
        L.append(f"  half {j}: {h['n_episodes']} episodes / {h['n_examples']} examples; chosen C {h['chosen_C']}; "
                 f"out-of-fold accuracy {h['oof_accuracy_at_chosen_C']:.2f}% (chance {100 / len(r['groupings']):.1f}); "
                 f"refit warnings {h['refit_convergence_warnings']}")
        for row in h["cv_table"]:
            L.append(f"      C {row['C']:>6g}: log loss {row['mean_log_loss']:.5f}; oof acc {row['oof_accuracy']:.2f}; "
                     f"warnings {sum(row['convergence_warnings'])}; {row['fit_s']:.0f}s")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


def load_readers(config, smoke):
    """(pickle dict, json record, npz) of the trained half-readers (SHA-256 of pkl and npz checked)."""
    import sklearn
    rec, z = load_json_npz(reader_paths(config, smoke), smoke, f"rb_build.py train --config {config}")
    with open(reader_paths(config, smoke)["pkl"], "rb") as f:
        pk = pickle.load(f)
    if pk["config"] != config or tuple(pk["groupings"]) != CONFIGS[config] or pk["rule_sha256"] != C.RULE_SHA:
        raise SystemExit(f"rb_reader_{config}.pkl: other configuration or rule")
    if pk["sklearn_version"] != sklearn.__version__:
        raise SystemExit(f"rb_reader_{config}.pkl was written with scikit-learn {pk['sklearn_version']}, "
                         f"running {sklearn.__version__}")
    return pk, rec, z


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("halves")
    s.add_argument("--smoke", action="store_true")
    s = sub.add_parser("heads")
    s.add_argument("--grouping", required=True, choices=GROUPINGS)
    s.add_argument("--smoke", action="store_true")
    s = sub.add_parser("bank")
    s.add_argument("--config", required=True, choices=tuple(CONFIGS))
    s.add_argument("--half", type=int, choices=(0, 1), default=None, help="build one half only (default both)")
    s.add_argument("--smoke", action="store_true")
    s = sub.add_parser("train")
    s.add_argument("--config", required=True, choices=tuple(CONFIGS))
    s.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    C.assert_rule()
    {"halves": stage_halves, "heads": stage_heads, "bank": stage_bank, "train": stage_train}[args.stage](args)


if __name__ == "__main__":
    main()
