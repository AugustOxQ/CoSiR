"""Round 7 standard and cross-fitted CLIP heads (DECISION_RULE L10, L11; R1rule §4.2 item 2).

Labels and cross-fitted posteriors use local scorer-train positions; fits use global rows. Standard
posteriors use global rows, finite only on selection. No episode or evaluation label is loaded.
Full standard fits reuse r6_heads; cross-fits reuse rb_build.fit_head_pair (never its stages).
build_heads writes heads/{post,head,xheads}_X and JSON records through r7_common. A missing
draw class writes only head_X.json with dropped=True, before fitting anything. T has no cross-fit
or reader. Smoke uses 5,000 draw / 1,000 check rows and never asserts stored full-head equality.
Existing stage-0 files are validated and preserved; only missing files are built.
"""
import r7_common as C

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

MODALITIES = ("img", "txt")
HEAD_ROWS, CHECK_ROWS = 60_000, 10_000
SMOKE_HEAD_ROWS, SMOKE_CHECK_ROWS = 5_000, 1_000
BASE_HEADS = ("affect", "image", "caption", "csd")


def _rows(rows, n, what):
    rows = np.asarray(rows)
    assert rows.dtype == np.int64 and rows.ndim == 1 and len(rows), f"{what}: expected int64 rows"
    assert rows[0] >= 0 and rows[-1] < n and (np.diff(rows) > 0).all(), f"{what}: invalid row order"
    return rows


def _labels(local, st):
    local = np.asarray(local)
    assert local.dtype == np.int64 and local.shape == st.shape, "labels: expected local int64 array"
    classes = np.unique(local)
    assert len(classes) >= 2 and np.array_equal(classes, np.arange(len(classes))), "labels: noncompact classes"
    return local, len(classes)


def _cross_rows(h, st, n):
    assert np.array_equal(h["scorer_train"], st), "cross-fit: scorer-train differs"
    membership = h["half_of_local_row"]
    assert membership.dtype == np.int8 and membership.shape == st.shape, "cross-fit: membership shape/dtype"
    assert np.isin(membership, (0, 1)).all(), "cross-fit: invalid half membership"
    for k in (0, 1):
        loc = _rows(h[f"local_rows_half{k}"], len(st), f"half {k} local")
        glob = _rows(h[f"global_rows_half{k}"], n, f"half {k} global")
        assert np.array_equal(loc, np.flatnonzero(membership == k)), f"half {k}: local membership differs"
        assert np.array_equal(st[loc], glob), f"half {k}: local/global mismatch"


def load_halves(st, groups):
    """Read the frozen round-1 halves, checking painting and local/global alignment."""
    path = C.R1D / "results/rb_halves.npz"
    with np.load(path, allow_pickle=False) as z:
        keys = ("scorer_train", "g_st", "painting_of_local_row", "paintings_sorted", "half_of_local_row",
                "paintings_half0", "paintings_half1", "local_rows_half0", "local_rows_half1",
                "global_rows_half0", "global_rows_half1")
        assert set(keys) <= set(z.files), "halves: missing keys"
        h = {k: z[k] for k in keys}
    assert len(st) == 183_694 and np.array_equal(h["scorer_train"], st), "halves: scorer-train differs"
    paintings, g_st = np.unique(groups[st], return_inverse=True)
    assert len(paintings) == 36_518 and np.array_equal(h["paintings_sorted"], paintings), "halves: paintings differ"
    assert np.array_equal(h["g_st"], g_st), "halves: painting codes differ"
    assert np.array_equal(h["painting_of_local_row"], groups[st]), "halves: row paintings differ"
    perm = np.random.default_rng(0).permutation(paintings)
    assert h["half_of_local_row"].dtype == np.int8 and h["half_of_local_row"].shape == st.shape, "halves: shape/dtype"
    for k in (0, 1):
        expected = perm[k * 18_259:(k + 1) * 18_259]
        assert np.array_equal(h[f"paintings_half{k}"], expected), f"half {k}: painting draw differs"
        local = np.flatnonzero(np.isin(groups[st], expected)).astype(np.int64)
        assert np.array_equal(h[f"local_rows_half{k}"], local), f"half {k}: local rows differ"
        assert np.array_equal(h[f"global_rows_half{k}"], st[local]), f"half {k}: global rows differ"
        assert (h["half_of_local_row"][local] == k).all(), f"half {k}: membership differs"
    _cross_rows(h, st, len(groups))
    return h


def load_groupings(names, st, g_st, root=None):
    """Read R1's exact sources or round-7 grouping files, without rb_build.load_groupings."""
    rb = C.load("rb_build")
    root = Path(root) if root is not None else C.RESULTS
    assert st.dtype == np.int64 and st.shape == (183_694,) and (np.diff(st) > 0).all(), "groupings: scorer-train"
    assert g_st.dtype == np.int64 and g_st.shape == st.shape, "groupings: painting codes"
    out = {}
    for name in names:
        if name in BASE_HEADS:
            path, sha, key = rb.SOURCES[name]
            assert C.sha_file(path) == sha, f"{name}: source SHA differs"
        else:
            path = root / "groupings" / ("r7_grouping_N.npz" if name == "N" else "r7_groupings.npz")
            key = name
        with np.load(path, allow_pickle=False) as z:
            assert key in z.files, f"{path}: missing {key}"
            if name in ("image", "caption"):
                assert "local_groups" in z.files and np.array_equal(z["local_groups"], g_st), "E2: row alignment"
            elif name == "affect":
                e2path, sha, _ = rb.SOURCES["image"]
                assert C.sha_file(e2path) == sha, "affect: E2 SHA differs"
                with np.load(e2path, allow_pickle=False) as e2:
                    assert {"local_groups", "affect"} <= set(e2.files), "E2: missing keys"
                    assert "partition_R0" in z.files, "affect: missing partition_R0"
                    assert np.array_equal(e2["local_groups"], g_st), "affect: row alignment"
                    assert np.array_equal(z["partition_R0"], e2["affect"]), "affect: E2 partition differs"
            else:
                # N's shared interface contains only N; its alignment comes from the main grouping file.
                alignment = root / "groupings/r7_groupings.npz" if name == "N" else path
                with np.load(alignment, allow_pickle=False) as a:
                    assert {"scorer_train", "g_st"} <= set(a.files), f"{alignment}: missing alignment keys"
                    assert np.array_equal(a["scorer_train"], st) and np.array_equal(a["g_st"], g_st), "grouping: row alignment"
                if name == "N":
                    for k, expected in (("scorer_train", st), ("g_st", g_st)):
                        if k in z.files:
                            assert np.array_equal(z[k], expected), f"N: {k} differs"
            local, n_cls = _labels(z[key], st)
        if name in BASE_HEADS:
            assert n_cls == rb.N_GROUPS[name], f"{name}: class count differs"
        if name not in ("affect", "caption"):
            first = np.unique(g_st, return_index=True)[1]
            assert np.array_equal(local[first][g_st], local), f"{name}: not constant within painting"
        out[name] = local
    return out


def _draw_record(local, st, pool, head_rows):
    draw = np.random.default_rng(0).choice(pool, head_rows, replace=False)
    classes = np.unique(local[np.searchsorted(st, draw)])
    missing = np.setdiff1d(np.unique(local), classes)
    return {"missing_groups": missing.tolist(), "all_groups_in_draw": not len(missing)}


def fit_standard(data, local, st, selection, head_rows=HEAD_ROWS, check_rows=CHECK_ROWS):
    """Return (classifiers, global posteriors, provenance); one selection prediction per modality."""
    r6 = C.load("r6_heads")
    lab = np.full(len(data.img_features), -1, dtype=np.int64)
    lab[st] = local
    if check_rows == CHECK_ROWS:
        clfs, prov = r6.fit_one_head_r6(data, lab, st, n_rows=head_rows)
    else:
        # Same recipe with a smaller smoke check set, without mutating reused-module constants.
        from sklearn.linear_model import LogisticRegression
        draw = np.random.default_rng(0).choice(st, head_rows, replace=False)
        rest = np.setdiff1d(st, draw)
        check = np.random.default_rng(1).choice(rest, check_rows, replace=False)
        F = r6._features(data)
        clfs = {m: LogisticRegression(C=1.0, max_iter=300).fit(r6.rc.unit(F[m][draw]), lab[draw]) for m in MODALITIES}
        counts = np.bincount(lab[check])
        prov = {"n_classes": len(clfs["img"].classes_), "draw_rows_sha256": r6.rg.sha_array(np.sort(draw)),
                "heldout_accuracy": {m: 100 * float(clfs[m].score(r6.rc.unit(F[m][check]), lab[check])) for m in MODALITIES},
                "check_majority_share": 100 * float(counts.max() / counts.sum()),
                "uniform": 100 / len(clfs["img"].classes_)}
    for m in MODALITIES:
        assert np.array_equal(clfs[m].classes_, np.unique(local)), f"standard {m}: classes differ"
    post = r6.predict({"X": clfs}, data, selection)["X"]
    return clfs, post, prov


def fit_crossfit(data, local, st, halves, head_rows=HEAD_ROWS, check_rows=CHECK_ROWS):
    """Return (local cross-fitted arrays, per-half provenance), preserving R1 class columns."""
    rb = C.load("rb_build")
    _, n_cls = _labels(local, st)
    F = {"img": data.img_features, "txt": data.txt_features}
    lab = np.full(len(data.img_features), -1, dtype=np.int64)
    lab[st] = local
    _cross_rows(halves, st, len(lab))
    cf = {m: np.full((len(st), n_cls), np.nan, dtype=np.float32) for m in MODALITIES}
    filled = np.full(len(st), -1, dtype=np.int8)
    records = {}
    for k in (0, 1):
        own, other = halves[f"global_rows_half{k}"], halves[f"global_rows_half{1-k}"]
        loc = halves[f"local_rows_half{1-k}"]
        assert np.array_equal(st[loc], other), f"half {k}: local/global mismatch"
        post, prov, classes = rb.fit_head_pair(F, lab, own, head_rows, other, other_rows=other, check_rows=check_rows)
        assert np.array_equal(classes, np.arange(n_cls)), f"half {k}: missing class"
        for m in MODALITIES:
            assert post[m].dtype == np.float32 and post[m].shape == (len(loc), n_cls), f"half {k}: posterior shape"
            cf[m][loc] = post[m]
        cf[f"classes_half{k}"] = classes.astype(np.int64)
        filled[loc] = k
        records[str(k)] = prov
    assert np.array_equal(filled, 1 - halves["half_of_local_row"]), "cross-fit: wrong prediction half"
    assert all(np.isfinite(cf[m]).all() for m in MODALITIES), "cross-fit: unfilled posteriors"
    cf["filled_by_half"] = filled
    return cf, records


def _assert_csd(post, selection):
    r6 = C.load("r6_heads")
    path = C.TEST / r6.CSD_HEADS_REL
    r6.R.assert_input(r6.CSD_HEADS_REL)
    with np.load(path, allow_pickle=False) as z:
        assert {"selection", "style_csd__img", "style_csd__txt"} <= set(z.files), "stored csd: missing keys"
        assert np.array_equal(z["selection"], selection), "stored csd: selection differs"
        for m in MODALITIES:
            assert r6.compare_bits(post[m][selection], z[f"style_csd__{m}"])["equal"], f"csd/{m}: differs from step1"


def check_stored_head(name, clfs, post, data, selection):
    """Resume of a build: the stored head_X.pkl, run on the selection rows, must give the stored post_X.npz (no record
    holds a file SHA-256, and the records are written last, so the files are checked against each other)."""
    predicted = C.load("r6_heads").predict({"X": clfs}, data, selection)["X"]
    for m in MODALITIES:
        assert np.allclose(predicted[m][selection], post[m][selection], rtol=0, atol=1e-6), \
            f"stored head {name}/{m}: predictions differ from the stored post"


def build_heads(name, data, sp, local, halves=None, root=None, smoke=False,
                head_rows=None, check_rows=None):
    """Build one grouping. Return its head_X.json record, including any L10/L11 drop reason."""
    C.assert_rule()
    C.assert_threads()
    root = Path(root) if root is not None else (C.SMOKE_CHAIN if smoke else C.RESULTS)
    assert name and all(c.isalnum() or c == "_" for c in name), "invalid head name"
    head_rows = (SMOKE_HEAD_ROWS if smoke else HEAD_ROWS) if head_rows is None else head_rows
    check_rows = (SMOKE_CHECK_ROWS if smoke else CHECK_ROWS) if check_rows is None else check_rows
    assert smoke or (head_rows, check_rows) == (HEAD_ROWS, CHECK_ROWS), "real head sizes are fixed"
    assert head_rows > 0 and check_rows > 0, "head sizes must be positive"
    r6 = C.load("r6_heads")
    F = r6._features(data)
    n = len(F["img"])
    st, selection = _rows(sp.scorer_train, n, "scorer_train"), _rows(sp.selection, n, "selection")
    assert not np.intersect1d(st, selection).size, "selection overlaps scorer-train"
    local, n_cls = _labels(local, st)
    cross = name != "T"
    if cross:
        halves = load_halves(st, np.asarray(sp.groups)) if halves is None else halves
        _cross_rows(halves, st, n)
    if name in ("R0", "csd"):
        assert np.array_equal(local, r6.csd_labels(st, n)[st]), "R0/csd must use frozen csd labels"
    paths = {k: root / "heads" / f"{k}_{name}.{ext}" for k, ext in
             (("head", "pkl"), ("post", "npz"), ("xheads", "npz"))}
    records = {k: p.with_suffix(".json") for k, p in paths.items()}
    existing = {}
    for kind, path in records.items():
        if path.exists():
            record = json.loads(path.read_text())
            assert record.get("rule_sha256") == C.rule_sha(), f"{path}: rule differs"
            assert record.get("grouping") == name and record.get("smoke") == bool(smoke), f"{path}: role/grouping differs"
            for key, expected in (("n_groups", n_cls), ("label_sha256", r6.rg.sha_array(local)),
                                  ("head_rows", head_rows), ("check_rows", check_rows)):
                if key in record:
                    assert record[key] == expected, f"{path}: {key} differs"
            existing[kind] = record
    pools = {"standard": st}
    if cross:
        pools.update({f"half{k}": halves[f"global_rows_half{k}"] for k in (0, 1)})
    draws = {}
    for k, pool in pools.items():
        _rows(pool, n, k)
        assert np.isin(pool, st).all(), f"{k}: fit pool outside scorer-train"
        assert len(pool) >= head_rows + check_rows, f"{k}: insufficient draw/check rows"
        draws[k] = _draw_record(local, st, pool, head_rows)
    rec = {"grouping": name, "smoke": bool(smoke), "head_rows": head_rows, "check_rows": check_rows,
           "n_groups": n_cls, "label_sha256": r6.rg.sha_array(local), "draws": draws,
           "dropped": any(not d["all_groups_in_draw"] for d in draws.values())}
    if "head" in existing:
        assert existing["head"]["dropped"] == rec["dropped"], "head drop status changed"
    if rec["dropped"]:
        rec["reason"] = "missing groups in standard or cross-fitted draw (L10/L11)"
        if "head" not in existing:
            C.write_json_once(records["head"], rec)
        return existing.get("head", rec)
    clfs, post = None, None
    prov = existing.get("head", {}).get("standard", existing.get("post", {}))
    if paths["head"].exists():
        with paths["head"].open("rb") as f:
            clfs = pickle.load(f)
        assert set(clfs) == set(MODALITIES), "stored head: modality keys differ"
        for m in MODALITIES:
            assert np.array_equal(clfs[m].classes_, np.arange(n_cls)), f"stored head {m}: classes differ"
            assert clfs[m].n_features_in_ == F[m].shape[1], f"stored head {m}: feature count differs"
    if paths["post"].exists():
        with np.load(paths["post"], allow_pickle=False) as z:
            assert {"img", "txt", "rows"} <= set(z.files), "stored post: keys differ"
            assert np.array_equal(z["rows"], selection), "stored post: selection differs"
            post = {m: z[m] for m in MODALITIES}
        outside = np.ones(n, dtype=bool)
        outside[selection] = False
        for m in MODALITIES:
            assert post[m].dtype == np.float32 and post[m].shape == (n, n_cls), f"stored post {m}: shape/dtype"
            assert np.isfinite(post[m][selection]).all() and np.isnan(post[m][outside]).all(), f"stored post {m}: selection mask"
            assert np.allclose(post[m][selection].sum(1), 1, rtol=0, atol=1e-5), f"stored post {m}: sums"
    if clfs is not None and post is not None:
        check_stored_head(name, clfs, post, data, selection)
    if clfs is None:
        clfs, predicted, prov = fit_standard(data, local, st, selection, head_rows, check_rows)
        if post is None:
            post = predicted
        else:
            assert all(np.array_equal(post[m], predicted[m], equal_nan=True) for m in MODALITIES), "stored post differs from fitted head"
        C.write_pickle_once(paths["head"], clfs)
    if post is None:
        post = r6.predict({"X": clfs}, data, selection)["X"]
    if name in ("R0", "csd") and not smoke:
        _assert_csd(post, selection)
        rec["csd_equals_step1"] = True
    rec["standard"] = {k: v for k, v in prov.items() if k not in ("rule_sha256", "written", "grouping", "smoke")}
    if not paths["post"].exists():
        C.write_npz_once(paths["post"], **post, rows=selection)
    if "post" not in existing:
        C.write_json_once(records["post"], {"grouping": name, "smoke": bool(smoke), **rec["standard"]})
    if cross:
        if paths["xheads"].exists():
            # Import before any further reused-module load; this is the reader's
            # full class/row/half validation for the stage-0 cross-fitted file.
            import r7_reader
            r7_reader.load_crossfit(name, local, halves, root)
            rec["halves"] = existing.get("xheads", {}).get("halves", existing.get("head", {}).get("halves", {}))
        else:
            cf, rec["halves"] = fit_crossfit(data, local, st, halves, head_rows, check_rows)
            C.write_npz_once(paths["xheads"], **cf)
        if "xheads" not in existing:
            C.write_json_once(records["xheads"], {"grouping": name, "smoke": bool(smoke), "halves": rec["halves"]})
    if "head" not in existing:
        C.write_json_once(records["head"], rec)
    return existing.get("head", rec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grouping", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--root", type=Path, help="round-7 grouping/output root; default RESULTS or smoke/chain")
    p.add_argument("--head-rows", type=int)
    p.add_argument("--check-rows", type=int)
    args = p.parse_args()
    C.assert_rule()
    C.assert_threads()
    rb = C.load("rb_build")
    data, sp, st, groups = rb.splits()
    h = load_halves(st, groups)
    root = args.root or (C.SMOKE_CHAIN if args.smoke else C.RESULTS)
    local = load_groupings([args.grouping], st, h["g_st"], root)[args.grouping]
    rec = build_heads(args.grouping, data, sp, local, h, root, args.smoke, args.head_rows, args.check_rows)
    print(f"{args.grouping}: {'dropped' if rec['dropped'] else 'heads written'} (smoke={args.smoke})")


if __name__ == "__main__":
    main()
