"""A0+X readers (DECISION_RULE L11; R1rule §4.2 items 1–6), CPU only.

Banks and posteriors use local scorer-train positions and the frozen round-1 painting
halves. X is named csd only in the bank partition dictionary; feature/class order is
(affect, image, caption, X). Reused stages and writers are never called. bank_features
exposes the exact intermediate arrays for G3; reader_X.npz also stores them and OOF
probabilities in round 1's layout. R0 is A1, with csd renamed R0 in feature metadata.

--smoke uses 256 episodes per block; real banks use 16,384. --jobs permits at most
three parallel reader processes, each with OMP_NUM_THREADS=8 and MKL_NUM_THREADS=8.
The controller must budget 24 threads for --jobs 3. No evaluation seed is scored.
"""
import r7_common as C

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

BASE = ("affect", "image", "caption")
GROUPINGS = tuple(f"{p}{k}" for p in ("V", "Z") for k in range(1, 7)) + ("R0", "N")
BANK_N, SMOKE_BANK_N = 16_384, 256
BANK_SEEDS, MIN_PAINTINGS = (11_700, 11_800), 30
EPISODE_FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")


def _root(root, smoke):
    return Path(root) if root is not None else (C.SMOKE_CHAIN if smoke else C.RESULTS)


def _record(path, smoke):
    rec = json.loads(path.read_text())
    assert rec["rule_sha256"] == C.RULE_SHA256, f"{path}: rule differs"
    assert rec["smoke"] == bool(smoke), f"{path}: smoke differs"
    assert isinstance(rec["dropped"], bool), f"{path}: missing drop status"
    return rec


def load_crossfit(name, labels, halves, root):
    """Ticket-05 file boundary: class columns, rows and prediction halves must agree."""
    path = Path(root) / "heads" / f"xheads_{name}.npz"
    with np.load(path, allow_pickle=False) as z:
        keys = ("img", "txt", "filled_by_half", "classes_half0", "classes_half1")
        assert set(keys) <= set(z.files), f"{path}: missing keys"
        post = {k: z[k] for k in keys}
    classes = np.unique(labels)
    assert np.array_equal(classes, np.arange(len(classes))), f"{name}: noncompact labels"
    n = len(halves["scorer_train"])
    filled = post["filled_by_half"]
    assert filled.dtype == np.int8 and filled.shape == (n,), f"{path}: prediction half shape/dtype"
    assert np.array_equal(filled, 1 - halves["half_of_local_row"]), f"{path}: wrong cross-fit half"
    for m in ("img", "txt"):
        p = post[m]
        assert p.dtype == np.float32 and p.shape == (n, len(classes)), f"{path}: {m} shape/dtype"
        assert np.isfinite(p).all() and (p >= 0).all(), f"{path}: invalid {m} posterior"
        assert np.allclose(p.sum(1), 1, rtol=0, atol=1e-5), f"{path}: {m} posterior sums"
    for j in (0, 1):
        assert post[f"classes_half{j}"].dtype == np.int64, f"{path}: class dtype"
        assert np.array_equal(post[f"classes_half{j}"], classes), f"{path}: missing cross-fit class"
    return {m: post[m] for m in ("img", "txt")}


def build_banks(name, labels, halves, bank_n=BANK_N):
    """Return banks in local row space; RuntimeError means an unfillable L11 block."""
    rb, rf = C.load("rb_build"), C.load("rb_features")
    builder = C.load("src.train.pseudo_partitions")
    parts = BASE + (name,)
    assert set(labels) == set(parts), "bank: grouping keys differ"
    st = halves["scorer_train"]
    paint = halves["painting_of_local_row"]
    membership = halves["half_of_local_row"]
    assert st.dtype == np.int64 and st.ndim == 1 and (np.diff(st) > 0).all(), "bank: scorer-train rows"
    assert paint.shape == membership.shape == st.shape, "bank: painting/half row counts"
    assert membership.dtype == np.int8 and np.isin(membership, (0, 1)).all(), "bank: half membership"
    for h in parts:
        assert labels[h].dtype == np.int64 and labels[h].shape == st.shape, f"bank: {h} labels"
        assert (labels[h] >= 0).all(), f"bank: {h} negative labels"
    partitions = {h: labels[h] for h in BASE}
    partitions["csd"] = labels[name]
    blocks = rf.bank_blocks(partitions)
    assert blocks == rb.BLOCKS["A1"], "bank: blocks differ from round-1 A1"
    banks = {}
    for j in (0, 1):
        rows = halves[f"local_rows_half{j}"]
        assert rows.dtype == np.int64 and np.array_equal(rows, np.flatnonzero(membership == j)), "bank: half rows"
        assert np.array_equal(st[rows], halves[f"global_rows_half{j}"]), "bank: local/global alignment"
        ep = builder.build_episode_bank(partitions, paint, rows, n_per_pair=bank_n,
                                        seed=BANK_SEEDS[j], min_paintings=MIN_PAINTINGS)
        validate_bank(ep, j, halves, bank_n)
        banks[j] = ep
    return banks, blocks


def validate_bank(ep, half, halves, bank_n):
    n = 6 * bank_n
    for f in EPISODE_FIELDS:
        a = getattr(ep, f)
        shape = (n,) if f == "anchor" else (n, 13 if f == "candidates" else 4)
        assert a.dtype == np.int64 and a.shape == shape, f"bank half {half}: {f} shape/dtype"
    rows = ep.rows()
    assert rows.min() >= 0 and rows.max() < len(halves["scorer_train"]), "bank: row out of bounds"
    assert (halves["half_of_local_row"][rows] == half).all(), "bank: row outside half"


def bank_features(name, post, banks, blocks, bank_n):
    """G3 intermediates: F[a/b], X, y, episode and n_episodes, unchanged R1 arithmetic."""
    rf = C.load("rb_features")
    parts = BASE + (name,)
    assert blocks == C.load("rb_build").BLOCKS["A1"], "features: block order"
    renamed = [(name if a == "csd" else a, name if b == "csd" else b) for a, b in blocks]
    out = {}
    for j in (0, 1):
        F = rf.both_conditions(post, parts, banks[j])
        ya, yb = rf.bank_labels(renamed, bank_n, parts)
        X, y, epi = rf.stack_conditions(F["a"], F["b"], ya, yb)
        n = 6 * bank_n
        assert X.dtype == np.float64 and X.shape == (2 * n, 24) and np.isfinite(X).all(), "features: invalid X"
        assert y.dtype == epi.dtype == np.int64 and y.shape == epi.shape == (2 * n,), "features: labels/index"
        assert np.array_equal(np.bincount(y, minlength=4), np.full(4, 3 * bank_n)), "features: unbalanced classes"
        out[j] = {"F": F, "X": X, "y": y, "episode": epi, "n_episodes": n}
    return out


def build_reader(name, labels, halves, root=None, smoke=False, bank_n=None, post=None):
    """Build one reader; return its JSON record. post permits synthetic chain-smoke inputs."""
    import sklearn
    C.assert_rule()
    C.assert_threads()
    assert name in GROUPINGS, f"unsupported reader {name} (T has no reader)"
    root = _root(root, smoke)
    bank_n = (SMOKE_BANK_N if smoke else BANK_N) if bank_n is None else bank_n
    assert bank_n > 0 and (smoke or bank_n == BANK_N), "real bank size is fixed"
    paths = [root / "readers" / f"reader_{name}.{ext}" for ext in ("pkl", "json", "npz")]
    bank_paths = [root / "readers" / f"bank_{name}_half{j}.npz" for j in (0, 1)]
    for p in paths + bank_paths:
        assert not p.exists() or C.in_smoke(p), f"{p}: refusing to overwrite"
    rec = {"grouping": name, "config": "A1" if name == "R0" else f"A0+{name}",
           "groupings": list(BASE + (name,)), "smoke": bool(smoke), "bank_n": bank_n, "dropped": False}
    if post is None:
        heads = {h: _record(root / "heads" / f"head_{h}.json", smoke) for h in BASE + (name,)}
        for h, record in heads.items():
            assert record["grouping"] == h, f"{h}: head grouping differs"
            assert record["label_sha256"] == C.load("rb_build").sha_array(labels[h]), f"{h}: head labels differ"
        if any(r["dropped"] for r in heads.values()):
            rec.update(dropped=True, reason="missing standard/cross-fitted head groups (L10/L11)")
            return C.write_json_once(paths[1], rec)
        post = {h: load_crossfit(h, labels[h], halves, root) for h in BASE + (name,)}
        rec["heads_sha256"] = {h: C.sha_file(root / "heads" / f"xheads_{h}.npz") for h in post}
    try:
        banks, blocks = build_banks(name, labels, halves, bank_n)
    except RuntimeError as e:
        rec.update(dropped=True, reason=f"unfillable bank block (L11): {e}")
        return C.write_json_once(paths[1], rec)
    feats = bank_features(name, post, banks, blocks, bank_n)
    rb, rf = C.load("rb_build"), C.load("rb_features")
    readers, records, arrays = [], {}, {}
    for j in (0, 1):
        d = feats[j]
        scaler, model, record, oof = rb.fit_half_reader(d["X"], d["y"], d["episode"], d["n_episodes"], 4)
        readers.append({"scaler": scaler, "model": model, "C": record["chosen_C"]})
        records[str(j)] = record
        arrays.update({f"half{j}__{k}": d[k] for k in ("X", "y", "episode")})
        arrays.update({f"half{j}__oof_proba": oof, f"half{j}__fold": rf.episode_folds(d["n_episodes"])})
        C.write_npz_once(bank_paths[j], **{f: getattr(banks[j], f) for f in EPISODE_FIELDS},
                         block_pairs=np.array([f"{a}__{b}" for a, b in blocks]),
                         block_size=np.array(bank_n), seed=np.array(BANK_SEEDS[j]), half=np.array(j))
    names = rf.feature_names(BASE + (name,))
    pk = {"config": rec["config"], "groupings": rec["groupings"], "feature_names": names,
          "halves": readers, "sklearn_version": sklearn.__version__, "rule_sha256": C.RULE_SHA256, "smoke": bool(smoke)}
    C.write_pickle_once(paths[0], pk)
    C.write_npz_once(paths[2], **arrays)
    rec.update(feature_names=names, halves=records, sklearn_version=sklearn.__version__, numpy_version=np.__version__,
               scorer_train_sha256=rb.sha_array(halves["scorer_train"]),
               paintings_sha256=rb.sha_array(halves["painting_of_local_row"]),
               label_sha256={h: rb.sha_array(labels[h]) for h in BASE + (name,)},
               banks_sha256={str(j): C.sha_file(p) for j, p in enumerate(bank_paths)},
               reader_sha256=C.sha_file(paths[0]), arrays_sha256=C.sha_file(paths[2]))
    return C.write_json_once(paths[1], rec)


def _run_one(task):
    name, root, smoke, bank_n = task
    # Import before the reused loader changes sys.path.
    import r7_heads as H
    C.assert_rule()
    C.assert_threads()
    root = _root(root, smoke)
    rb = C.load("rb_build")
    _, _, st, groups = rb.splits()
    halves = H.load_halves(st, groups)
    labels = H.load_groupings(BASE + (name,), st, halves["g_st"], root)
    return build_reader(name, labels, halves, root, smoke, bank_n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grouping", nargs="+", required=True, choices=GROUPINGS)
    p.add_argument("--root", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--bank-n", type=int, help="smoke size override only")
    p.add_argument("--jobs", type=int, choices=(1, 2, 3), default=1)
    args = p.parse_args()
    assert len(set(args.grouping)) == len(args.grouping), "duplicate readers"
    tasks = [(name, args.root, args.smoke, args.bank_n) for name in args.grouping]
    if args.jobs == 1:
        records = [_run_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            records = list(pool.map(_run_one, tasks))
    for rec in records:
        print(f"{rec['grouping']}: {'dropped' if rec['dropped'] else 'reader written'}")


if __name__ == "__main__":
    main()
