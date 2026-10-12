"""A0+X readers (DECISION_RULE L11; R1rule §4.2 items 1–6), CPU only.

Banks and posteriors use local scorer-train positions and the frozen round-1 painting
halves. X is named csd only in the bank partition dictionary; feature/class order is
(affect, image, caption, X). Reused stages and writers are never called. bank_features
exposes the exact intermediate arrays for G3; reader_X.npz also stores them and OOF
probabilities in round 1's layout. R0 is A1, with csd renamed R0 in feature metadata.

--smoke uses 256 episodes per block; real banks use 16,384. --jobs permits at most
three parallel reader processes, each with OMP_NUM_THREADS=8 and MKL_NUM_THREADS=8.
The controller must budget 24 threads for --jobs 3. No evaluation seed is scored.
Preserved reader/bank files are checked before completing missing outputs.
"""
import r8_scoring_io as C

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import pickle
from pathlib import Path

import numpy as np

BASE = ("affect", "image", "caption")
GROUPINGS = C.GROUPINGS
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
        validate_bank(ep, j, halves, bank_n, partitions, blocks)
        banks[j] = ep
    return banks, blocks


def check_layout(arrays, partitions, blocks, bank_n, half):
    """L11, round 1's `stage_bank` check on every episode: block i holds episodes i*n to (i+1)*n - 1, and its pairs share
    the block's two groupings (pair images and texts agree within a group, the anchor's own group is excluded, and the
    first two candidates carry the anchor's groups)."""
    for i, (a, b) in enumerate(blocks):
        sl = slice(i * bank_n, (i + 1) * bank_n)
        la, lb = partitions[a], partitions[b]
        ok = bool((la[arrays["pairs_a_img"][sl]] == la[arrays["pairs_a_txt"][sl]]).all()
                  and (lb[arrays["pairs_b_img"][sl]] == lb[arrays["pairs_b_txt"][sl]]).all()
                  and (la[arrays["pairs_a_img"][sl]] != la[arrays["anchor"][sl]][:, None]).all()
                  and (la[arrays["candidates"][sl, 0]] == la[arrays["anchor"][sl]]).all()
                  and (lb[arrays["candidates"][sl, 1]] == lb[arrays["anchor"][sl]]).all())
        assert ok, f"bank half {half} block {i} ({a}, {b}): episodes do not follow the block layout"


def validate_bank(ep, half, halves, bank_n, partitions=None, blocks=None):
    n = 6 * bank_n
    arrays = ep if isinstance(ep, dict) else {f: getattr(ep, f) for f in EPISODE_FIELDS}
    for f in EPISODE_FIELDS:
        a = arrays[f]
        shape = (n,) if f == "anchor" else (n, 13 if f == "candidates" else 4)
        assert a.dtype == np.int64 and a.shape == shape, f"bank half {half}: {f} shape/dtype"
    rows = np.concatenate([arrays[f].ravel() for f in EPISODE_FIELDS])
    assert rows.min() >= 0 and rows.max() < len(halves["scorer_train"]), "bank: row out of bounds"
    assert (halves["half_of_local_row"][rows] == half).all(), "bank: row outside half"
    if partitions is not None:
        assert blocks is not None and len(blocks) == 6, "bank: blocks"
        check_layout(arrays, partitions, blocks, bank_n, half)


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


def validate_reader(name, labels, halves, root, smoke):
    """Validate preserved stage-0 reader files before prep skips them."""
    root = Path(root)
    path = root / "readers" / f"reader_{name}.json"
    rec = _record(path, smoke)
    assert rec["grouping"] == name, f"{path}: grouping differs"
    if rec["dropped"]:
        return rec
    bank_n = SMOKE_BANK_N if smoke else BANK_N
    assert rec["bank_n"] == bank_n, f"{path}: bank size differs"
    with path.with_suffix(".pkl").open("rb") as f:
        pk = pickle.load(f)
    import sklearn
    assert pk["sklearn_version"] == sklearn.__version__, f"{path}: sklearn differs"
    assert pk["rule_sha256"] == C.rule_sha() and pk["smoke"] == smoke, f"{path}: pickle role/rule differs"
    assert tuple(pk["groupings"]) == BASE + (name,), f"{path}: grouping order differs"
    assert pk["feature_names"] == C.load("rb_features").feature_names(BASE + (name,)), f"{path}: feature order differs"
    assert len(pk["halves"]) == 2, f"{path}: half count differs"
    partitions = {**{h: labels[h] for h in BASE}, "csd": labels[name]}  # as build_banks: X is named csd in the blocks
    blocks = C.load("rb_features").bank_blocks(partitions)
    assert blocks == C.load("rb_build").BLOCKS["A1"], f"{path}: blocks differ from round-1 A1"
    for h in (0, 1):
        half = pk["halves"][h]
        assert half["C"] == rec["halves"][str(h)]["chosen_C"], f"{path}: chosen C differs"
        assert np.array_equal(half["model"].classes_, np.arange(4)), f"{path}: classes differ"
        bank = root / "readers" / f"bank_{name}_half{h}.npz"
        with np.load(bank, allow_pickle=False) as z:
            assert set(EPISODE_FIELDS) <= set(z.files), f"{bank}: episode keys differ"
            validate_bank({k: z[k] for k in EPISODE_FIELDS}, h, halves, bank_n, partitions, blocks)
        if "banks_sha256" in rec:
            assert C.sha_file(bank) == rec["banks_sha256"][str(h)], f"{bank}: SHA differs"
    for key, target in (("reader_sha256", path.with_suffix(".pkl")), ("arrays_sha256", path.with_suffix(".npz"))):
        if key in rec:
            assert C.sha_file(target) == rec[key], f"{target}: SHA differs"
    if "label_sha256" in rec:
        for h in BASE + (name,):
            assert rec["label_sha256"][h] == C.load("rb_build").sha_array(labels[h]), f"{path}: {h} labels differ"
    return rec


def _save_arrays(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert set(arrays) <= set(z.files), f"{path}: missing keys"
            assert all(z[k].dtype == a.dtype and np.array_equal(z[k], a) for k, a in arrays.items()), f"{path}: existing arrays differ"
    else:
        C.write_npz_once(path, **arrays)


def build_reader(name, labels, halves, root=None, smoke=False, bank_n=None, post=None):
    """Build one reader; return its JSON record. post accepts preloaded cross-fitted posteriors."""
    import sklearn
    C.assert_rule()
    C.assert_threads()
    assert name in GROUPINGS, f"unsupported reader {name} (T has no reader)"
    root = _root(root, smoke)
    bank_n = (SMOKE_BANK_N if smoke else BANK_N) if bank_n is None else bank_n
    assert bank_n > 0 and (smoke or bank_n == BANK_N), "real bank size is fixed"
    paths = [root / "readers" / f"reader_{name}.{ext}" for ext in ("pkl", "json", "npz")]
    bank_paths = [root / "readers" / f"bank_{name}_half{j}.npz" for j in (0, 1)]

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
        _save_arrays(bank_paths[j], {**{f: getattr(banks[j], f) for f in EPISODE_FIELDS},
                         "block_pairs": np.array([f"{a}__{b}" for a, b in blocks]),
                         "block_size": np.array(bank_n), "seed": np.array(BANK_SEEDS[j]), "half": np.array(j)})
    names = rf.feature_names(BASE + (name,))
    pk = {"config": rec["config"], "groupings": rec["groupings"], "feature_names": names,
          "halves": readers, "sklearn_version": sklearn.__version__, "rule_sha256": C.RULE_SHA256, "smoke": bool(smoke)}
    if paths[0].exists():
        with paths[0].open("rb") as f:
            old = pickle.load(f)
        for key in ("groupings", "feature_names", "sklearn_version", "rule_sha256", "smoke"):
            assert old[key] == pk[key], f"{paths[0]}: {key} differs"
        assert len(old["halves"]) == 2, "stored reader: half count differs"
        for j in (0, 1):
            saved, fitted = old["halves"][j], pk["halves"][j]
            assert saved["C"] == fitted["C"], "stored reader: chosen C differs"
            X = feats[j]["X"]
            assert np.allclose(saved["model"].predict_proba(saved["scaler"].transform(X)),
                               fitted["model"].predict_proba(fitted["scaler"].transform(X)),
                               rtol=0, atol=1e-12), "stored reader: probabilities differ"
    else:
        C.write_pickle_once(paths[0], pk)
    _save_arrays(paths[2], arrays)
    rec.update(feature_names=names, halves=records, sklearn_version=sklearn.__version__, numpy_version=np.__version__,
               scorer_train_sha256=rb.sha_array(halves["scorer_train"]),
               paintings_sha256=rb.sha_array(halves["painting_of_local_row"]),
               label_sha256={h: rb.sha_array(labels[h]) for h in BASE + (name,)},
               banks_sha256={str(j): C.sha_file(p) for j, p in enumerate(bank_paths)},
               reader_sha256=C.sha_file(paths[0]), arrays_sha256=C.sha_file(paths[2]))
    if paths[1].exists():
        old = _record(paths[1], smoke)
        for key in ("grouping", "bank_n", "reader_sha256", "arrays_sha256", "banks_sha256", "label_sha256"):
            assert old[key] == rec[key], f"{paths[1]}: {key} differs"
        return old
    return C.write_json_once(paths[1], rec)
