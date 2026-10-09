"""Tests of r6_heads and run_r6_refit (ticket 03; rule §6 item 2, §5 item 5, §11).

Synthetic data has the real shapes and dtypes (308,723 rows, 512-d float32 image and caption features, scorer-train
183,694, selection 32,413 and held-sized 61,744 disjoint rows, 41 / 64 / 64 / 64 / 17 classes); only the head draw is
a small stand-in (3,000 rows for the 60,000). Real files are read only for labels on scorer-train rows (no feature, no
held row). Mutation tests load a copy of the module from tmp_path with one `# guard:<name>` statement replaced by
`pass` and show that the guard's scenario then goes through (never in place).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_heads.py
"""
import ast
import copy
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_heads as H  # noqa: E402
import run_r6_refit as RR  # noqa: E402

import numpy as np  # noqa: E402

N, D = R.N_ROWS, 512
N_SCORER, N_SEL, N_HELD = 183_694, 32_413, 61_744
N_FIT = 3_000
_COUNT = itertools.count()
MUTATED = set()          # guard names a mutation test below covers (checked against the markers)


# ---------------------------------------------------------------- helpers

def mutant(tmp_path, module, guard):
    """Import a copy of ``module`` (file name in HERE) with its `# guard:<guard>` statements replaced by `pass`."""
    src = (HERE / module).read_text()
    lines = src.splitlines(keepends=True)
    hits = [n for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.Expr, ast.Assign)) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
    assert hits, f"no statement of {module} carries # guard:{guard}"
    for n in hits:
        first = lines[n.lineno - 1]
        lines[n.lineno - 1] = first[:len(first) - len(first.lstrip())] + "pass\n"
        for i in range(n.lineno, n.end_lineno):
            lines[i] = "\n"
    path = tmp_path / f"{Path(module).stem}_mut{next(_COUNT)}.py"
    path.write_text("".join(lines))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.modules[path.stem] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(path.stem, None)
        sys.path[:] = saved
    MUTATED.add(guard)
    return mod


def bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype == np.float32 and np.array_equal(a.view(np.uint32),
                                                                                      b.view(np.uint32))


def _labels(rng, img, scorer_train, k):
    """Global labels (-1 outside scorer-train), learnable from the image features, every class present."""
    W = rng.standard_normal((D, k), dtype=np.float32)
    y = np.argmax(img[scorer_train] @ W + rng.standard_normal((len(scorer_train), k), dtype=np.float32), axis=1)
    lab = np.full(N, -1, dtype=np.int64)
    lab[scorer_train] = y
    return lab


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(20261125)
    img = rng.standard_normal((N, D), dtype=np.float32)
    txt = rng.standard_normal((N, D), dtype=np.float32)
    perm = rng.permutation(N).astype(np.int64)
    scorer_train = np.sort(perm[:N_SCORER])
    selection = np.sort(perm[N_SCORER:N_SCORER + N_SEL])
    held = np.sort(perm[N_SCORER + N_SEL:N_SCORER + N_SEL + N_HELD])
    labels = {h: _labels(rng, img, scorer_train, H.N_CLASSES[h]) for h in H.HEADS}
    data = SimpleNamespace(img_features=img, txt_features=txt)
    heads = H.fit_all(data, scorer_train, labels, N_FIT)
    return SimpleNamespace(data=data, scorer_train=scorer_train, selection=selection, held=held, labels=labels,
                           heads=heads)


@pytest.fixture(scope="module")
def post_sel(synth):
    return H.predict(synth.heads, synth.data, synth.selection)


# ---------------------------------------------------------------- constants

def test_constants_and_targets():
    assert H.HEADS == ("affect", "affect_km", "image", "caption", "csd")
    assert H.N_CLASSES == {"affect": 41, "affect_km": 64, "image": 64, "caption": 64, "csd": 17}
    assert H.HEAD_ROWS == 60_000 and H.n6.CHECK_ROWS == 10_000 and H.rc.PROBE_SEED == 0
    assert H.TARGETS[H.N6_REL]["affect__img"] == ("affect_km", "img")          # n6's affect is affect-km (hazard 9)
    assert tuple(H.TARGETS[H.CSD_HEADS_REL]) == ("style_csd__img", "style_csd__txt")
    assert not any(k.endswith("_src") for keys in H.TARGETS.values() for k in keys)
    assert len(H.ITEM_KEYS) == 8
    for rel in (*H.TARGETS, H.TOLD_REL, H.PARTITION_L_REL, H.E2_PARTITIONS_REL, H.CSD_GROUP_REL):
        assert rel in R.INPUT_SHA256, rel


def test_every_marked_guard_has_a_mutation_test():
    names = set()
    for module in ("r6_heads.py", "run_r6_refit.py"):
        names |= set(re.findall(r"# guard:([a-z_0-9]+)", (HERE / module).read_text()))
    tests = (HERE / "test_r6_heads.py").read_text()
    covered = set(re.findall(r'mutant\(tmp_path, "[a-z_0-9]+\.py", "([a-z_0-9]+)"\)', tests))
    assert names and names == covered, (names, covered)


# ---------------------------------------------------------------- the copies against the originals

def test_copies_reproduce_the_originals_bit_for_bit(synth, post_sel):
    d, st, sel = synth.data, synth.scorer_train, synth.selection
    in_sel = np.zeros(N, dtype=bool)
    in_sel[sel] = True
    ctx = SimpleNamespace(data=d, selection=sel, in_sel=in_sel, groups=np.zeros(N, dtype=np.int64))
    post_n6, prov_n6 = H.n6.fit_heads(ctx, {part: synth.labels[name] for name, part in H.E2_HEADS.items()}, st,
                                      N_FIT)
    post_aff, prov_aff = H.rto.fit_one_head(ctx, synth.labels["affect"], st, N_FIT)
    post_csd, prov_csd = H.rto.fit_one_head(ctx, synth.labels["csd"], st, N_FIT)
    orig = {"affect": post_aff, "csd": post_csd, **{name: post_n6[part] for name, part in H.E2_HEADS.items()}}
    for name in H.HEADS:
        for m in H.MODALITIES:
            assert bits_equal(post_sel[name][m], orig[name][m]), (name, m)       # whole array, NaN pattern too
            direct = synth.heads[name][m].predict_proba(H.rc.unit(getattr(d, f"{m}_features")[sel]))
            assert bits_equal(direct.astype(np.float32), orig[name][m][sel]), (name, m)
    assert H.rto.roundtrip(synth.heads["affect"]["prov"]) == H.rto.roundtrip(prov_aff)
    assert H.rto.roundtrip(synth.heads["csd"]["prov"]) == H.rto.roundtrip(prov_csd)
    for name, part in H.E2_HEADS.items():
        assert synth.heads[name]["prov"] == {"draw_rows_sha256": prov_n6["draw_rows_sha256"],
                                             "n_draw": prov_n6["n_draw"], **prov_n6[part]}
    assert prov_aff["draw_rows_sha256"] == prov_n6["draw_rows_sha256"] == prov_csd["draw_rows_sha256"]


# ---------------------------------------------------------------- predict

def test_predict_nan_outside_finite_inside(synth, post_sel):
    outside = np.ones(N, dtype=bool)
    outside[synth.selection] = False
    for name in H.HEADS:
        for m in H.MODALITIES:
            p = post_sel[name][m]
            assert p.shape == (N, H.N_CLASSES[name]) and p.dtype == np.float32
            assert np.isfinite(p[synth.selection]).all() and np.isnan(p[outside]).all()
            assert np.allclose(p[synth.selection].sum(axis=1), 1, atol=1e-5)


def _spied(heads):
    """Deep copies of the heads whose predict_proba records each call's input."""
    calls = []
    out = copy.deepcopy(heads)
    for name in out:
        for m in H.MODALITIES:
            clf = out[name][m]

            def spy(X, _clf=clf, _key=(name, m)):
                calls.append((_key, np.array(X, copy=True)))
                return type(_clf).predict_proba(_clf, X)
            clf.predict_proba = spy
    return out, calls


def test_one_call_per_head_and_modality_on_exactly_the_rows(synth, post_sel):
    heads, calls = _spied(synth.heads)
    for rows in (synth.selection, synth.held):              # two row sets: two separate predict calls
        calls.clear()
        post = H.predict(heads, synth.data, rows)
        assert [k for k, _ in calls] == [(h, m) for h in H.HEADS for m in H.MODALITIES]
        for (name, m), X in calls:
            want = H.rc.unit(getattr(synth.data, f"{m}_features")[rows])
            assert X.shape == (len(rows), D) and bits_equal(X, want), (name, m)
        if rows is synth.selection:
            for name in H.HEADS:
                for m in H.MODALITIES:
                    assert bits_equal(post[name][m], post_sel[name][m])
        else:
            for name in H.HEADS:
                assert np.isnan(post[name]["img"][synth.selection]).all()
                assert np.isfinite(post[name]["img"][synth.held]).all()


def test_concatenated_rows_are_refused(synth):
    both = np.concatenate([synth.selection, synth.held])
    with pytest.raises(AssertionError, match="never a concatenation"):
        H.predict(synth.heads, synth.data, both)
    with pytest.raises(AssertionError, match="sorted and unique"):
        H.predict(synth.heads, synth.data, np.repeat(synth.selection[:10], 2))


def test_rows_sorted_guard_matters(synth, tmp_path):
    M = mutant(tmp_path, "r6_heads.py", "rows_sorted")
    both = np.concatenate([synth.selection[:500], synth.held[:500]])
    heads, calls = _spied({"csd": synth.heads["csd"]})
    post = M.predict(heads, synth.data, both)           # without the guard a concatenation is predicted in one call
    assert len(calls) == 2 and calls[0][1].shape[0] == 1000
    assert np.isfinite(post["csd"]["img"][both]).all()


def test_finite_nan_guard_matters(synth, tmp_path):
    heads = {"csd": copy.deepcopy(synth.heads["csd"])}
    clf = heads["csd"]["img"]
    clf.predict_proba = lambda X, _c=clf: np.full((len(X), len(_c.classes_)), np.nan)
    rows = synth.selection[:100]
    with pytest.raises(AssertionError, match="finite on the rows and NaN elsewhere"):
        H.predict(heads, synth.data, rows)
    M = mutant(tmp_path, "r6_heads.py", "finite_nan")
    assert np.isnan(M.predict(heads, synth.data, rows)["csd"]["img"][rows]).all()


# ---------------------------------------------------------------- coefficient SHA-256s

def test_coef_sha256_changes_with_one_coefficient(synth):
    base = H.coef_sha256(synth.heads)
    assert list(base) == list(H.HEADS)
    assert all(re.fullmatch(r"[0-9a-f]{64}", s) for v in base.values() for s in v.values())
    assert len({s for v in base.values() for s in v.values()}) == 10
    assert H.coef_sha256(copy.deepcopy(synth.heads)) == base
    for attr, idx in (("coef_", (3, 100)), ("intercept_", (5,))):
        heads = copy.deepcopy(synth.heads)
        arr = getattr(heads["image"]["txt"], attr)
        arr[idx] = np.nextafter(arr[idx], np.inf)          # one ulp
        got = H.coef_sha256(heads)
        assert got["image"]["txt"] != base["image"]["txt"]
        got["image"]["txt"] = base["image"]["txt"]
        assert got == base


# ---------------------------------------------------------------- check_selection (on stored copies)

def _write_stored(tmp_path, monkeypatch, synth, post_sel, perturb=None, prov_affect=None, selection=None):
    """Stored-file copies built from the synthetic heads' selection posteriors (one key perturbed by one ulp if asked),
    patched in as the inputs check_selection reads."""
    sel = synth.selection if selection is None else selection
    tag = next(_COUNT)
    for rel, keys in H.TARGETS.items():
        arrays = {"selection": sel}
        for key, (h, m) in keys.items():
            a = np.array(post_sel[h][m][synth.selection], copy=True)
            if perturb == f"{Path(rel).stem}/{key}":
                a[17, 3] = np.nextafter(a[17, 3], np.float32(np.inf))
            arrays[key] = a
        if rel == H.CSD_HEADS_REL:      # the CSD-feature head: equal to the CLIP image head here, and never a target
            arrays["style_csd__img_src"] = np.array(post_sel["csd"]["img"][synth.selection], copy=True)
        path = tmp_path / f"{tag}_{Path(rel).name}"
        np.savez(path, **arrays)
        monkeypatch.setitem(R.INPUT_PATHS, rel, path)
        monkeypatch.setitem(R.INPUT_SHA256, rel, R.sha256_file(path))
    told = {"arms": {"L": {"head": H.rto.roundtrip(prov_affect or synth.heads["affect"]["prov"])}}}
    path = tmp_path / f"{tag}_told_oracle.json"
    path.write_text(json.dumps(told))
    monkeypatch.setitem(R.INPUT_PATHS, H.TOLD_REL, path)
    monkeypatch.setitem(R.INPUT_SHA256, H.TOLD_REL, R.sha256_file(path))


def test_check_selection_passes_on_exact_copies(synth, post_sel, tmp_path, monkeypatch):
    _write_stored(tmp_path, monkeypatch, synth, post_sel)
    chk = H.check_selection(synth.heads, synth.data, synth.selection)
    assert chk["passed"] and chk["affect_identity"]
    assert tuple(chk["items"]) == H.ITEM_KEYS
    assert all(it == {"equal": True, "n_diff": 0, "max_abs_diff": 0.0} for it in chk["items"].values())


@pytest.mark.parametrize("key", H.ITEM_KEYS)
def test_check_selection_reports_one_ulp(synth, post_sel, tmp_path, monkeypatch, key):
    _write_stored(tmp_path, monkeypatch, synth, post_sel, perturb=key)
    chk = H.check_selection(synth.heads, synth.data, synth.selection)
    assert not chk["passed"] and chk["affect_identity"]
    for k, it in chk["items"].items():
        if k == key:
            stem_key = k.split("/")[1]
            h, m = next(v for keys in H.TARGETS.values() for kk, v in keys.items() if kk == stem_key)
            x = post_sel[h][m][synth.selection][17, 3]
            ulp = float(np.nextafter(x, np.float32(np.inf))) - float(x)
            assert it == {"equal": False, "n_diff": 1, "max_abs_diff": ulp} and ulp > 0
        else:
            assert it["equal"], k


def test_affect_identity_reported(synth, post_sel, tmp_path, monkeypatch):
    prov = H.rto.roundtrip(synth.heads["affect"]["prov"])
    prov["heldout_accuracy"]["img"] += 0.01
    _write_stored(tmp_path, monkeypatch, synth, post_sel, prov_affect=prov)
    chk = H.check_selection(synth.heads, synth.data, synth.selection)
    assert not chk["passed"] and not chk["affect_identity"]
    assert all(it["equal"] for it in chk["items"].values())


def test_stored_selection_guard_matters(synth, post_sel, tmp_path, monkeypatch):
    other = np.sort(np.random.default_rng(3).choice(synth.held, N_SEL, replace=False))
    _write_stored(tmp_path, monkeypatch, synth, post_sel, selection=other)
    with pytest.raises(AssertionError, match="another selection row set"):
        H.check_selection(synth.heads, synth.data, synth.selection)
    M = mutant(tmp_path, "r6_heads.py", "stored_selection")
    assert isinstance(M.check_selection(synth.heads, synth.data, synth.selection), dict)


def test_compare_bits_shape_and_sign_of_zero():
    a = np.zeros((4, 3), dtype=np.float32)
    assert H.compare_bits(a, a.copy()) == {"equal": True, "n_diff": 0, "max_abs_diff": 0.0}
    b = a.copy()
    b[0, 0] = -0.0                           # equal as a number, different bits
    assert H.compare_bits(a, b)["n_diff"] == 1 and not H.compare_bits(a, b)["equal"]
    c = H.compare_bits(a, np.zeros((4, 2), dtype=np.float32))
    assert not c["equal"] and c["max_abs_diff"] is None and c["shape"] == [[4, 3], [4, 2]]
    assert not H.compare_bits(a, a.astype(np.float64))["equal"]


# ---------------------------------------------------------------- the fit guards

def test_n_classes_guard_matters(synth, tmp_path):
    labels = dict(synth.labels)
    lab = labels["csd"].copy()
    lab[lab == 16] = 15                      # one group missing from the draw
    labels["csd"] = lab
    with pytest.raises(AssertionError, match="classes are not 0..16"):
        H.fit_all(synth.data, synth.scorer_train, labels, N_FIT)
    M = mutant(tmp_path, "r6_heads.py", "n_classes")
    heads = M.fit_all(synth.data, synth.scorer_train, labels, N_FIT)
    assert len(heads["csd"]["img"].classes_) == 16


def test_raw_features_guard_matters(synth, tmp_path):
    masked = np.full_like(synth.data.img_features, np.nan)
    masked[synth.selection] = synth.data.img_features[synth.selection]     # an EvalContext-style masked array
    data = SimpleNamespace(img_features=masked, txt_features=synth.data.txt_features)
    with pytest.raises(AssertionError, match="hazard 18"):
        H.fit_all(data, synth.scorer_train, synth.labels, N_FIT)
    M = mutant(tmp_path, "r6_heads.py", "raw_features")
    with pytest.raises(ValueError, match="NaN"):         # without the guard only sklearn's input check stops it
        M.fit_all(data, synth.scorer_train, synth.labels, N_FIT)


def test_threads_guard_matters(tmp_path, monkeypatch):
    H.require_threads()
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(AssertionError, match="8 threads"):
        H.require_threads()
    with pytest.raises(AssertionError, match="8 threads"):
        RR.main(["--out", str(tmp_path / "x.json")])     # refused before any input or data is read
    M = mutant(tmp_path, "r6_heads.py", "threads")
    M.require_threads()


# ---------------------------------------------------------------- real label files (scorer-train rows only)

def test_head_labels_real():
    with np.load(R.INPUT_PATHS[R.PREPARE_REL]) as z:
        groups, scorer_train = np.asarray(z["groups"]), np.asarray(z["scorer_train"])
    labels = H.head_labels(groups, scorer_train)
    assert tuple(labels) == H.HEADS
    outside = np.ones(len(groups), dtype=bool)
    outside[scorer_train] = False
    for h, lab in labels.items():
        assert lab.dtype == np.int64 and lab.shape == (R.N_ROWS,)
        assert (lab[outside] == -1).all()
        assert np.array_equal(np.unique(lab[scorer_train]), np.arange(H.N_CLASSES[h])), h


def test_csd_scorer_train_guard_matters(tmp_path):
    with np.load(R.INPUT_PATHS[H.CSD_GROUP_REL]) as z:
        st = np.asarray(z["scorer_train"])
    lab = H.csd_labels(st, R.N_ROWS)
    assert (lab[st] >= 0).all() and (lab >= 0).sum() == len(st)
    shifted = st.copy()
    shifted[-1] += 1
    with pytest.raises(AssertionError, match="step1_group_style"):
        H.csd_labels(shifted, R.N_ROWS)
    M = mutant(tmp_path, "r6_heads.py", "csd_scorer_train")
    assert M.csd_labels(shifted, R.N_ROWS).shape == (R.N_ROWS,)


# ---------------------------------------------------------------- run_r6_refit

def _chk(equal=True, identity=True):
    items = {k: {"equal": True, "n_diff": 0, "max_abs_diff": 0.0} for k in H.ITEM_KEYS}
    if not equal:
        items[H.ITEM_KEYS[6]] = {"equal": False, "n_diff": 2, "max_abs_diff": 5.960464477539063e-08}
    return {"passed": equal and identity, "items": items, "affect_identity": identity}


@pytest.mark.parametrize("equal,identity,code", [(True, True, 0), (False, True, 3), (True, False, 3)])
def test_report_exit_code_and_files(synth, tmp_path, capsys, equal, identity, code):
    rec = RR.record(_chk(equal, identity), synth.heads, {"x": "y"})
    assert set(rec) >= {"passed", "items", "affect_identity", "coef_sha256", "module_sha256", "input_sha256", "time"}
    assert rec["passed"] is (equal and identity) and rec["coef_sha256"] == H.coef_sha256(synth.heads)
    out = tmp_path / "refit_check.json"
    assert RR.report(rec, out) == code
    assert json.loads(out.read_text())["passed"] is (code == 0)
    printed = capsys.readouterr().out.splitlines()
    for line in printed[:9]:
        assert re.fullmatch(r"[a-z0-9_/]+: (pass|FAIL)", line), line
    assert not any("e-08" in line or "max_abs" in line for line in printed)     # differences go to the file
    diff = tmp_path / RR.DIFF_NAME
    assert diff.exists() is (code == 3)
    if not equal:
        assert f"{H.ITEM_KEYS[6]}: n_diff 2, max_abs_diff 5.960464477539063e-08" in diff.read_text()
    if not identity:
        assert "affect_identity" in diff.read_text()


def test_record_refuses_an_inconsistent_passed_flag(synth):
    chk = _chk(False, True)
    chk["passed"] = True
    with pytest.raises(AssertionError, match="disagrees"):
        RR.record(chk, synth.heads, {})


def test_exit_code_guard_matters(synth, tmp_path, capsys):
    rec = RR.record(_chk(False, True), synth.heads, {})
    M = mutant(tmp_path, "run_r6_refit.py", "exit_code")
    assert M.report(rec, tmp_path / "r.json") == 0         # without the guard a difference would exit 0
    assert RR.report(rec, tmp_path / "r2.json") == RR.EXIT_DIFF == 3
