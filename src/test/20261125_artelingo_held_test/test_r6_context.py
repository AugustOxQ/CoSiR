"""Tests of r6_context and r6_bundle on synthetic data of the real shapes (ticket 05; rule section 5 item 5, section
6 item 1, section 11). The real-data tests (selection rows, seed 42) are in test_r6_bundle.py.

Synthetic data: 308,723 rows of 512-d float32 image and caption features in paintings of four rows; a held band of
61,744 rows (15,436 whole paintings), and outside it scorer-train 183,694, selection 32,413 and val 30,872 rows; 8
emotions, 23 styles, 10 genres; 4,096 episodes per pair (12,288 per seed, held seed 52); heads with 41 / 64 / 64 / 64
/ 17 classes (fitted on a 2,000-row draw, the one stand-in). Real files read: the A3 checkpoint, the readers, the
stored selection posteriors (only to show they are refused) and prepare.npz's scorer_train index array (the PM draw's
SHA-256); no held feature, label or episode. Mutation tests load a copy of the module from tmp_path with one
`# guard:<name>` statement replaced by `pass` and show that the guard's scenario then goes through (never in place).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_context.py
"""
import ast
import dataclasses
import hashlib
import importlib.util
import itertools
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_context as X  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_heads as H  # noqa: E402

import numpy as np  # noqa: E402

import src.eval.aspect_episodes as AE  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, uniform_probe_scores  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores  # noqa: E402
from src.eval.pair_metric_baselines import fit_pair_scaler, fit_pca_basis, rca_term  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

N, D = R.N_ROWS, 512
N_SCORER, N_SEL, N_VAL, N_HELD = 183_694, 32_413, 30_872, 61_744
HELD_START = 160_000
N_FIT = 2_000
SEED = 52
A3 = R.TEST / B.A3_REL
N6_POST = R.TEST / "20261108_new_method_quick_checks/results/n6_posteriors.npz"
_COUNT = itertools.count()
MUTATED = set()


class Reached(Exception):
    """Raised by a stub placed after a guard: the scenario went through the guard."""


def _reach(*a, **kw):
    raise Reached()


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


def bits(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def scores_bits(s, t) -> bool:
    return all(bits(s[c][d], t[c][d]) for c in CONDITIONS for d in DIRECTIONS)


def _head_labels(rng, img, scorer_train, k):
    """Global labels (-1 outside scorer-train), learnable from the image features, every class present."""
    W = rng.standard_normal((D, k), dtype=np.float32)
    y = np.argmax(img[scorer_train] @ W + rng.standard_normal((len(scorer_train), k), dtype=np.float32), axis=1)
    lab = np.full(N, -1, dtype=np.int64)
    lab[scorer_train] = y
    return lab


def ctx_args(s, **over):
    """RowContext's positional arguments for the held synthetic context, with overrides."""
    a = dict(mode="held", seed=SEED, data=s.data, split=s.split, labels=s.labels, heads=s.heads, value_sets=s.vs,
             n_per_pair=R.N_PER_PAIR)
    a.update(over)
    return a


# ---------------------------------------------------------------- synthetic data of the real shapes

@pytest.fixture(scope="module")
def syn():
    rng = np.random.default_rng(20261125)
    groups = np.arange(N, dtype=np.int64) // 4
    ng = int(groups[-1]) + 1
    style_g = rng.integers(0, 23, ng)
    genre_g = rng.integers(0, 10, ng)
    genre_g[rng.random(ng) < 0.03] = -1
    emotion = rng.integers(0, 8, N)
    emotion[rng.random(N) < 0.05] = -1
    labels = {"emotion": emotion.astype(np.int64), "style": style_g[groups].astype(np.int64),
              "genre": genre_g[groups].astype(np.int64)}
    held = np.arange(HELD_START, HELD_START + N_HELD, dtype=np.int64)
    rest = np.setdiff1d(np.arange(N, dtype=np.int64), held)
    scorer_train, selection, val = rest[:N_SCORER], rest[N_SCORER:N_SCORER + N_SEL], rest[N_SCORER + N_SEL:]
    assert len(val) == N_VAL
    split = SimpleNamespace(groups=groups, train=np.sort(np.concatenate([scorer_train, selection])), val=val,
                            held=held, scorer_train=scorer_train, selection=selection)
    img = rng.standard_normal((N, D), dtype=np.float32)
    txt = rng.standard_normal((N, D), dtype=np.float32)
    data = SimpleNamespace(img_features=img, txt_features=txt)
    heads = H.fit_all(data, scorer_train, {h: _head_labels(rng, img, scorer_train, H.N_CLASSES[h]) for h in H.HEADS},
                      N_FIT)
    vs = {"emotion": list(range(8)), "style": list(range(23)), "genre": list(range(10))}
    return SimpleNamespace(data=data, split=split, labels=labels, heads=heads, vs=vs, groups=groups)


@pytest.fixture(scope="module")
def held(syn):
    return X.RowContext(**ctx_args(syn))


@pytest.fixture(scope="module")
def sel(syn):
    return X.RowContext(**ctx_args(syn, mode="selection", seed=R.SMOKE_SEEDS[0], n_per_pair=R.N_SMOKE))


@pytest.fixture(scope="module")
def pm_syn(syn):
    """The PM fit of run_baselines on the synthetic scorer-train rows (fit_pm's SHA assertion is for the real
    scorer-train array; its test is below)."""
    rows = B.fit_rows(syn.split.scorer_train)
    fi, ft = (np.asarray(x[rows], np.float32) for x in (syn.data.img_features, syn.data.txt_features))
    fi, ft = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (fi, ft))
    return SimpleNamespace(basis=fit_pca_basis(fi, ft), scaler=fit_pair_scaler(fi, ft), fit_rows_sha256="synthetic")


@pytest.fixture(scope="module")
def readers():
    return B.load_readers()


@pytest.fixture(scope="module")
def held_bundle(held, readers, pm_syn):
    return B.build_bundle_r6(held, readers, pm_syn)


# ---------------------------------------------------------------- bookkeeping

def test_every_marked_guard_has_a_mutation_test():
    names = set()
    for module in ("r6_context.py", "r6_bundle.py"):
        names |= set(re.findall(r"# guard:([a-z_0-9]+)", (HERE / module).read_text()))
    covered = set(re.findall(r'mutant\(tmp_path, "r6_(?:context|bundle)\.py", "([a-z_0-9]+)"\)',
                             Path(__file__).read_text()))
    assert names and names == covered, (names - covered, covered - names)


def test_admitted_seeds_and_names():
    assert X.ADMITTED == {("selection", 42): 4096, ("selection", 9001): 64, ("selection", 9002): 64,
                          ("selection", 9003): 64, ("held", 52): 4096, ("held", 53): 4096, ("held", 54): 4096}
    assert B.A0 == ("affect", "image", "caption") and B.A1 == ("affect", "image", "caption", "csd")
    assert B.E2_PARTS == B.n6.PARTS and B.E2_FROM["affect"] == "affect_km"            # B uses affect-km (hazard 9)
    assert B.PM_NAMES == ("diag", "diag_relu", "bilinear", "kissme", "xing", "wang", "probe", "tip",
                          "value_prototype")
    for rel in (B.BASELINES42_REL, B.A3_REL, *B.READER_RELS):
        assert rel in R.INPUT_SHA256, rel


def test_term_order_is_run_baselines_order():
    """The PM terms are computed in run_baselines.py's order (its `terms` dict, parsed, not imported)."""
    tree = ast.parse((R.AB / "run_baselines.py").read_text())
    keys = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "terms" for t in node.targets):
            keys = [ast.literal_eval(k) for k in node.value.keys]
    assert keys is not None and tuple(keys[:10]) == B.TERM_ORDER, keys


def test_module_never_calls_the_forbidden_functions():
    """Hazards 14, 17: r6_context and r6_bundle never call build_bundle, _check_seed, run_baselines or
    load_posteriors, and copy no round 4 or 5 code (no r4_ / r5_ import)."""
    for module in ("r6_context.py", "r6_bundle.py"):
        tree = ast.parse((HERE / module).read_text())
        called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert not called & {"build_bundle", "_check_seed", "load_posteriors", "extend_a1", "_load_csd_heads",
                             "EvalContext"}, (module, called)
        imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not any(str(m).startswith(("r4_", "r5_", "run_baselines", "r3_bundle")) for m in imported), imported


# ---------------------------------------------------------------- the held context (synthetic, real shapes)

def test_held_context_rows_and_features(syn, held):
    assert held.mode == "held" and held.seed == SEED and not held.smoke
    assert np.array_equal(held.rows, syn.split.held) and held.rows.dtype == np.int64
    assert held.selection is held.rows                                  # the load_posteriors tripwire
    assert held.in_rows.sum() == N_HELD and held.in_rows[syn.split.held].all()
    for m in ("img", "txt"):
        x, raw = getattr(held, m), getattr(syn.data, f"{m}_features")
        assert x.dtype == np.float32 and x.shape == (N, D)
        assert np.isnan(x[~held.in_rows]).all() and np.isfinite(x[held.in_rows]).all()
        assert bits(x[held.rows], raw[held.rows])
    want = cosine_scores(EvalInputs(held.img, held.txt), held.pooled)
    assert scores_bits(held.cos, want)
    assert all(np.isfinite(held.cos[c][d]).all() for c in CONDITIONS for d in DIRECTIONS)


def test_held_context_posteriors(syn, held):
    assert tuple(held.post) == H.HEADS
    want = H.predict(syn.heads, syn.data, syn.split.held)
    for h in H.HEADS:
        for m in H.MODALITIES:
            p = held.post[h][m]
            assert p.dtype == np.float32 and p.shape == (N, H.N_CLASSES[h])
            assert np.isnan(p[~held.in_rows]).all() and np.isfinite(p[held.in_rows]).all()
            assert bits(p, want[h][m])
    assert held.coef_sha256 == H.coef_sha256(syn.heads)


def test_held_context_episodes(syn, held):
    assert held.n == 3 * R.N_PER_PAIR == len(held.pooled.anchor)
    assert [len(ep.anchor) for ep in held.eps.per_pair] == [R.N_PER_PAIR] * 3
    member = held.pooled.rows()
    assert held.in_rows[member].all() and np.isin(member, syn.split.held).all()
    assert np.array_equal(held.pair_index, np.repeat(np.arange(3), R.N_PER_PAIR))
    assert np.array_equal(held.parity, np.arange(held.n) % 2)
    assert np.array_equal(held.anchor, held.pooled.anchor)
    assert np.array_equal(held.anchor_group, syn.groups[held.pooled.anchor])
    assert held.episode_sha == {p: AE.episodes_sha256(ep) for p, ep in zip(R.PAIR_NAMES, held.eps.per_pair)}
    for f in E.FIELDS:
        assert np.array_equal(getattr(held.pooled, f),
                              np.concatenate([getattr(ep, f) for ep in held.eps.per_pair]))


def test_held_codes_on_rows_only(syn, held):
    ic, tc = held.encode(A3)
    model, _ = load_factor_checkpoint(A3, device="cpu")
    wi, wt = encode_rows(model, syn.data.img_features, syn.data.txt_features, rows=syn.split.held, device="cpu")
    for full, want in ((ic, wi), (tc, wt)):
        assert full.dtype == np.float32 and full.shape == (N, want.shape[1])
        assert np.isnan(full[~held.in_rows]).all() and np.isfinite(full[held.in_rows]).all()
        assert bits(full[held.rows], want)


def test_load_posteriors_refuses_the_held_context(held, sel):
    """ctx.selection is the evaluated rows, so the stored selection posteriors are refused on any other row set."""
    for ctx in (held, sel):
        with pytest.raises(AssertionError, match="another selection row set"):
            R.run_n6.load_posteriors(N6_POST, ctx)


def test_selection_mode_uses_the_selection_rows(syn, sel):
    assert sel.mode == "selection" and sel.smoke and sel.n == 3 * R.N_SMOKE
    assert np.array_equal(sel.rows, syn.split.selection) and sel.selection is sel.rows
    assert sel.in_rows[sel.pooled.rows()].all()
    assert np.isnan(sel.img[syn.split.held]).all() and np.isnan(sel.post["csd"]["txt"][syn.split.held]).all()


def test_episodes_are_built_and_reported_before_the_posteriors(syn, monkeypatch):
    order = []
    real_predict = H.predict

    def predict(*a, **kw):
        order.append("predict")
        return real_predict(*a, **kw)

    monkeypatch.setattr(H, "predict", predict)
    ctx = X.RowContext(**ctx_args(syn, mode="selection", seed=R.SMOKE_SEEDS[1], n_per_pair=R.N_SMOKE),
                       on_episodes=lambda eps: order.append(("episodes", dict(eps.sha))))
    assert order == [("episodes", ctx.episode_sha), "predict"]


# ---------------------------------------------------------------- the context's guards

def test_seed_mode_guard(syn, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "build_seed", _reach)            # anything past the guard stops here
    for over in ({"seed": 42}, {"mode": "selection", "seed": SEED}, {"n_per_pair": R.N_SMOKE},
                 {"mode": "selection", "seed": 42, "n_per_pair": R.N_SMOKE}, {"mode": "test"}):
        with pytest.raises(AssertionError, match="admits|mode"):
            X.RowContext(**ctx_args(syn, **over))
    mut = mutant(tmp_path, "r6_context.py", "seed_mode")
    with pytest.raises(Reached):
        mut.RowContext(**ctx_args(syn, seed=42))


def test_rows_split_guard(syn, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "build_seed", _reach)
    sp = syn.split
    bad_scorer = SimpleNamespace(**{**vars(sp), "scorer_train": np.sort(np.append(sp.scorer_train, sp.held[0]))})
    bad_sel = SimpleNamespace(**{**vars(sp), "selection": np.sort(np.append(sp.selection, sp.held[-1]))})
    for split in (bad_scorer, bad_sel):
        with pytest.raises(AssertionError, match="overlap"):
            X.RowContext(**ctx_args(syn, split=split))
    mut = mutant(tmp_path, "r6_context.py", "rows_split")
    for split in (bad_scorer, bad_sel):
        with pytest.raises(Reached):
            mut.RowContext(**ctx_args(syn, split=split))


def test_masked_guard(syn, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "build_seed", _reach)
    txt = syn.data.txt_features.copy()
    txt[syn.split.held[123], 7] = np.nan                        # a non-finite feature on an evaluated row
    data = SimpleNamespace(img_features=syn.data.img_features, txt_features=txt)
    with pytest.raises(AssertionError, match="masked features"):
        X.RowContext(**ctx_args(syn, data=data))
    mut = mutant(tmp_path, "r6_context.py", "masked")
    with pytest.raises(Reached):
        mut.RowContext(**ctx_args(syn, data=data))


def _with_member_outside(eps, outside_row):
    """build_seed's namespace with one candidate of the first episode replaced by ``outside_row``."""
    cand = eps.pooled.candidates.copy()
    cand[0, 5] = outside_row
    pooled = dataclasses.replace(eps.pooled, candidates=cand)
    return SimpleNamespace(**{**vars(eps), "pooled": pooled})


def test_episode_member_outside_rows_raises(syn, held, tmp_path, monkeypatch):
    """Held mode, real shapes: the context's own guard fires when an episode member (here a candidate) is not a held
    row, independently of build_seed's guard (tested with r6_episodes); with the guard deleted the context goes on to
    the posteriors."""
    tampered = _with_member_outside(held.eps, int(syn.split.selection[0]))
    monkeypatch.setattr(E, "build_seed", lambda *a, **kw: tampered)
    monkeypatch.setattr(H, "predict", _reach)
    with pytest.raises(AssertionError, match="1 episode member rows lie outside the held rows"):
        X.RowContext(**ctx_args(syn))
    mut = mutant(tmp_path, "r6_context.py", "episode_rows")
    with pytest.raises(Reached):
        mut.RowContext(**ctx_args(syn))


def test_post_mask_guard(syn, held, tmp_path, monkeypatch):
    post = {h: dict(held.post[h]) for h in held.post}
    leak = post["csd"]["img"].copy()
    leak[syn.split.selection[0]] = 1.0 / H.N_CLASSES["csd"]     # a finite posterior outside the held rows
    post["csd"] = {"img": leak, "txt": post["csd"]["txt"]}
    monkeypatch.setattr(E, "build_seed", lambda *a, **kw: held.eps)
    monkeypatch.setattr(H, "predict", lambda *a, **kw: post)
    with pytest.raises(AssertionError, match="post csd/img"):
        X.RowContext(**ctx_args(syn))
    mut = mutant(tmp_path, "r6_context.py", "post_mask")
    ctx = mut.RowContext(**ctx_args(syn))
    assert np.isfinite(ctx.post["csd"]["img"][syn.split.selection[0]]).all()


def test_codes_mask_guard(held, tmp_path, monkeypatch):
    def bad_encode(model, img, txt, rows=None, **kw):
        ic, tc = encode_rows(model, img, txt, rows=rows, **kw)
        ic[3, 0] = np.nan                                      # a non-finite code on an evaluated row
        return ic, tc

    monkeypatch.setattr(X, "encode_rows", bad_encode)
    with pytest.raises(AssertionError, match="codes"):
        held.encode(A3)
    mut = mutant(tmp_path, "r6_context.py", "codes_mask")
    monkeypatch.setattr(mut, "encode_rows", bad_encode)
    ctx = SimpleNamespace(rows=held.rows, in_rows=held.in_rows, data=held.data)
    ic, _ = mut.RowContext.encode(ctx, A3)
    assert np.isnan(ic[held.rows[3], 0])


# ---------------------------------------------------------------- fit_pm (rule section 11)

@pytest.fixture(scope="module")
def real_scorer_train():
    """prepare.npz's scorer_train index array only (no feature, no label, no held row)."""
    R.assert_input(R.PREPARE_REL)
    return R._read_npz_keys(R.INPUT_PATHS[R.PREPARE_REL], ("scorer_train",))["scorer_train"]


def test_fit_pm_asserts_the_unsorted_draw_sha(syn, real_scorer_train, tmp_path, monkeypatch):
    pm = B.fit_pm(syn.data, real_scorer_train)            # synthetic features on the real draw: the SHA holds
    rows = B.fit_rows(real_scorer_train)
    assert pm.fit_rows_sha256 == hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()
    assert pm.fit_rows_sha256 != hashlib.sha256(np.sort(rows).astype(np.int64).tobytes()).hexdigest()  # hazard 11
    assert pm.basis.components.shape == (32, D) and pm.scaler.mean.shape == (D,)
    other = real_scorer_train[1:]                         # another scorer-train array: another draw
    monkeypatch.setattr(B, "fit_pca_basis", _reach)
    with pytest.raises(AssertionError, match="unsorted draw"):
        B.fit_pm(syn.data, other)
    mut = mutant(tmp_path, "r6_bundle.py", "fit_rows_sha")
    monkeypatch.setattr(mut, "fit_pca_basis", _reach)
    with pytest.raises(Reached):
        mut.fit_pm(syn.data, other)


def test_fit_pm_refuses_masked_features(syn, real_scorer_train, tmp_path, monkeypatch):
    rows = B.fit_rows(real_scorer_train)
    img = syn.data.img_features.copy()
    img[rows[17]] = np.nan                                 # as if the fit got a masked array (hazard 18)
    data = SimpleNamespace(img_features=img, txt_features=syn.data.txt_features)
    with pytest.raises(AssertionError, match="hazard 18"):
        B.fit_pm(data, real_scorer_train)
    mut = mutant(tmp_path, "r6_bundle.py", "pm_raw")
    monkeypatch.setattr(mut, "fit_pca_basis", _reach)
    with pytest.raises(Reached):
        mut.fit_pm(data, real_scorer_train)


# ---------------------------------------------------------------- the bundle's guards

def test_readers_guard(readers, tmp_path):
    B.check_readers(readers)
    bad = {**readers, "feature_names": list(readers["feature_names"])[::-1]}
    with pytest.raises(AssertionError, match="readers"):
        B.check_readers(bad)
    mut = mutant(tmp_path, "r6_bundle.py", "readers")
    mut.check_readers(bad)


def _fake_ctx(held, **over):
    keys = ("mode", "seed", "n", "n_per_pair", "pooled", "eps", "pair_index", "parity", "anchor", "rows", "in_rows",
            "selection", "anchor_group", "split")
    d = {k: getattr(held, k) for k in keys}
    d.update(over)
    return SimpleNamespace(**d)


def test_check_context_passes_the_held_context(held):
    assert np.array_equal(B.check_context(held), held.anchor_group)


def test_per_seed_guard(held, tmp_path):
    """Hazard 6: three seeds' episodes pooled (36,864, posing as 12,288 per pair) are not one seed's."""
    pooled3 = AE.concat_episodes([held.pooled] * 3)
    n3 = len(pooled3.anchor)
    fake = _fake_ctx(held, n=n3, n_per_pair=n3 // 3, pooled=pooled3, eps=SimpleNamespace(pooled=pooled3),
                     pair_index=np.repeat(np.arange(3), n3 // 3), parity=np.arange(n3) % 2, anchor=pooled3.anchor,
                     anchor_group=held.split.groups[pooled3.anchor])
    with pytest.raises(AssertionError, match="one seed's pooled episodes"):
        B.check_context(fake)
    mut = mutant(tmp_path, "r6_bundle.py", "per_seed")
    assert len(mut.check_context(fake)) == n3


def test_bundle_rows_guard(syn, held, tmp_path):
    tampered = _with_member_outside(held.eps, int(syn.split.selection[0])).pooled
    fake = _fake_ctx(held, pooled=tampered, eps=SimpleNamespace(pooled=tampered))
    with pytest.raises(AssertionError, match="outside the held rows"):
        B.check_context(fake)
    mut = mutant(tmp_path, "r6_bundle.py", "bundle_rows")
    mut.check_context(fake)


def test_cl_groups_guard(held, tmp_path):
    """Round 3's final review N11: cl == groups[anchor]."""
    cl = held.anchor_group.copy()
    cl[7] += 1
    fake = _fake_ctx(held, anchor_group=cl)
    with pytest.raises(AssertionError, match="N11"):
        B.check_context(fake)
    mut = mutant(tmp_path, "r6_bundle.py", "cl_groups")
    assert np.array_equal(mut.check_context(fake), cl)


def test_scores_guard(held, tmp_path):
    good = {c: {d: np.zeros((held.n, 13), np.float32) for d in DIRECTIONS} for c in CONDITIONS}
    B.check_scores(good, held.n, "good", np.float32)
    nan = {c: {d: x.copy() for d, x in good[c].items()} for c in CONDITIONS}
    nan["b"]["t2i"][5, 3] = np.nan
    f64 = {c: {d: x.astype(np.float64) for d, x in good[c].items()} for c in CONDITIONS}
    for bad, dtype in ((nan, None), (f64, np.float32)):
        with pytest.raises(AssertionError, match="must be finite"):
            B.check_scores(bad, held.n, "bad", dtype)
    mut = mutant(tmp_path, "r6_bundle.py", "scores")
    for bad, dtype in ((nan, None), (f64, np.float32)):
        mut.check_scores(bad, held.n, "bad", dtype)


def test_build_bundle_checks_the_context_first(held, readers, pm_syn, monkeypatch):
    monkeypatch.setattr(B, "check_context", _reach)
    with pytest.raises(Reached):
        B.build_bundle_r6(held, readers, pm_syn)


# ---------------------------------------------------------------- the held bundle (synthetic, real shapes)

def test_held_bundle_fields(held, held_bundle):
    b = held_bundle
    assert all(hasattr(b, f) for f in B.FIELDS)
    assert b.n == held.n and b.mode == "held" and b.seed == SEED
    assert np.array_equal(b.cl, held.split.groups[b.anchor]) and np.array_equal(b.anchor, held.pooled.anchor)
    assert np.array_equal(b.parity, np.arange(b.n) % 2)
    assert np.array_equal(b.pair_index, np.repeat(np.arange(3), R.N_PER_PAIR))
    assert b.episodes_sha256 == held.episode_sha
    assert b.module_sha256 == R.r6_module_shas() and "src/test/20261125_artelingo_held_test/r6_bundle.py" in \
        b.module_sha256
    assert b.coef_sha256 == held.coef_sha256 and b.fit_rows_sha256 == "synthetic"
    assert b.input_sha256 == R.assert_inputs()
    assert b.checks and all(v is True for v in b.checks.values())
    assert set(b.post) == set(B.A0) and all(b.post[h] is held.post[h] for h in B.A0)    # D1 affect, not affect-km
    assert tuple(b.pm_terms) == ("rca",) + B.PM_NAMES
    assert B.validate(b)


def test_held_bundle_terms(held, held_bundle, pm_syn):
    b, ep = held_bundle, held.pooled
    # T_N1u from A3 codes of the held rows only
    ic, tc = held.encode(A3)
    assert scores_bits(b.t_n1u, centered_term(EvalInputs(held.img, held.txt, ic, tc), ep, uniform=True))
    # B's T_6u is run_n6's T_6u (n6_terms) on the refit E2 heads (affect = affect-km); B0's and B1's over A0 and A1
    e2 = {"affect": held.post["affect_km"], "image": held.post["image"], "caption": held.post["caption"]}
    assert scores_bits(b.t6u_B, B.n6.n6_terms(e2, ep)[2])
    a0 = {h: held.post[h] for h in B.A0}
    assert scores_bits(b.t6u_B0, uniform_probe_scores(a0, ep, B.A0))
    assert scores_bits(b.t6u_B1, uniform_probe_scores({**a0, "csd": held.post["csd"]}, ep, B.A1))
    assert not scores_bits(b.t6u_B, b.t6u_B0) and not scores_bits(b.t6u_B0, b.t6u_B1)
    assert scores_bits(b.cos, held.cos)
    assert scores_bits(b.pm_terms["rca"], rca_term(EvalInputs(held.img, held.txt), ep, pm_syn.basis))
