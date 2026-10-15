"""Tests of r5_bundle.py (rule DECISION_RULE.md of this folder: D5, D11, §4 item 3, §5 items 4 and 5, §10 list A item 4).
Run from this folder, output redirected (rule §10: the log is read only for pass/fail, Error and Traceback):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r5_bundle.py \
        > results/smoke/test_r5_bundle.log 2>&1

Two kinds of test:
  synthetic  round-4-shaped bundles whose stack, F and B'(A0) are computed by round 1's and round 3's own functions
             from synthetic posteriors on synthetic episodes; the affect posteriors sit in r3_bundle._HEADS in the real
             shape ({"post": {"img", "txt"}}, the same dict object as bundle.post["affect"]); synthetic 'ge'
             placements minted by r5_guard.ge_from_file from a synthetic file (GE_POST_SHA, N_ROWS and N_SELECTION
             patched), the guard released from a synthetic results/regression_check.json in a temporary folder.
  smoke      the real round-4 builder on smoke seed 9001 with the bundle's own CLIP placement only (no GE placement on
             real data): the extension reproduces the bundle's stack, F and B'(A0) exactly, leaves round 3's and round
             4's fields, bundle.post and _HEADS unchanged, prints no value, and its cache round trip is exact.
Every assertion on real data asserts a plain bool with a message, so a failure prints no array or metric value.
"""
import contextlib
import io
import json
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402
import r5_guard as R5G  # noqa: E402
import r5_bundle as R5B  # noqa: E402

from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores  # noqa: E402

R3, RB3, RB4, C = R5.R3, R5.RB3, R5.RB4, R5.C
rbe, rf = R3.rbe, R3.rf
A0, A1 = R3.A0, RB4.A1
SMOKE_SEED = 9001
K41 = R5.N_CLASSES
# any decimal number: "0.5", ".5", "19.1", "5.", "5e-03", "2E+3" (round 4's pattern plus a trailing-dot form)
LEAK = re.compile(r"\.\d|\d\.(?!\w)|\d[eE][-+]?\d")


# ---------------------------------------------------------------- synthetic helpers (also used by test_r5_fusion)

def cf_scores(rng, n, k=13):
    base = {d: rng.normal(size=(n, k)).astype(np.float32) for d in DIRECTIONS}
    return {c: {d: base[d].copy() for d in DIRECTIONS} for c in CONDITIONS}


def post_side(rng, rows, k, sel, read_only=False):
    """A posterior array: float32 (rows, k), Dirichlet rows on the selection rows, NaN elsewhere (as the real heads)."""
    x = np.full((rows, k), np.nan, np.float32)
    x[sel] = rng.dirichlet(np.full(k, 0.5), size=len(sel)).astype(np.float32)
    if read_only:
        x.flags.writeable = False
    return x


def readers_on(F, seed=0):
    """Two half-readers (StandardScaler + LogisticRegression, round 1's pickle-dict layout) fitted on jittered copies of
    the bundle's own features, labels = arg max of the three Delta columns, so picks and margins vary with F."""
    rng = np.random.default_rng(seed)
    X0 = np.vstack([F["a"], F["b"]])
    halves = []
    for _ in range(2):
        X = np.vstack([X0 + rng.normal(scale=X0.std(axis=0, keepdims=True) * 0.3 + 1e-6, size=X0.shape)
                       for _ in range(4)])
        y = np.argmax(X[:, [2, 8, 14]] + rng.normal(scale=0.01, size=(len(X), 3)), axis=1)
        sc = StandardScaler().fit(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            model = LogisticRegression(max_iter=500).fit(sc.transform(X), y)
        halves.append({"scaler": sc, "model": model})
    return {"halves": halves, "feature_names": rf.feature_names(A0), "config": "A0"}


def synth_bundle(n=24, rows=50, seed=0, bundle_seed=SMOKE_SEED, smoke=True):
    """A synthetic round-4 bundle (round 3's fields plus round 4's A1 fields) with a context whose episodes index
    selection rows only; stack, F and B'(A0) are round 1's and round 3's own functions of the A0 posteriors. The affect
    posterior dict is registered in r3_bundle._HEADS (real shape) and IS bundle.post["affect"]."""
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(rows, size=rows - 8, replace=False)).astype(np.int64)

    def draw(*shape):
        return rng.choice(sel, size=shape).astype(np.int64)

    ep = AspectEpisodes("emotion", "style", draw(n), draw(n, 13), draw(n, 4), draw(n, 4), draw(n, 4), draw(n, 4))
    aff = {"img": post_side(rng, rows, K41, sel, True), "txt": post_side(rng, rows, K41, sel, True)}
    post = {"affect": aff,
            "image": {"img": post_side(rng, rows, 7, sel), "txt": post_side(rng, rows, 7, sel)},
            "caption": {"img": post_side(rng, rows, 5, sel), "txt": post_side(rng, rows, 5, sel)},
            "csd": {"img": post_side(rng, rows, 6, sel), "txt": post_side(rng, rows, 6, sel)}}
    key = f"synthetic:{seed}:{bundle_seed}:{n}:{rows}"
    RB3._HEADS[key] = {"post": aff, "prov": {"n_classes": K41}, "selection": sel.copy(),
                       "scorer_train": np.arange(3, dtype=np.int64), "n_rows_all": rows}
    parity = np.arange(n) % 2
    cos, t_n1u, B = cf_scores(rng, n), cf_scores(rng, n), cf_scores(rng, n)
    cl = rng.integers(0, 6, size=n).astype(np.int64)
    ctx = SimpleNamespace(pooled=ep, cos=cos, parity=parity, selection=sel, groups=np.arange(rows) // 3, n=n,
                          anchor_group=cl, pair_index=np.arange(n) % 3)
    pA0 = {h: post[h] for h in A0}
    stack = C.grouping_stack(pA0, ep, A0)
    F = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=pA0), A0)[0]
    Bp = crossfit_condition_free(cos, t_n1u, uniform_probe_scores(pA0, ep, A0), parity)[0]
    F1 = {c: np.concatenate([F[c], rng.normal(size=(n, 6))], axis=1) for c in CONDITIONS}
    F1["b"][:, 20] = -F1["a"][:, 20]
    Bp1 = cf_scores(rng, n)
    b = SimpleNamespace(
        seed=bundle_seed, smoke=smoke, n=n, ctx=ctx, cl=cl, parity=parity, pair_index=ctx.pair_index,
        anchor=np.asarray(ep.anchor), cos=cos, B=B, pB=per_anchor(B), Bp=Bp, pBp=per_anchor(Bp), t_n1u=t_n1u,
        post=post, stack=stack, F=F, affect_head={"n_classes": K41}, affect_heads_reused=False,
        readers=readers_on(F, seed), readers_record={"rule_sha256": "synthetic", "smoke": False},
        episodes_sha256={"emotion__style": "synthetic"}, input_sha256={"x": "y"}, checks={"synthetic": True},
        from_cache=False, build_s=0.0,
        F1=F1, readers_a1=None, readers_a1_record=None, Bp1=Bp1, pBp1=per_anchor(Bp1),
        v=np.minimum(F["a"][:, 6], F["a"][:, 7]), checks_a1={"synthetic": True}, input_sha256_a1={"q": "r"})
    RB3.validate(b)
    return b


def make_ge(tmp_path, monkeypatch, bundle, seed=7, rows=None, name=None, n_rows=None):
    """A synthetic GE file for the bundle's rows (default: its selection rows), minted by r5_guard.ge_from_file with
    r5_common.GE_POST_SHA, N_ROWS (default: the bundle's row count) and N_SELECTION patched to it."""
    n_rows = int(bundle.post["affect"]["img"].shape[0]) if n_rows is None else int(n_rows)
    rows = np.asarray(bundle.ctx.selection if rows is None else rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    p = tmp_path / (name or f"ge_{seed}.npz")
    np.savez(p, post_sel=rng.dirichlet(np.full(K41, 0.5), size=len(rows)).astype(np.float32), rows=rows,
             classes=np.arange(K41))
    monkeypatch.setattr(R5, "GE_POST_SHA", R5.sha256_file(p))
    monkeypatch.setattr(R5, "N_ROWS", n_rows)
    monkeypatch.setattr(R5, "N_SELECTION", len(rows))
    return R5G.ge_from_file(p)


def release_guard(tmp_path):
    """Release the guard from a synthetic results/regression_check.json in a temporary folder."""
    d = tmp_path / "results"
    d.mkdir(exist_ok=True)
    p = d / "regression_check.json"
    p.write_text(json.dumps({"rule_sha256": R5.RULE_SHA, "all_passed": True,
                             "items": {str(i): {"passed": True} for i in range(1, 5)}}))
    R5G.release(p)


@pytest.fixture(autouse=True)
def isolate():
    """Each test starts with the guard closed and leaves _HEADS as it found it."""
    saved = dict(RB3._HEADS)
    R5G._reset_for_tests()
    yield
    RB3._HEADS.clear()
    RB3._HEADS.update(saved)
    R5G._reset_for_tests()


@pytest.fixture
def ge_env(tmp_path, monkeypatch):
    """(bundle, ge placement, ge extension) on a synthetic bundle, the guard released."""
    b = synth_bundle(seed=1)
    pl = make_ge(tmp_path, monkeypatch, b, seed=11)
    release_guard(tmp_path)
    return b, pl, R5B.extend(b, pl)


def _independent_post(b, Q):
    return {"affect": {"img": b.post["affect"]["img"], "txt": Q}, "image": b.post["image"],
            "caption": b.post["caption"]}


# ---------------------------------------------------------------- post_Q and the extension (synthetic)

def test_post_q_is_a_new_dict_in_a0_order():
    b = synth_bundle()
    Q = np.zeros_like(b.post["affect"]["txt"])
    before = R5B.shared_state(b)
    pq = R5B.post_q(b.post, Q)
    assert tuple(pq) == A0 and pq is not b.post and pq["affect"] is not b.post["affect"]
    assert pq["affect"]["txt"] is Q and pq["affect"]["img"] is b.post["affect"]["img"]
    assert pq["image"] is b.post["image"] and pq["caption"] is b.post["caption"]
    assert tuple(b.post) == A1 and b.post["affect"]["txt"] is not Q
    assert R5B.shared_state(b) == before


def test_extend_with_the_clip_placement_returns_the_bundles_own_fields():
    b = synth_bundle()
    ext = R5B.extend(b, R5G.clip_from_bundle(b))
    same = R5B.same_as_bundle(b, ext)
    assert same["all_pass"] and all(v is True for v in same.values())
    assert ext.kind == "clip" and (ext.seed, ext.smoke, ext.n) == (b.seed, b.smoke, b.n)
    assert all(v is True for v in ext.checks.values())


def test_extend_with_a_synthetic_ge_q_changes_only_the_affect_parts(ge_env):
    b, pl, ext = ge_env
    assert ext.kind == "ge" and ext.placement is pl and ext.placement_sha256 == pl.sha256
    for d in DIRECTIONS:
        np.testing.assert_array_equal(ext.stack[d][:, 1:], b.stack[d][:, 1:])          # image and caption unchanged
        assert np.any(ext.stack[d][:, 0] != b.stack[d][:, 0])                           # affect slice changed
    for c in CONDITIONS:
        np.testing.assert_array_equal(ext.F[c][:, 6:], b.F[c][:, 6:])                  # columns 6 to 17 unchanged
        assert np.any(ext.F[c][:, :6] != b.F[c][:, :6])                                 # affect columns changed
        np.testing.assert_array_equal(ext.F["b"][:, 2], -ext.F["a"][:, 2])
    # B'_Q is round 3's recipe on post_Q, recomputed here from an independently built post_Q
    pq = _independent_post(b, pl.Q)
    ctx, ep = b.ctx, b.ctx.pooled
    Bp, picks = crossfit_condition_free(ctx.cos, b.t_n1u, uniform_probe_scores(pq, ep, A0), ctx.parity)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            np.testing.assert_array_equal(ext.Bp[c][d], Bp[c][d])
            np.testing.assert_array_equal(ext.Bp[c][d], ext.Bp["a"][d])                 # condition-free
    pa = per_anchor(Bp)
    for m in METRICS:
        np.testing.assert_array_equal(ext.pBp[m], pa[m])
    assert (ext.pBp["gain"] == 0).all()
    assert ext.Bp_picks == {h: [float(x) for x in v] for h, v in picks.items()}
    stack = C.grouping_stack(pq, ep, A0)
    F = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=pq), A0)[0]
    for d in DIRECTIONS:
        np.testing.assert_array_equal(ext.stack[d], stack[d])
    for c in CONDITIONS:
        np.testing.assert_array_equal(ext.F[c], F[c])


def test_positive_check_passes_on_a_ge_extension_and_is_boolean_only(ge_env):
    b, pl, ext = ge_env
    out = R5B.positive_check(b, ext, pl)
    assert set(out) == {"placement_is_ge", "extension_built_from_this_placement",
                        "affect_slice_equals_independent_einsum", "affect_slice_differs_from_bundle_on_an_episode",
                        "F_columns_0_to_5_differ_from_bundle_on_an_episode", "all_pass"}
    assert all(type(v) is bool for v in out.values())
    assert all(out.values())


def test_an_extension_that_ignores_its_q_fails_the_positive_check(tmp_path, monkeypatch):
    b = synth_bundle(seed=2)
    pl = make_ge(tmp_path, monkeypatch, b, seed=12)
    release_guard(tmp_path)
    real = R5B.post_q
    monkeypatch.setattr(R5B, "post_q", lambda post, Q: real(post, post["affect"]["txt"]))   # the mutant ignores Q
    ext = R5B.extend(b, pl)
    out = R5B.positive_check(b, ext, pl)
    assert out["all_pass"] is False
    assert out["affect_slice_equals_independent_einsum"] is False
    assert out["affect_slice_differs_from_bundle_on_an_episode"] is False
    assert out["F_columns_0_to_5_differ_from_bundle_on_an_episode"] is False
    assert out["extension_built_from_this_placement"] is True


def test_positive_check_with_the_clip_placement_is_not_a_pass():
    b = synth_bundle(seed=3)
    pl = R5G.clip_from_bundle(b)
    out = R5B.positive_check(b, R5B.extend(b, pl), pl)
    assert out["placement_is_ge"] is False and out["all_pass"] is False
    assert out["affect_slice_equals_independent_einsum"] is True                       # the einsum is the stack's own
    assert out["affect_slice_differs_from_bundle_on_an_episode"] is False


def test_positive_check_refuses_a_ge_placement_with_a_clip_extension_before_release(tmp_path, monkeypatch):
    """positive_check's own placement guard (review I2): the CLIP extension passes require_ext, so only the guard on
    the placement argument stops the Q_GE einsum before release."""
    b = synth_bundle(seed=4)
    clip_ext = R5B.extend(b, R5G.clip_from_bundle(b))
    pl = make_ge(tmp_path, monkeypatch, b, seed=13)                         # minted; the guard stays closed
    called = []
    real = np.einsum
    monkeypatch.setattr(R5B.np, "einsum", lambda *a, **k: called.append(1) or real(*a, **k))
    with pytest.raises(R5G.GuardError, match=re.escape("r5_bundle.positive_check")):
        R5B.positive_check(b, clip_ext, pl)
    assert not called, "the Q_GE einsum was computed before release"


def test_positive_check_refuses_an_extension_of_another_placement(tmp_path, monkeypatch):
    b = synth_bundle(seed=4)
    clip_ext = R5B.extend(b, R5G.clip_from_bundle(b))
    pl = make_ge(tmp_path, monkeypatch, b, seed=13)
    release_guard(tmp_path)
    out = R5B.positive_check(b, clip_ext, pl)
    assert out["extension_built_from_this_placement"] is False and out["all_pass"] is False


# ---------------------------------------------------------------- D5: nothing assigned into bundle.post or _HEADS

def _key_of(b):
    return next(k for k, v in RB3._HEADS.items() if v["post"] is b.post["affect"])


MUTATIONS = ["assign_bundle_post_affect_txt", "assign_HEADS_post_txt", "new_HEADS_entry", "HEADS_selection",
             "HEADS_prov", "HEADS_entry_replaced", "csd_in_place", "image_post_in_place", "new_post_key",
             "round4_a1_field", "round3_field", "rebind_bundle_post"]


@pytest.mark.parametrize("what", MUTATIONS)
def test_a_mutation_of_the_shared_state_fires(tmp_path, monkeypatch, what):
    b = synth_bundle(seed=5)
    pl = make_ge(tmp_path, monkeypatch, b, seed=14)
    release_guard(tmp_path)
    key = _key_of(b)
    aff, txt0 = b.post["affect"], b.post["affect"]["txt"]
    real = R5B.post_q

    def mutant(post, Q):
        if what == "assign_bundle_post_affect_txt":
            post["affect"]["txt"] = Q                                    # the shared dict: bundle and _HEADS
        elif what == "assign_HEADS_post_txt":
            RB3._HEADS[key]["post"]["txt"] = np.array(Q, copy=True)
        elif what == "new_HEADS_entry":
            RB3._HEADS["intruder"] = {"post": {"img": aff["img"], "txt": Q}}
        elif what == "HEADS_selection":                                      # parts of _HEADS not shared with post
            RB3._HEADS[key]["selection"] = RB3._HEADS[key]["selection"][::-1].copy()
        elif what == "HEADS_prov":
            RB3._HEADS[key]["prov"]["n_classes"] = 40
        elif what == "HEADS_entry_replaced":
            RB3._HEADS[key] = dict(RB3._HEADS[key])                          # same content, another entry object
        elif what == "csd_in_place":
            post["csd"]["img"][b.ctx.selection[0], 0] += np.float32(1.0)
        elif what == "image_post_in_place":
            post["image"]["txt"][b.ctx.selection[1], 2] += np.float32(1.0)
        elif what == "new_post_key":
            post["extra"] = post["affect"]
        elif what == "round4_a1_field":
            b.F1["a"][0, 0] += 1.0
        elif what == "round3_field":
            b.B["a"]["i2t"][0, 0] += np.float32(1.0)
        elif what == "rebind_bundle_post":
            b.post = dict(post)
        return real(post, Q)

    monkeypatch.setattr(R5B, "post_q", mutant)
    try:
        with pytest.raises(AssertionError, match="rule D5"):
            R5B.extend(b, pl)
    finally:
        aff["txt"] = txt0


def test_the_shared_state_covers_heads_and_the_whole_post():
    b = synth_bundle(seed=6)
    s0 = R5B.shared_state(b)
    assert s0 == R5B.shared_state(b)
    assert set(s0) == {"round3_fields", "round4_a1_fields", "bundle_post", "r3_bundle_HEADS", "identities"}
    assert set(s0["round3_fields"]) == set(RB4.R3_FIELDS) and set(s0["round4_a1_fields"]) == set(RB4.A1_FIELDS)
    assert b.post["affect"] is RB3._HEADS[_key_of(b)]["post"]
    b.post["csd"]["txt"][b.ctx.selection[0], 1] += np.float32(0.5)
    s1 = R5B.shared_state(b)
    assert s1["bundle_post"] != s0["bundle_post"] and s1["round3_fields"] == s0["round3_fields"]


def test_the_extension_holds_no_reference_into_post_q_and_leaves_post_as_it_was(ge_env):
    b, pl, ext = ge_env
    assert tuple(b.post) == A1 and b.post["affect"]["txt"] is not pl.Q
    assert not np.shares_memory(b.post["affect"]["txt"], pl.Q)
    assert set(vars(ext)) == {"kind", "placement", "placement_sha256", "seed", "smoke", "n", "stack", "F", "Bp", "pBp",
                              "Bp_picks", "checks", "from_cache"}


@pytest.mark.parametrize("what", ["bundle_post_itself", "shared_affect_dict", "other_order"])
def test_extend_refuses_a_post_q_that_is_not_a_new_a0_dict(monkeypatch, what):
    """extend's own post_Q assertion (review minor 3): each variant would otherwise run to the end unnoticed."""
    b = synth_bundle(seed=22)
    pl = R5G.clip_from_bundle(b)

    def mutant(post, Q):
        if what == "bundle_post_itself":
            return post
        if what == "shared_affect_dict":
            return {"affect": post["affect"], "image": post["image"], "caption": post["caption"]}
        return {"caption": post["caption"], "image": post["image"], "affect": {"img": post["affect"]["img"], "txt": Q}}

    monkeypatch.setattr(R5B, "post_q", mutant)
    with pytest.raises(AssertionError, match="new dict in A0 order"):
        R5B.extend(b, pl)


def test_extend_refuses_a_placement_with_another_row_count(tmp_path, monkeypatch):
    """_check_q's shape check (review minor 3): a GE file minted for more rows than the bundle has."""
    b = synth_bundle(seed=23)
    pl = make_ge(tmp_path, monkeypatch, b, seed=24, n_rows=int(b.post["affect"]["img"].shape[0]) + 5)
    release_guard(tmp_path)
    with pytest.raises(ValueError, match=re.escape("must be float32")):
        R5B.extend(b, pl)


# ---------------------------------------------------------------- features=False (rule §4 item 5, §6.4)

def test_extend_without_features_never_computes_F_Q(tmp_path, monkeypatch):
    b = synth_bundle(seed=25)
    pl = make_ge(tmp_path, monkeypatch, b, seed=26)
    release_guard(tmp_path)
    full = R5B.extend(b, pl)
    calls = []
    real = rbe.seed42_features
    monkeypatch.setattr(rbe, "seed42_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    nf = R5B.extend(b, pl, features=False)
    assert calls == [], "seed42_features was called on post_Q with features=False"
    assert nf.F is None and R5B.has_features(nf) is False and nf.checks["F_Q_computed"] is False
    assert R5B.has_features(full) is True and full.checks["F_Q_computed"] is True
    for d in DIRECTIONS:                                                     # stack_G and B'_G as with features
        np.testing.assert_array_equal(nf.stack[d], full.stack[d])
    for c in CONDITIONS:
        for d in DIRECTIONS:
            np.testing.assert_array_equal(nf.Bp[c][d], full.Bp[c][d])
    assert nf.Bp_picks == full.Bp_picks
    assert all(v is True for k, v in R5B.validate_ext(nf).items())
    assert set(R5B.check_pair(b, nf)) == {"stack_image_caption_slices_equal_bundle"}
    eq = R5B.exts_equal(nf, full)
    assert eq["has_F"] is False and eq["F__a"] is False and eq["stack__i2t"] is True
    for fn in (lambda: R5B.same_as_bundle(b, nf), lambda: R5B.positive_check(b, nf, pl)):
        with pytest.raises(ValueError, match="F_Q"):
            fn()
    with pytest.raises(ValueError, match="features"):
        R5B.extend(b, pl, features=1)


def test_ext_cache_round_trip_without_features(tmp_path, monkeypatch):
    b = synth_bundle(seed=27)
    pl = make_ge(tmp_path, monkeypatch, b, seed=28)
    release_guard(tmp_path)
    nf = R5B.extend(b, pl, features=False)
    r4 = _dummy_r4(R5.res_dir(True), "nofeatures")
    try:
        sha = R5B.save_ext(nf, r4["path"], r4["shas"])
        got = R5B.load_ext(r4["path"], r4["shas"], sha, pl, b)
        assert got.F is None and got.cache_meta["has_F"] is False
        eq = R5B.exts_equal(nf, got)
        assert all(eq.values()) and not any(k.startswith("F__") for k in eq), eq
        p = R5B.ext_path(r4["path"], "ge")
        with np.load(p) as z:                                                # a record that claims F it lacks
            arrays = {k: z[k] for k in z.files}
        assert not any(k.startswith("F__") for k in arrays)
        meta = json.loads(str(arrays["meta"][()]))
        meta["has_F"] = True
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(p, **arrays)
        with pytest.raises(SystemExit, match="has_F"):
            R5B.load_ext(r4["path"], r4["shas"], R5.sha256_file(p), pl, b)
    finally:
        _cleanup(r4)


def test_leak_pattern_catches_any_decimal_number():
    for s in ("bar 0.5", ".5", "x .5 y", "19.1", "-1.25", "5e-03", "1e-12", "2E+3", "[0.4, 0.9]", "margin 5.",
              "0.5%"):
        assert LEAK.search(s), s
    for s in ("[16:10:03] [r5_bundle] clip extension done: stack (192, 3, 13), F (192, 18)",
              "[r5_bundle] ge extension done: stack (192, 3, 13), F not computed", "CHECK item4 PASS",
              "dev_seed42.json written", "r5_bundle.extend"):
        assert not LEAK.search(s), s


# ---------------------------------------------------------------- D11: the guard

def test_extend_refuses_a_ge_placement_before_release_and_computes_nothing(tmp_path, monkeypatch):
    b = synth_bundle(seed=7)
    pl = make_ge(tmp_path, monkeypatch, b, seed=15)

    def boom(*a, **k):
        raise AssertionError("computed before the guard")

    for mod, name in ((C, "grouping_stack"), (rbe, "seed42_features"), (R5B, "crossfit_condition_free"),
                      (R5B, "uniform_probe_scores"), (R5B, "post_q")):
        monkeypatch.setattr(mod, name, boom)
    with pytest.raises(R5G.GuardError):
        R5B.extend(b, pl)
    with pytest.raises(R5G.GuardError):
        R5B.positive_check(b, SimpleNamespace(kind="ge", placement=pl, placement_sha256=pl.sha256), pl)


def test_every_function_refuses_a_ge_extension_once_the_guard_is_closed(tmp_path, monkeypatch, ge_env):
    b, pl, ext = ge_env
    r4 = _dummy_r4(tmp_path, "guard")
    sha = None
    monkeypatch.setattr(R5, "SMOKE", tmp_path)
    sha = R5B.save_ext(ext, r4["path"], r4["shas"])
    R5G._reset_for_tests()                                                   # the guard closes again
    calls = {"extend": lambda: R5B.extend(b, pl),
             "positive_check": lambda: R5B.positive_check(b, ext, pl),
             "same_as_bundle": lambda: R5B.same_as_bundle(b, ext),
             "validate_ext": lambda: R5B.validate_ext(ext),
             "check_pair": lambda: R5B.check_pair(b, ext),
             "require_ext": lambda: R5B.require_ext(ext, "test"),
             "exts_equal": lambda: R5B.exts_equal(ext, ext),
             "save_ext": lambda: R5B.save_ext(ext, r4["path"], r4["shas"]),
             "load_ext": lambda: R5B.load_ext(r4["path"], r4["shas"], sha, pl, b)}
    for name, call in calls.items():                                        # each function refuses by itself
        with pytest.raises(R5G.GuardError, match=re.escape("test" if name == "require_ext" else f"r5_bundle.{name}")):
            call()


def test_objects_that_are_not_minted_placements_are_refused():
    b = synth_bundle(seed=8)
    pl = R5G.clip_from_bundle(b)
    fake = SimpleNamespace(kind="clip", Q=pl.Q, sha256=pl.sha256)
    with pytest.raises(R5G.GuardError):
        R5B.extend(b, fake)
    ext = R5B.extend(b, pl)
    for bad in (SimpleNamespace(**{**vars(ext), "placement": fake}),
                SimpleNamespace(**{**vars(ext), "placement_sha256": "0" * 64}),
                SimpleNamespace(**{**vars(ext), "kind": "ge"})):
        with pytest.raises(R5G.GuardError):
            R5B.validate_ext(bad)
        with pytest.raises(R5G.GuardError):
            R5B.check_pair(b, bad)


# ---------------------------------------------------------------- refusals of the inputs

def test_extend_refuses_a_cache_or_a_pure_round3_bundle():
    b = synth_bundle(seed=9)
    pl = R5G.clip_from_bundle(b)
    nb = SimpleNamespace(**{**vars(b), "ctx": None})
    with pytest.raises(ValueError, match="cache"):
        R5B.extend(nb, pl)
    pure = SimpleNamespace(**{k: getattr(b, k) for k in RB4.R3_FIELDS})
    with pytest.raises(ValueError, match="round 4"):
        R5B.extend(pure, pl)
    a0_post = SimpleNamespace(**{**vars(b), "post": {h: b.post[h] for h in A0}})
    with pytest.raises(ValueError, match="groupings"):
        R5B.extend(a0_post, pl)


@pytest.mark.parametrize("what", ["missing_row", "shifted_row", "extra_row"])
def test_extend_refuses_a_placement_misaligned_with_the_selection_rows(tmp_path, monkeypatch, what):
    b = synth_bundle(seed=10)
    sel = np.asarray(b.ctx.selection)
    outside = np.setdiff1d(np.arange(b.post["affect"]["img"].shape[0]), sel)
    if what == "missing_row":
        rows = sel[1:]                                                       # a selection row without a posterior
    elif what == "shifted_row":
        rows = np.sort(np.concatenate([sel[1:], outside[:1]]))               # one row shifted outside the selection
    else:
        rows = np.sort(np.concatenate([sel, outside[:1]]))                   # a posterior on a non-selection row
    pl = make_ge(tmp_path, monkeypatch, b, seed=16, rows=rows)
    release_guard(tmp_path)
    with pytest.raises(ValueError, match="selection"):
        R5B.extend(b, pl)


@pytest.mark.parametrize("what", ["stack_float64", "stack_shape", "F_float32", "F_nonfinite", "delta", "Bp_not_cf",
                                  "Bp_b_scaled", "pBp_gain", "pBp_not_per_anchor", "Bp_dtype"])
def test_validate_ext_refusals(ge_env, what):
    b, pl, ext = ge_env
    e = SimpleNamespace(**vars(ext))
    e.stack = {d: np.array(ext.stack[d], copy=True) for d in DIRECTIONS}
    e.F = {c: np.array(ext.F[c], copy=True) for c in CONDITIONS}
    e.Bp = {c: {d: np.array(ext.Bp[c][d], copy=True) for d in DIRECTIONS} for c in CONDITIONS}
    e.pBp = {m: np.array(ext.pBp[m], copy=True) for m in METRICS}
    if what == "stack_float64":
        e.stack["i2t"] = e.stack["i2t"].astype(np.float64)
    elif what == "stack_shape":
        e.stack["t2i"] = e.stack["t2i"][:, :, :12]
    elif what == "F_float32":
        e.F["a"] = e.F["a"].astype(np.float32)
    elif what == "F_nonfinite":
        e.F["b"][0, 3] = np.nan
    elif what == "delta":
        e.F["b"][1, 2] += 1.0
    elif what == "Bp_not_cf":
        e.Bp["b"]["i2t"][0, 0] += np.float32(1.0)
    elif what == "Bp_b_scaled":                     # same ranking (so the same per-anchor arrays), not condition-free
        e.Bp["b"] = {d: (e.Bp["b"][d] * np.float32(2.0)).astype(np.float32) for d in DIRECTIONS}
        pa = per_anchor(e.Bp)
        assert all(np.array_equal(pa[m], ext.pBp[m]) for m in METRICS)   # precondition: only condition-freeness breaks
    elif what == "pBp_gain":
        e.pBp["gain"][0] = 0.25
    elif what == "pBp_not_per_anchor":
        e.pBp["r1"][0] = 1.0 - e.pBp["r1"][0]
    elif what == "Bp_dtype":
        e.Bp = {c: {d: e.Bp[c][d].astype(np.float64) for d in DIRECTIONS} for c in CONDITIONS}
    with pytest.raises(ValueError):
        R5B.validate_ext(e)


@pytest.mark.parametrize("what", ["seed", "smoke", "n", "stack_image", "stack_caption", "F_col7", "F_col17"])
def test_check_pair_refusals(ge_env, what):
    b, pl, ext = ge_env
    e = SimpleNamespace(**vars(ext))
    if what in ("seed", "smoke", "n"):
        setattr(e, what, {"seed": 9002, "smoke": False, "n": ext.n + 1}[what])
    elif what.startswith("stack"):
        s = np.array(ext.stack["t2i"], copy=True)
        s[0, 1 if what == "stack_image" else 2, 0] += np.float32(1.0)
        e.stack = {**ext.stack, "t2i": s}
    else:
        f = np.array(ext.F["a"], copy=True)
        f[0, 7 if what == "F_col7" else 17] += 1.0
        e.F = {**ext.F, "a": f}
    with pytest.raises(AssertionError):
        R5B.check_pair(b, e)
    assert R5B.check_pair(b, ext)


# ---------------------------------------------------------------- the per-seed cache (synthetic)

def _dummy_r4(folder, stem):
    """Two stand-in files for round 4's cache pair (only their SHA-256s bind the extension file)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"test_r5_bundle_{stem}.npz"
    p.write_bytes(b"round-3 cache stand-in " + stem.encode())
    RB4.a1_path(p).write_bytes(b"round-4 A1 cache stand-in " + stem.encode())
    return {"path": p, "shas": {"r3": R5.sha256_file(p), "a1": R5.sha256_file(RB4.a1_path(p))}}


def _cleanup(r4):
    for k in ("clip", "ge"):
        R5B.ext_path(r4["path"], k).unlink(missing_ok=True)
    r4["path"].unlink(missing_ok=True)
    RB4.a1_path(r4["path"]).unlink(missing_ok=True)


def test_ext_cache_round_trip_in_results_smoke(ge_env):
    b, pl, ext = ge_env
    r4 = _dummy_r4(R5.res_dir(True), "roundtrip")
    try:
        sha = R5B.save_ext(ext, r4["path"], r4["shas"])
        assert R5B.ext_path(r4["path"], "ge").parent.resolve() == R5.SMOKE.resolve()
        got = R5B.load_ext(r4["path"], r4["shas"], sha, pl, b)
        eq = R5B.exts_equal(ext, got)
        assert all(eq.values()), [k for k, v in eq.items() if not v]
        assert got.from_cache is True and got.placement is pl
        meta = got.cache_meta
        assert meta["r3_cache_sha256"] == r4["shas"]["r3"] and meta["a1_cache_sha256"] == r4["shas"]["a1"]
        assert meta["rule_sha256"] == R5.RULE_SHA and meta["placement_sha256"] == pl.sha256
        R5B.save_ext(ext, r4["path"], r4["shas"])                             # smoke files may be overwritten
    finally:
        _cleanup(r4)


@pytest.mark.parametrize("what", ["r3_changed", "a1_changed", "ext_sha", "other_seed_bundle", "wrong_r4_shas",
                                  "missing_a1", "ext_missing"])
def test_ext_cache_refusals(ge_env, what):
    b, pl, ext = ge_env
    r4 = _dummy_r4(R5.res_dir(True), f"refuse_{what}")
    try:
        sha = R5B.save_ext(ext, r4["path"], r4["shas"])
        shas, want = dict(r4["shas"]), sha
        bundle = b
        if what == "r3_changed":
            r4["path"].write_bytes(b"another round-3 file")
        elif what == "a1_changed":
            RB4.a1_path(r4["path"]).write_bytes(b"another A1 file")
        elif what == "ext_sha":
            want = "0" * 64
        elif what == "other_seed_bundle":
            bundle = SimpleNamespace(**{**vars(b), "seed": 9002})
        elif what == "wrong_r4_shas":
            shas = {"r3": shas["a1"], "a1": shas["r3"]}
        elif what == "missing_a1":
            RB4.a1_path(r4["path"]).unlink()
        elif what == "ext_missing":
            R5B.ext_path(r4["path"], "ge").unlink()
        with pytest.raises(SystemExit):
            R5B.load_ext(r4["path"], shas, want, pl, bundle)
    finally:
        _cleanup(r4)


def test_ext_cache_refuses_another_bundle_of_the_same_seed(ge_env):
    """The record's seed, smoke flag and n agree, but the bundle is another one: check_pair refuses it."""
    b, pl, ext = ge_env
    other = synth_bundle(seed=21)
    assert (other.seed, other.smoke, other.n) == (b.seed, b.smoke, b.n)
    r4 = _dummy_r4(R5.res_dir(True), "other_bundle")
    try:
        sha = R5B.save_ext(ext, r4["path"], r4["shas"])
        with pytest.raises(AssertionError, match="D5"):
            R5B.load_ext(r4["path"], r4["shas"], sha, pl, other)
    finally:
        _cleanup(r4)


def test_ext_cache_refuses_another_placement(tmp_path, monkeypatch, ge_env):
    b, pl, ext = ge_env
    r4 = _dummy_r4(R5.res_dir(True), "other_placement")
    try:
        sha = R5B.save_ext(ext, r4["path"], r4["shas"])
        other = make_ge(tmp_path, monkeypatch, b, seed=99, name="ge_other.npz")        # GE_POST_SHA now the other's
        with pytest.raises(SystemExit, match="another placement"):
            R5B.load_ext(r4["path"], r4["shas"], sha, other, b)
    finally:
        _cleanup(r4)


def test_ext_cache_binding_requires_round4_files_in_this_rounds_results(tmp_path, monkeypatch, ge_env):
    b, pl, ext = ge_env
    elsewhere = _dummy_r4(tmp_path / "elsewhere", "x")
    with pytest.raises(ValueError, match="live in"):
        R5B.save_ext(ext, elsewhere["path"], elsewhere["shas"])
    r4 = _dummy_r4(R5.res_dir(True), "bad_shas")
    try:
        with pytest.raises(SystemExit):
            R5B.save_ext(ext, r4["path"], {"r3": "0" * 64, "a1": r4["shas"]["a1"]})
        with pytest.raises(ValueError):
            R5B.save_ext(ext, r4["path"], {"r3": r4["shas"]["r3"]})
    finally:
        _cleanup(r4)


def test_a_non_smoke_ext_file_is_never_overwritten(tmp_path, monkeypatch):
    b = synth_bundle(seed=12, bundle_seed=42, smoke=False)
    pl = make_ge(tmp_path, monkeypatch, b, seed=17)
    release_guard(tmp_path)
    ext = R5B.extend(b, pl)
    res = tmp_path / "results"
    monkeypatch.setattr(R5, "RESULTS", res)
    r4 = _dummy_r4(res, "nonsmoke")
    sha = R5B.save_ext(ext, r4["path"], r4["shas"])
    before = R5B.ext_path(r4["path"], "ge").read_bytes()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        R5B.save_ext(ext, r4["path"], r4["shas"])
    assert R5B.ext_path(r4["path"], "ge").read_bytes() == before
    assert R5B.load_ext(r4["path"], r4["shas"], sha, pl, b).seed == 42
    smoke_dir = _dummy_r4(R5.res_dir(True), "nonsmoke_in_smoke")
    try:
        with pytest.raises(ValueError, match="live in"):                       # a non-smoke file never in results/smoke/
            R5B.save_ext(ext, smoke_dir["path"], smoke_dir["shas"])
    finally:
        _cleanup(smoke_dir)


# ---------------------------------------------------------------- the real builder on smoke seed 9001 (clip only)

@pytest.fixture(scope="module")
def smoke_build():
    eps = R3.AB / "results" / "smoke" / f"episodes_seed{SMOKE_SEED}.npz"
    if not eps.exists():
        pytest.fail(f"smoke seed {SMOKE_SEED} is not built: run run_baselines.py --smoke --episodes-seed {SMOKE_SEED}")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        b = RB4.build_bundle(SMOKE_SEED, True)
    return b


def test_smoke_clip_extension_is_the_bundles_own_and_prints_no_value(smoke_build):
    b = smoke_build
    pl = R5G.clip_from_bundle(b)
    assert pl.kind == "clip" and np.shares_memory(pl.Q, b.post["affect"]["txt"]), "Q_CLIP is a view of the bundle's"
    before = R5B.shared_state(b)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ext = R5B.extend(b, pl)
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    assert all("[r5_bundle]" in ln for ln in lines) and len(lines) >= 1, "extend printed a foreign line or none"
    assert not any(LEAK.search(ln) for ln in lines), "extend printed a decimal number"
    assert R5B.shared_state(b) == before, "round 3's or round 4's fields, bundle.post or _HEADS changed"
    assert b.post["affect"] is RB3._HEADS[60000]["post"], "bundle.post['affect'] is no longer the cached head dict"
    same = R5B.same_as_bundle(b, ext)
    assert same["all_pass"] is True, f"clip extension differs from the bundle in {[k for k, v in same.items() if not v]}"
    assert all(v is True for v in ext.checks.values()), "an extension check is not True"
    pc = R5B.positive_check(b, ext, pl)
    assert pc["affect_slice_equals_independent_einsum"] is True, "the einsum of D5 differs from round 3's stack"
    assert pc["placement_is_ge"] is False and pc["all_pass"] is False, "the positive check passed on Q_CLIP"
    assert ext.n == b.n and ext.seed == SMOKE_SEED and ext.smoke is True, "seed, smoke flag or n"


def test_smoke_clip_extension_cache_round_trip(smoke_build):
    b = smoke_build
    pl = R5G.clip_from_bundle(b)
    ext = R5B.extend(b, pl)
    r4 = _dummy_r4(R5.res_dir(True), f"smoke{SMOKE_SEED}")
    try:
        sha = R5B.save_ext(ext, r4["path"], r4["shas"])
        got = R5B.load_ext(r4["path"], r4["shas"], sha, pl, b)
        eq = R5B.exts_equal(ext, got)
        assert all(eq.values()), f"cache round trip differs: {[k for k, v in eq.items() if not v]}"
    finally:
        _cleanup(r4)
