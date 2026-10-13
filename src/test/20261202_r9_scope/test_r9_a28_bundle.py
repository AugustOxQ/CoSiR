"""Tests of r9_a28_context and r9_bundle.build_bundle_a28 (ticket 18; rule section 5 items 4 and 5, section 6 item 5).

Part 1, synthetic (a few thousand A28-shaped rows, 512-d, the real A3 checkpoint, stand-in heads fitted on a 2,000-row
draw): the NaN masks (a finite value outside the read rows fails each one; a non-finite value on a read row fails
it), a member row outside the read rows, the admitted seeds, the inputs-only mode (features, codes, posteriors,
masks; no label, episode or score). Mutation tests load a copy of r9_a28_context with one `# guard:<name>` statement
replaced by `pass` and show that the guard's scenario then goes through (never in place).

Part 2, the CLIP equivalence test (real ArtELingo features, the real A3 checkpoint, readers and pins; stand-in heads):
the selection rows of CLIP become an A28-shaped row space (position k = index in the selection rows), the seed-9001
episodes (64 per pair) are mapped into it, and the bundle and the frozen scores of the A28 path equal the ArtELingo
path's bit for bit.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_a28_bundle.py
"""
import ast
import dataclasses
import importlib.util
import inspect
import itertools
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_a28_context as AC  # noqa: E402
import r9_a28_episodes as AEp  # noqa: E402
import r9_bundle as B  # noqa: E402
import r9_context as X  # noqa: E402
import r9_episodes as E  # noqa: E402
import r9_heads as H  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_score as S  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src.eval.aspect_episodes as AE  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

D = 512
N_A28 = 6_000                 # synthetic A28 rows
N_FIT = 2_000
SEED = 9001                   # the synthetic runs: the smoke seed, admitted as ("a28", 9001) -> 64 in r9_context.ADMITTED
N_EP = R.N_SMOKE
A3 = R.TEST / B.A3_REL
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
    """Labels (-1 outside scorer-train), learnable from the (centred) image features, every class present."""
    X_ = img[scorer_train]
    X_ = (X_ - X_.mean(0)) / X_.std(0)
    W = rng.standard_normal((img.shape[1], k), dtype=np.float32)
    y = np.argmax(X_ @ W + rng.standard_normal((len(scorer_train), k), dtype=np.float32), axis=1)
    lab = np.full(len(img), -1, dtype=np.int64)
    lab[scorer_train] = y
    return lab


def stand_in_heads(data, scorer_train, seed=1):
    rng = np.random.default_rng(seed)
    labels = {h: _head_labels(rng, data.img_features, scorer_train, H.N_CLASSES[h]) for h in H.HEADS}
    heads = H.fit_all(data, scorer_train, labels, N_FIT)
    for head in heads.values():                       # the pinned heads carry their fit inputs; the stand-ins say CLIP
        head["fit_inputs"] = {"backbone": "clip_b32"}
    return heads


def ctx_args(s, **over):
    a = dict(a28=s.a28, eps=s.eps, seed=SEED, n_per_pair=N_EP, heads=s.heads)
    a.update(over)
    return a


# ---------------------------------------------------------------- synthetic A28-shaped data

LANGS = ("Hindi", "Thai", "arabic")


@pytest.fixture(scope="module")
def syn():
    rng = np.random.default_rng(20261213)
    n = N_A28
    pidx = np.arange(n, dtype=np.int64) // 3                  # three rows per painting, one per language
    npaint = int(pidx[-1]) + 1
    style_p, genre_p = rng.integers(0, 8, npaint), rng.integers(0, 8, npaint)
    labels = {"emotion": rng.integers(0, 8, n).astype(np.int64), "style": style_p[pidx].astype(np.int64),
              "genre": genre_p[pidx].astype(np.int64)}
    dropped = rng.random(n) < 0.05
    table = pd.DataFrame({"row_id": np.arange(n) * 2 + 5, "painting": [f"w{p:05d}" for p in pidx],
                          "language": [LANGS[i % 3] for i in range(n)], **labels,
                          "held": np.zeros(n, dtype=bool), "dropped": dropped})
    kept = ~dropped
    img = rng.standard_normal((n, D), dtype=np.float32)
    txt = rng.standard_normal((n, D), dtype=np.float32)
    img[dropped] = np.nan
    txt[dropped] = np.nan
    language = np.arange(n, dtype=np.int64) % 3 + 1           # l = 1 to 3 in LANGS' sort order (as load_a28's language)
    assert [LANGS[l - 1] for l in language[:3]] == sorted(LANGS)
    a28 = SimpleNamespace(backbone="clip_b32", n=n, img=img, txt=txt, kept=kept, dropped=dropped,
                          painting_index=pidx, language=language, read_set_sha256="0" * 64, **labels)
    train = np.flatnonzero(kept)
    raw = SimpleNamespace(img_features=np.nan_to_num(img), txt_features=np.nan_to_num(txt))   # stand-in ArtELingo
    heads = stand_in_heads(raw, train[:4_000])
    vs = {a: list(range(8)) for a in R.ASPECTS}
    ep_ctx = AEp.prepare(table, expected_languages=LANGS)
    assert np.array_equal(ep_ctx.cl, pidx)                    # the bootstrap code is the painting index
    eps = AEp.build_seed_a28(ep_ctx, vs, SEED, N_EP)
    return SimpleNamespace(a28=a28, heads=heads, eps=eps, labels=labels, vs=vs, rows=np.flatnonzero(kept), raw=raw)


@pytest.fixture(scope="module")
def ctx(syn):
    return AC.A28Context(**ctx_args(syn))


# ---------------------------------------------------------------- bookkeeping

def test_every_marked_guard_has_a_mutation_test():
    names = set(re.findall(r"# guard:([a-z_0-9]+)", (HERE / "r9_a28_context.py").read_text()))
    covered = set(re.findall(r'mutant\(tmp_path, "r9_a28_context\.py", "([a-z_0-9]+)"\)', Path(__file__).read_text()))
    assert names and names == covered, (names - covered, covered - names)


def test_admitted_seeds():
    """The a28 seeds live in the one shared table (r9_context.ADMITTED): the read seeds at 4,096, the smoke at 64."""
    a28 = {k: v for k, v in X.ADMITTED.items() if k[0] == "a28"}
    assert a28 == {("a28", 55): 4096, ("a28", 56): 4096, ("a28", 57): 4096,
                   ("a28", 9001): 64, ("a28", 9002): 64, ("a28", 9003): 64}
    assert AC.MODE == "a28" and not hasattr(AC, "A28_ADMITTED")
    assert "admitted" not in inspect.signature(AC.A28Context.__init__).parameters


def test_row_context_still_refuses_mode_a28():
    with pytest.raises(AssertionError, match="mode 'a28' is not one of"):
        X.RowContext("a28", 55, None, None, None, None, None, R.N_PER_PAIR)


def test_inputs_only_takes_no_label_episode_or_score():
    params = set(inspect.signature(AC.A28Inputs.__init__).parameters)
    assert params == {"self", "a28", "heads"}, params


# ---------------------------------------------------------------- the context

def test_context_rows_masks_and_fields(syn, ctx):
    a = syn.a28
    assert ctx.mode == "a28" and ctx.seed == SEED and ctx.n_per_pair == N_EP and ctx.backbone == "clip_b32"
    assert np.array_equal(ctx.rows, np.flatnonzero(a.kept)) and ctx.selection is ctx.rows
    assert ctx.in_rows.shape == (N_A28,) and np.array_equal(ctx.in_rows, a.kept)
    for m in ("img", "txt"):
        x = getattr(ctx, m)
        assert x.dtype == np.float32 and x.shape == (N_A28, D)
        assert np.isnan(x[~ctx.in_rows]).all() and np.isfinite(x[ctx.in_rows]).all()
        assert bits(x[ctx.rows], getattr(a, m)[ctx.rows])
    assert ctx.n == 3 * N_EP and ctx.pooled is ctx.eps.pooled
    assert ctx.in_rows[ctx.pooled.rows()].all()
    assert np.array_equal(ctx.pair_index, np.repeat(np.arange(3), N_EP))
    assert np.array_equal(ctx.parity, np.arange(ctx.n) % 2)
    assert np.array_equal(ctx.anchor, ctx.pooled.anchor)
    assert np.array_equal(ctx.anchor_group, a.painting_index[ctx.anchor])
    assert ctx.split.groups is a.painting_index or np.array_equal(ctx.split.groups, a.painting_index)
    assert ctx.episode_sha == dict(syn.eps.sha) and tuple(ctx.episode_sha) == R.PAIR_NAMES
    assert not ctx.smoke or ctx.seed in R.SMOKE_SEEDS
    want = cosine_scores(EvalInputs(ctx.img, ctx.txt), ctx.pooled)
    assert scores_bits(ctx.cos, want)


def test_context_posteriors_come_from_the_given_heads_on_read_rows(syn, ctx):
    want = H.predict(syn.heads, SimpleNamespace(img_features=syn.a28.img, txt_features=syn.a28.txt), ctx.rows)
    assert tuple(ctx.post) == H.HEADS
    for h in H.HEADS:
        for m in H.MODALITIES:
            p = ctx.post[h][m]
            assert p.dtype == np.float32 and p.shape == (N_A28, H.N_CLASSES[h])
            assert np.isnan(p[~ctx.in_rows]).all() and np.isfinite(p[ctx.in_rows]).all()
            assert bits(p, want[h][m])
    assert ctx.coef_sha256 == H.coef_sha256(syn.heads)


def test_context_codes_on_read_rows_only(syn, ctx):
    ic, tc = ctx.encode(A3)
    model, _ = load_factor_checkpoint(A3, device="cpu")
    wi, wt = encode_rows(model, syn.a28.img, syn.a28.txt, rows=ctx.rows, device="cpu")
    for full, want in ((ic, wi), (tc, wt)):
        assert full.dtype == np.float32 and full.shape == (N_A28, want.shape[1])
        assert np.isnan(full[~ctx.in_rows]).all() and np.isfinite(full[ctx.in_rows]).all()
        assert bits(full[ctx.rows], want)


# ---------------------------------------------------------------- inputs-only mode (rule section 6 item 5)

def test_inputs_only_has_features_codes_posteriors_and_masks_only(syn):
    inp = AC.A28Inputs(syn.a28, syn.heads)
    assert np.array_equal(inp.rows, np.flatnonzero(syn.a28.kept))
    assert np.isnan(inp.img[~inp.in_rows]).all() and np.isfinite(inp.img[inp.in_rows]).all()
    assert tuple(inp.post) == H.HEADS and inp.coef_sha256 == H.coef_sha256(syn.heads)
    ic, tc = inp.encode(A3)
    assert np.isnan(ic[~inp.in_rows]).all() and np.isfinite(ic[inp.in_rows]).all() and tc.shape == ic.shape
    eval_inputs = B.model_inputs(inp, "clip_b32")          # the bundle's own A3 path (pin checks) on the inputs
    assert eval_inputs.img_codes.shape[0] == N_A28 and np.isnan(eval_inputs.img_codes[~inp.in_rows]).all()
    for field in ("eps", "pooled", "cos", "labels", "anchor", "episode_sha", "n", "seed"):
        assert not hasattr(inp, field), field            # no label, episode or score


# ---------------------------------------------------------------- the guards

def test_seed_mode_guard(syn, tmp_path, monkeypatch):
    monkeypatch.setattr(AC.X, "check_masked", _reach)      # anything past the guard stops here
    for over in ({"seed": 42}, {"seed": 52}, {"seed": 56}, {"n_per_pair": 4096}, {"seed": 9004}, {"seed": 58}):
        with pytest.raises(AssertionError, match="admits"):
            AC.A28Context(**ctx_args(syn, **over))
    for s in (55, 56, 57):                                 # the read seeds: 4,096 per pair, not 64
        with pytest.raises(AssertionError, match="admits"):
            AC.A28Context(**ctx_args(syn, seed=s))
    for s in (9001, 9002, 9003):                           # the smoke seeds: 64 per pair, not 4,096
        with pytest.raises(AssertionError, match="admits"):
            AC.A28Context(**ctx_args(syn, seed=s, n_per_pair=4096))
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_seed_mode")
    monkeypatch.setattr(mut.X, "check_masked", _reach)
    with pytest.raises(Reached):
        mut.A28Context(**ctx_args(syn, seed=42))


def test_masked_guard_finite_outside_the_read_rows(syn, tmp_path, monkeypatch):
    """A finite feature on a dropped row, and a non-finite feature on a read row, each fail the feature mask."""
    monkeypatch.setattr(H, "predict", _reach)
    dropped_row = int(np.flatnonzero(syn.a28.dropped)[0])
    kept_row = int(syn.rows[10])
    for name, row in (("finite outside", dropped_row), ("nan inside", kept_row)):
        for m in ("img", "txt"):
            x = getattr(syn.a28, m).copy()
            x[row, 3] = 0.5 if name == "finite outside" else np.nan
            a = SimpleNamespace(**{**vars(syn.a28), m: x})
            with pytest.raises(AssertionError, match="A28 features"):
                AC.A28Context(**ctx_args(syn, a28=a))
            with pytest.raises(AssertionError, match="A28 features"):
                AC.A28Inputs(a, syn.heads)
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_masked")
    monkeypatch.setattr(mut.H, "predict", _reach)
    x = syn.a28.img.copy()
    x[dropped_row, 3] = 0.5
    with pytest.raises(Reached):
        mut.A28Inputs(SimpleNamespace(**{**vars(syn.a28), "img": x}), syn.heads)


def test_post_mask_guard(syn, ctx, tmp_path, monkeypatch):
    post = {h: dict(ctx.post[h]) for h in ctx.post}
    leak = post["csd"]["img"].copy()
    leak[int(np.flatnonzero(syn.a28.dropped)[0])] = 1.0 / H.N_CLASSES["csd"]   # finite outside the read rows
    post["csd"] = {"img": leak, "txt": post["csd"]["txt"]}
    monkeypatch.setattr(H, "predict", lambda *a, **kw: post)
    with pytest.raises(AssertionError, match="post csd/img"):
        AC.A28Inputs(syn.a28, syn.heads)
    bad = {h: dict(ctx.post[h]) for h in ctx.post}
    nan_in = bad["image"]["txt"].copy()
    nan_in[syn.rows[5]] = np.nan                                                   # non-finite on a read row
    bad["image"] = {"img": bad["image"]["img"], "txt": nan_in}
    monkeypatch.setattr(H, "predict", lambda *a, **kw: bad)
    with pytest.raises(AssertionError, match="post image/txt"):
        AC.A28Inputs(syn.a28, syn.heads)
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_post_mask")
    monkeypatch.setattr(mut.H, "predict", lambda *a, **kw: post)
    inp = mut.A28Inputs(syn.a28, syn.heads)
    assert np.isfinite(inp.post["csd"]["img"][int(np.flatnonzero(syn.a28.dropped)[0])]).all()


def test_codes_mask_guard(syn, ctx, tmp_path, monkeypatch):
    def nan_inside(model, img, txt, rows=None, **kw):
        ic, tc = encode_rows(model, img, txt, rows=rows, **kw)
        ic[3, 0] = np.nan                                  # a non-finite code on a read row
        return ic, tc

    def one_extra(model, img, txt, rows=None, **kw):
        ic, tc = encode_rows(model, img, txt, rows=rows, **kw)
        return ic[:-1], tc[:-1]                            # fewer codes than read rows

    monkeypatch.setattr(AC, "encode_rows", nan_inside)
    with pytest.raises(AssertionError, match="codes"):
        ctx.encode(A3)
    monkeypatch.setattr(AC, "encode_rows", one_extra)
    with pytest.raises(AssertionError, match="codes"):
        ctx.encode(A3)
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_codes_mask")
    monkeypatch.setattr(mut, "encode_rows", nan_inside)
    shim = SimpleNamespace(rows=ctx.rows, in_rows=ctx.in_rows, data=ctx.data, n_rows=ctx.n_rows)
    ic, _ = mut.A28Inputs.encode(shim, A3)
    assert np.isnan(ic[ctx.rows[3], 0])


def _with_member_outside(eps, outside_row):
    """The episodes with one candidate of the first episode replaced by ``outside_row``."""
    cand = eps.pooled.candidates.copy()
    cand[0, 5] = outside_row
    pooled = dataclasses.replace(eps.pooled, candidates=cand)
    return SimpleNamespace(**{**vars(eps), "pooled": pooled})


def test_member_row_outside_the_read_rows_fails(syn, tmp_path, monkeypatch):
    dropped_row = int(np.flatnonzero(syn.a28.dropped)[0])
    tampered = _with_member_outside(syn.eps, dropped_row)
    with pytest.raises(AssertionError, match="1 episode member rows lie outside the read rows"):
        AC.A28Context(**ctx_args(syn, eps=tampered))
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_episode_rows")
    monkeypatch.setattr(mut, "cosine_scores", _reach)      # the next step after the guard
    with pytest.raises(Reached):
        mut.A28Context(**ctx_args(syn, eps=tampered))


def _shifted(eps, rows):
    """The episodes with every member row moved to the next read row (a row space that differs from the features')."""
    nxt = np.zeros(int(rows.max()) + 1, dtype=np.int64)
    nxt[rows] = np.roll(rows, -1)
    per_pair = [AE.AspectEpisodes(ep.aspect_a, ep.aspect_b, *(nxt[getattr(ep, f)] for f in E.FIELDS))
                for ep in eps.per_pair]
    return SimpleNamespace(**{**vars(eps), "per_pair": per_pair, "pooled": AE.concat_episodes(per_pair)})


def test_synthetic_episodes_are_language_homogeneous(syn):
    lang = np.concatenate([syn.eps.language[p] for p in R.PAIR_NAMES])
    for f in E.FIELDS:
        members = getattr(syn.eps.pooled, f).reshape(len(lang), -1)
        assert (syn.a28.language[members] == lang[:, None]).all(), f
    assert len(set(lang.tolist())) == 3                    # all three languages occur


def test_episodes_in_another_row_space_fail_the_language_join(syn, tmp_path, monkeypatch):
    """Episodes moved to the next read row are still read rows, but many members then have another language."""
    moved = _shifted(syn.eps, syn.rows)
    assert syn.a28.kept[moved.pooled.rows()].all()          # the member-row guard alone cannot see this
    with pytest.raises(AssertionError, match="language"):
        AC.A28Context(**ctx_args(syn, eps=moved))
    single = _other_language_member(syn)                    # one example row of one episode in another language
    with pytest.raises(AssertionError, match="language"):
        AC.A28Context(**ctx_args(syn, eps=single))
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_language_join")
    monkeypatch.setattr(mut, "cosine_scores", _reach)      # the next step after the guard
    for bad in (moved, single):
        with pytest.raises(Reached):
            mut.A28Context(**ctx_args(syn, eps=bad))


def _other_language_member(syn):
    """The episodes with the first image row of example pair 0 (aspect a) of episode 0 replaced by a read row of
    another language (still a read row, so the member-row guard passes)."""
    lang0 = int(syn.eps.language[R.PAIR_NAMES[0]][0])
    other = int(syn.rows[syn.a28.language[syn.rows] != lang0][0])
    pa = syn.eps.pooled.pairs_a_img.copy()
    pa[0, 0] = other
    return SimpleNamespace(**{**vars(syn.eps), "pooled": dataclasses.replace(syn.eps.pooled, pairs_a_img=pa)})


def test_inputs_and_episodes_without_a_language_are_refused(syn):
    no_col = SimpleNamespace(**{k: v for k, v in vars(syn.a28).items() if k != "language"})
    with pytest.raises(AssertionError, match="language"):
        AC.A28Inputs(no_col, syn.heads)
    no_eps = SimpleNamespace(**{k: v for k, v in vars(syn.eps).items() if k != "language"})
    with pytest.raises(AssertionError, match="language"):
        AC.A28Context(**ctx_args(syn, eps=no_eps))


def test_bundle_check_context_reads_the_shared_admitted_table(ctx):
    """check_context looks (mode, seed) up in r9_context.ADMITTED: an a28 seed outside it, or the wrong size, fails."""
    B.check_context(ctx)
    for over in ({"seed": 9004}, {"seed": 55}, {"seed": 58}):
        with pytest.raises(AssertionError, match="admitted episodes per pair"):
            B.check_context(SimpleNamespace(**{**vars(ctx), **over}))


def test_bundle_check_context_fails_on_a_member_outside_the_read_rows(ctx):
    """The bundle's own check (guard bundle_rows of r9_bundle) on an A28Context whose episodes were tampered with
    after construction."""
    B.check_context(ctx)                                    # the clean context passes the generalised check
    bad = SimpleNamespace(**vars(ctx))
    bad.pooled = _with_member_outside(ctx.eps, int(np.flatnonzero(~ctx.in_rows)[0])).pooled
    bad.eps = SimpleNamespace(**{**vars(ctx.eps), "pooled": bad.pooled})
    with pytest.raises(AssertionError, match="outside"):
        B.check_context(bad)


def test_episode_shape_guard(syn, tmp_path, monkeypatch):
    monkeypatch.setattr(AC, "cosine_scores", _reach)
    short = SimpleNamespace(**{**vars(syn.eps), "parity": np.zeros(syn.eps.n, dtype=np.int64)})
    badseed = SimpleNamespace(**{**vars(syn.eps), "seed": SEED + 1})
    badsha = SimpleNamespace(**{**vars(syn.eps), "sha": dict(list(syn.eps.sha.items())[:2])})
    for bad in (short, badseed, badsha):
        with pytest.raises(AssertionError, match="episodes"):
            AC.A28Context(**ctx_args(syn, eps=bad))
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_eps_shape")
    monkeypatch.setattr(mut, "cosine_scores", _reach)
    for bad in (short, badseed, badsha):
        with pytest.raises(Reached):
            mut.A28Context(**ctx_args(syn, eps=bad))


def test_heads_dim_guard(syn, tmp_path, monkeypatch):
    small = SimpleNamespace(**{**vars(syn.a28), "img": syn.a28.img[:, :256], "txt": syn.a28.txt[:, :256]})
    monkeypatch.setattr(H, "predict", _reach)
    with pytest.raises(AssertionError, match="heads"):
        AC.A28Inputs(small, syn.heads)
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_heads_dim")
    monkeypatch.setattr(mut.H, "predict", _reach)
    with pytest.raises(Reached):
        mut.A28Inputs(small, syn.heads)


def test_heads_fitted_on_another_backbone_or_without_fit_inputs_are_refused(syn, tmp_path, monkeypatch):
    other = {h: {**v, "fit_inputs": {"backbone": "qwen3vl_emb_2b"}} for h, v in syn.heads.items()}
    bare = {h: {k: x for k, x in v.items() if k != "fit_inputs"} for h, v in syn.heads.items()}   # as H.fit_all makes
    assert all("fit_inputs" in v for v in syn.heads.values())
    for bad in (other, bare):
        with pytest.raises(AssertionError, match="heads"):
            AC.A28Inputs(syn.a28, bad)
    mut = mutant(tmp_path, "r9_a28_context.py", "a28_heads_backbone")
    monkeypatch.setattr(mut.H, "predict", _reach)
    for bad in (other, bare):
        with pytest.raises(Reached):
            mut.A28Inputs(syn.a28, bad)


def test_module_does_not_fit_anything():
    """Nothing is fitted on A28 rows: the module never calls a fit function or builds labels/episodes in inputs-only."""
    tree = ast.parse((HERE / "r9_a28_context.py").read_text())
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"fit", "fit_all", "fit_heads_r9", "fit_pm", "fit_pca_basis", "build_seed", "build_seed_a28",
                         "build_aspect_episodes_r6"}, called


# ---------------------------------------------------------------- the bundle on the synthetic context

def test_build_bundle_a28_refuses_another_mode_or_backbone(ctx):
    other = SimpleNamespace(**{**vars(ctx), "mode": "held"})
    with pytest.raises(AssertionError, match="a28"):
        B.build_bundle_a28(other, None, None, "clip_b32")
    with pytest.raises(AssertionError, match="a28"):
        B.build_bundle_a28(ctx, None, None, "qwen3vl_emb_2b")


# ---------------------------------------------------------------- CLIP equivalence (real ArtELingo rows)

@pytest.fixture(scope="module")
def equiv():
    """The ArtELingo path (RowContext, selection, smoke seed 9001, 64 per pair) and the A28 path on the same rows."""
    rows = R.load_rows("clip_b32")
    sel = rows.split.selection
    heads = stand_in_heads(rows.data, rows.split.scorer_train, seed=7)
    ref_ctx = X.RowContext("selection", SEED, rows.data, rows.split, rows.labels, heads, rows.value_sets, N_EP)
    pm = B.fit_pm(rows.data, rows.split.scorer_train, "clip_b32")
    readers = B.load_readers("clip_b32")
    ref = B.build_bundle_r6(ref_ctx, readers, pm, "clip_b32")

    # the A28-shaped row space: position k = index of the row in the selection rows. The rows that no episode uses are
    # dropped (NaN features, not kept), as most of the real read set's dropped rows are: the A28 path must be NaN-safe
    groups_sel = rows.split.groups[sel]
    painting_index = np.unique(groups_sel, return_inverse=True)[1].astype(np.int64)

    def to_a28(ep):
        return AE.AspectEpisodes(ep.aspect_a, ep.aspect_b,
                                 *(np.searchsorted(sel, getattr(ep, f)) for f in E.FIELDS))

    mapped = [to_a28(ep) for ep in ref_ctx.eps.per_pair]
    kept = np.zeros(len(sel), dtype=bool)
    kept[AE.concat_episodes(mapped).rows()] = True
    assert 0 < kept.sum() < 0.5 * len(sel)                    # most positions are dropped, some are read rows
    img, txt = rows.data.img_features[sel].copy(), rows.data.txt_features[sel].copy()
    img[~kept] = np.nan
    txt[~kept] = np.nan
    a28 = SimpleNamespace(
        backbone="clip_b32", n=len(sel), img=img, txt=txt, kept=kept, dropped=~kept, painting_index=painting_index,
        language=np.ones(len(sel), dtype=np.int64), read_set_sha256="a1" * 32)
    for old, new in zip(ref_ctx.eps.per_pair, mapped):
        assert np.array_equal(sel[new.anchor], old.anchor) and np.array_equal(sel[new.candidates], old.candidates)
    eps = SimpleNamespace(seed=SEED, per_pair=mapped, pooled=AE.concat_episodes(mapped),
                          sha={p: AE.episodes_sha256(ep) for p, ep in zip(R.PAIR_NAMES, mapped)}, n=3 * N_EP,
                          pair_index=ref_ctx.pair_index.copy(), parity=ref_ctx.parity.copy(),
                          language={p: np.ones(N_EP, dtype=np.int64) for p in R.PAIR_NAMES},
                          part_sha={p: {"L1": f"{i:02x}" * 32} for i, p in enumerate(R.PAIR_NAMES)},
                          shares={p: {"L1": N_EP} for p in R.PAIR_NAMES},
                          set_aside={p: [] for p in R.PAIR_NAMES},
                          set_aside_reasons={p: {} for p in R.PAIR_NAMES})
    a28_ctx = AC.A28Context(a28, eps, SEED, N_EP, heads)
    out = B.build_bundle_a28(a28_ctx, readers, pm, "clip_b32")
    return SimpleNamespace(rows=rows, sel=sel, ref_ctx=ref_ctx, ref=ref, a28_ctx=a28_ctx, out=out, heads=heads,
                           readers=readers, painting_index=painting_index)


def test_equivalence_bundle_fields(equiv):
    ref, out, sel = equiv.ref, equiv.out, equiv.sel
    assert out.n == ref.n and out.mode == "a28" and out.backbone == "clip_b32"
    for k in ("parity", "pair_index"):
        assert bits(getattr(out, k), getattr(ref, k)), k
    assert np.array_equal(sel[out.anchor], ref.anchor)
    # cl: the A28 code is the inverse index of the sorted leakage groups of the selection rows; ref.cl are the groups
    assert np.array_equal(np.unique(equiv.rows.split.groups[sel])[out.cl], ref.cl)
    for k in ("cos", "t_n1u", "t6u_B", "t6u_B0", "t6u_B1"):
        assert scores_bits(getattr(out, k), getattr(ref, k)), k
    kept = equiv.a28_ctx.in_rows
    assert out.n == 3 * N_EP and 0 < kept.sum() < len(sel)
    for h in ref.post:
        for m in H.MODALITIES:
            assert np.isnan(out.post[h][m][~kept]).all() and np.isfinite(out.post[h][m][kept]).all(), (h, m)
            assert bits(out.post[h][m][kept], ref.post[h][m][sel][kept]), (h, m)
    for d in DIRECTIONS:
        assert bits(out.stack[d], ref.stack[d]), d
    for c in CONDITIONS:
        assert bits(out.F[c], ref.F[c]), c
    for name in B.PM_TERM_NAMES:
        assert scores_bits(out.pm_terms[name], ref.pm_terms[name]), name
    assert out.coef_sha256 == ref.coef_sha256 and out.fit_rows_sha256 == ref.fit_rows_sha256


def test_equivalence_records_added_by_build_bundle_a28(equiv):
    """The records of ticket 17 and the read set that the A28 bundle adds to build_bundle_r6's (rule section 8 item 5:
    the per-language report and held_started_H8.json read them from the bundle)."""
    out, eps = equiv.out, equiv.a28_ctx.eps
    assert out.read_set_sha256 == "a1" * 32 == equiv.a28_ctx.a28.read_set_sha256
    assert out.language is eps.language and set(out.language) == set(R.PAIR_NAMES)
    assert all((out.language[p] == 1).all() and len(out.language[p]) == N_EP for p in R.PAIR_NAMES)
    for name in ("part_sha", "shares", "set_aside", "set_aside_reasons"):
        assert getattr(out, name) is getattr(eps, name), name
    assert out.part_sha[R.PAIR_NAMES[1]] == {"L1": "01" * 32} and out.shares[R.PAIR_NAMES[2]] == {"L1": N_EP}
    assert out.set_aside == {p: [] for p in R.PAIR_NAMES}


def test_equivalence_a_real_a28_bundle_passes_score_seed_admitted_guard(equiv):
    """r9_context.ADMITTED holds the a28 seeds, so the scorer's seed/size guard accepts an a28 bundle as it is."""
    out = equiv.out
    assert out.mode == "a28" and out.smoke
    assert len(R.PAIRS) * X.ADMITTED.get((out.mode, int(out.seed)), -1) == out.n         # guard:admitted's test
    picks = P.load_picks(backbone="clip_b32")
    got = S.score_seed(out, picks, P.frozen_lambdas("clip_b32"), equiv.readers, include_pm=False)
    assert got["pair_index"].shape == (out.n,)


def test_equivalence_frozen_scores(equiv):
    """The ArtELingo path's scoring code on the A28 bundle (relabelled to the admitted smoke seed, as the scoring
    code admits only round-6 seeds) gives the same per-anchor results as on the ArtELingo bundle."""
    picks = P.load_picks(backbone="clip_b32")
    lambdas = P.frozen_lambdas("clip_b32")
    ref = S.score_seed(equiv.ref, picks, lambdas, equiv.readers, include_pm=True)
    relabel = SimpleNamespace(**{**vars(equiv.out), "mode": "selection", "seed": SEED})
    got = S.score_seed(relabel, picks, lambdas, equiv.readers, include_pm=True)
    assert list(got) == list(ref)
    for k, v in ref.items():
        if k in ("cl",):
            assert np.array_equal(np.unique(equiv.rows.split.groups[equiv.sel])[got[k]], v)
        elif k in ("pair_index",):
            assert bits(got[k], v)
        elif k in ("gates", "reader"):
            flat = lambda o: o if isinstance(o, np.ndarray) else {kk: flat(vv) for kk, vv in o.items()}
            assert _same(flat(got[k]), flat(v)), k
        else:
            assert _same(got[k], v), k


def _same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    return bits(a, b)
