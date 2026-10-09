"""Tests of r6_episodes (ticket 02): the value-restricted episode builder, the per-seed builder, the identity check,
the hash guards and the episode file.

Synthetic data uses the real shapes: 308,723 rows in paintings of four rows, a held-like band of 61,744 rows, 8
emotions, 23 styles and 10 genres (plus a New_Realism-like 24th style where a test needs one), 4,096 episodes per pair
(12,288 per seed), 13 candidates, 4 example pairs a side. The real-data test reads selection rows only (and the
split's index arrays); no held feature, label or episode is built or read. Mutation tests load a copy of r6_episodes
from tmp_path with one `# guard:<name>` statement replaced by `pass`, and show that the guard's scenario then passes.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_episodes.py
"""
import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import itertools
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first: it puts MAIN's src in front and checks it)
import r6_episodes as E  # noqa: E402

import src.eval.aspect_episodes as AE  # noqa: E402

HERE_LINE = "HERE = Path(__file__).resolve().parent\n"
_COUNT = itertools.count()
NR = 23                       # the New_Realism-like style: eligible on the synthetic pool, outside the value set
N = R.N_PER_PAIR              # 4,096 per pair, the real count
SEED = 52


# ---------------------------------------------------------------- mutation helpers

def mutant_source(guard=None, replace=()) -> str:
    """r6_episodes' source with HERE pinned and the `# guard:<guard>` statements replaced by `pass`."""
    src = (HERE / "r6_episodes.py").read_text()
    assert src.count(HERE_LINE) == 1
    lines = src.splitlines(keepends=True)
    if guard is not None:
        hits = [n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Expr) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
        assert hits, f"no statement carries # guard:{guard}"
        for n in hits:
            indent = lines[n.lineno - 1][:len(lines[n.lineno - 1]) - len(lines[n.lineno - 1].lstrip())]
            lines[n.lineno - 1] = f"{indent}pass\n"
            for i in range(n.lineno, n.end_lineno):
                lines[i] = "\n"
    src = "".join(lines).replace(HERE_LINE, f"HERE = Path({str(E.HERE)!r})\n")
    for old, new in replace:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def load_copy(tmp_path, **kw):
    """Import a (possibly mutated) copy of r6_episodes from tmp_path under a fresh name (sys.path restored after)."""
    path = tmp_path / f"r6_episodes_copy{next(_COUNT)}.py"
    path.write_text(mutant_source(**kw))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(path.stem, None)
        sys.path[:] = saved
    return mod


def raised_message(fn, *a, **kw):
    """The message of the exception ``fn`` raises, or None."""
    try:
        fn(*a, **kw)
    except BaseException as e:   # SystemExit included
        return str(e)
    return None


def test_every_marked_guard_has_a_mutation_test():
    marked = set(re.findall(r"# guard:(\w+)", (HERE / "r6_episodes.py").read_text()))
    tested = set(re.findall(r"guard=\"(\w+)\"", Path(__file__).read_text()))
    assert marked == tested, (marked - tested, tested - marked)


# ---------------------------------------------------------------- synthetic data of the real shapes

def synthetic(nr_groups=40, seed=0):
    """Labels over 308,723 rows in paintings (groups) of four rows. Style and genre are per painting, emotion per row
    (as in ArtELingo); about 5% of rows lack an emotion and 3% of paintings a genre. The pool rows are a held-like
    band of 15,436 whole paintings (61,744 rows); the first ``nr_groups`` paintings of the band get style 23."""
    n = R.N_ROWS
    rng = np.random.default_rng(seed)
    groups = np.arange(n, dtype=np.int64) // 4
    ng = int(groups[-1]) + 1
    style_g = rng.integers(0, 23, ng)
    genre_g = rng.integers(0, 10, ng)
    genre_g[rng.random(ng) < 0.03] = -1
    emotion = rng.integers(0, 8, n)
    emotion[rng.random(n) < 0.05] = -1
    rows = np.arange(160_000, 160_000 + 61_744, dtype=np.int64)
    band = np.unique(groups[rows])
    style_g[band[:nr_groups]] = NR
    labels = {"emotion": emotion.astype(np.int64), "style": style_g[groups].astype(np.int64),
              "genre": genre_g[groups].astype(np.int64)}
    index = AE.PaintingValueIndex(labels, groups)
    dev = {"emotion": list(range(8)), "style": list(range(23)), "genre": list(range(10))}
    return dict(labels=labels, groups=groups, rows=rows, index=index, dev=dev)


def pool_of(s, rows=None):
    lab, rows = s["labels"], s["rows"] if rows is None else rows
    return rows[(lab["emotion"][rows] >= 0) & (lab["style"][rows] >= 0) & (lab["genre"][rows] >= 0)]


@pytest.fixture(scope="module")
def syn():
    s = synthetic()
    pool = pool_of(s)
    s["ok"] = {x: AE.eligible_values(s["labels"][x], s["groups"], pool, 30) for x in R.ASPECTS}
    assert s["ok"] == {"emotion": list(range(8)), "style": list(range(24)), "genre": list(range(10))}
    return s


@pytest.fixture(scope="module")
def orig(syn):
    """The original builder on the synthetic pool, each pair at 4,096 episodes."""
    return {f"{a}__{b}": AE.build_aspect_episodes(syn["labels"], syn["groups"], syn["rows"], a, b, N, SEED, third=t,
                                                  index=syn["index"]) for a, b, t in R.PAIRS}


@pytest.fixture(scope="module")
def dev_eps(syn):
    """build_seed with the development-like value sets (style 23 left out): 12,288 episodes."""
    return E.build_seed(syn["labels"], syn["groups"], syn["rows"], syn["index"], syn["dev"], SEED, N)


def same(x, y):
    return (x.aspect_a, x.aspect_b) == (y.aspect_a, y.aspect_b) and all(
        getattr(x, f).dtype == getattr(y, f).dtype == np.int64 and np.array_equal(getattr(x, f), getattr(y, f))
        for f in E.FIELDS)


# ---------------------------------------------------------------- the copy is the original plus one block

EXPECTED_BLOCK = (
    "if values is not None:\n"
    "    v_a, v_b = (_codes(v) for v in values)\n"
    "    _require(v_a <= set(ok_a), f'{aspect_a} x {aspect_b}: values {sorted(v_a - set(ok_a))} of {aspect_a} are not "
    "eligible on the pool')\n"
    "    _require(v_b <= set(ok_b), f'{aspect_a} x {aspect_b}: values {sorted(v_b - set(ok_b))} of {aspect_b} are not "
    "eligible on the pool')\n"
    "    ok_a = sorted(set(ok_a) & v_a)\n"
    "    ok_b = sorted(set(ok_b) & v_b)")


def test_copy_differs_from_the_original_only_by_the_values_block():
    orig_fn = ast.parse(textwrap.dedent(inspect.getsource(AE.build_aspect_episodes))).body[0]
    copy_fn = ast.parse(textwrap.dedent(inspect.getsource(E.build_aspect_episodes_r6))).body[0]
    body = list(copy_fn.body)
    assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)       # the docstring
    body = body[1:]
    blocks = [i for i, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == "values is not None"]
    assert len(blocks) == 1
    i = blocks[0]
    assert ast.unparse(body[i]) == EXPECTED_BLOCK
    assert ast.unparse(body[i - 1]) == "ok_b = eligible_values(lb, groups, pool, min_paintings)"
    assert ast.unparse(body[i + 1]).startswith("if len(ok_a) <= NUM_PAIRS or len(ok_b) <= NUM_PAIRS:")
    assert [ast.dump(s) for s in body[:i] + body[i + 1:]] == [ast.dump(s) for s in orig_fn.body]
    oa, ca = orig_fn.args, copy_fn.args
    assert [ast.dump(a) for a in ca.args[:-1]] == [ast.dump(a) for a in oa.args] and ca.args[-1].arg == "values"
    assert [ast.dump(d) for d in ca.defaults[:-1]] == [ast.dump(d) for d in oa.defaults]
    assert isinstance(ca.defaults[-1], ast.Constant) and ca.defaults[-1].value is None
    assert ast.dump(copy_fn.returns) == ast.dump(orig_fn.returns)
    # the names the body uses are the original module's objects
    for name in ("draw_distinct", "_pairs", "eligible_values", "PaintingValueIndex", "AspectEpisodes", "NUM_PAIRS",
                 "NUM_NEGATIVES"):
        assert getattr(E, name) is getattr(AE, name), name


def test_values_none_and_values_ok_give_the_originals_episodes(syn, orig):
    """values=None, and values equal to the eligible lists, reproduce the original on the real shape."""
    for a, b, t in R.PAIRS:
        o = orig[f"{a}__{b}"]
        assert o.anchor.shape == (N,) and o.candidates.shape == (N, 13) and o.pairs_a_img.shape == (N, 4)
        got = E.build_aspect_episodes_r6(syn["labels"], syn["groups"], syn["rows"], a, b, N, SEED, third=t,
                                         index=syn["index"])
        assert same(got, o) and AE.episodes_sha256(got) == AE.episodes_sha256(o)
        got = E.build_aspect_episodes_r6(syn["labels"], syn["groups"], syn["rows"], a, b, N, SEED, third=t,
                                         index=syn["index"], values=(syn["ok"][a], syn["ok"][b]))
        assert same(got, o)


# ---------------------------------------------------------------- the restriction (rule section 5 items 2 and 3)

def test_value_outside_the_set_is_never_anchor_or_shared_value_but_stays_in_the_pool(syn, orig, dev_eps):
    st = syn["labels"]["style"]
    assert NR in syn["ok"]["style"] and NR not in syn["dev"]["style"]          # eligible on the pool, outside V
    assert dev_eps.n == 3 * N and tuple(dev_eps.sha) == R.PAIR_NAMES
    by = dict(zip(R.PAIR_NAMES, dev_eps.per_pair))
    for name, own in (("emotion__style", "b"), ("style__genre", "a")):         # pairs where style is restricted
        ep, o = by[name], orig[name]
        third = {f"{a}__{b}": t for a, b, t in R.PAIRS}[name]
        AE.validate_aspect_episodes(ep, syn["labels"], syn["groups"], syn["index"], third=third)
        shared = np.concatenate([getattr(ep, f"pairs_{own}_img"), getattr(ep, f"pairs_{own}_txt")], axis=1)
        other = "a" if own == "b" else "b"
        others = np.concatenate([getattr(ep, f"pairs_{other}_img"), getattr(ep, f"pairs_{other}_txt")], axis=1)
        # never the anchor's style, never the shared value of a style example pair
        assert (st[ep.anchor] != NR).all()
        assert (st[shared] != NR).all()
        # its rows stay in the pool: negatives, and items of the other aspect's example pairs
        assert (st[ep.candidates[:, 2:]] == NR).sum() > 0
        assert (st[others] == NR).sum() > 0
        # without the restriction the same build does use it as an anchor value and as a shared value
        o_shared = np.concatenate([getattr(o, f"pairs_{own}_img"), getattr(o, f"pairs_{own}_txt")], axis=1)
        assert (st[o.anchor] == NR).sum() > 0 and (st[o_shared] == NR).sum() > 0
        assert dev_eps.sha[name] != AE.episodes_sha256(o)
    # emotion x genre: style is the third aspect, untouched; its anchors may carry style 23 (as in development)
    eg = by["emotion__genre"]
    assert same(eg, orig["emotion__genre"]) and (st[eg.anchor] == NR).sum() > 0


def test_build_seed_fields(syn, dev_eps):
    eps = dev_eps
    assert eps.seed == SEED and eps.n == 12_288 and len(eps.per_pair) == 3
    assert [f"{e.aspect_a}__{e.aspect_b}" for e in eps.per_pair] == list(R.PAIR_NAMES)
    assert eps.pair_index.dtype == np.int64 and np.array_equal(eps.pair_index, np.repeat(np.arange(3), N))
    assert eps.parity.dtype == np.int64 and np.array_equal(eps.parity, np.arange(3 * N) % 2)
    assert same(eps.pooled, AE.concat_episodes(eps.per_pair)) and eps.pooled.aspect_a == "mixed"
    assert all(eps.sha[p] == AE.episodes_sha256(e) for p, e in zip(R.PAIR_NAMES, eps.per_pair))
    in_rows = np.isin(E.members(eps.pooled), syn["rows"])
    assert in_rows.all() and E.members(eps.pooled).size == 12_288 * (1 + 13 + 16)


@pytest.mark.parametrize("bad,match", [
    ({"style": list(range(23)) + [24]}, r"values \[24\] of style are not eligible"),          # absent value
    ({"genre": list(range(10)) + [10]}, r"values \[10\] of genre are not eligible"),
    ({"emotion": [0, 1, 2, 3, 4, 5, 6, 7, 8]}, r"values \[8\] of emotion are not eligible"),
])
def test_values_outside_the_eligible_lists_raise(syn, bad, match):
    vs = {**syn["dev"], **bad}
    with pytest.raises(AssertionError, match=match):
        E.build_seed(syn["labels"], syn["groups"], syn["rows"], syn["index"], vs, SEED, R.N_SMOKE)


def test_value_with_29_paintings_raises_and_the_guard_matters(tmp_path):
    s = synthetic(nr_groups=29)                    # style 23 on 29 paintings: not eligible
    assert NR not in AE.eligible_values(s["labels"]["style"], s["groups"], pool_of(s), 30)
    vs = {**s["dev"], "style": list(range(24))}
    args = (s["labels"], s["groups"], s["rows"], "style", "genre", R.N_SMOKE, SEED)
    with pytest.raises(AssertionError, match=r"values \[23\] of style are not eligible on the pool"):
        E.build_aspect_episodes_r6(*args, third="emotion", index=s["index"], values=(vs["style"], vs["genre"]))
    mut = load_copy(tmp_path, guard="values_subset")
    ep = mut.build_aspect_episodes_r6(*args, third="emotion", index=s["index"], values=(vs["style"], vs["genre"]))
    assert len(ep.anchor) == R.N_SMOKE                                 # guard deleted: the bad value goes unnoticed


def test_values_must_be_integer_codes(syn):
    a, b, t = R.PAIRS[0]
    for bad in ([0.0, 1, 2, 3, 4, 5, 6, 7], [True] + list(range(1, 8)), ["0"] + list(range(1, 8))):
        with pytest.raises(AssertionError, match="is not an integer label code"):
            E.build_aspect_episodes_r6(syn["labels"], syn["groups"], syn["rows"], a, b, 4, SEED, third=t,
                                       index=syn["index"], values=(bad, syn["dev"][b]))
    ep = E.build_aspect_episodes_r6(syn["labels"], syn["groups"], syn["rows"], a, b, 4, SEED, third=t,
                                    index=syn["index"], values=(np.arange(8), np.arange(23, dtype=np.int32)))
    assert len(ep.anchor) == 4


# ---------------------------------------------------------------- build_seed's guards

def _swap_wrapper(fn, change):
    """A stand-in for build_aspect_episodes_r6 whose first pair's episodes get ``change`` applied."""
    calls = []

    def wrapped(*a, **kw):
        ep = fn(*a, **kw)
        calls.append(1)
        return change(ep) if len(calls) == 1 else ep
    return wrapped


def _outside_member(s, rows_sub):
    """Replace one negative by the out-of-rows fourth row of its painting (same painting, all labels known): only the
    membership guard can see it."""
    lab = s["labels"]

    def change(ep):
        cand = ep.candidates.copy()
        for i, j in itertools.product(range(len(cand)), range(2, 13)):
            r = 4 * int(s["groups"][cand[i, j]]) + 3
            if r not in rows_sub and all(lab[x][r] >= 0 for x in R.ASPECTS):
                cand[i, j] = r
                return dataclasses.replace(ep, candidates=cand)
        raise AssertionError("no swap found")
    return change


@pytest.fixture(scope="module")
def rows_sub(syn):
    return syn["rows"][syn["rows"] % 4 != 3]          # three rows of each painting; the fourth stays outside


def test_member_outside_rows_fires_and_the_guard_matters(syn, rows_sub, tmp_path, monkeypatch):
    change = _outside_member(syn, set(rows_sub.tolist()))
    a, b, t = R.PAIRS[0]
    ep = change(E.build_aspect_episodes_r6(syn["labels"], syn["groups"], rows_sub, a, b, R.N_SMOKE, SEED, third=t,
                                           index=syn["index"], values=(syn["dev"][a], syn["dev"][b])))
    AE.validate_aspect_episodes(ep, syn["labels"], syn["groups"], syn["index"], third=t)   # validation cannot see it
    args = (syn["labels"], syn["groups"], rows_sub, syn["index"], syn["dev"], SEED, R.N_SMOKE)
    E.build_seed(*args)                                                                    # the unchanged build passes
    monkeypatch.setattr(E, "build_aspect_episodes_r6", _swap_wrapper(E.build_aspect_episodes_r6, change))
    with pytest.raises(AssertionError, match="emotion__style: 1 member rows lie outside the given rows"):
        E.build_seed(*args)
    mut = load_copy(tmp_path, guard="members_in_rows")
    monkeypatch.setattr(mut, "build_aspect_episodes_r6", _swap_wrapper(mut.build_aspect_episodes_r6, change))
    assert mut.build_seed(*args).n == 3 * R.N_SMOKE                  # guard deleted: the outside row goes through


def test_invalid_episode_fires_validation_and_the_guard_matters(syn, tmp_path, monkeypatch):
    def change(ep):                                   # p_a := p_b of the same episode: a painting appears twice
        cand = ep.candidates.copy()
        cand[0, 0] = cand[0, 1]
        return dataclasses.replace(ep, candidates=cand)
    args = (syn["labels"], syn["groups"], syn["rows"], syn["index"], syn["dev"], SEED, R.N_SMOKE)
    monkeypatch.setattr(E, "build_aspect_episodes_r6", _swap_wrapper(E.build_aspect_episodes_r6, change))
    with pytest.raises(AssertionError, match="episode 0: a painting appears twice"):
        E.build_seed(*args)
    mut = load_copy(tmp_path, guard="validate")
    monkeypatch.setattr(mut, "build_aspect_episodes_r6", _swap_wrapper(mut.build_aspect_episodes_r6, change))
    assert mut.build_seed(*args).n == 3 * R.N_SMOKE                 # guard deleted: the bad episode goes through


def test_build_seed_refuses_a_foreign_index_and_unsorted_rows(syn):
    other = AE.PaintingValueIndex({**syn["labels"], "genre": np.roll(syn["labels"]["genre"], 4)}, syn["groups"])
    with pytest.raises(AssertionError, match="index is not PaintingValueIndex"):
        E.build_seed(syn["labels"], syn["groups"], syn["rows"], other, syn["dev"], SEED, 4)
    with pytest.raises(AssertionError, match="sorted, unique"):
        E.build_seed(syn["labels"], syn["groups"], syn["rows"][::-1], syn["index"], syn["dev"], SEED, 4)


# ---------------------------------------------------------------- recorded hashes and the distinctness guard

def _build_record(s):
    return json.loads((E.R3_RESULTS / f"build_seed{s}.json").read_text())


def test_recorded_hashes_real():
    rec = E.recorded_hashes()
    assert sorted(rec) == [42, 43, 45, 47, 48, 49, 50, 51]
    assert all(tuple(h) == R.PAIR_NAMES and all(len(v) == 64 for v in h.values()) for h in rec.values())
    for s in E.R3_BUILD_SEEDS:
        assert rec[s] == _build_record(s)["episode_pair_sha256"]
    assert rec[42] == E.identity_targets()[42]["episodes_sha256"]
    flat = [h for hs in rec.values() for h in hs.values()]
    assert len(set(flat)) == len(flat) == 24


@pytest.fixture
def rec_dirs(tmp_path):
    ab, r3 = tmp_path / "ab", tmp_path / "r3"
    (ab / "smoke").mkdir(parents=True)
    r3.mkdir()
    for s in E.RECORDED_AB_SEEDS:
        shutil.copy(E.AB_RESULTS / f"baselines_seed{s}.json", ab)
    shutil.copy(E.AB_RESULTS / "smoke/baselines_seed9001.json", ab / "smoke")
    for s in E.R3_BUILD_SEEDS:
        shutil.copy(E.R3_RESULTS / f"build_seed{s}.json", r3)
    return ab, r3


def test_recorded_hashes_read_the_build_records_and_skip_smoke(rec_dirs):
    ab, r3 = rec_dirs
    full = E.recorded_hashes(ab, r3)
    assert full == E.recorded_hashes() and 9001 not in full
    for s in E.R3_BUILD_SEEDS:                         # without AB's files of 49 to 51 they come from round 3's records
        (ab / f"baselines_seed{s}.json").unlink()
    got = E.recorded_hashes(ab, r3)
    assert got == full and all(got[s] == _build_record(s)["episode_pair_sha256"] for s in E.R3_BUILD_SEEDS)


def test_missing_record_fires_and_the_guard_matters(rec_dirs, tmp_path):
    ab, r3 = rec_dirs
    (ab / "baselines_seed45.json").unlink()
    with pytest.raises(AssertionError, match=r"AB records missing: \[45\]"):
        E.recorded_hashes(ab, r3)
    mut = load_copy(tmp_path, guard="recorded_complete")
    assert 45 not in mut.recorded_hashes(ab, r3)                # guard deleted: the missing record goes unnoticed


@pytest.mark.parametrize("seed", [43, 50])
def test_changed_record_file_fires_and_the_guard_matters(rec_dirs, tmp_path, seed):
    ab, r3 = rec_dirs
    p = ab / f"baselines_seed{seed}.json"
    p.write_bytes(p.read_bytes() + b"\n")                         # one byte more, the JSON unchanged
    with pytest.raises(SystemExit, match="differs from the recorded"):
        E.recorded_hashes(ab, r3)
    mut = load_copy(tmp_path, guard="recorded_file_sha")
    assert seed in mut.recorded_hashes(ab, r3)                 # guard deleted: the changed file goes unnoticed


def test_build_record_disagreeing_with_baselines_fires_and_the_guard_matters(rec_dirs, tmp_path):
    ab, r3 = rec_dirs
    p = r3 / "build_seed50.json"
    rec = json.loads(p.read_text())
    rec["episode_pair_sha256"]["emotion__genre"] = "0" * 64
    p.write_text(json.dumps(rec))
    with pytest.raises(AssertionError, match="seed 50: the build record and the baselines record disagree"):
        E.recorded_hashes(ab, r3)
    mut = load_copy(tmp_path, guard="recorded_agree")
    assert mut.recorded_hashes(ab, r3)[50]["emotion__genre"] != "0" * 64   # guard deleted: the conflict is dropped


def _fake(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


def _new(seeds=R.HELD_SEEDS):
    return {s: {p: _fake(f"{s}{p}") for p in R.PAIR_NAMES} for s in seeds}


def test_assert_distinct_passes_on_new_hashes_and_on_smoke_records():
    rec = E.assert_distinct(_new())
    assert rec["passed"] and rec["new_seeds"] == [52, 53, 54] and rec["recorded_seeds"] == [42, 43, 45, 47, 48, 49,
                                                                                            50, 51]
    smoke = {s: E.identity_targets()[s]["episodes_sha256"] for s in R.SMOKE_SEEDS}   # smoke records are not recorded
    assert E.assert_distinct(smoke)["passed"]


@pytest.mark.parametrize("where", ["same seed", "across seeds"])
def test_repeated_hash_fires_and_the_guard_matters(tmp_path, where):
    new = _new()
    if where == "same seed":
        new[53]["style__genre"] = new[53]["emotion__style"]
    else:
        new[54]["emotion__genre"] = new[52]["emotion__genre"]
    with pytest.raises(AssertionError, match="repeats the episode hash of seed"):
        E.assert_distinct(new)
    mut = load_copy(tmp_path, guard="distinct_new")
    assert mut.assert_distinct(new)["passed"]                       # guard deleted: the repeat goes unnoticed


@pytest.mark.parametrize("seed,pair", [(50, "emotion__genre"), (49, "style__genre"), (51, "emotion__style"),
                                       (42, "emotion__style"), (47, "style__genre")])
def test_recorded_hash_fires_and_the_guard_matters(tmp_path, seed, pair):
    """Seeds 49 to 51 are taken from round 3's build records, the others from AB's baselines records."""
    want = _build_record(seed)["episode_pair_sha256"][pair] if seed in E.R3_BUILD_SEEDS else \
        json.loads((E.AB_RESULTS / f"baselines_seed{seed}.json").read_text())["episodes_sha256"][pair]
    new = _new()
    new[53]["emotion__genre"] = want
    with pytest.raises(AssertionError, match=f"new seed 53 emotion__genre equals recorded seed {seed} {pair}"):
        E.assert_distinct(new)
    mut = load_copy(tmp_path, guard="distinct_recorded")
    assert mut.assert_distinct(new)["passed"]                       # guard deleted: the clash goes unnoticed


def test_assert_distinct_refuses_malformed_input():
    new = _new()
    new[52] = dict(reversed(list(new[52].items())))
    with pytest.raises(AssertionError, match="pairs"):
        E.assert_distinct(new)
    new = _new()
    new[52]["emotion__style"] = "abc"
    with pytest.raises(AssertionError, match="is not a SHA-256"):
        E.assert_distinct(new)


# ---------------------------------------------------------------- the episode file

def test_save_load_round_trip(dev_eps, tmp_path):
    path = E.save_episodes(tmp_path / "held_episodes_seed52.npz", dev_eps)
    assert path.is_file() and not list(tmp_path.glob("*.partial.npz"))
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == set(E.FILE_KEYS) and len(z.files) == 12     # seed, pair_index, 6 arrays, names, 3 SHAs
        assert all(z[k].dtype == np.int64 for k in ("seed", "pair_index", *E.FIELDS))
        assert z["candidates"].shape == (12_288, 13) and z["pairs_b_txt"].shape == (12_288, 4)
    got = E.load_episodes(path)
    assert got.seed == dev_eps.seed and got.n == dev_eps.n and got.sha == dev_eps.sha
    assert np.array_equal(got.pair_index, dev_eps.pair_index) and np.array_equal(got.parity, dev_eps.parity)
    assert all(same(x, y) for x, y in zip(got.per_pair, dev_eps.per_pair)) and same(got.pooled, dev_eps.pooled)
    assert {p: AE.episodes_sha256(e) for p, e in zip(R.PAIR_NAMES, got.per_pair)} == dev_eps.sha


def test_save_refuses_episodes_that_do_not_match_their_hashes(dev_eps, tmp_path):
    bad = dataclasses.replace(dev_eps.per_pair[1], anchor=dev_eps.per_pair[1].anchor[::-1].copy())
    eps = type(dev_eps)(**{**vars(dev_eps), "per_pair": [dev_eps.per_pair[0], bad, dev_eps.per_pair[2]]})
    with pytest.raises(AssertionError, match="emotion__genre: the per-pair episodes do not match their hash"):
        E.save_episodes(tmp_path / "x.npz", eps)


def test_tampered_file_fires_on_load_and_the_guard_matters(dev_eps, tmp_path):
    path = E.save_episodes(tmp_path / "ep.npz", dev_eps)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["anchor"] = d["anchor"].copy()
    d["anchor"][5000] = d["anchor"][5001]             # an emotion__genre episode
    np.savez(tmp_path / "tampered.npz", **d)
    with pytest.raises(AssertionError, match="emotion__genre episodes do not match their stored hash"):
        E.load_episodes(tmp_path / "tampered.npz")
    mut = load_copy(tmp_path, guard="load_sha")
    assert mut.load_episodes(tmp_path / "tampered.npz").n == 12_288  # guard deleted: the change goes unnoticed


# ---------------------------------------------------------------- the identity check on real selection rows

@pytest.fixture(scope="module")
def real():
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_aspect_labels
    data = load_artelingo()
    split = R.load_split(data)
    labels = artelingo_aspect_labels(data)
    del data
    index = AE.PaintingValueIndex(labels, split.groups)
    vs = R.development_value_sets(labels, split.groups, split.selection)
    return dict(split=split, labels=labels, index=index, vs=vs)


def test_identity_targets():
    t = E.identity_targets()
    assert sorted(t) == [42, 9001, 9002, 9003]
    assert t[42]["n_per_pair"] == 4096 and all(t[s]["n_per_pair"] == 64 for s in R.SMOKE_SEEDS)
    assert [t[42]["episodes_sha256"][p][:8] for p in R.PAIR_NAMES] == ["6932f553", "9196a1a8", "78fbccdc"]
    assert [t[9001]["episodes_sha256"][p][:8] for p in R.PAIR_NAMES] == ["b99f5193", "711da43b", "0fc38bdd"]
    smoke42 = json.loads((E.AB_RESULTS / "smoke/baselines_seed42.json").read_text())["episodes_sha256"]
    assert all(smoke42[p] != t[42]["episodes_sha256"][p] for p in R.PAIR_NAMES)     # AB's 64-per-pair file is not it


def test_identity_check_reproduces_the_twelve_stored_hashes_on_selection_rows(real):
    sp = real["split"]
    res = E.identity_check(real["labels"], sp.groups, sp.selection, real["index"], real["vs"])
    assert len(res["items"]) == 12
    bad = [k for k, v in res["items"].items() if not v["equal"]]
    assert res["passed"] and not bad, bad
    targets = E.identity_targets()
    for s, t in targets.items():
        for p in R.PAIR_NAMES:
            item = res["items"][f"episodes_seed{s}__{p}"]
            assert item["got"] == item["want"] == t["episodes_sha256"][p]
    # the smoke seeds pass the distinctness guard; seed 42 (a recorded non-smoke seed) does not
    got = {s: {p: res["items"][f"episodes_seed{s}__{p}"]["got"] for p in R.PAIR_NAMES} for s in targets}
    assert E.assert_distinct({s: got[s] for s in R.SMOKE_SEEDS})["passed"]
    with pytest.raises(AssertionError, match="equals recorded seed 42"):
        E.assert_distinct({42: got[42]})
