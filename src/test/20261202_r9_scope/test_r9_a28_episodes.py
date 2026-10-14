"""Tests of r9_a28_episodes (ticket 17): the ArtELingo-28 per-language episode builder (rule section 5.1) and its hash
guard (R6 rule section 5 item 3, applied by section 5.1 item 5).

Synthetic data: a few languages of different sizes over shared paintings, 8 values per aspect, the real builder
(`build_aspect_episodes_r6`) called on sub-arrays. Scenarios that need a particular exception use a wrapper around the
real builder that raises it for chosen languages. Mutation tests load a copy of r9_a28_episodes with one text
replacement and show that the guarded scenario then fails.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_a28_episodes.py
"""
import importlib.util
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first: it puts MAIN's src in front and checks it)
import r9_a28_episodes as A  # noqa: E402
import r9_episodes as E  # noqa: E402

import src.eval.aspect_episodes as AE  # noqa: E402

HERE_LINE = "HERE = Path(__file__).resolve().parent\n"
_COUNT = itertools.count()
PAIRS = R.PAIR_NAMES
V = {"emotion": list(range(8)), "style": list(range(8)), "genre": list(range(8))}
FAILED_DRAW = "emotion x style: could not fill 3 episodes (31 failed draws); the pools are too small for these constraints"
SUBSET_MSG = "emotion x genre: values [7] of genre are not eligible on the pool"


# ---------------------------------------------------------------- synthetic data

def make_table(sizes=(150, 120, 100, 80), langs=("Burmese", "Hindi", "Zulu", "arabic"), seed=0, per=2,
               genre_missing=(), tiny=()):
    """Rows interleaved over languages. Language i holds the first sizes[i] paintings, ``per`` rows each. Style and
    genre are per painting, emotion per row. ``genre_missing``: languages whose genre 7 becomes 6 (value 7 absent).
    ``tiny``: languages restricted to 12 paintings with every value present (a failed-draw language)."""
    rng = np.random.default_rng(seed)
    n_paint = max(sizes)
    names = np.array([f"w{i:04d}" for i in range(n_paint)])
    style_p = rng.integers(0, 8, n_paint)
    genre_p = rng.integers(0, 8, n_paint)
    style_p[:12] = np.arange(12) % 8
    genre_p[:12] = (np.arange(12) * 3) % 8
    rows = []
    for lang, size in zip(langs, sizes):
        size = 12 if lang in tiny else size
        for p in range(size):
            for _ in range(per):
                rows.append((p, lang))
    order = rng.permutation(len(rows))
    p = np.array([rows[i][0] for i in order])
    lang = np.array([rows[i][1] for i in order])
    emotion = rng.integers(0, 8, len(p))
    for t in tiny:
        m = lang == t
        emotion[m] = np.arange(m.sum()) % 8
    genre = genre_p[p].copy()
    for g in genre_missing:
        genre[(lang == g) & (genre == 7)] = 6
    return pd.DataFrame({"row_id": np.arange(len(p), dtype=np.int64) * 3 + 5, "painting": names[p], "language": lang,
                         "emotion": emotion.astype(np.int64), "style": style_p[p].astype(np.int64),
                         "genre": genre.astype(np.int64), "held": False, "dropped": False})


N = 40       # episodes per pair in the synthetic tests


# ---------------------------------------------------------------- mutation helpers

def mutant_source(replace=()) -> str:
    src = (HERE / "r9_a28_episodes.py").read_text()
    assert src.count(HERE_LINE) == 1
    src = src.replace(HERE_LINE, f"HERE = Path({str(HERE)!r})\n")
    for old, new in replace:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def load_copy(tmp_path, replace=()):
    path = tmp_path / f"r9_a28_episodes_copy{next(_COUNT)}.py"
    path.write_text(mutant_source(replace))
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


class Faulty:
    """Wraps the real builder; raises ``rules[(seed_mod_1000, pair)]`` (an exception or a callable n -> exception or
    None) for the given language index and pair; ``rules`` keys may use pair=None for every pair. Records calls."""

    def __init__(self, rules):
        self.rules = rules
        self.real = E.build_aspect_episodes_r6
        self.calls = []

    def __call__(self, **kw):
        l = kw["seed"] % 1000
        pair = f"{kw['aspect_a']}__{kw['aspect_b']}"
        self.calls.append(dict(kw, lang=l, pair=pair))
        for key in ((l, pair), (l, None)):
            if key in self.rules:
                exc = self.rules[key]
                exc = exc(kw["n_episodes"]) if callable(exc) else exc
                if exc is not None:
                    raise exc
        return self.real(**kw)


def prep(mod, table):
    """mod.prepare with the table's own language names as the expected names (the synthetic tables are not the 27)."""
    return mod.prepare(table, expected_languages=tuple(sorted(set(table.language))))


def run(mod, table, faulty=None, monkeypatch=None, seed=9001, n=N, **kw):
    """mod.build_seed_a28 on ``table`` with the faulty wrapper installed in the E module r9_a28_episodes uses."""
    if faulty is not None:
        monkeypatch.setattr(mod.E, "build_aspect_episodes_r6", faulty)
    return mod.build_seed_a28(prep(mod, table), V, seed, n_per_pair=n, **kw)


# ---------------------------------------------------------------- languages and shares

RULE_LANGUAGES = ("Burmese", "Darija", "Emakhuwa", "Hausa", "Hindi", "Igbo", "Indonesian", "IsiNdebele", "IsiXhosa",
                  "IsiZulu", "Kazakh", "Kinyarwanda", "Korean", "Kyrgyz", "Malay", "Setswana", "Swahili", "Tagalog",
                  "Tamil", "Thai", "Turkish", "Urdu", "Uzbek", "Vietnamese", "Yoruba", "arabic", "chinese")


def table_27(rename=None):
    t = make_table(sizes=(14,) * 27, langs=RULE_LANGUAGES, per=1)
    if rename:
        t["language"] = t["language"].replace(rename)
    return t


def test_the_rule_names_are_the_27_in_sort_order():
    assert A.LANGUAGES == RULE_LANGUAGES and list(A.LANGUAGES) == sorted(A.LANGUAGES)


def test_prepare_accepts_the_27_rule_names():
    assert A.prepare(table_27()).languages == list(RULE_LANGUAGES)


@pytest.mark.parametrize("rename", [{"Hindi": "Hindhi"}, {"Zzz": "x"}, {"arabic": "Arabic"}])
def test_prepare_refuses_a_wrong_language_name(rename):
    t = table_27(rename if "Zzz" not in rename else None)
    if "Zzz" in rename:
        t = t[t.language != "chinese"]               # a missing language
    with pytest.raises(AssertionError, match="language names"):
        A.prepare(t)


def test_prepare_refuses_an_extra_language():
    t = table_27()
    t.loc[t.index[0], "language"] = "Klingon"
    with pytest.raises(AssertionError, match="language names"):
        A.prepare(t)

def test_languages_are_in_python_sort_order():
    assert A.languages_of(pd.Series(["arabic", "Zulu", "Burmese", "chinese", "Hindi", "Burmese"])) == \
        ["Burmese", "Hindi", "Zulu", "arabic", "chinese"]


def test_shares_by_largest_remainders():
    # 10 * [5, 3, 2] / 10 exact
    assert A.shares([5, 3, 2], 10) == [5, 3, 2]
    # floors 3, 3, 3 (remainders 1/3 each of 10) -> one leftover, the tie goes to the lowest index
    assert A.shares([1, 1, 1], 10) == [4, 3, 3]
    # 7 * [5, 4, 1] / 10 = 3.5, 2.8, 0.7: floors 3, 2, 0; leftover 2 go to the remainders .8 and .7
    assert A.shares([5, 4, 1], 7) == [3, 3, 1]
    # 3 * [1, 1, 2] / 4 = .75, .75, 1.5: floors 0, 0, 1; the two leftovers go to remainders .75 and .75
    assert A.shares([1, 1, 2], 3) == [1, 1, 1]


def test_shares_tie_goes_to_the_lower_language():
    # 4096 * [1, 1] / 2 exact; with 3 equal pools of 4096 episodes: 1365.33 each -> leftover 1 to the lowest
    assert A.shares([7, 7, 7], 4096) == [1366, 1365, 1365]
    # two equal remainders, two leftovers, three candidates: the two lowest take them
    assert A.shares([1, 1, 1, 1, 1, 1], 8) == [2, 2, 1, 1, 1, 1]


def test_shares_sum_is_total_and_exact_on_large_pools():
    rng = np.random.default_rng(3)
    for _ in range(50):
        sizes = rng.integers(1, 20000, 27).tolist()
        s = A.shares(sizes, 4096)
        assert sum(s) == 4096 and all(x >= 0 for x in s)
        exact = [4096 * x / sum(sizes) for x in sizes]
        assert all(abs(a - b) < 1 for a, b in zip(s, exact))


def test_a_language_with_no_episodes_is_refused():
    with pytest.raises(AssertionError, match="no episodes"):
        A.build_seed_a28(prep(A, make_table()), V, 9001, n_per_pair=2)


def test_sum_guard_fires_when_shares_do_not_add_up(monkeypatch):
    monkeypatch.setattr(A, "shares", lambda sizes, total: [total - 1] + [0] * (len(sizes) - 1))
    with pytest.raises(AssertionError, match="4,096|sum"):
        A.build_seed_a28(prep(A, make_table()), V, 9001, n_per_pair=N)


# ---------------------------------------------------------------- the builder calls

@pytest.fixture(scope="module")
def table():
    return make_table()


def test_builder_called_with_the_rule_arguments(table, monkeypatch):
    t = table.copy()
    t.loc[t.index[:7], "dropped"] = True
    f = Faulty({})
    res = run(A, t, f, monkeypatch, seed=9002)
    langs = A.languages_of(t.language)
    assert len(f.calls) == 3 * len(langs)
    ctx = prep(A, t)
    for c in f.calls:
        lang = langs[c["lang"] - 1]
        R_l = np.flatnonzero((t.language == lang).to_numpy())
        # seed 1000 s + l, the same for the three pairs
        assert c["seed"] == 1000 * 9002 + c["lang"]
        assert c["min_paintings"] == 1 and c["values"] == (V[c["aspect_a"]], V[c["aspect_b"]])
        # sub-arrays: labels[a][R], groups[R] = global painting index, index over the sub-arrays
        cl = np.unique(t.painting.to_numpy(), return_inverse=True)[1]
        for a in R.ASPECTS:
            assert np.array_equal(c["labels"][a], t[a].to_numpy()[R_l])
            assert np.array_equal(c["index"].labels[a], t[a].to_numpy()[R_l])
        assert np.array_equal(c["groups"], cl[R_l]) and np.array_equal(c["index"].groups, cl[R_l])
        # rows = the kept positions in R_l (a dropped row is not a row but its labels stay in the index)
        kept = np.flatnonzero(~t.dropped.to_numpy()[R_l])
        assert np.array_equal(c["rows"], kept)
        assert c["third"] == dict((p, th) for (_, _, th), p in zip(R.PAIRS, PAIRS))[c["pair"]]
    # the first seven rows are dropped: some language's rows lost them, the sizes show it
    assert sum(len(c["rows"]) for c in f.calls[:len(langs)]) == len(t) - 7


def pool_table():
    """Seven dropped rows, ten unknown emotions, and about half of Burmese's rows dropped (its share must move)."""
    t = make_table()
    t.loc[t.index[:7], "dropped"] = True
    t.loc[t.index[20:30], "emotion"] = -1
    burmese = np.flatnonzero((t.language == "Burmese").to_numpy())
    t.loc[t.index[burmese[::2]], "dropped"] = True
    return t


def scn_pool_after_drops(mod, monkeypatch):
    """True when the shares come from the kept rows with all labels known (and differ from the all-rows shares)."""
    t = pool_table()
    f = Faulty({})
    res = run(mod, t, f, monkeypatch)
    langs = mod.languages_of(t.language)
    known = (t[list(R.ASPECTS)].to_numpy() >= 0).all(axis=1)
    keep = (~t.dropped.to_numpy()) & known
    sizes = [int((keep & (t.language == l).to_numpy()).sum()) for l in langs]
    all_rows = [int((known & (t.language == l).to_numpy()).sum()) for l in langs]
    want = mod.shares(sizes, N)
    assert want != mod.shares(all_rows, N)       # the data tell the two apart
    return all([res.shares[p][l] for l in langs] == want for p in PAIRS) and \
        [c["n_episodes"] for c in f.calls[:len(langs)]] == want


def test_shares_use_pool_sizes_after_drops_with_labels_known(monkeypatch):
    assert scn_pool_after_drops(A, monkeypatch) is True


def test_global_rows_are_the_sub_array_rows_mapped_by_R(table):
    res = A.build_seed_a28(prep(A, table), V, 9001, n_per_pair=N)
    langs = A.languages_of(table.language)
    labels = {a: table[a].to_numpy() for a in R.ASPECTS}
    cl = np.unique(table.painting.to_numpy(), return_inverse=True)[1]
    for (a, b, third), p in zip(R.PAIRS, PAIRS):
        parts = []
        for l, lang in enumerate(langs, start=1):
            R_l = np.flatnonzero((table.language == lang).to_numpy())
            sub = {k: v[R_l] for k, v in labels.items()}
            idx = AE.PaintingValueIndex(sub, cl[R_l])
            ep = E.build_aspect_episodes_r6(sub, cl[R_l], np.arange(len(R_l)), a, b, res.shares[p][lang],
                                            1000 * 9001 + l, third=third, min_paintings=1, index=idx,
                                            values=(V[a], V[b]))
            glob = AE.AspectEpisodes(a, b, *(R_l[getattr(ep, f)] for f in E.FIELDS))
            assert E.episodes_sha256(glob) == res.part_sha[p][lang]
            assert E.episodes_sha256(res.parts[p][lang]) == res.part_sha[p][lang]
            parts.append(glob)
            # every member row lies in R_l
            assert (table.language.to_numpy()[E.members(res.parts[p][lang])] == lang).all()
        cat = AE.concat_episodes(parts)
        pooled = res.per_pair[PAIRS.index(p)]
        for f in E.FIELDS:
            assert np.array_equal(getattr(pooled, f), getattr(cat, f))
        assert res.sha[p] == E.episodes_sha256(cat)


def test_assembly_shapes_and_bookkeeping(table):
    res = A.build_seed_a28(prep(A, table), V, 9003, n_per_pair=N)
    assert res.n == 3 * N and len(res.per_pair) == 3 and tuple(res.sha) == PAIRS
    assert np.array_equal(res.pair_index, np.repeat(np.arange(3), N))
    assert np.array_equal(res.parity, np.arange(3 * N) % 2)
    assert res.pooled.anchor.shape == (3 * N,) and res.pooled.candidates.shape == (3 * N, 13)
    langs = A.languages_of(table.language)
    for p in PAIRS:
        assert sum(res.shares[p].values()) == N and res.set_aside[p] == []
        # language of each episode: blocks in l order
        lang_of = res.language[p]
        assert lang_of.shape == (N,) and (np.diff(lang_of) >= 0).all()
        assert [int((lang_of == l).sum()) for l in range(1, len(langs) + 1)] == [res.shares[p][l] for l in langs]
    assert res.seed == 9003 and len(set(res.sha.values())) == 3


def test_every_part_is_validated(table, monkeypatch):
    seen = []
    real = AE.validate_aspect_episodes
    monkeypatch.setattr(A, "validate_aspect_episodes",
                        lambda ep, labels, groups, index, third=None: (seen.append(len(ep.anchor)),
                                                                      real(ep, labels, groups, index, third=third))[1])
    A.build_seed_a28(prep(A, table), V, 9001, n_per_pair=N)
    assert len(seen) == 3 * 4 and sum(seen) == 3 * N


# ---------------------------------------------------------------- set-aside passes

def test_real_values_subset_failure_sets_the_language_aside_for_the_pairs_it_touches(monkeypatch):
    t = make_table(genre_missing=("Zulu",))
    res = A.build_seed_a28(prep(A, t), V, 9001, n_per_pair=N)
    assert res.set_aside["emotion__style"] == []
    assert res.set_aside["emotion__genre"] == ["Zulu"] and res.set_aside["style__genre"] == ["Zulu"]
    assert "Zulu" not in res.shares["emotion__genre"] and "Zulu" in res.shares["emotion__style"]
    for p in PAIRS:
        assert sum(res.shares[p].values()) == N


def test_real_failed_draw_sets_the_language_aside_and_the_rest_are_re_split():
    t = make_table(tiny=("arabic",))
    res = A.build_seed_a28(prep(A, t), V, 9001, n_per_pair=N)
    langs = A.languages_of(t.language)
    keep = [l for l in langs if l != "arabic"]
    sizes = [int(((t.language == l)).sum()) for l in keep]
    for p in PAIRS:
        assert res.set_aside[p] == ["arabic"]
        assert [res.shares[p][l] for l in keep] == A.shares(sizes, N)
        assert res.language[p].max() == 3          # arabic is l = 4, not part of the episodes


def test_re_split_pass_uses_original_language_index_for_seeds(monkeypatch):
    t = make_table()
    boom = RuntimeError(FAILED_DRAW)
    f = Faulty({(2, None): boom})        # Hindi (l = 2) fails in every pair
    res = run(A, t, f, monkeypatch)
    assert res.set_aside["emotion__style"] == ["Hindi"]
    second = [c for c in f.calls if c["pair"] == "emotion__style" and c["lang"] != 2]
    # pass 1 builds l = 1, 2, 3, 4; pass 2 builds l = 1, 3, 4 with seeds 1000 s + l (not renumbered)
    assert [c["seed"] % 1000 for c in second] == [1, 3, 4, 1, 3, 4]
    assert [c["seed"] % 1000 for c in f.calls if c["pair"] == "emotion__style"] == [1, 2, 3, 4, 1, 3, 4]
    assert second[3]["n_episodes"] == res.shares["emotion__style"]["Burmese"]
    assert sum(res.shares["emotion__style"].values()) == N


def test_a_language_that_fails_only_at_the_larger_share_is_set_aside_in_the_next_pass(monkeypatch):
    t = make_table()
    # Burmese (l = 1) fails when its share is above 12; first pass Hindi fails, which raises Burmese's share
    limit = lambda n: RuntimeError(FAILED_DRAW) if n > 12 else None
    f = Faulty({(2, None): RuntimeError(FAILED_DRAW), (1, None): limit})
    res = run(A, t, f, monkeypatch)
    for p in PAIRS:
        assert res.set_aside[p] == ["Hindi", "Burmese"] or res.set_aside[p] == ["Burmese", "Hindi"] or \
            set(res.set_aside[p]) == {"Hindi", "Burmese"}
        assert sum(res.shares[p].values()) == N and set(res.shares[p]) == {"Zulu", "arabic"}


def test_two_set_aside_are_allowed_and_a_third_stops(monkeypatch):
    t = make_table()
    two = Faulty({(1, None): RuntimeError(FAILED_DRAW), (2, None): AssertionError(SUBSET_MSG)})
    res = run(A, t, two, monkeypatch)
    assert all(set(res.set_aside[p]) == {"Burmese", "Hindi"} for p in PAIRS)
    three = Faulty({(1, None): RuntimeError(FAILED_DRAW), (2, None): AssertionError(SUBSET_MSG),
                    (3, None): RuntimeError(FAILED_DRAW)})
    with pytest.raises(A.SetAsideStop, match="Burmese.*Hindi.*Zulu|more than two"):
        run(A, t, three, monkeypatch)


def test_a_third_set_aside_in_a_later_pass_stops_too(monkeypatch):
    t = make_table()
    # l = 4 fails only once the share is above its first-pass share
    f = Faulty({(1, None): RuntimeError(FAILED_DRAW), (2, None): RuntimeError(FAILED_DRAW),
                (3, None): lambda n: RuntimeError(FAILED_DRAW) if n > 14 else None})
    first_share = A.shares([int((t.language == l).sum()) for l in A.languages_of(t.language)], N)[2]
    assert first_share <= 14
    with pytest.raises(A.SetAsideStop):
        run(A, t, f, monkeypatch)


def test_stop_is_per_pair(monkeypatch):
    t = make_table()
    f = Faulty({(1, "emotion__genre"): RuntimeError(FAILED_DRAW), (2, "emotion__genre"): RuntimeError(FAILED_DRAW),
                (3, "emotion__style"): RuntimeError(FAILED_DRAW), (4, "style__genre"): RuntimeError(FAILED_DRAW)})
    res = run(A, t, f, monkeypatch)      # two in one pair, one in the others: no stop
    assert [len(res.set_aside[p]) for p in PAIRS] == [1, 2, 1]


@pytest.mark.parametrize("exc", [
    ValueError("anything"),
    KeyError("k"),
    RuntimeError("emotion x style: need more than 4 eligible values per aspect, got 3 and 8"),
    AssertionError("value 'x' is not an integer label code"),
    AssertionError("episode 3: a painting appears twice"),
    AssertionError(),
])
def test_other_exceptions_propagate(exc, monkeypatch):
    t = make_table()
    f = Faulty({(2, None): exc})
    with pytest.raises(type(exc)):
        run(A, t, f, monkeypatch)


def test_a_validation_failure_stops_the_step_and_sets_nothing_aside(monkeypatch):
    t = make_table()
    calls = []

    def bad(ep, labels, groups, index, third=None):
        calls.append(1)
        raise AssertionError(SUBSET_MSG)       # same text as a set-aside, but raised by validation

    monkeypatch.setattr(A, "validate_aspect_episodes", bad)
    with pytest.raises(AssertionError):
        A.build_seed_a28(prep(A, t), V, 9001, n_per_pair=N)
    assert len(calls) == 1


def test_a_bad_part_of_an_earlier_pass_stops_the_step(monkeypatch):
    """Hindi fails its draws in pass 1; Burmese (13 episodes in pass 1, 18 in pass 2) is built in both passes. A bad
    pass-1 part of Burmese must stop the step although its pass-2 part is fine (rule section 5.1 item 6)."""
    t = make_table()
    monkeypatch.setattr(A.E, "build_aspect_episodes_r6", Faulty({(2, None): RuntimeError(FAILED_DRAW)}))
    real = AE.validate_aspect_episodes
    first = []

    def check(ep, labels, groups, index, third=None):
        first.append(len(ep.anchor))
        if len(ep.anchor) == 13:                   # Burmese's pass-1 share
            raise AssertionError("episode 3: a painting appears twice")
        return real(ep, labels, groups, index, third=third)

    monkeypatch.setattr(A, "validate_aspect_episodes", check)
    with pytest.raises(AssertionError, match="appears twice"):
        A.build_seed_a28(prep(A, t), V, 9001, n_per_pair=N)
    assert first[0] == 13


def member_in_dropped_row(mod, monkeypatch):
    """A builder that returns an anchor on a dropped row (validation silenced so only the member check can see it)."""
    t = make_table()
    t.loc[t.index[:40], "dropped"] = True
    real = E.build_aspect_episodes_r6

    def leak(**kw):
        ep = real(**kw)
        out = np.setdiff1d(np.arange(len(kw["groups"])), kw["rows"])
        if len(out):
            ep.anchor[0] = out[0]
        return ep

    monkeypatch.setattr(mod.E, "build_aspect_episodes_r6", leak)
    monkeypatch.setattr(mod, "validate_aspect_episodes", lambda *a, **k: None)
    return mod.build_seed_a28(prep(mod, t), V, 9001, n_per_pair=N)


def test_member_row_on_a_dropped_row_fires(monkeypatch):
    with pytest.raises(AssertionError, match="outside"):
        member_in_dropped_row(A, monkeypatch)


def test_the_exception_classes_match_the_real_builder_messages():
    """The two set-aside exceptions are told from the others by message; the real builder's messages must match."""
    t = make_table(genre_missing=("Zulu",), tiny=("arabic",))
    ctx = prep(A, t)
    with pytest.raises(AssertionError) as e1:
        E.build_aspect_episodes_r6(**ctx_kwargs(ctx, "Zulu", "emotion", "genre", "style"))
    assert A.set_aside_reason(e1.value) == "values_subset"
    with pytest.raises(RuntimeError) as e2:
        E.build_aspect_episodes_r6(**ctx_kwargs(ctx, "arabic", "emotion", "style", "genre"))
    assert A.set_aside_reason(e2.value) == "failed_draws"
    assert A.set_aside_reason(RuntimeError("emotion x style: need more than 4 eligible values per aspect, got 1 and 2")) is None
    assert A.set_aside_reason(AssertionError("value 'x' is not an integer label code")) is None
    assert A.set_aside_reason(ValueError("x")) is None


def ctx_kwargs(ctx, lang, a, b, third, n=5):
    l = ctx.languages.index(lang)
    return dict(labels=ctx.labels[l], groups=ctx.groups[l], rows=ctx.kept[l], aspect_a=a, aspect_b=b, n_episodes=n,
                seed=1, third=third, min_paintings=1, index=ctx.index[l], values=(V[a], V[b]))


# ---------------------------------------------------------------- the hash guard

H = lambda c: c * 64
SEED_LEDGER = """# Episode-seed ledger

| Seed | Status | Scored by (date, purpose) |
|---|---|---|
| 42 | development and picks | E1 baselines |
| 49, 50, 51 | test (spent) | Reader-fix round 3 test |
| 55 | reserved | round 9, ArtELingo-28 reads, builder seeds 55001 to 55027 |
| 55002 | spent | an earlier round |
| 58 and later | free | none |
| 9001, 9002, 9003 (builder seeds 9001001 to 9001027) | smoke | round 9 smoke builder seeds |
| 9001, 9002, 9003 | smoke | Wiring smoke only, round 3 and round 6 |
"""
SEED_LEDGER = SEED_LEDGER.replace("| 55 | reserved", "| 55, 55001 | reserved")      # round 9's own row may list them


def held_ledger(hashes=None):
    hs = hashes or {s: {p: H(c) for p, c in zip(PAIRS, cs)} for s, cs in ((52, "abc"), (53, "def"), (54, "012"))}
    cell = "; ".join(f"seed {s}: emotion × style {hs[s]['emotion__style']}, emotion × genre {hs[s]['emotion__genre']}, "
                     f"style × genre {hs[s]['style__genre']}" for s in sorted(hs))
    return ("# Held-read ledger\n\n| # | Date | Dataset | Backbone | Purpose | Episode / data SHA-256 | Script | Report |\n"
            "|---|---|---|---|---|---|---|---|\n"
            f"| H4 | 2026-10-02 | CUB | CLIP | probe | n/a | see report | x |\n"
            f"| H5 | 2026-10-10 | ArtELingo held | CLIP | AFF held read | {cell} | a82c | x |\n")


@pytest.fixture
def ledgers(tmp_path):
    s, h = tmp_path / "seed_ledger.md", tmp_path / "held_ledger.md"
    s.write_text(SEED_LEDGER)
    h.write_text(held_ledger())
    return s, h


FRESH = (56, 57, 58)        # read seeds whose builder seeds (n_languages = 3) sit in no row of the test ledger


def new_hashes(chars="ghijklmno", read_seeds=(55, 56, 57)):
    cs = iter(chars)
    return {s: {p: H(next(cs)) for p in PAIRS} for s in read_seeds}


RECORDED = {42: {p: H("p") for p in PAIRS}, 43: {p: H("q") for p in PAIRS}}


def guard(mod, new, seeds, ledgers, recorded=RECORDED, n_languages=3):
    """``seeds`` None: the builder seeds of ``new``'s read seeds, as the read passes them."""
    s, h = ledgers
    seeds = mod.builder_seeds(sorted(new), n_languages) if seeds is None else seeds
    return mod.hash_guard(new, seeds, held_ledger=h, seed_ledger=s, recorded=recorded, n_languages=n_languages)


def test_builder_seeds_are_1000s_plus_l():
    seeds = A.builder_seeds((55, 56, 57), 27)
    assert len(seeds) == 81 and seeds[0] == 55001 and seeds[26] == 55027 and seeds[27] == 56001 and seeds[-1] == 57027
    assert len(set(seeds)) == 81


def test_guard_passes_on_fresh_hashes_and_seeds(ledgers):
    out = guard(A, new_hashes(read_seeds=FRESH), None, ledgers)
    assert out["passed"] and out["h5_hashes"] == 9 and out["builder_seeds"] == 9


def test_h5_nine_are_parsed_from_the_ledger(ledgers):
    got = A.h5_hashes(ledgers[1])
    assert sorted(got) == [52, 53, 54] and got[52]["emotion__style"] == H("a") and got[54]["style__genre"] == H("2")


def test_h5_with_other_than_nine_hashes_is_refused(tmp_path):
    p = tmp_path / "held.md"
    p.write_text(held_ledger().replace(f"style × genre {H('2')}", "style × genre 0000"))
    with pytest.raises(AssertionError, match="nine|H5"):
        A.h5_hashes(p)
    p.write_text(held_ledger().replace("| H5 |", "| H6 |"))
    with pytest.raises(AssertionError, match="H5"):
        A.h5_hashes(p)


def scn_h5(mod, ledgers):
    new = new_hashes(read_seeds=FRESH)
    new[56]["emotion__genre"] = H("d")          # H5 seed 53 emotion x style
    return raised(lambda: guard(mod, new, None, ledgers))


def raised(fn):
    try:
        fn()
    except AssertionError as e:
        return str(e)
    return None


def scn_recorded(mod, ledgers):
    new = new_hashes(read_seeds=FRESH)
    new[57]["style__genre"] = H("q")            # recorded seed 43
    return raised(lambda: guard(mod, new, None, ledgers))


def scn_distinct_new(mod, ledgers):
    new = new_hashes(read_seeds=FRESH)
    new[58]["emotion__style"] = new[56]["emotion__style"]
    return raised(lambda: guard(mod, new, None, ledgers))


def scn_seed_in_other_row(mod, ledgers):
    return raised(lambda: guard(mod, new_hashes(), None, ledgers))        # 55002 sits in another round's row


def scn_seed_in_plain_row(mod, ledgers):
    return raised(lambda: mod.assert_seeds_free([49], ledgers[0]))


def scn_round9_row_is_allowed(mod, ledgers):
    return raised(lambda: mod.assert_seeds_free([55001, 9001001, 55, 58], ledgers[0]))


def scn_free_row_is_allowed(mod, ledgers):
    return raised(lambda: mod.assert_seeds_free([58], ledgers[0]))


def scn_repeated_seed(mod, ledgers):
    return raised(lambda: mod.assert_seeds_free([60001, 60001], ledgers[0]))


@pytest.mark.parametrize("scn, text", [(scn_h5, "H5"), (scn_recorded, "recorded"), (scn_distinct_new, "repeats"),
                                       (scn_seed_in_other_row, "55002"), (scn_seed_in_plain_row, "49"),
                                       (scn_repeated_seed, "60001")])
def test_guard_fires(scn, text, ledgers):
    msg = scn(A, ledgers)
    assert msg is not None and text in msg, msg


def test_round_9_rows_and_free_rows_do_not_block(ledgers):
    assert scn_round9_row_is_allowed(A, ledgers) is None
    assert scn_free_row_is_allowed(A, ledgers) is None


def test_the_real_ledgers_pass_for_the_read_and_the_smoke_seeds():
    seeds = A.builder_seeds((55, 56, 57), 27) + A.builder_seeds((9001, 9002, 9003), 27)
    assert A.assert_seeds_free(seeds)["passed"]
    h5 = A.h5_hashes()
    assert sorted(h5) == [52, 53, 54] and all(len(v) == 64 for hs in h5.values() for v in hs.values())
    # seeds of other rounds are not free
    for s in (42, 43, 52, 54):
        with pytest.raises(AssertionError):
            A.assert_seeds_free([s])


def test_the_real_recorded_hashes_include_h5_in_the_guard():
    new = new_hashes()
    seeds = A.builder_seeds(sorted(new), 27)
    A.hash_guard(new, seeds)
    h5 = A.h5_hashes()
    bad = {55: dict(new[55]), 56: new[56], 57: new[57]}
    bad[55]["emotion__style"] = h5[53]["emotion__style"]
    with pytest.raises(AssertionError, match="H5"):
        A.hash_guard(bad, seeds)


def scn_seed_list_is_the_reads(mod, ledgers):
    """The seed list must be the read's own 1000 s + l: an empty, short, shifted or foreign list is refused."""
    new = new_hashes(read_seeds=FRESH)
    ok = mod.builder_seeds(sorted(new), 3)
    msgs = [raised(lambda: guard(mod, new, bad, ledgers)) for bad in
            ([], ok[:-1], ok[1:] + [ok[-1] + 1], list(reversed(ok)), mod.builder_seeds((56, 57, 59), 3))]
    return all(m is not None and "builder seed list" in m for m in msgs) and raised(lambda: guard(mod, new, ok, ledgers)) is None


def test_the_guard_refuses_a_seed_list_that_is_not_the_reads(ledgers):
    assert scn_seed_list_is_the_reads(A, ledgers) is True


def test_the_guard_refuses_the_empty_list_on_the_real_ledgers():
    new = new_hashes()
    with pytest.raises(AssertionError, match="builder seed list"):
        A.hash_guard(new, [])


# ---------------------------------------------------------------- mutants

def scn_values_subset(mod, tmp_path=None):
    t = make_table(genre_missing=("Zulu",))
    try:
        res = mod.build_seed_a28(prep(mod, t), V, 9001, n_per_pair=N)
    except BaseException:
        return False
    return res.set_aside["emotion__genre"] == ["Zulu"] and sum(res.shares["emotion__genre"].values()) == N


def scn_failed_draw(mod):
    t = make_table(tiny=("arabic",))
    try:
        res = mod.build_seed_a28(prep(mod, t), V, 9001, n_per_pair=N)
    except BaseException:
        return False
    return res.set_aside["emotion__style"] == ["arabic"] and sum(res.shares["emotion__style"].values()) == N


def _faulty_run(mod, rules, monkeypatch):
    f = Faulty(rules)
    monkeypatch.setattr(mod.E, "build_aspect_episodes_r6", f)
    return mod.build_seed_a28(prep(mod, make_table()), V, 9001, n_per_pair=N)


def scn_other_assertion_propagates(mod, monkeypatch):
    try:
        _faulty_run(mod, {(2, None): AssertionError("episode 3: a painting appears twice")}, monkeypatch)
    except AssertionError:
        return True
    return False


def scn_other_runtime_propagates(mod, monkeypatch):
    try:
        _faulty_run(mod, {(2, None): RuntimeError("emotion x style: need more than 4 eligible values per aspect, got 3 and 8")},
                    monkeypatch)
    except RuntimeError:
        return True
    return False


def scn_value_error_propagates(mod, monkeypatch):
    try:
        _faulty_run(mod, {(2, None): ValueError("x")}, monkeypatch)
    except ValueError:
        return True
    return False


def scn_third_stops(mod, monkeypatch):
    rules = {(l, None): RuntimeError(FAILED_DRAW) for l in (1, 2, 3)}
    try:
        _faulty_run(mod, rules, monkeypatch)
    except mod.SetAsideStop:
        return True
    return False


def scn_two_pass(mod, monkeypatch):
    rules = {(l, None): RuntimeError(FAILED_DRAW) for l in (1, 2)}
    try:
        res = _faulty_run(mod, rules, monkeypatch)
    except BaseException:
        return False
    return all(set(res.set_aside[p]) == {"Burmese", "Hindi"} for p in PAIRS)


def scn_resplit(mod, monkeypatch):
    res = _faulty_run(mod, {(2, None): RuntimeError(FAILED_DRAW)}, monkeypatch)
    return all(sum(res.shares[p].values()) == N and "Hindi" not in res.shares[p] for p in PAIRS)


def scn_later_pass_set_aside(mod, monkeypatch):
    rules = {(2, None): RuntimeError(FAILED_DRAW), (1, None): lambda n: RuntimeError(FAILED_DRAW) if n > 12 else None}
    res = _faulty_run(mod, rules, monkeypatch)
    return all(set(res.set_aside[p]) == {"Hindi", "Burmese"} for p in PAIRS)


def scn_original_seeds(mod, monkeypatch):
    f = Faulty({(2, None): RuntimeError(FAILED_DRAW)})
    monkeypatch.setattr(mod.E, "build_aspect_episodes_r6", f)
    mod.build_seed_a28(prep(mod, make_table()), V, 9001, n_per_pair=N)
    return [c["seed"] for c in f.calls if c["pair"] == "emotion__style"] == \
        [9001001, 9001002, 9001003, 9001004, 9001001, 9001003, 9001004]


def scn_member_in_dropped_row(mod, monkeypatch):
    msg = raised(lambda: member_in_dropped_row(mod, monkeypatch))
    return msg is not None and "outside" in msg


def scn_h5_clash(mod, tmp_path):
    return scn_h5(mod, ledger_files(tmp_path)) is not None


def scn_recorded_clash(mod, tmp_path):
    return scn_recorded(mod, ledger_files(tmp_path)) is not None


def scn_new_repeat(mod, tmp_path):
    return scn_distinct_new(mod, ledger_files(tmp_path)) is not None


def scn_seed_list_mutant(mod, tmp_path):
    return scn_seed_list_is_the_reads(mod, ledger_files(tmp_path))


def scn_seed_clash(mod, tmp_path):
    return scn_seed_in_other_row(mod, ledger_files(tmp_path)) is not None and \
        scn_seed_in_plain_row(mod, ledger_files(tmp_path)) is not None


def scn_round9_ok(mod, tmp_path):
    return scn_round9_row_is_allowed(mod, ledger_files(tmp_path)) is None


def scn_free_ok(mod, tmp_path):
    return scn_free_row_is_allowed(mod, ledger_files(tmp_path)) is None


def scn_seed_repeat(mod, tmp_path):
    return scn_repeated_seed(mod, ledger_files(tmp_path)) is not None


def scn_h5_nine(mod, tmp_path):
    p = tmp_path / "held_bad.md"
    p.write_text(held_ledger().replace(f"style × genre {H('2')}", "style × genre 0000"))
    return raised(lambda: mod.h5_hashes(p)) is not None


def ledger_files(tmp_path):
    s, h = tmp_path / "seed_ledger.md", tmp_path / "held_ledger.md"
    s.write_text(SEED_LEDGER)
    h.write_text(held_ledger())
    return s, h


# (name, old text, new text, scenario, fixtures the scenario takes); a scenario returns True when the behaviour is right
SUBSET_COND = 'if isinstance(exc, AssertionError) and "are not eligible on the pool" in str(exc):'
DRAW_COND = 'if isinstance(exc, RuntimeError) and "could not fill" in str(exc):'
STOP = "if len(set_aside) > MAX_SET_ASIDE:  # guard:stop"
MUTANTS = [
    ("catch_values_subset_off", SUBSET_COND, "if False:", scn_values_subset, "none"),
    ("catch_failed_draw_off", DRAW_COND, "if False:", scn_failed_draw, "none"),
    ("any_assertion_is_set_aside", SUBSET_COND, "if isinstance(exc, AssertionError):", scn_other_assertion_propagates, "mp"),
    ("any_runtime_error_is_set_aside", DRAW_COND, "if isinstance(exc, RuntimeError):", scn_other_runtime_propagates, "mp"),
    ("value_error_is_set_aside", "    return None  # mut:no_other",
     "    return 'failed_draws' if isinstance(exc, ValueError) else None", scn_value_error_propagates, "mp"),
    ("stop_at_four", STOP, "if len(set_aside) > MAX_SET_ASIDE + 1:  # guard:stop", scn_third_stops, "mp"),
    ("stop_at_two", STOP, "if len(set_aside) >= MAX_SET_ASIDE:  # guard:stop", scn_two_pass, "mp"),
    ("stop_removed", STOP, "if False:  # guard:stop", scn_third_stops, "mp"),
    ("shares_not_re_split", "sizes = [ctx.pool_size[l - 1] for l in active]",
     "sizes = [ctx.pool_size[l - 1] for l in range(1, L + 1)]", scn_resplit, "mp"),
    ("one_pass_only", "if not failed:  # mut:last_pass", "if True:  # mut:last_pass", scn_later_pass_set_aside, "mp"),
    ("renumbered_seeds", "seed = 1000 * int(s) + l", "seed = 1000 * int(s) + active.index(l) + 1", scn_original_seeds, "mp"),
    ("member_check_removed", "_require(in_kept[every].all(), ", "_require(True or in_kept[every].all(), ",
     scn_member_in_dropped_row, "mp"),
    ("pool_counts_dropped_rows", "out.pool_size.append(int(known.sum()))",
     'out.pool_size.append(int(((sub["emotion"] >= 0) & (sub["style"] >= 0) & (sub["genre"] >= 0)).sum()))',
     scn_pool_after_drops, "mp"),
    ("seed_list_not_checked", "_require(list(builder_seed_list) == expected_seeds,",
     "_require(True or list(builder_seed_list) == expected_seeds,", scn_seed_list_mutant, "tmp"),
    ("h5_not_checked", "_require(not h5_clash, ", "_require(True or not h5_clash, ", scn_h5_clash, "tmp"),
    ("recorded_not_used", "distinct = E.assert_distinct(new, recorded)  # guard:recorded",
     "distinct = E.assert_distinct(new, h5)  # guard:recorded", scn_recorded_clash, "tmp"),
    ("assert_distinct_removed", "distinct = E.assert_distinct(new, recorded)  # guard:recorded",
     "distinct = {}  # guard:recorded", scn_new_repeat, "tmp"),
    ("seeds_not_checked", "seeds = assert_seeds_free(builder_seed_list, seed_ledger)  # guard:seeds",
     "seeds = {'n_seeds': 0}  # guard:seeds", scn_seed_clash, "tmp"),
    ("ledger_clash_ignored", "_require(not clash, ", "_require(True or not clash, ", scn_seed_clash, "tmp"),
    ("round9_rows_block", 'if "round 9" in line.lower():  # guard:round9', 'if False:  # guard:round9', scn_round9_ok, "tmp"),
    ("free_rows_block", 'if status == "free":  # guard:free', 'if False:  # guard:free', scn_free_ok, "tmp"),
    ("repeated_seed_allowed", "_require(len(set(seeds)) == len(seeds)", "_require(True or len(set(seeds)) == len(seeds)",
     scn_seed_repeat, "tmp"),
    ("h5_count_not_checked", "_require(len(flat) == 9", "_require(True or len(flat) == 9", scn_h5_nine, "tmp"),
]


def call(scn, kind, mod, monkeypatch, tmp_path):
    return {"none": lambda: scn(mod), "mp": lambda: scn(mod, monkeypatch), "tmp": lambda: scn(mod, tmp_path)}[kind]()


@pytest.mark.parametrize("name, old, new, scn, kind", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_mutant_is_killed(name, old, new, scn, kind, tmp_path, monkeypatch):
    mod = load_copy(tmp_path, replace=[(old, new)])
    try:
        got = call(scn, kind, mod, monkeypatch, tmp_path)
    except Exception:
        got = False
    assert got is False, name


@pytest.mark.parametrize("name, old, new, scn, kind", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_the_unmutated_module_passes_the_mutant_scenario(name, old, new, scn, kind, tmp_path, monkeypatch):
    assert call(scn, kind, A, monkeypatch, tmp_path) is True, name
