"""Ticket 05 tests: the scoring runner on tiny synthetic stores (no data, CPU)."""
import numpy as np
import pytest

import tcl_paths as P
import tcl_score as S


# ------------------------------------------------------------------------------------------------ the mean flip
def test_means_pair_is_query_then_gallery_for_each_direction():
    t, i = np.full(3, 1, np.float32), np.full(3, 2, np.float32)
    q, g = S.query_gallery_means('t2i', t, i)
    assert q is t and g is i
    q, g = S.query_gallery_means('i2t', t, i)          # fit_means gives (text, image); i2t queries are images
    assert q is i and g is t
    q, g = S.query_gallery_means('t2t', t, i)          # one mean for text to text: the text mean, twice
    assert q is t and g is t


# ------------------------------------------------------------------------------------------------ synthetic store
def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def make_store(seed=0, n_i=6, d=16, gap=1.5, n_unlabelled=2):
    """A tiny set: 12 texts in non-adjacent order (two per painting), a modality gap so the means matter, some -1 labels."""
    rng = np.random.default_rng(seed)
    tp = np.array([3, 0, 5, 1, 2, 4, 0, 5, 3, 1, 4, 2], dtype=np.int64)[:2 * n_i]
    n_t = len(tp)
    base = rng.normal(size=(n_i, d))
    image = unit(base + 0.5 * rng.normal(size=(n_i, d)) + gap * np.eye(d)[0])
    text = unit(base[tp] + 0.8 * rng.normal(size=(n_t, d)) - gap * np.eye(d)[1])
    labels = {}
    for a, k in zip(P.ASPECTS, (3, 4, 2)):
        li = rng.integers(0, k, size=n_i).astype(np.int64)
        lt = li[tp].copy() if a != 'emotion' else rng.integers(0, k, size=n_t).astype(np.int64)
        li[rng.choice(n_i, n_unlabelled, replace=False)] = -1
        lt[rng.choice(n_t, n_unlabelled, replace=False)] = -1
        labels[a] = dict(t=lt, i=li)
    fr = dict(n_i=n_i, n_t=n_t, tp=tp, ip=np.arange(n_i, dtype=np.int64), labels=labels)
    flags = {a: rng.random(n_t) < 0.3 for a in P.ASPECTS}
    return fr, flags, text, image


def ref_orders(q, g, mq, mg, readout, self_match):
    """Brute force, from the spec: centre, normalise, rank, expand on the top 10 (the query plus its top 10 gallery rows,
    normalised), rank again; the query's own row never counts."""
    if readout != 'r0':
        q, g = unit(q - mq), unit(g - mg)
    orders = []
    for n in range(len(q)):
        def rank(vec):
            order = np.argsort(-(g @ vec).astype(np.float64), kind='stable')
            return order[order != n] if self_match else order
        order = rank(q[n])
        if readout == 'center_k10':
            order = rank(unit((q[n] + g[order[:10]].sum(0))[None])[0])
        orders.append(order)
    return orders


def ref_first_rank(q, g, qp, gp, mq, mg, readout, self_match):
    out = np.full(len(q), np.nan, np.float32)
    for n, order in enumerate(ref_orders(q, g, mq, mg, readout, self_match)):
        hits = np.nonzero(gp[order] == qp[n])[0]
        if len(hits):
            out[n] = hits[0] + 1
    return out


def ref_ap(orders, ql, gl, q_ok, g_ok, qp=None, gp=None):
    """Average precision per query by definition: over the valid gallery rows in rank order, the mean of the precision at
    each relevant row; NaN for a query that is unlabelled, filtered, or has nothing relevant."""
    out = np.full(len(ql), np.nan, np.float32)
    for n, order in enumerate(orders):
        valid = g_ok[order] & (True if qp is None else gp[order] != qp[n])
        rel = valid & (gl[order] == ql[n])
        if not q_ok[n] or not rel.any():
            continue
        prec = np.cumsum(rel) / np.maximum(np.cumsum(valid), 1)
        out[n] = prec[rel].mean()
    return out


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture
def store():
    return make_store()


def score(tmp_path, store, set_name='art_val', row='cstar', head='emotion', seed=0, text=None, image=None, **kw):
    fr, flags, t, i = store
    S.score_unit(set_name, 'caption', row, head, seed, t if text is None else text, i if image is None else image,
                 fr, flags, 'cpu', out=tmp_path, **kw)


# ------------------------------------------------------------------------------------------------ one head row
@pytest.mark.parametrize('direction', P.DIRECTIONS)
@pytest.mark.parametrize('readout', P.READOUTS)
def test_validation_file_ranks_with_its_own_validation_means(tmp_path, store, direction, readout):
    fr, flags, text, image = store
    score(tmp_path, store)
    got = load(P.score_path('art_val', 'caption', 'cstar', 'emotion', 0, direction, readout, out=tmp_path))
    mt, mi = text.mean(0, dtype=np.float64).astype(np.float32), image.mean(0, dtype=np.float64).astype(np.float32)
    q, g, qp, gp, mq, mg = {
        't2i': (text, image, fr['tp'], fr['ip'], mt, mi),
        'i2t': (image, text, fr['ip'], fr['tp'], mi, mt),
        't2t': (text, text, fr['tp'], fr['tp'], mt, mt)}[direction]
    assert np.array_equal(got['q_pid'], qp) and got['q_pid'].dtype == np.int64
    want = ref_first_rank(q, g, qp, gp, mq, mg, readout, direction == 't2t')
    assert got['first_rank'].dtype == np.float32
    assert np.array_equal(got['first_rank'], want, equal_nan=True)
    assert set(got) == {'q_pid', 'first_rank', 'ap__emotion', 'ap__emotion__no_leak', 'ap__emotion__cross_painting'}


# ------------------------------------------------------------------------------------------------ held sets
VAL_N = (12, 6)


@pytest.mark.parametrize('direction', P.DIRECTIONS)
def test_held_set_centres_with_the_validation_file_means_never_its_own_rows(tmp_path, direction):
    val, held = make_store(0), make_store(1)
    score(tmp_path, val)
    score(tmp_path, held, set_name='art_held', val_n=VAL_N)
    fr, _, text, image = held
    _, _, vtext, vimage = val
    mt, mi = (x.mean(0, dtype=np.float64).astype(np.float32) for x in (vtext, vimage))
    q, g, qp, gp, mq, mg = {
        't2i': (text, image, fr['tp'], fr['ip'], mt, mi),
        'i2t': (image, text, fr['ip'], fr['tp'], mi, mt),
        't2t': (text, text, fr['tp'], fr['tp'], mt, mt)}[direction]
    got = load(P.score_path('art_held', 'caption', 'cstar', 'emotion', 0, direction, 'center', out=tmp_path))
    assert np.array_equal(got['first_rank'], ref_first_rank(q, g, qp, gp, mq, mg, 'center', direction == 't2t'),
                          equal_nan=True)
    own = ref_first_rank(q, g, qp, gp, q.mean(0), q.mean(0) if direction == 't2t' else g.mean(0), 'center',
                         direction == 't2t')
    assert not np.array_equal(got['first_rank'], own, equal_nan=True)     # the held rows' own means would give other ranks
    assert not list((tmp_path / 'means').glob('art_held*'))               # a held set writes no means


def test_held_set_without_a_validation_means_file_crashes(tmp_path, store):
    with pytest.raises(FileNotFoundError):
        score(tmp_path, store, set_name='art_held', val_n=VAL_N)
    assert not (tmp_path / 'scores').exists()


def test_held_means_are_those_of_the_same_row_head_and_seed(tmp_path, store):
    score(tmp_path, store, seed=1)
    with pytest.raises(FileNotFoundError):
        score(tmp_path, store, set_name='art_held', seed=0, val_n=VAL_N)
    with pytest.raises(FileNotFoundError):
        score(tmp_path, store, set_name='art_held', seed=1, head='style', val_n=VAL_N)


def test_held_means_row_counts_must_match_the_validation_rows(tmp_path, store):
    score(tmp_path, store)
    with pytest.raises(AssertionError):
        score(tmp_path, store, set_name='art_held', val_n=(13, 6))
    with pytest.raises(AssertionError):
        score(tmp_path, store, set_name='art_held', val_n=(12, 7))
    with pytest.raises(AssertionError):
        score(tmp_path, store, set_name='art_held')            # a held set must say what it expects


# ------------------------------------------------------------------------------------------------ the three entries
@pytest.mark.parametrize('direction', P.DIRECTIONS)
@pytest.mark.parametrize('readout', ('r0', 'center_k10'))
def test_entries_follow_r22_with_unlabelled_rows_ranked_and_masked_only_at_ap(tmp_path, store, direction, readout):
    fr, flags, text, image = store
    score(tmp_path, store)
    got = load(P.score_path('art_val', 'caption', 'cstar', 'emotion', 0, direction, readout, out=tmp_path))
    mt, mi = text.mean(0, dtype=np.float64).astype(np.float32), image.mean(0, dtype=np.float64).astype(np.float32)
    q, g, qs, gs, mq, mg = {'t2i': (text, image, 't', 'i', mt, mi), 'i2t': (image, text, 'i', 't', mi, mt),
                            't2t': (text, text, 't', 't', mt, mt)}[direction]
    qp, gp = (fr['tp'] if qs == 't' else fr['ip']), (fr['tp'] if gs == 't' else fr['ip'])
    orders = ref_orders(q, g, mq, mg, readout, direction == 't2t')
    assert all(len(o) == len(g) - (direction == 't2t') for o in orders)          # unlabelled rows are in every ranking
    lab = fr['labels']['emotion']
    ql, gl = lab[qs], lab[gs]
    q_named = ~flags['emotion'] if qs == 't' else np.ones(len(q), bool)
    g_named = ~flags['emotion'] if gs == 't' else np.ones(len(g), bool)
    want = {'ap__emotion': ref_ap(orders, ql, gl, ql >= 0, gl >= 0),
            'ap__emotion__no_leak': ref_ap(orders, ql, gl, (ql >= 0) & q_named, (gl >= 0) & g_named),
            'ap__emotion__cross_painting': ref_ap(orders, ql, gl, ql >= 0, gl >= 0, qp, gp)}
    for key, value in want.items():
        assert got[key].dtype == np.float32
        assert np.allclose(got[key], value, atol=1e-6, equal_nan=True), key
        assert np.isnan(got[key]).any() and np.isfinite(got[key]).any(), key
    assert np.isnan(got['ap__emotion'][ql < 0]).all()                            # unlabelled queries: not scored
    assert np.isfinite(got['first_rank'][ql < 0]).any() or not (ql < 0).any()    # ...but they still have a pair rank


def test_entries_differ_from_the_bar_entry(tmp_path, store):
    score(tmp_path, store)
    got = load(P.score_path('art_val', 'caption', 'cstar', 'emotion', 0, 't2i', 'r0', out=tmp_path))
    assert not np.allclose(got['ap__emotion'], got['ap__emotion__cross_painting'], equal_nan=True)
    assert not np.allclose(got['ap__emotion'], got['ap__emotion__no_leak'], equal_nan=True)


def test_frame_and_flags_are_not_written(tmp_path, store):
    fr, flags, text, image = store
    before = (fr['tp'].copy(), {a: {k: v.copy() for k, v in d.items()} for a, d in fr['labels'].items()},
              {a: f.copy() for a, f in flags.items()}, text.copy(), image.copy())
    score(tmp_path, store)
    assert np.array_equal(fr['tp'], before[0]) and np.array_equal(text, before[3]) and np.array_equal(image, before[4])
    assert all(np.array_equal(fr['labels'][a][k], before[1][a][k]) for a in P.ASPECTS for k in 'ti')
    assert all(np.array_equal(flags[a], before[2][a]) for a in P.ASPECTS)


# ------------------------------------------------------------------------------------------------ keys per row type
def keys_of(aspects):
    return {'q_pid', 'first_rank'} | {f'ap__{a}{s}' for a in aspects for s in ('', '__no_leak', '__cross_painting')}


def files_under(tmp_path, set_name, row):
    return sorted(p.relative_to(tmp_path / 'scores' / set_name / 'caption' / row).as_posix()
                  for p in (tmp_path / 'scores' / set_name / 'caption' / row).glob('*.npz'))


def test_a_one_vector_row_reads_all_three_aspects(tmp_path, store):
    score(tmp_path, store, row='plain', head='all', seed='x')
    assert len(files_under(tmp_path, 'art_val', 'plain')) == 9
    for d in P.DIRECTIONS:
        for r in P.READOUTS:
            assert set(load(P.score_path('art_val', 'caption', 'plain', 'all', 'x', d, r, out=tmp_path))) == keys_of(P.ASPECTS)


def test_a_head_row_reads_its_own_heads_aspect_only(tmp_path, store):
    score(tmp_path, store, row='teacher', head='style', seed='x')
    assert files_under(tmp_path, 'art_val', 'teacher') == sorted(
        f'style__sx__{d}__{r}.npz' for d in P.DIRECTIONS for r in P.READOUTS)
    got = load(P.score_path('art_val', 'caption', 'teacher', 'style', 'x', 'i2t', 'center', out=tmp_path))
    assert set(got) == keys_of(('style',))


def test_roberta_ranks_text_to_text_on_emotion_with_no_image(tmp_path, store):
    fr, flags, text, _ = store
    S.score_unit('art_val', 'caption', 'roberta', 'all', 'x', text, None, fr, flags, 'cpu', out=tmp_path)
    assert files_under(tmp_path, 'art_val', 'roberta') == sorted(f'all__sx__t2t__{r}.npz' for r in P.READOUTS)
    got = load(P.score_path('art_val', 'caption', 'roberta', 'all', 'x', 't2t', 'center', out=tmp_path))
    assert set(got) == keys_of(('emotion',))
    mt = text.mean(0, dtype=np.float64).astype(np.float32)
    assert np.array_equal(got['first_rank'], ref_first_rank(text, text, fr['tp'], fr['tp'], mt, mt, 'center', True),
                          equal_nan=True)
    # its held set reads the same validation file, and needs no image count
    S.score_unit('art_held', 'caption', 'roberta', 'all', 'x', text, None, fr, flags, 'cpu', out=tmp_path, val_n=VAL_N)
    assert files_under(tmp_path, 'art_held', 'roberta') == files_under(tmp_path, 'art_val', 'roberta')


def test_late_fusion_means_act_on_the_whole_concatenated_vector(tmp_path, store):
    fr, flags, text, image = store
    rng = np.random.default_rng(5)
    wide = lambda x: unit(np.concatenate([np.sqrt(.3) * x, np.sqrt(.7) * unit(x + .3 * rng.normal(size=x.shape))], axis=1))
    lt, li = wide(text), wide(image)
    S.score_unit('art_val', 'caption', 'late', 'emotion', 2, lt, li, fr, flags, 'cpu', out=tmp_path)
    m = load(P.means_path('art_val', 'caption', 'late', 'emotion', 2, out=tmp_path))
    assert m['t'].shape == m['i'].shape == (32,) and m['t'].dtype == np.float32
    assert (int(m['n_t']), int(m['n_i'])) == (12, 6)
    assert np.allclose(m['t'], lt.mean(0), atol=1e-6) and np.allclose(m['i'], li.mean(0), atol=1e-6)
    got = load(P.score_path('art_val', 'caption', 'late', 'emotion', 2, 't2i', 'center', out=tmp_path))
    want = ref_first_rank(lt, li, fr['tp'], fr['ip'], m['t'], m['i'], 'center', False)
    assert np.array_equal(got['first_rank'], want, equal_nan=True)


def test_mismatched_rows_crash(tmp_path, store):
    fr, flags, text, image = store
    with pytest.raises(AssertionError):
        S.score_unit('art_val', 'caption', 'plain', 'all', 'x', text[:-1], image, fr, flags, 'cpu', out=tmp_path)
    with pytest.raises(AssertionError):
        S.score_unit('art_val', 'caption', 'plain', 'all', 'x', text, None, fr, flags, 'cpu', out=tmp_path)
    with pytest.raises(AssertionError):                 # a head row with the head `all`
        S.score_unit('art_val', 'caption', 'cstar', 'all', 0, text, image, fr, flags, 'cpu', out=tmp_path)


# ------------------------------------------------------------------------------------------------ resuming
def all_files(tmp_path):
    return sorted((tmp_path / 'scores').rglob('*'))


def stamp(paths):
    return {p: p.stat().st_mtime_ns for p in paths if p.is_file()}


def test_a_complete_unit_is_skipped_and_leaves_no_temporary_file(tmp_path, store):
    score(tmp_path, store)
    first = stamp(all_files(tmp_path)) | stamp(sorted((tmp_path / 'means').rglob('*')))
    assert len(first) == 10 and not [p for p in first if p.name.endswith('.tmp')]
    assert S.unit_done('art_val', 'caption', 'cstar', 'emotion', 0, out=tmp_path)
    score(tmp_path, store)
    assert stamp(all_files(tmp_path)) | stamp(sorted((tmp_path / 'means').rglob('*'))) == first


def test_only_the_missing_or_incomplete_files_are_redone(tmp_path, store):
    score(tmp_path, store)
    done = P.score_path('art_val', 'caption', 'cstar', 'emotion', 0, 't2i', 'r0', out=tmp_path)
    gone = P.score_path('art_val', 'caption', 'cstar', 'emotion', 0, 'i2t', 'center', out=tmp_path)
    half = P.score_path('art_val', 'caption', 'cstar', 'emotion', 0, 't2t', 'center_k10', out=tmp_path)
    want = {p: load(p) for p in (done, gone, half)}
    before = stamp([done])
    gone.unlink()
    np.savez(half, q_pid=want[half]['q_pid'])                                  # a file without its scores
    half.with_name(half.name + '.tmp').write_bytes(b'left by a killed job')
    assert not S.unit_done('art_val', 'caption', 'cstar', 'emotion', 0, out=tmp_path)
    score(tmp_path, store)
    assert stamp([done]) == before
    for p in (done, gone, half):
        assert set(load(p)) == set(want[p]) and all(np.array_equal(load(p)[k], want[p][k], equal_nan=True) for k in want[p])
    assert not list((tmp_path / 'scores').rglob('*.tmp'))
    assert S.unit_done('art_val', 'caption', 'cstar', 'emotion', 0, out=tmp_path)


def test_a_validation_unit_without_its_means_is_not_done(tmp_path, store):
    score(tmp_path, store)
    P.means_path('art_val', 'caption', 'cstar', 'emotion', 0, out=tmp_path).unlink()
    assert not S.unit_done('art_val', 'caption', 'cstar', 'emotion', 0, out=tmp_path)


# ------------------------------------------------------------------------------------------------ which rankings
def n_rankings(set_name, **kw):
    return sum(len(S._plan(row, head)[1]) * len(P.READOUTS) for row, head, seed in S.plan(set_name, **kw))


def test_plan_art_sets_score_bar_rows_and_side_rows_on_validation_and_held():
    for s in ('art_val', 'art_held'):
        units = S.plan(s)
        assert len(units) == 15 + 15 + 1 + 3 + 15 + 3 + 1       # cstar, late, plain, teacher; free, 3 plain side rows, roberta
        assert n_rankings(s) == 306 + 165                       # bar 135 + 135 + 9 + 27; side 135 + 3 * 9 + 3
        assert ('roberta', 'all', 'x') in units and ('free', 'style', 4) in units
    assert n_rankings('art_val', rows=('roberta',)) == 3


def test_plan_mmart_sets_score_the_bar_rows_only():
    for s in ('mm_val', 'mm_test'):
        units = S.plan(s)
        assert len(units) == 15 + 15 + 1 + 3 and n_rankings(s) == 306
        assert not {r for r, _, _ in units} & set(P.SIDE_ROWS)


def test_plan_rows_filter_keeps_order_and_refuses_unknown_rows():
    assert S.plan('art_val', rows=('plain', 'cstar'))[0] == ('cstar', 'emotion', 0)
    assert [u for u in S.plan('art_val', rows=('plain',))] == [('plain', 'all', 'x')]
    with pytest.raises(AssertionError):
        S.plan('mm_val', rows=('roberta',))
    with pytest.raises(AssertionError):
        S.plan('art_val', rows=('late_fusion',))


# ------------------------------------------------------------------------------------------------ against evaluate
@pytest.mark.parametrize('direction', P.DIRECTIONS)
@pytest.mark.parametrize('readout', P.READOUTS)
def test_validation_file_holds_what_evaluate_returns_without_means(tmp_path, store, direction, readout):
    """On validation, evaluate's own centring (means=None) and the fitted means agree: exactly at r0, which never reads the
    means, and to float rounding when centring."""
    from tcl_metrics import evaluate
    fr, flags, text, image = store
    score(tmp_path, store)
    got = load(P.score_path('art_val', 'caption', 'cstar', 'emotion', 0, direction, readout, out=tmp_path))
    q, g = (text if direction[0] == 't' else image), (text if direction[2] == 't' else image)
    pid = dict(t=fr['tp'], i=fr['ip'])
    want = evaluate(q, g, pid[direction[0]], pid[direction[2]], S._entries(direction, ('emotion',), fr, flags),
                    direction == 't2t', readout, 'cpu')
    assert np.array_equal(got['first_rank'], want['first_rank'], equal_nan=True)
    for key, value in want['ap'].items():
        assert np.allclose(got[f'ap__{key}'], value, atol=1e-6 if readout != 'r0' else 0, equal_nan=True)
