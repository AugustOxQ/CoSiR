"""Tests of r28_clay_crl: the encoder switch (CLAY, CRL) and the replay of an existing CLIP L/14 cell of each method.

Synthetic tests prove the wiring of the SigLIP 1 path (width, rank, text set, joins, the checks on the text files). The two
replay tests read the real stores and need the project's data; they check that the CLIP L/14 path still gives r18's CLAY cell
and r17's CRL cell within 1e-6 (the replay of all 24 cells is `r28_clay_crl.py replay`).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r28_clay_crl as m  # noqa: E402


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def write_texts(root, d, n=12, seed=0, mutate=None):
    """A fake text directory of set A and B for the three aspects, with the sidecars `text_rows` checks."""
    rng = np.random.RandomState(seed)
    for ts in m.TEXT_SETS:
        (Path(root) / ts).mkdir(parents=True, exist_ok=True)
        for a in m.ASPECTS:
            x = unit(rng.randn(n, d))
            texts = [f'{ts}-{a}-{i}' for i in range(n)]
            side = dict(texts=texts, shape=[n, d], sha256=m.om_data.array_sha256(x))
            if mutate:
                x, side = mutate(ts, a, x, side)
            np.save(Path(root) / ts / f'{a}_text.npy', x)
            (Path(root) / ts / f'{a}_text.json').write_text(json.dumps(side))


@pytest.fixture
def fake(monkeypatch, tmp_path):
    d, n = 24, 40
    rng = np.random.RandomState(1)
    rows = {}

    def keys(set_, split):
        return [f'{set_}-{split}-{i}' for i in range(n)]

    def plain(set_, split, enc):
        return rows.setdefault((set_, split, enc), unit(rng.randn(n, d)))

    monkeypatch.setattr(m.ns_data, 'keys', keys)
    monkeypatch.setattr(m.ns_data, 'plain', plain)
    monkeypatch.setattr(m, 'WIDTH', {'clip_l14': 768, 'siglip1': d})
    monkeypatch.setattr(m, 'source_texts', lambda ts, a: ([f'{ts}-{a}-{i}' for i in range(12)], Path('src')))
    m._MEMO.clear()
    write_texts(tmp_path, d)
    yield dict(root=tmp_path, d=d, n=n, plain=plain)
    m._MEMO.clear()


def test_siglip_clay_is_om_clay_math_on_siglip_rows(fake):
    for ts in m.TEXT_SETS:
        got = m.clay_vectors('artelingo', 'held', 'style', ts, 'siglip1', fake['root'])
        text = m.text_rows(ts, 'style', 'siglip1', fake['root'])
        P, mu = m.om_clay.subspace(text, m.op.clay_rank(text.shape[0]))
        want = m.om_clay.condition(fake['plain']('artelingo', 'held', 'siglip1'), P, mu)
        assert got.dtype == np.float32 and got.shape == (fake['n'], fake['d'])
        assert np.array_equal(got, want)
        assert np.abs(np.linalg.norm(got.astype(np.float64), axis=1) - 1).max() < 1e-5


def test_clay_alignment_mean_uses_only_the_split_passed(fake):
    a = m.clay_vectors('artelingo', 'held', 'emotion', 'A', 'siglip1', fake['root'])
    b = m.clay_vectors('artelingo', 'val', 'emotion', 'A', 'siglip1', fake['root'])
    assert not np.allclose(a, b)                      # another split: another rows, another mean, other vectors
    assert m.clay_vectors('artelingo', 'held', 'emotion', 'A', 'siglip1', fake['root']) is a    # memo hit, same array


def test_clay_text_sets_differ_and_aspects_differ(fake):
    a = m.clay_vectors('mmart', 'test', 'genre', 'A', 'siglip1', fake['root'])
    assert not np.allclose(a, m.clay_vectors('mmart', 'test', 'genre', 'B', 'siglip1', fake['root']))
    assert not np.allclose(a, m.clay_vectors('mmart', 'test', 'style', 'A', 'siglip1', fake['root']))


def test_siglip_crl_is_ac_crl_vector_on_set_a(fake, monkeypatch):
    monkeypatch.setattr(m.rt_crl, 'KEEP', 12)      # the fake text sets have 12 values, the real ones 50
    got = m.crl_vectors('mmart', 'val', 'emotion', 'siglip1', fake['root'])
    x = fake['plain']('mmart', 'val', 'siglip1').astype(np.float64)
    t = m.text_rows('A', 'emotion', 'siglip1', fake['root']).astype(np.float64)
    cos = x @ t.T
    want = (cos / np.linalg.norm(cos, axis=1, keepdims=True)).astype(np.float32)
    assert got.shape == (fake['n'], 12) and np.array_equal(got, want)


def test_unknown_encoder_is_refused(fake):
    with pytest.raises(AssertionError):
        m.clay_vectors('artelingo', 'held', 'style', 'A', 'siglip2', fake['root'])
    with pytest.raises(AssertionError):
        m.crl_vectors('artelingo', 'held', 'style', 'qwen', fake['root'])


def test_text_rows_refuse_a_changed_array_text_or_width(fake, tmp_path):
    root = tmp_path / 'bad'
    write_texts(root, fake['d'], mutate=lambda ts, a, x, side: (x + 0.0 if ts == 'B' else unit(x * 1.0 + 1e-3), side))
    m.text_rows('B', 'style', 'siglip1', root)                                   # untouched array passes
    with pytest.raises(AssertionError):
        m.text_rows('A', 'style', 'siglip1', root)                               # array no longer equals its sidecar hash
    root2 = tmp_path / 'bad2'
    write_texts(root2, fake['d'], mutate=lambda ts, a, x, side: (x, dict(side, texts=side['texts'][::-1])))
    with pytest.raises(AssertionError):
        m.text_rows('A', 'emotion', 'siglip1', root2)                            # the text list is not the source list
    root3 = tmp_path / 'bad3'
    write_texts(root3, fake['d'] + 1)
    with pytest.raises(AssertionError):
        m.text_rows('A', 'emotion', 'siglip1', root3)                            # wrong width


def test_compare_flags_a_difference_above_the_tolerance():
    cells = {'a': dict(value=0.5), 'b': dict(value=0.25)}
    ok = m.compare(cells, {'a': 0.5 + 5e-7, 'b': 0.25})
    assert ok['ok'] and ok['n'] == 2 and ok['max_abs_diff'] == pytest.approx(5e-7)
    bad = m.compare(cells, {'a': 0.5 + 2e-6, 'b': 0.25})
    assert not bad['ok']
    assert not m.compare(cells, {'zzz': 1.0})['ok']                              # nothing matched: not a pass


# ---- replay of existing CLIP L/14 cells (real data) ---------------------------------------------------------------
needs_data = pytest.mark.skipif(not m.op.OUT.exists() or not Path(m.OLD_PICKS).exists(), reason='r18 outputs not on this machine')


@needs_data
def test_clip_path_replays_one_clay_and_one_crl_cell():
    rec = m.recorded_values()
    picks = m.recorded_picks()
    cells = m.score_cells(picks, 'clip_l14', sets=('artelingo',), methods=('clay', 'crl'), aspects=('emotion',), tasks=('ami',))
    assert set(cells) == {'artelingo/emotion/ami/clay', 'artelingo/emotion/ami/crl'}
    for cid in cells:
        assert abs(cells[cid]['value'] - rec[cid]) <= m.TOL_REPLAY, cid


def test_default_encoder_is_clip_l14():
    import inspect
    for f in (m.clay_vectors, m.crl_vectors, m.images, m.text_rows, m.score_cells, m.make_picks):
        assert inspect.signature(f).parameters['enc'].default == 'clip_l14', f.__name__


def test_fix_records_renames_only_the_retrieval_unit(tmp_path):
    """The clustering text-set rule names no unit and is left as it is; the retrieval rule gets the encoder's own unit;
    table paths become relative to `out` (regression: the first fix wave asserted the unit on every rule)."""
    out, tables = tmp_path, tmp_path / 'siglip1'
    (tables / 'stage1').mkdir(parents=True)
    (tables / 'stage1' / 't.json').write_text('{}')
    clu = 'B only if the mean over aspects of validation AMI at 20 exceeds A by more than 0.001'
    picks = {'clay_textset/clu/artelingo': dict(set='A', rule=clu),
             'clay_textset/ret/artelingo': dict(set='A', rule='mAP / CLIP L/14 c* range exceeds A by 0.1'),
             'crl/artelingo': dict(pick='none+k10', table_path='stage1/t.json')}
    got = m.fix_records(picks, 'siglip1', tables, out)
    assert got['clay_textset/clu/artelingo']['rule'] == clu
    assert 'CLIP L/14' not in got['clay_textset/ret/artelingo']['rule']
    assert "siglip1's own c* validation mAP range" in got['clay_textset/ret/artelingo']['rule']
    assert got['crl/artelingo']['table_path'] == 'siglip1/stage1/t.json'
    keep = m.fix_records({'clay_textset/ret/x': dict(rule='a CLIP L/14 c* range b')}, 'clip_l14', tables, out)
    assert keep['clay_textset/ret/x']['rule'] == 'a CLIP L/14 c* range b'
