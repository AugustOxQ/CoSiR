"""Tests for ns_clu_cells (r17 ticket 07): which rows and partitions are paired, which r, which cell ids.

Verdict code: each test names the logic it guards and fails when that logic is broken. The loaders and the k-means are
replaced by small synthetic fakes that record how they were called; the bootstrap itself is ns_clu's (ticket 06).
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m pytest -q test_ns_clu_cells.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as P  # noqa: E402

P.add_rounds()
import ns_clu  # noqa: E402
import ns_clu_cells as CC  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402
import ns_outcome  # noqa: E402
import ns_photo  # noqa: E402

N, DRAWS = 200, 6
NOT_HELD = 'a held or test split was read by a range'


class Fake:
    """Replaces the loaders and the k-means of the cell code; records every call."""

    def __init__(self, monkeypatch):
        self.calls = []
        mp = monkeypatch
        mp.setattr(ns_clu, 'WORKERS', 1)
        mp.setattr(ns_data, 'keys', lambda s, sp: [f'k{i}' for i in range(N)])
        mp.setattr(ns_data, 'labels', self.labels_data)
        mp.setattr(ns_data, 'cstar', self.cstar)
        mp.setattr(ns_data, 'plain', lambda s, sp, e: self.rows('plain', (s, sp, e), 8))
        mp.setattr(ns_data, 'teacher', lambda s, sp, a: self.rows('teacher', (s, sp, a), 9))
        mp.setattr(ns_data, 'percept_probs', self.probs)
        mp.setattr(ns_data, 'crl', lambda s, sp, a: self.rows('crl', (s, sp, a), 50))
        mp.setattr(ns_photo, 'labels', self.labels_photo)
        mp.setattr(ns_photo, 'photo', self.photo)
        mp.setattr(ns_photo, 'plain', lambda s, sp, e: self.rows('pplain', (s, sp, e), 8))
        mp.setattr(ns_photo, 'teacher', lambda s, sp, e: self.rows('pteacher', (s, sp), 9))
        mp.setattr(ns_metrics, 'partitions', self.partitions)
        mp.setattr(ns_metrics, 'percept_partitions', self.percept_partitions)

    # data
    def log(self, name, *args, **kw):
        self.calls.append((name, args, kw))

    def rows(self, name, key, d):
        self.log(name, *key)
        rng = np.random.default_rng(abs(hash((name,) + key)) % 2 ** 32)
        x = rng.normal(size=(N, d)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def _y(self, k, unlabelled):
        y = np.arange(N) % k
        if unlabelled:
            y[::7] = -1
        return y.astype(np.int64)

    def labels_data(self, s, sp, a):
        self.log('labels', s, sp, a)
        return self._y(P.CLASS_COUNT[(s, a)], unlabelled=(a == 'genre'))

    def labels_photo(self, s, sp):
        self.log('labels', s, sp, 'emotion')
        return self._y(P.CLASS_COUNT[(s, 'emotion')], unlabelled=False)

    def cstar(self, s, sp, e, a, seed, unit=True):
        self.log('cstar', s, sp, e, a, seed, unit=unit)
        return self.rows('cstar_x', (s, sp, e, a, seed), 10)

    def photo(self, s, sp, e, seed):
        self.log('photo', s, sp, e, seed)
        return self.rows('photo_x', (s, sp, e, seed), 10)

    def probs(self, s, sp, e, seed):
        self.log('probs', s, sp, e, seed)
        rng = np.random.default_rng(abs(hash(('p', s, sp, e, seed))) % 2 ** 32)
        return rng.random((N, 40)).astype(np.float32)

    # k-means
    def _parts(self, x, k):
        rng = np.random.default_rng(int(np.abs(x.astype(np.float64)).sum() * 1000) % 2 ** 32)
        return rng.integers(0, k, size=(3, len(x))).astype(np.int32)

    def partitions(self, x, k, set_name):
        self.log('partitions', x.shape[1], k, set_name)
        return self._parts(x, k)

    def percept_partitions(self, probs, k, set_name):
        self.log('percept', probs.shape[1], k, set_name)
        return self._parts(probs, k)

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def fake(monkeypatch):
    return Fake(monkeypatch)


def ranges_for(set_, enc, value=0.01):
    """Stage 1 style ranges for every class-count range key of (set, enc)."""
    out = {}
    for c in ns_outcome.claim_cells():
        k = ns_outcome.range_key(c)
        if k and c['set'] == set_ and c['encoder'] == enc:
            out[k] = dict(r=value + 0.001 * len(out), per_seed={}, k=c['k'], n_val=1)
    return out


def claim_ids(set_, enc, scs):
    return [c['id'] for c in ns_outcome.claim_cells() if c['task'] == 'clu' and c['set'] == set_
            and c['encoder'] == enc and c['sc'] in scs]


# ------------------------------------------------------------------------------------------------ cell ids
@pytest.mark.parametrize('set_,enc,scs', [
    ('artelingo', 'qwen', ('SC1', 'SC2', 'SC3')),
    ('mmart', 'siglip1', ('SC1', 'SC2', 'SC3')),
    ('affection', 'clip_l14', ('SC4',)),
    ('emoset', 'qwen', ('SC4',)),
    ('artelingo', 'clip_l14', ('SC5',)),
    ('mmart', 'clip_l14', ('SC5',)),
])
def test_cell_ids_are_the_claim_cells_of_the_request(fake, set_, enc, scs):
    """Ids, order of fields and identity keys come from `ns_outcome.claim_cells()`; all of them, none else."""
    cells = CC.score_cells(set_, enc, scs, ranges_for(set_, enc), DRAWS)
    assert sorted(c['id'] for c in cells) == sorted(claim_ids(set_, enc, scs))
    by_id = {c['id']: c for c in ns_outcome.claim_cells()}
    for c in cells:
        for key in ns_outcome.IDENTITY_KEYS:
            assert c[key] == by_id[c['id']][key], (c['id'], key)
        ns_outcome.check_cell(c, smoke=True)          # the file boundary of ticket 02 accepts every cell


def test_scs_not_applicable_to_the_encoder_give_no_cell(fake):
    """SC5 is CLIP L/14 only, SC4 photo sets only: a request that names them elsewhere yields nothing, not a cell."""
    assert CC.score_cells('artelingo', 'qwen', ('SC5',), {}, DRAWS) == []
    assert CC.score_cells('artelingo', 'qwen', ('SC4',), {}, DRAWS) == []
    assert CC.score_cells('emoset', 'siglip2', ('SC4',), {}, DRAWS) == []
    with pytest.raises(AssertionError):
        CC.score_cells('artelingo', 'qwen', ('SC6',), {}, DRAWS)          # retrieval is not this module's


def test_cell_numbers_are_ns_clu_paired_cell(fake):
    """A cell is the paired cell of the right partitions: cand/comp/d/lo/hi of the metric, n_draws, r from the range."""
    ranges = ranges_for('artelingo', 'qwen')
    cells = CC.score_cells('artelingo', 'qwen', ('SC1',), ranges, DRAWS)
    k = P.CLASS_COUNT[('artelingo', 'emotion')]
    cand = np.stack([fake.partitions(fake.cstar('artelingo', 'held', 'qwen', 'emotion', s), k, 'artelingo')
                     for s in range(5)])
    comp = fake.partitions(fake.rows('plain', ('artelingo', 'held', 'qwen'), 8), k, 'artelingo')
    y = fake.labels_data('artelingo', 'held', 'emotion')
    want = ns_clu.paired_cell(cand, comp, y, k, draws=DRAWS)
    got = {c['metric']: c for c in cells if c['aspect'] == 'emotion'}
    assert set(got) == set(P.CLASS_METRICS)
    for m, c in got.items():
        for key in ('cand', 'comp', 'd', 'lo', 'hi'):
            assert c[key] == pytest.approx(want[m][key], abs=1e-12), (m, key)
        assert c['n_draws'] == DRAWS and c['n_rows'] == N


# ------------------------------------------------------------------------------------------------ rows and partitions
def test_cstar_rows_are_the_clustering_form_on_the_held_split(fake):
    """c* rows are asked unit=False (stored float16 cast, what r10, r14, r15 fed k-means), on the held/test split."""
    CC.score_cells('mmart', 'siglip2', ('SC1',), ranges_for('mmart', 'siglip2'), DRAWS)
    cs = fake.named('cstar')
    assert cs and all(c[2]['unit'] is False for c in cs)
    assert {c[1][1] for c in cs} == {'test'}
    assert all(c[1][1] == P.HELD_SPLIT['mmart'] for c in fake.named('labels'))


def test_k_is_the_class_count_and_threads_come_from_the_set(fake):
    """k-means at K (ArtELingo style 27) with the set's name, so ns_metrics picks the set's threads (D20)."""
    CC.score_cells('artelingo', 'qwen', ('SC1', 'SC3'), ranges_for('artelingo', 'qwen'), DRAWS)
    ks = {(c[1][1], c[1][2]) for c in fake.named('partitions')}
    assert {27, 9, 10} <= {k for k, _ in ks} and {s for _, s in ks} == {'artelingo'}
    assert 20 not in {k for k, _ in ks}
    CC.score_cells('affection', 'siglip2', ('SC4',), ranges_for('affection', 'siglip2'), DRAWS)
    assert {c[1][2] for c in fake.named('partitions')[-3:]} == {'affection'}


def test_sc2_uses_the_kmeans_reading_at_k_and_five_seeds_drawn_apart(fake):
    """SC2: PercepT only through percept_partitions at K, one fit per seed, never ns_metrics.partitions on 40-d
    probabilities; the comparator enters as [5, 3, n] so its seeds are drawn apart (D16)."""
    ranges = ranges_for('mmart', 'siglip2')
    cells = CC.score_cells('mmart', 'siglip2', ('SC2',), ranges, DRAWS)
    pe = fake.named('percept')
    ks = sorted({c[1][1] for c in pe})
    assert ks == [9, 10, 20] and len(pe) == 5 * 3                          # 5 seeds x 3 aspects, at K only
    assert [c[1][2] for c in pe] == ['mmart'] * 15
    assert all(c[1][0] != 40 for c in fake.named('partitions'))
    assert {c[1][3] for c in fake.named('probs')} == {0, 1, 2, 3, 4}
    assert {c[1][1] for c in fake.named('probs')} == {'test'}
    # equal to the paired cell with the SEEDED comparator, different from the same parts treated as unseeded
    k = P.CLASS_COUNT[('mmart', 'style')]
    cand = np.stack([fake.partitions(fake.cstar('mmart', 'test', 'siglip2', 'style', s), k, 'mmart') for s in range(5)])
    comp = np.stack([fake.percept_partitions(fake.probs('mmart', 'test', 'siglip2', s), k, 'mmart') for s in range(5)])
    y = fake.labels_data('mmart', 'test', 'style')
    want = ns_clu.paired_cell(cand, comp, y, k, draws=DRAWS)
    other = ns_clu.paired_cell(cand, comp[0], y, k, draws=DRAWS)
    got = next(c for c in cells if c['aspect'] == 'style' and c['metric'] == 'ami')
    assert got['lo'] == pytest.approx(want['ami']['lo'], abs=1e-12)
    assert got['comp'] == pytest.approx(want['ami']['comp'], abs=1e-12)
    assert abs(want['ami']['lo'] - other['ami']['lo']) > 1e-9


def test_sc3_teacher_is_per_aspect_and_sc4_photo_sets_use_the_photo_loaders(fake):
    CC.score_cells('artelingo', 'siglip1', ('SC3',), ranges_for('artelingo', 'siglip1'), DRAWS)
    assert [c[1][2] for c in fake.named('teacher')] == ['emotion', 'style', 'genre']
    fake.calls.clear()
    CC.score_cells('affection', 'siglip1', ('SC4',), ranges_for('affection', 'siglip1'), DRAWS)
    assert not fake.named('cstar') and len(fake.named('photo')) == 5
    assert {c[1][1] for c in fake.named('photo')} == {'test'}
    assert len(fake.named('pplain')) == 1 and len(fake.named('pteacher')) == 1
    assert {c[1][1] for c in fake.named('labels')} == {'test'}


def test_sc5_crl_is_clustered_on_its_vector_at_20_and_at_k(fake):
    """SC5: CRL's 50-d vector clustered at 20 for AMI@20 and at K for the four metrics; c* on CLIP L/14."""
    cells = CC.score_cells('artelingo', 'clip_l14', ('SC5',), ranges_for('artelingo', 'clip_l14'), DRAWS)
    crl_fits = [c for c in fake.named('partitions') if c[1][0] == 50]
    ks = [c[1][1] for c in crl_fits]
    assert sorted(set(ks)) == [9, 10, 20, 27]                              # K per aspect plus 20
    fixed = [c for c in cells if c['id'].split('/')[4] == 'ami@20']
    assert len(fixed) == 3 and all(c['k'] == 20 and c['metric'] == 'ami' for c in fixed)
    assert len([c for c in cells if c['id'].split('/')[4] == 'ami']) == 3   # AMI at K is a separate cell
    assert {c['encoder'] for c in cells} == {'clip_l14'}


def test_candidate_partitions_are_fitted_once_per_head_and_k_across_scs(fake):
    """SC1, SC2, SC3 and SC5 share c*'s partitions at K: one fit per (aspect, seed), not one per SC."""
    CC.score_cells('mmart', 'clip_l14', ('SC1', 'SC2', 'SC3', 'SC5'), ranges_for('mmart', 'clip_l14'), DRAWS)
    cstar_fits = [c for c in fake.named('partitions') if c[1][0] == 10]
    # 3 aspects x 5 seeds at K, plus the 20-cluster fits for emotion (9) and genre (10); style's K is 20 and is reused
    assert len(cstar_fits) == 15 + 10


# ------------------------------------------------------------------------------------------------ r
def test_class_count_r_is_the_stage1_range_of_that_cell(fake):
    ranges = ranges_for('artelingo', 'siglip2')
    cells = CC.score_cells('artelingo', 'siglip2', ('SC1', 'SC3'), ranges, DRAWS)
    assert cells
    for c in cells:
        key = ns_outcome.range_key({**c, 'kind': 'class_count'})
        assert c['r'] == ranges[key]['r'] and key.startswith('cstar/artelingo/siglip2/')
        assert key.endswith(f"/{c['aspect']}/{c['metric']}") and c['r_source'] == f'stage1:{key}'
    fake.calls.clear()
    ranges = ranges_for('emoset', 'qwen')
    cells = CC.score_cells('emoset', 'qwen', ('SC4',), ranges, DRAWS)
    assert all(c['r'] == ranges[f"photo/emoset/qwen/emotion/{c['metric']}"]['r'] for c in cells)


def test_sc5_ami_at_20_takes_the_recorded_range_not_stage_one(fake):
    """The six AMI-at-20 cells read CLIP L/14 c*'s recorded AMI ranges (r14 ArtELingo, r15 MMArt); the stage 1 range of
    the same aspect and AMI at K is not used for them, and stage 1 is not consulted for them at all."""
    r14 = json.loads((P.ROUNDS['r14'] / 'records/stage1_clip_l14.json').read_text())['artclu']['ranges']
    r15 = json.loads((P.ROUNDS['r15'] / 'records/stage1_clip_l14.json').read_text())['clu_ranges']['ranges']
    for set_, rec in (('artelingo', r14), ('mmart', r15)):
        ranges = ranges_for(set_, 'clip_l14', value=0.5)                    # stage 1 values far from the records
        cells = CC.score_cells(set_, 'clip_l14', ('SC5',), ranges, DRAWS)
        fixed = {c['aspect']: c for c in cells if c['id'].split('/')[4] == 'ami@20'}
        assert set(fixed) == {'emotion', 'style', 'genre'}
        for a, c in fixed.items():
            assert c['r'] == rec[a] and rec[a] < 0.1
            assert c['r_source'].startswith('recorded:')
        for c in cells:
            if c['id'].split('/')[4] != 'ami@20':
                assert c['r'] >= 0.5 and c['r_source'].startswith('stage1:')
        # an incomplete stage 1 record cannot touch the AMI-at-20 cells
        only20 = CC.score_cells(set_, 'clip_l14', ('SC5',), {}, DRAWS)
        assert {c['r'] for c in only20 if c['id'].split('/')[4] == 'ami@20'} == {c['r'] for c in fixed.values()}


def test_a_missing_stage1_range_leaves_r_none_with_its_source(fake):
    cells = CC.score_cells('artelingo', 'qwen', ('SC1',), {}, DRAWS)
    assert cells and all(c['r'] is None and c['r_source'].startswith('missing:') for c in cells)


def test_ranges_none_reads_the_committed_stage1_record(fake, monkeypatch):
    seen = []
    ranges = ranges_for('artelingo', 'qwen')

    def fake_read(name):
        seen.append(name)
        return ranges

    monkeypatch.setattr(P, 'read_committed_record', fake_read)
    cells = CC.score_cells('artelingo', 'qwen', ('SC1',), None, DRAWS)
    assert seen == ['stage1_ranges.json'] and all(c['r'] is not None for c in cells)


# ------------------------------------------------------------------------------------------------ stage 1
@pytest.mark.parametrize('set_,enc', [('artelingo', 'qwen'), ('mmart', 'clip_l14'), ('affection', 'siglip2'),
                                      ('emoset', 'qwen')])
def test_stage1_ranges_read_only_the_validation_split(fake, monkeypatch, set_, enc):
    """Every loader call inside `stage1_ranges` names the validation split; a held or test call raises."""
    def guard(real):
        def f(*a, **kw):
            assert a[1] == 'val', NOT_HELD
            return real(*a, **kw)
        return f

    for mod, names in ((ns_data, ('labels', 'cstar', 'plain', 'teacher', 'percept_probs', 'crl')),
                       (ns_photo, ('labels', 'photo', 'plain', 'teacher'))):
        for n in names:
            monkeypatch.setattr(mod, n, guard(getattr(mod, n)))
    out = CC.stage1_ranges(set_, enc)
    assert out and all(c[1][1] == 'val' for c in fake.calls if c[0] in ('labels', 'cstar', 'photo'))
    cand = 'photo' if set_ in P.PHOTO_SETS else 'cstar'
    want = {f'{cand}/{set_}/{enc}/{a}/{m}' for a in P.SET_ASPECTS[set_] for m in P.CLASS_METRICS}
    assert set(out) == want


def test_stage1_ranges_raise_when_a_held_split_is_read(fake, monkeypatch):
    """The guard itself bites: if the range code asked a held split, the loader call would raise inside it."""
    def raising(s, sp, *a, **kw):
        raise RuntimeError(NOT_HELD)

    monkeypatch.setattr(CC, 'RANGE_SPLIT', P.HELD_SPLIT['artelingo'])
    monkeypatch.setattr(ns_data, 'cstar', lambda s, sp, *a, **kw: raising(s, sp) if sp != 'val' else None)
    with pytest.raises((RuntimeError, AssertionError)):
        CC.stage1_ranges('artelingo', 'qwen')


def test_stage1_range_values_are_ns_clu_ranges_at_k_with_the_mask_after(fake):
    out = CC.stage1_ranges('artelingo', 'siglip1')
    own_cstar, own_parts = list(fake.named('cstar')), list(fake.named('partitions'))
    for a in P.ASPECTS:
        k = P.CLASS_COUNT[('artelingo', a)]
        y = fake.labels_data('artelingo', 'val', a)
        parts = np.stack([fake.partitions(fake.cstar('artelingo', 'val', 'siglip1', a, s), k, 'artelingo')
                          for s in range(5)])
        want = ns_clu.ranges(parts, y, k)
        for m in P.CLASS_METRICS:
            got = out[f'cstar/artelingo/siglip1/{a}/{m}']
            assert got['r'] == want[m]['r'] and got['per_seed'] == want[m]['per_seed']
            assert got['k'] == k and got['n_val'] == N
    assert len(own_parts) == 15 and all(c[1][0] == 10 and c[1][2] == 'artelingo' for c in own_parts)
    assert len(own_cstar) == 15 and all(c[2]['unit'] is False for c in own_cstar)


def test_stage1_refuses_an_encoder_the_set_does_not_score(fake):
    with pytest.raises(AssertionError):
        CC.stage1_ranges('emoset', 'siglip2')
    with pytest.raises(AssertionError):
        CC.stage1_ranges('artelingo', 'nope')
