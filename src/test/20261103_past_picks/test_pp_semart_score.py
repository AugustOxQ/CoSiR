"""Tests of pp_semart_score (r19 ticket 09): the SemArt scorer (SC11 to SC14), the units and picks record, gate (iv)'s
SemArt part.

The guarded logic: the read-out map is fitted on the ArtELingo validation rows and never on the scored rows; query
expansion reads the scored gallery with the pick's k; SC13's matched comparator is the larger of the two readings; rows
with no label are masked; every array is joined to the labels by key (a shuffled order changes nothing); a subset
re-scores on those keys only; the cells carry r19's labels and pass `pp_outcome.check_cell`; the units are checked
against r14's stored ones; picks come from the stage 1 records and are cross-checked; the committed record is read
through `pp_paths.read_committed_record`; gate (iv) compares every value within 1e-9. Real data are not read: arrays are
synthetic and the readers are fakes in the layout of the index (tickets 05 and 06)."""
import json
import sys
import types

import numpy as np
import pytest

import pp_paths as PP
import pp_register  # noqa: F401
import pp_outcome
import pp_semart_score as S

import bs_artret
import mt_clu
import mt_ret
import rt_data
from ac_select import readout_from_name
import rb_score
import rt_metrics
import rt_readout

ENC = 'clip_l14'
N, NVAL, D, KCLASS = 160, 90, 24, 10
SEEDS = PP.SEEDS


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class World:
    """A synthetic SemArt: keys, labels (some -1 when asked), and one row per key for every arm, plus validation rows."""

    def __init__(self, n=N, nval=NVAL, seed=3, masked=0, shift=1.5):
        rng = np.random.default_rng(seed)
        self.keys = [f'img{i:04d}.jpg' for i in rng.permutation(n)]            # not sorted: order is the caller's
        self.y = {k: int(v) for k, v in zip(self.keys, rng.integers(0, KCLASS, n))}
        for k in rng.choice(self.keys, size=masked, replace=False):
            self.y[str(k)] = -1
        centres = rng.normal(size=(KCLASS, D))
        lab = np.array([max(self.y[k], 0) for k in self.keys])
        offset = shift * rng.normal(size=D)                                   # the eval rows are off the validation mean

        def rows(noise, off=0.0):
            return unit_rows(centres[lab] + noise * rng.normal(size=(n, D)) + off)

        self.cstar = {s: dict(zip(self.keys, rows(0.7, offset))) for s in SEEDS}
        self.frozen = dict(zip(self.keys, rows(1.6, offset)))
        self.teacher = dict(zip(self.keys, rows(1.1, offset)))
        vlab = rng.integers(0, KCLASS, nval)
        self.val_cstar = {s: unit_rows(centres[vlab] + 0.7 * rng.normal(size=(nval, D))) for s in SEEDS}
        self.val_frozen = unit_rows(centres[vlab] + 1.6 * rng.normal(size=(nval, D)))
        self.val_teacher = unit_rows(centres[vlab] + 1.1 * rng.normal(size=(nval, D)))
        self.probs = {s: rng.dirichlet(np.ones(40) * 0.1, size=n).astype(np.float32) for s in SEEDS}
        self.prob_of = {s: dict(zip(self.keys, self.probs[s])) for s in SEEDS}


class FakeInputs(S.Inputs):
    """Index layout: every reader takes keys and returns rows in that order, joined by key."""

    def __init__(self, world, matched=None):
        self.w = world
        self.asked = []
        self._matched = matched

    def labels(self, keys):
        self.asked.append(('labels', tuple(keys)))
        return np.array([self.w.y[k] for k in keys], dtype=np.int64)

    def cstar_val(self, enc, seed):
        return self.w.val_cstar[seed]

    def cstar_eval(self, enc, seed, keys):
        self.asked.append(('cstar', tuple(keys)))
        return np.stack([self.w.cstar[seed][k] for k in keys])

    def frozen_val(self, enc):
        return self.w.val_frozen

    def frozen_eval(self, enc, keys):
        self.asked.append(('frozen', tuple(keys)))
        return np.stack([self.w.frozen[k] for k in keys])

    def teacher_val(self):
        return self.w.val_teacher

    def teacher_eval(self, keys):
        self.asked.append(('teacher', tuple(keys)))
        return np.stack([self.w.teacher[k] for k in keys])

    def matched(self, enc, keys):
        self.asked.append(('matched', tuple(keys)))
        if self._matched is not None:
            return self._matched(keys)
        probs = {s: np.stack([self.w.prob_of[s][k] for k in keys]) for s in SEEDS}
        return mt_clu.matched_from_probs(probs)


@pytest.fixture(scope='module', autouse=True)
def small_validation():
    """The synthetic validation set has NVAL rows where the real one has 6,152 (`rt_data.names('val')`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, 'n_val_rows', lambda: NVAL)
        yield


PICKS = dict(cstar='center+k10', frozen='center+k5', teacher='none+k3')
UNIT_CLU, UNIT_RET = 0.011, 0.0042


@pytest.fixture(scope='module')
def world():
    return World()


@pytest.fixture(scope='module')
def clu_cells(world):
    return S.score_clu(ENC, FakeInputs(world), world.keys, UNIT_CLU, n_draws=40, workers=2)


@pytest.fixture(scope='module')
def ret_cells(world):
    return S.score_ret(ENC, FakeInputs(world), world.keys, UNIT_RET, PICKS, n_draws=60, workers=2)


# ------------------------------------------------------------------------------------------------ the cells
def test_cell_labels_and_fields(clu_cells, ret_cells):
    assert [c['sc'] for c in clu_cells] == ['SC11', 'SC13', 'SC14']
    assert [c['sc'] for c in ret_cells] == ['SC12', 'SC14']
    for c in clu_cells + ret_cells:
        assert c['data'] == 'semart_new' and c['aspect'] == 'genre' and c['encoder'] == ENC
        assert c['n_rows'] == N and c['seeds'] == list(SEEDS)
        pp_outcome.check_cell(c, smoke=True)
        assert (c['r'] is None) == (c['sc'] != 'SC14')
    assert {c['task'] for c in clu_cells} == {'clu'} and {c['task'] for c in ret_cells} == {'ret'}
    assert clu_cells[2]['r'] == UNIT_CLU and ret_cells[1]['r'] == UNIT_RET
    assert clu_cells[0]['comparator'] == f'frozen_{ENC}' and clu_cells[1]['comparator'] == 'matched_M7'
    assert clu_cells[2]['comparator'] == 'teacher_genre' and ret_cells[1]['comparator'] == 'teacher_genre'


def test_cells_pass_the_protocol_draws_only_when_asked(world):
    with pytest.raises(AssertionError):
        S.draws('clu', 40)                  # outside smoke the protocol's counts only
    assert S.draws('clu') == 2000 and S.draws('ret') == 10000


# ------------------------------------------------------------------------------------------------ retrieval: the map
def manual_ap(x_val, x_eval, y, pick):
    ro = readout_from_name(pick)
    return rt_metrics.retrieval(rt_readout.stage5_apply(ro, x_val, x_eval), y, qe_k=ro.k)['ap']


def test_map_is_fitted_on_validation_rows_only(world, monkeypatch):
    seen = []
    real = rt_readout.stage5_apply

    def spy(ro, x_val, x_eval):
        seen.append((x_val.shape[0], x_eval.shape[0]))
        return real(ro, x_val, x_eval)
    monkeypatch.setattr(rt_readout, 'stage5_apply', spy)
    inputs = FakeInputs(world)
    arms = S.ret_arms(ENC, inputs, world.keys, PICKS, workers=2)
    assert seen and all(v == NVAL and e == N for v, e in seen), seen
    y = inputs.labels(world.keys)
    want = manual_ap(world.val_cstar[0], np.stack([world.cstar[0][k] for k in world.keys]), y, PICKS['cstar'])
    assert np.array_equal(arms['cstar']['ap'][0], want)
    # fitted on the scored rows instead, the result is another number: the test would notice
    ev = np.stack([world.cstar[0][k] for k in world.keys])
    wrong = manual_ap(ev, ev, y, PICKS['cstar'])
    assert np.abs(wrong - want).max() > 1e-4


def test_query_expansion_reads_the_scored_gallery_with_the_picks_k(world, monkeypatch):
    calls = []
    real = rt_metrics.retrieval

    def spy(x, labels, *a, **kw):
        calls.append((x.shape[0], kw.get('qe_k')))
        return real(x, labels, *a, **kw)
    monkeypatch.setattr(rt_metrics, 'retrieval', spy)
    S.ret_arms(ENC, FakeInputs(world), world.keys, PICKS, workers=2)
    ks = sorted({k for _, k in calls})
    assert ks == [3, 5, 10], ks                                   # teacher k3, frozen k5, c* k10
    assert {n for n, _ in calls} == {N}, calls                    # always the SemArt gallery, never the validation rows
    # the pick's k changes the number
    inputs = FakeInputs(world)
    a10 = S.ret_arms(ENC, inputs, world.keys, dict(PICKS, cstar='center+k10'), workers=2)['cstar']['ap']
    a0 = S.ret_arms(ENC, inputs, world.keys, dict(PICKS, cstar='center+k0'), workers=2)['cstar']['ap']
    assert np.abs(a10 - a0).max() > 1e-4


def test_validation_gallery_does_not_enter_expansion(world):
    inputs = FakeInputs(world)
    base = S.ret_arms(ENC, inputs, world.keys, dict(PICKS, cstar='none+k10'), workers=2)['cstar']['ap']
    moved = World()
    moved.val_cstar = {s: unit_rows(np.random.default_rng(9).normal(size=(NVAL, D))) for s in SEEDS}
    other = S.ret_arms(ENC, FakeInputs(moved), moved.keys, dict(PICKS, cstar='none+k10'), workers=2)['cstar']['ap']
    assert np.array_equal(base, other)            # the identity map ignores the validation rows altogether


# ------------------------------------------------------------------------------------------------ label masking
def test_retrieval_masks_unlabelled_rows():
    w = World(masked=25)
    pick = dict(PICKS, cstar='none+k0', frozen='none+k0', teacher='none+k0')
    cells = S.score_ret(ENC, FakeInputs(w), w.keys, UNIT_RET, pick, n_draws=40, workers=2)
    masked = [k for k in w.keys if w.y[k] < 0]
    assert len(masked) == 25 and cells[0]['n_queries'] <= N - 25
    for s in SEEDS:                                  # moving the masked rows changes nothing: not queries, not gallery
        for k in masked:
            w.cstar[s][k] = unit_rows(np.random.default_rng(5).normal(size=(1, D)))[0]
    again = S.score_ret(ENC, FakeInputs(w), w.keys, UNIT_RET, pick, n_draws=40, workers=2)
    assert [c['candidate_value'] for c in cells] == [c['candidate_value'] for c in again]
    assert cells[0]['n_rows'] == N                   # the population is every painting; the queries are the labelled ones


def test_clustering_masks_unlabelled_rows_in_ami():
    w = World(masked=25)
    inputs = FakeInputs(w)
    cells = S.score_clu(ENC, inputs, w.keys, UNIT_CLU, n_draws=40, workers=2)
    y = inputs.labels(w.keys)
    parts = mt_clu.cstar_fixed({s: inputs.cstar_eval(ENC, s, w.keys) for s in SEEDS})
    m = y >= 0
    ami = np.mean([np.mean([float(rb_score.ami(y[m], p[m])) for p in parts[s]['r20']]) for s in SEEDS])
    assert cells[0]['candidate_value'] == pytest.approx(ami, abs=1e-12)
    assert cells[0]['n_labelled'] == int(m.sum()) == N - 25 and cells[0]['n_rows'] == N


# ------------------------------------------------------------------------------------------------ SC13: the larger reading
def matched_with(world, better):
    """Matched partitions whose `own` reading equals the labels (best) or whose 20-cluster reading does."""
    def make(keys):
        y = np.array([max(world.y[k], 0) for k in keys])
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 20, len(keys))
        good, bad = y.astype(np.int32), noise.astype(np.int32)
        own, r20 = (good, bad) if better == 'own' else (bad, good)
        return {s: {'own': own[None, :], 'r20': np.stack([r20, r20, r20])} for s in SEEDS}
    return make


@pytest.mark.parametrize('better', ['own', 'r20'])
def test_sc13_comparator_is_the_larger_reading(world, better):
    cells = S.score_clu(ENC, FakeInputs(world, matched_with(world, better)), world.keys, UNIT_CLU, n_draws=40, workers=2)
    sc13 = cells[1]
    assert sc13['comparator_reading'] == f'M7-{better}'
    assert set(sc13['comparator_readings']) == {'M7-own', 'M7-r20'}
    other = 'r20' if better == 'own' else 'own'
    assert sc13['comparator_readings'][f'M7-{better}'] > sc13['comparator_readings'][f'M7-{other}'] + 0.5   # not a tie
    assert sc13['comparator_value'] == max(sc13['comparator_readings'].values())
    assert sc13['d'] == pytest.approx(sc13['candidate_value'] - sc13['comparator_value'], abs=1e-12)


def test_matched_arm_is_read_on_exactly_the_scored_rows(world):
    inputs = FakeInputs(world)
    S.score_clu(ENC, inputs, world.keys, UNIT_CLU, n_draws=40, workers=2)
    got = [keys for name, keys in inputs.asked if name == 'matched']
    assert got == [tuple(world.keys)]


# ------------------------------------------------------------------------------------------------ key joins
def test_shuffled_order_joins_by_key():
    w = World(shift=0.0)
    ordered = sorted(w.keys)
    shuffled = list(np.random.default_rng(7).permutation(w.keys))
    out = {}
    for name, order in (('sorted', ordered), ('shuffled', shuffled)):
        inputs = FakeInputs(w)
        pick = dict(PICKS, cstar='none+k0', frozen='none+k0', teacher='none+k0')
        clu = S.score_clu(ENC, inputs, order, UNIT_CLU, n_draws=40, workers=2)
        ret = S.score_ret(ENC, inputs, order, UNIT_RET, pick, n_draws=40, workers=2)
        asked = {n: k for n, k in inputs.asked}
        assert all(k == tuple(order) for n, k in inputs.asked), name          # every reader got the order, as keys
        out[name] = (clu, ret)
        assert clu[0]['candidate_value'] > 0.4          # rows and labels joined: a misjoined set has AMI near 0
    # mAP does not depend on the order of the rows (continuous scores: no ties)
    for a, b in zip(out['sorted'][1], out['shuffled'][1]):
        assert a['candidate_value'] == pytest.approx(b['candidate_value'], abs=1e-12)
        assert a['comparator_value'] == pytest.approx(b['comparator_value'], abs=1e-12)


def test_duplicate_or_missing_keys_are_refused(world):
    with pytest.raises(AssertionError):
        S.score_ret(ENC, FakeInputs(world), world.keys[:5] + world.keys[:1], UNIT_RET, PICKS, n_draws=10, workers=2)


def test_misaligned_reader_is_caught(world):
    class Short(FakeInputs):
        def frozen_eval(self, enc, keys):
            return super().frozen_eval(enc, keys)[:-1]
    with pytest.raises(AssertionError):
        S.score_ret(ENC, Short(world), world.keys, UNIT_RET, PICKS, n_draws=10, workers=2)


# ------------------------------------------------------------------------------------------------ subset (ticket 13)
def test_subset_rescores_on_those_keys_only(world):
    keep = set(world.keys[:120])
    inputs = FakeInputs(world)
    cells = S.score_clu(ENC, inputs, world.keys, UNIT_CLU, n_draws=40, workers=2, subset=keep)
    want = [k for k in world.keys if k in keep]
    assert all(c['n_rows'] == 120 for c in cells)
    assert all(keys == tuple(want) for _, keys in inputs.asked)           # in the order of `order`
    ret = S.score_ret(ENC, FakeInputs(world), world.keys, UNIT_RET, PICKS, n_draws=40, workers=2, subset=keep)
    assert all(c['n_rows'] == 120 for c in ret)
    with pytest.raises(AssertionError):
        S.score_ret(ENC, FakeInputs(world), world.keys, UNIT_RET, PICKS, n_draws=40, workers=2, subset=keep | {'nope.jpg'})


def test_ap_per_type_is_available_from_the_arms(world):
    arms = S.ret_arms(ENC, FakeInputs(world), world.keys, PICKS, workers=2)
    assert arms['cstar']['ap'].shape == (5, len(arms['cstar']['rows'])) and arms['frozen']['ap'].ndim == 1


# ------------------------------------------------------------------------------------------------ writing cells
def test_write_cells_checks_and_merges(tmp_path, clu_cells):
    path = S.write_cells('semart_clu', ENC, clu_cells, out=tmp_path, smoke=True)
    assert path == tmp_path / 'stage2' / 'cells_semart_clu.json'
    again = S.write_cells('semart_clu', ENC, clu_cells, out=tmp_path, smoke=True)
    assert len(json.loads(again.read_text())) == 3                         # replaced, not appended
    bad = [dict(c) for c in clu_cells]
    bad[2]['r'] = None                                                    # a teacher cell without its unit
    with pytest.raises(AssertionError):
        S.write_cells('semart_clu', ENC, bad, out=tmp_path, smoke=True)
    with pytest.raises(AssertionError):
        S.write_cells('semart_ret', ENC, clu_cells, out=tmp_path, smoke=True)      # clustering cells in the retrieval file
    n_draws = [dict(c, n_draws=40) for c in clu_cells]
    with pytest.raises(AssertionError):
        S.write_cells('semart_clu', ENC, n_draws, out=tmp_path, smoke=False)       # not the protocol's draws


# ------------------------------------------------------------------------------------------------ production order
def test_production_order_is_the_sha256_order_of_the_set(monkeypatch, world):
    import pp_semart
    keys = PP.order(world.keys)
    monkeypatch.setattr(pp_semart, 'keys', lambda name: list(keys))
    monkeypatch.setattr(PP, 'N_SEMART', len(keys))
    assert S.production_order() == keys
    monkeypatch.setattr(PP, 'N_SEMART', len(keys) + 1)             # the set must have the recorded size
    with pytest.raises(AssertionError):
        S.production_order()
    monkeypatch.setattr(PP, 'N_SEMART', len(keys))
    monkeypatch.setattr(pp_semart, 'keys', lambda name: list(world.keys))
    with pytest.raises(AssertionError):
        S.production_order()


# ------------------------------------------------------------------------------------------------ units
def fake_ranges(clu, ret):
    return (lambda root: dict(ranges=dict(emotion=0.1, style=0.2, genre=clu)),
            lambda root: dict(ranges=dict(emotion=0.1, style=0.2, genre=ret)))


def test_units_take_genre_from_the_range_code_gate_vii_checks():
    clu_fn, ret_fn = fake_ranges(0.0123, 0.0045)
    u = S.unit_values(ENC, root='x', clu_fn=clu_fn, ret_fn=ret_fn)
    assert (u['clu'], u['ret']) == (0.0123, 0.0045)


def test_units_must_equal_the_recorded_ones():
    clu_fn, ret_fn = fake_ranges(0.0123, 0.0045)
    S.unit_values(ENC, root='x', clu_fn=clu_fn, ret_fn=ret_fn, recorded=dict(clu=0.0123 + 5e-10, ret=0.0045))
    with pytest.raises(AssertionError):
        S.unit_values(ENC, root='x', clu_fn=clu_fn, ret_fn=ret_fn, recorded=dict(clu=0.0123 + 3e-9, ret=0.0045))
    with pytest.raises(AssertionError):
        S.unit_values(ENC, root='x', clu_fn=clu_fn, ret_fn=ret_fn, recorded=dict(clu=0.0123, ret=0.0046))
    with pytest.raises(AssertionError):                                   # a unit must be a positive finite number
        S.unit_values(ENC, root='x', clu_fn=lambda r: dict(ranges=dict(genre=0.0)), ret_fn=ret_fn)


R11 = {'cstar': 'center+k10', 'plain': 'center+k10', 'instructed': 'center+k10', 'clip': 'center+k10',
       'siglip2': 'center+k10', 'percept': 'center+k10', 'crl': 'none+k10'}


def record(enc, cstar='center+k10', frozen='center+k10', teacher='center+k10'):
    return dict(encoder=enc, artret_picks=dict(cstar=dict(pick=cstar), frozen=dict(pick=frozen),
                                               teacher=dict(pick=teacher)))


def test_qwen_picks_are_r11s():
    picks, source = S.arm_picks('qwen', None, dict(R11, cstar='center+k3', plain='none+k5', instructed='whiten64+k1'))
    assert picks == dict(cstar='center+k3', frozen='none+k5', teacher='whiten64+k1')
    assert set(source) == {'cstar', 'frozen', 'teacher'}


def test_other_picks_come_from_the_stage1_record_and_are_cross_checked():
    picks, _ = S.arm_picks('clip_l14', record('clip_l14', cstar='whiten256+k3'), R11)
    assert picks == dict(cstar='whiten256+k3', frozen='center+k10', teacher='center+k10')
    with pytest.raises(AssertionError):                                   # clip_l14's frozen pick is r11's 'clip' pick
        S.arm_picks('clip_l14', record('clip_l14', frozen='none+k0'), R11)
    with pytest.raises(AssertionError):                                   # every teacher pick is r11's 'instructed'
        S.arm_picks('siglip1', record('siglip1', teacher='none+k0'), R11)
    with pytest.raises(AssertionError):                                   # a record of another encoder
        S.arm_picks('siglip2', record('clip_l14'), R11)
    with pytest.raises(AssertionError):                                   # not a read-out of the menu
        S.arm_picks('dinov2_large', record('dinov2_large', cstar='center+k99'), R11)
    # SigLIP 1 and the new three have a computed frozen pick: it is whatever their record holds
    assert S.arm_picks('siglip1', record('siglip1', frozen='none+k5'), R11)[0]['frozen'] == 'none+k5'
    assert S.arm_picks('clip_b32', record('clip_b32', frozen='none+k5'), R11)[0]['frozen'] == 'none+k5'


def test_stage1_document_form():
    entry = S.stage1_entry(0.01, 0.002, dict(cstar='a', frozen='b', teacher='c'), 'u', dict(cstar='x', frozen='y', teacher='z'))
    assert entry == dict(units=dict(clu=0.01, ret=0.002), units_source='u', picks=dict(cstar='a', frozen='b', teacher='c'),
                         picks_source=dict(cstar='x', frozen='y', teacher='z'))
    with pytest.raises(AssertionError):
        S.check_stage1({e: entry for e in PP.ENCODERS[:6]})               # all seven encoders


def test_stage1_is_read_through_the_committed_record(monkeypatch):
    entry = S.stage1_entry(0.01, 0.002, dict(cstar='center+k10', frozen='center+k10', teacher='center+k10'), 'u',
                           dict(cstar='x', frozen='y', teacher='z'))
    asked = []
    monkeypatch.setattr(PP, 'read_committed_record', lambda name: asked.append(name) or {e: entry for e in PP.ENCODERS})
    units, picks = S.stage1_for('siglip1')
    assert asked == ['stage1_semart.json']
    assert units == dict(clu=0.01, ret=0.002) and picks['cstar'] == 'center+k10'


def test_write_stage1_is_frozen_once_written(tmp_path):
    entry = S.stage1_entry(0.01, 0.002, dict(cstar='center+k10', frozen='center+k10', teacher='center+k10'), 'u',
                           dict(cstar='x', frozen='y', teacher='z'))
    doc = {e: entry for e in PP.ENCODERS}
    path = S.write_stage1(doc, out=tmp_path)
    assert path == tmp_path / 'stage1' / 'semart.json'
    S.write_stage1(doc, out=tmp_path)                                     # the same content again is fine
    doc2 = dict(doc, qwen=S.stage1_entry(0.02, 0.002, entry['picks'], 'u', entry['picks_source']))
    with pytest.raises(AssertionError):
        S.write_stage1(doc2, out=tmp_path)


# ------------------------------------------------------------------------------------------------ gate (iv)
def stored_cell(sc, task, **kw):
    base = dict(sc=sc, encoder=ENC, aspect='genre', task=task, data='artelingo_held', candidate_value=0.3,
                comparator_value=0.2, d=0.1, lo=0.05, hi=0.15, r=None)
    base.update(kw)
    return base


def test_gate_compares_every_value_within_1e_9():
    ours = [stored_cell('SC11', 'clu'), stored_cell('SC13', 'clu', comparator_reading='M7-own'),
            stored_cell('SC14', 'clu', r=0.01)]
    theirs = [stored_cell('SC1', 'clu'), stored_cell('SC3', 'clu', comparator_reading='M7-own'),
              stored_cell('SC6', 'clu', r=0.01)]
    res = S.compare_gate(ours, theirs, full=True)
    assert res['passed'] and res['max_abs_diff'] == 0.0
    assert {i.split('/')[0] for i in res['items']} == {'SC1', 'SC3', 'SC6'}
    off = [dict(c) for c in ours]
    off[0]['hi'] += 2e-9
    assert not S.compare_gate(off, theirs, full=True)['passed']
    off = [dict(c) for c in ours]
    off[0]['hi'] += 5e-10
    assert S.compare_gate(off, theirs, full=True)['passed']
    off = [dict(c) for c in ours]
    off[1]['comparator_reading'] = 'M7-r20'
    assert not S.compare_gate(off, theirs, full=True)['passed']
    off = [dict(c) for c in ours]
    off[2]['r'] = 0.0100001
    assert not S.compare_gate(off, theirs, full=True)['passed']


def test_gate_with_reduced_draws_cannot_pass_but_checks_the_values():
    ours = [stored_cell('SC12', 'ret', n_queries=5), stored_cell('SC14', 'ret', r=0.01, n_queries=5)]
    theirs = [stored_cell('SC2', 'ret', n_queries=5), stored_cell('SC6', 'ret', r=0.01, n_queries=5)]
    ours[0]['lo'] = 123.0                                                  # intervals are not compared at reduced draws
    res = S.compare_gate(ours, theirs, full=False)
    assert res['values_ok'] and not res['passed']
    ours[0]['candidate_value'] += 1e-6
    assert not S.compare_gate(ours, theirs, full=False)['values_ok']
    ours[0]['n_queries'] = 4
    assert not S.compare_gate(ours, theirs, full=False)['values_ok']


def test_gate_refuses_a_missing_or_extra_cell():
    ours = [stored_cell('SC11', 'clu')]
    with pytest.raises(AssertionError):
        S.compare_gate(ours, [stored_cell('SC1', 'clu'), stored_cell('SC3', 'clu')], full=True)
    with pytest.raises(AssertionError):
        S.compare_gate(ours + [stored_cell('SC13', 'clu')], [stored_cell('SC1', 'clu')], full=True)


def test_gate_feeds_the_scorer_r14s_order_and_labels(world):
    """The gate runs the SemArt path (`score_clu`) with the ArtELingo held names in r14's order as `order`."""
    names = sorted(world.keys)                                             # r14's held order is sorted names
    inputs = FakeInputs(world)
    cells = S.score_clu(ENC, inputs, names, UNIT_CLU, n_draws=40, workers=2, data='artelingo_held')
    assert all(c['data'] == 'artelingo_held' for c in cells)
    assert all(keys == tuple(names) for _, keys in inputs.asked)


# ------------------------------------------------------------------------------------------------ fit rows of the maps
def test_fit_rows_must_be_the_validation_rows(world):
    class Short(FakeInputs):
        def __init__(self, w, which):
            super().__init__(w)
            self.which = which

        def cstar_val(self, enc, seed):
            x = super().cstar_val(enc, seed)
            return x[:-1] if self.which == 'cstar' else x

        def frozen_val(self, enc):
            x = super().frozen_val(enc)
            return x[:-1] if self.which == 'frozen' else x

        def teacher_val(self):
            x = super().teacher_val()
            return x[:-1] if self.which == 'teacher' else x
    S.ret_arms(ENC, FakeInputs(world), world.keys, PICKS, workers=2)
    for which in ('cstar', 'frozen', 'teacher'):
        with pytest.raises(AssertionError, match='fit rows'):
            S.ret_arms(ENC, Short(world, which), world.keys, PICKS, workers=2)


# ------------------------------------------------------------------------------------------------ Qwen's mAP unit
def test_qwen_map_unit_is_checked_against_r11s_stored_range(monkeypatch, tmp_path):
    (tmp_path / 'picks.json').write_text(json.dumps(R11))
    monkeypatch.setattr(S, 'R11_PICKS', tmp_path / 'picks.json')
    monkeypatch.setattr(S, 'unit_values', lambda enc, **kw: dict(clu=0.0123, ret=0.0045))
    monkeypatch.setattr(S, 'r11_map_range', lambda: 0.0045 + 5e-10)
    doc = S.stage1(('qwen',))
    assert doc['qwen']['units'] == dict(clu=0.0123, ret=0.0045) and 'r11' in doc['qwen']['units_source']
    monkeypatch.setattr(S, 'r11_map_range', lambda: 0.0045 + 3e-9)
    with pytest.raises(AssertionError, match="r11"):
        S.stage1(('qwen',))


def test_r11_map_range_reads_the_stored_genre_entry(monkeypatch, tmp_path):
    f = tmp_path / 'ranges.json'
    f.write_text(json.dumps({'map': {'genre': {'r': 0.25, 'per_seed': [1]}, 'style': {'r': 0.5}}}))
    monkeypatch.setattr(mt_ret.ac_paths, 'R11_RANGES_FILE', f)
    assert S.r11_map_range() == 0.25


# ------------------------------------------------------------------------------------------------ SemartInputs end to end
@pytest.fixture
def production_readers(world, monkeypatch):
    """The production readers faked with the real dtypes: float16 c* rows from `pp_export.load`, float32 unit rows from
    `pp_stores.load`, int32 matched partitions, int64 labels; ArtELingo validation rows from the older rounds' loaders."""
    import pp_matched
    import pp_semart
    import pp_stores
    calls = []

    def export_load(set_, enc, aspect, seed, keys, root=None, **kw):
        calls.append(('export', set_, enc, aspect, seed))
        assert set_ == 'semart' and aspect == 'genre' and seed in SEEDS
        return np.stack([world.cstar[seed][k] for k in keys]).astype(np.float16)

    def stores_load(dataset, view, keys, **kw):
        calls.append(('stores', dataset, view))
        assert dataset == 'semart'
        table = world.teacher if view == PP.TEACHER_VIEW['genre'] else world.frozen
        return np.stack([table[k] for k in keys]).astype(np.float32)

    def partitions(set_name, enc, keys, root=None):
        calls.append(('matched', set_name, enc))
        probs = {s: np.stack([world.prob_of[s][k] for k in keys]) for s in SEEDS}
        got = mt_clu.matched_from_probs(probs)
        return {s: {'own': np.asarray(got[s]['own']).astype(np.int32), 'r20': np.asarray(got[s]['r20']).astype(np.int32)}
                for s in SEEDS}
    try:
        import pp_export
    except ImportError:                       # ticket 06 is not merged: the module is a stand-in with `load`
        pp_export = types.ModuleType('pp_export')
        monkeypatch.setitem(sys.modules, 'pp_export', pp_export)
    monkeypatch.setattr(pp_export, 'load', export_load, raising=False)
    monkeypatch.setattr(pp_stores, 'load', stores_load)
    monkeypatch.setattr(pp_matched, 'partitions', partitions)
    monkeypatch.setattr(pp_semart, 'labels', lambda keys: np.array([world.y[k] for k in keys], dtype=np.int64))
    monkeypatch.setattr(bs_artret, 'cstar_embed', lambda root, split, aspect, seed: world.val_cstar[seed])
    monkeypatch.setattr(bs_artret, 'frozen_embed', lambda view, split: world.val_frozen)
    monkeypatch.setattr(rt_data, 'embed', lambda *a, **k: world.val_teacher)
    return calls


def test_semart_inputs_deliver_what_the_scorer_accepts(world, production_readers):
    clu = S.score_clu(ENC, S.SemartInputs('clu'), world.keys, UNIT_CLU, n_draws=40, workers=2)
    ret = S.score_ret(ENC, S.SemartInputs('ret'), world.keys, UNIT_RET, PICKS, n_draws=60, workers=2)
    assert [c['sc'] for c in clu] == ['SC11', 'SC13', 'SC14'] and [c['sc'] for c in ret] == ['SC12', 'SC14']
    assert {k[0] for k in production_readers} == {'export', 'stores', 'matched'}
    for c in clu + ret:
        pp_outcome.check_cell(c, smoke=True)


def test_semart_inputs_take_cstar_rows_the_inherited_way(world, production_readers):
    keys = world.keys
    x16 = np.stack([world.cstar[0][k] for k in keys]).astype(np.float16)
    assert x16.dtype == np.float16
    ret = S.SemartInputs('ret').cstar_eval(ENC, 0, keys)
    clu = S.SemartInputs('clu').cstar_eval(ENC, 0, keys)
    assert ret.dtype == clu.dtype == np.float32
    assert np.array_equal(ret, mt_ret.unit(x16))                          # r15's cstar_rows form: cast, divide by the norm
    assert np.array_equal(clu, x16.astype(np.float32))                    # r14 / mt_clu.cstar_fixed: cast only
    with pytest.raises(AssertionError):
        S.SemartInputs('both')


def test_semart_inputs_score_equals_the_float32_fake(world, production_readers):
    """The production wiring and the index-layout fake give the same cells once the rows are float16-rounded."""
    class Rounded(FakeInputs):
        def __init__(self, w, task):
            super().__init__(w)
            self.task = task

        def cstar_eval(self, enc, seed, keys):
            x16 = np.stack([self.w.cstar[seed][k] for k in keys]).astype(np.float16)
            return mt_ret.unit(x16) if self.task == 'ret' else x16.astype(np.float32)
    got = S.score_ret(ENC, S.SemartInputs('ret'), world.keys, UNIT_RET, PICKS, n_draws=60, workers=2)
    want = S.score_ret(ENC, Rounded(world, 'ret'), world.keys, UNIT_RET, PICKS, n_draws=60, workers=2)
    assert [c['candidate_value'] for c in got] == [c['candidate_value'] for c in want]
    got = S.score_clu(ENC, S.SemartInputs('clu'), world.keys, UNIT_CLU, n_draws=40, workers=2)
    want = S.score_clu(ENC, Rounded(world, 'clu'), world.keys, UNIT_CLU, n_draws=40, workers=2)
    assert [c['candidate_value'] for c in got] == [c['candidate_value'] for c in want]


# ================================================================================================ fix wave A (findings 3 and 12)
from test_pp_outcome import ok_status, failed  # noqa: E402

NEW3 = ('dinov2_large', 'clip_b32', 'clip_b16')
ALL7 = NEW3 + ('qwen', 'siglip2', 'siglip1', 'clip_l14')


class NoTeacher(FakeInputs):
    """The teacher store is absent (gate (i) genre_word failed): any read of it fails the test."""

    def teacher_val(self):
        raise AssertionError('the teacher was read (val)')

    def teacher_eval(self, keys):
        raise AssertionError('the teacher was read (eval)')


class NoMatched(FakeInputs):
    def matched(self, enc, keys):
        raise AssertionError('the matched arm was read')


def by_sc(cells):
    return {c['sc']: c for c in cells}


def test_clu_without_the_teacher_is_the_other_cells_of_the_full_list(world, clu_cells):
    got = S.score_clu(ENC, NoTeacher(world), world.keys, UNIT_CLU, n_draws=40, workers=2, use_teacher=False)
    assert [c['sc'] for c in got] == ['SC11', 'SC13']
    assert got == [c for c in clu_cells if c['sc'] != 'SC14']


def test_ret_without_the_teacher_is_the_other_cell_of_the_full_list(world, ret_cells):
    arms = S.ret_arms(ENC, NoTeacher(world), world.keys, dict(cstar=PICKS['cstar'], frozen=PICKS['frozen']), workers=2,
                      use_teacher=False)
    assert arms['teacher'] is None and arms['cstar'] is not None
    got = S.score_ret(ENC, NoTeacher(world), world.keys, UNIT_RET, dict(cstar=PICKS['cstar'], frozen=PICKS['frozen']),
                      n_draws=60, workers=2, use_teacher=False)
    assert [c['sc'] for c in got] == ['SC12']
    assert got[0] == ret_cells[0]


def test_clu_without_the_matched_arm_is_the_other_cells_of_the_full_list(world, clu_cells):
    got = S.score_clu(ENC, NoMatched(world), world.keys, UNIT_CLU, n_draws=40, workers=2, use_matched=False)
    assert [c['sc'] for c in got] == ['SC11', 'SC14']
    assert got == [c for c in clu_cells if c['sc'] != 'SC13']
    only = S.score_clu(ENC, NoTeacher(world), world.keys, UNIT_CLU, n_draws=40, workers=2, use_matched=False,
                       use_teacher=False)
    assert only == [c for c in clu_cells if c['sc'] == 'SC11']


def test_without_the_teacher_the_arms_are_still_asserted_without_a_pick_for_it(world):
    with pytest.raises(AssertionError):
        S.ret_arms(ENC, FakeInputs(world), world.keys, dict(cstar=PICKS['cstar']), workers=2)
    with pytest.raises(AssertionError):
        S.ret_arms(ENC, FakeInputs(world), world.keys, dict(cstar=PICKS['cstar'], frozen=PICKS['frozen']), workers=2)


@pytest.fixture
def stage2(world, monkeypatch, tmp_path):
    """Smoke `score()`: the SemArt set, the units and picks, and the cells writer are stand-ins; OUT/records is empty."""
    monkeypatch.setattr(PP, 'SMOKE', 1)
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    written = []
    monkeypatch.setattr(S, 'production_order', lambda: list(world.keys))
    monkeypatch.setattr(S, 'stage1_for', lambda enc: (dict(clu=UNIT_CLU, ret=UNIT_RET), dict(PICKS)))
    monkeypatch.setattr(S, 'write_cells', lambda part, enc, cells, *a, **k: written.append((part, enc, cells)))
    return written


def run_score(world, enc, inputs_cls=FakeInputs, **kw):
    out = []
    for part in ('semart_clu', 'semart_ret'):
        out += S.score(part, enc, n_draws=40 if part == 'semart_clu' else 60, workers=2, inputs=inputs_cls(world, **{}), **kw)
    return out


@pytest.mark.parametrize('enc', ALL7)
def test_score_with_genre_word_failed_writes_sc11_to_sc13_and_never_reads_the_teacher(world, stage2, enc):
    status = failed('gates', 'i_semart_genre_word')
    cells = run_score(world, enc, NoTeacher, status=status)
    assert [(c['sc'], c['task']) for c in cells] == [('SC11', 'clu'), ('SC13', 'clu'), ('SC12', 'ret')]
    assert [w[0] for w in stage2] == ['semart_clu', 'semart_ret']
    got = {pp_outcome.cell_key(c) for c in cells}
    missing = [k for k in pp_outcome.required_cell_keys(enc) if k[4] == 'semart_new' and k not in got]
    assert sorted(missing) == [('SC14', enc, 'genre', 'clu', 'semart_new'), ('SC14', enc, 'genre', 'ret', 'semart_new')]
    voids = pp_outcome.Voids(status)
    assert all(voids.cell(k) for k in missing)
    assert not [k for k in got if voids.cell(k)]


@pytest.mark.parametrize('enc', NEW3)
@pytest.mark.parametrize('how', [('runs', 'matched'), ('gates', 'v')])
def test_score_with_a_failed_matched_arm_writes_everything_but_sc13(world, stage2, enc, how):
    status = ok_status()
    if how[0] == 'runs':
        status['runs'][enc]['matched'] = 'failed'
    else:
        status['gates']['v'][enc] = 'failed'
    cells = run_score(world, enc, NoMatched, status=status)
    assert [(c['sc'], c['task']) for c in cells] == [('SC11', 'clu'), ('SC14', 'clu'), ('SC12', 'ret'), ('SC14', 'ret')]
    got = {pp_outcome.cell_key(c) for c in cells}
    missing = [k for k in pp_outcome.required_cell_keys(enc) if k[4] == 'semart_new' and k not in got]
    assert missing == [('SC13', enc, 'genre', 'clu', 'semart_new')]
    assert pp_outcome.Voids(status).cell(missing[0])


def test_score_with_both_failed_writes_sc11_and_sc12(world, stage2):
    status = failed('gates', 'i_semart_genre_word')
    status['runs']['clip_b32']['matched'] = 'failed'

    class Neither(NoTeacher, NoMatched):
        pass
    cells = run_score(world, 'clip_b32', Neither, status=status)
    assert [c['sc'] for c in cells] == ['SC11', 'SC12']


def test_old_encoders_have_no_run_record_and_keep_their_matched_cells(world, stage2):
    status = ok_status()
    for e in NEW3:
        status['runs'][e]['matched'] = 'failed'
    cells = run_score(world, 'qwen', status=status)
    assert [c['sc'] for c in cells] == ['SC11', 'SC13', 'SC14', 'SC12', 'SC14']


@pytest.mark.parametrize('enc', ['clip_l14', 'clip_b32', 'qwen'])
def test_a_healthy_status_gives_the_cells_of_a_run_without_a_status(world, stage2, clu_cells, ret_cells, enc):
    """The pair: healthy status and (under smoke) no status file give the same five cells as the scorers alone."""
    plain = run_score(world, enc)
    assert [c['sc'] for c in plain] == ['SC11', 'SC13', 'SC14', 'SC12', 'SC14']
    assert run_score(world, enc, status=ok_status()) == plain
    if enc == ENC:
        assert plain == clu_cells + ret_cells
    for c in plain:
        assert c['encoder'] == enc


def test_other_failed_gates_do_not_drop_the_teacher_or_matched_cells(world, stage2):
    status = failed('gates', 'vii', 'artelingo')
    assert len(run_score(world, 'clip_b32', status=status)) == 5


def test_a_missing_status_outside_smoke_raises_before_any_row(world, monkeypatch, tmp_path):
    monkeypatch.setattr(PP, 'SMOKE', 0)
    monkeypatch.setattr(PP, 'RECORDS', tmp_path)
    monkeypatch.setattr(S, 'stage1_for', lambda enc: (dict(clu=UNIT_CLU, ret=UNIT_RET), dict(PICKS)))
    monkeypatch.setattr(S, 'production_order', lambda: pytest.fail('rows were asked for before the status was read'))
    for part in ('semart_clu', 'semart_ret'):
        with pytest.raises(FileNotFoundError):
            S.score(part, 'clip_b32', inputs=NoTeacher(world))
