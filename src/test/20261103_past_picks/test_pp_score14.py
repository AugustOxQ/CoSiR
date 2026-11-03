"""Tests of pp_score14 (r19 ticket 07): r14's ArtELingo and Affection scores and stage 1 for DINOv2, CLIP B/32, CLIP B/16.

The guarded logic: a call for a new encoder reaches r14's statistic with the new encoder's own arm names; stage 2 reads
the stage 1 values only from the committed record and refuses before any held row when the record is not committed; the
cells carry r19's sc labels, tasks and `data` values; the frozen pick of a new encoder uses r11's / r12's ranges (D21),
never X-c*'s own. Real data are not read: arrays are synthetic and the readers are spies."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import pp_paths as PP
import pp_register  # noqa: F401
import pp_score14 as S

import ac_data
import ac_stats
import bs_affection
import bs_artclu
import bs_artret
import bs_paths
import mt_clu
import rt_data
import rt_readout

NEW = ('dinov2_large', 'clip_b32', 'clip_b16')
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
N = 90


# ------------------------------------------------------------------------------------------------ synthetic world
def labels_of(rng, k=5, missing=6):
    y = rng.integers(0, k, size=N).astype(np.int64)
    y[rng.choice(N, size=missing, replace=False)] = -1
    return y


def partitions_of(rng, y, noise):
    """Three int partitions [3, N] that follow the labels except for `noise` of the rows."""
    out = []
    for _ in range(3):
        p = np.where(y < 0, 0, y).copy()
        flip = rng.random(N) < noise
        p[flip] = rng.integers(0, 6, size=flip.sum())
        out.append(p)
    return np.stack(out).astype(np.int64)


def fixed_world(seed=0):
    rng = np.random.default_rng(seed)
    y = labels_of(rng)
    c = {s: {'r20': partitions_of(rng, y, 0.1)} for s in SEEDS}
    f = {0: {'r20': partitions_of(rng, y, 0.5)}}
    t = {0: {'r20': partitions_of(rng, y, 0.3)}}
    m = {s: {'r20': partitions_of(rng, y, 0.4), 'own': partitions_of(rng, y, 0.45)[:1]} for s in SEEDS}
    return y, c, f, m, t


RANGES = {'emotion': 0.004, 'style': 0.008, 'genre': 0.016}


def check(part, enc, cells):
    return S.check_cells(part, enc, cells, smoke=True)      # tests run with a few draws


def strip(cell):
    return {k: v for k, v in cell.items() if k not in ('n_rows', 'n_labelled')}


# ------------------------------------------------------------------------------------------------ clustering cells
@pytest.mark.parametrize('enc', NEW)
def test_clu_cells_are_r14s_statistics_with_the_new_encoders_arm_names(enc):
    y, c, f, m, t = fixed_world()
    got = S.clu_cells(enc, 'emotion', y, c, f, m, t, RANGES, n_draws=20, workers=2)
    spec = bs_artclu.ENC[enc]
    frozen_name = bs_paths.FROZEN_ARM[enc]
    margins = dict(frozen=bs_artclu.frozen_margin(c, f, y, 20, 2, 'C', frozen_name),
                   matched=bs_artclu.matched_margin(c, m, y, 20, 2, 'C', spec['matched']),
                   teacher=bs_artclu.teacher_margin(c, t, y, 20, 2))
    want = bs_artclu.aspect_cells(enc, 'emotion', 'C', margins, RANGES, 20, spec['matched'], frozen_name)
    assert [strip(x) for x in got] == want                      # same dicts as r14's, nothing recomputed differently
    assert [x['sc'] for x in got] == ['SC1', 'SC3', 'SC6']
    assert got[1]['comparator'] == f'matched_{PP.MATCHED[enc]}' and got[1]['comparator_reading'].startswith(PP.MATCHED[enc])
    assert got[0]['comparator'] == f'frozen_{enc}' and got[2]['comparator'] == 'teacher_emotion'
    assert all(x['n_rows'] == N and x['n_labelled'] == int((y >= 0).sum()) for x in got)


def test_clu_cells_fail_when_an_arm_is_in_the_wrong_slot():
    y, c, f, m, t = fixed_world()
    right = S.clu_cells('clip_b32', 'style', y, c, f, m, t, RANGES, n_draws=20, workers=2)
    swapped = S.clu_cells('clip_b32', 'style', y, c, t, m, f, RANGES, n_draws=20, workers=2)
    assert right[0]['comparator_value'] != swapped[0]['comparator_value']          # frozen slot reads the frozen arm
    assert right[2]['comparator_value'] != swapped[2]['comparator_value']          # teacher slot reads the teacher
    assert right[2]['r'] == RANGES['style'] and right[0]['r'] is None and right[1]['r'] is None


# ------------------------------------------------------------------------------------------------ the held-data runner
class Spy:
    def __init__(self):
        self.calls = []

    def log(self, name, *a, **k):
        self.calls.append((name, a, k))


def install_clu_world(monkeypatch, spy, n=N):
    """Synthetic readers under run_artelingo_clu; every reader logs its call."""
    rng = np.random.default_rng(1)
    names = [f'p{i}' for i in range(n)]
    labels = {a: labels_of(np.random.default_rng(3 + i)) for i, a in enumerate(ASPECTS)}

    def held_setup():
        spy.log('held_setup')
        return names, 'sha', labels

    def load_candidate(cand, scale, aspect, seed):
        spy.log('load_candidate', Path(cand.held_dir), cand.sidecar_arm, scale, aspect, seed)
        return rng.normal(size=(n, 6)).astype(np.float32), f'sha-{aspect}-{seed}'

    def store_loader(enc, nm):
        spy.log('store_loader', enc, tuple(nm))
        return rng.normal(size=(n, 6)).astype(np.float32)

    def instructed(split, aspect, nm):
        spy.log('instructed', split, aspect, tuple(nm))
        return rng.normal(size=(n, 6)).astype(np.float32)

    def kmeans(x):
        x = np.asarray(x)
        return np.stack([np.digitize(x[:, j], np.quantile(x[:, j], np.linspace(0, 1, 7)[1:-1])) for j in range(3)])

    def matched(enc):
        spy.log('matched_fixed', enc)
        return {s: {'r20': partitions_of(rng, labels['emotion'], 0.4), 'own': partitions_of(rng, labels['emotion'], 0.4)[:1]}
                for s in SEEDS}

    monkeypatch.setattr(bs_artclu, 'held_setup', held_setup)
    monkeypatch.setattr(bs_artclu.S0, 'load_candidate', load_candidate)
    monkeypatch.setattr(bs_artclu, 'store_loader', store_loader)
    monkeypatch.setattr(rt_data, '_instructed', instructed)
    monkeypatch.setattr(mt_clu, 'kmeans_partitions', kmeans)
    monkeypatch.setattr(S, 'matched_fixed', matched)
    return names, labels


def record_for(enc):
    """A stage 1 record in r14's form with the blocks the scorers read (synthetic values)."""
    per = {a: {str(s): 0.3 + 0.001 * s for s in SEEDS} for a in ASPECTS}
    ranges = dict(RANGES)
    r12_r = ac_stats.load_ranges(bs_affection.R12_RANGES)['map']['r']
    doc = {m: {'per_seed': [0.1 + 0.001 * i for i in range(5)], 'r': 0.004} for m in ('clu', 'map', 'ndcg')}
    for m in doc:
        doc[m]['r'] = ac_stats.seed_range(doc[m]['per_seed'])
    frozen_ranges = bs_artret.r11_ranges()
    return dict(
        encoder=enc, kind='stage1_record',
        artclu=dict(encoder=enc, kind='artelingo_clustering_validation_ranges', ranges=ranges, per_seed=per, sources={},
                    n_val=6152, labelled={}),
        artret_ranges=dict(encoder=enc, kind='artelingo_retrieval_validation_ranges', ranges=dict(ranges), per_seed=per,
                           sources={}, n_val=6152, labelled={}),
        artret_picks=dict(encoder=enc, kind='artelingo_retrieval_picks',
                          cstar=dict(pick='center+k10', table={}, ranges=dict(ranges), dropped={}),
                          frozen=dict(pick='none+k0', table={}, ranges=frozen_ranges,
                                      source="computed: r11's stage 4 rule with r11's r_a"),
                          teacher=dict(pick='center+k10', source='r11 stored pick, arm instructed'),
                          r11_ranges_sha256=bs_artret.sha_file(bs_artret.R11_RANGES_FILE)),
        affection=dict(encoder=enc, kind='affection_validation_ranges_and_readouts',
                       ranges={m: doc[m]['r'] for m in doc}, ranges_doc=doc,
                       photo_readout=dict(pick='center+k10', scores={'center+k10': dict(map={}, score=0.0, kept=True)}),
                       frozen=dict(pick='none+k0', source="r12's rule with r12's r_ret,mAP", r=r12_r, scores={'none+k0': {}},
                                   view=enc),
                       sources={f'val__s{s}': 'x' for s in SEEDS}))


@pytest.mark.parametrize('enc', NEW)
def test_run_clu_reads_the_new_encoders_own_inputs_and_returns_nine_cells(monkeypatch, enc):
    spy = Spy()
    names, labels = install_clu_world(monkeypatch, spy)
    cells = S.run_artelingo_clu(enc, record_for(enc), n_draws=10, workers=2)
    assert [(c['sc'], c['aspect']) for c in cells] == [(sc, a) for a in ASPECTS for sc in ('SC1', 'SC3', 'SC6')]
    assert {c['encoder'] for c in cells} == {enc} and {c['data'] for c in cells} == {'artelingo_held'}
    loads = [c for c in spy.calls if c[0] == 'load_candidate']
    assert len(loads) == 15
    assert {c[1][0] for c in loads} == {PP.CSTAR_ROOT[enc] / 'held' / 'k11=40'}      # X-c*'s own held exports
    assert {c[1][1] for c in loads} == {'k11=40'} and {c[1][2] for c in loads} == {'full'}
    assert [c[1][1:] for c in spy.calls if c[0] == 'store_loader'] == [(tuple(names),)]
    assert spy.calls[0][0] == 'held_setup' and ('matched_fixed', (enc,), {}) in spy.calls
    assert [c[1][0] for c in spy.calls if c[0] == 'instructed'] == ['held'] * 3     # the teacher: r10's instructed view
    assert all(c['n_rows'] == len(names) for c in cells)
    check('artelingo_clu', enc, cells)


def test_run_clu_sc6_unit_is_the_record_range_not_a_glossary_value(monkeypatch):
    spy = Spy()
    install_clu_world(monkeypatch, spy)
    rec = record_for('clip_b16')
    rec['artclu']['ranges'] = {'emotion': 0.011, 'style': 0.022, 'genre': 0.033}
    cells = S.run_artelingo_clu('clip_b16', rec, n_draws=10, workers=2)
    assert {c['aspect']: c['r'] for c in cells if c['sc'] == 'SC6'} == rec['artclu']['ranges']
    assert all(c['r_glossary'] == bs_artclu.GLOSSARY_R[c['aspect']] for c in cells if c['sc'] == 'SC6')


# ------------------------------------------------------------------------------------------------ matched arm inputs
def test_matched_probs_of_m3_and_m6_come_from_their_stored_held_outputs(monkeypatch):
    asked = []

    def load(path):
        asked.append(Path(path))
        return np.full((12281, 40), 1 / 40, dtype=np.float32)
    monkeypatch.setattr(S, '_load_prob', load)
    for enc, arm, root in (('clip_b32', 'M3', PP.RB_PART2), ('clip_b16', 'M6', PP.EG_PART2),
                           ('clip_l14', 'M7', bs_artclu.ENC['clip_l14']['root'])):
        asked.clear()
        probs = S.matched_probs(enc)
        assert sorted(probs) == list(SEEDS)
        assert asked == [root / f'full/held/{arm}/P3__s{s}.npz' for s in SEEDS]


def test_matched_probs_of_dinov2_come_from_pp_matched_in_held_order(monkeypatch):
    class FakeMatched:
        @staticmethod
        def held_probs(enc):
            assert enc == 'dinov2_large'
            return {s: np.full((12281, 40), 1 / 40, dtype=np.float32) for s in SEEDS}
    monkeypatch.setitem(__import__('sys').modules, 'pp_matched', FakeMatched)
    probs = S.matched_probs('dinov2_large')
    assert sorted(probs) == list(SEEDS) and probs[0].shape == (12281, 40)


@pytest.mark.parametrize('bad', ['rows', 'seeds', 'width'])
def test_matched_probs_refuse_a_misaligned_table(monkeypatch, bad):
    shape = {'rows': (12280, 40), 'width': (12281, 41)}.get(bad, (12281, 40))

    class FakeMatched:
        @staticmethod
        def held_probs(enc):
            keep = SEEDS[:4] if bad == 'seeds' else SEEDS
            return {s: np.full(shape, 1 / 40, dtype=np.float32) for s in keep}
    monkeypatch.setitem(__import__('sys').modules, 'pp_matched', FakeMatched)
    with pytest.raises(AssertionError):
        S.matched_probs('dinov2_large')


# ------------------------------------------------------------------------------------------------ retrieval
def fake_ap(rng, nq, shift):
    return dict(ap=(rng.random(nq) * 0.5 + shift).astype(np.float64), rows=np.arange(nq), p10=np.zeros(nq), dropped=0)


@pytest.mark.parametrize('enc', NEW)
def test_run_ret_scores_the_computed_frozen_pick_through_r14s_function(monkeypatch, enc):
    spy = Spy()
    rng = np.random.default_rng(5)
    nq = 40

    def cstar_held(root, pick, outputs, workers=8, stage=5):
        spy.log('cstar', Path(root), pick)
        return {a: dict(ap=(rng.random((5, nq)) * 0.5 + 0.3), rows=np.arange(nq), p10=0, dropped=0) for a in ASPECTS}

    def frozen_held(view, arm, pick, outputs, workers=8, stage=5):
        spy.log('frozen', view, arm, pick, Path(outputs))
        return {a: fake_ap(rng, nq, 0.2) for a in ASPECTS}

    def r11_stored(arm, aspect):
        spy.log('r11_stored', arm, aspect)
        return dict(fake_ap(rng, nq, 0.25), pick='center+k10')
    monkeypatch.setattr(bs_artret, 'score_cstar_held', cstar_held)
    monkeypatch.setattr(bs_artret, 'score_frozen_held', frozen_held)
    monkeypatch.setattr(bs_artret, 'r11_stored', r11_stored)
    rec = record_for(enc)
    cells = S.run_artelingo_ret(enc, rec, outputs='/nonexistent/out', n_draws=30, workers=2)
    assert [(c['sc'], c['aspect']) for c in cells] == [(sc, a) for a in ASPECTS for sc in ('SC2', 'SC6')]
    assert ('cstar', (PP.CSTAR_ROOT[enc], 'center+k10'), {}) in spy.calls          # X-c*'s own exports and its pick
    frozen = [c for c in spy.calls if c[0] == 'frozen']
    assert frozen == [('frozen', (enc, f'frozen_{enc}', 'none+k0', Path('/nonexistent/out') / enc / 'frozen'), {})]
    assert {c[1] for c in spy.calls if c[0] == 'r11_stored'} == {('instructed', a) for a in ASPECTS}   # the teacher only
    assert {c['data'] for c in cells} == {'artelingo_held'} and all(c['n_rows'] == 12281 for c in cells)
    assert {c['aspect']: c['r'] for c in cells if c['sc'] == 'SC6'} == rec['artret_ranges']['ranges']
    assert all(c['candidate_readout'] == 'center+k10' and c['comparator_readout'] in ('none+k0', 'center+k10') for c in cells)
    check('artelingo_ret', enc, cells)


# ------------------------------------------------------------------------------------------------ Affection
def test_run_affection_uses_the_records_picks_and_ranges(monkeypatch, tmp_path):
    spy = Spy()
    labels = np.arange(8501) % 9

    class FakeArm:
        def __init__(self, arm, root=None):
            spy.log('PhotoArm', arm, Path(root))
            self.keys = {'val': ['v0']}

        def test_keys(self):
            return ['t0']

        def candidate(self, readout):
            spy.log('candidate', readout)
            return type('C', (), dict(name='photo_x', readout=readout))()

    def frozen_comparator(enc, readout, val_keys, test_keys, labels_, partition_root, workers=8):
        spy.log('frozen_comparator', enc, readout)
        return dict(arm=f'frozen_{enc}')

    def score_cells(enc, cand, r, labels_, comparators, partition_root, draws_clu, draws_ret, workers):
        spy.log('score_cells', enc, cand.readout, dict(r), comparators['frozen']['arm'], comparators['teacher'], draws_clu,
                draws_ret)
        return dict(cells=[], partitions=None, picked=None)
    import pf_stage3
    monkeypatch.setattr(bs_affection, 'PhotoArm', FakeArm)
    monkeypatch.setattr(bs_affection, 'frozen_comparator', frozen_comparator)
    monkeypatch.setattr(bs_affection, 'score_cells', score_cells)
    monkeypatch.setattr(ac_data, 'labels', lambda split: labels)
    monkeypatch.setattr(pf_stage3, 'load_comparators', lambda picks, n: dict(teacher={'arm': 'teacher_artwork'}))
    rec = record_for('clip_b32')
    rec['affection']['photo_readout']['pick'] = 'center+k3'
    rec['affection']['frozen']['pick'] = 'whiten256+k1'
    rec['affection']['photo_readout']['scores'] = {'center+k3': {}}
    out = S.run_affection('clip_b32', rec, root=tmp_path, cache=tmp_path / 'c', n_clu=7, n_ret=9, workers=2)
    assert out['cells'] == []
    names = [c[0] for c in spy.calls]
    assert ('PhotoArm', ('photo_clip_b32', tmp_path), {}) in spy.calls               # X photo model's exports
    assert ('candidate', ('center+k3',), {}) in spy.calls                              # the record's photo pick
    assert ('frozen_comparator', ('clip_b32', 'whiten256+k1'), {}) in spy.calls       # the record's frozen pick
    (sc,) = [c for c in spy.calls if c[0] == 'score_cells']
    assert sc[1][:2] == ('clip_b32', 'center+k3') and sc[1][2] == {'clu': rec['affection']['ranges']['clu'],
                                                                   'map': rec['affection']['ranges']['map']}
    assert sc[1][5:] == (7, 9)
    assert names.index('PhotoArm') < names.index('score_cells')


# ------------------------------------------------------------------------------------------------ the stage 1 gate
@pytest.fixture
def tmp_records(monkeypatch):
    """PP.RECORDS pointed at a throw-away folder inside the repo (so `git show HEAD:` finds no such file)."""
    folder = PP.ROUND / '_test_records'
    folder.mkdir(exist_ok=True)
    monkeypatch.setattr(PP, 'RECORDS', folder)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def tripwires(monkeypatch):
    """Every held reader of the three scorers fails the test if it is reached."""
    def boom(*a, **k):
        raise AssertionError('a held reader was reached before the stage 1 record was read')
    monkeypatch.setattr(bs_artclu, 'held_setup', boom)
    monkeypatch.setattr(bs_artclu.S0, 'load_candidate', boom)
    monkeypatch.setattr(bs_artret, 'score_cstar_held', boom)
    monkeypatch.setattr(bs_artret, 'score_frozen_held', boom)
    monkeypatch.setattr(bs_affection, 'PhotoArm', boom)
    monkeypatch.setattr(ac_data, 'keys', boom)
    monkeypatch.setattr(ac_data, 'labels', boom)


@pytest.mark.parametrize('part', S.PARTS)
def test_stage2_refuses_an_uncommitted_record_before_any_held_row(monkeypatch, tmp_records, part, tmp_path):
    tripwires(monkeypatch)
    (tmp_records / 'stage1_clip_b32.json').write_text(json.dumps(record_for('clip_b32')))      # on disk, not in HEAD
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    with pytest.raises(RuntimeError, match='not committed'):
        S.score(part, 'clip_b32', n_draws=5)
    assert not list(tmp_path.glob('**/cells_*.json'))


@pytest.mark.parametrize('part', S.PARTS)
def test_stage2_refuses_a_missing_record(monkeypatch, tmp_records, part):
    tripwires(monkeypatch)
    with pytest.raises(FileNotFoundError):
        S.score(part, 'dinov2_large', n_draws=5)


@pytest.mark.parametrize('part', S.PARTS)
def test_stage2_reads_stage1_only_through_read_committed_record(monkeypatch, part, tmp_path):
    tripwires(monkeypatch)
    asked = []

    def read(name):
        asked.append(name)
        raise RuntimeError('stop here')
    monkeypatch.setattr(PP, 'read_committed_record', read)
    # a stage 1 file in OUT/stage1 must not be a way around the record
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    (tmp_path / 'stage1').mkdir()
    for f in ('artclu_ranges', 'artret_ranges', 'artret_picks', 'affection_ranges'):
        (tmp_path / 'stage1' / f'{f}_clip_b16.json').write_text('{}')
    with pytest.raises(RuntimeError, match='stop here'):
        S.score(part, 'clip_b16', n_draws=5)
    assert asked == ['stage1_clip_b16.json']


@pytest.mark.parametrize('part', S.PARTS)
def test_stage2_refuses_draw_counts_other_than_the_protocols_outside_smoke(monkeypatch, part, tmp_path):
    """A committed record is read, then a 5-draw run is refused before any held row (ticket 11 refuses such cells anyway)."""
    tripwires(monkeypatch)
    monkeypatch.setattr(S, 'committed_record', lambda enc: record_for(enc))
    monkeypatch.setattr(PP, 'SMOKE', 0)
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    with pytest.raises(AssertionError, match='only smoke runs'):
        S.score(part, 'clip_b32', n_draws=5)


def test_stage2_refuses_an_encoder_that_is_not_one_of_the_new_three():
    with pytest.raises(AssertionError):
        S.score('artelingo_clu', 'qwen', n_draws=5)


# ------------------------------------------------------------------------------------------------ the record's checks
def test_record_check_accepts_a_good_record_and_names_the_encoder():
    for enc in NEW:
        S.check_record(enc, record_for(enc))
    rec = record_for('clip_b32')
    with pytest.raises(AssertionError):
        S.check_record('clip_b16', rec)


def test_record_check_refuses_a_frozen_pick_made_with_the_encoders_own_ranges():
    """D21: the new frozen image picks with r11's r_a, never X-c*'s own ranges."""
    rec = record_for('clip_b32')
    rec['artret_picks']['frozen']['ranges'] = dict(rec['artret_ranges']['ranges'])
    with pytest.raises(AssertionError, match='r11'):
        S.check_record('clip_b32', rec)


def test_record_check_refuses_an_affection_frozen_pick_without_r12s_range():
    rec = record_for('clip_b16')
    rec['affection']['frozen']['r'] = rec['affection']['ranges']['map']            # the photo model's own, wrong
    with pytest.raises(AssertionError, match='r12'):
        S.check_record('clip_b16', rec)
    rec = record_for('clip_b16')
    rec['affection']['frozen']['r'] = None
    with pytest.raises(AssertionError, match='r12'):
        S.check_record('clip_b16', rec)


def test_record_check_refuses_picks_made_with_other_ranges():
    rec = record_for('dinov2_large')
    rec['artret_picks']['cstar']['ranges']['style'] += 1e-6
    with pytest.raises(AssertionError):
        S.check_record('dinov2_large', rec)


def test_record_check_keeps_an_old_encoders_stored_frozen_pick():
    """The gate replays CLIP L/14: its frozen pick is r11's stored one (no ranges), its Affection pick r12's (r null)."""
    rec = record_for('clip_l14')
    rec['artret_picks']['frozen'] = dict(pick='center+k10', source='r11 stored pick, arm clip')
    rec['affection']['frozen'] = dict(pick='center+k10', source='r12 pick', r=None, scores=None, view='clip_l14')
    S.check_record('clip_l14', rec)


# ------------------------------------------------------------------------------------------------ cells: labels and data
def cell(sc, aspect, task, data, **kw):
    base = dict(sc=sc, encoder='clip_b32', aspect=aspect, task=task, data=data, candidate='cstar_clip_b32', comparator='x',
                candidate_value=0.3, comparator_value=0.2, d=0.1, lo=0.05, hi=0.15, r=0.01 if sc == 'SC6' else None,
                r_source='s' if sc == 'SC6' else None, n_draws=10, seeds=[0, 1, 2, 3, 4], n_rows=100)
    base.update(kw)
    return base


def good(part):
    if part == 'artelingo_clu':
        return [cell(sc, a, 'clu', 'artelingo_held') for a in ASPECTS for sc in ('SC1', 'SC3', 'SC6')]
    if part == 'artelingo_ret':
        return [cell(sc, a, 'ret', 'artelingo_held') for a in ASPECTS for sc in ('SC2', 'SC6')]
    return [cell('SC4', 'emotion', 'clu', 'affection_test'), cell('SC5', 'emotion', 'ret', 'affection_test'),
            cell('SC6', 'emotion', 'clu', 'affection_test'), cell('SC6', 'emotion', 'ret', 'affection_test')]


@pytest.mark.parametrize('part', S.PARTS)
def test_check_cells_accepts_the_right_labels_and_data(part):
    check(part, 'clip_b32', good(part))


@pytest.mark.parametrize('part', S.PARTS)
def test_check_cells_refuses_a_wrong_data_value(part):
    cells = good(part)
    cells[0]['data'] = 'mmart_test'
    with pytest.raises(AssertionError):
        check(part, 'clip_b32', cells)


@pytest.mark.parametrize('part', S.PARTS)
def test_check_cells_refuses_a_missing_or_extra_or_duplicate_cell(part):
    with pytest.raises(AssertionError):
        check(part, 'clip_b32', good(part)[1:])
    with pytest.raises(AssertionError):
        check(part, 'clip_b32', good(part) + good(part)[:1])
    with pytest.raises(AssertionError):
        check(part, 'clip_b32', good(part)[:-1] + [dict(good(part)[0], sc='SC7')])


def test_check_cells_refuses_r14s_numbers_under_another_part():
    with pytest.raises(AssertionError):
        check('artelingo_ret', 'clip_b32', good('artelingo_clu'))
    with pytest.raises(AssertionError):
        check('affection', 'clip_b32', good('artelingo_ret'))


def test_check_cells_refuses_a_unit_on_a_win_bar_and_none_on_a_teacher_cell():
    for part in ('artelingo_clu', 'artelingo_ret', 'affection'):
        cells = good(part)
        i = next(i for i, c in enumerate(cells) if c['sc'] != 'SC6')
        cells[i]['r'] = 0.01
        with pytest.raises(AssertionError):
            check(part, 'clip_b32', cells)
        cells = good(part)
        i = next(i for i, c in enumerate(cells) if c['sc'] == 'SC6')
        cells[i]['r'] = None
        with pytest.raises(AssertionError):
            check(part, 'clip_b32', cells)


def test_check_cells_refuses_the_wrong_encoder_and_nonfinite_values():
    with pytest.raises(AssertionError):
        check('artelingo_clu', 'clip_b16', good('artelingo_clu'))
    cells = good('artelingo_clu')
    cells[2]['lo'] = float('nan')
    with pytest.raises(AssertionError):
        check('artelingo_clu', 'clip_b32', cells)


def test_check_cells_keeps_the_draw_count_of_one_run():
    cells = good('artelingo_clu')
    cells[3]['n_draws'] = 11
    with pytest.raises(AssertionError):
        check('artelingo_clu', 'clip_b32', cells)


def test_cells_file_names_follow_the_index():
    assert S.cells_path('artelingo_clu', out=Path('/o')) == Path('/o/stage2/cells_artelingo_clu.json')
    assert S.cells_path('artelingo_ret', out=Path('/o')) == Path('/o/stage2/cells_artelingo_ret.json')
    assert S.cells_path('affection', out=Path('/o')) == Path('/o/stage2/cells_affection.json')


def test_write_cells_replaces_one_encoders_cells_and_keeps_the_others(tmp_path):
    a = good('artelingo_clu')
    b = [dict(c, encoder='clip_b16', candidate='cstar_clip_b16') for c in good('artelingo_clu')]
    S.write_cells('artelingo_clu', 'clip_b32', a, out=tmp_path)
    S.write_cells('artelingo_clu', 'clip_b16', b, out=tmp_path)
    again = [dict(c, d=0.5) for c in a]
    S.write_cells('artelingo_clu', 'clip_b32', again, out=tmp_path)
    got = json.loads(S.cells_path('artelingo_clu', out=tmp_path).read_text())
    assert [c['d'] for c in got if c['encoder'] == 'clip_b32'] == [0.5] * 9
    assert [c['d'] for c in got if c['encoder'] == 'clip_b16'] == [0.1] * 9


# ------------------------------------------------------------------------------------------------ stage 1
def test_stage1_artelingo_runs_r14s_functions_on_the_new_encoders_root(monkeypatch):
    seen = {}
    ranges_c = dict(ranges=dict(RANGES), per_seed={}, sources={}, n_val=6152, labelled={})
    ranges_r = dict(ranges={'emotion': 0.001, 'style': 0.002, 'genre': 0.003}, per_seed={}, sources={}, n_val=6152,
                    labelled={})

    def clu(root, seeds=bs_artclu.SEEDS):
        seen['clu'] = Path(root)
        return ranges_c

    def ret(root, seeds=bs_artret.SEEDS):
        seen['ret'] = Path(root)
        return ranges_r

    def picks(enc, root, ranges, workers=8):
        seen['picks'] = (enc, Path(root), ranges)
        return dict(encoder=enc, kind='artelingo_retrieval_picks')
    monkeypatch.setattr(bs_artclu, 'clustering_ranges', clu)
    monkeypatch.setattr(bs_artret, 'validation_ranges', ret)
    monkeypatch.setattr(bs_artret, 'stage1_picks', picks)
    got = S.stage1_artelingo('clip_b16')
    root = PP.CSTAR_ROOT['clip_b16']
    assert seen == dict(clu=root, ret=root, picks=('clip_b16', root, ranges_r['ranges']))
    assert got['artclu'] == dict(encoder='clip_b16', kind='artelingo_clustering_validation_ranges', **ranges_c)
    assert got['artret_ranges'] == dict(encoder='clip_b16', kind='artelingo_retrieval_validation_ranges', **ranges_r)
    assert got['artret_picks']['encoder'] == 'clip_b16'
    assert set(got) == {'artclu', 'artret_ranges', 'artret_picks'}


def test_stage1_forms_are_what_r14s_writers_write(tmp_path):
    result = dict(ranges=dict(RANGES), per_seed={'emotion': {0: 0.1}}, sources={}, n_val=6152, labelled={})
    theirs = bs_artclu.write_ranges(tmp_path / 'a.json', 'clip_b32', result)
    assert json.loads(json.dumps(S.artclu_block('clip_b32', result))) == json.loads((tmp_path / 'a.json').read_text())
    assert theirs == S.artclu_block('clip_b32', result)
    bs_artret.write_ranges(tmp_path / 'b.json', 'clip_b32', result)
    assert json.loads(json.dumps(S.artret_block('clip_b32', result))) == json.loads((tmp_path / 'b.json').read_text())


def test_stage1_new_frozen_image_picks_by_r11s_rule_with_r11s_ranges(monkeypatch):
    """D21 through the registration: r14's `stage1_picks` computes the frozen pick of a new encoder with r11's r_a."""
    menu = rt_readout.menu(512)
    rows = {ro: {a: 0.3 + 0.001 * i for a in ASPECTS} for i, ro in enumerate(menu)}
    seen = {}

    def frozen_rows(view, workers=8):
        seen['view'] = view
        return rows
    monkeypatch.setattr(bs_artret, 'frozen_rows', frozen_rows)
    monkeypatch.setattr(bs_artret, 'pick_cstar', lambda root, ranges, workers=8: dict(pick='center+k10', table={},
                                                                                      ranges=ranges, dropped={}))
    own = {'emotion': 0.5, 'style': 0.6, 'genre': 0.7}
    rec = bs_artret.stage1_picks('clip_b32', PP.CSTAR_ROOT['clip_b32'], own)
    assert seen['view'] == 'clip_b32'
    assert rec['frozen']['source'].startswith('computed') and rec['frozen']['ranges'] == bs_artret.r11_ranges()
    assert rec['frozen']['ranges'] != own and rec['cstar']['ranges'] == own
    assert rec['teacher']['pick'] == 'center+k10'
    S.check_record('clip_b32', dict(record_for('clip_b32'), artret_picks=rec,
                                    artret_ranges=dict(record_for('clip_b32')['artret_ranges'], ranges=own)))


def test_stage1_affection_writes_nothing_and_uses_r12s_rule_for_the_new_frozen_image(monkeypatch, tmp_path):
    seen = {}
    for s in SEEDS:
        p = tmp_path / 'exports/affection/photo_clip_b32'
        p.mkdir(parents=True, exist_ok=True)
        (p / f'val__s{s}.json').write_text(json.dumps(dict(sha256=f'sha{s}')))

    class FakeArm:
        def __init__(self, arm, root=None):
            seen['arm'] = (arm, Path(root))

        def embed(self, split, seed):
            return np.zeros((2, 3), dtype=np.float32)

        def stored(self, split, seed):
            return np.zeros((2, 3), dtype=np.float16)
    doc = {m: {'per_seed': [0.1, 0.2, 0.3, 0.4, 0.5], 'r': 0.4} for m in ('clu', 'map', 'ndcg')}

    def photo_stage1(embed_val, stored_val, labels, dist, keys, *, r12_ranges, width=2048, workers=8):
        seen['photo'] = sorted(embed_val)
        return dict(ranges=doc, readout=dict(pick='center+k10', scores={'center+k10': {}}))

    def frozen_readout(enc, x_val, labels, keys, r12_ranges, r12_picks, workers=8, force_rule=False):
        seen['frozen'] = (enc, force_rule)
        return dict(pick='none+k0', source="r12's rule with r12's r_ret,mAP", r=float(r12_ranges['map']['r']), scores={})
    monkeypatch.setattr(ac_data, 'keys', lambda split: ['k'])
    monkeypatch.setattr(ac_data, 'labels', lambda split: np.zeros(1, dtype=np.int64))
    monkeypatch.setattr(ac_data, 'emotion_dist', lambda split: np.zeros((1, 9)))
    monkeypatch.setattr(bs_affection, 'PhotoArm', FakeArm)
    monkeypatch.setattr(bs_affection, 'photo_stage1', photo_stage1)
    monkeypatch.setattr(bs_affection, 'frozen_readout', frozen_readout)
    monkeypatch.setattr(bs_affection, 'frozen_embed', lambda enc, keys: np.zeros((1, 3), dtype=np.float32))
    rec = S.stage1_affection('clip_b32', root=tmp_path)
    assert seen['arm'] == ('photo_clip_b32', tmp_path) and seen['photo'] == list(SEEDS) and seen['frozen'] == ('clip_b32', False)
    assert rec['encoder'] == 'clip_b32' and rec['kind'] == 'affection_validation_ranges_and_readouts'
    assert rec['frozen']['view'] == 'clip_b32' and rec['sources'] == {f'val__s{s}': f'sha{s}' for s in SEEDS}
    assert not (tmp_path / 'stage1').exists()                                        # computing writes nothing


def test_stage1_files_follow_r14s_names_and_the_record_is_r14s_form(tmp_path):
    blocks = dict(artclu={'a': 1}, artret_ranges={'b': 2}, artret_picks={'c': 3}, affection={'d': 4})
    paths = S.write_stage1('clip_b32', blocks, out=tmp_path)
    assert {k: v.name for k, v in paths.items()} == dict(artclu='artclu_ranges_clip_b32.json',
                                                         artret_ranges='artret_ranges_clip_b32.json',
                                                         artret_picks='artret_picks_clip_b32.json',
                                                         affection='affection_ranges_clip_b32.json')
    assert all(v.parent == tmp_path / 'stage1' for v in paths.values())
    rec = S.stage1_record('clip_b32', out=tmp_path)
    assert rec == dict(encoder='clip_b32', kind='stage1_record', **blocks)
    with pytest.raises(AssertionError):                                              # picks are frozen once written
        S.write_stage1('clip_b32', dict(blocks, artclu={'a': 2}), out=tmp_path)
    S.write_stage1('clip_b32', blocks, out=tmp_path)                                  # the same content again is fine


def test_stage1_refuses_an_encoder_with_no_x_c_star():
    with pytest.raises(AssertionError):
        S.stage1_artelingo('qwen')
