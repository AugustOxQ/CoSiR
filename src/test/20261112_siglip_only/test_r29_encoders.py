"""Tests of r29_encoders (ticket 10): the stage 2 SC4 path for SigLIP 2 (E2) and CLIP L/14 (E3).

Records in the tests are built here (ticket 05's `prefix_pick.json` is not merged: the four keys of the index's
Shared interfaces). Heavy steps (trainer, scorers, ladder) are replaced by recorders; the store check runs the real
`r29_extract.check_store` on tiny stores, and `beside` reads the real frozen r17 and r14 verdicts."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_encoders as E  # noqa: E402
import r29_paths as P  # noqa: E402

ASP = ('emotion', 'style', 'genre')
K = {'emotion': 9, 'style': 27, 'genre': 10}


# ------------------------------------------------------------------------------------------------ records
@pytest.fixture
def rec(tmp_path, monkeypatch):
    """An empty records folder that the module reads as its records."""
    monkeypatch.setattr(E, 'records_dir', lambda: tmp_path)
    return tmp_path


def put(rec, name, doc):
    path = rec / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def pass_rule(rec, status='pass'):
    return put(rec, 'pass_rule.json', dict(status=status))


def prefix_pick(rec, form='plain', kstar=None, flag=None, **extra):
    kstar = kstar or ('siglip1' if form == 'plain' else 'siglip1_prefix')
    return put(rec, 'prefix_pick.json', dict(kstar=kstar, kstar_form=form, p1_form='plain', flag=flag, **extra))


def gate(rec, source, status='pass', tries=1):
    return put(rec, f'gates/viii_{source}.json', dict(gate=f'viii_{source}', status=status, tries=tries, details={}, written='x'))


# ------------------------------------------------------------------------------------------------ D3: pass rule
@pytest.mark.parametrize('status', ['fail', 'parked', 'fallback', None])
def test_require_pass_refuses_any_status_but_pass(rec, status):
    put(rec, 'pass_rule.json', dict(status=status))
    with pytest.raises(RuntimeError, match='pass'):
        E.require_pass()


def test_require_pass_refuses_an_absent_record(rec):
    with pytest.raises(RuntimeError, match='pass_rule.json'):
        E.require_pass()


def test_require_pass_accepts_pass(rec):
    pass_rule(rec)
    E.require_pass()


@pytest.mark.parametrize('argv', [
    ['stores', '--encoder', 'siglip2'],
    ['train', '--encoder', 'siglip2'],
    ['export', '--encoder', 'clip_l14'],
    ['picks', '--encoder', 'clip_l14'],
    ['cells', '--encoder', 'siglip2'],
    ['ladder', '--encoder', 'siglip2'],
])
def test_every_command_refuses_without_a_pass(rec, argv):
    prefix_pick(rec)          # the pass rule is the first thing read: nothing else is needed to refuse
    put(rec, 'pass_rule.json', dict(status='fail'))
    with pytest.raises(RuntimeError, match='pass'):
        E.main(argv)


# ------------------------------------------------------------------------------------------------ D13: caption source
def test_prefix_pick_keys_are_asserted(rec):
    put(rec, 'prefix_pick.json', dict(kstar='siglip1', kstar_form='plain', p1_form='plain'))     # no flag
    with pytest.raises(AssertionError, match='flag'):
        E.read_prefix_pick()


def test_prefix_pick_absent_names_ticket_05(rec):
    with pytest.raises(RuntimeError, match='prefix_pick.json'):
        E.read_prefix_pick()


@pytest.mark.parametrize('kstar, form', [('siglip1', 'prefixed'), ('siglip1_prefix', 'plain'), ('siglip2', 'plain'),
                                         ('siglip1', 'both')])
def test_prefix_pick_form_must_agree_with_kstar(rec, kstar, form):
    put(rec, 'prefix_pick.json', dict(kstar=kstar, kstar_form=form, p1_form='plain', flag=None))
    with pytest.raises(AssertionError):
        E.read_prefix_pick()


@pytest.mark.parametrize('enc, form, source, width', [
    ('siglip2', 'plain', 'siglip2', 1152), ('siglip2', 'prefixed', 'siglip2_prefix', 1152),
    ('clip_l14', 'plain', 'clipl14', 768), ('clip_l14', 'prefixed', 'clipl14_prefix', 768)])
def test_caption_source_follows_kstars_form(rec, enc, form, source, width):
    prefix_pick(rec, form)
    assert E.kstar_source(enc) == source
    assert P.CAPTION_SOURCES[source][1] == width
    assert source in P.ENCODER_CAPTION_SOURCES[enc]


@pytest.mark.parametrize('enc', ['siglip1', 'qwen', 'clipl14', 'clip_b16'])
def test_only_e2_and_e3_are_this_tickets_encoders(rec, enc):
    prefix_pick(rec)
    with pytest.raises(AssertionError, match='encoder'):
        E.kstar_source(enc)


def test_flagged_pick_is_carried_not_judged(rec):
    prefix_pick(rec, 'plain', flag='K2 out: K1 is K* with no comparison')
    assert E.read_prefix_pick()['flag'].startswith('K2 out')
    assert E.kstar_source('siglip2') == 'siglip2'


# ------------------------------------------------------------------------------------------------ gate (viii)
def ok_store(n_truncated=0, limit=64):
    return lambda source: dict(ok=True, n_texts=10, n_truncated=n_truncated, truncation_limit=limit, max_norm_dev=1e-4)


def ok_join(source):
    return dict(status='pass', details=dict(checks=dict(same_ids=True), n_captions=10))


def test_gate_viii_writes_a_pass_record_with_the_truncation_count(rec):
    out = E.gate_viii('clipl14_prefix', store_check=ok_store(3, 77), join_check=ok_join)
    assert (out['gate'], out['status'], out['tries']) == ('viii_clipl14_prefix', 'pass', 1)
    assert out['details']['store']['n_truncated'] == 3 and out['details']['store']['truncation_limit'] == 77
    assert 'cut' in out['details']['truncation_flag']          # D14: cut at the limit, count, flag
    assert json.loads((rec / 'gates/viii_clipl14_prefix.json').read_text()) == out
    assert out['written']


def test_gate_viii_no_flag_without_truncation(rec):
    out = E.gate_viii('siglip2', store_check=ok_store(0), join_check=ok_join)
    assert out['status'] == 'pass' and out['details']['truncation_flag'] is None


def test_gate_viii_fails_when_the_join_fails(rec):
    bad = lambda s: dict(status='fail', details=dict(checks=dict(same_ids=False)))   # noqa: E731
    out = E.gate_viii('siglip2', store_check=ok_store(), join_check=bad)
    assert out['status'] == 'fail' and out['details']['join']['checks']['same_ids'] is False


def test_gate_viii_records_a_failed_store_check_instead_of_raising(rec):
    def boom(source):
        raise AssertionError('finite/norm gate: ...')
    out = E.gate_viii('siglip2', store_check=boom, join_check=ok_join)
    assert out['status'] == 'fail' and 'finite/norm' in out['details']['error']


def test_gate_viii_three_tries_then_refuses(rec):
    boom = lambda s: (_ for _ in ()).throw(AssertionError('x'))   # noqa: E731
    tries = [E.gate_viii('siglip2', store_check=boom, join_check=ok_join)['tries'] for _ in range(3)]
    assert tries == [1, 2, 3]
    with pytest.raises(RuntimeError, match='three tries'):
        E.gate_viii('siglip2', store_check=ok_store(), join_check=ok_join)
    assert json.loads((rec / 'gates/viii_siglip2.json').read_text())['tries'] == 3


def test_gate_viii_is_not_rerun_after_a_pass(rec):
    E.gate_viii('siglip2', store_check=ok_store(), join_check=ok_join)
    with pytest.raises(RuntimeError, match='already passed'):
        E.gate_viii('siglip2', store_check=ok_store(), join_check=ok_join)


def test_gate_viii_accepts_only_the_four_store_names(rec):
    for source in ('siglip1', 'siglip1_prefix', 'qwen', 'viii_siglip2'):
        with pytest.raises(AssertionError, match='store'):
            E.gate_viii(source, store_check=ok_store(), join_check=ok_join)


def test_stores_command_gates_each_encoders_kstar_store(rec, monkeypatch):
    pass_rule(rec)
    prefix_pick(rec, 'prefixed')
    seen = []
    monkeypatch.setattr(E, 'gate_viii', lambda source, **kw: seen.append(source) or dict(status='pass', tries=1))
    assert E.main(['stores', '--encoder', 'all']) == 0
    assert seen == ['siglip2_prefix', 'clipl14_prefix']


def test_stores_command_exits_1_on_a_failed_gate(rec, monkeypatch):
    pass_rule(rec)
    prefix_pick(rec)
    monkeypatch.setattr(E, 'gate_viii', lambda source, **kw: dict(status='fail', tries=1))
    assert E.main(['stores', '--encoder', 'siglip2']) == 1


def test_a_missing_prefixed_store_names_the_extractor_command(rec):
    pass_rule(rec)
    prefix_pick(rec, 'prefixed')
    with pytest.raises(RuntimeError, match=r'r29_extract.py extract --encoder siglip2_so400m'):
        E.store_check('siglip2_prefix', folder=rec / 'no_such_store')


# the real check on tiny stores (r29_extract's own test fixtures)
def _tiny(tmp_path, cls_name, texts, prefix=''):
    import test_r29_extract as T
    from src.data import feature_extract as F
    cls = getattr(F, cls_name)
    enc = T.encoder(cls)
    ref = T.reference(len(texts))
    ref_path = tmp_path / 'ref.json'
    ref_path.write_text(json.dumps(ref))
    folder = tmp_path / 'store'
    import r29_extract
    r29_extract.extract_store(ref, texts, enc, folder, prefix=prefix)
    return folder, ref


def test_store_check_counts_clip_truncation_at_77(tmp_path):
    long = ' '.join(['word'] * 90)                      # 90 words + the end token: over CLIP's 77
    folder, ref = _tiny(tmp_path, 'ClipL14', ['a b', long, 'c d'], prefix=P.PREFIX)
    out = E.store_check('clipl14_prefix', folder=folder, reference=ref, width=3)
    assert out['ok'] and out['n_truncated'] == 1 and out['truncation_limit'] == 77 and out['n_texts'] == 3


def test_store_check_a_prefixed_store_must_carry_the_prefix(tmp_path):
    folder, ref = _tiny(tmp_path, 'Siglip2So400m', ['a b', 'c d', 'e f'], prefix='')
    with pytest.raises(AssertionError, match='prefix'):
        E.store_check('siglip2_prefix', folder=folder, reference=ref, width=3)
    assert E.store_check('siglip2', folder=folder, reference=ref, width=3)['n_truncated'] == 0


def test_store_check_keys_join_by_id_and_painting(tmp_path):
    folder, ref = _tiny(tmp_path, 'Siglip2So400m', ['a b', 'c d', 'e f'])
    wrong = json.loads(json.dumps(ref))
    wrong['texts'][1]['painting'] = 'zz'                 # same sample ID, another painting (C6)
    with pytest.raises(AssertionError, match='painting'):
        E.store_check('siglip2', folder=folder, reference=wrong, width=3)


def test_store_check_width_is_the_sources(tmp_path):
    folder, ref = _tiny(tmp_path, 'ClipL14', ['a b', 'c d', 'e f'])
    with pytest.raises(AssertionError, match='width'):
        E.store_check('clipl14', folder=folder, reference=ref, width=768)       # tiny store has dim 3


# ------------------------------------------------------------------------------------------------ training and exports
class FakeTrainer:
    ENCODERS = ('siglip1', 'siglip2', 'clip_l14')

    def __init__(self):
        self.calls = []

    def train(self, enc, source, seeds, device, root, **kw):
        self.calls.append(('train', enc, source, list(seeds), device, Path(root), kw))
        return {}

    def export(self, enc, source, seeds, root, **kw):
        self.calls.append(('export', enc, source, list(seeds), Path(root), kw))
        return {}


@pytest.fixture
def fake(rec, monkeypatch):
    f = FakeTrainer()
    monkeypatch.setattr(E, '_trainer', lambda: f)
    pass_rule(rec)
    return f


def test_train_uses_the_store_of_kstars_form_for_the_encoder(rec, fake):
    prefix_pick(rec, 'prefixed')
    gate(rec, 'clipl14_prefix')
    assert E.main(['train', '--encoder', 'clip_l14', '--seeds', '0', '3']) == 0
    (call,) = fake.calls
    assert call[:5] == ('train', 'clip_l14', 'clipl14_prefix', [0, 3], 'cuda')
    assert call[5] == P.OUT / 'cstar' / 'clip_l14'


def test_train_defaults_to_the_five_seeds(rec, fake):
    prefix_pick(rec, 'plain')
    gate(rec, 'siglip2')
    E.main(['train', '--encoder', 'siglip2'])
    assert fake.calls[0][3] == [0, 1, 2, 3, 4]


def test_train_refuses_without_the_store_gate(rec, fake):
    prefix_pick(rec, 'prefixed')
    gate(rec, 'clipl14')                                 # the plain store's gate is not the prefixed store's
    with pytest.raises(RuntimeError, match='viii_clipl14_prefix not run'):
        E.main(['train', '--encoder', 'clip_l14'])
    assert not fake.calls


@pytest.mark.parametrize('status', ['fail', 'parked'])
def test_train_refuses_a_store_that_did_not_pass(rec, fake, status):
    prefix_pick(rec, 'plain')
    gate(rec, 'siglip2', status=status, tries=3)
    with pytest.raises(RuntimeError, match='viii_siglip2'):
        E.main(['train', '--encoder', 'siglip2'])
    assert not fake.calls


def test_export_passes_val_and_held_and_the_same_source(rec, fake):
    prefix_pick(rec, 'plain')
    gate(rec, 'siglip2')
    E.main(['export', '--encoder', 'siglip2', '--seeds', '1'])
    (call,) = fake.calls
    assert call[:4] == ('export', 'siglip2', 'siglip2', [1]) and call[5]['kinds'] == ('val', 'held')


# ------------------------------------------------------------------------------------------------ picks, cells, ladder
class Recorder:
    def __init__(self):
        self.argv = None

    def main(self, argv):
        self.argv = list(argv)
        return 0


def test_picks_and_cells_run_ticket_06_on_the_encoders_own_arm_and_root(rec, monkeypatch):
    pass_rule(rec)
    prefix_pick(rec, 'prefixed')
    gate(rec, 'siglip2_prefix')
    r = Recorder()
    monkeypatch.setattr(E, '_image_cells', lambda: r)
    for cmd in ('picks', 'cells'):
        assert E.main([cmd, '--encoder', 'siglip2']) == 0
        assert r.argv == ['--workers', '8', cmd, '--arm', 'siglip2_prefix', '--encoder', 'siglip2', '--root',
                          str(P.OUT / 'cstar' / 'siglip2')]


def test_cells_refuse_without_the_store_gate(rec, monkeypatch):
    pass_rule(rec)
    prefix_pick(rec, 'plain')
    monkeypatch.setattr(E, '_image_cells', lambda: Recorder())
    with pytest.raises(RuntimeError, match='viii_clipl14 not run'):
        E.main(['cells', '--encoder', 'clip_l14'])


def test_ladder_is_ticket_08s_sc4_for_the_arm_and_encoder(rec, monkeypatch):
    pass_rule(rec)
    prefix_pick(rec, 'plain')
    gate(rec, 'clipl14')
    r = Recorder()
    monkeypatch.setattr(E, '_ladder', lambda: r)
    assert E.main(['ladder', '--encoder', 'clip_l14']) == 0
    assert r.argv == ['ladder', '--claim', 'SC4', '--arm', 'clipl14', '--encoder', 'clip_l14']


# ------------------------------------------------------------------------------------------------ r per encoder
def test_e2_and_e3_get_their_own_encoder_in_ticket_06s_wiring():
    """The LEVEL unit r and the plain comparator follow the wiring's encoder (never SigLIP 1's): the arm names map to
    their own encoder in both ticket 06 and ticket 08."""
    import r29_image_cells as IC
    import r29_ladder as L
    for src, enc in (('siglip2', 'siglip2'), ('siglip2_prefix', 'siglip2'), ('clipl14', 'clip_l14'), ('clipl14_prefix', 'clip_l14')):
        assert IC.encoder_of_arm(src) == L.encoder_of_arm(src) == enc
    for enc in E.ENCODERS:
        assert E.arm_encoder(E.arm_source(enc, 'plain')) == enc


# ------------------------------------------------------------------------------------------------ the current model beside
@pytest.mark.parametrize('enc', ['siglip2', 'clip_l14'])
def test_beside_shows_the_current_models_cells_of_that_encoder(rec, enc):
    rows = E.beside(enc)
    assert len(rows) == 12                                # 6 SC4 cells against plain, the same 6 against Q1
    assert {r['cell'] for r in rows} == {f'{a}/{m}/{c}' for a in ASP for m in ('ami', 'map') for c in ('plain', 'teacher')}
    for r in rows:
        cur = r['current']
        assert np.isfinite([cur['value'], cur['comparator_value'], cur['d'], cur['lo'], cur['hi']]).all(), r
        assert enc in cur['source'] and cur['outcome'] in ('WIN', 'LEVEL', 'BELOW')
        assert r['new'] is None                           # no new cell record yet


def test_beside_current_ami_plain_is_r17s_sc1_cell(rec):
    rows = {r['cell']: r for r in E.beside('clip_l14')}
    c = rows['genre/ami/plain']['current']
    assert c['source'] == 'r17 SC1/artelingo/clip_l14/genre/ami/10/plain'
    assert c['value'] == pytest.approx(0.5155, abs=5e-5) and c['outcome'] == 'WIN'
    t = rows['genre/map/teacher']['current']
    assert t['source'].startswith('r14 SC6') and t['value'] == pytest.approx(0.6177380783984918)


def test_beside_joins_the_new_cell_record_by_aspect_measure_comparator(rec):
    cells = []
    for a in ASP:
        for m, mid in (('ami', f'ami{K[a]}'), ('map', 'map')):
            for comp in ('plain', 'teacher'):
                cells.append(dict(id=f'artelingo/siglip2/siglip2/{a}/{mid}/{comp}', arm='siglip2', encoder='siglip2', aspect=a,
                                  comparator=comp, value=0.5, comparator_value=0.4, d=0.1, lo=0.05, hi=0.15, r=0.01, outcome='WIN'))
    put(rec, 'image_cells_siglip2.json', dict(arm='siglip2', encoder='siglip2', cells=cells))
    rows = E.beside('siglip2', arm='siglip2')
    assert all(r['new']['outcome'] == 'WIN' and r['new']['id'].startswith('artelingo/siglip2/siglip2/') for r in rows)
    assert {r['cell'] for r in rows if r['new']['id'].endswith('/ami10/plain')} == {'genre/ami/plain'}


def test_beside_refuses_another_encoders_cell_record(rec):
    put(rec, 'image_cells_siglip2.json', dict(arm='siglip2', encoder='clip_l14', cells=[]))
    with pytest.raises(AssertionError, match='encoder'):
        E.beside('siglip2', arm='siglip2')


# ------------------------------------------------------------------------------------------------ smoke (pure parts)
def test_smoke_join_checks_names_and_shapes():
    names = ['a', 'b', 'c']
    ok = E.join_report(names, names, head_shapes={'emotion__s0': (3, 2048)}, plain_shape=(3, 1152), cap_shape=(3, 1152),
                       plain_width=1152, cap_width=1152)
    assert ok['ok']
    with pytest.raises(AssertionError, match='names'):
        E.join_report(names, ['a', 'c', 'b'], head_shapes={}, plain_shape=(3, 1152), cap_shape=(3, 1152), plain_width=1152, cap_width=1152)
    with pytest.raises(AssertionError, match='shape'):
        E.join_report(names, names, head_shapes={'emotion__s0': (3, 1024)}, plain_shape=(3, 1152), cap_shape=(3, 1152),
                      plain_width=1152, cap_width=1152)
    with pytest.raises(AssertionError, match='shape'):
        E.join_report(names, names, head_shapes={}, plain_shape=(3, 1152), cap_shape=(3, 768), plain_width=1152, cap_width=1152)


@pytest.mark.parametrize('enc', ['siglip2', 'clip_l14'])
def test_ticket_06_range_on_r14s_exports_is_that_encoders_own_recorded_range(enc):
    """The LEVEL unit r of an E2 or E3 arm is its own five-seed range: on r14's exports of the encoder, ticket 06's validation range
    equals r14's recorded range of that encoder (and differs between the two encoders)."""
    import r29_image_cells as IC
    w = IC.Wiring(P.R14_OUT / 'cstar' / enc, P.K0_ARM, enc, splits=('val',))
    got = IC.validation_ranges(w)['ranges']
    rec = json.loads((P.R14_ROUND / f'records/stage1_{enc}.json').read_text())['artret_ranges']['ranges']
    assert all(abs(got[a] - rec[a]) < 1e-9 for a in ASP), (got, rec)
