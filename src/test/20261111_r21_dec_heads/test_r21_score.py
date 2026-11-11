"""Tests of r21_arms and r21_score: arm names, the stage 1 selection rule, the dev-path guard, the validation loader."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r21_arms  # noqa: E402


# ------------------------------------------------------------------------------------------------ arm names
@pytest.mark.parametrize('name, settings', [
    ('dec-off', dict(form='off', lam=None, warmup=None, stop=None, balance=None)),
    ('dec-plain_l1_w1_S1', dict(form='plain', lam=1.0, warmup=1, stop='S1', balance=False)),
    ('dec-teacher_l10_w5_S2_bal', dict(form='teacher', lam=10.0, warmup=5, stop='S2', balance=True)),
    ('dec-start_l0.5_w5_S1_bal', dict(form='start', lam=0.5, warmup=5, stop='S1', balance=True)),
])
def test_parse_arm_and_back(name, settings):
    assert r21_arms.parse_arm(name) == settings
    assert r21_arms.arm_name(settings) == name


@pytest.mark.parametrize('bad', ['dec', 'dec-plain', 'dec-plain_l1_w1', 'dec-plain_l01_w1_S1', 'dec-foo_l1_w1_S1',
                                 'dec-plain_l1_w1_S3', 'dec-plain_l1_w1_S1_bal_x', 'k11=40', 'dec-off_bal',
                                 'dec-plain_l1_w01_S1'])
def test_parse_arm_refuses(bad):
    with pytest.raises(AssertionError):
        r21_arms.parse_arm(bad)


def test_arm_name_refuses_off_with_values():
    with pytest.raises(AssertionError):
        r21_arms.arm_name(dict(form='off', lam=1.0, warmup=None, stop=None, balance=None))


# ------------------------------------------------------------------------------------------------ selection rule
import r21_score  # noqa: E402

ASP = ('emotion', 'style', 'genre')
REF = dict(emotion=0.10, style=0.40, genre=0.50)      # c* seed 0 validation AMI (stand-in values)


def row(arm, d_ami, mAP=0.3):
    """A sheet row whose three aspects each change by d_ami (a number, or one per aspect) from REF; mAP likewise."""
    d = d_ami if isinstance(d_ami, (list, tuple)) else [d_ami] * 3
    m = mAP if isinstance(mAP, (list, tuple)) else [mAP] * 3
    return dict(arm=arm, ami={a: REF[a] + x for a, x in zip(ASP, d)}, map=dict(zip(ASP, m)))


A = 'dec-plain_l1_w1_S1'
B = 'dec-plain_l10_w1_S1'
C = 'dec-plain_l1_w5_S1'


def test_selection_highest_mean_ami_change_wins_outside_the_band():
    got = r21_score.select([row(A, 0.010, 0.2), row(B, 0.020, 0.1)], REF)
    assert got['pick'] == B and not got['beyond_spec']
    assert got['candidates'][0]['mean_d_ami'] == pytest.approx(0.020)


def test_selection_inside_the_band_higher_map_wins_even_with_lower_ami():
    got = r21_score.select([row(A, 0.0100, 0.30), row(B, 0.0090, 0.31)], REF)
    assert got['pick'] == B


def test_selection_band_edge_included_and_just_outside_excluded():
    assert r21_score.select([row(A, 0.010, 0.30), row(B, 0.008, 0.31)], REF)['pick'] == B     # exactly 0.002 below
    assert r21_score.select([row(A, 0.010, 0.30), row(B, 0.0079, 0.31)], REF)['pick'] == A    # 0.0021 below


def test_selection_band_is_measured_from_the_best_not_chained():
    rows = [row(A, 0.0100, 0.30), row(B, 0.0085, 0.20), row(C, 0.0070, 0.90)]      # C is 0.003 below A
    assert r21_score.select(rows, REF)['pick'] == A


def test_selection_uses_the_mean_over_the_three_aspects():
    big_on_one = row(A, [0.03, -0.02, -0.02], 0.3)         # mean change -0.0033: one big gain does not carry it
    small_on_all = row(B, [0.0, 0.0, 0.01], 0.3)           # mean change +0.0033
    assert r21_score.select([big_on_one, small_on_all], REF)['pick'] == B


def test_selection_equal_ami_and_map_goes_to_smaller_lambda():
    rows = [row(B, 0.01, 0.3), row(A, 0.01, 0.3)]
    got = r21_score.select(rows, REF)
    assert got['pick'] == A and not got['beyond_spec']


def test_selection_ties_beyond_the_rule_are_flagged_and_go_to_smaller_warmup():
    rows = [row('dec-plain_l1_w5_S1', 0.01, 0.3), row('dec-plain_l1_w1_S2', 0.01, 0.3)]
    got = r21_score.select(rows, REF)
    assert got['pick'] == 'dec-plain_l1_w1_S2' and got['beyond_spec']


def test_selection_refuses_a_nan_row_and_a_missing_aspect():
    with pytest.raises(AssertionError):
        r21_score.select([row(A, float('nan'))], REF)
    bad = row(A, 0.01)
    del bad['map']['genre']
    with pytest.raises(AssertionError):
        r21_score.select([bad], REF)


def test_grids():
    g = r21_score.grid_1a()
    assert len(g) == len(set(g)) == 8 and all(r21_arms.parse_arm(n)['form'] == 'plain' for n in g)
    assert {r21_arms.parse_arm(n)['lam'] for n in g} == {1.0, 10.0} and 'dec-plain_l10_w5_S2' in g
    g1b = r21_score.grid_1b(r21_arms.parse_arm('dec-plain_l10_w5_S2'))
    assert len(g1b) == 6 and g1b[0] == 'dec-plain_l10_w5_S2'
    assert 'dec-teacher_l10_w5_S2_bal' in g1b and 'dec-start_l10_w5_S2' in g1b


def test_stage_picks_1a_then_per_form_after_1b():
    rows = {n: row(n, 0.01 * i, 0.3) for i, n in enumerate(r21_score.grid_1a())}      # last one (l10_w5_S2) is best
    best = r21_score.grid_1a()[-1]
    p1a = r21_score.pick_stage1a(rows, REF)
    assert p1a['pick'] == best
    for i, n in enumerate(r21_score.grid_1b(r21_arms.parse_arm(best))):
        rows.setdefault(n, row(n, 0.0, 0.3))
    rows[best + '_bal'] = row(best + '_bal', 0.15, 0.3)                                 # plain + balance beats plain
    rows['dec-start_l10_w5_S2'] = row('dec-start_l10_w5_S2', 0.02, 0.3)
    p1b = r21_score.pick_stage1b(rows, REF, p1a['pick'])
    assert p1b['plain']['pick'] == best + '_bal'
    assert p1b['start']['pick'] == 'dec-start_l10_w5_S2'
    assert set(p1b) == {'plain', 'start', 'teacher'}


def test_stage1a_needs_all_eight_runs():
    with pytest.raises(AssertionError):
        r21_score.pick_stage1a({A: row(A, 0.01)}, REF)


# ------------------------------------------------------------------------------------------------ validation only
import json  # noqa: E402

import numpy as np  # noqa: E402


@pytest.mark.parametrize('bad', [
    '/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held/k11=40/emotion__s0.npz',
    '/data/SSD2/cosir_r21/20261111_dec_heads/full/held/dec-off/emotion__s0.npz',
    '/data/SSD2/cosir_r21/20261111_dec_heads/full/train/dec-off/run__s0.json',
    '/x/train/dec-off/held/R6-5/emotion__Gres__s0.npz',
    '/x/train/dec-off/R6-5/emotion__Gres__s0.txt',
])
def test_dev_path_guard_refuses_test_and_other_paths(bad):
    with pytest.raises(AssertionError):
        r21_score.check_dev_path(Path(bad))


def test_dev_path_guard_accepts_validation_exports():
    r21_score.check_dev_path(Path('/data/SSD2/cosir_r21/20261111_dec_heads/full/train/dec-off/R6-5/emotion__Gres__s0.npz'))
    r21_score.check_dev_path(Path(r21_score.val_export_path('cstar', 'genre', 4)))
    r21_score.check_dev_path(Path(r21_score.val_export_path('dec-plain_l1_w1_S1', 'style', 0)))


@pytest.fixture(scope='module')
def fake_export(tmp_path_factory):
    """A synthetic validation export and its sidecar for dec-off, emotion, seed 0, in a temporary r21 tree."""
    r21_score._deps()
    root = tmp_path_factory.mktemp('r21') / 'train'
    names = r21_score.rt_data.names('val')
    x = np.random.default_rng(0).normal(size=(len(names), 2048)).astype(np.float32)
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
    path = r21_score.val_export_path('dec-off', 'emotion', 0, train_root=root)
    path.parent.mkdir(parents=True)
    np.savez(path, img=x, cap=x)
    cfg = dict(mode='full', smoke=False, aspect='emotion', seed=0, graph_row='Gres', output_rows=len(names),
               names_sha256=dict(dev=r21_score.rt_data.rs_vh.names_sha(names)))
    side = dict(config=cfg, best_epoch=4, epochs_run=19, history=[dict(epoch=1)],
                output_sha256=r21_score.rt_data.rs_vh.file_sha(path))
    path.with_suffix('.json').write_text(json.dumps(side))
    return root, path, x


def test_load_val_returns_the_checked_rows(fake_export):
    root, path, x = fake_export
    got, side = r21_score.load_val('dec-off', 'emotion', 0, train_root=root)
    assert got.dtype == np.float16 and got.shape == (6152, 2048) and np.array_equal(got, x)
    assert side['best_epoch'] == 4


import contextlib  # noqa: E402


@contextlib.contextmanager
def changed_sidecar(path, edit):
    """The sidecar with `edit(dict)` applied, restored afterwards."""
    old = path.with_suffix('.json').read_text()
    side = json.loads(old)
    edit(side)
    path.with_suffix('.json').write_text(json.dumps(side))
    try:
        yield
    finally:
        path.with_suffix('.json').write_text(old)


@pytest.mark.parametrize('edit', [
    lambda s: s['config'].update(seed=1), lambda s: s['config'].update(aspect='style'),
    lambda s: s['config'].update(output_rows=6151), lambda s: s['config'].update(smoke=True),
    lambda s: s['config'].update(graph_row='R0o'), lambda s: s['config']['names_sha256'].update(dev='0' * 64),
    lambda s: s.update(output_sha256='1' * 64)])
def test_load_val_refuses_a_sidecar_that_does_not_match(fake_export, edit):
    root, path, x = fake_export
    with changed_sidecar(path, edit):
        with pytest.raises(AssertionError):
            r21_score.load_val('dec-off', 'emotion', 0, train_root=root)
    r21_score.load_val('dec-off', 'emotion', 0, train_root=root)        # restored: loads again


def test_load_val_refuses_rows_that_are_not_unit(fake_export):
    root, path, x = fake_export
    side = json.loads(path.with_suffix('.json').read_text())
    saved = path.read_bytes(), json.dumps(side)
    np.savez(path, img=(x.astype(np.float32) * 1.5).astype(np.float16), cap=x)
    side['output_sha256'] = r21_score.rt_data.rs_vh.file_sha(path)
    path.with_suffix('.json').write_text(json.dumps(side))
    try:
        with pytest.raises(AssertionError):
            r21_score.load_val('dec-off', 'emotion', 0, train_root=root)
    finally:
        path.write_bytes(saved[0])
        path.with_suffix('.json').write_text(saved[1])


# ------------------------------------------------------------------------------------------------ cluster shares
def test_largest_share_per_state_and_mean():
    parts = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 2, 3]])       # largest clusters: 3/4, 3/4, 1/4
    got = r21_score.largest_share(parts)
    assert got == dict(mean=pytest.approx(0.5833333333), max=0.75)


# ------------------------------------------------------------------------------------------------ sheet rows, 0c line
def fake_val(arm, d=0.0, mAP=0.3, share=0.2):
    return dict(arm=arm, seed=0, aspects={a: dict(ami=REF[a] + d, map=mAP, largest_share=dict(mean=share, max=share),
                                                  k=9, n_queries=6152, input_sha256='x') for a in ASP})


def fake_sides(best=(4, 1, 18), run=(10, 16, 33), dec=(0, 0, 0)):
    return {a: dict(best_epoch=b, epochs_run=r, history=[dict(epoch=i + 1, **({'dec': {'changed': 0.1}} if i >= r - n else {}))
                                                         for i in range(r)])
            for a, b, r, n in zip(ASP, best, run, dec)}


def test_make_row_changes_dec_epochs_and_reason():
    rec = dict(heads={'emotion': dict(stop_reason='changed<0.001'), 'style': dict(stop_reason='patience'),
                      'genre': dict(stop_reason='cap')})
    r = r21_score.make_row('dec-plain_l1_w1_S1', fake_val('x', 0.01, 0.31), fake_sides(dec=(3, 5, 0)), rec, fake_val('c'))
    assert r['arm'] == 'dec-plain_l1_w1_S1' and r['settings']['form'] == 'plain'
    assert r['d_ami']['style'] == pytest.approx(0.01) and r['mean_d_ami'] == pytest.approx(0.01)
    assert r['d_map']['genre'] == pytest.approx(0.01)
    assert r['dec_epochs'] == dict(emotion=3, style=5, genre=0) and r['epochs_run']['style'] == 16
    assert r['stop_reason']['style'] == 'patience' and r['best_epoch']['genre'] == 18


def test_make_row_without_a_run_record_reason_is_not_available():
    r = r21_score.make_row('dec-off', fake_val('x'), fake_sides(), None, fake_val('c'))
    assert r['stop_reason'] == dict(emotion=None, style=None, genre=None) and r['dec_epochs'] == dict(emotion=0, style=0, genre=0)


def row_off(d=0.0, best=(4, 1, 18)):
    return r21_score.make_row('dec-off', fake_val('x', d), fake_sides(best=best), None, fake_val('c'))


def test_zero_c_line_pass_and_boundaries():
    cstar = r21_score.make_row('cstar', fake_val('c'), fake_sides(), None, fake_val('c'))
    assert r21_score.zero_c_line(row_off(0.004 - 1e-9, (6, 1, 16)), cstar)['status'] == 'PASS'       # 2 epochs, < 0.004
    assert r21_score.zero_c_line(row_off(0.0, (7, 1, 18)), cstar)['status'] == 'FAIL'                  # 3 epochs on emotion
    assert r21_score.zero_c_line(row_off(0.0041), cstar)['status'] == 'FAIL'                           # AMI outside 0.004
    assert r21_score.zero_c_line(row_off(-0.0041), cstar)['status'] == 'FAIL'


def test_zero_c_line_pending_without_the_rerun():
    cstar = r21_score.make_row('cstar', fake_val('c'), fake_sides(), None, fake_val('c'))
    assert r21_score.zero_c_line(None, cstar)['status'] == 'pending'


def test_sheet_markdown_has_every_run_the_0c_line_and_the_picks():
    cstar = r21_score.make_row('cstar', fake_val('c'), fake_sides(), None, fake_val('c'))
    off = row_off(0.001)
    run = r21_score.make_row('dec-plain_l1_w1_S1', fake_val('x', 0.02, 0.31, 0.9), fake_sides(dec=(2, 2, 2)), None, fake_val('c'))
    doc = dict(seed=0, cstar=cstar, rows=[off, run], zero_c=r21_score.zero_c_line(off, cstar), missing=['dec-start_l1_w1_S1'],
               picks=dict(stage1a=dict(pick='dec-plain_l1_w1_S1', band=['dec-plain_l1_w1_S1'], beyond_spec=False, candidates=[]),
                          stage1b=None))
    md = r21_score.sheet_markdown(doc)
    assert 'dec-off' in md and 'dec-plain_l1_w1_S1' in md and 'Gate 0c: PASS' in md and 'dec-start_l1_w1_S1' in md
    assert '0.90' in md or '0.900' in md          # the largest cluster share is shown
    assert 'stage 1a pick: dec-plain_l1_w1_S1' in md
