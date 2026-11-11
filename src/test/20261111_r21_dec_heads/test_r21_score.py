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
    # 0.0200 - 0.0180 computes (mean of three equal changes, REF added and removed) to a float just above 0.002: it is
    # inside only because of the 1e-12 epsilon, so strict `<` or `<=` without it fails this case.
    a, b = row(A, 0.0200, 0.30), row(B, 0.0180, 0.31)
    gap = r21_score.select([a, b], REF)['candidates']
    gap = gap[0]['mean_d_ami'] - gap[1]['mean_d_ami']
    assert 0.002 < gap < 0.002 + 1e-12, gap                                                  # the case is on the edge
    assert r21_score.select([a, b], REF)['pick'] == B
    assert r21_score.select([row(A, 0.0200, 0.30), row(B, 0.0180 - 1e-9, 0.31)], REF)['pick'] == A   # 1e-9 outside


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
    cfg = dict(name='R6-5', rg=dict(arm='dec-off'), mode='full', smoke=False, aspect='emotion', seed=0, graph_row='Gres', output_rows=len(names),
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
    lambda s: s['config'].update(graph_row='R0o'), lambda s: s['config']['rg'].update(arm='k11=40'),
    lambda s: s['config'].pop('rg'), lambda s: s['config']['names_sha256'].update(dev='0' * 64),
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


def fake_history(run, n_dec, reason='changed'):
    """History in ticket 02's format: `run` entries; with DEC on (n_dec > 0) the first run - n_dec are warm-up entries (the last
    of them also carries `start_epoch`), the rest DEC entries carrying `start_epoch`, the last one `stop_reason`. DEC off
    (n_dec == 0): no `dec` key at all."""
    if n_dec == 0:
        return [dict(epoch=i + 1) for i in range(run)]
    w, hist = run - n_dec, []
    for i in range(run):
        if i < w:
            dec = dict(phase='warmup', **(dict(start_epoch=w + 1, share_max=0.2) if i == w - 1 else {}))
        else:
            dec = dict(phase='dec', start_epoch=w + 1, kl=0.1, changed=0.1, **(dict(stop_reason=reason) if i == run - 1 else {}))
        hist.append(dict(epoch=i + 1, dec=dec))
    return hist


def fake_sides(best=(4, 1, 18), run=(10, 16, 33), dec=(0, 0, 0), reason=('changed', 'patience', 'dec_epochs')):
    return {a: dict(best_epoch=b, epochs_run=r, history=fake_history(r, n, why))
            for a, b, r, n, why in zip(ASP, best, run, dec, reason)}


PLAIN = 'dec-plain_l1_w1_S1'


def test_make_row_changes_dec_epochs_and_reason():
    rec = dict(heads={'emotion': dict(stop_reason='changed', dec_start_epoch=8), 'style': dict(stop_reason='patience'),
                      'genre': dict(stop_reason='dec_epochs')})
    r = r21_score.make_row(PLAIN, fake_val('x', 0.01, 0.31), fake_sides(dec=(3, 5, 4)), rec, fake_val('c'))
    assert r['arm'] == 'dec-plain_l1_w1_S1' and r['settings']['form'] == 'plain'
    assert r['d_ami']['style'] == pytest.approx(0.01) and r['mean_d_ami'] == pytest.approx(0.01)
    assert r['d_map']['genre'] == pytest.approx(0.01)
    # warm-up entries are not DEC epochs: 10 entries of which 3 DEC, 16 of which 5, 33 of which 4
    assert r['dec_epochs'] == dict(emotion=3, style=5, genre=4) and r['epochs_run']['style'] == 16
    assert r['dec_start_epoch'] == dict(emotion=8, style=12, genre=30)
    assert r['stop_reason'] == dict(emotion='changed', style='patience', genre='dec_epochs') and r['best_epoch']['genre'] == 18


def test_make_row_reads_the_reason_and_start_from_the_sidecar_without_a_run_record():
    r = r21_score.make_row(PLAIN, fake_val('x'), fake_sides(dec=(3, 5, 4)), None, fake_val('c'))
    assert r['stop_reason'] == dict(emotion='changed', style='patience', genre='dec_epochs')
    assert r['dec_start_epoch'] == dict(emotion=8, style=12, genre=30)


def test_make_row_dec_off_and_cstar_have_no_dec_epochs_and_no_reason_without_a_record():
    for arm in ('dec-off', 'cstar'):
        r = r21_score.make_row(arm, fake_val('x'), fake_sides(), None, fake_val('c'))
        assert r['stop_reason'] == dict(emotion=None, style=None, genre=None) and r['dec_epochs'] == dict(emotion=0, style=0, genre=0)
    rec = dict(heads={a: dict(stop_reason='patience', dec_start_epoch=None) for a in ASP})       # DEC off: c*'s reason, from the record
    assert r21_score.make_row('dec-off', fake_val('x'), fake_sides(), rec, fake_val('c'))['stop_reason']['genre'] == 'patience'


def test_make_row_refuses_dec_entries_in_a_dec_off_history():
    with pytest.raises(AssertionError):
        r21_score.make_row('dec-off', fake_val('x'), fake_sides(dec=(3, 0, 0)), None, fake_val('c'))


def test_make_row_refuses_a_dec_arm_without_a_stop_reason():
    sides = fake_sides(dec=(3, 5, 4))
    del sides['style']['history'][-1]['dec']['stop_reason']
    with pytest.raises(AssertionError):
        r21_score.make_row(PLAIN, fake_val('x'), sides, None, fake_val('c'))
    sides = fake_sides(dec=(3, 5, 4), reason=('changed', 'cap', 'dec_epochs'))        # not one of the trainer's four
    with pytest.raises(AssertionError):
        r21_score.make_row(PLAIN, fake_val('x'), sides, None, fake_val('c'))


def test_make_row_refuses_a_head_that_never_reached_dec():
    sides = fake_sides(dec=(3, 5, 4))
    sides['emotion']['history'] = fake_history(10, 3)[:7]          # stopped in warm-up: no DEC entry
    sides['emotion']['epochs_run'] = 7
    with pytest.raises(AssertionError):
        r21_score.make_row(PLAIN, fake_val('x'), sides, None, fake_val('c'))


def test_make_row_asserts_the_run_record_agrees_with_the_sidecar():
    ok = {a: dict(stop_reason=w, dec_start_epoch=s) for a, w, s in zip(ASP, ('changed', 'patience', 'dec_epochs'), (8, 12, 30))}
    for edit in (dict(emotion=dict(stop_reason='patience')), dict(style=dict(stop_reason='patience', dec_start_epoch=11))):
        rec = dict(heads=dict(ok, **edit))
        with pytest.raises(AssertionError):
            r21_score.make_row(PLAIN, fake_val('x'), fake_sides(dec=(3, 5, 4)), rec, fake_val('c'))
    r21_score.make_row(PLAIN, fake_val('x'), fake_sides(dec=(3, 5, 4)), dict(heads=ok), fake_val('c'))


def row_off(d=0.0, best=(4, 1, 18)):
    return r21_score.make_row('dec-off', fake_val('x', d), fake_sides(best=best), None, fake_val('c'))


def test_zero_c_line_pass_and_boundaries():
    cstar = r21_score.make_row('cstar', fake_val('c'), fake_sides(), None, fake_val('c'))
    assert r21_score.zero_c_line(row_off(0.004 - 1e-9, (6, 1, 16)), cstar)['status'] == 'PASS'       # 2 epochs, < 0.004
    assert r21_score.zero_c_line(row_off(0.0, (7, 1, 18)), cstar)['status'] == 'FAIL'                  # 3 epochs on emotion
    assert r21_score.zero_c_line(row_off(0.0, (4, 1, 15)), cstar)['status'] == 'FAIL'                  # 3 epochs on genre, below
    assert r21_score.zero_c_line(row_off(0.0, (2, 3, 20)), cstar)['status'] == 'PASS'                  # 2 epochs either way
    assert r21_score.zero_c_line(row_off(0.0041), cstar)['status'] == 'FAIL'                           # AMI outside 0.004
    assert r21_score.zero_c_line(row_off(-0.0041), cstar)['status'] == 'FAIL'


def edge_diff(d):
    """The AMI difference the 0c line computes for a dec-off row `d` above saved c* (mean-free: each aspect on its own)."""
    cstar = r21_score.make_row('cstar', fake_val('c'), fake_sides(), None, fake_val('c'))
    return r21_score.zero_c_line(row_off(d), cstar)['ami_diff']


def test_zero_c_ami_edge_is_inclusive_and_float_safe():
    cstar = r21_score.make_row('cstar', fake_val('c'), fake_sides(), None, fake_val('c'))
    # Exactly 0.004 on the nose and on the float side that computes just above 0.004 (inside only through the epsilon).
    for d in (0.004, -0.004):
        assert r21_score.zero_c_line(row_off(d), cstar)['status'] == 'PASS', d
    over = [d for d in (0.004 + k * 2 ** -60 for k in range(0, 200000, 1000)) if max(abs(x) for x in edge_diff(d).values()) > 0.004]
    assert over, 'need a float case whose computed difference lands just above 0.004'
    d = over[0]
    assert 0.004 < max(abs(x) for x in edge_diff(d).values()) < 0.004 + 1e-12
    assert r21_score.zero_c_line(row_off(d), cstar)['status'] == 'PASS'                                  # epsilon makes it inside
    assert r21_score.zero_c_line(row_off(0.004 + 1e-9), cstar)['status'] == 'FAIL'
    assert r21_score.zero_c_line(row_off(-0.004 - 1e-9), cstar)['status'] == 'FAIL'


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


# ------------------------------------------------------------------------------------------------ the validation-score cache
def test_val_cached_recomputes_on_a_code_hash_mismatch_and_reuses_when_equal(tmp_path, monkeypatch):
    calls = []

    def fake_scores(arm, seed, aspects, train_root, workers):
        calls.append(arm)
        return dict(arm=arm, seed=seed, aspects={a: dict(input_sha256='h') for a in ASP})

    monkeypatch.setattr(r21_score, 'val_scores', fake_scores)
    monkeypatch.setattr(r21_score, 'load_val', lambda arm, a, seed, train_root=None: (None, dict(output_sha256='h')))
    monkeypatch.setattr(r21_score, 'code_hashes', lambda: dict(r21_score='a', ns_metrics='b'))
    r21_score.val_cached('dec-off', 0, tmp_path)
    assert calls == ['dec-off']
    r21_score.val_cached('dec-off', 0, tmp_path)                       # same inputs, same code: the cache is read
    assert calls == ['dec-off']
    monkeypatch.setattr(r21_score, 'code_hashes', lambda: dict(r21_score='a', ns_metrics='c'))
    r21_score.val_cached('dec-off', 0, tmp_path)                       # one scoring file changed: recomputed
    assert calls == ['dec-off', 'dec-off']
    assert json.loads((tmp_path / 'val' / 'dec-off__s0.json').read_text())['code_sha256'] == dict(r21_score='a', ns_metrics='c')
    r21_score.val_cached('dec-off', 0, tmp_path)
    assert len(calls) == 2
    old = json.loads((tmp_path / 'val' / 'dec-off__s0.json').read_text())          # a cache file from before this field
    del old['code_sha256']
    (tmp_path / 'val' / 'dec-off__s0.json').write_text(json.dumps(old))
    r21_score.val_cached('dec-off', 0, tmp_path)
    assert len(calls) == 3


def test_code_hashes_cover_every_scoring_file():
    got = r21_score.code_hashes()
    assert set(got) == {'r21_score.py', 'ns_metrics.py', 'ns_data.py', 'rt_data.py', 'rt_metrics.py', 'r20_ret.py'}
    assert all(len(v) == 64 for v in got.values())


# ------------------------------------------------------------------------------------------------ collect_rows
def stage_tree(tmp_path, monkeypatch, record_edit=None, ranges_ok=True):
    """A tmp train tree with dec-off and one DEC arm at seed 0, and r21_score's readers replaced by fakes."""
    root = tmp_path / 'train'
    arms = {'dec-off': (fake_sides(), None), PLAIN: (fake_sides(dec=(3, 5, 4)), None)}
    for arm, (sides, _) in arms.items():
        for a in ASP:
            p = r21_score.val_export_path(arm, a, 0, train_root=root)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'')
        settings = r21_arms.parse_arm(arm)
        heads = {a: dict(stop_reason=None if arm == 'dec-off' else w, dec_start_epoch=None if arm == 'dec-off' else s)
                 for a, w, s in zip(ASP, ('changed', 'patience', 'dec_epochs'), (8, 12, 30))}
        rec = dict(arm=arm, settings=settings, seed=0, status='ok', heads=heads)
        if record_edit and arm == PLAIN:
            record_edit(rec)
        (root / arm / 'run__s0.json').write_text(json.dumps(rec))
    cs = fake_sides()

    def fake_load(arm, a, seed, train_root=None):
        return None, (cs if arm == 'cstar' else arms[arm][0])[a]

    monkeypatch.setattr(r21_score, 'val_cached', lambda arm, seed, *a, **k: fake_val(arm))
    monkeypatch.setattr(r21_score, 'load_val', fake_load)
    monkeypatch.setattr(r21_score, 'check_cstar_ranges', lambda ref: {a: dict(ok=ranges_ok) for a in ASP})
    return root


def test_collect_rows_reads_every_arm(tmp_path, monkeypatch):
    root = stage_tree(tmp_path, monkeypatch)
    cstar, rows, missing = r21_score.collect_rows(0, train_root=root)
    assert set(rows) == {'dec-off', PLAIN} and not missing and rows[PLAIN]['dec_epochs']['style'] == 5


def test_collect_rows_checks_the_cstar_reference_against_the_ranges(tmp_path, monkeypatch):
    root = stage_tree(tmp_path, monkeypatch, ranges_ok=False)
    with pytest.raises(AssertionError):
        r21_score.collect_rows(0, train_root=root)


@pytest.mark.parametrize('edit', [lambda r: r.update(seed=1), lambda r: r.update(arm='dec-off'),
                                  lambda r: r['settings'].update(lam=10.0), lambda r: r['settings'].update(balance=True)])
def test_collect_rows_refuses_a_run_record_of_another_arm_or_seed(tmp_path, monkeypatch, edit):
    root = stage_tree(tmp_path, monkeypatch, record_edit=edit)
    with pytest.raises(AssertionError):
        r21_score.collect_rows(0, train_root=root)


def test_collect_rows_refuses_a_record_whose_reason_differs_from_the_sidecar(tmp_path, monkeypatch):
    root = stage_tree(tmp_path, monkeypatch, record_edit=lambda r: r['heads']['style'].update(stop_reason='changed'))
    with pytest.raises(AssertionError):
        r21_score.collect_rows(0, train_root=root)


# ------------------------------------------------------------------------------------------------ picks: write-once, gate 0c
def sel(arm, band=None):
    return dict(pick=arm, band=band or [arm], beyond_spec=False, candidates=[])


def test_merge_picks_first_write_then_same_pick_kept_then_conflict():
    new = dict(stage1a=sel(A), stage1b=None)
    merged, conflicts, _ = r21_score.merge_picks(None, new, force=False)
    assert merged['stage1a']['pick'] == A and not conflicts
    again, conflicts, _ = r21_score.merge_picks(merged, dict(stage1a=sel(A, [A, B]), stage1b=None), force=False)
    assert not conflicts and again['stage1a']['band'] == [A]                       # the frozen record is kept
    _, conflicts, _ = r21_score.merge_picks(merged, dict(stage1a=sel(B), stage1b=None), force=False)
    assert conflicts == [dict(level='stage1a', frozen=A, new=B)]


def test_merge_picks_every_level_is_write_once_on_its_own():
    old = dict(stage1a=sel(A), stage1b=dict(plain=sel(A)))
    new = dict(stage1a=sel(A), stage1b=dict(plain=sel(A), start=sel('dec-start_l1_w1_S1'), teacher=sel('dec-teacher_l1_w1_S1')))
    merged, conflicts, _ = r21_score.merge_picks(old, new, force=False)
    assert not conflicts and set(merged['stage1b']) == {'plain', 'start', 'teacher'}      # new levels are added
    new['stage1b']['start'] = sel('dec-start_l1_w1_S1_bal')
    merged2, conflicts, _ = r21_score.merge_picks(merged, new, force=False)
    assert [c['level'] for c in conflicts] == ['stage1b.start']


def test_merge_picks_force_overrides_and_logs():
    old = dict(stage1a=sel(A), stage1b=None)
    merged, conflicts, overrides = r21_score.merge_picks(old, dict(stage1a=sel(B), stage1b=None), force=True)
    assert merged['stage1a']['pick'] == B and conflicts and merged['overrides'] == overrides
    assert overrides[0]['level'] == 'stage1a' and overrides[0]['frozen_pick'] == A and overrides[0]['new_pick'] == B
    merged, _, _ = r21_score.merge_picks(merged, dict(stage1a=sel(A), stage1b=None), force=True)
    assert len(merged['overrides']) == 2                                                  # the log accumulates


def all_rows(zero_c_ok=True, with_off=True):
    """Seed 0 sheet rows by `make_row`: the eight 1a arms (the last, l10_w5_S2, has the best AMI) and dec-off."""
    cs = r21_score.make_row('cstar', fake_val('c'), fake_sides(), None, fake_val('c'))
    rows = {n: r21_score.make_row(n, fake_val(n, 0.01 * i), fake_sides(dec=(3, 5, 4)), None, fake_val('c'))
            for i, n in enumerate(r21_score.grid_1a())}
    if with_off:
        rows['dec-off'] = row_off(0.0 if zero_c_ok else 0.02)
    return cs, rows


def test_build_sheet_has_no_picks_unless_gate_0c_passes():
    for ok, off, status in ((True, True, 'PASS'), (False, True, 'FAIL'), (True, False, 'pending')):
        cs, rows = all_rows(ok, off)
        doc = r21_score.build_sheet(cs, rows, [])
        assert doc['zero_c']['status'] == status
        assert (doc['picks']['stage1a'] is not None) == (status == 'PASS'), status


def run_sheet(monkeypatch, tmp_path, rows_fn, force=False):
    monkeypatch.setattr(r21_score, 'SCORES', tmp_path)
    cs, rows = rows_fn()
    monkeypatch.setattr(r21_score, 'collect_rows', lambda *a, **k: (cs, rows, []))
    args = type('Args', (), dict(workers=1, force=force))()
    r21_score.cmd_sheet(args)


def test_cmd_sheet_writes_picks_when_0c_passes_and_freezes_them(monkeypatch, tmp_path):
    run_sheet(monkeypatch, tmp_path, lambda: all_rows(True))
    picks = json.loads((tmp_path / 'stage1_picks.json').read_text())
    best = r21_score.grid_1a()[-1]
    assert picks['stage1a']['pick'] == best
    # a changed number for the same runs gives another pick: refused, the file is not touched
    def changed():
        cs, rows = all_rows(True)
        rows[best] = r21_score.make_row(best, fake_val(best, -0.5), fake_sides(dec=(3, 5, 4)), None, fake_val('c'))
        return cs, rows
    before = (tmp_path / 'stage1_picks.json').read_text()
    with pytest.raises(SystemExit) as e:
        run_sheet(monkeypatch, tmp_path, changed)
    assert e.value.code not in (0, None)
    assert (tmp_path / 'stage1_picks.json').read_text() == before
    run_sheet(monkeypatch, tmp_path, changed, force=True)
    after = json.loads((tmp_path / 'stage1_picks.json').read_text())
    assert after['stage1a']['pick'] != best and after['overrides'][0]['frozen_pick'] == best


@pytest.mark.parametrize('ok, off, status', [(False, True, 'FAIL'), (True, False, 'pending')])
def test_cmd_sheet_writes_no_picks_and_exits_non_zero_when_0c_is_not_pass(monkeypatch, tmp_path, ok, off, status, capsys):
    with pytest.raises(SystemExit) as e:
        run_sheet(monkeypatch, tmp_path, lambda: all_rows(ok, off))
    assert e.value.code not in (0, None)
    assert not (tmp_path / 'stage1_picks.json').exists()                                 # no picks file at all
    doc = json.loads((tmp_path / 'stage1_sheet.json').read_text())                       # the sheet is written
    assert doc['zero_c']['status'] == status and doc['picks'] == dict(stage1a=None, stage1b=None)
    assert (tmp_path / 'stage1_sheet.md').exists() and status in capsys.readouterr().out


def test_cmd_sheet_0c_fail_leaves_an_existing_picks_file_alone(monkeypatch, tmp_path):
    run_sheet(monkeypatch, tmp_path, lambda: all_rows(True))
    before = (tmp_path / 'stage1_picks.json').read_text()
    with pytest.raises(SystemExit):
        run_sheet(monkeypatch, tmp_path, lambda: all_rows(False))
    assert (tmp_path / 'stage1_picks.json').read_text() == before
