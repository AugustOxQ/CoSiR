"""Part 2 reported margins (ticket 13): refusal without a verdict, one end-to-end run on small synthetic outputs."""
import json

import numpy as np
import pytest

import rb_paths  # noqa: F401  (sys.path)
import rb_part2_report as rpt
import rb_score as score
from test_rb_score import GROUPS, SEEDS5, put_sidecar, synthetic, tiny_world, unit

N = 60


def world(tmp_path, monkeypatch):
    """tiny_world (K2, M2, B1 to B6 under a 60-painting held set) plus K4, the CT's K1 outputs, weights, labels."""
    out = tiny_world(tmp_path, monkeypatch, N)
    ct_root = score.CT_ROOT['sample']
    for s in SEEDS5:
        for aspect in ('emotion', 'style', 'genre'):
            path = score.output_path('sample', 'K4', s, aspect, out)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, img=unit(N, 2048, 100 + s))
            put_sidecar(path, 'K4', s, aspect)
        path = score.output_path('sample', 'K1', s, 'emotion')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, img=unit(N, 2048, 200 + s))
        path.with_suffix('.json').write_text(json.dumps(dict(held_sha256='H', rows=N)))
        for root in (ct_root / 'd2/weights', out / 'part2/sample/weights/K2', out / 'part2/sample/weights/K4'):
            for aspect in ('emotion', 'style', 'genre'):
                root.mkdir(parents=True, exist_ok=True)
                (root / f'{aspect}__s{s}.json').write_text(json.dumps(dict(best_epoch=10 + s, best_dev_loss=1.5)))
    rng = np.random.default_rng(1)
    labels = dict(emotion=np.arange(N) % 5, style=rng.integers(-1, 4, N), genre=rng.integers(-1, 3, N))
    monkeypatch.setattr(rpt, 'held_labels', lambda: (['x'] * N, 'H', labels))
    monkeypatch.setattr(rpt, 'frozen_image', lambda c, names: unit(N, 16, 7).astype(np.float32))
    monkeypatch.setattr(rpt, 'target_image', lambda aspect, names, sha: unit(N, 16, 8).astype(np.float32))
    monkeypatch.setattr(rpt, 'clip_names', lambda: [f'p{i}' for i in range(7)])
    a = unit(7, 8, 3).astype(np.float32)
    monkeypatch.setattr(rpt, 'clip_pair', lambda names: (a, a * 1.0))
    score.write_regression_items(out / 'part2/regression/stage2.json', GROUPS)
    score.run_stage(3, ['K2', 'K4'], out, workers=1, n_resamples=3)
    return out, labels


def test_refuses_to_run_without_the_stage_verdict_file(tmp_path):
    with pytest.raises(AssertionError, match='verdict file'):
        rpt.run(3, tmp_path, workers=1, n_resamples=3)


def test_generic_bootstrap_equals_rb_score_on_standard_readings():
    arms = ('K2', 'B1', 'B4')
    fixed, y = synthetic(arms)
    eligible = {'K2': SEEDS5, 'B1': SEEDS5, 'B4': [0, 2]}
    ours = rpt._bootstrap(fixed, y, eligible, arms, 8, 1)
    ref = score.paired_bootstrap(fixed, y, eligible, arms, 8, 1)
    assert ours['bootstrap_d'] == ref['bootstrap_d'] and ours['d'] == ref['d']
    assert rpt.margin(fixed, y, eligible, arms, 8, 1, lenient=True)['interval'] == ref['interval']


def test_margin_arm_lists_follow_the_rule_order_and_k4s_matched_arm_is_b1():
    g = rpt.margin_arms('K2')
    assert g['a'] == ('K2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6') and g['b'] == ('K2', 'M2')
    assert g['ab'] == g['a'] + ('M2',)
    g = rpt.margin_arms('K4')
    assert g['b'] == ('K4', 'B1') and g['ab'] == g['a']


def test_end_to_end_on_small_synthetic_outputs(tmp_path, monkeypatch):
    out, labels = world(tmp_path, monkeypatch)
    verdict = json.loads((out / 'part2/verdict_stage3.json').read_text())
    record = rpt.run(3, out, workers=1, n_resamples=4)
    assert (out / 'part2/reported_stage3.md').read_text().startswith('# Part 2 reported margins, stage 3')
    saved = json.loads((out / 'part2/reported_stage3.json').read_text())
    assert saved['blocks'].keys() == {'K2', 'K4', 'K1_with_K4_entry_set', 'clip_b32_cosine'} == record['blocks'].keys()
    k2 = saved['blocks']['K2']
    assert k2['verdict']['d'] == verdict['blocks']['K2']['d']
    assert set(k2['aspects']['emotion']['margins']) == {'c_frozen', 'T', 'leiden_vs_all'}
    assert set(k2['aspects']['style']['margins']) == {'a_B1_to_B6', 'b_matched', 'max_a_b', 'c_frozen', 'T',
                                                     'leiden_vs_all'}
    m = k2['aspects']['style']['margins']
    assert m['b_matched']['arms'] == ['K2', 'M2'] and m['c_frozen']['seedless'] == ['F2']
    assert m['max_a_b']['arms'] == verdict['blocks']['K2']['arms']
    # each margin is its own default_rng(0) bootstrap through rb_score: rerun one by hand
    fixed, eligible, _ = score.prepare_arms('sample', score.arms_for('K2'), score.read_rcol_eligible('sample')[0],
                                            'H', out, aspect='style', n_rows=N)
    hand = score.paired_bootstrap(fixed, labels['style'], {a: eligible[a] for a in ('K2', 'M2')}, ('K2', 'M2'), 4, 1)
    assert m['b_matched']['d'] == hand['d'] and m['b_matched']['interval'] == hand['interval']
    # the emotion statistic is the verdict's; seed spread and best epochs are there
    spread = k2['aspects']['emotion']['point_spread']
    assert set(spread['K2']['r20']['per_seed']) == {'0', '1', '2', '3', '4'} and spread['B1']['own']['sd'] >= 0
    assert k2['best_epochs']['genre']['3'] == dict(best_epoch=13, best_dev_loss=1.5)
    k1 = saved['blocks']['K1_with_K4_entry_set']
    assert k1['arms'] == ['K1', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'] and len(k1['interval']) == 2
    assert saved['blocks']['K4']['aspects']['style']['margins']['b_matched']['arms'] == ['K4', 'B1']
    cos = saved['blocks']['clip_b32_cosine']
    assert cos['n'] == 7 and abs(cos['mean'] - 1) < 1e-3
    with np.load(out / 'part2/clip_b32_cosine.npz') as z:
        assert z['names'].tolist() == [f'p{i}' for i in range(7)] and z['cosine'].shape == (7,)
    assert 'S1-' not in json.dumps({k: v for k, v in saved['blocks']['K2'].items() if k != 'verdict'})
    again = rpt.run(3, out, workers=1, n_resamples=4)           # identical rerun leaves the file untouched
    assert again['blocks'] == saved['blocks']
    cli_out = tmp_path / 'not_tmp'
    with pytest.raises(AssertionError, match='/tmp/claude-0'):
        rpt.main(['--stage', '3', '--smoke', '--outputs', str(cli_out)])
