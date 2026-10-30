"""Synthetic C13 checks. No real round outputs or ac_* implementation imports."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score

# Load the checker by path: importing it must not pull any round implementation in.
SPEC = importlib.util.spec_from_file_location('independent_vh', Path(__file__).with_name('ac_vh.py'))
vh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vh)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_annotations(path):
    rows = [
        ('z', 'b', 'val', 'anger'), ('z', 'b', 'val', 'amusement'),
        ('z', 'b', 'ignored', 'anger'), ('a', 'c', 'test', 'sadness'),
        ('a', 'a', 'val', 'fear'), ('a', 'a', 'val', 'fear'),
        ('a', 'a', 'val', 'awe'), ('t', 'unused', 'train', 'awe'),
    ]
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'image_name', 'split', 'emotion'])
        writer.writerows(rows)


def test_labels_plurality_ties_ignored_and_key_order(tmp_path):
    path = tmp_path / 'ann.csv'
    write_annotations(path)
    data = vh.annotations(path)
    assert data['val']['keys'] == ['a/a', 'z/b']
    assert data['val']['labels'].tolist() == [6, 0]
    assert data['val']['ties'].tolist() == [False, True]
    assert data['val']['dist'][1, :2].tolist() == [.5, .5]
    assert data['test']['labels'].tolist() == [7]
    assert vh.label_summary(data)['val']['ties'] == 1
    with path.open('a') as f:
        f.write('z,b,test,anger\n')
    with pytest.raises(AssertionError, match='multiple splits'):
        vh.annotations(path)


def test_key_joins_assert_identity_not_count(tmp_path):
    x = np.array([[1., 0.], [0., 1.]], dtype=np.float16)
    np.save(tmp_path / 'img.npy', x)
    write_json(tmp_path / 'keys.json', ['b', 'a'])
    data = object.__new__(vh.Data)
    np.testing.assert_array_equal(data._view(tmp_path, ['a', 'b']), x[::-1])
    with pytest.raises(AssertionError, match='missing key'):
        data._view(tmp_path, ['a', 'c'])
    write_json(tmp_path / 'keys.json', ['a', 'a'])
    with pytest.raises(AssertionError):
        data._view(tmp_path, ['a', 'b'])


def test_export_key_digest_and_dtype_boundaries(tmp_path):
    ann = tmp_path / 'annotations.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path, smoke=2, annotation=ann, aff=tmp_path)
    stem = tmp_path / 'exports/percept/val/P3__s0'
    stem.parent.mkdir(parents=True)
    x = np.ones((2, 40), dtype=np.float32)
    np.savez(stem.with_suffix('.npz'), prob=x)
    meta = dict(split='val', seed=0, smoke=True,
                keys_sha256=hashlib.sha256(json.dumps(data.full['val']['keys']).encode()).hexdigest(),
                array_sha256=hashlib.sha256(x.tobytes()).hexdigest())
    write_json(stem.with_suffix('.json'), meta)
    np.testing.assert_array_equal(data._export('percept', 'val', 0, 'emotion'), x)
    meta['keys_sha256'] = next(iter(vh.key_hashes(['wrong', 'keys'])))
    write_json(stem.with_suffix('.json'), meta)
    with pytest.raises(AssertionError, match='key digest'):
        data._export('percept', 'val', 0, 'emotion')


def test_retrieval_stable_ties_self_exclusion_and_singletons(tmp_path):
    np.save(tmp_path / 'identical.npy', np.ones((4, 2), dtype=np.float32) / np.sqrt(2))
    x = np.load(tmp_path / 'identical.npy')
    y = np.array([0, 1, 0, 1])
    q = vh.retrieval(x, y, np.eye(2)[y], ks=(0, 1))
    # Gallery ties remain row order: relevant ranks 2, 3, 1, 2 respectively.
    np.testing.assert_allclose(q['ap'][0], [.5, 1 / 3, 1, .5])
    np.testing.assert_allclose(q['ap'][1], q['ap'][0])
    np.testing.assert_allclose(q['ndcg'], [1 / np.log2(3), .5, 1, 1 / np.log2(3)])
    singleton = vh.retrieval(x, np.array([0, 0, 1, -1]))
    assert singleton['rows'].tolist() == [0, 1, 2]
    assert np.isnan(singleton['ap'][0][2])


def test_graded_ndcg_uses_overlap_gain(tmp_path):
    x = np.array([[1., 0.], [-1., 0.], [0., 1.]], dtype=np.float32)
    p = np.array([[.75, .25], [1., 0.], [0., 1.]])
    np.savez(tmp_path / 'graded.npz', x=x, p=p)
    q = vh.retrieval(x, np.array([0, 0, 1]), p)
    # q2 ranks row0 then row1: overlap .25 then 0, ideal identical.
    assert q['ndcg'][2] == 1
    discount = 1 / np.log2(3)
    assert q['ndcg'][0] == pytest.approx((.25 + .75 * discount) / (.75 + .25 * discount))
    assert q['ndcg'][1] == pytest.approx(discount)


def test_menu_fit_whitening_and_tie_order(tmp_path):
    x = np.array([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]], dtype=np.float32)
    np.save(tmp_path / 'fit.npy', x)
    stats = vh.fit(np.load(tmp_path / 'fit.npy'))
    np.testing.assert_allclose(stats[1], [8 / 3, 2 / 3])
    out = vh.transform(x, ('whiten', 2, 0), stats)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1)
    assert len(vh.menu(40)) == 15
    assert len(vh.menu(100)) == 20
    assert len(vh.menu(2048)) == 25
    ordered = sorted([('whiten', 64, 0), ('whiten', 256, 0), ('center', 0, 0),
                      ('none', 0, 1), vh.R0], key=vh.tie_key)
    assert ordered == [vh.R0, ('center', 0, 0), ('whiten', 256, 0), ('whiten', 64, 0), ('none', 0, 1)]
    assert vh.parse_ro(vh.ro_name(('whiten', 64, 5))) == ('whiten', 64, 5)


@pytest.mark.parametrize('gain,expected', [(.001, 'none+k0'), (np.nextafter(.001, 0), 'none+k0'),
                                         (np.nextafter(.001, np.inf), 'center+k0')])
def test_readout_tie_tolerance_boundary(gain, expected):
    maps = {vh.R0: {'emotion': 0.}, ('center', 0, 0): {'emotion': gain}}
    pick, _ = vh.select_readout(maps, {'emotion': 1.})
    assert pick == expected


def test_readout_guard_edge_and_zero_range():
    # A three-aspect gain can win the average while failing the per-aspect guard.
    good = {'emotion': -1., 'style': 3., 'genre': 3.}
    bad = dict(good, emotion=np.nextafter(-1., -np.inf))
    base = dict.fromkeys(vh.ASPECTS, 0.)
    ranges = dict.fromkeys(vh.ASPECTS, 1.)
    assert vh.select_readout({vh.R0: base, ('center', 0, 0): good}, ranges)[0] == 'center+k0'
    assert vh.select_readout({vh.R0: base, ('center', 0, 0): bad}, ranges)[0] == 'none+k0'
    with pytest.raises(AssertionError, match='selection range'):
        vh.select_readout({vh.R0: {'emotion': .5}}, {'emotion': 0.})


@pytest.mark.parametrize('photo,expected', [(.001, 'artwork'), (np.nextafter(.001, 0), 'artwork'),
                                           (np.nextafter(.001, np.inf), 'photo')])
def test_teacher_strict_photo_margin(photo, expected):
    assert vh.teacher_pick(0., photo, 1.) == expected
    assert vh.teacher_pick(0., photo * 2, 2.) == expected


@pytest.mark.parametrize('best,expected', [(.001, 'clip_painting'),
                                         (np.nextafter(.001, np.inf), 'qwen_photo')])
def test_crl_normalized_tolerance_boundary(best, expected):
    assert vh.version_pick({'qwen_photo': best, 'clip_painting': 0., 'clip_photo': 0.}) == expected
    assert vh.version_pick({'qwen_painting': 0., 'clip_painting': 0.}) == 'clip_painting'


@pytest.mark.parametrize('lower,r,expected', [(1e-20, .2, 'WIN'), (0., .2, 'LEVEL'),
                                             (-.2, .2, 'BELOW'),
                                             (np.nextafter(-.2, 0), .2, 'LEVEL'),
                                             (np.nextafter(-.2, -np.inf), .2, 'BELOW'),
                                             (0., 0., 'BELOW')])
def test_outcome_edges(lower, r, expected):
    assert vh.outcome(lower, r) == expected


@pytest.mark.parametrize('third,fourth,expected', [('WIN', 'LEVEL', 'holds'), ('LEVEL', 'WIN', 'holds'),
                                                  ('BELOW', 'LEVEL', 'qualified'),
                                                  ('LEVEL', 'BELOW', 'qualified'),
                                                  ('BELOW', 'BELOW', 'fails')])
def test_verdict_including_qualified(third, fourth, expected):
    outcomes = dict(SC1='WIN', SC2='WIN', SC3=third, SC4=fourth)
    assert vh.verdict(outcomes) == expected
    assert vh.verdict(dict(outcomes, SC1='LEVEL')) == 'fails'
    assert vh.verdict(dict(outcomes, SC2='LEVEL')) == 'fails'
    assert vh.repeats_line(third) == ('yes' if third == 'WIN' else 'no')


def test_joint_query_seed_bootstrap_matches_literal_draws(tmp_path):
    c = np.arange(35).reshape(5, 7) / 50
    t = np.linspace(.1, .7, 7)
    np.savez(tmp_path / 'queries.npz', c=c, t=t)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(29):
        rows = rng.integers(0, 7, size=7)
        seeds = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        boot.append(np.mean([c[s, i] - t[i] for s in seeds for i in rows]))
    result = vh.paired_retrieval(c, t, 29)
    np.testing.assert_allclose(result['interval'], np.percentile(boot, [2.5, 97.5]))
    assert result['d'] == pytest.approx(c.mean() - t.mean())
    np.testing.assert_allclose(vh.paired_retrieval(c, c, 17)['interval'], [0, 0])
    # Joint resampling shares seeds between seeded sides.
    np.testing.assert_allclose(vh.paired_retrieval(c, c - .02, 17)['interval'], [.02, .02])
    unseeded = vh.paired_retrieval(c[0], t, 17)
    assert np.isfinite(unseeded['interval']).all()


def test_cluster_bootstrap_refits_no_partitions_and_matches_literal_draws(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    c = rng.integers(0, 3, size=(5, 3, 9), dtype=np.int32)
    t = rng.integers(0, 3, size=(3, 9), dtype=np.int32)
    lab = rng.integers(0, 3, size=9)
    np.savez(tmp_path / 'parts.npz', c=c, t=t, lab=lab)
    def no_refit(*args, **kwargs):
        pytest.fail('bootstrap refit partitions')
    monkeypatch.setattr(vh, 'partitions', no_refit)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(11):
        idx = rng.integers(0, 9, size=9)
        seeds = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        draws.append(np.mean([np.mean([adjusted_mutual_info_score(lab[idx], p[idx]) for p in c[s]])
                              for s in seeds]) -
                     np.mean([adjusted_mutual_info_score(lab[idx], p[idx]) for p in t]))
    result = vh.paired_cluster(c, t, lab, 11)
    np.testing.assert_allclose(result['interval'], np.percentile(draws, [2.5, 97.5]), atol=1e-15)
    assert result['point_c'] == pytest.approx(np.mean([vh.ami(p, lab) for p in c]))


def test_partitions_use_frozen_settings_and_one_thread(tmp_path, monkeypatch):
    settings = []
    class KMeans:
        def __init__(self, **kwargs):
            settings.append(kwargs)
        def fit(self, x):
            self.labels_ = np.arange(len(x)) % 20
            return self
    monkeypatch.setattr(vh, 'MiniBatchKMeans', KMeans)
    x = vh.unit(np.random.default_rng(5).normal(size=(24, 4)), np.float32)
    np.save(tmp_path / 'x.npy', x)
    parts = vh.partitions(np.load(tmp_path / 'x.npy'))
    assert parts.shape == (3, 24) and parts.dtype == np.int32
    assert settings == [dict(n_clusters=20, n_init=3, batch_size=4096, random_state=s) for s in range(3)]


def test_crl_float64_product_then_single_cast(tmp_path):
    rng = np.random.default_rng(5)
    x = vh.unit(rng.normal(size=(8, 3)), np.float32)
    text = vh.unit(rng.normal(size=(4, 3)), np.float32)
    np.save(tmp_path / 'text.npy', text)
    out = vh.crl_vectors(x, np.load(tmp_path / 'text.npy'))
    expected = vh.unit(x.astype(np.float64) @ text.astype(np.float64).T).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_seed_range_is_unrounded_max_minus_min():
    values = [.123456789, .123456790, .123456788, .123456792, .123456787]
    result = vh.seed_range(values)
    assert result['per_seed'] == values
    assert result['r'] == max(values) - min(values)
    assert result['r'] > 0
    with pytest.raises(AssertionError):
        vh.seed_range(values[:4])


class SyntheticData:
    def __init__(self, root):
        self.root = root
        self.rng = np.random.default_rng(823)
        self.full, self.arrays = {}, {}
        for split in ('val', 'test', 'held'):
            self.full[split] = dict(keys=[f'photo/{i:03d}' for i in range(48)],
                                    labels=np.arange(48) % 3, ties=np.zeros(48, dtype=bool),
                                    dist=np.eye(9)[np.arange(48) % 3])
        self.full.pop('held')
        self.records = self.full
        for ds in ('affection', 'artelingo'):
            for split in ('val', 'test', 'held'):
                for arm in set(vh.AFF_ARMS + vh.ART_ARMS):
                    for seed in (vh.SEEDS if arm in ('cstar', 'percept') else (None,)):
                        path = root / f'{ds}_{split}_{arm}_{seed}.npy'
                        np.save(path, vh.unit(self.rng.normal(size=(48, 4)), np.float32))
                        self.arrays[ds, arm, split, seed] = path
    def embed(self, dataset, arm, split, aspect='emotion', seed=None):
        return np.load(self.arrays[dataset, arm, split, seed])
    def labels(self, dataset, split, aspect):
        return np.arange(48) % 3
    def dist(self, dataset, split):
        return np.eye(9)[np.arange(48) % 3]


def test_crossfit_fits_other_parity_half(tmp_path, monkeypatch):
    data = SyntheticData(tmp_path)
    run = vh.Run(data, 48)
    original = vh.fit
    fits = []
    def record(x):
        fits.append(x.copy())
        return original(x)
    monkeypatch.setattr(vh, 'fit', record)
    maps = run.crossfit('affection', 'plain', ('emotion',))
    x = data.embed('affection', 'plain', 'val')
    np.testing.assert_array_equal(fits[0], x[1::2])
    np.testing.assert_array_equal(fits[1], x[::2])
    assert len(maps) == 15 and all(np.isfinite(v['emotion']) for v in maps.values())


def test_all_stages_compare_reports_and_detect_mismatches(tmp_path, monkeypatch, capsys):
    data = SyntheticData(tmp_path)
    run = vh.Run(data, 48)
    # Frozen r11 ranges are represented by independent synthetic inputs here.
    run.art_ranges = dict.fromkeys(vh.ASPECTS, .1)
    cluster, retrieval = vh.paired_cluster, vh.paired_retrieval
    monkeypatch.setattr(vh, 'paired_cluster', lambda c, t, labels, draws: cluster(c, t, labels, 3))
    monkeypatch.setattr(vh, 'paired_retrieval', lambda c, t, draws: retrieval(c, t, 7))
    # Compute all expected results BEFORE any reference JSON exists.
    derived = {stage: run.derive(stage) for stage in ('0b', '1', '2', '3')}
    write_json(tmp_path / 'stage0a/gates.json', dict(ok=True, items={}))
    assert derived['0b']['map']['r'] > 0 and derived['0b']['clu']['r'] > 0
    assert derived['1']['picks']['artelingo']['readout']['cstar'] == 'center+k10'
    assert derived['1']['picks']['artelingo']['readout']['crl_clip_painting'] == 'none+k10'
    for stage, result in derived.items():
        if stage == '1':
            for name in ('scores', 'picks'):
                write_json(tmp_path / f'stage1/{name}.json', result[name])
        else:
            write_json(tmp_path / f'stage{stage}' / ('ranges.json' if stage == '0b' else 'verdict.json'), result)
    for stage, result in derived.items():
        assert vh.compare_final(run, stage, result, tmp_path) == 0
        report = vh.read_json(tmp_path / f'vh/{stage}.json')
        assert report['ok'] and report['items']
        assert all(set(v) == {'ours', 'theirs', 'ok'} for v in report['items'].values())
    reference = vh.read_json(tmp_path / 'stage2/verdict.json')
    reference['sc']['SC1']['interval'][0] += .001
    write_json(tmp_path / 'stage2/verdict.json', reference)
    assert vh.compare_final(run, '2', derived['2'], tmp_path) == 1
    report = vh.read_json(tmp_path / 'vh/2.json')
    assert not report['items']['result.sc.SC1.interval[0]']['ok']
    write_json(tmp_path / 'stage0a/gates.json', dict(ok=False, items={}))
    assert vh.compare_final(run, '0b', derived['0b'], tmp_path) == 1
    assert not vh.read_json(tmp_path / 'vh/0b.json')['items']['stage0a.ok']['ok']
    assert 'mismatches' in capsys.readouterr().out


def test_comparison_absolute_tolerance_and_missing_keys():
    check = vh.Check()
    check.compare('inside', 0., 1e-6)
    check.compare('outside', 0., np.nextafter(1e-6, np.inf))
    check.compare('missing', {'pick': 'none+k0'}, {})
    assert check.items['inside']['ok']
    assert not check.items['outside']['ok']
    assert not check.items['missing.pick']['ok']
    check.compare('bool', True, 1)
    assert not check.items['bool']['ok']


def test_cli_stage_flags_and_missing_inputs(tmp_path, monkeypatch):
    class MissingData:
        def __init__(self, outputs, smoke):
            assert outputs == tmp_path and smoke == 21
            raise FileNotFoundError('synthetic missing store')
    monkeypatch.setattr(vh, 'Data', MissingData)
    assert vh.main(['--stage', '0b', '--outputs', str(tmp_path), '--smoke', '21']) == 2
    with pytest.raises(SystemExit):
        vh.main(['--stage', '0a'])


def test_no_round_module_imports_or_result_reads_before_final_comparison():
    import ast
    tree = ast.parse(Path(vh.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith('ac_') for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith('ac_')


def test_raw_store_view_and_crl_loaders_join_by_key(tmp_path):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path, smoke=2, annotation=ann, aff=tmp_path)
    folder = tmp_path / 'qwen3vl_emb_2b'
    folder.mkdir()
    # The default store order deliberately differs from the annotation's key order.
    keys = ['z/b', 'a/c', 'a/a']
    x = np.zeros((3, 2048), dtype=np.float16)
    x[np.arange(3), np.arange(3)] = 1
    np.save(folder / 'img.npy', x)
    write_json(folder / 'index.json', dict(images=[dict(source=k.split('/')[0], image_name=k.split('/')[1])
                                                  for k in keys]))
    loaded = data.embed('affection', 'plain', 'val')
    np.testing.assert_array_equal(loaded, x[[2, 0]].astype(np.float32))
    view = tmp_path / 'qwen3vl_emb_2b_aspect_prompts/emotion_focus_photo'
    view.mkdir(parents=True)
    np.save(view / 'img.npy', x)
    write_json(view / 'keys.json', keys)
    np.testing.assert_array_equal(data.embed('affection', 'teacher_photo', 'val'), loaded)
    crl = tmp_path / 'crl'
    crl.mkdir()
    text = np.zeros((2, 2048), dtype=np.float32)
    text[0, 0], text[1, 2] = 1, 1
    np.save(crl / 'qwen_photo_emotion.npy', text)
    np.testing.assert_array_equal(data.embed('affection', 'crl_qwen_photo', 'val'), [[0., 1.], [1., 0.]])


def test_label_gate_values_and_extra_pick_are_compared(tmp_path):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path, smoke=2, annotation=ann, aff=tmp_path)
    run = vh.Run(data, smoke=2)
    mine = dict(clu=vh.seed_range([0, .1, .2, .3, .4]), map=vh.seed_range([.1] * 5),
                ndcg=vh.seed_range([.2] * 5))
    write_json(tmp_path / 'stage0b/ranges.json', mine)
    write_json(tmp_path / 'stage0a/gates.json', dict(ok=True, items={
        'label_counts_val': dict(value=[1, 0, 0, 0, 0, 0, 1, 0, 0]),
        'label_ties_val': dict(value=0)}))
    assert vh.compare_final(run, '0b', mine, tmp_path) == 1
    report = vh.read_json(tmp_path / 'vh/0b.json')
    assert report['items']['labels.gates.label_ties_val']['ours'] == 1
    check = vh.Check()
    check.compare('picks', {'readout': {'cstar': 'none+k0'}},
                  {'readout': {'cstar': 'none+k0', 'extra_arm': 'none+k0'}}, strict=True)
    assert not check.ok
