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
    data.smoke = 0
    write_json(tmp_path / 'meta.json', {'instruction': None, 'gate': {'ok': True}, 'smoke': False})
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
    def cluster_cstar(self, split, seed):
        return self.embed('affection', 'cstar', split, seed=seed)
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
    run = vh.Run(data, 48, draws_clu=3, draws_ret=7)
    # Frozen r11 ranges are represented by independent synthetic inputs here.
    run.art_ranges = dict.fromkeys(vh.ASPECTS, .1)
    # Compute all expected results BEFORE any reference JSON exists.
    derived = {stage: run.derive(stage) for stage in ('0b', '1', '2', '3')}
    # A smoke run reads the committed records from <outputs>/records (ticket 13), not from the round folder.
    write_json(tmp_path / 'records/stage0b_ranges.json', derived['0b'])
    write_json(tmp_path / 'records/stage1_picks.json', derived['1']['picks'])
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
        assert report['draws_clu'] == 3 and report['draws_ret'] == 7
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


def test_raw_store_view_and_crl_loaders_join_by_key(tmp_path, monkeypatch):
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
    write_json(view / 'meta.json', {
        'instruction': 'Represent the emotion that this photo, or this description of a photo, conveys. Focus only on the emotion.',
        'gate': {'ok': True}, 'smoke': False})
    np.testing.assert_array_equal(data.embed('affection', 'teacher_photo', 'val'), loaded)
    crl = tmp_path / 'crl'
    crl.mkdir()
    text = np.zeros((2, 2048), dtype=np.float32)
    text[0, 0], text[1, 2] = 1, 1
    np.save(crl / 'qwen_photo_emotion.npy', text)
    monkeypatch.setattr(vh, 'R11', tmp_path / 'r11')
    write_json(tmp_path / 'r11/crl/lists.json', {'aspects': {'emotion': {'values': ['first', 'second']}}})
    write_crl_metadata(data.outputs, 'qwen', 'photo', 'emotion')
    np.testing.assert_array_equal(data.embed('affection', 'crl_qwen_photo', 'val'), [[0., 1.], [1., 0.]])


def test_label_gate_values_and_extra_pick_are_compared(tmp_path, monkeypatch):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path, smoke=2, annotation=ann, aff=tmp_path)
    run = vh.Run(data, smoke=2)
    mine = dict(clu=vh.seed_range([0, .1, .2, .3, .4]), map=vh.seed_range([.1] * 5),
                ndcg=vh.seed_range([.2] * 5))
    write_json(tmp_path / 'records/stage0b_ranges.json', mine)    # smoke run: <outputs>/records (ticket 13)
    write_json(tmp_path / 'stage0b/ranges.json', mine)
    write_json(tmp_path / 'stage0a/gates.json', dict(ok=True, items={
        'label_counts_val': dict(value=[1, 0, 0, 0, 0, 0, 1, 0, 0]),
        'label_ties_val': dict(value=0)}))
    assert vh.compare_final(run, '0b', mine, tmp_path) == 1
    report = vh.read_json(tmp_path / 'vh/0b.json')
    assert report['items']['labels.gates.label_ties_val']['ours'] == 1
    write_json(tmp_path / 'stage0a/gates.json', dict(ok=True, items={       # the runner's smoke marker passes in smoke
        'label_counts_val': dict(value=vh.SKIPPED), 'label_ties_val': dict(value=vh.SKIPPED)}))
    assert vh.compare_final(run, '0b', mine, tmp_path) == 0
    monkeypatch.setattr(vh, 'ROUND', tmp_path / 'records')                  # and never in a full run
    assert vh.compare_final(vh.Run(data, smoke=0), '0b', mine, tmp_path) == 1
    report = vh.read_json(tmp_path / 'vh/0b.json')
    assert not report['items']['labels.gates.label_ties_val']['ok']
    assert not report['items']['labels.gates.label_counts_val']['ok']
    check = vh.Check()
    check.compare('picks', {'readout': {'cstar': 'none+k0'}},
                  {'readout': {'cstar': 'none+k0', 'extra_arm': 'none+k0'}}, strict=True)
    assert not check.ok


def test_artelingo_clip_crl_uses_frozen_texts_and_reports_round_mismatch(tmp_path, monkeypatch):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path / 'outputs', smoke=2, annotation=ann, aff=tmp_path)
    data.art_records = {'val': {'keys': ['painting']}}
    monkeypatch.setattr(vh, 'ART', tmp_path / 'art')
    monkeypatch.setattr(vh, 'R11', tmp_path / 'r11')
    clip = tmp_path / 'art/clip_l14'
    clip.mkdir(parents=True)
    np.save(clip / 'img.npy', np.array([[1., 0.]], dtype=np.float16))
    write_json(clip / 'index.json', {'images': [{'painting': 'painting'}]})
    frozen = np.eye(2, dtype=np.float32)
    frozen_path = tmp_path / 'r11/crl/emotion_text.npy'
    frozen_path.parent.mkdir(parents=True)
    np.save(frozen_path, frozen)
    local = data.outputs / 'crl/clip_painting_emotion.npy'
    local.parent.mkdir(parents=True)
    np.save(local, frozen[::-1])
    values = {'values': ['first', 'second']}
    write_json(tmp_path / 'r11/crl/lists.json', {'aspects': {'emotion': values, 'style': values}})
    for aspect in ('emotion', 'style'):                    # r11's own sidecars; the round has no local CLIP painting json
        write_json(tmp_path / f'r11/crl/{aspect}_text.json', dict(
            model='openai/clip-vit-large-patch14', texts=[f'a painting with the {aspect} of {v}' for v in values['values']]))
    np.save(tmp_path / 'r11/crl/style_text.npy', frozen)
    np.testing.assert_array_equal(data.embed('artelingo', 'crl_clip_painting', 'val'), [[1., 0.]])
    np.testing.assert_array_equal(data.embed('artelingo', 'crl_clip_painting', 'val', aspect='style'), [[1., 0.]])
    assert data.frozen_crl_aspects == {'emotion', 'style'}
    mine = {metric: vh.seed_range([.1, .2, .3, .4, .5]) for metric in ('clu', 'map', 'ndcg')}
    write_json(data.outputs / 'records/stage0b_ranges.json', mine)    # smoke run: <outputs>/records (ticket 13)
    write_json(data.outputs / 'stage0a/gates.json', {'ok': True, 'items': {}})
    write_json(data.outputs / 'stage0b/ranges.json', mine)
    run = vh.Run(data, smoke=2)
    assert vh.compare_final(run, '0b', mine, data.outputs) == 1
    report = vh.read_json(data.outputs / 'vh/0b.json')
    assert not report['items']['inputs.crl.clip_painting_emotion.equals_r11']['ok']
    assert 'inputs.crl.clip_painting_style.equals_r11' not in report['items']       # no local style copy to compare
    np.save(local, frozen)
    assert vh.compare_final(run, '0b', mine, data.outputs) == 0


def write_crl_metadata(root, encoder, wording, aspect):
    values = ['first', 'second']
    side = dict(texts=[f'a {wording} with the {aspect} of {v}' for v in values], encoder=encoder,
                model_id=('openai/clip-vit-large-patch14' if encoder == 'clip'
                          else 'Qwen/Qwen3-VL-Embedding-2B'),
                instruction=None if encoder == 'clip' else "Represent the user's input.")
    write_json(root / 'crl' / f'{encoder}_{wording}_{aspect}.json', side)
    return side


def crl_input_data(tmp_path, monkeypatch):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path / 'outputs', smoke=2, annotation=ann, aff=tmp_path)
    monkeypatch.setattr(vh, 'R11', tmp_path / 'r11')
    write_json(tmp_path / 'r11/crl/lists.json', {'aspects': {
        a: {'values': ['first', 'second']} for a in ('emotion', 'style', 'genre')}})
    keys = ['a/a', 'z/b', 'a/c']
    for folder, width in ((tmp_path / 'clip_l14_valtest', 768), (tmp_path / 'qwen3vl_emb_2b', 2048)):
        folder.mkdir()
        x = np.zeros((3, width), dtype=np.float16)
        x[np.arange(3), np.arange(3)] = 1
        np.save(folder / 'img.npy', x)
        if width == 768:
            write_json(folder / 'keys.json', keys)
            write_json(folder / 'meta.json', {'instruction': None, 'gate': {'ok': True}, 'smoke': False})
        else:
            write_json(folder / 'index.json', {'images': [
                {'source': k.split('/')[0], 'image_name': k.split('/')[1]} for k in keys]})
    return data


@pytest.mark.parametrize('encoder', ['clip', 'qwen'])
@pytest.mark.parametrize('wording', ['painting', 'photo'])
@pytest.mark.parametrize('aspect', ['emotion', 'style', 'genre'])
def test_crl_sidecar_accepts_spec_texts_in_frozen_value_order(tmp_path, monkeypatch, encoder, wording, aspect):
    data = crl_input_data(tmp_path, monkeypatch)
    write_crl_metadata(data.outputs, encoder, wording, aspect)
    text = np.zeros((2, 768 if encoder == 'clip' else 2048), dtype=np.float32)
    text[0, 0], text[1, 1] = 1, 1
    np.save(data.outputs / 'crl' / f'{encoder}_{wording}_{aspect}.npy', text)
    np.testing.assert_array_equal(data.embed('affection', f'crl_{encoder}_{wording}', 'val', aspect), np.eye(2))


@pytest.mark.parametrize('encoder', ['clip', 'qwen'])
@pytest.mark.parametrize('corruption', ['value_order', 'template', 'encoder', 'model_id', 'instruction', 'missing'])
def test_crl_sidecar_rejects_wrong_metadata(tmp_path, monkeypatch, encoder, corruption):
    data = crl_input_data(tmp_path, monkeypatch)
    side = write_crl_metadata(data.outputs, encoder, 'photo', 'emotion')
    text = np.zeros((2, 768 if encoder == 'clip' else 2048), dtype=np.float32)
    text[0, 0], text[1, 1] = 1, 1
    np.save(data.outputs / 'crl' / f'{encoder}_photo_emotion.npy', text)
    if corruption == 'value_order':
        side['texts'].reverse()
    elif corruption == 'template':
        side['texts'] = ['a painting with the emotion of first', 'a painting with the emotion of second']
    elif corruption == 'missing':
        side.pop('instruction')
    else:
        side[corruption] = 'wrong'
    write_json(data.outputs / 'crl' / f'{encoder}_photo_emotion.json', side)
    with pytest.raises((AssertionError, KeyError)):
        data.embed('affection', f'crl_{encoder}_photo', 'val')


@pytest.mark.parametrize('arm,folder,instruction,width', [
    ('teacher_artwork', 'qwen3vl_emb_2b_aspect_prompts/emotion_focus',
     'Represent the emotion that this artwork, or this description of an artwork, conveys. Focus only on the emotion.', 2048),
    ('teacher_photo', 'qwen3vl_emb_2b_aspect_prompts/emotion_focus_photo',
     'Represent the emotion that this photo, or this description of a photo, conveys. Focus only on the emotion.', 2048),
    ('clip', 'clip_l14_valtest', None, 768)])
@pytest.mark.parametrize('corruption', [None, 'instruction', 'gate', 'smoke', 'missing'])
def test_view_metadata_gates_and_verbatim_instruction(tmp_path, monkeypatch, arm, folder, instruction, width, corruption):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    monkeypatch.setattr(vh, 'EXPECTED_COUNTS', {'val': [1, 0, 0, 0, 0, 0, 1, 0, 0],
                                              'test': [0, 0, 0, 0, 0, 0, 0, 1, 0]})
    data = vh.Data(tmp_path, annotation=ann, aff=tmp_path)
    folder = tmp_path / folder
    folder.mkdir(parents=True)
    x = np.zeros((2, width), dtype=np.float16)
    x[0, 0], x[1, 1] = 1, 1
    np.save(folder / 'img.npy', x)
    write_json(folder / 'keys.json', ['a/a', 'z/b'])
    meta = {'instruction': instruction, 'gate': {'ok': True}, 'smoke': False}
    if corruption == 'instruction':
        meta['instruction'] = 'incorrect instruction'
    elif corruption == 'gate':
        meta['gate']['ok'] = False
    elif corruption == 'smoke':
        meta['smoke'] = True
    elif corruption == 'missing':
        meta.pop('instruction')
    write_json(folder / 'meta.json', meta)
    if corruption:
        with pytest.raises((AssertionError, KeyError)):
            data.embed('affection', arm, 'val')
    else:
        np.testing.assert_array_equal(data.embed('affection', arm, 'val'), x.astype(np.float32))


@pytest.mark.parametrize('flags,smoke,expected', [
    ([], 0, (2000, 10000)), (['--smoke', '48'], 48, (2000, 10000)),
    (['--smoke', '48', '--draws-clu', '11', '--draws-ret', '29'], 48, (11, 29))])
def test_cli_bootstrap_draw_counts_are_explicit(tmp_path, monkeypatch, flags, smoke, expected):
    captured = []
    class Run:
        def __init__(self, data, smoke, draws_clu=2000, draws_ret=10000):
            captured.append((smoke, draws_clu, draws_ret))
        def derive(self, stage):
            return {}
    monkeypatch.setattr(vh, 'Data', lambda outputs, smoke: None)
    monkeypatch.setattr(vh, 'Run', Run)
    monkeypatch.setattr(vh, 'compare_final', lambda *args: 0)
    assert vh.main(['--stage', '2', '--outputs', str(tmp_path)] + flags) == 0
    assert captured == [(smoke, *expected)]


def test_cstar_clustering_preserves_float16_rows_but_retrieval_normalizes(tmp_path, monkeypatch):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path, smoke=2, annotation=ann, aff=tmp_path)
    x = np.zeros((2, 2048), dtype=np.float16)
    x[:, :2] = [.6001, .8001]  # rounding leaves a small, valid norm deviation
    for seed in range(5):
        stem = tmp_path / f'exports/cstar/val/emotion__s{seed}'
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stem.with_suffix('.npz'), img=x)
        write_json(stem.with_suffix('.json'), dict(split='val', seed=seed, aspect='emotion', smoke=True,
                   keys_sha256=hashlib.sha256(json.dumps(['a/a', 'z/b']).encode()).hexdigest(),
                   array_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    raw = x.astype(np.float32)
    np.testing.assert_array_equal(data.cluster_cstar('val', 0), raw)
    normalized = data.embed('affection', 'cstar', 'val', seed=0)
    assert not np.array_equal(normalized, raw)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1., atol=1e-7)
    fitted = []
    class KMeans:
        def __init__(self, **kwargs):
            pass
        def fit(self, array):
            fitted.append(array.copy())
            self.labels_ = np.array([0, 1])
            return self
    monkeypatch.setattr(vh, 'MiniBatchKMeans', KMeans)
    assert vh.Run(data, smoke=2).parts('val', 'cstar').shape == (5, 3, 2)
    assert len(fitted) == 15
    for array in fitted:
        np.testing.assert_array_equal(array, raw)
    x[0, 0] = 1.1
    stem = tmp_path / 'exports/cstar/val/emotion__s0'
    np.savez(stem.with_suffix('.npz'), img=x)
    meta = vh.read_json(stem.with_suffix('.json'))
    meta['array_sha256'] = hashlib.sha256(x.tobytes()).hexdigest()
    write_json(stem.with_suffix('.json'), meta)
    with pytest.raises(AssertionError, match='unit norm'):
        data.cluster_cstar('val', 0)


def test_absent_json_key_is_not_explicit_null():
    missing = vh.Check()
    missing.compare('row', {'readout_c': None}, {})
    assert not missing.ok
    explicit_null = vh.Check()
    explicit_null.compare('row', {'readout_c': None}, {'readout_c': None})
    assert explicit_null.ok


@pytest.fixture
def comparison_files(tmp_path, monkeypatch):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path, smoke=2, annotation=ann, aff=tmp_path)
    run = vh.Run(data, smoke=2, draws_clu=3, draws_ret=7)
    run.ranges = {m: vh.seed_range([.1, .2, .3, .4, .5]) for m in ('clu', 'map', 'ndcg')}
    run.selection = {'picks': {'affection': {'readout': {'cstar': 'none+k0'}}}}
    mine = {'sc': {}, 'verdict': 'holds', 'repeats_artelingo_win': 'yes'}
    for i in range(1, 5):
        clustering = i in (1, 3)
        mine['sc'][f'SC{i}'] = dict(candidate='cstar', comparator='plain', dataset='affection', aspect='emotion',
             metric='ami' if clustering else 'map', readout_c=None if clustering else 'none+k0',
             readout_t=None if clustering else 'none+k0', point_c=.6, point_t=.5, d=.1,
             interval=[.01, .2], r=.4, outcome='WIN', **{'pass': True})
    write_json(tmp_path / 'stage0a/gates.json', {'ok': True, 'items': {}})
    write_json(tmp_path / 'stage0b/ranges.json', run.ranges)
    write_json(tmp_path / 'stage1/picks.json', run.selection['picks'])
    write_json(tmp_path / 'stage2/verdict.json', mine)
    # The fixture is a smoke run, so the committed records are the runner's scratch copies under <outputs>/records
    # (ticket 13); ROUND (full runs) is pointed at a folder with deliberately wrong records to prove it is not read.
    monkeypatch.setattr(vh, 'ROUND', tmp_path / 'committed', raising=False)
    write_json(tmp_path / 'committed/stage0b_ranges.json', {'wrong': True})
    write_json(tmp_path / 'committed/stage1_picks.json', {'wrong': True})
    write_json(tmp_path / 'records/stage0b_ranges.json', run.ranges)
    write_json(tmp_path / 'records/stage1_picks.json', run.selection['picks'])
    return run, mine, tmp_path


@pytest.mark.parametrize('mutation,item', [
    ('flipped_pick', 'picks.affection.readout.cstar'),
    ('changed_outcome', 'result.sc.SC1.outcome'),
    ('d_off_1e5', 'result.sc.SC1.d'),
    ('removed_key', 'result.sc.SC1.readout_c')])
def test_report_rejects_planted_pick_outcome_difference_and_removed_key(comparison_files, mutation, item):
    run, mine, outputs = comparison_files
    assert vh.compare_final(run, '2', mine, outputs) == 0
    if mutation == 'flipped_pick':
        picks = vh.read_json(outputs / 'stage1/picks.json')
        picks['affection']['readout']['cstar'] = 'center+k0'
        write_json(outputs / 'stage1/picks.json', picks)
    else:
        theirs = vh.read_json(outputs / 'stage2/verdict.json')
        if mutation == 'changed_outcome':
            theirs['sc']['SC1']['outcome'] = 'BELOW'
        elif mutation == 'd_off_1e5':
            theirs['sc']['SC1']['d'] += 1e-5
        else:
            theirs['sc']['SC1'].pop('readout_c')
        write_json(outputs / 'stage2/verdict.json', theirs)
    assert vh.compare_final(run, '2', mine, outputs) == 1
    report = vh.read_json(outputs / 'vh/2.json')
    assert not report['items'][item]['ok']


@pytest.mark.parametrize('copy,item', [('stage0b_ranges.json', 'committed.ranges.map.r'),
                                      ('stage1_picks.json', 'committed.picks.affection.readout.cstar')])
def test_committed_range_and_pick_copies_are_independently_compared(comparison_files, copy, item):
    run, mine, outputs = comparison_files
    assert vh.compare_final(run, '2', mine, outputs) == 0
    record = vh.read_json(outputs / 'records' / copy)
    if copy == 'stage0b_ranges.json':
        record['map']['r'] += 1e-5
    else:
        record['affection']['readout']['cstar'] = 'center+k0'
    write_json(outputs / 'records' / copy, record)
    assert vh.compare_final(run, '2', mine, outputs) == 1
    assert not vh.read_json(outputs / 'vh/2.json')['items'][item]['ok']


def test_committed_records_come_from_outputs_records_in_smoke_and_from_the_round_folder_otherwise(tmp_path):
    assert vh.records_dir(tmp_path, 7) == tmp_path / 'records'
    assert vh.records_dir(str(tmp_path), 1) == tmp_path / 'records'
    assert vh.records_dir(tmp_path, 0) == vh.ROUND
    assert (vh.ROUND / 'stage0b_ranges.json').parent == Path(vh.__file__).resolve().parent


def test_smoke_compare_does_not_read_the_round_folder(comparison_files):
    # ROUND holds {'wrong': True} records in this fixture; a smoke run that read them would mismatch.
    run, mine, outputs = comparison_files
    assert vh.compare_final(run, '2', mine, outputs) == 0
    items = vh.read_json(outputs / 'vh/2.json')['items']
    assert items['committed.ranges.map.r']['ok'] and items['committed.picks.affection.readout.cstar']['ok']
    (outputs / 'records/stage0b_ranges.json').unlink()
    with pytest.raises(FileNotFoundError):
        vh.compare_final(run, '2', mine, outputs)


# ---- --views-root (ticket 13): a smoke vh run reads the smoke views, everything else stays on the real folders ----

def test_views_are_read_from_views_root_while_the_store_stays_under_aff(tmp_path):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    aff, root = tmp_path / 'aff', tmp_path / 'scratch/smoke_views'
    data = vh.Data(tmp_path, smoke=2, annotation=ann, aff=aff, views_root=root)
    assert data.records['val']['keys'] == ['a/a', 'z/b']
    views = {'teacher_artwork': ('qwen3vl_emb_2b_aspect_prompts/emotion_focus', 2048),
             'teacher_photo': ('qwen3vl_emb_2b_aspect_prompts/emotion_focus_photo', 2048),
             'clip': ('clip_l14_valtest', 768)}
    for arm, (folder, width) in views.items():
        (root / folder).mkdir(parents=True)
        x = np.zeros((2, width), dtype=np.float16)
        x[0, 1], x[1, 0] = 1, 1                               # rows in keys.json order: z/b first
        np.save(root / folder / 'img.npy', x)
        write_json(root / folder / 'keys.json', ['z/b', 'a/a'])
        write_json(root / folder / 'meta.json',
                   {'instruction': vh.VIEW_INSTRUCTIONS[arm], 'gate': {'ok': True}, 'smoke': True})
        got = data.embed('affection', arm, 'val')
        np.testing.assert_array_equal(got[:, :2], [[1, 0], [0, 1]])        # joined by key: a/a first
    assert not (aff / 'qwen3vl_emb_2b_aspect_prompts').exists()          # nothing was read from aff for views


def test_views_root_defaults_to_the_real_aff_folders(tmp_path):
    ann = tmp_path / 'ann.csv'
    write_annotations(ann)
    data = vh.Data(tmp_path, smoke=2, annotation=ann, aff=tmp_path / 'aff')
    assert data.views_root == tmp_path / 'aff'
    assert vh.Data(tmp_path, smoke=2, annotation=ann).views_root == vh.AFF


def test_views_root_flag_is_passed_only_when_given(tmp_path, monkeypatch):
    seen = []

    class StopData:
        def __init__(self, outputs, smoke, **kwargs):
            seen.append((outputs, smoke, kwargs))
            raise FileNotFoundError('stop here')

    monkeypatch.setattr(vh, 'Data', StopData)
    assert vh.main(['--stage', '0b', '--outputs', str(tmp_path), '--smoke', '3']) == 2
    assert vh.main(['--stage', '0b', '--outputs', str(tmp_path), '--smoke', '3',
                    '--views-root', str(tmp_path / 'v')]) == 2
    assert seen == [(tmp_path, 3, {}), (tmp_path, 3, {'views_root': tmp_path / 'v'})]


def test_cli_smoke_must_equal_ac_smoke_when_that_is_set(tmp_path, monkeypatch):
    seen = []

    class StopData:
        def __init__(self, outputs, smoke):
            seen.append(smoke)
            raise FileNotFoundError('synthetic stop')
    monkeypatch.setattr(vh, 'Data', StopData)
    monkeypatch.setenv('AC_SMOKE', '21')
    base = ['--stage', '0b', '--outputs', str(tmp_path)]
    for flags in (['--smoke', '22'], []):                 # a different N, and a full run, against AC_SMOKE = 21
        with pytest.raises(SystemExit) as e:
            vh.main(base + flags)
        assert e.value.code == 2
    assert seen == []
    assert vh.main(base + ['--smoke', '21']) == 2 and seen == [21]     # equal: reaches the data
    monkeypatch.setenv('AC_SMOKE', '')                     # empty = 0: a full run and a standalone smoke are both allowed
    assert vh.main(base) == 2 and vh.main(base + ['--smoke', '21']) == 2 and seen == [21, 0, 21]
