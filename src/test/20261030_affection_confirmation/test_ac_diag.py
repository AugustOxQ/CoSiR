"""Diagnostics against the runner's synthetic stage files, and the Impressions figure on a tiny store."""
import json
from argparse import Namespace

import numpy as np
import pytest
from PIL import Image

import ac_cluster
import ac_data
import ac_diag
import ac_paths
import ac_run
import ac_select
import ac_stats
import rs_vh
import rt_data
import rt_readout

ARGS = Namespace(workers=1, draws_clu=2, draws_ret=2)


@pytest.fixture
def stages(tmp_path, monkeypatch):
    n = 32
    monkeypatch.setattr(ac_paths, 'OUTPUTS', tmp_path / 'out')
    monkeypatch.setattr(ac_paths, 'SMOKE', n)
    monkeypatch.setattr(ac_paths, 'N', dict(val=n, test=n))
    monkeypatch.setattr(ac_paths, 'RANGES_FILE', tmp_path / 'ranges.json')
    monkeypatch.setattr(ac_paths, 'PICKS_FILE', tmp_path / 'picks.json')
    rng = np.random.default_rng(123)
    xs = {}
    for arm in ac_data.ARMS:
        for split in ('val', 'test'):
            for seed in (ac_paths.SEEDS if arm in ac_data.SEEDED else (None,)):
                for head in ac_paths.ASPECTS:
                    x = rng.uniform(.01, .99, (n, 40)) if arm == 'percept' else rng.normal(size=(n, 8))
                    xs[arm, split, seed, head] = (ac_data._unit(x) * 1.0005).astype(np.float16)
    labels = (np.arange(n) % 4).astype(np.int64)
    dist = np.eye(9, dtype=np.float64)[labels]
    tied = np.arange(n) < 2

    def stored(arm, split, seed, *, head='emotion'):
        assert arm == 'cstar', arm
        return xs[arm, split, seed, head]

    def embed(arm, split, seed=None, *, head='emotion'):
        return ac_data._unit(xs[arm, split, seed, head])

    monkeypatch.setattr(ac_data, 'stored', stored)
    monkeypatch.setattr(ac_data, 'embed', embed)
    monkeypatch.setattr(ac_data, 'percept_prob', lambda split, seed: xs['percept', split, seed, 'emotion'].astype(np.float32))
    monkeypatch.setattr(ac_data, 'labels', lambda split: labels)
    monkeypatch.setattr(ac_data, 'emotion_dist', lambda split: dist)
    monkeypatch.setattr(ac_data, 'ties', lambda split: tied)
    monkeypatch.setattr(ac_data, 'keys', lambda split: [f'source/{i:03}.jpg' for i in range(n)])
    monkeypatch.setattr(ac_data, 'sources', lambda split: ['fi' if i % 3 == 0 else 'art' for i in range(n)])
    monkeypatch.setattr(rt_data, 'labels', lambda split: {a: labels for a in ac_paths.ASPECTS})
    monkeypatch.setattr(rt_data, 'embed', lambda arm, split, aspect, seed=None:
                        embed('cstar' if arm == 'cstar' else 'crl_clip_painting',
                              'val' if split == 'val' else 'test', seed, head=aspect))
    monkeypatch.setattr(ac_data, 'artelingo_crl_qwen', lambda split, aspect:
                        embed('crl_qwen_painting', 'val' if split == 'val' else 'test', head=aspect))
    monkeypatch.setattr(ac_stats, 'r11_r_a', lambda: {a: .01 for a in ac_paths.ASPECTS})
    monkeypatch.setattr(rt_readout, 'menu', lambda width: [ac_select.readout_from_name(name)
                        for name in ('none+k0', 'center+k0', 'none+k10', 'center+k10')])
    ac_run.write_json(ac_paths.OUTPUTS / 'stage0a/gates.json', dict(ok=True, smoke=n, items={}))
    ranges = ac_run.stage0b(ARGS)
    picks = ac_run.stage1(ARGS)
    s2, s3 = ac_run.stage2(ARGS), ac_run.stage3(ARGS)
    return dict(ranges=ranges, picks=picks, s2=s2, s3=s3, xs=xs, keep=np.flatnonzero(~tied))


def test_diag_reads_all_ten_arms_from_the_runner_and_refits_untied_stored_rows(stages, monkeypatch):
    real = ac_cluster.partitions
    seen = []

    def partitions(x):
        if len(seen) < 5:
            expected = stages['xs']['cstar', 'test', len(seen), 'emotion'][stages['keep']]
            assert x.dtype == np.float16, 'untied c* needs stored rows'
            np.testing.assert_array_equal(x.astype(np.float32), expected.astype(np.float32))
        else:
            assert x.dtype == np.float32, 'other arms stay unit float32'
        seen.append(x)
        return real(x)

    monkeypatch.setattr(ac_cluster, 'partitions', partitions)
    doc = ac_diag.diag(ARGS)
    assert len(seen) == 7, 'c* refit once per seed, plus two comparators'
    assert json.loads((ac_paths.OUTPUTS / 'diag/diag.json').read_text()) == doc, 'diagnostics saved'
    for field in ('r0_rows', 'ndcg', 'per_source_map'):
        assert set(doc[field]) == set(ac_data.ARMS), field
    for arm in ac_data.ARMS:
        row = stages['s2']['rows'][arm]
        assert doc['r0_rows'][arm]['map_r0'] == row['map_r0'], arm
        assert doc['ndcg'][arm]['readout'] == stages['picks']['affection']['readout'][arm], arm
        for kind, r0 in (('picked', False), ('r0', True)):
            ap, rows = ac_diag._perquery(2, arm, r0)
            src = np.asarray(ac_data.sources('test'))[rows]
            for source in ('fi', 'art'):
                got = doc['per_source_map'][arm][kind][source]
                assert got['map'] == pytest.approx(ac_diag._seed_mean(ap)[src == source].mean()), (arm, source)
                assert got['n_queries'] == int((src == source).sum()), (arm, source)
    assert set(doc['per_class_ap']) == {'cstar', 'teacher_' + stages['picks']['affection']['teacher_wording']['retrieval']}
    assert set(doc['switch_column']) == set(ac_paths.ASPECTS), 'all heads scored'
    assert len(doc['cstar_seed_spread']['ami']['per_seed']) == 5, 'saved partitions read'
    assert doc['untied']['n'] == 30 and set(doc['untied']['sc']) == set(ac_stats.STAGE2_IDS), 'untied verdict'
    for version in ac_paths.CRL_VERSIONS:
        row = doc['rows_not_picked']['crl_versions_test'][version]
        assert row['map_picked'] == stages['s3']['rows']['affection']['crl_' + version]['map_picked'], version
    assert doc['percept_core']['ami_own'] == stages['s2']['rows']['percept']['ami_own'], 'own reading'


def test_stage4_writes_rankings_and_figure_from_keyed_impressions_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(ac_paths, 'OUTPUTS', tmp_path / 'out')
    monkeypatch.setattr(ac_paths, 'SMOKE', 16)
    monkeypatch.setattr(ac_paths, 'IMPRESSIONS_STORE', tmp_path / 'store')
    monkeypatch.setattr(ac_paths, 'IMPRESSIONS_DIR', tmp_path / 'impressions')
    media = ac_paths.IMPRESSIONS_DIR / 'media'
    media.mkdir(parents=True)
    images = []
    for i in range(11):
        uuid = f'{i:08x}-0000-0000-0000-000000000000'
        for suffix in (('_cap2_0', '', '_cap1_0') if i == 0 else ('_cap2_0', '_cap1_0') if i == 1 else ('',)):
            name = uuid + suffix + '.png'
            Image.new('RGB', (4, 4), color=(i * 20, 10, 20)).save(media / name)
            images.append(dict(image=name, uuid=uuid, has_overlay=bool(suffix)))
    images.reverse()
    names = [r['image'] for r in images]
    ac_run.write_json(ac_paths.IMPRESSIONS_STORE / 'index.json',
                      dict(dataset='impressions', dtype='float16', dim=2048, n_images=len(images), images=images))
    raw = ac_data._unit(np.random.default_rng(21).normal(size=(len(images), 2048))).astype(np.float16)
    np.save(ac_paths.IMPRESSIONS_STORE / 'img.npy', raw)
    exp = ac_paths.OUTPUTS / 'exports/cstar/impressions/emotion__s0'
    exported = raw[::-1].copy()
    ac_run.write_json(exp.with_suffix('.json'), dict(smoke=True, split='impressions', seed=0, aspect='emotion',
                      keys=names[::-1], keys_sha256=rs_vh.names_sha(names[::-1]), array_sha256=rs_vh.array_sha(exported)))
    np.savez(exp.with_suffix('.npz'), img=exported)
    doc = ac_diag.stage4(ARGS)
    assert doc['n_uuids'] == 11 and doc['n_queries'] == 6 and doc['n_gallery'] == 5, 'UUID split'
    assert doc['n_uuids_with_plain_file'] == 10 and doc['n_uuids_overlay_only'] == 1, 'overlay choices'
    assert doc['queries'][0]['file'].endswith('000000000000.png'), 'prefer plain image'
    assert doc['queries'][1]['file'].endswith('_cap1_0.png'), 'first overlay in file order'
    for q in doc['queries']:
        assert q['cstar'] == q['plain'], 'both joins recover the same rows despite reversed export order'
        assert len(q['cstar']) == 5 and q['uuid'] not in {hit['uuid'] for hit in q['cstar']}, 'gallery excludes queries'
        assert [h['cos'] for h in q['cstar']] == sorted((h['cos'] for h in q['cstar']), reverse=True), 'cosine ranking'
    assert json.loads((ac_paths.OUTPUTS / 'stage4/impressions.json').read_text()) == doc, 'rankings saved'
    with Image.open(ac_paths.OUTPUTS / 'stage4/impressions.png') as figure:
        assert figure.format == 'PNG' and min(figure.size) > 100, 'figure rendered'
