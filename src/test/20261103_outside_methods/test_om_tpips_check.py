"""Synthetic store audit: all probe rows joined by key, cosine failures, active chunks and package changes."""
import json
import os

import numpy as np
import pytest

import om_paths as P
import om_tpips as tp
import om_tpips_check as check


@pytest.fixture
def audit(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, 'WIDTH', 4)
    keys = [f'probe-{i}' for i in range(48)]
    vectors = np.random.default_rng(8).normal(size=(48, 4))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
    probe, store = tmp_path / 'probe', tmp_path / 'store'
    for factor in tp.FACTORS:
        root = probe / factor
        root.mkdir(parents=True)
        np.save(root / 'emb.npy', vectors)
        (root / 'keys.json').write_text(json.dumps(keys))
        (root / 'meta.json').write_text(json.dumps(dict(batches=[1], fake=False, sha256=P.sha256_file(root / 'emb.npy'))))
        for i in range(6):
            (store / factor / 'shards' / f'{i}of6' / 'chunks').mkdir(parents=True)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(set='mmart', keys=keys, order=list(range(48)))))
    env = tmp_path / 'env'
    site = env / 'lib/python3.11/site-packages'
    site.mkdir(parents=True)

    def run():
        return check.check({'mmart': store}, probe, manifest, [], '2026-11-04T00:00:00+00:00',
                           environments=[env], now=2_000_000_000)

    def finish():
        for factor in tp.FACTORS:
            for i in range(6):
                root = store / factor / 'shards' / f'{i}of6'
                indices = list(range(i, 48, 6))[::-1]    # store order differs from reference
                np.save(root / 'emb.npy', vectors[indices])
                (root / 'keys.json').write_text(json.dumps([keys[j] for j in indices]))
                (root / 'meta.json').write_text('{}')

    return store, site, run, finish


def test_empty_running_shards_are_pending(audit):
    _, _, run, _ = audit
    report = run()
    assert report['pass'] and not report['complete']
    assert len(report['rows']) == 18
    assert all(r['n'] == 0 and r['expected'] == 8 and r['min_cos'] is None for r in report['rows'])


def test_finished_shards_join_all_48_probe_keys(audit):
    _, _, run, finish = audit
    finish()
    report = run()
    assert report['pass'] and report['complete']
    assert report['probe_n'] == 48 and report['probe_rows_by_shard'] == {f'{i}of6': 8 for i in range(6)}
    assert all(r['n'] == 8 and r['min_cos'] == pytest.approx(1) for r in report['rows'])


def test_one_changed_probe_row_fails(audit):
    store, _, run, finish = audit
    finish()
    path = store / 'emotion/shards/0of6/emb.npy'
    x = np.load(path)
    x[0] *= -1
    np.save(path, x)
    report = run()
    assert not report['pass']
    bad = [r for r in report['rows'] if not r['pass']]
    assert len(bad) == 1 and bad[0]['min_cos'] == pytest.approx(-1)


def test_finished_shard_cannot_omit_probe_keys(audit):
    store, _, run, finish = audit
    finish()
    root = store / 'emotion/shards/0of6'
    keys = json.loads((root / 'keys.json').read_text())[:-1]
    (root / 'keys.json').write_text(json.dumps(keys))
    np.save(root / 'emb.npy', np.load(root / 'emb.npy')[:-1])
    report = run()
    assert not report['pass']
    assert any('finished shard lacks 1' in p for p in report['problems'])


def test_new_dist_info_fails_and_is_listed(audit):
    _, site, run, _ = audit
    new = site / 'changed-1.dist-info'
    old = site / 'old-1.dist-info'
    new.mkdir()
    old.mkdir()
    os.utime(new, (2_000_000_000, 2_000_000_000))
    os.utime(old, (1, 1))
    report = run()
    assert not report['pass']
    changes = [p for rows in report['package_changes'].values() for p in rows]
    assert [p['path'] for p in changes] == [str(new)]


def test_active_unfinished_chunk_is_reported_without_failure(audit):
    store, _, run, _ = audit
    path = store / 'emotion/shards/0of6/chunks/0000000.npy'
    path.write_bytes(b'in progress')
    os.utime(path, (2_000_000_000, 2_000_000_000))
    report = run()
    assert report['pass']
    assert report['rows'][0]['active_chunks'] == ['0000000.npy']


def test_stable_partial_chunks_compare_finished_probe_rows(audit):
    store, _, run, _ = audit
    path = store / 'emotion/shards/0of6/chunks/0000000.npy'
    np.save(path, np.array([[1, 0, 0, 0]], np.float32))
    path.with_suffix('.json').write_text('["not-a-probe-row"]')
    report = run()
    assert report['pass'] and report['rows'][0]['n'] == 0
    # Copy a real reference row into an unfinished shard's stable chunk.
    reference = check.read(store.parent / 'probe/emotion/keys.json')
    x = np.load(store.parent / 'probe/emotion/emb.npy')
    np.save(path, x[:1])
    path.with_suffix('.json').write_text(json.dumps(reference[:1]))
    report = run()
    assert report['pass'] and report['rows'][0]['n'] == 1 and report['rows'][0]['state'] == 'pending'


def test_start_comes_from_first_run_log_line(tmp_path):
    log = tmp_path / 'run.log'
    log.write_text('shard 1 start 2026-11-04 04:30\nshard 0 start 2026-11-04 01:38\n')
    stamp, source = check.shard_start({}, [log])
    assert stamp == check.datetime.fromisoformat('2026-11-04T00:38:00+00:00').timestamp()
    assert 'shard 0 start' in source
