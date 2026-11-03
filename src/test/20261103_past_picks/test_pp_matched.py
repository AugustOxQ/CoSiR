"""Tests of pp_matched (r19 ticket 05): the reading code on a tiny array, the D23 tolerance check, the CPU replay's
tolerance, the exports and key joins in shuffled order, the output-root guard, the settings check.

Tiny arrays and tmp folders test the logic; the `real` tests read the stored 10-25 / 10-26 outputs (read only)."""
import json
import hashlib

import numpy as np
import pytest

import pp_paths as PP  # noqa: F401  (first: sets the older rounds' output roots)
import pp_register  # noqa: F401
import pp_matched as M


# ---------------------------------------------------------------------------------------------- readings
def topic_prob(n_groups=20, per=4, seed=0):
    """n_groups one-hot-ish groups (each with its own dominant topic), `per` rows each; returns prob [n, 40], group id."""
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.0, 0.001, size=(n_groups * per, PP.N_TOPICS)).astype(np.float32)
    group = np.repeat(np.arange(n_groups), per)
    prob[np.arange(len(group)), group] = 0.9 + rng.uniform(0, 0.02, size=len(group))
    return prob, group


def test_readings_own_is_argmax_and_r20_has_three_states():
    prob, group = topic_prob()
    got = M.readings(prob)
    assert got['own'].shape == (1, len(prob)) and got['r20'].shape == (3, len(prob))
    assert np.array_equal(got['own'][0], prob.argmax(axis=1))
    assert np.array_equal(got['own'][0], group)


def test_readings_r20_recovers_twenty_groups_and_means_score_one():
    prob, group = topic_prob()
    got = M.readings(prob)
    for state in range(3):                       # 20 well-separated groups: every state finds (nearly) all of them
        assert len(np.unique(got['r20'][state])) == 20
    means = M.reading_means({0: got}, group)
    assert means['r20'] > 0.95 and means['own'] == pytest.approx(1.0)


def test_reading_means_masks_negative_labels_and_averages_seeds():
    prob, group = topic_prob()
    got = M.readings(prob)
    labels = group.copy()
    labels[:8] = -1                              # masked rows do not count
    full = M.reading_means({0: got}, labels)
    assert full['own'] == pytest.approx(1.0)
    bad = {'own': (got['own'] + 1) % 40, 'r20': got['r20'][:, ::-1] * 0}   # a constant partition: AMI 0
    two = M.reading_means({0: got, 1: {'own': got['own'], 'r20': bad['r20']}}, group)
    assert two['own'] == pytest.approx(1.0)
    assert two['r20'] == pytest.approx(M.reading_means({0: got}, group)['r20'] / 2)   # mean over the two seeds


def test_reading_table_is_per_aspect():
    prob, group = topic_prob()
    other = (group % 2).astype(np.int64)
    table = M.reading_table({0: prob}, {'emotion': group, 'style': other, 'genre': group})
    assert set(table) == {'emotion', 'style', 'genre'} and set(table['emotion']) == {'r20', 'own'}
    assert table['emotion']['own'] == pytest.approx(1.0) and table['style']['own'] < 1.0


# ---------------------------------------------------------------------------------------------- D23
def stored_doc(seed0=0.5, lo=0.49, hi=0.52):
    return {a: {r: dict(seed0=seed0, min=lo, max=hi) for r in ('r20', 'own')} for a in PP.ASPECTS}


def rerun_doc(value):
    return {a: {r: value for r in ('r20', 'own')} for a in PP.ASPECTS}


def test_d23_inside_and_at_the_edge_passes():
    res = M.d23_check(rerun_doc(0.5 + 0.03), stored_doc())          # tolerance 0.03: the difference equals it
    assert res['pass'] is True and res['items']['emotion/r20']['tolerance'] == pytest.approx(0.03)
    assert M.d23_check(rerun_doc(0.5), stored_doc())['pass'] is True


def test_d23_outside_the_arms_range_fails_and_records_the_difference():
    res = M.d23_check(rerun_doc(0.5 + 0.0301), stored_doc())
    assert res['pass'] is False
    item = res['items']['genre/own']
    assert item['ok'] is False and item['abs_diff'] == pytest.approx(0.0301)
    assert res['max_abs_diff'] == pytest.approx(0.0301)


def test_d23_one_failing_item_fails_the_gate():
    rerun = rerun_doc(0.5)
    rerun['style']['own'] = 0.5 - 0.031                              # below, by more than the range
    res = M.d23_check(rerun, stored_doc())
    assert res['pass'] is False
    assert [k for k, v in res['items'].items() if not v['ok']] == ['style/own']


def test_d23_is_two_sided_and_asserts_complete_inputs():
    assert M.d23_check(rerun_doc(0.5 - 0.0301), stored_doc())['pass'] is False
    short = rerun_doc(0.5)
    del short['genre']['r20']
    with pytest.raises(AssertionError):
        M.d23_check(short, stored_doc())
    with pytest.raises(AssertionError):                              # seed 0 must lie in the arm's own range
        M.d23_check(rerun_doc(0.5), stored_doc(seed0=0.6))
    nan = rerun_doc(0.5)
    nan['emotion']['own'] = float('nan')
    with pytest.raises(AssertionError):
        M.d23_check(nan, stored_doc())


def test_stored_sample_readings_real():
    got = M.stored_sample_readings()
    assert got['emotion']['r20']['seed0'] == pytest.approx(0.06654731820756389, abs=1e-12)
    assert got['genre']['own']['seed0'] == pytest.approx(0.39113782522734697, abs=1e-12)
    tol = {(a, r): v['max'] - v['min'] for a, d in got.items() for r, v in d.items()}
    assert tol[('emotion', 'r20')] == pytest.approx(0.0023940917527821015, abs=1e-12)
    assert tol[('style', 'own')] == pytest.approx(0.006070583508229227, abs=1e-12)
    assert tol[('genre', 'r20')] == pytest.approx(0.007154730874510462, abs=1e-12)
    assert all(v['min'] <= v['seed0'] <= v['max'] for d in got.values() for v in d.values())


# ---------------------------------------------------------------------------------------------- CPU replay
def test_replay_probs_tolerance_is_1e_5_max_abs():
    stored = np.random.default_rng(0).uniform(0.01, 0.99, size=(30, 40)).astype(np.float32)
    ok = M.replay_probs({0: stored + np.float32(0.9e-5)}, {0: stored})
    assert ok['passed'] is True and ok['failed_seeds'] == []
    bad = stored.copy()
    bad[7, 3] += np.float32(2e-5)                                    # one cell off by more than 1e-5
    res = M.replay_probs({0: stored, 1: bad}, {0: stored, 1: stored})
    assert res['passed'] is False and res['failed_seeds'] == [1]
    assert res['max_abs_diff'] == pytest.approx(2e-5, rel=1e-3)
    with pytest.raises(AssertionError):                              # seeds must match
        M.replay_probs({0: stored}, {1: stored})


def test_settings_diff_allows_only_the_input_and_the_commit():
    a = {'lr': 0.003, 'epochs': 100, 'input': 'clip_b16 default image', 'input_dim': 512, 'source_commit': 'x', 'seed': 0}
    b = dict(a, input='dinov2_large default image', input_dim=1024, source_commit='y')
    assert M.settings_diff(a, b) == []
    assert M.settings_diff(a, dict(b, lr=0.01)) == ['lr']
    assert M.settings_diff(a, dict(b, epochs=50, seed=1)) == ['epochs', 'seed']


# ---------------------------------------------------------------------------------------------- exports
def fake_keys(n, prefix='k'):
    return PP.order([f'{prefix}{i}.jpg' for i in range(n)])


@pytest.fixture
def fake_mapper(monkeypatch):
    """A mapper whose weights hash is fixed; the exports bind to it."""
    monkeypatch.setattr(M, 'mapper_weights_sha256', lambda enc, seed: f'sha-{enc}-{seed}')


def fake_probs(keys, seed):
    """probabilities that are a function of the key, so a mis-joined row shows."""
    rows = []
    for k in keys:
        h = int(hashlib.sha256(f'{k}{seed}'.encode()).hexdigest()[:8], 16)
        r = np.random.default_rng(h).uniform(0.001, 0.999, size=PP.N_TOPICS).astype(np.float32)
        rows.append(r)
    return np.stack(rows)


def write_fake(root, set_name, enc, keys):
    for s in PP.SEEDS:
        M.write_export(root, set_name, enc, s, keys, fake_probs(keys, s), mapper='m', mapper_sha256=f'sha-{enc}-{s}',
                       view=enc)


def test_load_probs_joins_by_key_in_shuffled_order(tmp_path, fake_mapper):
    keys = fake_keys(60)
    write_fake(tmp_path, 'semart', 'siglip2', keys)
    asked = list(np.random.default_rng(1).permutation(keys))
    got = M.load_probs('semart', 'siglip2', 3, asked, root=tmp_path)
    assert np.array_equal(got, fake_probs(asked, 3))
    subset = asked[:17]                                              # a subset is fine
    assert np.array_equal(M.load_probs('semart', 'siglip2', 0, subset, root=tmp_path), fake_probs(subset, 0))


def test_load_probs_refuses_missing_key_tampering_and_stale_mapper(tmp_path, fake_mapper, monkeypatch):
    keys = fake_keys(30)
    write_fake(tmp_path, 'mmart_new', 'clip_b32', keys)
    with pytest.raises(AssertionError, match='not exported'):
        M.load_probs('mmart_new', 'clip_b32', 0, keys + ['absent.jpg'], root=tmp_path)
    path = M.export_path(tmp_path, 'mmart_new', 'clip_b32', 2)
    meta = json.loads(path.with_suffix('.json').read_text())
    meta['sha256'] = '0' * 64
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match='sha256'):
        M.load_probs('mmart_new', 'clip_b32', 2, keys, root=tmp_path)
    monkeypatch.setattr(M, 'mapper_weights_sha256', lambda enc, seed: 'other')
    with pytest.raises(AssertionError, match='mapper'):
        M.load_probs('mmart_new', 'clip_b32', 0, keys, root=tmp_path)


def test_write_export_asserts_shape_range_and_unique_keys(tmp_path):
    keys = fake_keys(5)
    good = fake_probs(keys, 0)
    kw = dict(mapper='m', mapper_sha256='s', view='x')
    with pytest.raises(AssertionError):
        M.write_export(tmp_path, 'semart', 'qwen', 0, keys, good.astype(np.float64), **kw)
    with pytest.raises(AssertionError):
        M.write_export(tmp_path, 'semart', 'qwen', 0, keys, good[:4], **kw)
    with pytest.raises(AssertionError):
        M.write_export(tmp_path, 'semart', 'qwen', 0, keys, good * 2, **kw)
    with pytest.raises(AssertionError):
        M.write_export(tmp_path, 'semart', 'qwen', 0, keys[:4] + keys[:1], good, **kw)
    with pytest.raises(AssertionError):                              # the set and the encoder must be a listed pair
        M.write_export(tmp_path, 'mmart_new', 'qwen', 0, keys, good, **kw)
    with pytest.raises(AssertionError):
        M.write_export(tmp_path, 'artelingo', 'qwen', 0, keys, good, **kw)


def test_partitions_follow_the_requested_rows_in_shuffled_order(tmp_path, fake_mapper):
    keys = fake_keys(80)
    write_fake(tmp_path, 'semart', 'dinov2_large', keys)
    asked = list(np.random.default_rng(2).permutation(keys))
    out = M.partitions('semart', 'dinov2_large', asked, root=tmp_path)
    assert sorted(out) == list(PP.SEEDS)
    for s, got in out.items():
        assert got['own'].shape == (1, 80) and got['own'].dtype == np.int32
        assert got['r20'].shape == (3, 80) and got['r20'].dtype == np.int32
        assert np.array_equal(got['own'][0], fake_probs(asked, s).argmax(axis=1))     # own = argmax, row by row


def test_export_writes_one_file_per_seed_through_the_loader(tmp_path, fake_mapper, monkeypatch):
    keys = fake_keys(40)
    calls = []

    def loader(dataset, enc, ks):
        calls.append((dataset, enc, len(ks)))
        return np.random.default_rng(0).normal(size=(len(ks), PP.WIDTH[enc])).astype(np.float32)

    def infer(path, images):
        assert images.shape == (40, PP.WIDTH['clip_b16'])
        return fake_probs(keys, int(str(path)[-1]))

    monkeypatch.setattr(M, 'mapper_path', lambda enc, seed: f'mapper{seed}')
    paths = M.export('mmart_new', 'clip_b16', root=tmp_path, keys=keys, loader=loader, infer=infer)
    assert calls == [('mmart', 'clip_b16', 40)] and len(paths) == 5 and all(p.is_file() for p in paths)
    assert np.array_equal(M.load_probs('mmart_new', 'clip_b16', 4, keys, root=tmp_path), fake_probs(keys, 4))
    with pytest.raises(AssertionError, match='sha256 order'):        # the evaluation order is sha256 of the key
        M.export('mmart_new', 'clip_b16', root=tmp_path / 'x', keys=sorted(keys), loader=loader, infer=infer)
    with pytest.raises(AssertionError):                              # MMArt's new keys: the new three only
        M.export('mmart_new', 'siglip2', root=tmp_path / 'y', keys=keys, loader=loader, infer=infer)


# ---------------------------------------------------------------------------------------------- guards
def test_output_root_guard_refuses_frozen_rounds(tmp_path):
    for bad in (PP.EG_PART2.parent, PP.EG_PART2, PP.RB_PART2.parent, PP.R14_OUT, PP.R15_OUT, PP.R10_OUT,
                PP.EG_ROUND / 'outputs', PP.MAIN / 'src/test/20261020_claim_test/outputs'):
        with pytest.raises(AssertionError):
            M.check_out_root(bad)
    assert M.check_out_root(tmp_path) == tmp_path


# ---------------------------------------------------------------------------------------------- real stored data
def test_held_probs_real_matches_the_stored_files():
    got = M.held_probs('clip_b32')
    assert sorted(got) == list(PP.SEEDS) and all(v.shape == (12281, 40) and v.dtype == np.float32 for v in got.values())
    with np.load(PP.RB_PART2 / 'full/held/M3/P3__s2.npz') as z:
        assert np.array_equal(got[2], z['prob'])


def test_held_probs_refuses_a_missing_arm(tmp_path, monkeypatch):
    import bs_artclu
    monkeypatch.setitem(bs_artclu.ENC['dinov2_large'], 'root', tmp_path)     # an empty M5 root: fail, never a partial answer
    with pytest.raises(AssertionError):
        M.held_probs('dinov2_large')


def test_held_probs_refuses_a_file_of_another_held_list(tmp_path, monkeypatch):
    import bs_artclu
    monkeypatch.setitem(bs_artclu.ENC['dinov2_large'], 'root', tmp_path)
    folder = tmp_path / 'full/held/M5'
    folder.mkdir(parents=True)
    for s in PP.SEEDS:
        np.savez_compressed(folder / f'P3__s{s}.npz', prob=np.full((12281, 40), 0.5, np.float32))
        (folder / f'P3__s{s}.json').write_text(json.dumps(dict(list_sha256='0' * 64, rows=12281)))
    with pytest.raises(AssertionError, match='held list'):
        M.held_probs('dinov2_large')
