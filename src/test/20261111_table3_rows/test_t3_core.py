"""Unit tests (synthetic data, seconds) for t3_core and t3_inputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t3_core as C  # noqa: E402
import t3_inputs as I  # noqa: E402

N, D = 200, 40


def unit(x):
    x = x.astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def make(seed_count=5, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    labels = {}
    for a, k in zip(C.ASPECTS, (3, 4, 2)):
        y = np.arange(N) % k
        rng.shuffle(y)
        y[rng.random(N) < 0.1] = -1
        labels[a] = y.astype(np.int64)
    centers = rng.normal(size=(4, D))
    base = {a: unit(centers[np.maximum(labels[a], 0)] + rng.normal(size=(N, D)) * 2.0) for a in C.ASPECTS}
    seeded = {s: {a: unit(base[a] + rng.normal(size=(N, D)) * 0.3) for a in C.ASPECTS} for s in range(seed_count)}
    return labels, base, seeded


UNIT = {a: 0.01 for a in C.ASPECTS}


def test_parse_seeds():
    assert C.parse_seeds('0,1,3') == (0, 1, 3)
    assert C.parse_seeds('4') == (4,)
    for bad in ('', '1,0', '0,0', '5', '0,x'):
        with pytest.raises((AssertionError, ValueError)):
            C.parse_seeds(bad)


def test_seeded_pick_equals_arm_table_on_five_seeds():
    labels, _, seeded = make()
    xs = {(a, s): seeded[s][a] for s in range(5) for a in C.ASPECTS}
    table, ro = C.ac_select.arm_table(xs, labels, UNIT, workers=2)
    mine = C.pick_readout(lambda a, s: seeded[s][a], labels, UNIT, C.ALL_SEEDS, workers=2)
    assert mine['pick'] == C.rt_readout.name(ro)
    assert list(mine['table']) == list(table)
    for name in table:
        for a in C.ASPECTS:
            assert abs(mine['table'][name]['map'][a] - table[name]['map'][a]) <= 1e-12
        assert abs(mine['table'][name]['score'] - table[name]['score']) <= 1e-12
        assert mine['table'][name]['kept'] == table[name]['kept']


def test_unseeded_pick_equals_seeded_with_one_seed():
    labels, _, seeded = make()
    a_ = C.pick_readout(lambda a, s: seeded[0][a], labels, UNIT, None, workers=2)
    b_ = C.pick_readout(lambda a, s: seeded[0][a], labels, UNIT, (0,), workers=2)
    assert a_['pick'] == b_['pick'] and a_['table'].keys() == b_['table'].keys()
    for name in a_['table']:
        for a in C.ASPECTS:
            assert abs(a_['table'][name]['map'][a] - b_['table'][name]['map'][a]) <= 1e-12


def test_seed_subset_pick_uses_mean_over_kept_seeds():
    labels, _, seeded = make()
    res = C.pick_readout(lambda a, s: seeded[s][a], labels, UNIT, (0, 2), workers=2)
    assert set(res['per_seed_map']) == {'0', '2'}
    for name, row in res['table'].items():
        for a in C.ASPECTS:
            want = np.mean([res['per_seed_map'][s][name][a] for s in ('0', '2')])
            assert abs(row['map'][a] - want) <= 1e-12


def test_score_seeded_matches_arm_ap_on_five_seeds_and_subset_is_mean():
    labels, _, seeded = make()
    val = lambda a, s: seeded[s][a]
    test = lambda a, s: seeded[(s + 1) % 5][a]            # any other rows of the same shape
    out = C.score_test(val, test, 'center+k10', labels, C.ALL_SEEDS, workers=2)
    for a in C.ASPECTS:
        ref = C.ns_ret.arm_ap(lambda s: val(a, s), lambda s: test(a, s), 'center+k10', labels[a], C.ALL_SEEDS, 2)
        assert abs(out[a]['map'] - ref['point']) <= 1e-15
        assert [out[a]['per_seed'][str(s)] for s in range(5)] == ref['per_seed']
        assert out[a]['n_queries'] == len(ref['rows'])
    sub = C.score_test(val, test, 'center+k10', labels, (1, 4), workers=2)
    for a in C.ASPECTS:
        assert abs(sub[a]['map'] - np.mean([out[a]['per_seed']['1'], out[a]['per_seed']['4']])) <= 1e-15


def test_score_unseeded_matches_arm_ap():
    labels, base, _ = make()
    out = C.score_test(lambda a, s: base[a], lambda a, s: base[a], 'none+k10', labels, None, workers=2)
    for a in C.ASPECTS:
        ref = C.ns_ret.arm_ap(lambda s: base[a], lambda s: base[a], 'none+k10', labels[a], None, 2)
        assert out[a]['map'] == ref['point'] and out[a]['per_seed'] is None


def test_align_reorders_by_name_and_rejects_other_keys():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = I.align(['d', 'a', 'c', 'b'], x, ['a', 'b', 'c', 'd'], 't')
    assert np.array_equal(y, x[[1, 3, 2, 0]])
    with pytest.raises(AssertionError):
        I.align(['a', 'b', 'c', 'z'], x, ['a', 'b', 'c', 'd'], 't')
    with pytest.raises(AssertionError):
        I.align(['a', 'a', 'c', 'd'], x, ['a', 'b', 'c', 'd'], 't')


def _write_store(root, aspect, names, emb, n=None, split=None):
    d = Path(root) / aspect
    d.mkdir(parents=True)
    np.save(d / 'emb.npy', emb)
    (d / 'keys.json').write_text(json.dumps(names))
    meta = dict(aspect=aspect, n=len(names) if n is None else n, limit=None, emb_sha256=C.file_sha(d / 'emb.npy'),
                instruction=f'inst {aspect}', model='m', revision='r')
    if split:
        meta['split'] = split
    (d / 'meta.json').write_text(json.dumps(meta))


def test_vlm2vec_store_reorders_and_hashes(tmp_path, monkeypatch):
    names = [f'p{i:03d}' for i in range(20)]
    monkeypatch.setattr(C, 'keys', lambda split: list(names))
    monkeypatch.setattr(I, 'VLM2VEC_DIM', 8)
    rng = np.random.default_rng(1)
    emb = unit(rng.normal(size=(20, 8)))
    perm = rng.permutation(20)
    _write_store(tmp_path, 'emotion', [names[i] for i in perm], emb[perm])
    st = I.Vlm2VecStore(tmp_path, 'val', 20)
    x = st.rows('emotion')
    assert np.array_equal(x, emb)
    prov = st.provenance()['emotion']
    assert prov['reordered'] and len(prov['emb_sha256']) == 64
    with pytest.raises(AssertionError, match='missing'):
        st.rows('style')


def test_vlm2vec_store_rejects_bad_hash_and_count(tmp_path, monkeypatch):
    names = [f'p{i:03d}' for i in range(20)]
    monkeypatch.setattr(C, 'keys', lambda split: list(names))
    monkeypatch.setattr(I, 'VLM2VEC_DIM', 8)
    emb = unit(np.random.default_rng(2).normal(size=(20, 8)))
    _write_store(tmp_path, 'emotion', names, emb)
    meta = json.loads((tmp_path / 'emotion' / 'meta.json').read_text())
    meta['emb_sha256'] = '0' * 64
    (tmp_path / 'emotion' / 'meta.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match='hash'):
        I.Vlm2VecStore(tmp_path, 'val', 20).rows('emotion')
    _write_store(tmp_path, 'style', names, emb, n=19)
    with pytest.raises(AssertionError):
        I.Vlm2VecStore(tmp_path, 'val', 20).rows('style')


def test_percept_files_unit_rows_and_names(tmp_path, monkeypatch):
    names = [f'p{i:03d}' for i in range(20)]
    monkeypatch.setattr(C, 'keys', lambda split: list(names))
    rng = np.random.default_rng(3)
    prob = rng.dirichlet(np.ones(40), size=20).astype(np.float32)
    perm = rng.permutation(20)
    np.savez(tmp_path / 'L1__s3.npz', prob=prob[perm], names=np.array([names[i] for i in perm]))
    pf = I.PerceptFiles(tmp_path, 'L1__s{seed}.npz', 'val', (3,), 20)
    x = pf.rows(3)
    assert np.array_equal(x, C.ns_ret._unit(prob))
    assert pf.provenance()['3']['reordered']
    np.savez(tmp_path / 'L1__s4.npz', prob=prob, names=np.array(names[:-1] + ['zzz']))
    with pytest.raises(AssertionError):
        I.PerceptFiles(tmp_path, 'L1__s{seed}.npz', 'val', (4,), 20).rows(4)
    with pytest.raises(AssertionError, match='missing'):
        pf.rows(0)


def test_write_json_rejects_nan(tmp_path):
    C.write_json(tmp_path / 'a.json', {'x': 1.0})
    assert json.loads((tmp_path / 'a.json').read_text()) == {'x': 1.0}
    with pytest.raises(ValueError):
        C.write_json(tmp_path / 'b.json', {'x': float('nan')})
