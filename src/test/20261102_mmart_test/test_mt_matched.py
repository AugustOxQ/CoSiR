"""Tests of mt_matched (ticket 03): CPU inference wrap, the 0(iv) checkers, the stage-4 reader, `partitions` row order,
the export file format. Synthetic fixtures; the real-data gate is the CLI `gate-iv` (see the ticket hand-back)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mt_matched as M  # noqa: E402
import mt_paths as P  # noqa: E402

K = 40


def fixture_mapper(tmp_path, device='cuda', width=16, seed=0, hidden=32):
    """A tiny sealed Stage 2 mapper in the CT's format (`base.save_output`), as the trainer wrote the real ones."""
    base = M.base
    config = dict(settings='core', cut='k40', arm='P3', seed=seed, device=device, input='fake default image',
                  input_dim=width, mapper='mlp', hidden_dim=hidden, eval_paintings=7)
    torch.manual_seed(seed)
    module = base.build_mapper(config, K)
    path = tmp_path / f'P3__s{seed}.npz'
    base.save_output(path, dict(eval_prob=np.full((7, K), .5, np.float32)), dict(mapper=module.state_dict()),
                     dict(config=config, K=K))
    return path, module


def images(n=11, width=16, seed=1, scale=3.0):
    """Float32 rows that are NOT unit norm, so the second normalisation matters."""
    return (np.random.default_rng(seed).normal(size=(n, width)) * scale).astype(np.float32)


# ---- inference ------------------------------------------------------------------------------------------------
def test_infer_cpu_accepts_a_cuda_trained_mapper_where_the_trainer_refuses(tmp_path):
    path, module = fixture_mapper(tmp_path, device='cuda')
    x = images()
    with pytest.raises(AssertionError, match='inference device'):
        M.rb_matched.infer(path, x, 'cpu')
    got = M.infer_cpu(path, x)
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)           # the mapper's own second normalisation stays
    with torch.no_grad():
        want = module(torch.from_numpy(unit)).sigmoid().numpy()
    assert got.dtype == np.float32 and got.shape == (len(x), K)
    assert np.array_equal(got, want)


def test_infer_cpu_equals_the_trainers_infer_on_a_cpu_mapper(tmp_path):
    path, _ = fixture_mapper(tmp_path, device='cpu')
    x = images()
    assert np.array_equal(M.infer_cpu(path, x), M.rb_matched.infer(path, x, 'cpu'))


def test_infer_cpu_normalises_twice_not_once(tmp_path):
    path, module = fixture_mapper(tmp_path, device='cuda')
    x = images()
    raw = module(torch.from_numpy(x)).sigmoid().detach().numpy()
    assert not np.allclose(M.infer_cpu(path, x), raw, atol=1e-3)


def test_infer_cpu_refuses_bad_input(tmp_path):
    path, _ = fixture_mapper(tmp_path, device='cuda')
    with pytest.raises(AssertionError):
        M.infer_cpu(path, images(width=15))
    with pytest.raises(AssertionError):
        M.infer_cpu(path, images().astype(np.float64))
    bad = images()
    bad[2, 3] = np.nan
    with pytest.raises(AssertionError):
        M.infer_cpu(path, bad)


# ---- gate (iv)(a) ---------------------------------------------------------------------------------------------
def test_compare_probs_tolerance_is_1e5_max_absolute():
    stored = np.random.default_rng(0).uniform(.1, .9, size=(50, K)).astype(np.float32)
    near, far = stored.copy(), stored.copy()
    near[3, 4] += 9e-6
    far[3, 4] += 2e-5
    far[10, 1] -= 1e-6
    assert M.compare_probs(near, stored)['ok']
    res = M.compare_probs(far, stored)
    assert not res['ok'] and abs(res['max_abs_diff'] - 2e-5) < 2e-6
    with pytest.raises(AssertionError):
        M.compare_probs(stored[:-1], stored)


# ---- readings -------------------------------------------------------------------------------------------------
def test_readings_cluster_the_l2_normalised_probabilities_not_the_raw_ones():
    """r20 is `ct_score.partitions(l2(p), 20)`, the route `output_partitions(p, 'percept')` takes; raw p differs."""
    rng = np.random.default_rng(5)
    scale = rng.uniform(.05, 1, size=(300, 1))                       # rows of very different length, similar direction
    p = np.clip(scale * rng.uniform(.2, 1, size=(300, K)), 0, 1).astype(np.float32)
    got = M.readings(p)['r20']
    want = np.asarray(M.ct_score.partitions(M.ct_score.l2(p), 20))
    raw = np.asarray(M.ct_score.partitions(p, 20))
    assert np.array_equal(got, want) and np.array_equal(got, M.ct_score.output_partitions(p, 'percept')['r20'])
    assert not np.array_equal(got, raw)


# ---- gate (iv)(b) ---------------------------------------------------------------------------------------------
def readings_fixture(n=240, seed=0):
    rng = np.random.default_rng(seed)
    prob = (1 / (1 + np.exp(-rng.normal(size=(n, K)) * 2))).astype(np.float32)
    parts = M.readings(prob)
    labels = {a: rng.integers(-1, 4, size=n).astype(np.int64) for a in P.ASPECTS}
    return prob, parts, labels


def stage4_doc(matched, means):
    """A `reported_stage4.json` in the stored format: emotion in the seed spread, style and genre in b_matched."""
    return {'blocks': {'KX': {'aspects': {
        'emotion': {'point_spread': {matched: {r: {'mean': means['emotion'][r]} for r in ('r20', 'own')}}},
        'style': {'margins': {'b_matched': {'A_e': {f'{matched}-{r}': means['style'][r] for r in ('r20', 'own')}}}},
        'genre': {'margins': {'b_matched': {'A_e': {f'{matched}-{r}': means['genre'][r] for r in ('r20', 'own')}}}},
    }}}}


def test_reported_readings_reads_the_stored_format():
    means = {a: {'r20': .1 + i / 100, 'own': .2 + i / 100} for i, a in enumerate(P.ASPECTS)}
    doc = stage4_doc('M2', means)
    got = M.reported_readings(doc, 'KX', 'M2')
    assert got == means
    with pytest.raises(KeyError):
        M.reported_readings(doc, 'KX', 'M7')


def test_gate_b_passes_on_the_stored_readings_and_catches_each_kind_of_mismatch():
    prob, parts, labels = readings_fixture()
    probs, stored = {0: prob}, {0: parts}
    means = {a: M.reading_means(stored, labels[a]) for a in P.ASPECTS}
    assert M.gate_b(probs, stored, labels, means)['passed']

    changed = {0: {'r20': parts['r20'].copy(), 'own': parts['own']}}      # one row moved in one state
    changed[0]['r20'][1, 5] = (changed[0]['r20'][1, 5] + 1) % 20
    res = M.gate_b(probs, changed, labels, means)
    assert not res['passed'] and not res['partitions']['0/r20']['equal'] and res['partitions']['0/own']['equal']

    off = {a: {r: v + 1e-8 for r, v in means[a].items()} for a in P.ASPECTS}
    res = M.gate_b(probs, stored, labels, off)
    assert not res['passed'] and res['partitions_equal'] and not res['readings_ok']
    tiny = {a: {r: v + 1e-10 for r, v in means[a].items()} for a in P.ASPECTS}
    assert M.gate_b(probs, stored, labels, tiny)['passed']


def test_reading_means_is_the_mean_over_seeds_and_states_of_masked_ami():
    prob, parts, labels = readings_fixture(seed=3)
    prob2, parts2, _ = readings_fixture(seed=4)
    two = M.reading_means({0: parts, 1: parts2}, labels['style'])
    one_a = M.reading_means({0: parts}, labels['style'])
    one_b = M.reading_means({1: parts2}, labels['style'])
    for r in ('r20', 'own'):
        assert abs(two[r] - (one_a[r] + one_b[r]) / 2) < 1e-12
    mask = labels['style'] >= 0
    assert mask.sum() < len(mask)                                    # the mask is exercised
    from sklearn.metrics import adjusted_mutual_info_score as ami
    want = np.mean([ami(labels['style'][mask], p[mask]) for p in parts['r20']])
    assert abs(one_a['r20'] - want) < 1e-12


# ---- exports and `partitions` ---------------------------------------------------------------------------------
def fake_export(root, enc, monkeypatch, n=90, seeds=(0, 1, 2, 3, 4)):
    """Exports for `enc` with a fake current mapper sidecar per seed (`mapper_path` redirected under `root`)."""
    mdir = Path(root) / 'mappers' / enc
    mdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(M, 'mapper_path', lambda e, s: Path(root) / 'mappers' / e / f'P3__s{s}.npz')
    rng = np.random.default_rng(7)
    keys = [f'Style/p{i:03d}.jpg' for i in range(n)]
    probs = {}
    for s in seeds:
        probs[s] = (1 / (1 + np.exp(-rng.normal(size=(n, K)) * 2))).astype(np.float32)
        mp = M.mapper_path(enc, s)
        mp.with_suffix('.json').write_text(json.dumps({'weights_sha256': f'w-{enc}-{s}'}))
        M.write_export(root, enc, s, keys, probs[s], mapper=mp, mapper_sha256=f'w-{enc}-{s}', view=P.DEFAULT_VIEW[enc])
    return keys, probs


def test_partitions_row_order_follows_keys(tmp_path, monkeypatch):
    keys, probs = fake_export(tmp_path, 'siglip2', monkeypatch)
    pick = [keys[i] for i in (17, 3, 60, 41, 2, 88, 9, 70, 31, 55, 12, 80, 25, 6, 47, 66, 1, 90 - 1)]
    pick = pick + [keys[i] for i in range(20, 60)]
    out = M.partitions('siglip2', pick, root=tmp_path)
    idx = [keys.index(k) for k in pick]
    assert sorted(out) == [0, 1, 2, 3, 4]
    for s in range(5):
        assert out[s]['own'].dtype == np.int32 and out[s]['own'].shape == (len(pick),)
        assert out[s]['k20'].dtype == np.int32 and out[s]['k20'].shape == (3, len(pick))
        assert np.array_equal(out[s]['own'], probs[s][idx].argmax(axis=1))
        want = M.readings(probs[s][idx])                              # fitted on exactly these rows, in this order
        assert np.array_equal(out[s]['k20'], want['r20'])
    rev = M.partitions('siglip2', pick[::-1], root=tmp_path)
    assert np.array_equal(rev[0]['own'], out[0]['own'][::-1])


def test_partitions_refuses_missing_keys_and_changed_files(tmp_path, monkeypatch):
    keys, probs = fake_export(tmp_path, 'clip_l14', monkeypatch)
    with pytest.raises(AssertionError):
        M.partitions('clip_l14', keys[:5] + ['Style/absent.jpg'], root=tmp_path)
    f = tmp_path / 'exports/new/clip_l14/matched/probs__s2.npz'
    np.savez_compressed(f, keys=np.asarray(keys), probs=probs[2] * np.float32(.5))
    with pytest.raises(AssertionError, match='sha256'):
        M.partitions('clip_l14', keys[:30], root=tmp_path)


def rewrite_sidecar(root, enc, which, changes):
    f = Path(root) / f'exports/new/{enc}/matched/probs__s{which}.json'
    meta = json.loads(f.read_text())
    meta.update(changes)
    f.write_text(json.dumps(meta))


@pytest.mark.parametrize('field, value', [('encoder', 'siglip1'), ('seed', 3), ('view', 'clip_l14'),
                                          ('mapper_sha256', 'some-other-mapper')])
def test_load_probs_refuses_an_export_from_another_mapper_seed_encoder_or_view(tmp_path, monkeypatch, field, value):
    keys, _ = fake_export(tmp_path, 'siglip2', monkeypatch)
    assert M.load_probs('siglip2', 2, keys[:10], root=tmp_path).shape == (10, K)          # untouched: loads
    rewrite_sidecar(tmp_path, 'siglip2', 2, {field: value})
    with pytest.raises(AssertionError, match=field):
        M.load_probs('siglip2', 2, keys[:10], root=tmp_path)


def test_load_probs_refuses_when_the_current_mapper_changed(tmp_path, monkeypatch):
    keys, _ = fake_export(tmp_path, 'qwen', monkeypatch)
    M.mapper_path('qwen', 1).with_suffix('.json').write_text(json.dumps({'weights_sha256': 'retrained'}))
    with pytest.raises(AssertionError, match='mapper_sha256'):
        M.load_probs('qwen', 1, keys[:10], root=tmp_path)
    assert M.load_probs('qwen', 0, keys[:10], root=tmp_path).shape == (10, K)


def test_export_sidecar_records_the_provenance(tmp_path, monkeypatch):
    keys, _ = fake_export(tmp_path, 'qwen', monkeypatch, n=20, seeds=(0,))
    meta = json.loads((tmp_path / 'exports/new/qwen/matched/probs__s0.json').read_text())
    for k in ('sha256', 'mapper', 'mapper_sha256', 'view', 'n', 'seed', 'device'):
        assert k in meta
    assert meta['n'] == 20 and meta['device'] == 'cpu'
    with np.load(tmp_path / 'exports/new/qwen/matched/probs__s0.npz') as z:
        assert z['probs'].dtype == np.float32 and z['keys'].tolist() == keys


def test_write_export_refuses_bad_probs(tmp_path):
    keys = [f'S/{i}.jpg' for i in range(5)]
    bad = np.full((5, K), 1.5, np.float32)
    with pytest.raises(AssertionError):
        M.write_export(tmp_path, 'qwen', 0, keys, bad, mapper=Path('/x'), mapper_sha256='a', view='v')
    with pytest.raises(AssertionError):
        M.write_export(tmp_path, 'qwen', 0, keys[:4], np.full((5, K), .5, np.float32), mapper=Path('/x'),
                       mapper_sha256='a', view='v')
