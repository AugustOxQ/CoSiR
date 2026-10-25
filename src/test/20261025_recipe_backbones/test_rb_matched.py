"""Matched arm seams (ticket 11): width reaches only the first Linear, the P3 recipe on another image, saved
inference, output sidecars, the regression items. All on CPU with a tiny fake split."""
import json
import os

import numpy as np
import pytest
import torch

import rb_paths  # noqa: F401  (sys.path)
import ct_percept
import rb_matched as m
import rb_score

K = 40   # the CT verifier fixes K = 40
N_FIT, N_DEV, N_EVAL = 30, 10, 10
N_PAINT = N_FIT + N_DEV + N_EVAL
N_HELD, N_A28 = 7, 5


def fake_split(seed=0):
    """A sample-mode split: two captions per painting, roles fit / dev / eval = subset_pid, topics in [0, K)."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(N_PAINT), 2).astype(np.int64)
    topic = rng.integers(0, K, size=len(pid)).astype(np.int16)
    topic[:K] = np.arange(K)   # every topic used
    s = dict(mode='sample', pid=pid, first=np.arange(0, len(pid), 2), names_sha256={},
             all_painting_ids=np.asarray([f'p{i:03d}' for i in range(N_PAINT)]),
             fit=np.arange(N_FIT), dev=np.arange(N_FIT, N_FIT + N_DEV),
             subset_pid=np.arange(N_FIT + N_DEV, N_PAINT))
    return s, topic


def unit(n, width, seed):
    x = np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def train(width, seed=0, images=None, s_topic=None, epochs=None):
    s, topic = s_topic or fake_split()
    images = unit(N_PAINT, width, 1) if images is None else images
    return m.train_stage2(s, topic, K, images, 'fake', seed, 'cpu', smoke=False, epochs=epochs), s, images


# ---- the recipe: width changes only the first Linear -------------------------------------------------------

def test_width_reaches_only_the_first_linear():
    narrow, _, _ = train(16, epochs=2)
    wide, _, _ = train(40, epochs=2)
    shapes = lambda r: {k: tuple(v.shape) for k, v in r['state'].items()}
    assert shapes(narrow) == {'0.weight': (1024, 16), '0.bias': (1024,), '2.weight': (K, 1024), '2.bias': (K,)}
    assert shapes(wide) == {'0.weight': (1024, 40), '0.bias': (1024,), '2.weight': (K, 1024), '2.bias': (K,)}
    for key in ('targets',):
        for part in ('fit', 'dev'):
            np.testing.assert_array_equal(narrow[key][part], wide[key][part])
    keep = lambda r: {k: v for k, v in r['config'].items() if k not in ('input', 'input_dim')}
    assert keep(narrow) == keep(wide)
    assert narrow['config']['input_dim'] == 16 and wide['config']['input_dim'] == 40


def test_recipe_is_p3s_and_follows_the_spec_literally():
    """100 epochs, full batch, Adam lr .003, mean BCE, seeded by the PercepT seed; written out from the spec."""
    result, s, images = train(16, seed=3)
    cfg = result['config']
    assert (cfg['epochs'], cfg['lr'], cfg['optimizer'], cfg['seed'], cfg['hidden_dim'], cfg['activation']) == \
        (100, .003, 'Adam', 3, 1024, 'ReLU')
    x = torch.from_numpy(images / np.linalg.norm(images, axis=1, keepdims=True))   # the second division of P3
    y = {part: torch.from_numpy(result['targets'][part]) for part in ('fit', 'dev')}
    torch.manual_seed(3)
    net = torch.nn.Sequential(torch.nn.Linear(16, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, K))
    opt = torch.optim.Adam(net.parameters(), lr=.003)
    for _ in range(100):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x[s['fit']]), y['fit'])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        expected = net(x[s['subset_pid']]).sigmoid().numpy()
    np.testing.assert_array_equal(result['eval_prob'], expected)


def test_targets_are_the_union_of_own_partition_caption_topics():
    result, s, _ = train(8, epochs=1)
    s, topic = fake_split()
    fit = result['targets']['fit']
    assert fit.shape == (N_FIT, K) and fit.dtype == np.float32
    for p in (0, 7, 29):
        assert set(np.flatnonzero(fit[p])) == set(topic[2 * p:2 * p + 2].tolist())


def test_training_reads_the_given_images_and_is_repeatable():
    a, _, images = train(16, epochs=3)
    b, _, _ = train(16, epochs=3)
    c, _, _ = train(16, epochs=3, images=unit(N_PAINT, 16, 2))
    np.testing.assert_array_equal(a['eval_prob'], b['eval_prob'])
    assert not np.array_equal(a['eval_prob'], c['eval_prob'])


def test_images_must_match_the_split_and_be_float32():
    s, topic = fake_split()
    with pytest.raises(AssertionError):
        m.train_stage2(s, topic, K, unit(N_PAINT - 1, 8, 1), 'fake', 0, 'cpu', smoke=False, epochs=1)
    with pytest.raises(AssertionError):
        m.train_stage2(s, topic, K, unit(N_PAINT, 8, 1).astype(np.float64), 'fake', 0, 'cpu', smoke=False, epochs=1)


# ---- saved mapper and inference ------------------------------------------------------------------------------

def test_saved_mapper_infers_its_own_training_eval_rows_exactly(tmp_path):
    result, s, images = train(16, seed=1, epochs=4)
    path = tmp_path / 'P3__s1.npz'
    m.save_stage2(path, result)
    prob = m.infer(path, images[s['subset_pid']])
    assert prob.dtype == np.float32 and prob.shape == (N_EVAL, K)
    np.testing.assert_array_equal(prob, result['eval_prob'])


def test_infer_refuses_wrong_width_and_other_device(tmp_path):
    result, s, images = train(16, epochs=1)
    path = tmp_path / 'P3__s0.npz'
    m.save_stage2(path, result)
    with pytest.raises(AssertionError):
        m.infer(path, unit(3, 17, 0))
    with pytest.raises(AssertionError, match='device'):
        m.infer(path, images[:3], device='cuda')


def test_infer_normalizes_twice_for_qwen_trained_mappers_once_for_clip(tmp_path):
    result, s, images = train(16, epochs=2)
    path = tmp_path / 'P3__s0.npz'
    m.save_stage2(path, result)
    # a small off-norm image set: P3-style mappers divide by the norm again, as in training
    scaled = images[:4] * 3
    np.testing.assert_allclose(m.infer(path, scaled), m.infer(path, images[:4]), rtol=1e-5, atol=1e-6)
    assert m.renormalizes('qwen_default') and m.renormalizes('siglip2 default image')
    assert not m.renormalizes('clip_image')


# ---- outputs and sidecars -------------------------------------------------------------------------------------

def test_written_output_passes_the_scorers_sidecar_check(tmp_path):
    prob = np.random.default_rng(0).uniform(size=(N_HELD, K)).astype(np.float32)
    path = tmp_path / 'part2/sample/held/M2/P3__s2.npz'
    m.write_output(path, prob, list_sha256='ab' * 32, scale='sample', arm='M2', seed=2)
    json.loads(path.with_suffix('.json').read_text())
    x = rb_score.load_output(path, 'M2', n_rows=N_HELD, list_sha256='ab' * 32, scale='sample', seed=2,
                             aspect='percept')
    np.testing.assert_array_equal(x, prob)
    with pytest.raises(AssertionError):
        rb_score.load_output(path, 'M2', n_rows=N_HELD, list_sha256='cd' * 32, scale='sample', seed=2,
                             aspect='percept')


def test_write_output_rejects_bad_probabilities(tmp_path):
    good = np.full((3, K), .5, np.float32)
    for bad in (good.astype(np.float64), np.where(good > 0, 1.5, 0).astype(np.float32),
                np.full((3, K), np.nan, np.float32)):
        with pytest.raises(AssertionError):
            m.write_output(tmp_path / 'x.npz', bad, list_sha256='ab' * 32, scale='sample', arm='M2', seed=0)


# ---- world: orchestration of one matched arm and the CT-mapper inference ----------------------------------------

WIDTHS = {'qwen3vl_emb_2b': 24, 'clip_b32': 12, 'siglip2_so400m': 20}


def fake_world(tmp_path, scale='sample'):
    s, topic = fake_split()
    held = [f'h{i}' for i in range(N_HELD)]
    a28 = [f'a{i}' for i in range(N_A28)]
    store = {}

    def full_images(backbone, names):
        names = list(names)
        key = (backbone, 'full')
        pool = store.setdefault(key, {})
        for n in names:
            pool.setdefault(n, unit(1, WIDTHS[backbone], abs(hash((n, backbone))) % 1000)[0])
        return np.stack([pool[n] for n in names]).astype(np.float32)

    def a28_images(backbone):
        return full_images(backbone, a28)

    return m.World(scale=scale, ct_root=tmp_path / 'ct', s=s, topics=lambda seed: (topic, K, 'cc' * 32),
                   held_names=held, held_sha256='aa' * 32, full_images=full_images,
                   a28_names=a28, a28_sha256='bb' * 32, a28_images=a28_images,
                   clip_images=lambda names: full_images('clip_b32', names))


def make_ct_files(world, seed=0, device='cpu'):
    """Stand-ins for the CT's saved P3 and P2 mappers and held outputs: written with the CT's own writer."""
    s, topic = world.s, world.topics(seed)[0]
    for arm, backbone in (('P3', 'qwen3vl_emb_2b'), ('P2', 'clip_b32')):
        images = world.full_images(backbone, s['all_painting_ids'])
        result = m.train_stage2(s, topic, K, images, backbone, seed, device, smoke=False, epochs=3, arm=arm)
        path = world.ct_root / f'percept/core/stage2/k40/{arm}__s{seed}.npz'
        m.save_stage2(path, result)
        held = m.infer(path, world.full_images(backbone, world.held_names))
        out = world.ct_root / f'held/percept/core__k40/{arm}__s{seed}.npz'
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, prob=held)


def test_run_matched_writes_held_and_a28_outputs_for_every_seed_given(tmp_path):
    world = fake_world(tmp_path)
    out = tmp_path / 'out'
    m.run_matched(world, 'M2', [0], out, 'cpu', smoke=False, epochs=2, gate=False)
    held = out / 'part2/sample/held/M2/P3__s0.npz'
    a28 = out / 'part2/sample/a28/M2/P3__s0.npz'
    assert (out / 'part2/sample/matched/M2/P3__s0.pt').exists()
    h = rb_score.load_output(held, 'M2', N_HELD, world.held_sha256, 'sample', 0, 'percept')
    a = rb_score.load_output(a28, 'M2', N_A28, world.a28_sha256, 'sample', 0, 'percept')
    assert h.shape == (N_HELD, K) and a.shape == (N_A28, K)
    # held output is the saved mapper's reading of the candidate's own held image
    prob = m.infer(out / 'part2/sample/matched/M2/P3__s0.npz', world.full_images('siglip2_so400m', world.held_names))
    np.testing.assert_array_equal(h, prob)


def test_run_matched_gate_blocks_before_any_training(tmp_path):
    world = fake_world(tmp_path)
    with pytest.raises(Exception):
        m.run_matched(world, 'M3', [0], tmp_path / 'out', 'cpu', smoke=False, epochs=1, gate=True)
    assert not (tmp_path / 'out/part2/sample').exists()


def test_arm_names_map_to_their_backbone():
    assert m.MATCHED == {'M2': 'siglip2_so400m', 'M3': 'clip_b32', 'M5': 'dinov2_large'}
    assert m.CANDIDATE == {'M2': 'K2', 'M3': 'K3', 'M5': 'K5'}


def test_ct_arms_on_a28_use_the_ct_mapper_and_the_a28_store_of_their_input(tmp_path):
    world = fake_world(tmp_path)
    make_ct_files(world)
    out = tmp_path / 'out'
    m.run_ct_a28(world, 'B1', [0], out, gate=False)
    m.run_ct_a28(world, 'B6', [0], out, gate=False)
    for arm, number, backbone in (('B1', 3, 'qwen3vl_emb_2b'), ('B6', 2, 'clip_b32')):
        path = out / f'part2/sample/a28/{arm}/P{number}__s0.npz'
        got = np.load(path)['prob']
        rb_score.check_sidecar(path, got, arm, N_A28, world.a28_sha256, 'sample', 0, 'percept')
        expected = m.infer(world.ct_root / f'percept/core/stage2/k40/P{number}__s0.npz', world.a28_images(backbone))
        np.testing.assert_array_equal(got, expected)


# ---- regressions --------------------------------------------------------------------------------------------------

def test_regression_passes_when_the_qwen_input_reproduces_the_ct_files(tmp_path):
    world = fake_world(tmp_path)
    make_ct_files(world)
    out = tmp_path / 'out'
    record = m.regress(world, out, 'cpu', epochs=3)
    path = out / 'part2/regression/stage2.json'
    assert json.loads(path.read_text()) == record
    assert record['pass'] is True
    assert record['items'] and all(k.startswith('matched/') for k in record['items'])
    assert {'matched/qwen_s0/eval_prob', 'matched/qwen_s0/held_prob', 'matched/qwen_s0/mapper_state',
            'matched/ct_mapper_infer/P3_s0', 'matched/ct_mapper_infer/P2_s0'} <= set(record['items'])


def test_regression_fails_on_one_flipped_bit_and_keeps_other_tickets_items(tmp_path):
    world = fake_world(tmp_path)
    make_ct_files(world)
    out = tmp_path / 'out'
    rb_score.write_regression_items(out / 'part2/regression/stage2.json', {'student/x': {'pass': True}})
    ct_held = world.ct_root / 'held/percept/core__k40/P3__s0.npz'
    prob = np.load(ct_held)['prob'].copy()
    prob[2, 3] = np.nextafter(prob[2, 3], np.float32(2))
    np.savez_compressed(ct_held, prob=prob)
    record = m.regress(world, out, 'cpu', epochs=3)
    assert record['pass'] is False
    assert record['items']['matched/qwen_s0/held_prob']['pass'] is False
    assert record['items']['matched/qwen_s0/eval_prob']['pass'] is True
    assert record['items']['student/x'] == {'pass': True}


def test_regression_refuses_another_device_than_the_ct_trained_on(tmp_path):
    world = fake_world(tmp_path)
    make_ct_files(world)
    with pytest.raises(AssertionError, match='device'):
        m.regress(world, tmp_path / 'out', 'cuda', epochs=3)


def test_regression_stage_names_map_to_scales():
    assert m.REGRESSION == {'stage2': 'sample', 'stage3a': 'full'}


def test_regress_cli_refuses_smoke_and_smoke_refuses_outside_tmp(tmp_path):
    with pytest.raises(AssertionError, match='smoke'):
        m.main(['--regress', 'stage2', '--smoke', '--outputs', str(tmp_path)])
    with pytest.raises(AssertionError, match='/tmp/claude-0/'):
        m.main(['--smoke', '--scale', 'sample', '--arm', 'M2', '--outputs', '/project/x'])


# ---- real frozen data (slow) -------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get('RB_SLOW'), reason='reads the real frozen sample (set RB_SLOW=1)')
def test_real_ct_stage1_topics_load_at_sample_scale():
    world = m.load_world('sample')
    topic, k, sha = world.topics(0)
    assert len(sha) == 64
    assert k == 40 and topic.shape == world.s['pid'].shape and topic.dtype == np.int16
    assert world.held_sha256 == rb_paths.HELD_SHA256 and len(world.held_names) == 12281
