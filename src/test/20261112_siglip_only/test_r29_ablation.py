"""SC7: stage guard, r28 training equivalence, paired removal contrasts and replay."""
import json
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_ablation as A


def test_train_refuses_without_a_pass_before_reading_any_training_input(tmp_path, monkeypatch):
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    for record in (None, {'status': 'fail'}, {'status': 'inconclusive'}, {'pass': True}):
        if record is not None:
            (tmp_path / 'pass_rule.json').write_text(json.dumps(record))
        with pytest.raises(AssertionError, match='pass_rule'):
            A.train('buddy', [0], 'cpu', A.P.SMOKE_OUT / 't14-unit', smoke=True)


@pytest.mark.parametrize('arm,form', [('siglip1', 'plain'), ('siglip1_prefix', 'prefixed')])
def test_kstar_asserts_the_fixed_interface_and_matching_form(tmp_path, monkeypatch, arm, form):
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    path = tmp_path / 'prefix_pick.json'
    path.write_text(json.dumps(dict(kstar=arm, kstar_form=form)))
    assert A.kstar() == arm
    for bad in ({'kstar': arm}, {'kstar_form': form}, {'kstar': 'qwen', 'kstar_form': 'plain'},
                {'kstar': arm, 'kstar_form': 'prefixed' if form == 'plain' else 'plain'}):
        path.write_text(json.dumps(bad))
        with pytest.raises(AssertionError):
            A.kstar()


@pytest.mark.parametrize('row', A.REMOVALS)
def test_qwen_equals_r28_state_loss_and_gradients_on_fixed_batch(tmp_path, monkeypatch, row):
    # Same real B.train -> rg_full.train_cohort seam as ticket 03; only tiny
    # input data, file writers and the epoch loop are replaced. Real models,
    # removal contexts, dropout and every head's loss/backward run on CPU.
    import test_r29_trainer as fixtures
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    (tmp_path / 'pass_rule.json').write_text('{"status": "pass"}')
    monkeypatch.setattr(A.P, 'SMOKE_OUT', tmp_path / 'smoke')
    monkeypatch.setattr(A.B, 'code_state', lambda: ('test-commit', False))
    monkeypatch.setattr(A.B.eg, 'atomic_record', lambda *args: None)
    monkeypatch.setattr(A.B, '_record', lambda *a, **kw: dict(status='ok', seconds=0, best_epoch={}))
    monkeypatch.setattr(A.R, 'BASE_RECORD', A.B._record)
    monkeypatch.setattr(A.T.C, 'store_hash', lambda source: 'test-store')
    s, inputs, img = fixtures.data()
    monkeypatch.setattr(A.B, 'load_data', lambda ctx: (s, inputs, {}, {}, img))
    root = A.P.SMOKE_OUT / 't14-fixed-no-write'
    expected, actual = {}, {}
    with A.R.trainer_patches(row):
        monkeypatch.setattr(A.T.ct2_d2, 'run', fixtures.fixed_batch_run(expected))
        A.B.train('siglip1', row, [0], 'cpu', root, smoke=True)
    monkeypatch.setattr(A.T.ct2_d2, 'run', fixtures.fixed_batch_run(actual))
    A.train(row, [0], 'cpu', root, smoke=True, source='qwen')
    assert actual['state'].keys() == expected['state'].keys()
    for key, value in actual['state'].items():
        assert torch.equal(value, expected['state'][key]), key
    for aspect in A.P.ASPECTS:
        assert torch.equal(actual['loss'][aspect], expected['loss'][aspect])
        for key, grad in actual['grads'][aspect].items():
            other = expected['grads'][aspect][key]
            assert other is None if grad is None else torch.equal(grad, other), (aspect, key)


def test_paired_intervals_share_training_seeds_and_restore_frozen_scorer(monkeypatch):
    rng = np.random.default_rng(8)
    y = np.tile(np.arange(3), 4).astype(np.int64)
    parts = rng.integers(0, 3, size=(5, 3, len(y)))
    ap = rng.random((5, len(y)))
    item = dict(parts=parts, ret=dict(ap=ap, query_ids=np.array([f'p{i}' for i in range(len(y))]), dropped=0))
    original = A.F.ns_clu.draws_iter
    monkeypatch.setattr(A.F.ns_clu, 'WORKERS', 1)
    result = A.paired_change(item, item, y, 3, ami_draws=24, map_draws=32)
    for metric in ('ami', 'map'):
        assert result[metric]['d'] == result[metric]['lo'] == result[metric]['hi'] == 0
    assert A.F.ns_clu.draws_iter is original
    # Reversing different arms reverses points and both interval endpoints.
    other = dict(parts=rng.integers(0, 3, size=parts.shape), ret=dict(item['ret'], ap=ap * 0.5))
    forward = A.paired_change(item, other, y, 3, ami_draws=24, map_draws=32)
    reverse = A.paired_change(other, item, y, 3, ami_draws=24, map_draws=32)
    for metric in ('ami', 'map'):
        assert forward[metric]['d'] == pytest.approx(-reverse[metric]['d'])
        assert forward[metric]['lo'] == pytest.approx(-reverse[metric]['hi'])
        assert forward[metric]['hi'] == pytest.approx(-reverse[metric]['lo'])


def test_paired_map_rejects_different_query_painting_order(monkeypatch):
    monkeypatch.setattr(A.F.ns_clu, 'WORKERS', 1)
    y = np.tile(np.arange(3), 4).astype(np.int64)
    item = dict(parts=np.tile(y, (5, 3, 1)), ret=dict(ap=np.ones((5, len(y))), query_ids=np.arange(len(y)), dropped=0))
    other = dict(item, ret=dict(item['ret'], query_ids=np.arange(len(y))[::-1]))
    with pytest.raises(AssertionError, match='query painting order'):
        A.paired_change(item, other, y, 3, ami_draws=2, map_draws=2)


def write_export(root, arm, *, source='siglip1', seed=0, **overrides):
    import hashlib
    x = np.arange(24, dtype=np.float16).reshape(6, 4)
    path = root / 'held' / arm / f'emotion__s{seed}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, img=x)
    meta = dict(kind='held', arm=arm, aspect='emotion', seed=seed, rows=6, list_sha256='names',
                arrays=dict(img=dict(shape=[6, 4], dtype='float16', sha256=hashlib.sha256(x.tobytes()).hexdigest())),
                cap_source=source, cap_width=A.P.CAPTION_SOURCES[source][1], cap_store_sha256='store',
                encoder=A.B.backbone('siglip1'))
    meta.update(overrides)
    path.with_suffix('.json').write_text(json.dumps(meta))
    return path, x


def test_held_loader_checks_names_array_and_kstar_caption_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(A.T.C, 'store_hash', lambda _: 'store')
    path, x = write_export(tmp_path, 'siglip1_prefix', source='siglip1_prefix')
    got = A.load_rows(tmp_path, 'siglip1_prefix', 'emotion', 0, 'names', {}, source='siglip1_prefix', n=6, width=4)
    assert np.array_equal(got, x.astype(np.float32))
    original = json.loads(path.with_suffix('.json').read_text())
    for bad in (dict(list_sha256='bad'), dict(cap_source='siglip1'), dict(cap_width=2048),
                dict(cap_store_sha256='changed'), dict(encoder='qwen3vl_emb_2b'), dict(arm='buddy')):
        path.with_suffix('.json').write_text(json.dumps(dict(original, **bad)))
        with pytest.raises(AssertionError):
            A.load_rows(tmp_path, 'siglip1_prefix', 'emotion', 0, 'names', {}, source='siglip1_prefix', n=6, width=4)
    path.with_suffix('.json').write_text(json.dumps(original))
    np.savez(path, img=np.zeros_like(x))
    with pytest.raises(AssertionError):
        A.load_rows(tmp_path, 'siglip1_prefix', 'emotion', 0, 'names', {}, source='siglip1_prefix', n=6, width=4)


def test_scorer_keeps_r28_float16_clustering_and_plain_cosine_readout(tmp_path, monkeypatch):
    y = np.tile(np.arange(9), 3).astype(np.int64)
    names = np.array([f'p{i}' for i in range(len(y))])
    rows = {s: A.F.r20_ret.unit(np.random.default_rng(s).normal(size=(len(y), 12))).astype(np.float16).astype(np.float32)
            for s in A.P.SEEDS}
    monkeypatch.setattr(A.F, 'load_rows', lambda arm, aspect, seed, *a, **kw: rows[seed])
    monkeypatch.setattr(A, 'load_rows', lambda root, arm, aspect, seed, *a, **kw: rows[seed])
    expected = A.F.score_arm_aspect('buddy', 'emotion', y, names, 'names', A.P.SEEDS, {}, lambda _: None)
    actual = A.score_aspect(tmp_path, 'buddy', 'emotion', y, names, 'names', {}, source='siglip1')
    for key in ('ami_per_seed', 'ami_mean', 'map_per_seed', 'map_mean', 'n_labelled', 'n_queries', 'dropped', 'seeds', 'k'):
        assert actual['cell'][key] == expected[key], key
    assert actual['parts'].shape == (5, 3, len(y))
    assert actual['ret']['ap'].shape == (5, len(y))


def test_gate_replay_checks_every_seed_mean_query_count_and_names():
    frozen = json.loads((A.P.R28_RECORDS / 'ablation_ref_replay.json').read_text())
    cells = {a: dict(seeds=list(A.P.SEEDS), k=A.F.CLASS_COUNT[a],
                    ami_per_seed=frozen['ami'][a]['per_seed'], ami_mean=frozen['ami'][a]['mean'],
                    map_per_seed=frozen['map_plain_cosine'][a]['per_seed'], map_mean=frozen['map_plain_cosine'][a]['mean'],
                    n_queries=frozen['map_plain_cosine'][a]['n_queries'], dropped=frozen['map_plain_cosine'][a]['dropped'])
             for a in A.P.ASPECTS}
    assert A.compare_replay(cells, frozen['list_sha256'], frozen)['pass']
    cells['style']['ami_per_seed'] = list(cells['style']['ami_per_seed'])
    cells['style']['ami_per_seed'][2] += 2e-9
    assert not A.compare_replay(cells, frozen['list_sha256'], frozen)['pass']
    cells['style']['ami_per_seed'][2] -= 2e-9
    cells['emotion']['n_queries'] -= 1
    assert not A.compare_replay(cells, frozen['list_sha256'], frozen)['pass']
    assert not A.compare_replay(cells, 'wrong names', frozen)['pass']


@pytest.mark.parametrize('failures', [1, 2])
def test_d21_retries_same_seed_once_then_records_inconclusive(tmp_path, monkeypatch, failures):
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    monkeypatch.setattr(A.P, 'SMOKE_OUT', tmp_path / 'smoke')
    (tmp_path / 'pass_rule.json').write_text('{"status":"pass"}')
    calls = []
    def crashing_train(enc, row, seeds, device, root, **kwargs):
        calls.append((enc, row, list(seeds), str(root)))
        partial = root / 'weights' / row / 'emotion__s0.pt'
        if len(calls) == 1:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b'crashed checkpoint')
        else:
            assert not partial.exists(), 'a rerun starts fresh even if all previous weights were written'
            assert list((root / 'replaced').rglob('emotion__s0.pt')), 'keep the previous checkpoint'
        if len(calls) <= failures:
            raise RuntimeError('simulated training crash')
        return {0: A.B.Run(dict(status='ok'), None, None)}
    monkeypatch.setattr(A.B, 'train', crashing_train)
    root = A.P.SMOKE_OUT / 'd21'
    result = A.train('buddy', [0], 'cpu', root, smoke=True, source='qwen')
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result[0].record['status'] == ('ok' if failures == 1 else 'inconclusive')
    journal = json.loads((root / 'train/buddy/attempts__s0.json').read_text())
    assert len(journal['attempts']) == 2
    assert journal['r29_code_sha256'] == A.code_hashes()
    if failures == 2:
        with pytest.raises(AssertionError, match='inconclusive'):
            A.train('buddy', [0], 'cpu', root, smoke=True, source='qwen')
        assert len(calls) == 2


def test_held_export_uses_the_removal_context_and_ticket03_caption_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    monkeypatch.setattr(A.P, 'OUT', tmp_path / 'out')
    (tmp_path / 'pass_rule.json').write_text('{"status":"pass"}')
    (tmp_path / 'prefix_pick.json').write_text('{"kstar":"siglip1_prefix","kstar_form":"prefixed"}')
    seen = []
    original = A.B.training_context
    def export(enc, source, seeds, root, **kwargs):
        assert A.B.training_context is not original
        with A.B.training_context(enc, A.B.REF, export=True) as ctx:
            assert ctx.arm == 'buddy' and ctx.cfg['remove'] == ('buddy',)
            seen.append((enc, source, list(seeds), kwargs['arm'], kwargs['kinds']))
    monkeypatch.setattr(A.T, 'export', export)
    A.export('buddy', [0], tmp_path / 'out', kinds=('held',), device='cpu')
    assert seen == [('siglip1', 'siglip1_prefix', [0], 'buddy', ('held',))]
    assert A.B.training_context is original


def test_full_scoring_has_sheet_rows_with_paired_intervals_and_no_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    monkeypatch.setattr(A.P, 'OUT', tmp_path / 'out')
    (tmp_path / 'pass_rule.json').write_text('{"status":"pass"}')
    (tmp_path / 'prefix_pick.json').write_text('{"kstar":"siglip1_prefix","kstar_form":"prefixed"}')
    monkeypatch.setattr(A, 'require_committed', lambda _: None)
    monkeypatch.setattr(A, 'require_gate', lambda: None)
    labels = {a: np.tile(np.arange(A.F.CLASS_COUNT[a]), 2).astype(np.int64) for a in A.P.ASPECTS}
    monkeypatch.setattr(A.F.t5, 'bind_labels', lambda: (np.array(['p']), 'names', labels))
    calls = []
    def score(root, arm, aspect, *args, **kwargs):
        calls.append((str(root), arm, kwargs['source']))
        point = 0.7 if arm == 'siglip1_prefix' else 0.5
        cell = dict(k=A.F.CLASS_COUNT[aspect], seeds=list(A.P.SEEDS), ami_per_seed=[point] * 5,
                    ami_mean=point, map_per_seed=[point] * 5, map_mean=point, n_queries=1, dropped=0, n_labelled=1)
        return dict(cell=cell)
    monkeypatch.setattr(A, 'score_aspect', score)
    monkeypatch.setattr(A, 'paired_change', lambda *args: {m: dict(d=-0.2, lo=-0.3, hi=-0.1) for m in ('ami', 'map')})
    document = A.score()
    saved = json.loads((tmp_path / 'ablation.json').read_text())
    assert document == saved and saved['complete'] and saved['claim'] == 'SC7'
    assert saved['kstar'] == 'siglip1_prefix' and saved['cap_width'] == 1152
    assert len(saved['cells']) == 24 and len(saved['arms']) == 5
    for cell in saved['cells']:
        assert cell['comparator'] == 'siglip1_prefix'
        assert cell['d'] == -0.2 and cell['lo'] == -0.3 and cell['hi'] == -0.1
        assert cell['cand'] == 0.5 and cell['comp'] == 0.7
        assert cell['value'] == cell['cand'] and cell['comparator_value'] == cell['comp']
        assert cell['outcome'] is None and cell['r'] is None
        assert cell['id'] == f'artelingo/siglip1/{cell["arm"]}/{cell["aspect"]}/{cell["measure"]}/siglip1_prefix'
    assert calls[0][0] == str(tmp_path / 'out/cstar/siglip1')
    assert all(source == 'siglip1_prefix' for _, _, source in calls)


def test_full_score_refuses_before_any_held_read_without_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    with pytest.raises(AssertionError, match='pass_rule'):
        A.score()


@pytest.mark.parametrize('trainer_ok', [True, False])
def test_gate_writes_separate_record_and_requires_trainer_and_replay(tmp_path, monkeypatch, trainer_ok):
    from types import SimpleNamespace
    monkeypatch.setattr(A.P, 'GATES', tmp_path / 'gates')
    monkeypatch.setattr(A.P, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(A.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0 if trainer_ok else 1,
                                                                         stdout='4 fixed-batch tests', stderr=''))
    frozen = json.loads((A.P.R28_RECORDS / 'ablation_ref_replay.json').read_text())
    monkeypatch.setattr(A.F.t5, 'bind_labels', lambda: (np.array(['p']), frozen['list_sha256'], dict.fromkeys(A.P.ASPECTS, None)))
    called = []
    def score(root, arm, aspect, *a, **kw):
        called.append((root, arm))
        assert kw['source'] is None and root == A.P.K0_ROOT and arm == A.P.K0_ARM
        return dict(cell=dict(k=A.F.CLASS_COUNT[aspect], seeds=list(A.P.SEEDS),
                             ami_per_seed=frozen['ami'][aspect]['per_seed'], ami_mean=frozen['ami'][aspect]['mean'],
                             map_per_seed=frozen['map_plain_cosine'][aspect]['per_seed'],
                             map_mean=frozen['map_plain_cosine'][aspect]['mean'],
                             n_queries=frozen['map_plain_cosine'][aspect]['n_queries'], dropped=0))
    monkeypatch.setattr(A, 'score_aspect', score)
    result = A.gate_vii()
    gate = json.loads((A.P.GATES / 'vii_ablation.json').read_text())
    assert result == gate and gate['gate'] == 'vii_ablation' and gate['tries'] == 1
    assert gate['status'] == ('pass' if trainer_ok else 'fail')
    assert gate['details']['trainer']['pass'] == trainer_ok
    assert len(called) == (3 if trainer_ok else 0)
    assert 'written' in gate
    if trainer_ok:
        A.require_gate()
        gate['details']['code_sha256'] = {}
        (A.P.GATES / 'vii_ablation.json').write_text(json.dumps(gate))
        with pytest.raises(AssertionError, match='code changed'):
            A.require_gate()


def test_cli_smoke_keeps_qwen_standin_and_all_files_under_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(A.P, 'SMOKE_OUT', tmp_path / 'smoke')
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    (tmp_path / 'pass_rule.json').write_text('{"status":"pass"}')
    seen = []
    def train(row, seeds, device, root, **kwargs):
        seen.append(('train', row, list(seeds), root, kwargs))
        return {0: A.B.Run(dict(status='ok'), None, None)}
    monkeypatch.setattr(A, 'train', train)
    monkeypatch.setattr(A, 'smoke_checkpoint', lambda root, device: seen.append(('checkpoint', root)))
    monkeypatch.setattr(A, 'export', lambda row, seeds, root, **kwargs: seen.append(('export', root, kwargs)))
    assert A.main(['train', '--row', 'buddy', '--seeds', '0', '--smoke', '--k0-standin', '--device', 'cpu']) == 0
    root = tmp_path / 'smoke/ablation/siglip1/qwen'
    assert seen[0] == ('train', 'buddy', [0], root, dict(smoke=True, source='qwen', max_epochs=None))
    assert seen[1] == ('export', root, dict(smoke=True, source='qwen', device='cpu'))
    assert seen[2] == ('checkpoint', root)


@pytest.mark.parametrize('source,form', [('siglip1', 'plain'), ('siglip1_prefix', 'prefixed')])
def test_selected_kstar_input_reaches_real_removal_loss_at_1152(tmp_path, monkeypatch, source, form):
    import test_r29_trainer as fixtures
    monkeypatch.setattr(A.P, 'RECORDS', tmp_path)
    monkeypatch.setattr(A.P, 'SMOKE_OUT', tmp_path / 'smoke')
    (tmp_path / 'pass_rule.json').write_text('{"status":"pass"}')
    (tmp_path / 'prefix_pick.json').write_text(json.dumps(dict(kstar=source, kstar_form=form)))
    monkeypatch.setattr(A.B, 'code_state', lambda: ('test', False))
    monkeypatch.setattr(A.B.eg, 'atomic_record', lambda *args: None)
    monkeypatch.setattr(A.R, 'BASE_RECORD', lambda *a, **kw: dict(status='ok'))
    monkeypatch.setattr(A.T.C, 'store_hash', lambda source: 'store')
    s, inputs, img = fixtures.data()
    monkeypatch.setattr(A.B, 'load_data', lambda ctx: (s, inputs, {}, {}, img))
    asked = []
    means = fixtures.siglip_means()
    monkeypatch.setattr(A.T.C, 'painting_means', lambda source, names: asked.append((source, list(names))) or means)
    log = {}
    monkeypatch.setattr(A.T.ct2_d2, 'run', fixtures.fixed_batch_run(log))
    A.train('relational', [0], 'cpu', A.P.SMOKE_OUT / 'width', smoke=True)
    assert asked == [(source, list(s['all_painting_ids']))]
    assert log['state']['cap.trunk.0.weight'].shape == (1024, 1152)
    for aspect in A.P.ASPECTS:
        assert np.array_equal(log['inputs'][aspect]['cap'], means)
        assert np.array_equal(log['inputs'][aspect]['img'], img)
        assert inputs[aspect]['cap'].shape[1] == 2048  # Qwen graph data was not mutated
        assert torch.isfinite(log['loss'][aspect])
        assert log['grads'][aspect]['cap.trunk.0.weight'] is not None
