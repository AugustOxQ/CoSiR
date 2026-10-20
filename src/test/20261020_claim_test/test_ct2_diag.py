"""Ticket 08: stage 2 diagnostics gate, fit inference and report keys on synthetic roots."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import ct_diag
import ct_score as sc
import ct2_diag as diag


def test_refuses_before_the_stage2_verdict_and_never_reads_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(ct_diag.ct_held, 'read_held_list', lambda *a: pytest.fail('read before verdict'))
    with pytest.raises(AssertionError, match='verdict_stage2.json first'):
        diag.run(tmp_path / 'verdict_stage2.json', tmp_path, tmp_path / 'diag_stage2.json')


def test_refuses_stage1_verdict_and_stage1_output_names(tmp_path):
    v1 = tmp_path / 'verdict_stage1.json'
    v1.write_text(json.dumps(dict(stage=1)))
    with pytest.raises(AssertionError, match='stage 2 verdict path'):
        diag.run(v1, tmp_path, tmp_path / 'diag_stage2.json')
    v2 = tmp_path / 'verdict_stage2.json'
    v2.write_text('{}')
    with pytest.raises(AssertionError, match='diagnostics path'):
        diag.run(v2, tmp_path, tmp_path / 'diag_stage1.json')
    with pytest.raises(AssertionError, match='diagnostics path'):
        diag.run(v2, tmp_path, v2)
    v2.write_text(json.dumps(dict(stage=1, outcome_id='S1-NO')))
    with pytest.raises(AssertionError):
        diag.read_verdict(v2)


def test_fit_inference_blocks_patches_and_checks_full_mode(tmp_path, monkeypatch):
    import torch
    import ct_d2 as d2
    import ct_percept as pc
    names = [f'fit-{i:02}' for i in range(5)]
    engine = diag.FitInference.__new__(diag.FitInference)
    engine.d2, engine.pc, engine.torch = d2, pc, torch
    engine.mode, engine.root, engine.names, engine.sources = 'full', tmp_path, names, {}
    engine.sample_sha256 = 'sample'
    engine.inputs = {}
    monkeypatch.setattr(diag, 'PATCH_BLOCK', 2)
    calls = []
    def patches(block):
        calls.append(list(block))
        return np.ones((len(block), 50, 3), dtype=np.float32)
    monkeypatch.setattr(diag.ct_held, 'clip_patches', patches)
    config = dict(settings='paper', cut='strict', arm='P1', seed=0, smoke=False, mode='full',
                  mapper='attention', input_dim=3, hidden_dim=4, device='cuda')
    seen = []
    class Model(torch.nn.Module):
        def load_state_dict(self, state): pass
        def forward(self, x):
            seen.append(tuple(x.shape))
            return torch.zeros(len(x), 7)
    monkeypatch.setattr(pc, 'build_mapper', lambda c, k: Model())
    path = tmp_path / 'percept/paper/stage2/strict/P1__s0.npz'
    path.parent.mkdir(parents=True)
    torch.save(dict(mapper={}), path.with_suffix('.pt'))
    meta = dict(config=config, K=7, sample_sha256='sample', weights_sha256=sc._sha_file(path.with_suffix('.pt')))
    path.with_suffix('.json').write_text(json.dumps(meta))
    prob = engine.predict('B2', 'emotion', 0)
    assert prob.shape == (5, 7) and np.allclose(prob, .5)
    assert calls == [names[:2], names[2:4], names[4:]]
    assert all(shape[1:] == (50, 3) for shape in seen)
    assert engine.sources['B2/emotion/0']['inference_device'] == 'cpu'
    meta['config']['mode'] = 'sample'
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        engine.predict('B2', 'emotion', 0)


def test_run_writes_stage2_report_with_keys_and_leaves_verdict_alone(tmp_path, monkeypatch):
    n = 24
    monkeypatch.setattr(sc, 'HELD_ROWS', n)
    monkeypatch.setattr(sc, 'SEEDS', [0])
    names = [f'held-{i:02}' for i in range(n)]
    y = np.arange(n) % 3
    x = ct_diag.normalize(np.random.default_rng(13).normal(size=(n, 8)))
    monkeypatch.setattr(ct_diag.ct_held, 'read_held_list', lambda path: (names, 'names'))
    monkeypatch.setattr(ct_diag.ct_held, 'held_names', lambda: names)
    monkeypatch.setattr(ct_diag.ct_held, 'painting_labels', lambda names: {a: y for a in ct_diag.PROMPTS})
    monkeypatch.setattr(ct_diag.ct_held, 'qwen_default_images', lambda names: x)
    monkeypatch.setattr(ct_diag, 'target_images', lambda *args: x)
    eligible = {arm: [0] if arm in ('K1', 'B1', 'B4') else [] for arm in sc.ARM_ORDER}
    sources = {arm: {'0': dict(status='finite' if eligible[arm] else 'rcol_collapsed_no_output')}
               for arm in sc.ARM_ORDER}
    verdict = dict(stage=2, outcome_id='S2-NO', held_sha256='names', eligibility=dict(eligible_seeds=eligible),
                   sources=dict(held_outputs=sources, rcol=dict(record=dict(runs=[]))))
    root = tmp_path / 'outputs'
    for aspect in ct_diag.PROMPTS:
        for arm in sc.ARM_ORDER if aspect == 'emotion' else ('K1', 'B1'):
            if aspect == 'emotion' and not eligible[arm]:
                continue
            stem = sc.HELD_STEMS[arm]
            if aspect != 'emotion':
                stem = f'd2/{aspect}' if arm == 'K1' else f'percept/core__k40/{"P2" if aspect == "style" else "P3"}'
            path = root / 'held' / f'{stem}__s0.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
            path.with_suffix('.json').write_text(json.dumps(dict(weights_sha256='weights')))
            if aspect == 'emotion':
                sources[arm]['0'].update(kind='k1' if arm == 'K1' else 'percept',
                                         output_sha256=sc._sha_file(path),
                                         sidecar_sha256=sc._sha_file(path.with_suffix('.json')),
                                         partitions='fixture')
    verdict_path = tmp_path / 'verdict_stage2.json'
    verdict_path.write_text(json.dumps(verdict))
    before = verdict_path.read_bytes()
    def predict(arm, aspect, seed):
        inference.sources[f'{arm}/{aspect}/{seed}'] = dict(weights_sha256='weights')
        return x
    inference = SimpleNamespace(names=['fit-a'], labels={a: y for a in ct_diag.PROMPTS},
                                predict=predict, sources={})
    modes = []
    monkeypatch.setattr(diag, 'FitInference', lambda out, names, mode: modes.append(mode) or inference)
    monkeypatch.setattr(sc, 'load_held_output', lambda *args, **kwargs: x)
    def fixed(kind):
        p = dict(r20=np.array([y] * 3))
        if kind != 'k1':
            p['own'] = y[None, :]
        return p
    monkeypatch.setattr(ct_diag, 'saved_partitions', lambda path, source, hash: fixed(source['kind']))
    def diagnose(x, labels, kind, prior=None, fit=None, fit_labels=None):
        partitions = fixed(kind)
        record = dict(held={key: ct_diag.partition_metrics(ps, labels) for key, ps in partitions.items()},
                      leiden_counts=[3, 3, 3])
        record['held']['leiden3a'] = record['held']['r20']
        record['held']['kmeans_at_leiden_count'] = record['held']['r20']
        if fit is not None:
            record['fit'] = dict(r20=record['held']['r20'])
            record['held']['leiden3b'] = record['held']['r20']
        return record, partitions['r20'], partitions
    monkeypatch.setattr(ct_diag, 'diagnose_vectors', diagnose)
    report = diag.run(verdict_path, root, tmp_path / 'diag_stage2.json', workers=1, n_resamples=2)
    assert modes == ['full']
    assert verdict_path.read_bytes() == before
    assert not (tmp_path / 'diag_stage1.json').exists()
    assert set(report) >= {'records', 'margins', 'paper_runs', 'fit_sources', 'verdict_sha256', 'eligibility'}
    assert set(report['margins']) == {'leiden_emotion', 'style', 'genre'}
    assert len(report['records']) == 13
    assert json.loads((tmp_path / 'diag_stage2.json').read_text())['diagnostic_only']
    assert '| target | style |' in (tmp_path / 'diag_stage2.md').read_text()
