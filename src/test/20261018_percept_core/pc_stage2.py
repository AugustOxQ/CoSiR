"""Train image-only PercepT core topic mappers from frozen Stage 1 captions."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import pc_data
import pc_fixes
import pc_sept

HERE = Path(__file__).resolve().parent
ARMS = dict(P1=('clip_roberta', 'patch', 512), P2=('clip_roberta', 'clip_image', 512),
            P3=('qwen_qwen', 'qwen_default', 2048), P4=('qwen_qwen', 'patch', 512),
            P5=('qwen_roberta', 'qwen_default', 2048))


def load_stage1(s, path, version, seed):
    """Validate the Stage 1 identity and read A3's 40 caption topics."""
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['sample_sha256'] == pc_data.fingerprint(s), 'Stage 1 sample mismatch'
    assert meta['config']['version'] == version, 'Stage 1 version mismatch'
    assert meta['config']['seed'] == seed, 'Stage 1 seed mismatch'
    assert set(meta['K_gamma']) == {'1.5', '2', '3', '5'}
    k = meta['K']
    assert type(k) is int and k == 40
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'center_norms', 'fit_counts', 'kept', 'topic', 'q_paint_eval'}
        assert z['center_norms'].shape == z['fit_counts'].shape == z['kept'].shape == (100,)
        assert z['center_norms'].dtype == np.float32 and np.isfinite(z['center_norms']).all()
        assert z['fit_counts'].dtype == np.int64 and (z['fit_counts'] >= 0).all()
        assert z['kept'].dtype == np.bool_ and k == 40 == int(z['kept'].sum())
        topic = z['topic']
        assert topic.shape == s['pid'].shape and topic.dtype == np.int16
        assert ((topic >= 0) & (topic < k)).all()
        q = z['q_paint_eval']
        assert q.shape == (len(s['subset_pid']), k) and q.dtype == np.float32
        assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
        assert np.allclose(q.sum(axis=1), 1, atol=1e-6)
    return topic, k


def caption_targets(s, topic, k, smoke=False):
    """Union fit/dev captions separately; keep evaluation in subset_pid order."""
    partition = pc_data.partitions(s)
    assert partition.shape == (len(s['first']),)
    ids = {part: np.flatnonzero(partition == part).astype(np.int64)
           for part in ('fit', 'dev')}
    ids['eval'] = s['subset_pid'].copy()
    assert (partition[ids['eval']] == 'eval').all()
    if smoke:
        assert len(ids['fit']) >= 200
        ids['fit'] = ids['fit'][:200]
    targets = {}
    for part in ('fit', 'dev'):
        rows = pc_fixes.pair_rows(s['pid'], partition, part)
        targets[part] = pc_fixes.multi_hot(s['pid'][rows], topic[rows], ids[part], k)
    return ids, targets


def image_inputs(s, arm, ids, device):
    """Use stored patches or normalized global embeddings, joined by painting."""
    _, kind, width = ARMS[arm]
    tensors = {}
    if kind != 'patch':
        global_x = pc_data.image_features(s, kind)
        assert global_x.shape == (len(s['first']), width) and global_x.dtype == np.float32
        assert np.isfinite(global_x).all()
        norms = np.linalg.norm(global_x, axis=1, keepdims=True)
        assert (norms > 0).all()
        global_x = global_x / norms
    for part, pids in ids.items():
        x = pc_data.patch_tokens(s, pids) if kind == 'patch' else global_x[pids]
        shape = (len(pids), 50, width) if kind == 'patch' else (len(pids), width)
        assert x.shape == shape and x.dtype == np.float32 and np.isfinite(x).all()
        tensors[part] = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    return tensors


def make_config(s, arm, seed, smoke, device):
    version, kind, width = ARMS[arm]
    partition = pc_data.partitions(s)
    counts = {part: int((partition == part).sum()) for part in ('fit', 'dev', 'eval')}
    if not smoke:
        assert tuple(counts.values()) == (13400, 1489, 5000)
    return dict(arm=arm, version=version, seed=seed, smoke=smoke, device=str(device),
                source_commit='1b70d7f', input=kind, input_dim=width,
                mapper='attention' if kind == 'patch' else 'mlp',
                hidden_dim=None if kind == 'patch' else 1024,
                activation=None if kind == 'patch' else 'ReLU',
                normalization='stored patches' if kind == 'patch' else 'l2 global vector',
                targets='multi-hot union of own partition caption topics; A3 K=40',
                fit_paintings=min(200, counts['fit']) if smoke else counts['fit'],
                dev_paintings=counts['dev'], eval_paintings=counts['eval'],
                smoke_fit_selection='first 200 fit paintings in frozen order' if smoke else None,
                epochs=3 if smoke else 100, lr=.003, optimizer='Adam',
                adam_betas=[.9, .999], adam_eps=1e-8, weight_decay=0,
                batch='full fit paintings', loss='mean BCEWithLogitsLoss',
                scheduler=None, early_stopping=False,
                dev_metric='September per-topic ROC AUC; skip constant; unweighted macro',
                eval_order='subset_pid', numpy_seed=seed, torch_seed=seed, cuda_seed=seed,
                cudnn_deterministic=True, cudnn_benchmark=False,
                deterministic_algorithms=True, deterministic_warn_only=True,
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'], cpu_threads=8)


def train_mapper(inputs, targets, config, k):
    """Reseed each mapper and run September's full-batch BCE/Adam recipe."""
    seed, device = config['seed'], config['device']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config['mapper'] == 'attention':
        mapper = pc_sept.AttentionPoolingMapper(n_topics=k)
    else:
        mapper = nn.Sequential(nn.Linear(config['input_dim'], 1024), nn.ReLU(),
                               nn.Linear(1024, k))
    mapper.to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=config['lr'])
    target = torch.from_numpy(targets['fit']).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    history, epoch_seconds = [], []
    mapper.train()
    for epoch in range(1, config['epochs'] + 1):
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        tick = time.perf_counter()
        loss = loss_fn(mapper(inputs['fit']), target)
        assert torch.isfinite(loss), 'nonfinite mapper loss'
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - tick)
        print(f"{config['arm']}: epoch={epoch} mean_bce={history[-1]:.6f} "
              f'seconds_per_epoch={epoch_seconds[-1]:.4f}', flush=True)
    mapper.eval()
    with torch.no_grad():
        dev_prob = mapper(inputs['dev']).sigmoid().cpu().numpy()
        eval_prob = mapper(inputs['eval']).sigmoid().cpu().numpy()
    return eval_prob, dev_prob, history, epoch_seconds


def write_sidecar(path, meta):
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def run(s, arm, seed, smoke=None, device=None):
    """Read the arm's frozen topics, train on fit paintings and seal predictions."""
    assert arm in ARMS and seed in (0, 1, 2)
    start = time.perf_counter()
    root = Path(smoke) if smoke is not None else HERE / 'outputs'
    version, _, _ = ARMS[arm]
    stage1 = root / 'stage1' / f'{version}__s{seed}.npz'
    stage1_sha256 = hashlib.sha256(stage1.read_bytes()).hexdigest()
    topic, k = load_stage1(s, stage1, version, seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = make_config(s, arm, seed, smoke is not None, device)
    folder = root / 'stage2'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{arm}__s{seed}.npz'
    meta = dict(config=config, K=k, stage1_sha256=stage1_sha256,
                dev_auc=None, dev_valid_topics=0, loss_history=[], seconds=0.)
    assert k == 40
    assert smoke is not None or str(device).startswith('cuda'), 'full Stage 2 requires GPU'
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    ids, targets = caption_targets(s, topic, k, smoke is not None)
    with threadpool_limits(limits=8):
        inputs = image_inputs(s, arm, ids, device)
        eval_prob, dev_prob, history, epoch_seconds = train_mapper(inputs, targets, config, k)
    for part, prob in (('eval', eval_prob), ('dev', dev_prob)):
        assert prob.shape == (len(ids[part]), k) and prob.dtype == np.float32
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    positives = targets['dev'].sum(axis=0)
    valid = (positives > 0) & (positives < len(targets['dev']))
    if valid.any():
        aucs, skipped = pc_sept.evaluate_auc(dev_prob, targets['dev'],
                                            lambda line: print(line, flush=True), arm + ' dev')
        meta.update(dev_auc=pc_sept.auc_summary(aucs)['macro'], dev_valid_topics=len(aucs),
                    dev_skipped_topics=skipped)
    else:
        meta.update(dev_auc_note='No dev topic has both positive and negative labels',
                    dev_skipped_topics=list(range(k)))
        print(f"{arm}: {meta['dev_auc_note']}", flush=True)
    meta.update(loss_history=history, epoch_seconds=epoch_seconds)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, eval_prob=eval_prob)
    meta['seconds'] = time.perf_counter() - start
    pending.replace(path)
    write_sidecar(path.with_suffix('.json'), meta)
    print(f"Wrote {path}; dev_auc={meta['dev_auc']}; seconds={meta['seconds']:.3f}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--smoke', type=Path,
                        help='input/output root; first 200 fit paintings, 3 epochs')
    args = parser.parse_args()
    run(pc_data.load_frozen(), args.arm, args.seed, args.smoke)


if __name__ == '__main__':
    main()
