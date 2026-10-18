"""CPU image-only buddy-pair students. No label access or caption targets in training."""
import argparse
import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from pbp_graphs import (HERE, SMOKE, atomic_json, default_images, fingerprint,
                        load_frozen, load_graphs, root, sha_file)
from itp_train import CONFIG, Student

VARIANTS = dict(S_E='E', S_P='P', S_D='D', S_PN='P')
LOSS_VERSION = 'symmetric pair InfoNCE; PN up to4 unique N-neighbours/anchor; fixed dev draws; v1'


def adjacency(pairs, n):
    lists = [[] for _ in range(n)]
    for a, b in pairs:
        lists[a].append(int(b))
        lists[b].append(int(a))
    return [np.asarray(sorted(v), dtype=np.int64) for v in lists]


def sample_negatives(anchors, neighbours, rng):
    sampled = np.full((len(anchors), 4), -1, dtype=np.int64)
    for row, anchor in enumerate(anchors):
        ns = neighbours[int(anchor)]
        size = min(4, len(ns))
        if size:
            sampled[row, :size] = rng.choice(ns, size=size, replace=False)
    return sampled


def hard_logits(model, x, z, sampled):
    """Only valid negative embeddings are evaluated; absent columns are -inf."""
    rows, cols = np.nonzero(sampled >= 0)
    extra = z.new_full(sampled.shape, -torch.inf)
    if len(rows):
        unique, inverse = np.unique(sampled[rows, cols], return_inverse=True)
        emb = model(x[torch.as_tensor(unique)])
        vals = (z[torch.as_tensor(rows)] * emb[torch.as_tensor(inverse)]).sum(1) / CONFIG['temperature']
        extra[torch.as_tensor(rows), torch.as_tensor(cols)] = vals
    return extra


def direction_logits(base, extra=None):
    return base if extra is None else torch.cat((base, extra), dim=1)


def pair_loss(model, x, pairs, neighbours=None, rng=None):
    a, b = pairs[:, 0], pairs[:, 1]
    zi, zj = model(x[torch.as_tensor(a)]), model(x[torch.as_tensor(b)])
    base = zi @ zj.T / CONFIG['temperature']
    ei = ej = None
    if neighbours is not None:
        ei = hard_logits(model, x, zi, sample_negatives(a, neighbours, rng))
        ej = hard_logits(model, x, zj, sample_negatives(b, neighbours, rng))
    labels = torch.arange(len(pairs))
    return (F.cross_entropy(direction_logits(base, ei), labels) +
            F.cross_entropy(direction_logits(base.T, ej), labels)) / 2


def make_config(s, manifest, variant, seed, smoke):
    name = VARIANTS[variant]
    return dict(CONFIG, variant=variant, graph=name, seed=seed, smoke=smoke, device='cpu',
        max_epochs=2 if smoke else CONFIG['max_epochs'],
        sample_sha256=fingerprint(s), loss=LOSS_VERSION, hard_negatives=4 if variant == 'S_PN' else 0,
        graph_identity=manifest['identity'], graph_archives=manifest['archives'],
        source_sha256=manifest['source_sha256'],
        fit_pair_count=manifest['counts']['fit'][name], dev_pair_count=manifest['counts']['dev'][name],
        fit_negative_pair_count=manifest['counts']['fit']['N'] if variant == 'S_PN' else 0,
        dev_negative_pair_count=manifest['counts']['dev']['N'] if variant == 'S_PN' else 0)


def verify_output(path, config):
    """Output array is the completion marker; incompatible sealed output raises."""
    path = path if hasattr(path, 'exists') else HERE / path
    if not path.exists():
        return None
    info = json.loads(path.with_suffix('.json').read_text())
    assert info['config'] == config, f'{path}: resume config mismatch'
    assert sha_file(path) == info['output_sha256'], f'{path}: output checksum mismatch'
    saved = np.load(path, mmap_mode='r')
    assert saved.shape == (5000, 2048) and saved.dtype == np.float16
    assert np.isfinite(saved).all()
    assert np.max(np.abs(np.linalg.norm(saved.astype(np.float32), axis=1) - 1)) < 0.01
    assert 1 <= info['best_epoch'] <= info['epochs_run'] <= config['max_epochs']
    assert np.isfinite(info['best_dev_loss'])
    return info


def run(s, x, graphs, manifest, variant, seed, smoke=False):
    config = make_config(s, manifest, variant, seed, smoke)
    name = f'{variant}__s{seed}'
    outdir = root(smoke) / 'outputs'
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{name}.npy'
    graph = VARIANTS[variant]
    fit_pairs, dev_pairs = graphs['fit'][graph], graphs['dev'][graph]
    if len(dev_pairs) < 200:
        raise RuntimeError(f'{name}: dev {graph} has {len(dev_pairs)} pairs (<200); stop, no substitute')
    if not len(fit_pairs):
        raise RuntimeError(f'{name}: empty fit graph')
    old = verify_output(path, config)
    if old is not None:
        print(f'{name}: verified identical output; skip', flush=True)
        return old
    np.random.seed(seed)
    torch.manual_seed(seed)
    order_rng = np.random.default_rng(seed)
    negative_rng = np.random.default_rng(seed)
    model = Student().cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    xf = torch.from_numpy(np.ascontiguousarray(x[graphs['fit']['node_ids']]))
    xd = torch.from_numpy(np.ascontiguousarray(x[graphs['dev']['node_ids']]))
    nf = adjacency(graphs['fit']['N'], len(xf)) if variant == 'S_PN' else None
    nd = adjacency(graphs['dev']['N'], len(xd)) if variant == 'S_PN' else None
    history, steps = [], 0
    best, best_epoch, stale, best_state = float('inf'), 0, 0, None
    start = time.monotonic()
    for epoch in range(1, config['max_epochs'] + 1):
        tick = time.monotonic()
        model.train()
        order = order_rng.permutation(len(fit_pairs))
        total = 0.
        for a in range(0, len(order), CONFIG['batch_size']):
            pairs = fit_pairs[order[a:a + CONFIG['batch_size']]]
            opt.zero_grad(set_to_none=True)
            loss = pair_loss(model, xf, pairs, nf, negative_rng)
            assert torch.isfinite(loss), 'nonfinite train loss'
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(pairs)
            steps += 1
        train_loss = total / len(fit_pairs)
        model.eval()
        total = 0.
        # Fixed pair order, batch boundaries and negative draws on every epoch.
        dev_rng = np.random.default_rng(seed)
        with torch.no_grad():
            for a in range(0, len(dev_pairs), CONFIG['batch_size']):
                pairs = dev_pairs[a:a + CONFIG['batch_size']]
                total += float(pair_loss(model, xd, pairs, nd, dev_rng)) * len(pairs)
        dev_loss = total / len(dev_pairs)
        assert np.isfinite(dev_loss)
        history.append(dict(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss,
                            seconds=time.monotonic() - tick))
        if dev_loss < best:
            best, best_epoch, stale = dev_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if smoke or epoch % 10 == 0:
            print(f'{name}: epoch={epoch} train={train_loss:.6f} dev={dev_loss:.6f} '
                  f'steps={steps} seconds={history[-1]["seconds"]:.2f}', flush=True)
        if stale >= CONFIG['patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    outputs = np.empty((5000, 2048), dtype=np.float16)
    with torch.no_grad():
        for a in range(0, 5000, CONFIG['batch_size']):
            xx = torch.from_numpy(np.ascontiguousarray(x[s['subset_pid'][a:a + CONFIG['batch_size']]]))
            outputs[a:a + len(xx)] = model(xx).numpy().astype(np.float16)
    assert np.isfinite(outputs).all()
    assert np.max(np.abs(np.linalg.norm(outputs.astype(np.float32), axis=1) - 1)) < 0.01
    pending = path.with_name(path.stem + '.pending.npy')
    np.save(pending, outputs)
    info = dict(config=config, best_epoch=best_epoch, epochs_run=epoch, steps=steps,
        best_dev_loss=best, final_train_loss=train_loss, final_dev_loss=dev_loss,
        fit_size=len(xf), dev_size=len(xd), history=history,
        seconds=time.monotonic() - start, output_sha256=sha_file(pending))
    atomic_json(path.with_suffix('.json'), info)
    pending.replace(path)
    verify_output(path, config)
    print(f'{name}: saved; best_epoch={best_epoch}, best_dev={best:.6f}, '
          f'seconds={info["seconds"]:.2f}', flush=True)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--seeds', type=int, nargs='+', choices=[0, 1, 2], default=[0, 1, 2])
    p.add_argument('--smoke', action='store_true', help='isolated /tmp; 2000 fit; two epochs; seed 0; all variants')
    args = p.parse_args()
    if args.smoke and args.variants != list(VARIANTS):
        p.error('--smoke runs all four variants')
    torch.set_num_threads(8)
    print('CPU only; labels never enter training or early stopping.', flush=True)
    s = load_frozen()
    graphs, manifest = load_graphs(s, args.smoke)
    # Fail before any training when any requested dev graph is too small.
    for variant in args.variants:
        n = len(graphs['dev'][VARIANTS[variant]])
        if n < 200:
            raise RuntimeError(f'{variant}: only {n} dev pairs (<200); stopping')
    x = default_images(s)[0]
    for side in ('fit', 'dev'):
        actual = hashlib.sha256(np.ascontiguousarray(x[graphs[side]['node_ids']]).tobytes()).hexdigest()
        assert actual == manifest['source_sha256'][f'{side}.I'], 'image inputs changed since graph construction'
    for variant in args.variants:
        for seed in ([0] if args.smoke else args.seeds):
            run(s, x, graphs, manifest, variant, seed, args.smoke)


if __name__ == '__main__':
    main()
