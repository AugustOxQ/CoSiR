"""Fixed image-only student; no label access. Controller runs full GPU jobs."""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from itp_data import (HERE, SMOKE, TRAIN_IMAGES, atomic_json, default_images,
                      emotion_images, fingerprint, load_frozen, split, targets)

CONFIG = dict(architecture=[2048, 1024, 2048], activation='GELU', dropout=0.1,
              temperature=0.05, optimizer='AdamW', lr=1e-3, weight_decay=0.05,
              batch_size=512, max_epochs=200, patience=15, dev_seed=0, dev_size=1489)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 2048))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def loss_fn(pred, target):
    assert not target.requires_grad
    logits = pred @ target.T / CONFIG['temperature']
    labels = torch.arange(len(pred), device=pred.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def run(s, x, y, input_key, target_key, seed, device, smoke=False):
    outdir = SMOKE / 'train' if smoke else HERE / 'outputs'
    outdir.mkdir(parents=True, exist_ok=True)
    name = f'{input_key}__{target_key}__s{seed}'
    path = outdir / f'{name}.npy'
    meta_path = outdir / f'{name}.json'
    config = dict(CONFIG, input=input_key, target=target_key, seed=seed, smoke=smoke,
                  sample_sha256=fingerprint(s), device=str(device))
    if smoke:
        config.update(max_epochs=3, fit_size=2000)
    if path.exists():
        import json
        old = json.loads(meta_path.read_text())
        # Changing CPU/GPU on resume does not change the fixed experiment.
        assert {k: v for k, v in old['config'].items() if k != 'device'} == {
            k: v for k, v in config.items() if k != 'device'}
        saved = np.load(path, mmap_mode='r')
        assert saved.shape == (5000, 2048) and saved.dtype == np.float16
        assert np.isfinite(saved).all()
        assert np.max(np.abs(np.linalg.norm(saved.astype(np.float32), axis=1) - 1)) < 0.01
        print(f'{name}: verified outputs; skip', flush=True)
        return old
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _, fit, dev = split(s)
    if smoke:
        fit = fit[:2000]
    assert not np.intersect1d(fit, s['subset_pid']).size
    assert not np.intersect1d(dev, s['subset_pid']).size
    model = Student().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device).detach()
    dev_index = torch.as_tensor(dev, device=device)
    best, best_epoch, stale, best_state = float('inf'), 0, 0, None
    history = []
    start = time.monotonic()
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = rng.permutation(fit)
        total = 0.0
        for a in range(0, len(order), CONFIG['batch_size']):
            ix = torch.as_tensor(order[a:a + CONFIG['batch_size']], device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[ix]), yt[ix])
            assert torch.isfinite(loss), 'nonfinite train loss'
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ix)
        train_loss = total / len(fit)
        model.eval()
        with torch.no_grad():
            dev_loss = float(loss_fn(model(xt[dev_index]), yt[dev_index]))
        assert np.isfinite(dev_loss)
        history.append(dict(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss))
        if dev_loss < best:
            best, best_epoch, stale = dev_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f'{name}: epoch={epoch} train={train_loss:.6f} dev={dev_loss:.6f} best={best:.6f}', flush=True)
        if stale >= CONFIG['patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    outputs = np.empty((len(s['subset_pid']), 2048), dtype=np.float16)
    with torch.no_grad():
        for a in range(0, len(outputs), CONFIG['batch_size']):
            ix = torch.as_tensor(s['subset_pid'][a:a + CONFIG['batch_size']], device=device)
            outputs[a:a + len(ix)] = model(xt[ix]).cpu().numpy().astype(np.float16)
    norms = np.linalg.norm(outputs.astype(np.float32), axis=1)
    assert np.isfinite(outputs).all() and np.max(np.abs(norms - 1)) < 0.01
    if smoke:
        assert history[-1]['train_loss'] < history[0]['train_loss'], history
    info = dict(config=config, best_epoch=best_epoch, epochs_run=epoch,
                final_train_loss=train_loss, best_dev_loss=best,
                seconds=time.monotonic() - start, fit_size=len(fit), dev_size=len(dev),
                history=history)
    # Output is the commit marker. No model state is written to disk.
    atomic_json(meta_path, info)
    pending = path.with_name(path.stem + '.pending.npy')
    np.save(pending, outputs)
    pending.replace(path)
    print(f'{name}: best_epoch={best_epoch} epochs={epoch} train={train_loss:.6f} '
          f'best_dev={best:.6f} seconds={info["seconds"]:.1f}; saved {path}', flush=True)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--inputs', nargs='+', choices=['default', 'emotion_focus'], default=None)
    p.add_argument('--targets', nargs='+', choices=['emotion_focus', 'default'], default=['emotion_focus', 'default'])
    p.add_argument('--seeds', type=int, nargs='+', choices=[0, 1, 2], default=[0, 1, 2])
    p.add_argument('--smoke', action='store_true', help='CPU only, seed 0, 2000 fit paintings, 3 epochs, isolated /tmp')
    args = p.parse_args()
    if args.smoke and (args.device != 'cpu' or args.inputs not in (None, ['default'])):
        p.error('--smoke requires --device cpu and default input')
    torch.set_num_threads(8)
    s = load_frozen()
    ys = {key: targets(s, key) for key in (['emotion_focus'] if args.smoke else args.targets)}
    for input_key in args.inputs or (['default'] if args.smoke else ['default', 'emotion_focus']):
        if input_key == 'emotion_focus' and not (TRAIN_IMAGES / 'done.json').exists():
            print('emotion_focus: no sealed training-image extraction; skip', flush=True)
            if args.inputs is not None:
                raise FileNotFoundError(TRAIN_IMAGES / 'done.json')
            continue
        x = default_images(s)[0] if input_key == 'default' else emotion_images(s)
        for target_key, y in ys.items():
            for seed in ([0] if args.smoke else args.seeds):
                run(s, x, y, input_key, target_key, seed, torch.device(args.device), args.smoke)


if __name__ == '__main__':
    main()
