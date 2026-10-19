"""Separate and shared-trunk cross-modal heads; CPU stage-1 parity and CUDA training."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
GRAPH_BASE = Path('/project/CoSiR/src/test/20261018_aspect_graph_sheet')
SMOKE = Path('/tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/02')
for folder in ('20261017_image_transfer_pilot', '20261018_purified_buddy_pilot',
               '20261018_aspect_graph_sheet'):
    sys.path.insert(0, str(HERE.parent / folder))
import ags_head as head
import itp_data
import itp_train
from ms_rows import ASPECTS, ROWS, validate

CONFIG = itp_train.CONFIG


class TrunkSide(nn.Module):
    def __init__(self, aspects):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({a: nn.Linear(1024, 2048) for a in aspects})


class TrunkHead(nn.Module):
    """A view onto shared parameters; its state contains just this head and its trunk."""
    def __init__(self, side, aspect):
        super().__init__()
        self.trunk, self.output = side.trunk, side.heads[aspect]

    def forward(self, x):
        return F.normalize(self.output(self.trunk(x)), dim=1)


def build_model(row, aspects, device):
    """The only model factory; preserve side/layer creation and optimizer parameter order."""
    if row['share'] == 'separate':
        assert len(aspects) == 1
        models = nn.ModuleDict({side: itp_train.Student().to(device) for side in ('img', 'cap')})
        return models, {aspects[0]: models}
    assert row['share'] == 'trunk'
    models = nn.ModuleDict({side: TrunkSide(aspects).to(device) for side in ('img', 'cap')})
    views = {a: nn.ModuleDict({side: TrunkHead(models[side], a) for side in ('img', 'cap')})
             for a in aspects}
    return models, views


def symmetric(x, y, w=None):
    if x.device.type == 'cpu':
        return head.symmetric(x, y, w)
    logits = x @ y.T / CONFIG['temperature']
    labels = torch.arange(len(x), device=x.device)
    ce = (F.cross_entropy(logits, labels, reduction='none') +
          F.cross_entropy(logits.T, labels, reduction='none')) / 2
    return ce.mean() if w is None else (w * ce).sum() / w.sum()


def head_loss(models, img, cap, pairs, weights, row, aspect):
    """One head's loss; ticket 03 extends forms/teachers here, leaving the loop alone."""
    assert row['form'] == 'a' and row['graph_terms'] and row['distill'] == 'none'
    if img.device.type == 'cpu':
        return head.pair_loss(models, img, cap, pairs, weights)
    a = torch.as_tensor(pairs[:, 0].copy(), device=img.device)
    b = torch.as_tensor(pairs[:, 1].copy(), device=img.device)
    ia, ib = models['img'](torch.cat((img[a], img[b]))).chunk(2)
    ca, cb = models['cap'](torch.cat((cap[a], cap[b]))).chunk(2)
    w = torch.as_tensor(weights, device=img.device)
    buddy = (symmetric(ia, ib, w) + symmetric(ia, cb, w) +
             symmetric(ca, ib, w) + symmetric(ca, cb, w)) / 4
    own = (symmetric(ia, ca) + symmetric(ib, cb)) / 2
    return buddy + own


def code_version():
    paths = (Path(__file__), HERE / 'ms_rows.py', Path(head.__file__),
             Path(itp_train.__file__), Path(itp_data.__file__))
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def make_config(s, row, name, aspect, aspects, manifests, seed, device, smoke, max_epochs):
    graph_row = row['graphs'][aspect]
    manifest = manifests[aspect]
    config = head.make_config(s, manifest, aspect, graph_row, seed, smoke)
    config.update(row=copy.deepcopy(row), name=name, graph_row=graph_row,
                  active_aspects=list(aspects), device=str(device), max_epochs=max_epochs,
                  code_version=code_version(),
                  graph_archives={a: manifests[a]['archives'][a] for a in aspects})
    return config


def output_path(outdir, name, aspect, graph_row, seed):
    return Path(outdir) / name / f'{aspect}__{graph_row or "none"}__s{seed}.npz'


def save_output(s, view, inputs, state, path, config, start):
    view.load_state_dict(state['best_state'])
    view.eval()
    outputs = {}
    device = torch.device(config['device'])
    with torch.no_grad():
        for side, model in view.items():
            out = np.empty((5000, 2048), dtype=np.float16)
            for a in range(0, 5000, CONFIG['batch_size']):
                ids = s['subset_pid'][a:a + CONFIG['batch_size']]
                xx = torch.from_numpy(np.ascontiguousarray(inputs[side][ids])).to(device)
                out[a:a + len(ids)] = model(xx).cpu().numpy().astype(np.float16)
            head.check_view(out)
            outputs[side] = out
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, **outputs)
    info = dict(config=config, best_epoch=state['best_epoch'], epochs_run=len(state['history']),
                best_dev_loss=state['best'], history=state['history'],
                seconds=time.monotonic() - start, output_sha256=head.sha_file(pending))
    head.atomic_json(path.with_suffix('.json'), info)
    pending.replace(path)
    head.verify_output(path, config)
    print(f'{path.stem}: saved; best_epoch={info["best_epoch"]}; seconds={info["seconds"]:.2f}', flush=True)
    return info


def run(s, inputs, graphs, manifests, row, name, aspects, seed, device, outdir,
        smoke=False, max_epochs=200):
    """One independent separate head, or one joint cohort with per-head epoch clocks."""
    validate(row)
    assert aspects and aspects == [a for a in ASPECTS if a in aspects]
    assert row['share'] != 'separate' or len(aspects) == 1
    assert 1 <= max_epochs <= CONFIG['max_epochs']
    configs = {a: make_config(s, row, name, a, aspects, manifests, seed, device, smoke, max_epochs)
               for a in aspects}
    paths = {a: output_path(outdir, name, a, row['graphs'][a], seed) for a in aspects}
    old = {a: head.verify_output(paths[a], configs[a]) for a in aspects}
    if all(old.values()):
        print(f'{name} s{seed} {aspects}: verified identical outputs; skip', flush=True)
        return old
    # A partial shared run replays the whole cohort, preserving its shared-layer trajectory.
    np.random.seed(seed)
    torch.manual_seed(seed)
    rngs = {a: np.random.default_rng(seed if len(aspects) == 1 else [seed, ASPECTS.index(a)])
            for a in aspects}
    models, views = build_model(row, aspects, device)
    opt = torch.optim.AdamW(models.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    tensors, states = {}, {}
    for aspect in aspects:
        graph = graphs[aspect]
        tensors[aspect] = {
            part: {side: torch.from_numpy(np.ascontiguousarray(x[graph[f'{part}_node_ids']])).to(device)
                   for side, x in inputs[aspect].items()} for part in ('fit', 'dev')}
        fit_pairs, fit_w = graph['rows'][row['graphs'][aspect]]['fit']
        dev_pairs, dev_w = graph['rows'][row['graphs'][aspect]]['dev']
        if smoke:
            fit_pairs, fit_w = fit_pairs[:5000], fit_w[:5000]
        states[aspect] = dict(fit_pairs=fit_pairs, fit_w=fit_w, dev_pairs=dev_pairs, dev_w=dev_w,
                              history=[], best=float('inf'), best_epoch=0, stale=0, best_state=None,
                              epoch=1, pos=0, total=0., order=None, tick=None, active=True)
    start, step = time.monotonic(), 0
    while any(st['active'] for st in states.values()):
        models.train()
        active = [a for a in aspects if states[a]['active']]
        for aspect in active:
            st = states[aspect]
            if st['order'] is None:
                st['tick'] = time.monotonic()
                st['order'] = rngs[aspect].permutation(len(st['fit_pairs']))
        opt.zero_grad(set_to_none=True)
        losses, sizes = [], {}
        for aspect in active:
            st = states[aspect]
            ix = st['order'][st['pos']:st['pos'] + CONFIG['batch_size']]
            loss = head_loss(views[aspect], **tensors[aspect]['fit'],
                             pairs=st['fit_pairs'][ix], weights=st['fit_w'][ix], row=row, aspect=aspect)
            assert torch.isfinite(loss), 'nonfinite train loss'
            losses.append(loss)
            sizes[aspect] = len(ix)
        # Do not introduce even an add-to-zero operation on the single-head parity path.
        loss = losses[0]
        for other in losses[1:]:
            loss = loss + other
        loss.backward()
        opt.step()
        step += 1
        for aspect, value in zip(active, losses):
            st = states[aspect]
            st['total'] += float(value.detach()) * sizes[aspect]
            st['pos'] += sizes[aspect]
        for aspect in active:
            st = states[aspect]
            if st['pos'] < len(st['order']):
                continue
            models.eval()
            total = 0.
            with torch.no_grad():
                for a in range(0, len(st['dev_pairs']), CONFIG['batch_size']):
                    ps = st['dev_pairs'][a:a + CONFIG['batch_size']]
                    ws = st['dev_w'][a:a + CONFIG['batch_size']]
                    total += float(head_loss(views[aspect], **tensors[aspect]['dev'],
                                             pairs=ps, weights=ws, row=row, aspect=aspect)) * len(ps)
            dev_loss = total / len(st['dev_pairs'])
            assert np.isfinite(dev_loss)
            st['history'].append(dict(epoch=st['epoch'], step=step,
                                      train_loss=st['total'] / len(st['fit_pairs']), dev_loss=dev_loss,
                                      seconds=time.monotonic() - st['tick']))
            if dev_loss < st['best']:
                st['best'], st['best_epoch'], st['stale'] = dev_loss, st['epoch'], 0
                st['best_state'] = copy.deepcopy(views[aspect].state_dict())
            else:
                st['stale'] += 1
            h = st['history'][-1]
            print(f'{name}/{aspect} s{seed}: epoch={st["epoch"]} step={step} '
                  f'train={h["train_loss"]:.6f} dev={dev_loss:.6f} seconds={h["seconds"]:.2f}', flush=True)
            st['active'] = st['stale'] < CONFIG['patience'] and st['epoch'] < max_epochs
            st.update(epoch=st['epoch'] + 1, pos=0, total=0., order=None)
    result = dict(old)
    for aspect in aspects:
        if old[aspect] is None:
            result[aspect] = save_output(s, views[aspect], inputs[aspect], states[aspect],
                                         paths[aspect], configs[aspect], start)
    return result


def selected_rows(args):
    if args.rows is not None:
        assert not args.set and args.name is None, '--set/--name require --row'
        assert len(set(args.rows)) == len(args.rows)
        return {name: validate(copy.deepcopy(ROWS[name])) for name in args.rows}
    assert args.name, '--row requires --name'
    row = copy.deepcopy(ROWS[args.row])
    for assignment in args.set:
        key, value = assignment.split('=', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # Bare enum/graph strings are the CLI shorthand.
        target = row
        parts = key.split('.')
        for part in parts[:-1]:
            target = target[part]
        assert parts[-1] in target, f'unknown row field: {key}'
        target[parts[-1]] = value
    return {args.name: validate(row)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--rows', nargs='+', choices=list(ROWS))
    mode.add_argument('--row', choices=list(ROWS))
    p.add_argument('--set', nargs='+', action='extend', default=[], metavar='KEY=VALUE')
    p.add_argument('--name')
    p.add_argument('--seeds', type=int, nargs='+', choices=[0, 1, 2], default=[0])
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--aspects', nargs='+', choices=list(ASPECTS))
    p.add_argument('--outdir', type=Path)
    p.add_argument('--smoke', action='store_true', help='scratchpad outputs; first 5000 fit pairs; 2 epochs by default')
    p.add_argument('--max-epochs', type=int)
    args = p.parse_args()
    rows = selected_rows(args)
    assert all(name and name not in ('.', '..') and
               all(c.isalnum() or c in '-_.' for c in name) for name in rows), 'invalid output name'
    assert len(set(args.seeds)) == len(args.seeds)
    aspects = [a for a in ASPECTS if args.aspects is None or a in args.aspects]
    if args.aspects is not None:
        assert len(set(args.aspects)) == len(args.aspects)
        assert all(row['share'] == 'separate' for row in rows.values()), '--aspects is separate-only'
    max_epochs = args.max_epochs if args.max_epochs is not None else (2 if args.smoke else CONFIG['max_epochs'])
    assert 1 <= max_epochs <= CONFIG['max_epochs']
    outdir = args.outdir if args.outdir is not None else (SMOKE if args.smoke else HERE) / 'outputs'
    if args.smoke:
        assert outdir.resolve().is_relative_to(SMOKE.resolve()), '--smoke outputs must be in the scratchpad'
    torch.set_num_threads(8)
    device = torch.device(args.device)
    s = head.load_frozen()
    assert s['subset_pid'].shape == (5000,) and len(np.unique(s['subset_pid'])) == 5000
    inputs = head.load_inputs(s, aspects[0], 'default')
    for name, row in rows.items():
        graphs, manifests = {}, {}
        for aspect in aspects:
            graphs[aspect], manifests[aspect] = head.load_graph(s, GRAPH_BASE, aspect, [row['graphs'][aspect]])
        cohorts = [[a] for a in aspects] if row['share'] == 'separate' else [aspects]
        for seed in args.seeds:
            for cohort in cohorts:
                run(s, {a: inputs for a in cohort}, graphs, manifests, row, name, cohort,
                    seed, device, outdir, args.smoke, max_epochs)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
