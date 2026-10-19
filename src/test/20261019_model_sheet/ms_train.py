"""Model-sheet graph heads, masks and image distillation; CPU parity and CUDA training."""
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
SMOKE = Path('/tmp/claude-0/-project-CoSiR/56b9b751-6946-4c76-b52d-3761e520be8f/scratchpad/ms_smoke/03')
for folder in ('20261017_image_transfer_pilot', '20261018_purified_buddy_pilot',
               '20261018_aspect_graph_sheet'):
    sys.path.insert(0, str(HERE.parent / folder))
import ags_head as head
import ags_trainimg
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


class MaskHead(nn.Module):
    def __init__(self, side, mask):
        super().__init__()
        self.net, self.mask = side.net, mask

    def forward(self, x):
        return F.normalize(self.net(x) * F.relu(self.mask), dim=1)


def build_model(row, aspects, device):
    """The only model factory; preserve side/layer creation and optimizer parameter order."""
    if row['share'] == 'separate':
        assert len(aspects) == 1
        models = nn.ModuleDict({side: itp_train.Student().to(device) for side in ('img', 'cap')})
        return models, {aspects[0]: models}
    if row['share'] == 'mask':
        models = nn.ModuleDict({side: itp_train.Student().to(device) for side in ('img', 'cap')})
        models['masks'] = nn.ParameterDict({a: nn.Parameter(torch.empty(2048, device=device))
                                           for a in aspects})
        for a in aspects:
            nn.init.normal_(models['masks'][a], 0.9, 0.7)
        views = {a: nn.ModuleDict({side: MaskHead(models[side], models['masks'][a])
                                  for side in ('img', 'cap')}) for a in aspects}
        return models, views
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


def extra_symmetric(x, y, extra_y, extra_x, valid_a, valid_b, w):
    """Each anchor gets its own additional column, independently in both directions."""
    logits = x @ y.T / CONFIG['temperature']
    forward = (x * extra_y).sum(1) / CONFIG['temperature']
    backward = (y * extra_x).sum(1) / CONFIG['temperature']
    forward = torch.cat((logits, forward.masked_fill(~valid_a, -torch.inf)[:, None]), dim=1)
    backward = torch.cat((logits.T, backward.masked_fill(~valid_b, -torch.inf)[:, None]), dim=1)
    labels = torch.arange(len(x), device=x.device)
    ce = (F.cross_entropy(forward, labels, reduction='none') +
          F.cross_entropy(backward, labels, reduction='none')) / 2
    return (w * ce).sum() / w.sum()


def distill_loss(z, teacher, ids, kind):
    assert teacher.shape == z.shape and not teacher.requires_grad
    same = ids[:, None] == ids[None, :]
    if kind == 'direct':
        labels = torch.arange(len(z), device=z.device)
        same.fill_diagonal_(False)
        logits = (z @ teacher.T / 0.05).masked_fill(same, -torch.inf)
        return F.cross_entropy(logits, labels)
    assert kind == 'relational'
    assert (~same).any(dim=1).all(), 'relational rows need other paintings'
    student = (z @ z.T / 0.05).masked_fill(same, -torch.inf)
    target = (teacher @ teacher.T / 0.05).masked_fill(same, -torch.inf)
    logp, logq = F.log_softmax(target, dim=1), F.log_softmax(student, dim=1)
    # Zero the excluded log values before multiplication: 0 * -inf would be NaN.
    return (logp.exp() * (logp.masked_fill(same, 0) - logq.masked_fill(same, 0))).sum(1).mean()


def head_loss(models, img, cap, pairs, weights, row, aspect, teacher=None, extras=None):
    """Graph, teacher and mask terms; leave the original CPU form-a call untouched."""
    if (row['form'] == 'a' and row['graph_terms'] and row['distill'] == 'none'
            and img.device.type == 'cpu'):
        return head.pair_loss(models, img, cap, pairs, weights)
    if row['graph_terms']:
        a, b = torch.as_tensor(pairs, device=img.device).unbind(1)
        ids = torch.cat((a, b))
        z = models['img'](img[ids])
        ia, ib = z.chunk(2)
        ca, cb = models['cap'](cap[ids]).chunk(2)
        w = torch.as_tensor(weights, device=img.device)
        if row['form'] == 'b':
            assert extras is not None and extras.shape == (len(a), 2)
            # The same sampled node per endpoint is embedded on the term's column side.
            ex = torch.cat((extras[:, 0], extras[:, 1]))
            ei_a, ei_b = models['img'](img[ex.clamp_min(0)]).chunk(2)
            ec_a, ec_b = models['cap'](cap[ex.clamp_min(0)]).chunk(2)
            va, vb = extras[:, 0] >= 0, extras[:, 1] >= 0
            buddy = (extra_symmetric(ia, ib, ei_a, ei_b, va, vb, w) +
                     extra_symmetric(ia, cb, ec_a, ei_b, va, vb, w) +
                     extra_symmetric(ca, ib, ei_a, ec_b, va, vb, w) +
                     extra_symmetric(ca, cb, ec_a, ec_b, va, vb, w)) / 4
        else:
            buddy = (symmetric(ia, ib, w) + symmetric(ia, cb, w) +
                     symmetric(ca, ib, w) + symmetric(ca, cb, w)) / 4
        own = (symmetric(ia, ca) + symmetric(ib, cb)) / 2
        loss = buddy + own
        if row['distill'] != 'none':
            loss = loss + row['distill_weight'] * distill_loss(z, teacher[ids], ids, row['distill'])
    else:
        ids = torch.as_tensor(pairs, device=img.device)
        z = models['img'](img[ids])
        loss = row['distill_weight'] * distill_loss(z, teacher[ids], ids, row['distill'])
    if row['form'] == 'c':
        loss = loss + 5e-4 * F.relu(models['img'].mask).abs().sum()
    return loss


def disjoint_split(s):
    _, fit, dev = itp_data.split(s)
    evaluation = s['subset_pid']
    for ids, n in ((fit, 13400), (dev, 1489), (evaluation, 5000)):
        assert ids.dtype == np.int64 and len(ids) == len(np.unique(ids)) == n
    assert not np.intersect1d(fit, dev).size
    assert not np.intersect1d(fit, evaluation).size
    assert not np.intersect1d(dev, evaluation).size
    assert len(np.unique(np.concatenate((fit, dev, evaluation)))) == len(s['first'])
    return fit, dev


def negative_candidates(graphs, row, aspect, part):
    """CSR candidate lists in this head's node order, joined by painting ID."""
    from scipy.sparse import csr_matrix
    nodes = graphs[aspect][f'{part}_node_ids']
    assert len(np.unique(nodes)) == len(nodes)
    lookup = {int(pid): i for i, pid in enumerate(nodes)}
    adjacency = {}
    for other in ASPECTS:
        graph = graphs[other]
        other_nodes = graph[f'{part}_node_ids']
        assert len(np.unique(other_nodes)) == len(nodes)
        assert set(other_nodes) == set(nodes), f'{part}: different graph node sets'
        mapping = np.asarray([lookup[int(pid)] for pid in other_nodes], dtype=np.int64)
        pairs = mapping[graph['rows'][row['graphs'][other]][part][0]]
        assert pairs.ndim == 2 and pairs.shape[1] == 2
        src, dst = np.concatenate((pairs[:, 0], pairs[:, 1])), np.concatenate((pairs[:, 1], pairs[:, 0]))
        adjacency[other] = csr_matrix((np.ones(len(src), dtype=bool), (src, dst)),
                                      shape=(len(nodes), len(nodes)))
    own = adjacency[aspect]
    others = [a for a in ASPECTS if a != aspect]
    union = adjacency[others[0]].maximum(adjacency[others[1]])
    candidates = (union.astype(np.int8) - union.multiply(own).astype(np.int8)).tocsr()
    candidates.setdiag(0)
    candidates.eliminate_zeros()
    candidates.sort_indices()
    assert not candidates.multiply(own).nnz and not candidates.diagonal().any()
    return candidates.indptr, candidates.indices


def draw_extras(csr, pairs, rng, device):
    """Sample outside the training step, then keep all negative indices on its device."""
    ptr, nodes = csr
    starts, counts = ptr[pairs], ptr[pairs + 1] - ptr[pairs]
    offsets = rng.integers(np.maximum(counts, 1))
    result = np.full(pairs.shape, -1, dtype=np.int64)
    valid = counts > 0
    result[valid] = nodes[(starts + offsets)[valid]]
    return torch.from_numpy(result).to(device)


def code_version():
    paths = (Path(__file__), HERE / 'ms_rows.py', Path(head.__file__),
             Path(itp_train.__file__), Path(itp_data.__file__), Path(ags_trainimg.__file__))
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def make_config(s, row, name, aspect, aspects, manifests, seed, device, smoke, max_epochs):
    graph_row = row['graphs'][aspect] if row['graphs'] is not None else 'none'
    input_key = head.ASPECTS[aspect] if row['inputs'] == 'instructed' else 'default'
    if row['graphs'] is None:
        manifest = dict(identity='none', archives={aspect: 'none'},
                        counts={aspect: {part: {'none': 0} for part in ('fit', 'dev')}})
    else:
        manifest = manifests[aspect]
    config = head.make_config(s, manifest, aspect, graph_row, seed, smoke, input_key=input_key)
    config.update(row=copy.deepcopy(row), name=name, graph_row=graph_row,
                  active_aspects=list(aspects), device=str(device), max_epochs=max_epochs,
                  code_version=code_version(),
                  graph_archives={a: manifests[a]['archives'][a] for a in manifests})
    if not row['graph_terms']:
        config.update(batch_size=1024, fit_painting_count=13400, dev_painting_count=1489)
    if row['form'] != 'a' or not row['graph_terms'] or row['distill'] != 'none':
        terms = [f'form {row["form"]} buddy+own-caption'] if row['graph_terms'] else []
        if row['distill'] != 'none':
            terms.append(f'{row["distill_weight"]} * {row["distill"]} image distillation T=0.05')
        if row['form'] == 'c':
            terms.append('5e-4 * L1(relu(mask))')
        config['loss'] = ' + '.join(terms)
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
    if config['row']['form'] == 'c':
        mask = F.relu(view['img'].mask).detach().cpu().numpy()
        assert mask.shape == (2048,) and np.isfinite(mask).all() and (mask >= 0).all()
        info['mask'] = mask.tolist()
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
    paths = {a: output_path(outdir, name, a, configs[a]['graph_row'], seed) for a in aspects}
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
    if row['share'] == 'mask':
        masks = list(models['masks'].parameters())
        mask_ids = {id(p) for p in masks}
        params = [p for p in models.parameters() if id(p) not in mask_ids]
        opt = torch.optim.AdamW([dict(params=params), dict(params=masks, weight_decay=0.0)],
                                lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    else:
        opt = torch.optim.AdamW(models.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    tensors, states = {}, {}
    batch_size = CONFIG['batch_size'] if row['graph_terms'] else 1024
    neg_rngs = {a: np.random.default_rng([seed, ASPECTS.index(a), 1]) for a in aspects}
    if row['distill'] != 'none':
        fit, dev = disjoint_split(s)  # Before reading any teacher row.
    for aspect in aspects:
        graph = graphs[aspect]
        tensors[aspect] = {
            part: {side: torch.from_numpy(np.ascontiguousarray(x[graph[f'{part}_node_ids']])).to(device)
                   for side, x in inputs[aspect].items()} for part in ('fit', 'dev')}
        if row['distill'] != 'none':
            for part, ids in (('fit', fit), ('dev', dev)):
                assert set(graph[f'{part}_node_ids']) == set(ids)
            teacher = (inputs[aspect]['img'] if row['inputs'] == 'instructed' else
                       ags_trainimg.aspect_images(s, head.ASPECTS[aspect]))
            assert teacher.shape == (19889, 2048) and teacher.dtype == np.float32
            assert np.isfinite(teacher).all()
            for part in ('fit', 'dev'):
                ids = graph[f'{part}_node_ids']
                tensors[aspect][part]['teacher'] = torch.from_numpy(
                    np.ascontiguousarray(teacher[ids])).to(device)
            del teacher
        if row['graph_terms']:
            fit_pairs, fit_w = graph['rows'][row['graphs'][aspect]]['fit']
            dev_pairs, dev_w = graph['rows'][row['graphs'][aspect]]['dev']
            if smoke:
                fit_pairs, fit_w = fit_pairs[:5000], fit_w[:5000]
        else:
            fit_pairs, dev_pairs = np.arange(len(fit)), np.arange(len(dev))
            fit_w, dev_w = np.ones(len(fit), dtype=np.float32), np.ones(len(dev), dtype=np.float32)
        states[aspect] = dict(fit_pairs=fit_pairs, fit_w=fit_w, dev_pairs=dev_pairs, dev_w=dev_w,
                              history=[], best=float('inf'), best_epoch=0, stale=0, best_state=None,
                              epoch=1, pos=0, total=0., order=None, tick=None, active=True)
        if row['form'] == 'b' and row['graph_terms']:
            states[aspect]['candidates'] = {part: negative_candidates(graphs, row, aspect, part)
                                           for part in ('fit', 'dev')}
    start, step = time.monotonic(), 0
    while any(st['active'] for st in states.values()):
        models.train()
        active = [a for a in aspects if states[a]['active']]
        for aspect in active:
            st = states[aspect]
            if st['order'] is None:
                st['tick'] = time.monotonic()
                st['order'] = rngs[aspect].permutation(len(st['fit_pairs']))
                if row['form'] == 'b' and row['graph_terms']:
                    st['extras'] = draw_extras(st['candidates']['fit'], st['fit_pairs'],
                                              neg_rngs[aspect], device)
        opt.zero_grad(set_to_none=True)
        losses, sizes = [], {}
        for aspect in active:
            st = states[aspect]
            ix = st['order'][st['pos']:st['pos'] + batch_size]
            extra_args = {}
            if row['form'] == 'b' and row['graph_terms']:
                extra_args['extras'] = st['extras'][torch.as_tensor(ix, device=device)]
            loss = head_loss(views[aspect], **tensors[aspect]['fit'],
                             pairs=st['fit_pairs'][ix], weights=st['fit_w'][ix], row=row, aspect=aspect,
                             **extra_args)
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
            if row['form'] == 'b' and row['graph_terms']:
                dev_extras = draw_extras(st['candidates']['dev'], st['dev_pairs'],
                                        np.random.default_rng([10000, ASPECTS.index(aspect)]), device)
            with torch.no_grad():
                for a in range(0, len(st['dev_pairs']), batch_size):
                    ps = st['dev_pairs'][a:a + batch_size]
                    ws = st['dev_w'][a:a + batch_size]
                    extra_args = {}
                    if row['form'] == 'b' and row['graph_terms']:
                        extra_args['extras'] = dev_extras[a:a + batch_size]
                    total += float(head_loss(views[aspect], **tensors[aspect]['dev'],
                                             pairs=ps, weights=ws, row=row, aspect=aspect,
                                             **extra_args)) * len(ps)
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
    fit, dev = disjoint_split(s)
    inputs = head.load_inputs(s, aspects[0], 'default')
    for name, row in rows.items():
        graphs, manifests = {}, {}
        graph_aspects = list(ASPECTS) if row['form'] == 'b' else aspects
        for aspect in graph_aspects:
            if row['graphs'] is None:
                graphs[aspect] = dict(fit_node_ids=fit, dev_node_ids=dev)
            else:
                graphs[aspect], manifests[aspect] = head.load_graph(
                    s, GRAPH_BASE, aspect, [row['graphs'][aspect]])
        row_inputs = ({a: head.load_inputs(s, a, 'prompted') for a in aspects}
                      if row['inputs'] == 'instructed' else {a: inputs for a in aspects})
        cohorts = [[a] for a in aspects] if row['share'] == 'separate' else [aspects]
        for seed in args.seeds:
            for cohort in cohorts:
                run(s, {a: row_inputs[a] for a in cohort}, graphs, manifests, row, name, cohort,
                    seed, device, outdir, args.smoke, max_epochs)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
