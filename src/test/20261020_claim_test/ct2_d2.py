"""D2 at sample/full scale; the sheet loop with only its inline size gates lifted.

run is copied from ms_train.run: the teacher provider/row assertion, split,
config and saver are generalized; optimization, RNG streams and math are intact.
Sample configs retain the sheet recipe. Added fields are mode, names_sha256,
data_sha256, output_rows and stage2_code_sha256. Graph identity and archive
hashes differ because ct2_graphs saves only the selected row per aspect.
"""
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

import ct_d2
import ct2_data
import ct2_graphs

ms_train = ct_d2.ms_train
ASPECTS, ROWS = ct_d2.ASPECTS, ct_d2.ROWS
head = ms_train.head
_sheet_make_config = ms_train.make_config
HERE = Path(__file__).resolve().parent


def make_config(s, row, name, aspect, aspects, manifests, seed, device, smoke, max_epochs):
    config = _sheet_make_config(s, row, name, aspect, aspects, manifests, seed, device, smoke, max_epochs)
    config.update(mode=s['mode'], names_sha256=s['names_sha256'], data_sha256=s['data_sha256'],
                  output_rows=len(s['dev'] if s['mode'] == 'full' else s['subset_pid']),
                  stage2_code_sha256=head.sha_file(Path(__file__)))
    return config


def check_view(x, rows):
    assert x.shape == (rows, 2048) and x.dtype == np.float16 and np.isfinite(x).all()
    assert np.max(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1)) < 1e-3


def verify_output(path, config):
    """The sheet verifier with a config-bound output row count."""
    if not path.exists():
        return None
    info = json.loads(path.with_suffix('.json').read_text())
    assert info['config'] == config, f'{path}: resume config mismatch'
    assert head.sha_file(path) == info['output_sha256'], f'{path}: output checksum mismatch'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img', 'cap'}
        for side in ('img', 'cap'):
            check_view(z[side], config['output_rows'])
    assert 1 <= info['best_epoch'] <= info['epochs_run'] <= config['max_epochs']
    assert len(info['history']) == info['epochs_run'] and np.isfinite(info['best_dev_loss'])
    return info


def save_output(s, view, inputs, state, path, config, start):
    evaluation = s['dev'] if s['mode'] == 'full' else s['subset_pid']
    replacing = path.exists()
    view.load_state_dict(state['best_state'])
    view.eval()
    outputs = {}
    device = torch.device(config['device'])
    with torch.no_grad():
        for side, model in view.items():
            out = np.empty((len(evaluation), 2048), dtype=np.float16)
            for a in range(0, len(evaluation), ms_train.CONFIG['batch_size']):
                ids = evaluation[a:a + ms_train.CONFIG['batch_size']]
                xx = torch.from_numpy(np.ascontiguousarray(inputs[side][ids])).to(device)
                out[a:a + len(ids)] = model(xx).cpu().numpy().astype(np.float16)
            check_view(out, len(evaluation))
            outputs[side] = out
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, **outputs)
    info = dict(config=config, best_epoch=state['best_epoch'], epochs_run=len(state['history']),
                best_dev_loss=state['best'], history=state['history'],
                seconds=time.monotonic() - start, output_sha256=ms_train.head.sha_file(pending))
    if config['row']['form'] == 'c':
        mask = F.relu(view['img'].mask).detach().cpu().numpy()
        assert mask.shape == (2048,) and np.isfinite(mask).all() and (mask >= 0).all()
        info['mask'] = mask.tolist()
    ms_train.head.atomic_json(path.with_suffix('.json'), info)
    pending.replace(path)
    verify_output(path, config)
    print(f'{path}: {"replaced" if replacing else "saved"}; best_epoch={info["best_epoch"]}; seconds={info["seconds"]:.2f}', flush=True)
    return info


def run(s, inputs, graphs, manifests, row, name, aspects, seed, device, outdir,
        smoke=False, max_epochs=200):
    """One independent separate head, or one joint cohort with per-head epoch clocks."""
    ms_train.validate(row)
    assert aspects and aspects == [a for a in ms_train.ASPECTS if a in aspects]
    assert row['share'] != 'separate' or len(aspects) == 1
    assert 1 <= max_epochs <= ms_train.CONFIG['max_epochs']
    configs = {a: make_config(s, row, name, a, aspects, manifests, seed, device, smoke, max_epochs)
               for a in aspects}
    paths = {a: ms_train.output_path(outdir, name, a, configs[a]['graph_row'], seed) for a in aspects}
    # Verify existing files against their recorded config before replacing stale cohorts.
    old = {a: verify_output(paths[a], json.loads(paths[a].with_suffix('.json').read_text())['config'])
           if paths[a].exists() else None for a in aspects}
    if all(old[a] is not None and old[a]['config'] == configs[a] for a in aspects):
        print(f'{name} s{seed} {aspects}: verified identical outputs; skip', flush=True)
        return old
    # A missing or stale head replays and replaces the whole cohort.
    np.random.seed(seed)
    torch.manual_seed(seed)
    rngs = {a: np.random.default_rng(seed if len(aspects) == 1 else [seed, ms_train.ASPECTS.index(a)])
            for a in aspects}
    models, views = ms_train.build_model(row, aspects, device)
    if row['share'] == 'mask':
        masks = list(models['masks'].parameters())
        mask_ids = {id(p) for p in masks}
        params = [p for p in models.parameters() if id(p) not in mask_ids]
        opt = torch.optim.AdamW([dict(params=params), dict(params=masks, weight_decay=0.0)],
                                lr=ms_train.CONFIG['lr'], weight_decay=ms_train.CONFIG['weight_decay'])
    else:
        opt = torch.optim.AdamW(models.parameters(), lr=ms_train.CONFIG['lr'], weight_decay=ms_train.CONFIG['weight_decay'])
    tensors, states = {}, {}
    batch_size = ms_train.CONFIG['batch_size'] if row['graph_terms'] else 1024
    neg_rngs = {a: np.random.default_rng([seed, ms_train.ASPECTS.index(a), 1]) for a in aspects}
    if row['distill'] != 'none':
        fit, dev = s['fit'], s['dev']  # Before reading any teacher row.
    for aspect in aspects:
        graph = graphs[aspect]
        tensors[aspect] = {
            part: {side: torch.from_numpy(np.ascontiguousarray(x[graph[f'{part}_node_ids']])).to(device)
                   for side, x in inputs[aspect].items()} for part in ('fit', 'dev')}
        if row['distill'] != 'none':
            for part, ids in (('fit', fit), ('dev', dev)):
                assert set(graph[f'{part}_node_ids']) == set(ids)
            teacher = (inputs[aspect]['img'] if row['inputs'] == 'instructed' else
                       ct2_data.instructed_images(s, ms_train.head.ASPECTS[aspect]))
            assert teacher.shape == (len(s['first']), 2048) and teacher.dtype == np.float32
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
            states[aspect]['candidates'] = {part: ms_train.negative_candidates(graphs, row, aspect, part)
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
                    st['extras'] = ms_train.draw_extras(st['candidates']['fit'], st['fit_pairs'],
                                              neg_rngs[aspect], device)
        opt.zero_grad(set_to_none=True)
        losses, sizes = [], {}
        for aspect in active:
            st = states[aspect]
            ix = st['order'][st['pos']:st['pos'] + batch_size]
            extra_args = {}
            if row['form'] == 'b' and row['graph_terms']:
                extra_args['extras'] = st['extras'][torch.as_tensor(ix, device=device)]
            loss = ms_train.head_loss(views[aspect], **tensors[aspect]['fit'],
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
                dev_extras = ms_train.draw_extras(st['candidates']['dev'], st['dev_pairs'],
                                        np.random.default_rng([10000, ms_train.ASPECTS.index(aspect)]), device)
            with torch.no_grad():
                for a in range(0, len(st['dev_pairs']), batch_size):
                    ps = st['dev_pairs'][a:a + batch_size]
                    ws = st['dev_w'][a:a + batch_size]
                    extra_args = {}
                    if row['form'] == 'b' and row['graph_terms']:
                        extra_args['extras'] = dev_extras[a:a + batch_size]
                    total += float(ms_train.head_loss(views[aspect], **tensors[aspect]['dev'],
                                             pairs=ps, weights=ws, row=row, aspect=aspect,
                                             include_mask_penalty=False,
                                             **extra_args)) * len(ps)
            dev_loss = total / len(st['dev_pairs'])
            assert np.isfinite(dev_loss)
            st['history'].append(dict(epoch=st['epoch'], step=step,
                                      train_loss=st['total'] / len(st['fit_pairs']), dev_loss=dev_loss,
                                      seconds=time.monotonic() - st['tick']))
            if row['form'] == 'c':
                st['history'][-1]['mask_penalty'] = float((5e-4 * F.relu(views[aspect]['img'].mask).abs().sum()).detach())
            if dev_loss < st['best']:
                st['best'], st['best_epoch'], st['stale'] = dev_loss, st['epoch'], 0
                st['best_state'] = copy.deepcopy(views[aspect].state_dict())
            else:
                st['stale'] += 1
            h = st['history'][-1]
            print(f'{name}/{aspect} s{seed}: epoch={st["epoch"]} step={step} '
                  f'train={h["train_loss"]:.6f} dev={dev_loss:.6f} seconds={h["seconds"]:.2f}', flush=True)
            st['active'] = st['stale'] < ms_train.CONFIG['patience'] and st['epoch'] < max_epochs
            st.update(epoch=st['epoch'] + 1, pos=0, total=0., order=None)
    result = {}
    for aspect in aspects:
        result[aspect] = save_output(s, views[aspect], inputs[aspect], states[aspect],
                                     paths[aspect], configs[aspect], start)
    return result



def _sha_array(x):
    import hashlib
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def _graph_identity(s):
    return dict(method=ct2_graphs.METHOD, fit_size=len(s['fit']), dev_size=len(s['dev']),
                mode=s['mode'], sample_sha256=ms_train.itp_data.fingerprint(s),
                names_sha256=s['names_sha256'])


def _read_graphs(s, spaces, folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['schema'] == 1 and manifest['identity'] == _graph_identity(s), 'graph identity mismatch'
    assert manifest['source_sha256'] == {k: _sha_array(v) for k, v in spaces.items()}, 'graph source mismatch'
    graphs = {}
    for aspect, row in ct2_graphs.ROWS.items():
        path = folder / f'{aspect}.npz'
        assert head.sha_file(path) == manifest['archives'][aspect], 'graph archive checksum mismatch'
        graph = dict(rows={row: {}})
        with np.load(path, allow_pickle=False) as z:
            expected = {f'{p}_node_ids' for p in ('fit', 'dev')}
            for part in ('fit', 'dev'):
                nodes = z[f'{part}_node_ids']
                assert nodes.dtype == np.int64 and np.array_equal(nodes, s[part]), 'graph node order mismatch'
                pairs, weights = z[f'{row}__{part}'], z[f'{row}__{part}_w']
                assert pairs.dtype == np.int64 and pairs.ndim == 2 and pairs.shape[1] == 2
                assert len(pairs) > 0 and (pairs[:, 0] >= 0).all()
                assert (pairs[:, 0] < pairs[:, 1]).all() and (pairs[:, 1] < len(nodes)).all()
                assert (np.diff(pairs[:, 0] * len(nodes) + pairs[:, 1]) > 0).all()
                assert weights.dtype == np.float32 and weights.shape == (len(pairs),)
                assert np.isfinite(weights).all() and (weights == 1).all()
                assert len(pairs) == manifest['counts'][aspect][part][row], 'graph count mismatch'
                graph[f'{part}_node_ids'] = nodes
                graph['rows'][row][part] = pairs, weights
                expected.update((f'{row}__{part}', f'{row}__{part}_w'))
            assert set(z.files) == expected
        graphs[aspect] = graph
    return graphs, dict.fromkeys(ASPECTS, manifest)


def load_training_data(mode, root):
    """Load aligned providers and build or verify selected graphs under root/d2/graphs."""
    s = ct2_data.split2(mode)
    inputs = dict(img=ct2_data.default_images(s), cap=ct2_data.default_captions(s))
    spaces = dict(D=inputs['cap'], I=inputs['img'],
                  E=ct2_data.instructed_captions(s, 'emotion_focus'),
                  **{'style.img': ct2_data.instructed_images(s, 'style_word'),
                     'genre.img': ct2_data.instructed_images(s, 'genre_word')})
    s['data_sha256'] = {k: _sha_array(x) for k, x in spaces.items()}
    s['data_sha256'].update(img=s['data_sha256']['I'], cap=s['data_sha256']['D'],
                            **{'emotion.img': _sha_array(ct2_data.instructed_images(s, 'emotion_focus'))})
    s['data_sha256'].update({p: _sha_array(s[p]) for p in ('fit', 'dev', 'fit_pairs')})
    for x in spaces.values():
        assert x.shape == (len(s['first']), 2048) and x.dtype == np.float32 and np.isfinite(x).all()
    folder = Path(root) / 'd2/graphs'
    if not (folder / 'manifest.json').exists():
        graphs, manifest = ct2_graphs.build(spaces, s['fit'], s['dev'])
        manifest['identity'] = _graph_identity(s)
        ct2_graphs.save(graphs, manifest, folder)
    graphs, manifests = _read_graphs(s, spaces, folder)
    return s, dict.fromkeys(ASPECTS, inputs), graphs, manifests


def output_root(mode, root=None):
    assert mode in ('sample', 'full'), mode
    return Path(root) if root is not None else HERE / 'outputs/stage2' / ('s2a' if mode == 'sample' else 'full')


def _require_pass(device):
    # Ticket 07 supplies the host/device/code-bound stage 2a gate.
    import ct2_stage2a
    ct2_stage2a.require_pass(device=str(device))


@threadpool_limits.wrap(limits=8)
def train(mode, seeds=range(5), device='cuda', root=None):
    """Train D2 cohorts; preserve incomplete artifacts and retain each best view."""
    seeds = list(seeds)
    assert seeds and len(set(seeds)) == len(seeds) and all(s in range(5) for s in seeds)
    root, device = output_root(mode, root), torch.device(device)
    if mode == 'full':
        _require_pass(device)
    torch.set_num_threads(8)
    s, inputs, graphs, manifests = load_training_data(mode, root)
    original_save = save_output

    def save_weights(s, view, inputs, state, path, config, start):
        info = original_save(s, view, inputs, state, path, config, start)
        weight = ct_d2._weight_path(root, config['aspect'], config['seed'])
        weight.parent.mkdir(parents=True, exist_ok=True)
        pending = weight.with_suffix('.pending.pt')
        torch.save({key: value.detach().cpu() for key, value in state['best_state'].items()}, pending)
        pending.replace(weight)
        head.atomic_json(weight.with_suffix('.json'), dict(
            weights_sha256=head.sha_file(weight), output_sha256=info['output_sha256'],
            config=config, best_epoch=info['best_epoch'], best_dev_loss=info['best_dev_loss']))
        return info

    result = {}
    # Reuse stage 1's strict cohort resume checks and backup layout at either size.
    with patch.object(ms_train, 'make_config', make_config), \
            patch.object(head, 'verify_output', verify_output), \
            patch(__name__ + '.save_output', save_weights):
        for seed in seeds:
            verified = ct_d2._verified_cohort(s, manifests, seed, device, root)
            if verified is not None:
                result[seed] = verified
                continue
            assert not any((root / 'held/d2').glob('*.npz')), 'V4: cannot retrain after held outputs exist'
            ct_d2._preserve_cohort(root, seed)
            result[seed] = run(s, inputs, graphs, manifests, ROWS['D2'], 'D2', list(ASPECTS),
                               seed, device, root / 'd2/train', max_epochs=ms_train.CONFIG['max_epochs'])
    return result


def _weights_info(mode, root, aspect, seed):
    info = ct_d2._weights_info(root, aspect, seed)
    assert info['config']['mode'] == mode, 'weights mode mismatch'
    assert info['config']['names_sha256'] and info['config']['data_sha256']
    assert info['config']['stage2_code_sha256'] == head.sha_file(Path(__file__)), 'weights code mismatch'
    return info


def infer(mode, aspect, seed, names, side='img', *, device=None, root=None):
    """Replay the saved best view through ct_d2.infer, with its exact dtype/batches."""
    root = output_root(mode, root)
    weights = _weights_info(mode, root, aspect, seed)
    recorded = ct_d2._inference_device(weights, device)
    if mode == 'full':
        _require_pass(recorded)
    return ct_d2.infer(aspect, seed, names, side, device=device, outdir=root)


def write_held(mode, seeds=range(5), device=None, root=None):
    """Gate held reads, then use stage 1's inference, layout and sidecar checks."""
    root, seeds = output_root(mode, root), list(seeds)
    assert seeds and len(set(seeds)) == len(seeds) and all(s in range(5) for s in seeds)
    devices = set()
    for seed in seeds:
        for aspect in ASPECTS:
            info = _weights_info(mode, root, aspect, seed)
            devices.add(str(ct_d2._inference_device(info, device)))
    for recorded in sorted(devices):
        _require_pass(recorded)
    return ct_d2.write_held(seeds, device, root)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['sample', 'full'])
    parser.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    parser.add_argument('--device', help='training defaults to cuda; inference uses the weights device')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--held', action='store_true')
    args = parser.parse_args(argv)
    assert args.train or args.held, 'choose --train and/or --held'
    assert len(set(args.seeds)) == len(args.seeds)
    root = output_root(args.mode, args.root)
    if args.train:
        train(args.mode, args.seeds, args.device or 'cuda', root)
    if args.held:
        write_held(args.mode, args.seeds, args.device, root)


if __name__ == '__main__':
    main()
