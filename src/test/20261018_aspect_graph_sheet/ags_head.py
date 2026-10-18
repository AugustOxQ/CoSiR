"""CPU cross-modal graph heads and cached 5,000-painting AMI sheets."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
for folder in ('20261017_image_transfer_pilot', '20261018_purified_buddy_pilot'):
    sys.path.insert(0, str(HERE.parent / folder))
from itp_data import (atomic_json, default_images, fingerprint, load_frozen,
                      prompted, split, targets)
from itp_train import CONFIG, Student
from pbp_graphs import label_vectors, sha_file
from pav_score import ami, partitions, summary

SMOKE = Path('/tmp/claude-0/-project-CoSiR/a3bf264f-92cc-43df-a16f-1f6fb0f2917c/scratchpad/ags_fu_smoke')
ASPECTS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
LABELS = ('emotion', 'style', 'genre')
LOSS = 'mean(4 weighted symmetric buddy CE)+mean(2 unweighted own-caption CE); v1'
SCORING = 'pav partitions k20 seeds012; y>=0; student population sd only with 3 seeds; v1'


def root(smoke):
    return SMOKE if smoke else HERE


def atomic_text(path, value):
    pending = path.with_name(path.stem + '.pending' + path.suffix)
    pending.write_text(value)
    pending.replace(path)


def check_view(x):
    assert x.shape == (5000, 2048) and np.isfinite(x).all()
    assert np.max(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1)) < .01


def load_graph(s, base, aspect, rows=None):
    folder = base / 'graphs'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['identity']['sample_sha256'] == fingerprint(s), 'graph sample mismatch'
    assert manifest['identity']['method'], 'missing graph method'
    path = folder / f'{aspect}.npz'
    assert sha_file(path) == manifest['archives'][aspect], 'graph archive checksum mismatch'
    _, fit, dev = split(s)
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z['fit_node_ids'], fit), 'fit node order mismatch'
        assert np.array_equal(z['dev_node_ids'], dev), 'dev node order mismatch'
        available = [k[:-5] for k in z.files if k.endswith('__fit')]
        assert available and len(set(available)) == len(available)
        result = dict(fit_node_ids=fit, dev_node_ids=dev, rows={})
        for row in rows if rows is not None else available:
            result['rows'][row] = {}
            for side, nodes in (('fit', fit), ('dev', dev)):
                pairs, weights = z[f'{row}__{side}'], z[f'{row}__{side}_w']
                assert pairs.dtype == np.int64 and pairs.ndim == 2 and pairs.shape[1] == 2
                assert len(pairs) > 0 and (pairs[:, 0] >= 0).all()
                assert (pairs[:, 0] < pairs[:, 1]).all() and (pairs[:, 1] < len(nodes)).all()
                codes = pairs[:, 0] * len(nodes) + pairs[:, 1]
                assert (np.diff(codes) > 0).all(), 'pairs must be sorted and unique'
                assert weights.dtype == np.float32 and weights.shape == (len(pairs),)
                assert np.isfinite(weights).all() and ((weights > 0) & (weights <= 1)).all()
                assert manifest['counts'][aspect][side][row] == len(pairs), 'pair count mismatch'
                result['rows'][row][side] = (pairs, weights)
            assert len(result['rows'][row]['dev'][0]) >= 200, 'too few dev pairs'
    return result, manifest


def selected_rows(base, aspects, picked=False):
    path = base / ('head_seed_rows.json' if picked else 'graphs/selected.json')
    rows = json.loads(path.read_text())
    for aspect in aspects:
        assert aspect in rows and rows[aspect] and len(rows[aspect]) == len(set(rows[aspect]))
    return {aspect: rows[aspect] for aspect in aspects}


def make_config(s, manifest, aspect, row, seed, smoke, input_key='default'):
    counts = {side: manifest['counts'][aspect][side][row] for side in ('fit', 'dev')}
    return dict(CONFIG, aspect=aspect, row=row, seed=seed, smoke=smoke, device='cpu', threads=8,
                input_img=input_key, input_cap=input_key, sample_sha256=fingerprint(s), loss=LOSS,
                graph_identity=manifest['identity'], graph_sha256=manifest['archives'][aspect],
                fit_pair_count=counts['fit'], dev_pair_count=counts['dev'],
                fit_pairs_used=min(5000, counts['fit']) if smoke else counts['fit'],
                max_epochs=2 if smoke else CONFIG['max_epochs'])


def verify_output(path, config):
    """The npz seals completion; incompatible existing outputs crash on resume."""
    if not path.exists():
        return None
    info = json.loads(path.with_suffix('.json').read_text())
    assert info['config'] == config, f'{path}: resume config mismatch'
    assert sha_file(path) == info['output_sha256'], f'{path}: output checksum mismatch'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img', 'cap'}
        for side in ('img', 'cap'):
            assert z[side].dtype == np.float16
            check_view(z[side])
    assert 1 <= info['best_epoch'] <= info['epochs_run'] <= config['max_epochs']
    assert len(info['history']) == info['epochs_run'] and np.isfinite(info['best_dev_loss'])
    return info


def symmetric(x, y, w=None):
    logits = x @ y.T / CONFIG['temperature']
    labels = torch.arange(len(x))
    ce = (F.cross_entropy(logits, labels, reduction='none') +
          F.cross_entropy(logits.T, labels, reduction='none')) / 2
    return ce.mean() if w is None else (w * ce).sum() / w.sum()


def pair_loss(models, img, cap, pairs, weights):
    a, b = torch.from_numpy(pairs[:, 0].copy()), torch.from_numpy(pairs[:, 1].copy())
    # Concatenate endpoints to call each MLP once; dropout is independent per element.
    ia, ib = models['img'](torch.cat((img[a], img[b]))).chunk(2)
    ca, cb = models['cap'](torch.cat((cap[a], cap[b]))).chunk(2)
    w = torch.from_numpy(weights)
    buddy = (symmetric(ia, ib, w) + symmetric(ia, cb, w) +
             symmetric(ca, ib, w) + symmetric(ca, cb, w)) / 4
    own = (symmetric(ia, ca) + symmetric(ib, cb)) / 2
    return buddy + own


def run(s, inputs, graph, manifest, aspect, row, seed, smoke, time_epoch=False,
        input_key='default', outdir='outputs'):
    config = make_config(s, manifest, aspect, row, seed, smoke, input_key)
    name = f'{aspect}__{row}' + ('' if input_key == 'default' else '__prompted') + f'__s{seed}'
    outdir = root(smoke) / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{name}.npz'
    if not time_epoch:
        old = verify_output(path, config)
        if old is not None:
            print(f'{name}: verified identical output; skip', flush=True)
            return old
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    models = torch.nn.ModuleDict({side: Student().cpu() for side in ('img', 'cap')})
    opt = torch.optim.AdamW(models.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    tensors = {part: {side: torch.from_numpy(np.ascontiguousarray(x[graph[f'{part}_node_ids']]))
                      for side, x in inputs.items()} for part in ('fit', 'dev')}
    fit_pairs, fit_w = graph['rows'][row]['fit']
    dev_pairs, dev_w = graph['rows'][row]['dev']
    if smoke and not time_epoch:
        fit_pairs, fit_w = fit_pairs[:5000], fit_w[:5000]
    history = []
    best, best_epoch, stale, best_state = float('inf'), 0, 0, None
    start = time.monotonic()
    for epoch in range(1, (1 if time_epoch else config['max_epochs']) + 1):
        tick = time.monotonic()
        models.train()
        order = rng.permutation(len(fit_pairs))
        total = 0.
        for a in range(0, len(order), CONFIG['batch_size']):
            ix = order[a:a + CONFIG['batch_size']]
            opt.zero_grad(set_to_none=True)
            loss = pair_loss(models, **tensors['fit'], pairs=fit_pairs[ix], weights=fit_w[ix])
            assert torch.isfinite(loss), 'nonfinite train loss'
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ix)
        train_loss = total / len(fit_pairs)
        train_seconds = time.monotonic() - tick
        models.eval()
        total = 0.
        with torch.no_grad():
            for a in range(0, len(dev_pairs), CONFIG['batch_size']):
                ps = dev_pairs[a:a + CONFIG['batch_size']]
                ws = dev_w[a:a + CONFIG['batch_size']]
                total += float(pair_loss(models, **tensors['dev'], pairs=ps, weights=ws)) * len(ps)
        dev_loss = total / len(dev_pairs)
        assert np.isfinite(dev_loss)
        history.append(dict(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss,
                            train_seconds=train_seconds, seconds=time.monotonic() - tick))
        if dev_loss < best:
            best, best_epoch, stale = dev_loss, epoch, 0
            best_state = copy.deepcopy(models.state_dict())
        else:
            stale += 1
        print(f'{name}: epoch={epoch} train={train_loss:.6f} dev={dev_loss:.6f} '
              f'seconds={history[-1]["seconds"]:.2f}', flush=True)
        if stale >= CONFIG['patience']:
            break
    if time_epoch:
        timing = dict(aspect=aspect, row=row, seed=seed, fit_pairs=len(fit_pairs), dev_pairs=len(dev_pairs),
                      batch_size=CONFIG['batch_size'], threads=8, device='cpu', history=history,
                      seconds=time.monotonic() - start, graph_sha256=config['graph_sha256'])
        atomic_json(root(smoke) / 'head_full_epoch_timing.json', timing)
        return timing
    models.load_state_dict(best_state)
    models.eval()
    outputs = {}
    with torch.no_grad():
        for side, model in models.items():
            out = np.empty((5000, 2048), dtype=np.float16)
            for a in range(0, 5000, CONFIG['batch_size']):
                ids = s['subset_pid'][a:a + CONFIG['batch_size']]
                xx = torch.from_numpy(np.ascontiguousarray(inputs[side][ids]))
                out[a:a + len(ids)] = model(xx).numpy().astype(np.float16)
            check_view(out)
            outputs[side] = out
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, **outputs)
    info = dict(config=config, best_epoch=best_epoch, epochs_run=epoch, best_dev_loss=best,
                history=history, seconds=time.monotonic() - start, output_sha256=sha_file(pending))
    atomic_json(path.with_suffix('.json'), info)
    pending.replace(path)
    verify_output(path, config)
    print(f'{name}: saved; best_epoch={best_epoch}; seconds={info["seconds"]:.2f}', flush=True)
    return info


def ami_scores(x, labels):
    x = np.asarray(x, dtype=np.float32)
    check_view(x)
    ls = partitions(x, 20)
    return {label: summary([ami(y[y >= 0], lab[y >= 0]) for lab in ls])
            for label, y in labels.items()}


def target_ami(scores, aspect):
    return (scores['img'][aspect]['mean'] + scores['cap'][aspect]['mean']) / 2


def aggregate(r):
    out = {}
    for aspect, rows in r['selected'].items():
        out[aspect] = {}
        for row in rows:
            runs = [v for v in r['outputs'].values() if v['config']['aspect'] == aspect and
                    v['config']['row'] == row and v.get('complete')]
            if not runs:
                continue
            def stats(xs):
                return dict(mean=float(np.mean(xs)), sd=float(np.std(xs)) if len(xs) == 3 else None)
            out[aspect][row] = dict(seeds=sorted(v['config']['seed'] for v in runs),
                ami={side: {label: stats([v['ami'][side][label]['mean'] for v in runs])
                            for label in LABELS} for side in ('img', 'cap')},
                target_ami=stats([v['target_ami'] for v in runs]),
                fit_pairs=runs[0]['config']['fit_pair_count'], dev_pairs=runs[0]['config']['dev_pair_count'],
                epochs={str(v['config']['seed']): v['epochs_run'] for v in runs},
                best_epochs={str(v['config']['seed']): v['best_epoch'] for v in runs})
    return out


def fmt(v):
    return f"{v['mean']:.6f}" + (f" ± {v['sd']:.6f}" if v['sd'] is not None else '')


def markdown(r):
    lines = ['# Aspect graph head sheet' + (' (SMOKE)' if r['identity']['smoke'] else ''), '',
             'AMI k=20: three k-means seeds per view. Students: mean over runs; population sd only with '
             'three student seeds. References: sd over k-means seeds. Target = mean of both sides for the aspect label.', '']
    cols = ' | '.join(f'{side}/{label}' for side in ('img', 'cap') for label in LABELS)
    for aspect, rows in r['selected'].items():
        lines += [f'## {aspect}', '', f'| Row | Seeds | {cols} | Target AMI | Fit/dev pairs | Epochs (best) |',
                  '|---|---|' + '---:|' * 6 + '---:|---:|---|']
        for row in rows:
            g = r['aggregates'].get(aspect, {}).get(row)
            if g is None:
                lines.append(f'| {row} | pending | ' + '— | ' * 8 + '— |')
                continue
            epochs = ', '.join(f"s{k}:{v} ({g['best_epochs'][k]})" for k, v in g['epochs'].items())
            lines.append(f"| {row} | {','.join(map(str, g['seeds']))} | " +
                         ' | '.join(fmt(g['ami'][side][label]) for side in ('img', 'cap') for label in LABELS) +
                         f" | {fmt(g['target_ami'])} | {g['fit_pairs']}/{g['dev_pairs']} | {epochs} |")
        lines += ['', '| Reference | Emotion AMI | Style AMI | Genre AMI |', '|---|---:|---:|---:|']
        for key in ('default', ASPECTS[aspect]):
            for side in ('img', 'cap'):
                v = r['references'].get(f'{key}/{side}')
                if v:
                    lines.append(f'| {key}/{side} | ' + ' | '.join(fmt(v['ami'][label]) for label in LABELS) + ' |')
        lines.append('')
    lines += [f"Missing outputs: {', '.join(r['missing_outputs']) or 'none'}.", '']
    return '\n'.join(lines)


def score(s, base, aspects, smoke):
    start = time.monotonic()
    selected = selected_rows(base, aspects)
    labels = label_vectors(s, s['subset_pid'])
    identity = dict(sample_sha256=fingerprint(s), smoke=smoke, scoring=SCORING,
                    labels_sha256=hashlib.sha256(b''.join(s[k].tobytes() for k in ('emo', 'sty', 'gen'))).hexdigest())
    path = base / 'ags_head_results.json'
    r = json.loads(path.read_text()) if path.exists() else dict(
        schema=1, identity=identity, selected={}, outputs={}, references={}, missing_outputs=[])
    assert r['schema'] == 1 and r['identity'] == identity, 'incompatible score cache'
    r['selected'].update(selected)
    def save():
        r['aggregates'] = aggregate(r)
        atomic_json(path, r)
        atomic_text(base / 'ags_head_tables.md', markdown(r))
    missing = []
    picked_path = base / 'head_seed_rows.json'
    picked = json.loads(picked_path.read_text()) if picked_path.exists() else {}
    for aspect, rows in selected.items():
        graph, manifest = load_graph(s, base, aspect, rows)
        for row in rows:
            for seed in (0, 1, 2):
                name = f'{aspect}__{row}__s{seed}'
                output = base / 'outputs' / f'{name}.npz'
                config = make_config(s, manifest, aspect, row, seed, smoke)
                meta = verify_output(output, config)
                if meta is None:
                    if seed == 0 or (not smoke and row in picked.get(aspect, [])):
                        missing.append(name)
                    assert name not in r['outputs'], 'previously scored output disappeared'
                    continue
                v = r['outputs'].setdefault(name, dict(config=config, source_sha256=meta['output_sha256'],
                    epochs_run=meta['epochs_run'], best_epoch=meta['best_epoch'], ami={}))
                assert v['config'] == config and v['source_sha256'] == meta['output_sha256'], 'stale output score'
                if v.get('complete'):
                    print(f'{name}: score cached', flush=True)
                    continue
                with np.load(output, allow_pickle=False) as z:
                    for side in ('img', 'cap'):
                        if side not in v['ami']:
                            v['ami'][side] = ami_scores(z[side], labels)
                            save()
                v['target_ami'] = target_ami(v['ami'], aspect)
                v['complete'] = True
                save()
                print(f'{name}: target AMI={v["target_ami"]:.6f}', flush=True)
    for key in dict.fromkeys(['default'] + [ASPECTS[a] for a in aspects]):
        for side in ('img', 'cap'):
            name = f'{key}/{side}'
            # Verify source hashes on resume; cached rescoring does no k-means.
            if side == 'img':
                x = default_images(s)[0][s['subset_pid']] if key == 'default' else prompted(s, key)[0]
            else:
                x = targets(s, key)[s['subset_pid']]
            source = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            if name in r['references']:
                assert r['references'][name]['source_sha256'] == source, 'reference changed'
                continue
            r['references'][name] = dict(source_sha256=source, ami=ami_scores(x, labels))
            save()
            print(f'{name}: reference scored', flush=True)
    r['missing_outputs'] = [name for name in r['missing_outputs'] if name.split('__')[0] not in aspects] + missing
    save()
    print(f'Score complete: {time.monotonic() - start:.2f}s; {path}', flush=True)


def pick_seeds(s, base, aspects, smoke):
    r = json.loads((base / 'ags_head_results.json').read_text())
    assert r['identity']['sample_sha256'] == fingerprint(s) and r['identity']['smoke'] == smoke
    assert r['identity']['scoring'] == SCORING
    selected = selected_rows(base, aspects)
    picked = {}
    for aspect, rows in selected.items():
        assert len(rows) >= 2
        values = {}
        graph, manifest = load_graph(s, base, aspect, rows)
        for row in rows:
            name = f'{aspect}__{row}__s0'
            v = r['outputs'][name]
            config = make_config(s, manifest, aspect, row, 0, smoke)
            meta = verify_output(base / 'outputs' / f'{name}.npz', config)
            assert meta and v.get('complete') and v['config'] == config
            assert v['source_sha256'] == meta['output_sha256']
            values[row] = target_ami(v['ami'], aspect)
        # Stable ties retain graph selection order.
        picked[aspect] = sorted(rows, key=lambda row: -values[row])[:2]
    path = base / 'head_seed_rows.json'
    if path.exists():
        previous = json.loads(path.read_text())
        previous.update(picked)
        picked = previous
    atomic_json(path, picked)
    print('Seed rows: ' + json.dumps(picked), flush=True)


def load_inputs(s, aspect, input_kind):
    assert input_kind in ('default', 'prompted')
    if input_kind == 'default':
        inputs = dict(img=default_images(s)[0], cap=targets(s, 'default'))
    else:
        from ags_trainimg import aspect_images
        key = ASPECTS[aspect]
        inputs = dict(img=aspect_images(s, key), cap=targets(s, key))
    for x in inputs.values():
        assert x.shape == (19889, 2048) and x.dtype == np.float32 and np.isfinite(x).all()
    return inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--score', action='store_true')
    mode.add_argument('--pick-seeds', action='store_true')
    mode.add_argument('--time-epoch', action='store_true', help='one full-size epoch; timing only in smoke folder')
    p.add_argument('--aspects', nargs='+', choices=list(ASPECTS))
    p.add_argument('--rows', nargs='+', help='default: graphs/selected.json rows')
    p.add_argument('--seed-rows', action='store_true', help='train head_seed_rows.json instead of selected.json')
    p.add_argument('--seeds', type=int, nargs='+', choices=[0, 1, 2], default=[0])
    p.add_argument('--smoke', action='store_true', help='isolated scratchpad; first 5000 fit pairs, 2 epochs')
    args = p.parse_args()
    if args.time_epoch and not args.smoke:
        p.error('--time-epoch requires --smoke')
    if (args.score or args.pick_seeds) and (args.rows or args.seed_rows):
        p.error('--score/--pick-seeds use graphs/selected.json')
    torch.set_num_threads(8)
    assert not torch.cuda.is_available(), 'set CUDA_VISIBLE_DEVICES to empty'
    s = load_frozen()
    base = root(args.smoke)
    aspects = args.aspects or (['emotion'] if args.smoke else list(ASPECTS))
    if args.score:
        score(s, base, aspects, args.smoke)
    elif args.pick_seeds:
        pick_seeds(s, base, aspects, args.smoke)
    else:
        rows = {a: args.rows for a in aspects} if args.rows else selected_rows(base, aspects, args.seed_rows)
        graphs = {a: load_graph(s, base, a, rs) for a, rs in rows.items()}
        inputs = load_inputs(s, aspects[0], 'default')
        for aspect, rs in rows.items():
            graph, manifest = graphs[aspect]
            for row in rs:
                for seed in args.seeds:
                    run(s, inputs, graph, manifest, aspect, row, seed, args.smoke, args.time_epoch)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
