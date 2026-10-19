"""Image-ranked model number sheet, diagnostics and queue selection manifests (CPU)."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import ms_score as sc
from ms_rows import ROWS

HERE = Path(__file__).resolve().parent


def stats(xs):
    return dict(mean=float(np.mean(xs)), sd=float(np.std(xs)) if len(xs) == 3 else None)


def heads(s, labels, outputs, rescore):
    targets = {a: sc.check_view(sc.prompted(s, sc.PROMPTS[a])[0]) for a in sc.ASPECTS}
    assert all(x.shape == (5000, 2048) for x in targets.values())
    runs = {}
    for folder in sorted(outputs.iterdir()):
        if not folder.is_dir() or folder.name in ('stage0', 'cache'):
            continue
        for path in sorted(folder.glob('*.npz')):
            if '.pending' in path.name:
                continue
            meta = json.loads(path.with_suffix('.json').read_text())
            c = meta['config']
            a, seed = c['aspect'], c['seed']
            assert a in sc.ASPECTS and seed in (0, 1, 2)
            assert c['name'] == folder.name and c['sample_sha256'] == sc.fingerprint(s)
            assert path.stem == f'{a}__{c["graph_row"] or "none"}__s{seed}'
            sha = sc.sha_file(path)
            assert sha == meta['output_sha256'], path
            assert 1 <= meta['best_epoch'] <= meta['epochs_run']
            identity = dict(output_sha256=sha, scoring=sc.SCORING, sklearn=sc.sklearn.__version__,
                            scorer_version=sc.SCORER_VERSION,
                            labels={k: sc.array_sha(v) for k, v in labels.items()},
                            target_sha256=sc.array_sha(targets[a]))
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cache = outputs / 'cache' / f'head_{key}.json'
            with np.load(path, allow_pickle=False) as z:
                assert set(z.files) == {'img', 'cap'}
                for side in ('img', 'cap'):
                    assert z[side].shape == (5000, 2048) and z[side].dtype == np.float16
                    sc.check_view(z[side])
                if cache.exists():
                    run = json.loads(cache.read_text())
                    assert run['identity'] == identity, f'stale head cache: {path}'
                else:
                    assert rescore, f'run --score first: {path}'
                    start = time.monotonic()
                    scores = {side: sc.measures(z[side], labels, outputs / 'cache') for side in ('img', 'cap')}
                    run = dict(identity=identity, scores=scores,
                               cosine=float(np.mean(np.sum(z['img'].astype(np.float32) * targets[a], axis=1))),
                               score_seconds=time.monotonic() - start)
                    sc.atomic_json(cache, run)
                    print(f'{folder.name}/{path.stem}: both sides {run["score_seconds"]:.2f}s', flush=True)
            run = dict(run, config=c, best_epoch=meta['best_epoch'])
            if 'mask' in meta:
                mask = np.asarray(meta['mask'], dtype=np.float32)
                assert mask.shape == (2048,) and np.isfinite(mask).all() and (mask >= 0).all() and np.linalg.norm(mask) > 0
                run['mask'] = mask.tolist()
            key = f'{folder.name}/{a}/s{seed}'
            assert key not in runs
            runs[key] = run
    return runs


def aggregate(runs):
    rows = {a: {} for a in sc.ASPECTS}
    for name in sorted({v['config']['name'] for v in runs.values()}):
        versions = sorted({v['config']['code_version'] for v in runs.values() if v['config']['name'] == name})
        assert len(versions) == 1, f'{name}: mixed code_version values: {versions}'
        for a in sc.ASPECTS:
            group = sorted([v for v in runs.values() if v['config']['name'] == name and v['config']['aspect'] == a],
                           key=lambda v: v['config']['seed'])
            if not group:
                continue
            assert len({json.dumps(v['config']['row'], sort_keys=True) for v in group}) == 1
            rows[a][name] = dict(seeds=[v['config']['seed'] for v in group],
                seed0_image_ami=next((v['scores']['img'][a]['ami']['mean'] for v in group
                                      if v['config']['seed'] == 0), None),
                scores={side: {label: {m: stats([v['scores'][side][label][m]['mean'] for v in group])
                         for m in sc.METRICS} for label in sc.ASPECTS} for side in ('img', 'cap')},
                cosine=stats([v['cosine'] for v in group]),
                best_epochs={str(v['config']['seed']): v['best_epoch'] for v in group},
                caption_untrained=group[0]['config']['row']['graph_terms'] is False,
                config=group[0]['config']['row'])
    return rows


def mask_overlap(runs):
    result = {}
    for left, right in itertools.combinations(runs.values(), 2):
        c, d = left['config'], right['config']
        if 'mask' not in left or 'mask' not in right or c['name'] != d['name'] or c['seed'] != d['seed']:
            continue
        if c['aspect'] == d['aspect']:
            continue
        x, y = np.array(left['mask']), np.array(right['mask'])
        result[f'{c["name"]}/s{c["seed"]}/{c["aspect"]}:{d["aspect"]}'] = dict(
            cosine=float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))), shared_active_dims=int(((x > 0) & (y > 0)).sum()))
    return result


def pick_seeds(runs, outputs):
    present = {(v['config']['name'], v['config']['aspect']) for v in runs.values() if v['config']['seed'] == 0}
    missing = [f'{name}/{a}/s0' for name in ROWS for a in sc.ASPECTS if (name, a) not in present]
    assert not missing, 'Missing canonical seed-0 heads:\n' + '\n'.join(missing)
    picks = {'A1', 'T1'}
    top = {}
    for a in sc.ASPECTS:
        candidates = {v['config']['name']: v['scores']['img'][a]['ami']['mean'] for v in runs.values()
                      if v['config']['aspect'] == a and v['config']['seed'] == 0
                      and v['config']['row']['inputs'] == 'default' and v['config']['name'] != 'D17'}
        top[a] = sorted(candidates, key=lambda n: (-candidates[n], n))[:2]
        picks.update(top[a])
    result = dict(rows=sorted(picks), seeds=[1, 2], top_by_aspect=top)
    sc.atomic_json(outputs / 'seed_rows.json', result)
    print('Seed rows: ' + ', '.join(result['rows']))


def d17(rows, outputs):
    wins = {}
    for name in ('T4', 'T5'):
        margins = []
        for a in sc.ASPECTS:
            t1, candidate = rows[a].get('T1'), rows[a].get(name)
            if t1 and candidate and t1['scores']['img'][a]['ami']['sd'] is not None:
                base = t1['scores']['img'][a]['ami']
                margin = candidate['scores']['img'][a]['ami']['mean'] - base['mean'] - base['sd']
                if margin > 0:
                    margins.append(margin)
        if len(margins) >= 2:
            wins[name] = sum(margins)
    result = None
    if wins:
        form = max(wins, key=lambda n: (wins[n], n == 'T4'))
        candidates = {n: float(np.mean([rows[a][n]['seed0_image_ami'] for a in sc.ASPECTS]))
                      for n in ('D2', 'D3', 'D4', 'D5')
                      if all(n in rows[a] and rows[a][n]['seed0_image_ami'] is not None for a in sc.ASPECTS)}
        assert candidates, 'D17 triggered but no complete D2-D5 row'
        base = max(candidates, key=lambda n: (candidates[n], n))
        result = dict(name='D17', base=base, set=['form=' + ('b' if form == 'T4' else 'c'),
                                                            'share=' + ('trunk' if form == 'T4' else 'mask')])
    sc.atomic_json(outputs / 'd17.json', result if result else 'none')
    print('D17: ' + json.dumps(result if result else 'none'))


def fmt(v):
    if v is None:
        return '—'
    return f'{v["mean"]:.6f}' + (f' ± {v["sd"]:.6f}' if v['sd'] is not None else '')


def sheet(s, labels, runs, rows, outputs):
    bars = {}
    for name, aspects in sc.bar_vectors(s, outputs / 'stage0').items():
        bars[name] = {a: sc.aggregate(x, labels, outputs / 'cache')[a] if x is not None else None
                      for a, x in aspects.items()}
    probe = sc.probe_ami()
    bars['linear_probe'] = {a: dict(ami=probe[a]) for a in sc.ASPECTS}
    threshold = bars['percept']['emotion']['ami']['mean'] + .01
    reached = any(v['config']['inputs'] == 'default' and
                  v['scores']['img']['emotion']['ami']['mean'] >= threshold for v in rows['emotion'].values())
    gate = (f'At least one default-input row has image-side emotion AMI ≥ {threshold:.6f}.' if reached else
            f'No default-input row has image-side emotion AMI ≥ {threshold:.6f}; '
            'the 10-29 emotion gate is at risk on development data.')
    tie = {a: rows[a].get('T1', {}).get('scores', {}).get('img', {}).get(a, {}).get('ami', {}).get('sd') for a in sc.ASPECTS}
    result = dict(sample_sha256=sc.fingerprint(s), scoring=sc.SCORING, rows=rows, bars=bars,
                  runs=runs, mask_overlap=mask_overlap(runs), tie_band=tie,
                  emotion_gate_reached=reached, emotion_gate_threshold=threshold)
    sc.atomic_json(outputs / 'ms_results.json', result)
    lines = ['# Model number sheet', '', gate, '',
             'Development exploration. Row means over model seeds; population SD shown only with three seeds. '
             'Bars use k-means SD for one vector or model-seed SD for three vectors. AMI decides all picks; form a stays on a tie.', '']
    for a in sc.ASPECTS:
        other = [b for b in sc.ASPECTS if b != a]
        lines += [f'## {a}', '', f'T1 tie band: {tie[a] if tie[a] is not None else "pending three seeds"}.', '',
            '| Index | Row | Seeds | Inputs | Image AMI | V | ARI | FMI | Caption AMI | ' +
            ' | '.join('Image AMI/' + b for b in other) + ' | Cosine | Best epochs |',
            '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for name, values in bars.items():
            v = values[a]
            cells = [fmt(v.get(m)) for m in sc.METRICS] if v else ['pending'] * 4
            lines.append(f'| Bars | {name} | — | — | ' + ' | '.join(cells) + ' | — | — | — | — | — |')
        for name in sorted(rows[a], key=lambda n: (-rows[a][n]['scores']['img'][a]['ami']['mean'], n)):
            v = rows[a][name]
            cells = [fmt(v['scores']['img'][a][m]) for m in sc.METRICS]
            cells += ['untrained' if v['caption_untrained'] else fmt(v['scores']['cap'][a]['ami'])]
            cells += [fmt(v['scores']['img'][b]['ami']) for b in other]
            cells += [fmt(v['cosine']), ', '.join(f's{k}:{ep}' for k, ep in v['best_epochs'].items())]
            lines.append(f'| Row | {name} | {",".join(map(str, v["seeds"]))} | {v["config"]["inputs"]} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Mask overlap', '', '```json', json.dumps(result['mask_overlap'], indent=2), '```', '']
    pending = outputs / 'ms_tables.pending.md'
    pending.write_text('\n'.join(lines))
    pending.replace(outputs / 'ms_tables.md')
    print(gate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=HERE / 'outputs')
    for flag in ('score', 'sheet', 'pick-seeds', 'd17'):
        p.add_argument('--' + flag, action='store_true')
    args = p.parse_args()
    assert args.score or args.sheet or args.pick_seeds or args.d17, 'choose an action'
    args.outputs.mkdir(parents=True, exist_ok=True)
    s = sc.load_frozen()
    labels = sc.label_vectors(s, s['subset_pid'])
    runs = heads(s, labels, args.outputs, args.score)
    rows = aggregate(runs)
    if args.sheet:
        sheet(s, labels, runs, rows, args.outputs)
    if args.pick_seeds:
        pick_seeds(runs, args.outputs)
    if args.d17:
        d17(rows, args.outputs)


if __name__ == '__main__':
    with threadpool_limits(8):
        main()
