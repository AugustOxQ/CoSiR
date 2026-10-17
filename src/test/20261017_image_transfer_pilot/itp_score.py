"""CPU scoring only. Labels enter here, never the student or its stopping rule.

Checkpoint each metric and resume completed views. Baseline AMI gates are diagnostic:
failures are recorded and printed, and scoring continues. Seed sd is population sd.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from itp_data import (HERE, PAV, PROMPTS, atomic_json, default_images, fingerprint,
                      l2, load_frozen, prompted, targets)
from apc_data import paint_first, paint_majority
from pav_score import ami, graph_stats, mutual_pairs, partitions, probe, summary

BASELINES = dict(default_image=('default', 'image'), ef_image=('emotion_focus', 'image'),
                 teacher=('emotion_focus', 'mean_caption'), control_target=('default', 'mean_caption'))
BAR = 'emotion AMI >= 0.12 AND teacher-student minus control-student > max(the two seed sds)'


def top20_neighbours(x):
    """Exact cosine top-20, blocked like pav_score.mutual_pairs; self excluded."""
    x = l2(np.asarray(x, dtype=np.float32))
    assert len(x) > 20 and np.isfinite(x).all()
    out = np.empty((len(x), 20), dtype=np.int64)
    for a in range(0, len(x), 256):
        b = min(a + 256, len(x))
        sim = x[a:b] @ x.T
        sim[np.arange(b - a), np.arange(a, b)] = -np.inf
        out[a:b] = np.argpartition(-sim, 19, axis=1)[:, :20]
    return out


def neighbour_overlap(x, reference):
    return overlap_indices(top20_neighbours(x), top20_neighbours(reference))


def overlap_indices(a, b):
    assert a.shape == b.shape and a.shape[1] == 20
    return float(np.mean(np.sum(np.any(a[:, :, None] == b[:, None, :], axis=2), axis=1) / 20))


def flatten(value, prefix=''):
    out = {}
    for k, v in value.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (float, int)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def metrics(v):
    return flatten({k: v[k] for k in ('painting_ami', 'emotion_probe', 'overlap', 'buddies') if k in v})


def aggregate(r):
    out = {}
    for inp in ('default', 'emotion_focus'):
        for target in ('emotion_focus', 'default'):
            names = [f'{inp}__{target}__s{seed}' for seed in (0, 1, 2)]
            rows = [metrics(r['views'][n]) for n in names if r['views'].get(n, {}).get('complete')]
            if rows:
                keys = set.intersection(*(set(row) for row in rows))
                out[f'{inp}__{target}'] = dict(n_seeds=len(rows), metrics={
                    k: summary([row[k] for row in rows]) for k in sorted(keys)})
    return out


def bars(r, emit=False):
    out = {}
    for inp, baseline in [('default', 'default_image'), ('emotion_focus', 'ef_image')]:
        teacher = r['aggregates'].get(f'{inp}__emotion_focus')
        control = r['aggregates'].get(f'{inp}__default')
        if not teacher or not control or min(teacher['n_seeds'], control['n_seeds']) != 3:
            continue
        t = teacher['metrics']['painting_ami.emotion.mean']
        c = control['metrics']['painting_ami.emotion.mean']
        gap, max_sd = t['mean'] - c['mean'], max(t['sd'], c['sd'])
        passed = t['mean'] >= 0.12 and gap > max_sd
        changes = {}
        if baseline in r['views']:
            for label in ('genre', 'style'):
                changes[label] = teacher['metrics'][f'painting_ami.{label}.mean']['mean'] - r['views'][baseline]['painting_ami'][label]['mean']
        out[inp] = dict(pass_bar=passed, student=t, control=c, difference=gap, max_sd=max_sd,
                        ami_change_relative_to=baseline, ami_changes=changes)
        if emit:
            print(f"BAR {inp} {'PASS' if passed else 'FAIL'}: student={t['mean']:.6f}±{t['sd']:.6f} "
                  f"control={c['mean']:.6f}±{c['sd']:.6f} gap={gap:.6f} max_sd={max_sd:.6f} "
                  f"threshold=0.120000; genre_delta={changes.get('genre', float('nan')):+.6f} "
                  f"style_delta={changes.get('style', float('nan')):+.6f} vs {baseline}", flush=True)
    return out


def markdown(r):
    lines = ['# Image transfer pilot', '', f'Prespecified bar: {BAR}.', '',
             'Baseline AMI sd is over three k-means seeds. Student AMI is the mean of each run\'s three k-means scores, then mean ± population sd over student seeds. Other student scalars are also aggregated across student seeds. Partial groups are marked by their seed count.', '']
    columns = {}
    for name in BASELINES:
        v = r['views'].get(name)
        if v:
            columns[name] = {k: f'{val:.6f}' for k, val in metrics(v).items()}
    for name, v in r['aggregates'].items():
        columns[f"{name} ({v['n_seeds']}/3)"] = {k: f"{val['mean']:.6f} ± {val['sd']:.6f}" for k, val in v['metrics'].items()}
    sections = [
        ('Painting AMI (k=20)', [(label.title(), f'painting_ami.{label}.mean')
                                 for label in ('emotion', 'genre', 'style')]),
        ('Emotion probe and neighbour overlap', [
            ('Emotion accuracy', 'emotion_probe.accuracy.mean'),
            ('Teacher overlap', 'overlap.teacher'), ('Control overlap', 'overlap.control_target')]),
        ('Single-view buddy graph', [('Pairs', 'buddies.n_pairs')] + [
            (f'{label.title()} lift', f'buddies.labels.{label}.lift') for label in ('emotion', 'genre', 'style')])]
    for title, fields in sections:
        lines += [f'## {title}', '', '| View | ' + ' | '.join(label for label, _ in fields) + ' |',
                  '|---|' + '---:|' * len(fields)]
        for name, row in columns.items():
            cells = []
            for _, key in fields:
                cell = row.get(key, '—')
                if name in BASELINES and key.endswith('.mean'):
                    spread = row.get(key[:-4] + 'sd')
                    if spread is not None:
                        cell += ' ± ' + spread
                cells.append(cell)
            lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## All scalar metrics', '']
    keys = sorted(set().union(*(set(row) for row in columns.values()))) if columns else []
    lines += ['| Metric | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * len(columns)]
    for key in keys:
        lines.append('| ' + key + ' | ' + ' | '.join(row.get(key, '—') for row in columns.values()) + ' |')
    lines += ['', f"Top-20 chance overlap: {r['chance_overlap']:.8f} (20 / 4,999).", '',
              'Each metric is shown, including within-run k-means/fold sd and graph denominators. A student row for a within-run sd reports its mean and sd across student seeds.', '']
    for name, gate in r.get('consistency', {}).items():
        lines.append(f"Consistency {name}: {'PASS' if gate['ok'] else 'FAIL'}; max AMI mean delta={gate['max_mean_delta']:.8f}.")
    for name, v in r.get('bars', {}).items():
        lines += ['', f"BAR {name}: {'PASS' if v['pass_bar'] else 'FAIL'}; student={v['student']['mean']:.6f}; control={v['control']['mean']:.6f}; gap={v['difference']:.6f}; max sd={v['max_sd']:.6f}; genre/style changes versus {v['ami_change_relative_to']}: {v['ami_changes']}."]
    if not r.get('bars'):
        lines += ['', 'Bar pending: needs all three teacher and control student seeds for an input.']
    return '\n'.join(lines) + '\n'


def consistency(name, values, previous):
    key, side = BASELINES[name]
    old = previous['views'][key]['painting_ami'][side]
    comparison = {label: dict(current=values[label], previous=old[label],
        mean_delta=abs(values[label]['mean'] - old[label]['mean']),
        sd_delta=abs(values[label]['sd'] - old[label]['sd'])) for label in values}
    # The requested per-label gate compares the reported mean painting AMI.
    # Keep seed-sd deltas as diagnostics; store re-encoding can change partitions.
    ok = all(v['mean_delta'] <= 0.003 for v in comparison.values())
    print(f"consistency {name}: {'PASS' if ok else 'FAIL'} " + '; '.join(
        f"{label}={v['current']['mean']:.6f} vs {v['previous']['mean']:.6f} delta={v['mean_delta']:.8f}"
        for label, v in comparison.items()), flush=True)
    return dict(ok=ok, tolerance=0.003, statistic='mean', comparison=comparison,
                max_mean_delta=max(v['mean_delta'] for v in comparison.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baselines-only', action='store_true')
    p.add_argument('--views', nargs='+', choices=list(BASELINES), help='reduced baseline check; students omitted')
    p.add_argument('--results', type=Path, default=HERE / 'itp_results.json')
    p.add_argument('--tables', type=Path, help='default: itp_tables.md next to --results')
    args = p.parse_args()
    start = time.monotonic()
    print(f'Prespecified bar: {BAR}', flush=True)
    s = load_frozen()
    label_hash = hashlib.sha256(b''.join(s[k].tobytes() for k in ('emo', 'gen', 'sty'))).hexdigest()
    run_id = dict(sample_sha256=fingerprint(s), labels_sha256=label_hash,
                  prompts={k: PROMPTS[k] for k in ('default', 'emotion_focus')},
                  conventions='pav_score functions unchanged; k=20; three k-means seeds; population sd; GroupKFold(5)')
    r = json.loads(args.results.read_text()) if args.results.exists() else dict(
        schema=1, identity=run_id, chance_overlap=20 / 4999, views={}, consistency={})
    assert r['schema'] == 1 and r['identity'] == run_id, 'incompatible score cache'
    table_path = args.tables or args.results.with_name('itp_tables.md')
    def save():
        r['aggregates'] = aggregate(r)
        r['bars'] = bars(r)
        atomic_json(args.results, r)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pending = table_path.with_name(table_path.stem + '.pending.md')
        pending.write_text(markdown(r))
        pending.replace(table_path)
    subset, count = s['subset_pid'], len(s['first'])
    labels = dict(emotion=paint_majority(s['emo'], s['pid'], count, 9)[subset],
                  genre=paint_first(s['gen'], s['pid'], count)[subset],
                  style=paint_first(s['sty'], s['pid'], count)[subset])
    baseline = {}
    for key, image_name, target_name in [('default', 'default_image', 'control_target'),
                                         ('emotion_focus', 'ef_image', 'teacher')]:
        image, txt = prompted(s, key)
        baseline[image_name] = default_images(s, image)[0][subset] if key == 'default' else np.asarray(image, dtype=np.float32)
        baseline[target_name] = targets(s, key, txt)[subset]
    neighbours = {key: top20_neighbours(baseline[key]) for key in ('teacher', 'control_target')}
    previous = json.loads((PAV / 'results.json').read_text())
    def score(name, x, source=None):
        v = r['views'].setdefault(name, {})
        if source is not None:
            assert v.get('source_sha256', source) == source, f'{name}: output changed since scoring'
            v['source_sha256'] = source
        if v.get('complete'):
            print(f'{name}: scored; skip', flush=True)
            if name in BASELINES:
                r['consistency'][name] = consistency(name, v['painting_ami'], previous)
                save()
            return
        print(f'[{time.monotonic() - start:.1f}s] {name}', flush=True)
        assert x.shape == (5000, 2048) and np.isfinite(x).all()
        if 'painting_ami' not in v:
            ls = partitions(x, 20)
            v['painting_ami'] = {label: summary([ami(y[y >= 0], lab[y >= 0]) for lab in ls])
                                 for label, y in labels.items()}
            save()
        if name in BASELINES:
            r['consistency'][name] = consistency(name, v['painting_ami'], previous)
            save()
        if 'emotion_probe' not in v:
            v['emotion_probe'] = probe(x, labels['emotion'], subset)
            save()
        if 'overlap' not in v:
            nn = neighbours[name] if name in neighbours else top20_neighbours(x)
            v['overlap'] = {key: overlap_indices(nn, ref) for key, ref in neighbours.items()}
            save()
        if 'buddies' not in v:
            v['buddies'] = graph_stats(mutual_pairs(x), labels)
            save()
        v['complete'] = True
        save()
        print(f'{name}: complete; emotion AMI={v["painting_ami"]["emotion"]["mean"]:.6f}', flush=True)
    for name in args.views or BASELINES:
        score(name, baseline[name])
    if not args.baselines_only and not args.views:
        for inp in ('default', 'emotion_focus'):
            for target in ('emotion_focus', 'default'):
                for seed in (0, 1, 2):
                    name = f'{inp}__{target}__s{seed}'
                    path = HERE / 'outputs' / f'{name}.npy'
                    if not path.exists():
                        print(f'{name}: outputs absent; skip', flush=True)
                        continue
                    meta = json.loads(path.with_suffix('.json').read_text())
                    cfg = meta['config']
                    from itp_train import CONFIG
                    assert all(cfg[k] == val for k, val in CONFIG.items())
                    assert (cfg['input'], cfg['target'], cfg['seed'], cfg['smoke'], cfg['sample_sha256']) == (
                        inp, target, seed, False, fingerprint(s))
                    raw = np.load(path, mmap_mode='r')
                    assert raw.dtype == np.float16
                    x = np.asarray(raw, dtype=np.float32)
                    assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < 0.01
                    score(name, x, hashlib.sha256(path.read_bytes()).hexdigest())
    save()
    bars(r, emit=True)
    print(f'Finished in {time.monotonic() - start:.1f}s; {args.results}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
