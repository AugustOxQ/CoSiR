"""Full fixed-subset scoring; reference rows are imported, never recomputed."""
import argparse
import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from pbp_graphs import (HERE, ITP, atomic_json, fingerprint, label_vectors, load_frozen,
                        load_graphs, root, sha_file, stats_markdown, targets)
from pbp_train import VARIANTS, make_config, verify_output
from itp_score import flatten, overlap_indices, top20_neighbours
from pav_score import ami, partitions, probe, summary

REFERENCE = ITP / 'itp_results.json'
BAR = 'S_P emotion AMI >= 0.12 AND mean(S_P)-mean(S_D) > max(sd(S_P),sd(S_D))'


def aggregate(views):
    out = {}
    for variant in VARIANTS:
        rows = [flatten(views[f'{variant}__s{seed}']) for seed in (0, 1, 2)
                if views.get(f'{variant}__s{seed}', {}).get('complete')]
        if rows:
            keys = set.intersection(*(set(row) for row in rows))
            out[variant] = dict(n_seeds=len(rows), metrics={key: summary([r[key] for r in rows])
                for key in sorted(keys) if key.startswith(('painting_ami.', 'emotion_probe.', 'overlap.'))})
    return out


def bar_numbers(p, d):
    gap = p['mean'] - d['mean']
    spread = max(p['sd'], d['sd'])
    return dict(pass_bar=bool(p['mean'] >= .12 and gap > spread),
                S_P=p, S_D=d, difference=gap, max_sd=spread, threshold=.12)


def effects(aggregates):
    out = {}
    for name, a, b in [('purification', 'S_P', 'S_E'), ('hard_negative', 'S_PN', 'S_P'),
                        ('instruction_full_graph', 'S_E', 'S_D')]:
        if a in aggregates and b in aggregates:
            va = aggregates[a]['metrics']['painting_ami.emotion.mean']
            vb = aggregates[b]['metrics']['painting_ami.emotion.mean']
            out[name] = dict(a=a, b=b, a_mean=va['mean'], b_mean=vb['mean'],
                            difference=va['mean'] - vb['mean'], a_sd=va['sd'], b_sd=vb['sd'],
                            a_seeds=aggregates[a]['n_seeds'], b_seeds=aggregates[b]['n_seeds'])
    return out


def derive_results(r):
    r['aggregates'] = aggregate(r['views'])
    ag = r['aggregates']
    r['bar'] = dict(status='PENDING', reason='Requires all three S_P and S_D student seeds')
    if 'S_P' in ag and 'S_D' in ag:
        nums = bar_numbers(ag['S_P']['metrics']['painting_ami.emotion.mean'],
                           ag['S_D']['metrics']['painting_ami.emotion.mean'])
        if ag['S_P']['n_seeds'] == ag['S_D']['n_seeds'] == 3 and not r['smoke']:
            r['bar'] = dict(nums, status='PASS' if nums['pass_bar'] else 'FAIL')
        elif r['smoke']:
            r['smoke_bar_diagnostic'] = nums
    r['effects'] = effects(ag)
    baseline = r['references']['views']['default_image']['painting_ami']
    r['genre_style_changes'] = {variant: {
        label: dict(difference=g['metrics'][f'painting_ami.{label}.mean']['mean'] - baseline[label]['mean'],
                    student=g['metrics'][f'painting_ami.{label}.mean'], reference=baseline[label])
        for label in ('genre', 'style')} for variant, g in ag.items()}


def fmt(v):
    return '—' if v is None else f"{v['mean']:.6f} ± {v['sd']:.6f}"


def markdown(r):
    lines = ['# Purified buddy pilot' + (' — SMOKE ONLY' if r['smoke'] else ''), '',
        f'Prespecified bar: {BAR}.', '',
        'Student AMI: mean of each run\'s three k-means seeds, then mean ± population sd across student seeds. '
        'Reference individual-row sd is across k-means seeds. Probe sd is across GroupKFold folds within a run. '
        'Partial student groups cannot decide the research bar.', '',
        f"Reference rows copied unchanged from `{REFERENCE}` (SHA256 `{r['references']['sha256']}`); no reference metric recomputed.", '',
        '| View | Seeds | Emotion AMI | Genre AMI | Style AMI | Probe accuracy | Emo overlap | Dcap overlap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in r['references']['views'].items():
        ami_row = row['painting_ami']
        overlaps = row.get('overlap', {})
        lines.append(f'| reference/{name} | — | ' + ' | '.join(fmt(ami_row[k]) for k in ('emotion', 'genre', 'style')) +
            f" | {fmt(row.get('emotion_probe', {}).get('accuracy'))} | {overlaps.get('teacher', '—')} | {overlaps.get('control_target', '—')} |")
    for name, g in r['references']['aggregates'].items():
        m = g['metrics']
        lines.append(f"| reference/itp/{name} | {g['n_seeds']}/3 | " +
            ' | '.join(fmt(m[f'painting_ami.{k}.mean']) for k in ('emotion', 'genre', 'style')) +
            f" | {fmt(m.get('emotion_probe.accuracy.mean'))} | {fmt(m.get('overlap.teacher'))} | {fmt(m.get('overlap.control_target'))} |")
    for name, g in r['aggregates'].items():
        m = g['metrics']
        lines.append(f"| {name} | {g['n_seeds']}/3 | " +
            ' | '.join(fmt(m[f'painting_ami.{k}.mean']) for k in ('emotion', 'genre', 'style')) +
            f" | {fmt(m['emotion_probe.accuracy.mean'])} | {fmt(m['overlap.teacher'])} | {fmt(m['overlap.control_target'])} |")
    lines += ['', f"BAR {r['bar']['status']}: " + (r['bar'].get('reason') or
        f"S_P={fmt(r['bar']['S_P'])}; S_D={fmt(r['bar']['S_D'])}; gap={r['bar']['difference']:.6f}; max_sd={r['bar']['max_sd']:.6f}; threshold=0.12"), '']
    if 'smoke_bar_diagnostic' in r:
        b = r['smoke_bar_diagnostic']
        lines += [f"SMOKE diagnostic {'PASS' if b['pass_bar'] else 'FAIL'} (one seed; not a research decision): "
            f"S_P={fmt(b['S_P'])}; S_D={fmt(b['S_D'])}; gap={b['difference']:.6f}; max_sd={b['max_sd']:.6f}.", '']
    lines += ['| Effect (emotion AMI) | Difference | First student sd | Second student sd | Seed counts |',
              '|---|---:|---:|---:|---|']
    for e in r['effects'].values():
        lines.append(f"| {e['a']} - {e['b']} | {e['difference']:+.6f} | {e['a_sd']:.6f} | {e['b_sd']:.6f} | {e['a_seeds']}, {e['b_seeds']} |")
    lines += ['', '| Variant | Genre Δ vs default image | Style Δ vs default image |', '|---|---:|---:|']
    for name, changes in r['genre_style_changes'].items():
        lines.append(f"| {name} | {changes['genre']['difference']:+.6f} | {changes['style']['difference']:+.6f} |")
    lines += ['', '## Fit graph statistics', '']
    if r.get('graph_stats'):
        lines += [stats_markdown(r['graph_stats'])]
    else:
        lines += ['Graph statistics pending.']
    lines += ['', '## Per-run metrics', '', '| Run | Emotion AMI | Genre AMI | Style AMI | Probe accuracy | Emo overlap | Dcap overlap |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in r['views'].items():
        if not row.get('complete'):
            continue
        lines.append(f'| {name} | ' + ' | '.join(fmt(row['painting_ami'][k]) for k in ('emotion', 'genre', 'style')) +
            f" | {fmt(row['emotion_probe']['accuracy'])} | {row['overlap']['teacher']:.6f} | {row['overlap']['control_target']:.6f} |")
    lines += ['', f"Exact top-20 chance overlap: {20/4999:.8f}. Missing outputs are recorded in JSON. Full queue remains controller-owned.", '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true', help='score only isolated seed-0 smoke outputs; full 5000 painting metrics')
    p.add_argument('--references-only', action='store_true', help='import reference results and graph stats; do not score students')
    args = p.parse_args()
    start = time.monotonic()
    print(f'Prespecified bar: {BAR}', flush=True)
    s = load_frozen()
    base = root(args.smoke)
    result_path = base / 'pbp_results.json'
    table_path = base / 'pbp_tables.md'
    label_hash = hashlib.sha256(b''.join(s[k].tobytes() for k in ('emo', 'gen', 'sty'))).hexdigest()
    identity = dict(sample_sha256=fingerprint(s), labels_sha256=label_hash,
                    reference_sha256=sha_file(REFERENCE), smoke=args.smoke,
                    scoring='unchanged pav partitions(k20,seeds012)/probe/summary; exact itp top20 overlap; v1')
    original = json.loads(REFERENCE.read_text())
    assert original['identity']['sample_sha256'] == fingerprint(s)
    assert original['identity']['labels_sha256'] == label_hash
    r = json.loads(result_path.read_text()) if result_path.exists() else dict(
        schema=1, identity=identity, smoke=args.smoke, views={},
        references=dict(file=str(REFERENCE), sha256=identity['reference_sha256'],
                        views=original['views'], aggregates=original['aggregates']))
    assert r['identity'] == identity, 'incompatible score cache'
    r['missing_outputs'] = [f'{variant}__s{seed}' for variant in VARIANTS
        for seed in ([0] if args.smoke else (0, 1, 2))
        if not (base / 'outputs' / f'{variant}__s{seed}.npy').exists()]
    stats_path = base / 'graphs' / 'stats.json'
    r['graph_stats'] = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if r['graph_stats']:
        assert r['graph_stats']['labels_sha256'] == label_hash
    def save():
        derive_results(r)
        atomic_json(result_path, r)
        pending = table_path.with_name('pbp_tables.pending.md')
        pending.write_text(markdown(r))
        pending.replace(table_path)
    save()
    if not args.references_only:
        _, manifest = load_graphs(s, args.smoke)
        labels = label_vectors(s, s['subset_pid'])
        neighbours = {name: top20_neighbours(targets(s, key)[s['subset_pid']])
                      for name, key in [('teacher', 'emotion_focus'), ('control_target', 'default')]}
        missing = []
        for variant in VARIANTS:
            for seed in ([0] if args.smoke else (0, 1, 2)):
                name = f'{variant}__s{seed}'
                path = base / 'outputs' / f'{name}.npy'
                cfg = make_config(s, manifest, variant, seed, args.smoke)
                meta = verify_output(path, cfg)
                if meta is None:
                    missing.append(name)
                    print(f'{name}: absent; skip', flush=True)
                    continue
                v = r['views'].setdefault(name, dict(source_sha256=meta['output_sha256'], config=cfg))
                assert v['source_sha256'] == meta['output_sha256'] and v['config'] == cfg
                if v.get('complete'):
                    print(f'{name}: scored; skip', flush=True)
                    continue
                x = np.asarray(np.load(path), dtype=np.float32)
                if 'painting_ami' not in v:
                    ls = partitions(x, 20)
                    v['painting_ami'] = {label: summary([ami(y[y >= 0], lab[y >= 0]) for lab in ls])
                                         for label, y in labels.items()}
                    save()
                if 'emotion_probe' not in v:
                    v['emotion_probe'] = probe(x, labels['emotion'], s['subset_pid'])
                    save()
                if 'overlap' not in v:
                    nn = top20_neighbours(x)
                    v['overlap'] = {key: overlap_indices(nn, ref) for key, ref in neighbours.items()}
                    save()
                v['complete'] = True
                save()
                print(f'{name}: score PASS emotion={v["painting_ami"]["emotion"]["mean"]:.6f}', flush=True)
        r['missing_outputs'] = missing
    save()
    b = r['bar']
    if b['status'] == 'PENDING':
        print(f"BAR PENDING: {b['reason']}", flush=True)
    else:
        print(f"BAR {b['status']}: S_P={fmt(b['S_P'])}; S_D={fmt(b['S_D'])}; "
              f"gap={b['difference']:.6f}; max_sd={b['max_sd']:.6f}; threshold=0.120000", flush=True)
    if r.get('smoke_bar_diagnostic'):
        b = r['smoke_bar_diagnostic']
        print(f"SMOKE diagnostic {'PASS' if b['pass_bar'] else 'FAIL'} (one seed, not research bar): "
              f"S_P={fmt(b['S_P'])}; S_D={fmt(b['S_D'])}; gap={b['difference']:.6f}; max_sd={b['max_sd']:.6f}", flush=True)
    for e in r['effects'].values():
        print(f"{e['a']} - {e['b']}: {e['difference']:+.6f}; sds={e['a_sd']:.6f}, {e['b_sd']:.6f}", flush=True)
    for name, changes in r['genre_style_changes'].items():
        print(f'{name} vs default_image: ' + '; '.join(
            f"{k} delta={v['difference']:+.6f}, student_sd={v['student']['sd']:.6f}, reference_kmeans_sd={v['reference']['sd']:.6f}"
            for k, v in changes.items()), flush=True)
    print(f'Scoring complete in {time.monotonic()-start:.1f}s; {result_path}', flush=True)
    if args.smoke and not args.references_only:
        assert not r.get('missing_outputs'), 'Smoke requires every variant output'


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
