"""CPU scoring, checkpointed per metric. Missing masked views are reported and skipped.

Uses the pav clustering/probe/summary functions without changes. A failed unmasked
consistency gate is recorded and printed; remaining metrics still run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import adjusted_mutual_info_score as ami
from ewm_masks import (HERE, PAV, KEYS, EMOTION_NAMES, MASK_SPEC_SHA256, assert_emotion_map,
                       atomic_json, build_masks, json_hash, load_captions, load_frozen,
                       save_masks, word_statistics)
from ewm_extract import validate_masked
from pav_extract import OUTPUT, identity, validate_key
from pav_prompts import PROMPTS
from pav_score import partitions, probe, summary
from apc_data import l2, paint_first, paint_majority, paint_mean

TEXTS = ('unmasked', 'masked_label', 'masked_broad')


def fmt(value):
    return f"{value['mean']:.6f} ± {value['sd']:.6f}" if value else '—'


def markdown(r):
    lines = ['# Emotion word masking', '',
        'Train-side check. AMI: mean ± population sd across seeds 0/1/2. Probe: five grouped folds.',
        'Labels enter scoring only. Missing views are skipped. Unmasked probes are copied from pav results.json.', '']
    stats = r.get('word_statistics')
    if stats:
        lines += ['## Word statistics', '', '| Mask | Captions matched | Share |', '|---|---:|---:|']
        for mask, v in stats['masks'].items():
            lines.append(f"| {mask} | {v['matched_n']} / {stats['n']} | {v['share']:.6f} |")
        lines += ['', stats['label_agreement_definition'],
                  f"Agreement: {stats['label_agreement_n']} captions; share {stats['label_agreement_share']:.6f}.", '',
                  '| Own emotion | n | Label share | Broad share | Own-label agreement among label matches |',
                  '|---|---:|---:|---:|---:|']
        for name in EMOTION_NAMES:
            a = stats['masks']['label']['per_emotion'][name]
            b = stats['masks']['broad']['per_emotion'][name]
            agreement = a['own_label_agreement_share']
            lines.append(f"| {name} | {a['n']} | {a['share']:.6f} | {b['share']:.6f} | "
                         + ('—' if agreement is None else f'{agreement:.6f}') + ' |')
        for mask, v in stats['masks'].items():
            lines += ['', f'### {mask}: top 30 matches (occurrences, lowercased)', '', '| Word | Count |', '|---|---:|']
            lines += [f"| {v['word']} | {v['count']} |" for v in v['top30']]
        lines += ['', '## Full and no-word label distributions', '',
                  '| Group | n | ' + ' | '.join(EMOTION_NAMES) + ' |', '|---|---:|' + '---:|' * 9]
        for group, v in {'full': stats['full_distribution'], **stats['subgroup_distributions']}.items():
            entries = [f'{n} ({share:.4f})' if share is not None else str(n)
                       for n, share in zip(v['counts'], v['shares'])]
            lines.append(f"| {group} | {v['n']} | " + ' | '.join(entries) + ' |')
    lines += ['', '## Caption emotion AMI', '', '| View | k=9 | k=20 | k=40 |', '|---|---:|---:|---:|']
    for name, v in r['views'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(v.get('caption_ami', {}).get(str(k))) for k in (9, 20, 40)) + ' |')
    lines += ['', '## Mean-caption painting AMI (5,000 paintings, k=20)', '',
              '| View | Emotion | Genre | Style |', '|---|---:|---:|---:|']
    for name, v in r['views'].items():
        vals = v.get('painting_ami', {}).get('mean_caption', {})
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(vals.get(label)) for label in ('emotion', 'genre', 'style')) + ' |')
    lines += ['', '## Caption emotion probe (subset captions, grouped by painting)', '',
              '| View | Accuracy | Majority rate | Source |', '|---|---:|---:|---|']
    for name, v in r['views'].items():
        val = v.get('probes', {}).get('caption/emotion')
        if val:
            lines.append(f"| {name} | {fmt(val['accuracy'])} | {val['majority_class_rate']:.6f} | {v['probe_source']} |")
    lines += ['', '## No-word subgroups: unmasked caption emotion AMI', '',
              '| Group | Key | n | k=9 | k=20 | k=40 |', '|---|---|---:|---:|---:|---:|']
    for group, g in r.get('subgroups', {}).items():
        for key, v in g.get('views', {}).items():
            lines.append(f"| {group} | {key} | {g['n']} | " + ' | '.join(fmt(v.get(str(k))) for k in (9, 20, 40)) + ' |')
    lines += ['', '## Unmasked consistency gates (tolerance 0.002 for mean and sd)', '',
              '| Metric | Status | Current | Earlier |', '|---|---|---:|---:|']
    for name, gate in r.get('consistency', {}).items():
        lines.append(f"| {name} | {'PASS' if gate['ok'] else 'FAIL'} | {fmt(gate['observed'])} | {fmt(gate['expected'])} |")
    lines += ['', '## Availability', '']
    lines += [f'- {name}: {status}' for name, status in r.get('availability', {}).items()]
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--keys', nargs='+', choices=KEYS, default=list(KEYS))
    p.add_argument('--texts', nargs='+', choices=TEXTS, default=list(TEXTS))
    p.add_argument('--ks', nargs='+', type=int, choices=(9, 20, 40), default=[9, 20, 40])
    p.add_argument('--stage', choices=('all', 'caption', 'words'), default='all')
    p.add_argument('--no-subgroups', action='store_true', help='omit subgroup clustering for a bounded smoke check')
    p.add_argument('--results', type=Path, default=HERE / 'ewm_results.json',
                   help='checkpoint; artifacts and ewm_tables.md are written beside it')
    args = p.parse_args()
    start = time.monotonic()
    s = load_frozen()
    assert_emotion_map(s)
    bundle = build_masks(load_captions(s))
    reference = json.loads((PAV / 'results.json').read_text())
    baseline = {key: {metric: reference['views'][key][metric] for metric in ('caption_ami', 'painting_ami', 'probes')}
                for key in KEYS}
    fingerprint = hashlib.sha256(b''.join(np.ascontiguousarray(s[k]).tobytes() for k in
        ('sample_ids', 'pid', 'subset_pid', 'subset_image_indices', 'emo', 'gen', 'sty'))).hexdigest()
    provenance = dict(sample_sha256=fingerprint, source_caption_sha256=bundle['source_sha256'],
                      masked_caption_sha256=bundle['masked_sha256'], mask_spec_sha256=MASK_SPEC_SHA256,
                      prompts={key: PROMPTS[key] for key in KEYS}, baseline_sha256=json_hash(baseline))
    r = json.loads(args.results.read_text()) if args.results.exists() else dict(
        schema=1, provenance=provenance, n_captions=len(s['emo']), n_paintings=len(s['first']),
        n_subset=len(s['subset_pid']), views={}, consistency={}, subgroups={}, availability={}, metric_seconds={},
        conventions='Exact pav_score partitions/probe/summary; raw float32 captions; L2 mean captions; labels scoring only')
    assert r['schema'] == 1 and r['provenance'] == provenance, 'checkpoint provenance differs; use a new --results path'
    args.results.parent.mkdir(parents=True, exist_ok=True)

    def save():
        atomic_json(args.results, r)
        table = args.results.parent / 'ewm_tables.md'
        pending = table.with_suffix('.pending.md')
        pending.write_text(markdown(r), encoding='utf-8')
        pending.replace(table)

    def measure(out, field, name, compute):
        if field not in out:
            before = time.monotonic()
            out[field] = compute()
            r['metric_seconds'][name] = time.monotonic() - before
            save()
        print(f'{name}: {out[field]}', flush=True)

    def gate(name, actual, expected):
        delta = {field: abs(actual[field] - expected[field]) for field in ('mean', 'sd')}
        r['consistency'][name] = dict(ok=all(v <= .002 for v in delta.values()),
                                     observed=actual, expected=expected, absolute_difference=delta, tolerance=.002)
        print(f"gate {name}: {'PASS' if r['consistency'][name]['ok'] else 'FAIL'} "
              f'current={actual} reference={expected} delta={delta}', flush=True)
        save()

    save_masks(bundle, s, args.results.parent)
    if 'word_statistics' not in r:
        r['word_statistics'] = word_statistics(bundle, s['emo'])
        save()
    for mask, value in r['word_statistics']['masks'].items():
        print(f"words {mask}: {value['matched_n']}/{len(s['emo'])} = {value['share']:.6f}; top10={value['top30'][:10]}", flush=True)
    if args.stage == 'words':
        print(f'Finished word statistics in {time.monotonic()-start:.1f}s', flush=True)
        return
    subset, pcount = s['subset_pid'], len(s['first'])
    # These are exactly the mean_caption features and labels from pav_score.main.
    labels = dict(emotion=paint_majority(s['emo'], s['pid'], pcount, 9)[subset],
                  genre=paint_first(s['gen'], s['pid'], pcount)[subset],
                  style=paint_first(s['sty'], s['pid'], pcount)[subset])
    cm = np.isin(s['pid'], subset)
    r['n_subset_captions'] = int(cm.sum())
    for key in args.keys:
        for text in args.texts:
            name = f'{key}/{text}'
            path = OUTPUT / key if text == 'unmasked' else OUTPUT / text / key
            required = ('done.json', 'ids.npz', 'txt.npy') + (('img.npy',) if text == 'unmasked' else ())
            if not all((path / file).exists() for file in required):
                r['availability'][name] = 'missing/incomplete; skipped'
                print(f'{name}: missing/incomplete; skip ({path})', flush=True)
                save()
                continue
            if text == 'unmasked':
                _, raw = validate_key(path, identity(s, 5000, 99999), PROMPTS[key])
            else:
                mask = text.removeprefix('masked_')
                raw = validate_masked(path, identity(s, 0, 99999), PROMPTS[key], mask, bundle['masked_sha256'][mask])
            r['availability'][name] = 'verified'
            cap = np.asarray(raw, dtype=np.float32)
            v = r['views'].setdefault(name, {})
            out = v.setdefault('caption_ami', {})
            for k in args.ks:
                metric = f'{name}/caption_ami/{k}'
                measure(out, str(k), metric, lambda k=k: summary([ami(s['emo'], lab) for lab in partitions(cap, k)]))
                if text == 'unmasked':
                    gate(metric, out[str(k)], baseline[key]['caption_ami'][str(k)])
            if args.stage == 'all':
                means = v.setdefault('painting_ami', {}).setdefault('mean_caption', {})
                if len(means) < 3:
                    x = l2(paint_mean(cap, s['pid'], pcount)[subset])
                    before = time.monotonic()
                    ls = partitions(x, 20)
                    r['metric_seconds'][f'{name}/painting_partitions'] = time.monotonic() - before
                    for label, y in labels.items():
                        measure(means, label, f'{name}/mean_caption/{label}',
                                lambda y=y: summary([ami(y[y >= 0], lab[y >= 0]) for lab in ls]))
                if text == 'unmasked':
                    for label in labels:
                        gate(f'{name}/mean_caption/{label}', means[label], baseline[key]['painting_ami']['mean_caption'][label])
                probes = v.setdefault('probes', {})
                if text == 'unmasked':
                    v['probe_source'] = 'copied from pav results.json: views[key].probes[caption/emotion]'
                    if 'caption/emotion' not in probes:
                        probes['caption/emotion'] = baseline[key]['probes']['caption/emotion']
                        save()
                else:
                    v['probe_source'] = 'computed with pav_score.probe'
                    measure(probes, 'caption/emotion', f'{name}/probe', lambda: probe(cap[cm], s['emo'][cm], s['pid'][cm]))
            if text == 'unmasked' and not args.no_subgroups:
                for mask in ('broad', 'label'):
                    group = f'no_{mask}_word'
                    rows = ~bundle[f'has_{mask}_word']
                    g = r['subgroups'].setdefault(group, dict(n=int(rows.sum()), views={}))
                    out = g['views'].setdefault(key, {})
                    if any(str(k) not in out for k in args.ks):
                        x = cap[rows]
                        y = s['emo'][rows]
                        for k in args.ks:
                            measure(out, str(k), f'{group}/{key}/caption_ami/{k}',
                                    lambda k=k: summary([ami(y, lab) for lab in partitions(x, k)]))
                        del x
            save()
            del cap, raw
    save()
    failed = [name for name, value in r['consistency'].items() if not value['ok']]
    print(f'Finished in {time.monotonic()-start:.1f}s; failed gates={failed}; {args.results}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
