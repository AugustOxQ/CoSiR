"""Follow-up default and prompted graph heads, with a combined cached AMI sheet."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import ags_head as head

FU = {
    'emotion': {'default': {'R0': [1, 2], 'R1': [1, 2]},
                'prompted': {'R0': [0, 1, 2], 'Fres1': [0, 1, 2]}},
    'style': {'default': {'R0': [1, 2], 'R1': [1, 2]},
              'prompted': {'R0': [0, 1, 2], 'Fres05': [0, 1, 2]}},
    'genre': {'default': {'R1': [1, 2], 'R0o': [0, 1, 2]},
              'prompted': {'R0': [0, 1, 2], 'Fside025': [0, 1, 2]}},
}
STAGE1 = Path('/project/CoSiR/src/test/20261018_aspect_graph_sheet/ags_head_results.json')


def input_key(aspect, kind):
    return 'default' if kind == 'default' else head.ASPECTS[aspect]


def output_name(aspect, kind, row, seed):
    return f'{aspect}__{row}' + ('' if kind == 'default' else '__prompted') + f'__s{seed}'


def runs(aspect, kind, smoke=False):
    items = [(row, seed) for row, seeds in FU[aspect][kind].items() for seed in seeds]
    return items[:1] if smoke else items


def train(s, base, aspects, smoke):
    for aspect in aspects:
        rows = list(dict.fromkeys(row for kind in FU[aspect] for row in FU[aspect][kind]))
        graph, manifest = head.load_graph(s, base, aspect, rows)
        for kind in ('default', 'prompted'):
            inputs = head.load_inputs(s, aspect, kind)
            for row, seed in runs(aspect, kind, smoke):
                head.run(s, inputs, graph, manifest, aspect, row, seed, smoke,
                         input_key=input_key(aspect, kind), outdir='outputs_fu')
            del inputs


def aggregate(s, r, stage1, manifest):
    result = {}
    for aspect, kinds in r['selected'].items():
        result[aspect] = {}
        for kind, rows in kinds.items():
            outputs = []
            if kind == 'default' and not r['identity']['smoke']:
                for v in stage1['outputs'].values():
                    c = v['config']
                    if c['aspect'] != aspect or c['row'] not in rows or not v.get('complete'):
                        continue
                    expected = head.make_config(s, manifest, aspect, c['row'], c['seed'], False)
                    if c == expected:
                        outputs.append(v)
            for row, seed in runs(aspect, kind):
                v = r['outputs'].get(output_name(aspect, kind, row, seed))
                if v and v.get('complete'):
                    outputs.append(v)
            ids = [(v['config']['row'], v['config']['seed']) for v in outputs]
            assert len(ids) == len(set(ids)), f'{aspect}/{kind}: duplicate seed across stages'
            combined = dict(selected={aspect: rows}, outputs=dict(enumerate(outputs)))
            groups = head.aggregate(combined)[aspect]
            for row, g in groups.items():
                g['target_ami_by_seed'] = {str(v['config']['seed']): v['target_ami']
                                           for v in outputs if v['config']['row'] == row}
            result[aspect][kind] = groups
    return result


def markdown(r):
    lines = ['# Aspect graph head follow-up sheet' + (' (SMOKE)' if r['identity']['smoke'] else ''), '',
             'AMI k=20: three k-means seeds per view. Students: mean over runs; population sd only with '
             'three student seeds. References: sd over k-means seeds. Target = mean of both sides for the aspect label.', '']
    cols = ' | '.join(f'{side}/{label}' for side in ('img', 'cap') for label in head.LABELS)
    for aspect, kinds in r['selected'].items():
        lines += [f'## {aspect}', '', f'| Input | Row | Seeds | {cols} | Target AMI | Fit/dev pairs | Epochs (best) |',
                  '|---|---|---|' + '---:|' * 8 + '---|']
        for kind in ('default', 'prompted'):
            for row in kinds[kind]:
                g = r['aggregates'][aspect][kind].get(row)
                if g is None:
                    lines.append(f'| {kind} | {row} | pending | ' + '— | ' * 8 + '— |')
                    continue
                epochs = ', '.join(f"s{k}:{v} ({g['best_epochs'][k]})" for k, v in g['epochs'].items())
                lines.append(f"| {kind} | {row} | {','.join(map(str, g['seeds']))} | " +
                             ' | '.join(head.fmt(g['ami'][side][label])
                                        for side in ('img', 'cap') for label in head.LABELS) +
                             f" | {head.fmt(g['target_ami'])} | {g['fit_pairs']}/{g['dev_pairs']} | {epochs} |")
        lines += ['', '| Reference | Emotion AMI | Style AMI | Genre AMI |', '|---|---:|---:|---:|']
        for key in ('default', head.ASPECTS[aspect]):
            for side in ('img', 'cap'):
                v = r['references'][f'{key}/{side}']
                lines.append(f'| {key}/{side} | ' +
                             ' | '.join(head.fmt(v['ami'][label]) for label in head.LABELS) + ' |')
        lines.append('')
        for kind in ('prompted', 'default'):
            key = input_key(aspect, kind)
            value = np.mean([r['references'][f'{key}/{side}']['ami'][aspect]['mean']
                             for side in ('img', 'cap')])
            lines.append(f'{kind} view, mean of both sides ({aspect} AMI): {value:.6f}.')
        lines.append('')
    lines += [f"Missing outputs: {', '.join(r['missing_outputs']) or 'none'}.", '']
    return '\n'.join(lines)


def score(s, base, aspects, smoke):
    start = time.monotonic()
    stage_path = head.HERE / 'ags_head_results.json'
    stage1 = json.loads((stage_path if stage_path.exists() else STAGE1).read_text())
    identity = dict(sample_sha256=head.fingerprint(s), smoke=smoke, scoring=head.SCORING,
                    labels_sha256=hashlib.sha256(b''.join(s[k].tobytes() for k in ('emo', 'sty', 'gen'))).hexdigest())
    assert stage1['identity']['scoring'] == head.SCORING, 'stage 1 scoring mismatch'
    assert stage1['identity'] == dict(identity, smoke=False), 'stage 1 identity mismatch'
    manifest = json.loads((base / 'graphs/manifest.json').read_text())
    path = base / 'ags_fu_results.json'
    r = json.loads(path.read_text()) if path.exists() else dict(
        schema=1, identity=identity, selected={}, outputs={}, references={}, missing_outputs=[])
    assert r['schema'] == 1 and r['identity'] == identity, 'incompatible score cache'
    for aspect in aspects:
        r['selected'][aspect] = {
            'default': list(dict.fromkeys(stage1['selected'][aspect] + list(FU[aspect]['default']))),
            'prompted': list(FU[aspect]['prompted'])}
    r['references'] = stage1['references']
    labels = head.label_vectors(s, s['subset_pid'])

    def save():
        r['aggregates'] = aggregate(s, r, stage1, manifest)
        head.atomic_json(path, r)
        head.atomic_text(base / 'ags_fu_tables.md', markdown(r))

    missing = []
    for aspect in aspects:
        rows = list(dict.fromkeys(row for kind in FU[aspect] for row in FU[aspect][kind]))
        _, graph_manifest = head.load_graph(s, base, aspect, rows)
        for kind in ('default', 'prompted'):
            for row, seed in runs(aspect, kind):
                name = output_name(aspect, kind, row, seed)
                output = base / 'outputs_fu' / f'{name}.npz'
                config = head.make_config(s, graph_manifest, aspect, row, seed, smoke, input_key(aspect, kind))
                meta = head.verify_output(output, config)
                if meta is None:
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
                            v['ami'][side] = head.ami_scores(z[side], labels)
                            save()
                v['target_ami'] = head.target_ami(v['ami'], aspect)
                v['complete'] = True
                save()
                print(f'{name}: target AMI={v["target_ami"]:.6f}', flush=True)
    r['missing_outputs'] = [name for name in r['missing_outputs'] if name.split('__')[0] not in aspects] + missing
    save()
    print(f'Missing outputs: {", ".join(missing) or "none"}', flush=True)
    print(f'Score complete: {time.monotonic() - start:.2f}s; {path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--score', action='store_true')
    p.add_argument('--aspects', nargs='+', choices=list(FU), default=list(FU))
    p.add_argument('--smoke', action='store_true', help='first default and prompted runs, 5000 fit pairs, 2 epochs')
    args = p.parse_args()
    torch.set_num_threads(8)
    assert not torch.cuda.is_available(), 'set CUDA_VISIBLE_DEVICES to empty'
    s = head.load_frozen()
    base = head.root(args.smoke)
    if args.train:
        train(s, base, args.aspects, args.smoke)
    else:
        score(s, base, args.aspects, args.smoke)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
