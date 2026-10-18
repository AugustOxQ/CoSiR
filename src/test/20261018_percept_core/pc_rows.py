"""Frozen-head ablation rows 6, 10, 13 and the genre regression gate."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261018_aspect_graph_sheet'))
import ags_head as head

ARCHIVE = Path('/project/CoSiR/src/test/20261018_aspect_graph_sheet')
PILOT = Path('/project/CoSiR/src/test/20261018_purified_buddy_pilot/graphs')
METHOD = 'pc rows: Ddef=pilot D (emotion/genre), I (style); R0o=ags archive; v1'
GATE_TARGET = 0.40609876614182405
SETS = dict(row6=('Ddef', 'default', ('emotion', 'style', 'genre')),
            row10=('R0o', 'default', ('emotion', 'style')),
            row13=('R0o', 'prompted', ('emotion', 'style', 'genre')),
            gate=('R0', 'default', ('genre',)))


def build_graphs(s, base):
    base = Path(base).resolve()
    folder = base / 'graphs'
    folder.mkdir(parents=True, exist_ok=True)
    _, fit, dev = head.split(s)
    parts = dict(fit=fit, dev=dev)
    pilot = json.loads((PILOT / 'manifest.json').read_text())
    assert pilot['identity']['sample_sha256'] == head.fingerprint(s)
    assert not pilot['identity']['smoke']
    defaults = {}
    for side, nodes in parts.items():
        path = PILOT / f'{side}.npz'
        assert head.sha_file(path) == pilot['archives'][side], 'pilot checksum mismatch'
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['node_ids'], nodes), 'pilot node order mismatch'
            defaults[side] = {key: z[key] for key in ('D', 'I')}
            for key, pairs in defaults[side].items():
                assert len(pairs) == pilot['counts'][side][key]
    arrays, sources, counts = {}, dict(pilot=pilot['archives']), {}
    for aspect in head.ASPECTS:
        graph, source = head.load_graph(s, ARCHIVE, aspect, ['R0o'])
        sources[aspect] = source['archives'][aspect]
        arrays[aspect] = {f'{side}_node_ids': nodes for side, nodes in parts.items()}
        counts[aspect] = {}
        for side in parts:
            pairs = defaults[side]['I' if aspect == 'style' else 'D']
            other, weights = graph['rows']['R0o'][side]
            arrays[aspect].update({f'Ddef__{side}': pairs,
                f'Ddef__{side}_w': np.ones(len(pairs), dtype=np.float32),
                f'R0o__{side}': other, f'R0o__{side}_w': weights})
            counts[aspect][side] = dict(Ddef=len(pairs), R0o=len(other))
    identity = dict(method=METHOD, sample_sha256=head.fingerprint(s),
                    fit_size=len(fit), dev_size=len(dev))
    marker = folder / 'manifest.json'
    if marker.exists():
        manifest = json.loads(marker.read_text())
        assert manifest['identity'] == identity and manifest['source_sha256'] == sources
        assert manifest['counts'] == counts
    else:
        manifest = dict(identity=identity, archives={}, counts=counts, source_sha256=sources)
        for aspect, values in arrays.items():
            path = folder / f'{aspect}.npz'
            assert not path.exists(), 'unsealed graph archive exists'
            pending = path.with_name(path.stem + '.pending.npz')
            np.savez(pending, **values)
            pending.replace(path)
            manifest['archives'][aspect] = head.sha_file(path)
        head.atomic_json(marker, manifest)
    for aspect, values in arrays.items():
        head.load_graph(s, base, aspect)
        with np.load(folder / f'{aspect}.npz', allow_pickle=False) as z:
            assert set(z.files) == set(values)
            for key, expected in values.items():
                assert np.array_equal(z[key], expected), f'{aspect}/{key}: source mismatch'
    return manifest


def gate_result(scores, smoke=False):
    actual = head.target_ami(scores, 'genre')
    assert np.isfinite(actual)
    difference = actual - GATE_TARGET
    passed = abs(difference) <= 1e-6 and not smoke
    return dict(status='SMOKE' if smoke else ('PASS' if passed else 'FAIL'),
                pass_gate=passed, target_ami=actual, expected=GATE_TARGET,
                difference=difference, tolerance=1e-6, within_002=abs(difference) <= .002)


def row_set(config):
    aspect, row, key = config['aspect'], config['row'], config['input_img']
    assert config['input_cap'] == key and aspect in head.ASPECTS
    assert config['seed'] in (0, 1, 2)
    if row == 'Ddef':
        assert key == 'default'
        return 'row6'
    if row == 'R0o':
        assert key in ('default', head.ASPECTS[aspect])
        if key == 'default':
            assert aspect in ('emotion', 'style')
            return 'row10'
        return 'row13'
    assert row == 'R0' and aspect == 'genre' and config['seed'] == 0 and key == 'default'
    return 'gate'


def aggregate(outputs):
    result = {name: {} for name in ('row6', 'row10', 'row13')}
    grouped = {}
    for value in outputs.values():
        if not value.get('complete'):
            continue
        name = row_set(value['config'])
        if name != 'gate':
            grouped.setdefault((name, value['config']['aspect']), []).append(value)
    for (name, aspect), runs in grouped.items():
        seeds = sorted(v['config']['seed'] for v in runs)
        assert len(seeds) == len(set(seeds)), 'duplicate head seed'
        def stats(xs):
            assert np.isfinite(xs).all()
            return dict(mean=float(np.mean(xs)), sd=float(np.std(xs)))
        ami = {side: {label: stats([v['ami'][side][label]['mean'] for v in runs])
                      for label in head.LABELS} for side in ('img', 'cap')}
        ami['mean'] = {label: stats([(v['ami']['img'][label]['mean'] +
                                     v['ami']['cap'][label]['mean']) / 2 for v in runs])
                       for label in head.LABELS}
        result[name][aspect] = dict(seeds=seeds, complete=seeds == [0, 1, 2], ami=ami,
                                   target_ami=stats([head.target_ami(v['ami'], aspect) for v in runs]))
    return result


def jobs(name, aspects=None, seeds=None):
    row, kind, allowed = SETS[name]
    aspects = list(allowed) if aspects is None else list(dict.fromkeys(aspects))
    seeds = ([0] if name == 'gate' else [0, 1, 2]) if seeds is None else list(dict.fromkeys(seeds))
    assert aspects and set(aspects) <= set(allowed), f'{name}: unsupported aspects'
    assert seeds and set(seeds) <= ({0} if name == 'gate' else {0, 1, 2})
    return [(aspect, row, 'default' if kind == 'default' else head.ASPECTS[aspect], seed)
            for aspect in aspects for seed in seeds]


def run_base(base, smoke):
    base = Path(base).resolve()
    return base / 'smoke' if smoke else base


def output_name(aspect, row, key, seed):
    return f'{aspect}__{row}' + ('' if key == 'default' else '__prompted') + f'__s{seed}'


def verify_head(s, base, path, smoke):
    config = json.loads(path.with_suffix('.json').read_text())['config']
    name = row_set(config)
    aspect, row, seed, key = (config[k] for k in ('aspect', 'row', 'seed', 'input_img'))
    assert path.stem == output_name(aspect, row, key, seed), 'head filename mismatch'
    _, manifest = head.load_graph(s, ARCHIVE if name == 'gate' else base, aspect, [row])
    expected = head.make_config(s, manifest, aspect, row, seed, smoke, key)
    return head.verify_output(path, expected)


def gate_check(s, base, smoke=False):
    folder = run_base(base, smoke)
    path = folder / 'heads/genre__R0__s0.npz'
    assert path.exists(), 'local genre R0 seed 0 rerun required before gate-check'
    info = verify_head(s, base, path, smoke)
    labels = head.label_vectors(s, s['subset_pid'])
    with np.load(path, allow_pickle=False) as z:
        scores = {side: head.ami_scores(z[side], labels) for side in ('img', 'cap')}
    verdict = dict(gate_result(scores, smoke), config=info['config'],
                   output_sha256=info['output_sha256'], ami=scores, scoring=head.SCORING)
    verdict['cause'] = ('smoke run: full regression not checked' if smoke else
                        None if verdict['pass_gate'] else
                        'target AMI mismatch; numerical cause not established; head rows stopped')
    head.atomic_json(folder / 'gate.json', verdict)
    print(f"Gate {verdict['status']}: target={verdict['target_ami']:.12f}; "
          f"difference={verdict['difference']:+.12f}; "
          f"within 0.002: {'yes' if verdict['within_002'] else 'no'}", flush=True)
    return verdict


def train(s, base, name, aspects=None, seeds=None, smoke=False):
    selected = jobs(name, aspects, [0] if smoke and seeds is None else seeds)
    if name != 'gate' and not smoke:
        marker = Path(base) / 'gate.json'
        assert marker.exists(), 'stage 2 gate-check must PASS before head rows'
        assert json.loads(marker.read_text())['pass_gate'], 'stage 2 gate failed; head rows stopped'
        assert gate_check(s, base)['pass_gate'], 'stage 2 gate failed; head rows stopped'
    inputs, previous = None, None
    for aspect, row, key, seed in selected:
        kind = 'default' if key == 'default' else 'prompted'
        source = ARCHIVE if name == 'gate' else base
        graph, manifest = head.load_graph(s, source, aspect, [row])
        if previous != key:
            inputs = head.load_inputs(s, aspect, kind)
            previous = key
        outdir = run_base(base, smoke) / 'heads'
        assert outdir.is_absolute()
        head.run(s, inputs, graph, manifest, aspect, row, seed, smoke,
                 input_key=key, outdir=outdir)


def markdown(result):
    lines = ['# Frozen-head ablation rows' + (' (SMOKE)' if result['identity']['smoke'] else ''), '',
             'Stage 1 AMI at k=20, three k-means seeds per view. Each cell is the mean ± population sd '
             'over available head seeds. A row is complete only with seeds 0, 1, 2. '
             'The two-side mean is computed within each seed before aggregation.', '',
             '| Row | Aspect | Seeds | Image AMI | Caption AMI | Mean AMI | Complete |',
             '|---|---|---|---:|---:|---:|---|']
    for name in ('row6', 'row10', 'row13'):
        for aspect in SETS[name][2]:
            value = result['aggregates'][name].get(aspect)
            if value is None:
                lines.append(f'| {name} | {aspect} | pending | — | — | — | no |')
                continue
            cells = [head.fmt(value['ami'][side][aspect]) for side in ('img', 'cap', 'mean')]
            lines.append(f"| {name} | {aspect} | {','.join(map(str, value['seeds']))} | " +
                         ' | '.join(cells) + f" | {'yes' if value['complete'] else 'no'} |")
    lines += ['', 'All three label scores for each side are stored in pc_rows_results.json.', '',
              f"Missing outputs: {', '.join(result['missing_outputs']) or 'none'}.", '']
    return '\n'.join(lines)


def score(s, base, smoke=False):
    folder = run_base(base, smoke)
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(sample_sha256=head.fingerprint(s), smoke=smoke, scoring=head.SCORING,
        aggregation='mean and population sd over available head seeds; v1',
        labels_sha256=hashlib.sha256(b''.join(s[k].tobytes() for k in ('emo', 'sty', 'gen'))).hexdigest())
    path = folder / 'pc_rows_results.json'
    result = json.loads(path.read_text()) if path.exists() else dict(
        schema=1, identity=identity, outputs={}, missing_outputs=[])
    assert result['schema'] == 1 and result['identity'] == identity, 'incompatible score cache'
    labels = head.label_vectors(s, s['subset_pid'])
    def save():
        result['aggregates'] = aggregate(result['outputs'])
        head.atomic_json(path, result)
        head.atomic_text(folder / 'pc_rows_tables.md', markdown(result))
    seen = set()
    for output in sorted((folder / 'heads').glob('*.npz')):
        if output.name.endswith('.pending.npz'):
            continue
        info = verify_head(s, base, output, smoke)
        name = output.stem
        seen.add(name)
        value = result['outputs'].setdefault(name, dict(config=info['config'],
            source_sha256=info['output_sha256'], epochs_run=info['epochs_run'],
            best_epoch=info['best_epoch'], ami={}))
        assert value['config'] == info['config'] and value['source_sha256'] == info['output_sha256'], \
            'stale output score'
        if value.get('complete'):
            continue
        with np.load(output, allow_pickle=False) as z:
            for side in ('img', 'cap'):
                if side not in value['ami']:
                    value['ami'][side] = head.ami_scores(z[side], labels)
                    save()
        value['target_ami'] = head.target_ami(value['ami'], info['config']['aspect'])
        value['complete'] = True
        save()
        print(f"{name}: target AMI={value['target_ami']:.6f}", flush=True)
    assert set(result['outputs']) <= seen, 'previously scored output disappeared'
    expected = {output_name(*job) for name in SETS for job in jobs(name)}
    complete = {name for name, value in result['outputs'].items() if value.get('complete')}
    result['missing_outputs'] = sorted(expected - complete)
    save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build-graphs', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--gate-check', action='store_true')
    mode.add_argument('--score', action='store_true')
    parser.add_argument('--set', choices=list(SETS))
    parser.add_argument('--aspects', nargs='+', choices=list(head.ASPECTS))
    parser.add_argument('--seeds', type=int, nargs='+', choices=[0, 1, 2])
    parser.add_argument('--smoke', action='store_true', help='two epochs, 5000 fit pairs; outputs/smoke/heads')
    args = parser.parse_args()
    if args.train != (args.set is not None):
        parser.error('--train requires --set, and --set requires --train')
    if not args.train and (args.aspects is not None or args.seeds is not None):
        parser.error('--aspects/--seeds require --train')
    if args.train:
        jobs(args.set, args.aspects, args.seeds)
    head.torch.set_num_threads(8)
    assert not head.torch.cuda.is_available(), 'set CUDA_VISIBLE_DEVICES to empty'
    s = head.load_frozen()
    base = HERE / 'outputs'
    if args.build_graphs:
        build_graphs(s, base)
        print(f'Graphs PASS: {base / "graphs"}', flush=True)
    elif args.train:
        train(s, base, args.set, args.aspects, args.seeds, args.smoke)
    elif args.gate_check:
        verdict = gate_check(s, base, args.smoke)
        if not args.smoke and not verdict['pass_gate']:
            raise SystemExit(1)
    else:
        score(s, base, args.smoke)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
