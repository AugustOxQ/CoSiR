"""r29 ticket 05: validation-only decisions (spec §§3.4–3.6, §6; D17, D20, D23, D25).

No held row is read. Frozen K0 bounds come from r28 records, never from exports.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np

import r29_paths as P

ASPECTS = P.ASPECTS
SEEDS = list(P.SEEDS)
IMAGE_CELLS = tuple(f'{m}/{a}' for m in ('ami', 'map') for a in ASPECTS)
TEXT_CELLS = tuple(f'{a}/{d}' for a in ASPECTS for d in ('t2i', 'i2t', 't2t'))
CANDIDATES = ('siglip1', 'siglip1_prefix')


def read_json(path):
    doc = json.loads(Path(path).read_text())
    assert isinstance(doc, dict), path
    return doc


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    return doc


def now():
    return datetime.now(ZoneInfo('Europe/Amsterdam')).isoformat(timespec='seconds')


def numbers(values, n):
    assert isinstance(values, list) and len(values) == n, values
    assert all(type(v) in (float, int) for v in values) and np.isfinite(values).all(), values
    return values


def read_text(records, arm):
    assert arm in (*CANDIDATES, 'qwen', 'gate_i_a_new', 'p1_siglip1', 'p1_siglip1_prefix'), arm
    doc = read_json(Path(records) / f'text_{arm}.json')
    assert doc['arm'] == arm, (arm, doc['arm'])
    seeds = doc['seeds']
    if arm.startswith('p1_'):
        assert seeds == ['x'], seeds
    else:
        assert seeds and all(type(s) is int and s in SEEDS for s in seeds), seeds
        assert seeds == sorted(set(seeds)), seeds
        if arm == 'gate_i_a_new':
            assert seeds == [0], seeds
    group = numbers(doc['val']['group_r0']['per_seed'], len(seeds))
    cells = doc['val']['cells']
    assert set(cells) == set(ASPECTS), cells.keys()
    flat = {}
    for a in ASPECTS:
        assert set(cells[a]) == {'t2i', 'i2t', 't2t'}, cells[a].keys()
        for d in ('t2i', 'i2t', 't2t'):
            flat[f'{a}/{d}'] = numbers(cells[a][d]['per_seed']['r0'], len(seeds))
    return dict(arm=arm, seeds=seeds, per_seed=group, cells=flat)


def sc1(records=P.RECORDS):
    """SC1 seed 0; §3.5 one arm out; D20 P1 form chosen only when K2 wins."""
    records = Path(records)
    values = {}
    for arm in CANDIDATES:
        if (records / f'text_{arm}.json').exists():
            text = read_text(records, arm)
            assert text['seeds'][0] == 0, f'{arm}: SC1 requires seed 0 first'
            values[arm] = text['per_seed'][0]
    if not values:
        return write_json(records / 'prefix_pick.json', dict(status='inconclusive', arm=None,
                          no_comparison=True, reason='both arms out', written=now()))
    arm = max(values, key=values.get)  # insertion order makes exact ties K1
    p1_values, form = {}, 'siglip1'
    if arm == 'siglip1_prefix':
        for f in CANDIDATES:
            p1_values[f] = read_text(records, 'p1_' + f)['per_seed'][0]
        form = max(p1_values, key=p1_values.get)  # D20 exact tie: plain
    return write_json(records / 'prefix_pick.json', dict(status='picked', arm=arm, seed=0,
                      text_group=values, p1_form=form, p1_text_group=p1_values,
                      no_comparison=len(values) == 1, missing_arms=[a for a in CANDIDATES if a not in values],
                      readout='r0', written=now()))


def sc2_level(image, text, baseline):
    """§6: inclusive L1/L2, strict L3, using unrounded group means and bounds."""
    assert np.isfinite([image, text]).all()
    i, t = baseline['image'], baseline['text']
    if image > i['best'] and text > t['best']:
        return 3
    if image >= i['m'] - i['r'] and text >= t['m'] - t['r']:
        return 2 if text >= t['m'] else 1
    return 0


def group_cells(cells, names, seeds):
    assert set(cells) == set(names), cells.keys()
    for c in names:
        numbers(cells[c], len(seeds))
    # Match ticket 04's group_r0 reduction order: one mean per seed, then seeds.
    per_seed = [float(np.mean([cells[c][i] for c in names])) for i in range(len(seeds))]
    return dict(cells=cells, per_seed=per_seed, mean=float(np.mean(per_seed)))


def k0_baseline(r28_records=P.R28_RECORDS):
    """Replay D17 from frozen records. r28 text arrays use its fixed seed order 0..4."""
    r28_records = Path(r28_records)
    image_doc = read_json(r28_records / 'valpick.json')
    assert image_doc['kind'] == 'r28_valpick' and image_doc['seeds'] == SEEDS
    assert image_doc['arm'] == P.K0_ARM + ' (c*)'
    encoder = image_doc['encoders']['siglip1']
    assert encoder['encoder'] == 'siglip1' and encoder['n_val'] == 6152
    image = group_cells({c: v['per_seed'] for c, v in encoder['cells'].items()}, IMAGE_CELLS, SEEDS)
    # m is the recorded r28 score; re-averaging seed groups changes it by one ULP.
    image['mean'] = numbers([encoder['score']], 1)[0]
    text_doc = read_json(r28_records / 'text.json')
    picks = text_doc['ours_siglip1']['picks']
    assert set(picks) == set(ASPECTS)
    for a in ASPECTS:
        assert set(picks[a]) == {'t2i', 'i2t', 't2t'}
    text = group_cells({f'{a}/{d}': picks[a][d]['per_seed']['r0']
                        for a in ASPECTS for d in ('t2i', 'i2t', 't2t')}, TEXT_CELLS, SEEDS)
    text['mean'] = float(np.mean(numbers([picks[a][d]['means']['r0']
        for a in ASPECTS for d in ('t2i', 'i2t', 't2t')], 9)))
    result = {}
    for name, group, expected_mean, expected_range in (
        ('image', image, P.K0_IMAGE_MEAN, P.K0_IMAGE_SEEDS),
        ('text', text, P.K0_TEXT_MEAN, P.K0_TEXT_SEEDS),
    ):
        values = group['per_seed']
        m, lo, hi = group['mean'], min(values), max(values)
        assert round(m, 10) == expected_mean, (name, m)
        assert (round(lo, 10), round(hi, 10)) == expected_range, (name, lo, hi)
        r = hi - lo
        result[name] = dict(m=m, r=r, best=hi, lower=m-r, per_seed=values, cells=group['cells'])
    return result


def sc2(records=P.RECORDS):
    records = Path(records)
    pick = read_json(records / 'prefix_pick.json')  # missing pick refuses before any scoring
    assert pick['status'] == 'picked' and pick['arm'] in CANDIDATES, pick
    arm = pick['arm']
    groups_doc = read_json(records / f'groups_{arm}.json')
    assert groups_doc['arm'] == arm and groups_doc['seeds'] == SEEDS, 'SC2 requires all five seeds'
    values = {g: numbers(groups_doc[g]['per_seed'], 5) for g in ('image', 'text')}
    means = {g: float(np.mean(v)) for g, v in values.items()}
    baseline = k0_baseline()
    level = sc2_level(means['image'], means['text'], baseline)
    return write_json(records / 'pass_rule.json', dict(arm=arm, seeds=SEEDS, level=level,
                      status='pass' if level >= 1 else 'fail', mean=means, per_seed=values,
                      m={g: b['m'] for g, b in baseline.items()}, r={g: b['r'] for g, b in baseline.items()},
                      bounds={g: dict(lower=b['lower'], best=b['best']) for g, b in baseline.items()},
                      k0=baseline, readout='r0', written=now()))


def arm_location(arm):
    assert arm in (*CANDIDATES, 'qwen', 'gate_i_a_new'), arm
    if arm == 'qwen':
        return P.K0_ROOT, P.K0_ARM
    if arm == 'gate_i_a_new':
        return P.OUT / 'gate_i_a/new', 'qwen'
    return P.OUT / 'cstar/siglip1', arm


def image_groups(root, arm, seeds):
    """Import r28's scorer unchanged; ticket 06's checked loader supplies validation exports.

    Only its five-seed-only summary and historical gate are adapted for a requested seed subset.
    Clustering, retrieval, labels, and joining are r28's original computation.
    """
    assert seeds and seeds == sorted(set(seeds)) and all(type(s) is int and s in SEEDS for s in seeds)
    assert arm in (*CANDIDATES, P.K0_ARM, 'qwen'), arm
    P.use_older_rounds()
    import r28_valpick as V
    from r29_image_cells import Wiring
    w = Wiring(root, arm, 'siglip1', splits=('val',))

    def sidecar(enc, aspect, seed, ns_data, rt_data):
        assert enc == 'siglip1'
        path = w.path('val', aspect, seed)
        arm_read, a, s, digest = w._identity(read_json(path.with_suffix('.json')))
        assert (arm_read, a, s) == (arm, aspect, seed), path
        return digest, path

    with w, mock.patch.object(V, 'SEEDS', tuple(seeds)), \
            mock.patch.object(V, 'sidecar_dev_sha', sidecar), \
            mock.patch.object(V, 'gates', lambda enc, cells: {}):
        result = V.score_encoder('siglip1', seeds=seeds, log=lambda *args: None)
    P.check_older_imports()
    assert result['encoder'] == 'siglip1'
    out = group_cells({c: v['per_seed'] for c, v in result['cells'].items()}, IMAGE_CELLS, seeds)
    out['names_sha256'] = result['names_sha256']
    out['files'] = {f'{a}/s{s}': str(w.path('val', a, s)) for a in ASPECTS for s in seeds}
    return out


def groups(arm, seeds, root=None, records=P.RECORDS):
    text = read_text(records, arm)
    assert seeds and seeds == sorted(set(seeds)) and all(s in text['seeds'] for s in seeds), seeds
    indices = [text['seeds'].index(s) for s in seeds]
    selected = dict(per_seed=[text['per_seed'][i] for i in indices],
                    cells={c: [v[i] for i in indices] for c, v in text['cells'].items()})
    selected['mean'] = float(np.mean(selected['per_seed']))
    default_root, folder = arm_location(arm)
    image = image_groups(root or default_root, folder, seeds)
    return write_json(Path(records) / f'groups_{arm}.json', dict(arm=arm, seeds=seeds, image=image,
                      text=selected, root=str(root or default_root), readout='r0', written=now()))


def gate_i_a_verdict(weights, k0_cells, new_cells=None):
    """D23: equal weights, or observed r14 nondeterminism AND both fallback tests."""
    assert weights['gate'] == 'i_a_weights'
    details = weights['details']
    assert details['seed'] == 0 and details['encoder'] == 'siglip1'
    comparisons = details['comparisons']
    for name, comp in comparisons.items():
        assert name in ('new_vs_r14', 'r14_vs_r14_b')
        assert comp['runs'] == (['new', 'r14'] if name == 'new_vs_r14' else ['r14', 'r14_b'])
        assert set(comp['heads']) == set(ASPECTS)
        assert type(comp['equal']) is bool
        assert comp['equal'] == all(h['tensors_equal'] for h in comp['heads'].values())
        for h in comp['heads'].values():
            assert h['recorded_sha_ok'] == [True, True], 'weights differ from their sidecars'
            assert len(h['best_epoch']) == 2 and all(type(v) is int for v in h['best_epoch'])
    main = comparisons['new_vs_r14']
    result = dict(cuda_fallback=False, comparisons=comparisons)
    if main['equal']:
        return dict(status='pass', details=result)
    repeat = comparisons.get('r14_vs_r14_b')
    if repeat is None or repeat['equal']:
        result['reason'] = 'r14 nondeterminism not demonstrated'
        return dict(status='fail', details=result)
    assert new_cells is not None, 'CUDA fallback requires new-trainer validation cells'
    assert set(k0_cells) == set(new_cells) == set(IMAGE_CELLS)
    epochs = {a: abs(h['best_epoch'][0] - h['best_epoch'][1]) for a, h in main['heads'].items()}
    cells = {}
    for c in IMAGE_CELLS:
        old, new = numbers(k0_cells[c], 5), numbers(new_cells[c], 1)[0]
        lo, hi = min(old), max(old)
        cells[c] = dict(value=new, min=lo, max=hi, within=lo <= new <= hi)
    ok = all(d <= 2 for d in epochs.values()) and all(c['within'] for c in cells.values())
    result.update(cuda_fallback=ok, best_epoch_abs_diff=epochs, cells=cells)
    return dict(status='fallback' if ok else 'fail', details=result)


def write_gate(records, part, verdict):
    path = Path(records) / 'gates' / f'{part}.json'
    previous = read_json(path) if path.exists() else None
    if previous:
        assert previous['gate'] == part
    tries = previous['tries'] + 1 if previous else 1
    assert 1 <= tries <= 3, f'{part}: at most three recorded tries (§3.5)'
    return write_json(path, dict(gate=part, **verdict, tries=tries, written=now()))


def gate_i_a(records=P.RECORDS):
    weights = read_json(Path(records) / 'gates/i_a_weights.json')
    assert weights['gate'] == 'i_a_weights'
    comparisons = weights['details']['comparisons']
    main = comparisons['new_vs_r14']
    repeat = comparisons.get('r14_vs_r14_b')
    new = None
    baseline = k0_baseline()
    if not main['equal'] and repeat and not repeat['equal']:
        new = image_groups(P.OUT / 'gate_i_a/new', 'qwen', [0])['cells']
    return write_gate(records, 'i_a', gate_i_a_verdict(weights, baseline['image']['cells'], new))


def valpick_verdict(cells, frozen):
    assert set(cells) == set(frozen) == set(IMAGE_CELLS)
    diffs = {c: abs(numbers(cells[c], 1)[0] - numbers(frozen[c], 5)[0]) for c in IMAGE_CELLS}
    return dict(status='pass' if all(d <= 1e-6 for d in diffs.values()) else 'fail',
                details=dict(seed=0, tolerance=1e-6, max_abs_diff=max(diffs.values()),
                             abs_diff=diffs, cells=cells, expected_seed0={c: frozen[c][0] for c in IMAGE_CELLS}))


def gate_i_c_valpick(root=None, records=P.RECORDS):
    records = Path(records)
    if root is None:
        export = read_json(records / 'gates/i_c.json')
        assert export['gate'] == 'i_c' and export['details']['arm'] == P.K0_ARM
        assert export['details']['seeds'][0] == 0
        root = Path(export['details']['out_root'])
    image = image_groups(root, P.K0_ARM, [0])
    baseline = k0_baseline()
    verdict = valpick_verdict(image['cells'], baseline['image']['cells'])
    verdict['details'].update(root=str(root), k0=baseline, image=image)
    return write_gate(records, 'i_c_valpick', verdict)


# A standalone gate file cannot hide absent parts. i_c is the trainer/export part.
GATE_PARTS = {'iv': ('iv_store', 'iv_hash'), 'i_a': ('i_a_weights', 'i_a'),
              'i_c': ('i_c', 'i_c_valpick')}


def gates(records=P.RECORDS):
    records = Path(records)
    parts = {}
    for path in sorted((records / 'gates').glob('*.json')):
        doc = read_json(path)
        assert doc['gate'] == path.stem and doc['status'] in ('pass', 'fail', 'fallback', 'parked'), path
        assert type(doc['tries']) is int and 1 <= doc['tries'] <= 3
        assert isinstance(doc['details'], dict) and isinstance(doc['written'], str)
        parts[path.stem] = doc
    member_gate = {part: gate for gate, members in GATE_PARTS.items() for part in members}
    names = set(GATE_PARTS) | {member_gate.get(p, p) for p in parts}
    merged = {}
    for gate in sorted(names):
        required = GATE_PARTS.get(gate, (gate,))
        present = {p: parts[p] for p in required if p in parts}
        missing = [p for p in required if p not in parts]
        statuses = [p['status'] for p in present.values()]
        status = ('parked' if 'parked' in statuses else 'fail') if missing or any(
            s not in ('pass', 'fallback') for s in statuses) else ('fallback' if 'fallback' in statuses else 'pass')
        merged[gate] = dict(gate=gate, status=status, missing_parts=missing, parts=present,
                            tries=max((p['tries'] for p in present.values()), default=0))
    return write_json(records / 'gates.json', dict(gates=merged, written=now()))


def read_groups(records, arm, seeds):
    doc = read_json(Path(records) / f'groups_{arm}.json')
    assert doc['arm'] == arm and doc['seeds'] == seeds, (arm, doc['seeds'])
    for group, cells in (('image', IMAGE_CELLS), ('text', TEXT_CELLS)):
        numbers(doc[group]['per_seed'], len(seeds))
        assert set(doc[group]['cells']) == set(cells)
        for values in doc[group]['cells'].values():
            numbers(values, len(seeds))
    return doc


def best_epochs(root, arm, seeds):
    result = {}
    for seed in seeds:
        path = Path(root) / f'train/{arm}/run__s{seed}.json'
        doc = read_json(path)
        assert (doc['name'], doc['encoder'], doc['seed'], doc['status']) == (arm, 'siglip1', seed, 'ok'), path
        assert set(doc['best_epoch']) == set(ASPECTS)
        assert all(type(v) is int and v >= 0 for v in doc['best_epoch'].values()), path
        result[str(seed)] = doc['best_epoch']
    return result


def cell_changes(candidate, reference, seed0=False):
    assert set(candidate) == set(reference)
    result = {}
    for c in candidate:
        # Means over seeds for (a); matched seed 0 for local minus node402 (e).
        new = candidate[c][0] if seed0 else float(np.mean(candidate[c]))
        old = reference[c][0] if seed0 else float(np.mean(reference[c]))
        result[c] = dict(candidate=new, k0=old, difference=new-old)
    return result


def diagnostics(records=P.RECORDS):
    records = Path(records)
    rule = read_json(records / 'pass_rule.json')
    assert rule['arm'] in CANDIDATES and rule['seeds'] == SEEDS
    assert type(rule['level']) is int and rule['level'] in (0, 1, 2, 3)
    assert rule['status'] == ('pass' if rule['level'] >= 1 else 'fail')
    arm = rule['arm']
    candidate = read_groups(records, arm, SEEDS)
    local = read_groups(records, 'gate_i_a_new', [0])
    assert Path(local['root']) == P.OUT / 'gate_i_a/new', 'diagnostic (e) must use the gate new run'
    baseline = k0_baseline()
    plain = read_text(records, 'p1_siglip1')  # (c): plain input, even if D20 chose prefixed P1
    a = {g: cell_changes(candidate[g]['cells'], baseline[g]['cells']) for g in ('image', 'text')}
    e = {g: cell_changes(local[g]['cells'], baseline[g]['cells'], seed0=True) for g in ('image', 'text')}
    e.update(arm='gate_i_a_new', seed=0, root=str(P.OUT / 'gate_i_a/new'), reference='K0 node402 seed 0')
    root, folder = arm_location(arm)
    assert Path(candidate['root']) == root, 'candidate groups must name the training root'
    b = dict(candidate=best_epochs(root, folder, SEEDS), k0=best_epochs(P.K0_ROOT, P.K0_ARM, SEEDS))
    c = {}
    for aspect in ASPECTS:
        key = f'{aspect}/t2t'
        new, old = float(np.mean(candidate['text']['cells'][key])), plain['cells'][key][0]
        c[aspect] = dict(candidate=new, plain=old, difference=new-old)
    return write_json(records / 'diagnostics_val.json', dict(arm=arm, a=a, b=b, c=c, e=e,
                      pass_rule=rule, readout='r0', written=now()))


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('groups', 'sc1', 'sc2', 'gate-i-a', 'gate-i-c-valpick', 'gates', 'diagnostics'):
        p = sub.add_parser(command)
        p.add_argument('--records', type=Path, default=P.RECORDS)
        if command == 'groups':
            p.add_argument('--arm', choices=(*CANDIDATES, 'qwen', 'gate_i_a_new'), required=True)
            p.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, required=True)
            p.add_argument('--root', type=Path)
        elif command == 'gate-i-c-valpick':
            p.add_argument('--root', type=Path, help='new export root; default: gates/i_c.json details.out_root')
    return parser


def main(argv=None):
    args = cli_parser().parse_args(argv)
    if args.command == 'groups':
        result = groups(args.arm, args.seeds, args.root, args.records)
    elif args.command == 'gate-i-c-valpick':
        result = gate_i_c_valpick(args.root, args.records)
    else:
        result = {'sc1': sc1, 'sc2': sc2, 'gate-i-a': gate_i_a,
                  'gates': gates, 'diagnostics': diagnostics}[args.command](args.records)
    print(args.command, result.get('status', 'written'),
          f'level={result["level"]}' if 'level' in result else '')
    return 1 if result.get('status') in ('fail', 'inconclusive') else 0


if __name__ == '__main__':
    raise SystemExit(main())
