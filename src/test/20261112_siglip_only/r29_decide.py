"""r29 validation-only decisions (spec §§3.4–3.6, §6; D17, D20, D23, D25; SC9/A2).

No held row is read. Frozen K0 bounds come from r28 records, never from exports.
SC9: `sc9-replay --encoder siglip2|clip_l14` precedes `image-groups --encoder E [--root DIR]`;
`encoder-pick` reuses K*'s SC2 groups and the saved E2/E3 groups, with r28's Qwen as information only.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np

import r29_paths as P
import r29_captions as C   # the shared gate-record writer (three-try limit); also puts older rounds on sys.path via P

ASPECTS = P.ASPECTS
SEEDS = list(P.SEEDS)
IMAGE_CELLS = tuple(f'{m}/{a}' for m in ('ami', 'map') for a in ASPECTS)
TEXT_CELLS = tuple(f'{a}/{d}' for a in ASPECTS for d in ('t2i', 'i2t', 't2t'))
CANDIDATES = ('siglip1', 'siglip1_prefix')
SC9_ENCODERS = ('siglip1', 'siglip2', 'clip_l14')
SC9_NEW_ENCODERS = SC9_ENCODERS[1:]


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
    return datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M')   # Amsterdam time, as r29_captions


def write_once(path, doc, force=None):
    """Verdict records are written once. An existing one is refused unless `force` (a reason) is given; the record then
    keeps exactly its interface keys, and the reason and the replaced record go to `<name>_forced.json` beside it
    (an earlier side file is kept under `earlier`), because other tickets' readers assert the exact key set."""
    path = Path(path)
    if path.exists():
        assert force, f'{path.name} exists: a verdict record is written once (--force REASON replaces it)'
        side = path.with_name(f'{path.stem}_forced.json')
        note = dict(force_reason=force, replaced=read_json(path), written=now())
        if side.exists():
            note['earlier'] = read_json(side)
        write_json(side, note)
    else:
        assert not force, f'{path.name} does not exist: nothing to force'
    return write_json(path, doc)


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
    # The stored group value is the mean of the nine r0 cells, seed by seed (to 1e-12: ticket 04 reduces in its own
    # order, which can differ from numpy's in the last place; a wrong group differs by far more).
    for i in range(len(seeds)):
        mean = float(np.mean([flat[c][i] for c in TEXT_CELLS]))
        assert abs(mean - group[i]) <= 1e-12, f'{arm} seed {seeds[i]}: group_r0 {group[i]} != mean of nine cells {mean}'
    return dict(arm=arm, seeds=seeds, per_seed=group, cells=flat)


# Gate (i) is its parts i_a, i_b (a test: its record is read only if one exists) and i_c; then (ii) and (iii).
STAGE1_GATES = ('i_a', 'i_b', 'i_c', 'ii', 'iii')
CRASH_KEYS = {'arm', 'seed', 'status', 'tries', 'details', 'written'}
PICK_KEYS = {'kstar', 'kstar_form', 'p1_form', 'flag'}


def read_gates(records):
    doc = read_json(Path(records) / 'gates.json')
    assert isinstance(doc['gates'], dict), 'gates.json: no gates'
    return doc['gates']


def gate_blocker(name, gate):
    """None when the merged gate passes or falls back. Else (kind, text): `parked` (a part is parked, or failed at
    its third try: §3.5 allows three tries per gate), `open` (a part failed before try 3: a fix and a rerun can still
    pass it), `incomplete` (a part is missing and none failed). Only the parts that set the status count."""
    if gate['status'] in ('pass', 'fallback'):
        return None
    counted = [p for p in STATUS_FROM.get(name, tuple(gate['parts'])) if p in gate['parts']]
    failing = {p: gate['parts'][p] for p in counted if gate['parts'][p]['status'] not in ('pass', 'fallback')}
    for part, doc in sorted(failing.items()):
        if doc['status'] == 'parked' or doc['tries'] >= 3:
            return 'parked', f'gate {name} is parked ({part}: {doc["status"]} at try {doc["tries"]}; §3.5)'
    if failing:
        tries = max(d['tries'] for d in failing.values())
        return 'open', f'gate {name} still open (fail at try {tries} of 3)'
    return 'incomplete', f'gate {name} is incomplete (missing {gate["missing_parts"]})'


def require_gates(records):
    """SC1 and SC2 run only when gates (i), (ii) and (iii) pass or fall back (§3.5: a failed gate parks them, but only
    after its third try: a `fail` before try 3 is still open and refuses with that message)."""
    gates = read_gates(records)
    for name in STAGE1_GATES:
        if name == 'i_b' and name not in gates:
            continue
        assert name in gates, f'gate {name} not in gates.json: run `gates` first'
        blocked = gate_blocker(name, gates[name])
        assert blocked is None, blocked[1]
    return gates


def read_crash(records, arm):
    """`records/crash_<arm>.json` (ticket 15's runner, D21's second crash), or None."""
    path = Path(records) / f'crash_{arm}.json'
    if not path.exists():
        return None
    doc = read_json(path)
    assert set(doc) == CRASH_KEYS, (path.name, sorted(doc))
    assert doc['arm'] == arm and doc['status'] == 'out' and doc['tries'] == 2, doc
    assert type(doc['seed']) is int and isinstance(doc['details'], (str, dict)) and isinstance(doc['written'], str), doc
    return doc


def arm_out(records, gates, arm):
    """Why an arm is out, or None. Only on evidence (never a missing `text_<arm>.json`): its crash record, or for K2
    a parked gate (iv) part, or one that failed at its third try. A `fail` before try 3 is still open and a part
    missing with none failed is no evidence: both refuse."""
    crash = read_crash(records, arm)
    if crash:
        return f'second crash (D21), seed {crash["seed"]}'
    if arm == 'siglip1_prefix':
        blocked = gate_blocker('iv', gates['iv'])
        if blocked and blocked[0] == 'parked':
            return blocked[1]
        assert blocked is None, f'{blocked[1]}: K2 is not out'
    return None


def read_pick(records):
    doc = read_json(Path(records) / 'prefix_pick.json')
    assert set(doc) == PICK_KEYS, sorted(doc)
    assert doc['kstar'] in CANDIDATES and doc['kstar_form'] == ('plain' if doc['kstar'] == 'siglip1' else 'prefixed')
    assert doc['p1_form'] in ('plain', 'prefixed')
    assert doc['flag'] is None or isinstance(doc['flag'], str)
    return doc


def sc1(records=P.RECORDS, force=None):
    """SC1 seed 0; §3.5 row 1a (an arm out only on evidence); D20 P1 form chosen only when K2 is K*.
    Writes `prefix_pick.json` = {kstar, kstar_form, p1_form, flag}. Both arms out: no pick, stage 1 inconclusive."""
    records = Path(records)
    gates = require_gates(records)
    out = {arm: arm_out(records, gates, arm) for arm in CANDIDATES}
    out = {arm: why for arm, why in out.items() if why}
    if len(out) == 2:
        print(f'stage 1 is inconclusive: both arms out ({out}); no pick written')
        return dict(status='inconclusive', out=out)
    values = {}
    for arm in CANDIDATES:
        if arm in out:
            continue
        assert (records / f'text_{arm}.json').exists(), \
            f'text_{arm}.json missing: {arm} is not scored and has no crash or gate (iv) record, so it is not out'
        text = read_text(records, arm)
        assert text['seeds'][0] == 0, f'{arm}: SC1 requires seed 0 first'
        values[arm] = text['per_seed'][0]
    kstar = max(values, key=values.get)  # insertion order makes exact ties K1
    p1_form = 'plain'
    if kstar == 'siglip1_prefix':
        p1 = {f: read_text(records, 'p1_' + f)['per_seed'][0] for f in CANDIDATES}
        p1_form = 'prefixed' if p1['siglip1_prefix'] > p1['siglip1'] else 'plain'   # D20 exact tie: plain
    flag = None
    if out:
        (gone, why), = out.items()
        flag = f'{gone} out ({why}): {kstar} is K* with no comparison (§3.5 row 1a)'
    print('SC1 seed 0 text group:', values)
    return write_once(records / 'prefix_pick.json', dict(kstar=kstar,
                      kstar_form='plain' if kstar == 'siglip1' else 'prefixed', p1_form=p1_form, flag=flag), force)


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


def sc2(records=P.RECORDS, force=None):
    records = Path(records)
    require_gates(records)
    pick = read_pick(records)  # missing pick (or both arms out) refuses before any scoring
    arm = pick['kstar']
    groups_doc = read_json(records / f'groups_{arm}.json')
    assert groups_doc['arm'] == arm and groups_doc['seeds'] == SEEDS, 'SC2 requires all five seeds'
    values = {g: numbers(groups_doc[g]['per_seed'], 5) for g in ('image', 'text')}
    means = {g: float(np.mean(v)) for g, v in values.items()}
    baseline = k0_baseline()
    level = sc2_level(means['image'], means['text'], baseline)
    return write_once(records / 'pass_rule.json', dict(arm=arm, seeds=SEEDS, level=level,
                      status='pass' if level >= 1 else 'fail', mean=means, per_seed=values,
                      m={g: b['m'] for g, b in baseline.items()}, r={g: b['r'] for g, b in baseline.items()},
                      bounds={g: dict(lower=b['lower'], best=b['best']) for g, b in baseline.items()},
                      k0=baseline, readout='r0', written=now()), force)


def encoder_pick_verdict(per_seed):
    """SC9 (A2): mean of five seed groups, exact SigLIP 1 tie, matched-seed report."""
    assert set(per_seed) == set(SC9_ENCODERS), per_seed.keys()
    encoders = {}
    for enc in SC9_ENCODERS:
        values = numbers(per_seed[enc], 5)
        encoders[enc] = dict(score=float(np.mean(values)), per_seed=values,
                             seed_range=max(values)-min(values), role='candidate')
    # Stable order also fixes the runner-up when the two lower scores tie.
    ranking = sorted(SC9_ENCODERS, key=lambda enc: -encoders[enc]['score'])
    leader, runner = ranking[:2]
    tied = [enc for enc in ranking if encoders[enc]['score'] == encoders[leader]['score']]
    assert len(tied) == 1 or 'siglip1' in tied, f'SC9 top tie without SigLIP 1: {tied}'
    margin = encoders[leader]['score'] - encoders[runner]['score']
    ahead = [s for s in SEEDS if per_seed[leader][s] > per_seed[runner][s]]
    return dict(pick=leader, ranking=ranking, tie=tied if len(tied) > 1 else None,
                margin=margin, seeds_ahead=ahead, close_call=margin < encoders[leader]['seed_range'], encoders=encoders)


def sc9_replay_verdict(image, frozen):
    """D23 validation replay: every cell/seed and r28's recorded encoder score, inclusive 1e-6."""
    assert set(image['cells']) == set(frozen['cells']) == set(IMAGE_CELLS)
    expected = {c: numbers(frozen['cells'][c]['per_seed'], 5) for c in IMAGE_CELLS}
    diffs = {c: [abs(v-e) for v, e in zip(numbers(image['cells'][c], 5), expected[c])] for c in IMAGE_CELLS}
    score = numbers([image['mean']], 1)[0]
    expected_score = numbers([frozen['score']], 1)[0]
    score_diff = abs(score - expected_score)
    maximum = max(score_diff, *(d for values in diffs.values() for d in values))
    return dict(status='pass' if maximum <= 1e-6 else 'fail',
                details=dict(tolerance=1e-6, abs_diff=diffs, score_abs_diff=score_diff, max_abs_diff=maximum,
                             cells=image['cells'], expected_cells=expected, score=score, expected_score=expected_score))


def arm_location(arm):
    assert arm in (*CANDIDATES, 'qwen', 'gate_i_a_new'), arm
    if arm == 'qwen':
        return P.K0_ROOT, P.K0_ARM
    if arm == 'gate_i_a_new':
        return P.OUT / 'gate_i_a/new', 'qwen'
    return P.OUT / 'cstar/siglip1', arm


def image_groups(root, arm, seeds, encoder='siglip1'):
    """Import r28's scorer unchanged; ticket 06's checked loader supplies validation exports.

    Only its five-seed-only summary and historical gate are adapted for a requested seed subset.
    Clustering, retrieval, labels, and joining are r28's original computation.
    """
    assert seeds and seeds == sorted(set(seeds)) and all(type(s) is int and s in SEEDS for s in seeds)
    assert encoder in SC9_ENCODERS, encoder
    if encoder == 'siglip1':
        assert arm in (*CANDIDATES, P.K0_ARM, 'qwen'), arm
    else:
        assert arm in (*P.ENCODER_CAPTION_SOURCES[encoder][1:], P.K0_ARM, 'qwen'), (encoder, arm)
    P.use_older_rounds()
    import r28_valpick as V
    from r29_image_cells import Wiring
    w = Wiring(root, arm, encoder, splits=('val',))

    def sidecar(enc, aspect, seed, ns_data, rt_data):
        assert enc == encoder
        path = w.path('val', aspect, seed)
        arm_read, a, s, digest = w._identity(read_json(path.with_suffix('.json')))
        assert (arm_read, a, s) == (arm, aspect, seed), path
        return digest, path

    with w, mock.patch.object(V, 'SEEDS', tuple(seeds)), \
            mock.patch.object(V, 'sidecar_dev_sha', sidecar), \
            mock.patch.object(V, 'gates', lambda enc, cells: {}):
        result = V.score_encoder(encoder, seeds=seeds, log=lambda *args: None)
    P.check_older_imports()
    assert result['encoder'] == encoder
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


def frozen_image_encoder(encoder, r28_records):
    """r28's validation-only source for replay and Qwen's information row."""
    doc = read_json(Path(r28_records) / 'valpick.json')
    assert doc['kind'] == 'r28_valpick' and doc['seeds'] == SEEDS
    assert doc['arm'] == P.K0_ARM + ' (c*)' and doc['n_val'] == 6152
    row = doc['encoders'][encoder]
    assert row['encoder'] == encoder and row['n_val'] == 6152
    return row


def sc9_replay(encoder, records=P.RECORDS, r28_records=P.R28_RECORDS):
    """Replay r14's c* on E2/E3 before any new image group; no candidate or test export is read."""
    assert encoder in SC9_NEW_ENCODERS, encoder
    records = Path(records)
    part = f'sc9_replay_{encoder}'
    path = records / f'gates/{part}.json'
    if path.exists():
        prior = read_json(path)
        assert prior['gate'] == part
        assert prior['tries'] < 3, f'gate {part}: a fourth recorded try (§3.5 allows three)'
    frozen = frozen_image_encoder(encoder, r28_records)
    root = P.R14_OUT / 'cstar' / encoder
    image = image_groups(root, P.K0_ARM, SEEDS, encoder=encoder)
    verdict = sc9_replay_verdict(image, frozen)
    verdict['details'].update(encoder=encoder, arm=P.K0_ARM, seeds=SEEDS, root=str(root),
                              source=str(Path(r28_records) / 'valpick.json'), image=image)
    return write_gate(records, part, verdict)


def require_sc9_replay(records, encoder):
    assert encoder in SC9_NEW_ENCODERS, encoder
    part = f'sc9_replay_{encoder}'
    doc = read_json(Path(records) / f'gates/{part}.json')
    assert set(doc) == {'gate', 'status', 'tries', 'details', 'written'}, part
    assert doc['gate'] == part and doc['status'] == 'pass', f'{part} must pass before SC9'
    assert type(doc['tries']) is int and 1 <= doc['tries'] <= 3, part
    details = doc['details']
    assert (details['encoder'], details['arm'], details['seeds'], details['tolerance']) == \
           (encoder, P.K0_ARM, SEEDS, 1e-6), f'{part}: replay identity or tolerance'
    return doc


def encoder_image_groups(encoder, root=None, records=P.RECORDS, force=None):
    """SC9 E2/E3 group: validation only, five seeds, K*'s caption form (D13)."""
    gate = P.stage2_gate(records)
    require_sc9_replay(records, encoder)  # regression before the first new number
    import r29_encoders as E
    arm = E.kstar_source(encoder, records=records)
    root = Path(root) if root is not None else P.OUT / 'cstar' / encoder
    image = image_groups(root, arm, SEEDS, encoder=encoder)
    return write_once(Path(records) / f'image_groups_{arm}.json', dict(arm=arm, encoder=encoder,
                      seeds=SEEDS, image=image, root=str(root), written=now(), **P.a1_marks(gate)), force)


def encoder_pick(records=P.RECORDS, force=None, r28_records=P.R28_RECORDS, expected_roots=None):
    """SC9: reuse SC2's K* seed groups; E2/E3's saved groups; Qwen is information only (A2).
    expected_roots can supply temporary training roots for tests; production uses image-groups' defaults."""
    gate = P.stage2_gate(records)
    records = Path(records)
    if expected_roots is None:
        expected_roots = {enc: P.OUT / 'cstar' / enc for enc in SC9_NEW_ENCODERS}
    for enc in SC9_NEW_ENCODERS:
        require_sc9_replay(records, enc)
    pick = read_pick(records)
    arm = pick['kstar']
    kstar = read_json(records / f'groups_{arm}.json')
    rule = read_json(records / 'pass_rule.json')
    assert kstar['arm'] == rule['arm'] == arm and kstar['seeds'] == rule['seeds'] == SEEDS
    values = {'siglip1': numbers(kstar['image']['per_seed'], 5)}
    assert values['siglip1'] == numbers(rule['per_seed']['image'], 5), 'K* image groups differ from SC2'
    group_paths = {'siglip1': str(records / f'groups_{arm}.json')}
    import r29_encoders as E
    for enc in SC9_NEW_ENCODERS:
        source = E.kstar_source(enc, records=records)
        path = records / f'image_groups_{source}.json'
        doc = read_json(path)
        assert (doc['arm'], doc['encoder'], doc['seeds']) == (source, enc, SEEDS), path
        assert Path(doc['root']) == Path(expected_roots[enc]), f'{path}: unexpected encoder root {doc["root"]}'
        image = doc['image']
        checked = group_cells(image['cells'], IMAGE_CELLS, SEEDS)
        assert image['per_seed'] == checked['per_seed'] and image['mean'] == checked['mean'], path
        values[enc] = numbers(image['per_seed'], 5)
        group_paths[enc] = str(path)
    verdict = encoder_pick_verdict(values)
    qwen = frozen_image_encoder('qwen', r28_records)
    info = group_cells({c: v['per_seed'] for c, v in qwen['cells'].items()}, IMAGE_CELLS, SEEDS)
    verdict['encoders']['qwen'] = dict(score=numbers([qwen['score']], 1)[0], per_seed=info['per_seed'],
                                      seed_range=max(info['per_seed'])-min(info['per_seed']), role='information')
    verdict.update(kind='r29_encoder_pick', split='artelingo validation', seeds=SEEDS, kstar=arm,
                   pick_candidates=list(SC9_ENCODERS), information_rows=['qwen'], written=now(),
                   sources=dict(prefix_pick=str(records / 'prefix_pick.json'),
                                pass_rule=str(records / 'pass_rule.json'), groups=group_paths,
                                replay_gates={e: str(records / f'gates/sc9_replay_{e}.json') for e in SC9_NEW_ENCODERS},
                                valpick=str(Path(r28_records) / 'valpick.json')))
    verdict.update(P.a1_marks(gate))
    return write_once(records / 'encoder_pick.json', verdict, force)


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
    assert repeat is not None, 'r14_b repeat missing: run r14\'s trainer once more first (§3.5); no try spent'
    if repeat['equal']:
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
    """The shared writer: `records/gates/<part>.json`, tries counted, a fourth recorded try raises (§3.5)."""
    return C.write_gate_record(Path(records) / 'gates', verdict, gate=part)


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


# A standalone gate file cannot hide absent parts. i_c is the trainer/export part. Every part must be present; the
# status comes from every part (iv, i_c: co-equal) or, for i_a, from ticket 05's verdict alone: ticket 03's
# `i_a_weights` says `fail` whenever the weights differ, which D23's fallback can still pass.
GATE_PARTS = {'iv': ('iv_store', 'iv_hash'), 'i_a': ('i_a_weights', 'i_a'),
              'i_c': ('i_c', 'i_c_valpick')}
STATUS_FROM = {'i_a': ('i_a',)}


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
        statuses = [present[p]['status'] for p in STATUS_FROM.get(gate, required) if p in present]
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
    pick = read_pick(records)
    assert pick['kstar'] == arm, (pick['kstar'], arm)
    # (c): "plain SigLIP 1" is the frozen encoder's caption embedding in the form K*'s tower reads (`kstar_form`:
    # prefixed when K* is K2), at r0 -- not D20's `p1_form`, which is the test comparator's form.
    plain = read_text(records, 'p1_siglip1_prefix' if pick['kstar_form'] == 'prefixed' else 'p1_siglip1')
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
    return write_json(records / 'diagnostics_val.json', dict(arm=arm, a=a, b=b, c=c, c_input_form=pick['kstar_form'], e=e,
                      pass_rule=rule, readout='r0', written=now()))


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('groups', 'sc1', 'sc2', 'gate-i-a', 'gate-i-c-valpick', 'gates', 'diagnostics',
                    'sc9-replay', 'image-groups', 'encoder-pick'):
        p = sub.add_parser(command)
        p.add_argument('--records', type=Path, default=P.RECORDS)
        if command in ('sc1', 'sc2', 'encoder-pick', 'image-groups'):
            p.add_argument('--force', metavar='REASON', help='replace an existing record; the reason is kept in a side file')
        if command == 'groups':
            p.add_argument('--arm', choices=(*CANDIDATES, 'qwen', 'gate_i_a_new'), required=True)
            p.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, required=True)
            p.add_argument('--root', type=Path)
        elif command == 'gate-i-c-valpick':
            p.add_argument('--root', type=Path, help='new export root; default: gates/i_c.json details.out_root')
        elif command in ('sc9-replay', 'image-groups'):
            p.add_argument('--encoder', choices=SC9_NEW_ENCODERS, required=True)
            if command == 'image-groups':
                p.add_argument('--root', type=Path, help='default: OUT/cstar/<encoder>')
    return parser


def main(argv=None):
    args = cli_parser().parse_args(argv)
    if args.command == 'groups':
        result = groups(args.arm, args.seeds, args.root, args.records)
    elif args.command == 'gate-i-c-valpick':
        result = gate_i_c_valpick(args.root, args.records)
    elif args.command == 'sc9-replay':
        result = sc9_replay(args.encoder, args.records)
    elif args.command == 'image-groups':
        result = encoder_image_groups(args.encoder, args.root, args.records, args.force)
    elif args.command == 'encoder-pick':
        result = encoder_pick(args.records, args.force)
    elif args.command in ('sc1', 'sc2'):
        result = {'sc1': sc1, 'sc2': sc2}[args.command](args.records, args.force)
    else:
        result = {'gate-i-a': gate_i_a, 'gates': gates, 'diagnostics': diagnostics}[args.command](args.records)
    print(args.command, result.get('status', 'written'),
          f'level={result["level"]}' if 'level' in result else '')
    return 1 if result.get('status') in ('fail', 'inconclusive') else 0


if __name__ == '__main__':
    raise SystemExit(main())
