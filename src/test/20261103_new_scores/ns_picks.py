"""r17 ticket 09: stage 1 read-out picks and stage 0 gate (vi) (spec §3.4 "Retrieval", D8, D9, D10, D17, §3.5 (vi)).

r11's read-out rule on the set's VALIDATION split: `rt_readout`'s cross-fit halves (even and odd position of the rows
as given), the menu at the arm's width, the guard and the tie order, through `ac_select.arm_table`. One pick per arm
for its three aspects, in units of X-c*'s recorded told validation mAP range on that set (D17), never recomputed.

Arms (16 picks; keys `<arm>/<set>/<enc>`, the same as `ns_outcome.pick_key`):
    untold/<set>/<enc>      artelingo and mmart x four encoders (8): `ns_data.untold` of seeds 0 to 4, aspect-free, so one
                            array per seed under all three aspects (the eigendecompositions run once); width 6,144.
    matched/<set>/<enc>     SigLIP 2, SigLIP 1, CLIP L/14 on artelingo; four encoders on mmart (7; Qwen on ArtELingo keeps
                            r11's pick): the 40 validation probabilities (`ns_data.percept_probs(set, 'val', ...)`,
                            ticket 05's files) divided by their float32 row norm (r11's form), seeds 0 to 4, aspect-free.
    crl/mmart/clip_l14      CRL on MMArt (D10): `ns_data.crl` per aspect (r11's vector), unseeded, width 50.
Row order decides the cross-fit halves: ArtELingo sorted painting names, MMArt sha256 order (`mt_data.order`); the order is
asserted and `keys=` is never passed to `ac_select.arm_table` (it asserts sorted keys; r15 trap 1).

Units (D17): ArtELingo Qwen r11's `stage0a_ranges.json` (`map`); ArtELingo SigLIP 2, SigLIP 1, CLIP L/14 r14's
`records/stage1_<enc>.json['artret_ranges']`; MMArt r15's `records/stage1_<enc>.json['ret_ranges']`. Each is X-c*'s own
range on that set (max minus min over seeds 0 to 4 of validation mAP at R0), unrounded; the stored per-seed values are
checked to give the stored range.

    pick(arm, set_, enc, workers=8, out=None) -> {pick, w, k, scores, unit, table_path, ...}
        pick 'center+k10' (r11's name); w the map ('none', 'center', 'whiten256'); k the query-expansion neighbours;
        scores {'map': {aspect: cross-fit mAP at the pick}, 'score', 'kept', 'r0_map'}; unit {aspect: r};
        table_path the full table (all read-outs, relative to OUT) written to OUT/stage1/tables/.
    write_pick(key, record, out=None)   merge one record into OUT/stage1/picks.json (locked, atomic).
    gate_vi(arms=None, workers=8) -> {'pass', 'max_abs_diff', ...}   r11's stage 4 recomputed through the same rule.

CLI: python ns_picks.py list | pick --key untold/artelingo/qwen | gate-vi [--arms ...]   (NS_OUTPUTS sets OUT).
"""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import ac_paths  # noqa: E402
import ac_select  # noqa: E402
import mt_data  # noqa: E402
import rt_data  # noqa: E402
import rt_readout  # noqa: E402
import ns_data  # noqa: E402

ASPECTS = ns_paths.ASPECTS
SEEDS = ns_paths.SEEDS
TOL = ns_paths.TOL_REPLAY
SPLIT = ns_paths.DEV_SPLIT                       # 'val': the only split this module reads
PICKS = (tuple(('untold', s, e) for s in ns_paths.PAINTING_SETS for e in ns_paths.ENCODERS)
         + tuple(('matched', s, e) for s in ns_paths.PAINTING_SETS for e in ns_paths.ENCODERS
                 if (s, e) != ('artelingo', 'qwen'))
         + (('crl', 'mmart', ns_paths.CRL_ENCODER),))
assert len(PICKS) == 16


def key_of(arm, set_, enc):
    return f'{arm}/{set_}/{enc}'


def check_arm(arm, set_, enc):
    assert (arm, set_, enc) in PICKS, f'{key_of(arm, set_, enc)} is not one of the 16 new picks'


# ------------------------------------------------------------------------------------------------ units (D17)
def _range_file(set_, enc):
    """(path, reader) of X-c*'s recorded told validation mAP range on `set_`."""
    if set_ == 'artelingo' and enc == 'qwen':
        return ns_paths.ROUNDS['r11'] / 'stage0a_ranges.json', 'r11'
    if set_ == 'artelingo':
        return ns_paths.ROUNDS['r14'] / 'records' / f'stage1_{enc}.json', 'r14'
    return ns_paths.ROUNDS['r15'] / 'records' / f'stage1_{enc}.json', 'r15'


def ranges(set_, enc):
    """({aspect: r}, source) from the recorded file, unrounded; the stored per-seed values must give the stored range."""
    assert set_ in ns_paths.PAINTING_SETS and enc in ns_paths.ENCODERS, (set_, enc)
    path, kind = _range_file(set_, enc)
    doc = json.loads(Path(path).read_text())
    r, per_seed = {}, {}
    for a in ASPECTS:
        if kind == 'r11':
            r[a], per_seed[a] = float(doc['map'][a]['r']), [float(v) for v in doc['map'][a]['per_seed']]
        else:
            rec = doc['artret_ranges' if kind == 'r14' else 'ret_ranges']
            assert rec['encoder'] == enc, (path, rec['encoder'])
            r[a] = float(rec['ranges'][a])
            per_seed[a] = [float(rec['per_seed'][a][str(s)]) for s in SEEDS]
        assert len(per_seed[a]) == len(SEEDS) and np.isfinite(r[a]) and r[a] > 0, (path, a)
        assert abs(max(per_seed[a]) - min(per_seed[a]) - r[a]) <= 1e-12, f'{path} {a}: per-seed values do not give the range'
    return r, str(path)


# ------------------------------------------------------------------------------------------------ inputs
def unit32(p):
    """float32 rows divided by max(float32 row norm, 1e-12), nothing else: r11's `rt_data.embed('percept')`."""
    p = np.asarray(p).astype(np.float32)
    assert p.ndim == 2 and np.isfinite(p).all()
    return (p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def check_order(set_, keys):
    """The rows are in the evaluation order of the set: ArtELingo sorted names; MMArt sha256 (`mt_data.order`)."""
    keys = list(keys)
    assert keys and len(set(keys)) == len(keys), 'duplicate or no keys'
    if set_ == 'artelingo':
        assert keys == sorted(keys), 'ArtELingo rows must be in sorted painting-name order'
    else:
        assert keys == list(mt_data.order(keys)), 'MMArt rows must be in mt_data.order (sha256) order'
    assert len(keys) == ns_paths.N_ROWS[(set_, SPLIT)], (set_, len(keys))


def inputs(arm, set_, enc, split=SPLIT, data=ns_data):
    """(xs, labels): {(aspect, seed or None): float32 unit rows}, {aspect: int64 labels}, validation rows only. Aspect-free
    arms put ONE array object under the three aspects. `data` is `ns_data` (tests pass a stand-in)."""
    assert split == SPLIT, 'picks read the validation split only'
    check_arm(arm, set_, enc)
    keys = data.keys(set_, SPLIT)
    check_order(set_, keys)
    labels = {a: np.asarray(data.labels(set_, SPLIT, a), dtype=np.int64) for a in ASPECTS}
    xs = {}
    if arm == 'crl':
        for a in ASPECTS:
            xs[(a, None)] = data.crl(set_, SPLIT, a)
    else:
        for s in SEEDS:
            x = data.untold(set_, SPLIT, enc, s) if arm == 'untold' else unit32(data.percept_probs(set_, SPLIT, enc, s))
            for a in ASPECTS:
                xs[(a, s)] = x                          # one array: the eigendecompositions are shared
    width = {'untold': ns_paths.UNTOLD_WIDTH, 'matched': ns_paths.N_TOPICS, 'crl': ns_paths.N_CRL_VALUES}[arm]
    for k, x in xs.items():
        assert x.dtype == np.float32 and x.shape == (len(keys), width), (k, x.dtype, x.shape)
    assert all(y.shape == (len(keys),) for y in labels.values())
    return xs, labels


# ------------------------------------------------------------------------------------------------ the rule
def readout_doc(xs, labels, unit, workers=8):
    """One arm's validation table and pick by r11's rule (`ac_select.arm_table`, no `keys=`). unit {aspect: r_a}.
    Returns {'pick': name, 'readout': ReadOut, 'table': {name: {'map', 'score', 'kept'}}, 'ranges', 'dropped'}."""
    assert set(labels) == set(ASPECTS) and set(unit) == set(ASPECTS), (tuple(labels), tuple(unit))
    assert all(np.isfinite(v) and v > 0 for v in unit.values()), unit
    table, ro = ac_select.arm_table(xs, labels, unit, workers=workers)
    return dict(pick=rt_readout.name(ro), readout=ro, table=table, ranges={a: float(unit[a]) for a in ASPECTS},
                dropped=rt_readout.half_dropped(labels))


def _record(doc, source, n_val, table_rel):
    ro, row, base = doc['readout'], doc['table'][doc['pick']], doc['table']['none+k0']
    w = ro.w + (str(ro.size) if ro.w == 'whiten' else '')
    return dict(pick=doc['pick'], w=w, k=int(ro.k),
                scores=dict(map=row['map'], score=row['score'], kept=row['kept'], r0_map=base['map']),
                unit=doc['ranges'], unit_source=source, n_val=int(n_val), table_path=table_rel)


def pick(arm, set_, enc, workers=8, out=None, data=ns_data):
    """The new pick of one arm: the record in the index format (see the module docstring). Writes the full table."""
    check_arm(arm, set_, enc)
    out = Path(ns_paths.OUT if out is None else out)
    unit, source = ranges(set_, enc)
    xs, labels = inputs(arm, set_, enc, data=data)
    doc = readout_doc(xs, labels, unit, workers)
    n_val = len(next(iter(labels.values())))
    rel = f'stage1/tables/{arm}__{set_}__{enc}.json'
    ns_paths.write_json(out / rel, dict(arm=arm, set=set_, encoder=enc, pick=doc['pick'], table=doc['table'],
                                        ranges=doc['ranges'], unit_source=source, dropped=doc['dropped'], n_val=n_val))
    return _record(doc, source, n_val, rel)


def write_pick(key, record, out=None):
    """Merge `record` under `key` into OUT/stage1/picks.json (file lock, atomic replace): arms are queued separately."""
    assert key in {key_of(*p) for p in PICKS}, key
    path = Path(ns_paths.OUT if out is None else out) / 'stage1' / 'picks.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name('picks.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        have = json.loads(path.read_text()) if path.exists() else {}
        have[key] = record
        tmp = path.with_name('picks.pending.json')
        ns_paths.write_json(tmp, have)
        os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------------------------ gate (vi)
def replay_item(name, ours, theirs):
    return dict(name=name, ours=float(ours), theirs=float(theirs), abs_diff=abs(float(ours) - float(theirs)))


def gate_vi(arms=None, workers=8, data=rt_data, stored_dir=None, ranges_file=None):
    """Stage 0 gate (vi): the rule (`readout_doc`, the function every pick calls), fed r11's stored validation outputs
    (`rt_data.embed` on the ArtELingo validation paintings, sorted order, no `keys=`) with r11's r_a for every arm, gives
    r11's `stage4/scores.json` (cross-fit mAP per aspect, score, guard flag of every read-out) and `stage4/picks.json`
    within TOL_REPLAY. A subset of arms gives `values_ok` but cannot pass. Returns {'pass', 'values_ok', 'max_abs_diff', ...}."""
    stored_dir = Path(ns_paths.R11_OUT / 'stage4' if stored_dir is None else stored_dir)
    scores = json.loads((stored_dir / 'scores.json').read_text())
    picks = json.loads((stored_dir / 'picks.json').read_text())
    doc = json.loads(Path(ac_paths.R11_RANGES_FILE if ranges_file is None else ranges_file).read_text())
    r_a = {a: float(doc['map'][a]['r']) for a in ASPECTS}
    arms = tuple(data.ARMS if arms is None else arms)
    assert set(arms) <= set(data.ARMS) and len(set(arms)) == len(arms), arms
    labels = {a: np.asarray(v, dtype=np.int64) for a, v in data.labels(SPLIT).items()}
    items, got, pick_ok = [], {}, True
    for arm in arms:
        xs = {}
        for s in (SEEDS if arm in data.SEEDED else (None,)):
            if arm in data.ASPECT_FREE:
                x = data.embed(arm, SPLIT, 'emotion', s)
                xs.update({(a, s): x for a in ASPECTS})                # one matrix: its eigendecomposition is shared
            else:
                xs.update({(a, s): data.embed(arm, SPLIT, a, s) for a in ASPECTS})
        res = readout_doc(xs, labels, r_a, workers)
        assert set(res['table']) == set(scores[arm]), (arm, 'read-out menu differs')
        for name, t in res['table'].items():
            for a in ASPECTS:
                items.append(replay_item(f'{arm}/{name}/{a}/crossfit mAP', t['map'][a], scores[arm][name]['map'][a]))
            items.append(replay_item(f'{arm}/{name}/score', t['score'], scores[arm][name]['score']))
            pick_ok &= t['kept'] == scores[arm][name]['kept']
        pick_ok &= res['pick'] == picks[arm]
        got[arm] = res['pick']
    worst = max(i['abs_diff'] for i in items)
    values_ok = bool(worst <= TOL and pick_ok)
    complete = set(arms) == set(data.ARMS)
    return {'pass': bool(values_ok and complete), 'values_ok': values_ok, 'complete': complete, 'max_abs_diff': float(worst),
            'tol': TOL, 'n_values': len(items), 'picks': got, 'picks_and_flags_equal': bool(pick_ok), 'arms': list(arms),
            'items': items}


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='the 16 pick keys')
    s = sub.add_parser('pick', help='one arm -> OUT/stage1/picks.json (merged) and OUT/stage1/tables/')
    s.add_argument('--key', required=True, choices=[key_of(*x) for x in PICKS])
    g = sub.add_parser('gate-vi', help='r11 stage 4 recomputed -> OUT/stage0/gate_vi.json (about 1,040 s at 8 threads)')
    g.add_argument('--arms', nargs='+', default=None, choices=list(rt_data.ARMS))
    args = p.parse_args(argv)
    t0 = time.monotonic()
    if args.cmd == 'list':
        print('\n'.join(key_of(*x) for x in PICKS))
        return 0
    if args.cmd == 'pick':
        arm, set_, enc = args.key.split('/')
        rec = pick(arm, set_, enc, args.workers)
        path = write_pick(args.key, rec)
        print(f'{args.key}: pick {rec["pick"]} score {rec["scores"]["score"]:.4f} -> {path} ({time.monotonic() - t0:.0f}s)')
        return 0
    res = gate_vi(args.arms, args.workers)
    path = ns_paths.OUT / 'stage0' / 'gate_vi.json'
    ns_paths.write_json(path, res)
    print(f'gate-vi: {"PASS" if res["pass"] else "FAIL"} max_abs_diff {res["max_abs_diff"]:.3g} values_ok {res["values_ok"]} '
          f'({time.monotonic() - t0:.0f}s) {path}')
    return 0 if res['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
