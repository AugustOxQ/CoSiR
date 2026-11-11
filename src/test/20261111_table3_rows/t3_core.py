"""Table 3 rows, core (round r27): one generic read-out pick and one generic test scorer, both thin wrappers on the
project's existing code, so the numbers match the existing table to the last digit.

Reused, not rewritten:
    pick, unseeded   `om_stage1.readout_doc` -> `ac_select.arm_table` (r11's cross-fit rule, one pick for the three aspects)
    pick, seeds      `rt_readout.crossfit_all` per seed (it accepts seeds None or 0..4 only), mean over the kept seeds, then
                     `ac_select.readout_scores / readout_guard / pick_readout`: the same last step as `arm_table`
    test scoring     `ns_ret.arm_ap(get_val, get_test, pick, labels, seeds=None)`: the map is fitted on all validation rows
                     and applied to the test rows; queries and gallery are the labelled test rows. A seed subset is scored
                     per seed (seeds=None each time) and averaged, because `arm_ap` accepts only None or (0..4).
Splits: the code calls the test split `held` (`op.HELD_SPLIT['artelingo']`); printed text says "test".

Row getters are `get(aspect, seed) -> float32 unit rows [n, d]` (seed None for an unseeded arm). Callers cache them.
"""
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OM_ROUND = HERE.parent / '20261103_outside_methods'
if str(OM_ROUND) not in sys.path:
    sys.path.insert(0, str(OM_ROUND))

import om_paths as op  # noqa: E402  (first: puts r17's folder and the older rounds on sys.path, as om_stage1 does)
import om_stage1 as s1  # noqa: E402

import ac_select  # noqa: E402
import ns_paths  # noqa: E402
import ns_picks  # noqa: E402
import ns_ret  # noqa: E402
import rt_readout  # noqa: E402

SET = 'artelingo'
DEV_SPLIT = op.DEV_SPLIT                      # 'val'
TEST_SPLIT = op.HELD_SPLIT[SET]               # 'held': the paper's test split (12,281 paintings)
ASPECTS = tuple(op.ASPECTS)
ALL_SEEDS = tuple(op.SEEDS)                   # (0, 1, 2, 3, 4)
OUT = HERE / 'out'
LOGS = HERE / 'logs'


# ================================================================================================ seeds
def parse_seeds(text):
    """'0,1,3' -> (0, 1, 3): a non-empty, sorted, duplicate-free subset of 0..4."""
    seeds = tuple(int(t) for t in str(text).split(',') if t.strip() != '')
    assert seeds and list(seeds) == sorted(set(seeds)) and set(seeds) <= set(ALL_SEEDS), \
        f'--seeds must be a sorted subset of 0..4, got {text!r}'
    return seeds


# ================================================================================================ the pick
def _table_from_rows(rows, unit):
    """arm_table's last step: {name: {map, score, kept}} and the picked ReadOut, from {ReadOut: {aspect: mAP}}."""
    sc, keep = ac_select.readout_scores(rows, unit), ac_select.readout_guard(rows, unit)
    table = {rt_readout.name(x): {'map': {a: float(m[a]) for a in m}, 'score': float(sc[x]), 'kept': bool(keep[x])}
             for x, m in rows.items()}
    return table, ac_select.pick_readout(rows, unit)


def pick_readout(get_val, labels, unit, seeds=None, workers=8):
    """r11's read-out pick on the validation split: one read-out for the three aspects, units `unit` {aspect: r}.
    seeds None: unseeded arm (`om_stage1.readout_doc`). seeds a tuple: mean over those seeds of the cross-fit mAP.
    Returns {pick, readout (w, size, k), table, dropped, per_seed_map (None or {seed: {name: {aspect: mAP}}})}."""
    assert set(labels) == set(ASPECTS) and set(unit) == set(ASPECTS)
    if seeds is None:
        xs = {(a, None): get_val(a, None) for a in ASPECTS}
        doc = s1.readout_doc(xs, labels, unit, workers=workers)
        table, ro, per_seed = doc['table'], doc['readout'], None
        dropped = doc['dropped']
    else:
        seeds = tuple(seeds)
        width = {get_val(a, s).shape[1] for a in ASPECTS for s in seeds}
        assert len(width) == 1, ('one width per arm', width)
        ros = rt_readout.menu(width.pop())
        per, per_seed = [], {}
        for s in seeds:
            xs = {(a, None): get_val(a, s) for a in ASPECTS}          # seed None: crossfit_all takes 0..4 or None only
            res = rt_readout.crossfit_all(xs, labels, ros, workers=workers)
            per.append(res)
            per_seed[str(s)] = {rt_readout.name(x): {a: float(m[a]) for a in m} for x, m in res.items()}
        rows = {x: {a: float(np.mean([p[x][a] for p in per])) for a in ASPECTS} for x in ros}
        table, ro = _table_from_rows(rows, unit)
        dropped = rt_readout.half_dropped(labels)
    return dict(pick=rt_readout.name(ro), readout=dict(w=ro.w, size=None if ro.size is None else int(ro.size), k=int(ro.k)),
                table=table, dropped=dropped, per_seed_map=per_seed)


# ================================================================================================ the test score
def score_test(get_val, get_test, pick, labels_test, seeds=None, workers=8):
    """Test mAP per aspect at `pick`. Returns {aspect: {map, per_seed ({seed: mAP} or None), n_queries, dropped}}.
    A seed tuple is scored per seed through `ns_ret.arm_ap(seeds=None)` and averaged (equal to `arm_ap`'s own mean over
    seeds 0..4 when all five are kept: its `point` is np.mean of the per-seed mAPs)."""
    out = {}
    for a in ASPECTS:
        y = np.asarray(labels_test[a], dtype=np.int64)
        if seeds is None:
            r = ns_ret.arm_ap(lambda _s, a=a: get_val(a, None), lambda _s, a=a: get_test(a, None), pick, y, None, workers)
            out[a] = dict(map=float(r['point']), per_seed=None, n_queries=int(len(r['rows'])), dropped=int(r['dropped']))
            continue
        per, nq, dropped = {}, None, None
        for s in seeds:
            r = ns_ret.arm_ap(lambda _s, a=a, s=s: get_val(a, s), lambda _s, a=a, s=s: get_test(a, s), pick, y, None, workers)
            per[str(s)] = float(r['point'])
            assert nq in (None, len(r['rows'])) and dropped in (None, r['dropped']), 'queries differ between seeds'
            nq, dropped = int(len(r['rows'])), int(r['dropped'])
        out[a] = dict(map=float(np.mean(list(per.values()))), per_seed=per, n_queries=nq, dropped=dropped)
    return out


# ================================================================================================ units and labels
def tpips_unit():
    """Qwen c*'s validation mAP ranges (r11's `stage0a_ranges.json`), as TPIPS: (unit {aspect: r}, source)."""
    return s1.units('tpips', SET)


def clip_l14_unit():
    """CLIP L/14 c*'s validation mAP ranges (r14's record), as r17's matched PercepT pick: (unit, source)."""
    return ns_picks.ranges(SET, 'clip_l14')


def labels(split):
    import om_data
    return {a: np.asarray(om_data.labels(SET, split, a), dtype=np.int64) for a in ASPECTS}


def keys(split):
    import om_data
    return list(om_data.keys(SET, split))


def check_validation_keys():
    """The pick reads validation rows only: sorted painting names (the cross-fit halves are right only then), the expected
    count, and no key shared with the test split."""
    val, test = keys(DEV_SPLIT), keys(TEST_SPLIT)
    ns_picks.check_order(SET, val)
    shared = set(val) & set(test)
    assert not shared, f'{len(shared)} validation keys are also test keys'
    return val


# ================================================================================================ output
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)
    return path


def check_rows(x, n, what, width=None):
    """float32 [n, d] unit rows, finite (norm within 1e-4 of 1, as `om_stage1._inputs`)."""
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[0] == n, (what, x.dtype, x.shape)
    assert width is None or x.shape[1] == width, (what, x.shape, width)
    assert np.isfinite(x).all(), f'{what}: non-finite values'
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-4, f'{what}: rows are not unit rows'
    return x
