"""r17 retrieval scorer (ticket 08): arm AP, the paired bootstrap with three seed rules, SC6 to SC8 cells, gate (ii).

Spec: docs/superpowers/specs/2026-11-03-new-scores-design.md §6 SC6 to SC8, §3.4 "Retrieval", "Untold retrieval",
"Intervals", §3.5 stage 0 (ii), §3.7, D8, D9, D17, D19. Older code does the work (the replays prove it): the map and
the AP are `ac_stats.arm_retrieval` (r12; r11's `rt_run.stage5` steps: map fitted on ALL validation rows, applied to the
scored rows, query expansion on the scored gallery with k fixed on validation), the bootstrap is `rt_stats.paired` (r11).

Frozen interface (index "Retrieval scorer (08)"; a later need adds a function and never changes a signature):
  arm_ap(get_val, get_eval, pick, labels, seeds) -> {ap, point, per_seed, rows} (+ dropped)
  paired_ret(cand_ap, comp_ap, r=None, draws=10000, comp_seeds='none'|'apart'|'shared', rng_seed=0)
      -> {cand, comp, d, lo, hi, r, outcome, n_draws}
        'none'   unseeded comparator: exactly r11's draws (`rt_stats.paired`, bit for bit at rng_seed 0)
        'apart'  two seeded arms: per draw the rows, then c*'s five seeds, then the comparator's five seeds (D19)
        'shared' two seeded arms, one seed draw for both (`rt_stats.paired`'s rule; told against untold, same runs)
  score_cells(set, enc, scs, picks, draws) -> list of cells (format of the index "Cells"; ids from
      `ns_outcome.claim_cells()`); optional `workers=8`.

Arms (all on the aspect's labelled rows as queries and gallery; relevance by that aspect's label):
  cstar    X-c*'s own head per aspect, seeds 0 to 4 (`ns_data.cstar`, unit rows)
  untold   X-c*'s three heads joined (`ns_data.untold`), scored with each aspect's labels (D8)
  plain    X's default image (`ns_data.plain`), unseeded
  matched  X's matched PercepT: the 40 probabilities divided by their float32 norm (r11's form), seeds 0 to 4
  crl      CLIP L/14 CRL (`ns_data.crl`, r11's form), unseeded, aspect-specific
Picks: reused ones from their rounds' records (`reused_pick`); new ones (untold, matched but Qwen on ArtELingo, CRL on
MMArt) from the committed `stage1_picks.json` through `ns_paths.read_committed_record`, key `<arm>/<set>/<enc>`.
r per cell: X-c*'s recorded told validation mAP range on that set (`recorded_range`, D17).
The Qwen ArtELingo PercepT validation rows are r11's stored ones (its pick and probabilities are kept, D9).

Gate (ii) (§3.5): `gate_ii_scorer`, `gate_ii_crl`, `gate_ii_percept` (each {'pass', 'max_abs_diff', 'items', ...}) and
`gate_ii` (all three). CLI: `python ns_ret.py gate-ii [--part scorer|crl|percept] [--draws N] [--out FILE]`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import ac_stats  # noqa: E402  (r12: arm_retrieval)
import ns_data as data  # noqa: E402
import rt_data  # noqa: E402
import rt_stats  # noqa: E402

SEEDS = tuple(ns_paths.SEEDS)
ASPECTS = tuple(ns_paths.ASPECTS)
_CHUNK = 250                                  # draws held in memory at once, as rt_stats
SEED_MODES = ('none', 'apart', 'shared')
RETRIEVAL_SCS = ('SC6', 'SC7', 'SC8')
R11_OUT = ns_paths.R11_OUT


# ================================================================================================ arm AP
def arm_ap(get_val, get_eval, pick, labels, seeds, workers=8):
    """Per-query AP of one arm under its picked read-out (`ac_stats.arm_retrieval`): the map is fitted on ALL validation
    rows (`get_val(seed)`, float32 unit rows, no labels) and applied to the scored rows (`get_eval(seed)`); query expansion
    with the pick's k runs on the scored gallery. `labels` int64 [n] of the scored rows (-1 = none): queries and gallery
    are the labelled rows. `seeds`: None (unseeded arm) or (0, 1, 2, 3, 4). Returns {'ap': float64 [nq] or [5, nq],
    'point': mean mAP (over seeds), 'per_seed': [5 mAPs] or None, 'rows': int64 [nq], 'dropped': int}."""
    labels = np.asarray(labels)
    assert labels.ndim == 1 and labels.dtype == np.int64, f'labels must be int64 [n], got {labels.dtype} {labels.shape}'
    assert seeds is None or tuple(seeds) == SEEDS, f'seeds must be None or {SEEDS}, got {seeds}'

    def checked(get):
        def f(seed):
            x = get(seed)
            assert x.dtype == np.float32 and x.ndim == 2, (x.dtype, x.shape)
            return x
        return f

    res = ac_stats.arm_retrieval(checked(get_val), checked(get_eval), pick, labels,
                                 seeds=None if seeds is None else SEEDS, workers=workers)
    ap = res['ap']
    assert ap.dtype == np.float64 and ap.shape[-1] == len(res['rows']) and (ap.ndim == 1) == (seeds is None)
    return {k: res[k] for k in ('ap', 'point', 'per_seed', 'rows', 'dropped')}


# ================================================================================================ bootstrap
def _side(x, name):
    assert isinstance(x, np.ndarray) and x.dtype == np.float64, f'{name}: float64 array needed'
    assert x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 5), f'{name}: shape {x.shape}, need [n] or [5, n]'
    return x


def _word(lo, r):
    """r11's rule (`rt_stats.outcome`): WIN lo > 0; LEVEL lo > -r; BELOW. Without r: 'WIN' or 'not WIN'."""
    if lo > 0:
        return 'WIN'
    return 'not WIN' if r is None else rt_stats.outcome(lo, r)


def _boot(cand, comp, draws, mode, rng_seed):
    """Joint draws of candidate minus comparator. One default_rng(rng_seed); per draw, in order: I = integers(0, n, n);
    if the candidate is seeded S_c = choice([0..4], 5); if the comparator is seeded: 'shared' reuses S_c, 'apart' draws
    S_m = choice([0..4], 5) after S_c. The mean over I is a count-matrix product (rt_stats.paired's chunks)."""
    n = cand.shape[-1]
    cols = np.vstack([np.atleast_2d(cand), np.atleast_2d(comp)])
    kc = 1 if cand.ndim == 1 else 5
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(draws, dtype=np.float64)
    for lo in range(0, draws, _CHUNK):
        m = min(_CHUNK, draws - lo)
        counts = np.empty((m, n), dtype=np.float64)
        sel_c = np.empty((m, 5), dtype=np.int64) if cand.ndim == 2 else None
        sel_m = np.empty((m, 5), dtype=np.int64) if comp.ndim == 2 else None
        for b in range(m):
            counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
            if cand.ndim == 2:
                sel_c[b] = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
            if comp.ndim == 2:
                sel_m[b] = rng.choice([0, 1, 2, 3, 4], size=5, replace=True) if mode == 'apart' else sel_c[b]
        means = counts @ cols.T / n
        mc, mm = means[:, :kc], means[:, kc:]
        mc = np.take_along_axis(mc, sel_c, axis=1).mean(axis=1) if cand.ndim == 2 else mc[:, 0]
        mm = np.take_along_axis(mm, sel_m, axis=1).mean(axis=1) if comp.ndim == 2 else mm[:, 0]
        boot[lo:lo + m] = mc - mm
    return boot


def paired_ret(cand_ap, comp_ap, r=None, draws=10000, comp_seeds='none', rng_seed=0):
    """Paired bootstrap of candidate minus comparator over queries and seeds (module docstring for the three rules).
    Both float64 [n] or [5, n] with the same n (queries in the same order: the caller asserts equal `rows`). The point
    value d is the difference of the means over all queries and seeds; the interval is the 2.5 and 97.5 percentiles.
    Returns {cand, comp, d, lo, hi, r, outcome, n_draws}; outcome 'WIN' / 'LEVEL' / 'BELOW' (with r) or 'WIN' / 'not WIN'."""
    assert comp_seeds in SEED_MODES, f'comp_seeds {comp_seeds!r} not in {SEED_MODES}'
    cand, comp = _side(cand_ap, 'cand'), _side(comp_ap, 'comp')
    n = cand.shape[-1]
    assert comp.shape[-1] == n and n > 0, f'query counts differ: {cand.shape} vs {comp.shape}'
    if comp_seeds == 'none':
        assert comp.ndim == 1, "comp_seeds='none' is an unseeded comparator"
    else:
        assert cand.ndim == 2 and comp.ndim == 2, f"comp_seeds={comp_seeds!r} needs two seeded arms"
    if comp_seeds == 'none' and rng_seed == 0:
        p = rt_stats.paired(cand, comp, draws)                    # r11's function itself: bit for bit
        lo, hi = p['interval']
        d = p['d']
    else:
        lo, hi = np.percentile(_boot(cand, comp, draws, comp_seeds, rng_seed), [2.5, 97.5]).tolist()
        d = float(cand.mean() - comp.mean())
    return dict(cand=float(cand.mean()), comp=float(comp.mean()), d=float(d), lo=float(lo), hi=float(hi),
                r=None if r is None else float(r), outcome=_word(lo, r), n_draws=int(draws))


# ================================================================================================ records
def _json(path):
    return json.loads(Path(path).read_text())


def _r11_picks():
    return _json(R11_OUT / 'stage4/picks.json')


def _stage1_record(round_, enc):
    return _json(ns_paths.ROUNDS[round_] / f'records/stage1_{enc}.json')


def reused_pick(arm, set_, enc):
    """The recorded read-out name of a reused arm, or None when the arm's pick is new (stage 1 of r17).
    arm in cstar | plain | matched | crl. ArtELingo Qwen: r11's `stage4/picks.json` (cstar, plain, percept = matched);
    ArtELingo SigLIP 2, SigLIP 1, CLIP L/14: r14's stage 1 record (`artret_picks` cstar, frozen = plain); CRL on ArtELingo:
    r11's `crl`. MMArt: r15's stage 1 record (`picks.arms` cstar, frozen = plain)."""
    assert arm in ('cstar', 'plain', 'matched', 'crl', 'untold'), arm
    assert set_ in ns_paths.PAINTING_SETS and enc in ns_paths.ENCODERS, (set_, enc)
    if arm == 'untold':
        return None
    if arm == 'crl':
        assert enc == ns_paths.CRL_ENCODER, enc
        return _r11_picks()['crl'] if set_ == 'artelingo' else None
    if set_ == 'artelingo' and enc == 'qwen':
        return _r11_picks()[{'matched': 'percept'}.get(arm, arm)]
    if arm == 'matched':
        return None
    key = {'cstar': 'cstar', 'plain': 'frozen'}[arm]
    if set_ == 'artelingo':
        return _stage1_record('r14', enc)['artret_picks'][key]['pick']
    return _stage1_record('r15', enc)['picks']['arms'][key]['pick']


def recorded_range(set_, enc, aspect):
    """(r, source): X-c*'s recorded told validation mAP range on `set_` (D17), unrounded. ArtELingo Qwen: r11's
    `stage0a_ranges.json`; ArtELingo others: r14's stage 1 record `artret_ranges`; MMArt: r15's `ret_ranges`."""
    assert aspect in ASPECTS
    if set_ == 'artelingo' and enc == 'qwen':
        r = _json(ns_paths.ROUNDS['r11'] / 'stage0a_ranges.json')['map'][aspect]['r']
        src = 'r11 stage0a_ranges.json (Qwen c*, ArtELingo validation, mAP at R0)'
    elif set_ == 'artelingo':
        r = _stage1_record('r14', enc)['artret_ranges']['ranges'][aspect]
        src = f'r14 stage1_{enc}.json artret_ranges (X-c* ArtELingo validation, mAP at R0)'
    else:
        assert set_ == 'mmart'
        r = _stage1_record('r15', enc)['ret_ranges']['ranges'][aspect]
        src = f'r15 stage1_{enc}.json ret_ranges (X-c* MMArt validation, mAP at R0)'
    assert np.isfinite(r) and r > 0, (set_, enc, aspect, r)
    return float(r), src


def new_pick(record, arm, set_, enc):
    key = f'{arm}/{set_}/{enc}'
    if key not in record:
        raise KeyError(f'no stage 1 pick {key!r} in the picks record')
    return record[key]['pick']


# ================================================================================================ arm rows
def _unit(x):
    """float32 rows divided by max(float32 row norm, 1e-12): r11's `rt_data.embed('percept')`, nothing else."""
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def r11_percept_val(seed):
    """r11's stored Qwen PercepT validation rows (ArtELingo), unit float32: the rows its pick was made and applied with."""
    return rt_data.embed('percept', 'val', 'emotion', seed)


def _percept_rows(set_, split, enc, seed):
    if (set_, enc, split) == ('artelingo', 'qwen', 'val'):
        return r11_percept_val(seed)
    return _unit(data.percept_probs(set_, split, enc, seed))


class _Rows:
    """Getters (get_val, get_eval) of one arm; arms with one matrix per set (plain) keep it between aspects."""

    def __init__(self, set_, enc):
        self.set, self.enc, self.held = set_, enc, ns_paths.HELD_SPLIT[set_]
        self._plain = {}

    def _plain_rows(self, split):
        if split not in self._plain:
            x = data.plain(self.set, split, self.enc)
            # MMArt: r15's retrieval form divides once more (`mt_ret.unit(features)`, `view_rows`); ArtELingo rows are r11's
            self._plain[split] = _unit(x) if self.set == 'mmart' else x
        return self._plain[split]

    def seeds(self, arm):
        return SEEDS if arm in ('cstar', 'untold', 'matched') else None

    def getter(self, arm, aspect, split):
        s_, e_ = self.set, self.enc
        if arm == 'cstar':
            return lambda s: data.cstar(s_, split, e_, aspect, s)
        if arm == 'untold':
            return lambda s: data.untold(s_, split, e_, s)
        if arm == 'plain':
            return lambda s: self._plain_rows(split)
        if arm == 'matched':
            return lambda s: _percept_rows(s_, split, e_, s)
        if arm == 'crl':
            return lambda s: data.crl(s_, split, aspect)
        raise AssertionError(arm)

    def ap(self, arm, aspect, pick, y, workers):
        return arm_ap(self.getter(arm, aspect, 'val'), self.getter(arm, aspect, self.held), pick, y, self.seeds(arm), workers)


# ================================================================================================ cells
# comparison per SC: (candidate arm, comparator arm, seed rule)
SC_ARMS = {'SC6': ('cstar', 'crl', 'none'), 'SC7': ('untold', 'plain', 'none'), 'SC8': ('cstar', 'matched', 'apart')}


def _pick(arm, set_, enc, record_box):
    p = reused_pick(arm, set_, enc)
    if p is not None:
        return p
    if record_box[0] is None:
        record_box[0] = ns_paths.read_committed_record('stage1_picks.json')
    return new_pick(record_box[0], arm, set_, enc)


def score_cells(set, enc, scs, picks, draws, workers=8):
    """The cells of `scs` (subset of SC6, SC7, SC8) for encoder `enc` on `set` ('artelingo' held, 'mmart' test), aspect
    by aspect, `draws` bootstrap draws each. SC6 (CLIP L/14 only): c* against CRL; SC7: untold c* against the plain image;
    SC8: told c* against matched PercepT, seeds drawn apart. Reused picks come from their rounds' records, new ones
    from the committed `stage1_picks.json` (`picks` None); a given `picks` dict (that record's form) is for smoke runs
    and tests only. r: X-c*'s recorded told validation mAP range on `set` (D17). A cell has the index's keys plus
    `candidate_readout`, `comparator_readout`, `n_queries`; its `id` is the claim cell's."""
    import ns_outcome
    assert set in ns_paths.PAINTING_SETS and enc in ns_paths.ENCODERS, (set, enc)
    scs = tuple(scs)
    assert scs and frozenset(scs) <= frozenset(RETRIEVAL_SCS) and len(frozenset(scs)) == len(scs), scs
    assert 'SC6' not in scs or enc == ns_paths.CRL_ENCODER, 'SC6 is c* on CLIP L/14 against CRL on CLIP L/14'
    if picks is not None:
        assert ns_paths.SMOKE, 'picks are read from the committed record; passing them is for smoke runs and tests'
    box = [picks]
    claim = {(c['sc'], c['set'], c['encoder'], c['aspect']): c for c in ns_outcome.claim_cells() if c['task'] == 'ret'}
    rows = _Rows(set, enc)
    cells = []
    for aspect in ASPECTS:
        y = data.labels(set, ns_paths.HELD_SPLIT[set], aspect)
        r, r_source = recorded_range(set, enc, aspect)
        done = {}
        for sc in scs:
            cand_arm, comp_arm, mode = SC_ARMS[sc]
            res, picked = [], []
            for arm in (cand_arm, comp_arm):
                if (arm, aspect) not in done:
                    pk = _pick(arm, set, enc, box)
                    done[(arm, aspect)] = (rows.ap(arm, aspect, pk, y, workers), pk)
                res.append(done[(arm, aspect)][0])
                picked.append(done[(arm, aspect)][1])
            cand, comp = res
            assert np.array_equal(cand['rows'], comp['rows']), (sc, set, enc, aspect, 'the arms were scored on other queries')
            p = paired_ret(cand['ap'], comp['ap'], r, draws, comp_seeds=mode)
            cc = claim[(sc, set, enc, aspect)]
            cells.append(dict(
                id=cc['id'], sc=sc, set=set, encoder=enc, aspect=aspect, task='ret', metric='map', k=None,
                candidate=cc['candidate'], comparator=cc['comparator'], cand=p['cand'], comp=p['comp'], d=p['d'],
                lo=p['lo'], hi=p['hi'], r=p['r'], r_source=r_source, n_draws=p['n_draws'], n_rows=int(len(y)),
                candidate_readout=picked[0], comparator_readout=picked[1], n_queries=int(len(cand['rows']))))
    return cells


# ================================================================================================ gate (ii)
TOL = ns_paths.TOL_REPLAY
PERQUERY = R11_OUT / 'stage5/perquery'


def _item(name, ours, theirs):
    return dict(name=name, ours=float(ours), theirs=float(theirs), abs_diff=abs(float(ours) - float(theirs)))


def _result(items, complete=True, **extra):
    worst = max(i['abs_diff'] for i in items)
    return {'pass': bool(complete and worst <= TOL), 'max_abs_diff': float(worst), 'tol': TOL, 'n_values': len(items),
            'complete': bool(complete), 'items': items, **extra}


def _stored_ap(name):
    with np.load(PERQUERY / name, allow_pickle=False) as z:
        return z['ap'], z['rows']


def _cell_items(prefix, mine, stored, with_interval):
    """mine: paired_ret dict + n_queries; stored: a cell of an older round (candidate_value, ...)."""
    items = [_item(f'{prefix}/candidate_value', mine['cand'], stored['candidate_value']),
             _item(f'{prefix}/comparator_value', mine['comp'], stored['comparator_value']),
             _item(f'{prefix}/n_queries', mine['n_queries'], stored['n_queries'])]
    if with_interval:
        items += [_item(f'{prefix}/{k}', mine[k], stored[k]) for k in ('d', 'lo', 'hi')]
    return items


def _paired_cell(cand, comp, r, draws):
    assert np.array_equal(cand['rows'], comp['rows'])
    p = paired_ret(cand['ap'], comp['ap'], r, draws, comp_seeds='none')
    p['n_queries'] = len(cand['rows'])
    return p


def _gate_cstar_vs_plain(set_, round_cells, prefix, draws, workers):
    """SC2 of an older round for CLIP L/14: c* against the plain image, each at its recorded pick, scored through this
    module's arm_ap, paired_ret and the data loaders."""
    enc = 'clip_l14'
    rows = _Rows(set_, enc)
    items = []
    for a in ASPECTS:
        y = data.labels(set_, ns_paths.HELD_SPLIT[set_], a)
        cand = rows.ap('cstar', a, reused_pick('cstar', set_, enc), y, workers)
        comp = rows.ap('plain', a, reused_pick('plain', set_, enc), y, workers)
        mine = _paired_cell(cand, comp, None, draws)
        (st,) = [c for c in round_cells if c['aspect'] == a]
        items += _cell_items(f'{prefix}/{a}', mine, st, draws == 10000)
        assert st['candidate_readout'] == reused_pick('cstar', set_, enc)
    return items


def gate_ii_scorer(draws=10000, workers=8, parts=('r15', 'r14', 'r11')):
    """(ii) the scorer and the bootstrap: r15's CLIP L/14 SC2 values and intervals (MMArt test; committed
    `R15/records/verdicts.json`), r14's CLIP L/14 SC2 c* mAP (ArtELingo held; `R14/records/verdicts.json`, with the plain
    image and the interval), r11's stage 5 SC10 to SC15 values and intervals (c* against the plain and the instructed
    image; `stage5/verdict.json`) with the c*, plain and instructed per-query AP. Fewer draws skip the intervals and the
    gate cannot pass."""
    items = []
    if 'r15' in parts:
        cells = [c for c in _json(ns_paths.ROUNDS['r15'] / 'records/verdicts.json')['cells']
                 if c['task'] == 'ret' and c['sc'] == 'SC2' and c['encoder'] == 'clip_l14' and c['data'] == 'mmart_test']
        assert len(cells) == 3
        items += _gate_cstar_vs_plain('mmart', cells, 'r15/clip_l14/SC2', draws, workers)
    if 'r14' in parts:
        cells = [c for c in _json(ns_paths.ROUNDS['r14'] / 'records/verdicts.json')['cells']
                 if c['task'] == 'ret' and c['sc'] == 'SC2' and c['encoder'] == 'clip_l14']
        assert len(cells) == 3
        items += _gate_cstar_vs_plain('artelingo', cells, 'r14/clip_l14/SC2', draws, workers)
    if 'r11' in parts:
        v = _json(R11_OUT / 'stage5/verdict.json')
        assert v['draws'] == 10000
        picks = _r11_picks()
        want = {a: {'r': recorded_range('artelingo', 'qwen', a)[0]} for a in ASPECTS}   # the cells' own read
        rows = _Rows('artelingo', 'qwen')
        by = {}

        def teacher_rows(a, split):
            return lambda s: data.teacher('artelingo', split, a)

        for a in ASPECTS:
            y = data.labels('artelingo', 'held', a)
            res = {'cstar': rows.ap('cstar', a, picks['cstar'], y, workers),
                   'plain': rows.ap('plain', a, picks['plain'], y, workers),
                   'instructed': arm_ap(teacher_rows(a, 'val'), teacher_rows(a, 'held'), picks['instructed'], y, None, workers)}
            for s in SEEDS:
                items.append(_item(f'r11/cstar/{a}/mAP s{s}', res['cstar']['per_seed'][s], v['map']['cstar'][a]['per_seed'][s]))
                st, srows = _stored_ap(f'cstar__{a}__h{a}__s{s}.npz')
                assert np.array_equal(srows, res['cstar']['rows'])
                items.append(_item(f'r11/cstar/{a}/s{s}/per-query max abs', np.abs(res['cstar']['ap'][s] - st).max(), 0.0))
            items.append(_item(f'r11/cstar/{a}/mAP mean', res['cstar']['point'], v['map']['cstar'][a]['mean']))
            for arm in ('plain', 'instructed'):
                st, srows = _stored_ap(f'{arm}__{a}__s-.npz' if arm == 'plain' else f'{arm}__{a}__h{a}__s-.npz')
                assert np.array_equal(srows, res[arm]['rows'])
                items.append(_item(f'r11/{arm}/{a}/mAP', res[arm]['point'], v['map'][arm][a]['mean']))
                items.append(_item(f'r11/{arm}/{a}/per-query max abs', np.abs(res[arm]['ap'] - st).max(), 0.0))
            by[a] = res
        if draws == 10000:
            for sid, entry in v['sc'].items():
                (c,) = entry['comparisons']
                a = c['aspect']
                p = _paired_cell(by[a]['cstar'], by[a][c['comparator'][0]], want[a]['r'], draws)
                items += [_item(f'r11/{sid}/d', p['d'], c['d']), _item(f'r11/{sid}/lo', p['lo'], c['interval'][0]),
                          _item(f'r11/{sid}/hi', p['hi'], c['interval'][1]), _item(f'r11/{sid}/r', p['r'], c['r'])]
                assert p['outcome'] == c['outcome'], (sid, p['outcome'], c['outcome'])
    return _result(items, complete=bool(draws == 10000 and frozenset(parts) == {'r15', 'r14', 'r11'}), parts=list(parts), n_draws=draws)


def gate_ii_crl(workers=8):
    """(ii) CRL rebuild: `ns_data.crl` (held, 3 aspects) equals r11's `rt_data.embed('crl')` bit for bit, and at r11's pick
    its held mAP and per-query AP equal r11's stage 5 `verdict.json` map and `perquery/crl__<a>__h<a>__s-.npz`."""
    v = _json(R11_OUT / 'stage5/verdict.json')
    pick = _r11_picks()['crl']
    rows = _Rows('artelingo', 'clip_l14')
    items = []
    for a in ASPECTS:
        for split in ('held', 'val'):
            ours, theirs = data.crl('artelingo', split, a), rt_data.embed('crl', split, a)
            items.append(_item(f'crl/{a}/{split}/vector max abs', np.abs(ours.astype(np.float64) - theirs).max(), 0.0))
        y = data.labels('artelingo', 'held', a)
        res = rows.ap('crl', a, pick, y, workers)
        st, srows = _stored_ap(f'crl__{a}__h{a}__s-.npz')
        assert np.array_equal(srows, res['rows'])
        items.append(_item(f'crl/{a}/mAP', res['point'], v['map']['crl'][a]['mean']))
        items.append(_item(f'crl/{a}/per-query max abs', np.abs(res['ap'] - st).max(), 0.0))
    return _result(items, pick=pick)


def gate_ii_percept(workers=8):
    """(ii) PercepT per-query AP: Qwen matched PercepT on ArtELingo held at r11's pick, per seed: mAP against `verdict.json`
    (per seed and mean) and per-query AP against `perquery/percept__<a>__s<seed>.npz`. Held probabilities through
    `ns_data.percept_probs` divided by their float32 norm; validation rows are r11's stored ones."""
    v = _json(R11_OUT / 'stage5/verdict.json')
    pick = _r11_picks()['percept']
    rows = _Rows('artelingo', 'qwen')
    items = []
    for a in ASPECTS:
        y = data.labels('artelingo', 'held', a)
        res = rows.ap('matched', a, pick, y, workers)
        for s in SEEDS:
            st, srows = _stored_ap(f'percept__{a}__s{s}.npz')
            assert np.array_equal(srows, res['rows'])
            items.append(_item(f'percept/{a}/mAP s{s}', res['per_seed'][s], v['map']['percept'][a]['per_seed'][s]))
            items.append(_item(f'percept/{a}/s{s}/per-query max abs', np.abs(res['ap'][s] - st).max(), 0.0))
        items.append(_item(f'percept/{a}/mAP mean', res['point'], v['map']['percept'][a]['mean']))
    return _result(items, pick=pick)


def gate_ii(draws=10000, workers=8):
    """{'ii_scorer', 'ii_crl', 'ii_percept'}: the three status entries of gate (ii)."""
    return dict(ii_scorer=gate_ii_scorer(draws, workers), ii_crl=gate_ii_crl(workers), ii_percept=gate_ii_percept(workers))


# ================================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gate-ii', help='replay gate (ii) -> JSON')
    g.add_argument('--part', choices=['scorer', 'crl', 'percept'], default=None, help='default: all three')
    g.add_argument('--scorer-parts', nargs='+', default=['r15', 'r14', 'r11'], choices=['r15', 'r14', 'r11'])
    g.add_argument('--draws', type=int, default=10000)
    g.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    g.add_argument('--out', default=None, help='output JSON (default: OUT/stage0/gate_ii_<part>.json)')
    args = p.parse_args(argv)
    t0 = time.monotonic()
    parts = [args.part] if args.part else ['scorer', 'crl', 'percept']
    for part in parts:
        t1 = time.monotonic()
        res = {'scorer': lambda: gate_ii_scorer(args.draws, args.workers, tuple(args.scorer_parts)),
               'crl': lambda: gate_ii_crl(args.workers), 'percept': lambda: gate_ii_percept(args.workers)}[part]()
        res['seconds'] = time.monotonic() - t1
        out = Path(args.out) if args.out else ns_paths.OUT / f'stage0/gate_ii_{part}.json'
        ns_paths.write_json(out, res)
        print(f'gate (ii) {part}: pass={res["pass"]} max_abs_diff={res["max_abs_diff"]:.3e} n={res["n_values"]} '
              f'complete={res["complete"]} ({res["seconds"]:.0f}s) -> {out}')
    print(f'total {time.monotonic() - t0:.0f}s')


if __name__ == '__main__':
    main()
