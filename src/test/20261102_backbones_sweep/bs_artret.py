"""ArtELingo retrieval scores and stage 1 picks of r14 (ticket 06): SC2 and SC6 (retrieval) for X-c*, X-c*'s validation
mAP seed ranges and read-out picks, the frozen-image bars' picks (D19), part b's ablation row function, and the replays
0(vi) and 0(xii) (r11 parts). CPU only.

Spec: docs/superpowers/specs/2026-11-02-backbones-sweep-design.md, SC2, SC6, §3.4 ("ArtELingo retrieval", "Seed ranges for
LEVEL"), §3.5 0(vi), 0(xii), §6 "Selection rules", D19. Nothing is rewritten from r11: each step calls the function that
produced the replayed number.

  protocol   `rt_metrics.retrieval` (aspect head as query and gallery, relevance by the aspect's label, mAP, P@10),
             `rt_readout` (menu, cross-fit, guard, score, tie rule, `stage5_apply`), `rt_run.score_arm` (held scoring and
             per-query files), `rt_stats.paired` (10,000 joint query/seed draws, `default_rng(0)`), `rt_data` (readers).
  ranges     validation mAP at R0 (`rt_stage0.validation_seed`'s computation: whole validation set, no cross-fit), max
             minus min over seeds 0 to 4, per aspect, unrounded: `OUT/stage1/artret_ranges_<enc>.json`.
  picks      X-c*: r11's rule (stage 4: cross-fit, guard, score, tie order) with X-c*'s own ranges.
             Frozen image of CLIP L/14 and SigLIP 2: r11's stored picks (`stage4/picks.json`) and stored held per-query
             outputs (`stage5/perquery`), as `reused` below says. SigLIP 1: r11's stage 4 rule with r11's r_a
             (`20261029_retrieval/stage0a_ranges.json`); never X-c*'s ranges (D19).
             Teacher: r11's stored instructed-image pick and per-query outputs.
             `OUT/stage1/artret_picks_<enc>.json`.
  cells      SC2: X-c* against X's frozen image, both at their picks, WIN-only (r null). SC6: X-c* against the
             teacher of the aspect, LEVEL or WIN, r = X-c*'s own validation range. `OUT/stage3/cells_artelingo_ret.json`.
  stand-in   encoder `qwen` is r10's c* (root `R10_OUT/full`) with r11's plain Qwen image as the frozen image: not an
             encoder of part a; its cells must give r11's stage 5 SC10 to SC15.

Held data are read only by `score_encoder` and the replays; `validation_ranges`, `pick_cstar` and `pick_frozen` read
validation only. CLI: `python bs_artret.py -h` (ranges | pick | score | replay GATE). Outputs under `bs_paths.OUT`
(BS_OUTPUTS for a smoke run).
"""
import argparse
import contextlib
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

import bs_paths

for _name in ('20261027_recipe_sweep', '20261029_retrieval'):       # this checkout's copies of the inherited code
    if str(bs_paths.TEST / _name) not in sys.path:
        sys.path.insert(0, str(bs_paths.TEST / _name))
import rs_paths  # noqa: E402,F401  (puts r10's and the older rounds' folders on sys.path)
import rt_data  # noqa: E402
import rt_metrics  # noqa: E402
import rt_readout  # noqa: E402
import rt_run  # noqa: E402
import rt_stats  # noqa: E402
from rt_readout import R0, ReadOut  # noqa: E402

SEEDS = tuple(bs_paths.SEEDS)
ASPECTS = tuple(bs_paths.ASPECTS)
N_DRAWS = 10000                                                     # r11's joint query/seed draws
TOL = 1e-9
CELLS_NAME = 'cells_artelingo_ret.json'
CELL_KEYS = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator', 'candidate_value',
             'comparator_value', 'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'seeds')
R_SOURCE = 'X-c* own validation seed range (retrieval mAP at R0, seeds 0 to 4)'
ABLATION_READOUT = ReadOut('center', None, 10)                      # c*'s r11 pick (centering + k = 10)
R11_RANGES_FILE = bs_paths.TEST / '20261029_retrieval/stage0a_ranges.json'
R10_FULL = bs_paths.R10_OUT / 'full'

# Per encoder: its default-image view in `bs_stores`, and the r11 arm whose stored pick and held per-query outputs are
# the frozen-image bar (None = no r11 arm: the pick is computed with r11's rule and r_a). `qwen` is the stand-in.
ENC = {
    'siglip2': dict(view='siglip2', r11_arm='siglip2'),
    'clip_l14': dict(view='clip_l14', r11_arm='clip'),
    'siglip1': dict(view='siglip1', r11_arm=None),
    'qwen': dict(view='qwen_default', r11_arm='plain'),
}
GRAPHS = rt_data.GRAPHS


# ============================================================================================ ranges and picks (pure)
def own_ranges(per_seed):
    """X-c*'s seed range per aspect: max - min over seeds 0 to 4 of the validation mAP at R0, unrounded.
    per_seed: {aspect: {seed: mAP}}."""
    assert set(per_seed) == set(ASPECTS), sorted(per_seed)
    out = {}
    for a in ASPECTS:
        assert sorted(per_seed[a]) == list(SEEDS), f'{a}: seeds {sorted(per_seed[a])}, expected {list(SEEDS)}'
        out[a] = rt_stats.seed_range([per_seed[a][s] for s in SEEDS])
        assert np.isfinite(out[a]) and out[a] > 0, (a, out[a])
    return out


def r11_ranges():
    """r11's r_a: the validation R0 mAP seed ranges of r10's c*, from the committed `stage0a_ranges.json`."""
    doc = json.loads(R11_RANGES_FILE.read_text())
    r = {a: float(doc['map'][a]['r']) for a in ASPECTS}
    assert all(np.isfinite(v) and v > 0 for v in r.values()), r
    return r


def _table(rows, ranges):
    sc, keep = rt_readout.score(rows, ranges), rt_readout.guard(rows, ranges)
    return {rt_readout.name(ro): dict(map={a: float(m[a]) for a in ASPECTS}, score=float(sc[ro]), kept=bool(keep[ro]))
            for ro, m in rows.items()}


def cstar_pick(rows, ranges):
    """Stage 1 pick of X-c*: r11's selection rule (guard, score, .001 tie, tie order) with X-c*'s OWN ranges.
    rows: {ReadOut: {aspect: cross-fit mAP, seeds averaged}}. Returns {'pick': name, 'table': {...}, 'ranges': {...}}."""
    ranges = {a: float(ranges[a]) for a in ASPECTS}
    return dict(pick=rt_readout.name(rt_readout.select(rows, ranges)), table=_table(rows, ranges), ranges=ranges)


def frozen_pick(rows):
    """Pick of a frozen-image bar without a stored r11 pick (SigLIP 1; D19): r11's stage 4 rule with r11's r_a. There is
    no ranges argument: the encoder's own ranges must never reach it."""
    ranges = r11_ranges()
    return dict(pick=rt_readout.name(rt_readout.select(rows, ranges)), table=_table(rows, ranges), ranges=ranges)


# ============================================================================================ cells (pure)
def make_cell(sc, encoder, aspect, candidate, comparator, candidate_value, comparator_value, d, lo, hi, r, r_source,
              n_draws, seeds=SEEDS, data='artelingo_held', **extra):
    """One cell of `cells_artelingo_ret` (index format). `r` is the unrounded LEVEL unit, None for a WIN-only bar."""
    assert sc in ('SC2', 'SC6') and aspect in ASPECTS
    values = [candidate_value, comparator_value, d, lo, hi]
    assert all(np.isfinite(v) for v in values), (sc, encoder, aspect, values)
    assert (r is None) == (sc == 'SC2'), 'only SC6 cells carry a LEVEL unit'
    assert r is None or (np.isfinite(r) and r > 0)
    cell = dict(sc=sc, encoder=encoder, aspect=aspect, task='ret', data=data, candidate=candidate,
                comparator=comparator, candidate_value=float(candidate_value), comparator_value=float(comparator_value),
                d=float(d), lo=float(lo), hi=float(hi), r=None if r is None else float(r), r_source=r_source,
                n_draws=int(n_draws), seeds=list(seeds), **extra)
    assert set(CELL_KEYS) <= set(cell)
    return cell


def _side(x, name):
    assert set(x) >= {'ap', 'rows'}, (name, sorted(x))
    return np.asarray(x['ap'], dtype=np.float64), np.asarray(x['rows'])


def aspect_cells(enc, aspect, cand, frozen, teacher, ranges, n_draws, readouts):
    """The two cells of one aspect. cand/frozen/teacher: dict(ap, rows) of the held per-query AP (cand float64[5, nq],
    the others float64[nq]); the three must have the same query rows. ranges: X-c*'s own validation ranges (SC6's r).
    readouts: dict(cstar, frozen, teacher) read-out names, kept beside the values.
    SC2 and SC6 intervals: `rt_stats.paired(cand, comparator, n_draws)` (r11's 10,000-draw joint query/seed bootstrap)."""
    (ac, rc), (af, rf), (at, rt) = _side(cand, 'cand'), _side(frozen, 'frozen'), _side(teacher, 'teacher')
    assert ac.ndim == 2 and ac.shape[0] == 5 and af.ndim == at.ndim == 1
    assert np.array_equal(rc, rf) and np.array_equal(rc, rt), (enc, aspect, 'query rows differ')
    out = []
    for sc, comp, comp_name, comp_ro in (('SC2', af, f'frozen_{enc}', readouts['frozen']),
                                          ('SC6', at, f'teacher_{aspect}', readouts['teacher'])):
        p = rt_stats.paired(ac, comp, n_draws)
        out.append(make_cell(sc, enc, aspect, f'cstar_{enc}', comp_name, ac.mean(), comp.mean(), p['d'],
                             p['interval'][0], p['interval'][1], None if sc == 'SC2' else float(ranges[aspect]),
                             None if sc == 'SC2' else R_SOURCE, n_draws, candidate_readout=readouts['cstar'],
                             comparator_readout=comp_ro, n_queries=int(len(rc))))
    return out


def write_cells(path, cells, encoders):
    """Merge `cells` into the list file: cells of `encoders` are replaced, other encoders' cells kept (atomic)."""
    path = Path(path)
    old = json.loads(path.read_text()) if path.exists() else []
    assert isinstance(old, list)
    kept = [c for c in old if c['encoder'] not in set(encoders)]
    for c in cells:
        assert set(CELL_KEYS) <= set(c) and c['encoder'] in set(encoders)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(kept + list(cells), indent=1, allow_nan=False) + '\n')
    tmp.replace(path)
    return kept + list(cells)


# ============================================================================================ part b's ablation rows
def ablation_retrieval(x_val, x_held, y_val, y_held, aspect, workers=8):
    """Part b's ablation row for one run, aspect and seed (§3.4; ticket 11 calls it): R0 and the read-out of c*'s r11 pick
    (centering + k = 10) fitted on the run's OWN validation export `x_val`, applied to the held export `x_held`.
    x_*: float32 unit rows [n, d] in r11's sorted painting order; y_*: {aspect: int labels, < 0 unlabelled}.
    Returns {'readout': 'center+k10',
             'held': {'r0': retrieval dict, 'center_k10': retrieval dict}   (ap, p10, rows, dropped: r11's `retrieval`),
             'val': {'r0_map': whole-validation R0 mAP, 'center_k10_crossfit_map': r11's cross-fit mAP}}."""
    assert aspect in ASPECTS and x_val.dtype == x_held.dtype == np.float32
    held_r0 = rt_metrics.retrieval(x_held, y_held[aspect], workers=workers)
    xa = rt_readout.stage5_apply(ABLATION_READOUT, x_val, x_held)
    held_c = rt_metrics.retrieval(xa, y_held[aspect], qe_k=ABLATION_READOUT.k, workers=workers)
    assert np.array_equal(held_r0['rows'], held_c['rows'])
    val_r0 = float(rt_metrics.retrieval(x_val, y_val[aspect], workers=workers)['ap'].mean())
    cf = rt_readout.crossfit_all({(aspect, None): x_val}, {aspect: y_val[aspect]}, [ABLATION_READOUT],
                                 workers=workers)[ABLATION_READOUT][aspect]
    return dict(readout=rt_readout.name(ABLATION_READOUT), held=dict(r0=held_r0, center_k10=held_c),
                val=dict(r0_map=val_r0, center_k10_crossfit_map=float(cf)))


# ============================================================================================ readers
@contextlib.contextmanager
def cstar_root(root):
    """r11's reader with only the root changed (`rt_data.CSTAR_ROOT`; index, "X-c* runs"), restored afterwards."""
    old = rt_data.CSTAR_ROOT
    rt_data.CSTAR_ROOT = Path(root)
    try:
        yield
    finally:
        rt_data.CSTAR_ROOT = old


def cstar_embed(root, split, aspect, seed):
    """X-c*'s unit float32 rows, r11's `rt_data.embed('cstar', ...)` at `root`: held `held/k11=40/<aspect>__s<seed>.npz`,
    validation the training-time file `train/k11=40/R6-5/<aspect>__<graph>__s<seed>.npz` (not `val/<arm>/`)."""
    with cstar_root(root):
        return rt_data.embed('cstar', split, aspect, seed)


def frozen_embed(view, split):
    """X's default ArtELingo image on r11's sorted `split` painting names, from the r14 store reader, float32 unit rows
    (the normalization of `rt_data.embed`). bs_stores keys ArtELingo rows by `image`; names map through the store's index."""
    import bs_stores
    (folder,) = bs_stores.folders('artelingo', view)
    index = json.loads((Path(folder) / 'index.json').read_text())['images']
    to_image = {m['painting']: m['image'] for m in index}
    assert len(to_image) == len(index), f'{folder}: duplicate painting names'
    names = rt_data.names(split)
    missing = [n for n in names if n not in to_image]
    assert not missing, f'{folder}: {len(missing)} painting names not in the index, e.g. {missing[:2]}'
    x = bs_stores.load('artelingo', view, [to_image[n] for n in names]).astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    assert x.dtype == np.float32 and x.shape[0] == len(names) and np.isfinite(x).all()
    return x


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def r11_stored(arm, aspect):
    """r11's stored stage 5 held per-query AP of a frozen or instructed arm (`stage5/perquery`) with its recorded mAP, and
    its stage 4 pick. The stored mean must equal r11's verdict value (asserted)."""
    head = '' if arm in rt_data.ASPECT_FREE else f'__h{aspect}'
    z = np.load(bs_paths.R11_OUT / f'stage5/perquery/{arm}__{aspect}{head}__s-.npz')
    ap, rows = z['ap'].astype(np.float64), z['rows']
    verdict = read_json(bs_paths.R11_OUT / 'stage5/verdict.json')
    assert ap.ndim == 1 and float(ap.mean()) == verdict['map'][arm][aspect]['mean'], (arm, aspect, 'stored per-query')
    pick = read_json(bs_paths.R11_OUT / 'stage4/picks.json')[arm]
    assert verdict['readouts'][arm] == pick, (arm, 'verdict and picks disagree')
    return dict(ap=ap, rows=rows, pick=pick)


# ============================================================================================ stage 1
def validation_ranges(root, seeds=SEEDS):
    """X-c*'s validation mAP at R0 per aspect and seed (whole validation set, r11's stage 0a computation) and the seed
    ranges. Validation only. Returns {'ranges', 'per_seed', 'sources', ...}."""
    labels = rt_data.labels('val')
    per_seed, sources = {}, {}
    for a in ASPECTS:
        per_seed[a], sources[a] = {}, {}
        for s in seeds:
            x = cstar_embed(root, 'val', a, s)
            per_seed[a][s] = float(rt_metrics.retrieval(x, labels[a])['ap'].mean())
            f = Path(root) / f'train/k11=40/R6-5/{a}__{GRAPHS[a]}__s{s}.npz'
            sources[a][str(s)] = dict(path=str(f), sha256=sha_file(f))
    return dict(ranges=own_ranges(per_seed), per_seed=per_seed, sources=sources, n_val=len(next(iter(labels.values()))),
                labelled={a: int((labels[a] >= 0).sum()) for a in ASPECTS})


def write_ranges(path, enc, result):
    write_json(path, dict(encoder=enc, kind='artelingo_retrieval_validation_ranges', **result))


def read_ranges(path, enc):
    rec = read_json(path)
    assert rec['encoder'] == enc and set(rec['ranges']) == set(ASPECTS), path
    assert all(np.isfinite(v) and v > 0 for v in rec['ranges'].values()), path
    return rec['ranges']


def cstar_rows(root, workers=8):
    """Cross-fit mAP of every read-out of X-c*'s menu (validation, seeds averaged), r11's stage 4 call."""
    labels = rt_data.labels('val')
    xs = {(a, s): cstar_embed(root, 'val', a, s) for a in ASPECTS for s in SEEDS}
    widths = {x.shape[1] for x in xs.values()}
    assert widths == {rt_data.WIDTHS['cstar']}, widths
    return rt_readout.crossfit_all(xs, labels, rt_readout.menu(widths.pop()), workers=workers), labels


def frozen_rows(view, workers=8):
    """The same for a frozen image (one matrix, no seeds)."""
    labels = rt_data.labels('val')
    x = frozen_embed(view, 'val')
    xs = {(a, None): x for a in ASPECTS}                    # one matrix: its eigendecomposition is shared
    return rt_readout.crossfit_all(xs, labels, rt_readout.menu(x.shape[1]), workers=workers)


def pick_cstar(root, ranges, workers=8):
    rows, labels = cstar_rows(root, workers)
    return dict(cstar_pick(rows, ranges), dropped=rt_readout.half_dropped(labels))


def pick_frozen(view, workers=8):
    return frozen_pick(frozen_rows(view, workers))


def stage1_picks(enc, root, ranges, workers=8):
    """The record of one encoder's picks: X-c* (own ranges), the frozen image (r11's stored pick, or computed with r11's
    r_a for SigLIP 1) and the teacher (r11's stored pick). Validation only."""
    spec = ENC[enc]
    rec = dict(encoder=enc, kind='artelingo_retrieval_picks', cstar=pick_cstar(root, ranges, workers))
    if spec['r11_arm']:
        arm = spec['r11_arm']
        rec['frozen'] = dict(pick=read_json(bs_paths.R11_OUT / 'stage4/picks.json')[arm], source=f'r11 stored pick, arm {arm}')
    else:
        rec['frozen'] = dict(pick_frozen(spec['view'], workers), source="computed: r11's stage 4 rule with r11's r_a")
    rec['teacher'] = dict(pick=read_json(bs_paths.R11_OUT / 'stage4/picks.json')['instructed'],
                          source='r11 stored pick, arm instructed')
    rec['r11_ranges_sha256'] = sha_file(R11_RANGES_FILE)
    return rec


def readout_of(name, width):
    by_name = {rt_readout.name(ro): ro for ro in rt_readout.menu(width)}
    assert name in by_name, (name, 'not in the menu of width', width)
    return by_name[name]


# ============================================================================================ stage 3 scoring
def _args(outputs, workers):
    return types.SimpleNamespace(outputs=Path(outputs), smoke=None, workers=workers)


def score_cstar_held(root, pick, outputs, workers=8, stage=5):
    """X-c*'s held per-query AP at its picked read-out, seeds 0 to 4, through `rt_run.score_arm` (fit on the seed's own
    validation export, applied to its held export). Returns {aspect: dict(ap [5, nq], p10, rows, dropped)}."""
    ro = readout_of(pick, rt_data.WIDTHS['cstar'])
    labels = rt_data.labels('held')

    def get_x(head, seed):
        return rt_readout.stage5_apply(ro, cstar_embed(root, 'val', head, seed), cstar_embed(root, 'held', head, seed))
    cells = [(a, a) for a in ASPECTS]
    res = rt_run.score_arm(_args(outputs, workers), stage, 'cstar', cells, get_x, labels, qe_k=ro.k)
    return {a: res[(a, a)] for a in ASPECTS}


def score_frozen_held(view, arm, pick, outputs, workers=8, stage=5):
    """A frozen image's held per-query AP at `pick` (fit on all validation rows, applied to held), via `score_arm`."""
    x_val, x_held = frozen_embed(view, 'val'), frozen_embed(view, 'held')
    ro = readout_of(pick, x_val.shape[1])
    xh = rt_readout.stage5_apply(ro, x_val, x_held)
    res = rt_run.score_arm(_args(outputs, workers), stage, arm, [(None, a) for a in ASPECTS], lambda h, s: xh,
                           rt_data.labels('held'), qe_k=ro.k)
    return {a: res[(None, a)] for a in ASPECTS}


def score_encoder(enc, root, ranges, picks, outputs, n_draws=N_DRAWS, workers=8):
    """The six cells of an encoder. `ranges` and `picks` are the stage 1 records (written before this runs)."""
    spec = ENC[enc]
    cand = score_cstar_held(root, picks['cstar']['pick'], Path(outputs) / enc / 'cstar', workers)
    if spec['r11_arm']:
        frozen = {a: r11_stored(spec['r11_arm'], a) for a in ASPECTS}
        assert all(frozen[a]['pick'] == picks['frozen']['pick'] for a in ASPECTS)
    else:
        frozen = score_frozen_held(spec['view'], f'frozen_{enc}', picks['frozen']['pick'], Path(outputs) / enc / 'frozen',
                                   workers)
    teacher = {a: r11_stored('instructed', a) for a in ASPECTS}
    readouts = dict(cstar=picks['cstar']['pick'], frozen=picks['frozen']['pick'], teacher=picks['teacher']['pick'])
    assert readouts['teacher'] == teacher['emotion']['pick']
    cells = []
    for a in ASPECTS:
        cells += aspect_cells(enc, a, cand[a], frozen[a], teacher[a], ranges, n_draws, readouts)
    return cells


# ============================================================================================ replays
def _item(name, ours, theirs):
    return dict(name=name, ours=ours, theirs=theirs, abs_diff=abs(float(ours) - float(theirs)))


def _result(items):
    worst = max(i['abs_diff'] for i in items)
    return dict(passed=bool(worst <= TOL), max_abs_diff=float(worst), tol=TOL, n_values=len(items), items=items)


def replay_0vi(n_draws=N_DRAWS, workers=8, root=R10_FULL, outputs=None):
    """0(vi), r11's stage 5: c* and teacher held mAP (c* per seed and mean, at r11's stored pick, recomputed through this
    module's scoring path; the teacher from r11's stored per-query outputs), and the intervals of SC10 to SC15 (c* against
    the plain image and against the instructed image: d, lo, hi from `rt_stats.paired` at n_draws)."""
    outputs = Path(outputs or bs_paths.OUT / 'replays/vi_perquery')
    v = read_json(bs_paths.R11_OUT / 'stage5/verdict.json')
    assert n_draws == v['draws'], 'the interval replay needs r11\'s draws'
    pick = read_json(bs_paths.R11_OUT / 'stage4/picks.json')['cstar']
    cand = score_cstar_held(root, pick, outputs, workers)
    items = []
    for a in ASPECTS:
        per = [float(cand[a]['ap'][s].mean()) for s in range(5)]
        for s in range(5):
            items.append(_item(f'cstar/{a}/mAP s{s}', per[s], v['map']['cstar'][a]['per_seed'][s]))
        items.append(_item(f'cstar/{a}/mAP mean', float(np.mean(per)), v['map']['cstar'][a]['mean']))
        for arm in ('instructed', 'plain'):
            st = r11_stored(arm, a)
            items.append(_item(f'{arm}/{a}/mAP', float(st['ap'].mean()), v['map'][arm][a]['mean']))
    for sid, entry in v['sc'].items():
        (c,) = entry['comparisons']
        a, arm = c['aspect'], c['comparator'][0]
        st = r11_stored(arm, a)
        assert np.array_equal(st['rows'], cand[a]['rows'])
        p = rt_stats.paired(cand[a]['ap'], st['ap'], n_draws)
        items += [_item(f'{sid}/d', p['d'], c['d']), _item(f'{sid}/lo', p['interval'][0], c['interval'][0]),
                  _item(f'{sid}/hi', p['interval'][1], c['interval'][1])]
    return dict(_result(items), pick_used=pick)


def replay_0vi_ranges(root=R10_FULL, workers=8):
    """The stand-in's validation ranges against r11's stage 0a (c*'s r_a): same data, same computation."""
    got = validation_ranges(root)
    want = r11_ranges()
    return _result([_item(f'range/{a}', got['ranges'][a], want[a]) for a in ASPECTS])


def replay_0vi_pick(root=R10_FULL, workers=8, from_file=None):
    """c*'s pick: the stage 1 procedure with the stand-in's own ranges (which equal r11's r_a on r10's c*) against r11's
    `stage4/scores.json` (every read-out's cross-fit mAP and score, guard flags) and `picks.json`. `from_file`: a picks
    record already written by `pick` (not recomputed)."""
    ranges = replay_ranges_for_standin(root) if from_file is None else read_json(from_file)['cstar']['ranges']
    rec = pick_cstar(root, ranges, workers) if from_file is None else read_json(from_file)['cstar']
    return _compare_pick(rec, read_json(bs_paths.R11_OUT / 'stage4/scores.json')['cstar'],
                         read_json(bs_paths.R11_OUT / 'stage4/picks.json')['cstar'])


def replay_ranges_for_standin(root):
    return validation_ranges(root)['ranges']


def _compare_pick(rec, stored, stored_pick):
    items = []
    assert set(rec['table']) == set(stored), (sorted(rec['table']), sorted(stored))
    for name, t in rec['table'].items():
        for a in ASPECTS:
            items.append(_item(f'{name}/{a}/crossfit mAP', t['map'][a], stored[name]['map'][a]))
        items.append(_item(f'{name}/score', t['score'], stored[name]['score']))
        assert t['kept'] == stored[name]['kept'], (name, 'guard flag')
    assert rec['pick'] == stored_pick, ('pick', rec['pick'], stored_pick)
    return dict(_result(items), pick=rec['pick'])


def replay_0xii(workers=8):
    """0(xii), r11's stage 5 CLIP L/14 and SigLIP 2 frozen-image mAPs: recomputed from the r14 store reader (validation
    fit at r11's stored pick, held scored through this module's path) and read from r11's stored per-query outputs, both
    against r11's verdict (the stored values must agree with the verdict; the recomputed ones within 1e-9)."""
    v = read_json(bs_paths.R11_OUT / 'stage5/verdict.json')
    items = []
    outputs = bs_paths.OUT / 'replays/xii_perquery'
    for enc in ('clip_l14', 'siglip2'):
        arm = ENC[enc]['r11_arm']
        pick = read_json(bs_paths.R11_OUT / 'stage4/picks.json')[arm]
        res = score_frozen_held(ENC[enc]['view'], f'frozen_{enc}', pick, outputs, workers)
        for a in ASPECTS:
            st = r11_stored(arm, a)
            assert np.array_equal(st['rows'], res[a]['rows'])
            items.append(_item(f'{enc}/{a}/mAP recomputed', float(res[a]['ap'].mean()), v['map'][arm][a]['mean']))
            items.append(_item(f'{enc}/{a}/mAP stored', float(st['ap'].mean()), v['map'][arm][a]['mean']))
            items.append(_item(f'{enc}/{a}/per-query max abs', float(np.abs(res[a]['ap'] - st['ap']).max()), 0.0))
    return _result(items)


def replay_0xii_pick(workers=8):
    """The frozen-image pick procedure (what SigLIP 1 uses) run on CLIP L/14 and SigLIP 2 with r11's r_a: the cross-fit
    table and the pick against r11's stage 4 `scores.json` and `picks.json` (arms clip, siglip2)."""
    scores = read_json(bs_paths.R11_OUT / 'stage4/scores.json')
    picks = read_json(bs_paths.R11_OUT / 'stage4/picks.json')
    items, picked = [], {}
    for enc in ('clip_l14', 'siglip2'):
        arm = ENC[enc]['r11_arm']
        rec = pick_frozen(ENC[enc]['view'], workers)
        r = _compare_pick(rec, scores[arm], picks[arm])
        items += [dict(i, name=f'{enc}/{i["name"]}') for i in r['items']]
        picked[enc] = rec['pick']
    return dict(_result(items), picks=picked)


GATES = {'vi': replay_0vi, 'vi-ranges': replay_0vi_ranges, 'vi-pick': replay_0vi_pick, 'xii': replay_0xii,
         'xii-pick': replay_0xii_pick}


# ============================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--n-draws', type=int, default=N_DRAWS)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name, help_ in (('ranges', 'X-c* validation mAP seed ranges -> OUT/stage1/artret_ranges_<enc>.json'),
                        ('pick', 'stage 1 picks (X-c*, frozen image, teacher) -> OUT/stage1/artret_picks_<enc>.json'),
                        ('score', 'SC2 and SC6 cells -> OUT/stage3/cells_artelingo_ret.json')):
        s = sub.add_parser(name, help=help_)
        s.add_argument('--enc', required=True, choices=sorted(ENC))
        s.add_argument('--root', type=Path, help='X-c* run root; default OUT/cstar/<enc>, the qwen stand-in R10_OUT/full')
    rp = sub.add_parser('replay', help='0(vi) and 0(xii) (r11 parts); see GATES')
    rp.add_argument('gate', choices=sorted(GATES))
    rp.add_argument('--from-file', type=Path, help='vi-pick: compare a picks record written by `pick`')
    args = p.parse_args(argv)
    t0 = time.monotonic()
    out = bs_paths.OUT
    assert not str(out).startswith(str(bs_paths.R10_OUT)), 'never write into an older round'
    if args.cmd in ('ranges', 'pick', 'score'):
        root = args.root or (R10_FULL if args.enc == 'qwen' else out / 'cstar' / args.enc)
        rpath = out / f'stage1/artret_ranges_{args.enc}.json'
        ppath = out / f'stage1/artret_picks_{args.enc}.json'
        if args.cmd == 'ranges':
            write_ranges(rpath, args.enc, validation_ranges(root))
            print(f'{args.enc}: ranges {read_ranges(rpath, args.enc)} -> {rpath} ({time.monotonic() - t0:.0f}s)')
            return 0
        ranges = read_ranges(rpath, args.enc)           # stage 1 first: no held number before ranges and picks exist
        if args.cmd == 'pick':
            rec = stage1_picks(args.enc, root, ranges, args.workers)
            write_json(ppath, rec)
            print(f'{args.enc}: picks cstar {rec["cstar"]["pick"]}, frozen {rec["frozen"]["pick"]}, '
                  f'teacher {rec["teacher"]["pick"]} -> {ppath} ({time.monotonic() - t0:.0f}s)')
            return 0
        picks = read_json(ppath)
        assert picks['encoder'] == args.enc
        cells = score_encoder(args.enc, root, ranges, picks, out / 'stage3/artret', args.n_draws, args.workers)
        write_cells(out / 'stage3' / CELLS_NAME, cells, [args.enc])
        print(f'{args.enc}: {len(cells)} cells -> {out / "stage3" / CELLS_NAME} ({time.monotonic() - t0:.0f}s)')
        return 0
    kw = dict(workers=args.workers)
    if args.gate == 'vi':
        kw['n_draws'] = args.n_draws
    if args.gate == 'vi-pick':
        kw['from_file'] = args.from_file
    res = GATES[args.gate](**kw)
    path = out / f'replays/replay_{args.gate}.json'
    write_json(path, dict(gate=args.gate, **res))
    print(f'replay {args.gate}: {"PASS" if res["passed"] else "FAIL"} max_abs_diff {res["max_abs_diff"]:.3g} '
          f'({time.monotonic() - t0:.0f}s) {path}')
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
