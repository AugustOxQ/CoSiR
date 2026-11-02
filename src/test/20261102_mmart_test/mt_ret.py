"""Retrieval on MMArt (r15 ticket 05): stage 1 seed ranges and read-out picks, the SC2 and SC4 (mAP) scorer, and the
replays 0(iii) (retrieval part) and 0(vi). CPU only.

Spec: docs/superpowers/specs/2026-11-02-mmart-test-design.md, §3.3 (populations), §3.4 (seed ranges, read-out picks),
§3.5 0(iii) and 0(vi), §6 SC2 and SC4, D12, D16. It inherits r14 part a (`bs_artret`, `bs_emoset_score`), r12
(`ac_select`, `ac_stats`) and r11 (`rt_readout`, `rt_metrics`, `rt_stats`). Nothing is rewritten from them; each statistic
calls the function that produced the replayed number (the replays prove it):

  protocol   `rt_metrics.retrieval` (the aspect head as query and gallery, relevance by the aspect's label, mAP; queries and
             gallery are the labelled rows), `ac_stats.arm_retrieval` (the map is fitted on ALL validation rows and applied to the
             scored set; query expansion runs on the scored set's labelled gallery, k fixed on validation),
             `rt_stats.paired` (10,000 joint query/seed draws, `default_rng(0)`).
  ranges     r_X,a,mAP: max minus min over c* seeds 0 to 4 of the validation mAP at R0 (`bs_artret.own_ranges`), unrounded:
             `OUT/stage1/ret_ranges_<enc>.json`.
  picks      `ac_select.arm_table` (r11's cross-fit by parity of the POSITION in the rows given, guard, score
             (1/3) sum_a (mAP_a - mAP_a(R0)) / r_a, tie rule) on the 5,769 validation rows in `mt_data.order` (sha256); no
             `keys=` is passed (it asserts sorted keys), `check_order` asserts the order instead. X-c* and X's frozen image with
             X's ranges (`OUT/stage1/picks_<enc>.json`, arms `cstar` and `frozen`); the teacher, one arm with three views
             (emotion_focus, style_word, genre_word), once with QWEN's ranges for all encoders (D12;
             `OUT/stage1/picks_teacher.json`, arm `teacher`).
  scorer     `score_encoder`: SC2 (X-c* against X's frozen image, WIN-only, r null) and SC4 (X-c* against the teacher, r =
             r_X,a,mAP from the committed stage 1 record), each at its own picked read-out, 10,000 joint draws.

Stage 1 functions read validation only (keys 'val', c* exports of keyset 'val', default and teacher views of the validation
keys). The scorer reads labels only of the scored set; ranges and picks come from the records. mt_data (ticket 01) and mt_export
(ticket 02) are imported lazily (`_data`, `_export`), so the module imports without them.

CLI (`python mt_ret.py -h`): ranges | pick | pick-teacher | score | replay-iii | replay-vi. Outputs under `mt_paths.OUT`.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mt_paths as MP  # noqa: E402

for _name in ('20261102_backbones_sweep', '20261030_affection_confirmation', '20261029_retrieval',
              '20261027_recipe_sweep'):                                  # this checkout's copies of the inherited code
    if str(MP.TEST / _name) not in sys.path:
        sys.path.insert(0, str(MP.TEST / _name))
import ac_paths  # noqa: E402,F401  (puts r11's, r10's and the older rounds' folders on sys.path)
import ac_select  # noqa: E402
import ac_stats  # noqa: E402
import bs_artret  # noqa: E402
import rt_data  # noqa: E402
import rt_metrics  # noqa: E402
import rt_readout  # noqa: E402
import rt_stats  # noqa: E402

ASPECTS = tuple(MP.ASPECTS)
SEEDS = tuple(MP.SEEDS)
TOL = 1e-9
R_SOURCE = 'X-c* own validation seed range (MMArt validation, mAP at R0, seeds 0 to 4), stage 1 record'
CELL_DATA = {'test': 'mmart_test', 'far': 'mmart_far', 'unseen_artist': 'mmart_unseen_artist',
             'shared_held': 'mmart_shared_held'}
# the c* exports exist for keysets val, new and shared_held; the test, far and unseen-artist sets are subsets of `new`
EXPORT_KEYSET = {'val': 'val', 'shared_held': 'shared_held', 'new': 'new', 'test': 'new', 'far': 'new',
                 'unseen_artist': 'new'}
CELL_KEYS = bs_artret.CELL_KEYS + ('n_rows',)

own_ranges = bs_artret.own_ranges
read_json, write_json, sha_file = bs_artret.read_json, bs_artret.write_json, bs_artret.sha_file


# ============================================================================================ boundaries
def _data():
    return importlib.import_module('mt_data')


def _export():
    return importlib.import_module('mt_export')


def check_order(keys):
    """The rows are in D9's order: unique keys sorted by the sha256 of the key string (`mt_data.order`). r12's sorted-key order
    would group labels in the parity halves of the cross-fit."""
    keys = list(keys)
    assert keys and len(set(keys)) == len(keys) and keys == list(_data().order(keys)), \
        'rows must be in mt_data.order (sha256) order'


def unit(x):
    """float32 unit rows from stored rows (r11's `rt_data.embed`: cast to float32, divide by the row norm)."""
    x = np.asarray(x).astype(np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def export_keyset(keyset):
    assert keyset in EXPORT_KEYSET, f'unknown keyset {keyset!r}'
    return EXPORT_KEYSET[keyset]


def cell_data(keyset):
    assert keyset in CELL_DATA, f'no cell data tag for keyset {keyset!r}'
    return CELL_DATA[keyset]


def cstar_rows(enc, aspect, seed, keyset, keys):
    """X-c*'s unit float32 rows (head `aspect`, `seed`) on `keys` of `keyset`: the export, joined by key."""
    x = _export().load(export_keyset(keyset), enc, aspect, seed, keys)
    assert x.shape[0] == len(keys), (enc, aspect, seed, x.shape)
    return unit(x)


def view_rows(view, keys):
    """A frozen image or teacher view on `keys` (mt_data.features: float32 unit rows, joined by key)."""
    x = _data().features(view, keys)
    assert x.shape[0] == len(keys) and x.dtype == np.float32, (view, x.shape, x.dtype)
    return unit(x)


def val_labels(keys):
    return {a: np.asarray(_data().labels(keys, a), dtype=np.int64) for a in ASPECTS}


def stage1_dir():
    return Path(MP.OUT) / 'stage1'


def readout_of(name, width):
    return bs_artret.readout_of(name, width)


# ============================================================================================ ranges
def r0_map_per_seed(load, labels, workers=8):
    """Validation mAP at R0 per aspect and seed (whole set, labelled rows only, no cross-fit; r11's stage 0a computation).
    load(aspect, seed) -> unit float32 rows; labels {aspect: int64 [n], -1 unlabelled}. Returns {aspect: {seed: mAP}}."""
    return {a: {s: float(rt_metrics.retrieval(load(a, s), labels[a], workers=workers)['ap'].mean()) for s in SEEDS}
            for a in ASPECTS}


def validation_ranges(enc, workers=8):
    """X-c*'s seed ranges on MMArt validation (labels: MMArt's, via mt_data), from the 'val' exports. Validation only.
    Returns {'ranges', 'per_seed', 'sources', 'n_val', 'labelled'}."""
    assert enc in MP.ENCODERS
    keys = list(_data().keys('val'))
    check_order(keys)
    labels = val_labels(keys)
    per_seed = r0_map_per_seed(lambda a, s: cstar_rows(enc, a, s, 'val', keys), labels, workers)
    sources = {}
    for a in ASPECTS:
        sources[a] = {}
        for s in SEEDS:
            f = Path(MP.OUT) / f'exports/val/{enc}/cstar/{a}__s{s}.npz'
            sources[a][str(s)] = dict(path=str(f), sha256=sha_file(f) if f.is_file() else None)
    return dict(ranges=own_ranges(per_seed), per_seed=per_seed, sources=sources, n_val=len(keys),
                labelled={a: int((labels[a] >= 0).sum()) for a in ASPECTS})


def write_ranges(enc, result):
    path = stage1_dir() / f'ret_ranges_{enc}.json'
    write_json(path, dict(encoder=enc, kind='mmart_retrieval_validation_ranges', **result))
    return path


def read_ranges(enc, path=None):
    path = Path(path) if path else stage1_dir() / f'ret_ranges_{enc}.json'
    rec = read_json(path)
    assert rec['encoder'] == enc and set(rec['ranges']) == set(ASPECTS), path
    assert all(np.isfinite(v) and v > 0 for v in rec['ranges'].values()), path
    return rec['ranges']


# ============================================================================================ picks
def readout_doc(xs, labels, ranges, workers=8):
    """One arm's validation table and pick: r11's rule through `ac_select.arm_table`. The cross-fit halves are the parity of the
    POSITION in the rows given, which the caller checks to be `mt_data.order` (no `keys=`). xs: {(aspect, seed or None): unit
    float32 [n, d]} (a frozen arm passes ONE array for the three aspects); labels {aspect: int64 [n]}; ranges {aspect: r_a}.
    Returns {'pick': 'center+k10', 'table': {name: {'map': {aspect: f}, 'score': f, 'kept': b}}, 'ranges', 'dropped'}."""
    assert set(labels) == set(ASPECTS) and set(ranges) == set(ASPECTS), (tuple(labels), tuple(ranges))
    assert all(np.isfinite(v) and v > 0 for v in ranges.values()), ranges
    table, pick = ac_select.arm_table(xs, labels, ranges, workers=workers)
    return dict(pick=rt_readout.name(pick), table=table, ranges={a: float(ranges[a]) for a in ASPECTS},
                dropped=rt_readout.half_dropped(labels))


def pick_encoder(enc, workers=8, ranges=None):
    """Stage 1 record of X's two arms on MMArt validation: X-c* (five seeds, its own head per aspect) and X's frozen image, both
    with X's ranges. Validation only."""
    assert enc in MP.ENCODERS
    ranges = read_ranges(enc) if ranges is None else ranges
    keys = list(_data().keys('val'))
    check_order(keys)
    labels = val_labels(keys)
    xs = {(a, s): cstar_rows(enc, a, s, 'val', keys) for a in ASPECTS for s in SEEDS}
    cstar = readout_doc(xs, labels, ranges, workers)
    del xs
    x = view_rows(MP.DEFAULT_VIEW[enc], keys)
    frozen = readout_doc({(a, None): x for a in ASPECTS}, labels, ranges, workers)    # one array: its eigendecomposition is shared
    return dict(encoder=enc, kind='mmart_retrieval_picks', n_val=len(keys), arms=dict(cstar=cstar, frozen=frozen))


def pick_teacher(workers=8):
    """Stage 1 record of the teacher (one arm, three views), picked once with QWEN's ranges for every encoder (D12). Runs after
    Qwen's ranges exist. Validation only."""
    ranges = read_ranges('qwen')
    keys = list(_data().keys('val'))
    check_order(keys)
    labels = val_labels(keys)
    xs = {(a, None): view_rows(MP.TEACHER_VIEW[a], keys) for a in ASPECTS}
    return dict(kind='mmart_retrieval_picks_teacher', ranges_encoder='qwen', n_val=len(keys),
                arms=dict(teacher=readout_doc(xs, labels, ranges, workers)))


def write_picks(enc, rec):
    assert rec['encoder'] == enc and set(rec['arms']) == {'cstar', 'frozen'}
    path = stage1_dir() / f'picks_{enc}.json'
    write_json(path, rec)
    return path


def write_teacher_picks(rec):
    assert set(rec['arms']) == {'teacher'} and rec['ranges_encoder'] == 'qwen'
    path = stage1_dir() / 'picks_teacher.json'
    write_json(path, rec)
    return path


# ============================================================================================ cells
def make_cell(sc, encoder, aspect, candidate, comparator, candidate_value, comparator_value, d, lo, hi, r, r_source,
              n_draws, n_rows, seeds=SEEDS, data='mmart_test', **extra):
    """One cell of `cells_ret` (index format). SC2 is WIN-only (r null); SC4 carries X's unrounded seed range."""
    assert sc in ('SC2', 'SC4') and aspect in ASPECTS and encoder in MP.ENCODERS
    assert all(np.isfinite(v) for v in (candidate_value, comparator_value, d, lo, hi)), (sc, encoder, aspect)
    assert (r is None) == (sc == 'SC2'), 'only SC4 cells carry a LEVEL unit'
    assert r is None or (np.isfinite(r) and r > 0)
    cell = dict(sc=sc, encoder=encoder, aspect=aspect, task='ret', data=data, candidate=candidate,
                comparator=comparator, candidate_value=float(candidate_value), comparator_value=float(comparator_value),
                d=float(d), lo=float(lo), hi=float(hi), r=None if r is None else float(r), r_source=r_source,
                n_draws=int(n_draws), seeds=list(seeds), n_rows=int(n_rows), **extra)
    assert set(CELL_KEYS) <= set(cell)
    return cell


def aspect_cells(enc, aspect, cand, frozen, teacher, r, n_draws, readouts, n_rows, data='mmart_test', scs=('SC2', 'SC4'),
                 task_names=None):
    """The cells of one aspect from per-query AP (`ac_stats.arm_retrieval` results: cand float64 [5, nq], the others [nq]); the
    arms must have the same query rows. SC2 against `frozen`, SC4 against `teacher` (r = the encoder's range for the aspect).
    Intervals: `rt_stats.paired` (r11's 10,000-draw joint query/seed bootstrap)."""
    ap = np.asarray(cand['ap'], dtype=np.float64)
    assert ap.ndim == 2 and ap.shape[0] == 5
    out = []
    for sc, comp, name, ro in (('SC2', frozen, f'frozen_{enc}', readouts.get('frozen')),
                               ('SC4', teacher, f'teacher_{aspect}', readouts.get('teacher'))):
        if sc not in scs:
            continue
        assert comp is not None, (sc, 'comparator missing')
        assert np.array_equal(cand['rows'], comp['rows']), (enc, aspect, sc, 'query rows differ')
        cp = np.asarray(comp['ap'], dtype=np.float64)
        assert cp.ndim == 1
        p = rt_stats.paired(ap, cp, n_draws)
        out.append(make_cell(sc, enc, aspect, f'cstar_{enc}', name, ap.mean(), cp.mean(), p['d'], p['interval'][0],
                             p['interval'][1], None if sc == 'SC2' else r, None if sc == 'SC2' else R_SOURCE, n_draws,
                             n_rows, data=data, candidate_readout=readouts['cstar'], comparator_readout=ro,
                             n_queries=int(len(cand['rows']))))
    return out


def arm_ap(get_val, get_eval, pick, labels, seeded, workers=8):
    """Per-query AP of one arm under its picked read-out (`ac_stats.arm_retrieval`): the map is fitted on ALL validation rows
    (`get_val(seed)`, no labels) and applied to the scored rows (`get_eval(seed)`); query expansion with the pick's k runs on the
    scored set's labelled gallery. Seeded arm: seeds 0 to 4. Returns arm_retrieval's dict (ap, rows, dropped, point, per_seed)."""
    return ac_stats.arm_retrieval(get_val, get_eval, pick, labels, seeds=SEEDS if seeded else None, workers=workers)


# ============================================================================================ stage 2 scorer
def _arms_of(doc):
    doc = doc.get('picks', doc)
    return doc['arms']


def _ranges_of(doc):
    doc = doc.get('ranges', doc) if isinstance(doc.get('ranges'), dict) else doc
    assert set(doc) == set(ASPECTS) and all(np.isfinite(v) and v > 0 for v in doc.values()), doc
    return {a: float(doc[a]) for a in ASPECTS}


def load_stage1(enc, ranges=None, picks=None, teacher=None):
    """X's ranges, X's picks and the teacher's picks from the COMMITTED stage 1 records (`mt_paths.read_committed_record`:
    `stage1_<enc>.json` keys `ret_ranges` and `picks`, `stage1_teacher.json`), unless given. Returns (ranges, arms, teacher arm)."""
    if ranges is None or picks is None:
        assert ranges is None and picks is None, 'give both ranges and picks, or neither'
        rec = MP.read_committed_record(f'stage1_{enc}.json')
        ranges, picks = rec['ret_ranges'], rec['picks']
    if teacher is None:
        teacher = MP.read_committed_record('stage1_teacher.json')
    arms = _arms_of(picks)
    if 'encoder' in picks.get('picks', picks):
        assert picks.get('picks', picks)['encoder'] == enc
    assert {'cstar', 'frozen'} <= set(arms)
    return _ranges_of(ranges), arms, _arms_of(teacher)['teacher']


def score_encoder(enc, keyset='test', scs=('SC2', 'SC4'), n_draws=10000, workers=8, ranges=None, picks=None, teacher=None):
    """The SC2 and SC4 (mAP) cells of encoder X on `keyset` (default the MMArt test cut), aspect by aspect. Each picked map is
    fitted on all 5,769 validation features (the seed's own c* export; X's frozen image; the teacher's view) and applied to the
    keyset's rows. Queries and gallery are the aspect's labelled rows. Ranges and picks come from the committed stage 1 records
    (or are passed); the validation labels are not read here. Returns a list of cells (aspect by aspect: SC2, SC4)."""
    assert enc in MP.ENCODERS and set(scs) <= {'SC2', 'SC4'} and scs, scs
    data_tag = cell_data(keyset)
    ranges, arms, teach = load_stage1(enc, ranges, picks, teacher)
    d = _data()
    val_keys, keys = list(d.keys('val')), list(d.keys(keyset))
    check_order(val_keys)
    check_order(keys)
    ek = export_keyset(keyset)
    view = MP.DEFAULT_VIEW[enc]
    if 'SC2' in scs:
        fz_val, fz_eval = view_rows(view, val_keys), view_rows(view, keys)
    cells = []
    for a in ASPECTS:
        y = np.asarray(d.labels(keys, a), dtype=np.int64)
        assert y.shape == (len(keys),)
        cand = arm_ap(lambda s: cstar_rows(enc, a, s, 'val', val_keys), lambda s: cstar_rows(enc, a, s, ek, keys),
                      arms['cstar']['pick'], y, True, workers)
        frozen = teach_res = None
        if 'SC2' in scs:
            frozen = arm_ap(lambda s: fz_val, lambda s: fz_eval, arms['frozen']['pick'], y, False, workers)
        if 'SC4' in scs:
            tv, te = view_rows(MP.TEACHER_VIEW[a], val_keys), view_rows(MP.TEACHER_VIEW[a], keys)
            teach_res = arm_ap(lambda s: tv, lambda s: te, teach['pick'], y, False, workers)
        cells += aspect_cells(enc, a, cand, frozen, teach_res, ranges[a], n_draws,
                              dict(cstar=arms['cstar']['pick'], frozen=arms['frozen']['pick'], teacher=teach['pick']),
                              len(keys), data_tag, scs)
    return cells


def cells_path(keyset):
    return (Path(MP.OUT) / 'stage2/cells_ret.json') if keyset == 'test' else \
        Path(MP.OUT) / f'diag/cells_ret_{cell_data(keyset)}.json'


# ============================================================================================ replays
def replay_item(name, ours, theirs):
    return dict(name=name, ours=float(ours), theirs=float(theirs), abs_diff=abs(float(ours) - float(theirs)))


def replay_result(items, **extra):
    worst = max(i['abs_diff'] for i in items)
    return dict(passed=bool(worst <= TOL), max_abs_diff=float(worst), tol=TOL, n_values=len(items), items=items, **extra)


def _compare_cells(got, stored, prefix, intervals=True):
    """Items of `got` cells against the stored cells with the same (sc, aspect): values, and d, lo, hi when `intervals`."""
    by = {(c['sc'], c['aspect']): c for c in stored}
    items = []
    for c in got:
        s = by[(c['sc'], c['aspect'])]
        names = ['candidate_value', 'comparator_value', 'n_queries'] + (['d', 'lo', 'hi'] if intervals else [])
        items += [replay_item(f'{prefix}/{c["sc"]}/{c["aspect"]}/{n}', c[n], s[n]) for n in names]
        if c['r'] is not None or s['r'] is not None:
            items.append(replay_item(f'{prefix}/{c["sc"]}/{c["aspect"]}/r', c['r'], s['r']))
    return items


def replay_iii_ret(n_draws=10000, workers=8, parts=('r14', 'r11'), aspects=ASPECTS):
    """0(iii), retrieval part, on ArtELingo with their recorded picks and validation fits:
    r14 (SigLIP 2): the seed ranges (`r0_map_per_seed`, `own_ranges`), then SC2 and SC6 cells (`aspect_cells`: c* at the recorded
    pick against the frozen image and the teacher, held mAP, d and intervals) against `stage3/cells_artelingo_ret.json`;
    r11 (Qwen): the stage 0a ranges, then c*, plain and instructed held mAP (c* per seed) and SC10 to SC15 against stage 5's
    `verdict.json`; and the per-query AP of plain and instructed against `stage5/perquery`. All through this module's `arm_ap`.
    `n_draws` below the stored 10,000 skips the intervals and the result cannot pass (complete False)."""
    full = n_draws == 10000 and tuple(aspects) == ASPECTS
    items, extra = [], {}
    labels = rt_data.labels('held')
    if 'r14' in parts:
        root = MP.R14_OUT / 'cstar/siglip2'
        picks = read_json(MP.R14_OUT / 'stage1/artret_picks_siglip2.json')
        rng_rec = read_json(MP.R14_OUT / 'stage1/artret_ranges_siglip2.json')
        got = r0_map_per_seed(lambda a, s: bs_artret.cstar_embed(root, 'val', a, s), rt_data.labels('val'), workers)
        for a in ASPECTS:
            for s in SEEDS:
                items.append(replay_item(f'r14/siglip2/range/{a}/mAP s{s}', got[a][s], rng_rec['per_seed'][a][str(s)]))
        ranges = own_ranges(got)
        items += [replay_item(f'r14/siglip2/range/{a}', ranges[a], rng_rec['ranges'][a]) for a in ASPECTS]
        stored = [c for c in read_json(MP.R14_OUT / 'stage3/cells_artelingo_ret.json') if c['encoder'] == 'siglip2']
        fv, fh = bs_artret.frozen_embed('siglip2', 'val'), bs_artret.frozen_embed('siglip2', 'held')
        cells = []
        for a in aspects:
            cand = arm_ap(lambda s: bs_artret.cstar_embed(root, 'val', a, s), lambda s: bs_artret.cstar_embed(root, 'held', a, s),
                          picks['cstar']['pick'], labels[a], True, workers)
            frozen = arm_ap(lambda s: fv, lambda s: fh, picks['frozen']['pick'], labels[a], False, workers)
            tch = arm_ap(lambda s: rt_data.embed('instructed', 'val', a), lambda s: rt_data.embed('instructed', 'held', a),
                         picks['teacher']['pick'], labels[a], False, workers)
            cells += aspect_cells('siglip2', a, cand, frozen, tch, rng_rec['ranges'][a], n_draws,
                                  dict(cstar=picks['cstar']['pick'], frozen=picks['frozen']['pick'],
                                       teacher=picks['teacher']['pick']), len(labels[a]), 'artelingo_held')
        items += _compare_cells(cells, stored, 'r14/siglip2', intervals=n_draws == 10000)
    if 'r11' in parts:
        v = read_json(MP.R11_OUT / 'stage5/verdict.json')
        picks = read_json(MP.R11_OUT / 'stage4/picks.json')
        assert v['draws'] == 10000
        want = read_json(ac_paths.R11_RANGES_FILE)['map']
        got = r0_map_per_seed(lambda a, s: rt_data.embed('cstar', 'val', a, s), rt_data.labels('val'), workers)
        r = own_ranges(got)
        items += [replay_item(f'r11/range/{a}', r[a], want[a]['r']) for a in ASPECTS]
        cells = []
        for a in aspects:
            res = {
                'cstar': arm_ap(lambda s: rt_data.embed('cstar', 'val', a, s), lambda s: rt_data.embed('cstar', 'held', a, s),
                                picks['cstar'], labels[a], True, workers),
                'plain': arm_ap(lambda s: rt_data.embed('plain', 'val', 'emotion'), lambda s: rt_data.embed('plain', 'held', 'emotion'),
                                picks['plain'], labels[a], False, workers),
                'instructed': arm_ap(lambda s: rt_data.embed('instructed', 'val', a), lambda s: rt_data.embed('instructed', 'held', a),
                                     picks['instructed'], labels[a], False, workers)}
            for s in SEEDS:
                items.append(replay_item(f'r11/cstar/{a}/mAP s{s}', res['cstar']['per_seed'][s], v['map']['cstar'][a]['per_seed'][s]))
            items.append(replay_item(f'r11/cstar/{a}/mAP mean', res['cstar']['point'], v['map']['cstar'][a]['mean']))
            for arm in ('plain', 'instructed'):
                st = bs_artret.r11_stored(arm, a)
                assert np.array_equal(st['rows'], res[arm]['rows'])
                items.append(replay_item(f'r11/{arm}/{a}/mAP', res[arm]['point'], v['map'][arm][a]['mean']))
                items.append(replay_item(f'r11/{arm}/{a}/per-query max abs', np.abs(res[arm]['ap'] - st['ap']).max(), 0.0))
            cells += aspect_cells('qwen', a, res['cstar'], res['plain'], res['instructed'], 1.0, n_draws,
                                  dict(cstar=picks['cstar'], frozen=picks['plain'], teacher=picks['instructed']),
                                  len(labels[a]), 'artelingo_held')
        if n_draws == 10000:
            for sid, entry in v['sc'].items():
                (c,) = entry['comparisons']
                if c['aspect'] not in aspects:
                    continue
                sc = 'SC2' if c['comparator'][0] == 'plain' else 'SC4'
                mine = next(x for x in cells if x['sc'] == sc and x['aspect'] == c['aspect'])
                items += [replay_item(f'r11/{sid}/d', mine['d'], c['d']), replay_item(f'r11/{sid}/lo', mine['lo'], c['interval'][0]),
                          replay_item(f'r11/{sid}/hi', mine['hi'], c['interval'][1])]
    res = replay_result(items, complete=bool(full and set(parts) == {'r14', 'r11'}), parts=list(parts),
                        n_draws=n_draws, aspects=list(aspects))
    res['values_ok'], res['passed'] = res['passed'], bool(res['passed'] and res['complete'])      # a reduced run cannot pass
    return res


def replay_vi(arms=None, workers=8):
    """0(vi): the read-out rule (`readout_doc`, the function stage 1 calls) fed r11's stored validation outputs (`rt_data.embed`
    on the ArtELingo validation paintings, r11's sorted order, no `keys=`) gives r11's `stage4/scores.json` (cross-fit mAP, score,
    guard flag of every read-out) and `picks.json`, with r11's r_a for every arm, as `ac_select.replay_r11_stage4`."""
    scores = read_json(MP.R11_OUT / 'stage4/scores.json')
    picks = read_json(MP.R11_OUT / 'stage4/picks.json')
    r_a = {a: float(read_json(ac_paths.R11_RANGES_FILE)['map'][a]['r']) for a in ASPECTS}
    arms = tuple(rt_data.ARMS if arms is None else arms)
    labels = rt_data.labels('val')
    items, got = [], {}
    for arm in arms:
        xs = {}
        for s in (SEEDS if arm in rt_data.SEEDED else (None,)):
            if arm in rt_data.ASPECT_FREE:
                x = rt_data.embed(arm, 'val', 'emotion', s)
                xs.update({(a, s): x for a in ASPECTS})                # one matrix: its eigendecomposition is shared
            else:
                xs.update({(a, s): rt_data.embed(arm, 'val', a, s) for a in ASPECTS})
        doc = readout_doc(xs, labels, r_a, workers)
        assert set(doc['table']) == set(scores[arm]), (arm, 'read-out menu differs')
        for name, t in doc['table'].items():
            for a in ASPECTS:
                items.append(replay_item(f'{arm}/{name}/{a}/crossfit mAP', t['map'][a], scores[arm][name]['map'][a]))
            items.append(replay_item(f'{arm}/{name}/score', t['score'], scores[arm][name]['score']))
            assert t['kept'] == scores[arm][name]['kept'], (arm, name, 'guard flag')
        assert doc['pick'] == picks[arm], (arm, 'pick', doc['pick'], picks[arm])
        got[arm] = doc['pick']
    return replay_result(items, picks=got, arms=list(arms))


# ============================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='cmd', required=True)
    for name, h in (('ranges', 'X-c* validation mAP seed ranges -> OUT/stage1/ret_ranges_<enc>.json'),
                    ('pick', 'X-c* and frozen image picks -> OUT/stage1/picks_<enc>.json (needs the ranges file)'),
                    ('score', 'SC2 and SC4 cells of X on a keyset -> OUT/stage2/cells_ret.json (or OUT/diag/)')):
        s = sub.add_parser(name, help=h)
        s.add_argument('--enc', required=True, choices=MP.ENCODERS)
        if name == 'score':
            s.add_argument('--keyset', default='test', choices=sorted(CELL_DATA))
            s.add_argument('--scs', nargs='+', default=['SC2', 'SC4'], choices=['SC2', 'SC4'])
            s.add_argument('--n-draws', type=int, default=None, help='default 10,000 (smoke: mt_paths.SMOKE_DRAWS)')
    sub.add_parser('pick-teacher', help="the teacher's pick with Qwen's ranges -> OUT/stage1/picks_teacher.json")
    r3 = sub.add_parser('replay-iii', help='0(iii) retrieval part -> OUT/stage0/gate_iii_ret.json')
    r3.add_argument('--n-draws', type=int, default=10000)
    r3.add_argument('--parts', nargs='+', default=['r14', 'r11'], choices=['r14', 'r11'])
    r3.add_argument('--aspects', nargs='+', default=list(ASPECTS), choices=ASPECTS)
    r6 = sub.add_parser('replay-vi', help="0(vi) -> OUT/stage0/gate_vi.json")
    r6.add_argument('--arms', nargs='+', default=None, choices=list(rt_data.ARMS))
    args = p.parse_args(argv)
    t0 = time.monotonic()
    out = Path(MP.OUT)
    assert not str(out).startswith(('/data/SSD2/cosir_r10', '/data/SSD2/cosir_r11', '/data/SSD2/cosir_r14')), 'never write into an older round'
    if args.cmd == 'ranges':
        path = write_ranges(args.enc, validation_ranges(args.enc, args.workers))
        print(f'{args.enc}: ranges {read_ranges(args.enc)} -> {path} ({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'pick':
        path = write_picks(args.enc, pick_encoder(args.enc, args.workers))
        rec = read_json(path)
        print(f'{args.enc}: picks cstar {rec["arms"]["cstar"]["pick"]}, frozen {rec["arms"]["frozen"]["pick"]} -> {path} '
              f'({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'pick-teacher':
        path = write_teacher_picks(pick_teacher(args.workers))
        print(f'teacher: pick {read_json(path)["arms"]["teacher"]["pick"]} -> {path} ({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'score':
        n_draws = args.n_draws or (MP.SMOKE_DRAWS['ret'] if MP.SMOKE else MP.N_DRAWS['ret'])
        cells = score_encoder(args.enc, args.keyset, tuple(args.scs), n_draws, args.workers)
        path = cells_path(args.keyset)
        bs_artret.write_cells(path, cells, [args.enc])
        print(f'{args.enc}: {len(cells)} cells on {args.keyset} -> {path} ({time.monotonic() - t0:.0f}s)')
    else:
        if args.cmd == 'replay-iii':
            res, name = replay_iii_ret(args.n_draws, args.workers, tuple(args.parts), tuple(args.aspects)), 'gate_iii_ret.json'
        else:
            res, name = replay_vi(args.arms, args.workers), 'gate_vi.json'
            res['complete'] = args.arms is None
            res['values_ok'], res['passed'] = res['passed'], bool(res['passed'] and res['complete'])
        path = out / 'stage0' / name
        write_json(path, res)
        print(f'{args.cmd}: {"PASS" if res["passed"] else "FAIL"} max_abs_diff {res["max_abs_diff"]:.3g} '
              f'({time.monotonic() - t0:.0f}s) {path}')
        return 0 if res['passed'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
