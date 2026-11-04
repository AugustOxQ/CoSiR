"""r19 ticket 09: the SemArt scorer (SC11 to SC14, all seven encoders), SemArt's units and picks record, gate (iv)'s SemArt part.

Spec: docs/superpowers/specs/2026-11-03-past-picks-design.md, SC11 to SC14, §3.3 (SemArt set, order, read-out maps), §3.4
"SemArt", §3.5 (iv), D9, D17, D18, D20. Genre only, every arm frozen, on the 11,159 paintings of `pp_semart.keys('set')`
(labels: SemArt's 10 types). Nothing of r14's or r15's statistics is rewritten; each score is the function that produced the
replayed number, called on arrays in the row order given as an argument (`order`):

  clustering  SC11 (X-c* against X's frozen image), SC13 (against the matched arm, the larger of its two readings taken again
              in each draw), SC14 (against the teacher; LEVEL unit r): `mt_clu.score_aspect` (r15; -> r14's
              `bs_artclu.frozen_margin`, `matched_margin`, `teacher_margin`), k-means with 20 clusters and states 0 to 2 on the
              rows of the scored set (`ct_score.output_partitions(x, 'k1')['r20']`), 2,000 joint draws. Its cells are
              relabelled SC1, SC3, SC4 -> SC11, SC13, SC14. The matched arm's partitions are `pp_matched.partitions`.
  retrieval   SC12 (against the frozen image), SC14 (against the teacher; LEVEL unit r): `mt_ret.arm_ap` (->
              `ac_stats.arm_retrieval`: the read-out map is fitted on all ArtELingo validation rows of the arm and applied
              to the SemArt rows; query expansion with the pick's k reads the SemArt gallery) and `mt_ret.aspect_cells`
              (`rt_stats.paired`, 10,000 joint query and seed draws); relabelled SC2, SC4 -> SC12, SC14. `mt_ret.aspect_cells`
              asserts the encoder against `mt_paths.ENCODERS`, which `pp_register` extends to all seven.
  units       X-c*'s genre range over seeds 0 to 4 on ArtELingo validation (AMI at 20 clusters; mAP at R0): the SAME range
              code gate (vii) checks (`bs_artclu.clustering_ranges`, `bs_artret.validation_ranges`) on the five validation
              exports; r14's three and the new three are asserted equal to their stage 1 records, Qwen's is computed (D20).
  picks       each arm's existing ArtELingo pick (§3.4): r11's for Qwen (c*, plain, instructed), r14's records for SigLIP 2,
              SigLIP 1, CLIP L/14 (frozen: r11's stored pick for SigLIP 2 and CLIP L/14, r14's computed pick for SigLIP 1),
              this round's committed stage 1 record for the new three; the teacher's pick is r11's 'instructed' for all.
  stage 2     reads the units and picks only through `pp_paths.read_committed_record('stage1_semart.json')`.

Readers (the index layout, tickets 05, 06 and 01): `Inputs` names six readers that return rows in the order of the keys they
are given, joined by key. `SemartInputs` is the production wiring (c* SemArt exports `pp_export.load('semart', ...)`, the
ArtELingo validation exports of every c* from `rt_data._cstar` through `bs_artret.cstar_embed`, frozen and teacher rows from
`pp_stores`, matched partitions from `pp_matched`). `GateInputs` feeds the same scoring functions ArtELingo validation and
held genre rows in r14's order, for gate (iv).

`subset=` (keys) re-scores on those keys only (ticket 13: D14's row): k-means, the matched arm's partitions, the gallery and the
bootstrap all run on the subset rows, in the order of `order`.

CLI (`python pp_semart_score.py -h`): stage1 | score | gate-iv. Smoke: PP_OUTPUTS and PP_SMOKE=1 (fewer draws).
"""
import pp_paths as PP  # noqa: I001  (first: sets the older rounds' output roots)
import pp_register  # noqa: F401  (all seven encoders in r14's and r15's tables)
import pp_void_scope

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import ac_select
import bs_artclu
import bs_artret
import mt_clu
import mt_ret
import pp_outcome
import rt_data

ENCODERS = tuple(PP.ENCODERS)
SEEDS = tuple(PP.SEEDS)
ASPECT = 'genre'
DATA = 'semart_new'
TOL = 1e-9                                           # gate (iv) and the stored units
SC_CLU = {'SC1': 'SC11', 'SC3': 'SC13', 'SC4': 'SC14'}      # mt_clu's labels -> r19's
SC_RET = {'SC2': 'SC12', 'SC4': 'SC14'}
OLD_SC = {'SC11': 'SC1', 'SC12': 'SC2', 'SC13': 'SC3', 'SC14': 'SC6'}   # r14's names of the same cells (gate (iv))
PARTS = {'semart_clu': 'clu', 'semart_ret': 'ret'}
R_SOURCE_CLU = ('X-c* own validation seed range (clustering AMI at 20 clusters, seeds 0 to 4, ArtELingo validation, genre '
                'labels), stage 1 record')
R_SOURCE_RET = 'X-c* own validation seed range (mAP at R0, seeds 0 to 4, ArtELingo validation, genre labels), stage 1 record'
R11_PICKS = PP.R11_OUT / 'stage4/picks.json'
R11_FROZEN_ARM = {'qwen': 'plain', 'siglip2': 'siglip2', 'clip_l14': 'clip'}    # encoders whose frozen pick r11 stored
OLD_RECORD = ('siglip2', 'siglip1', 'clip_l14')       # r14's stage 1 records
STAGE1_NAME = 'stage1_semart.json'
ARMS = ('cstar', 'frozen', 'teacher')


# ============================================================================================ small helpers
def draws(task, n_draws=None):
    """The draws of a production call: the protocol's counts; smoke may take others (default `PP.SMOKE_DRAWS`)."""
    protocol = PP.N_DRAWS[task]
    if PP.SMOKE:
        return PP.SMOKE_DRAWS[task] if n_draws is None else int(n_draws)
    assert n_draws in (None, protocol), f'{task}: {n_draws} draws; outside smoke the protocol\'s {protocol}'
    return protocol


def restrict(order, subset=None):
    """The keys to score: `order` itself, or its members that are in `subset`, in the order of `order`."""
    order = [str(k) for k in order]
    assert order and len(set(order)) == len(order), 'order: keys must be unique'
    if subset is None:
        return order
    subset = {str(k) for k in subset}
    stray = subset - set(order)
    assert not stray, f'subset: {len(stray)} keys are not in the order, e.g. {sorted(stray)[:2]}'
    keys = [k for k in order if k in subset]
    assert keys, 'subset: no key left'
    return keys


def production_order():
    """The SemArt set's keys in sha256 order (`pp_semart.keys('set')`), asserted to be in that order."""
    import pp_semart
    keys = [str(k) for k in pp_semart.keys('set')]
    assert len(keys) == PP.N_SEMART, f'{len(keys)} SemArt keys, {PP.N_SEMART} expected'
    assert keys == PP.order(keys), 'pp_semart.keys("set") is not in the sha256 order (D19)'
    return keys


def n_val_rows():
    """The number of ArtELingo validation rows (6,152): the fit set of every read-out map."""
    return len(rt_data.names('val'))


def _rows(x, n, name):
    """Float32 unit rows, one per key (checked at the reader boundary)."""
    x = np.asarray(x)
    assert x.ndim == 2 and x.shape[0] == n, f'{name}: {x.shape} rows for {n} keys'
    assert x.dtype == np.float32, f'{name}: dtype {x.dtype}, float32 expected'
    assert np.isfinite(x).all(), f'{name}: non-finite values'
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-3, f'{name}: rows are not unit'
    return x


def _matched(parts, n, name):
    """Matched partitions {seed: {'own': int [1, n], 'r20': int [3, n]}} (the form of `pp_matched.partitions`)."""
    assert sorted(parts) == list(SEEDS), f'{name}: seeds {sorted(parts)}'
    out = {}
    for s in SEEDS:
        own, r20 = np.asarray(parts[s]['own']), np.asarray(parts[s]['r20'])
        assert own.shape == (1, n) and r20.shape == (3, n), f'{name} seed {s}: own {own.shape}, r20 {r20.shape} for {n} keys'
        assert np.issubdtype(own.dtype, np.integer) and np.issubdtype(r20.dtype, np.integer)
        out[s] = {'own': own, 'r20': r20}
    return out


# ============================================================================================ readers
class Inputs:
    """The readers the scorer needs; every one returns rows in the order of the keys it is given (joined by key, C6).
    `*_val` rows are the ArtELingo validation rows (6,152, `rt_data.names('val')` order), the fit set of the read-out map."""

    def labels(self, keys):                     # int64 [n]: SemArt's type index of each key (-1: no label)
        raise NotImplementedError

    def cstar_val(self, enc, seed):             # float32 [6152, 2048]: seed s's own ArtELingo validation genre export
        raise NotImplementedError

    def cstar_eval(self, enc, seed, keys):      # float32 [n, 2048]: X-c*'s genre export of seed s on the scored rows
        raise NotImplementedError

    def frozen_val(self, enc):                  # float32 [6152, w]: X's default ArtELingo image
        raise NotImplementedError

    def frozen_eval(self, enc, keys):           # float32 [n, w]: X's default image on the scored rows
        raise NotImplementedError

    def teacher_val(self):                      # float32 [6152, 2048]: Qwen `genre_word` on ArtELingo validation
        raise NotImplementedError

    def teacher_eval(self, keys):               # float32 [n, 2048]: Qwen `genre_word` on the scored rows
        raise NotImplementedError

    def matched(self, enc, keys):               # {seed: {'own': int [1, n], 'r20': int [3, n]}}: the matched arm's readings
        raise NotImplementedError


class SemartInputs(Inputs):
    """Production wiring: the SemArt set (ticket 03), exports (06), probabilities (05), stores (01). `task` ('clu' | 'ret')
    picks the form of the X-c* rows on the scored set: `pp_export.load` returns float16 stored rows, and each task takes
    them the way its inherited code does. Retrieval: r15's `cstar_rows` form, `mt_ret.unit` (cast to float32, divide by the
    row norm). Clustering: the cast-only route of `mt_clu.cstar_fixed` (r14's `S0.load_candidate`; float32, no division)."""

    def __init__(self, task):
        assert task in ('clu', 'ret'), task
        self.task = task

    def labels(self, keys):
        import pp_semart
        return pp_semart.labels(keys)

    def cstar_val(self, enc, seed):
        return bs_artret.cstar_embed(PP.CSTAR_ROOT[enc], 'val', ASPECT, seed)

    def cstar_eval(self, enc, seed, keys):
        import pp_export
        x = pp_export.load('semart', enc, ASPECT, seed, keys)
        return mt_ret.unit(x) if self.task == 'ret' else np.asarray(x).astype(np.float32)

    def frozen_val(self, enc):
        return bs_artret.frozen_embed(bs_artret.ENC[enc]['view'], 'val')

    def frozen_eval(self, enc, keys):
        import pp_stores
        return pp_stores.load('semart', enc, keys)

    def teacher_val(self):
        return rt_data.embed('instructed', 'val', ASPECT)

    def teacher_eval(self, keys):
        import pp_stores
        return pp_stores.load('semart', PP.TEACHER_VIEW[ASPECT], keys)

    def matched(self, enc, keys):
        import pp_matched
        return pp_matched.partitions('semart', enc, keys)


# ============================================================================================ scoring
def score_clu(enc, inputs, order, unit, n_draws, workers=8, subset=None, data=DATA, matched_name=None, use_matched=True,
              use_teacher=True):
    """SC11, SC13, SC14 (clustering) of encoder `enc` on the rows `order` (or its `subset`): the cells of
    `mt_clu.score_aspect` relabelled. `unit`: X-c*'s genre AMI range (the LEVEL unit of SC14).
    `use_matched=False` (the status voids the matched arm, §3.5 row (v)): the matched arm is not read, no SC13.
    `use_teacher=False` (gate (i) genre_word failed, row (i)): the teacher is not read, no SC14."""
    keys = restrict(order, subset)
    n = len(keys)
    y = np.asarray(inputs.labels(keys))
    assert y.shape == (n,) and np.issubdtype(y.dtype, np.integer) and (y >= 0).any(), f'labels {y.shape} for {n} keys'
    c_fixed = mt_clu.cstar_fixed({s: _rows(inputs.cstar_eval(enc, s, keys), n, f'{enc} c* s{s}') for s in SEEDS})
    frozen = mt_clu.single_fixed(_rows(inputs.frozen_eval(enc, keys), n, f'{enc} frozen'))
    teacher = mt_clu.single_fixed(_rows(inputs.teacher_eval(keys), n, 'teacher')) if use_teacher else None
    matched = _matched(inputs.matched(enc, keys), n, f'{enc} matched') if use_matched else None
    scs = tuple(sc for sc in SC_CLU if (use_matched or sc != 'SC3') and (use_teacher or sc != 'SC4'))
    cells = mt_clu.score_aspect(enc, ASPECT, y.astype(np.int64), c_fixed, frozen=frozen, matched=matched, teacher=teacher,
                                scs=scs, r=unit, r_source=R_SOURCE_CLU, n_draws=n_draws, workers=workers,
                                data=data, matched_name=matched_name or PP.MATCHED[enc])
    for c in cells:
        c['sc'] = SC_CLU[c['sc']]
    return cells


def ret_arms(enc, inputs, order, picks, workers=8, subset=None, use_teacher=True):
    """Per-query AP of the three arms on the scored rows (`mt_ret.arm_ap`): each arm at its own pick, the map fitted on all
    ArtELingo validation rows of the arm (c*: seed s's own export), query expansion on the scored gallery with the pick's k.
    Returns {'cstar': ap [5, nq], 'frozen': ap [nq], 'teacher': ap [nq]} as `arm_ap` dicts (ap, rows, dropped, point, per_seed).
    `use_teacher=False` (gate (i) genre_word failed): the teacher's rows and pick are not read; 'teacher' is None."""
    keys = restrict(order, subset)
    n = len(keys)
    y = np.asarray(inputs.labels(keys))
    assert y.shape == (n,) and np.issubdtype(y.dtype, np.integer) and (y >= 0).any()
    y = y.astype(np.int64)
    assert set(picks) >= set(ARMS if use_teacher else ARMS[:2])

    def fit_rows(x, name):
        x = np.asarray(x)
        want = n_val_rows()
        assert x.ndim == 2 and x.shape[0] == want, f'{name}: {x.shape[0]} fit rows, {want} ArtELingo validation rows expected'
        return x
    cand = mt_ret.arm_ap(lambda s: fit_rows(inputs.cstar_val(enc, s), f'{enc} c* val s{s}'),
                         lambda s: _rows(inputs.cstar_eval(enc, s, keys), n, f'{enc} c* s{s}'), picks['cstar'], y, True, workers)
    fz_val, fz_eval = fit_rows(inputs.frozen_val(enc), f'{enc} frozen val'), _rows(inputs.frozen_eval(enc, keys), n, f'{enc} frozen')
    frozen = mt_ret.arm_ap(lambda s: fz_val, lambda s: fz_eval, picks['frozen'], y, False, workers)
    teacher = None
    if use_teacher:
        t_val, t_eval = fit_rows(inputs.teacher_val(), 'teacher val'), _rows(inputs.teacher_eval(keys), n, 'teacher')
        teacher = mt_ret.arm_ap(lambda s: t_val, lambda s: t_eval, picks['teacher'], y, False, workers)
    return dict(cstar=cand, frozen=frozen, teacher=teacher)


def score_ret(enc, inputs, order, unit, picks, n_draws, workers=8, subset=None, data=DATA, use_teacher=True):
    """SC12, SC14 (retrieval) of encoder `enc`: `mt_ret.aspect_cells` on `ret_arms`, relabelled. `unit`: X-c*'s genre mAP
    range at R0 (the LEVEL unit of SC14). `use_teacher=False` (gate (i) genre_word failed): no teacher read, no SC14."""
    keys = restrict(order, subset)
    arms = ret_arms(enc, inputs, order, picks, workers, subset, use_teacher)
    cells = mt_ret.aspect_cells(enc, ASPECT, arms['cstar'], arms['frozen'], arms['teacher'], unit, n_draws, dict(picks),
                                len(keys), data, tuple(sc for sc in SC_RET if use_teacher or sc != 'SC4'))
    for c in cells:
        c['sc'] = SC_RET[c['sc']]
        if c['sc'] == 'SC14':
            c['r_source'] = R_SOURCE_RET
    return cells


# ============================================================================================ cells on disk
def cells_path(part, out=None):
    assert part in PARTS, part
    return Path(PP.OUT if out is None else out) / 'stage2' / f'cells_{part}.json'


def write_cells(part, enc, cells, out=None, smoke=None):
    """Check every cell (`pp_outcome.check_cell`: fields, SC, task, data, `r`, the protocol's draws) and merge them into
    `OUT/stage2/cells_<part>.json` (a cell replaces the stored one of the same encoder, sc, aspect and data)."""
    smoke = bool(PP.SMOKE) if smoke is None else smoke
    task = PARTS[part]
    assert cells, 'no cells'
    for c in cells:
        pp_outcome.check_cell(c, smoke=smoke)
        assert c['encoder'] == enc and c['data'] == DATA and c['task'] == task, (part, enc, pp_outcome.cell_key(c))
    path = cells_path(part, out)
    with PP.file_lock(path):
        mt_clu.write_cells(path, cells)
    return path


# ============================================================================================ stage 1: units and picks
def unit_values(enc, root=None, clu_fn=None, ret_fn=None, recorded=None):
    """X-c*'s genre LEVEL units {'clu', 'ret'} from the five runs' ArtELingo validation exports under `root` (default
    `PP.CSTAR_ROOT[enc]`), with the range code gate (vii) checks: `bs_artclu.clustering_ranges` (AMI at 20 clusters) and
    `bs_artret.validation_ranges` (mAP at R0), genre entry. `recorded` ({'clu', 'ret'}): the stage 1 record's values, asserted
    equal within 1e-9."""
    root = PP.CSTAR_ROOT[enc] if root is None else root
    clu_fn = bs_artclu.clustering_ranges if clu_fn is None else clu_fn
    ret_fn = bs_artret.validation_ranges if ret_fn is None else ret_fn
    unit = dict(clu=float(clu_fn(root)['ranges'][ASPECT]), ret=float(ret_fn(root)['ranges'][ASPECT]))
    for k, v in unit.items():
        assert math.isfinite(v) and v > 0, f'{enc}: {k} unit {v!r}'
    if recorded is not None:
        for k in ('clu', 'ret'):
            assert abs(unit[k] - float(recorded[k])) <= TOL, \
                f'{enc}: {k} unit {unit[k]!r} differs from the stage 1 record\'s {recorded[k]!r}'
    return unit


def r11_map_range():
    """r11's stored genre mAP range (`ac_paths.R11_RANGES_FILE`, map, genre, r): what `mt_ret.replay_iii_ret` compares to."""
    return float(mt_ret.read_json(mt_ret.ac_paths.R11_RANGES_FILE)['map'][ASPECT]['r'])


def load_record(enc):
    """The encoder's ArtELingo stage 1 record: r14's committed `records/stage1_<enc>.json` (SigLIP 2, SigLIP 1, CLIP L/14), this
    round's committed record (the new three, through `read_committed_record`), None for Qwen (r10's c*, no r14 record)."""
    if enc == 'qwen':
        return None
    if enc in OLD_RECORD:
        return json.loads((PP.R14_RECORDS / f'stage1_{enc}.json').read_text())
    assert enc in PP.NEW_ENCODERS, enc
    return PP.read_committed_record(f'stage1_{enc}.json')


def recorded_units(record):
    return dict(clu=float(record['artclu']['ranges'][ASPECT]), ret=float(record['artret_ranges']['ranges'][ASPECT]))


def arm_picks(enc, record, r11):
    """The three ArtELingo read-out picks of the SemArt arms and where each comes from: {'cstar', 'frozen', 'teacher'}, sources.
    Qwen: r11's cstar, plain, instructed. Others: the stage 1 record's `artret_picks`; checked against r11's stored picks
    where r11 stored them (teacher: 'instructed'; frozen: SigLIP 2 'siglip2', CLIP L/14 'clip'). Every pick must be in the menu."""
    if enc == 'qwen':
        assert record is None
        picks = dict(cstar=r11['cstar'], frozen=r11['plain'], teacher=r11['instructed'])
        source = dict(cstar="r11 stored pick, arm cstar (r10's c*)", frozen='r11 stored pick, arm plain',
                      teacher='r11 stored pick, arm instructed')
    else:
        assert record is not None and record['encoder'] == enc, f'{enc}: stage 1 record of {record and record["encoder"]}'
        p = record['artret_picks']
        picks = dict(cstar=p['cstar']['pick'], frozen=p['frozen']['pick'], teacher=p['teacher']['pick'])
        assert picks['teacher'] == r11['instructed'], f'{enc}: teacher pick {picks["teacher"]!r} is not r11\'s'
        arm = R11_FROZEN_ARM.get(enc)
        if arm:
            assert picks['frozen'] == r11[arm], f'{enc}: frozen pick {picks["frozen"]!r} is not r11\'s ({arm})'
        where = ('records/stage1_%s.json (r14)' if enc in OLD_RECORD else "this round's committed records/stage1_%s.json") % enc
        source = dict(cstar=f'{where}, artret_picks.cstar (X-c*\'s own ranges)',
                      frozen=(f'r11 stored pick, arm {arm}; in {where}' if arm else
                              f"{where}, artret_picks.frozen (computed: r11's stage 4 rule with r11's r_a)"),
                      teacher=f'r11 stored pick, arm instructed; in {where}')
    width = dict(cstar=2048, frozen=PP.WIDTH[enc], teacher=2048)
    for arm_name, pick in picks.items():
        bs_artret.readout_of(pick, width[arm_name])           # in the menu of that arm's width (asserts)
    return picks, source


def stage1_entry(clu, ret, picks, units_source, picks_source):
    return dict(units=dict(clu=clu, ret=ret), units_source=units_source, picks=dict(picks), picks_source=dict(picks_source))


def check_stage1(doc):
    """The form of `records/stage1_semart.json`: all seven encoders, finite positive units, three picks each."""
    assert set(doc) == set(ENCODERS), f'encoders {sorted(doc)}'
    for enc, e in doc.items():
        assert set(e) == {'units', 'units_source', 'picks', 'picks_source'}, (enc, sorted(e))
        assert set(e['units']) == {'clu', 'ret'} and all(math.isfinite(v) and v > 0 for v in e['units'].values()), (enc, e['units'])
        assert set(e['picks']) == set(ARMS) == set(e['picks_source']), (enc, e['picks'])
        assert all(isinstance(p, str) and '+k' in p for p in e['picks'].values()), (enc, e['picks'])
    return doc


def stage1(encs=ENCODERS):
    """SemArt's units and picks for all seven encoders (ArtELingo validation only; no SemArt row is read). The three-line
    entries are checked against the stage 1 records as described in the module docstring."""
    r11 = json.loads(R11_PICKS.read_text())
    doc = {}
    for enc in encs:
        t0 = time.monotonic()
        rec = load_record(enc)
        recorded = None if rec is None else recorded_units(rec)
        u = unit_values(enc, recorded=recorded)
        r11_checked = None
        if enc == 'qwen':                        # no r14 record: the mAP unit is r11's own stored range (D20)
            r11_checked = r11_map_range()
            assert abs(u['ret'] - r11_checked) <= TOL, f'qwen: mAP unit {u["ret"]!r} differs from r11\'s stored {r11_checked!r}'
        picks, picks_source = arm_picks(enc, rec, r11)
        how = ("computed with bs_artclu.clustering_ranges and bs_artret.validation_ranges on the five ArtELingo validation "
               f"exports of {PP.CSTAR_ROOT[enc]}, genre entry")
        if recorded is not None:
            who = 'r14' if enc in OLD_RECORD else "this round's"
            how += f"; equal within 1e-9 to {who} committed records/stage1_{enc}.json"
        if r11_checked is not None:
            how += f"; the mAP unit equals r11's stored range ({mt_ret.ac_paths.R11_RANGES_FILE.name}, map, genre, r = {r11_checked!r}) within 1e-9"
        doc[enc] = stage1_entry(u['clu'], u['ret'], picks, how, picks_source)
        print(f'{enc}: units clu {u["clu"]:.6g} ret {u["ret"]:.6g}, picks {picks} ({time.monotonic() - t0:.0f}s)', flush=True)
    return doc


def write_stage1(doc, out=None):
    """Write `OUT/stage1/semart.json` (ticket 12 merges it into records/stage1_semart.json). A file that exists with other
    content is refused: picks are frozen once written."""
    check_stage1(doc)
    path = Path(PP.OUT if out is None else out) / 'stage1' / 'semart.json'
    text = json.dumps(doc, indent=1, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == text, f'{path} holds other content: picks are frozen once written'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.pending.json')
        tmp.write_text(text)
        tmp.replace(path)
    return path


def stage1_for(enc):
    """(units, picks) of `enc` from the COMMITTED record (`pp_paths.read_committed_record`: refuses unless equal to git HEAD)."""
    assert enc in ENCODERS, enc
    doc = check_stage1(PP.read_committed_record(STAGE1_NAME))
    return dict(doc[enc]['units']), dict(doc[enc]['picks'])


# ============================================================================================ stage 2
def score(part, enc, n_draws=None, workers=8, inputs=None, subset=None, status=None):
    """Production: the cells of `part` ('semart_clu' | 'semart_ret') for `enc` on the SemArt set in sha256 order, units and
    picks from the committed stage 1 record; written to `OUT/stage2/cells_<part>.json`. `subset`: keys (ticket 13 passes it
    and writes its own cells; this function then returns the cells without writing).
    The committed status is read next (`pp_void_scope.committed_status`; `status` overrides it for tests; a missing file raises
    outside smoke), before any row: gate (i) genre_word failed -> the teacher is not read and SC14 is not written (row (i));
    the matched arm void for `enc` (runs.<enc>.matched or gate (v)) -> the matched arm is not read and SC13 is not written."""
    task = PARTS[part]
    n_draws = draws(task, n_draws)
    units, picks = stage1_for(enc)
    status = pp_void_scope.resolve(status)
    use_teacher = not pp_void_scope.teacher_void(status, enc, 'semart')
    use_matched = not pp_void_scope.matched_void(status, enc, 'semart')
    order = production_order()
    inputs = SemartInputs(task) if inputs is None else inputs
    if task == 'clu':
        cells = score_clu(enc, inputs, order, units['clu'], n_draws, workers, subset, use_matched=use_matched,
                          use_teacher=use_teacher)
    else:
        cells = score_ret(enc, inputs, order, units['ret'], picks, n_draws, workers, subset, use_teacher=use_teacher)
    if subset is None:
        write_cells(part, enc, cells)
    return cells


# ============================================================================================ gate (iv): the SemArt path on ArtELingo
class GateInputs(Inputs):
    """ArtELingo validation and held genre rows in r14's order, through the loaders that produced r14's numbers (r15's
    replays: `mt_clu.replay_iii` for clustering, `mt_ret.replay_iii_ret` for retrieval). `keys` must be the held names."""

    def __init__(self, task):
        assert task in ('clu', 'ret')
        self.task = task

    def _held(self, keys):
        names = bs_artclu.held_setup()[0] if self.task == 'clu' else rt_data.names('held')
        assert list(keys) == list(names), 'the gate scores the held paintings in r14\'s order'

    def labels(self, keys):
        self._held(keys)
        return bs_artclu.held_setup()[2][ASPECT] if self.task == 'clu' else rt_data.labels('held')[ASPECT]

    def cstar_val(self, enc, seed):
        return bs_artret.cstar_embed(PP.CSTAR_ROOT[enc], 'val', ASPECT, seed)

    def cstar_eval(self, enc, seed, keys):
        self._held(keys)
        root = PP.CSTAR_ROOT[enc]
        if self.task == 'ret':
            return bs_artret.cstar_embed(root, 'held', ASPECT, seed)
        cand = bs_artclu.S0.Candidate('C', root / 'held' / PP.CSTAR_ARM, PP.CSTAR_ARM)
        return bs_artclu.S0.load_candidate(cand, 'full', ASPECT, seed)[0]

    def frozen_val(self, enc):
        return bs_artret.frozen_embed(bs_artret.ENC[enc]['view'], 'val')

    def frozen_eval(self, enc, keys):
        self._held(keys)
        if self.task == 'ret':
            return bs_artret.frozen_embed(bs_artret.ENC[enc]['view'], 'held')
        return bs_artclu.store_loader(enc, keys)

    def teacher_val(self):
        return rt_data.embed('instructed', 'val', ASPECT)

    def teacher_eval(self, keys):
        self._held(keys)
        if self.task == 'ret':
            return rt_data.embed('instructed', 'held', ASPECT)
        return mt_clu._unit32(rt_data._instructed('held', ASPECT, keys))

    def matched(self, enc, keys):
        self._held(keys)
        spec = bs_artclu.ENC[enc]
        probs = {}
        for s in SEEDS:
            with np.load(Path(spec['root']) / f'full/held/{spec["matched"]}/P3__s{s}.npz', allow_pickle=False) as z:
                probs[s] = z['prob']
        return mt_clu.matched_from_probs(probs)


def _item(ours, theirs):
    diff = abs(float(ours) - float(theirs))
    return dict(ours=float(ours), theirs=float(theirs), abs_diff=diff, ok=bool(diff <= TOL))


def compare_gate(ours, theirs, full):
    """Our SemArt-path cells (SC11 to SC14, run on ArtELingo held genre) against r14's stored cells (SC1, SC2, SC3, SC6), by
    (task, sc): candidate value, comparator value, d always; lo and hi when `full` (the protocol's draws); `r` where either has
    one; the matched reading's name; `n_queries` where both have it. The two sides must hold the same cells. At reduced draws
    `passed` is False whatever the values (`values_ok` tells the values)."""
    mine = {(c['task'], OLD_SC[c['sc']]): c for c in ours}
    old = {(c['task'], c['sc']): c for c in theirs}
    assert len(mine) == len(ours) and len(old) == len(theirs), 'duplicate cells'
    assert set(mine) == set(old), f'cells differ: ours {sorted(mine)}, r14 {sorted(old)}'
    items = {}
    for (task, sc), c in sorted(mine.items()):
        s = old[(task, sc)]
        tag = f'{sc}/{task}'
        fields = ['candidate_value', 'comparator_value', 'd'] + (['lo', 'hi'] if full else [])
        for f in fields:
            items[f'{tag}/{f}'] = _item(c[f], s[f])
        if c.get('r') is not None or s.get('r') is not None:
            items[f'{tag}/r'] = _item(c['r'], s['r']) if c.get('r') is not None and s.get('r') is not None \
                else dict(ours=c.get('r'), theirs=s.get('r'), abs_diff=math.inf, ok=False)
        if 'comparator_reading' in s or 'comparator_reading' in c:
            same = c.get('comparator_reading') == s.get('comparator_reading')
            items[f'{tag}/comparator_reading'] = dict(ours=c.get('comparator_reading'), theirs=s.get('comparator_reading'),
                                                      abs_diff=0.0 if same else 1.0, ok=same)
        if 'n_queries' in s and 'n_queries' in c:
            items[f'{tag}/n_queries'] = _item(c['n_queries'], s['n_queries'])
    values_ok = bool(all(i['ok'] for i in items.values()))
    return dict(values_ok=values_ok, passed=bool(values_ok and full), max_abs_diff=max(i['abs_diff'] for i in items.values()),
                tol=TOL, intervals_compared=bool(full), items=items)


def stored_r14_cells(task):
    """r14's stored CLIP L/14 genre cells (`stage3/cells_artelingo_{clu,ret}.json`)."""
    f = PP.R14_OUT / 'stage3' / f'cells_artelingo_{task}.json'
    return [c for c in json.loads(f.read_text()) if c['encoder'] == 'clip_l14' and c['aspect'] == ASPECT]


def gate_iv(n_clu=None, n_ret=None, workers=8, tasks=('clu', 'ret'), out=None):
    """Gate (iv), SemArt part (spec §3.5): the SemArt path (`score_clu`, `score_ret`: genre only, maps fitted on ArtELingo
    validation, row order an argument) fed ArtELingo validation and held genre inputs in r14's order, with r14's CLIP L/14
    picks and M7's stored held probabilities, against r14's CLIP L/14 genre cells (SC1, SC2, SC3, SC6): values and intervals
    within 1e-9. The LEVEL unit passed in is r14's stored one, compared back as a cell field. Writes
    `OUT/stage0/gate_iv_semart.json`."""
    enc = 'clip_l14'
    n_clu, n_ret = draws('clu', n_clu), draws('ret', n_ret)
    rec = load_record(enc)
    picks, _ = arm_picks(enc, rec, json.loads(R11_PICKS.read_text()))
    result = dict(gate='iv_semart', encoder=enc, n_draws=dict(clu=n_clu, ret=n_ret), picks=picks, parts={})
    for task in tasks:
        stored = stored_r14_cells(task)
        unit = next(c['r'] for c in stored if c['sc'] == 'SC6')
        gi = GateInputs(task)
        if task == 'clu':
            order = list(bs_artclu.held_setup()[0])
            ours = score_clu(enc, gi, order, unit, n_clu, workers, data='artelingo_held')
            full = n_clu == PP.N_DRAWS['clu']
        else:
            order = list(rt_data.names('held'))
            ours = score_ret(enc, gi, order, unit, picks, n_ret, workers, data='artelingo_held')
            full = n_ret == PP.N_DRAWS['ret']
        stored = [c for c in stored if (c['sc'], task) in {(OLD_SC[o['sc']], task) for o in ours}]
        result['parts'][task] = compare_gate(ours, stored, full)
        print(f'gate (iv) SemArt path, {task}: values_ok={result["parts"][task]["values_ok"]} '
              f'max_abs_diff={result["parts"][task]["max_abs_diff"]:.3g}', flush=True)
    parts = result['parts'].values()
    result.update(values_ok=all(p['values_ok'] for p in parts), complete=set(tasks) == {'clu', 'ret'},
                  max_abs_diff=max(p['max_abs_diff'] for p in parts))
    result['passed'] = bool(result['complete'] and all(p['passed'] for p in parts))
    path = Path(PP.OUT if out is None else out) / 'stage0' / 'gate_iv_semart.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    result['path'] = str(path)
    return result


# ============================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('stage1', help='units and picks of all seven encoders -> OUT/stage1/semart.json')
    s = sub.add_parser('score', help='SC11 to SC14 of one encoder -> OUT/stage2/cells_semart_{clu,ret}.json')
    s.add_argument('--enc', required=True, choices=ENCODERS)
    s.add_argument('--part', required=True, choices=sorted(PARTS))
    s.add_argument('--n-draws', type=int, default=None)
    g = sub.add_parser('gate-iv', help='gate (iv), SemArt part -> OUT/stage0/gate_iv_semart.json')
    g.add_argument('--n-clu', type=int, default=None)
    g.add_argument('--n-ret', type=int, default=None)
    g.add_argument('--tasks', nargs='+', default=['clu', 'ret'], choices=('clu', 'ret'))
    args = p.parse_args(argv)
    t0 = time.monotonic()
    if args.cmd == 'stage1':
        path = write_stage1(stage1())
        print(f'stage 1 (SemArt units and picks) -> {path} ({time.monotonic() - t0:.0f}s)')
        return 0
    if args.cmd == 'score':
        cells = score(args.part, args.enc, args.n_draws, args.workers)
        print(f'{args.enc} {args.part}: {len(cells)} cells -> {cells_path(args.part)} ({time.monotonic() - t0:.0f}s)')
        return 0
    res = gate_iv(args.n_clu, args.n_ret, args.workers, tuple(args.tasks))
    verdict = 'PASS' if res['passed'] else ('VALUES OK (reduced draws or one task)' if res['values_ok'] else 'FAIL')
    print(f'gate (iv) SemArt part: {verdict} max_abs_diff {res["max_abs_diff"]:.3g} ({time.monotonic() - t0:.0f}s) {res["path"]}')
    return 0 if res['values_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
