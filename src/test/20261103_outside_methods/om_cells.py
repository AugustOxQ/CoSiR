"""r18 stage 2: cells and labels (ticket 10; spec §6 SC1 to SC6 with §3.4 and §3.5, D11, D13, D15, D24).

Every cell goes through r17's paired scorer (`ns_clu.paired_cell`, `ns_ret.arm_ap`, `ns_ret.paired_ret`), imported
read-only. The cells (337), id form `SC/set/encoder/aspect/metric/k/comparator` (k `-` for map):

    SC1  Qwen c* (k-means, [5, 3, n]) against TPIPS k-means on the aspect's vector ([3, n]); class count (4 metrics), 20 (AMI)
    SC2  Qwen c* mAP at c*'s recorded pick against TPIPS at its stage 1 pick
    SC3  Qwen and CLIP L/14 c* (k-means) against CLAY k-means (CLAY's clustering text set from stage 1)
    SC4  Qwen and CLIP L/14 c* mAP against CLAY at its stage 1 pick and retrieval text set
    SC5  X-c* + TURTLE ([5, 1, n] repeated to [5, 3, n]) against X plain + TURTLE ([3, n]), four encoders
    SC6  X-c* (k-means) against X plain + TURTLE, four encoders
All on the set's held split (ArtELingo `held`, MMArt `test`). Fits exist only there.

Steps (CLI; OUT = `om_paths.OUT`):
    cells    [--sc SC] [--set S] [--enc E] [--draws N] [--force]
             scores every unit (SC, set, encoder) not yet done -> OUT/stage2/parts/<SC>__<set>__<enc>.json (numbers only);
             needs the committed stage 1 records (§3.5: stage 1 is committed before any stage 2 number)
    assemble -> OUT/stage2/cells.json: r17's cell form plus `label`, `void`, `sentence`; r from the committed stage 1 range
             record, voids from the committed `records/status.json` and r17's status (§3.5's table), label by §3.4's rule
    labels   -> records/labels.json; refuses unless OUT/c13/c13_cells.json exists and agrees with cells.json on every pick,
             value, d, lo, hi and label within `om_paths.TOL_C13`. The only writer of labels.json.
    list     counts

r of a cell: `records/stage1_om_ranges.json[range_key(cell)]['r']`; r null (or r17's own range failed) gives
`inconclusive` for a cell that is not WIN; WIN stands. A void (failed gate, missing pick, a cell not scored) keeps
numbers if there are any but no label and no sentence.

c13_cells.json (written by the C13 code, from the saved outputs): {"picks": {<key>: str}, "cells": [{id, cand, comp, d, lo,
hi, label (null for a void cell), candidate_readout, comparator_readout (retrieval cells)}]}. `pick_summary` names the keys.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import om_data
import om_paths as op
import om_stage1 as s1

import ns_clu  # noqa: E402  (om_paths put r17's folder and the older rounds on sys.path)
import ns_clu_cells  # noqa: E402
import ns_metrics  # noqa: E402
import ns_outcome  # noqa: E402
import ns_ret  # noqa: E402

ENCODERS, ASPECTS, SETS, SEEDS = op.ENCODERS, op.ASPECTS, op.SETS, op.SEEDS
OK = 'ok'
STATUS_VALUES = ns_outcome.STATUS_VALUES
OUR_GATES = ('i', 'ii', 'iii_a', 'iii_b', 'iii_c', 'iv')
WIN, LEVEL, BELOW, INCONCLUSIVE = 'WIN', 'LEVEL', 'BELOW', 'inconclusive'
# cell id -> unit: SC -> (task, encoders, candidate arm, comparator arm)
SC_DEFS = {
    'SC1': ('clu', ('qwen',), 'cstar', 'tpips'),
    'SC2': ('ret', ('qwen',), 'cstar', 'tpips'),
    'SC3': ('clu', tuple(e for e in ENCODERS if e in ('qwen', 'clip_l14')), 'cstar', 'clay'),
    'SC4': ('ret', tuple(e for e in ENCODERS if e in ('qwen', 'clip_l14')), 'cstar', 'clay'),
    'SC5': ('clu', ENCODERS, 'cstar_turtle', 'plain_turtle'),
    'SC6': ('clu', ENCODERS, 'cstar', 'plain_turtle'),
}
CELL_KEYS = ns_outcome.CELL_KEYS + ('label', 'void', 'sentence')
# gates (ours) -> SCs voided (§3.5's table)
GATE_VOIDS = {'i': ('SC3', 'SC4'), 'ii': ('SC5', 'SC6'), 'iii_a': ('SC1', 'SC3', 'SC5', 'SC6'), 'iii_b': ('SC5', 'SC6'),
              'iii_c': ('SC2', 'SC4'), 'iv': ('SC1', 'SC2')}
R17_SCORER_CLU = ('SC1', 'SC3', 'SC5', 'SC6')          # cells r17's clustering scorer produces here
R17_SCORER_RET = ('SC2', 'SC4')                        # cells r17's retrieval scorer produces here
TURTLE_STATES = 3


class MissingInput(Exception):
    """A part a cell needs does not exist (a pick, a store, a fit): the cell is reported void, not scored."""


# ================================================================================================ the cell list
def _cell(sc, set_, enc, aspect, metric, k):
    task, _, cand, comp = SC_DEFS[sc]
    kid = '-' if k is None else str(k)
    return dict(id='/'.join((sc, set_, enc, aspect, metric, kid, comp)), sc=sc, set=set_, encoder=enc, aspect=aspect, task=task,
                metric=metric, k=k, candidate=cand, comparator=comp)


def cell_specs():
    """The 337 cells, in the order SC, set, encoder, aspect, then class count before 20 (`om_stage1.class_ks`)."""
    out = []
    for sc, (task, encs, _, _) in SC_DEFS.items():
        for set_ in SETS:
            for enc in encs:
                for aspect in ASPECTS:
                    if task == 'ret':
                        out.append(_cell(sc, set_, enc, aspect, 'map', None))
                        continue
                    for k, metrics in s1.class_ks(set_, aspect):
                        out.extend(_cell(sc, set_, enc, aspect, m, k) for m in metrics)
    assert len(out) == 337 and len({c['id'] for c in out}) == 337
    return out


def units():
    """(sc, set, encoder) in the order of `cell_specs`: 28 units."""
    seen = []
    for c in cell_specs():
        u = (c['sc'], c['set'], c['encoder'])
        if u not in seen:
            seen.append(u)
    return seen


def range_key(cell):
    """The stage 1 range key of a cell (`om_stage1.range_keys()` form): `<set>/<enc>/<aspect>/<metric>/<k>` or `.../map`.
    It follows the rule of the cells' r: c*'s own range (encoder = the cell's encoder, whatever the comparator)."""
    if cell['task'] == 'ret':
        return f"{cell['set']}/{cell['encoder']}/{cell['aspect']}/map"
    return f"{cell['set']}/{cell['encoder']}/{cell['aspect']}/{cell['metric']}/{cell['k']}"


def is_class_count(cell):
    """r17's 'class_count' kind: the cell's k is the aspect's class count (MMArt style: 20)."""
    return cell['task'] == 'clu' and cell['k'] == op.CLASS_COUNT[(cell['set'], cell['aspect'])]


# ================================================================================================ the TURTLE candidate
def repeat_states(parts):
    """int [5, 1, n] (X-c* + TURTLE, one fit per seed) -> [5, 3, n] for `ns_clu._boot`, which wants 3 states: the same
    partition three times keeps the mean over states and the draws of seeds (`np.repeat(parts, 3, axis=1)`)."""
    parts = np.asarray(parts)
    assert parts.ndim == 3 and parts.shape[0] == len(SEEDS) and parts.shape[1] == 1, \
        f'X-c* + TURTLE partitions must be [5, 1, n], got {parts.shape}'
    return np.repeat(parts, TURTLE_STATES, axis=1)


# ================================================================================================ pins
def text_sha(text_set, aspect):
    import om_clay
    return om_data.array_sha256(om_clay.text_set(text_set, aspect))


def check_pin(picks, text_set, aspect):
    """Stage 1 pinned the text rows of every set it read: recompute and compare (a null pin never passes)."""
    pins = picks[s1.PIN_KEY]
    pinned = pins[text_set][aspect]
    assert pinned is not None, f'pin: text set {text_set} has no pinned hash for {aspect}'
    got = text_sha(text_set, aspect)
    assert got == pinned, f'pin: text set {text_set}/{aspect} hashes to {got}, stage 1 pinned {pinned}'


def _picks_get(picks, key):
    if key not in picks:
        raise MissingInput(f'stage 1 record {key} is absent')
    return picks[key]


# ================================================================================================ loaders
class Loaders:
    """The real inputs, all on the held split. Tests pass a stand-in with the same methods."""

    def __init__(self):
        self._part_cache, self._ap_cache, self._rows = {}, {}, {}

    def _held(self, set_):
        return op.HELD_SPLIT[set_]

    def labels(self, set_, aspect):
        return np.asarray(om_data.labels(set_, self._held(set_), aspect))

    def _kmeans(self, key, rows, k, set_):
        if key not in self._part_cache:
            self._part_cache[key] = ns_metrics.partitions(rows(), k, set_)
        return self._part_cache[key]

    def cstar_parts(self, set_, enc, aspect, k):
        """int32 [5, 3, n], r17's candidate form (`ns_clu_cells._cand_parts`)."""
        key = ('cstar', set_, enc, aspect, k)
        if key not in self._part_cache:
            self._part_cache[key] = ns_clu_cells._cand_parts(set_, self._held(set_), enc, aspect, k)
        return self._part_cache[key]

    def _x(self, key, load):
        if key not in self._rows:
            self._rows.clear()                      # one matrix at a time (TPIPS rows are 4,096 wide)
            self._rows[key] = load()
        return self._rows[key]

    def tpips_parts(self, set_, aspect, k):
        rows = lambda: self._x(('tpips', set_, aspect), lambda: om_data.tpips(set_, self._held(set_), aspect))
        return self._kmeans(('tpips', set_, aspect, k), rows, k, set_)

    def clay_parts(self, set_, aspect, k, text_set):
        rows = lambda: self._x(('clay', set_, aspect, text_set), lambda: om_data.clay(set_, self._held(set_), aspect, text_set))
        return self._kmeans(('clay', set_, aspect, text_set, k), rows, k, set_)

    def turtle_plain(self, set_, enc, k):
        return self._turtle(set_, enc, 'plain', k)

    def turtle_cstar(self, set_, enc, aspect, k):
        return self._turtle(set_, enc, 'cstar', k, aspect)

    def _turtle(self, set_, enc, arm, k, aspect=None):
        try:
            return om_data.turtle_parts(set_, enc, arm, k, aspect)
        except FileNotFoundError as e:
            raise MissingInput(f'TURTLE partitions {set_}/{enc}/{arm}/{aspect}/k{k}: {e}') from e

    def ap(self, arm, set_, enc, aspect, pick, text_set=None):
        """`ns_ret.arm_ap` of one arm on the aspect's labelled rows: c* (seeds 0 to 4, unit rows), TPIPS and CLAY (unseeded)."""
        key = (arm, set_, enc, aspect, pick, text_set)
        if key in self._ap_cache:
            return self._ap_cache[key]
        held, y = self._held(set_), self.labels(set_, aspect).astype(np.int64)
        if arm == 'cstar':
            def get(split):
                return lambda s: om_data.cstar(set_, split, enc, aspect, s)
            seeds = SEEDS
        elif arm == 'tpips':
            def get(split):
                return lambda s: self._x(('tpips', set_, aspect, split), lambda: om_data.tpips(set_, split, aspect))
            seeds = None
        else:
            assert arm == 'clay', arm

            def get(split):
                return lambda s: self._x(('clay', set_, aspect, text_set, split), lambda: om_data.clay(set_, split, aspect, text_set))
            seeds = None
        res = ns_ret.arm_ap(get(op.DEV_SPLIT), get(held), pick, y, seeds)
        self._ap_cache[key] = res
        return res


# ================================================================================================ scoring
def default_draws(task):
    return (op.SMOKE_DRAWS if op.SMOKE else op.N_DRAWS)[task]


def _numbers(res, extra):
    return dict(cand=res['cand'], comp=res['comp'], d=res['d'], lo=res['lo'], hi=res['hi'], n_draws=res['n_draws'], **extra)


def _blank(spec, why):
    return dict(spec, cand=None, comp=None, d=None, lo=None, hi=None, n_draws=None, n_rows=None, missing=why)


def _clay_text_set(picks, kind, set_):
    return _picks_get(picks, f'clay_textset/{kind}/{set_}')['set']


def _clu_parts(sc, ld, set_, enc, aspect, k, picks):
    """(candidate, comparator, extra fields) of one clustering group."""
    extra = {}
    if sc in ('SC1', 'SC3', 'SC6'):
        cand = ld.cstar_parts(set_, enc, aspect, k)
    else:
        cand = repeat_states(ld.turtle_cstar(set_, enc, aspect, k))
    if sc == 'SC1':
        comp = ld.tpips_parts(set_, aspect, k)
    elif sc == 'SC3':
        ts = _clay_text_set(picks, 'clu', set_)
        check_pin(picks, ts, aspect)
        comp = ld.clay_parts(set_, aspect, k, ts)
        extra['text_set'] = ts
    else:
        comp = ld.turtle_plain(set_, enc, k)
    return cand, comp, extra


def _score_clu(sc, set_, enc, ld, picks, draws):
    out = []
    for aspect in ASPECTS:
        y = ld.labels(set_, aspect)
        for k, metrics in s1.class_ks(set_, aspect):
            group = [_cell(sc, set_, enc, aspect, m, k) for m in metrics]
            try:
                cand, comp, extra = _clu_parts(sc, ld, set_, enc, aspect, k, picks)
            except (MissingInput, FileNotFoundError) as e:
                out.extend(_blank(c, f'{type(e).__name__}: {e}') for c in group)
                continue
            res = ns_clu.paired_cell(cand, comp, y, k, r=None, metrics=metrics, draws=draws, rng_seed=op.BOOT_SEED)
            for c in group:
                out.append(dict(c, **_numbers(res[c['metric']], dict(extra, n_rows=int(len(y)), n_labelled=int((y >= 0).sum()),
                                                                     missing=None))))
    return out


def _score_ret(sc, set_, enc, ld, picks, draws):
    out = []
    for aspect in ASPECTS:
        spec = _cell(sc, set_, enc, aspect, 'map', None)
        y = ld.labels(set_, aspect)
        try:
            ts = None
            if sc == 'SC2':
                comp_arm, comp_pick = 'tpips', _picks_get(picks, f'tpips/{set_}')['pick']
            else:
                comp_arm, rec = 'clay', _picks_get(picks, f'clay/{set_}')
                comp_pick, ts = rec['pick'], _clay_text_set(picks, 'ret', set_)
                assert rec['text_set'] == ts, f'text set of clay/{set_} ({rec["text_set"]}) differs from clay_textset/ret ({ts})'
                check_pin(picks, ts, aspect)
            cand_pick = ns_ret.reused_pick('cstar', set_, enc)
            cand = ld.ap('cstar', set_, enc, aspect, cand_pick)
            comp = ld.ap(comp_arm, set_, enc, aspect, comp_pick, ts)
        except (MissingInput, FileNotFoundError) as e:
            out.append(_blank(spec, f'{type(e).__name__}: {e}'))
            continue
        assert np.array_equal(cand['rows'], comp['rows']), f'{spec["id"]}: the arms were scored on other queries'
        p = ns_ret.paired_ret(cand['ap'], comp['ap'], r=None, draws=draws, comp_seeds='none')
        extra = dict(n_rows=int(len(y)), n_queries=int(len(cand['rows'])), candidate_readout=cand_pick,
                     comparator_readout=comp_pick, missing=None)
        if ts is not None:
            extra['text_set'] = ts
        out.append(dict(spec, **_numbers(p, extra)))
    return out


def score_unit(sc, set_, enc, ld=None, picks=None, draws=None):
    """The raw cells (numbers, no r, no label) of one unit (SC, set, encoder), in `cell_specs` order. A part that does not
    exist gives cells with numbers None and `missing` (void at assembly); a changed text set pin or other query rows crash."""
    task, encs, _, _ = SC_DEFS[sc]
    assert set_ in SETS and enc in encs, (sc, set_, enc)
    ld = ld or Loaders()
    picks = picks if picks is not None else op.read_committed_record('stage1_om_picks.json')
    draws = draws or default_draws(task)
    raw = (_score_clu if task == 'clu' else _score_ret)(sc, set_, enc, ld, picks, draws)
    want = [c['id'] for c in cell_specs() if (c['sc'], c['set'], c['encoder']) == (sc, set_, enc)]
    assert [c['id'] for c in raw] == want, 'cells out of the expected order'
    return raw


# ================================================================================================ labels, voids, sentences
def label_of(lo, r):
    """§3.4: WIN when lo > 0; LEVEL when not WIN and lo > -r; BELOW otherwise. r None: `inconclusive` unless WIN."""
    word = ns_outcome.word_of(lo, r)
    return INCONCLUSIVE if word == ns_outcome.NOT_WIN else word


def check_status(status):
    """Our `records/status.json`: gates exactly i, ii, iii_a, iii_b, iii_c, iv; `stage1` {key: s} (a pick or range key;
    absent keys are decided from the records); `r17` {path, sha256}. Values ok | failed | missing."""
    assert isinstance(status, dict) and set(status) == {'gates', 'stage1', 'r17'}, \
        f'status keys {sorted(status) if isinstance(status, dict) else status!r}'
    assert set(status['gates']) == set(OUR_GATES), f'status gates {sorted(status["gates"])}, need {sorted(OUR_GATES)}'
    for name, v in {**status['gates'], **status['stage1']}.items():
        assert v in STATUS_VALUES, f'status {name}: {v!r} is not one of {STATUS_VALUES}'
    return status


def load_r17_status(status):
    """r17's status.json named by our status ({path, sha256}); refuses a file whose hash differs."""
    ref = status['r17']
    path = Path(ref['path'])
    got = op.sha256_file(path)
    assert got == ref['sha256'], f'{path}: sha256 {got} differs from records/status.json ({ref["sha256"]})'
    return ns_outcome.check_status(json.loads(path.read_text()))


def _stage1_state(status, picks, key):
    if key in status['stage1']:
        return status['stage1'][key]
    return OK if key in picks else 'missing'


def void_reasons(cell, status, picks, r17):
    """Why `cell` is void: our failed gates, a missing stage 1 pick or text set, r17's failed parts (§3.5's table)."""
    out = []
    for gate, scs in GATE_VOIDS.items():
        if cell['sc'] in scs and status['gates'][gate] != OK:
            out.append(f'gate {gate} {status["gates"][gate]}')
    s = cell['set']
    items = {'SC2': (f'tpips/{s}',), 'SC4': (f'clay/{s}', f'clay_textset/ret/{s}'), 'SC3': (f'clay_textset/clu/{s}',)}
    for key in items.get(cell['sc'], ()):
        st = _stage1_state(status, picks, key)
        if st != OK:
            out.append(f'stage 1 {key} {st}')
    g = r17['gates']
    rules = []
    if cell['sc'] in R17_SCORER_CLU:
        rules.append(('r17 i_a', g['i_a']))
        rules.append((f'r17 iii_metric[{cell["metric"]}]', g['iii_metric'].get(cell['metric'])))
        if is_class_count(cell):
            rules.append(('r17 i_d', g['i_d']))
            rules.append((f'r17 iii_count[{s}/{cell["aspect"]}]', g['iii_count'].get(f'{s}/{cell["aspect"]}')))
    if cell['sc'] in R17_SCORER_RET:
        rules.append(('r17 ii_scorer', g['ii_scorer']))
    out.extend(f'{name} {v}' for name, v in rules if v != OK)
    return out


def r_of(cell, ranges, r17, family):
    """(r, source) of a cell from the stage 1 range record; None when the range is missing, absent or (class count) r17's own
    range failed in its status."""
    key = range_key(cell)
    entry = ranges.get(key)
    if entry is None:
        return None, f'missing:{key}'
    if family.get(key) == 'r17':
        rkey = f"cstar/{cell['set']}/{cell['encoder']}/{cell['aspect']}/{cell['metric']}"
        st = r17['stage1']['ranges'].get(rkey, 'missing')    # r17's rule: an absent key counts as missing
        if st != OK:
            return None, f'missing:{key}: r17 range {rkey} {st}'
    if entry['r'] is None:
        return None, entry.get('source') or f'missing:{key}'
    r = float(entry['r'])
    assert math.isfinite(r) and r >= 0, (key, r)
    return r, entry['source']


ARM_NAME = {'tpips': 'TPIPS', 'clay': 'CLAY'}
TURTLE_WORDS = "its released code's default settings, without its hyperparameter selection"


def sentence_of(cell, label):
    """§6's licensed sentence of one cell: WIN 'better than', LEVEL 'level with'; BELOW only when the upper bound is below 0
    (then the comparator is the better one); else None."""
    if label not in (WIN, LEVEL, BELOW):
        return None
    if label == BELOW and not cell['hi'] < 0:
        return None
    S, X = ns_outcome.SET_NAME[cell['set']], op.ENCODER_NAME[cell['encoder']]
    a, sc, comp = cell['aspect'], cell['sc'], cell['comparator']
    ret = cell['task'] == 'ret'
    verb = 'retrieves' if ret else 'clusters'
    tail = ' (mAP)' if ret else f" ({ns_outcome.METRIC_NAME[cell['metric']]}, {cell['k']} clusters)"
    word = {WIN: 'better than', LEVEL: 'level with'}.get(label)
    if sc == 'SC5':
        pre = f'On {S}, with TURTLE ({TURTLE_WORDS}) on both sides, '
        plain = f'the plain {X} image'
        if label == BELOW:
            return f'{pre}{plain} {verb} paintings by {a} better than c* on {X}{tail}.'
        return f'{pre}c* on {X} {verb} paintings by {a} {word} {plain}{tail}.'
    if sc == 'SC6':
        plain = f'the plain {X} image clustered with TURTLE at {TURTLE_WORDS}'
        if label == BELOW:
            return f'On {S}, {plain} clusters paintings by {a} better than c* on {X}{tail}.'
        return f'On {S}, c* on {X} clusters paintings by {a} {word} {plain}{tail}.'
    arm = ARM_NAME[comp]
    if label == BELOW:
        return f'On {S}, {arm} {verb} paintings by {a} better than c* on {X}{tail}.'
    return f'On {S}, c* on {X} {verb} paintings by {a} {word} {arm}{tail}.'


def assemble(raw, ranges, picks, status, r17, expected=None):
    """r17's cell form plus label, void, sentence, for every expected cell (default all 337), in order. `raw`: score_unit's
    cells (any subset; an id outside `expected` or twice is refused); a cell not scored is void ('not scored')."""
    expected = cell_specs() if expected is None else expected
    check_status(status)
    by = {}
    for c in raw:
        assert c['id'] not in by, f'duplicate cell {c["id"]}'
        by[c['id']] = c
    ids = {c['id'] for c in expected}
    stray = sorted(set(by) - ids)
    assert not stray, f'cells outside the list: {stray[:3]}'
    family = {k['key']: k['source'] for k in s1.range_keys()}
    out = []
    for spec in expected:
        c = dict(by.get(spec['id']) or _blank(spec, 'not scored'))
        for key in ('n_draws', 'n_rows', 'missing'):
            c.setdefault(key, None)
        c['r'], c['r_source'] = r_of(spec, ranges, r17, family)
        void = void_reasons(spec, status, picks, r17)
        if c.get('missing'):
            void.append(f'missing input: {c["missing"]}' if c['missing'] != 'not scored' else 'not scored')
        c['void'] = '; '.join(void) if void else None
        if void:
            c['label'] = None
        else:
            assert all(c[k] is not None for k in ('d', 'lo', 'hi')), spec['id']
            c['label'] = label_of(c['lo'], c['r'])
        c['sentence'] = None if void else sentence_of(c, c['label'])
        out.append(c)
    return out


# ================================================================================================ the C13 guard
def pick_summary(picks):
    """The picks a C13 file must name, flat {key: str}: TPIPS's and CLAY's read-outs, CLAY's text sets, per set."""
    out = {}
    for s in SETS:
        for key, field in ((f'tpips/{s}', 'pick'), (f'clay/{s}', 'pick'), (f'clay/{s}/text_set', None),
                           (f'clay_textset/clu/{s}', 'set'), (f'clay_textset/ret/{s}', 'set')):
            if field is None:
                rec = picks.get(f'clay/{s}')
                if rec is not None:
                    out[key] = rec['text_set']
            elif key in picks:
                out[key] = picks[key][field]
    return out


C13_NUMBERS = ('cand', 'comp', 'd', 'lo', 'hi')
C13_READOUTS = ('candidate_readout', 'comparator_readout')


def c13_disagreements(cells, picks, c13, tol=op.TOL_C13):
    """What differs between our cells/picks and the C13 file: [] when they agree on every pick, value, d, lo, hi and label
    within `tol` (readouts of retrieval cells by name). Missing and extra ids count."""
    bad = []
    want = pick_summary(picks)
    got = c13.get('picks', {})
    for key in sorted(set(want) | set(got)):
        if want.get(key) != got.get(key):
            bad.append(f'pick {key}: ours {want.get(key)!r}, C13 {got.get(key)!r}')
    theirs = {}
    for c in c13.get('cells', []):
        assert c['id'] not in theirs, f'C13 file has {c["id"]} twice'
        theirs[c['id']] = c
    ours = {c['id']: c for c in cells}
    bad += [f'cell {i}: missing from the C13 file' for i in ours if i not in theirs]
    bad += [f'cell {i}: in the C13 file, not in ours' for i in theirs if i not in ours]
    for i, c in ours.items():
        t = theirs.get(i)
        if t is None:
            continue
        if c.get('label') != t.get('label'):
            bad.append(f'cell {i}: label ours {c.get("label")!r}, C13 {t.get("label")!r}')
        for f in C13_NUMBERS:
            a, b = c.get(f), t.get(f)
            if a is None or b is None:
                if a is not b:
                    bad.append(f'cell {i}: {f} ours {a!r}, C13 {b!r}')
            elif not abs(a - b) <= tol:
                bad.append(f'cell {i}: {f} ours {a!r}, C13 {b!r}, difference {abs(a - b):.3g} > {tol}')
        for f in C13_READOUTS:
            if f in c and c[f] != t.get(f):
                bad.append(f'cell {i}: {f} ours {c[f]!r}, C13 {t.get(f)!r}')
    return bad


def labels_path():
    return op.record_path('labels.json')


def run_labels(cells_path, c13_path, out_path, picks):
    """The guarded step: refuses (SystemExit) unless the C13 file exists and agrees; then writes the labels record."""
    cells_path, c13_path = Path(cells_path), Path(c13_path)
    if not c13_path.exists():
        raise SystemExit(f'labels refuses: {c13_path} does not exist (C13 must re-derive the cells first)')
    cells = json.loads(cells_path.read_text())
    bad = c13_disagreements(cells, picks, json.loads(c13_path.read_text()))
    if bad:
        raise SystemExit(f'labels refuses: C13 disagrees on {len(bad)} items, first: {bad[:5]}')
    rec = dict(cells_sha256=op.sha256_file(cells_path), c13_sha256=op.sha256_file(c13_path), tol=op.TOL_C13,
               picks=pick_summary(picks), cells=[{k: c.get(k) for k in CELL_KEYS} for c in cells])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    op.write_json(out_path, rec)
    return out_path


# ================================================================================================ CLI
def parts_dir(out=None):
    return Path(op.OUT if out is None else out) / 'stage2' / 'parts'


def cmd_cells(args):
    op.read_committed_record('stage1_om_ranges.json')     # refuses uncommitted stage 1 (§3.5); r is read at assembly
    picks = op.read_committed_record('stage1_om_picks.json')
    ld, todo = Loaders(), [u for u in units() if (args.sc in (None, u[0])) and (args.set in (None, u[1])) and (args.enc in (None, u[2]))]
    for sc, set_, enc in todo:
        path = parts_dir() / f'{sc}__{set_}__{enc}.json'
        if path.exists() and not args.force:
            print(f'skip {path.name} (exists)')
            continue
        raw = score_unit(sc, set_, enc, ld, picks, args.draws)
        op.write_json(path, raw)
        print(f'{path.name}: {len(raw)} cells, {sum(c["missing"] is not None for c in raw)} without numbers')
    return 0


def cmd_assemble(args):
    ranges = op.read_committed_record('stage1_om_ranges.json')
    picks = op.read_committed_record('stage1_om_picks.json')
    status = check_status(op.read_committed_record('status.json'))
    r17 = load_r17_status(status)
    raw = []
    for sc, set_, enc in units():
        path = parts_dir() / f'{sc}__{set_}__{enc}.json'
        if path.exists():
            raw += json.loads(path.read_text())
    cells = assemble(raw, ranges, picks, status, r17)
    dst = Path(op.OUT) / 'stage2' / 'cells.json'
    op.write_json(dst, cells)
    from collections import Counter
    print(f'{len(cells)} cells -> {dst}; labels {dict(Counter(str(c["label"]) for c in cells))}; void {sum(bool(c["void"]) for c in cells)}')
    return 0


def cmd_labels(args):
    picks = op.read_committed_record('stage1_om_picks.json')
    out = run_labels(Path(op.OUT) / 'stage2' / 'cells.json', Path(op.OUT) / 'c13' / 'c13_cells.json', labels_path(), picks)
    print(f'labels -> {out}')
    return 0


def cmd_list(args):
    from collections import Counter
    print(dict(Counter(c['sc'] for c in cell_specs())), len(units()), 'units')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cells')
    p.add_argument('--sc', choices=sorted(SC_DEFS))
    p.add_argument('--set', choices=SETS)
    p.add_argument('--enc', choices=ENCODERS)
    p.add_argument('--draws', type=int, default=None)
    p.add_argument('--force', action='store_true')
    for name in ('assemble', 'labels', 'list'):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    return {'cells': cmd_cells, 'assemble': cmd_assemble, 'labels': cmd_labels, 'list': cmd_list}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
