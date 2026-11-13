"""r29 stage 2, C13 check (spec §3.4 "Order and C13"): every pick, range, SC9 field and test cell that OUR scorers wrote
under records/ against Codex's independent re-derivation under rederive/, tolerance 1e-6 (D23's validation tolerance).
Model: r29_c1_compare.py (stage 1c). Writes records/s2_compare.json.

usage: python r29_s2_compare.py --phase val|held [--pieces e2e3 sc9 mmart photo ablation]
  val pieces:  e2e3 sc9 mmart photo        held pieces: e2e3 mmart photo ablation
  default pieces: every piece of the phase whose input files all exist; a piece with a file missing on either side (or a
  Codex file that says `complete: false`, or no cells) is listed under `missing`, not as a mismatch, and is reported.
  Exit status 1 when any compared piece of this run has a mismatch, else 0. Each run rewrites only its own phase's sections of
  the pieces it ran (the rest of an existing s2_compare.json is kept).

Rules (as r29_c1_compare.py): a number differing by more than 1e-6, a pick, outcome or categorical field that differs, or a
field missing on one side (both files present) goes under `mismatches` with the id, both values and the difference; that
pick, range or cell is inconclusive until the cause is fixed. The script never picks a side and never edits a record.
Codex's outcome of a test cell is recomputed from ITS lo and r by §3.4's rule (WIN: lo > 0; LEVEL: not WIN and lo > -r; BELOW
otherwise) and compared with ours; its own `outcome` field (INCONCLUSIVE where its replay certification failed) and replay /
rebuild flags are recorded as information. Ladder counts (SC4 per encoder, SC5, SC6) are compared as information only.

Key mapping (ours -> Codex). Arms: E2 = SigLIP 2, arm siglip2_prefix, encoder key siglip2; E3 = CLIP L/14, arm clipl14_prefix,
encoder key clip_l14; K* = siglip1_prefix. Aspects emotion, style, genre.
  val e2e3   records/image_picks_<arm>.json                      s2e_val.json encoders.<E>
    ami_ranges.<a>.r                                              picks.ami_range.<a>
    map_ranges.<a>.r                                              picks.image_map.<a>.r
    readouts.cstar.pick  (Q1 side, "told")                        picks.image_map.<a>.pick_q1_side
    readouts.untold.pick (plain side, joined heads)               picks.image_map.<a>.pick_plain_side
    ami_ranges.<a>.per_seed / map_ranges.<a>.per_seed             groups.cells['ami/<a>'] / ['map/<a>'] (5 seed values)
    readouts.{cstar,untold} picked row (map, score, kept)         image_tables.{q1,plain}.{table,scores,kept}[pick]  (info)
    unpaired: ours readouts.{plain,teacher} (comparator picks, copied from older rounds), validation_files, root;
              Codex forward_check, replay_gates, replays, image_tables rows other than the pick
  val sc9    records/image_groups_<arm>.json image.{cells,per_seed,mean}   s2e_val.json encoders.<E>.groups.{cells,per_seed,mean}
    records/encoder_pick.json                                     s2e_val.json sc9
    encoders.siglip1.per_seed                                     kstar_per_seed
    encoders.<enc>.score                                          means.<enc>
    pick / ranking[1] / margin / seeds_ahead / close_call         pick / runner_up / margin / seeds_ahead / close_call
    encoders.<leader>.seed_range                                  leader_range
    encoders.qwen.score (information row)                         qwen_info (info)
    unpaired: ours ranking, tie, sources, role; Codex n_seeds_ahead (info: list length)
  val mmart  records/mmart_picks_siglip1_prefix.json             s2mp_val.json mmart
    ami_ranges.<a>.r / .per_seed                                  ranges.<a>.ami.r / .per_seed
    map_ranges.<a>.r / .per_seed                                  ranges.<a>.map.r / .per_seed
    readouts.untold.pick (plain side, P1)                         picks.p1.pick
    readouts.cstar.pick  (Q1 side)                                picks.q1.pick
    readouts.{untold,cstar}.table[ro].{map,score,kept}            picks.{p1,q1}.table[ro].{map,score,kept}  (info)
    unpaired: ours readouts.{plain,teacher} (older rounds' comparator picks), n_val, validation_files;
              Codex picks.*.units, ranges.*.n, n_labelled
  val photo  records/photo_picks_siglip1_prefix.json             s2mp_val.json photo
    sets.<d>.ami_range.r / .per_seed                              ranges.<d>.ami.r / .per_seed
    sets.<d>.r_map (affection: own R0 range; emoset: r14's r_E)   ranges.<d>.map.r
    sets.affection.map_range.per_seed                             ranges.affection.map.per_seed
    r_E.value                                                     ranges.emoset.map.r
    sets.<d>.readout.pick                                         picks.<d>.pick
    sets.<d>.readout.val_scores[ro]                               picks.<d>.table[ro].map.emotion  (info)
    unpaired: ours comparator_picks, f1, val_files; Codex ranges.emoset.map.own_r0_range / per_seed / r_source
  held e2e3  records/image_cells_<arm>.json cells                s2e_held.json encoders.<E>.cells
    cell id artelingo/<enc>/<arm>/<a>/<ami<k>|map>/<plain|teacher>   <E>/<ami|map>/<a>/<plain|q1>      (teacher = Q1)
    ladder (info): records/SC4_<E>_<arm>.json                     encoders.<E>.sc4
  held mmart records/mmart_cells_siglip1_prefix.json cells       s2mp_held.json mmart.cells
    mmart/siglip1/<arm>/<a>/<ami<k>|map>/<plain|teacher>            mmart/<ami|map>/<a>/<p1|q1>         (plain = P1)
    ladder (info): records/SC5_siglip1_prefix.json                mmart.sc5
  held photo records/photo_cells_siglip1_prefix.json cells       s2mp_held.json photo.cells
    <set>/siglip1/<arm>/emotion/<ami<k>|map>/<plain|teacher>        <affection|emoset>/<ami|map>/<p1|q1>
    ladder (info): records/SC6_siglip1_prefix.json                photo.sc6
  held ablation records/ablation.json cells                      s2abl_held.json cells
    artelingo/siglip1/<row>/<a>/<ami<k>|map>/<K*>                   ablation/<row>/<a>/<ami|map>
    value (removal mean) / comparator_value (K* mean)             removal_mean / kstar_mean
    arms.<row>.<a>.<m>_per_seed / arms.<K*>.<a>.<m>_per_seed      removal_per_seed / kstar_per_seed
    d, lo, hi                                                     d, lo, hi
    cell.per_seed (removal minus K*) (info)                       d_per_seed;   n_rows / arms n_queries, n_draws (info)   n, n_draws
    unpaired: ours r, outcome (None: SC7 has no outcome); Codex kstar_check (info, recorded)
  every cell compared: value, comparator_value, d, lo, hi, r (1e-6); outcome (ours vs the rule on Codex's lo and r, and
  ours vs the rule on our own lo and r); n and n_draws as information.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REC = HERE / 'records'
RD = HERE / 'rederive'
TOL = 1e-6
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
ENCODERS = {'siglip2': 'siglip2_prefix', 'clip_l14': 'clipl14_prefix'}     # E2, E3: encoder key -> arm
KSTAR = 'siglip1_prefix'
NUM_KEYS = ('value', 'comparator_value', 'd', 'lo', 'hi', 'r')
VAL_PIECES = ('e2e3', 'sc9', 'mmart', 'photo')
HELD_PIECES = ('e2e3', 'mmart', 'photo', 'ablation')
PHASE_PIECES = {'val': VAL_PIECES, 'held': HELD_PIECES}
ROWS = ('instructed_graphs', 'content_correction', 'buddy', 'relational')
MISSING = object()


def rule(lo, r):
    return 'WIN' if lo > 0 else ('LEVEL' if lo > -r else 'BELOW')


def load(p):
    return json.loads(Path(p).read_text())


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M')


def dig(d, *path):
    """d[path[0]][path[1]]..., MISSING when any step is absent (or not a container)."""
    for k in path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, (list, tuple)) and isinstance(k, int) and -len(d) <= k < len(d):
            d = d[k]
        else:
            return MISSING
    return d


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def seed_values(x):
    """Five per-seed values from a list or from a {'0': v, ...} dict (seed order); MISSING stays MISSING."""
    if isinstance(x, dict):
        try:
            return [x[str(s)] for s in SEEDS] if all(str(s) in x for s in SEEDS) else MISSING
        except (KeyError, TypeError):
            return MISSING
    return x


class Missing(Exception):
    """The piece cannot be compared: a side is absent, incomplete or pending (listed under `missing`, never a mismatch)."""

    def __init__(self, reason, codex_status=None):
        super().__init__(reason)
        self.reason, self.codex_status = reason, codex_status or {}


class Check:
    """Collector of one piece: numeric and categorical comparisons, information rows, unpaired fields."""

    def __init__(self):
        self.items, self.mismatches, self.unpaired, self.info = [], [], [], []
        self.max_diff = 0.0
        self.n = 0
        self.cells = []

    def _miss(self, id_, ours, codex):
        side = [n for n, v in (('ours', ours), ('codex', codex)) if v is MISSING]
        self.mismatches.append(dict(id=id_, ours=None if ours is MISSING else ours, codex=None if codex is MISSING else codex,
                                    diff=None, note='field missing on ' + ' and '.join(side)))
        self.items.append(dict(item=id_, ours=None if ours is MISSING else ours, codex=None if codex is MISSING else codex,
                               diff=None, missing_side=side))

    def num(self, id_, ours, codex):
        self.n += 1
        if ours is MISSING or codex is MISSING:
            return self._miss(id_, ours, codex)
        if not (is_num(ours) and is_num(codex)):
            self.mismatches.append(dict(id=id_, ours=ours, codex=codex, diff=None, note='not a number on at least one side'))
            self.items.append(dict(item=id_, ours=ours, codex=codex, diff=None))
            return None
        diff = abs(ours - codex)
        self.items.append(dict(item=id_, ours=ours, codex=codex, diff=diff))
        if diff == diff:
            self.max_diff = max(self.max_diff, diff)
        if not diff <= TOL:
            self.mismatches.append(dict(id=id_, ours=ours, codex=codex, diff=diff, note=f'diff {diff:.3e} > {TOL:g}'))
        return diff

    def nums(self, id_, ours, codex):
        if ours is MISSING or codex is MISSING:
            return self._miss(id_, ours, codex)
        if not (isinstance(ours, (list, tuple)) and isinstance(codex, (list, tuple))) or len(ours) != len(codex):
            self.n += 1
            self.mismatches.append(dict(id=id_, ours=ours, codex=codex, diff=None, note='lists differ in shape or length'))
            self.items.append(dict(item=id_, ours=ours, codex=codex, diff=None))
            return None
        for i, (o, c) in enumerate(zip(ours, codex)):
            self.num(f'{id_}[{i}]', o, c)

    def same(self, id_, ours, codex):
        self.n += 1
        if ours is MISSING or codex is MISSING:
            return self._miss(id_, ours, codex)
        equal = ours == codex
        self.items.append(dict(item=id_, ours=ours, codex=codex, equal=equal))
        if not equal:
            self.mismatches.append(dict(id=id_, ours=ours, codex=codex, diff=None, note='differs'))

    def note(self, id_, ours, codex, tol=TOL):
        """Information only: recorded with its difference, never a mismatch."""
        row = dict(item=id_, ours=None if ours is MISSING else ours, codex=None if codex is MISSING else codex)
        if ours is MISSING or codex is MISSING:
            row['equal'] = None
        elif is_num(ours) and is_num(codex) and not isinstance(ours, bool):
            row['diff'] = abs(ours - codex)
            row['equal'] = row['diff'] <= tol
        else:
            row['equal'] = ours == codex
        self.info.append(row)

    def unpair(self, side, field, why=''):
        self.unpaired.append(dict(side=side, field=field, note=why))


def result(chk, extra_files, codex_status, extra=None):
    out = dict(status='mismatch' if chk.mismatches else 'pass', n_compared=chk.n,
               max_abs_diff=chk.max_diff if chk.n else None, mismatches=chk.mismatches, unpaired=chk.unpaired, info=chk.info,
               codex_status=codex_status, items=chk.items)
    if chk.cells:
        out['cells'] = chk.cells
        out['max_abs_diff_cells'] = max((c['max_diff'] for c in chk.cells if c.get('max_diff') is not None), default=None)
    out['extra_files'] = [str(f) for f in extra_files]
    out.update(extra or {})
    return out


# ------------------------------------------------------------------------------------------------ shared pieces
def codex_complete(doc, what):
    """Codex says `complete: false` (or names pending inputs): the piece is missing."""
    if doc.get('complete') is False or doc.get('pending'):
        raise Missing(f'{what}: Codex file says complete {doc.get("complete")}, pending {doc.get("pending")!r}',
                      dict(complete=doc.get('complete'), pending=doc.get('pending'), mode=doc.get('mode', doc.get('kind'))))


def definition_notes(doc, words=('ahead', 'close call', 'close_call', 'margin', 'seeds')):
    """Codex's notes that define SC9's terms (recorded, so a different definition is visible)."""
    notes = doc.get('notes', [])
    notes = notes if isinstance(notes, list) else [notes]
    return [n for n in notes if isinstance(n, str) and any(w in n.lower() for w in words)]


def compare_cell(chk, cid, o, c):
    """One test cell: value, comparator value, d, lo, hi, r at TOL, the outcome by §3.4's rule, information fields."""
    row = dict(cell=cid, ours_id=o.get('id'), codex_id=c.get('id'), ours_outcome=o.get('outcome'),
               codex_outcome_field=c.get('outcome'))
    o_none = all(o.get(k) is None for k in NUM_KEYS if k != 'r')
    c_none = all(c.get(k) is None for k in NUM_KEYS if k != 'r')
    if o_none and c_none:                                   # both sides parked: nothing to compare
        row.update(parked=True, ours_reason=o.get('reason'), codex_reason=c.get('reason'))
        chk.cells.append(row)
        return
    if o_none != c_none:
        side = 'ours' if o_none else 'codex'
        chk.n += 1
        chk.mismatches.append(dict(id=f'cell {cid}', ours=o.get('outcome'), codex=c.get('outcome'), diff=None,
                                   note=f'{side} cell has no numbers (outcome {(o if o_none else c).get("outcome")!r}, '
                                        f'reason {(o if o_none else c).get("reason")!r})'))
        row.update(parked=side)
        chk.cells.append(row)
        return
    diffs = {}
    for k in NUM_KEYS:
        d = chk.num(f'cell {cid} {k}', o.get(k, MISSING), c.get(k, MISSING))
        if d is not None:
            diffs[k] = d
    row['diffs'] = diffs
    row['max_diff'] = max(diffs.values()) if diffs else None
    if all(is_num(c.get(k)) for k in ('lo', 'r')) and all(is_num(o.get(k)) for k in ('lo', 'r')):
        row['codex_outcome_rule'] = rule(c['lo'], c['r'])
        row['ours_outcome_rule'] = rule(o['lo'], o['r'])
        chk.n += 2
        if o.get('outcome') != row['codex_outcome_rule']:
            chk.mismatches.append(dict(id=f'cell {cid} outcome', ours=o.get('outcome'), codex=row['codex_outcome_rule'], diff=None,
                                       note='ours vs the §3.4 rule on Codex\'s lo and r'))
        if o.get('outcome') != row['ours_outcome_rule']:
            chk.mismatches.append(dict(id=f'cell {cid} outcome (ours)', ours=o.get('outcome'), codex=row['ours_outcome_rule'],
                                       diff=None, note='our outcome is not the rule word for our own lo and r'))
    else:
        chk.n += 1
        chk.mismatches.append(dict(id=f'cell {cid} outcome', ours=o.get('outcome'), codex=c.get('outcome'), diff=None,
                                   note='lo or r missing: the outcome cannot be recomputed'))
    chk.note(f'cell {cid} n', o.get('n_rows'), c.get('n'))
    chk.note(f'cell {cid} n_draws', o.get('n_draws'), c.get('n_draws'))
    for k in ('reason', 'comparator_rebuild', 'scoring_check', 'candidate_readout', 'comparator_readout'):
        if k in c:
            row['codex_' + k] = c[k]
    chk.cells.append(row)


def compare_cells(chk, ours_cells, codex_cells, expected, label):
    """ours_cells, codex_cells: key -> cell dict; expected: the keys both must carry. Absent cells are mismatches (both files
    exist); keys outside `expected` are listed as unpaired."""
    for key in expected:
        cid = '/'.join(key)
        o, c = ours_cells.get(key), codex_cells.get(key)
        if o is None or c is None:
            chk.n += 1
            chk.mismatches.append(dict(id=f'cell {cid}', ours=None if o is None else o.get('outcome'),
                                       codex=None if c is None else c.get('outcome'), diff=None,
                                       note=f'{label}: no cell of ' + ('ours' if o is None else 'Codex')))
            continue
        compare_cell(chk, cid, o, c)
    for side, cells in (('ours', ours_cells), ('codex', codex_cells)):
        for key in cells:
            if key not in expected:
                chk.unpair(side, '/'.join(key), f'{label}: cell outside the expected {len(expected)}')


def ladder_info(chk, name, ours, codex):
    """Ladder counts, information only. ours: a records/SC*.json document or None; codex: its sc4/sc5/sc6 dict or None."""
    if ours is None or codex in (None, MISSING):
        chk.info.append(dict(item=f'ladder/{name}', ours='absent' if ours is None else 'present',
                             codex='absent' if codex in (None, MISSING) else 'present', equal=None))
        return
    c = ours.get('counts', {})
    pairs = [('level', ours.get('level'), codex.get('level')),
             ('p1_wins', c.get('p1_win'), codex.get('wins_plain', codex.get('wins_p1'))),
             ('q1_wins_or_level', c.get('q1_win_or_level'), codex.get('wins_or_level_q1')),
             ('inconclusive_above', ours.get('inconclusive_above'), codex.get('inconclusive_above'))]
    if 'inconclusive' in codex:
        pairs.append(('inconclusive_total', (c.get('p1_inconclusive') or 0) + (c.get('q1_inconclusive') or 0), codex['inconclusive']))
    else:
        pairs += [('p1_inconclusive', c.get('p1_inconclusive'), codex.get('inconclusive_p1')),
                  ('q1_inconclusive', c.get('q1_inconclusive'), codex.get('inconclusive_q1'))]
    for k, o, x in pairs:
        chk.note(f'ladder/{name}/{k}', MISSING if o is None else o, MISSING if x is None else x)


def optional(path):
    return load(path) if Path(path).exists() else None


# ------------------------------------------------------------------------------------------------ val pieces
def files_val_e2e3(rec, rd):
    return [rec / f'image_picks_{arm}.json' for arm in ENCODERS.values()] + [rd / 's2e_val.json']


def val_e2e3(rec, rd):
    cx = load(rd / 's2e_val.json')
    codex_complete(cx, 's2e_val.json')
    status = dict(complete=cx.get('complete'), kind=cx.get('kind'), replay_gates=cx.get('replay_gates'),
                  forward_check_pass={e: {a: dig(cx, 'forward_check', e, a, 'pass') for a in ASPECTS} for e in ENCODERS
                                      if dig(cx, 'forward_check', e) is not MISSING})
    chk = Check()
    for E, arm in ENCODERS.items():
        ours, x = load(rec / f'image_picks_{arm}.json'), dig(cx, 'encoders', E)
        if x is MISSING:
            chk.n += 1
            chk.mismatches.append(dict(id=f'{E}', ours='present', codex=None, diff=None, note='no encoders.' + E + ' in s2e_val.json'))
            continue
        chk.same(f'{E}/ours_arm', dig(ours, 'arm'), arm)
        chk.same(f'{E}/ours_encoder', dig(ours, 'encoder'), E)
        for a in ASPECTS:
            chk.num(f'{E}/range/ami/{a}', dig(ours, 'ami_ranges', a, 'r'), dig(x, 'picks', 'ami_range', a))
            chk.num(f'{E}/range/image_map/{a}', dig(ours, 'map_ranges', a, 'r'), dig(x, 'picks', 'image_map', a, 'r'))
            chk.same(f'{E}/pick/image_map_q1_side/{a}', dig(ours, 'readouts', 'cstar', 'pick'),
                     dig(x, 'picks', 'image_map', a, 'pick_q1_side'))
            chk.same(f'{E}/pick/image_map_plain_side/{a}', dig(ours, 'readouts', 'untold', 'pick'),
                     dig(x, 'picks', 'image_map', a, 'pick_plain_side'))
            for m, key in (('ami', 'ami_ranges'), ('map', 'map_ranges')):
                o = dig(ours, key, a, 'per_seed')
                chk.nums(f'{E}/per_seed/{m}/{a}', o if o is MISSING else seed_values(o), dig(x, 'groups', 'cells', f'{m}/{a}'))
        for side, rd_key in (('q1', 'cstar'), ('plain', 'untold')):
            pick = dig(ours, 'readouts', rd_key, 'pick')
            row = dig(ours, 'readouts', rd_key)             # the picked row's map, score, kept sit on the read-out itself
            ctab = dig(x, 'image_tables', side)
            if pick is MISSING or ctab is MISSING or dig(ctab, 'pick') != pick:
                chk.info.append(dict(item=f'{E}/table/{side}', ours=None if pick is MISSING else pick, codex=dig(ctab, 'pick') if ctab is not MISSING else None,
                                     equal=None, note='picks differ or a side is absent: picked rows not compared'))
                continue
            for a in ASPECTS:
                chk.note(f'{E}/table/{side}/{pick}/map/{a}', dig(row, 'map', a), dig(ctab, 'table', pick, a))
            chk.note(f'{E}/table/{side}/{pick}/score', dig(row, 'score'), dig(ctab, 'scores', pick))
            chk.note(f'{E}/table/{side}/{pick}/kept', dig(row, 'kept'), dig(ctab, 'kept', pick))
        chk.unpair('ours', f'{E}: readouts.plain, readouts.teacher, validation_files, root', 'comparator picks copied from older rounds; provenance')
        chk.unpair('codex', f'{E}: forward_check, replay_gates, replays, image_tables (other rows)', 'Codex certification')
    return result(chk, [], status)


def files_val_sc9(rec, rd):
    return [rec / f'image_groups_{arm}.json' for arm in ENCODERS.values()] + [rec / 'encoder_pick.json', rd / 's2e_val.json']


def val_sc9(rec, rd):
    cx = load(rd / 's2e_val.json')
    codex_complete(cx, 's2e_val.json')
    if 'sc9' not in cx:
        raise Missing('s2e_val.json has no `sc9` (Codex wrote it for the full selection only)',
                      dict(complete=cx.get('complete'), kind=cx.get('kind')))
    status = dict(complete=cx.get('complete'), kind=cx.get('kind'), definition_notes=definition_notes(cx))
    chk = Check()
    for E, arm in ENCODERS.items():
        g = dig(load(rec / f'image_groups_{arm}.json'), 'image')
        x = dig(cx, 'encoders', E, 'groups')
        for cell in [f'{m}/{a}' for a in ASPECTS for m in ('ami', 'map')]:
            chk.nums(f'{E}/group/cell/{cell}', dig(g, 'cells', cell), dig(x, 'cells', cell))
        chk.nums(f'{E}/group/per_seed', dig(g, 'per_seed'), dig(x, 'per_seed'))
        chk.num(f'{E}/group/mean', dig(g, 'mean'), dig(x, 'mean'))
    ep, s = load(rec / 'encoder_pick.json'), cx['sc9']
    enc = dig(ep, 'encoders')
    chk.nums('sc9/kstar_per_seed', dig(enc, 'siglip1', 'per_seed'), dig(s, 'kstar_per_seed'))
    for e in ('siglip1', *ENCODERS):
        chk.num(f'sc9/mean/{e}', dig(enc, e, 'score'), dig(s, 'means', e))
    chk.same('sc9/pick', dig(ep, 'pick'), dig(s, 'pick'))
    chk.same('sc9/runner_up', dig(ep, 'ranking', 1), dig(s, 'runner_up'))
    chk.num('sc9/margin', dig(ep, 'margin'), dig(s, 'margin'))
    chk.same('sc9/seeds_ahead', dig(ep, 'seeds_ahead'), dig(s, 'seeds_ahead'))
    leader = dig(ep, 'pick')
    chk.num('sc9/leader_range', dig(enc, leader, 'seed_range') if leader is not MISSING else MISSING, dig(s, 'leader_range'))
    chk.same('sc9/close_call', dig(ep, 'close_call'), dig(s, 'close_call'))
    chk.note('sc9/qwen_info', dig(enc, 'qwen', 'score'), dig(s, 'qwen_info'))
    nahead = dig(ep, 'seeds_ahead')
    chk.note('sc9/n_seeds_ahead', len(nahead) if isinstance(nahead, list) else MISSING, dig(s, 'n_seeds_ahead'))
    chk.unpair('ours', 'encoder_pick: ranking, tie, sources, role, written', 'ranking[1] is read as runner_up')
    chk.unpair('codex', 'sc9.n_seeds_ahead; forward_check, replays', 'information')
    return result(chk, [], status)


def piece_of(doc, name):
    p = doc.get(name)
    if not isinstance(p, dict):
        raise Missing(f'no `{name}` in the Codex file', dict(mode=doc.get('mode')))
    return p


def tables(chk, E, ours_ro, codex_pick_doc, side):
    """Ours readouts.<untold|cstar> vs Codex picks.<p1|q1>: the picked row's map, score, kept and (when ours has it) the table."""
    pick = dig(ours_ro, 'pick')
    if pick is MISSING or pick != dig(codex_pick_doc, 'pick'):
        chk.info.append(dict(item=f'{E}/table/{side}', ours=None if pick is MISSING else pick, codex=dig(codex_pick_doc, 'pick'),
                             equal=None, note='picks differ or are absent: table rows not compared'))
        return
    ours_table = dig(ours_ro, 'table')
    rows = list(ours_table) if ours_table is not MISSING else [pick]
    for ro in rows:
        o = dig(ours_table, ro) if ours_table is not MISSING else ours_ro
        c = dig(codex_pick_doc, 'table', ro)
        for a in ASPECTS:
            chk.note(f'{E}/table/{side}/{ro}/map/{a}', dig(o, 'map', a), dig(c, 'map', a))
        chk.note(f'{E}/table/{side}/{ro}/score', dig(o, 'score'), dig(c, 'score'))
        chk.note(f'{E}/table/{side}/{ro}/kept', dig(o, 'kept'), dig(c, 'kept'))


def files_val_mmart(rec, rd):
    return [rec / f'mmart_picks_{KSTAR}.json', rd / 's2mp_val.json']


def val_mmart(rec, rd):
    cxd = load(rd / 's2mp_val.json')
    cx = piece_of(cxd, 'mmart')
    codex_complete(cx, 's2mp_val.json mmart')
    ours = load(rec / f'mmart_picks_{KSTAR}.json')
    status = dict(complete=cx.get('complete'), mode=cxd.get('mode'), file_complete=cxd.get('complete'))
    chk = Check()
    chk.same('mmart/ours_arm', dig(ours, 'arm'), KSTAR)
    for a in ASPECTS:
        for m, key in (('ami', 'ami_ranges'), ('map', 'map_ranges')):
            chk.num(f'mmart/range/{m}/{a}', dig(ours, key, a, 'r'), dig(cx, 'ranges', a, m, 'r'))
            o = dig(ours, key, a, 'per_seed')
            chk.nums(f'mmart/per_seed/{m}/{a}', o if o is MISSING else seed_values(o), seed_values(dig(cx, 'ranges', a, m, 'per_seed')))
    chk.same('mmart/pick/p1_plain_side', dig(ours, 'readouts', 'untold', 'pick'), dig(cx, 'picks', 'p1', 'pick'))
    chk.same('mmart/pick/q1_side', dig(ours, 'readouts', 'cstar', 'pick'), dig(cx, 'picks', 'q1', 'pick'))
    tables(chk, 'mmart', dig(ours, 'readouts', 'untold'), dig(cx, 'picks', 'p1'), 'p1')
    tables(chk, 'mmart', dig(ours, 'readouts', 'cstar'), dig(cx, 'picks', 'q1'), 'q1')
    chk.unpair('ours', 'readouts.plain, readouts.teacher, n_val, validation_files, root', 'older rounds\' comparator picks; provenance')
    chk.unpair('codex', 'picks.*.units, ranges.*.n, ranges.*.n_labelled', 'information')
    return result(chk, [], status)


def files_val_photo(rec, rd):
    return [rec / f'photo_picks_{KSTAR}.json', rd / 's2mp_val.json']


def val_photo(rec, rd):
    cxd = load(rd / 's2mp_val.json')
    cx = piece_of(cxd, 'photo')
    codex_complete(cx, 's2mp_val.json photo')
    ours = load(rec / f'photo_picks_{KSTAR}.json')
    status = dict(complete=cx.get('complete'), mode=cxd.get('mode'), file_complete=cxd.get('complete'))
    chk = Check()
    chk.same('photo/ours_arm', dig(ours, 'arm'), KSTAR)
    for d in ('affection', 'emoset'):
        sd = dig(ours, 'sets', d)
        chk.num(f'photo/{d}/range/ami', dig(sd, 'ami_range', 'r'), dig(cx, 'ranges', d, 'ami', 'r'))
        o = dig(sd, 'ami_range', 'per_seed')
        chk.nums(f'photo/{d}/per_seed/ami', o if o is MISSING else seed_values(o), seed_values(dig(cx, 'ranges', d, 'ami', 'per_seed')))
        chk.num(f'photo/{d}/range/map', dig(sd, 'r_map'), dig(cx, 'ranges', d, 'map', 'r'))
        if d == 'affection':
            o = dig(sd, 'map_range', 'per_seed')
            chk.nums('photo/affection/per_seed/map', o if o is MISSING else seed_values(o),
                     seed_values(dig(cx, 'ranges', 'affection', 'map', 'per_seed')))
        else:
            chk.num('photo/emoset/r_E', dig(ours, 'r_E', 'value'), dig(cx, 'ranges', 'emoset', 'map', 'r'))
            chk.unpair('codex', 'ranges.emoset.map.own_r0_range, .per_seed, .r_source', 'F1\'s own R0 range, not the LEVEL unit (D19)')
        chk.same(f'photo/{d}/pick/image_map', dig(sd, 'readout', 'pick'), dig(cx, 'picks', d, 'pick'))
        pick = dig(sd, 'readout', 'pick')
        if pick is not MISSING and pick == dig(cx, 'picks', d, 'pick'):
            for ro, v in (dig(sd, 'readout', 'val_scores') if isinstance(dig(sd, 'readout', 'val_scores'), dict) else {}).items():
                chk.note(f'photo/{d}/table/{ro}/map/emotion', v, dig(cx, 'picks', d, 'table', ro, 'map', 'emotion'))
    chk.unpair('ours', 'comparator_picks, f1, val_files, code_sha256', 'older rounds\' comparator picks; provenance')
    return result(chk, [], status)


# ------------------------------------------------------------------------------------------------ held pieces
def ours_cells_by_key(doc, comp_names, kind):
    """key -> cell. kind: 'e2e3' (aspect, metric, comp), 'mmart' (aspect, metric, comp), 'photo' (set, metric, comp)."""
    out = {}
    for c in doc.get('cells', []):
        parts = c['id'].split('/')
        s, enc, arm, aspect, measure, comp = parts
        metric = 'ami' if measure.startswith('ami') else measure
        key = (s if kind == 'photo' else aspect, metric, comp_names.get(comp, comp))
        out[key] = c
    return out


def files_held_e2e3(rec, rd):
    return [rec / f'image_cells_{arm}.json' for arm in ENCODERS.values()] + [rd / 's2e_held.json']


def held_e2e3(rec, rd):
    cx = load(rd / 's2e_held.json')
    codex_complete(cx, 's2e_held.json')
    status = dict(complete=cx.get('complete'), kind=cx.get('kind'), picks_sha256=cx.get('picks_sha256'), inconclusive={})
    chk = Check()
    extra_inputs = []
    for E, arm in ENCODERS.items():
        x = dig(cx, 'encoders', E)
        if x is MISSING or not dig(x, 'cells'):
            raise Missing(f's2e_held.json has no cells of {E}', status)
        ours = ours_cells_by_key(load(rec / f'image_cells_{arm}.json'), {'plain': 'plain', 'teacher': 'q1'}, 'e2e3')
        theirs = {}
        for c in x['cells']:
            _, m, a, side = c['id'].split('/')
            theirs[(a, m, side)] = c
        expected = [(a, m, side) for a in ASPECTS for m in ('ami', 'map') for side in ('plain', 'q1')]
        compare_cells(chk, ours, theirs, expected, E)
        status['inconclusive'][E] = {c['id']: c.get('reason') for c in x['cells'] if c.get('outcome') == 'INCONCLUSIVE'}
        lp = rec / f'SC4_{E}_{arm}.json'
        ladder_info(chk, f'SC4/{E}', optional(lp), dig(x, 'sc4'))
        if lp.exists():
            extra_inputs.append(lp)
    return result(chk, extra_inputs, status)


def files_held_mp(rec, rd, name):
    return [rec / f'{name}_cells_{KSTAR}.json', rd / 's2mp_held.json']


def held_mp(rec, rd, name):
    cxd = load(rd / 's2mp_held.json')
    cx = piece_of(cxd, name)
    status = dict(mode=cxd.get('mode'), pending=cxd.get('pending'), picks_committed_attestation=cxd.get('picks_committed_attestation'),
                  validation_sha256=cxd.get('validation_sha256'))
    if cxd.get('mode') == 'controller_pending' or not cx.get('cells'):
        raise Missing(f's2mp_held.json {name}: no cells (mode {cxd.get("mode")!r}, pending {cxd.get("pending")!r})', status)
    ours = ours_cells_by_key(load(rec / f'{name}_cells_{KSTAR}.json'), {'plain': 'p1', 'teacher': 'q1'}, name)
    theirs = {}
    for c in cx['cells']:
        parts = c['id'].split('/')
        theirs[(parts[2], parts[1], parts[3]) if name == 'mmart' else (parts[0], parts[1], parts[2])] = c
    if name == 'mmart':
        expected = [(a, m, c) for a in ASPECTS for m in ('ami', 'map') for c in ('p1', 'q1')]
    else:
        expected = [(d, m, c) for d in ('affection', 'emoset') for m in ('ami', 'map') for c in ('p1', 'q1')]
    chk = Check()
    compare_cells(chk, ours, theirs, expected, name)
    status['inconclusive'] = {c['id']: c.get('reason') for c in cx['cells'] if c.get('outcome') == 'INCONCLUSIVE'}
    ladder = 'sc5' if name == 'mmart' else 'sc6'
    lp = rec / f'{ladder.upper()}_{KSTAR}.json'
    ladder_info(chk, ladder.upper(), optional(lp), dig(cx, ladder))
    return result(chk, [lp] if lp.exists() else [], status)


def held_mmart(rec, rd):
    return held_mp(rec, rd, 'mmart')


def held_photo(rec, rd):
    return held_mp(rec, rd, 'photo')


def files_held_ablation(rec, rd):
    return [rec / 'ablation.json', rd / 's2abl_held.json']


def held_ablation(rec, rd):
    ours = load(rec / 'ablation.json')
    cx = load(rd / 's2abl_held.json')
    status = dict(complete=cx.get('complete'), mode=cx.get('mode'), rows=cx.get('rows'), replay_certified=cx.get('replay_certified'),
                  kstar_check=cx.get('kstar_check'))
    if cx.get('complete') is False or not cx.get('cells'):
        raise Missing('s2abl_held.json: complete false or no cells', status)
    if not ours.get('cells'):
        raise Missing(f'ablation.json has no cells (status {ours.get("status")!r}, cause {ours.get("cause")!r})', status)
    source = ours.get('kstar')
    chk = Check()
    o_cells, c_cells = {}, {}
    for c in ours['cells']:
        parts = c['id'].split('/')
        o_cells[(parts[2], parts[3], 'ami' if parts[4].startswith('ami') else parts[4])] = c
    for c in cx['cells']:
        _, row, a, m = c['id'].split('/')
        c_cells[(row, a, m)] = c
    inconclusive = ours.get('inconclusive_removals') or {}
    status['ours_inconclusive_removals'] = inconclusive
    for row in ROWS:
        for a in ASPECTS:
            for m in ('ami', 'map'):
                key = (row, a, m)
                cid = f'ablation/{row}/{a}/{m}'
                o, c = o_cells.get(key), c_cells.get(key)
                if o is None and c is None:
                    chk.unpair('both', cid, f'not scored on either side (ours: {inconclusive.get(row, "no cause recorded")})')
                    continue
                if o is None or c is None:
                    chk.n += 1
                    chk.mismatches.append(dict(id=cid, ours=None if o is None else 'cell', codex=None if c is None else 'cell',
                                               diff=None, note='no cell of ' + ('ours' if o is None else 'Codex') +
                                               (f' (ours inconclusive: {inconclusive[row]})' if o is None and row in inconclusive else '')))
                    continue
                n0 = len(chk.items)
                chk.num(f'{cid} removal_mean', dig(o, 'value'), dig(c, 'removal_mean'))
                chk.num(f'{cid} kstar_mean', dig(o, 'comparator_value'), dig(c, 'kstar_mean'))
                chk.nums(f'{cid} removal_per_seed', dig(ours, 'arms', row, a, f'{m}_per_seed'), dig(c, 'removal_per_seed'))
                chk.nums(f'{cid} kstar_per_seed', dig(ours, 'arms', source, a, f'{m}_per_seed') if source else MISSING,
                         dig(c, 'kstar_per_seed'))
                for k in ('d', 'lo', 'hi'):
                    chk.num(f'{cid} {k}', dig(o, k), dig(c, k))
                chk.note(f'{cid} d_per_seed', dig(o, 'per_seed'), dig(c, 'd_per_seed'))
                chk.note(f'{cid} n', dig(o, 'n_rows') if m == 'ami' else dig(ours, 'arms', row, a, 'n_queries'), dig(c, 'n'))
                chk.note(f'{cid} n_draws', dig(o, 'n_draws'), dig(c, 'n_draws'))
                diffs = [i['diff'] for i in chk.items[n0:] if i.get('diff') is not None]
                chk.cells.append(dict(cell=cid, max_diff=max(diffs) if diffs else None))
    for side, cells in (('ours', o_cells), ('codex', c_cells)):
        for key in cells:
            if key[0] not in ROWS or key[1] not in ASPECTS or key[2] not in ('ami', 'map'):
                chk.unpair(side, '/'.join(key), 'cell outside the 24 expected')
    chk.unpair('ours', 'cells.r, cells.outcome', 'None: SC7 has no outcome')
    chk.unpair('codex', 'kstar_check', 'Codex\'s own consistency check (recorded under codex_status)')
    return result(chk, [], status)


PIECES = {
    'val': {'e2e3': (files_val_e2e3, val_e2e3), 'sc9': (files_val_sc9, val_sc9), 'mmart': (files_val_mmart, val_mmart),
            'photo': (files_val_photo, val_photo)},
    'held': {'e2e3': (files_held_e2e3, held_e2e3), 'mmart': (lambda r, d: files_held_mp(r, d, 'mmart'), held_mmart),
             'photo': (lambda r, d: files_held_mp(r, d, 'photo'), held_photo), 'ablation': (files_held_ablation, held_ablation)},
}


# ------------------------------------------------------------------------------------------------ run
def run_piece(phase, name, rec, rd):
    files_fn, fn = PIECES[phase][name]
    files = files_fn(rec, rd)
    absent = [str(f) for f in files if not Path(f).exists()]
    if absent:
        return dict(status='missing', reason='input file missing: ' + ', '.join(absent), missing_files=absent, written=now())
    inputs = {str(f): sha256(f) for f in files}
    try:
        out = fn(rec, rd)
    except Missing as m:
        return dict(status='missing', reason=m.reason, codex_status=m.codex_status, inputs=inputs, written=now())
    extra = out.pop('extra_files', [])
    out['inputs'] = {str(f): sha256(f) for f in [*files, *extra]}
    out['written'] = now()
    return out


def run(phase, pieces, rec=REC, rd=RD, out_path=None):
    """Returns (document, pieces run, exit code). Updates only the phase's sections of the pieces run in the existing file."""
    rec, rd = Path(rec), Path(rd)
    out_path = Path(out_path) if out_path else rec / 's2_compare.json'
    doc = load(out_path) if out_path.exists() else {}
    doc.setdefault('tolerance', TOL)
    for k in ('pieces', 'mismatches', 'missing', 'written'):
        doc.setdefault(k, {})
    sections = {k: doc[k].setdefault(phase, {}) for k in ('pieces', 'mismatches', 'missing')}
    ran = {}
    for name in pieces:
        r = run_piece(phase, name, rec, rd)
        ran[name] = r
        sections['pieces'][name] = r
        sections['mismatches'][name] = r.get('mismatches', []) if r['status'] != 'missing' else []
        if r['status'] == 'missing':
            sections['missing'][name] = r['reason']
        else:
            sections['missing'].pop(name, None)
    doc['written'][phase] = now()
    out_path.write_text(json.dumps(doc, indent=1) + '\n')
    bad = any(r['status'] == 'mismatch' for r in ran.values())
    return doc, ran, (1 if bad else 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('val', 'held'), required=True)
    ap.add_argument('--pieces', nargs='+', choices=sorted(set(VAL_PIECES) | set(HELD_PIECES)))
    ap.add_argument('--records', default=str(REC), help='folder of our records (tests)')
    ap.add_argument('--rederive', default=str(RD), help='folder of Codex\'s files (tests)')
    ap.add_argument('--out', default=None, help='output file (default records/s2_compare.json)')
    args = ap.parse_args(argv)
    pieces = args.pieces
    if pieces:
        bad = [p for p in pieces if p not in PHASE_PIECES[args.phase]]
        if bad:
            ap.error(f'piece(s) {bad} do not exist in phase {args.phase} (pieces: {PHASE_PIECES[args.phase]})')
    else:   # every piece of the phase: those with both sides present are compared, the others are reported under `missing`
        pieces = list(PHASE_PIECES[args.phase])
    doc, ran, code = run(args.phase, pieces, args.records, args.rederive, args.out)
    for name, r in ran.items():
        if r['status'] == 'missing':
            print(f'{args.phase}/{name}: MISSING ({r["reason"][:160]})')
        else:
            md = r['max_abs_diff']
            print(f'{args.phase}/{name}: {r["status"]}; compared {r["n_compared"]}; max diff {"-" if md is None else format(md, ".2e")}; '
                  f'mismatches {len(r["mismatches"])}')
            for m in r['mismatches']:
                print(f'   {m["id"]}: ours {m["ours"]!r} codex {m["codex"]!r} diff {m["diff"]} ({m["note"]})')
    return code


if __name__ == '__main__':
    sys.exit(main())
