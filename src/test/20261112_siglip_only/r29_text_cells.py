"""r29 ticket 07: the nine ArtELingo text test cells of a new model against P1 and Q1, by r23's protocol.

Spec docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md §3.4 (cells table, row "ArtELingo text, three directions";
outcome rule; order and C13), D18, D20, D23, §3.5 gate (v) text part.

A cell = one aspect (emotion, style, genre) x one direction (t2i, i2t, t2t), for one comparator. Each cell is r23's
`tcl_outcome.label_cell` (imported, not copied): the candidate's five seeds of per-query AP (head = aspect) at the candidate's
validation pick against a comparator's per-query AP at the comparator's own pick; the interval is r23's
`tcl_metrics.painting_interval` (painting unit, every caption query of a drawn painting carried with it, one shared seed draw,
10,000 draws, RNG 0); WIN when the lower bound is above 0, LEVEL when above -r, else BELOW; r is the candidate's own range at
its pick (r23: not the comparator's). `label_cell` asserts that candidate seeds and comparator agree in query IDs and scored
rows (an arm that differs raises).

  candidate  row `cstar`, head = aspect, seeds 0 to 4, files of ticket 04's score root `r29_text.score_root(arm)`;
             pick, mean and range from `records/text_<arm>.json` (read through ticket 04's `require_pick_record`: committed
             unchanged in HEAD, store and weights as scored)
  Q1         r23's `teacher` rows (head = aspect) at r23's pick (`records/stage1_art.json`), r23's output folder
  P1 plain   r23's SigLIP 1 side row `siglip_so400m` (head `all`) at r23's pick, r23's output folder. A comparator in its
             own right: `label_cell` takes the row name as given, so the row is not routed as a teacher
  P1 prefix  ticket 04's `p1_siglip1_prefix` rows (row `siglip_so400m`, head `all`) at the pick of `records/text_p1_siglip1_
             prefix.json`, in ticket 04's score root. The form is read from `records/prefix_pick.json` (D20: key `p1_form`, `plain`
             | `prefixed`; the same file names K* in `kstar`, and a cells run for any other arm is refused)
`label_cell` reads every file through `tcl_paths.score_path` with one output root; `routed` points that function at one root per
row (candidate `cstar` -> the arm's root, the comparator row -> its own) while a cell is scored, and `mock.patch` restores it.

`cells` refuses, before any number, unless `records/pass_rule.json` is present and committed unchanged (SC2 first, §3.5 row 1b;
its status is not read, D9) and `records/prefix_pick.json` is committed unchanged and names the arm as K*.

Commands (from the round folder):
    python r29_text_cells.py cells --arm siglip1|siglip1_prefix       -> records/text_cells_<arm>.json
    python r29_text_cells.py gate-v-text [--aspects ..] [--draws N]   -> records/gates/v_text.json (a reduced run writes
                                                                         OUT/gates_partial/v_text.json instead)

Record `records/text_cells_<arm>.json`: arm, encoder, p1_form, pick_record / held_record / prefix_pick (sha256 of the files read),
written, cells = 18 cells with the field names of ticket 06's image cells (`r29_image_cells._cell`): id
`artelingo/<encoder>/<arm>/<aspect>/<direction>/<p1|q1>`, type `text`, protocol `r23_label`, set, encoder, arm, aspect, metric,
k, comparator (`p1` | `q1`, equal to the id's last part), value (candidate mean AP over seeds and scored queries),
comparator_value, d (value minus comparator_value), lo, hi (the 95% interval), r, outcome, n_draws, plus direction,
comparator_row (r23's row name), candidate_readout, comparator_readout, n_queries, n_units, and p1_form on the P1 cells.

Gate (v) text: on the Qwen3 c* files r23 scored, the same code (candidate row `cstar` at r23's picks and ranges; the two
comparators r23 used, `teacher` and `plain`: `plain` is a one-vector row, head `all`, as the SigLIP 1 side row is) gives r23's
18 ArtELingo label cells of SC1 (stage 2 `cells.json`: d, lower, upper, outcome, r) within 1e-9; the cell means are also checked
against r28's `records/text.json` (`ours_qwen3`, `instructed_qwen3`, `plain_qwen3`), which holds them. The record stores each
replayed cell's values (d, lower, upper, r, the means) beside r23's and r28's. A second part runs the P1 plain route (r23's
`siglip_so400m` row at r23's pick, against the same c* files) and checks its comparator means against r28's `plain_siglip1`.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_paths as P29  # noqa: E402

P29.use_older_rounds()
import r29_text as X  # noqa: E402  (ticket 04; imports r28_text, which puts r23's folder on sys.path)
import tcl_paths as P23  # noqa: E402
import tcl_outcome as O23  # noqa: E402
P29.check_older_imports()  # every older-round module loaded so far is the main checkout's copy

KIND = P23.CAPTION
ASPECTS, DIRECTIONS = tuple(P23.ASPECTS), tuple(P23.DIRECTIONS)
assert ASPECTS == tuple(P29.ASPECTS)
ROW = 'cstar'                          # r23's name of the row `label_cell` reads as the candidate (head = aspect, seeds 0 to 4)
SIDE_ROW = X.IMAGE_ROW                 # 'siglip_so400m': r23's SigLIP 1 side row, and the row name of ticket 04's P1 files
R23_OUT = Path(X.T.R23_OUT)
R23_STAGE1 = Path(X.T.R23_STAGE1)
N_DRAWS = P23.N_DRAWS                  # 10,000 (r23)
TOL = 1e-9                             # D23: replayed test cells
SET = 'artelingo'
ARMS = ('siglip1', 'siglip1_prefix')   # K1, K2 (tickets 10 and 11 extend with the other encoders' arms)
ENCODER_OF = {'siglip1': 'siglip1', 'siglip1_prefix': 'siglip1'}
P1_FORMS = ('plain', 'prefixed')
P1_PREFIX_ARM = 'p1_siglip1_prefix'
PROTOCOL = 'r23_label'
R28_TEXT = Path(P29.R28_RECORDS) / 'text.json'


# ================================================================================================ one cell
@contextmanager
def routed(roots):
    """While a cell is scored, `tcl_paths.score_path` puts each row's files under its own output root: roots = {row: folder}.
    An output root given by the caller, or a row not in `roots`, is a crash (the files of other arms are never read by chance)."""
    orig = P23.score_path

    def score_path(set_name, kind, row, head, seed, direction, readout, out=None):
        assert out is None and row in roots, (row, out)
        return orig(set_name, kind, row, head, seed, direction, readout, out=Path(roots[row]))

    with mock.patch.object(P23, 'score_path', score_path):
        yield


def paired(cand_root, cand_pick, r, comp_row, comp_root, comp_pick, aspect, direction, draws=N_DRAWS):
    """One cell by r23's `label_cell`: the candidate (row `cstar`, head = aspect, seeds 0 to 4, read-out `cand_pick`, files under
    `cand_root`) against the comparator row (files under `comp_root`, read-out `comp_pick`), LEVEL unit `r` (the candidate's).
    Returns value, comparator_value (means over the scored queries; the candidate's over seeds too), d, lo, hi, outcome, r,
    n_units, n_query and the two read-outs."""
    assert comp_row != ROW and aspect in ASPECTS and direction in DIRECTIONS
    record = dict(picks={KIND: {ROW: {aspect: {direction: cand_pick}}, comp_row: {aspect: {direction: comp_pick}}}},
                  ranges={KIND: {ROW: {aspect: {direction: r}}}})
    seen = {}
    interval = O23._interval

    def capture(cand, comp, pid, draws_):
        seen.update(value=float(cand.mean()), comparator_value=float(comp.mean()))
        return interval(cand, comp, pid, draws_)

    with routed({ROW: cand_root, comp_row: comp_root}), mock.patch.object(O23, '_interval', capture):
        row = O23.label_cell('art', ROW, comp_row, aspect, direction, record, draws)
    assert row['candidate_readout'] == cand_pick and row['comparator_readout'] == comp_pick and row['seed_range'] == r
    assert abs(row['point'] - (seen['value'] - seen['comparator_value'])) <= 1e-15
    return dict(value=seen['value'], comparator_value=seen['comparator_value'], d=row['point'], lo=row['lower'], hi=row['upper'],
                outcome=row['outcome'], r=row['seed_range'], n_units=row['n_units'], n_query=row['n_query'],
                candidate_readout=row['candidate_readout'], comparator_readout=row['comparator_readout'])


def score_cells(arm, encoder, cand, p1, q1, draws=N_DRAWS, aspects=ASPECTS, names=('p1', 'q1')):
    """The label cells of one candidate against P1 and Q1 (or any two comparators): per aspect and direction, `p1` then `q1`.
    cand = {root, picks: {aspect: {direction: {pick, range, ..}}}}; p1, q1 = {row, root, picks: {aspect: {direction: pick}},
    form (P1 only)}. `names` limits the comparators scored (the gate's extra P1 check scores `p1` alone)."""
    out = []
    for a in aspects:
        for d in DIRECTIONS:
            pk = cand['picks'][a][d]
            for name, comp in ((n, c) for n, c in (('p1', p1), ('q1', q1)) if n in names):
                res = paired(cand['root'], pk['pick'], pk['range'], comp['row'], comp['root'], comp['picks'][a][d], a, d, draws)
                cell = dict(id=f'{SET}/{encoder}/{arm}/{a}/{d}/{name}', type='text', protocol=PROTOCOL, set=SET, encoder=encoder,
                            arm=arm, aspect=a, direction=d, metric='map', k=None, comparator=name, comparator_row=comp['row'],
                            value=res['value'], comparator_value=res['comparator_value'], d=res['d'], lo=res['lo'], hi=res['hi'],
                            r=float(res['r']), outcome=res['outcome'], n_draws=int(draws), n_queries=int(res['n_query']),
                            n_units=int(res['n_units']), candidate_readout=res['candidate_readout'],
                            comparator_readout=res['comparator_readout'])
                if name == 'p1':
                    cell['p1_form'] = comp.get('form')
                assert all(np.isfinite(cell[k]) for k in ('value', 'comparator_value', 'd', 'lo', 'hi', 'r')), cell['id']
                out.append(cell)
    return out


# ================================================================================================ comparators
def q1_spec(r23):
    """Q1: r23's teacher rows at r23's picks (head = aspect), in r23's output folder."""
    return dict(row='teacher', root=R23_OUT, picks=r23['picks'][KIND]['teacher'])


def p1_spec(form, r23_stage1, p1_pick_record):
    """P1 by its form (D20). plain: r23's SigLIP 1 side row at r23's pick. prefixed: ticket 04's `p1_siglip1_prefix` rows at the
    picks of its committed pick record (`p1_pick_record`, from `require_pick_record`). Neither is routed as a teacher."""
    assert form in P1_FORMS, form
    if form == 'plain':
        return dict(row=SIDE_ROW, root=R23_OUT, picks=r23_stage1['picks'][KIND][SIDE_ROW], form=form)
    assert p1_pick_record is not None, 'the prefixed P1 needs the pick record of ' + P1_PREFIX_ARM
    picks = {a: {d: p1_pick_record['picks'][a][d]['pick'] for d in DIRECTIONS} for a in ASPECTS}
    return dict(row=SIDE_ROW, root=X.score_root(P1_PREFIX_ARM), picks=picks, form=form)


def read_p1_form(path, arm):
    """P1's caption form (D20) from `records/prefix_pick.json` (ticket 05; keys `kstar`, `kstar_form`, `p1_form`, `flag`). The
    scored arm must be K* (`kstar`): a losing arm is refused. `p1_form` is 'plain' or 'prefixed'; when K1 is K* (arm
    `siglip1`) D20 never picks a form and P1 is the plain one."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'no prefix_pick record {path}: ticket 05 writes it before stage 1b (spec §3.5)')
    assert arm in ARMS, arm
    doc = json.loads(path.read_text())
    for key in ('kstar', 'kstar_form', 'p1_form', 'flag'):
        assert key in doc, f'{path}: key {key} missing (keys {sorted(doc)})'
    assert doc['kstar'] in ARMS, f'{path}: kstar {doc["kstar"]!r} not in {ARMS}'
    if doc['kstar'] != arm:
        raise RuntimeError(f'{path}: K* is {doc["kstar"]!r}, so arm {arm!r} gets no test cells (spec §3.4: K* only)')
    assert doc['kstar_form'] in P1_FORMS, f'{path}: kstar_form {doc["kstar_form"]!r} not in {P1_FORMS}'
    form = doc['p1_form']
    assert form in P1_FORMS, f'{path}: p1_form {form!r} not in {P1_FORMS}'
    assert arm != 'siglip1' or form == 'plain', f'{path}: K1 is the arm, so P1 is the plain form, not {form!r}'
    return form


def require_committed(path):
    """The record is committed unchanged in git HEAD (outside smoke): the order rule's check for records this step reads."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'no record {path}: {path.name} is written and committed before any test number (spec §3.4, §3.5)')
    if X.is_smoke():
        return
    rel = path.resolve().relative_to(P29.REPO.resolve())
    head = subprocess.run(['git', '-C', str(P29.REPO), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout != path.read_text():
        raise RuntimeError(f'{rel} is not committed unchanged in HEAD: commit it before the test cells (spec §3.4)')


# ================================================================================================ checks against records
def check_value(name, got, want):
    assert np.isfinite(got) and np.isfinite(want) and abs(got - want) <= TOL, f'{name}: {got} against the record {want}'


def check_held_record(pick, held):
    """The held record (ticket 04's `cells`) was made at the picks of the pick record."""
    for a in ASPECTS:
        for d in DIRECTIONS:
            assert held['picks'][a][d] == pick['picks'][a][d]['pick'], \
                f'held record picks {held["picks"][a][d]}, pick record {pick["picks"][a][d]["pick"]} ({a}/{d})'


def check_candidate_values(cells, held):
    """The candidate's value of every cell equals the held record's mean at that aspect and direction (1e-9)."""
    for c in cells:
        check_value(f'{c["id"]} candidate value', c['value'], held['cells'][c['aspect']][c['direction']]['mean'])


def check_comparator_values(cells, refs):
    """Each cell's comparator value equals the mean a record holds. refs = {comparator: {aspect: {direction: mean}}}."""
    for c in cells:
        check_value(f'{c["id"]} comparator value', c['comparator_value'], refs[c['comparator']][c['aspect']][c['direction']])


def r28_means(key):
    """{aspect: {direction: mean}} of one row of r28's records/text.json (r23's files at r23's picks)."""
    doc = json.loads(R28_TEXT.read_text())[key]['cells']
    return {a: {d: doc[a][d]['mean'] for d in DIRECTIONS} for a in ASPECTS}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ================================================================================================ cells
def cells(arm, draws=N_DRAWS):
    """The 18 cells of arm A (9 against P1, 9 against Q1) -> records/text_cells_<A>.json. Refuses unless the arm's pick record
    is on record, committed unchanged and made for the store and weights as scored (ticket 04's `require_pick_record`), and
    the held scores exist (`r29_text.py cells --arm A`). Also refuses unless `records/pass_rule.json` is on record and committed
    unchanged (SC2 before any test number, §3.5 row 1b; D9: it is checked for presence, not for its status, because the test
    cells run on a pass and on a fail) and unless `records/prefix_pick.json` names the arm as K* (§3.4: K* only)."""
    assert arm in ARMS, f'arm {arm!r}: one of {ARMS}'
    require_committed(Path(X.RECORDS_DIR) / 'pass_rule.json')
    pick = X.require_pick_record(arm)
    held_path = X.record_path(arm, held=True)
    if not held_path.exists():
        raise RuntimeError(f'no held record {held_path}: run `r29_text.py cells --arm {arm}` first')
    held = json.loads(held_path.read_text())
    check_held_record(pick, held)
    prefix_path = Path(X.RECORDS_DIR) / 'prefix_pick.json'
    require_committed(prefix_path)
    form = read_p1_form(prefix_path, arm)
    r23 = json.loads(R23_STAGE1.read_text())
    assert r23['group'] == 'art'
    p1_pick = p1_held = None
    refs = {'q1': r28_means('instructed_qwen3')}
    if form == 'prefixed':
        p1_pick = X.require_pick_record(P1_PREFIX_ARM)
        p1_held_path = X.record_path(P1_PREFIX_ARM, held=True)
        if not p1_held_path.exists():
            raise RuntimeError(f'no held record {p1_held_path}: run `r29_text.py cells --arm {P1_PREFIX_ARM}` first')
        p1_held = json.loads(p1_held_path.read_text())
        check_held_record(p1_pick, p1_held)
        refs['p1'] = {a: {d: p1_held['cells'][a][d]['mean'] for d in DIRECTIONS} for a in ASPECTS}
    else:
        refs['p1'] = r28_means('plain_siglip1')
    p1, q1 = p1_spec(form, r23, p1_pick), q1_spec(r23)
    cand = dict(root=X.score_root(arm), picks=pick['picks'])
    encoder = ENCODER_OF[arm]
    out = score_cells(arm, encoder, cand, p1, q1, draws)
    check_candidate_values(out, held)
    check_comparator_values(out, refs)
    rec = dict(arm=arm, encoder=encoder, p1_form=form, written=X.now(), pick_record=sha_file(X.record_path(arm)),
               held_record=sha_file(held_path), prefix_pick=sha_file(prefix_path), cells=out)
    path = Path(X.RECORDS_DIR) / f'text_cells_{arm}.json'
    X.write_json(path, rec)
    return path, rec


# ================================================================================================ gate (v) text
def compare_to_r23(cells_, recorded, check_interval):
    """Our cells against r23's recorded label cells (stage 2 `cells.json` rows), matched by aspect, direction and comparator row,
    every one exactly once. d, r and (with `check_interval`) lower and upper within 1e-9, outcome and read-outs equal."""
    key = lambda aspect, direction, comp: (aspect, direction, comp)  # noqa: E731
    by = {}
    for r in recorded:
        k = key(r['aspect'], r['direction'], r['comparator'])
        assert k not in by, f'r23 record: {k} twice'
        by[k] = r
    assert len(by) == len(cells_), f'{len(cells_)} cells, {len(by)} recorded'
    items = []
    for c in cells_:
        rec = by[key(c['aspect'], c['direction'], c['comparator_row'])]
        want = dict(d=rec['point'], r=rec['seed_range'])
        got = dict(d=c['d'], r=c['r'])
        if check_interval:
            want.update(lo=rec['lower'], hi=rec['upper'])
            got.update(lo=c['lo'], hi=c['hi'])
        diffs = {k: abs(got[k] - want[k]) for k in want}
        same = c['candidate_readout'] == rec['candidate_readout'] and c['comparator_readout'] == rec['comparator_readout']
        ok = max(diffs.values()) <= TOL and same and (not check_interval or c['outcome'] == rec['outcome'])
        items.append(dict(id=c['id'], comparator_row=c['comparator_row'], max_abs_diff=float(max(diffs.values())), diffs=diffs,
                          replayed={k: float(v) for k, v in got.items()}, recorded={k: float(v) for k, v in want.items()},
                          outcome=c['outcome'], recorded_outcome=rec['outcome'], readouts_equal=bool(same), ok=bool(ok)))
    return items


def gate_v_text(aspects=ASPECTS, draws=N_DRAWS):
    """Gate (v), text part: r23's own c* files through this module's code. Returns (ok, details).
    (1) The 18 SC1 label cells of r23 (Q1 route `teacher`, one-vector route `plain`): d, lower, upper, r, outcome, read-outs
    against r23's `stage2/cells.json`, and the cell means against r28's `text.json`. Each cell's replayed values are stored,
    not only the differences. (2) The P1 plain route (`p1_spec('plain')`: r23's SigLIP 1 side row at r23's pick) on the same
    c* files: its comparator values against r28's `plain_siglip1` means, and its cells' outcomes."""
    r23 = json.loads(R23_STAGE1.read_text())
    recorded = [c for c in json.loads((R23_OUT / 'stage2/cells.json').read_text())
                if c['sc'] == 'SC1' and c['group'] == 'art' and c['cell'] == 'label' and c['aspect'] in aspects]
    assert len(recorded) == 2 * len(aspects) * len(DIRECTIONS), len(recorded)
    cand_picks = {a: {d: dict(pick=r23['picks'][KIND][ROW][a][d], range=r23['ranges'][KIND][ROW][a][d]) for d in DIRECTIONS}
                  for a in ASPECTS}
    cand = dict(root=R23_OUT, picks=cand_picks)
    plain = dict(row='plain', root=R23_OUT, picks=r23['picks'][KIND]['plain'], form='r23 Qwen3 plain row (a one-vector row, as P1)')
    t0 = time.monotonic()
    got = score_cells('r23_cstar', 'qwen', cand, plain, q1_spec(r23), draws, aspects)
    check_interval = draws == N_DRAWS
    items = compare_to_r23(got, recorded, check_interval)
    refs = dict(cand=r28_means('ours_qwen3'), plain=r28_means('plain_qwen3'), teacher=r28_means('instructed_qwen3'))
    for c, item in zip(got, items):
        comp = refs['plain' if c['comparator_row'] == 'plain' else 'teacher'][c['aspect']][c['direction']]
        item['value_diff'] = abs(c['value'] - refs['cand'][c['aspect']][c['direction']])
        item['comparator_value_diff'] = abs(c['comparator_value'] - comp)
        item['replayed_means'] = dict(value=c['value'], comparator_value=c['comparator_value'])
        item['r28_means'] = dict(value=refs['cand'][c['aspect']][c['direction']], comparator_value=comp)
        item['ok'] = bool(item['ok'] and item['value_diff'] <= TOL and item['comparator_value_diff'] <= TOL)
    # the P1 plain route, which r23's records do not hold: its comparator means are r28's `plain_siglip1`
    p1_ref = r28_means('plain_siglip1')
    p1_cells = score_cells('r23_cstar', 'qwen', cand, p1_spec('plain', r23, None), q1_spec(r23), draws, aspects, names=('p1',))
    p1_items = []
    for c in p1_cells:
        want = p1_ref[c['aspect']][c['direction']]
        diff, vdiff = abs(c['comparator_value'] - want), abs(c['value'] - refs['cand'][c['aspect']][c['direction']])
        p1_items.append(dict(id=c['id'], comparator_row=c['comparator_row'], value=c['value'], comparator_value=c['comparator_value'],
                             r28_value=refs['cand'][c['aspect']][c['direction']], r28_comparator_value=want, value_diff=vdiff,
                             comparator_value_diff=diff, d=c['d'], lo=c['lo'], hi=c['hi'], r=c['r'], outcome=c['outcome'],
                             ok=bool(diff <= TOL and vdiff <= TOL)))
    p1_check = dict(n_cells=len(p1_items), outcomes={o: sum(i['outcome'] == o for i in p1_items) for o in ('WIN', 'LEVEL', 'BELOW')},
                    max_abs_diff=float(max(max(i['value_diff'], i['comparator_value_diff']) for i in p1_items)),
                    comparator='r23 siglip_so400m (head all) at r23 picks, against r28 plain_siglip1', cells=p1_items)
    ok = (all(i['ok'] for i in items) and len(items) == 2 * len(aspects) * len(DIRECTIONS)
          and all(i['ok'] for i in p1_items) and len(p1_items) == len(aspects) * len(DIRECTIONS))
    worst = max(max(i['max_abs_diff'], i['value_diff'], i['comparator_value_diff']) for i in items)
    return ok, dict(aspects=list(aspects), draws=draws, intervals_checked=check_interval, tol=TOL, n_cells=len(items),
                    max_abs_diff=float(max(worst, p1_check['max_abs_diff'])), seconds=round(time.monotonic() - t0), cells=items,
                    p1_plain_check=p1_check, candidate='r23 cstar (Qwen3) files at r23 picks and ranges',
                    comparators='teacher (Q1 route) and plain (one-vector P1 route), r23 files at r23 picks')


def write_gate(ok, details, partial_dir=None):
    """records/gates/v_text.json = {gate, status, tries, details, written}, written through `r29_captions.write_gate_record` (it counts
    the tries and raises on a fourth): `pass`; a failure is `fail` before try 3 and `parked` at try 3. A reduced check (a subset of
    aspects or fewer draws) goes to `partial_dir` (default OUT/gates_partial) and never to the gate record (it counts its own)."""
    full = set(details['aspects']) == set(ASPECTS) and details['intervals_checked']
    if full:
        folder = Path(X.RECORDS_DIR) / 'gates'
        path = folder / 'v_text.json'
        prior = json.loads(path.read_text()).get('tries', 0) if path.exists() else 0
        status = 'pass' if ok else ('parked' if prior == 2 else 'fail')
        return path, X.C.write_gate_record(folder, dict(status=status, details=details), gate='v_text')
    path = Path(partial_dir or P29.OUT / 'gates_partial') / 'v_text.json'
    tries = json.loads(path.read_text()).get('tries', 0) + 1 if path.exists() else 1
    rec = dict(gate='v_text', status='partial-pass' if ok else 'partial-fail', tries=tries, details=details, written=X.now())
    X.write_json(path, rec)
    return path, rec


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cells', help='the 18 text cells of an arm -> records/text_cells_<arm>.json (refuses without its pick record)')
    c.add_argument('--arm', required=True, choices=ARMS)
    g = sub.add_parser('gate-v-text', help="replay r23's nine ArtELingo text cells on its Qwen3 c* files")
    g.add_argument('--aspects', nargs='+', choices=ASPECTS, default=list(ASPECTS))
    g.add_argument('--draws', type=int, default=N_DRAWS)
    args = ap.parse_args(argv)
    assert not P23.SMOKE, 'TCL_SMOKE is set: r23 would read its smoke shapes'
    t0 = time.monotonic()
    if args.cmd == 'cells':
        path, rec = cells(args.arm)
        print(f'cells {args.arm}: {len(rec["cells"])} (p1 form {rec["p1_form"]}) -> {path} ({time.monotonic() - t0:.0f}s)')
        return 0
    ok, details = gate_v_text(tuple(args.aspects), args.draws)
    path, rec = write_gate(ok, details)
    print(f'gate v_text: {rec["status"]} max_abs_diff {details["max_abs_diff"]:.3g} ({time.monotonic() - t0:.0f}s) {path}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
