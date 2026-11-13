"""r29 ticket 08: the claim ladders of spec §6 (SC3 to SC6), the numbers sheet `paper_cells.md`, diagnostic (d).

Spec docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md §6 (levels, "most cells", sentences, inconclusive rule),
§3.4 "Cells per claim test", D9 (cells go to the sheet on a fail too), D12, D22, §3.6 (d); constitution row 1 (never edit a
level), row 8 (fill only the slots of the reached level's sentence).

Ladder (`ladder`): per claim, `n` cells, each with an outcome against P1 and against Q1 (WIN, LEVEL, BELOW, or INCONCLUSIVE:
the cell is not run, its gate is parked, or its record says so). "Most cells" `most` is fixed per claim in `CLAIMS` (12 of 15,
5 of 6, 5 of 6, 3 of 4; spec §6, §3.4), never recomputed.
  level 1  WIN over P1 in >= most cells        level 2  WIN over P1 in all n cells
  level 3  level 2 and WIN or LEVEL over Q1 in >= most cells
An inconclusive cell counts as neither WIN nor LEVEL. "level L, inconclusive above" (D22) when counting every inconclusive
cell as a WIN (P1 side) and as a WIN or LEVEL (Q1 side) would give a higher level. Each Q1 cell has its own claim word: WIN
"above", LEVEL "level with", BELOW none (§6).

Cell records read (written by tickets 06, 07; 11 and 13 write the same shape, file names fixed here):
  image_cells_<arm>.json (06), text_cells_<arm>.json (07), mmart_cells_<arm>.json (11), photo_cells_<arm>.json (13): each
  {arm, encoder, cells: [..]}; a cell holds id `<set>/<encoder>/<arm>/<aspect>/<measure>/<comparator>` (measure ami<k>, map,
  t2i, i2t, t2t; comparator: image cells plain / teacher, text cells p1 / q1; P1 = plain = p1, Q1 = teacher = q1, the other
  form in a file is refused) and type, protocol, comparator, aspect, arm, value, d, r, outcome, lo, hi (no `interval` key). Every read asserts these; the outcome must be the rule's
  word for lo and r (`outcome`, as r11 and ticket 06). A missing cell inside a file is "not run", counted as neither; a missing
  FILE refuses the ladder (unless a gate parks the claim, or `--missing-ok <reason>`, which goes into the record).
Order and gates: SC3 to SC6 need records/pass_rule.json present and committed unchanged in HEAD (SC2 parked: refused, 1c does not
  run; SC4 to SC6 also status pass, D3). Gate parking: `GATE_PARKS` (spec §3.5); an absent gate record the claim needs raises
  "gate <id> not run"; status fail or parked parks the claim's cells.

CLI (from the round folder):
  ladder --claim SC3|SC4|SC5|SC6 --arm A [--encoder E] [--missing-ok REASON]   records/<claim>_<arm>.json (SC4: SC4_<encoder>_<arm>.json), once
  sheet [--out FILE]                                     rewrite paper_cells.md from every cell record present
  diag-d --arm A [--claim SC3|SC4] [--encoder E]         records/diag_d_<claim>_<arm>.json, after the ladder record exists
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_paths as P  # noqa: E402  (older rounds are put on sys.path only by P.use_older_rounds(), called lazily below)

ASPECTS = tuple(P.ASPECTS)
IMG_MEASURES = ('ami', 'map')
TXT_MEASURES = ('t2i', 'i2t', 't2t')
W, LV, B, INC = 'WIN', 'LEVEL', 'BELOW', 'INCONCLUSIVE'
OUTCOMES = (W, LV, B, INC)
COMPARATOR = {'plain': 'P1', 'p1': 'P1', 'teacher': 'Q1', 'q1': 'Q1'}
ENCODER_OF = {'siglip1': 'siglip1', 'siglip2': 'siglip2', 'clipl14': 'clip_l14'}     # caption source prefix -> image encoder
TZ = ZoneInfo('Europe/Amsterdam')
TOL = 1e-9
STEM = 'With no Qwen3 pass at test time, training adds over the plain embedding in'

# Claims (spec §6, §3.4 "Cells per claim test"). `cells`: the expected (set, aspect, measure) keys; a photo cell has no aspect
# (None). `source`: measure -> the cell file that holds it. `gates`: source -> the gate files that park its cells (§3.5).
_ART = [('artelingo', a, m) for a in ASPECTS for m in IMG_MEASURES + TXT_MEASURES]
_ART_IMG = [('artelingo', a, m) for a in ASPECTS for m in IMG_MEASURES]
CLAIMS = {
    'SC3': dict(cells=_ART, most=12, stage2=False, data='ArtELingo test', sets=('artelingo',),
                source={m: ('image' if m in IMG_MEASURES else 'text') for m in IMG_MEASURES + TXT_MEASURES}),
    'SC4': dict(cells=_ART_IMG, most=5, stage2=True, data='ArtELingo test', sets=('artelingo',),
                source={m: 'image' for m in IMG_MEASURES}),
    'SC5': dict(cells=[('mmart', a, m) for a in ASPECTS for m in IMG_MEASURES], most=5, stage2=True, data='MMArt test',
                sets=('mmart',), source={m: 'mmart' for m in IMG_MEASURES}),
    'SC6': dict(cells=[(s, None, m) for s in ('affection', 'emoset') for m in IMG_MEASURES], most=3, stage2=True,
                data='Affection and EmoSet test', sets=('affection', 'emoset'), source={m: 'photo' for m in IMG_MEASURES}),
}
# Gate parking (spec §3.5 "What a failed gate parks"; ruling of build 4). One table: gate id -> the claims whose outcomes it parks.
# A gate part with status `fail` or `parked` parks ALL cells of each claim listed (all cells of a claim count inconclusive).
#   (v) image, (v) text   the outcomes of SC3 to SC6 ("a failed (v) parks the outcomes of SC3 to SC6")
#   (v) MMArt rebuild, (vi) MMArt   SC5 ("a failed rebuild parks only the cells that use that comparator"; (vi) parks SC5)
#   (v) photo rebuild     SC6; the record cannot say which comparator's cells, and every SC6 cell set has a P1 and a Q1 comparator
#                         of the rebuilt kind, so all SC6 cells (alternative: only the cells of the failed comparator)
#   (vii) trainer, (vii) photo scorer, (viii) Affection store (ticket 12 and 13's parts)   SC6 (§3.5: (vii) parks SC6, (viii)
#                         the piece whose store fails)
#   (viii) store of encoder E's K* arm: SC4 of E only (not SigLIP 1), the gate `viii_<cap_source>` named by `store_gate`
GATE_PARKS = {
    'v_image': ('SC3', 'SC4', 'SC5', 'SC6'),
    'v_text': ('SC3', 'SC4', 'SC5', 'SC6'),
    'v_mmart': ('SC5',),
    'vi': ('SC5',),
    'v_photo': ('SC6',),
    'vii_trainer': ('SC6',),
    'vii_photo': ('SC6',),
    'viii_affection': ('SC6',),
}
GATE_STATUSES = ('pass', 'fail', 'parked')       # `fallback` is gate i_*'s status only; ticket 08 reads no i_* gate


def now():
    return datetime.now(TZ).strftime('%F %H:%M')


def records_dir():
    """Committed records, or OUT/records under a smoke run (R29_OUTPUTS set), as ticket 06."""
    return P.RECORDS if P.OUT == P.DEFAULT_OUT else P.OUT / 'records'


def encoder_of_arm(arm):
    base = arm.removesuffix('_prefix')
    assert base in ENCODER_OF, f'arm {arm!r}: not a caption source of siglip1, siglip2, clipl14'
    return ENCODER_OF[base]


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ================================================================================================ the ladder (pure)
def outcome(lo, r):
    """§3.4: WIN when lo > 0; LEVEL when not WIN and lo > -r; BELOW otherwise (ticket 06's and r11's rule)."""
    assert r is not None and np.isfinite(r) and r >= 0 and np.isfinite(lo), (lo, r)
    return W if lo > 0 else (LV if lo > -r else B)


def q1_claim(o):
    """§6: each Q1 cell is its own claim: WIN 'above', LEVEL 'level with', BELOW (and a cell not decided) none."""
    return {W: 'above', LV: 'level with'}.get(o)


def _level(w1, wq, n, most):
    if w1 == n:
        return 3 if wq >= most else 2
    return 1 if w1 >= most else 0


def ladder(p1, q1, most):
    """The level from the per-cell outcomes against P1 and Q1 (lists of the same length n = the claim's cell count)."""
    n = len(p1)
    assert len(q1) == n and 0 < most <= n and set(p1) | set(q1) <= set(OUTCOMES), (len(p1), len(q1), most)
    c = dict(p1_win=p1.count(W), p1_inconclusive=p1.count(INC), q1_win_or_level=q1.count(W) + q1.count(LV),
             q1_win=q1.count(W), q1_level=q1.count(LV), q1_inconclusive=q1.count(INC))
    level = _level(c['p1_win'], c['q1_win_or_level'], n, most)
    best = _level(c['p1_win'] + c['p1_inconclusive'], c['q1_win_or_level'] + c['q1_inconclusive'], n, most)   # D22
    above = best > level
    return dict(level=level, inconclusive_above=above, counts=c, n_cells=n, most=most,
                report=f'level {level}' + (', inconclusive above' if above else ''))


def sentence(level, n, w1, wq):
    """§6's sentence of the level reached with only its slots filled (constitution row 8). Level 1: N of n (N = WIN over P1).
    Level 2: no slot. Level 3: N of n (N = WIN or LEVEL over Q1). Level 0: no sentence."""
    if level == 0:
        return None
    if level == 1:
        assert wq is None
        return f'{STEM} {w1} of {n} cells.'
    assert w1 == n
    if level == 2:
        assert wq is None
        return f'{STEM} every cell.'
    return f'{STEM} every cell and it is level with or above the instructed Qwen3 in {wq} of {n} cells.'


# ================================================================================================ reading cell records
def cell_name(key):
    s, a, m = key
    return f'{s}/{a}/{m}' if a else f'{s}/{m}'


def normalize(raw, arm, encoder, claim, source):
    """One cell record -> (key, 'P1'|'Q1', dict). Asserts the fields and the id (ticket 06 and 07 shapes)."""
    for f in ('id', 'type', 'protocol', 'comparator', 'aspect', 'arm', 'value', 'd', 'r', 'outcome'):
        assert f in raw, f'cell {raw.get("id")!r} lacks field {f!r}'
    parts = raw['id'].split('/')
    assert len(parts) == 6, f'cell id {raw["id"]!r}: not <set>/<encoder>/<arm>/<aspect>/<measure>/<comparator>'
    set_, enc, arm_, aspect, mid, comp = parts
    assert (enc, arm_, aspect, comp) == (encoder, arm, raw['aspect'], raw['comparator']) and raw['arm'] == arm, raw['id']
    for f, v in (('set', set_), ('encoder', enc)):
        assert f not in raw or raw[f] == v, (raw['id'], f)
    assert comp in COMPARATOR, f'cell {raw["id"]}: comparator {comp!r}'
    assert (comp in ('p1', 'q1')) == (source == 'text'), \
        f'cell {raw["id"]}: comparator {comp!r}; text cells say p1 / q1, image cells plain / teacher (P1 = plain = p1, Q1 = teacher = q1)'
    measure = 'ami' if mid.startswith('ami') and mid[3:].isdigit() else mid
    assert measure in IMG_MEASURES + TXT_MEASURES, f'cell {raw["id"]}: measure {mid!r}'
    spec = CLAIMS[claim]
    assert set_ in spec['sets'], f'cell {raw["id"]}: set {set_!r} is not in {claim}'
    key = (set_, None if claim == 'SC6' else aspect, measure)
    assert key in spec['cells'], f'cell {raw["id"]}: not one of the {len(spec["cells"])} cells of {claim}'
    o = str(raw['outcome']).upper()
    assert o in OUTCOMES, f'cell {raw["id"]}: outcome {raw["outcome"]!r}'
    assert 'lo' in raw and 'hi' in raw, f'cell {raw["id"]}: no interval (lo, hi)'
    lo, hi = raw['lo'], raw['hi']
    assert 'direction' not in raw or raw['direction'] == measure, (raw['id'], raw['direction'])
    if o != INC:
        nums = [raw['value'], raw['d'], raw['r'], lo, hi]
        assert all(v is not None and np.isfinite(v) for v in nums), f'cell {raw["id"]}: non-finite numbers {nums}'
        assert o == outcome(lo, raw['r']), f'cell {raw["id"]}: outcome {o} is not the rule\'s word for lo {lo}, r {raw["r"]}'
    cell = dict(id=raw['id'], outcome=o, value=raw['value'], d=raw['d'], lo=lo, hi=hi, r=raw['r'], type=raw['type'],
                protocol=raw['protocol'])
    return key, COMPARATOR[comp], cell


def read_source(source, claim, arm, encoder, records):
    """The cells of one cell file: {(key, 'P1'|'Q1'): cell}, or None when the file is absent (not run)."""
    path = Path(records) / f'{source}_cells_{arm}.json'
    if not path.exists():
        return None, None
    rec = json.loads(path.read_text())
    assert rec.get('arm') == arm and rec.get('encoder') == encoder and isinstance(rec.get('cells'), list), \
        f'{path}: arm {rec.get("arm")!r}, encoder {rec.get("encoder")!r}; asked {arm!r}, {encoder!r}'
    out = {}
    for raw in rec['cells']:
        key, comp, cell = normalize(raw, arm, encoder, claim, source)
        assert CLAIMS[claim]['source'][key[2]] == source, f'{path}: cell {raw["id"]} belongs to another cell file'
        assert (key, comp) not in out, f'{path}: cell {raw["id"]} twice'
        out[(key, comp)] = cell
    return out, dict(path=str(path), sha256=sha_file(path), n_cells=len(out))


def gate_status(records, gate, open_ok=False):
    """Status of records/gates/<gate>.json: pass, fail or parked. An absent record raises "gate <id> not run"; any other status
    raises (`fallback` is gate i_*'s only). A `fail` with tries < 3 is still open (§3.5: at most three tries; a gate that still
    fails parks what it serves): it raises "still open" (the ladder is write-once, so it must not freeze a claim the run can still
    fix), or with `open_ok` (the sheet) returns 'open'. A `fail` with tries >= 3 parks, as `parked` does."""
    path = Path(records) / 'gates' / f'{gate}.json'
    if not path.exists():
        raise RuntimeError(f'gate {gate} not run: no {path}')
    g = json.loads(path.read_text())
    assert g.get('gate') == gate and 'status' in g, path
    assert g['status'] in GATE_STATUSES, f'{path}: status {g["status"]!r}, one of {GATE_STATUSES}'
    tries = int(g.get('tries', 0))
    if g['status'] == 'fail' and tries < 3:
        if open_ok:
            return 'open'
        raise RuntimeError(f'gate {gate} still open (fail at try {tries} of 3): fix it or let it park before the ladder')
    return g['status']


def read_kstar_form(records, arm):
    """`records/prefix_pick.json` (ticket 05; keys kstar, kstar_form, p1_form, flag): K*'s caption form for SC4's store gate."""
    path = Path(records) / 'prefix_pick.json'
    if not path.exists():
        raise RuntimeError(f'no {path}: ticket 05 writes it (SC4 needs K*\'s caption form to name its store gate)')
    doc = json.loads(path.read_text())
    for key in ('kstar', 'kstar_form', 'p1_form', 'flag'):
        assert key in doc, f'{path}: key {key} missing (keys {sorted(doc)})'
    assert doc['kstar_form'] in ('plain', 'prefixed'), f'{path}: kstar_form {doc["kstar_form"]!r}'
    return doc['kstar_form']


def store_gate(encoder, form):
    """SC4's store gate `viii_<cap_source>`: the `r29_paths.CAPTION_SOURCES` name of the store that encoder's K* arm reads, through
    `r29_paths.ENCODER_CAPTION_SOURCES` (qwen, plain, prefixed) and K*'s form. None for SigLIP 1 (its store gates are ticket 02's)."""
    if encoder == 'siglip1':
        return None
    _qwen, plain, prefixed = P.ENCODER_CAPTION_SOURCES[encoder]
    src = {'plain': plain, 'prefixed': prefixed}[form]
    assert src in P.CAPTION_SOURCES, src
    return f'viii_{src}', src


def needed_gates(claim, arm, encoder, records, strict=True):
    """The gate ids that can park `claim` for this arm, in the order of GATE_PARKS then the SC4 store gate. A sheet (not strict)
    skips the store gate when prefix_pick.json is absent."""
    gates = [g for g, claims in GATE_PARKS.items() if claim in claims]
    if claim == 'SC4' and encoder != 'siglip1':
        if strict or (Path(records) / 'prefix_pick.json').exists():
            gate, src = store_gate(encoder, read_kstar_form(records, arm))
            assert arm == src, f'SC4 arm {arm!r}: encoder {encoder!r} with K*\'s form reads the store {src!r}'
            gates.append(gate)
    return gates


def claim_gates(claim, arm, encoder, records, strict=True):
    """{gate id: status} of the gates the claim needs. strict: an absent record raises "gate <id> not run" (the ladder); the sheet
    shows what exists."""
    out = {}
    for g in needed_gates(claim, arm, encoder, records, strict):
        if strict or (Path(records) / 'gates' / f'{g}.json').exists():
            out[g] = gate_status(records, g, open_ok=not strict)
    return out


def parked_by(gates):
    """The first gate whose part is `parked`, or `fail` at 3 tries (an open fail never gets here: `gate_status`), or None."""
    return next((g for g, st in gates.items() if st in ('fail', 'parked')), None)


def collect(claim, arm, encoder, records):
    """Every cell of the claim as {(key, comp): cell or None}, the source statuses and file hashes."""
    spec = CLAIMS[claim]
    got, status, files = {}, {}, {}
    for source in sorted(set(spec['source'].values())):
        cells, info = read_source(source, claim, arm, encoder, records)
        status[source] = 'not run' if cells is None else 'read'
        if info:
            files[source] = info
        got.update(cells or {})
    return {(k, c): got.get((k, c)) for k in spec['cells'] for c in ('P1', 'Q1')}, status, files


def is_committed(path, repo=None):
    """True unless `path` is in the checkout `repo` and differs from its content in git HEAD (a file elsewhere, as in a smoke run
    or a test folder, is not checked)."""
    path, repo = Path(path).resolve(), Path(repo or P.REPO).resolve()
    if not path.is_relative_to(repo):
        return True
    head = subprocess.run(['git', '-C', str(repo), 'show', f'HEAD:{path.relative_to(repo)}'], capture_output=True, text=True)
    return head.returncode == 0 and head.stdout == path.read_text()


def check_pass_rule(records, stage2, repo=None):
    """SC3 to SC6 refuse unless records/pass_rule.json is present and committed unchanged in HEAD (§3.5 row 1b: SC2 before any test
    number; presence and commit, not status, as ticket 07's `cells`, D9). A parked SC2 refuses: 1c and stage 2 do not run (§6).
    Stage 2 claims (SC4 to SC6) also need status pass (D3), or fail under a committed A1 record with R29_AMENDMENT=A1 (`P.stage2_gate`).
    Returns "pass" or "A1" for a stage 2 claim, None for SC3."""
    path = Path(records) / 'pass_rule.json'
    if not path.exists():
        raise RuntimeError(f'no {path}: pass_rule.json is written and committed before any test number (spec §3.5 row 1b)')
    if not is_committed(path, repo):
        raise RuntimeError(f'{path.name} is not committed unchanged in HEAD: commit it before the claim tests (spec §3.4, §3.5)')
    status = json.loads(path.read_text()).get('status')
    if status == 'parked':
        raise RuntimeError(f'{path}: SC2 is parked, so stage 1 is inconclusive: 1c does not run (spec §6)')
    if not stage2:
        return None
    try:
        return P.stage2_gate(records)
    except RuntimeError as exc:
        raise RuntimeError(f'{path}: stage 2 (SC4 to SC6) needs records/pass_rule.json with status pass, or fail under A1 (D3): {exc}') from exc


def build_record(claim, arm, encoder=None, records=None, missing_ok=None, repo=None):
    """The ladder record of one claim and arm (nothing written). `missing_ok`: a reason that lets a cell file be absent (it goes
    into the record); a cell file that is absent otherwise refuses the run, unless a gate parks the claim."""
    records = Path(records or records_dir())
    spec = CLAIMS[claim]
    enc = encoder_of_arm(arm)
    assert encoder is None or encoder == enc, f'arm {arm!r} is on encoder {enc!r}, asked {encoder!r}'
    if claim == 'SC3':
        assert enc == 'siglip1', f'SC3 is K*, on SigLIP 1; arm {arm!r} is on {enc!r}'
    if claim in ('SC5', 'SC6'):
        assert enc == 'siglip1', f'{claim} is K* / F1 on SigLIP 1; arm {arm!r} is on {enc!r}'
    gate = check_pass_rule(records, spec['stage2'], repo)
    gates = claim_gates(claim, arm, enc, records)
    parking = parked_by(gates)
    cells, status, files = collect(claim, arm, enc, records)
    if parking is None:                      # a cell missing from a file that exists is a writer bug (a whole missing file is below)
        for key in spec['cells']:
            for comp in ('P1', 'Q1'):
                if status[spec['source'][key[2]]] == 'read' and cells[(key, comp)] is None:
                    src = spec['source'][key[2]]
                    raise RuntimeError(f'{src}_cells_{arm}.json exists but lacks the cell {cell_name(key)} against {comp}: a writer bug, '
                                       'not a cell that was not run')
    absent = sorted(s for s, v in status.items() if v == 'not run')
    if absent and parking is None and not missing_ok:
        raise RuntimeError(f'{claim} {arm}: cell file(s) {absent} not written yet (`{absent[0]}_cells_{arm}.json`): run those cells first, '
                           'or give --missing-ok <reason> (the reason goes into the record)')
    rows, p1, q1 = [], [], []
    for key in spec['cells']:
        row = dict(cell=cell_name(key))
        for comp, lst in (('P1', p1), ('Q1', q1)):
            c = cells[(key, comp)]
            if parking is not None:
                o, why = INC, f'gate {parking} parked'
            elif c is None:
                o, why = INC, 'not run'
            else:
                o, why = c['outcome'], 'recorded inconclusive' if c['outcome'] == INC else None
            lst.append(o)
            row[comp.lower()] = o if why is None else f'{o} ({why})'
            row[f'{comp.lower()}_id'] = None if c is None else c['id']
        row['q1_claim'] = q1_claim(q1[-1])
        rows.append(row)
    lad = ladder(p1, q1, spec['most'])
    c, n, lvl = lad['counts'], lad['n_cells'], lad['level']
    assert n == len(spec['cells'])
    ids = lambda comp, outs: [r['cell'] for r, o in zip(rows, comp) if o in outs]
    rec = dict(claim=claim, arm=arm, encoder=enc, scope=f'{spec["data"]}, encoder {enc}', written=now(), n_cells=n,
               most=spec['most'], level=lvl, inconclusive_above=lad['inconclusive_above'], report=lad['report'],
               counts=c, sentence=sentence(lvl, n, c['p1_win'] if lvl else None, c['q1_win_or_level'] if lvl == 3 else None),
               counted=dict(p1_win=ids(p1, (W,)), q1_win_or_level=ids(q1, (W, LV)), p1_inconclusive=ids(p1, (INC,)),
                            q1_inconclusive=ids(q1, (INC,))),
               cells=rows, sources=status, cell_files=files, gates=gates, parked_by=parking)
    if missing_ok:
        rec['missing_ok'] = missing_ok
    rec.update(P.a1_marks(gate))
    return rec


def ladder_path(claim, arm, encoder, records):
    name = f'{claim}_{encoder}_{arm}.json' if claim == 'SC4' else f'{claim}_{arm}.json'
    return Path(records) / name


def write_ladder(claim, arm, encoder=None, records=None, missing_ok=None, repo=None):
    """records/<claim>_<arm>.json, once (a level is never edited: a new try needs the file moved aside by the controller)."""
    records = Path(records or records_dir())
    rec = build_record(claim, arm, encoder, records, missing_ok, repo)
    path = ladder_path(claim, arm, rec['encoder'], records)
    if path.exists():
        raise FileExistsError(f'{path} exists: a ladder record is written once (constitution row 1); the controller moves it for a new try')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + '\n')
    return path, rec


# ================================================================================================ K0's cells (frozen records)
def _j(path):
    return json.loads(Path(path).read_text())


def k0_values(encoder='siglip1'):
    """K0's test value per cell key: {(set, aspect, measure): dict(value, source)}, from r28's records (the sources of r28's
    paper_cells.md: per_seed.json, mmart.json, photos.json, r20 cells.json). For another encoder (SC4, descriptive) the
    ArtELingo AMI is r17's SC1 `cand` and the image mAP is r14's SC6 told cell `candidate_value`."""
    out = {}
    if encoder == 'siglip1':
        ps = _j(P.R28_RECORDS / 'per_seed.json')['ours_siglip1']
        for a in ASPECTS:
            for m, name in (('ami', 'ami'), ('map', 'i2i'), ('t2i', 't2i'), ('i2t', 'i2t'), ('t2t', 't2t')):
                out[('artelingo', a, m)] = dict(value=ps[f'{name}/{a}']['mean'], source=f'r28/per_seed.json :: ours_siglip1/{name}/{a}/mean')
        mm = _j(P.R28_RECORDS / 'mmart.json')['siglip1']
        for i, a in enumerate(ASPECTS):
            for m in IMG_MEASURES:
                out[('mmart', a, m)] = dict(value=mm[f'cstar_{m}']['values'][i], source=f'r28/mmart.json :: siglip1/cstar_{m}/values[{i}]')
        r20 = _j(P.OLDER / '20261111_encoder_photo_tables/records/cells.json')
        ph = _j(P.R28_RECORDS / 'photos.json')['cells']['a']
        for m in IMG_MEASURES:
            out[('affection', None, m)] = dict(value=r20[f'photo/affection/siglip1/photo/{m}']['value'],
                                               source=f'r20/cells.json :: photo/affection/siglip1/photo/{m}/value')
            out[('emoset', None, m)] = dict(value=ph[m]['mean'], source=f'r28/photos.json :: cells/a/{m}/mean')
        return out
    r17 = {c['id']: c for c in _j(P.R17_ROUND / 'records/verdicts.json')['cells']}
    r14 = _j(P.R14_ROUND / 'records/verdicts.json')['cells']
    for a in ASPECTS:
        k = {'emotion': 9, 'style': 27, 'genre': 10}[a]
        rid = f'SC1/artelingo/{encoder}/{a}/ami/{k}/plain'
        out[('artelingo', a, 'ami')] = dict(value=r17[rid]['cand'], source=f'r17/verdicts.json :: id={rid} => cand')
        (c,) = [c for c in r14 if (c['sc'], c['encoder'], c['aspect'], c['data'], c['task']) == ('SC6', encoder, a, 'artelingo_held', 'ret')]
        out[('artelingo', a, 'map')] = dict(value=c['candidate_value'], source=f'r14/verdicts.json :: SC6 {encoder}/{a}/ret candidate_value')
    return out


# ================================================================================================ the sheet
def f3(x):
    return f'{x:.3f}'


def f4(x):
    return f'{x:+.4f}'


def _cell_text(c, parked=None):
    if parked:
        return f'INCONCLUSIVE (gate {parked} parked)'
    if c is None:
        return 'not run'
    if c['outcome'] == INC:
        return 'INCONCLUSIVE'
    return f'{c["outcome"]} d {f4(c["d"])} [{f4(c["lo"])}, {f4(c["hi"])}] r {c["r"]:.4f}'


def discover_arms(records):
    """{source: [arm, ..]} of the cell files present."""
    out = {}
    for source in ('image', 'text', 'mmart', 'photo'):
        out[source] = sorted(p.name[len(source) + 7:-5] for p in Path(records).glob(f'{source}_cells_*.json'))
    return out


def _claim_arms(claim, arms):
    spec = CLAIMS[claim]
    found = sorted({a for s in set(spec['source'].values()) for a in arms[s]})
    enc = lambda a: encoder_of_arm(a)
    return [a for a in found if (enc(a) != 'siglip1') == (claim == 'SC4')]


def claim_section(claim, arm, records, k0_cache):
    spec = CLAIMS[claim]
    enc = encoder_of_arm(arm) if arm else ('siglip1' if claim != 'SC4' else None)
    k0 = k0_cache.setdefault(enc, k0_values(enc))
    lines = []
    parking = None
    if arm is None:
        cells = {(k, c): None for k in spec['cells'] for c in ('P1', 'Q1')}
        lines.append('New cells: not run (no cell record yet).\n')
        status = {}
    else:
        cells, status, _ = collect(claim, arm, enc, records)
        parking = parked_by(claim_gates(claim, arm, enc, records, strict=False))     # the ladder counts a parked claim inconclusive
        lines.append(f'Arm `{arm}` (encoder {enc}); cell files: ' + ', '.join(f'{s} {v}' for s, v in status.items()) + '.\n')
    dpath = Path(records) / (f'diag_d_{claim}_{enc}_{arm}.json' if claim == 'SC4' else f'diag_d_{claim}_{arm}.json')
    diag = {c['cell']: c for c in _j(dpath)['cells']} if arm and dpath.exists() else None
    head = '| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim |' + (' K* - K0 (d) |' if diag else '')
    lines += [head, '|---|---|---|---|---|---|' + ('---|' if diag else '')]
    for key in spec['cells']:
        p, q = cells[(key, 'P1')], cells[(key, 'Q1')]
        base = q or p
        if parking:
            new = ptxt = qtxt = f'(gate {parking} parked)'
            qclaim = 'inconclusive'
        else:
            new = 'not run' if base is None else ('inconclusive' if base['outcome'] == INC else f3(base['value']))
            ptxt, qtxt = _cell_text(p), _cell_text(q)
            if p is not None and q is not None and p['outcome'] != INC and abs(p['value'] - q['value']) > 1e-12:
                ptxt += f' (cell value {f3(p["value"])})'
            qclaim = 'not run' if q is None else ('inconclusive' if q['outcome'] == INC else (q1_claim(q['outcome']) or 'none'))
        k0v = k0.get(key)
        row = f'| {key[1] or key[0]} {key[2]} | '
        row += f'{f3(k0v["value"]) if k0v else "n/a"} | {new} | {ptxt} | {qtxt} | {qclaim} |'
        if diag:
            dd = diag.get(cell_name(key))
            row += f' {f4(dd["d"])} [{f4(dd["lo"])}, {f4(dd["hi"])}] |' if dd else ' not run |'
        lines.append(row)
    lpath = ladder_path(claim, arm, enc, records) if arm else None
    if lpath is not None and lpath.exists():
        lr = _j(lpath)
        lines.append(f'\nLadder ({claim}, `{lpath.name}`): **{lr["report"]}**; {lr["counts"]["p1_win"]} of {lr["n_cells"]} WIN over P1, '
                     f'{lr["counts"]["q1_win_or_level"]} WIN or LEVEL over Q1 (level 1 needs {lr["most"]} of {lr["n_cells"]}). '
                     + (f'Sentence: "{lr["sentence"]}"' if lr['sentence'] else 'No sentence (no level reached).'))
    else:
        lines.append(f'\nLadder ({claim}): not run (no ladder record).')
    return '\n'.join(lines) + '\n'


T5_ARMS = (('instructed_graphs', 'Instructed graphs (removal)'), ('content_correction', 'Content correction (removal)'),
           ('buddy', 'Graph-positive loss (removal)'), ('relational', 'Relational term (removal)'))


def _t5_star_rows(doc, d3=False):
    """The K* side of Table 5 from `records/ablation.json` (ticket 14; shapes of r29_ablation.score): {row: (AMI text, mAP text)}
    for 'ref' (K*'s own means) and each removal, plus 'notes'. A removal the record lists as inconclusive shows its cause and no
    number; a whole-record `inconclusive` (gate parked, or no removal scorable) shows one line with the cause."""
    names = ['ref'] + [a for a, _ in T5_ARMS]
    if doc is None:
        txt = 'not run (D3)' if d3 else 'not run'
        return {**{n: (txt, txt) for n in names}, 'notes': []}
    assert doc.get('claim') == 'SC7' and doc.get('status') in ('ok', 'partial', 'inconclusive'), doc.get('status')
    listed = doc.get('inconclusive_removals', {})
    notes = []
    if doc['status'] == 'inconclusive':
        notes.append(f'\nK* side of Table 5: INCONCLUSIVE ({doc["cause"]}).')
        notes += [f'- {a}: {c}' for a, c in listed.items()]
        return {**{n: ('inconclusive', 'inconclusive') for n in names}, 'notes': notes + ['']}
    star = doc['arms'][doc['kstar']]
    out = {'ref': tuple(' / '.join(f3(star[a][f'{m}_mean']) for a in ASPECTS) for m in ('ami', 'map'))}
    for arm_name, _ in T5_ARMS:
        if arm_name in listed:
            out[arm_name] = (f'INCONCLUSIVE: {listed[arm_name]}', 'inconclusive')
            notes.append(f'- {arm_name}: INCONCLUSIVE, no number: {listed[arm_name]}')
            continue
        d = doc['deltas_vs_kstar'][arm_name]
        out[arm_name] = tuple(' / '.join(f'{f4(d[a][m]["delta"])} [{f4(d[a][m]["lo"])}, {f4(d[a][m]["hi"])}]' for a in ASPECTS)
                              for m in ('ami', 'map'))
    return {**out, 'notes': (['\nInconclusive removals (D21: not scored; the others are):'] + notes + ['']) if notes else []}


def sheet_text(records):
    records = Path(records)
    arms = discover_arms(records)
    k0 = {'siglip1': k0_values('siglip1')}
    out = ['# r29 numbers sheet: the SigLIP-only adapter beside the current SigLIP 1 model (K0)\n',
           f'Written {now()} by `r29_ladder.py sheet` from the cell records present in `records/`; a missing piece shows `not run`. '
           'K0 values are r28\'s (`per_seed.json`, `mmart.json`, `photos.json`, r20 `cells.json`; for SC4 r17 SC1 and r14 SC6): '
           'five-seed means on test, three decimals. New cells: value (Q1 cell\'s value; for image mAP the aspect head, the P1 cell '
           'is the untold candidate and shows its own value), the paired difference d from the comparator with its 95% interval, '
           'r (the LEVEL unit) and the outcome by spec §3.4: WIN, LEVEL, BELOW, or INCONCLUSIVE. Q1 claim: WIN "above", LEVEL '
           '"level with", BELOW none. A failed pass rule (D9) does not hide the cells of SC3: they stay on this sheet. The sheet decides nothing: the levels '
           'are the ladder records\'.\n',
           '"Level" lines come from `records/<claim>_<arm>.json`; "K* - K0" is diagnostic (d), measured after the levels.\n']
    rule = _j(records / 'pass_rule.json') if (records / 'pass_rule.json').exists() else None
    d3 = rule is not None and rule['status'] != 'pass'          # D3: SC4 to SC7 do not run unless the pass rule passes ...
    a1 = False
    if d3:                                                      # ... or fails under a committed A1 record (R29_AMENDMENT=A1)
        try:
            a1 = P.stage2_gate(records) == 'A1'
        except RuntimeError:
            pass
        d3 = not a1
    out.append('Pass rule (SC2): ' + ('not run.' if rule is None else f'status `{rule["status"]}`, level {rule.get("level")}.'
               + (' SC4 to SC7: not run (D3).' if d3 else f' {P.A1_LINE}.' if a1 else '')) + '\n')
    titles = {'SC3': 'SC3 (ArtELingo test, 15 cells; Tables 2 and 3): K* against P1 (plain embedding) and Q1 (instructed Qwen3)',
              'SC4': 'SC4 (Table 6, ArtELingo test, 6 cells per encoder; K0 on that encoder shown beside, descriptive)',
              'SC5': 'SC5 (Table 4, MMArt test, 6 cells)',
              'SC6': 'SC6 (Table 7, photo model F1, 4 cells: Affection and EmoSet test)'}
    for claim in ('SC3', 'SC4', 'SC5', 'SC6'):
        out.append(f'## {titles[claim]}\n')
        if d3 and claim != 'SC3':
            out.append('Not run (D3): the pass rule did not pass.\n')
        claim_arms = _claim_arms(claim, arms) or [None]
        for arm in claim_arms:
            if claim == 'SC4' and arm is None:
                out.append('No encoder arm yet: not run.\n')
                out.append(f'| Cell | K0 (r28) | new value | vs P1 | vs Q1 | Q1 claim |\n|---|---|---|---|---|---|')
                kk = k0_values('siglip2')
                for key in CLAIMS['SC4']['cells']:
                    out.append(f'| {key[1]} {key[2]} | {f3(kk[key]["value"])} (SigLIP 2 K0) | not run | not run | not run | not run |')
                out.append('\nLadder (SC4): not run.\n')
                continue
            out.append(claim_section(claim, arm, records, k0))
    # Table 5 and Table 8: K0's rows; the new side is written by tickets 14 and 09
    new5 = records / 'ablation.json'
    doc5 = _j(new5) if new5.exists() else None
    star_title = 'not run (D3)' if d3 and doc5 is None else ('not run' if doc5 is None else 'from `ablation.json`')
    out.append(f'## Table 5 (ablation, ArtELingo test): K0 rows; removals of K*: {star_title}\n')
    ref, abl = _j(P.R28_RECORDS / 'ablation_ref_replay.json'), _j(P.R28_RECORDS / 'ablation_siglip1.json')
    out += ['| Row | K0 AMI emotion / style / genre | K0 mAP emotion / style / genre | K* AMI emotion / style / genre | K* mAP emotion / style / genre |',
            '|---|---|---|---|---|']
    star = _t5_star_rows(doc5, d3)
    out.append('| CoSiR (ours) = c* | ' + ' / '.join(f3(ref['ami'][a]['mean']) for a in ASPECTS) + ' | '
               + ' / '.join(f3(ref['map_plain_cosine'][a]['mean']) for a in ASPECTS) + f' | {star["ref"][0]} | {star["ref"][1]} |')
    for arm_name, label in T5_ARMS:
        d = abl['deltas_vs_cstar'][arm_name]
        out.append(f'| {label}, change from c* (K*: change from K*, paired 95% interval) | '
                   + ' / '.join(f4(d[a]['ami']['delta']) for a in ASPECTS) + ' | '
                   + ' / '.join(f4(d[a]['map']['delta']) for a in ASPECTS) + f' | {star[arm_name][0]} | {star[arm_name][1]} |')
    out += star['notes']
    out.append('\n## Table 8 (cost, ms per caption, one RTX 3090, batch 32)\n')
    tt, new8 = _j(P.R28_RECORDS / 'time_text.json'), records / 'time_text.json'
    n8 = _j(new8) if new8.exists() else None
    ms = lambda d, k: f3(d[k]['ms_per_caption_mean_of_repeats']) if d and k in d else 'not run'
    form8 = f' (K* caption form: {n8["kstar_form"]})' if n8 and 'kstar_form' in n8 else ''
    out += ['| Row | K0 (r28) | new |', '|---|---|---|',
            f'| Qwen3 text pass | {ms(tt, "qwen_text_pass")} | n/a (no Qwen3 pass) |',
            f'| SigLIP 1 text pass{form8} | n/a | {ms(n8, "siglip1_text_pass")} |',
            f'| Caption tower, three aspects | {ms(tt, "caption_tower_three_aspects")} | {ms(n8, "caption_tower_three_aspects")} |', '']
    return '\n'.join(out)


def write_sheet(records=None, out=None):
    records = Path(records or records_dir())
    out = Path(out or (P.ROUND / 'paper_cells.md' if records == P.RECORDS else P.OUT / 'paper_cells.md'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet_text(records))
    return out


# ================================================================================================ diagnostic (d)
def painting_interval_seeded(cand, comp, pid, draws=10000):
    """(d, lo, hi, n_units): paired bootstrap of candidate minus comparator, both float64 [k, nq] (k seeds), over paintings then
    seeds, as r23's `tcl_metrics.painting_interval` but with a seeded comparator: per draw the paintings (all of a painting's
    queries move together), then the candidate's seed draw, then the comparator's (the two arms are different trainings, so
    their seeds are drawn apart: r17's 'apart' rule). Same RNG: default_rng(0), draws in chunks of 250."""
    assert cand.ndim == 2 and comp.ndim == 2 and cand.shape == comp.shape and cand.dtype == comp.dtype == np.float64
    k, nq = cand.shape
    assert pid.shape == (nq,) and pid.dtype == np.int64 and np.isfinite(cand).all() and np.isfinite(comp).all()
    units, inverse, counts = np.unique(pid, return_inverse=True, return_counts=True)
    m = len(units)
    counts = counts.astype(np.float64)
    cols = np.vstack([np.stack([np.bincount(inverse, weights=row, minlength=m) for row in arm]) for arm in (cand, comp)])
    rng = np.random.default_rng(0)
    boot = np.empty(draws)
    for lo in range(0, draws, 250):
        n = min(250, draws - lo)
        weights, sc, sm = np.empty((n, m)), np.empty((n, k), dtype=np.int64), np.empty((n, k), dtype=np.int64)
        for b in range(n):
            weights[b] = np.bincount(rng.integers(0, m, size=m), minlength=m)
            sc[b] = rng.choice(k, size=k, replace=True)
            sm[b] = rng.choice(k, size=k, replace=True)
        values = (weights @ cols.T) / (weights @ counts)[:, None]
        boot[lo:lo + n] = np.take_along_axis(values[:, :k], sc, axis=1).mean(1) - np.take_along_axis(values[:, k:], sm, axis=1).mean(1)
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return float(cand.mean() - comp.mean()), float(lower), float(upper), m


def diag_d(claim, arm, encoder=None, records=None, differs=None):
    """Diagnostic (d): per test cell, K* minus K0 with the paired interval of that cell's type. Refuses unless the ladder record
    of the claim exists (§3.6: measured after the levels). `differs`: measure -> f(key, arm, k0_entry) -> dict(value, d, lo, hi,
    n_draws, protocol) with value = K*'s value and d = value - K0's (asserted within 1e-9 against r28's record)."""
    records = Path(records or records_dir())
    enc = encoder_of_arm(arm)
    lpath = ladder_path(claim, arm, enc, records)
    if not lpath.exists():
        raise FileNotFoundError(f'no ladder record {lpath}: run `ladder --claim {claim} --arm {arm}` first (§3.6: (d) follows the levels)')
    lad = _j(lpath)
    assert lad['claim'] == claim and lad['arm'] == arm
    k0 = k0_values(enc)
    differs = differs if differs is not None else real_differs(claim, arm, enc, records)
    rows = []
    for key in CLAIMS[claim]['cells']:
        res = differs[key[2]](key, arm, k0[key])
        for f in ('value', 'd', 'lo', 'hi', 'n_draws', 'protocol'):
            assert f in res, (key, f)
        assert abs(res['value'] - k0[key]['value'] - res['d']) <= TOL, \
            f'{cell_name(key)}: K* {res["value"]} minus K0 {k0[key]["value"]} is not d {res["d"]}'
        assert np.isfinite([res['d'], res['lo'], res['hi']]).all() and res['lo'] <= res['hi'], (key, res)
        rows.append(dict(cell=cell_name(key), k_star=res['value'], k0=k0[key]['value'], k0_source=k0[key]['source'], d=res['d'],
                         lo=res['lo'], hi=res['hi'], n_draws=res['n_draws'], protocol=res['protocol']))
    rec = dict(claim=claim, arm=arm, encoder=enc, ladder_record=str(lpath), ladder_level=lad['level'], written=now(), cells=rows)
    path = Path(records) / (f'diag_d_{claim}_{enc}_{arm}.json' if claim == 'SC4' else f'diag_d_{claim}_{arm}.json')
    path.write_text(json.dumps(rec, indent=1) + '\n')
    return path, rec


def real_differs(claim, arm, encoder, records, workers=8, draws=None):
    """The scorers of diagnostic (d) on ArtELingo test: K* (this round's held exports) against K0 (r14's, the model on that
    encoder). Cell types as ticket 06 and 07 score them, with K0 as the comparator of the same type:
      ami   r17 `ns_clu.paired_cell`, both arms' five seeds' partitions (comparator seeded: drawn apart), 2,000 draws
      map   the aspect head at r11's stage-4 rule's pick (K*: ticket 06's picks record; K0: r14's recorded pick), per-query AP,
            `ns_ret.paired_ret(comp_seeds='apart')`, 10,000 draws (r14 Q1's cell type, the one beside K0's i2i value)
      t2i, i2t, t2t  r23's per-query held files at each arm's own picked read-out, `painting_interval_seeded`, 10,000 draws."""
    assert claim in ('SC3', 'SC4'), 'diagnostic (d) is built for ArtELingo cells (SC3, SC4); MMArt and photo come with tickets 11 and 13'
    P.use_older_rounds()
    import r29_image_cells as ic       # puts the older rounds on sys.path through r29_paths
    P.check_older_imports()
    root0 = P.K0_ROOT if encoder == 'siglip1' else P.R14_OUT / 'cstar' / encoder
    wk = ic.Wiring(P.OUT / 'cstar' / encoder, arm, encoder)
    w0 = ic.Wiring(root0, P.K0_ARM, encoder)
    picks = ic.read_picks(arm, records=records, require_committed=P.OUT == P.DEFAULT_OUT)
    pick0 = _j(P.R14_ROUND / f'records/stage1_{encoder}.json')['artret_picks']['cstar']['pick']
    draws = dict(ic.N_DRAWS, **(draws or {}))
    ns_clu, ns_clu_cells, ns_data, ns_paths, ns_ret = ic.ns_clu, ic.ns_clu_cells, ic.ns_data, ic.ns_paths, ic.ns_ret
    cache = {}

    def ami(key, arm_, k0):
        a = key[1]
        k = ns_paths.CLASS_COUNT[(ic.SET, a)]
        y = ns_data.labels(ic.SET, ic.SPLIT_HELD, a)
        with wk:
            cand = ns_clu_cells._cand_parts(ic.SET, ic.SPLIT_HELD, encoder, a, k)
        with w0:
            base = ns_clu_cells._cand_parts(ic.SET, ic.SPLIT_HELD, encoder, a, k)
        r = ns_clu.paired_cell(cand, base, y, k, r=None, metrics=('ami',), draws=draws['ami'], rng_seed=ic.AMI_RNG_SEED)['ami']
        return dict(value=r['cand'], d=r['d'], lo=r['lo'], hi=r['hi'], n_draws=r['n_draws'], protocol='r17 paired_cell, comparator seeded (apart)')

    def head_ap(w, pick, tag):
        if (tag, pick) not in cache:
            ro = ic.bs_artret.readout_of(pick, ic.rt_data.WIDTHS['cstar'])
            labels = ic.rt_data.labels(ic.SPLIT_HELD)
            get_x = lambda head, seed: ic.rt_readout.stage5_apply(ro, w.embed(w.root, 'val', head, seed), w.embed(w.root, 'held', head, seed))
            with w:
                cache[(tag, pick)] = ic.bs_artret.rt_run.score_arm(ic.bs_artret._args(P.OUT / 'diag_d' / encoder / tag / 'perquery', workers), 5,
                                                                    'cstar', [(a, a) for a in ASPECTS], get_x, labels, qe_k=ro.k)
        return cache[(tag, pick)]

    def mp(key, arm_, k0):
        a = key[1]
        ck, c0 = head_ap(wk, picks['readouts']['cstar']['pick'], arm)[(a, a)], head_ap(w0, pick0, 'k0')[(a, a)]
        assert np.array_equal(ck['rows'], c0['rows']), (a, 'K* and K0 were scored on other queries')
        p = ns_ret.paired_ret(ck['ap'], c0['ap'], None, draws['map'], comp_seeds='apart')
        return dict(value=p['cand'], d=p['d'], lo=p['lo'], hi=p['hi'], n_draws=p['n_draws'], protocol='r17 paired_ret, comp_seeds apart')

    def text(key, arm_, k0):
        return _text_diff(key, arm, records)
    return {'ami': ami, 'map': mp, 't2i': text, 'i2t': text, 't2t': text}


def _load_per_query(root, row, head, seed, direction, readout, aspect):
    P.use_older_rounds()
    import tcl_paths as P23
    P.check_older_imports()
    path = P23.score_path('art_held', P23.CAPTION, row, head, seed, direction, readout, out=root)
    with np.load(path, allow_pickle=False) as z:
        assert {'q_pid', f'ap__{aspect}'} <= set(z.files), path
        return z['q_pid'], z[f'ap__{aspect}']


def _text_diff(key, arm, records, draws=10000):
    """K* minus K0 on one text cell: per-query held AP of both at their own picked read-outs (K*: `text_<arm>_held.json`; K0:
    r28's `text.json`), same queries and scored masks (asserted), `painting_interval_seeded`."""
    P.use_older_rounds()
    import r29_text as T29
    P.check_older_imports()
    a, d = key[1], key[2]
    held = _j(Path(records) / f'text_{arm}_held.json')
    assert held['arm'] == arm
    ro_k = held['picks'][a][d]
    ro_0 = _j(P.R28_RECORDS / 'text.json')['ours_siglip1']['cells'][a][d]['readout']
    rows, pid, mask = {}, None, None
    for tag, root, ro in (('k', T29.score_root(arm), ro_k), ('0', Path(T29.T.OUT), ro_0)):
        ap = []
        for s in P.SEEDS:
            q, x = _load_per_query(root, 'cstar', a, s, d, ro, a)
            pid = q if pid is None else pid
            assert np.array_equal(q, pid), f'{tag}: query paintings differ ({a} {d} seed {s})'
            m = np.isfinite(x)
            mask = m if mask is None else mask
            assert np.array_equal(m, mask), f'{tag}: scored queries differ ({a} {d} seed {s})'
            ap.append(x)
        rows[tag] = np.stack(ap)
    dd, lo, hi, _ = painting_interval_seeded(rows['k'][:, mask].astype(np.float64), rows['0'][:, mask].astype(np.float64),
                                             pid[mask].astype(np.int64), draws)
    return dict(value=float(rows['k'][:, mask].astype(np.float64).mean()), d=dd, lo=lo, hi=hi, n_draws=draws,
                protocol=f'r23 painting unit, seeded comparator (apart); K* {ro_k}, K0 {ro_0}')


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('ladder', help='records/<claim>_<arm>.json from the cell records (written once)')
    s.add_argument('--claim', required=True, choices=sorted(CLAIMS))
    s.add_argument('--arm', required=True)
    s.add_argument('--encoder')
    s.add_argument('--missing-ok', metavar='REASON', help='let a cell file be absent; the reason goes into the record')
    sub.add_parser('sheet', help='rewrite paper_cells.md from every cell record present').add_argument('--out', type=Path)
    d = sub.add_parser('diag-d', help='K* minus K0 per test cell, after the ladder record (SC3 and SC4 only: SC5 and SC6 have no K0 paired path)')
    d.add_argument('--arm', required=True)
    d.add_argument('--claim', default='SC3', choices=['SC3', 'SC4'])
    d.add_argument('--encoder')
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    if args.cmd == 'ladder':
        path, rec = write_ladder(args.claim, args.arm, args.encoder, missing_ok=args.missing_ok)
        print(f'{args.claim} {args.arm}: {rec["report"]}; P1 WIN {rec["counts"]["p1_win"]}/{rec["n_cells"]}, Q1 WIN or LEVEL '
              f'{rec["counts"]["q1_win_or_level"]}/{rec["n_cells"]} -> {path}')
    elif args.cmd == 'sheet':
        print('wrote', write_sheet(out=args.out))
    else:
        path, rec = diag_d(args.claim, args.arm, args.encoder)
        print(f'diag (d) {args.claim} {args.arm}: {len(rec["cells"])} cells -> {path} ({time.monotonic() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
