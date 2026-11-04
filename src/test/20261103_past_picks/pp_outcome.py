"""r19 outcome code (ticket 11): cell outcomes, the five verdict kinds, the D17 teacher claims, the voids table, the
licensed and limit sentences.

Spec: docs/superpowers/specs/2026-11-03-past-picks-design.md, section 6 (whole), section 3.4 "Outcomes", the voids table
under section 3.5, D4, D15, D17, D24. Pattern: r15's `mt_outcome` (`Voids`, `decide`, copied in form, not imported: it keys
everything on r15's four encoders). Imported from r14's `bs_outcome`: `cell_outcome`, `paintings_verdict`,
`photos_verdict`, `sc6_claim`, `aspect_list` and the paintings and photos sentence texts.

CLI (no GPU, reads files only):
  python pp_outcome.py decide [--root OUT] [--status FILE] [--out FILE] [--smoke]
                              [--mmart-n N] [--semart-n N] [--semart-artists N]
    reads  OUT/stage2/cells_<part>.json for the parts artelingo_clu, artelingo_ret, affection, mmart_clu, mmart_ret,
           semart_clu, semart_ret (a missing file is fine when its cells are voided; any other cells_*.json is refused),
           OUT/stage3/cells_cost.json (one SC15 cell per encoder),
           records/status.json (outside smoke through `pp_paths.read_committed_record`; --status reads a file as it is)
    writes OUT/stage4/outcomes.json
  --smoke (or PP_SMOKE=1) lets `n_draws` and the counts differ from the protocol's. Nothing else changes.

CELLS (the shared interface, index "Cells"): r14's cell dict {sc, encoder, aspect, task, data, candidate, comparator,
candidate_value, comparator_value, d, lo, hi, r, r_source, n_draws, seeds, n_rows}. `r` is null on a WIN-only bar
(SC1 to SC5, SC7 to SC9, SC11 to SC13) and the unrounded LEVEL unit (> 0) on a teacher cell (SC6, SC10, SC14).
Outcome: WIN when lo > 0; LEVEL when not WIN and lo > -r; BELOW otherwise; a WIN-only bar is 'WIN' or 'not WIN'.
  SC1 to SC3 ArtELingo held, 3 aspects (SC1, SC3 clu; SC2 ret); SC6 ArtELingo held 3 aspects x clu, ret
  SC4 (clu), SC5 (ret), SC6 (clu, ret): Affection test, emotion
  SC7 to SC9 (as SC1 to SC3), SC10 (as SC6): MMArt test          -> SC1 to SC10: the new three encoders only
  SC11 to SC13 (clu, ret, clu), SC14 (clu, ret): SemArt, genre, all seven encoders
  SC15: cost, one cell per encoder (`check_cost_cell`).

VERDICT KINDS (section 6; each (encoder, set) its own, D15 (4)):
  artelingo (SC1 to SC3) and mmart (SC7 to SC9), new three: holds when all nine WIN; qualified when the six frozen-image
    cells (SC1, SC2 / SC7, SC8) WIN and one or more PercepT cells (SC3 / SC9) do not; fails otherwise
  photos (SC4, SC5), new three: holds when both WIN, else fails
  semart (SC11 to SC13), all seven: holds when all three WIN; qualified when SC11 and SC12 WIN and SC13 does not; else fails
  cost (SC15), all seven: holds when c*'s slowest of three repeats is below the teacher's fastest, else fails

STATUS (written by ticket 12; values 'ok' | 'failed'; refused if a key is missing or extra):
  {"gates": {"i_store": {enc: {set: str}}      # dinov2_large: mmart, affection, semart; clip_b32: mmart, affection;
                                               # clip_b16: mmart, affection, semart; siglip1, clip_l14: semart
             "i_art_parity": {enc: str},       # the new three, and siglip1, clip_l14 (their new SemArt stores)
             "i_semart_parity": {"clip_b32", "qwen", "siglip2": str}, "i_semart_genre_word": str,
             "i_aff_l14_parity": str, "ii": str, "iii": {"r10", "r14": str}, "iv": str, "v": {new three: str},
             "vi": str, "vii": {"artelingo", "affection", "mmart": str}, "viii": str},
   "runs": {new three: {"cstar", "photo", "matched": str}}}       # a run is failed when it failed twice (r14 D18)
`runs[X].cstar` failed with `runs[X].photo` ok is refused (the photo model starts from X-c*'s emotion checkpoint).

VOIDS (section 3.5 table; D24). A voided verdict is 'inconclusive' (never 'qualified', no licensed sentence, no limit
sentence); a voided cell keeps its row with its outcome and status 'inconclusive'; a voided teacher claim is
'inconclusive'. Each row below voids what is listed (V = the verdicts, C = the cells; "teacher" = SC6, SC10, SC14):
  i_store[X][set]          V and C: X's verdict and teacher cells on that set (set: mmart, affection -> photos, semart)
  i_semart_genre_word      C: SC14 of all seven; the SemArt verdicts stand
  i_art_parity[X]          V and C: X on every set that has a new X store (the i_store entries of X); ArtELingo stands
  i_semart_parity[any]     V and C: every SemArt verdict and cell (incl. SC14), all seven
  i_aff_l14_parity         V and C: the new three's photos
  ii                       V and C: all four sets of the new three
  iii[r10 | r14]           as ii, plus SemArt of qwen (r10) or of siglip2, siglip1, clip_l14 (r14)
  iv                       V and C: everything except cost
  v[X], runs[X].matched    V: X's artelingo, mmart, semart; C: X's SC3, SC9, SC13 only (photos and teacher claims stand)
  vi                       V and C: every SemArt verdict and cell
  vii[artelingo]           V: the new three's artelingo and semart; C: their retrieval cells (SC2, SC6 ret; SC12, SC14
                           ret) on those sets; AMI teacher claims stand; the existing encoders' SemArt stands
  vii[affection]           V: the new three's photos; C: their retrieval cells (SC5, SC6 ret)
  vii[mmart]               V: the new three's mmart; C: their retrieval cells (SC8, SC10 ret)
  viii                     V and C: SC15 of all seven
  runs[X].cstar            V and C: X on all four sets and SC15 of X (the heads pass times the seed-0 c* network, which a
                           failed c* run does not leave)
  runs[X].photo            V and C: X's photos
A required cell may be missing only when a row above voids it; otherwise `decide` raises.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pp_paths as PP                                     # first: sets the older rounds' output roots
import bs_outcome as BO                                   # noqa: E402  r14: cell rule, verdict rules, sentence texts

assert tuple(BO.ASPECTS) == tuple(PP.ASPECTS)

ENCODERS, NEW, OLD, ASPECTS = PP.ENCODERS, PP.NEW_ENCODERS, PP.OLD_ENCODERS, PP.ASPECTS
ENCODER_NAME = PP.ENCODER_NAME
WIN, LEVEL, BELOW, NOT_WIN = BO.WIN, BO.LEVEL, BO.BELOW, BO.NOT_WIN
HOLDS, QUALIFIED, FAILS, INCONCLUSIVE = BO.HOLDS, BO.QUALIFIED, BO.FAILS, BO.INCONCLUSIVE
GO_OF = {HOLDS: 'GO', QUALIFIED: 'GO for the qualified claim', FAILS: 'NO-GO', INCONCLUSIVE: 'inconclusive (no claim)'}
cell_outcome = BO.cell_outcome
paintings_verdict = BO.paintings_verdict                  # holds: nine WIN; qualified: SC1, SC2 all WIN; else fails
photos_verdict = BO.photos_verdict
sc6_claim = BO.sc6_claim                                  # LEVEL or WIN

AE, AF, MM, SA = PP.DATA                                  # artelingo_held, affection_test, mmart_test, semart_new
KIND_OF_DATA = {AE: 'artelingo', AF: 'photos', MM: 'mmart', SA: 'semart'}
DATA_OF_KIND = {v: k for k, v in KIND_OF_DATA.items()}
KINDS = ('artelingo', 'photos', 'mmart', 'semart', 'cost')
KINDS_OF = {e: ('artelingo', 'photos', 'mmart', 'semart') for e in NEW} | {e: ('semart',) for e in OLD}
TEACHER = ('SC6', 'SC10', 'SC14')
MATCHED_SC = ('SC3', 'SC9', 'SC13')
CELL_KEYS = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator', 'candidate_value',
             'comparator_value', 'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'seeds', 'n_rows')
COST_KEYS = ('sc', 'encoder', 'cstar_ms', 'teacher_ms', 'cstar_ms_max', 'teacher_ms_min', 'cstar_ms_mean',
             'teacher_ms_mean', 'cstar_ms_e2e', 'teacher_ms_e2e')
SCS = {                          # sc: (tasks, datas, aspects (None: by data), encoders)
    'SC1': (('clu',), (AE,), ASPECTS, NEW), 'SC2': (('ret',), (AE,), ASPECTS, NEW),
    'SC3': (('clu',), (AE,), ASPECTS, NEW),
    'SC4': (('clu',), (AF,), ('emotion',), NEW), 'SC5': (('ret',), (AF,), ('emotion',), NEW),
    'SC6': (('clu', 'ret'), (AE, AF), None, NEW),
    'SC7': (('clu',), (MM,), ASPECTS, NEW), 'SC8': (('ret',), (MM,), ASPECTS, NEW),
    'SC9': (('clu',), (MM,), ASPECTS, NEW), 'SC10': (('clu', 'ret'), (MM,), ASPECTS, NEW),
    'SC11': (('clu',), (SA,), ('genre',), ENCODERS), 'SC12': (('ret',), (SA,), ('genre',), ENCODERS),
    'SC13': (('clu',), (SA,), ('genre',), ENCODERS), 'SC14': (('clu', 'ret'), (SA,), ('genre',), ENCODERS)}
PART_FILES = {'artelingo_clu': (AE, 'clu'), 'artelingo_ret': (AE, 'ret'), 'affection': (AF, None),
              'mmart_clu': (MM, 'clu'), 'mmart_ret': (MM, 'ret'), 'semart_clu': (SA, 'clu'), 'semart_ret': (SA, 'ret')}
COST_FILE = 'cells_cost.json'
N_MMART = 16384                                           # spec 3.3: r15's cut test set
# the new stores by encoder and set (the i_store entries; i_art_parity of X voids these sets)
STORE_SETS = {'dinov2_large': ('mmart', 'affection', 'semart'), 'clip_b32': ('mmart', 'affection'),
              'clip_b16': ('mmart', 'affection', 'semart'), 'siglip1': ('semart',), 'clip_l14': ('semart',)}
KIND_OF_STORE_SET = {'mmart': 'mmart', 'affection': 'photos', 'semart': 'semart'}
CKPT_FAMILY = {'r10': ('qwen',), 'r14': ('siglip2', 'siglip1', 'clip_l14')}
TASK_WORD = {'clu': 'clustering', 'ret': 'retrieval'}
DATA_WORD = {AE: '12,281 held ArtELingo paintings', AF: '8,501 held Affection photos', MM: 'MMArt',
             SA: "SemArt's new artists"}
SET_WORD = {'artelingo': 'ArtELingo held', 'photos': 'Affection test photos', 'mmart': 'MMArt test',
            'semart': "SemArt's new-artist set"}
CLAIM_BETTER, CLAIM_LEVEL, CLAIM_NONE, CLAIM_VOID = 'better than', 'level with or better than', None, 'inconclusive'
DECIDED, VOID = 'decided', 'inconclusive'


# ------------------------------------------------------------------------------------------------ licensed sentences
# Section 6, copied from the spec; the paintings and photos texts are r14's (`bs_outcome`), MMArt's head is r15's
# (`mt_outcome.HEAD`), the encoder sentence is item 16's. A test finds the fixed parts in the spec text.
ENC_SENTENCE = ("<X> is one of three encoders that failed or were unclear in an earlier held emotion test on ArtELingo "
                "with an earlier version of our model.")
PAINT_TAIL = " (AMI with 20 clusters; mAP with read-outs picked on validation; 5 seeds). " + ENC_SENTENCE
MMART_HEAD = ("On <n> WikiArt paintings outside ArtELingo (MMArt; exact and near copies of ArtELingo train and "
              "validation paintings removed), c*'s recipe with the <X> image as its only input at test (frozen; never "
              "trained or picked on these paintings) clusters and retrieves paintings by emotion (MMArt's labels), "
              "style and genre ")
MMART_TAIL = (" (AMI with 20 clusters; mAP with read-outs picked on validation paintings with MMArt labels; 5 seeds). "
              + ENC_SENTENCE)
SEMART_HEAD = ("On <n> paintings by <m> artists who have no painting in ArtELingo's train set (SemArt, from the Web "
               "Gallery of Art; artists matched by name; images at cosine ≥ 0.95 to an ArtELingo train or "
               "validation painting removed), frozen c* on <X> clusters and retrieves paintings by type (SemArt's 10 "
               "classes, close to genre) ")
SEMART_HOLDS_BODY = "better than the plain <X> image, and clusters them better than PercepT on the <X> image"
SEMART_QUAL_BODY = "better than the plain <X> image; against PercepT on the <X> image it is not better"
SEMART_TAIL = " (AMI with 20 clusters; mAP with read-outs picked on ArtELingo validation; 5 seeds)."
TEACHER_SENTENCE = "<model> is <claim> the Qwen image instructed for <aspect> in <task> on <data>."
COST_SENTENCE = ("On one RTX 3090 at batch 32, c* on <X> embeds a painting for emotion, style and genre in <a> ms (one "
                 "<X> pass and three small heads), against <b> ms for the Qwen image instructed per aspect (three "
                 "passes); model time, images decoded beforehand (<c> and <d> ms with decoding).")


def _fill(text, **kw):
    for k, v in kw.items():
        text = text.replace(f'<{k}>', str(v))
    assert '<' not in text, f'unfilled placeholder in {text!r}'
    return text


def paintings_sentence(enc, verdict, sc3_win_aspects=()):
    """ArtELingo (r14's form with item 16's encoder sentence)."""
    X = ENCODER_NAME[enc]
    if verdict == HOLDS:
        return _fill(BO.PAINT_HEAD + BO.PAINT_HOLDS_BODY + PAINT_TAIL, X=X)
    assert verdict == QUALIFIED
    if not sc3_win_aspects:
        return _fill(BO.PAINT_HEAD + BO.PAINT_QUAL_NONE + PAINT_TAIL, X=X)
    return _fill(BO.PAINT_HEAD + BO.PAINT_QUAL_BODY + PAINT_TAIL, X=X, aspects=BO.aspect_list(sc3_win_aspects))


def mmart_sentence(enc, verdict, n, sc9_win_aspects=()):
    """MMArt (r15's form with item 16's encoder sentence)."""
    X, n_txt = ENCODER_NAME[enc], f'{n:,}'
    if verdict == HOLDS:
        return _fill(MMART_HEAD + BO.PAINT_HOLDS_BODY + MMART_TAIL, n=n_txt, X=X)
    assert verdict == QUALIFIED
    if not sc9_win_aspects:
        return _fill(MMART_HEAD + BO.PAINT_QUAL_NONE + MMART_TAIL, n=n_txt, X=X)
    return _fill(MMART_HEAD + BO.PAINT_QUAL_BODY + MMART_TAIL, n=n_txt, X=X, aspects=BO.aspect_list(sc9_win_aspects))


def photos_sentence(enc):
    return _fill(BO.PHOTO_SENTENCE, X=ENCODER_NAME[enc])


def semart_sentence(enc, verdict, n, artists):
    X = ENCODER_NAME[enc]
    body = SEMART_HOLDS_BODY if verdict == HOLDS else SEMART_QUAL_BODY
    assert verdict in (HOLDS, QUALIFIED)
    return _fill(SEMART_HEAD + body + SEMART_TAIL, n=f'{n:,}', m=f'{artists:,}', X=X)


def teacher_sentence(enc, data, aspect, task, claim):
    assert claim in (CLAIM_BETTER, CLAIM_LEVEL)
    model = ('The photo model on ' if data == AF else 'c* on ') + ENCODER_NAME[enc]
    return _fill(TEACHER_SENTENCE.replace('<model>', model), claim=claim, aspect=aspect, task=TASK_WORD[task],
                 data=DATA_WORD[data])


def cost_sentence(enc, cell):
    f = lambda x: f'{x:.1f}'                                                      # noqa: E731
    return _fill(COST_SENTENCE, X=ENCODER_NAME[enc], a=f(cell['cstar_ms_mean']), b=f(cell['teacher_ms_mean']),
                 c=f(cell['cstar_ms_e2e']), d=f(cell['teacher_ms_e2e']))


def _and(items):
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


COMPARATOR_OF = {'SC1': 'plain', 'SC2': 'plain', 'SC4': 'plain', 'SC5': 'plain', 'SC7': 'plain', 'SC8': 'plain',
                 'SC11': 'plain', 'SC12': 'plain', 'SC3': 'percept', 'SC9': 'percept', 'SC13': 'percept'}


def limit_sentence(kind, enc, verdict, misses):
    """D4: the limit sentence of a fails or qualified verdict, naming the encoder and the cells that did not WIN.
    `misses`: [(sc, aspect, task)] of the cells that are not WIN."""
    assert verdict in (FAILS, QUALIFIED) and misses
    X = ENCODER_NAME[enc]
    subject = f'the photo model on {X}' if kind == 'photos' else f'c* on {X}'
    groups = []
    for cmp_, text in (('plain', f'the plain {X} image'), ('percept', f'PercepT on the {X} image')):
        cells = [f'{TASK_WORD[t]} for {a} ({sc})' for sc, a, t in misses if COMPARATOR_OF[sc] == cmp_]
        if cells:
            groups.append(f'{text} in {_and(cells)}')
    return f'Limit ({X}, {SET_WORD[kind]}, {verdict}): {subject} does not beat ' + '; and does not beat '.join(groups) + '.'


# ------------------------------------------------------------------------------------------------ cells
def cell_key(c):
    return (c['sc'], c['encoder'], c['aspect'], c['task'], c['data'])


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_cell(c, smoke=False):
    """Asserts at the file boundary: fields, SC / task / data / aspect / encoder pairing, finite numbers, `r` (null on a
    WIN-only bar, > 0 on a teacher cell), the protocol's draws."""
    missing = [k for k in CELL_KEYS if k not in c]
    assert not missing, f'cell lacks {missing}: {c}'
    where = cell_key(c)
    assert c['sc'] in SCS, f'unknown SC {where}'
    tasks, datas, aspects, encs = SCS[c['sc']]
    assert c['task'] in tasks, f'task {c["task"]!r} does not belong to {c["sc"]}: {where}'
    assert c['data'] in datas, f'data {c["data"]!r} does not belong to {c["sc"]}: {where}'
    if aspects is None:
        aspects = ASPECTS if c['data'] == AE else ('emotion',)
    assert c['aspect'] in aspects, f'aspect {c["aspect"]!r} does not belong to {c["sc"]} on {c["data"]}: {where}'
    assert c['encoder'] in encs, f'encoder {c["encoder"]!r} has no {c["sc"]} cell in this round: {where}'
    for k in ('candidate_value', 'comparator_value', 'd', 'lo', 'hi'):
        assert _num(c[k]), f'{k} not finite {where}'
    assert c['lo'] <= c['hi'], f'lo > hi {where}'
    if c['sc'] in TEACHER:
        assert _num(c['r']) and c['r'] > 0, f'{c["sc"]} is a teacher cell and needs a LEVEL unit r > 0, got {c["r"]!r}: {where}'
    else:
        assert c['r'] is None, f'{c["sc"]} is a WIN-only bar: r must be null, got {c["r"]!r}: {where}'
    assert isinstance(c['n_rows'], int) and not isinstance(c['n_rows'], bool) and c['n_rows'] > 0, f'n_rows {where}'
    if not smoke:
        assert c['n_draws'] == PP.N_DRAWS[c['task']], (
            f'n_draws {c["n_draws"]} != {PP.N_DRAWS[c["task"]]} for {where} (protocol draws; smoke allows others)')
    else:
        assert isinstance(c['n_draws'], int) and c['n_draws'] > 0, f'n_draws {where}'
    return where


def check_cost_cell(c):
    """SC15: three timings per side, the recorded maximum, minimum and means must follow from them."""
    missing = [k for k in COST_KEYS if k not in c]
    assert not missing, f'cost cell lacks {missing}: {c.get("encoder")}'
    enc = c['encoder']
    assert c['sc'] == 'SC15' and enc in ENCODERS, f'cost cell {c["sc"]} {enc}'
    for side in ('cstar_ms', 'teacher_ms'):
        v = c[side]
        assert isinstance(v, list) and len(v) == PP.PROBE_REPEATS and all(_num(x) and x > 0 for x in v), \
            f'{enc}: {side} must be {PP.PROBE_REPEATS} positive finite timings, got {v!r}'
    for k in ('cstar_ms_e2e', 'teacher_ms_e2e'):
        assert _num(c[k]) and c[k] > 0, f'{enc}: {k} {c[k]!r}'
    for k, f in (('cstar_ms_max', max(c['cstar_ms'])), ('teacher_ms_min', min(c['teacher_ms'])),
                 ('cstar_ms_mean', sum(c['cstar_ms']) / PP.PROBE_REPEATS),
                 ('teacher_ms_mean', sum(c['teacher_ms']) / PP.PROBE_REPEATS)):
        assert _num(c[k]) and abs(c[k] - f) <= 1e-6 * max(1.0, abs(f)), f'{enc}: {k} {c[k]!r} does not follow from the timings ({f})'
    return ('SC15', enc)


def cost_verdict(c):
    """Holds when c*'s slowest repeat is below the teacher's fastest (strict)."""
    return HOLDS if max(c['cstar_ms']) < min(c['teacher_ms']) else FAILS


def semart_verdict(sc11, sc12, sc13):
    """Holds: all three WIN. Qualified: SC11 and SC12 WIN and SC13 does not. Fails otherwise."""
    if (sc11, sc12) != (WIN, WIN):
        return FAILS
    return HOLDS if sc13 == WIN else QUALIFIED


def required_cell_keys(enc):
    """Every verdict and teacher cell of `enc`, in report order (new three 39, existing four 5)."""
    keys = []
    if enc in NEW:
        keys += [(sc, enc, a, SCS[sc][0][0], AE) for sc in ('SC1', 'SC2', 'SC3') for a in ASPECTS]
        keys += [('SC6', enc, a, t, AE) for a in ASPECTS for t in ('clu', 'ret')]
        keys += [('SC4', enc, 'emotion', 'clu', AF), ('SC5', enc, 'emotion', 'ret', AF),
                 ('SC6', enc, 'emotion', 'clu', AF), ('SC6', enc, 'emotion', 'ret', AF)]
        keys += [(sc, enc, a, SCS[sc][0][0], MM) for sc in ('SC7', 'SC8', 'SC9') for a in ASPECTS]
        keys += [('SC10', enc, a, t, MM) for a in ASPECTS for t in ('clu', 'ret')]
    keys += [(sc, enc, 'genre', SCS[sc][0][0], SA) for sc in ('SC11', 'SC12', 'SC13')]
    keys += [('SC14', enc, 'genre', t, SA) for t in ('clu', 'ret')]
    return keys


def load_cells(root):
    """The cell files under `root` (the run's OUT): (cells, cost cells, file names). Each stage 2 file must hold only
    its part's cells; a file that is not part of the rule is refused (`*.pending.json` temporaries are skipped)."""
    root = Path(root)
    cells, cost, files = [], [], []
    stage2, stage3 = root / 'stage2', root / 'stage3'
    for d, known in ((stage2, {f'cells_{p}.json' for p in PART_FILES}), (stage3, {COST_FILE})):
        if d.exists():
            stray = sorted(f.name for f in d.glob('cells_*.json') if not f.name.endswith('.pending.json')
                           and f.name not in known)
            assert not stray, f'cell files that are not part of the rule: {stray} in {d}'
    for part, (data, task) in PART_FILES.items():
        f = stage2 / f'cells_{part}.json'
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        assert isinstance(rows, list), f'{f}: not a list'
        for c in rows:
            assert c.get('data') == data and (task is None or c.get('task') == task), \
                f'{f.name} holds a cell of {c.get("data")!r}, {c.get("task")!r}: {c.get("sc")} {c.get("encoder")}'
        cells += rows
        files.append(f.name)
    f = stage3 / COST_FILE
    if f.exists():
        cost = json.loads(f.read_text())
        assert isinstance(cost, list), f'{f}: not a list'
        files.append(f.name)
    return cells, cost, files


# ------------------------------------------------------------------------------------------------ status
def check_status(status):
    """Every gate and run key present, values 'ok' | 'failed', no extra keys; photo runs cannot be ok when c* failed."""
    def keys(d, want, what):
        assert isinstance(d, dict) and set(d) == set(want), \
            f'status {what}: keys {sorted(d) if isinstance(d, dict) else d!r}, need {sorted(want)}'

    def val(v, what):
        assert v in ('ok', 'failed'), f'status {what}: {v!r} is not "ok" or "failed"'

    keys(status, ('gates', 'runs'), 'top level')
    g = status['gates']
    keys(g, ('i_store', 'i_art_parity', 'i_semart_parity', 'i_semart_genre_word', 'i_aff_l14_parity', 'ii', 'iii',
             'iv', 'v', 'vi', 'vii', 'viii'), 'gates')
    for k in ('i_semart_genre_word', 'i_aff_l14_parity', 'ii', 'iv', 'vi', 'viii'):
        val(g[k], f'gates.{k}')
    keys(g['i_store'], STORE_SETS, 'gates.i_store')
    for e, sets in g['i_store'].items():
        keys(sets, STORE_SETS[e], f'gates.i_store.{e}')
        for s, v in sets.items():
            val(v, f'gates.i_store.{e}.{s}')
    keys(g['i_art_parity'], set(NEW) | {'siglip1', 'clip_l14'}, 'gates.i_art_parity')
    keys(g['i_semart_parity'], ('clip_b32', 'qwen', 'siglip2'), 'gates.i_semart_parity')
    keys(g['iii'], ('r10', 'r14'), 'gates.iii')
    keys(g['v'], NEW, 'gates.v')
    keys(g['vii'], ('artelingo', 'affection', 'mmart'), 'gates.vii')
    for k in ('i_art_parity', 'i_semart_parity', 'iii', 'v', 'vii'):
        for e, v in g[k].items():
            val(v, f'gates.{k}.{e}')
    keys(status['runs'], NEW, 'runs')
    for e, r in status['runs'].items():
        keys(r, ('cstar', 'photo', 'matched'), f'runs.{e}')
        for k, v in r.items():
            val(v, f'runs.{e}.{k}')
        assert not (r['cstar'] == 'failed' and r['photo'] == 'ok'), \
            f'runs.{e}: photo runs cannot be ok when the c* runs failed (they start from its emotion checkpoint)'
    return status


class Voids:
    """What the failed gates and runs void (the voids table of section 3.5), as reason lists.

    A rule has a verdict scope {(kind, enc)} and a cell scope {(kind, enc)} with a mode: 'all' (every cell of the pair),
    'matched' (SC3, SC9, SC13), 'ret' (retrieval cells), 'teacher' (SC6, SC10, SC14)."""

    def __init__(self, status):
        check_status(status)
        g, runs = status['gates'], status['runs']
        self.rules = []                                        # (reason, verdict scope, cell scope, mode)
        bad = lambda v: v == 'failed'                          # noqa: E731

        def add(reason, verdicts, scope=None, mode='all'):
            self.rules.append((reason, set(verdicts), set(verdicts) if scope is None else set(scope), mode))

        def pairs(kinds, encs):
            return {(k, e) for e in encs for k in kinds if k in KINDS_OF[e]}

        for e, sets in g['i_store'].items():
            for s, v in sets.items():
                if bad(v):
                    add(f'gate i_store failed for {e} on {s}', {(KIND_OF_STORE_SET[s], e)})
        if bad(g['i_semart_genre_word']):
            add('gate i_semart_genre_word failed (SC14 of every encoder)', set(), pairs(('semart',), ENCODERS), 'teacher')
        for e, v in g['i_art_parity'].items():
            if bad(v):
                add(f'gate i_art_parity failed for {e}', {(KIND_OF_STORE_SET[s], e) for s in STORE_SETS[e]})
        for e, v in g['i_semart_parity'].items():
            if bad(v):
                add(f'gate i_semart_parity failed for {e}', pairs(('semart',), ENCODERS))
        if bad(g['i_aff_l14_parity']):
            add('gate i_aff_l14_parity failed', pairs(('photos',), NEW))
        if bad(g['ii']):
            add('gate ii failed', pairs(KINDS[:4], NEW) | {('cost', e) for e in NEW})    # "every verdict": SC15 too
        for fam, v in g['iii'].items():
            if bad(v):
                add(f'gate iii failed for {fam} checkpoints',
                    pairs(KINDS[:4], NEW) | {('cost', e) for e in NEW} | pairs(('semart',), CKPT_FAMILY[fam]))
        if bad(g['iv']):
            add('gate iv failed', pairs(KINDS[:4], ENCODERS))
        for e, v in g['v'].items():
            if bad(v):
                add(f'gate v failed for {e} (matched arm)', pairs(('artelingo', 'mmart', 'semart'), (e,)), mode='matched')
        if bad(g['vi']):
            add('gate vi failed (SemArt set)', pairs(('semart',), ENCODERS))
        for path, kinds in (('artelingo', ('artelingo', 'semart')), ('affection', ('photos',)), ('mmart', ('mmart',))):
            if bad(g['vii'][path]):
                add(f'gate vii failed on the {path} path (ranges or picks)', pairs(kinds, NEW), mode='ret')
                if path == 'artelingo':       # the existing encoders' SC14 mAP unit is a range from this path (D20)
                    add('gate vii failed on the artelingo path (SC14 retrieval units of the existing encoders)', set(),
                        pairs(('semart',), OLD), 'teacher_ret')
        if bad(g['viii']):
            add('gate viii failed (probe)', {('cost', e) for e in ENCODERS})
        for e, r in runs.items():
            if bad(r['cstar']):
                add(f'{e}: c* runs failed twice (r14 D18)', pairs(KINDS[:4], (e,)) | {('cost', e)})   # probe: seed-0 c*
            if bad(r['photo']):
                add(f'{e}: photo runs failed twice (r14 D18)', pairs(('photos',), (e,)))
            if bad(r['matched']):
                add(f'{e}: matched-arm runs failed twice (r14 D18)', pairs(('artelingo', 'mmart', 'semart'), (e,)),
                    mode='matched')

    def verdict(self, kind, enc):
        return [r for r, v, _, _ in self.rules if (kind, enc) in v]

    def cell(self, key):
        """Reasons for a verdict or teacher cell key (sc, enc, aspect, task, data)."""
        sc, enc, _, task, data = key
        pair = (KIND_OF_DATA[data], enc)
        out = []
        for reason, _, scope, mode in self.rules:
            hit = {'all': True, 'matched': sc in MATCHED_SC, 'ret': task == 'ret', 'teacher': sc in TEACHER,
                   'teacher_ret': sc in TEACHER and task == 'ret'}[mode]
            if pair in scope and hit:
                out.append(reason)
        return out

    def cost(self, enc):
        return self.verdict('cost', enc)


# ------------------------------------------------------------------------------------------------ decide
def teacher_claim(enc, aspect, task, data, outcome):
    """D17: a D17 cell at WIN gives 'better than'; a D17 cell at LEVEL, and any other cell at LEVEL or WIN, give 'level
    with or better than'; BELOW gives no claim."""
    assert outcome in (WIN, LEVEL, BELOW), f'teacher cell outcome {outcome!r}'
    if not sc6_claim(outcome):
        return CLAIM_NONE
    if outcome == WIN and (enc, aspect, task, data) in PP.D17:
        return CLAIM_BETTER
    return CLAIM_LEVEL


def decide(cells, status, cost_cells=(), counts=None, smoke=False):
    """The whole output (a dict, JSON-able). `cells`: verdict and teacher cells; `cost_cells`: SC15 cells; `counts`:
    {mmart_n, semart_n, semart_artists} for the sentences (needed only when a verdict that uses one is holds or
    qualified; outside smoke mmart_n must equal 16,384 and match the cells' n_rows, semart_n the SemArt cells')."""
    voids = Voids(status)
    counts = dict(counts or {})
    by_key = {}
    for c in cells:
        key = check_cell(c, smoke)
        assert key not in by_key, f'duplicate cell {key}'
        by_key[key] = c
    cost_by = {}
    for c in cost_cells:
        enc = check_cost_cell(c)[1]
        assert enc not in cost_by, f'duplicate cost cell {enc}'
        cost_by[enc] = c
    if not smoke:
        for key, c in by_key.items():
            if key[4] == MM and 'mmart_n' in counts:
                assert c['n_rows'] == counts['mmart_n'], f'{key}: n_rows {c["n_rows"]} != mmart_n {counts["mmart_n"]}'
            if key[4] == SA and 'semart_n' in counts:
                assert c['n_rows'] == counts['semart_n'], f'{key}: n_rows {c["n_rows"]} != semart_n {counts["semart_n"]}'
        if 'mmart_n' in counts:
            assert counts['mmart_n'] == N_MMART, f'mmart_n {counts["mmart_n"]} != {N_MMART} (spec 3.3; smoke allows others)'
    res = dict(smoke=bool(smoke), counts=counts, verdicts={k: {} for k in KINDS}, claims={}, cells=[], cost_cells=[])
    required = {e: required_cell_keys(e) for e in ENCODERS}
    allowed = {k for ks in required.values() for k in ks}
    rest = [k for k in by_key if k not in allowed]
    assert not rest, f'cells not part of the rule: {rest}'
    for e in ENCODERS:
        absent = [k for k in required[e] if k not in by_key and not voids.cell(k)]
        assert not absent, f'{e}: cells missing that no failed gate excuses: {absent}'
    absent = [e for e in ENCODERS if e not in cost_by and not voids.cost(e)]
    assert not absent, f'cost cells missing that no failed gate excuses: {absent}'

    def need(n, what):
        assert isinstance(counts.get(n), int) and not isinstance(counts.get(n), bool) and counts[n] > 0, \
            f'the licensed sentence of {what} needs counts[{n!r}] (a positive int), got {counts.get(n)!r}'
        return counts[n]

    oc = {}
    for e in ENCODERS:
        for k in required[e]:
            if k not in by_key:
                continue
            c = by_key[k]
            oc[k] = cell_outcome(c)
            why = voids.cell(k)
            res['cells'].append(dict(c, outcome=oc[k], status=VOID if why else DECIDED, void=why))
    for e in ENCODERS:
        for kind in KINDS_OF[e]:
            data = DATA_OF_KIND[kind]
            why = voids.verdict(kind, e)
            scs = {'artelingo': ('SC1', 'SC2', 'SC3'), 'mmart': ('SC7', 'SC8', 'SC9'), 'photos': ('SC4', 'SC5'),
                   'semart': ('SC11', 'SC12', 'SC13')}[kind]
            aspects = ('emotion',) if kind == 'photos' else ('genre',) if kind == 'semart' else ASPECTS
            vkeys = [(sc, e, a, SCS[sc][0][0], data) for sc in scs for a in aspects]
            if why:
                res['verdicts'][kind][e] = dict(verdict=INCONCLUSIVE, go=GO_OF[INCONCLUSIVE], reason=why,
                                                sentence=None, limit=None, outcomes=None)
                continue
            assert all(k in by_key and not voids.cell(k) for k in vkeys), f'{kind} {e}: verdict not void but a cell is'
            o = {(k[0], k[2]): oc[k] for k in vkeys}
            third_wins = [a for a in aspects if kind in ('artelingo', 'mmart') and o[(scs[2], a)] == WIN]
            if kind in ('artelingo', 'mmart'):
                v = paintings_verdict({(f'SC{i + 1}', a): o[(scs[i], a)] for i in range(3) for a in ASPECTS})
            elif kind == 'photos':
                v = photos_verdict(o[('SC4', 'emotion')], o[('SC5', 'emotion')])
            else:
                v = semart_verdict(o[('SC11', 'genre')], o[('SC12', 'genre')], o[('SC13', 'genre')])
            sentence = limit = None
            if v in (HOLDS, QUALIFIED):
                if kind == 'artelingo':
                    sentence = paintings_sentence(e, v, third_wins)
                elif kind == 'mmart':
                    sentence = mmart_sentence(e, v, need('mmart_n', f'{e} on MMArt'), third_wins)
                elif kind == 'photos':
                    sentence = photos_sentence(e)
                else:
                    sentence = semart_sentence(e, v, need('semart_n', f'{e} on SemArt'),
                                               need('semart_artists', f'{e} on SemArt'))
            if v in (FAILS, QUALIFIED):
                misses = [(k[0], k[2], k[3]) for k in vkeys if oc[k] != WIN]
                limit = limit_sentence(kind, e, v, misses)
            res['verdicts'][kind][e] = dict(verdict=v, go=GO_OF[v], reason=[], sentence=sentence, limit=limit,
                                            outcomes={f'{sc}/{a}': x for (sc, a), x in o.items()})
    for e in ENCODERS:
        claims = []
        for k in required[e]:
            if k[0] not in TEACHER:
                continue
            why = voids.cell(k)
            row = dict(sc=k[0], aspect=k[2], task=k[3], data=k[4], d17=(e, k[2], k[3], k[4]) in PP.D17,
                       outcome=oc.get(k), status=VOID if why else DECIDED, void=why)
            if why:
                row.update(claim=CLAIM_VOID, sentence=None)
            else:
                cl = teacher_claim(e, k[2], k[3], k[4], oc[k])
                row.update(claim=cl, sentence=teacher_sentence(e, k[4], k[2], k[3], cl) if cl else None)
            claims.append(row)
        res['claims'][e] = claims
    for e in ENCODERS:
        why = voids.cost(e)
        c = cost_by.get(e)
        if c is not None:
            res['cost_cells'].append(dict(c, status=VOID if why else DECIDED, void=why))
        if why:
            res['verdicts']['cost'][e] = dict(verdict=INCONCLUSIVE, go=GO_OF[INCONCLUSIVE], reason=why, sentence=None,
                                              limit=None, outcomes=None)
            continue
        v = cost_verdict(c)
        limit = None
        if v == FAILS:
            limit = (f"Limit ({ENCODER_NAME[e]}, cost, fails): c*'s slowest repeat ({max(c['cstar_ms']):.1f} ms) is not "
                     f"below the fastest repeat of the Qwen image instructed per aspect ({min(c['teacher_ms']):.1f} ms) "
                     "(SC15).")
        res['verdicts']['cost'][e] = dict(verdict=v, go=GO_OF[v], reason=[], sentence=cost_sentence(e, c) if v == HOLDS else None,
                                          limit=limit, outcomes=None)
    return res


# ------------------------------------------------------------------------------------------------ CLI
def table(res):
    lines = ['encoder       ' + ' '.join(f'{k:<13}' for k in KINDS) + 'teacher claims (better / level-or-better / none / void)']
    for e in ENCODERS:
        cells = []
        for k in KINDS:
            v = res['verdicts'][k].get(e)
            cells.append(f'{v["verdict"] if v else "-":<13}')
        cl = res['claims'][e]
        cnt = [sum(1 for c in cl if c['claim'] == x) for x in (CLAIM_BETTER, CLAIM_LEVEL, CLAIM_NONE, CLAIM_VOID)]
        lines.append(f'{e:<13} ' + ' '.join(cells) + f'{cnt[0]} / {cnt[1]} / {cnt[2]} / {cnt[3]}')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('decide', help='cells + records/status.json -> OUT/stage4/outcomes.json')
    p.add_argument('--root', default=str(PP.OUT))
    p.add_argument('--status', default=None, help='a status file read as it is (default: the committed record)')
    p.add_argument('--out', default=None)
    p.add_argument('--smoke', action='store_true', help="allow n_draws and counts other than the protocol's")
    p.add_argument('--mmart-n', type=int, default=N_MMART)
    p.add_argument('--semart-n', type=int, default=PP.N_SEMART, help='after the md5 step (pp_semart.record())')
    p.add_argument('--semart-artists', type=int, default=PP.SEMART_GATE['artists'])
    args = ap.parse_args(argv)
    smoke = bool(args.smoke or PP.SMOKE)
    cells, cost, files = load_cells(args.root)
    if args.status:
        status = json.loads(Path(args.status).read_text())
    elif smoke:
        status = json.loads(PP.record_path('status.json').read_text())
    else:
        status = PP.read_committed_record('status.json')
    counts = dict(mmart_n=args.mmart_n, semart_n=args.semart_n, semart_artists=args.semart_artists)
    res = decide(cells, status, cost, counts=counts, smoke=smoke)
    res = dict(cell_files=files, status=status, **res)
    out = Path(args.out) if args.out else Path(args.root) / 'stage4' / 'outcomes.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    print(table(res))
    print(f'{len(res["cells"])} cells and {len(res["cost_cells"])} cost cells from {len(files)} files -> {out}')


if __name__ == '__main__':
    main()
