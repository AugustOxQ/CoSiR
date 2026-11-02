"""r14 outcome code (ticket 09): cell outcomes, verdicts, inconclusive paths and licensed sentences.

Spec: docs/superpowers/specs/2026-11-02-backbones-sweep-design.md, §6 (whole), §3.4 "Outcomes" and "Seed ranges for
LEVEL"; D3, D4, D18. Rule review Q4 lists the patterns the tests cover.

CLI (no GPU, reads files only):
  python bs_outcome.py decide [--smoke] [--root OUT] [--status FILE] [--nearcopy FILE] [--out FILE]
    reads  OUT/stage3/cells_*.json   every file found (tickets 05 to 08; the parts are not listed here, the required
                                     cells are: a missing cell raises unless status excuses it)
           RECORDS/status.json       see STATUS SCHEMA
           RECORDS/nearcopy_emoset.json  only for <n> of part c's sentence (splits.test.n_kept)
    writes OUT/stage3/outcomes.json  every cell with its outcome, the verdicts, claims and sentences
  --smoke lets `n_draws` differ from the protocol's (2,000 clustering; 10,000 retrieval). Nothing else changes.

CELL OUTCOME (§3.4): WIN when lo > 0; LEVEL when not WIN and lo > -r; BELOW otherwise (strict, unrounded r).
A cell without `r` (a WIN-only bar: SC1 to SC5, SC7, SC8) is recorded 'WIN' or 'not WIN': LEVEL and BELOW need a unit.
SC6, SC9 and SC10 must carry `r` (SC6 is a LEVEL-or-WIN claim; SC9 and SC10 decide `qualified`): a cell without it raises.

STATUS SCHEMA (ticket 12 writes `RECORDS/status.json` exactly so; values 'ok' | 'failed'; 'failed' for runs means
failed twice, §3.4; for a gate it means the stage 0 or stage 2 gate failed):
  {"encoders": {"<enc>": {"cstar_runs":   "ok"|"failed",       # the five X-c* runs
                          "photo_runs":   "ok"|"failed",       # the five X photo runs
                          "stage0": {"paintings": "ok"|"failed",   # stage 0 gates on the ArtELingo path
                                     "photos":    "ok"|"failed"}}},  # stage 0 gates on the Affection photo path
                        for <enc> in siglip2, siglip1, clip_l14 (exactly these),
   "part_c": {"stage0": "ok"|"failed", "stage2": "ok"|"failed"}}
An X photo model starts from X-c*'s emotion checkpoint: `cstar_runs` failed with `photo_runs` ok raises.

INCONCLUSIVE (D18, §6): cstar_runs or stage0.paintings failed -> the encoder's paintings verdict and all 8 of its SC6
cells are inconclusive (no claim; the rows that exist are reported). photo_runs or stage0.photos failed -> its photos
verdict and its 2 photo SC6 cells are inconclusive; the paintings verdict stands. part_c: stage0 or stage2 failed ->
part c inconclusive. A path that is not failed needs all its cells: a missing cell raises.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import bs_paths as BP

ENCODERS = BP.NEW_ENCODERS
ASPECTS = BP.ASPECTS
ENC_NAME = {'siglip2': 'SigLIP 2', 'siglip1': 'SigLIP 1', 'clip_l14': 'CLIP L/14'}
CELL_KEYS = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator', 'candidate_value',
             'comparator_value', 'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'seeds')
N_DRAWS = {'clu': 2000, 'ret': 10000}                      # the protocols' draws (§3.4)
TASK_OF_SC = {'SC1': 'clu', 'SC2': 'ret', 'SC3': 'clu', 'SC4': 'clu', 'SC5': 'ret', 'SC7': 'clu', 'SC8': 'ret',
              'SC9': 'clu', 'SC10': 'ret'}                  # SC6 has both
DATA_OF_SC = {'SC1': 'artelingo_held', 'SC2': 'artelingo_held', 'SC3': 'artelingo_held', 'SC4': 'affection_test',
              'SC5': 'affection_test', 'SC7': 'emoset_test', 'SC8': 'emoset_test', 'SC9': 'emoset_test',
              'SC10': 'emoset_test'}                        # SC6: artelingo_held or affection_test
SC6_DATA = ('artelingo_held', 'affection_test')
NEEDS_R = ('SC6', 'SC9', 'SC10')
WIN, LEVEL, BELOW, NOT_WIN = 'WIN', 'LEVEL', 'BELOW', 'not WIN'
HOLDS, QUALIFIED, FAILS, INCONCLUSIVE = 'holds', 'qualified', 'fails', 'inconclusive'
GO_OF = {HOLDS: 'GO', QUALIFIED: 'GO for the qualified claim', FAILS: 'NO-GO', INCONCLUSIVE: 'inconclusive (no claim)'}

# ------------------------------------------------------------------------------------------------ licensed sentences
# §6, copied from the spec (a test compares each fragment with the spec text; placeholders as in the spec).
PAINT_HEAD = ("On 12,281 held ArtELingo paintings, c*'s recipe with the <X> image as its only input at test "
              "(Qwen3-VL-Embedding-2B's instructed graphs, captions and teacher at training) clusters and retrieves "
              "paintings by emotion, style and genre ")
PAINT_HOLDS_BODY = "better than the plain <X> image, and clusters them better than PercepT on the <X> image"
PAINT_QUAL_BODY = "better than the plain <X> image; against PercepT on the <X> image, better in <aspects>"
PAINT_QUAL_NONE = "better than the plain <X> image; against PercepT on the <X> image it is better in no aspect"
PAINT_TAIL = (" (AMI with 20 clusters; mAP with read-outs picked on validation; 5 seeds). <X> is one of three encoders "
              "that passed an earlier held emotion test.")
PHOTO_SENTENCE = ("On 8,501 held Affection photos, the same model on <X>, fine-tuned on Affection's unlabeled train "
                  "photos with the settings picked for Qwen, clusters and retrieves photos by emotion better than the "
                  "plain <X> image.")
SC6_SENTENCE = "<model on X> is level with or better than the Qwen image instructed for <aspect> in <task> on <data>."
PARTC_HEAD = ("On <n> EmoSet-118K test photos (photos with a near-copy in Affection's train set removed), the photo "
              "model (c*, trained on ArtELingo paintings, then fine-tuned on Affection's unlabeled train photos; "
              "frozen; never trained on EmoSet) clusters and retrieves photos by emotion ")
PARTC_HOLDS_BODY = ("better than the plain Qwen3-VL-Embedding-2B image and better than the Qwen image instructed for "
                    "emotion")
PARTC_PLAIN = "better than the plain Qwen3-VL-Embedding-2B image"
PARTC_QUAL_BODY = "better than the instructed image in <task> and level with it in <other task>"
PARTC_TAIL = (" (AMI with 20 clusters; mAP; wording and read-outs picked on EmoSet validation by the rule fixed in "
              "r12; 5 seeds).")
TASK_WORD = {'clu': 'clustering', 'ret': 'retrieval'}
DATA_WORD = {'artelingo_held': '12,281 held ArtELingo paintings', 'affection_test': '8,501 held Affection photos'}


def _fill(text, **kw):
    for k, v in kw.items():
        text = text.replace(f'<{k}>', v)
    assert '<' not in text, f'unfilled placeholder in {text!r}'
    return text


def aspect_list(aspects):
    """'emotion', 'emotion and style', 'emotion, style and genre' (the spec's own form, in aspect order)."""
    a = [x for x in ASPECTS if x in set(aspects)]
    assert a and len(a) == len(set(aspects)), aspects
    return a[0] if len(a) == 1 else ', '.join(a[:-1]) + ' and ' + a[-1]


def paintings_sentence(enc, verdict, sc3_win_aspects=()):
    X = ENC_NAME[enc]
    if verdict == HOLDS:
        return _fill(PAINT_HEAD + PAINT_HOLDS_BODY + PAINT_TAIL, X=X)
    assert verdict == QUALIFIED
    if not sc3_win_aspects:
        return _fill(PAINT_HEAD + PAINT_QUAL_NONE + PAINT_TAIL, X=X)
    return _fill(PAINT_HEAD + PAINT_QUAL_BODY + PAINT_TAIL, X=X, aspects=aspect_list(sc3_win_aspects))


def photos_sentence(enc):
    return _fill(PHOTO_SENTENCE, X=ENC_NAME[enc])


def sc6_sentence(enc, data, aspect, task):
    model = ('c* on ' if data == 'artelingo_held' else 'The photo model on ') + ENC_NAME[enc]
    return _fill(SC6_SENTENCE.replace('<model on X>', model), aspect=aspect, task=TASK_WORD[task], data=DATA_WORD[data])


def part_c_sentence(verdict, n, win_task=None):
    """`win_task` ('clu' | 'ret'): for `qualified`, the task of SC9 / SC10 that is a WIN (the other is LEVEL)."""
    n_txt = f'{int(n):,}'
    if verdict == HOLDS:
        return _fill(PARTC_HEAD + PARTC_HOLDS_BODY + PARTC_TAIL, n=n_txt)
    assert verdict == QUALIFIED and win_task in TASK_WORD
    other = 'ret' if win_task == 'clu' else 'clu'
    body = PARTC_PLAIN + ', and ' + PARTC_QUAL_BODY
    return _fill(PARTC_HEAD + body + PARTC_TAIL, n=n_txt, task=TASK_WORD[win_task], **{'other task': TASK_WORD[other]})


# ------------------------------------------------------------------------------------------------ cell outcome
def cell_outcome(cell):
    """WIN: lo > 0. LEVEL: not WIN and lo > -r. BELOW otherwise. Strict, unrounded. Without `r`: WIN or 'not WIN'."""
    lo, r = cell['lo'], cell.get('r')
    if lo > 0:
        return WIN
    if r is None:
        return NOT_WIN
    return LEVEL if lo > -r else BELOW


def check_cell(cell, smoke=False):
    """Cheap asserts at the file boundary: fields, finite numbers, the SC's task and data, the protocol's draws."""
    missing = [k for k in CELL_KEYS if k not in cell]
    assert not missing, f'cell lacks {missing}: {cell}'
    sc, task, data = cell['sc'], cell['task'], cell['data']
    where = (sc, cell['encoder'], cell['aspect'], task, data)
    assert sc in TASK_OF_SC or sc == 'SC6', f'unknown SC {where}'
    assert task in N_DRAWS, f'unknown task {where}'
    if sc == 'SC6':
        assert data in SC6_DATA, f'SC6 data {where}'
    else:
        assert task == TASK_OF_SC[sc] and data == DATA_OF_SC[sc], f'task or data wrong for {where}'
    for k in ('candidate_value', 'comparator_value', 'd', 'lo', 'hi'):
        assert isinstance(cell[k], (int, float)) and math.isfinite(cell[k]), f'{k} not finite {where}'
    assert cell['lo'] <= cell['hi'], f'lo > hi {where}'
    if cell['r'] is None:
        assert sc not in NEEDS_R, f'{sc} needs a LEVEL unit r (null) {where}'
    else:
        assert isinstance(cell['r'], (int, float)) and math.isfinite(cell['r']) and cell['r'] > 0, f'r {where}'
    if not smoke:
        assert cell['n_draws'] == N_DRAWS[task], (f'n_draws {cell["n_draws"]} != {N_DRAWS[task]} for {where} '
                                                  '(protocol draws; --smoke allows others)')
    return where


def cell_key(cell):
    return (cell['sc'], cell['encoder'], cell['aspect'], cell['task'], cell['data'])


def load_cells(stage3, smoke=False):
    """Every `cells_*.json` in `stage3`: a list of cells each; returns (cells, part names). Duplicates raise."""
    stage3 = Path(stage3)
    files = sorted(stage3.glob('cells_*.json'))
    cells, seen = [], {}
    for f in files:
        part = json.loads(f.read_text())
        assert isinstance(part, list), f'{f}: not a list'
        for c in part:
            key = check_cell(c, smoke)
            assert key not in seen, f'duplicate cell {key} in {f.name} and {seen[key]}'
            seen[key] = f.name
            cells.append(c)
    return cells, [f.name for f in files]


# ------------------------------------------------------------------------------------------------ status
def check_status(status):
    assert set(status) == {'encoders', 'part_c'}, f'status keys {sorted(status)}'
    assert set(status['encoders']) == set(ENCODERS), f'status encoders {sorted(status["encoders"])}'
    ok = ('ok', 'failed')
    for enc, s in status['encoders'].items():
        assert set(s) == {'cstar_runs', 'photo_runs', 'stage0'} and set(s['stage0']) == {'paintings', 'photos'}, enc
        for v in (s['cstar_runs'], s['photo_runs'], s['stage0']['paintings'], s['stage0']['photos']):
            assert v in ok, (enc, v)
        assert not (s['cstar_runs'] == 'failed' and s['photo_runs'] == 'ok'), \
            f'{enc}: photo runs cannot be ok when X-c* runs failed (they start from its checkpoint)'
    pc = status['part_c']
    assert set(pc) == {'stage0', 'stage2'} and all(v in ok for v in pc.values()), pc
    return status


def path_state(status, enc):
    """Which parts of an encoder are inconclusive: (paintings_inconclusive, photos_inconclusive, reasons)."""
    s = status['encoders'][enc]
    why_p = [n for n, bad in (('X-c* runs failed twice', s['cstar_runs'] == 'failed'),
                              ('stage 0 gate failed on the paintings path', s['stage0']['paintings'] == 'failed')) if bad]
    why_h = [n for n, bad in (('photo runs failed twice', s['photo_runs'] == 'failed'),
                              ('stage 0 gate failed on the photos path', s['stage0']['photos'] == 'failed')) if bad]
    return why_p, why_h


# ------------------------------------------------------------------------------------------------ verdicts (pure)
def paintings_verdict(outcome):
    """`outcome`: {(sc, aspect): outcome} for SC1, SC2, SC3 over the three aspects. holds: nine WIN. qualified: SC1 and
    SC2 all WIN, one or more SC3 not WIN. fails otherwise."""
    for sc in ('SC1', 'SC2', 'SC3'):
        for a in ASPECTS:
            assert (sc, a) in outcome, f'paintings verdict needs {(sc, a)}'
    if any(outcome[(sc, a)] != WIN for sc in ('SC1', 'SC2') for a in ASPECTS):
        return FAILS
    return HOLDS if all(outcome[('SC3', a)] == WIN for a in ASPECTS) else QUALIFIED


def photos_verdict(sc4, sc5):
    return HOLDS if (sc4, sc5) == (WIN, WIN) else FAILS


def part_c_verdict(sc7, sc8, sc9, sc10):
    if (sc7, sc8) != (WIN, WIN):
        return FAILS
    if (sc9, sc10) == (WIN, WIN):
        return HOLDS
    if {sc9, sc10} == {WIN, LEVEL}:
        return QUALIFIED
    return FAILS


def sc6_claim(outcome):
    """LEVEL or WIN gives the claim; BELOW gives none (the row stays)."""
    return outcome in (WIN, LEVEL)


# ------------------------------------------------------------------------------------------------ decide
def required_keys(enc, path):
    if path == 'paintings':
        keys = [(sc, enc, a, TASK_OF_SC[sc], 'artelingo_held') for sc in ('SC1', 'SC2', 'SC3') for a in ASPECTS]
        keys += [('SC6', enc, a, t, 'artelingo_held') for a in ASPECTS for t in ('clu', 'ret')]
    else:
        keys = [('SC4', enc, 'emotion', 'clu', 'affection_test'), ('SC5', enc, 'emotion', 'ret', 'affection_test'),
                ('SC6', enc, 'emotion', 'clu', 'affection_test'), ('SC6', enc, 'emotion', 'ret', 'affection_test')]
    return keys


def decide(cells, status, nearcopy_n=None):
    """The whole output (a dict, JSON-able). `cells`: checked cells; `nearcopy_n`: <n> for part c's sentence (needed
    only if part c holds or is qualified)."""
    check_status(status)
    by_key = {}
    for c in cells:
        assert cell_key(c) not in by_key, f'duplicate cell {cell_key(c)}'
        by_key[cell_key(c)] = c
    pc_cells = {c['sc']: c for c in cells if c['data'] == 'emoset_test'}
    known = {c['encoder'] for c in cells if c['data'] != 'emoset_test'}
    assert known <= set(ENCODERS), f'cells of unknown encoders {sorted(known - set(ENCODERS))}'
    out = dict(paintings={}, photos={}, sc6={}, part_c=None)
    used = set()

    def row(c, inconclusive):
        used.add(cell_key(c))
        return dict(c, outcome=cell_outcome(c), status='inconclusive' if inconclusive else 'decided')

    rows = []
    for enc in ENCODERS:
        why_p, why_h = path_state(status, enc)
        sc6_all_bad = bool(why_p)             # paintings-side failure takes every SC6 cell of the encoder
        for path, bad in (('paintings', bool(why_p)), ('photos', bool(why_h))):
            keys = required_keys(enc, path)
            if not bad:
                absent = [k for k in keys if k not in by_key and not (k[0] == 'SC6' and sc6_all_bad)]
                assert not absent, f'{enc}: cells missing for a path that did not fail: {absent}'
            for k in keys:
                if k in by_key:
                    rows.append(row(by_key[k], bad or (k[0] == 'SC6' and sc6_all_bad)))
        ocm = {k: cell_outcome(by_key[k]) for k in by_key if k[1] == enc}
        # paintings
        if why_p:
            out['paintings'][enc] = dict(verdict=INCONCLUSIVE, go=GO_OF[INCONCLUSIVE], sentence=None, reason=why_p)
        else:
            o = {(sc, a): ocm[(sc, enc, a, TASK_OF_SC[sc], 'artelingo_held')] for sc in ('SC1', 'SC2', 'SC3') for a in ASPECTS}
            v = paintings_verdict(o)
            wins = [a for a in ASPECTS if o[('SC3', a)] == WIN]
            out['paintings'][enc] = dict(
                verdict=v, go=GO_OF[v], reason=[], sc3_win_aspects=wins,
                sentence=None if v == FAILS else paintings_sentence(enc, v, wins))
        # photos
        if why_h:
            out['photos'][enc] = dict(verdict=INCONCLUSIVE, go=GO_OF[INCONCLUSIVE], sentence=None, reason=why_h)
        else:
            v = photos_verdict(ocm[('SC4', enc, 'emotion', 'clu', 'affection_test')],
                               ocm[('SC5', enc, 'emotion', 'ret', 'affection_test')])
            out['photos'][enc] = dict(verdict=v, go=GO_OF[v], reason=[],
                                      sentence=photos_sentence(enc) if v == HOLDS else None)
        # SC6: each cell its own claim
        sc6 = []
        for k in required_keys(enc, 'paintings')[9:] + required_keys(enc, 'photos')[2:]:
            if k not in by_key:
                continue
            bad = bool(why_p) or (bool(why_h) and k[4] == 'affection_test')
            oc = ocm[k]
            claim = (not bad) and sc6_claim(oc)
            sc6.append(dict(aspect=k[2], task=k[3], data=k[4], outcome=oc, status='inconclusive' if bad else 'decided',
                            claim=claim, sentence=sc6_sentence(enc, k[4], k[2], k[3]) if claim else None))
        out['sc6'][enc] = sc6
    # part c
    pc = status['part_c']
    why_c = [n for n, bad in (('stage 0 gate failed on the part c path', pc['stage0'] == 'failed'),
                              ('stage 2 gate failed', pc['stage2'] == 'failed')) if bad]
    pk = {'SC7', 'SC8', 'SC9', 'SC10'}
    if not why_c:
        absent = sorted(pk - set(pc_cells))
        assert not absent, f'part c: cells missing {absent}'
    assert set(pc_cells) <= pk, f'unknown EmoSet cells {sorted(set(pc_cells) - pk)}'
    for c in pc_cells.values():
        rows.append(row(c, bool(why_c)))
    if why_c:
        out['part_c'] = dict(verdict=INCONCLUSIVE, go=GO_OF[INCONCLUSIVE], sentence=None, reason=why_c)
    else:
        assert len({c['encoder'] for c in pc_cells.values()}) == 1, 'part c cells name different encoders'
        o = {sc: cell_outcome(c) for sc, c in pc_cells.items()}
        v = part_c_verdict(o['SC7'], o['SC8'], o['SC9'], o['SC10'])
        sentence, win_task = None, None
        if v in (HOLDS, QUALIFIED):
            assert nearcopy_n is not None, 'part c sentence needs <n> (near-copy record, test split n_kept)'
            if v == QUALIFIED:
                win_task = 'clu' if o['SC9'] == WIN else 'ret'
            sentence = part_c_sentence(v, nearcopy_n, win_task)
        out['part_c'] = dict(verdict=v, go=GO_OF[v], reason=[], sentence=sentence, n=nearcopy_n,
                             outcomes={sc: o[sc] for sc in ('SC7', 'SC8', 'SC9', 'SC10')})
    rest = [k for k in by_key if k not in used]
    assert not rest, f'cells not part of the rule: {rest}'
    out['cells'] = rows
    return out


def nearcopy_n_test(path):
    rec = json.loads(Path(path).read_text())
    return int(rec['splits']['test']['n_kept'])


def table(res):
    lines = ['encoder   paintings                         photos                            SC6 claims']
    for enc in ENCODERS:
        p, h = res['paintings'][enc], res['photos'][enc]
        n = sum(1 for r in res['sc6'][enc] if r['claim'])
        lines.append(f'{enc:<9} {p["verdict"] + " / " + p["go"]:<33} {h["verdict"] + " / " + h["go"]:<33} {n} of {len(res["sc6"][enc])}')
    pc = res['part_c']
    lines.append(f'part c    {pc["verdict"]} / {pc["go"]}')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('decide', help='cells + status -> OUT/stage3/outcomes.json')
    p.add_argument('--smoke', action='store_true', help='allow n_draws other than the protocol\'s')
    p.add_argument('--root', default=str(BP.OUT))
    p.add_argument('--status', default=str(BP.RECORDS / 'status.json'))
    p.add_argument('--nearcopy', default=str(BP.RECORDS / 'nearcopy_emoset.json'))
    p.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    stage3 = Path(args.root) / 'stage3'
    cells, files = load_cells(stage3, args.smoke)
    status = json.loads(Path(args.status).read_text())
    n = nearcopy_n_test(args.nearcopy) if Path(args.nearcopy).exists() else None
    res = decide(cells, status, n)
    res = dict(smoke=bool(args.smoke), cell_files=files, status=status, **res)
    out = Path(args.out) if args.out else stage3 / 'outcomes.json'
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    print(table(res))
    print(f'{len(res["cells"])} cells from {len(files)} files -> {out}')


if __name__ == '__main__':
    main()
