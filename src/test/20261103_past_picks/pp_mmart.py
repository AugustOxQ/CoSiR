"""r19 ticket 08: MMArt scores and stage 1 for DINOv2, CLIP B/32 and CLIP B/16 (SC7 to SC10), gates (iv) and (vii), r15 part.

Spec: docs/superpowers/specs/2026-11-03-past-picks-design.md SC7 to SC10, §3.4 (r15 §3.4 inherited), §3.5 (iv) and (vii), D21.
Everything inherited runs through r15's own functions (`mt_clu`, `mt_ret`, `mt_export`, `mt_data`) with the new three
registered by `pp_register`; this module only wires them to r19's records, matched probabilities and labels. Nothing of
r15's statistics is rewritten. Two orchestration loops are copies, line for line, because the originals read r15's own
records and matched probabilities: `score_clu` (mt_clu.py:227 `score_encoder`) and `score_ret` (mt_ret.py:308
`score_encoder`, with `load_stage1` at :286).

  stage 1   `stage1_block(enc)`: r15's `clu_ranges`, `ret_ranges` and `picks` (arms `cstar` and `frozen`) from the
            validation exports `OUT/mmart/exports/val/<enc>/cstar/` and the ArtELingo default store of the 5,769
            paintings. `OUT/stage1/mmart_<enc>.json` (ticket 12 merges it into `records/stage1_<enc>.json`, key `mmart`).
            The teacher's pick is not recomputed: r15's `stage1_teacher.json` (picked once with Qwen's ranges, D12).
  stage 2   `score_clu` (SC7 AMI against the frozen image, SC9 against the matched arm's larger reading, SC10 against
            the teacher) and `score_ret` (SC8 mAP against the frozen image at its pick, SC10 against the teacher at
            r15's pick) on MMArt test, r15's 16,384 paintings in sha256 order. r15's SC1 to SC4 become SC7 to SC10.
            SC10's LEVEL unit is X-c*'s own MMArt validation range (AMI for clustering, mAP at R0 for retrieval).
            The ranges and picks are read through `pp_paths.read_committed_record` before any test row.
            Cells: `OUT/stage2/cells_mmart_clu.json`, `cells_mmart_ret.json`.
  gates     `replay_iv` (r15's CLIP L/14 SC1 to SC4 cells within 1e-9, from r15's stored exports, matched probabilities
            and records, through the same two scorers) and `replay_vii` (r15's stored validation exports through
            `stage1_block` give r15's `stage1_clip_l14.json` within 1e-9). `OUT/stage0/gate_iv_r15.json`, `gate_vii_r15.json`.
            Keep r15's double normalisation of frozen and teacher retrieval rows (`mt_ret.view_rows`): the replay
            replays r15's numbers.

CLI (`python pp_mmart.py -h`): stage1 | score | replay-iv | replay-vii.
"""
import pp_paths as PP  # first: sets the older rounds' output roots
import pp_register  # noqa: F401  (the new three in r14's and r15's tables, store readers on pp_stores)

import argparse
import contextlib
import importlib
import json
import time
from pathlib import Path

import numpy as np

import mt_clu
import mt_matched
import mt_paths
import mt_ret

ASPECTS = tuple(PP.ASPECTS)
SEEDS = tuple(PP.SEEDS)
NEW = tuple(PP.NEW_ENCODERS)
TOL = 1e-9
SC_NEW = {'SC1': 'SC7', 'SC2': 'SC8', 'SC3': 'SC9', 'SC4': 'SC10'}      # r15's labels -> r19's (§6)
BLOCK_KEYS = ('clu_ranges', 'ret_ranges', 'picks')
N_TEST = mt_paths.N_TEST                                               # 16,384
N_VAL = mt_paths.SPLIT_COUNTS['val']                                   # 5,769
REPLAY_ENC = 'clip_l14'


# ============================================================================================ small helpers
def _data():
    return importlib.import_module('mt_data')


def jsonable(doc):
    """The document as the JSON file holds it (seed keys as text, numbers as floats); NaN refused."""
    return json.loads(json.dumps(doc, allow_nan=False))


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)
    return path


def relabel(cells):
    """r15's cells (SC1 to SC4) with r19's labels (SC7 to SC10). Refuses a cell that is not in r15's labels."""
    out = []
    for c in cells:
        assert c['sc'] in SC_NEW, f'cell label {c["sc"]!r} is not one of r15\'s {sorted(SC_NEW)}'
        out.append(dict(c, sc=SC_NEW[c['sc']]))
    return out


def check_block(enc, block):
    """A stage 1 `mmart` block: r15's three parts, each for this encoder."""
    assert set(BLOCK_KEYS) <= set(block), f'stage 1 block lacks {sorted(set(BLOCK_KEYS) - set(block))}'
    assert block.get('encoder', enc) == enc, (enc, block.get('encoder'))
    for k in BLOCK_KEYS:
        assert block[k].get('encoder', enc) == enc, f'{k}: record of {block[k].get("encoder")}, wanted {enc}'
    arms = block['picks'].get('arms', {})
    assert {'cstar', 'frozen'} <= set(arms), sorted(arms)
    return block


def read_block(enc):
    """Stage 1 of encoder `enc` from `records/stage1_<enc>.json`, key `mmart`: refuses unless committed unchanged in HEAD
    (§3.5), so it runs before any test row."""
    rec = PP.read_committed_record(f'stage1_{enc}.json')
    assert rec.get('encoder', enc) == enc and 'mmart' in rec, f'stage1_{enc}.json: no `mmart` block for {enc}'
    return check_block(enc, rec['mmart'])


def read_teacher():
    """r15's teacher pick (`stage1_teacher.json`, committed in r15's records; picked once with Qwen's ranges, D12)."""
    doc = mt_paths.read_committed_record('stage1_teacher.json')
    return check_teacher(doc)


def check_teacher(doc):
    assert doc.get('ranges_encoder') == 'qwen', 'the teacher is picked once, with Qwen\'s ranges (D12)'
    assert set(doc['arms']) == {'teacher'} and 'pick' in doc['arms']['teacher']
    return doc


def check_matched(got, n):
    """{seed: {'own': int [1, n], 'r20': int [3, n]}} for seeds 0 to 4 (ticket 05's `pp_matched.partitions`)."""
    assert sorted(got) == list(SEEDS), f'matched seeds {sorted(got)}'
    out = {}
    for s in SEEDS:
        own, r20 = np.asarray(got[s]['own']), np.asarray(got[s]['r20'])
        assert own.shape == (1, n) and r20.shape == (3, n), f'matched seed {s}: own {own.shape}, r20 {r20.shape}, want n={n}'
        assert np.issubdtype(own.dtype, np.integer) and np.issubdtype(r20.dtype, np.integer)
        assert 0 <= own.min() and own.max() < PP.N_TOPICS and 0 <= r20.min() and r20.max() < PP.K
        out[s] = {'r20': r20, 'own': own}
    return out


def matched_new(enc, keys):
    """The matched arm on MMArt for the new three: ticket 05's partitions of the `mmart_new` probabilities, recomputed on
    exactly `keys` (the test rows, in order), never a slice of a fit on all of `new`."""
    import pp_matched
    return check_matched(pp_matched.partitions('mmart_new', enc, list(keys)), len(keys))


def matched_r15(enc, keys):
    """r15's route (gate (iv) replay): `mt_matched.partitions` (stored probabilities of r15's outputs root)."""
    return check_matched(mt_clu.matched_fixed(mt_matched.partitions(enc, list(keys))), len(keys))


@contextlib.contextmanager
def r15_outputs():
    """Point r15's code at r15's stored outputs (exports, matched probabilities) for a replay."""
    old = mt_paths.OUT
    mt_paths.OUT = PP.R15_OUT
    try:
        yield
    finally:
        mt_paths.OUT = old


# ============================================================================================ stage 1
def stage1_block(enc, workers=8, expect_n=N_VAL):
    """r15's stage 1 for `enc` on MMArt validation: clustering ranges (`mt_clu.clustering_ranges`), retrieval ranges
    (`mt_ret.validation_ranges`), and the read-out picks of X-c* and the frozen image with X's retrieval ranges
    (`mt_ret.pick_encoder`, D12/D21). Validation only. Returns {encoder, clu_ranges, ret_ranges, picks} as the JSON holds it."""
    assert enc in mt_paths.ENCODERS, enc
    clu = mt_clu.clustering_ranges(enc)
    assert expect_n is None or clu['n_val'] == expect_n, clu['n_val']
    clu_doc = jsonable(dict(encoder=enc, kind='mmart_clustering_validation_ranges', ranges=clu['ranges'],
                            per_seed={a: {str(s): v for s, v in d.items()} for a, d in clu['per_seed'].items()},
                            sources=clu['sources'], n_val=clu['n_val'], labelled=clu['labelled']))
    ret = mt_ret.validation_ranges(enc, workers)
    assert ret['n_val'] == clu['n_val'] and ret['labelled'] == clu['labelled'], 'validation sets differ'
    ret_doc = jsonable(dict(encoder=enc, kind='mmart_retrieval_validation_ranges', **ret))
    picks = jsonable(mt_ret.pick_encoder(enc, workers, ranges=ret_doc['ranges']))
    return dict(encoder=enc, clu_ranges=clu_doc, ret_ranges=ret_doc, picks=picks)


def stage1_path(enc):
    return PP.OUT / 'stage1' / f'mmart_{enc}.json'


def write_stage1(enc, block):
    assert enc in NEW, f'stage 1 files are written for {NEW}'
    return write_json(stage1_path(enc), check_block(enc, block))


# ============================================================================================ stage 2
def _n_draws(n_draws, task):
    return n_draws or (PP.SMOKE_DRAWS[task] if PP.SMOKE else PP.N_DRAWS[task])


def score_clu(enc, *, block=None, matched=None, n_draws=None, workers=8, aspects=ASPECTS, expect_n=N_TEST):
    """SC7, SC9, SC10 (AMI) cells of X-c* on MMArt test, aspect by aspect. A copy, line for line, of `mt_clu.score_encoder`
    (mt_clu.py:227) for keyset 'test' with three changes: the stage 1 record is r19's (`read_block`, committed in HEAD, read
    before any row), the matched arm is `matched(enc, keys)` (default `matched_new`), and the cells carry r19's labels.
    SC10's LEVEL unit is X-c*'s own MMArt validation AMI range from the record."""
    assert enc in mt_paths.ENCODERS and set(aspects) <= set(ASPECTS)
    n_draws = _n_draws(n_draws, 'clu')
    block = read_block(enc) if block is None else check_block(enc, block)       # §3.5: before any test row
    ranges = mt_clu.record_ranges(block, enc)
    md, ex = _data(), importlib.import_module('mt_export')
    keys = mt_clu._ordered_keys('test')
    assert expect_n is None or len(keys) == expect_n, f'test set has {len(keys)} keys, expected {expect_n}'
    frozen = mt_clu.single_fixed(md.features(mt_paths.DEFAULT_VIEW[enc], keys))
    matched_fixed = (matched or matched_new)(enc, keys)
    cells = []
    for aspect in aspects:
        y = np.asarray(md.labels(keys, aspect))
        assert y.shape == (len(keys),) and y.dtype == np.int64
        c_fixed = mt_clu.cstar_fixed({s: ex.load(mt_clu.EXPORT_SET['test'], enc, aspect, s, keys) for s in SEEDS})
        teacher = mt_clu.single_fixed(md.features(mt_paths.TEACHER_VIEW[aspect], keys))
        got = mt_clu.score_aspect(enc, aspect, y, c_fixed, frozen=frozen, matched=matched_fixed, teacher=teacher,
                                  scs=mt_clu.SCS, r=ranges[aspect], n_draws=n_draws, workers=workers,
                                  data=mt_clu.DATA_NAME['test'], matched_name=mt_paths.MATCHED[enc])
        cells += got
        print(f'{enc} {aspect}: ' + ', '.join(f'{SC_NEW[c["sc"]]} d={c["d"]:+.4f}' for c in got), flush=True)
    return relabel(cells)


def _ret_stage1(enc, block, teacher):
    """Ranges, X's picks and the teacher's arm from the records (`mt_ret.load_stage1`, mt_ret.py:286, on r19's block)."""
    block = check_block(enc, block)
    ranges = mt_ret._ranges_of(block['ret_ranges'])
    arms = mt_ret._arms_of(block['picks'])
    assert {'cstar', 'frozen'} <= set(arms)
    return ranges, arms, mt_ret._arms_of(check_teacher(teacher))['teacher']


def score_ret(enc, *, block=None, teacher=None, n_draws=None, workers=8, aspects=ASPECTS, expect_n=N_TEST):
    """SC8 and SC10 (mAP) cells of X-c* on MMArt test, aspect by aspect: a copy, line for line, of `mt_ret.score_encoder`
    (mt_ret.py:308) for keyset 'test'. Each picked map is fitted on all 5,769 validation features (the seed's own c*
    export; X's frozen image; the teacher's view) and applied to the test rows; queries and gallery are the aspect's
    labelled rows. Ranges and picks come from r19's committed record, the teacher's pick from r15's."""
    assert enc in mt_paths.ENCODERS and set(aspects) <= set(ASPECTS)
    n_draws = _n_draws(n_draws, 'ret')
    block = read_block(enc) if block is None else check_block(enc, block)       # §3.5: before any test row
    teacher = read_teacher() if teacher is None else teacher
    ranges, arms, teach = _ret_stage1(enc, block, teacher)
    d = _data()
    val_keys, keys = list(d.keys('val')), list(d.keys('test'))
    mt_ret.check_order(val_keys)
    mt_ret.check_order(keys)
    assert expect_n is None or (len(keys) == expect_n and len(val_keys) == N_VAL), (len(keys), len(val_keys))
    ek = mt_ret.export_keyset('test')
    fz_val, fz_eval = (mt_ret.view_rows(mt_paths.DEFAULT_VIEW[enc], k) for k in (val_keys, keys))
    cells = []
    for a in aspects:
        y = np.asarray(d.labels(keys, a), dtype=np.int64)
        assert y.shape == (len(keys),)
        cand = mt_ret.arm_ap(lambda s: mt_ret.cstar_rows(enc, a, s, 'val', val_keys),
                             lambda s: mt_ret.cstar_rows(enc, a, s, ek, keys), arms['cstar']['pick'], y, True, workers)
        frozen = mt_ret.arm_ap(lambda s: fz_val, lambda s: fz_eval, arms['frozen']['pick'], y, False, workers)
        tv, te = (mt_ret.view_rows(mt_paths.TEACHER_VIEW[a], k) for k in (val_keys, keys))
        teach_res = mt_ret.arm_ap(lambda s: tv, lambda s: te, teach['pick'], y, False, workers)
        cells += mt_ret.aspect_cells(enc, a, cand, frozen, teach_res, ranges[a], n_draws,
                                     dict(cstar=arms['cstar']['pick'], frozen=arms['frozen']['pick'], teacher=teach['pick']),
                                     len(keys), mt_ret.cell_data('test'), ('SC2', 'SC4'))
        print(f'{enc} {a}: ' + ', '.join(f'{SC_NEW[c["sc"]]} d={c["d"]:+.4f}' for c in cells[-2:]), flush=True)
    return relabel(cells)


def cells_path(part):
    assert part in ('clu', 'ret')
    return PP.OUT / 'stage2' / f'cells_mmart_{part}.json'


def write_cells(part, cells):
    """Merge into `OUT/stage2/cells_mmart_<part>.json`: a cell replaces the stored one of the same (encoder, sc, aspect,
    data). Cells must carry r19's labels and the index's fields."""
    for c in cells:
        assert c['sc'] in SC_NEW.values() and c['data'] == 'mmart_test' and c['task'] == part, c
        assert {'encoder', 'aspect', 'candidate', 'comparator', 'candidate_value', 'comparator_value', 'd', 'lo', 'hi', 'r',
                'r_source', 'n_draws', 'seeds', 'n_rows'} <= set(c)
    return mt_clu.write_cells(cells_path(part), cells)


# ============================================================================================ gates
def compare_docs(ours, theirs, prefix=''):
    """Items {name, ours, theirs, abs_diff, ok} of two JSON documents, leaf by leaf: numbers within `TOL`, everything else
    equal. A key or length that differs is an item that is not ok."""
    items = []

    def add(name, a, b, ok, diff):
        items.append(dict(name=name, ours=a, theirs=b, abs_diff=diff, ok=bool(ok)))

    if isinstance(ours, dict) and isinstance(theirs, dict):
        if set(ours) != set(theirs):
            add(f'{prefix}/<keys>', sorted(ours), sorted(theirs), False, None)
        for k in sorted(set(ours) & set(theirs)):
            items += compare_docs(ours[k], theirs[k], f'{prefix}/{k}')
    elif isinstance(ours, list) and isinstance(theirs, list):
        if len(ours) != len(theirs):
            add(f'{prefix}/<len>', len(ours), len(theirs), False, None)
        for i, (a, b) in enumerate(zip(ours, theirs)):
            items += compare_docs(a, b, f'{prefix}[{i}]')
    elif (isinstance(ours, (int, float)) and isinstance(theirs, (int, float))
          and not isinstance(ours, bool) and not isinstance(theirs, bool)):
        diff = abs(float(ours) - float(theirs))
        add(prefix, float(ours), float(theirs), diff <= TOL, diff)
    else:
        same = type(ours) is type(theirs) and ours == theirs
        add(prefix, ours, theirs, same, 0.0 if same else None)
    return items


def compare_cells(got, stored, prefix, intervals=True):
    """Items of relabelled `got` cells against r15's stored cells (r15's labels, mapped through `SC_NEW`), matched by
    (sc, aspect, task, data). Compared: the two values, d, the interval (when `intervals`), the LEVEL unit, the row and
    query counts, the readouts and the matched arm's reading. Every `got` cell needs its stored cell."""
    by = {(SC_NEW[c['sc']], c['aspect'], c['task'], c['data']): c for c in stored}
    assert len(by) == len(stored), 'stored cells repeat a (sc, aspect, task, data)'
    fields = ['candidate_value', 'comparator_value', 'r', 'n_rows', 'n_draws', 'n_labelled', 'n_queries', 'candidate',
              'comparator', 'comparator_reading', 'candidate_readout', 'comparator_readout']
    fields += ['d', 'lo', 'hi'] if intervals else []
    items = []
    for c in got:
        key = (c['sc'], c['aspect'], c['task'], c['data'])
        assert key in by, f'no stored cell for {key}'
        s = by[key]
        items += compare_docs({f: c.get(f) for f in fields if f in s or f in c},
                              {f: s.get(f) for f in fields if f in s or f in c}, f'{prefix}/{key[0]}/{key[1]}/{key[2]}')
    assert len({(c['sc'], c['aspect'], c['task']) for c in got}) == len(got)
    return items


def _result(gate, items, **extra):
    nums = [i['abs_diff'] for i in items if i['abs_diff'] is not None]
    return dict(gate=gate, tol=TOL, n_items=len(items), n_not_ok=sum(not i['ok'] for i in items),
                max_abs_diff=max(nums) if nums else None, items=items, **extra)


def _stored_cells(part):
    path = PP.R15_OUT / 'stage2' / f'cells_{part}.json'
    cells = [c for c in json.loads(path.read_text()) if c['encoder'] == REPLAY_ENC and c['data'] == 'mmart_test']
    assert cells, f'{path}: no {REPLAY_ENC} test cells'
    return cells


def replay_iv(parts=('clu', 'ret'), aspects=ASPECTS, n_draws_clu=None, n_draws_ret=None, workers=8):
    """Gate (iv), r15 part: r15's CLIP L/14 SC1 to SC4 values and intervals (`stage2/cells_{clu,ret}.json`) within 1e-9,
    through `score_clu` and `score_ret` fed r15's stored exports (`r15_outputs`), r15's matched probabilities
    (`matched_r15`) and r15's committed records (`stage1_clip_l14.json`, `stage1_teacher.json`). Intervals are compared at
    the stored draw counts only (2,000 and 10,000); a reduced run (draws, parts or aspects) cannot pass."""
    assert set(parts) <= {'clu', 'ret'} and parts and set(aspects) <= set(ASPECTS)
    n_clu, n_ret = n_draws_clu or PP.N_DRAWS['clu'], n_draws_ret or PP.N_DRAWS['ret']
    items, drawn = [], {}
    with r15_outputs():
        rec = mt_paths.read_committed_record(f'stage1_{REPLAY_ENC}.json')
        block = check_block(REPLAY_ENC, rec)
        for part in parts:
            stored = _stored_cells(f'{part}')
            stored = [c for c in stored if c['aspect'] in aspects]
            if part == 'clu':
                got = score_clu(REPLAY_ENC, block=block, matched=matched_r15, n_draws=n_clu, workers=workers,
                                aspects=tuple(aspects))
            else:
                got = score_ret(REPLAY_ENC, block=block, teacher=read_teacher(), n_draws=n_ret, workers=workers,
                                aspects=tuple(aspects))
            full = (n_clu if part == 'clu' else n_ret) == PP.N_DRAWS[part]
            drawn[part] = dict(n_draws=n_clu if part == 'clu' else n_ret, intervals_compared=full)
            stored_keys = {(SC_NEW[c['sc']], c['aspect']) for c in stored}
            assert stored_keys == {(c['sc'], c['aspect']) for c in got}, 'cells replayed differ from the stored cells'
            items += compare_cells(got, stored, f'{REPLAY_ENC}/{part}', intervals=full)
    complete = bool(set(parts) == {'clu', 'ret'} and set(aspects) == set(ASPECTS) and all(d['intervals_compared'] for d in drawn.values()))
    res = _result('iv_r15', items, parts=list(parts), aspects=list(aspects), draws=drawn, complete=complete)
    res['values_ok'] = res['n_not_ok'] == 0
    res['passed'] = bool(res['values_ok'] and complete)
    return res


def replay_vii(enc=REPLAY_ENC, workers=8):
    """Gate (vii), r15 part: r15's stored validation exports (`exports/val/<enc>/cstar`) through `stage1_block` give r15's
    stage 1 record of `enc` (`stage1_<enc>.json`: clu_ranges, ret_ranges, picks) within 1e-9, every leaf (ranges,
    per-seed values, export hashes, picks, read-out tables) compared."""
    with r15_outputs():
        stored = check_block(enc, mt_paths.read_committed_record(f'stage1_{enc}.json'))
        ours = stage1_block(enc, workers)
    items = compare_docs(ours, stored, enc)
    res = _result('vii_r15', items, encoder=enc, picks={arm: ours['picks']['arms'][arm]['pick'] for arm in ('cstar', 'frozen')})
    res['passed'] = res['n_not_ok'] == 0
    return res


# ============================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='cmd', required=True)
    s1 = sub.add_parser('stage1', help='ranges and picks on MMArt validation -> OUT/stage1/mmart_<enc>.json')
    s1.add_argument('--enc', required=True, choices=NEW)
    s2 = sub.add_parser('score', help='SC7 to SC10 on MMArt test -> OUT/stage2/cells_mmart_{clu,ret}.json')
    s2.add_argument('--enc', required=True, choices=NEW)
    s2.add_argument('--parts', nargs='+', default=['clu', 'ret'], choices=['clu', 'ret'])
    s2.add_argument('--n-draws-clu', type=int, default=None, help=f'default {PP.N_DRAWS["clu"]} (smoke {PP.SMOKE_DRAWS["clu"]})')
    s2.add_argument('--n-draws-ret', type=int, default=None, help=f'default {PP.N_DRAWS["ret"]} (smoke {PP.SMOKE_DRAWS["ret"]})')
    r4 = sub.add_parser('replay-iv', help="gate (iv), r15 part -> OUT/stage0/gate_iv_r15.json")
    r4.add_argument('--parts', nargs='+', default=['clu', 'ret'], choices=['clu', 'ret'])
    r4.add_argument('--aspects', nargs='+', default=list(ASPECTS), choices=ASPECTS)
    r4.add_argument('--n-draws-clu', type=int, default=None)
    r4.add_argument('--n-draws-ret', type=int, default=None)
    sub.add_parser('replay-vii', help="gate (vii), r15 part -> OUT/stage0/gate_vii_r15.json")
    args = p.parse_args(argv)
    t0 = time.monotonic()
    if args.cmd == 'stage1':
        block = stage1_block(args.enc, args.workers)
        path = write_stage1(args.enc, block)
        picks = {arm: block['picks']['arms'][arm]['pick'] for arm in ('cstar', 'frozen')}
        print(f'{args.enc}: clu ranges {block["clu_ranges"]["ranges"]} ret ranges {block["ret_ranges"]["ranges"]} picks {picks} '
              f'-> {path} ({time.monotonic() - t0:.0f}s)')
        return 0
    if args.cmd == 'score':
        for part in args.parts:
            if part == 'clu':
                cells = score_clu(args.enc, n_draws=args.n_draws_clu, workers=args.workers)
            else:
                cells = score_ret(args.enc, n_draws=args.n_draws_ret, workers=args.workers)
            write_cells(part, cells)
            print(f'{args.enc}: {len(cells)} {part} cells -> {cells_path(part)} ({time.monotonic() - t0:.0f}s)', flush=True)
        return 0
    if args.cmd == 'replay-iv':
        res, name = replay_iv(tuple(args.parts), tuple(args.aspects), args.n_draws_clu, args.n_draws_ret, args.workers), 'gate_iv_r15.json'
    else:
        res, name = replay_vii(workers=args.workers), 'gate_vii_r15.json'
    path = write_json(PP.OUT / 'stage0' / name, res)
    verdict = 'PASS' if res['passed'] else ('VALUES OK (reduced run)' if res.get('values_ok') else 'FAIL')
    print(f'{args.cmd}: {verdict} max_abs_diff {res["max_abs_diff"]} items {res["n_items"]} not ok {res["n_not_ok"]} '
          f'({time.monotonic() - t0:.0f}s) {path}')
    return 0 if res['passed'] or res.get('values_ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
