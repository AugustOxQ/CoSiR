"""r18 ticket 07: stage 0 gate (iii), path replays within 1e-9 (spec §3.5 (iii) a, b, c; D23).

Our loaders (`om_data`, which re-exports r17's `ns_data`) feed r17's scorer (`ns_metrics`, `ns_clu`, `ns_ret`, read only).
The expected numbers are the older rounds' committed records and outputs, never a summary:
  r15 `records/verdicts.json` (CLIP L/14 and Qwen, MMArt test: SC1 clustering, SC2 retrieval cells),
  r14 `records/verdicts.json` (SigLIP 2, ArtELingo held: SC1), r10 `stage4/partitions` (Qwen c* held, k 20),
  r11 `stage5/verdict.json` (Qwen c* against the plain image, SC10, ArtELingo held).

  (a) k-means at 20. AMI value, comparator value, d, and the interval of every SC1 cell of r15 CLIP L/14 and Qwen on MMArt
      and r14 SigLIP 2 on ArtELingo (3 aspects each); partitions from `ns_metrics.partitions` on `om_data` rows in the
      clustering form (c* unit=False); and r10's stage 4 Qwen c* held partitions at 20, entry for entry.
  (b) The partitions of (a) through the partition files of TURTLE and TEMI. The plain image's k-means states 0 to 2 are
      written as three starts of the unseeded arm (`om_data.save_parts`, method `replay`, K 20) and read back by
      `om_data.turtle_parts` as int32 [3, n]; the cell is scored with that array as comparator (`ns_clu.paired_cell`).
      c*'s seeds have three states each and the c* file form has one start, so the c* files hold state 0 of each seed
      ([5, 1, n] on reading); that read is checked entry for entry against the in-memory state 0, and the (a) cell
      keeps the in-memory [5, 3, n] candidate. Every file read back must equal what was written.
  (c) Retrieval. r15's CLIP L/14 and Qwen SC2 cells on MMArt test (3 aspects each) and r11's SC10 (emotion; Qwen c*
      against the plain image, ArtELingo held), each at its recorded pick, through `ns_ret.arm_ap` and `ns_ret.paired_ret`.

A part passes only with the full draws (2,000 clustering, 10,000 retrieval), every value within `om_paths.TOL_REPLAY`
and nothing left out (aspects, cells). Fewer draws compare the point values only (the stored intervals come from the
full draws) and never pass.

(b) writes replay partitions to `OUT/parts/replay/...` (`OUT` = `OM_OUTPUTS` or the real folder); the fields `loss`,
`entropy` and `seconds` of the sidecars are 0.0 (no fit ran).

CLI: python om_gate_iii.py [--part a|b|c] [--draws N] [--workers W] [--cells C ...] [--aspects A ...]
Writes `OUT/stage0/gate_iii.json`: `a`, `b`, `c` (each `pass`, `max_abs_diff`, `items` with expected and got values) and
`pass` (all three present, passing, from the same code hash). One run of the parts a and b together shares the k-means fits.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import om_data as od
import om_paths as op

import ns_clu  # noqa: E402  (om_paths put r17's folder and the older rounds on sys.path)
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402
import ns_ret  # noqa: E402
import rg_held  # noqa: E402  (r10: the stored partition reader, as r17's replay)

TOL = op.TOL_REPLAY
FULL = {'clu': op.N_DRAWS['clu'], 'ret': op.N_DRAWS['ret']}
ASPECTS = tuple(op.ASPECTS)
SEEDS = tuple(op.SEEDS)
# the replayed cells: (round of the stored record, set, encoder)
CLU_CELLS = (('r15', 'mmart', 'qwen'), ('r15', 'mmart', 'clip_l14'), ('r14', 'artelingo', 'siglip2'))
RET_CELLS = (('r15', 'mmart', 'qwen'), ('r15', 'mmart', 'clip_l14'))
R11_ASPECT = 'emotion'
DATA_TAG = {'mmart': 'mmart_test', 'artelingo': 'artelingo_held'}
# every replayed unit by name (`--cells` picks a subset, which cannot pass): the cells above as round:set:enc, r10's
# stage 4 partitions (part a) and r11's SC10 (part c)
ALL_CELLS = tuple(':'.join(c) for c in CLU_CELLS) + ('r10', 'r11')


# ------------------------------------------------------------------------------------------------ items and results
def code_files():
    """The source files a replay number depends on: this module and the loaders and scorer it calls (om_data, om_paths,
    ns_clu, ns_metrics, ns_ret, ns_data, rg_held)."""
    mods = (sys.modules[__name__], od, op, ns_clu, ns_metrics, ns_ret, ns_data, rg_held)
    return [Path(m.__file__) for m in mods]


def code_sha256():
    """SHA-256 over the names and bytes of `code_files()`: a part computed before any of them changed does not count."""
    h = hashlib.sha256()
    for f in code_files():
        h.update(f.name.encode() + b'\0' + f.read_bytes() + b'\0')
    return h.hexdigest()


def item(name, source, ours, theirs, kind='value'):
    diff = float(abs(float(ours) - float(theirs)))
    return dict(name=name, source=str(source), kind=kind, ours=float(ours), theirs=float(theirs), abs_diff=diff,
                ok=bool(diff <= TOL))


def partition_item(name, source, ours, theirs):
    """Entry-for-entry comparison of two partition arrays; abs_diff counts the differing entries."""
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    assert ours.shape == theirs.shape, (name, ours.shape, theirs.shape)
    diff = int((ours != theirs).sum())
    return dict(name=name, source=str(source), kind='partitions', ours=f'{diff} entries differ', theirs='equal',
                abs_diff=float(diff), ok=diff == 0)


def cell_items(tag, source, got, stored, full):
    """Items of one cell: got {cand, comp, d, lo, hi} against a stored cell (candidate_value, comparator_value, d, lo,
    hi). The interval is compared only when `full` (the stored one is from the full draws)."""
    pairs = [('cand', 'candidate_value'), ('comp', 'comparator_value'), ('d', 'd')]
    if full:
        pairs += [('lo', 'lo'), ('hi', 'hi')]
    return [item(f'{tag}/{key}', f'{source}::{key}', got[ours], stored[key]) for ours, key in pairs]


def finish(items, draws, full_draws, complete=True):
    """{'pass', 'max_abs_diff', 'wiring_ok', 'draws', 'complete', 'items'}: pass needs the full draws, every item within the
    tolerance and nothing left out; wiring_ok is the item check alone."""
    assert items, 'a replay with no item'
    ok = all(i['ok'] for i in items)
    return {'pass': bool(ok and draws == full_draws and complete), 'max_abs_diff': float(max(i['abs_diff'] for i in items)),
            'wiring_ok': bool(ok), 'draws': int(draws), 'complete': bool(complete), 'tol': TOL, 'items': items}


def check_pick(ours, recorded, what):
    assert ours == recorded, f'{what}: pick {ours!r} differs from the recorded {recorded!r}'


def _json(path):
    return json.loads(Path(path).read_text())


def stored_cell(round_, task, sc, enc, aspect, data):
    path = op.ROUNDS[round_] / 'records/verdicts.json'
    (c,) = [c for c in _json(path)['cells'] if (c['task'], c['sc'], c['encoder'], c['aspect'], c['data']) ==
            (task, sc, enc, aspect, data)]
    return c, path


# ------------------------------------------------------------------------------------------------ (a) clustering
def chosen(cells, names):
    """The replayed cells (round, set, enc) whose name round:set:enc is in `names`."""
    return [c for c in cells if ':'.join(c) in names]


def clu_cell(cand, comp, y, draws):
    """AMI of c* against a comparator at 20 clusters: {cand, comp, d, lo, hi, ...} (`ns_clu.paired_cell`)."""
    return ns_clu.paired_cell(cand, comp, y, op.K20, metrics=('ami',), draws=draws)['ami']


def cand_parts(set_, enc, aspect, cache):
    """c* seeds 0 to 4, k-means at 20: int32 [5, 3, n] from `om_data.cstar` (clustering form, unit=False)."""
    key = ('cstar', set_, enc, aspect)
    if key not in cache:
        split = op.HELD_SPLIT[set_]
        cache[key] = np.stack([ns_metrics.partitions(od.cstar(set_, split, enc, aspect, s, unit=False), op.K20, set_)
                               for s in SEEDS])
    return cache[key]


def plain_parts(set_, enc, cache):
    """The plain image, k-means at 20: int32 [3, n]."""
    key = ('plain', set_, enc)
    if key not in cache:
        cache[key] = ns_metrics.partitions(od.plain(set_, op.HELD_SPLIT[set_], enc), op.K20, set_)
    return cache[key]


def labels_of(set_, aspect):
    return od.labels(set_, op.HELD_SPLIT[set_], aspect)


def replay_a(draws=FULL['clu'], aspects=ASPECTS, cache=None, cells=ALL_CELLS):
    cache = {} if cache is None else cache
    items = []
    for a in (aspects if 'r10' in cells else ()):           # r10 stage 4, Qwen c* on ArtELingo held: exactly
        for s in SEEDS:
            path = op.R10_OUT / f'stage4/partitions/full/k11=40/{a}__s{s}__r20.npz'
            ours = ns_metrics.partitions(od.cstar('artelingo', 'held', 'qwen', a, s, unit=False), op.K20, 'artelingo')
            theirs = rg_held.read_frozen_partitions(path, ours.shape[1])
            items.append(partition_item(f'r10/qwen/artelingo/{a}/s{s}/partitions', f'{path}::labels', ours, theirs))
    for round_, set_, enc in chosen(CLU_CELLS, cells):
        comp = plain_parts(set_, enc, cache)
        for a in aspects:
            y = labels_of(set_, a)
            got = clu_cell(cand_parts(set_, enc, a, cache), comp, y, draws)
            st, path = stored_cell(round_, 'clu', 'SC1', enc, a, DATA_TAG[set_])
            assert st['n_draws'] == FULL['clu'], (round_, enc, a, st['n_draws'])
            items += cell_items(f'{round_}/{enc}/{set_}/{a}/SC1', path, got, st, draws == FULL['clu'])
    return finish(items, draws, FULL['clu'], tuple(aspects) == ASPECTS and set(cells) >= set(ALL_CELLS[:3] + ('r10',)))


# ------------------------------------------------------------------------------------------------ (b) partition files
def _write(labels, **kw):
    return od.save_parts(labels, method='replay', k=op.K20, device='cpu', loss=0.0, entropy=0.0, seconds=0.0,
                         n_clusters=int(len(np.unique(labels))), overwrite=True, **kw)


def write_replay_parts(set_, enc, cand, comp, aspect, out=None):
    """The plain image's states 0 to 2 as starts 0 to 2 of the unseeded arm, and c*'s state 0 of each seed (aspect)."""
    assert comp.shape[0] == len(op.TURTLE_PLAIN_STARTS) and cand.shape[:2] == (len(SEEDS), comp.shape[0])
    for s in op.TURTLE_PLAIN_STARTS:
        _write(comp[s], set_=set_, enc=enc, arm='plain', start=s, input_sha256=od.array_sha256(comp), out=out)
    for s in SEEDS:
        _write(cand[s, 0], set_=set_, enc=enc, arm='cstar', aspect=aspect, start=op.TURTLE_CSTAR_START, seed=s,
               input_sha256=od.array_sha256(cand), out=out)


def read_replay_parts(set_, enc, out=None, aspect=None):
    """(plain int32 [3, n], c* int32 [5, 1, n] or None) through the reader of TURTLE's partitions."""
    plain = od.turtle_parts(set_, enc, 'plain', op.K20, method='replay', out=out)
    cstar = None if aspect is None else od.turtle_parts(set_, enc, 'cstar', op.K20, aspect=aspect, method='replay', out=out)
    return plain, cstar


def replay_b_cell(set_, enc, cand, comp, y, draws, out=None, aspect='emotion'):
    """Write the partitions, read them back and score the cell with the file array as comparator. Returns
    (cell dict, the plain array read back, the c* array read back [5, 1, n]). Raises when what was read back differs from
    what was written."""
    write_replay_parts(set_, enc, cand, comp, aspect, out)
    plain, cstar = read_replay_parts(set_, enc, out, aspect)
    assert np.array_equal(plain, comp), f'{set_}/{enc}: plain partitions read back differ from those written'
    assert np.array_equal(cstar[:, 0], cand[:, 0]), f'{set_}/{enc}/{aspect}: c* partitions read back differ from those written'
    return clu_cell(cand, plain, y, draws), plain, cstar


def replay_b(draws=FULL['clu'], aspects=ASPECTS, cache=None, out=None, cells=ALL_CELLS):
    cache = {} if cache is None else cache
    items = []
    for round_, set_, enc in chosen(CLU_CELLS, cells):
        comp = plain_parts(set_, enc, cache)
        for a in aspects:
            cand = cand_parts(set_, enc, a, cache)
            got, back, back_c = replay_b_cell(set_, enc, cand, comp, labels_of(set_, a), draws, out, a)
            st, path = stored_cell(round_, 'clu', 'SC1', enc, a, DATA_TAG[set_])
            tag = f'{round_}/{enc}/{set_}/{a}/SC1/files'
            items.append(partition_item(f'{tag}/plain_read_back', 'OUT/parts/replay', back, comp))
            items.append(partition_item(f'{tag}/cstar_read_back', 'OUT/parts/replay', back_c[:, 0], cand[:, 0]))
            items += cell_items(tag, path, got, st, draws == FULL['clu'])
    return finish(items, draws, FULL['clu'], tuple(aspects) == ASPECTS and set(cells) >= set(ALL_CELLS[:3]))


# ------------------------------------------------------------------------------------------------ (c) retrieval
def ret_cell(cstar, plain, draws):
    assert np.array_equal(cstar['rows'], plain['rows'])
    p = ns_ret.paired_ret(cstar['ap'], plain['ap'], None, draws, comp_seeds='none')
    p['n_queries'] = int(len(cstar['rows']))
    return p


def replay_c(draws=FULL['ret'], aspects=ASPECTS, workers=8, cells=ALL_CELLS):
    full = draws == FULL['ret']
    items = []
    for round_, set_, enc in chosen(RET_CELLS, cells):
        arms = ns_ret._Rows(set_, enc)
        pick = {arm: ns_ret.reused_pick(arm, set_, enc) for arm in ('cstar', 'plain')}
        for a in aspects:
            y = labels_of(set_, a)
            got = ret_cell(arms.ap('cstar', a, pick['cstar'], y, workers), arms.ap('plain', a, pick['plain'], y, workers), draws)
            st, path = stored_cell(round_, 'ret', 'SC2', enc, a, DATA_TAG[set_])
            check_pick(pick['cstar'], st['candidate_readout'], f'{round_}/{enc}/{a} c*')
            check_pick(pick['plain'], st['comparator_readout'], f'{round_}/{enc}/{a} plain')
            assert st['n_draws'] == FULL['ret'], (round_, enc, a, st['n_draws'])
            tag = f'{round_}/{enc}/{set_}/{a}/SC2'
            items += cell_items(tag, path, got, st, full)
            items.append(item(f'{tag}/n_queries', f'{path}::n_queries', got['n_queries'], st['n_queries']))
    # r11 stage 5, Qwen c* against the plain image on ArtELingo held (SC10 = emotion)
    if 'r11' in cells:
        items += _replay_r11(draws, workers)
    return finish(items, draws, FULL['ret'], tuple(aspects) == ASPECTS and set(cells) >= set(ALL_CELLS[:2] + ('r11',)))


def _replay_r11(draws, workers):
    full = draws == FULL['ret']
    items = []
    path = op.R11_OUT / 'stage5/verdict.json'
    v = _json(path)
    assert v['draws'] == FULL['ret']
    arms = ns_ret._Rows('artelingo', 'qwen')
    a = R11_ASPECT
    y = labels_of('artelingo', a)
    pick = {arm: ns_ret.reused_pick(arm, 'artelingo', 'qwen') for arm in ('cstar', 'plain')}
    for arm in pick:
        check_pick(pick[arm], v['readouts'][arm], f'r11/qwen/artelingo/{a} {arm}')
    cstar = arms.ap('cstar', a, pick['cstar'], y, workers)
    plain = arms.ap('plain', a, pick['plain'], y, workers)
    got = ret_cell(cstar, plain, draws)
    (cmp_,) = v['sc']['SC10']['comparisons']
    assert cmp_['aspect'] == a and cmp_['candidate'] == ['cstar', a] and cmp_['comparator'][0] == 'plain'
    stored = dict(candidate_value=v['map']['cstar'][a]['mean'], comparator_value=v['map']['plain'][a]['mean'], d=cmp_['d'],
                  lo=cmp_['interval'][0], hi=cmp_['interval'][1])
    tag = f'r11/qwen/artelingo/{a}/SC10'
    items += cell_items(tag, f'{path}::sc.SC10 and map', got, stored, full)
    items.append(item(f'{tag}/n_queries', f'{path}::n_queries', got['n_queries'], v['n_queries'][f'cstar/{a}/{a}'][0]))
    return items


# ------------------------------------------------------------------------------------------------ the gate file
def gate_path():
    return Path(op.OUT) / 'stage0' / 'gate_iii.json'


def write_gate(path, part, result):
    """Merge `result` as part a, b or c into the gate file; `pass` needs all three, passing, from the same code hash as
    this module (a part from older code does not count). Returns the file's content."""
    assert part in ('a', 'b', 'c'), part
    path = Path(path)
    res = json.loads(path.read_text()) if path.exists() else {}
    res[part] = result
    here = code_sha256()
    res['pass'] = bool(all(p in res and res[p]['pass'] and res[p].get('code_sha256') == here for p in ('a', 'b', 'c')))
    op.write_json(path, res)
    return res


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--part', choices=('a', 'b', 'c'), default=None, help='one part (default: all three)')
    ap.add_argument('--draws', type=int, default=None, help='bootstrap draws (default: the full 2,000 and 10,000)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cells', nargs='+', default=None, choices=ALL_CELLS, help='a subset of the cells: the part cannot pass')
    ap.add_argument('--aspects', nargs='+', default=None, choices=ASPECTS, help='a subset: the part cannot pass')
    args = ap.parse_args(argv)
    ns_clu.WORKERS = args.workers
    aspects = ASPECTS if args.aspects is None else tuple(args.aspects)
    cells = ALL_CELLS if args.cells is None else tuple(args.cells)
    parts = ('a', 'b', 'c') if args.part is None else (args.part,)
    cache = {}
    rc = 0
    here = code_sha256()       # taken before any part runs: a source edit during the run leaves a stale stamp
    for part in parts:
        t0 = time.monotonic()
        d = args.draws
        if part == 'a':
            res = replay_a(d or FULL['clu'], aspects, cache, cells=cells)
        elif part == 'b':
            res = replay_b(d or FULL['clu'], aspects, cache, cells=cells)
        else:
            res = replay_c(d or FULL['ret'], aspects, args.workers, cells=cells)
        res['seconds'] = round(time.monotonic() - t0, 1)
        res['code_sha256'] = here
        out = write_gate(gate_path(), part, res)
        bad = [i for i in res['items'] if not i['ok']]
        print(f'gate (iii) {part}: pass={res["pass"]} wiring_ok={res["wiring_ok"]} max_abs_diff={res["max_abs_diff"]:.3g} '
              f'draws={res["draws"]} complete={res["complete"]} items={len(res["items"])} {res["seconds"]}s; '
              f'overall pass={out["pass"]}', flush=True)
        for i in bad[:10]:
            print(f'  DIFF {i["name"]}: ours {i["ours"]} theirs {i["theirs"]} abs_diff {i["abs_diff"]:.3g}', flush=True)
        rc |= 0 if res['wiring_ok'] else 1
    return rc


if __name__ == '__main__':
    sys.exit(main())
