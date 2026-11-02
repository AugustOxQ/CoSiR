"""MMArt clustering scores of r15 (ticket 04): SC1, SC3 and SC4 (AMI cells), the validation seed ranges, replay 0(iii) (clustering).

Spec: docs/superpowers/specs/2026-11-02-mmart-test-design.md, SC1, SC3, SC4, §3.3 (populations), §3.4, §3.5 0(iii), D16.
The protocol is r14 part a's (r14 §3.4, `bs_artclu.py`); nothing of its statistics is rewritten. Each one is the function
that produced r14's numbers, imported here and called on arrays in an evaluation order:

  SC1  X-c* against X's frozen image   `bs_artclu.frozen_margin` (-> `rb_part2_report.margin`, seedless frozen arm): the
       frozen image has one partition set and draws no seeds; per draw the paintings, then five seeds for X-c*.
  SC3  X-c* against X's matched arm    `bs_artclu.matched_margin` (10-26's `b_matched`): one `default_rng(0)`; per draw the
       paintings, the seeds of X-c*, the seeds of the matched arm (independent); the larger of the matched arm's own and
       20-cluster readings is taken again in each draw (`rb_score.score_indices`).
  SC4  X-c* against the teacher        `bs_artclu.teacher_margin` (-> `rg_held.bootstrap_d_t`): paintings, then seeds for
       X-c*; the teacher draws nothing. The LEVEL unit r is X-c*'s validation range (stage 1 record).
  ranges  max - min over seeds 0 to 4 of the validation AMI (`bs_artclu.validation_ami`: `ct_score.output_partitions(x,
       'k1')['r20']`, three states, mean AMI), unrounded; MMArt validation labels, all 5,769 rows; validation only.

Populations (D16): k-means with 20 clusters (states 0 to 2) runs on every row of the set; AMI and its bootstrap rows run
over all rows and rows without the aspect's label (-1) are masked after indexing. Every set is in `mt_data.order`.
Held data (MMArt test) are read by `score_encoder` only; `clustering_ranges` reads validation only.

CLI (`python mt_clu.py -h`): ranges | score | replay-iii. Outputs go under `mt_paths.OUT` (MT_OUTPUTS for a smoke run).
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import mt_paths

if str(mt_paths.BS_ROUND) not in sys.path:
    sys.path.insert(0, str(mt_paths.BS_ROUND))
import bs_artclu  # noqa: E402  (r14's scorer; puts the older rounds' folders on sys.path)
import ct_score  # noqa: E402
import rb_score  # noqa: E402
import rg_held  # noqa: E402
import rt_data  # noqa: E402

from bs_artclu import frozen_margin, matched_margin, teacher_margin, seed_range, validation_ami  # noqa: E402,F401

OUT = mt_paths.OUT
ASPECTS = tuple(mt_paths.ASPECTS)
SEEDS = tuple(mt_paths.SEEDS)
SCS = ('SC1', 'SC3', 'SC4')
TOL = bs_artclu.TOL                                                 # 0(iii): 1e-9
FULL_DRAWS = mt_paths.N_DRAWS['clu']                                # 2,000
R_SOURCE = 'X-c* own validation seed range (clustering AMI, seeds 0 to 4, MMArt labels)'

# keyset -> the export keyset its c* rows are read from (exports exist for val, new and shared_held; the test, far and
# unseen-artist sets are subsets of `new`, read by key), and the `data` field of its cells.
EXPORT_SET = {'val': 'val', 'new': 'new', 'test': 'new', 'far': 'new', 'unseen_artist': 'new', 'shared_held': 'shared_held'}
DATA_NAME = {'test': 'mmart_test', 'far': 'mmart_far', 'unseen_artist': 'mmart_unseen_artist',
             'shared_held': 'mmart_shared_held'}
MATCHED_SETS = ('test', 'far', 'unseen_artist')                     # the matched arm is exported on MMArt `new` only


# ============================================================================================ small helpers
def _data():
    return importlib.import_module('mt_data')


def _unit32(x):
    """Float32 rows divided by their norm (the convention of `mt_data.features`, as `rb_encoders.full_images`)."""
    x = np.asarray(x).astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norm > 0).all()
    return (x / norm).astype(np.float32)


def kmeans_partitions(x):
    """Three k-means partitions (20 clusters, states 0 to 2) of all rows of `x`: r14 part a's route, int [3, n]."""
    return np.asarray(ct_score.output_partitions(np.asarray(x, dtype=np.float32), 'k1')['r20'])


def cstar_fixed(xs):
    """{seed: {'r20': int [3, n]}} from {seed: export rows}; seeds 0 to 4."""
    assert sorted(xs) == list(SEEDS), f'seeds {sorted(xs)}'
    return {s: {'r20': kmeans_partitions(np.asarray(xs[s]).astype(np.float32))} for s in SEEDS}


def single_fixed(x):
    """The frozen image or the teacher: one partition set, pseudo-seed 0."""
    return {0: {'r20': kmeans_partitions(x)}}


def matched_fixed(parts):
    """{seed: {'r20': [3, n], 'own': [1, n]}} from `mt_matched.partitions` ({'own': [n], 'k20': [3, n]} per seed)."""
    assert sorted(parts) == list(SEEDS), f'seeds {sorted(parts)}'
    out = {}
    for s in SEEDS:
        own, k20 = np.asarray(parts[s]['own']), np.asarray(parts[s]['k20'])
        assert own.ndim == 1 and k20.shape == (3, len(own)), (s, own.shape, k20.shape)
        assert np.issubdtype(own.dtype, np.integer) and np.issubdtype(k20.dtype, np.integer)
        out[s] = {'r20': k20, 'own': own[None, :]}
    return out


def matched_from_probs(probs_by_seed):
    """The same from stored probabilities [n, 40] per seed, through `ct_score.output_partitions(probs, 'percept')`
    (the replay's route for the matched arm)."""
    assert sorted(probs_by_seed) == list(SEEDS)
    out = {}
    for s in SEEDS:
        got = ct_score.output_partitions(np.asarray(probs_by_seed[s], dtype=np.float32), 'percept')
        out[s] = {'r20': np.asarray(got['r20']), 'own': np.asarray(got['own'])}
    return out


# ============================================================================================ cells
def make_cell(sc, encoder, aspect, candidate, comparator, candidate_value, comparator_value, d, lo, hi, r, r_source,
              n_draws, data, n_rows, n_labelled, **extra):
    """One clustering cell (index format). `r` is the unrounded LEVEL unit for SC4 and None for SC1, SC3 (WIN-only bars)."""
    assert sc in SCS and aspect in ASPECTS
    values = [candidate_value, comparator_value, d, lo, hi]
    assert all(np.isfinite(v) for v in values), (sc, encoder, aspect, values)
    assert (r is None) == (sc != 'SC4'), 'only SC4 cells carry a LEVEL unit'
    assert r is None or (np.isfinite(r) and r > 0)
    return dict(sc=sc, encoder=encoder, aspect=aspect, task='clu', data=data, candidate=candidate, comparator=comparator,
                candidate_value=float(candidate_value), comparator_value=float(comparator_value), d=float(d), lo=float(lo),
                hi=float(hi), r=None if r is None else float(r), r_source=r_source, n_draws=int(n_draws),
                seeds=list(SEEDS), n_rows=int(n_rows), n_labelled=int(n_labelled), **extra)


def score_aspect(enc, aspect, labels, c_fixed, *, frozen=None, matched=None, teacher=None, scs=SCS, r=None,
                 r_source=R_SOURCE, n_draws=FULL_DRAWS, workers=8, data='mmart_test', matched_name=None):
    """The cells of one aspect on arrays: `labels` int64 [n] (-1 = no label), fixed partitions of X-c* (`cstar_fixed`), the
    frozen image (`single_fixed`), the matched arm (`matched_fixed`) and the teacher (`single_fixed`), all on the same n
    rows in one evaluation order. The statistics are r14's; each cell is returned in the order SC1, SC3, SC4."""
    labels = np.asarray(labels)
    assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer) and (labels >= 0).any()
    assert set(scs) <= set(SCS) and 1 <= workers <= 8
    matched_name = matched_name or mt_paths.MATCHED.get(enc, 'M')
    n, n_lab = len(labels), int((labels >= 0).sum())
    cand = f'cstar_{enc}'
    cells = []
    if 'SC1' in scs:
        assert frozen is not None, 'SC1 needs the frozen image'
        res = frozen_margin(c_fixed, frozen, labels, n_draws, workers, 'C', 'F')
        cells.append(make_cell('SC1', enc, aspect, cand, f'frozen_{enc}', res['A_cand'], res['A_e']['F-r20'], res['d'],
                               res['interval'][0], res['interval'][1], None, None, n_draws, data, n, n_lab))
    if 'SC3' in scs:
        assert matched is not None, 'SC3 needs the matched arm'
        res = matched_margin(c_fixed, matched, labels, n_draws, workers, 'C', 'M')
        readings = {k.replace('M-', f'{matched_name}-', 1): v for k, v in res['A_e'].items()}
        reading = res['max_entry'].replace('M-', f'{matched_name}-', 1)
        cells.append(make_cell('SC3', enc, aspect, cand, f'matched_{matched_name}', res['A_cand'], readings[reading],
                               res['d'], res['interval'][0], res['interval'][1], None, None, n_draws, data, n, n_lab,
                               comparator_reading=reading, comparator_readings=readings))
    if 'SC4' in scs:
        assert teacher is not None, 'SC4 needs the teacher'
        assert r is not None, 'SC4 needs the LEVEL unit (stage 1 record)'
        res = teacher_margin(c_fixed, teacher, labels, n_draws, workers)
        cells.append(make_cell('SC4', enc, aspect, cand, f'teacher_{aspect}', res['A_cand'], res['A_T'], res['d'],
                               res['interval'][0], res['interval'][1], r, r_source, n_draws, data, n, n_lab))
    return cells


def write_cells(path, cells):
    """Merge `cells` into the list file: a cell replaces the stored one with the same (encoder, sc, aspect, data), the
    others are kept (atomic write)."""
    path = Path(path)
    ident = lambda c: (c['encoder'], c['sc'], c['aspect'], c['data'])  # noqa: E731
    old = json.loads(path.read_text()) if path.exists() else []
    assert isinstance(old, list)
    new = {ident(c) for c in cells}
    merged = [c for c in old if ident(c) not in new] + list(cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(merged, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)
    return merged


def cells_path(keyset):
    """Verdict cells (test) in `stage2/cells_clu.json`; diagnostics under `diag/`."""
    return OUT / 'stage2' / 'cells_clu.json' if keyset == 'test' else OUT / 'diag' / f'cells_clu_{keyset}.json'


# ============================================================================================ ranges
def _export_sha(keyset, enc, aspect, seed):
    """SHA-256 of the c* export, from its sidecar (ticket 02)."""
    side = json.loads((mt_paths.OUT / 'exports' / keyset / enc / 'cstar' / f'{aspect}__s{seed}.json').read_text())
    return side['sha256']


def _ordered_keys(keyset):
    md = _data()
    keys = list(md.keys(keyset))
    assert len(set(keys)) == len(keys) and keys, keyset
    assert keys == list(md.order(keys)), f'{keyset}: keys are not in the sha256 order (D9)'
    return keys


def clustering_ranges(enc, keyset='val'):
    """X-c*'s validation clustering range per aspect: AMI (20 clusters, three states, all 5,769 rows, MMArt labels, -1
    masked) of each seed's validation export, then max - min over seeds 0 to 4, unrounded. Validation only.
    Returns {'ranges', 'per_seed' {aspect: {seed: ami}}, 'sources' {aspect: {seed text: export sha256}}, 'n_val', 'labelled'}."""
    assert keyset == 'val', 'ranges read validation only'
    md, ex = _data(), importlib.import_module('mt_export')
    keys = _ordered_keys('val')
    per_seed, sources, labelled = {}, {}, {}
    for aspect in ASPECTS:
        y = np.asarray(md.labels(keys, aspect))
        assert y.shape == (len(keys),) and (y >= 0).any()
        labelled[aspect] = int((y >= 0).sum())
        per_seed[aspect], sources[aspect] = {}, {}
        for s in SEEDS:
            x = np.asarray(ex.load('val', enc, aspect, s, keys)).astype(np.float32)
            assert x.shape[0] == len(keys)
            per_seed[aspect][s] = validation_ami(x, y)
            sources[aspect][str(s)] = _export_sha('val', enc, aspect, s)
    return dict(ranges={a: seed_range(per_seed[a]) for a in ASPECTS}, per_seed=per_seed, sources=sources, n_val=len(keys),
                labelled=labelled)


def write_ranges(enc, result):
    """`OUT/stage1/clu_ranges_<enc>.json` (ticket 07 merges it into the committed `records/stage1_<enc>.json`, key
    `clu_ranges`)."""
    path = OUT / 'stage1' / f'clu_ranges_{enc}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(encoder=enc, kind='mmart_clustering_validation_ranges', ranges=result['ranges'],
                  per_seed={a: {str(s): v for s, v in d.items()} for a, d in result['per_seed'].items()},
                  sources=result['sources'], n_val=result['n_val'], labelled=result['labelled'])
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(record, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)
    return path


def record_ranges(rec, enc=None):
    """{aspect: r} from a stage 1 record (`clu_ranges` is the ranges file's content; a plain aspect map also passes).
    With `enc`, the record (and the block, when it names one) must belong to that encoder."""
    block = rec['clu_ranges']
    if enc is not None:
        assert rec.get('encoder', enc) == enc and block.get('encoder', enc) == enc, (enc, rec.get('encoder'))
    ranges = block['ranges'] if 'ranges' in block else block
    assert set(ranges) == set(ASPECTS), sorted(ranges)
    out = {a: float(ranges[a]) for a in ASPECTS}
    assert all(np.isfinite(v) and v > 0 for v in out.values()), out
    return out


# ============================================================================================ scoring
def score_encoder(enc, keyset='test', scs=SCS, n_draws=None, workers=8, write=True):
    """SC1, SC3, SC4 (AMI) cells of X-c* on MMArt `keyset` (test: the cut set; diagnostics: far, unseen_artist,
    shared_held), three aspects, every row of the set in `mt_data.order`. c* rows: the exports (ticket 02); frozen image
    and teacher: `mt_data.features`; matched arm: `mt_matched.partitions` on exactly these rows; labels: `mt_data.labels`.
    Stage 1 record `records/stage1_<enc>.json`: read (committed in HEAD) before any row, for every set and SC; SC4 uses its r. Cells go to `cells_path(keyset)`; returned in aspect order."""
    assert enc in mt_paths.ENCODERS and keyset in DATA_NAME and set(scs) <= set(SCS)
    assert 'SC3' not in scs or keyset in MATCHED_SETS, f'no matched arm on {keyset}'
    n_draws = n_draws or (mt_paths.SMOKE_DRAWS['clu'] if mt_paths.SMOKE else FULL_DRAWS)
    md, ex = _data(), importlib.import_module('mt_export')
    keys = _ordered_keys(keyset)
    rec = mt_paths.read_committed_record(f'stage1_{enc}.json')          # §3.5: before any test row, whatever SCs are asked
    ranges = record_ranges(rec, enc) if 'SC4' in scs else {}
    assert rec.get('encoder', enc) == enc, (enc, rec.get('encoder'))
    frozen = single_fixed(md.features(mt_paths.DEFAULT_VIEW[enc], keys)) if 'SC1' in scs else None
    matched = matched_fixed(importlib.import_module('mt_matched').partitions(enc, keys)) if 'SC3' in scs else None
    cells = []
    for aspect in ASPECTS:
        y = np.asarray(md.labels(keys, aspect))
        assert y.shape == (len(keys),) and y.dtype == np.int64
        c_fixed = cstar_fixed({s: ex.load(EXPORT_SET[keyset], enc, aspect, s, keys) for s in SEEDS})
        teacher = single_fixed(md.features(mt_paths.TEACHER_VIEW[aspect], keys)) if 'SC4' in scs else None
        got = score_aspect(enc, aspect, y, c_fixed, frozen=frozen, matched=matched, teacher=teacher, scs=scs,
                           r=ranges.get(aspect), n_draws=n_draws, workers=workers, data=DATA_NAME[keyset],
                           matched_name=mt_paths.MATCHED[enc])
        cells += got
        print(f'{enc} {aspect}: ' + ', '.join(f'{c["sc"]} d={c["d"]:+.4f}' for c in got), flush=True)
    if write:
        write_cells(cells_path(keyset), cells)
    return cells


# ============================================================================================ replay 0(iii)
def _item(ours, theirs):
    diff = float(abs(ours - theirs))
    return dict(ours=float(ours), theirs=float(theirs), abs_diff=diff, ok=bool(diff < TOL))


def compare_cell(ours, theirs, tag, full_draws):
    """Items for one cell: the two values and d always; the interval (lo, hi) only at the stored 2,000 draws."""
    keys = ['candidate_value', 'comparator_value', 'd'] + (['lo', 'hi'] if full_draws else [])
    return {f'{tag}/{k}': _item(ours[k], theirs[k]) for k in keys}


def _result(items, n_draws):
    full = n_draws == FULL_DRAWS
    worst = max(i['abs_diff'] for i in items.values())
    ok = bool(all(i['ok'] for i in items.values()))
    return dict(gate='iii_clu', n_draws=n_draws, intervals_compared=full, max_abs_diff=worst, wiring_ok=ok,
                passed=ok and full, items=items)


def _held():
    names, sha, labels = bs_artclu.held_setup()
    return names, labels


def _stored_k(held_dir, names, aspect):
    """X-c*'s five held exports as float32 through r10's checked reader (sidecar, shape, unit norm)."""
    cand = bs_artclu.S0.Candidate('C', Path(held_dir), 'k11=40')
    return {s: bs_artclu.S0.load_candidate(cand, 'full', aspect, s)[0] for s in SEEDS}


def _teacher_x(names, aspect):
    return _unit32(rt_data._instructed('held', aspect, names))


def replay_iii(n_draws=FULL_DRAWS, workers=8, encs=('siglip2', 'qwen')):
    """0(iii), clustering part. The scorer is fed ArtELingo held data in r14's (r10's) order and recomputes every partition
    from the stored exports and features through this round's route:
    siglip2: r14's SC1, SC3, SC6 cells (`stage3/cells_artelingo_clu.json`): X-c* held exports, the default image through
             r14's `store_loader`, M2's stored held probabilities through `ct_score.output_partitions(probs, 'percept')`,
             the teacher's held view; values and (at 2,000 draws) intervals; the matched reading chosen too.
    qwen:    report 12-28's c* and teacher held AMI and d with its interval (r10 `stage4/verdict.json`)."""
    names, labels = _held()
    items = {}
    full = n_draws == FULL_DRAWS
    for enc in encs:
        assert enc in ('siglip2', 'qwen'), enc
        held_dir = mt_paths.CSTAR_ROOT[enc] / 'held' / mt_paths.CSTAR_ARM
        if enc == 'siglip2':
            stored = {(c['sc'], c['aspect']): c for c in json.loads(
                (mt_paths.R14_OUT / 'stage3' / 'cells_artelingo_clu.json').read_text()) if c['encoder'] == enc}
            spec = bs_artclu.ENC[enc]
            frozen = single_fixed(bs_artclu.store_loader(enc, names))
            probs = {}
            for s in SEEDS:
                with np.load(Path(spec['root']) / f'full/held/{spec["matched"]}/P3__s{s}.npz', allow_pickle=False) as z:
                    probs[s] = z['prob']
            matched = matched_from_probs(probs)
            scs = SCS
        else:
            verdict = json.loads((mt_paths.R10_OUT / 'stage4/verdict.json').read_text())['aspects']
            frozen = matched = None
            scs = ('SC4',)
        for aspect in ASPECTS:
            c_fixed = cstar_fixed(_stored_k(held_dir, names, aspect))
            teacher = single_fixed(_teacher_x(names, aspect))
            r = stored[('SC6', aspect)]['r'] if enc == 'siglip2' else 1.0     # the unit is not under test here
            cells = {c['sc']: c for c in score_aspect(
                enc, aspect, labels[aspect], c_fixed, frozen=frozen, matched=matched, teacher=teacher, scs=scs, r=r,
                n_draws=n_draws, workers=workers, data='artelingo_held',
                matched_name=mt_paths.MATCHED[enc] if enc == 'siglip2' else None)}
            if enc == 'siglip2':
                for sc, old_sc in (('SC1', 'SC1'), ('SC3', 'SC3'), ('SC4', 'SC6')):
                    items.update(compare_cell(cells[sc], stored[(old_sc, aspect)], f'{enc}/{aspect}/{sc}', full))
                same = cells['SC3']['comparator_reading'] == stored[('SC3', aspect)]['comparator_reading']
                items[f'{enc}/{aspect}/SC3/comparator_reading'] = dict(
                    ours=cells['SC3']['comparator_reading'], theirs=stored[('SC3', aspect)]['comparator_reading'],
                    abs_diff=0.0 if same else 1.0, ok=bool(same))
            else:
                v = verdict[aspect]
                theirs = dict(candidate_value=v['A_cand'], comparator_value=v['A_T'], d=v['d'], lo=v['lo'], hi=v['hi'])
                items.update(compare_cell(cells['SC4'], theirs, f'{enc}/{aspect}/SC4', full))
            print(f'{enc} {aspect}: replayed', flush=True)
    return _result(items, n_draws)


# ============================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--n-draws', type=int, default=None, help=f'default {FULL_DRAWS}')
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('ranges', help='X-c* validation seed ranges -> OUT/stage1/clu_ranges_<enc>.json')
    r.add_argument('--enc', required=True, choices=mt_paths.ENCODERS)
    s = sub.add_parser('score', help='SC1, SC3, SC4 cells -> OUT/stage2/cells_clu.json (test) or OUT/diag/')
    s.add_argument('--enc', required=True, choices=mt_paths.ENCODERS)
    s.add_argument('--keyset', default='test', choices=sorted(DATA_NAME))
    s.add_argument('--scs', nargs='+', default=list(SCS), choices=SCS)
    rp = sub.add_parser('replay-iii', help='0(iii) clustering part -> OUT/stage0/gate_iii_clu.json')
    rp.add_argument('--encs', nargs='+', default=['siglip2', 'qwen'], choices=('siglip2', 'qwen'))
    args = p.parse_args(argv)
    t0 = time.monotonic()
    if args.cmd == 'ranges':
        res = clustering_ranges(args.enc)
        path = write_ranges(args.enc, res)
        print(f'{args.enc}: ranges {res["ranges"]} -> {path} ({time.monotonic() - t0:.0f}s)')
        return 0
    if args.cmd == 'score':
        cells = score_encoder(args.enc, args.keyset, tuple(args.scs), args.n_draws, args.workers)
        print(f'{args.enc}: {len(cells)} cells -> {cells_path(args.keyset)} ({time.monotonic() - t0:.0f}s)')
        return 0
    res = replay_iii(args.n_draws or FULL_DRAWS, args.workers, tuple(args.encs))
    path = OUT / 'stage0' / 'gate_iii_clu.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    print(f'replay-iii (clustering): {"PASS" if res["passed"] else ("WIRING OK (reduced draws)" if res["wiring_ok"] else "FAIL")}'
          f' max_abs_diff {res["max_abs_diff"]:.3g} ({time.monotonic() - t0:.0f}s) {path}')
    return 0 if res['wiring_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
