"""r23 scoring replay, gate (iii) (ticket 06; spec §3.5 (iii), C12). Run from the round folder: `python tcl_replay.py cuda`.

New scoring code fed r22's validation rows (means fitted on those rows, passed to `tcl_metrics.evaluate`), against numbers
read from r22's own files. Writes OUT/stage0/gate_iii.json, {"clauses": {name: {"pass", "worst", "n"}}, "overall"}:

    core_label     c*, plain Qwen, teacher's text, seed 0: label mAP, 9 cells x 3 read-outs, vs results/core.json     1e-5
    late_label     late fusion at a = 0, 0.7, 1, seed 0 heads: label mAP (best read-out) per cell, vs followup.json     1e-5
    late_pair      the same: R@10 of the picked head and read-out per direction, and the pick itself                    1e-5
    mmart_label    MMArt validation, unified text, text to image, raw and centring, c* and plain: label mAP, mmart.json 1e-5
    mmart_pair     the same: pair R@10 and MRR                                                                         1e-5
    interval_hand  `replicate` on a hand case of 6 paintings (2 and 3 seeds, hand I and S)                              1e-12
    interval_tie   `painting_interval` against `replicate` rebuilt draw by draw on the same case and generator           1e-12
    table2         `painting_interval`, k = 1, on r22's per-query differences vs r22's Table 2 bounds                    3e-4

A difference of head or read-out in late_pair counts as 1. The MMArt labels are first checked against r22's own coding,
joined by key (`check_label_join`). TCL_SMOKE does not apply: the rows are full validation rows (`full_rows`).
"""
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

import tcl_paths as P
import tcl_data
import tcl_forward
import tcl_metrics
import tcl_score

TOL_REPLAY, TOL_HAND, TOL_TABLE2 = 1e-5, 1e-12, 3e-4


def compare(got, want, tol):
    """One clause: `got` and `want` map the same keys to floats. Returns {"pass", "worst", "n"}; a missing key or a
    non-finite value crashes, a wrong value fails."""
    assert got.keys() == want.keys() and got, 'clause keys differ or are empty'
    diffs = np.asarray([abs(got[k] - want[k]) for k in got], dtype=np.float64)
    assert np.isfinite(diffs).all(), 'non-finite value in a clause'
    worst = float(diffs.max())
    return {'pass': bool(worst <= tol), 'worst': worst, 'n': len(diffs)}


def gate_path(out=None):
    return Path(P.OUT if out is None else out) / 'stage0/gate_iii.json'


def write_gate(clauses, out=None):
    """OUT/stage0/gate_iii.json: {"clauses": {name: {"pass", "worst", "n"}}, "overall": bool}."""
    assert clauses and all(set(c) == {'pass', 'worst', 'n'} for c in clauses.values())
    doc = {'clauses': clauses, 'overall': bool(all(c['pass'] for c in clauses.values()))}
    path = gate_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    return doc


# ------------------------------------------------------------------------------------------------ interval clauses
# Hand case: 6 paintings with 1, 2, 3, 1, 2, 4 scored queries; per-painting query-value sums of five seeds (the first two
# rows are worked by hand below; the rest only feed the tie) and of the comparator. Draw: I = [0, 1, 1, 4, 5, 5], so the
# weights are [1, 2, 0, 0, 1, 2] and the draw holds 15 queries. Seed numerators: 6.2 (row 0), 7.8 (row 1), the comparator 6.5.
HAND_COUNTS = np.asarray([1, 2, 3, 1, 2, 4], dtype=np.float64)
HAND_SUM_C = np.asarray([[0.2, 1.0, 1.5, 0.4, 1.6, 1.2],
                         [0.8, 0.6, 2.1, 0.9, 0.2, 2.8],
                         [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                         [0.3, 1.2, 0.6, 0.7, 0.1, 3.0],
                         [0.5, 0.2, 1.1, 0.8, 1.9, 2.2]], dtype=np.float64)
HAND_SUM_M = np.asarray([0.1, 0.8, 1.2, 0.6, 0.8, 2.0], dtype=np.float64)
HAND_I = np.asarray([0, 1, 1, 4, 5, 5], dtype=np.int64)
# (seeds used, S, hand value): (2 v0 + v1) / 3 minus the comparator is 7/450 for S = [0, 0, 1], and so on.
HAND_VALUES = (
    (2, [1, 1], 13 / 150), (2, [0, 1], 1 / 30), (2, [0, 0], -0.02),
    (3, [0, 0, 1], 7 / 450), (3, [1, 1, 0], 23 / 450),
)


def clause_interval_hand(replicate=tcl_metrics.replicate):
    """`replicate` on the hand case, within 1e-12 of the hand values."""
    got, want = {}, {}
    for j, (k, S, value) in enumerate(HAND_VALUES):
        got[j] = replicate(HAND_SUM_C[:k], HAND_SUM_M, HAND_COUNTS, HAND_I, np.asarray(S, dtype=np.int64))
        want[j] = value
    return compare(got, want, TOL_HAND)


def hand_queries():
    """Per-query rows of the hand case: each painting's sum split evenly over its queries, paintings named by gapped ids,
    queries interleaved, so the grouping has to use the ids."""
    units = np.asarray([2, 4, 9, 20, 60, 100], dtype=np.int64)
    pid = np.repeat(units, HAND_COUNTS.astype(int))
    value = np.repeat(1 / HAND_COUNTS, HAND_COUNTS.astype(int))
    order = np.random.default_rng(3).permutation(len(pid))
    pid, value = pid[order], value[order]
    cand = np.stack([value * np.repeat(row, HAND_COUNTS.astype(int))[order] for row in HAND_SUM_C])
    comp = value * np.repeat(HAND_SUM_M, HAND_COUNTS.astype(int))[order]
    return cand, comp, pid


def clause_interval_tie(interval=tcl_metrics.painting_interval, draws=300):
    """`painting_interval` has its own vectorised sum; `replicate` is rebuilt draw by draw from the same generator stream
    (I, then S, as spec §3.4) with per-painting sums taken by a plain loop. Point, bounds and unit count agree to 1e-12."""
    cand, comp, pid = hand_queries()
    units = np.unique(pid)
    counts = np.asarray([(pid == u).sum() for u in units], dtype=np.float64)
    sum_m = np.asarray([comp[pid == u].sum() for u in units])
    got, want = {}, {}
    for k in (2, 3, 5):
        sum_c = np.asarray([[row[pid == u].sum() for u in units] for row in cand[:k]])
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(draws):
            I = rng.integers(0, len(units), size=len(units))
            S = rng.choice(list(range(k)), size=k, replace=True)
            boot.append(tcl_metrics.replicate(sum_c, sum_m, counts, I, S))
        low, high = np.percentile(boot, [2.5, 97.5])
        for name, value, expected in zip(('point', 'lower', 'upper', 'units'), interval(cand[:k], comp, pid, draws=draws),
                                         (cand[:k].mean() - comp.mean(), low, high, len(units))):
            got[k, name], want[k, name] = float(value), float(expected)
    return compare(got, want, TOL_HAND)


# ------------------------------------------------------------------------------------------------ Table 2
R22_SUMMARY = P.R22_ROUND / 'results/summary.json'


def table2_items(summary=R22_SUMMARY, perquery=P.R22_OUT / 'perquery'):
    """r22's Table 2 (the `cstar` rule; seed 0 only): for each of the nine cells and the two comparators, the saved
    per-query AP of both sides at the readouts r22 picked, the painting of each query, and r22's mean and bounds."""
    doc = json.loads(Path(summary).read_text())
    rule = doc['rules'][0]
    assert rule['row'] == 'cstar' and not rule['smoke'] and len(rule['cells']) == 9
    items = []
    for cell in rule['cells']:
        a, d = cell['aspect'], cell['direction']
        for comparator in ('plain', 'teacher'):
            comparison = cell['comparisons'][comparator]
            sides = {}
            for row, head, readout in (('cstar', a, cell['readout']),
                                       (comparator, 'all' if comparator == 'plain' else a, comparison['readout'])):
                assert readout in P.READOUTS
                with np.load(Path(perquery) / f'{row}__{head}__{d}__{readout}.npz', allow_pickle=False) as z:
                    sides[row] = z[f'ap__{a}'].astype(np.float64), z['q_pid']
            (cand, pid), (comp, comp_pid) = sides['cstar'], sides[comparator]
            assert cand.shape == comp.shape == pid.shape and np.array_equal(pid, comp_pid) and pid.dtype == np.int64
            items.append(dict(key=(a, d, comparator), cand=cand, comp=comp, pid=pid, mean=comparison['mean'],
                              lo=comparison['lo'], hi=comparison['hi'], n_paired=comparison['n_paired']))
    return items


def clause_table2(items, interval=tcl_metrics.painting_interval, draws=P.N_DRAWS):
    """`painting_interval` with k = 1 on each item's scored queries, its bounds within 3e-4 of r22's."""
    got, want = {}, {}
    for item in items:
        keep = np.isfinite(item['cand']) & np.isfinite(item['comp'])
        assert int(keep.sum()) == item['n_paired'], ('paired queries differ from r22', item['key'])
        point, lower, upper, _ = interval(item['cand'][keep][None], item['comp'][keep], item['pid'][keep], draws=draws)
        assert abs(point - item['mean']) < 1e-6, ('point differs from r22', item['key'])
        for name, value in (('lo', lower), ('hi', upper)):
            got[item['key'], name], want[item['key'], name] = value, item[name]
    return compare(got, want, TOL_TABLE2)


# ------------------------------------------------------------------------------------------------ scoring through the new code
def finite_mean(x):
    return float(x[np.isfinite(x)].mean())


def score_all(frame, text, image, aspects, device, readouts=P.READOUTS, directions=P.DIRECTIONS):
    """Rank every (direction, readout) with the new `evaluate`, means fitted on these rows (`fit_means`), labels from the
    frame. Returns {(direction, readout): {'ap': {aspect: label mAP}, 'pair': summarize_pair(first hit ranks)}}."""
    assert len(text) == frame['n_t'] and len(image) == frame['n_i']
    text_mean, image_mean = tcl_metrics.fit_means(text, image)         # the pair is ordered by tcl_score, as in the verdict
    out = {}
    for d in directions:
        q, q_pid = (text, frame['tp']) if d[0] == 't' else (image, frame['ip'])
        g, g_pid = (text, frame['tp']) if d[2] == 't' else (image, frame['ip'])
        labels = {a: dict(q=frame['labels'][a][d[0]], g=frame['labels'][a][d[2]]) for a in aspects}
        for ro in readouts:
            got = tcl_metrics.evaluate(q, g, q_pid, g_pid, labels, d == 't2t', ro, device,
                                       means=tcl_score.query_gallery_means(d, text_mean, image_mean))
            out[d, ro] = {'ap': {a: finite_mean(v) for a, v in got['ap'].items()},
                          'pair': tcl_metrics.summarize_pair(got['first_rank'])}
    return out


def clause_core(src, ref, device):
    """r22's label mAP for c*, plain Qwen and the teacher's text, seed 0, nine cells at every read-out. `ref` maps
    (row, aspect, direction, readout) to r22's number."""
    got = {}
    for row in ('cstar', 'plain', 'teacher'):
        for head in ((None,) if row == 'plain' else P.ASPECTS):
            text, image = src.rows(row, head)
            aspects = P.ASPECTS if head is None else (head,)
            for (d, ro), r in score_all(src.frame, text, image, aspects, device).items():
                for a in aspects:
                    got[row, a, d, ro] = r['ap'][a]
    return compare(got, ref, TOL_REPLAY)


R22_RESULTS = P.R22_OUT / 'results'


def core_reference_r22(path=R22_RESULTS / 'core.json'):
    """r22's label mAP of c*, plain Qwen and the teacher's text (seed 0): {(row, aspect, direction, readout): map}, 81
    numbers. c* and the teacher are read at their own head; plain Qwen has one vector (aspect None in r22's records)."""
    doc = json.loads(Path(path).read_text())
    assert not doc['smoke'] and doc['n_paintings'] == P.N_PAINTINGS['art_val']
    ref = {}
    for r in doc['records']:
        if r['row'] not in ('cstar', 'plain', 'teacher') or r['entry'] not in P.ASPECTS or r['direction'] not in P.DIRECTIONS:
            continue
        if r['row'] != 'plain' and r['aspect'] != r['entry']:
            continue
        key = (r['row'], r['entry'], r['direction'], r['readout'])
        assert key not in ref, ('doubled r22 record', key)
        assert r['readout'] in P.READOUTS
        ref[key] = float(r['map'])
    assert len(ref) == 3 * 9 * len(P.READOUTS), len(ref)
    return ref


# ------------------------------------------------------------------------------------------------ late fusion
GATE_ALPHAS = (0.0, 0.7, 1.0)


def late_scores(src, device, alphas=GATE_ALPHAS):
    """{(a, head): score_all of the late rows}: seed 0 of each head, built through `tcl_forward.fuse` with `a` passed in.
    At a = 0 and a = 1 one block has weight zero, as in r22's check_joined; the read-outs act on the whole vector."""
    plain_t, plain_i = src.rows('plain', None)
    out = {}
    for head in P.ASPECTS:
        c_t, c_i = src.rows('cstar', head)
        for a in alphas:
            out[a, head] = score_all(src.frame, tcl_forward.fuse(c_t, plain_t, a), tcl_forward.fuse(c_i, plain_i, a), (head,), device)
    return out


def clause_late_label(late, ref):
    """Label mAP of each head's late fusion per cell: the best read-out's value, as r22's `label`."""
    got = {}
    for (a, head) in late:
        for d in P.DIRECTIONS:
            values = [late[a, head][d, ro]['ap'][head] for ro in P.READOUTS]
            assert np.isfinite(values).all(), ('non-finite label mAP at a read-out', a, head, d)     # max would drop a NaN
            got[a, head, d] = max(values)
    return compare(got, ref['label'], TOL_REPLAY)


def best_head(late, a, d):
    """r22's Table 2 pick: each head's read-out by MRR (first on ties), then the head by R@10 (first on ties)."""
    picks = {h: max(P.READOUTS, key=lambda ro: late[a, h][d, ro]['pair']['mrr']) for h in P.ASPECTS}
    head = max(picks, key=lambda h: late[a, h][d, picks[h]]['pair']['r10'])
    return head, picks[head], late[a, head][d, picks[head]]['pair']['r10']


def clause_late_pair(late, ref):
    """Pair R@10 of the picked head and read-out per alpha and direction, and the pick itself (a different head or
    read-out counts as a difference of 1)."""
    assert set(ref['pair']) == {(a, d) for a, _ in late for d in P.DIRECTIONS}, 'late and r22 have different cells'
    got, want = {}, {}
    for a in sorted({a for a, _ in late}):
        for d in P.DIRECTIONS:
            head, readout, r10 = best_head(late, a, d)
            expected = ref['pair'][a, d]
            got[a, d, 'r10'], want[a, d, 'r10'] = r10, expected['r10']
            got[a, d, 'pick'], want[a, d, 'pick'] = float((head, readout) != (expected['head'], expected['readout'])), 0.0
    return compare(got, want, TOL_REPLAY)


def late_reference_r22(path=R22_RESULTS / 'followup.json'):
    """r22's short check, joined rows at a = 0, 0.7, 1: label[(a, aspect, direction)] (best read-out) and
    pair[(a, direction)] = {head, readout, r10}."""
    rows = json.loads(Path(path).read_text())['joined']['rows']
    label, pair = {}, {}
    for a in GATE_ALPHAS:
        found = [r for r in rows if abs(r['alpha'] - a) < 1e-12]
        assert len(found) == 1, ('alpha rows in r22 followup', a, len(found))
        for key, value in found[0]['label'].items():
            aspect, d = key.split('/')
            label[a, aspect, d] = float(value)
        for d, value in found[0]['pair'].items():
            assert value['head'] in P.ASPECTS and value['readout'] in P.READOUTS
            pair[a, d] = dict(value)
    assert len(label) == 27 and len(pair) == 9
    return dict(label=label, pair=pair)


# ------------------------------------------------------------------------------------------------ MMArt
MM_READOUTS = ('r0', 'center')            # r22's MMArt run had no expansion
MM_KIND = 'unified'


def check_label_join(names, frame_labels, classes, r22_keys, r22_codes, r22_codebooks):
    """r22 coded MMArt labels locally (sorted class strings, rows in the index's order). Join by key, then for each aspect:
    the same rows are unlabelled; r22's classes and the frame's classes pair one to one; for emotion and style the paired
    classes have the same name. `frame_labels` {aspect: int64 by `names`}, `classes` {aspect: names of those codes}."""
    assert len(set(r22_keys)) == len(r22_keys) and len(set(names)) == len(names)
    lookup = {k: i for i, k in enumerate(r22_keys)}
    assert set(names) == set(lookup), 'MMArt keys of r22 and of the frame differ'
    rows = np.asarray([lookup[k] for k in names], dtype=np.int64)
    for aspect, ours in frame_labels.items():
        theirs = np.asarray(r22_codes[aspect])[rows]
        assert len(ours) == len(theirs)
        assert np.array_equal(ours < 0, theirs < 0), f'{aspect}: unlabelled rows differ'
        pairs = {(int(a), int(b)) for a, b in zip(theirs, ours) if a >= 0}
        assert len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs}), f'{aspect}: classes do not pair one to one'
        if aspect != 'genre':           # genre has integer codes in r22 and names in r15; the pairing is all there is to match
            for a, b in pairs:
                assert r22_codebooks[aspect][a] == classes[aspect][b], (aspect, r22_codebooks[aspect][a], classes[aspect][b])


def mmart_scores(src, device):
    """{(row, head): score_all}: plain Qwen and c* (seed 0, each head) on unified text, text to image, r0 and center, with
    every aspect's labels. The labels are first checked against r22's own coding."""
    src.check_labels()
    out = {}
    for row, head in [('plain', None)] + [('cstar', h) for h in P.ASPECTS]:
        text, image = src.rows(row, head)
        out[row, head] = score_all(src.frame, text, image, P.ASPECTS, device, readouts=MM_READOUTS, directions=('t2i',))
    return out


def clause_mmart_label(mm, ref):
    got = {(row, head, entry, ro): r['ap'][entry]
           for (row, head), scores in mm.items() for (d, ro), r in scores.items() for entry in P.ASPECTS}
    return compare(got, ref['label'], TOL_REPLAY)


def clause_mmart_pair(mm, ref):
    got = {(row, head, ro, name): r['pair'][name]
           for (row, head), scores in mm.items() for (d, ro), r in scores.items() for name in ('r10', 'mrr')}
    return compare(got, ref['pair'], TOL_REPLAY)


def mmart_reference_r22(path=R22_RESULTS / 'mmart.json'):
    """r22's MMArt validation numbers on the unified text (column e_unified), text to image, c* at seed 0 (each head) and
    plain Qwen: label[(row, head, entry, readout)] = label mAP, pair[(row, head, readout, 'r10' | 'mrr')]."""
    doc = json.loads(Path(path).read_text())
    assert not doc['smoke']
    label, pair = {}, {}
    for r in doc['records']:
        if r['row'] in ('plain', 'cstar') and r['column'] == 'e_' + MM_KIND and r['direction'] == 't2i' and r['readout'] in MM_READOUTS:
            key = (r['row'], r['aspect'], r['entry'], r['readout'])
            assert key not in label and r['entry'] in P.ASPECTS, ('doubled or unknown r22 record', key)
            label[key] = float(r['map'])
    for r in doc['pairs']:
        if r['row'] in ('plain', 'cstar') and r['column'] == 'e_' + MM_KIND and r['direction'] == 't2i' and r['readout'] in MM_READOUTS:
            assert r['n_query'] == P.N_PAINTINGS['mm_val']
            for name in ('r10', 'mrr'):
                key = (r['row'], r['aspect'], r['readout'], name)
                assert key not in pair, ('doubled r22 record', key)
                pair[key] = float(r[name])
    assert len(label) == 4 * 3 * len(MM_READOUTS) and len(pair) == 4 * 2 * len(MM_READOUTS), (len(label), len(pair))
    return dict(label=label, pair=pair)


# ------------------------------------------------------------------------------------------------ the gate
class Source:
    """One validation set read through the data door and the forward module: `frame`, and `rows(row, head)` giving
    (text, image) for plain Qwen (head None), c* and the teacher's text (head = aspect), all at seed 0."""

    def __init__(self, set_name, kind, device):
        self.set_name, self.kind, self.device = set_name, kind, device
        self.frame = tcl_data.frame(set_name, kind)
        self.cache = {}

    def rows(self, row, head):
        if (row, head) not in self.cache:
            assert (head is None) == (row == 'plain') and row in ('plain', 'cstar', 'teacher')
            self.cache[row, head] = tcl_forward.rows(self.set_name, self.kind, row, 'all' if head is None else head,
                                                     0 if row == 'cstar' else 'x', self.device)
        return self.cache[row, head]

    def check_labels(self):
        """MMArt only: r15's labels and order against r22's own coding of its val index, joined by key."""
        import tx_data
        import tx_extras
        import mt_paths
        assert self.set_name == 'mm_val'
        index = tx_data.mmart_index()
        codes, books = tx_extras.mmart_labels(index)
        check_label_join(self.frame['names'], {a: self.frame['labels'][a]['i'] for a in P.ASPECTS}, mt_paths.CLASSES,
                         index['key'].tolist(), codes, books)


@contextlib.contextmanager
def full_rows():
    """Gate (iii) replays full validation rows: TCL_SMOKE does not apply to them."""
    saved, P.SMOKE = P.SMOKE, 0
    try:
        yield
    finally:
        P.SMOKE = saved


def run_gate(art, mm, refs, items, device, draws=P.N_DRAWS):
    """The eight clauses, in order. `art` and `mm` are sources, `refs` r22's numbers {core, late, mmart}, `items` Table 2's."""
    clauses = {}

    def done(name, result):
        clauses[name] = result
        print(f"gate iii {name}: {'pass' if result['pass'] else 'FAIL'} worst {result['worst']:.3g} n {result['n']}", flush=True)

    done('core_label', clause_core(art, refs['core'], device))
    late = late_scores(art, device)
    done('late_label', clause_late_label(late, refs['late']))
    done('late_pair', clause_late_pair(late, refs['late']))
    mmart = mmart_scores(mm, device)
    done('mmart_label', clause_mmart_label(mmart, refs['mmart']))
    done('mmart_pair', clause_mmart_pair(mmart, refs['mmart']))
    done('interval_hand', clause_interval_hand())
    done('interval_tie', clause_interval_tie())
    done('table2', clause_table2(items, draws=draws))
    return clauses


def gate_iii(device='cuda', out=None):
    """Spec §3.5 (iii) on the full validation sets (TCL_SMOKE does not apply). Writes OUT/stage0/gate_iii.json.
    An older file is removed first: a crash must not leave an earlier pass on disk."""
    gate_path(out).unlink(missing_ok=True)
    with full_rows():
        refs = dict(core=core_reference_r22(), late=late_reference_r22(), mmart=mmart_reference_r22())
        clauses = run_gate(Source('art_val', P.CAPTION, device), Source('mm_val', MM_KIND, device), refs, table2_items(), device)
    return write_gate(clauses, out)


def main(argv=None):
    """Run the gate, print the overall result, return the exit code: 0 for a pass, 1 for a fail (a crash raises)."""
    argv = sys.argv[1:] if argv is None else argv
    result = gate_iii(argv[0] if argv else 'cuda')
    print('gate iii overall:', 'pass' if result['overall'] else 'FAIL')
    return 0 if result['overall'] else 1


if __name__ == '__main__':
    sys.exit(main())
