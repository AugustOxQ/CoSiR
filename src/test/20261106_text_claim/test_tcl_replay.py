"""Gate (iii) checks (spec §3.5 (iii)): each clause fails on a wrong value, passes on a right one.

Synthetic cases stand in for the full validation rows; the oracle for the scoring clauses is r22's own `tx_metrics`. The
full gate (GPU, full rows) runs once through the queue and is not part of this file.
"""
import json

import numpy as np
import pytest

import tcl_paths as P
import tcl_metrics
import tcl_replay as R


# ------------------------------------------------------------------------------------------------ compare, gate file
def test_compare_reports_worst_difference_and_count():
    got = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    want = {'a': 1.0 + 4e-6, 'b': 2.0 - 9e-6, 'c': 3.0}
    assert R.compare(got, want, 1e-5) == {'pass': True, 'worst': pytest.approx(9e-6, rel=1e-6), 'n': 3}


def test_compare_fails_just_over_the_tolerance_and_never_widens_it():
    got = {'a': 0.5}
    assert R.compare(got, {'a': 0.5 + 1.1e-5}, 1e-5)['pass'] is False
    assert R.compare(got, {'a': 0.5 + 0.9e-5}, 1e-5)['pass'] is True


def test_compare_crashes_on_key_mismatch_or_nonfinite():
    with pytest.raises(AssertionError):
        R.compare({'a': 1.0}, {'a': 1.0, 'b': 2.0}, 1e-5)
    with pytest.raises(AssertionError):
        R.compare({'a': float('nan')}, {'a': 1.0}, 1e-5)
    with pytest.raises(AssertionError):
        R.compare({}, {}, 1e-5)


def test_gate_file_has_the_schema_and_overall_is_all_clauses(tmp_path):
    ok = dict(R.compare({'a': 1.0}, {'a': 1.0}, 1e-5))
    bad = dict(R.compare({'a': 1.0}, {'a': 2.0}, 1e-5))
    R.write_gate({'one': ok, 'two': bad}, tmp_path)
    doc = json.loads((tmp_path / 'stage0/gate_iii.json').read_text())
    assert set(doc) == {'clauses', 'overall'} and doc['overall'] is False
    assert doc['clauses']['one'] == {'pass': True, 'worst': 0.0, 'n': 1}
    assert doc['clauses']['two']['pass'] is False and doc['clauses']['two']['worst'] == 1.0
    R.write_gate({'one': ok}, tmp_path)
    assert json.loads((tmp_path / 'stage0/gate_iii.json').read_text())['overall'] is True


# ------------------------------------------------------------------------------------------------ interval clauses
def _dedup(sum_c, sum_m, counts, I, S):
    """A wrong replicate: repeats of S count once."""
    w = np.bincount(I, minlength=len(counts)).astype(np.float64)
    den = w @ counts
    values = (sum_c @ w) / den
    return float(values[np.unique(S)].mean() - (w @ sum_m) / den)


def test_interval_hand_clause_passes_on_the_hand_values():
    got = R.clause_interval_hand()
    assert got['pass'] is True and got['worst'] <= 1e-12 and got['n'] == 5


def test_interval_hand_clause_fails_when_repeats_of_s_are_dropped():
    got = R.clause_interval_hand(replicate=_dedup)
    assert got['pass'] is False and got['worst'] > 1e-3


def test_interval_tie_clause_passes_for_the_shipped_code():
    got = R.clause_interval_tie()
    assert got['pass'] is True and got['worst'] <= 1e-12 and got['n'] == 3 * 4


def test_interval_tie_clause_fails_when_painting_interval_disagrees_with_replicate():
    def shifted(cand, comp, pid, draws=10000):
        point, lower, upper, m = tcl_metrics.painting_interval(cand, comp, pid, draws=draws)
        return point, lower + 1e-9, upper, m
    got = R.clause_interval_tie(interval=shifted)
    assert got['pass'] is False and got['worst'] == pytest.approx(1e-9, rel=1e-3)


# ------------------------------------------------------------------------------------------------ Table 2
def synthetic_item(shift=0.0, seed=5, n=400, n_pid=60):
    rng = np.random.default_rng(seed)
    pid = rng.integers(0, n_pid, size=n).astype(np.int64)
    comp = rng.random(n)
    cand = np.clip(comp + rng.normal(0.05, 0.1, size=n), 0, 1)
    cand[::37], comp[::37] = np.nan, np.nan          # unscored queries leave the interval
    keep = np.isfinite(cand)
    point, lower, upper, m = tcl_metrics.painting_interval(cand[keep][None], comp[keep], pid[keep], draws=2000)
    return dict(key=('emotion', 't2i', 'plain'), cand=cand, comp=comp, pid=pid, mean=point, n_paired=int(keep.sum()),
                lo=lower + shift, hi=upper + shift)


def test_table2_clause_passes_inside_and_fails_outside_the_bound():
    assert R.clause_table2([synthetic_item(2.9e-4)], draws=2000) == {'pass': True, 'worst': pytest.approx(2.9e-4, abs=1e-9), 'n': 2}
    got = R.clause_table2([synthetic_item(3.1e-4)], draws=2000)
    assert got['pass'] is False and got['worst'] == pytest.approx(3.1e-4, abs=1e-9)


def test_table2_clause_checks_each_bound_of_each_interval():
    good, bad = synthetic_item(0.0), synthetic_item(0.0, seed=6)
    bad['key'] = ('style', 'i2t', 'teacher')
    bad['hi'] += 1e-3
    got = R.clause_table2([good, bad], draws=2000)
    assert got['pass'] is False and got['n'] == 4 and got['worst'] == pytest.approx(1e-3, abs=1e-9)


def test_table2_clause_crashes_when_the_paired_queries_differ():
    item = synthetic_item()
    item['n_paired'] += 1
    with pytest.raises(AssertionError):
        R.clause_table2([item], draws=2000)


def test_table2_items_are_read_from_r22s_own_files():
    items = R.table2_items()
    assert len(items) == 18 and {i['key'][2] for i in items} == {'plain', 'teacher'}
    assert {i['key'][:2] for i in items} == {(a, d) for a in P.ASPECTS for d in P.DIRECTIONS}
    teacher = next(i for i in items if i['key'] == ('emotion', 't2i', 'teacher'))
    assert teacher['lo'] == pytest.approx(-0.0070661890624830375, abs=1e-15)
    assert teacher['n_paired'] == 30872 == len(teacher['cand']) == len(teacher['pid'])


# ------------------------------------------------------------------------------------------------ scoring clauses
import tx_metrics   # r22's own ranking: the oracle for the synthetic cases


class FakeSource:
    """A small stand-in for a set: `frame` and `rows(row, head)` as the real source gives them. Text and image rows carry
    different offsets, so a flipped pair of means changes the centred rankings."""

    def __init__(self, n_i=14, n_t=40, dim=8, seed=0, one_text_per_painting=False):
        rng = np.random.default_rng(seed)
        self.rng, self.dim = rng, dim
        self.cache = {}
        tp = np.arange(n_i, dtype=np.int64) if one_text_per_painting else \
            np.concatenate([np.arange(n_i), rng.integers(0, n_i, size=n_t - n_i)]).astype(np.int64)
        labels = {}
        for a, n_class in zip(P.ASPECTS, (3, 4, 2)):
            i = rng.integers(0, n_class, size=n_i).astype(np.int64)
            i[::5] = -1
            t = i[tp].copy() if a != 'emotion' else rng.integers(-1, n_class, size=len(tp)).astype(np.int64)
            labels[a] = dict(t=t, i=i)
        self.frame = dict(n_i=n_i, n_t=len(tp), tp=tp, ip=np.arange(n_i, dtype=np.int64), labels=labels)
        self.shared = rng.normal(size=(n_i, dim))      # texts and images of one painting are alike

    def check_labels(self):
        self.checked = True

    def unit_rows(self, offset, n_rows, painting):
        x = self.shared[painting] + self.rng.normal(size=(n_rows, self.dim)) * 0.7 + offset
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    def rows(self, row, head):
        if (row, head) not in self.cache:
            self.cache[row, head] = (self.unit_rows(1.5 + len(self.cache) * 0.1, self.frame['n_t'], self.frame['tp']),
                                     self.unit_rows(-1.0 + len(self.cache) * 0.1, self.frame['n_i'], self.frame['ip']))
        return self.cache[row, head]


def finite_mean(x):
    return float(x[np.isfinite(x)].mean())


def oracle(src, text, image, aspects, direction, readout):
    """r22's evaluate, fitting its own means on the rows: {aspect: label mAP}, first-hit ranks."""
    f = src.frame
    q, qp = (text, f['tp']) if direction[0] == 't' else (image, f['ip'])
    g, gp = (text, f['tp']) if direction[2] == 't' else (image, f['ip'])
    entries = {a: dict(q=f['labels'][a][direction[0]], g=f['labels'][a][direction[2]]) for a in aspects}
    got = tx_metrics.evaluate(q, g, qp, gp, entries, direction == 't2t', readout, 'cpu')
    return {a: finite_mean(v) for a, v in got['ap'].items()}, got['first_rank']


def core_reference(src):
    ref = {}
    for row in ('cstar', 'plain', 'teacher'):
        for a in P.ASPECTS:
            text, image = src.rows(row, None if row == 'plain' else a)
            for d in P.DIRECTIONS:
                for ro in P.READOUTS:
                    ref[row, a, d, ro] = oracle(src, text, image, (a,), d, ro)[0][a]
    return ref


def test_core_clause_passes_on_r22s_numbers_for_nine_cells_times_three_readouts_and_three_rows():
    src = FakeSource()
    ref = core_reference(src)
    got = R.clause_core(src, ref, 'cpu')
    assert got['pass'] is True and got['n'] == 81 and got['worst'] < 1e-5


@pytest.mark.parametrize('cell', [('cstar', 'emotion', 't2i', 'r0'), ('plain', 'genre', 'i2t', 'center'),
                                  ('teacher', 'style', 't2t', 'center_k10')])
def test_core_clause_fails_when_one_number_is_wrong(cell):
    src = FakeSource()
    ref = core_reference(src)
    ref[cell] += 2e-5
    got = R.clause_core(src, ref, 'cpu')
    assert got['pass'] is False and got['worst'] == pytest.approx(2e-5, rel=0.05)


def test_core_clause_catches_a_flipped_means_pair_in_image_to_text(monkeypatch):
    src = FakeSource()
    ref = core_reference(src)
    monkeypatch.setattr(R, 'means_for', lambda d, tm, im: (tm, im) if d != 't2t' else (tm, tm))
    got = R.clause_core(src, ref, 'cpu')
    assert got['pass'] is False and got['worst'] > 1e-3


def test_core_reference_is_read_from_r22s_core_json():
    ref = R.core_reference_r22()
    assert len(ref) == 81 and {k[0] for k in ref} == {'cstar', 'plain', 'teacher'}
    assert ref['plain', 'emotion', 't2i', 'r0'] == 0.21786029636859894


def test_core_reference_crashes_on_a_missing_or_doubled_record(tmp_path):
    doc = json.loads((P.R22_OUT / 'results/core.json').read_text())
    short = dict(doc, records=doc['records'][1:])
    path = tmp_path / 'core.json'
    path.write_text(json.dumps(short))
    with pytest.raises(AssertionError):
        R.core_reference_r22(path)
    path.write_text(json.dumps(dict(doc, records=doc['records'] + doc['records'][:1])))
    with pytest.raises(AssertionError):
        R.core_reference_r22(path)


# ------------------------------------------------------------------------------------------------ late fusion clauses
def late_reference(src, alphas=(0.0, 0.7, 1.0)):
    """r22's check_joined, restated: the late vector by r22's own concatenation, ranked by r22's evaluate."""
    plain_t, plain_i = src.rows('plain', None)
    label, pair = {}, {}
    for x in alphas:
        w_c, w_p = float(np.sqrt(1 - x)), float(np.sqrt(x))
        per_head = {}
        for a in P.ASPECTS:
            c_t, c_i = src.rows('cstar', a)
            t = np.concatenate((w_c * c_t, w_p * plain_t), axis=1).astype(np.float32)
            i = np.concatenate((w_c * c_i, w_p * plain_i), axis=1).astype(np.float32)
            per_head[a] = {}
            for d in P.DIRECTIONS:
                per_head[a][d] = {}
                for ro in P.READOUTS:
                    ap, first = oracle(src, t, i, (a,), d, ro)
                    per_head[a][d][ro] = dict(ap=ap[a], pair=tx_metrics.summarize_pair(first))
        for a in P.ASPECTS:
            for d in P.DIRECTIONS:
                label[x, a, d] = max(per_head[a][d][ro]['ap'] for ro in P.READOUTS)
        for d in P.DIRECTIONS:
            picks = {a: max(P.READOUTS, key=lambda ro: per_head[a][d][ro]['pair']['mrr']) for a in P.ASPECTS}
            head = max(picks, key=lambda a: per_head[a][d][picks[a]]['pair']['r10'])
            pair[x, d] = dict(head=head, readout=picks[head], r10=per_head[head][d][picks[head]]['pair']['r10'])
    return dict(label=label, pair=pair)


def test_late_clauses_pass_on_r22s_formula_at_a_0_07_1():
    src = FakeSource(seed=2)
    ref = late_reference(src)
    late = R.late_scores(src, 'cpu')
    label, pair = R.clause_late_label(late, ref), R.clause_late_pair(late, ref)
    assert label['pass'] and label['n'] == 27 and label['worst'] < 1e-5
    assert pair['pass'] and pair['n'] == 9 * 2 and pair['worst'] < 1e-5


def test_late_label_clause_fails_on_one_wrong_number():
    src = FakeSource(seed=2)
    ref = late_reference(src)
    ref['label'][0.7, 'style', 'i2t'] -= 2e-5
    got = R.clause_late_label(R.late_scores(src, 'cpu'), ref)
    assert got['pass'] is False and got['worst'] == pytest.approx(2e-5, rel=0.05)


def test_late_pair_clause_fails_on_a_wrong_r10_head_or_readout():
    src = FakeSource(seed=2)
    ref = late_reference(src)
    late = R.late_scores(src, 'cpu')
    wrong_r10 = {**ref, 'pair': {**ref['pair'], (1.0, 't2i'): dict(ref['pair'][1.0, 't2i'], r10=ref['pair'][1.0, 't2i']['r10'] + 2e-5)}}
    got = R.clause_late_pair(late, wrong_r10)
    assert got['pass'] is False and got['worst'] == pytest.approx(2e-5, rel=0.05)
    other = next(a for a in P.ASPECTS if a != ref['pair'][0.7, 'i2t']['head'])
    wrong_head = {**ref, 'pair': {**ref['pair'], (0.7, 'i2t'): dict(ref['pair'][0.7, 'i2t'], head=other)}}
    assert R.clause_late_pair(late, wrong_head)['pass'] is False
    other = next(r for r in P.READOUTS if r != ref['pair'][0.0, 't2t']['readout'])
    wrong_readout = {**ref, 'pair': {**ref['pair'], (0.0, 't2t'): dict(ref['pair'][0.0, 't2t'], readout=other)}}
    assert R.clause_late_pair(late, wrong_readout)['pass'] is False


def test_late_scores_use_the_a_that_is_passed_in(monkeypatch):
    src = FakeSource(seed=2)
    seen = []
    import tcl_forward
    real = tcl_forward.fuse
    monkeypatch.setattr(R.tcl_forward, 'fuse', lambda c, p, a=P.A: seen.append(a) or real(c, p, a))
    R.late_scores(src, 'cpu')
    assert sorted(set(seen)) == [0.0, 0.7, 1.0] and len(seen) == 3 * 3 * 2


def test_late_reference_is_read_from_r22s_followup_json():
    ref = R.late_reference_r22()
    assert len(ref['label']) == 27 and len(ref['pair']) == 9
    assert ref['label'][0.0, 'emotion', 't2i'] == 0.2649883031845093
    assert set(ref['pair'][0.0, 't2i']) == {'head', 'readout', 'r10'} and ref['pair'][0.0, 't2i']['head'] == 'genre'


def test_late_reference_crashes_when_an_alpha_is_missing(tmp_path):
    doc = json.loads((P.R22_OUT / 'results/followup.json').read_text())
    doc['joined']['rows'] = [r for r in doc['joined']['rows'] if r['alpha'] != 0.7]
    path = tmp_path / 'followup.json'
    path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        R.late_reference_r22(path)


# ------------------------------------------------------------------------------------------------ MMArt clauses
def mmart_reference(src):
    label, pair = {}, {}
    for row in ('plain', 'cstar'):
        for head in ((None,) if row == 'plain' else P.ASPECTS):
            text, image = src.rows(row, head)
            for ro in ('r0', 'center'):
                ap, first = oracle(src, text, image, P.ASPECTS, 't2i', ro)
                for entry in P.ASPECTS:
                    label[row, head, entry, ro] = ap[entry]
                summary = tx_metrics.summarize_pair(first)
                pair[row, head, ro, 'r10'], pair[row, head, ro, 'mrr'] = summary['r10'], summary['mrr']
    return dict(label=label, pair=pair)


def test_mmart_clauses_pass_on_r22s_numbers_and_count_their_cells():
    src = FakeSource(seed=3, one_text_per_painting=True)
    ref = mmart_reference(src)
    mm = R.mmart_scores(src, 'cpu')
    label, pair = R.clause_mmart_label(mm, ref), R.clause_mmart_pair(mm, ref)
    assert label['pass'] and label['n'] == 4 * 3 * 2 and label['worst'] < 1e-5
    assert pair['pass'] and pair['n'] == 4 * 2 * 2 and pair['worst'] < 1e-5


def test_mmart_clauses_fail_on_one_wrong_number():
    src = FakeSource(seed=3, one_text_per_painting=True)
    ref = mmart_reference(src)
    mm = R.mmart_scores(src, 'cpu')
    ref['label']['cstar', 'genre', 'style', 'center'] += 2e-5
    ref['pair']['plain', None, 'r0', 'r10'] -= 2e-5
    assert R.clause_mmart_label(mm, ref)['pass'] is False
    assert R.clause_mmart_pair(mm, ref)['pass'] is False


def test_mmart_scores_check_the_labels_first():
    class Mismatched(FakeSource):
        def check_labels(self):
            raise AssertionError('labels differ')
    with pytest.raises(AssertionError, match='labels differ'):
        R.mmart_scores(Mismatched(one_text_per_painting=True), 'cpu')


def join_case():
    """r22's key order and codes differ from the frame's; the labelled partitions agree."""
    names = ['k3', 'k1', 'k2', 'k0']
    classes = {'emotion': ('amusement', 'awe', 'fear'), 'style': ('baroque', 'cubism'), 'genre': ('a', 'b', 'c')}
    frame_labels = {'emotion': np.asarray([2, 0, -1, 1]), 'style': np.asarray([1, 0, 0, -1]),
                    'genre': np.asarray([0, 2, 1, 1])}
    r22_keys = ['k0', 'k1', 'k2', 'k3']
    r22_codes = {'emotion': np.asarray([0, 1, -1, 2]), 'style': np.asarray([-1, 1, 1, 0]), 'genre': np.asarray([5, 7, 5, 4])}
    r22_books = {'emotion': ['awe', 'amusement', 'fear'], 'style': ['cubism', 'baroque'], 'genre': ['0', '1', '2', '3', '4', '5', '6', '7']}
    return names, classes, frame_labels, r22_keys, r22_codes, r22_books


def test_mmart_label_join_matches_by_key_and_class_name():
    names, classes, frame_labels, r22_keys, r22_codes, r22_books = join_case()
    # k0: frame emotion 1 (awe) vs r22 code 0 = awe; k1: 0 (amusement) vs 1 = amusement; k3: 2 (fear) vs 2 = fear.
    R.check_label_join(names, frame_labels, classes, r22_keys, r22_codes, r22_books)


@pytest.mark.parametrize('mutate', ['emotion name', 'unlabelled', 'merged classes', 'missing key'])
def test_mmart_label_join_crashes_when_the_labels_differ(mutate):
    names, classes, frame_labels, r22_keys, r22_codes, r22_books = join_case()
    if mutate == 'emotion name':
        r22_books['emotion'] = ['amusement', 'awe', 'fear']
    elif mutate == 'unlabelled':
        r22_codes['style'][0] = 0
    elif mutate == 'merged classes':
        r22_codes['genre'][:] = 5
    else:
        r22_keys[0] = 'other'
    with pytest.raises(AssertionError):
        R.check_label_join(names, frame_labels, classes, r22_keys, r22_codes, r22_books)


def test_mmart_reference_is_read_from_r22s_mmart_json():
    ref = R.mmart_reference_r22()
    assert len(ref['label']) == 4 * 3 * 2 and len(ref['pair']) == 4 * 2 * 2
    assert ref['pair']['cstar', 'genre', 'center', 'r10'] == 0.7339226902409429
    assert ref['label']['plain', None, 'genre', 'center'] == 0.4726029336452484


def test_mmart_reference_crashes_on_a_missing_record(tmp_path):
    doc = json.loads((P.R22_OUT / 'results/mmart.json').read_text())
    first = next(i for i, r in enumerate(doc['records']) if r['column'] == 'e_unified' and r['row'] == 'cstar')
    doc['records'] = doc['records'][:first] + doc['records'][first + 1:]
    path = tmp_path / 'mmart.json'
    path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        R.mmart_reference_r22(path)


# ------------------------------------------------------------------------------------------------ the whole gate
CLAUSES = ('core_label', 'late_label', 'late_pair', 'mmart_label', 'mmart_pair', 'interval_hand', 'interval_tie', 'table2')


def fake_gate_inputs():
    art, mm = FakeSource(seed=2), FakeSource(seed=3, one_text_per_painting=True)
    refs = dict(core=core_reference(art), late=late_reference(art), mmart=mmart_reference(mm))
    return art, mm, refs, [synthetic_item()]


def test_run_gate_has_one_flag_per_clause_and_overall_follows_them(tmp_path):
    art, mm, refs, items = fake_gate_inputs()
    clauses = R.run_gate(art, mm, refs, items, 'cpu', draws=2000)
    assert tuple(clauses) == CLAUSES and all(c['pass'] for c in clauses.values())
    assert R.write_gate(clauses, tmp_path)['overall'] is True


@pytest.mark.parametrize('where, name', [('core', 'core_label'), ('late', 'late_label'), ('mmart', 'mmart_label')])
def test_run_gate_fails_the_clause_whose_reference_is_wrong(tmp_path, where, name):
    art, mm, refs, items = fake_gate_inputs()
    table = refs[where]['label'] if where != 'core' else refs['core']
    key = next(iter(table))
    table[key] += 1e-4
    clauses = R.run_gate(art, mm, refs, items, 'cpu', draws=2000)
    assert [n for n, c in clauses.items() if not c['pass']] == [name]
    assert R.write_gate(clauses, tmp_path)['overall'] is False


def test_full_rows_lifts_smoke_inside_and_restores_it():
    before = P.SMOKE
    P.SMOKE = 7
    try:
        with R.full_rows():
            assert P.SMOKE == 0
        assert P.SMOKE == 7
    finally:
        P.SMOKE = before
