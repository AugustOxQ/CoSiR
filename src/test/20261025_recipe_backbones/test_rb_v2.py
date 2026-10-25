"""Seams of rb_v2 (ticket 15): the label rules, R-COL eligibility, the bootstrap draw order, the statistic, the
comparison with a verdict file, and a synthetic end-to-end run. The acceptance test reproduces the CT's saved
verdict_stage1.json from its saved outputs (real data; RB_SLOW=1 adds the full-scale one)."""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_mutual_info_score

import rb_paths  # noqa: F401  (sys.path)
import rb_v2

slow = pytest.mark.skipif(not os.environ.get('RB_SLOW'), reason='real full-scale outputs: set RB_SLOW=1')
CT_DIR = rb_paths.MAIN_TEST / '20261020_claim_test'

NINE = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness', 'something else']


# ---- labels (rule §3.2 and §3.3) ---------------------------------------------------------------------------

def annotations():
    """Every one of the nine names occurs; painting 'tie' has awe x2 and anger x2; 'ses' has something else x2, fear."""
    rows = [('a', 'amusement')] * 3 + [('a', 'sadness')]
    rows += [('tie', 'awe'), ('tie', 'anger'), ('tie', 'awe'), ('tie', 'anger')]
    rows += [('ses', 'something else'), ('ses', 'fear'), ('ses', 'something else')]
    rows += [('c', 'contentment'), ('d', 'disgust'), ('e', 'excitement'), ('x', 'amusement'), ('x', 'anger'),
             ('x', 'awe'), ('x', 'contentment'), ('x', 'disgust'), ('x', 'excitement'), ('x', 'fear'),
             ('x', 'sadness'), ('x', 'something else'), ('x', 'sadness')]
    return [dict(painting=p, emotion=e, caption='c', art_style='s', image='i', image_id='j') for p, e in rows]


def test_held_labels_plurality_ties_to_smallest_code():
    names = sorted(['a', 'tie', 'ses', 'c', 'd', 'e', 'x'])
    got = rb_v2.held_emotion_labels(names, annotations())
    want = {'a': 0, 'tie': 1, 'ses': 8, 'c': 3, 'd': 4, 'e': 5, 'x': 7}   # tie: anger (1) < awe (2); x: sadness x2
    np.testing.assert_array_equal(got, [want[n] for n in names])
    assert got.dtype == np.int64


def test_held_labels_assert_the_rest():
    ann = annotations()
    with pytest.raises(AssertionError):
        rb_v2.held_emotion_labels(['a', 'zzz'], ann)                              # painting without a caption
    with pytest.raises(AssertionError):
        rb_v2.held_emotion_labels(['tie', 'a'], ann)                              # not sorted
    bad = ann + [dict(painting='a', emotion='joy')]
    with pytest.raises(AssertionError):
        rb_v2.held_emotion_labels(['a'], bad)                                     # a tenth name


def a28_tables():
    """Parquet rows join to the CSV by row_id = CSV data-row position; 'other' becomes 'something else'."""
    csv = pd.DataFrame({
        'emotion': ['anger', 'other', 'awe', 'other', 'awe', 'fear', 'sadness', 'x'],
        'art_style': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2', 'S1', 'S1'],
        'genre': ['g1', 'g1', 'g1', 'g2', 'g2', 'g2', 'g1', 'g1'],
        'painting': ['p1', 'p1', 'p1', 'p0', 'p0', 'p0', 'p1', 'zz'],
    })
    # p0: other, awe, fear -> (something else 8, awe 2, fear 6) all one vote: ties to the smallest code = awe (2)
    # p1: anger, other, awe, (row 6 sadness not in the parquet) -> anger (1) < awe (2) < other=8: anger
    parquet = pd.DataFrame({'row_id': [3, 4, 5, 0, 1, 2], 'painting': ['p0', 'p0', 'p0', 'p1', 'p1', 'p1']})
    return parquet, csv


def test_a28_labels_join_by_row_id_other_to_something_else_ties_smallest():
    parquet, csv = a28_tables()
    names, labels, info = rb_v2.a28_emotion_labels(parquet, csv)
    assert names == ['p0', 'p1']
    np.testing.assert_array_equal(labels, [2, 1])
    assert info == dict(paintings=2, ties=2, unique=0)


def test_a28_labels_assert_a_painting_mismatch():
    parquet, csv = a28_tables()
    parquet.loc[0, 'row_id'] = 7        # CSV row 7 is painting 'zz', the parquet says p0
    with pytest.raises(AssertionError):
        rb_v2.a28_emotion_labels(parquet, csv)


# ---- R-COL eligibility (CT rule §4) -----------------------------------------------------------------------

def run(seed, cut, kept, share, version='clip_roberta'):
    return dict(version=version, seed=seed, cut=cut, kept_count=kept, share=share, collapsed=None)


def test_rcol_strict_and_k40_rules_at_the_thresholds():
    runs = []
    cases = {0: (9, .1), 1: (10, .5), 2: (90, .1), 3: (91, .1), 4: (50, .5000001)}
    for seed, (kept, share) in cases.items():
        runs.append(run(seed, 'strict', kept, share))
    for seed, (_, share) in cases.items():
        runs.append(run(seed, 'k40', 40, share))                       # kept count does not matter for k40
    runs.append(run(0, 'strict', 1, 1.0, version='qwen_qwen'))         # another Stage 1 version: ignored
    assert rb_v2.rcol_eligible_seeds(runs, 'strict') == [1, 2]
    assert rb_v2.rcol_eligible_seeds(runs, 'k40') == [0, 1, 2, 3]


def test_rcol_nonfinite_held_outputs_count_as_collapsed():
    runs = [run(s, 'k40', 40, .1) for s in range(5)]
    assert rb_v2.rcol_eligible_seeds(runs, 'k40', nonfinite=(1, 3)) == [0, 2, 4]


def test_rcol_one_run_per_seed_and_cut():
    runs = [run(0, 'k40', 40, .1), run(0, 'k40', 40, .1)]
    with pytest.raises(AssertionError):
        rb_v2.rcol_eligible_seeds(runs, 'k40')


# ---- partitions --------------------------------------------------------------------------------------------

def test_own_is_argmax_lowest_index_on_ties():
    prob = np.array([[.2, .7, .7], [.9, .1, .9], [.1, .1, .1]], dtype=np.float32)
    own = rb_v2.own_partition(prob)
    np.testing.assert_array_equal(own, [[1, 0, 0]])
    assert own.dtype == np.int64


def test_r20_partition_shape_dtype_and_unit_norm_checks():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    part = rb_v2.r20_partition(x.astype(np.float16))                   # saved float16 -> float32 cast
    assert part.shape == (3, 300) and part.dtype == np.int64 and part.max() < 20
    with pytest.raises(AssertionError):
        rb_v2.r20_partition((x * 1.01).astype(np.float16))              # |norm - 1| < 1e-3 is checked, not repaired
    p = rng.random(size=(300, 40)).astype(np.float32)
    got = rb_v2.r20_partition(rb_v2.l2_probabilities(p))
    assert got.shape == (3, 300)
    np.testing.assert_allclose(np.linalg.norm(rb_v2.l2_probabilities(p), axis=1), 1, atol=1e-6)
    assert rb_v2.l2_probabilities(p).dtype == np.float32


# ---- the draw order (rule §3.2) ----------------------------------------------------------------------------

def test_draw_plan_order_and_arms_without_seeds_draw_nothing():
    n, draws = 17, 4
    arm_seeds = {'K2': [0, 1, 2, 3, 4], 'B1': [0, 1, 2, 3, 4], 'B2': [], 'B3': [], 'B4': [1, 3], 'B6': [0, 2, 4],
                 'M2': [0, 1, 2, 3, 4]}
    idx, seeds = rb_v2.draw_plan(n, arm_seeds, draws)
    rng = np.random.default_rng(0)                                      # the rule's text, replayed by hand
    for b in range(draws):
        np.testing.assert_array_equal(idx[b], rng.integers(0, n, size=n))
        for name in ('K2', 'B1', 'B4', 'B6', 'M2'):
            pool = arm_seeds[name]
            assert list(seeds[b][name]) == list(rng.choice(pool, size=len(pool), replace=True))
        assert 'B2' not in seeds[b] and 'B3' not in seeds[b]


# ---- the statistic -----------------------------------------------------------------------------------------

def toy(n=120, seeds=(0, 1), rng_seed=3):
    """Labels over 4 classes; the candidate follows them closely, the entries less so."""
    rng = np.random.default_rng(rng_seed)
    labels = rng.integers(0, 4, size=n)

    def noisy(p):
        out = labels.copy()
        flip = rng.random(n) < p
        out[flip] = rng.integers(0, 6, size=flip.sum())
        return out

    def arm(name, p, own=False):
        parts = {}
        for s in seeds:
            parts[s] = {'r20': np.stack([noisy(p) for _ in range(3)])}
            if own:
                parts[s]['own'] = noisy(p)[None, :]
        return rb_v2.Arm(name, parts)

    return labels, arm('K2', .1), [arm('B1', .5, True), arm('B4', .3, True), arm('M2', .6, True)]


def naive(labels, cand, entries, draws):
    """The rule's text as plain loops."""
    def a(arm, reading, seeds, I):
        return np.mean([np.mean([adjusted_mutual_info_score(labels[I], p[I]) for p in arm.parts[s][reading]])
                        for s in seeds])
    I0 = np.arange(len(labels))
    ents = [(e, r) for e in entries for r in ('r20', 'own')]
    A = {f'{e.name}-{r}': a(e, r, sorted(e.parts), I0) for e, r in ents}
    Ak = a(cand, 'r20', sorted(cand.parts), I0)
    d = Ak - max(A.values())
    rng = np.random.default_rng(0)
    db = []
    for _ in range(draws):
        I = rng.integers(0, len(labels), size=len(labels))
        S = {}
        for arm in [cand] + entries:
            pool = sorted(arm.parts)
            S[arm.name] = rng.choice(pool, size=len(pool), replace=True)
        ak = a(cand, 'r20', S['K2'], I)
        db.append(ak - max(a(e, r, S[e.name], I) for e, r in ents))
    return Ak, A, d, db


def test_rederive_matches_the_plain_loops_serial_and_parallel():
    labels, cand, entries = toy()
    Ak, A, d, db = naive(labels, cand, entries, 6)
    for workers in (1, 2):
        res = rb_v2.rederive(labels, cand, entries, draws=6, workers=workers)
        assert res['A_cand'] == pytest.approx(Ak, abs=1e-12)
        assert res['d'] == pytest.approx(d, abs=1e-12)
        assert set(res['A_e']) == set(A)
        for k in A:
            assert res['A_e'][k] == pytest.approx(A[k], abs=1e-12)
        np.testing.assert_allclose(res['bootstrap_d'], db, atol=1e-12)
        lo, hi = np.percentile(db, [2.5, 97.5])
        assert res['interval'] == pytest.approx([lo, hi], abs=1e-12)


def test_rederive_entries_without_seeds_are_left_out():
    labels, cand, entries = toy()
    ghost = rb_v2.Arm('B2', {})
    res = rb_v2.rederive(labels, cand, [entries[0], ghost, *entries[1:]], draws=3, workers=1)
    ref = rb_v2.rederive(labels, cand, entries, draws=3, workers=1)
    assert res['d'] == ref['d'] and res['bootstrap_d'] == ref['bootstrap_d']
    assert res['eligible_seeds']['B2'] == [] and 'B2-r20' not in res['A_e']


# ---- comparison with a verdict file ------------------------------------------------------------------------

def fake_result():
    return dict(A_cand=.1, d=.012, A_e={'B1-r20': .088, 'B1-own': .085}, interval=[.009, .015],
                eligible_seeds={'K2': [0, 1], 'B1': [0, 1], 'B2': []}, bootstrap_d=[.01, .011])


def write_verdict(path, block, name='K2'):
    path.write_text(json.dumps({'schema': 1, 'stage': 3, 'candidates': {name: block}}))


def good_block():
    return dict(d=.012, A_K2=.1, A_e={'B1-r20': .088, 'B1-own': .085}, interval=[.009, .015],
                eligibility=dict(eligible_seeds={'K2': [0, 1], 'B1': [0, 1], 'B2': []}), bootstrap_d=[.01, .011])


def test_compare_agrees_within_tolerance_and_reports_a_planted_difference(tmp_path):
    f = tmp_path / 'v.json'
    write_verdict(f, good_block())
    ok = rb_v2.compare(fake_result(), rb_v2.read_block(f, 'K2'))
    assert ok['agree'] and not ok['mismatches']
    block = good_block()
    block['d'] += 5e-7
    write_verdict(f, block)
    assert rb_v2.compare(fake_result(), rb_v2.read_block(f, 'K2'))['agree']
    for key, value in [('d', .0121), ('interval', [.009, .0151]), ('A_e', {'B1-r20': .0881, 'B1-own': .085}),
                       ('eligibility', dict(eligible_seeds={'K2': [0, 1], 'B1': [0, 1, 2], 'B2': []}))]:
        block = good_block()
        block[key] = value
        write_verdict(f, block)
        out = rb_v2.compare(fake_result(), rb_v2.read_block(f, 'K2'))
        assert not out['agree'] and len(out['mismatches']) == 1, key
    block = good_block()
    del block['A_e']['B1-own']
    write_verdict(f, block)
    out = rb_v2.compare(fake_result(), rb_v2.read_block(f, 'K2'))
    assert not out['agree'] and any('B1-own' in m for m in out['mismatches'])


def test_read_block_needs_the_candidate(tmp_path):
    f = tmp_path / 'v.json'
    write_verdict(f, good_block())
    with pytest.raises(AssertionError):
        rb_v2.read_block(f, 'K3')


# ---- synthetic end to end: files, sidecars, partitions, the written re-derivation -------------------------

N = 60
WIDTH = {'K2': 2048}


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(path, key, arr, side):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: arr})
    side = dict(side, sha256=sha_array(arr))
    path.with_suffix('.json').write_text(json.dumps(side))


class World:
    """Tiny CT-like root, a candidate K2 and its matched arm M2 under an outputs folder."""

    def __init__(self, tmp):
        self.names = sorted(f'p{i:03d}' for i in range(N))
        self.sha = hashlib.sha256('\n'.join(self.names).encode()).hexdigest()
        rng = np.random.default_rng(5)
        self.rng = rng
        emotions = [NINE[i % 9] for i in range(9)]
        self.ann = [dict(painting=n, emotion=emotions[(i * 7) % 9], caption='c') for i, n in enumerate(self.names)
                    for _ in range(3)]
        self.ann += [dict(painting='zz', emotion=e, caption='c') for e in NINE]    # keeps all nine names present
        self.labels = rb_v2.held_emotion_labels(self.names, self.ann)
        self.ct = tmp / 'ct'
        self.out = tmp / 'out'
        (self.ct / 'held').mkdir(parents=True)
        (self.ct / 'held/held_paintings.json').write_text(json.dumps(dict(names=self.names, sha256=self.sha)))
        self.roots = rb_v2.Roots(outputs=self.out, ct={'sample': self.ct, 'full': self.ct}, held_sha=self.sha,
                                 rcol={'sample': self.ct / 'rcol.json', 'full': self.ct / 'rcol.json'},
                                 annotations=self.ann)
        self.build()

    def probs(self, quality):
        base = np.eye(40, dtype=np.float32)[self.labels % 40] * quality
        return (base + self.rng.random((N, 40)).astype(np.float32)).astype(np.float32)

    def ct_prob(self, folder, p, s, quality):
        save(self.ct / f'held/percept/{folder}/{p}__s{s}.npz', 'prob', self.probs(quality),
             dict(held_sha256=self.sha, rows=N))

    def build(self):
        runs = []
        for s in range(5):
            for folder, p, q in [('core__k40', 'P3', 3), ('core__k40', 'P2', 2), ('paper__strict', 'P1', 1),
                                 ('paper__strict', 'P4', 1), ('paper__k40', 'P1', 2), ('paper__k40', 'P4', 2)]:
                self.ct_prob(folder, p, s, q)
            runs += [run(s, 'strict', 1 if s == 2 else 50, .1), run(s, 'k40', 40, .1)]
            x = np.eye(2048, dtype=np.float32)[self.labels * 3]
            x = x + 0.3 * self.rng.normal(size=x.shape).astype(np.float32)
            x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
            side = dict(list_sha256=self.sha, rows=N, width=2048, dtype='float16', scale='sample', arm='K2', seed=s,
                        aspect='emotion')
            save(self.out / f'part2/sample/held/K2/emotion__s{s}.npz', 'img', x, side)
            side = dict(list_sha256=self.sha, rows=N, width=40, dtype='float32', scale='sample', arm='M2', seed=s,
                        aspect='percept')
            save(self.out / f'part2/sample/held/M2/P3__s{s}.npz', 'prob', self.probs(1.5), side)
        for r in runs:
            r['collapsed'] = r['cut'] == 'strict' and r['kept_count'] < 10
        (self.ct / 'rcol.json').write_text(json.dumps(dict(runs=runs, entries={
            'B2': dict(eligible=True, seeds=[0, 1, 3, 4]), 'B3': dict(eligible=True, seeds=[0, 1, 3, 4]),
            'B4': dict(eligible=True, seeds=[0, 1, 2, 3, 4]), 'B5': dict(eligible=True, seeds=[0, 1, 2, 3, 4])})))


def test_run_stage_end_to_end_synthetic(tmp_path):
    w = World(tmp_path)
    res = rb_v2.run_candidate('K2', 3, w.roots, draws=8, workers=2)
    assert res['eligible_seeds']['B2'] == [0, 1, 3, 4] and res['eligible_seeds']['B5'] == [0, 1, 2, 3, 4]
    assert set(res['A_e']) == {f'{b}-{r}' for b in ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'M2') for r in ('r20', 'own')}
    assert res['label_counts'] == {str(k): int(v) for k, v in enumerate(np.bincount(w.labels, minlength=9))}
    assert res['d'] == pytest.approx(res['A_cand'] - max(res['A_e'].values()), abs=1e-12)
    assert len(res['bootstrap_d']) == 8
    # the saved partitions of the new arms, when present, are compared; here none exist yet
    assert res['partitions']['checked'] == 0 and res['partitions']['unchecked'] > 0
    # now the new round's own saved partitions: equal, then one planted difference
    parts = res['_partitions']
    for arm in ('K2', 'M2'):
        for s in range(5):
            name = 'emotion' if arm == 'K2' else 'percept'
            for reading, arr in parts[arm][s].items():
                f = w.out / f'part2/partitions/sample/{arm}/{name}__s{s}__{reading}.npz'
                f.parent.mkdir(parents=True, exist_ok=True)
                np.savez(f, labels=arr)
    res2 = rb_v2.run_candidate('K2', 3, w.roots, draws=8, workers=2)
    assert res2['partitions']['checked'] > 0 and not res2['partitions']['mismatches']
    f = w.out / 'part2/partitions/sample/M2/percept__s1__own.npz'
    bad = np.load(f)['labels'].copy()
    bad[0, 0] = (bad[0, 0] + 1) % 40
    np.savez(f, labels=bad)
    res3 = rb_v2.run_candidate('K2', 3, w.roots, draws=8, workers=2)
    assert len(res3['partitions']['mismatches']) == 1 and 'M2' in res3['partitions']['mismatches'][0]


def test_run_candidate_asserts_a_sidecar_that_disagrees(tmp_path):
    w = World(tmp_path)
    f = w.out / 'part2/sample/held/K2/emotion__s0.json'
    side = json.loads(f.read_text())
    side['seed'] = 3
    f.write_text(json.dumps(side))
    with pytest.raises(AssertionError):
        rb_v2.run_candidate('K2', 3, w.roots, draws=2, workers=1)


def test_run_candidate_asserts_a_non_unit_student_row(tmp_path):
    w = World(tmp_path)
    f = w.out / 'part2/sample/held/K2/emotion__s0.npz'
    x = np.load(f)['img'].astype(np.float32) * 1.01
    save(f, 'img', x.astype(np.float16), json.loads(f.with_suffix('.json').read_text()))
    with pytest.raises(AssertionError):
        rb_v2.run_candidate('K2', 3, w.roots, draws=2, workers=1)


def test_stage_file_compares_with_the_verdict_and_reports_the_planted_difference(tmp_path):
    w = World(tmp_path)
    res = rb_v2.run_candidate('K2', 3, w.roots, draws=8, workers=2)
    verdict = w.out / 'part2/verdict_stage3.json'
    verdict.parent.mkdir(parents=True, exist_ok=True)
    block = dict(d=res['d'], A_K2=res['A_cand'], A_e=res['A_e'], interval=res['interval'],
                 eligibility=dict(eligible_seeds=res['eligible_seeds']), bootstrap_d=res['bootstrap_d'])
    write_verdict(verdict, block)
    summary = rb_v2.run_stage(3, w.roots, candidates=['K2'], draws=8, workers=2)
    saved = json.loads((w.out / 'part2/rederive_v2_stage3.json').read_text())
    assert saved['agree'] and saved['candidates']['K2']['comparison']['agree']
    assert summary['agree']
    block['A_e'] = dict(block['A_e'], **{'B4-own': block['A_e']['B4-own'] + 1e-3})
    write_verdict(verdict, block)
    summary = rb_v2.run_stage(3, w.roots, candidates=['K2'], draws=8, workers=2)
    saved = json.loads((w.out / 'part2/rederive_v2_stage3.json').read_text())
    assert not summary['agree'] and not saved['agree']
    assert any('B4-own' in m for m in saved['candidates']['K2']['comparison']['mismatches'])


def test_smoke_refuses_an_outputs_path_outside_the_scratch_area():
    with pytest.raises(AssertionError):
        rb_v2.check_smoke_outputs('/project/CoSiR/x')
    rb_v2.check_smoke_outputs('/tmp/claude-0/x')


# ---- real data: ArtELingo-28 labels, and the CT's own verdicts ---------------------------------------------

def test_a28_labels_on_the_real_tables():
    names, labels, info = rb_v2.a28_emotion_labels(*rb_v2.load_a28_tables())
    assert len(names) == 746 and names == sorted(names)
    assert info['unique'] == 729 and info['ties'] == 17            # ticket 14's expectation
    assert labels.min() >= 0 and labels.max() <= 8


def ct_check(stage, workers=8):
    t0 = time.time()
    verdict = json.loads((CT_DIR / f'verdict_stage{stage}.json').read_text())
    res = rb_v2.rederive_ct(stage, workers=workers)
    return verdict, res, time.time() - t0


def assert_ct_agrees(verdict, res):
    assert res['d'] == pytest.approx(verdict['d'], abs=1e-6)
    assert res['A_cand'] == pytest.approx(verdict['A_K1'], abs=1e-6)
    assert set(res['A_e']) == set(verdict['A_e'])
    for k, v in verdict['A_e'].items():
        assert res['A_e'][k] == pytest.approx(v, abs=1e-6), k
    assert res['interval'] == pytest.approx(verdict['interval'], abs=1e-6)
    assert res['eligible_seeds'] == {k: v for k, v in verdict['eligibility']['eligible_seeds'].items()}
    assert max(abs(a - b) for a, b in zip(res['bootstrap_d'], verdict['bootstrap_d'])) < 1e-6
    assert res['partitions']['mismatches'] == [] and res['partitions']['checked'] > 0
    assert res['label_counts'] == {k: v for k, v in verdict['label_counts']['emotion']['classes'].items()}
    assert res['eligibility_notes'] == []


def test_reproduces_ct_verdict_stage1():
    """Acceptance: K1 and B1, B4, B5 (B2, B3 not eligible) from the CT's saved sample-scale outputs."""
    verdict, res, seconds = ct_check(1)
    assert_ct_agrees(verdict, res)
    assert seconds < 240, seconds


@slow
def test_reproduces_ct_verdict_stage2():
    verdict, res, _ = ct_check(2)
    assert_ct_agrees(verdict, res)


# ---- stage 5, synthetic ------------------------------------------------------------------------------------

def test_stage5_end_to_end_synthetic(tmp_path):
    n = 30
    names = sorted(f'q{i:02d}' for i in range(n))
    sha = hashlib.sha256('\n'.join(names).encode()).hexdigest()
    rng = np.random.default_rng(2)
    csv = pd.DataFrame({'emotion': [NINE[(i // 3 * 5 + (i % 3 == 2)) % 9] if i % 7 else 'other' for i in range(3 * n)],
                        'painting': [nm for nm in names for _ in range(3)]})
    parquet = pd.DataFrame({'row_id': np.arange(3 * n)[::-1].copy(), 'painting': csv['painting'].to_numpy()[::-1]})
    got_names, labels, info = rb_v2.a28_emotion_labels(parquet, csv)
    assert got_names == names and info['paintings'] == n
    out = tmp_path / 'out'
    roots = rb_v2.Roots(outputs=out, ct={'sample': tmp_path, 'full': tmp_path}, held_sha='x',
                        rcol={}, annotations=[], a28=(parquet, csv), a28_paintings=n)

    def prob(q):
        return (np.eye(40, dtype=np.float32)[labels % 40] * q + rng.random((n, 40)).astype(np.float32)).astype(np.float32)

    for s in range(5):
        x = np.eye(2048, dtype=np.float32)[labels * 3] + 0.3 * rng.normal(size=(n, 2048)).astype(np.float32)
        x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
        base = dict(list_sha256=sha, rows=n, scale='sample', seed=s)
        save(out / f'part2/sample/a28/K2/emotion__s{s}.npz', 'img', x,
             dict(base, width=2048, dtype='float16', arm='K2', aspect='emotion'))
        for arm, p, q in [('B1', 'P3', 2), ('B6', 'P2', 1), ('M2', 'P3', 3)]:
            save(out / f'part2/sample/a28/{arm}/{p}__s{s}.npz', 'prob', prob(q),
                 dict(base, width=40, dtype='float32', arm=arm, aspect='percept'))
    res = rb_v2.run_candidate('K2', 5, roots, draws=6, workers=2, scale='sample')
    assert set(res['A_e']) == {f'{b}-{r}' for b in ('B1', 'B6', 'M2') for r in ('r20', 'own')}
    assert res['n'] == n and res['list_sha256'] == sha and 'B2' not in res['eligible_seeds']
    block = dict(d_A28=res['d'], A_K2=res['A_cand'], A_e=res['A_e'], interval=res['interval'],
                 eligible_seeds=res['eligible_seeds'], bootstrap_d=res['bootstrap_d'])
    v = out / 'part2/verdict_stage5.json'
    v.write_text(json.dumps({'models': {'K2': block}}))
    summary = rb_v2.run_stage(5, roots, candidates=['K2'], draws=6, workers=2, scales={'K2': 'sample'})
    assert summary['agree']
    block['interval'] = [block['interval'][0], block['interval'][1] + 1e-3]
    v.write_text(json.dumps({'models': {'K2': block}}))
    assert not rb_v2.run_stage(5, roots, candidates=['K2'], draws=6, workers=2, scales={'K2': 'sample'})['agree']
    side = out / 'part2/sample/a28/B1/P3__s0.json'
    d = json.loads(side.read_text())
    d['list_sha256'] = 'bad'
    side.write_text(json.dumps(d))
    with pytest.raises(AssertionError):
        rb_v2.run_candidate('K2', 5, roots, draws=2, workers=1, scale='sample')
