"""Ticket 10 tests: stage 3 diagnostics on synthetic score files (no data, CPU)."""
import json
import zlib

import numpy as np
import pytest
from scipy.stats import spearmanr

import tcl_data
import tcl_diag as D
import tcl_extract
import tcl_forward
import tcl_paths as P
import tcl_score
from tcl_metrics import evaluate, fit_means
from test_tcl_score import make_store, unit


# ------------------------------------------------------------------------------------------------ pure parts
def test_four_way_split_and_late_hit_rate_by_hand():
    #            q0     q1     q2     q3     q4     q5
    c = np.array([1, 1, 1, 0, 0, 0], bool)
    p = np.array([1, 1, 0, 1, 0, 0], bool)
    late = np.array([1, 0, 1, 1, 1, 0], bool)
    g = D.four_way(c, p, late)
    assert [g[k]['n'] for k in D.GROUPS4] == [2, 1, 1, 2]
    assert g['both']['share'] == pytest.approx(2 / 6) and g['neither']['share'] == pytest.approx(2 / 6)
    assert g['both']['late_rate'] == 0.5            # q0 hit, q1 miss
    assert g['only_cstar']['late_rate'] == 1.0      # q2
    assert g['only_plain']['late_rate'] == 1.0      # q3
    assert g['neither']['late_rate'] == 0.5         # q4 hit, q5 miss
    assert sum(v['share'] for v in g.values()) == pytest.approx(1)


def test_four_way_empty_group_has_no_rate():
    g = D.four_way([1, 1], [1, 1], [1, 0])
    assert g['only_cstar']['n'] == 0 and np.isnan(g['only_cstar']['late_rate'])
    assert g['both']['late_rate'] == 0.5
    out = D.clean(g)
    assert out['only_cstar']['late_rate'] is None
    json.dumps(out, allow_nan=False)


def test_pair_hit_rule_counts_nan_as_miss_and_rank_10_as_hit():
    assert D.hit(np.array([1, 10, 11, np.nan], np.float32)).tolist() == [True, True, False, False]


def test_above_median_split_ties_and_nan():
    nan = np.nan
    ap = np.array([0.1, 0.5, 0.5, 0.9, nan, 0.7], np.float32)       # scored: 0.1 0.5 0.5 0.9 0.7, median 0.5
    above, scored, med = D.above_median(ap)
    assert med == pytest.approx(0.5)
    assert above.tolist() == [False, False, False, True, False, True]     # 0.5 ties are not above; NaN is not scored
    assert scored.tolist() == [True, True, True, True, False, True]


def test_label_split_uses_each_rows_own_median_and_checks_scored_queries():
    nan = np.nan
    c = np.array([0.9, 0.8, 0.2, 0.1, nan], np.float32)      # median 0.5 -> above: q0 q1
    p = np.array([0.30, 0.20, 0.90, 0.1, nan], np.float32)   # median 0.25 -> above: q0 q2
    late = np.array([0.9, 0.2, 0.8, 0.1, nan], np.float32)   # median 0.5 -> above: q0 q2
    g, med, n = D.label_split(c, p, late)
    assert n == 4 and med == dict(cstar=pytest.approx(0.5), plain=pytest.approx(0.25), late=pytest.approx(0.5))
    assert [g[k]['n'] for k in D.GROUPS4] == [1, 1, 1, 1]
    assert g['both']['late_rate'] == 1 and g['only_cstar']['late_rate'] == 0
    assert g['only_plain']['late_rate'] == 1 and g['neither']['late_rate'] == 0
    with pytest.raises(AssertionError):
        D.label_split(c, np.array([0.3, 0.2, nan, 0.1, nan], np.float32), late)


def test_spread_ratio_and_the_late_cosine_it_assumes():
    assert D.spread_ratio(2.0, 4.0, a=0.7) == pytest.approx(0.7 * 2 / (0.3 * 4))
    assert np.isnan(D.spread_ratio(1.0, 0.0))
    rng = np.random.default_rng(1)
    c1, c2, p1, p2 = (unit(rng.normal(size=(5, 8))) for _ in range(4))
    a = P.A
    late1, late2 = tcl_forward.fuse(c1, p1, a), tcl_forward.fuse(c2, p2, a)
    # tcl_forward's late vector: its cosine is (1 - a) * c* cosine + a * plain cosine, so a acts on the plain spread
    assert np.allclose(late1 @ late2.T, (1 - a) * (c1 @ c2.T) + a * (p1 @ p2.T), atol=1e-6)


def test_spearman_hand_cases_and_ties_agree_with_scipy():
    x = np.array([[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]])
    y = np.array([[4., 3., 2., 1.], [1., 3., 2., 4.], [10., 20., 30., 40.]])
    got = D.spearman_rows(x, y)
    assert got[0] == pytest.approx(-1) and got[2] == pytest.approx(1)
    assert got[1] == pytest.approx(1 - 6 * 2 / (4 * 15))              # 1 - 6 sum d^2 / (n (n^2 - 1)), sum d^2 = 2 -> 0.8
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 6, size=(7, 40)).astype(np.float32), rng.integers(0, 6, size=(7, 40)).astype(np.float32)
    ref = [spearmanr(a[i], b[i])[0] for i in range(7)]
    assert np.allclose(D.spearman_rows(a, b), ref, atol=1e-12)
    assert np.isnan(D.spearman_rows(np.ones((1, 4)), x[:1])[0])        # constant row: no correlation
    mean, q25, q50, q75 = D.quartiles([0.0, 1.0, 2.0, 3.0, np.nan])
    assert (mean, q25, q50, q75) == (1.5, 0.75, 1.5, 2.25)


def test_per_seed_average_runs_over_the_seeds_that_have_a_rate():
    s0 = D.four_way([1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0])
    s1 = D.four_way([1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 1])      # only_plain empty in this seed
    avg = D.average_groups([s0, s1])
    assert avg['both']['share'] == pytest.approx((0.25 + 0.5) / 2)
    assert avg['both']['share_per_seed'] == [0.25, 0.5]
    assert avg['only_plain']['share'] == pytest.approx((0.25 + 0) / 2)
    assert avg['only_plain']['late_rate'] == s0['only_plain']['late_rate'] and avg['only_plain']['late_rate_n_seeds'] == 1
    assert avg['both']['late_rate'] == pytest.approx((1 + 0.5) / 2) and avg['both']['late_rate_n_seeds'] == 2
    assert D.mean_defined([np.nan, np.nan]) [1] == 0


def test_query_sample_is_fixed_sorted_and_capped():
    a, b = D.sample_queries(1000), D.sample_queries(1000)
    assert np.array_equal(a, b) and len(a) == 500 and len(set(a.tolist())) == 500 and (np.diff(a) > 0).all()
    assert np.array_equal(a, np.sort(np.random.default_rng(0).choice(1000, size=500, replace=False)))
    assert np.array_equal(D.sample_queries(7), np.arange(7))


def test_sample_cosines_drop_the_query_itself_and_spread_matches_numpy():
    x = unit(np.random.default_rng(2).normal(size=(9, 6)))
    idx = np.array([1, 4, 7])
    sim = D.sample_cosines(x, x, idx, True, 'cpu').numpy()
    assert sim.shape == (3, 8)
    full = x[idx] @ x.T
    assert np.allclose(sim, np.stack([np.delete(full[k], idx[k]) for k in range(3)]), atol=1e-6)
    std, within = D.spreads(D.sample_cosines(x, x, idx, False, 'cpu'))
    assert std == pytest.approx(np.std(full.astype(np.float64)), abs=1e-6)
    assert within == pytest.approx(np.std(full.astype(np.float64), axis=1).mean(), abs=1e-6)


def test_clean_turns_nan_into_null_and_json_refuses_nan():
    out = D.clean(dict(a=np.float32('nan'), b=[np.float64(1.5), np.inf], c=np.int64(3), d=np.bool_(True)))
    assert out == dict(a=None, b=[1.5, None], c=3, d=True)
    with pytest.raises(ValueError):
        json.dumps(dict(a=float('nan')), allow_nan=False)


# ------------------------------------------------------------------------------------------------ synthetic tree
KIND = dict(art_val='caption', mm_val='narrative')


def fake_frame(set_name, kind):
    fr, _, _, _ = make_store(seed=0)
    if P.GROUP[set_name] == 'art':
        return fr
    n = fr['n_i']
    labels = {a: dict(t=v['i'].copy(), i=v['i'].copy()) for a, v in fr['labels'].items()}
    return dict(n_i=n, n_t=n, tp=np.arange(n, dtype=np.int64), ip=np.arange(n, dtype=np.int64), labels=labels)


def fake_rows(set_name, kind, row, head, seed, device):
    """Deterministic synthetic rows of a row, head and seed in the frame's order (c* is plain plus row-specific noise)."""
    fr = fake_frame(set_name, kind)
    n_t, n_i, d = fr['n_t'], fr['n_i'], 16
    key = lambda *k: zlib.crc32(repr(k).encode())
    base = np.random.default_rng(key(set_name, 'base')).normal(size=(n_i, d))
    gap = 1.5
    noise = lambda tag, n: np.random.default_rng(key(set_name, tag)).normal(size=(n, d))
    t0, i0 = unit(base[fr['tp']] + 0.8 * noise('pt', n_t) - gap * np.eye(d)[1]), unit(base + 0.5 * noise('pi', n_i) + gap * np.eye(d)[0])
    if row == 'plain':
        return t0, i0
    assert row == 'cstar'
    tag = (head, seed)
    return (unit(t0 + 0.7 * np.random.default_rng(key(set_name, 'ct', tag)).normal(size=(n_t, d))),
            unit(i0 + 0.7 * np.random.default_rng(key(set_name, 'ci', tag)).normal(size=(n_i, d))))


RECS = None


def make_recs():
    ro = P.READOUTS
    late = {a: {d: ro[(i + j) % 3] for j, d in enumerate(P.DIRECTIONS)} for i, a in enumerate(P.ASPECTS)}
    plain = {a: {d: ro[(i + j + 1) % 3] for j, d in enumerate(P.DIRECTIONS)} for i, a in enumerate(P.ASPECTS)}
    pair = lambda kind: {kind: {'t2i': dict(late=dict(head='genre', readout='r0'), plain=dict(readout='center')),
                                'i2t': dict(late=dict(head='style', readout='center_k10'), plain=dict(readout='r0'))}}
    rec = lambda group, kind: dict(group=group, picks={kind: dict(late=late, plain=plain, cstar=late)},
                                   pair_picks=pair(kind), values={}, ranges={})
    return {'art': rec('art', 'caption'), 'mm': rec('mm', 'narrative')}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(P, 'SMOKE', 6)
    monkeypatch.setattr(tcl_data, 'frame', fake_frame)
    monkeypatch.setattr(tcl_forward, 'rows', fake_rows)
    monkeypatch.setattr(tcl_score, 'held_guard', lambda name: None, raising=False)
    return tmp_path


def write_late_files(out, set_name, seeds=P.SEEDS[1:]):
    kind, fr = KIND[set_name], fake_frame(set_name, KIND[set_name])
    flags = {a: np.random.default_rng(3).random(fr['n_t']) < 0.3 for a in P.ASPECTS}
    text_p, image_p = fake_rows(set_name, kind, 'plain', 'all', 'x', 'cpu')
    for head in P.ASPECTS:
        for seed in seeds:
            c_t, c_i = fake_rows(set_name, kind, 'cstar', head, seed, 'cpu')
            tcl_score.score_unit(set_name, kind, 'late', head, seed, tcl_forward.fuse(c_t, text_p), tcl_forward.fuse(c_i, image_p),
                                 fr, flags, 'cpu', out=out)


def test_curve_equals_stored_late_files_at_a_07_and_plain_at_a_1(world):
    for s in KIND:
        write_late_files(world, s)
    ctx = D.Context(make_recs(), 'cpu')
    res = D.part2(ctx)
    chk = res['self_check']
    assert chk['n_checked'] == 2 * 4 * (9 + 2) and chk['max_abs_diff'] <= 1e-6      # 9 label + 2 pair values, 4 seeds, 2 sets
    assert res['missing'] == [] and res['rankings_run'] > 0
    assert D.A_GRID[7] == P.A and len(D.A_GRID) == 11
    lab = [r for r in res['sections']['label'] if r['set'] == 'art_val' and r['a'] == 1.0 and r['head'] == 'style'
           and r['direction'] == 'i2t'][0]
    # a = 1: the late vector is (0, plain), so the curve is plain Qwen with means fitted on it
    fr = fake_frame('art_val', 'caption')
    t, i = fake_rows('art_val', 'caption', 'plain', 'all', 'x', 'cpu')
    mt, mi = fit_means(t, i)
    want = evaluate(i, t, fr['ip'], fr['tp'], {'style': dict(q=fr['labels']['style']['i'], g=fr['labels']['style']['t'])},
                    False, lab['readout'], 'cpu', means=tcl_score.query_gallery_means('i2t', mt, mi))
    assert lab['map'] == pytest.approx(D.mean_ap(want['ap']['style']), abs=1e-7)
    assert lab['seeds'] == [1, 2, 3, 4] and len(lab['map_per_seed']) == 4 and lab['n_query'] == D.n_scored(want['ap']['style'])
    pair = [r for r in res['sections']['pair'] if r['set'] == 'art_val' and r['direction'] == 't2i']
    assert {r['head'] for r in pair} == {'genre'} and {r['readout'] for r in pair} == {'r0'} and len(pair) == 11


def test_curve_self_check_catches_a_changed_stored_file(world):
    for s in KIND:
        write_late_files(world, s)
    path = P.score_path('art_val', 'caption', 'late', 'emotion', 2, 't2t', make_recs()['art']['picks']['caption']['late']['emotion']['t2t'])
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    ap = arrays['ap__emotion']
    ap[np.isfinite(ap)] *= 0.9
    np.savez(path, **arrays)
    with pytest.raises(AssertionError, match='differs from the stored late file'):
        D.part2(D.Context(make_recs(), 'cpu'))


def test_missing_file_is_listed_under_smoke_and_crashes_outside(world, monkeypatch):
    R = D.Reader()
    assert R.score('art_val', 'caption', 'plain', 'all', 'x', 't2i', 'r0') is None
    assert R.missing == ['scores/art_val/caption/plain/all__sx__t2i__r0.npz']
    monkeypatch.setattr(P, 'SMOKE', 0)
    with pytest.raises(FileNotFoundError):
        D.Reader().score('art_val', 'caption', 'plain', 'all', 'x', 't2i', 'r0')


def test_read_checks_reject_misaligned_query_ids(world):
    write_late_files(world, 'mm_val', seeds=(1,))
    path = P.score_path('mm_val', 'narrative', 'late', 'genre', 1, 't2i', 'r0')
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays['q_pid'] = arrays['q_pid'][::-1].copy()
    np.savez(path, **arrays)
    with pytest.raises(AssertionError, match='painting order'):
        D.Reader().score('mm_val', 'narrative', 'late', 'genre', 1, 't2i', 'r0')


def snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob('*')) if p.is_file()}


def test_main_writes_only_stage3_and_leaves_records_alone(world, monkeypatch, capsys):
    for s in KIND:
        write_late_files(world, s, seeds=P.SEEDS)
    recs = make_recs()
    (world / 'records').mkdir()
    for g, r in recs.items():
        (world / 'records' / f'stage1_{g}.json').write_text(json.dumps(r))
    done = world / 'store' / 'done.json'
    done.parent.mkdir()
    done.write_text(json.dumps(dict(count=10, n_truncated=1, longest_tokens=600, token_limit=512, smoke=True)))
    monkeypatch.setattr(tcl_extract, 'stores_to_check',
                        lambda: [('art_held/caption/emotion_focus/teacher', 'art_held', 'caption', 'emotion_focus', 'teacher',
                                  done.parent, 'ids')])
    before = snapshot(world)
    assert D.main(['--device', 'cpu', '--parts', '3', '8']) == 0
    after = snapshot(world)
    new = sorted(set(after) - set(before))
    assert all(k.startswith('stage3/') for k in new), new
    assert {k for k in before if k.startswith('records/')} == {k for k in after if k.startswith('records/')}
    assert all(after[k] == before[k] for k in before), 'an existing file changed'
    assert not any(k.startswith('stage2/') for k in after)
    assert {'stage3/part3_pair_all_rows.json', 'stage3/part8_truncation.json', 'stage3/summary.md'} <= set(new)
    part3 = json.loads((world / 'stage3/part3_pair_all_rows.json').read_text())
    rec = [r for r in part3['sections']['pair'] if (r['set'], r['row'], r['head'], r['direction'], r['readout']) ==
           ('art_val', 'late', 'emotion', 't2i', 'r0')][0]
    # the stored first ranks, read straight from the files
    ranks = [np.load(P.score_path('art_val', 'caption', 'late', 'emotion', s, 't2i', 'r0'))['first_rank'] for s in P.SEEDS]
    assert rec['seeds'] == [0, 1, 2, 3, 4] and rec['r10_per_seed'] == [float(np.mean(r <= 10)) for r in ranks]
    assert rec['n_query'] == 12 and rec['r10'] == pytest.approx(np.mean(rec['r10_per_seed']))
    assert part3['missing'] and part3['smoke'] is True
    p8 = json.loads((world / 'stage3/part8_truncation.json').read_text())
    assert p8['sections']['stores'][0]['n_truncated'] == 1 and p8['sections']['stores'][0]['share_truncated'] == 0.1


def test_main_refuses_without_verdicts_outside_smoke(world, monkeypatch, capsys):
    monkeypatch.setattr(P, 'SMOKE', 0)
    monkeypatch.setattr(P, 'RECORDS', world / 'committed_records')
    called = []
    monkeypatch.setattr(tcl_score, 'held_guard', lambda name: called.append(name), raising=False)
    assert D.main(['--device', 'cpu', '--parts', '8']) == 2
    assert called == ['art_held', 'mm_test']                           # the guard runs first, then the refusal
    assert not (world / 'stage3').exists()
    assert not (world / 'committed_records').exists()


@pytest.mark.parametrize('missing', ['art', 'mm'])
def test_main_skips_only_the_group_without_a_record(world, monkeypatch, capsys, missing):
    kept = 'mm' if missing == 'art' else 'art'
    records = world / 'records'
    records.mkdir()
    (records / f'stage1_{kept}.json').write_text(json.dumps(make_recs()[kept]))
    guarded, ran = [], []
    monkeypatch.setattr(tcl_score, 'held_guard', lambda name: guarded.append(name))

    def group_part(n, ctx):
        assert set(ctx.recs) == {kept}
        ran.append(n)
        return D.result(n, D.NAMES[n], D.Reader(), {})

    for n in (5, 6):
        monkeypatch.setitem(D.PARTS, n, lambda ctx, n=n: group_part(n, ctx))
    assert D.main(['--device', 'cpu', '--parts', '3', '5', '6']) == 0
    assert guarded == [dict(art='art_held', mm='mm_test')[kept]]
    assert ran == [dict(art=5, mm=6)[kept]]
    assert f'part {dict(art=5, mm=6)[missing]}: skipped' in capsys.readouterr().out
    part3 = json.loads((world / 'stage3/part3_pair_all_rows.json').read_text())
    assert part3['missing']
    assert {path.split('/')[1] for path in part3['missing']} == {s for s in P.SETS if P.GROUP[s] == kept}

    monkeypatch.setattr(P, 'SMOKE', 0)
    monkeypatch.setattr(P, 'RECORDS', records)
    assert D.main(['--device', 'cpu', '--parts', '3', '5', '6']) == 2
    assert guarded == [dict(art='art_held', mm='mm_test')[kept]] * 2
    assert ran == [dict(art=5, mm=6)[kept]]


def test_main_keeps_guard_when_neither_record_exists(world, monkeypatch):
    def guard(name):
        assert name == 'art_held'
        raise FileNotFoundError('stage1_art.json')

    monkeypatch.setattr(tcl_score, 'held_guard', guard)
    with pytest.raises(FileNotFoundError, match='stage1_art.json'):
        D.main(['--device', 'cpu', '--parts', '8'])


@pytest.mark.parametrize('smoke', [0, 6])
@pytest.mark.parametrize('voided', [[], [dict(what='diagnostic', key='mm_test/formal', why='store failed'),
                                      dict(what='cell_claims', key='art_held', why='gate iv failed')]])
def test_voided_reaches_part_and_summary_from_outcomes(world, monkeypatch, smoke, voided):
    records = world / 'records'
    records.mkdir()
    for group, rec in make_recs().items():
        (records / f'stage1_{group}.json').write_text(json.dumps(rec))
    (records / 'verdicts.json').write_text('{}')
    monkeypatch.setattr(P, 'SMOKE', smoke)
    monkeypatch.setattr(P, 'RECORDS', records)
    monkeypatch.setattr(P, 'read_committed_record', lambda name: json.loads((records / name).read_text()))
    monkeypatch.setitem(D.PARTS, 8, lambda ctx: D.result(8, D.NAMES[8], D.Reader(), dict(stores=[])))
    args = ['--device', 'cpu', '--parts', '8']
    part = world / 'stage3/part8_truncation.json'
    summary = world / 'stage3/summary.md'
    if smoke:
        assert D.main(args) == 0
        assert json.loads(part.read_text())['voided'] == []
        assert '## Voided (§6)\n\nnone\n' in summary.read_text()
    else:
        with pytest.raises(AssertionError, match='outcomes.json'):
            D.main(args)
    outcomes = world / 'stage2/outcomes.json'
    outcomes.parent.mkdir()
    outcomes.write_text(json.dumps(dict(voided=voided)))
    assert D.main(args) == 0
    assert json.loads(part.read_text())['voided'] == voided
    outcomes.unlink()
    D.write_summary()                                             # a standalone part carries the void list
    text = summary.read_text()
    assert text.index('## Voided (§6)') < text.index('| part |')
    for entry in voided:
        assert f'- {entry["what"]}: {entry["key"]}: {entry["why"]}' in text
    if not voided:
        assert '## Voided (§6)\n\nnone\n' in text
