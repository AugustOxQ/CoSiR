"""Stage ii scorer (ticket 08, rule §3.2, §4.2): lifts, intervals, outcomes, VL1 (a); small synthetic inputs."""
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score as ami

import eg_paths  # noqa: F401  (sys.path)
import eg_lift
import rb_ami


def test_outcome_boundaries():
    assert eg_lift.outcome(0.01, 0.001) == 'L-HOLDS'
    assert eg_lift.outcome(0.01, 0.0) == 'L-QUAL'          # lower bound exactly 0 is not > 0
    assert eg_lift.outcome(0.01, -0.02) == 'L-QUAL'
    assert eg_lift.outcome(0.0, -0.02) == 'L-FAILS'        # lift exactly 0 is not > 0
    assert eg_lift.outcome(-0.01, 0.5) == 'L-FAILS'        # a negative lift fails whatever the bound
    with pytest.raises(AssertionError):
        eg_lift.outcome(float('nan'), 0.0)


# ------------------------------------------------------------------ draws and the per-draw maximum
def test_draw_plan_order_is_painting_draw_then_student_seeds_then_own_seeds():
    rng = np.random.default_rng(0)
    plan = eg_lift.draw_plan(7, 4)
    for b in range(4):
        pos = rng.integers(0, 7, size=7)
        s_k = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        s_o = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        np.testing.assert_array_equal(plan['pos'][b], pos)
        np.testing.assert_array_equal(plan['seeds_student'][b], s_k)
        np.testing.assert_array_equal(plan['seeds_own'][b], s_o)
    assert plan['pos'].shape == (4, 7) and plan['seeds_own'].shape == (4, 5)


def perfect_or_null(labels, perfect_seeds, m=3):
    """Five seeds of [m, n] partitions: the labels themselves (AMI 1) for the listed seeds, one cluster (AMI 0) else."""
    return [np.tile(labels if s in perfect_seeds else np.zeros_like(labels), (m, 1)) for s in range(5)]


def test_per_draw_maximum_over_the_two_own_readings():
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 3, 40)
    student = perfect_or_null(labels, {0, 1, 2, 3, 4})            # A = 1 in every draw
    own_r20 = perfect_or_null(labels, {0, 1})                      # A = share of seeds 0, 1 in S_b
    own_own = perfect_or_null(labels, {2, 3, 4}, m=1)              # A = share of seeds 2 to 4 in S_b
    out = eg_lift.bootstrap(labels, student, own_r20, own_own, n_draws=50, workers=1)
    plan = eg_lift.draw_plan(40, 50)
    share01 = np.isin(plan['seeds_own'], [0, 1]).mean(axis=1)
    np.testing.assert_allclose(out['own_r20'], share01)
    np.testing.assert_allclose(out['own_own'], 1 - share01)
    np.testing.assert_allclose(out['student'], 1.0)
    np.testing.assert_allclose(out['lift'], 1 - np.maximum(share01, 1 - share01))
    # the max of the two means over draws is not the mean of the per-draw maxima
    assert out['lift'].mean() < 1 - max(out['own_r20'].mean(), out['own_own'].mean()) - 1e-9


def test_score_aspect_point_estimates_interval_and_masked_labels():
    rng = np.random.default_rng(11)
    n = 90
    labels = rng.integers(-1, 3, n)                                  # genre-like: some rows masked
    labelled = labels >= 0
    student = perfect_or_null(labels, {0, 1, 2, 3, 4})
    own_r20 = perfect_or_null(labels, {0, 1})                        # A = 2/5 on the point (seeds 0 to 4 once each)
    own_own = perfect_or_null(labels, {0}, m=1)                      # A = 1/5
    frozen = np.tile(rng.integers(0, 5, n), (3, 1))
    r = eg_lift.score_aspect(labels, student, own_r20, own_own, frozen, n_draws=40, workers=1)
    assert r['n_labelled'] == int(labelled.sum()) and r['n_draws'] == 40
    assert r['A'] == pytest.approx(dict(student=1.0, own_r20=0.4, own_own=0.2,
                                        frozen=np.mean([ami(labels[labelled], frozen[0][labelled])] * 3)))
    assert r['own_best'] == 'own_r20' and r['lift'] == pytest.approx(0.6)
    assert r['student_minus_frozen'] == pytest.approx(1.0 - r['A']['frozen'])
    boot = eg_lift.bootstrap(labels, student, own_r20, own_own, 40, 1)
    assert r['interval'] == pytest.approx(list(np.percentile(boot['lift'], [2.5, 97.5])))
    assert 'outcome' not in r                                        # emotion decides, at the encoder level


def test_score_aspect_takes_the_point_maximum_over_the_two_readings_and_fails_on_a_loss():
    rng = np.random.default_rng(12)
    labels = rng.integers(0, 3, 60)
    student = perfect_or_null(labels, {0})                           # A = 1/5
    own_r20 = perfect_or_null(labels, {0, 1, 2, 3, 4})               # A = 1
    own_own = perfect_or_null(labels, {0, 1, 2}, m=1)                # A = 3/5
    r = eg_lift.score_aspect(labels, student, own_r20, own_own, np.tile(labels, (3, 1)), n_draws=20, workers=1)
    assert r['own_best'] == 'own_r20' and r['lift'] == pytest.approx(0.2 - 1.0)
    assert eg_lift.outcome(r['lift'], r['interval'][0]) == 'L-FAILS'


def test_score_aspect_takes_the_arg_max_reading_when_it_is_the_better_one():
    rng = np.random.default_rng(13)
    labels = rng.integers(0, 3, 60)
    student = perfect_or_null(labels, {0, 1})                        # A = 2/5
    own_r20 = perfect_or_null(labels, {0})                           # A = 1/5
    own_own = perfect_or_null(labels, {0, 1, 2, 3}, m=1)             # A = 4/5
    r = eg_lift.score_aspect(labels, student, own_r20, own_own, np.tile(labels, (3, 1)), n_draws=20, workers=1)
    assert r['A']['own_own'] > r['A']['own_r20']
    assert r['own_best'] == 'own_own' and r['lift'] == pytest.approx(0.4 - 0.8)      # not student - own_r20 = +0.2


def test_bootstrap_matches_a_plain_loop_on_random_partitions_and_ignores_workers():
    rng = np.random.default_rng(3)
    n = 80
    labels = rng.integers(-1, 4, n)                                 # about a fifth masked
    mk = lambda m: [rng.integers(0, 6, (m, n)) for _ in range(5)]
    student, own_r20, own_own = mk(3), mk(3), mk(1)
    out1 = eg_lift.bootstrap(labels, student, own_r20, own_own, n_draws=12, workers=1)
    out2 = eg_lift.bootstrap(labels, student, own_r20, own_own, n_draws=12, workers=2)
    np.testing.assert_array_equal(out1['lift'], out2['lift'])
    labelled = np.flatnonzero(labels >= 0)
    plan = eg_lift.draw_plan(len(labelled), 12)
    seed_a = lambda parts, idx, s: np.mean([ami(labels[idx], p[idx]) for p in parts[s]])
    for b in range(12):
        idx = labelled[plan['pos'][b]]
        a_k = np.mean([seed_a(student, idx, s) for s in plan['seeds_student'][b]])
        a_r = np.mean([seed_a(own_r20, idx, s) for s in plan['seeds_own'][b]])
        a_o = np.mean([seed_a(own_own, idx, s) for s in plan['seeds_own'][b]])
        assert out1['lift'][b] == pytest.approx(a_k - max(a_r, a_o), abs=1e-12)


# ------------------------------------------------------------------ the run on synthetic stage ii exports
import eg_encoders  # noqa: E402

N, W = 90, 8
P_SHA = 'p' * 64
TOY = (eg_encoders.ENCODERS[0], eg_encoders.ENCODERS[1], eg_encoders.ENCODERS[3])   # K4/B1, K2/M2, K6/M6 (new); toy widths


def toy_labels():
    emotion = np.repeat(np.arange(3), N // 3)
    genre = emotion.copy()
    genre[::7] = -1                               # masked
    return dict(emotion=emotion, style=(emotion + 1) % 3, genre=genre)


def blobs_of(codes, seed, noise=0.05):
    """Unit-norm float32 rows: one centre per code plus noise."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((codes.max() + 1, W))
    x = centres[codes] + noise * rng.standard_normal((len(codes), W))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def write_student(outputs, arm, aspect, seed, x, p_sha=P_SHA, changes=None):
    path = eg_paths.p_student(outputs, arm, aspect, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, img=x.astype(np.float16))
    side = dict(p_sha256=p_sha, arm=arm, seed=seed, aspect=aspect, rows=x.shape[0], width=x.shape[1])
    path.with_suffix('.json').write_text(json.dumps({**side, **(changes or {})}))


def write_own(outputs, arm, seed, prob, p_sha=P_SHA, changes=None):
    path = eg_paths.p_own(outputs, arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, prob=prob.astype(np.float32))
    side = dict(p_sha256=p_sha, arm=arm, seed=seed, rows=prob.shape[0], width=prob.shape[1])
    path.with_suffix('.json').write_text(json.dumps({**side, **(changes or {})}))


def aligned_prob(codes, seed):
    """Sigmoid-like [N, 40] vectors whose arg-max topic is the code (so the own partition equals the labels)."""
    rng = np.random.default_rng(seed)
    prob = 0.05 * rng.random((len(codes), 40))
    prob[np.arange(len(codes)), codes] = 0.9
    return prob


def random_prob(seed):
    return np.random.default_rng(seed).random((N, 40))


def vl1_scorer_record(**changes):
    """A passing `vl1_scorer.json` as `check_vl1a` writes it, for this code and the toy P."""
    return {**{'pass': True}, **dict(p_sha256=P_SHA, rb_ami_sha256=eg_lift.rb_ami_sha256(), versions=eg_lift.versions()),
            **changes}


def plant(tmp_path, labels, *, student_aligned, own_aligned, missing=None, vl1=True, vl1_scorer=True):
    """Write exports of K4/B1 (student aligned with emotion, own random), K2/M2 (student random, own aligned), K6/M6 missing."""
    out = tmp_path / 'outputs'
    (out / 'stage_ii/p').mkdir(parents=True)
    (out / 'stage_ii/p/index.json').write_text(json.dumps(dict(missing=missing or {})))
    (out / 'stage_ii/vl1.json').write_text(json.dumps(dict(**{'pass': vl1})))
    (out / 'stage_ii/vl1_scorer.json').write_text(json.dumps(vl1_scorer_record(**{'pass': vl1_scorer})))
    for enc in TOY:
        if enc.student in (missing or {}):
            continue
        for seed in range(5):
            aligned = enc.student in student_aligned
            for aspect in eg_paths.ASPECTS:
                x = blobs_of(labels[aspect] if aligned else np.random.default_rng(seed).integers(0, 3, N), seed + 10 * len(aspect))
                write_student(out, enc.student, aspect, seed, x)
            write_own(out, enc.own, seed, aligned_prob(labels['emotion'], seed) if enc.own in own_aligned else random_prob(seed))
    return out


def frozen_of(store):
    return blobs_of(np.random.default_rng(7).integers(0, 3, N), 7)


def run_toy(out, labels, **kw):
    return eg_lift.run(out, labels, frozen_images=frozen_of, n_rows=N, width=W, p_sha=P_SHA, n_draws=20, workers=1,
                       encoders=TOY, **kw)


def test_run_scores_present_encoders_and_reports_a_missing_one_without_outcome(tmp_path):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'no stage 3 weights'})
    result = run_toy(out, labels)
    assert json.loads((out / 'stage_ii/lift.json').read_text()) == json.loads(json.dumps(result))
    k4, k2, k6 = (result['encoders'][s] for s in ('K4', 'K2', 'K6'))
    assert k6['status'] == 'missing' and k6['reason'] == 'no stage 3 weights' and k6['outcome'] is None
    assert k4['status'] == 'scored' and k2['status'] == 'scored'
    assert k4['outcome'] in ('L-HOLDS', 'L-QUAL') and k4['aspects']['emotion']['lift'] > 0     # aligned student, random own
    assert k2['outcome'] == 'L-FAILS' and k2['aspects']['emotion']['lift'] < 0                  # random student, aligned own
    assert k4['outcome'] == eg_lift.outcome(k4['aspects']['emotion']['lift'], k4['aspects']['emotion']['interval'][0])
    assert set(k4['aspects']) == set(eg_paths.ASPECTS) and k4['aspects']['genre']['n_labelled'] == int((labels['genre'] >= 0).sum())
    assert k4['aspects']['style']['n_draws'] == 20 and set(k4['aspects']['emotion']['A']) == {'student', 'own_r20', 'own_own', 'frozen'}
    assert result['p_sha256'] == P_SHA and result['n_draws'] == 20 and 'sklearn' in result['versions'] and 'code_commit' in result


def test_partitions_are_cached_under_stage_ii_and_reused_and_other_inputs_are_refused(tmp_path, monkeypatch):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    first = run_toy(out, labels)
    cache = out / 'stage_ii/partitions'
    assert (cache / 'K4/emotion__s0.npz').is_file() and (cache / 'B1/P3__s0__r20.npz').is_file()
    assert (cache / 'B1/P3__s0__own.npz').is_file() and (cache / 'F4/default.npz').is_file()
    with np.load(cache / 'B1/P3__s0__own.npz') as z:
        assert z['labels'].shape == (1, N) and z['labels'].dtype == np.int64
    (out / 'stage_ii/lift.json').unlink()
    monkeypatch.setattr(rb_ami, 'partitions', lambda x: pytest.fail('partitions recomputed'))
    again = run_toy(out, labels)
    monkeypatch.undo()
    assert again == first
    write_student(out, 'K4', 'emotion', 0, blobs_of(np.random.default_rng(99).integers(0, 3, N), 99))   # a changed export
    (out / 'stage_ii/lift.json').unlink()
    with pytest.raises(AssertionError, match='other inputs'):
        run_toy(out, labels)


def test_lift_json_is_not_overwritten_by_a_run_with_other_results(tmp_path):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    run_toy(out, labels)
    run_toy(out, labels)                                               # same results: fine
    with pytest.raises(AssertionError, match='other results'):
        eg_lift.run(out, labels, frozen_images=frozen_of, n_rows=N, width=W, p_sha=P_SHA, n_draws=21, workers=1,
                    encoders=TOY)


@pytest.mark.parametrize('vl1,vl1_scorer', [(False, True), (True, False)])
def test_refuses_unless_vl1_passes(tmp_path, vl1, vl1_scorer):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'}, vl1=vl1, vl1_scorer=vl1_scorer)
    with pytest.raises(AssertionError, match='VL1'):
        run_toy(out, labels)
    assert not (out / 'stage_ii/lift.json').exists()


def test_refuses_without_vl1_json(tmp_path):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    (out / 'stage_ii/vl1.json').unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        run_toy(out, labels)


@pytest.mark.parametrize('change', [dict(p_sha256='q' * 64), dict(rb_ami_sha256='0' * 64), dict(versions=dict(sklearn='0.0', numpy='0.0'))])
def test_refuses_a_vl1_a_record_made_for_other_code_or_data(tmp_path, change):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    (out / 'stage_ii/vl1_scorer.json').write_text(json.dumps(vl1_scorer_record(**change)))
    with pytest.raises(AssertionError, match='VL1.*differs'):
        run_toy(out, labels)
    assert not (out / 'stage_ii/lift.json').exists()


@pytest.mark.parametrize('key', ['p_sha256', 'rb_ami_sha256', 'versions'])
def test_refuses_a_vl1_a_record_without_a_key(tmp_path, key):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    record = vl1_scorer_record()
    del record[key]
    (out / 'stage_ii/vl1_scorer.json').write_text(json.dumps(record))
    with pytest.raises(AssertionError, match='VL1'):
        run_toy(out, labels)


@pytest.mark.parametrize('missing', [{'K3': 'x'}, {'F3': 'x'}, {'M2': 'x'}, {'qwen3vl_emb_2b': 'x'}, ['K4'], {'K6': 'x', 'B1': 'y'}])
def test_only_new_encoders_can_be_listed_missing(tmp_path, missing):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    (out / 'stage_ii/p/index.json').write_text(json.dumps(dict(missing=missing)))
    with pytest.raises(AssertionError, match='new encoder'):
        run_toy(out, labels)
    assert not (out / 'stage_ii/lift.json').exists()


def test_new_encoders_can_be_listed_missing_by_any_of_their_names():
    for enc in eg_encoders.ENCODERS:
        if enc.new:
            for name in (enc.student, enc.own, enc.frozen, enc.store):
                assert eg_lift.check_missing({name: 'x'}) == {name: 'x'}


@pytest.mark.parametrize('arm,change', [('K2', dict(arm='K4')), ('K2', dict(seed=2)), ('K2', dict(aspect='emotion')),
                                         ('K2', dict(rows=N + 1)), ('K2', dict(width=W + 1)),
                                         ('M2', dict(arm='B1')), ('M2', dict(seed=2)), ('M2', dict(rows=N + 1)),
                                         ('M2', dict(width=41))])
def test_a_sidecar_that_differs_from_the_arm_seed_aspect_or_array_shape_stops_the_run(tmp_path, arm, change):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    if arm == 'K2':
        write_student(out, 'K2', 'style', 3, blobs_of(np.zeros(N, dtype=int), 1), changes=change)
    else:
        write_own(out, 'M2', 3, random_prob(3), changes=change)
    with pytest.raises(AssertionError, match='sidecar'):
        run_toy(out, labels)


@pytest.mark.parametrize('arm,key', [('K2', 'arm'), ('K2', 'seed'), ('K2', 'aspect'), ('K2', 'rows'), ('K2', 'width'),
                                     ('M2', 'arm'), ('M2', 'seed'), ('M2', 'rows'), ('M2', 'width')])
def test_a_sidecar_without_a_key_stops_the_run(tmp_path, arm, key):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    path = eg_paths.p_student(out, 'K2', 'style', 3) if arm == 'K2' else eg_paths.p_own(out, 'M2', 3)
    side = json.loads(path.with_suffix('.json').read_text())
    del side[key]
    path.with_suffix('.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='sidecar'):
        run_toy(out, labels)


def test_an_export_of_another_p_or_a_not_unit_student_or_a_missing_file_stops_the_run(tmp_path):
    labels = toy_labels()
    out = plant(tmp_path, labels, student_aligned={'K4'}, own_aligned={'M2'}, missing={'K6': 'x'})
    write_student(out, 'K2', 'style', 3, blobs_of(np.zeros(N, dtype=int), 1), p_sha='q' * 64)
    with pytest.raises(AssertionError, match='P sha256'):
        run_toy(out, labels)
    write_student(out, 'K2', 'style', 3, 2 * blobs_of(np.zeros(N, dtype=int), 1))
    with pytest.raises(AssertionError, match='unit norm'):
        run_toy(out, labels)
    write_student(out, 'K2', 'style', 3, blobs_of(np.zeros(N, dtype=int), 1))
    eg_paths.p_own(out, 'M2', 4).unlink()
    with pytest.raises(FileNotFoundError):
        run_toy(out, labels)


# ------------------------------------------------------------------ VL1 (a)
from sklearn.cluster import MiniBatchKMeans  # noqa: E402


def direct_a(x, y):
    """A for vectors x against labels y by plain sklearn calls: the mean over k-means states 0 to 2 of the AMI."""
    x32 = np.asarray(x).astype(np.float32)
    parts = [MiniBatchKMeans(n_clusters=20, random_state=s, n_init=3, batch_size=4096).fit_predict(x32) for s in (0, 1, 2)]
    m = y >= 0
    return float(np.mean([ami(y[m], p[m]) for p in parts]))


def vl1a_fixture(tmp_path):
    labels = toy_labels()
    exports = {(r, a): blobs_of(labels[a] if r == 'R6-5' else np.random.default_rng(len(a)).integers(0, 3, N), 5 + len(a)).astype(np.float16)
               for r in ('R0-5', 'R6-5') for a in eg_paths.ASPECTS}
    references = {k: (lambda k=k: blobs_of(labels['emotion'] if k == 'default' else np.arange(N) % 4, len(k))) for k in
                  ('default', 'emotion_focus', 'style_word', 'genre_word')}
    pick = dict(recipes={r: {a: dict(A=direct_a(exports[r, a], labels[a])) for a in eg_paths.ASPECTS} for r in ('R0-5', 'R6-5')})
    reported = dict(references_p={k: {a: direct_a(f(), labels[a]) for a in eg_paths.ASPECTS} for k, f in references.items()})
    return labels, exports, references, pick, reported


def test_vl1a_passes_when_the_scorer_reproduces_the_saved_values(tmp_path):
    labels, exports, references, pick, reported = vl1a_fixture(tmp_path)
    out = tmp_path / 'outputs'
    rec = eg_lift.check_vl1a(out, labels, load_export=lambda r, a: exports[r, a], references=references,
                             pick=pick, reported=reported, p_sha=P_SHA)
    assert rec['pass'] is True and rec['max_abs_diff'] < 1e-9 and rec['tol'] == 1e-6
    assert len(rec['items']) == 2 * 3 + 4 * 3 and all(i['ok'] for i in rec['items'])
    assert json.loads((out / 'stage_ii/vl1_scorer.json').read_text()) == json.loads(json.dumps(rec))
    assert (out / 'stage_ii/partitions/vl1a').is_dir()                        # caches under --outputs


def test_vl1a_fails_beyond_1e6_and_still_writes_the_record(tmp_path):
    labels, exports, references, pick, reported = vl1a_fixture(tmp_path)
    reported['references_p']['style_word']['genre'] += 2e-6
    pick['recipes']['R0-5']['emotion']['A'] += 5e-7                            # inside the tolerance
    rec = eg_lift.check_vl1a(tmp_path / 'outputs', labels, load_export=lambda r, a: exports[r, a], references=references,
                             pick=pick, reported=reported, p_sha=P_SHA)
    bad = [i['name'] for i in rec['items'] if not i['ok']]
    assert rec['pass'] is False and bad == ['references_p/style_word/genre']
    assert json.loads((tmp_path / 'outputs/stage_ii/vl1_scorer.json').read_text())['pass'] is False
