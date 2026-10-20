"""Ticket 03 seams: partitions, indexed AMI, bootstrap draws and verdict files."""
import numpy as np
import pytest

import ct_score as score


def test_percept_own_ties_take_the_lowest_topic():
    prob = np.tile(np.array([[.9, .9, .1], [.2, .8, .8]], dtype=np.float32), (20, 1))
    parts = score.output_partitions(prob, 'percept')
    np.testing.assert_array_equal(parts['own'], [np.tile([0, 1], 20)])
    assert parts['r20'].shape == (3, len(prob))


def test_partition_inputs_match_the_sheet_without_renormalizing_k1(monkeypatch):
    from threadpoolctl import threadpool_info
    seen = []

    def sheet_partitions(x, k):
        assert k == 20
        assert all(p['num_threads'] <= 8 for p in threadpool_info())
        seen.append(x.copy())
        return np.zeros((3, len(x)), dtype=np.int64)

    monkeypatch.setattr(score, 'partitions', sheet_partitions)
    x = np.tile(np.array([.9995, 0], dtype=np.float16), (40, 1))
    score.output_partitions(x, 'k1')
    np.testing.assert_array_equal(seen[-1], x.astype(np.float32))
    assert seen[-1].dtype == np.float32
    prob = np.tile(np.array([.3, .4], dtype=np.float32), (40, 1))
    score.output_partitions(prob, 'percept')
    np.testing.assert_array_equal(seen[-1], np.tile(np.array([.6, .8], np.float32), (40, 1)))
    with pytest.raises(AssertionError):
        score.output_partitions(x * 2, 'k1')
    with pytest.raises(AssertionError):
        score.output_partitions(np.full((40, 2), np.nan, np.float32), 'percept')
    with pytest.raises(AssertionError):
        score.output_partitions(np.full((40, 2), 1.1, np.float32), 'percept')


def hand_case():
    # For [0,0,1,1]: perfect AMI=1, constant=0, crossed=-1/2.
    perfect = [0, 0, 1, 1, 9]
    constant = [0, 0, 0, 0, 9]
    crossed = [0, 1, 0, 1, 9]
    parts = {
        'K1': {0: dict(r20=np.array([perfect, constant, crossed])),
               1: dict(r20=np.array([perfect] * 3))},
        'B1': {0: dict(r20=np.array([crossed] * 3), own=np.array([constant])),
               1: dict(r20=np.array([constant] * 3), own=np.array([perfect]))},
        'B2': {4: dict(r20=np.array([perfect, constant, constant]), own=np.array([crossed]))},
    }
    return parts, np.array([0, 0, 1, 1, -1]), {'K1': [0, 1], 'B1': [0, 1], 'B2': [4]}


def test_point_estimate_means_partitions_then_seeds_and_masks_after_indexing():
    parts, y, seeds = hand_case()
    for rows in (np.arange(5), np.array([4, 4, 0, 1, 2, 3])):
        result = score.score_indices(parts, y, rows, seeds)
        assert result['A_K1'] == pytest.approx(7 / 12)
        assert result['A_e'] == pytest.approx({'B1-r20': -.25, 'B1-own': .5,
                                             'B2-r20': 1 / 3, 'B2-own': -.5})
        assert result['d'] == pytest.approx(1 / 12)
        assert result['per_seed']['K1'][0]['r20'] == pytest.approx([1, 0, -.5])
    # Duplicated model seeds are weighted as draws; both readings share the draw.
    result = score.score_indices(parts, y, np.arange(5), {'K1': [0, 0], 'B1': [1, 1], 'B2': [4]})
    assert result['A_K1'] == pytest.approx(1 / 6)
    assert result['A_e']['B1-own'] == pytest.approx(1)
    assert result['d'] == pytest.approx(-5 / 6)


def test_bootstrap_draws_rows_then_nonempty_arms_in_rule_order():
    # Literal default_rng(0) transcript; mapping insertion order is deliberately reversed.
    eligible = {'B5': [0, 3, 4], 'B4': [2], 'B3': [], 'B2': [1, 4], 'B1': [0, 1], 'K1': [0, 1]}
    draws = score.draw_bootstrap(4, eligible, n_resamples=3)
    expected = [
        ([3, 2, 2, 1], {'K1': [0, 0], 'B1': [0, 0], 'B2': [1, 4], 'B4': [2], 'B5': [3, 4, 3]}),
        ([2, 3, 2, 2], {'K1': [1, 1], 'B1': [1, 0], 'B2': [4, 4], 'B4': [2], 'B5': [0, 3, 4]}),
        ([2, 0, 3, 2], {'K1': [1, 0], 'B1': [0, 1], 'B2': [1, 4], 'B4': [2], 'B5': [0, 0, 3]}),
    ]
    for (rows, seeds), (want_rows, want_seeds) in zip(draws, expected, strict=True):
        np.testing.assert_array_equal(rows, want_rows)
        assert list(seeds) == list(want_seeds)
        assert {k: v.tolist() for k, v in seeds.items()} == want_seeds
    assert len(score.draw_bootstrap(4, eligible)) == 2000
    with pytest.raises(AssertionError):
        score.draw_bootstrap(4, dict(K1=[1, 0], B1=[0]))


def switching_case():
    perfect = [0, 0, 1, 1]
    constant = [0, 0, 0, 0]
    same = dict(r20=np.array([perfect] * 3))
    good_r20 = dict(r20=np.array([perfect] * 3), own=np.array([constant]))
    good_own = dict(r20=np.array([constant] * 3), own=np.array([perfect]))
    bad = dict(r20=np.array([constant] * 3), own=np.array([constant]))
    return {'K1': {0: same, 1: same}, 'B1': {0: good_r20, 1: bad}, 'B2': {0: good_own, 1: bad}}


def test_paired_bootstrap_reselects_the_winner_and_uses_linear_percentiles():
    fixed = switching_case()
    eligible = {'K1': [0, 1], 'B1': [0, 1], 'B2': [0, 1], 'B3': []}
    expected = [0, 0, .5, 0, 0, .5, 1, 0, .5, 0, 0, .5, .5, 0, 0, 0]
    for workers in (1, 2):
        result = score.paired_bootstrap(fixed, np.array([0, 0, 1, 1]), eligible,
                                        n_resamples=16, workers=workers)
        assert result['d'] == pytest.approx(.5)
        np.testing.assert_allclose(result['bootstrap_d'], expected, atol=1e-14)
        assert result['interval'] == pytest.approx([0, .8125])
    with pytest.raises(AssertionError):
        score.paired_bootstrap(fixed, np.array([0, 0, 1, 1]), eligible, workers=9)


def test_all_random_indices_are_drawn_before_any_ami(monkeypatch):
    actual_rng, actual_ami = np.random.default_rng, score.ami
    calls = []

    class RandomTranscript:
        def __init__(self, seed):
            assert seed == 0
            self.rng = actual_rng(seed)

        def integers(self, *args, **kwargs):
            calls.append('I')
            return self.rng.integers(*args, **kwargs)

        def choice(self, *args, **kwargs):
            calls.append('S')
            return self.rng.choice(*args, **kwargs)

    def checked_ami(*args):
        assert calls == ['I', 'S', 'S', 'S'] * 3
        return actual_ami(*args)

    monkeypatch.setattr(np.random, 'default_rng', RandomTranscript)
    monkeypatch.setattr(score, 'ami', checked_ami)
    score.paired_bootstrap(switching_case(), np.array([0, 0, 1, 1]),
                           {'K1': [0, 1], 'B1': [0, 1], 'B2': [0, 1]}, n_resamples=3, workers=1)


def test_score_dev_returns_each_partition_ami_with_the_sheet_label_mask():
    y = np.arange(5000) % 20
    x = np.eye(20, dtype=np.float16)[y]
    masked = y.copy()
    masked[::3] = -1
    result = score.score_dev(x, {'emotion': y, 'genre': masked})
    assert set(result) == {'emotion', 'genre'}
    assert result['emotion'] == pytest.approx([1, 1, 1])
    assert result['genre'] == pytest.approx([1, 1, 1])
    with pytest.raises(AssertionError):
        score.score_dev(x[:-1], {'emotion': y[:-1]})
    with pytest.raises(AssertionError):
        score.score_dev(x, {'emotion': y[:-1]})


@pytest.fixture
def held_identity():
    import hashlib
    names = [f'painting_{i:05}' for i in range(12281)]
    return names, hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def write_output(path, x, held_sha256, **meta):
    import json
    np.savez_compressed(path, **({'img': x} if path.parent.name == 'd2' else {'prob': x}))
    path.with_suffix('.json').write_text(json.dumps(dict(held_sha256=held_sha256, rows=len(x), **meta)))


def test_k1_file_boundary_enforces_v3_and_preserves_saved_values(tmp_path, held_identity):
    names, sha = held_identity
    path = tmp_path / 'd2' / 'emotion__s0.npz'
    path.parent.mkdir()
    x = np.zeros((12281, 2048), dtype=np.float16)
    x[:, 0] = .9995
    write_output(path, x, sha, weights_sha256='a' * 64)
    actual = score.load_held_output(path, 'K1', names, sha)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, x.astype(np.float32))
    with pytest.raises(AssertionError):
        score.load_held_output(path, 'K1', names[::-1], sha)
    with pytest.raises(AssertionError):
        score.load_held_output(path, 'K1', names, 'f' * 64)
    for bad in (x[:-1], x.astype(np.float32), x * 2):
        write_output(path, bad, sha, weights_sha256='a' * 64)
        with pytest.raises(AssertionError):
            score.load_held_output(path, 'K1', names, sha)
    x[0, 0] = np.nan
    write_output(path, x, sha, weights_sha256='a' * 64)
    with pytest.raises(AssertionError):
        score.load_held_output(path, 'K1', names, sha)


def test_nonfinite_paper_output_is_collapsed_while_core_fails_v3(tmp_path, held_identity):
    names, sha = held_identity
    path = tmp_path / 'P1__s0.npz'
    prob = np.ones((12281, 40), dtype=np.float32)
    prob[2, 0] = np.inf
    write_output(path, prob, sha)
    assert score.load_held_output(path, 'B2', names, sha) is None
    with pytest.raises(AssertionError):
        score.load_held_output(path, 'B1', names, sha)
    # V4 does not excuse a wrong order/hash or missing rows for a paper seed.
    with pytest.raises(AssertionError):
        score.load_held_output(path, 'B2', names, 'f' * 64)
    write_output(path, prob[:-1], sha)
    with pytest.raises(AssertionError):
        score.load_held_output(path, 'B2', names, sha)


def test_held_partitions_are_saved_once_on_every_row_and_bound_to_the_output(tmp_path, monkeypatch,
                                                                          held_identity):
    names, sha = held_identity
    prob = np.tile(np.array([.6, .8], np.float32), (len(names), 1))
    calls = []

    def sheet_partitions(x, k):
        calls.append(x.shape)
        return np.array([np.arange(len(x)) % k] * 3)

    monkeypatch.setattr(score, 'partitions', sheet_partitions)
    path = tmp_path / 'partitions' / 'B1__s0.npz'
    first = score.fixed_partitions(prob, 'percept', path, sha, 'a' * 64)
    second = score.fixed_partitions(prob, 'percept', path, sha, 'a' * 64)
    assert calls == [(12281, 2)]
    with np.load(path, allow_pickle=False) as saved:
        for reading in ('r20', 'own'):
            np.testing.assert_array_equal(saved[reading], first[reading])
            np.testing.assert_array_equal(second[reading], first[reading])
        assert saved['r20'].shape == (3, 12281)
        assert saved['own'].shape == (1, 12281)
    with pytest.raises(AssertionError):
        score.fixed_partitions(prob, 'percept', path, sha, 'b' * 64)


def make_held_files(root, sha):
    stems = {'K1': 'd2/emotion', 'B1': 'percept/core__k40/P3',
             'B2': 'percept/paper__strict/P1', 'B3': 'percept/paper__strict/P4',
             'B4': 'percept/paper__k40/P1', 'B5': 'percept/paper__k40/P4'}
    x = np.zeros((12281, 2048), np.float16)
    x[:, 0] = 1
    prob = np.tile(np.full(40, .2, np.float32), (12281, 1))
    for arm in ('K1', 'B1', 'B2'):
        for seed in range(5):
            path = root / 'held' / f'{stems[arm]}__s{seed}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            write_output(path, x if arm == 'K1' else prob, sha, weights_sha256='a' * 64)
    return stems, prob


def test_prepare_held_keeps_core_seeds_and_records_paper_nonfinite_collapse(tmp_path, monkeypatch,
                                                                         held_identity):
    names, sha = held_identity
    stems, prob = make_held_files(tmp_path, sha)
    prob[0, 0] = np.nan
    write_output(tmp_path / 'held' / f'{stems["B2"]}__s1.npz', prob, sha)
    # The external sheet estimator is replaced to keep the full-row fixture cheap.
    monkeypatch.setattr(score, 'partitions', lambda x, k: np.zeros((3, len(x)), np.int64))
    fixed, eligibility, sources = score.prepare_held(names, sha,
        {'B2': [0, 1], 'B3': [], 'B4': [], 'B5': []}, tmp_path)
    assert eligibility['eligible_seeds'] == {'K1': list(range(5)), 'B1': list(range(5)),
                                           'B2': [0], 'B3': [], 'B4': [], 'B5': []}
    assert eligibility['rcol_eligible_seeds']['B2'] == [0, 1]
    assert eligibility['nonfinite_held']['B2'] == [1]
    assert set(fixed['K1']) == set(range(5))
    assert set(fixed['B1']) == set(range(5))
    assert 1 not in fixed['B2']
    assert sources['K1']['0']['output_sha256']
    assert sources['B2']['1']['status'] == 'nonfinite_held'
    # A missing required core seed must fail, never reduce its seed list.
    (tmp_path / 'held' / f'{stems["K1"]}__s4.npz').rename(tmp_path / 'absent.npz')
    with pytest.raises(AssertionError):
        score.prepare_held(names, sha, {'B2': [0, 1], 'B3': [], 'B4': [], 'B5': []}, tmp_path)


@pytest.mark.parametrize(('d', 'lower', 'outcome'), [
    (.01, 1e-15, 'S1-GO'), (.01, 0, 'S1-MM'), (.02, -.001, 'S1-MM'),
    (.009999999999, .001, 'S1-IN'), (1e-15, -.001, 'S1-IN'),
    (0, .001, 'S1-NO'), (-.01, -.02, 'S1-NO'),
])
def test_stage1_outcome_uses_unrounded_boundaries(d, lower, outcome):
    assert score.stage1_outcome(d, lower) == outcome


def test_verdict_file_contains_unrounded_statistics_and_audit_inputs(tmp_path, held_identity):
    import json
    _, sha = held_identity
    fixed, labels, seeds = hand_case()
    result = score.paired_bootstrap(fixed, labels, seeds, n_resamples=4, workers=1)
    path = tmp_path / 'verdict_stage1.json'
    eligibility = {'eligible_seeds': seeds, 'rcol_eligible_seeds': {'B2': [4]}, 'nonfinite_held': {}}
    verdict = score.write_verdict(path, result, eligibility, sha,
                                 {'emotion': labels, 'genre': np.array([0, 0, -1, -1, -1])},
                                 'a' * 40, sources={'held_outputs': 'fixture'})
    saved = json.loads(path.read_text())
    assert saved == verdict
    assert saved['d'] == result['d']
    assert saved['interval'] == result['interval']
    assert saved['A_K1'] == pytest.approx(7 / 12)
    assert saved['A_e']['B1-own'] == pytest.approx(.5)
    assert saved['per_seed']['K1']['0']['r20'] == pytest.approx([1, 0, -.5])
    assert saved['eligibility'] == eligibility
    assert saved['held_sha256'] == sha
    assert saved['outcome_id'] == score.stage1_outcome(result['d'], result['interval'][0])
    assert saved['label_counts'] == {'emotion': {'classes': {'0': 2, '1': 2}, 'missing': 1},
                                     'genre': {'classes': {'0': 2}, 'missing': 3}}
    assert saved['code_commit'] == 'a' * 40
    assert saved['sources'] == {'held_outputs': 'fixture'}
    assert saved['bootstrap']['resamples'] == 4
    assert not list(tmp_path.glob('*.pending.*'))
    assert not list(tmp_path.glob('*diag*'))


def test_score_dev_accepts_raw_percept_probabilities_with_explicit_kind():
    y = np.arange(5000) % 20
    prob = np.full((5000, 20), .05, np.float32)
    prob[np.arange(len(y)), y] = .9
    assert score.score_dev(prob, {'emotion': y}, kind='percept')['emotion'] == pytest.approx([1, 1, 1])


def test_default_bootstrap_has_exactly_2000_resamples():
    perfect = np.array([[0, 0, 1, 1]] * 3)
    constant = np.zeros((3, 4), dtype=np.int64)
    fixed = {'K1': {0: {'r20': perfect}},
             'B1': {0: {'r20': constant, 'own': constant[:1]}}}
    result = score.paired_bootstrap(fixed, np.array([0, 0, 1, 1]),
                                    {'K1': [0], 'B1': [0]}, workers=2)
    assert len(result['bootstrap_d']) == 2000
    assert result['d'] == pytest.approx(1)
    assert result['interval'] == pytest.approx([0, 1])


def test_core_k40_file_has_exactly_40_probabilities(tmp_path, held_identity):
    names, sha = held_identity
    path = tmp_path / 'P3__s0.npz'
    write_output(path, np.full((12281, 39), .5, np.float32), sha)
    with pytest.raises(AssertionError):
        score.load_held_output(path, 'B1', names, sha)
