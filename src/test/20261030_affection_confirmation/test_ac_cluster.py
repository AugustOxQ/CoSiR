"""Tests of ac_cluster (ticket 06): partitions, AMI, the paired bootstrap, the partition cache, the r10 replay."""
import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

import ac_cluster


def unit_rows(n, d, seed):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope='module')
def x_small():
    return unit_rows(300, 8, 0)


@pytest.fixture(scope='module')
def labels_small():
    return np.random.default_rng(1).integers(0, 5, size=300).astype(np.int64)


def fake_parts(n, seed, seeded=False, k=6):
    """Random partitions standing in for fitted ones: int32 [3, n] or [5, 3, n]."""
    shape = (5, 3, n) if seeded else (3, n)
    return np.random.default_rng(seed).integers(0, k, size=shape).astype(np.int32)


# ---- partitions -------------------------------------------------------------------------------------------------

def test_partitions_use_r10s_kmeans_settings(x_small):
    got = ac_cluster.partitions(x_small)
    assert got.shape == (3, 300) and got.dtype == np.int32
    for state in (0, 1, 2):
        km = MiniBatchKMeans(n_clusters=20, random_state=state, n_init=3, batch_size=4096)
        np.testing.assert_array_equal(got[state], km.fit(x_small).labels_)


def test_partitions_refuse_settings_other_than_r10s(x_small):
    with pytest.raises(AssertionError):
        ac_cluster.partitions(x_small, k=10)
    with pytest.raises(AssertionError):
        ac_cluster.partitions(x_small, states=(0, 1))


def test_partitions_refuse_rows_that_are_not_unit_norm(x_small):
    with pytest.raises(AssertionError):
        ac_cluster.partitions(x_small * 2)


# ---- AMI ----------------------------------------------------------------------------------------------------------

def test_ami_is_the_mean_over_states_of_scikit_learns(labels_small):
    parts = fake_parts(300, 2)
    expected = np.mean([adjusted_mutual_info_score(labels_small, p) for p in parts])
    assert ac_cluster.ami(parts, labels_small) == pytest.approx(expected, abs=1e-12)


def test_ami_of_a_seeded_side_is_the_mean_over_seeds_of_the_state_means(labels_small):
    parts = fake_parts(300, 3, seeded=True)
    expected = np.mean([[adjusted_mutual_info_score(labels_small, p) for p in seed] for seed in parts])
    assert ac_cluster.ami(parts, labels_small) == pytest.approx(expected, abs=1e-12)


def test_ami_masks_negative_labels(labels_small):
    parts = fake_parts(300, 4)
    y = labels_small.copy()
    y[:40] = -1
    expected = np.mean([adjusted_mutual_info_score(y[40:], p[40:]) for p in parts])
    assert ac_cluster.ami(parts, y) == pytest.approx(expected, abs=1e-12)


# ---- the paired bootstrap -----------------------------------------------------------------------------------------

def mean_ami(y, rows, parts):
    """sklearn's AMI by hand: mean over the states of the AMI of the drawn rows."""
    return np.mean([adjusted_mutual_info_score(y[rows], p[rows]) for p in parts])


def hand_draws(cand, comp, y, draws, comp_seeded=False):
    """The rule's transcript by hand: default_rng(0); per draw rows = integers(0, n, size=n), then seeds = choice(5,
    size=5, replace=True) for the candidate (and the comparator takes the same seeds when it is seeded)."""
    rng = np.random.default_rng(0)
    n, ds = len(y), []
    for _ in range(draws):
        rows = rng.integers(0, n, size=n)
        s = rng.choice(5, size=5, replace=True)
        a_c = np.mean([mean_ami(y, rows, cand[i]) for i in s])
        a_t = np.mean([mean_ami(y, rows, comp[i]) for i in s]) if comp_seeded else mean_ami(y, rows, comp)
        ds.append(a_c - a_t)
    return ds


@pytest.fixture(scope='module')
def y_small():
    return np.random.default_rng(5).integers(0, 4, size=120).astype(np.int64)


def test_paired_draws_images_then_seeds_as_a_hand_loop(y_small):
    cand, comp = fake_parts(120, 6, seeded=True), fake_parts(120, 7)
    got = ac_cluster.paired(cand, comp, y_small, draws=5, workers=1)
    expected = hand_draws(cand, comp, y_small, 5)
    np.testing.assert_allclose(got['bootstrap_d'], expected, atol=1e-12)
    np.testing.assert_allclose(got['interval'], np.percentile(expected, [2.5, 97.5]), atol=1e-12)
    a_c = np.mean([mean_ami(y_small, np.arange(120), c) for c in cand])
    a_t = mean_ami(y_small, np.arange(120), comp)
    assert got['A_cand'] == pytest.approx(a_c, abs=1e-12) and got['A_comp'] == pytest.approx(a_t, abs=1e-12)
    assert got['d'] == pytest.approx(a_c - a_t, abs=1e-12)


def test_paired_gives_the_same_draws_with_other_workers_and_on_a_second_call(y_small):
    cand, comp = fake_parts(120, 6, seeded=True), fake_parts(120, 7)
    first = ac_cluster.paired(cand, comp, y_small, draws=5, workers=1)
    again = ac_cluster.paired(cand, comp, y_small, draws=5, workers=2)
    assert first['bootstrap_d'] == again['bootstrap_d']          # a fresh default_rng(0) per statistic


def test_one_rng_per_statistic_the_comparator_does_not_move_the_draws(y_small):
    cand = fake_parts(120, 6, seeded=True)
    for seed in (7, 8):                                          # two comparators, each equal to its own hand loop
        comp = fake_parts(120, seed)
        got = ac_cluster.paired(cand, comp, y_small, draws=4, workers=1)
        np.testing.assert_allclose(got['bootstrap_d'], hand_draws(cand, comp, y_small, 4), atol=1e-12)


def test_a_seeded_comparator_shares_the_candidates_seed_draw(y_small):
    cand, comp = fake_parts(120, 6, seeded=True), fake_parts(120, 9, seeded=True)
    got = ac_cluster.paired(cand, comp, y_small, draws=4, workers=1)
    np.testing.assert_allclose(got['bootstrap_d'], hand_draws(cand, comp, y_small, 4, comp_seeded=True), atol=1e-12)


def test_paired_checks_shapes_and_seededness(y_small):
    seeded, plain = fake_parts(120, 6, seeded=True), fake_parts(120, 7)
    with pytest.raises(AssertionError):
        ac_cluster.paired(plain, plain, y_small, draws=2)            # the candidate must carry five seeds
    with pytest.raises(AssertionError):
        ac_cluster.paired(seeded[:4], plain, y_small, draws=2)       # four seeds
    with pytest.raises(AssertionError):
        ac_cluster.paired(seeded, plain[:, :100], y_small, draws=2)  # wrong row count
    with pytest.raises(AssertionError):
        ac_cluster.paired(seeded, plain.astype(np.float32), y_small, draws=2)


def test_identical_partitions_give_d_zero(y_small):
    comp = fake_parts(120, 7)
    cand = np.stack([comp] * 5)
    got = ac_cluster.paired(cand, comp, y_small, draws=20, workers=1)
    assert got['d'] == pytest.approx(0, abs=1e-12)
    assert got['interval'] == pytest.approx([0, 0], abs=1e-12)


# ---- PercepT core's two readings ----------------------------------------------------------------------------------

def test_percept_readings_are_r10s_r20_on_l2_probabilities_and_the_argmax():
    prob = np.random.default_rng(10).uniform(0.01, 0.99, size=(200, 40)).astype(np.float32)
    got = ac_cluster.percept_partitions(prob)
    assert set(got) == {'r20', 'own'}
    assert got['r20'].shape == (3, 200) and got['own'].shape == (1, 200)
    assert got['r20'].dtype == got['own'].dtype == np.int32
    unit = prob / np.linalg.norm(prob, axis=1, keepdims=True)
    for state in (0, 1, 2):
        km = MiniBatchKMeans(n_clusters=20, random_state=state, n_init=3, batch_size=4096)
        np.testing.assert_array_equal(got['r20'][state], km.fit(unit).labels_)
    np.testing.assert_array_equal(got['own'][0], prob.argmax(axis=1))


# ---- the cache ----------------------------------------------------------------------------------------------------

def test_cache_writes_once_and_a_hit_returns_the_same_partitions(x_small, tmp_path):
    first = ac_cluster.cached_partitions(x_small, 'd__test__arm__s0', root=tmp_path)
    path = tmp_path / 'd__test__arm__s0.npz'
    assert path.is_file() and (tmp_path / 'd__test__arm__s0.json').is_file()
    stamp = path.stat().st_mtime_ns
    again = ac_cluster.cached_partitions(x_small, 'd__test__arm__s0', root=tmp_path)
    np.testing.assert_array_equal(first, again)
    assert again.dtype == np.int32 and path.stat().st_mtime_ns == stamp          # no rewrite on a hit
    with np.load(path) as z:
        assert z.files == ['labels']


def test_a_hit_is_decided_by_the_array_hash_not_the_name(x_small, tmp_path):
    ac_cluster.cached_partitions(x_small, 'same_name', root=tmp_path)
    other = unit_rows(300, 8, 99)
    got = ac_cluster.cached_partitions(other, 'same_name', root=tmp_path)
    np.testing.assert_array_equal(got, ac_cluster.partitions(other))               # not the stale partitions of x_small
    assert not np.array_equal(got, ac_cluster.partitions(x_small))


def test_a_changed_cache_file_is_refused(x_small, tmp_path):
    ac_cluster.cached_partitions(x_small, 'broken', root=tmp_path)
    np.savez(tmp_path / 'broken.npz', labels=np.zeros((3, 300), dtype=np.int32))
    with pytest.raises(AssertionError):
        ac_cluster.cached_partitions(x_small, 'broken', root=tmp_path)


def test_cache_hashes_the_float32_rows_so_float16_and_float32_of_one_array_agree(x_small, tmp_path):
    x16 = x_small.astype(np.float16)
    a = ac_cluster.cached_partitions(x16, 'half', root=tmp_path)
    stamp = (tmp_path / 'half.npz').stat().st_mtime_ns
    b = ac_cluster.cached_partitions(x16.astype(np.float32), 'half', root=tmp_path)
    np.testing.assert_array_equal(a, b)
    assert (tmp_path / 'half.npz').stat().st_mtime_ns == stamp


# ---- spec §3.5 0a(iii), r10 part: the replay on r10's held emotion inputs (slow; run by the controller) --------------

R10_BLOCK = dict(A_cand=0.10658729378204021, A_T=0.09207439052277983, d=0.014512903259260385,
                 interval=[0.011288090988306501, 0.017671180133450672])


@pytest.mark.real_data
def test_replay_plumbing_on_a_prefix_of_the_held_rows_returns_the_block_keys():
    got = ac_cluster.r10_emotion_replay(workers=2, draws=3, limit=400)
    assert set(got) == {'A_cand', 'A_T', 'd', 'interval'}
    assert got['d'] == pytest.approx(got['A_cand'] - got['A_T'], abs=1e-12)
    assert got['interval'][0] <= got['interval'][1]


@pytest.mark.slow
def test_replay_gives_r10s_stage_4_emotion_block_within_1e_9():
    got = ac_cluster.r10_emotion_replay(workers=8)
    for key in ('A_cand', 'A_T', 'd'):
        assert got[key] == pytest.approx(R10_BLOCK[key], abs=1e-9), key
    assert got['interval'] == pytest.approx(R10_BLOCK['interval'], abs=1e-9)
