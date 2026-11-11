"""Tests of r21_table (ticket 06): the word rule, arm and seed pairing, the change cells, silhouette and cluster shares,
the DEC statistics from training sidecars, and two real-file checks on r10's saved tree (sidecars; fixed partitions)."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r21_table as T  # noqa: E402

SEEDS = T.SEEDS
ASPECTS = T.ASPECTS


# ------------------------------------------------------------------------------------------------ the word rule (D14)
@pytest.mark.parametrize('lo, hi, word', [
    (0.001, 0.002, 'raises'),
    (1e-18, 0.5, 'raises'),                 # strictly above zero, however small
    (-0.002, -0.001, 'lowers'),
    (-0.5, -1e-18, 'lowers'),
    (-0.001, 0.001, 'no clear change'),
    (0.0, 0.003, 'no clear change'),        # the interval touches zero from above: not "raises"
    (-0.003, 0.0, 'no clear change'),       # touches zero from below: not "lowers"
    (0.0, 0.0, 'no clear change'),          # c* against itself: [0, 0]
])
def test_word_rule_at_the_edges(lo, hi, word):
    assert T.word(lo, hi) == word


@pytest.mark.parametrize('lo, hi', [(float('nan'), 0.1), (0.0, float('inf')), (0.2, 0.1), (None, 0.1)])
def test_word_refuses_bad_intervals(lo, hi):
    with pytest.raises(AssertionError):
        T.word(lo, hi)


def test_cell_carries_word_and_values():
    c = T.cell(0.01, 0.002, 0.02)
    assert c == dict(change=0.01, lo=0.002, hi=0.02, word='raises')
    with pytest.raises(AssertionError):
        T.cell(float('nan'), 0.0, 0.1)


# ------------------------------------------------------------------------------------------------ pairing
SHA = 'a' * 64


def fake_metas(arm, shas=None, drop=None, extra=None):
    m = {(a, s): dict(arm=arm, aspect=a, seed=s, rows=T.N_HELD, list_sha256=(shas or {}).get((a, s), SHA),
                      array_sha256=f'{arm}{a}{s}') for a in ASPECTS for s in SEEDS}
    if drop:
        del m[drop]
    if extra:
        m[extra] = dict(arm=arm, aspect=extra[0], seed=extra[1], rows=T.N_HELD, list_sha256=SHA, array_sha256='x')
    return m


def test_pairing_accepts_a_full_matching_grid():
    T.check_pairing(fake_metas('dec-off'), fake_metas('k11=40'), SHA)


def test_pairing_refuses_a_missing_seed():
    with pytest.raises(AssertionError, match='missing|grid'):
        T.check_pairing(fake_metas('dec-off', drop=('style', 3)), fake_metas('k11=40'), SHA)
    with pytest.raises(AssertionError, match='missing|grid'):
        T.check_pairing(fake_metas('dec-off'), fake_metas('k11=40', drop=('genre', 0)), SHA)


def test_pairing_refuses_an_extra_seed_or_aspect():
    with pytest.raises(AssertionError):
        T.check_pairing(fake_metas('dec-off', extra=('emotion', 5)), fake_metas('k11=40'), SHA)
    with pytest.raises(AssertionError):
        T.check_pairing(fake_metas('dec-off', extra=('colour', 0)), fake_metas('k11=40'), SHA)


def test_pairing_refuses_a_mismatched_held_order():
    other = {('emotion', 2): 'b' * 64}
    with pytest.raises(AssertionError, match='held'):
        T.check_pairing(fake_metas('dec-off', shas=other), fake_metas('k11=40'), SHA)
    with pytest.raises(AssertionError, match='held'):
        T.check_pairing(fake_metas('dec-off'), fake_metas('k11=40', shas=other), SHA)


def test_pairing_refuses_a_wrong_row_count_or_a_mislabelled_entry():
    bad = fake_metas('dec-off')
    bad[('style', 1)]['rows'] = 12280
    with pytest.raises(AssertionError):
        T.check_pairing(bad, fake_metas('k11=40'), SHA)
    bad = fake_metas('dec-off')
    bad[('style', 1)]['seed'] = 2                       # a sidecar of another seed under this key
    with pytest.raises(AssertionError):
        T.check_pairing(bad, fake_metas('k11=40'), SHA)


def test_pairing_refuses_the_same_files_twice_under_two_names():
    with pytest.raises(AssertionError, match='same'):
        a = fake_metas('dec-off')
        T.check_pairing(a, {k: dict(v, arm='k11=40') for k, v in a.items()}, SHA)


def test_queries_must_match_between_arm_and_cstar():
    ap = np.random.default_rng(0).random((5, 30))
    one = dict(ap=ap, query_ids=np.array([f'p{i}' for i in range(30)]), dropped=0)
    T.map_cell(one, one, 50)
    with pytest.raises(AssertionError, match='query'):
        T.map_cell(one, dict(one, query_ids=one['query_ids'][::-1].copy()), 50)
    with pytest.raises(AssertionError, match='dropped'):
        T.map_cell(one, dict(one, dropped=1), 50)


# ------------------------------------------------------------------------------------------------ the change cells
def noisy(y, k, rate, rng):
    p = y.copy()
    flip = rng.random(len(y)) < rate
    p[flip] = rng.integers(0, k, flip.sum())
    return p


@pytest.fixture(scope='module')
def synth():
    rng = np.random.default_rng(1)
    n, k = 600, 5
    y = rng.integers(0, k, n)
    y[:40] = -1                                          # unlabelled rows are masked in scoring only
    good = {s: np.stack([noisy(np.where(y < 0, 0, y), k, 0.2, rng) for _ in range(3)]) for s in SEEDS}
    bad = {s: np.stack([noisy(np.where(y < 0, 0, y), k, 0.7, rng) for _ in range(3)]) for s in SEEDS}
    return y, good, bad


def test_ami_cell_against_itself_is_zero_and_no_clear_change(synth):
    y, good, _ = synth
    c = T.ami_cell(good, good, y, 20, 2)
    assert c['change'] == 0.0 and c['lo'] == 0.0 and c['hi'] == 0.0 and c['word'] == 'no clear change'
    assert c['arm'] == c['cstar'] and c['arm_seed'] == c['cstar_seed']


def test_ami_cell_signs_and_absolute_values(synth):
    y, good, bad = synth
    up = T.ami_cell(good, bad, y, 40, 2)
    down = T.ami_cell(bad, good, y, 40, 2)
    assert up['word'] == 'raises' and down['word'] == 'lowers'
    assert up['change'] == pytest.approx(-down['change'], abs=1e-12)
    assert up['change'] == pytest.approx(up['arm'] - up['cstar'], abs=1e-12)
    assert sorted(up['arm_seed']) == [str(s) for s in SEEDS]
    # the point estimate is the mean over seeds of the mean over the three states of sklearn's AMI on labelled rows
    from sklearn.metrics import adjusted_mutual_info_score as ami
    keep = y >= 0
    want = np.mean([np.mean([ami(y[keep], p[keep]) for p in good[s]]) for s in SEEDS])
    assert up['arm'] == pytest.approx(want, abs=1e-12)


def test_ami_cell_needs_the_five_seeds(synth):
    y, good, bad = synth
    with pytest.raises(AssertionError):
        T.ami_cell({s: good[s] for s in (0, 1, 2, 3)}, bad, y, 10, 2)
    with pytest.raises(AssertionError):
        T.ami_cell(good, {s: bad[s] for s in (1, 2, 3, 4)}, y, 10, 2)


def ap_pair(rng, shift):
    q = np.array([f'p{i}' for i in range(60)])
    base = rng.random((5, 60)) * 0.5
    return (dict(ap=base + shift, query_ids=q, dropped=2), dict(ap=base, query_ids=q, dropped=2))


def test_map_cell_against_itself_and_signs():
    rng = np.random.default_rng(2)
    arm, cs = ap_pair(rng, 0.0)
    z = T.map_cell(cs, cs, 200)
    assert z['change'] == 0.0 and z['lo'] == 0.0 and z['hi'] == 0.0 and z['word'] == 'no clear change'
    arm, cs = ap_pair(rng, 0.05)
    up = T.map_cell(arm, cs, 200)
    assert up['word'] == 'raises' and up['change'] == pytest.approx(0.05, abs=1e-12)
    assert up['arm'] == pytest.approx(arm['ap'].mean()) and up['cstar'] == pytest.approx(cs['ap'].mean())
    assert up['n_queries'] == 60 and up['dropped'] == 2
    down = T.map_cell(cs, arm, 200)
    assert down['word'] == 'lowers'


def test_map_cell_follows_the_reference_bootstrap():
    """The interval is `r20_ret.paired`'s (ns_ret shared-seed bootstrap, default_rng(0)), unchanged."""
    T._deps()
    import r20_ret
    rng = np.random.default_rng(3)
    arm, cs = ap_pair(rng, 0.01)
    ref = r20_ret.paired(arm, cs, 300)
    got = T.map_cell(arm, cs, 300)
    assert (got['change'], got['lo'], got['hi']) == (ref['d'], ref['lo'], ref['hi'])


# ------------------------------------------------------------------------------------------------ silhouette, shares
def test_silhouette_equals_sklearn_cosine():
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 16)) + np.repeat(rng.normal(size=(4, 16)) * 2, 75, axis=0)
    lab = np.repeat(np.arange(4), 75)
    lab[::7] = rng.integers(0, 4, len(lab[::7]))
    assert T.silhouette_cosine(x, lab) == pytest.approx(silhouette_score(x, lab, metric='cosine'), abs=1e-9)
    # scale of the rows does not matter, and labels need not be 0..k-1
    assert T.silhouette_cosine(x * 3.0, lab * 5 + 2) == pytest.approx(T.silhouette_cosine(x, lab), abs=1e-12)


def test_silhouette_singleton_cluster_scores_zero_as_sklearn():
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 8))
    lab = np.array([0] * 20 + [1] * 19 + [2])
    assert T.silhouette_cosine(x, lab) == pytest.approx(silhouette_score(x, lab, metric='cosine'), abs=1e-9)


def test_silhouette_needs_two_clusters():
    with pytest.raises(AssertionError):
        T.silhouette_cosine(np.ones((5, 3)), np.zeros(5, dtype=int))


def test_cluster_shares_count_empty_clusters():
    s = T.cluster_shares(np.array([0, 0, 0, 1, 3]), 5)
    assert s['largest'] == 0.6 and s['smallest'] == 0.0 and s['n_empty'] == 2 and s['n_used'] == 3
    s = T.cluster_shares(np.array([0, 1, 1, 2]), 3)
    assert s['largest'] == 0.5 and s['smallest'] == 0.25 and s['n_empty'] == 0
    with pytest.raises(AssertionError):
        T.cluster_shares(np.array([0, 1, 7]), 3)         # a label outside the class count


# ------------------------------------------------------------------------------------------------ DEC statistics
def dec_history(n_warm=2, n_dec=3, reason='changed'):
    h = [dict(epoch=e + 1, dev_loss=1.0, dec=dict(phase='warmup')) for e in range(n_warm)]
    h[-1]['dec'].update(start_epoch=n_warm + 1, share_max=0.2, share_min=0.05)
    for i in range(n_dec):
        h.append(dict(epoch=n_warm + i + 1, dev_loss=1.0,
                      dec=dict(phase='dec', start_epoch=n_warm + 1, changed=0.1 / (i + 1), away=0.1 * (i + 1),
                               share_max=0.3, share_min=0.01)))
    h[-1]['dec']['stop_reason'] = reason
    return h


def test_dec_stats_counts_only_dec_phase_epochs():
    side = dict(history=dec_history(), best_epoch=4, epochs_run=5)
    d = T.dec_stats(side)
    assert d['dec_on'] is True and d['dec_epochs'] == 3 and d['dec_start_epoch'] == 3 and d['warmup_epochs'] == 2
    assert d['stop_epoch'] == 5 and d['best_epoch'] == 4 and d['stop_reason'] == 'changed'
    assert d['away'] == pytest.approx(0.3) and d['share_max'] == 0.3 and d['share_min'] == 0.01


def test_dec_stats_for_the_rerun_arm_has_no_dec_entries():
    h = [dict(epoch=e + 1, dev_loss=1.0) for e in range(6)]
    d = T.dec_stats(dict(history=h, best_epoch=4, epochs_run=6))
    assert d['dec_on'] is False and d['dec_epochs'] == 0 and d['away'] is None and d['stop_epoch'] == 6


def test_dec_stats_refuses_inconsistent_sidecars():
    with pytest.raises(AssertionError):                                   # epochs_run differs from the history length
        T.dec_stats(dict(history=dec_history(), best_epoch=4, epochs_run=9))
    h = dec_history()
    del h[-1]['dec']['stop_reason']
    with pytest.raises(AssertionError, match='stop'):
        T.dec_stats(dict(history=h, best_epoch=4, epochs_run=5))
    h = dec_history()
    h[1]['dec']['phase'] = 'dec'                                           # a DEC entry before the last warm-up one
    with pytest.raises(AssertionError):
        T.dec_stats(dict(history=h, best_epoch=4, epochs_run=5))
    h = dec_history()
    h[2]['dec'] = dict(phase='burn')
    with pytest.raises(AssertionError):
        T.dec_stats(dict(history=h, best_epoch=4, epochs_run=5))


def test_run_record_cross_check_flags_disagreement_without_hiding_it():
    side = dict(history=dec_history(), best_epoch=4, epochs_run=5)
    d = T.dec_stats(side)
    ok = T.record_agrees(d, dict(stop_reason='changed', stop_epoch=5, dec_start_epoch=3))
    assert ok == dict(stop_reason=True, stop_epoch=True, dec_start_epoch=True)
    bad = T.record_agrees(d, dict(stop_reason='patience', stop_epoch=5, dec_start_epoch=4))
    assert bad == dict(stop_reason=False, stop_epoch=True, dec_start_epoch=False)
    assert T.record_agrees(d, None) is None


# ------------------------------------------------------------------------------------------------ arms and defaults
def test_arm_names_are_checked_for_the_real_tree_only():
    T.check_arm_names(['dec-off', 'dec-plain_l1_w1_S1'], T.R21_FULL)
    with pytest.raises(AssertionError):
        T.check_arm_names(['k11=40-no_buddy'], T.R21_FULL)                # a removal arm is not an r21 arm
    T.check_arm_names(['k11=40-no_buddy'], T.R10_FULL)                     # the regression reads r10's tree
    with pytest.raises(AssertionError):
        T.check_arm_names(['dec-off', 'dec-off'], T.R21_FULL)


def test_picked_arms_read_the_stage1b_picks(tmp_path):
    p = tmp_path / 'picks.json'
    p.write_text(json.dumps(dict(stage1a=None, stage1b=dict(
        plain=dict(pick='dec-plain_l1_w1_S1'), start=dict(pick='dec-start_l1_w1_S1_bal'),
        teacher=dict(pick='dec-teacher_l1_w1_S1')))))
    assert T.picked_arms(p) == ['dec-off', 'dec-plain_l1_w1_S1', 'dec-start_l1_w1_S1_bal', 'dec-teacher_l1_w1_S1']
    p.write_text(json.dumps(dict(stage1a=None, stage1b=None)))
    with pytest.raises(AssertionError):
        T.picked_arms(p)


# ------------------------------------------------------------------------------------------------ real files (r10's tree)
R10_HELD = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held')
needs_r10 = pytest.mark.skipif(not R10_HELD.exists(), reason='r10 outputs not mounted')


@needs_r10
def test_real_sidecars_pair_cstar_with_a_removal_arm():
    c = T.read_metas(T.R10_FULL, 'k11=40')
    x = T.read_metas(T.R10_FULL, 'k11=40-no_buddy')
    T.check_pairing(x, c, T.held_sha())
    assert len(c) == 15 and c[('genre', 4)]['rows'] == 12281
    bad = copy.deepcopy(x)
    bad[('genre', 4)]['list_sha256'] = 'f' * 64
    with pytest.raises(AssertionError):
        T.check_pairing(bad, c, T.held_sha())


@needs_r10
def test_real_partitions_equal_r10s_saved_fixed_partitions_at_20(tmp_path):
    """Our clustering at 20 (the class-count code path, k = 20) equals the partitions r10's ablation saved."""
    root = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/ablation/cache/held')
    x, meta = T.load_held(T.R10_FULL, 'k11=40', 'emotion', 3)
    got = T.get_partitions(tmp_path, 'k11=40', 'emotion', 3, 20, meta, x)
    want = np.load(root / 'k11=40/emotion__s3__r20.npz')['labels']
    assert got.shape == (3, 12281) and np.array_equal(got, want)
    again = T.get_partitions(tmp_path, 'k11=40', 'emotion', 3, 20, meta, None)      # read back from the cache, no array
    assert np.array_equal(again, got)
