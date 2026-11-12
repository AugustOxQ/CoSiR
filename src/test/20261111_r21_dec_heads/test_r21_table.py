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
IMG = '5af25f2a' + 'c' * 56


def fake_metas(arm, shas=None, drop=None, extra=None, imgs=None):
    m = {(a, s): dict(arm=arm, aspect=a, seed=s, rows=T.N_HELD, list_sha256=(shas or {}).get((a, s), SHA),
                      image_sha256=(imgs or {}).get((a, s), IMG), array_sha256=f'{arm}{a}{s}') for a in ASPECTS for s in SEEDS}
    if drop:
        del m[drop]
    if extra:
        m[extra] = dict(arm=arm, aspect=extra[0], seed=extra[1], rows=T.N_HELD, list_sha256=SHA, image_sha256=IMG, array_sha256='x')
    return m


def test_pairing_accepts_a_full_matching_grid():
    assert T.check_pairing(fake_metas('dec-off'), fake_metas('k11=40'), SHA) is False       # not byte for byte equal


def test_pairing_refuses_other_input_rows():
    """F2: the npz has no IDs; the sidecar's image_sha256 (the rows fed to inference) must equal c*'s per aspect and seed."""
    other = {('genre', 3): 'd' * 64}
    with pytest.raises(AssertionError, match='image_sha256|input rows'):
        T.check_pairing(fake_metas('dec-off', imgs=other), fake_metas('k11=40'), SHA)
    with pytest.raises(AssertionError, match='image_sha256|input rows'):
        T.check_pairing(fake_metas('dec-off'), fake_metas('k11=40', imgs=other), SHA)
    with pytest.raises(AssertionError, match='image_sha256'):                         # a sidecar without the field
        bad = fake_metas('dec-off')
        bad[('style', 0)]['image_sha256'] = None
        T.check_pairing(bad, fake_metas('k11=40'), SHA)


def test_sidecar_meta_reads_image_sha256(tmp_path):
    arr = dict(img=dict(sha256='e' * 64))
    side = dict(arm='dec-off', aspect='style', seed=2, rows=5, list_sha256=SHA, image_sha256=IMG, arrays=arr)
    (tmp_path / 'style__s2.npz').write_bytes(b'x')
    (tmp_path / 'style__s2.json').write_text(json.dumps(side))
    m = T._sidecar_meta(tmp_path / 'style__s2.npz')
    assert m['image_sha256'] == IMG and m['array_sha256'] == 'e' * 64 and m['list_sha256'] == SHA


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
        a = fake_metas('dec-plain_l1_w1_S1')
        T.check_pairing(a, {k: dict(v, arm='k11=40') for k, v in a.items()}, SHA)


def test_pairing_lets_the_cstar_rerun_equal_cstar_byte_for_byte():
    """The `dec-off` arm is c* rerun: identical arrays are allowed and reported; a DEC arm with even one equal array is refused."""
    a = fake_metas('dec-off')
    assert T.check_pairing(a, {k: dict(v, arm='k11=40') for k, v in a.items()}, SHA) is True
    part = {k: dict(v, arm='k11=40') if k == ('emotion', 0) else dict(v, arm='k11=40', array_sha256='z' + v['array_sha256'])
            for k, v in a.items()}
    assert T.check_pairing(a, part, SHA) is False
    d = fake_metas('dec-teacher_l1_w1_S1')
    with pytest.raises(AssertionError, match='same'):
        T.check_pairing(d, {k: dict(v, arm='k11=40') if k == ('emotion', 0) else dict(v, arm='k11=40', array_sha256='z' + v['array_sha256'])
                            for k, v in d.items()}, SHA)


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


def _blobs():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(200, 8)) + np.repeat(rng.normal(size=(4, 8)) * 2, 50, axis=0)
    return x, np.repeat(np.arange(4), 50)


def test_internal_scores_equal_sklearn_on_unit_rows():
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    x, lab = _blobs()
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    got = T.internal_scores_unit(x, lab)
    assert got['dbi'] == pytest.approx(davies_bouldin_score(u, lab), abs=1e-9)
    assert got['chi'] == pytest.approx(calinski_harabasz_score(u, lab), rel=1e-9)


def test_internal_scores_ignore_row_scale_and_label_values():
    x, lab = _blobs()
    a, b = T.internal_scores_unit(x, lab), T.internal_scores_unit(x * 3.0, lab * 5 + 2)
    assert b['dbi'] == pytest.approx(a['dbi'], abs=1e-9) and b['chi'] == pytest.approx(a['chi'], rel=1e-9)


def test_internal_scores_need_two_clusters():
    with pytest.raises(AssertionError):
        T.internal_scores_unit(np.ones((5, 3)), np.zeros(5, dtype=int))


def test_diag_markdown_has_internal_scores_and_paired_change():
    def src(si, dbi, chi):
        return {a: dict(mean=dict(silhouette=np.mean(si), dbi=np.mean(dbi), chi=np.mean(chi), largest=0.2, smallest=0.1),
                        seeds={str(i): dict(silhouette=si[i], dbi=dbi[i], chi=chi[i]) for i in range(2)}) for a in ASPECTS}
    doc = dict(sources=dict(cstar=src([0.10, 0.20], [3.0, 3.0], [100., 100.]), arm1=src([0.15, 0.30], [2.5, 2.0], [110., 130.])),
               dec={}, val={})
    md = T.diag_markdown(doc)
    i = md.index('## Internal cluster scores (test)')
    assert i > md.index('## Silhouette and cluster shares (test)') and i < md.index('## Change from c*, paired by seed')
    row = [l for l in md.split('## Change from c*')[1].splitlines() if l.startswith('| arm1 |')][0]
    assert '+0.0750 [+0.0500, +0.1000]' in row and '-0.7500 [-1.0000, -0.5000]' in row and '+20.0 [+10.0, +30.0]' in row
    assert '| cstar | ' not in md.split('## Change from c*')[1]


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


ARM = 'dec-plain_l1_w2_S1'


def test_dec_stats_counts_only_dec_phase_epochs():
    side = dict(history=dec_history(), best_epoch=4, epochs_run=5)
    d = T.dec_stats(ARM, 'emotion', side)
    assert d['dec_on'] is True and d['dec_epochs'] == 3 and d['dec_start_epoch'] == 3 and d['warmup_epochs'] == 2
    assert d['stop_epoch'] == 5 and d['best_epoch'] == 4 and d['stop_reason'] == 'changed'
    assert d['away'] == pytest.approx(0.3) and d['share_max'] == 0.3 and d['share_min'] == 0.01
    assert (d['dec_epochs'], d['dec_start_epoch'], d['stop_reason']) == T.SC.head_dec(ARM, 'emotion', side)    # F4: one reading


def test_dec_stats_for_the_rerun_arm_has_no_dec_entries():
    h = [dict(epoch=e + 1, dev_loss=1.0) for e in range(6)]
    d = T.dec_stats('dec-off', 'emotion', dict(history=h, best_epoch=4, epochs_run=6))
    assert d['dec_on'] is False and d['dec_epochs'] == 0 and d['away'] is None and d['stop_epoch'] == 6 and d['stop_reason'] is None
    with pytest.raises(AssertionError):                                    # DEC entries under a DEC-off arm
        T.dec_stats('dec-off', 'emotion', dict(history=dec_history(), best_epoch=4, epochs_run=5))


def test_dec_stats_reads_as_strictly_as_the_stage1_sheet():
    """F4: a DEC-on arm whose history has no `dec` entries, no DEC epoch, or an unknown stop reason is refused (it is not DEC off)."""
    h = [dict(epoch=e + 1, dev_loss=1.0) for e in range(6)]
    with pytest.raises(AssertionError):
        T.dec_stats(ARM, 'emotion', dict(history=h, best_epoch=4, epochs_run=6))
    with pytest.raises(AssertionError):                                    # warm-up only: zero DEC epochs
        T.dec_stats(ARM, 'emotion', dict(history=dec_history(n_dec=0), best_epoch=1, epochs_run=2))
    with pytest.raises(AssertionError):
        T.dec_stats(ARM, 'emotion', dict(history=dec_history(reason='bored'), best_epoch=4, epochs_run=5))


def test_dec_stats_refuses_inconsistent_sidecars():
    with pytest.raises(AssertionError):                                   # epochs_run differs from the history length
        T.dec_stats(ARM, 'emotion', dict(history=dec_history(), best_epoch=4, epochs_run=9))
    h = dec_history()
    del h[-1]['dec']['stop_reason']
    with pytest.raises(AssertionError, match='stop'):
        T.dec_stats(ARM, 'emotion', dict(history=h, best_epoch=4, epochs_run=5))
    h = dec_history()
    h[1]['dec']['phase'] = 'dec'                                           # a DEC entry before the last warm-up one
    with pytest.raises(AssertionError):
        T.dec_stats(ARM, 'emotion', dict(history=h, best_epoch=4, epochs_run=5))
    h = dec_history()
    h[2]['dec'] = dict(phase='burn')
    with pytest.raises(AssertionError):
        T.dec_stats(ARM, 'emotion', dict(history=h, best_epoch=4, epochs_run=5))


def test_run_record_cross_check_flags_disagreement_without_hiding_it():
    side = dict(history=dec_history(), best_epoch=4, epochs_run=5)
    d = T.dec_stats(ARM, 'emotion', side)
    ok = T.record_agrees(d, dict(stop_reason='changed', stop_epoch=5, dec_start_epoch=3))
    assert ok == dict(stop_reason=True, stop_epoch=True, dec_start_epoch=True)
    bad = T.record_agrees(d, dict(stop_reason='patience', stop_epoch=5, dec_start_epoch=4))
    assert bad == dict(stop_reason=False, stop_epoch=True, dec_start_epoch=False)
    assert T.record_agrees(d, None) is None


def test_run_record_cross_check_for_dec_off_ignores_the_reason():
    """F3: DEC off has no stop reason in the history, the run record gives c*'s (`patience` / `max_epochs`): not compared."""
    h = [dict(epoch=e + 1, dev_loss=1.0) for e in range(6)]
    d = T.dec_stats('dec-off', 'emotion', dict(history=h, best_epoch=4, epochs_run=6))
    ok = T.record_agrees(d, dict(stop_reason='patience', stop_epoch=6, dec_start_epoch=None))
    assert ok == dict(stop_epoch=True, dec_start_epoch=True) and all(ok.values())
    assert T.record_agrees(d, dict(stop_reason='patience', stop_epoch=7, dec_start_epoch=None)) == dict(stop_epoch=False, dec_start_epoch=True)


def test_diag_markdown_shows_the_best_epoch_and_flags_only_real_disagreement():
    """F7 and F3 in the markdown: DEC epochs / stop epoch / best epoch / reason / away; "(run record differs)" only if some field differs."""
    side = dict(history=dec_history(), best_epoch=4, epochs_run=5)
    on = T.dec_stats(ARM, 'emotion', side)
    h = [dict(epoch=e + 1, dev_loss=1.0) for e in range(6)]
    off = T.dec_stats('dec-off', 'emotion', dict(history=h, best_epoch=2, epochs_run=6))
    rec = lambda stats, head: dict(stats, run_record=head, run_record_agrees=T.record_agrees(stats, head))
    doc = dict(sources={}, val={}, dec={
        ARM: {a: {'0': rec(on, dict(stop_reason='changed', stop_epoch=5, dec_start_epoch=3))} for a in ASPECTS},
        'dec-off': {a: {'0': rec(off, dict(stop_reason='max_epochs', stop_epoch=6, dec_start_epoch=None))} for a in ASPECTS}})
    text = T.diag_markdown(doc)
    assert '| 3 / 5 / 4 / changed / 0.300 |' in text                       # best epoch 4 is shown
    assert '| 0 / 6 / 2 / n/a / n/a |' in text and 'run record differs' not in text
    doc['dec'][ARM]['style']['0'] = rec(on, dict(stop_reason='changed', stop_epoch=6, dec_start_epoch=3))
    assert text.count('run record differs') == 0 and T.diag_markdown(doc).count('run record differs') == 1


# ------------------------------------------------------------------------------------------------ arms and defaults
def test_arm_names_are_checked_for_the_real_tree_only():
    T.check_arm_names(['dec-off', 'dec-plain_l1_w1_S1'], T.R21_FULL)
    with pytest.raises(AssertionError):
        T.check_arm_names(['k11=40-no_buddy'], T.R21_FULL)                # a removal arm is not an r21 arm
    T.check_arm_names(['k11=40-no_buddy'], T.R10_FULL)                     # the regression reads r10's tree
    with pytest.raises(AssertionError):
        T.check_arm_names(['dec-off', 'dec-off'], T.R21_FULL)


def picks_doc(**over):
    d = dict(stage1a=dict(pick='dec-plain_l1_w1_S1'), stage1b=dict(
        plain=dict(pick='dec-plain_l1_w1_S1'), start=dict(pick='dec-start_l1_w1_S1_bal'), teacher=dict(pick='dec-teacher_l1_w1_S1')))
    d.update(over)
    return d


def test_picked_arms_read_the_stage1b_picks(tmp_path):
    p = tmp_path / 'picks.json'
    p.write_text(json.dumps(picks_doc()))
    assert T.picked_arms(p) == ['dec-off', 'dec-plain_l1_w1_S1', 'dec-start_l1_w1_S1_bal', 'dec-teacher_l1_w1_S1']
    p.write_text(json.dumps(dict(stage1a=None, stage1b=None)))
    with pytest.raises(AssertionError):
        T.picked_arms(p)


def test_picked_arms_check_form_and_shared_settings(tmp_path):
    """F6: each pick is an arm of its own form; the three share 1a's λ, W and stop rule."""
    p = tmp_path / 'picks.json'
    bad = {
        'form': picks_doc(stage1b=dict(plain=dict(pick='dec-plain_l1_w1_S1'), start=dict(pick='dec-teacher_l1_w1_S1'),
                                       teacher=dict(pick='dec-teacher_l1_w1_S1'))),
        'lam': picks_doc(stage1b=dict(plain=dict(pick='dec-plain_l1_w1_S1'), start=dict(pick='dec-start_l10_w1_S1'),
                                      teacher=dict(pick='dec-teacher_l1_w1_S1'))),
        'warmup': picks_doc(stage1b=dict(plain=dict(pick='dec-plain_l1_w5_S1'), start=dict(pick='dec-start_l1_w1_S1'),
                                         teacher=dict(pick='dec-teacher_l1_w1_S1'))),
        'stop': picks_doc(stage1b=dict(plain=dict(pick='dec-plain_l1_w1_S1'), start=dict(pick='dec-start_l1_w1_S1'),
                                       teacher=dict(pick='dec-teacher_l1_w1_S2'))),
        'no 1a': picks_doc(stage1a=None),
        'not an arm': picks_doc(stage1b=dict(plain=dict(pick='dec-plain_l1_w1_S1'), start=dict(pick='x'), teacher=dict(pick='dec-teacher_l1_w1_S1'))),
    }
    for what, doc in bad.items():
        p.write_text(json.dumps(doc))
        with pytest.raises(AssertionError):
            T.picked_arms(p)
    p.write_text(json.dumps(picks_doc(stage1b=dict(plain=dict(pick='dec-plain_l1_w1_S1'), start=dict(pick='dec-start_l1_w1_S1'),
                                                   teacher=dict(pick='dec-teacher_l1_w1_S1_bal')))))   # balance may differ per form
    assert len(T.picked_arms(p)) == 4


# ------------------------------------------------------------------------------------------------ c* against the saved figures
def saved_files(tmp_path, monkeypatch, aspect='emotion', k=9, ami_k=0.31, ami_20=0.27, mapv=0.45):
    r17 = tmp_path / 'clu.json'
    r17.write_text(json.dumps([dict(id=f'SC3/{T.SET}/qwen/{aspect}/ami/{k}/teacher', cand=ami_k)]))
    r10 = tmp_path / 'held.json'
    r10.write_text(json.dumps(dict(results={'a': dict(status='ok', aspects={aspect: dict(final_ami=ami_20)}),
                                            'b': dict(status='ok', aspects={aspect: dict(final_ami=ami_20)})})))
    r20 = tmp_path / 'ret.json'
    r20.write_text(json.dumps(dict(aspects={aspect: dict(arms={T.CSTAR_ARM: dict(map=mapv)})})))
    monkeypatch.setattr(T.SC, 'R17_CLU', r17)
    monkeypatch.setattr(T, 'R10_HELD_RECORD', r10)
    monkeypatch.setattr(T.SC, 'ABLATION_RET', r20)


def test_cstar_reference_passes_when_c_star_equals_the_saved_figures(tmp_path, monkeypatch):
    saved_files(tmp_path, monkeypatch)
    out = T.cstar_reference('emotion', 9, 0.31 + 1e-8, 0.27, 0.45)
    assert all(c['ok'] for c in out.values()) and out['ami_20']['diff'] == pytest.approx(0.0)


@pytest.mark.parametrize('ami_k, ami_20, mapv', [(0.32, 0.27, 0.45), (0.31, 0.2701, 0.45), (0.31, 0.27, 0.4501)])
def test_cstar_reference_is_enforced(tmp_path, monkeypatch, ami_k, ami_20, mapv):
    """A c* here that differs from the saved c* (in any of the three) is refused: the changes would be against another comparator."""
    saved_files(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match='differs from the saved'):
        T.cstar_reference('emotion', 9, ami_k, ami_20, mapv)


# ------------------------------------------------------------------------------------------------ table(), end to end on synthetic exports
N_SYN = 150
DIM = 16
KS = dict(emotion=4, style=5, genre=3)
K_FIXED = 7                                   # not a class count of any aspect, so a cell fed the wrong partitions shows
DEC_ARMS = ['dec-plain_l1_w1_S1', 'dec-start_l1_w1_S1', 'dec-teacher_l1_w1_S1']


def synth_x(centers, y, noise, rng):
    x = centers[y] + noise * rng.normal(size=(len(y), DIM))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A tiny stand-in for the held set: 150 paintings, three aspects of 4, 5 and 3 classes, exports of c*, `dec-off` (the same arrays
    as c*) and three DEC arms (other arrays) under temporary roots that replace the real trees."""
    T._deps()
    rng = np.random.default_rng(7)
    names = np.array([f'p{i:04d}' for i in range(N_SYN)])
    labels = {a: rng.integers(0, k, N_SYN).astype(np.int64) for a, k in KS.items()}
    labels['genre'][:10] = -1
    r21, r10 = tmp_path / 'r21', tmp_path / 'r10'
    store, metas = {}, {}
    for a, k in KS.items():
        centers = rng.normal(size=(k, DIM))
        for s in SEEDS:
            y = np.where(labels[a] < 0, 0, labels[a])
            base = synth_x(centers, y, 1.2, rng)
            for root, arm, x in [(r10, T.CSTAR_ARM, base), (r21, 'dec-off', base)] + [
                    (r21, d, synth_x(centers, y, 0.6 + 0.2 * i, rng)) for i, d in enumerate(DEC_ARMS)]:
                path = Path(root) / 'held' / arm / f'{a}__s{s}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f'{arm}{a}{s}'.encode())
                store[(str(root), arm, a, s)] = x
                metas.setdefault((str(root), arm), {})[(a, s)] = dict(
                    arm=arm, aspect=a, seed=s, rows=N_SYN, list_sha256=SHA, image_sha256=IMG, path=str(path),
                    array_sha256=__import__('hashlib').sha256(x.tobytes()).hexdigest())
    monkeypatch.setattr(T, 'N_HELD', N_SYN)
    monkeypatch.setattr(T, 'FIXED_K', K_FIXED)
    monkeypatch.setattr(T, 'R21_FULL', r21)
    monkeypatch.setattr(T, 'R10_FULL', r10)
    monkeypatch.setattr(T, 'SCORES', tmp_path / 'scores')
    monkeypatch.setattr(T, 'N_AMI_DRAWS', 12)
    monkeypatch.setattr(T, 'N_MAP_DRAWS', 12)
    monkeypatch.setattr(T, 'held_sha', lambda: SHA)
    monkeypatch.setattr(T, 'read_metas', lambda root, arm: copy.deepcopy(metas[(str(root), arm)]))
    monkeypatch.setattr(T, 'load_held', lambda root, arm, a, s: (store[(str(root), arm, a, s)], metas[(str(root), arm)][(a, s)]))
    monkeypatch.setattr(T.rb_part2_report, 'held_labels', lambda: (names, SHA, labels))
    monkeypatch.setattr(T.ns_data, 'labels', lambda set_, split, aspect: labels[aspect].copy())
    for a, k in KS.items():
        monkeypatch.setitem(T.ns_paths.CLASS_COUNT, (T.SET, a), k)
    picks = tmp_path / 'picks.json'
    picks.write_text(json.dumps(picks_doc(stage1b=dict(zip(T.FORMS, [dict(pick=d) for d in DEC_ARMS])))))

    def fake_reference(aspect, k, ami_k, ami_20, map_val):             # the real one reads the saved figures: tested above
        return {key: dict(ours=v, saved=v, diff=0.0, ok=True) for key, v in
                (('ami_class_count', ami_k), ('ami_20', ami_20), ('map', map_val))}
    monkeypatch.setattr(T, 'cstar_reference', fake_reference)
    return dict(r21=r21, r10=r10, picks=picks, store=store, labels=labels, names=names, out=tmp_path / 'out', scores=tmp_path / 'scores')


def run_table(w, arms, out=None, aspects=ASPECTS, picks='default', root=None, cstar_root=None, name='stage2_table', **kw):
    return T.table(arms, root or w['r21'], cstar_root or w['r10'], out or w['out'], 12, 12, 2, aspects, name,
                   w['picks'] if picks == 'default' else picks, **kw)


def test_table_end_to_end_complete_cells_and_pairing(world):
    w = world
    arms = ['dec-off'] + DEC_ARMS
    doc = run_table(w, arms)
    assert doc['complete'] is True and doc['arms'] == arms
    assert doc['picks'] == dict(path=str(w['picks']), sha256=T._file_sha(w['picks']))
    assert doc['identical_to_cstar'] == {'dec-off': True, **{d: False for d in DEC_ARMS}}
    assert (w['out'] / 'stage2_table.json').is_file() and 'Identical to c*' in (w['out'] / 'stage2_table.md').read_text()
    from sklearn.metrics import adjusted_mutual_info_score as ami
    for aspect, k in KS.items():
        y = w['labels'][aspect]
        keep = y >= 0
        for arm in arms:
            res = doc['aspects'][aspect]['arms'][arm]
            for key, kk in (('ami_class_count', k), ('ami_20', K_FIXED)):
                want = {}
                for who, root, name in (('arm', w['r21'], arm), ('cstar', w['r10'], T.CSTAR_ARM)):
                    parts = [T.ns_metrics.partitions(w['store'][(str(root), name, aspect, s)], kk, T.SET) for s in SEEDS]
                    want[who] = float(np.mean([np.mean([ami(y[keep], p[keep]) for p in ps]) for ps in parts]))
                assert res[key]['arm'] == pytest.approx(want['arm'], abs=1e-12), (aspect, arm, key)
                assert res[key]['cstar'] == pytest.approx(want['cstar'], abs=1e-12), (aspect, arm, key)
                assert res[key]['change'] == pytest.approx(want['arm'] - want['cstar'], abs=1e-12)
                assert res[key]['cstar'] == doc['cstar_check'][aspect][key]['ours']
            assert res['map']['cstar'] == pytest.approx(doc['cstar_check'][aspect]['map']['ours'], abs=1e-12)
        off = doc['aspects'][aspect]['arms']['dec-off']
        assert all(off[key]['change'] == 0.0 and off[key]['word'] == 'no clear change' for key in ('ami_class_count', 'ami_20', 'map'))


def test_table_stops_when_a_cell_is_not_against_the_checked_cstar(world, monkeypatch):
    """F5: the c* inside each cell must be the c* that was checked against the saved figures."""
    real = T.ami_cell

    def wrong(arm_parts, cstar_parts, y, n_draws, workers):
        return dict(real(arm_parts, cstar_parts, y, n_draws, workers), cstar=0.123)
    monkeypatch.setattr(T, 'ami_cell', wrong)
    with pytest.raises(AssertionError, match="checked c"):
        run_table(world, ['dec-off'], aspects=('emotion',))


def test_table_is_complete_only_for_the_set_trees_all_aspects_full_draws_and_the_picked_arms(world, monkeypatch):
    """F1: every departure turns `complete` off (the record and the markdown say so)."""
    w = world
    full = ['dec-off'] + DEC_ARMS
    doc = run_table(w, ['dec-off'], aspects=('emotion',))                                # a few arms, one aspect: the record says so
    assert doc['complete'] is False and doc['identical_to_cstar'] == {'dec-off': True}
    assert json.loads((w['out'] / 'stage2_table.json').read_text())['complete'] is False
    assert 'Not the full table' in (w['out'] / 'stage2_table.md').read_text()
    ok = lambda **kw: T.is_complete(**{**dict(root=w['r21'], cstar_root=w['r10'], aspects=ASPECTS, arms=full, n_ami_draws=12,
                                              n_map_draws=12, picks_path=w['picks']), **kw})
    assert ok() is True
    assert ok(aspects=('emotion', 'style')) is False and ok(arms=['dec-off']) is False and ok(arms=full[::-1]) is False
    assert ok(arms=full + ['dec-extra']) is False and ok(picks_path=None) is False and ok(picks_path=w['out'] / 'nope.json') is False
    assert ok(n_ami_draws=11) is False and ok(n_map_draws=13) is False
    assert ok(root=Path('/elsewhere/r21')) is False and ok(cstar_root=Path('/elsewhere/r10')) is False
    bad = w['out'] / 'bad_picks.json'                                                    # a picks file that does not read as valid
    bad.write_text(json.dumps(picks_doc(stage1b=dict(plain=dict(pick='dec-plain_l1_w1_S1'), start=dict(pick='dec-plain_l1_w1_S1'),
                                                     teacher=dict(pick='dec-teacher_l1_w1_S1')))))
    assert ok(picks_path=bad) is False


def test_table_refuses_the_paper_facing_file_unless_complete(world):
    w = world
    with pytest.raises(AssertionError, match='refusing'):
        run_table(w, ['dec-off'], out=w['scores'])
    assert not (w['scores'] / 'stage2_table.json').exists() and not (w['scores'] / 'stage2_table.md').exists()
    with pytest.raises(AssertionError, match='refusing'):                                # also when the folder is reached by another path
        run_table(w, ['dec-off'], out=w['scores'] / '..' / 'scores')
    run_table(w, ['dec-off'], out=w['scores'], name='scratch', aspects=('emotion',))     # another stem is allowed
    assert (w['scores'] / 'scratch.json').is_file()
    doc = run_table(w, ['dec-off'] + DEC_ARMS, out=w['scores'])                          # the complete table is written
    assert doc['complete'] is True and (w['scores'] / 'stage2_table.json').is_file()
    assert json.loads((w['scores'] / 'stage2_table.json').read_text())['complete'] is True


def test_table_refuses_other_input_rows(world, monkeypatch):
    w = world
    real = T.read_metas

    def other(root, arm):
        m = real(root, arm)
        if arm == 'dec-off':
            m[('style', 2)]['image_sha256'] = 'f' * 64
        return m
    monkeypatch.setattr(T, 'read_metas', other)
    with pytest.raises(AssertionError, match='input rows'):
        run_table(w, ['dec-off'], aspects=('emotion',))


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
