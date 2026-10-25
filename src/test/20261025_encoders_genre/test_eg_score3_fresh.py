"""Stage iii fresh seeds (rule §3.3 "Fresh seeds", §4.3 C outcomes), on small synthetic inputs.

Seams (the ticket's list): the paired draw order, the C boundaries, the parity refusal, a failed fresh run.
"""
import json

import numpy as np
import pytest

import eg_paths
import eg_score3 as s3
import eg_variants as V
import rb_ami

PSHA = eg_paths.P_SHA256
SEEDS = (1, 2, 3)


# ------------------------------------------------------------------ the paired draw order
def literal_draws(n_labelled, n_draws):
    """The rule's sentence, written as the loop it describes: one default_rng(0); per b, I_b then S_b."""
    rng = np.random.default_rng(0)
    idx, seeds = [], []
    for _ in range(n_draws):
        idx.append(rng.integers(0, n_labelled, size=n_labelled))
        seeds.append(rng.choice([1, 2, 3], size=3, replace=True))
    return np.array(idx), np.array(seeds)


def test_draws_follow_the_rule_order_per_draw_index_then_seeds():
    idx, seeds = s3.fresh_draws(50, 7)
    want_idx, want_seeds = literal_draws(50, 7)
    assert np.array_equal(idx, want_idx) and np.array_equal(seeds, want_seeds)
    assert seeds.shape == (7, 3) and set(np.unique(seeds)) <= {1, 2, 3}


def test_draws_are_not_the_seed_zero_draws():
    """Interleaving S_b between the I_b moves every I_b after the first off 12-24's draws."""
    idx, _ = s3.fresh_draws(50, 4)
    old = rb_ami.draws(50, 4)
    assert np.array_equal(idx[0], old[0]) and not np.array_equal(idx[1], old[1])


def test_interval_is_the_paired_mean_over_the_seeds_drawn():
    """e_b = mean over S_b of A(v*, I_b, s) - A(R6-5, I_b, s): one S_b for both arms."""
    n = 5
    b = np.arange(n, dtype=float)
    boot_v = {1: b + 10, 2: b + 20, 3: b + 30}              # v* is 10, 20, 30 above R6-5 on seeds 1, 2, 3
    boot_r = {1: b, 2: b, 3: b}
    S = np.array([[1, 1, 1], [3, 3, 3], [1, 2, 3], [1, 1, 3], [2, 2, 2]])
    e_b = s3.paired_gains(boot_v, boot_r, S)
    assert np.allclose(e_b, [10, 30, 20, (10 + 10 + 30) / 3, 20])
    boot_r_other = {1: b, 2: b + 5, 3: b}                   # a different R6-5 in seed 2 only
    assert np.allclose(s3.paired_gains(boot_v, boot_r_other, S), [10, 30, (10 + 15 + 30) / 3, (10 + 10 + 30) / 3, 15])


def synthetic(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, size=n)
    y[rng.random(n) < 0.2] = -1
    informative = np.where(y >= 0, y, 0)

    def part(noise):
        return np.stack([np.where(rng.random(n) < noise, rng.integers(0, 5, size=n), informative)
                         for _ in range(3)]).astype(np.int64)
    return y.astype(np.int64), part


def test_aspect_gain_equals_a_literal_bootstrap_of_the_rule():
    y, part = synthetic()
    parts = {'V': {s: part(0.1) for s in SEEDS}, 'R6-5': {s: part(0.6) for s in SEEDS}}
    n_draws = 12
    got = s3.fresh_aspect(y, parts, 'V', 'R6-5', SEEDS, n_draws, workers=1)
    L = np.flatnonzero(y >= 0)
    idx, seeds = literal_draws(len(L), n_draws)
    e_b = []
    for b in range(n_draws):
        d = [rb_ami.score(y, parts['V'][s], L[idx[b]]) - rb_ami.score(y, parts['R6-5'][s], L[idx[b]]) for s in seeds[b]]
        e_b.append(np.mean(d))
    lo, hi = np.percentile(e_b, [2.5, 97.5])
    abar = {n: np.mean([rb_ami.score(y, parts[n][s], L) for s in SEEDS]) for n in parts}
    assert got['lo'] == pytest.approx(lo, abs=1e-12) and got['hi'] == pytest.approx(hi, abs=1e-12)
    assert got['e'] == pytest.approx(abar['V'] - abar['R6-5'], abs=1e-12)
    assert got['Abar']['V'] == pytest.approx(abar['V'], abs=1e-12) and got['A']['V'][2] == pytest.approx(
        rb_ami.score(y, parts['V'][2], L), abs=1e-12)
    assert got['e'] > 0 and got['lo'] > 0


# ------------------------------------------------------------------ the C boundaries
R = V.RANGES


def eff(e_emotion=0.0, e_style=0.0, e_genre=0.0, lo=0.01):
    return {'emotion': dict(e=e_emotion, lo=-1, hi=1), 'style': dict(e=e_style, lo=-1, hi=1),
            'genre': dict(e=e_genre, lo=lo, hi=1)}


def test_c_holds_at_exactly_the_range_with_the_lower_bound_above_zero():
    assert s3.c_outcome(eff(e_genre=R['genre'], lo=1e-9)) == 'C-HOLDS'
    assert s3.c_outcome(eff(e_genre=R['genre'], lo=0.0)) == 'C-QUAL'                  # lower bound must be > 0
    assert s3.c_outcome(eff(e_genre=np.nextafter(R['genre'], 0), lo=0.01)) == 'C-QUAL'
    assert s3.c_outcome(eff(e_genre=0.05, lo=0.02)) == 'C-HOLDS'


def test_c_guards_on_the_three_seed_means_are_greater_or_equal_minus_range():
    ok = dict(e_emotion=-R['emotion'], e_style=-R['style'])
    assert s3.c_outcome(eff(e_genre=0.05, **ok)) == 'C-HOLDS'
    assert s3.c_outcome(eff(e_genre=0.05, e_emotion=np.nextafter(-R['emotion'], -1))) == 'C-FAILS'
    assert s3.c_outcome(eff(e_genre=0.05, e_style=np.nextafter(-R['style'], -1))) == 'C-FAILS'
    assert s3.c_outcome(eff(e_genre=0.001, e_emotion=-0.0047)) == 'C-FAILS'          # a failed guard is never QUAL


def test_c_qual_needs_a_positive_genre_gain_and_c_fails_otherwise():
    assert s3.c_outcome(eff(e_genre=1e-9, lo=-0.01)) == 'C-QUAL'
    assert s3.c_outcome(eff(e_genre=0.0, lo=-0.01)) == 'C-FAILS'
    assert s3.c_outcome(eff(e_genre=-0.01, lo=-0.02)) == 'C-FAILS'


def test_genre_against_the_targets_compares_unrounded_values():
    t = V.T_G
    assert s3.genre_vs_targets(t)['reaches_t_g'] and s3.genre_vs_targets(t)['reaches_promising']
    g = s3.genre_vs_targets(np.nextafter(t, 0))
    assert not g['reaches_t_g'] and g['reaches_promising']
    p = t - R['genre']
    assert s3.genre_vs_targets(p)['reaches_promising'] is True
    assert s3.genre_vs_targets(np.nextafter(p, 0))['reaches_promising'] is False
    assert s3.genre_vs_targets(0.4989)['reaches_t_g'] is False                       # the rounded 0.4989 is below T_g


def test_reading_both_sentence():
    assert 'not repeated on fresh seeds' in s3.reading_both('III-GO', 'C-FAILS')
    assert 'not repeated on fresh seeds' in s3.reading_both('III-PROM', 'C-FAILS')
    assert 'repeat it' in s3.reading_both('III-GO', 'C-HOLDS') and 'not repeated' not in s3.reading_both('III-GO', 'C-HOLDS')
    for c in ('C-HOLDS', 'C-QUAL', 'C-FAILS'):                      # III-NONE: neither GO nor promising, whatever C says
        text = s3.reading_both('III-NONE', c)
        assert 'neither' in text and c in text and 'not repeated' not in text
    assert 'no C outcome' in s3.reading_both('III-GO', None)


# ------------------------------------------------------------------ files: parity, failed runs, the whole command
WINNER = 'GT-4'


def labels_ctx(n=240):
    ys = {a: synthetic(n, seed=i)[0] for i, a in enumerate(eg_paths.ASPECTS)}
    return dict(p_sha=PSHA, labels=ys, names=None, pick=None)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_score_all(out, winner=WINNER, fresh=True, outcome='III-PROM', smoke=True, n_draws=6):
    put_json(s3.stage_dir(out) / 'score_all.json', dict(
        stage='all', smoke=smoke, n_draws=n_draws, winner=winner, outcome=outcome, A_genre_winner=0.49,
        e_genre_winner=0.02, fresh_seeds=fresh, identity=dict(p_sha256=PSHA)))


def write_parity(out, ok=(True, True, True), cause=None):
    seeds = {str(s): {'pass': bool(o), 'run_status': 'ok' if o or cause is None else 'failed',
                      'run_cause': None if o or cause is None else cause,
                      'weights_equal': {a: bool(o) for a in eg_paths.ASPECTS}} for s, o in zip(SEEDS, ok)}
    put_json(s3.stage_dir(out) / 'fresh_parity.json', {'pass': all(ok), 'seeds': seeds, 'code_commit': 'abc',
                                                      'reference': 'K4'})


def write_run_json(out, variant, seed, status='ok', cause=None):
    put_json(eg_paths.iii_run(out, variant, seed) / 'run.json', dict(
        variant=variant, seed=seed, switches=V.switches(variant), status=status, cause=cause, p_sha256=PSHA,
        best_epoch={a: 4 for a in eg_paths.ASPECTS}))


def onehot_images(y, noise, seed, width=16):
    rng = np.random.default_rng(seed)
    cls = np.where(rng.random(len(y)) < noise, rng.integers(0, 8, size=len(y)), np.where(y >= 0, y, 0))
    x = np.zeros((len(y), width), np.float16)
    x[np.arange(len(y)), cls] = 1.0
    return x


def stub_read_run(monkeypatch, ctx, noise_of, statuses=None):
    """read_run replaced by a reader of small synthetic exports; `statuses` {(variant, seed): cause of a failed run}."""
    statuses = statuses or {}

    def read(outputs, variant, p_sha, declared_failed=None, seed=0):
        if (variant, seed) in statuses:
            return dict(status='failed', cause=statuses[(variant, seed)], best_epoch=None, run_sha256='f' * 64, img=None)
        img = {a: onehot_images(ctx['labels'][a], noise_of[variant], 1000 * seed + i)
               for i, a in enumerate(eg_paths.ASPECTS)}
        return dict(status='ok', cause=None, best_epoch={a: 4 for a in eg_paths.ASPECTS}, run_sha256=f'{seed}' * 64,
                    img=img)
    monkeypatch.setattr(s3, 'read_run', read)


def test_a_parity_failure_leaves_no_c_outcome_and_names_the_cause(tmp_path, monkeypatch):
    write_score_all(tmp_path)
    write_parity(tmp_path, ok=(True, False, True))
    monkeypatch.setattr(s3, 'read_run', lambda *a, **k: pytest.fail('no run is read after a parity failure'))
    rec = s3.run_fresh(tmp_path, 6, 1, smoke=True, ctx=labels_ctx())
    assert rec['c_outcome'] is None and rec['effects'] is None
    assert 'seed 2' in rec['no_outcome_cause'] and 'K4' in rec['no_outcome_cause']
    assert rec['parity']['pass'] is False and 'no C outcome' in rec['reading']
    assert json.loads((s3.stage_dir(tmp_path) / 'score_fresh.json').read_text())['c_outcome'] is None


def test_parity_failure_from_a_failed_r6_5_run_gives_its_cause(tmp_path):
    write_score_all(tmp_path)
    write_parity(tmp_path, ok=(True, True, False), cause='non-finite loss at epoch 9')
    rec = s3.run_fresh(tmp_path, 6, 1, smoke=True, ctx=labels_ctx())
    assert rec['c_outcome'] is None and 'non-finite loss at epoch 9' in rec['no_outcome_cause']


def test_fresh_refuses_without_a_trigger_or_a_parity_file(tmp_path):
    write_score_all(tmp_path, fresh=False)
    write_parity(tmp_path)
    with pytest.raises(AssertionError, match='not triggered'):
        s3.run_fresh(tmp_path, 6, 1, smoke=True, ctx=labels_ctx())
    write_score_all(tmp_path, winner=None, fresh=False, outcome='III-NONE')
    with pytest.raises(AssertionError, match='not triggered'):
        s3.run_fresh(tmp_path, 6, 1, smoke=True, ctx=labels_ctx())
    other = tmp_path / 'other'
    write_score_all(other)
    with pytest.raises(AssertionError, match='fresh_parity.json'):
        s3.run_fresh(other, 6, 1, smoke=True, ctx=labels_ctx())


def test_a_failed_fresh_run_leaves_no_c_outcome(tmp_path, monkeypatch):
    write_score_all(tmp_path)
    write_parity(tmp_path)
    ctx = labels_ctx()
    stub_read_run(monkeypatch, ctx, {WINNER: 0.05, 'R6-5': 0.7}, {(WINNER, 2): 'non-finite loss at epoch 3'})
    rec = s3.run_fresh(tmp_path, 6, 1, smoke=True, ctx=ctx)
    assert rec['c_outcome'] is None and rec['effects'] is None
    assert WINNER in rec['no_outcome_cause'] and 'seed 2' in rec['no_outcome_cause'] and 'non-finite' in rec['no_outcome_cause']
    assert rec['runs'][WINNER]['2']['status'] == 'failed' and rec['runs']['R6-5']['1']['status'] == 'ok'
    assert 'no C outcome' in rec['reading']


def test_read_run_reads_the_fresh_seed_it_is_asked_for(tmp_path):
    write_run_json(tmp_path, WINNER, 2, status='failed', cause='non-finite loss')
    r = s3.read_run(tmp_path, WINNER, PSHA, seed=2)
    assert r['status'] == 'failed' and r['img'] is None
    with pytest.raises(AssertionError):
        s3.read_run(tmp_path, WINNER, PSHA, seed=1)                     # no run.json for seed 1
    write_run_json(tmp_path, 'R6-5', 3)
    folder = eg_paths.iii_run(tmp_path, 'R6-5', 1)
    folder.mkdir(parents=True)
    (folder / 'run.json').write_text((eg_paths.iii_run(tmp_path, 'R6-5', 3) / 'run.json').read_text())
    with pytest.raises(AssertionError):                                 # a seed 3 record under seed 1
        s3.read_run(tmp_path, 'R6-5', PSHA, seed=1)


def test_other_variants_fresh_folders_stop_the_run(tmp_path):
    write_run_json(tmp_path, WINNER, 1)
    write_run_json(tmp_path, 'R6-5', 1)
    s3.check_fresh_dirs(tmp_path, WINNER)
    write_run_json(tmp_path, 'GR-2', 2)
    with pytest.raises(AssertionError, match='VG4'):
        s3.check_fresh_dirs(tmp_path, WINNER)


def test_the_whole_command_on_small_synthetic_runs(tmp_path, monkeypatch):
    write_score_all(tmp_path)
    write_parity(tmp_path)
    ctx = labels_ctx()
    stub_read_run(monkeypatch, ctx, {WINNER: 0.05, 'R6-5': 0.7})
    rec = s3.run_fresh(tmp_path, 6, 1, smoke=True, ctx=ctx)
    assert rec['c_outcome'] in ('C-HOLDS', 'C-QUAL', 'C-FAILS') and rec['seeds'] == [1, 2, 3]
    g = rec['effects']['genre']
    assert g['e'] == pytest.approx(g['Abar'][WINNER] - g['Abar']['R6-5']) and g['e'] > 0
    assert rec['genre_vs_targets']['A_bar'] == g['Abar'][WINNER] and rec['seed0_outcome'] == 'III-PROM'
    assert rec['reading'] and rec['runs'][WINNER]['1']['best_epoch']['genre'] == 4
    for v in (WINNER, 'R6-5'):                               # the cache layout of seed 0, one folder per fresh seed
        for s in SEEDS:
            for a in eg_paths.ASPECTS:
                base = s3.cache_dir(tmp_path, v, s)
                assert (base / f'{a}__partitions.npz').exists() and (base / f'{a}__boot.json').exists()
    monkeypatch.setattr(s3, 'fresh_boot_scores', lambda *a, **k: pytest.fail('bootstrap recomputed'))
    assert s3.run_fresh(tmp_path, 6, 1, smoke=True, ctx=ctx) == rec              # same inputs: kept, nothing redone
