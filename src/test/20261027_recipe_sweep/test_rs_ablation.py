"""The ablation scorer (rule §2.5, §3.5, §4.5; ticket 10): F's resolution from the stage files, the eight removal arms, the
development effects (the joint bootstrap with F in the reference's place), the held effects (one S_b shared by the arm and F),
a failed arm, and F = G's frozen sources. Everything runs on tiny arrays in tmp_path: no export of P and no held output is
read here (the real files are the controller's smoke and run)."""
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score as ami

import rs_ablation as A
import rs_paths
import rs_sheet as H
import rs_stage3 as T
import rs_sweep as S
import rb_pick

ASP = H.ASPECTS
R = H.RANGES
SEEDS = (1, 2, 3)
FULL = (0, 1, 2, 3, 4)
N_ROWS, N_DRAWS = 40, 12


# ------------------------------------------------------------------ the stage files F is read from
def put_stage3(outputs, outcome='PICK', pick='k6=0.0005', smoke=False, runs=None, n_draws=None):
    rec = dict(stage='stage3', smoke=smoke, outcome=outcome, pick=pick if outcome == 'PICK' else None, p_sha256=rs_paths.P_SHA256,
               n_draws=n_draws or (T.SMOKE_DRAWS if smoke else T.N_DRAWS), inputs=dict(runs=runs or {}), results={})
    S._json_dump(T.verdict_path(outputs), rec)
    return rec


def put_stage1(outputs, outcome, machine='local', smoke=False):
    S._json_dump(S.sweep_path(outputs, machine, 'stage1'), dict(stage='stage1', machine=machine, smoke=smoke, outcome=outcome))


def put_stage2(outputs, outcome, machine='local', smoke=False):
    S._json_dump(S.sweep_path(outputs, machine, 'stage2'), dict(stage='stage2', machine=machine, smoke=smoke, outcome=outcome))


def test_final_is_c_star_under_pick(tmp_path):
    put_stage3(tmp_path, 'PICK', 'k2=8+k6=0.0005')
    f = A.resolve_final(tmp_path, smoke=False)
    assert (f['name'], f['outcome'], f['pick'], f['run_key']) == ('k2=8+k6=0.0005', 'PICK', 'k2=8+k6=0.0005', 'k2=8+k6=0.0005')
    assert f['config'] == H.config({'k2': 8, 'k6': 0.0005}) and not f['is_g']
    assert f['stage3_sha256'] == S._file_sha256(T.verdict_path(tmp_path))


def test_final_is_g_under_keep_g(tmp_path):
    put_stage3(tmp_path, 'KEEP-G')
    f = A.resolve_final(tmp_path, smoke=False)
    assert (f['name'], f['outcome'], f['pick'], f['run_key'], f['is_g']) == ('G', 'KEEP-G', None, 'GS-0.02', True)
    assert f['config'] == H.config()


def test_final_is_g_after_sweep_none_without_a_stage_3_file(tmp_path):
    put_stage1(tmp_path, 'SWEEP-NONE')
    f = A.resolve_final(tmp_path, smoke=False)
    assert (f['name'], f['outcome'], f['is_g']) == ('G', 'SWEEP-NONE', True) and f['stage3_sha256'] is None
    assert f['source_sha256'] == S._file_sha256(S.sweep_path(tmp_path, 'local', 'stage1'))


def test_final_is_g_after_top_none_without_a_stage_3_file(tmp_path):
    put_stage1(tmp_path, 'GRID-2')
    put_stage2(tmp_path, 'TOP-NONE')
    f = A.resolve_final(tmp_path, smoke=False)
    assert (f['name'], f['outcome']) == ('G', 'TOP-NONE')
    assert f['source_sha256'] == S._file_sha256(S.sweep_path(tmp_path, 'local', 'stage2'))


def test_a_missing_stage_3_file_without_a_none_outcome_is_refused(tmp_path):
    with pytest.raises(AssertionError, match='no stage 3 verdict'):
        A.resolve_final(tmp_path, smoke=False)                           # nothing at all
    put_stage1(tmp_path, 'GRID-3')
    with pytest.raises(AssertionError, match='no stage 3 verdict'):
        A.resolve_final(tmp_path, smoke=False)                           # stage 1 found knobs, stage 2 not run
    put_stage2(tmp_path, 'TOP')
    with pytest.raises(AssertionError, match='stage 3'):
        A.resolve_final(tmp_path, smoke=False)                           # stage 2 gave TOP: stage 3 must have run


def test_stage_files_of_two_machines_need_a_machine_argument(tmp_path):
    put_stage1(tmp_path, 'SWEEP-NONE', 'local')
    put_stage1(tmp_path, 'SWEEP-NONE', 'node402')
    with pytest.raises(AssertionError, match='--machine'):
        A.resolve_final(tmp_path, smoke=False)
    assert A.resolve_final(tmp_path, smoke=False, machine='node402')['machine'] == 'node402'


def test_final_must_equal_the_stage_files(tmp_path):
    put_stage3(tmp_path, 'PICK', 'k6=0.0005')
    f = A.resolve_final(tmp_path, smoke=False)
    A.check_final(f, 'k6=0.0005')
    with pytest.raises(AssertionError, match=r"--final 'G'.*'k6=0.0005'"):
        A.check_final(f, 'G')
    put_stage3(tmp_path / 'b', 'KEEP-G')
    with pytest.raises(AssertionError, match='--final'):
        A.check_final(A.resolve_final(tmp_path / 'b', smoke=False), 'k6=0.0005')
    A.check_final(A.resolve_final(tmp_path / 'b', smoke=False), 'G')


def test_the_stage_files_must_be_of_the_same_kind_as_the_run(tmp_path):
    put_stage3(tmp_path, 'PICK', smoke=True)
    with pytest.raises(AssertionError, match='smoke'):
        A.resolve_final(tmp_path, smoke=False)
    put_stage1(tmp_path / 'b', 'SWEEP-NONE', smoke=True)
    with pytest.raises(AssertionError, match='smoke'):
        A.resolve_final(tmp_path / 'b', smoke=False)


def test_a_pick_that_is_not_a_sheet_name_is_refused(tmp_path):
    put_stage3(tmp_path, 'PICK', 'k6=0.0005000')
    with pytest.raises(AssertionError):
        A.resolve_final(tmp_path, smoke=False)


def test_held_mode_needs_stage_4_to_name_c_star(tmp_path):
    f_pick = dict(A.resolve_final(put_and(tmp_path, 'PICK'), smoke=False))
    with pytest.raises(AssertionError, match='stage4'):
        A.require_stage4_candidate(tmp_path, f_pick)                     # file missing
    path = tmp_path / 'stage4' / 'verdict.json'
    S._json_dump(path, dict(candidate='k2=8'))
    with pytest.raises(AssertionError, match='k2=8'):
        A.require_stage4_candidate(tmp_path, f_pick)
    S._json_dump(path, dict(candidate='k6=0.0005'))
    assert A.require_stage4_candidate(tmp_path, f_pick) == S._file_sha256(path)
    put_stage3(tmp_path / 'g', 'KEEP-G')
    assert A.require_stage4_candidate(tmp_path, A.resolve_final(tmp_path / 'g', smoke=False)) is None   # F = G: G's own files


def put_and(outputs, outcome):
    put_stage3(outputs, outcome)
    return outputs


# ------------------------------------------------------------------ the eight removal arms
@pytest.mark.parametrize('changes, base', [({}, 'G'), ({'k6': 0.0005}, 'k6=0.0005'), ({'k2': 8, 'k6': 0.0005}, 'k2=8+k6=0.0005')])
def test_each_arm_is_f_with_one_removal_and_nothing_retuned(changes, base):
    cfg = H.config(changes)
    final = dict(name=base, config=cfg)
    arms = A.arm_names(final)
    assert list(arms) == list(H.REMOVALS) and len(arms) == 8
    for x, name in arms.items():
        assert name == f'{base}-no_{x}'
        assert H.parse(name) == A.arm_config(final, x) == dict(cfg, remove=(x,))


def test_a_failing_graph_check_stops_exactly_the_two_graph_removals(tmp_path):
    assert all(A.graph_gate_cause(tmp_path, x) is None for x in H.REMOVALS)          # no gate file: the arms' own files decide
    S._json_dump(tmp_path / 'gates' / 'ablation_graphs.json', dict(name='ablation_graphs', **{'pass': False}, items=dict(
        default_view={'pass': False}, uncorrected={'pass': True})))
    assert 'default_view' in A.graph_gate_cause(tmp_path, 'instructed_graphs')
    assert A.graph_gate_cause(tmp_path, 'content_correction') is None
    assert all(A.graph_gate_cause(tmp_path, x) is None for x in H.REMOVALS if x != 'instructed_graphs')


# ------------------------------------------------------------------ development: F's runs
def fake_part(rng):
    return rng.integers(0, 4, (3, N_ROWS)).astype(np.int64)


def fake_run(rng, status='ok', cause=None, tag='r'):
    if status == 'failed':
        return dict(status='failed', cause=cause, img=None, run_sha256=tag)
    return dict(status='ok', cause=None, img={a: fake_part(rng) for a in ASP}, run_sha256=tag)


def verdict_inputs(name, runs):
    return {name: {str(s): dict(run_sha256=r['run_sha256'], img_sha256={a: rb_pick.array_sha256(r['img'][a]) for a in ASP})
                   for s, r in runs.items()}}


def pick_final(tmp_path, runs, **inputs_edit):
    inputs = verdict_inputs('k6=0.0005', runs)
    for key, value in inputs_edit.items():
        inputs['k6=0.0005']['1'][key] = value
    put_stage3(tmp_path, 'PICK', 'k6=0.0005', runs=inputs)
    return A.resolve_final(tmp_path, smoke=False)


def test_f_runs_under_pick_are_stage_3s_and_checked_against_its_verdict(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    runs = {s: fake_run(rng, tag=f'sha{s}') for s in SEEDS}
    calls = []
    monkeypatch.setattr(T, 'read_config_runs', lambda outputs, names, smoke: calls.append((names, smoke)) or {names[0]: runs})
    monkeypatch.setattr(A.T, 'require_v3a', lambda *a: pytest.fail('stage 3 ran with a gate; F = c* needs no G'))
    got = A.read_final_dev_runs(tmp_path, pick_final(tmp_path, runs), smoke=False)
    assert got == runs and calls == [(['k6=0.0005'], False)]


def test_f_run_that_is_not_the_one_stage_3_scored_is_refused(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    runs = {s: fake_run(rng, tag=f'sha{s}') for s in SEEDS}
    monkeypatch.setattr(T, 'read_config_runs', lambda outputs, names, smoke: {names[0]: runs})
    with pytest.raises(AssertionError, match='s1.*run.json'):
        A.read_final_dev_runs(tmp_path, pick_final(tmp_path, runs, run_sha256='other'), smoke=False)
    with pytest.raises(AssertionError, match='s1 emotion'):
        A.read_final_dev_runs(tmp_path, pick_final(tmp_path, runs, img_sha256={'emotion': 'other', 'style': 'x', 'genre': 'y'}), smoke=False)


def test_f_runs_under_pick_must_be_ok(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    runs = {1: fake_run(rng), 2: fake_run(rng, 'failed', 'nonfinite'), 3: fake_run(rng)}
    monkeypatch.setattr(T, 'read_config_runs', lambda outputs, names, smoke: {names[0]: runs})
    put_stage3(tmp_path, 'PICK', 'k6=0.0005', runs={})
    with pytest.raises(AssertionError, match='failed'):
        A.read_final_dev_runs(tmp_path, A.resolve_final(tmp_path, smoke=False), smoke=False)


def g_final(tmp_path, outcome='KEEP-G'):
    put_stage3(tmp_path, outcome)
    return A.resolve_final(tmp_path, smoke=False)


def test_f_equals_g_reads_12_27s_runs_only_through_stage_3s_hash_bound_reader(tmp_path, monkeypatch):
    gate = dict(g_exports='the-gate')
    calls = []

    def sentinel(outputs, smoke):
        calls.append(('gate', smoke))
        return gate
    monkeypatch.setattr(A.T, 'require_v3a', sentinel)
    monkeypatch.setattr(A.T, 'read_g_runs', lambda g: calls.append(('g', g)) or 'G-RUNS')
    monkeypatch.setattr(A.T, 'read_frozen_run', lambda *a: pytest.fail('G was read around read_g_runs'))
    monkeypatch.setattr(A.T, 'read_config_runs', lambda *a: pytest.fail('G has no run folder of this round'))
    assert A.read_final_dev_runs(tmp_path, g_final(tmp_path), smoke=False) == 'G-RUNS'
    assert calls == [('gate', False), ('g', gate)]
    assert T.g_folder(2) == rs_paths.RG_TRAIN / 'GS-0.02' / 's2'           # the folders read_g_runs reads: 12-27's seeds 1 to 3


def test_f_equals_g_after_sweep_none_uses_the_same_sources(tmp_path, monkeypatch):
    put_stage1(tmp_path, 'SWEEP-NONE')
    monkeypatch.setattr(A.T, 'require_v3a', lambda outputs, smoke: 'gate')
    monkeypatch.setattr(A.T, 'read_g_runs', lambda g: {'from': g})
    assert A.read_final_dev_runs(tmp_path, A.resolve_final(tmp_path, smoke=False), smoke=False) == {'from': 'gate'}


# ------------------------------------------------------------------ development: the arms and the scoring
def put_run(outputs, name, seed, **run):
    folder = rs_paths.run_dir(outputs, name, seed)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = H.parse(name)
    rec_ = dict(name=name, config=json.loads(json.dumps(cfg)), seed=seed, status='ok', cause=None, machine='local',
                p_sha256=rs_paths.P_SHA256, best_epoch=dict(emotion=3, style=2, genre=4), smoke=False)
    rec_.update(run)
    (folder / 'run.json').write_text(json.dumps(rec_))


def failed_run(outputs, name, seed):
    put_run(outputs, name, seed, status='failed', cause='non-finite loss in the genre head', failure_kind='nonfinite_loss')


def test_a_removal_arm_with_a_failed_seed_is_failed_with_the_cause_and_its_exports_are_not_read(tmp_path, monkeypatch):
    put_stage3(tmp_path, 'KEEP-G')
    final = A.resolve_final(tmp_path, smoke=False)
    for s in SEEDS:
        failed_run(tmp_path, 'G-no_buddy', s) if s == 2 else put_run(tmp_path, 'G-no_buddy', s)
    monkeypatch.setattr(A.S, 'read_exports', lambda folder: pytest.fail(f'exports of a failed arm read: {folder}'))
    arms = A.dev_arms(tmp_path, final, ['buddy'], smoke=False)
    assert arms['buddy']['name'] == 'G-no_buddy' and arms['buddy']['runs'] is None
    assert arms['buddy']['cause'] == 'seed 2: non-finite loss in the genre head'


def test_a_removal_arm_that_is_ok_on_all_seeds_is_read_through_stage_sweeps_run_reader(tmp_path, monkeypatch):
    put_stage3(tmp_path, 'KEEP-G')
    final = A.resolve_final(tmp_path, smoke=False)
    for s in SEEDS:
        put_run(tmp_path, 'G-no_buddy', s)
    calls = []
    monkeypatch.setattr(A.S, 'read_run', lambda outputs, name, seed, machine, smoke: calls.append((name, seed, machine, smoke)) or dict(status='ok', img={}))
    arms = A.dev_arms(tmp_path, final, ['buddy'], smoke=False)
    assert arms['buddy']['cause'] is None and sorted(arms['buddy']['runs']) == [1, 2, 3]
    assert calls == [('G-no_buddy', s, 'local', False) for s in SEEDS]


def test_an_unfinished_removal_arm_is_refused_not_listed_as_failed(tmp_path):
    put_stage3(tmp_path, 'KEEP-G')
    final = A.resolve_final(tmp_path, smoke=False)
    with pytest.raises(AssertionError, match='G-no_buddy'):
        A.dev_arms(tmp_path, final, ['buddy'], smoke=False)


def test_a_graph_removal_whose_check_failed_is_failed_and_its_runs_are_not_looked_for(tmp_path):
    put_stage3(tmp_path, 'KEEP-G')
    S._json_dump(tmp_path / 'gates' / 'ablation_graphs.json', dict(name='ablation_graphs', **{'pass': False}, items=dict(
        default_view={'pass': False}, uncorrected={'pass': True})))
    arms = A.dev_arms(tmp_path, A.resolve_final(tmp_path, smoke=False), ['instructed_graphs'], smoke=False)
    assert arms['instructed_graphs']['runs'] is None and 'default_view' in arms['instructed_graphs']['cause']


def tiny_ctx(seed=0):
    rng = np.random.default_rng(100 + seed)
    labels = dict(emotion=rng.integers(0, 4, N_ROWS), style=rng.integers(0, 5, N_ROWS), genre=rng.integers(0, 3, N_ROWS))
    labels['genre'][[1, 4, 5, 9, 20, 21, 30]] = -1
    return dict(p_sha='p' * 64, labels=labels)


def stub_partitions(monkeypatch):
    """`fixed_partitions` of the stub runs: the 'export' is the partition array itself."""
    calls = []

    def fixed(folder, name, aspect, img, p_sha, seed):
        calls.append((str(folder), name, aspect, seed))
        return img
    monkeypatch.setattr(A.S, 'fixed_partitions', fixed)
    return calls


def two_arm_world(tmp_path, monkeypatch, outcome='KEEP-G', smoke=False):
    rng = np.random.default_rng(3)
    put_stage3(tmp_path, outcome, smoke=smoke)
    final = A.resolve_final(tmp_path, smoke=smoke)
    f_runs = {s: fake_run(rng) for s in SEEDS}
    arms = {'buddy': dict(name='G-no_buddy', cause=None, runs={s: fake_run(rng) for s in SEEDS}),
            'direct': dict(name='G-no_direct', cause=None, runs={s: fake_run(rng) for s in SEEDS})}
    return final, f_runs, arms


def test_dev_effect_is_the_three_seed_mean_difference_with_f_in_the_reference_place(tmp_path, monkeypatch):
    stub_partitions(monkeypatch)
    final, f_runs, arms = two_arm_world(tmp_path, monkeypatch)
    ctx = tiny_ctx()
    out = A.score_dev(tmp_path, final, f_runs, arms, ctx, N_DRAWS, 1)
    assert list(out) == ['buddy', 'direct']
    for x, arm in arms.items():
        rec = out[x]
        assert rec['status'] == 'ok' and rec['cause'] is None and rec['arm'] == arm['name']
        parts = {a: {arm['name']: {s: arm['runs'][s]['img'][a] for s in SEEDS}, 'GS-0.02': {s: f_runs[s]['img'][a] for s in SEEDS}}
                 for a in ASP}
        want = T.joint_bootstrap(ctx['labels'], parts, arm['name'], 'GS-0.02', SEEDS, N_DRAWS, workers=1)    # own default_rng(0) each
        for a in ASP:
            r, w = rec['aspects'][a], want['effects'][a]
            assert r['effect'] == w['e'] and r['lo'] == w['lo'] and r['hi'] == w['hi']
            assert r['arm_ami'] == w['Abar'][arm['name']] and r['final_ami'] == w['Abar']['GS-0.02']
            assert r['effect'] == pytest.approx(r['arm_ami'] - r['final_ami'], abs=1e-12)
            assert r['arm_seed_ami'] == {str(s): w['A'][arm['name']][s] for s in SEEDS}
            assert r['final_seed_ami'] == {str(s): w['A']['GS-0.02'][s] for s in SEEDS}
        assert rec['M'] == want['M']


def test_dev_arm_equal_to_f_gives_exactly_zero_which_needs_the_same_i_b_and_s_b(tmp_path, monkeypatch):
    stub_partitions(monkeypatch)
    final, f_runs, arms = two_arm_world(tmp_path, monkeypatch)
    arms['buddy']['runs'] = {s: dict(f_runs[s], img={a: f_runs[s]['img'][a].copy() for a in ASP}) for s in SEEDS}
    out = A.score_dev(tmp_path, final, f_runs, arms, tiny_ctx(), N_DRAWS, 1)
    assert all(out['buddy']['aspects'][a][k] == 0 for a in ASP for k in ('effect', 'lo', 'hi'))
    assert out['direct']['aspects']['emotion']['hi'] != 0


def test_dev_failed_arm_gets_no_effect_and_the_others_go_on(tmp_path, monkeypatch):
    stub_partitions(monkeypatch)
    final, f_runs, arms = two_arm_world(tmp_path, monkeypatch)
    arms['buddy'] = dict(name='G-no_buddy', cause='seed 2: non-finite loss in the genre head', runs=None)
    boot_calls = []
    real = T.joint_bootstrap

    def spy(labels, parts, arm, ref, *a, **kw):
        boot_calls.append((arm, ref))
        return real(labels, parts, arm, ref, *a, **kw)
    monkeypatch.setattr(A.T, 'joint_bootstrap', spy)
    out = A.score_dev(tmp_path, final, f_runs, arms, tiny_ctx(), N_DRAWS, 1)
    assert out['buddy'] == dict(arm='G-no_buddy', status='failed', cause='seed 2: non-finite loss in the genre head', aspects=None, M=None)
    assert out['direct']['status'] == 'ok' and boot_calls == [('G-no_direct', 'GS-0.02')]
    assert A.failed_list(out) == [dict(removal='buddy', arm='G-no_buddy', cause='seed 2: non-finite loss in the genre head')]


def test_f_partitions_are_stage_3s_when_they_exist_and_every_other_run_is_in_the_ablation_cache(tmp_path):
    f = T.cache_root(tmp_path) / 'GS-0.02' / 's1' / 'emotion__partitions.npz'
    f.parent.mkdir(parents=True)
    f.write_bytes(b'x')
    assert A.part_folder(tmp_path, 'GS-0.02', 1, 'emotion', shared='GS-0.02') == f.parent
    assert A.part_folder(tmp_path, 'GS-0.02', 2, 'emotion', shared='GS-0.02') == A.cache_root(tmp_path) / 'GS-0.02' / 's2'
    assert A.part_folder(tmp_path, 'G-no_buddy', 1, 'emotion', shared='GS-0.02') == A.cache_root(tmp_path) / 'G-no_buddy' / 's1'
    assert A.cache_root(tmp_path) == tmp_path / 'ablation' / 'cache' and not A.cache_root(tmp_path).is_relative_to(T.cache_root(tmp_path))


def test_dev_f_scores_are_read_from_stage_3_and_new_ones_go_to_the_ablation_cache(tmp_path, monkeypatch):
    stub_partitions(monkeypatch)
    (T.cache_root(tmp_path) / 'GS-0.02').mkdir(parents=True)
    final, f_runs, arms = two_arm_world(tmp_path, monkeypatch)
    ctx = tiny_ctx()
    labels_sha = {a: rb_pick.array_sha256(ctx['labels'][a]) for a in ASP}
    img_sha = {a: {s: rb_pick.array_sha256(f_runs[s]['img'][a]) for s in SEEDS} for a in ASP}
    I, _ = T.joint_draws(N_ROWS, N_DRAWS, SEEDS)
    seen = []
    real = T.boot_scores
    for a in ASP:                                                        # what stage 3 / V3a left for G
        for s in SEEDS:
            boots = T.Boots(T.cache_root(tmp_path), a, s, ctx['p_sha'], labels_sha[a], {'GS-0.02': img_sha[a][s]}, N_DRAWS)
            boots.put('GS-0.02', f'P-n{N_ROWS}', real(ctx['labels'][a], {'GS-0.02': f_runs[s]['img'][a]}, I, 1)['GS-0.02'])

    def spy(labels, parts, I_, workers):
        seen.append(sorted(parts))
        return real(labels, parts, I_, workers)
    monkeypatch.setattr(A.T, 'boot_scores', spy)
    out = A.score_dev(tmp_path, final, f_runs, arms, ctx, N_DRAWS, 1)
    assert seen and all(names == ['G-no_buddy'] or names == ['G-no_direct'] for names in seen)      # F's scores were not recomputed
    assert list(A.cache_root(tmp_path).glob('G-no_buddy/s*/*__boot_*.npz')) and not list(A.cache_root(tmp_path).glob('GS-0.02/**/*boot*'))
    # and the numbers are those of a run without any cache
    want = T.joint_bootstrap(ctx['labels'], {a: {'G-no_buddy': {s: arms['buddy']['runs'][s]['img'][a] for s in SEEDS},
                                                 'GS-0.02': {s: f_runs[s]['img'][a] for s in SEEDS}} for a in ASP},
                             'G-no_buddy', 'GS-0.02', SEEDS, N_DRAWS, workers=1)
    assert out['buddy']['aspects']['genre']['lo'] == want['effects']['genre']['lo']


# ------------------------------------------------------------------ held: the draws and the statistic
def test_held_draws_are_i_then_one_s_over_seeds_0_to_4_from_one_generator():
    draws = A.held_draws(30, 5)
    rng = np.random.default_rng(0)
    assert len(draws) == 5
    for rows, seeds in draws:
        assert np.array_equal(rows, rng.integers(0, 30, size=30))
        s = rng.choice(list(FULL), size=5, replace=True)
        assert np.array_equal(seeds['X'], s) and np.array_equal(seeds['F'], s)    # one S_b, shared by x and F
        assert seeds['X'] is seeds['F'] or np.array_equal(seeds['X'], seeds['F'])


def test_held_draws_on_a_tiny_case_are_pinned():
    (r1, s1), (r2, s2) = A.held_draws(5, 2)
    assert r1.tolist() == [4, 3, 2, 1, 1] and r2.tolist() != r1.tolist()
    rng = np.random.default_rng(0)
    rng.integers(0, 5, size=5)
    assert s1['X'].tolist() == rng.choice([0, 1, 2, 3, 4], size=5, replace=True).tolist()
    assert set(np.unique(s1['X'])) <= set(FULL)


def test_every_aspect_draws_from_its_own_generator_and_gets_the_same_numbers():
    a, b = A.held_draws(30, 4), A.held_draws(30, 4)
    for (r1, s1), (r2, s2) in zip(a, b):
        assert np.array_equal(r1, r2) and np.array_equal(s1['X'], s2['X'])


def held_tiny(seed=0, n=60):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    y[[3, 7, 8, 20]] = -1
    fx = {s: {'r20': rng.integers(0, 4, (3, n))} for s in FULL}
    ff = {s: {'r20': rng.integers(0, 4, (3, n))} for s in FULL}
    return y, fx, ff


def brute_held(y, fx, ff, n_draws):
    rng = np.random.default_rng(0)
    n = len(y)
    d = []

    def A_(fixed, rows, seeds):
        yy = y[rows]
        m = yy >= 0
        return np.mean([np.mean([ami(yy[m], p[rows][m]) for p in fixed[s]['r20']]) for s in seeds])
    for _ in range(n_draws):
        rows = rng.integers(0, n, size=n)
        sb = rng.choice(list(FULL), size=5, replace=True)
        d.append(A_(fx, rows, sb) - A_(ff, rows, sb))
    point_x, point_f = A_(fx, np.arange(n), FULL), A_(ff, np.arange(n), FULL)
    return point_x, point_f, np.percentile(d, [2.5, 97.5])


def test_held_effect_equals_a_brute_force_loop_with_one_shared_s_b():
    y, fx, ff = held_tiny()
    out = A.held_effect(fx, ff, y, 10, workers=1)
    ax, af, (lo, hi) = brute_held(y, fx, ff, 10)
    assert out['arm_ami'] == pytest.approx(ax, abs=1e-12) and out['final_ami'] == pytest.approx(af, abs=1e-12)
    assert out['effect'] == pytest.approx(ax - af, abs=1e-12)                # A_x(I_0, S_0) - A_F(I_0, S_0)
    assert out['lo'] == pytest.approx(lo, abs=1e-12) and out['hi'] == pytest.approx(hi, abs=1e-12)
    assert sorted(out['arm_seed_ami']) == sorted(out['final_seed_ami']) == ['0', '1', '2', '3', '4']
    assert np.mean(list(out['arm_seed_ami'].values())) == pytest.approx(ax, abs=1e-12)


def test_held_interval_is_exactly_zero_for_an_arm_equal_to_f_because_s_b_is_shared():
    y, fx, _ = held_tiny(1)
    same = {s: {'r20': fx[s]['r20'].copy()} for s in FULL}
    out = A.held_effect(fx, same, y, 10, workers=1)
    assert (out['effect'], out['lo'], out['hi']) == (0, 0, 0)               # independent S_b for x and F would not give 0 here


def test_held_workers_do_not_change_the_numbers():
    y, fx, ff = held_tiny(2)
    one, two = A.held_effect(fx, ff, y, 10, workers=1), A.held_effect(fx, ff, y, 10, workers=2)
    assert one == two


# ------------------------------------------------------------------ held: F's sources, failed arms
def test_f_equals_g_held_outputs_are_12_27s_frozen_ones(tmp_path):
    root, arm = A.held_final_source(tmp_path, g_final(tmp_path))
    assert root == rs_paths.RG_FULL / 'held' / 'GS-0.02' and arm == 'GS-0.02'
    put_stage1(tmp_path / 'n', 'SWEEP-NONE')
    assert A.held_final_source(tmp_path / 'n', A.resolve_final(tmp_path / 'n', smoke=False)) == (rs_paths.RG_FULL / 'held' / 'GS-0.02', 'GS-0.02')


def test_f_equals_c_star_held_outputs_are_stage_4s(tmp_path):
    put_stage3(tmp_path, 'PICK', 'k6=0.0005')
    root, arm = A.held_final_source(tmp_path, A.resolve_final(tmp_path, smoke=False))
    assert root == rs_paths.full_root(tmp_path) / 'held' / 'k6=0.0005' and arm == 'k6=0.0005'


def full_dirs(outputs, name):
    train = rs_paths.full_root(outputs) / 'train' / name
    held = rs_paths.full_root(outputs) / 'held' / name
    train.mkdir(parents=True, exist_ok=True)
    held.mkdir(parents=True, exist_ok=True)
    return train, held


def test_held_arm_failure_by_a_non_finite_loss_a_failed_run_record_or_a_second_vc3_failure(tmp_path):
    assert A.held_arm_failure(tmp_path, 'G-no_buddy') is None                  # nothing recorded: not failed (the outputs decide)
    train, held = full_dirs(tmp_path, 'G-no_buddy')
    (train / 'failed__s3.json').write_text(json.dumps(dict(status='failed', failure_kind='nonfinite_loss', cause='non-finite loss in the style head')))
    assert A.held_arm_failure(tmp_path, 'G-no_buddy') == 'seed 3: non-finite loss in the style head'
    train, held = full_dirs(tmp_path, 'G-no_direct')
    (train / 'run__s1.json').write_text(json.dumps(dict(status='failed', cause='non-finite loss in the genre head')))
    (train / 'run__s0.json').write_text(json.dumps(dict(status='ok', cause=None)))
    assert A.held_arm_failure(tmp_path, 'G-no_direct') == 'seed 1: non-finite loss in the genre head'
    train, held = full_dirs(tmp_path, 'G-no_sharp_row')
    (held / 'failed__genre__s2.json').write_text(json.dumps(dict(failure_kind='vc3', problems=['nonfinite rows', 'nonfinite rows'])))
    cause = A.held_arm_failure(tmp_path, 'G-no_sharp_row')
    assert cause.startswith('genre seed 2:') and 'VC3' in cause and 'nonfinite rows' in cause


def test_held_arm_failure_lists_every_failed_seed(tmp_path):
    train, _ = full_dirs(tmp_path, 'G-no_buddy')
    for s, aspect in ((1, 'emotion'), (4, 'genre')):
        (train / f'failed__s{s}.json').write_text(json.dumps(dict(cause=f'non-finite loss in the {aspect} head')))
    assert A.held_arm_failure(tmp_path, 'G-no_buddy') == 'seed 1: non-finite loss in the emotion head; seed 4: non-finite loss in the genre head'


def held_world(tmp_path, monkeypatch, failed=(), smoke=False):
    """F = G (frozen sources stubbed), two arms; partitions are the stub rows themselves."""
    rng = np.random.default_rng(5)
    n = 60
    labels = {a: rng.integers(0, 4, n) for a in ASP}
    labels['genre'][[2, 3, 9]] = -1
    parts = {}

    def held_partitions(outputs, root, arm, aspect, seed, sha):
        p = parts.setdefault((arm, aspect, seed), rng.integers(0, 4, (3, n)))
        return p, dict(path=f'{arm}/{aspect}/{seed}', output_sha256='0' * 64, vc3_attempts=1)
    monkeypatch.setattr(A, 'held_partitions', held_partitions)
    put_stage3(tmp_path, 'KEEP-G', smoke=smoke)
    final = A.resolve_final(tmp_path, smoke=smoke)
    arms = {x: dict(name=f'G-no_{x}', cause=('seed 4: non-finite loss in the genre head' if x in failed else None)) for x in ('buddy', 'direct')}
    return final, arms, labels, parts


def test_score_held_effects_per_aspect_and_a_failed_arm_gets_none(tmp_path, monkeypatch):
    final, arms, labels, parts = held_world(tmp_path, monkeypatch, failed=('buddy',))
    results, sources = A.score_held(tmp_path, final, arms, labels, 'h' * 64, 6, 1)
    assert results['buddy'] == dict(arm='G-no_buddy', status='failed', cause='seed 4: non-finite loss in the genre head', aspects=None)
    rec = results['direct']
    assert rec['status'] == 'ok' and rec['cause'] is None and list(rec['aspects']) == list(ASP)
    for a in ASP:
        fx = {s: {'r20': parts[('G-no_direct', a, s)]} for s in FULL}
        ff = {s: {'r20': parts[('GS-0.02', a, s)]} for s in FULL}
        want = A.held_effect(fx, ff, labels[a], 6, 1)
        assert rec['aspects'][a] == want
        assert set(want) == {'effect', 'lo', 'hi', 'arm_ami', 'final_ami', 'arm_seed_ami', 'final_seed_ami'}
    assert ('G-no_buddy', 'genre', 0) not in parts                          # a failed arm's outputs are never partitioned
    assert set(sources) == {'GS-0.02', 'G-no_direct'} and sorted(sources['GS-0.02']) == sorted(ASP)


def test_unfinished_held_arm_is_refused(tmp_path):
    with pytest.raises(AssertionError, match='G-no_buddy.*not finished'):
        A.require_held_outputs(tmp_path, 'G-no_buddy')


# ------------------------------------------------------------------ the records and the CLI
def test_failed_list_carries_the_removal_the_arm_and_the_cause():
    results = {'buddy': dict(arm='G-no_buddy', status='failed', cause='c'), 'direct': dict(arm='G-no_direct', status='ok', cause=None)}
    assert A.failed_list(results) == [dict(removal='buddy', arm='G-no_buddy', cause='c')]


def test_dev_record_keys_and_the_file_is_kept_on_a_rerun(tmp_path, monkeypatch):
    stub_partitions(monkeypatch)
    final, f_runs, arms = two_arm_world(tmp_path, monkeypatch, smoke=True)
    S._json_dump(tmp_path / 'gates' / 'v3a.json', dict(stage='v3a'))
    monkeypatch.setattr(A.E, 'load_context', lambda: tiny_ctx())
    monkeypatch.setattr(A, 'read_final_dev_runs', lambda outputs, f, smoke: f_runs)
    monkeypatch.setattr(A, 'dev_arms', lambda outputs, f, removals, smoke: {x: arms[x] for x in removals})
    rec = A.run_dev(tmp_path, 'G', smoke=True, workers=1, removals=['buddy', 'direct'])
    assert {'stage', 'smoke', 'written', 'rule_commit', 'ranges', 'seeds', 'n_draws', 'draw_mode', 'final', 'removals', 'results', 'failed',
            'inputs', 'p_sha256', 'code_sha256'} <= set(rec)
    assert rec['stage'] == 'ablation_dev' and rec['seeds'] == [1, 2, 3] and rec['n_draws'] == T.SMOKE_DRAWS
    assert rec['final']['name'] == 'G' and rec['final']['outcome'] == 'KEEP-G' and rec['removals'] == ['direct', 'buddy']
    assert set(rec['results']['buddy']) == {'arm', 'status', 'cause', 'aspects', 'M'}
    assert set(rec['results']['buddy']['aspects']['emotion']) == {'effect', 'lo', 'hi', 'arm_ami', 'final_ami', 'arm_seed_ami', 'final_seed_ami'}
    assert rec['failed'] == [] and rec['inputs']['v3a_sha256'] == S._file_sha256(tmp_path / 'gates' / 'v3a.json')
    path = tmp_path / 'ablation' / 'dev.json'
    assert json.loads(path.read_text())['results'] == json.loads(json.dumps(rec['results']))
    again = A.run_dev(tmp_path, 'G', smoke=True, workers=1, removals=['buddy', 'direct'])
    assert again['results'] == rec['results']


def test_a_full_run_has_all_eight_removals_and_a_subset_only_with_smoke(tmp_path):
    with pytest.raises(AssertionError, match='smoke'):
        A.check_removals(['buddy'], smoke=False)
    assert A.check_removals(None, smoke=False) == list(H.REMOVALS)
    assert A.check_removals(['buddy', 'direct'], smoke=True) == ['direct', 'buddy']      # in the sheet's order
    with pytest.raises(AssertionError):
        A.check_removals(['nonsense'], smoke=True)


def test_final_argument_must_match_before_anything_is_read(tmp_path, monkeypatch):
    put_stage3(tmp_path, 'PICK', 'k6=0.0005')
    monkeypatch.setattr(A, 'read_final_dev_runs', lambda *a: pytest.fail('read before --final was checked'))
    with pytest.raises(AssertionError, match='--final'):
        A.run_dev(tmp_path, 'G', smoke=False, workers=1)


def test_cli_modes_and_smoke_outputs_must_be_scratch(tmp_path):
    with pytest.raises(SystemExit):
        A.main(['both', '--final', 'G'])
    with pytest.raises(AssertionError, match='smoke'):
        A.main(['dev', '--final', 'G', '--smoke', '--outputs', str(tmp_path)])        # not under /tmp/claude-0


def test_held_arms_list_a_failed_arm_require_outputs_of_the_others_and_refuse_an_unfinished_one(tmp_path):
    put_stage3(tmp_path, 'KEEP-G')
    final = A.resolve_final(tmp_path, smoke=False)
    train, _ = full_dirs(tmp_path, 'G-no_buddy')
    (train / 'failed__s3.json').write_text(json.dumps(dict(cause='non-finite loss in the style head')))
    _, held = full_dirs(tmp_path, 'G-no_direct')
    for a in ASP:
        for s in FULL:
            (held / f'{a}__s{s}.npz').write_bytes(b'x')
    arms = A.held_arms(tmp_path, final, ['direct', 'buddy'])
    assert arms == {'direct': dict(name='G-no_direct', cause=None),
                    'buddy': dict(name='G-no_buddy', cause='seed 3: non-finite loss in the style head')}
    with pytest.raises(AssertionError, match='G-no_sharp_row.*not finished'):
        A.held_arms(tmp_path, final, ['sharp_row'])


def test_held_record_keys_and_the_file_is_kept_on_a_rerun(tmp_path, monkeypatch):
    final, arms, labels, parts = held_world(tmp_path, monkeypatch, failed=('buddy',), smoke=True)
    monkeypatch.setattr(A.rb_part2_report, 'held_labels', lambda: (['p'] * rs_paths.N_HELD, rs_paths.HELD_SHA256, labels))
    monkeypatch.setattr(A, 'held_arms', lambda outputs, f, removals: {x: arms[x] for x in removals})
    monkeypatch.setattr(A, 'held_final_source', lambda outputs, f: ('froot', 'GS-0.02'))
    rec = A.run_held(tmp_path, 'G', smoke=True, workers=1, removals=['buddy', 'direct'])
    assert {'stage', 'smoke', 'written', 'rule_commit', 'ranges', 'seeds', 'n_draws', 'held_sha256', 'draws', 'final', 'final_key',
            'removals', 'results', 'failed', 'inputs', 'code_sha256'} <= set(rec)
    assert rec['stage'] == 'ablation_held' and rec['seeds'] == [0, 1, 2, 3, 4] and rec['n_draws'] == T.SMOKE_DRAWS
    assert rec['held_sha256'] == rs_paths.HELD_SHA256 and rec['final_key'] == 'GS-0.02' and rec['removals'] == ['direct', 'buddy']
    assert rec['results']['buddy'] == dict(arm='G-no_buddy', status='failed', cause='seed 4: non-finite loss in the genre head', aspects=None)
    assert set(rec['results']['direct']['aspects']['genre']) == {'effect', 'lo', 'hi', 'arm_ami', 'final_ami', 'arm_seed_ami', 'final_seed_ami'}
    assert rec['failed'] == [dict(removal='buddy', arm='G-no_buddy', cause='seed 4: non-finite loss in the genre head')]
    assert rec['inputs']['stage4_sha256'] is None and set(rec['inputs']['outputs']) == {'GS-0.02', 'G-no_direct'}
    assert json.loads((tmp_path / 'ablation' / 'held.json').read_text())['results'] == rec['results']
    assert A.run_held(tmp_path, 'G', smoke=True, workers=1, removals=['buddy', 'direct'])['results'] == rec['results']
