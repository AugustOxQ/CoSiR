"""Final-review regressions on scratch trees; no training or real outputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import eg_encoders
import eg_lift
import eg_matched
import eg_paths
import eg_pinfer
import eg_run
import eg_score
import eg_v2
import eg_v5


def put(out, rel, value):
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def gates(out):
    put(out, 'stores/x0.json', {'pass': True, 'stores': {s: {'equal': True} for s in eg_run.CALIBRATION}})
    for part in eg_run.X_PARTS:
        put(out, f'part2/regression/x1_{part}.json', {'pass': True, 'items': {'test': {'pass': True}}})


def stage3(out, **blocks):
    return put(out, 'part2/verdict_stage3.json', dict(stage=3, scale='sample', held_sha256='H', blocks=blocks))


def bare_run(out, *, smoke=False, forced=None):
    run = eg_pinfer.Run.__new__(eg_pinfer.Run)
    run.outputs = out
    run.args = SimpleNamespace(smoke=smoke, arms=None, seeds=None, forced_vl3=forced, device='cpu')
    run.base = out / 'stage_ii/p'
    run.index_path = run.base / 'index.json'
    run.p_sha, run.n_rows, run.commit, run.code_sha = 'P', 1, 'COMMIT', 'CODE'
    run.aspects, run.seeds = eg_paths.ASPECTS, eg_paths.SEEDS
    return run


@pytest.mark.parametrize('forced', [None, 'failed VL3'])
@pytest.mark.parametrize('change', ['arm', 'seed'])
def test_vl4_refuses_added_arm_or_seed_even_when_forced(tmp_path, monkeypatch, forced, change):
    run = bare_run(tmp_path, smoke=True, forced=forced)
    monkeypatch.setattr(eg_pinfer, 'plan', lambda *a: (['K6'], {}))
    monkeypatch.setattr(run, 'student_one', lambda *a: 'kept')
    monkeypatch.setattr(run, 'own_one', lambda *a: 'kept')
    put(tmp_path, 'stage_ii/p/index.json', dict(arms=[] if change == 'arm' else ['K6'], missing={},
        seeds=list(eg_paths.SEEDS[:-1] if change == 'seed' else eg_paths.SEEDS),
        aspects=list(eg_paths.ASPECTS), p_sha256='P', complete=True))
    with pytest.raises(AssertionError, match='VL4: an arm or seed'):
        run.infer()


@pytest.mark.parametrize('state', ['absent', 'partial', 'missing_x1'])
def test_non_smoke_refuses_premature_plan_lock(tmp_path, monkeypatch, state):
    run = bare_run(tmp_path)
    monkeypatch.setattr(eg_pinfer, 'plan', lambda *a: ([], {}))
    if state != 'absent':
        gates(tmp_path)
        stage3(tmp_path, K6={'outcome_id': 'S1-GO'})
    if state == 'missing_x1':
        (tmp_path / 'part2/regression/x1_scorer.json').rename(tmp_path / 'x1.off')
    with pytest.raises((AssertionError, eg_run.Stop), match='stage 3'):
        run.infer()
    assert not run.index_path.exists()


@pytest.mark.parametrize('state', ['complete', 'deferred', 'failed_x0', 'failed_x1'])
def test_plan_lock_allowed_after_recorded_completion_or_actual_failure(tmp_path, monkeypatch, state):
    gates(tmp_path)
    if state in ('complete', 'deferred'):
        stage3(tmp_path, **{c: {'outcome_id': None, 'v_failure': 'V5'} for c in eg_encoders.CANDIDATES
                           if state != 'deferred' or c != 'K5'})
    if state == 'deferred':
        put(tmp_path, 'part2/deferral.json', dict(deferred=['K5'], history=[]))
    if state.startswith('failed'):
        put(tmp_path, 'stores/x0.json' if state == 'failed_x0' else 'part2/regression/x1_scorer.json', {'pass': False})
    run = bare_run(tmp_path)
    monkeypatch.setattr(eg_pinfer, 'plan', lambda *a: ([], {}))
    run.infer()
    assert json.loads(run.index_path.read_text())['complete']


@pytest.mark.parametrize('state', ['no_outcome', 'absent', 'deferred', 'failed_x0'])
def test_weights_do_not_rescue_a_missing_stage3_outcome(tmp_path, monkeypatch, state):
    enc = eg_encoders.BY_STUDENT['K6']
    stage3(tmp_path, K6={'outcome_id': None if state == 'no_outcome' else 'S1-GO', 'v_failure': 'V3 held inference failed'})
    if state == 'absent':
        stage3(tmp_path)
    if state == 'deferred':
        put(tmp_path, 'part2/deferral.json', dict(deferred=['K6'], history=[]))
    if state == 'failed_x0':
        put(tmp_path, 'stores/x0.json', {'pass': False})
    monkeypatch.setattr(eg_pinfer, 'student_files', lambda *a: [])
    monkeypatch.setattr(eg_pinfer, 'own_files', lambda *a: [])
    arms, missing = eg_pinfer.plan(tmp_path, encoders=(enc,))
    assert arms == [] and set(missing) == {'K6', 'M6'}
    if state == 'no_outcome':
        assert missing['K6'] == 'V3 held inference failed'


@pytest.mark.parametrize('scale', ['sample', 'full'])
def test_matched_held_gate_does_not_need_failed_a28_store(tmp_path, scale):
    gates(tmp_path)
    for part in eg_run.X_PARTS:
        put(tmp_path, f'part2/regression/x2_{part}.json', {'pass': True})
    put(tmp_path, 'stores/v5.json', {'stores': {'artelingo_full/clip_b16': {'pass': True},
        'artelingo28/clip_b16': {'pass': False, 'cause': 'index missing'}}})
    eg_matched.require_gate(tmp_path, scale, 'M6', kinds=('held',))
    if scale == 'sample':
        with pytest.raises(AssertionError, match='artelingo28'):
            eg_matched.require_gate(tmp_path, scale, 'M6')


def test_v5_records_pass_before_a28_exception_and_keeps_its_cause(tmp_path, monkeypatch):
    put(tmp_path, 'stores/v5.json', dict(calibration={s: .2 for s in eg_v5.CALIBRATION}, stores={}))
    def check(dataset, *args, **kw):
        if dataset == eg_v5.A28:
            saved = json.loads((tmp_path / 'stores/v5.json').read_text())
            assert saved['stores']['artelingo_full/clip_b16']['pass']
            raise FileNotFoundError('A28 index.json missing')
        return {'pass': True, 'checks': {'rows': True}}
    monkeypatch.setattr(eg_v5, 'check_store', check)
    monkeypatch.setattr(eg_paths, 'check_imports', lambda: None)
    monkeypatch.setattr('sys.argv', ['eg_v5.py', '--outputs', str(tmp_path), '--store', 'clip_b16',
                                   '--dataset', 'artelingo_full', '--dataset', 'artelingo28'])
    with pytest.raises(SystemExit) as err:
        eg_v5.main()
    assert err.value.code == 1
    saved = json.loads((tmp_path / 'stores/v5.json').read_text())['stores']
    assert saved['artelingo_full/clip_b16']['pass']
    assert saved['artelingo28/clip_b16']['pass'] is False
    assert saved['artelingo28/clip_b16']['cause'] == 'FileNotFoundError: A28 index.json missing'
    assert saved['artelingo28/clip_b16']['fails_on_28_only']


def test_failed_v5_cause_precedes_missing_held_output(tmp_path):
    put(tmp_path, 'stores/v5.json', {'stores': {'artelingo_full/clip_b16': {
        'pass': False, 'cause': 'caption mean is nonfinite', 'checks': {'finite': False}}}})
    block = eg_score.stage_block('K6', 'sample', eg_score.Roots(tmp_path, tmp_path), np.arange(2), 'H', {},
                                 tmp_path / 'rcol.json', 2, 1)
    assert block['outcome_id'] is None and 'caption mean is nonfinite' in block['v_failure']


def test_no_stage3_outcome_is_left_out_of_a28_plan_and_gets_a_no_outcome_block(tmp_path, monkeypatch):
    stage3(tmp_path, K6={'outcome_id': None, 'v_failure': 'V4 failed'}, K7={'outcome_id': 'S1-GO'})
    put(tmp_path, 'stores/v5.json', {'stores': {f'artelingo28/{s}': {'pass': True} for s in ('clip_b16', 'clip_l14')}})
    assert eg_run.a28_plan(tmp_path)[0] == ['K7']
    # Even if stale A28 outputs exist, the statistic must never be called for K6.
    monkeypatch.setattr(eg_score, 'a28_block', lambda *a: pytest.fail('A28 must not rescue stage 3'))
    setup = dict(names=['p'], sha256='A28')
    record = eg_score.run_verdict(['K6'], tmp_path, setup=setup, workers=1, n_resamples=1)
    assert record['blocks']['K6']['outcome_id'] is None
    assert record['blocks']['K6']['v_failure'] == 'no stage 3 outcome: V4 failed'


def test_failed_a28_v5_cause_precedes_outputs(tmp_path, monkeypatch):
    stage3(tmp_path, K6={'outcome_id': 'S1-GO'})
    put(tmp_path, 'stores/v5.json', {'stores': {'artelingo28/clip_b16': {
        'pass': False, 'cause': 'A28 index missing', 'checks': {}}}})
    monkeypatch.setattr(eg_score, 'a28_block', lambda *a: pytest.fail('failed V5 must block A'))
    record = eg_score.run_verdict(['K6'], tmp_path, setup=dict(names=['p'], sha256='A28'))
    assert record['blocks']['K6']['outcome_id'] is None and 'A28 index missing' in record['blocks']['K6']['v_failure']


def test_v2_no_outcomes_message_keeps_the_stop(tmp_path, capsys):
    stage3(tmp_path, K6={'outcome_id': None, 'v_failure': 'V5 failed'})
    assert eg_v2.main(['--stage', '3', '--outputs', str(tmp_path)]) == 1
    assert 'no candidate has an outcome' in capsys.readouterr().out
    put(tmp_path, 'part2/rederive_v2_stage3.json', dict(agree=False, candidates={}))
    with pytest.raises(eg_run.Stop, match='no candidate has an outcome'):
        eg_run.gate_v2(tmp_path, 3)


def test_recorded_deferral_controls_candidate_lists_and_check3(tmp_path, monkeypatch):
    gates(tmp_path)
    stage3(tmp_path, **{c: {'outcome_id': 'S1-GO'} for c in ('K6', 'K7', 'K8')})
    monkeypatch.setenv('EG_DEFER', 'K6')  # ignored: only the recorded deferral is authoritative
    put(tmp_path, 'part2/deferral.json', dict(deferred=['K5'], history=[{'estimate_end': '12:00'}]))
    assert eg_run.candidates(tmp_path, 3, todo=False) == ['K6', 'K7', 'K8']
    assert eg_run.main(['check-verdict', '3', '--outputs', str(tmp_path)]) == 0


@pytest.mark.parametrize('kind', ['student', 'own'])
@pytest.mark.parametrize('mismatch', ['weights', 'code', None])
def test_forced_export_rerun_requires_same_weights_and_code(tmp_path, monkeypatch, kind, mismatch):
    import rb_student
    import rb_matched
    run = bare_run(tmp_path, forced='VL3 nonfinite')
    enc = eg_encoders.BY_STUDENT['K6']
    out = (eg_paths.p_student(tmp_path, 'K6', 'emotion', 0) if kind == 'student'
           else eg_paths.p_own(tmp_path, 'M6', 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b'failed VL3')
    put(tmp_path, str(out.with_suffix('.json').relative_to(tmp_path)),
        dict(weights_sha256='OLD' if mismatch == 'weights' else 'W',
             code_commit='ANY COMMIT', code_sha256='OLD' if mismatch == 'code' else 'CODE'))
    monkeypatch.setattr(run, 'store_images', lambda *a: np.ones((1, enc.width), np.float32))
    monkeypatch.setattr(rb_student, 'weights_path', lambda *a: tmp_path / 'weights.pt')
    put(tmp_path, 'weights.json', dict(weights_sha256='W'))
    monkeypatch.setattr(rb_student.ms_train.head, 'sha_file', lambda *a: 'W')
    img = np.zeros((1, 2048), np.float16)
    img[0, 0] = 1
    monkeypatch.setattr(rb_student, 'infer', lambda *a: img)
    monkeypatch.setattr(rb_matched.base, 'verify_output', lambda *a: dict(weights_sha256='W', config={'device': 'cpu'}))
    monkeypatch.setattr(rb_matched, 'infer', lambda *a: np.zeros((1, 40), np.float32))
    call = lambda: run.student_one(enc, 'emotion', 0, run.args.forced_vl3) if kind == 'student' else run.own_one(enc, 0, run.args.forced_vl3)
    if mismatch:
        with pytest.raises(AssertionError, match='VL4.*same weights and code'):
            call()
        assert out.read_bytes() == b'failed VL3'
    else:
        assert call() == 'wrote'
        side = json.loads(out.with_suffix('.json').read_text())
        assert side['code_commit'] == 'COMMIT' and side['code_sha256'] == 'CODE'
        assert len((tmp_path / 'stage_ii/vl4_log.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('kind', ['student', 'own'])
def test_moved_export_after_complete_is_logged_as_a_rerun(tmp_path, monkeypatch, kind):
    # Exercise the same real one-export paths as forced reruns, with the NPZ moved aside.
    import rb_student
    import rb_matched
    run = bare_run(tmp_path)
    put(tmp_path, 'stage_ii/p/index.json', dict(complete=True))
    enc = eg_encoders.BY_STUDENT['K6']
    out = eg_paths.p_student(tmp_path, 'K6', 'emotion', 0) if kind == 'student' else eg_paths.p_own(tmp_path, 'M6', 0)
    put(tmp_path, str(out.with_suffix('.json').relative_to(tmp_path)), dict(weights_sha256='W', code_commit='COMMIT', code_sha256='CODE'))
    monkeypatch.setattr(run, 'store_images', lambda *a: np.ones((1, enc.width), np.float32))
    monkeypatch.setattr(rb_student, 'weights_path', lambda *a: tmp_path / 'weights.pt')
    put(tmp_path, 'weights.json', dict(weights_sha256='W'))
    monkeypatch.setattr(rb_student.ms_train.head, 'sha_file', lambda *a: 'W')
    img = np.zeros((1, 2048), np.float16)
    img[0, 0] = 1
    monkeypatch.setattr(rb_student, 'infer', lambda *a: img)
    monkeypatch.setattr(rb_matched.base, 'verify_output', lambda *a: dict(weights_sha256='W', config={'device': 'cpu'}))
    monkeypatch.setattr(rb_matched, 'infer', lambda *a: np.zeros((1, 40), np.float32))
    assert (run.student_one(enc, 'emotion', 0, None) if kind == 'student' else run.own_one(enc, 0, None)) == 'wrote'
    log = json.loads((tmp_path / 'stage_ii/vl4_log.jsonl').read_text())
    assert 'missing' in log['cause'] and log['file'] == str(out)


@pytest.mark.parametrize('scale', ['sample', 'full'])
def test_matched_held_only_run_never_loads_or_writes_a28(tmp_path, monkeypatch, scale):
    import rb_matched
    world = SimpleNamespace(scale=scale, s={}, held_names=['p'], held_sha256='H',
        topics=lambda seed: (None, 40, 'TOPIC'),
        full_images=lambda *a: np.ones((1, 512), np.float32),
        a28_images=lambda *a: pytest.fail('A28 store must not be loaded'))
    monkeypatch.setattr(eg_paths, 'check_imports', lambda: None)
    monkeypatch.setattr(rb_matched, 'load_world', lambda *a: world)
    monkeypatch.setattr(rb_matched, '_stage2_ok', lambda *a: True)
    monkeypatch.setattr(rb_matched, '_output_ok', lambda *a: False)
    monkeypatch.setattr(rb_matched.base, 'verify_output', lambda *a: {'weights_sha256': 'W'})
    monkeypatch.setattr(rb_matched, 'infer', lambda *a: np.zeros((1, 40), np.float32))
    paths = eg_matched.run(scale, 'M6', [0], tmp_path, 'cpu', gated=False, kinds=('held',))
    assert len(paths) == 1 and paths[0].is_file()
    assert not (tmp_path / f'part2/{scale}/a28').exists()


# cpu_i's margins share rb_score's workers; inspect all eight processes, including VL2's chunk scope.
def _init_thread_probe(kind, barrier):
    global _PROBE_KIND, _PROBE_BARRIER
    _PROBE_KIND, _PROBE_BARRIER = kind, barrier
    if kind == 'cpu_i':
        eg_score.rb_score._init_worker({}, np.arange(2), ())
    elif kind == 'lift':
        eg_lift._init_worker({})


def _probe_threads(_):
    import os
    from threadpoolctl import threadpool_info
    _PROBE_BARRIER.wait(timeout=20)
    if _PROBE_KIND == 'vl2':
        import eg_vl2
        observed = []
        def measured_mean(*args):
            observed.extend(threadpool_info())
            return 0.
        eg_vl2.mean_score = measured_mean
        eg_vl2._init_bootstrap(np.arange(2), None, None, None, np.arange(2))
        eg_vl2._bootstrap_chunk((0, 1))
    else:
        observed = threadpool_info()
    return os.getpid(), [(i['internal_api'], i['num_threads']) for i in observed]


@pytest.mark.parametrize('kind', ['cpu_i', 'lift', 'vl2'])
def test_every_cpu_worker_limits_blas_and_omp_to_one_thread(kind):
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(8, initializer=_init_thread_probe, initargs=(kind, ctx.Barrier(8))) as pool:
        rows = pool.map(_probe_threads, range(8))
    assert len({pid for pid, info in rows}) == 8
    assert all(info and all(n == 1 for api, n in info) for pid, info in rows), rows
