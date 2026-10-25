"""Queue dry runs: real gate/list driver, stubbed training/inference/scoring, scratch outputs only."""
import json
import os
from pathlib import Path
import subprocess

import pytest

import eg_encoders
import eg_paths
import eg_run
from test_eg_final_i_ii import gates, put, stage3

HERE = Path(__file__).parent.resolve()
PYTHON = '/root/miniconda3/envs/CoSiR/bin/python'
STUB = '''#!/root/miniconda3/envs/CoSiR/bin/python
import json, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-I':
    import os
    os.execv(sys.executable, [sys.executable] + args)
if args[0] == '-u': args.pop(0)
script = Path(args.pop(0))
sys.path.insert(0, str(script.parent))
import eg_run, eg_paths
out = Path(args[args.index('--outputs') + 1])
with (out / 'stub_calls.jsonl').open('a') as f: f.write(json.dumps([script.name] + args) + '\\n')
def write(rel, value):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))
def names():
    i = args.index('--candidates') + 1
    end = next((j for j in range(i, len(args)) if args[j].startswith('--')), len(args))
    return args[i:end]
if script.name == 'eg_run.py' and args[0] != 'lift-score':
    sys.exit(eg_run.main(args))
if script.name == 'eg_pinfer.py':
    psha = eg_paths.P_SHA256 if '--smoke' not in args or ('--n-rows' in args and args[args.index('--n-rows') + 1] == '29232') else 'WRONG_SMOKE_HASH'
    write('stage_ii/vl1.json', {'pass': True, 'p_sha256': psha})
elif script.name == 'eg_lift.py': write('stage_ii/vl1_scorer.json', {'pass': True, 'p_sha256': eg_paths.P_SHA256})
elif script.name == 'eg_run.py': write('stage_ii/lift.json', {'encoders': {}})
elif script.name == 'eg_vl2.py': write('stage_ii/rederive_vl2.json', {'agree': True})
elif script.name == 'eg_stats.py': write('stage_ii/' + ('stats.json' if args[0] == 'stats' else 'held_table.json'), {})
elif script.name == 'eg_score.py':
    stage = int(args[args.index('--stage') + 1])
    file = out / f'part2/verdict_stage{stage}.json'
    blocks = json.loads(file.read_text())['blocks'] if file.exists() else {}
    for c in names():
        failed = c == 'K7' or (stage == 5 and c == 'K6')
        blocks[c] = {'outcome_id': None if failed else f'S{1 if stage == 3 else 2}-GO', 'v_failure': 'V5 cause' if failed else None}
    write(f'part2/verdict_stage{stage}.json', dict(stage=stage, scale='sample' if stage == 3 else 'full', blocks=blocks))
elif script.name == 'eg_v2.py':
    stage = int(args[args.index('--stage') + 1])
    blocks = json.loads((out / f'part2/verdict_stage{stage}.json').read_text())['blocks']
    write(f'part2/rederive_v2_stage{stage}.json', {'agree': True, 'candidates': {c: {} for c,b in blocks.items() if b['outcome_id']}})
print('STUB', script.name, *args)
'''


@pytest.fixture
def stub(tmp_path):
    p = tmp_path / 'fakepython'
    p.write_text(STUB)
    p.chmod(0o755)
    return p


def run_queue(tmp_path, stub, queue, *args, smoke=False, dry=False, env_defer=''):
    env = dict(os.environ, EG_OUTPUTS=str(tmp_path), EG_PYTHON=str(stub), EG_DEVICE='cpu',
               EG_GPU_LOCK=str(tmp_path / 'gpu.lock'), EG_PENDING=str(tmp_path / 'pending.md'),
               EG_SMOKE='1' if smoke else '0', EG_DRY='1' if dry else '0', EG_DEFER=env_defer)
    result = subprocess.run(['bash', str(HERE / f'eg_queue_{queue}.sh'), *args], env=env,
                            text=True, capture_output=True, timeout=60)
    (tmp_path / f'dry_run_{queue}.log').write_text(result.stdout + result.stderr)
    file = tmp_path / 'stub_calls.jsonl'
    calls = [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
    return result, calls


@pytest.mark.parametrize('case, expected', [('empty', 'missing gate file'), ('no_blocks', 'stage 3 has no block'),
                                          ('missing_scorer', 'missing gate file')])
def test_stage_ii_never_calls_pinfer_before_inputs_are_ready(tmp_path, stub, case, expected):
    if case != 'empty': gates(tmp_path)
    if case == 'missing_scorer':
        (tmp_path / 'part2/regression/x1_scorer.json').rename(tmp_path / 'scorer.off')
    result, calls = run_queue(tmp_path, stub, 'stage_ii')
    assert result.returncode == 1 and expected in result.stdout
    assert not any(c[0] == 'eg_pinfer.py' for c in calls)


@pytest.mark.parametrize('failed', ['x0', 'x1'])
def test_failed_x0_or_x1_runs_old_arms_and_held_without_stage_i_reports(tmp_path, stub, failed):
    # X0 failure also works when X1 has never been run.
    if failed == 'x1': gates(tmp_path)
    put(tmp_path, 'stores/x0.json' if failed == 'x0' else 'part2/regression/x1_matched.json', {'pass': False})
    result, calls = run_queue(tmp_path, stub, 'stage_ii', smoke=True)
    assert result.returncode == 0, result.stdout
    assert any(c[0] == 'eg_pinfer.py' for c in calls)
    assert any(c[:2] == ['eg_stats.py', 'held'] for c in calls)
    assert (tmp_path / 'stage_ii/held_table.json').exists()
    assert 'waits for' not in result.stdout
    assert not (tmp_path / 'part2/reported_stage3.json').exists()


def test_stage_ii_smoke_reaches_lift_vl2_stats_and_held_with_full_p_hash(tmp_path, stub):
    gates(tmp_path)
    stage3(tmp_path, **{c: {'outcome_id': None if c == 'K7' else 'S1-GO', 'v_failure': 'V3 failed'} for c in eg_encoders.CANDIDATES})
    put(tmp_path, 'part2/reported_stage3.json', {})
    result, calls = run_queue(tmp_path, stub, 'stage_ii', smoke=True)
    assert result.returncode == 0, result.stdout
    pinfer, = [c for c in calls if c[0] == 'eg_pinfer.py']
    assert pinfer[pinfer.index('--n-rows') + 1] == '29232'
    assert any(c[:2] == ['eg_run.py', 'lift-score'] for c in calls)
    assert any(c[0] == 'eg_vl2.py' for c in calls)
    assert [c[1] for c in calls if c[0] == 'eg_stats.py'] == ['stats', 'held']


@pytest.mark.parametrize('failed', [False, True])
def test_held_only_waits_for_reports_unless_stage_i_failed(tmp_path, stub, failed):
    gates(tmp_path)
    put(tmp_path, 'stage_ii/stats.json', {})
    if failed: put(tmp_path, 'part2/regression/x1_scorer.json', {'pass': False})
    result, calls = run_queue(tmp_path, stub, 'stage_ii', 'held')
    assert result.returncode == 0
    assert any(c[:2] == ['eg_stats.py', 'held'] for c in calls) == failed
    assert ('waits for' in result.stdout) != failed


def v5(out):
    put(out, 'stores/v5.json', dict(stores={f'{ds}/{enc.store}': {'pass': not (
        enc.student == 'K7' and ds == 'artelingo_full' or enc.student == 'K6' and ds == 'artelingo28')}
        for enc in eg_encoders.ENCODERS if enc.new for ds in ('artelingo_full', 'artelingo28')}))


def test_stage_i_a28_only_failure_preserves_both_held_scales_and_writes_no_a_outcome(tmp_path, stub):
    gates(tmp_path)
    for part in eg_run.X_PARTS:
        put(tmp_path, f'part2/regression/x2_{part}.json', {'pass': True, 'items': {'test': {'pass': True}}})
    v5(tmp_path)
    # A real estimate-based deferral; the removed env route must not defer K6.
    eg_run.defer(tmp_path, '12:00', '11:00', None, eg_run.now_of('2026-10-25T09:00'))
    result, calls = run_queue(tmp_path, stub, 'stage_i', env_defer='K6')
    assert result.returncode == 0, result.stdout
    for scale in ('sample', 'full'):
        student, = [c for c in calls if c[:2] == ['eg_student.py', 'infer'] and c[c.index('--arm')+1] == 'K6'
                    and c[c.index('--scale')+1] == scale]
        matched, = [c for c in calls if c[0] == 'eg_matched.py' and '--arm' in c and c[c.index('--arm')+1] == 'M6'
                    and c[c.index('--scale')+1] == scale]
        assert student[student.index('--kinds')+1] == matched[matched.index('--kinds')+1] == 'held'
    assert not any('--arm' in c and c[c.index('--arm')+1] in ('K5', 'M5') for c in calls)
    a28, = [c for c in calls if c[0] == 'eg_score.py' and c[c.index('--stage')+1] == '5']
    assert 'K6' in a28 and 'K7' in a28 and 'K5' not in a28
    blocks = json.loads((tmp_path / 'part2/verdict_stage5.json').read_text())['blocks']
    assert blocks['K6']['outcome_id'] is None and blocks['K7']['outcome_id'] is None
    assert 'check3=0' in result.stdout and 'check5=0' in result.stdout


def test_stores_dry_run_uses_only_recorded_deferrals(tmp_path, stub):
    eg_run.defer(tmp_path, '12:00', '11:00', None, eg_run.now_of('2026-10-25T09:00'))
    result, calls = run_queue(tmp_path, stub, 'stores', dry=True, env_defer='K6')
    assert result.returncode == 0
    assert 'DRY clip_b16_full' in result.stdout and 'DRY dinov2_large_full' not in result.stdout
    assert 'stores for: K6 K7 K8' in result.stdout


def test_queue_shell_syntax():
    result = subprocess.run(['bash', '-n', *[str(HERE / f'eg_queue_{name}.sh')
        for name in ('common', 'stores', 'stage_i', 'stage_ii', 'cpu_i')]], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cpu_i_reports_v5_causes_when_no_candidate_has_an_outcome(tmp_path, stub, monkeypatch):
    stage3(tmp_path, K6={'outcome_id': None, 'v_failure': 'V5 missing index.json'})
    put(tmp_path, 'part2/rederive_v2_stage3.json', dict(agree=False, candidates={}))
    monkeypatch.setenv('EG_CPU_LOCK', str(tmp_path / 'cpu.lock'))
    result, calls = run_queue(tmp_path, stub, 'cpu_i', '3')
    assert result.returncode == 0, result.stdout
    assert any(c[0] == 'eg_report_i.py' for c in calls)
