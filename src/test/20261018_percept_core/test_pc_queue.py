"""Queue dependency and isolation checks without training or GPU access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pc_queue
import pc_stage0


def test_september_report_uses_measured_column():
    report = ('| held-out emotion AMI | 0.1225 | 0.1200 | 0.0025 | FAILED |\n'
              '| held-out genre AMI | 0.2274 | 0.2274 | 0.0000 | reproduced |\n'
              '| 42 | newly measured | 0.8296 | 0.6 | 0.8 | 0.9 | exceeds |\n')
    values = pc_stage0.report_values(report, stage2=True)
    assert values == {'held-out emotion AMI': .12, 'held-out genre AMI': .2274,
                      'seed42_macro_auc': .8296}
    assert pc_stage0.compare(values['held-out emotion AMI'], .1225)['status'] == 'FAIL'
    assert pc_stage0.compare(.829649, .8296)['status'] == 'PASS'
    assert pc_stage0.compare(.829651, .8296)['status'] == 'FAIL'
    assert pc_stage0.compare(None, .8296)['status'] == 'FAIL'


def test_timings_preserve_both_queues(tmp_path):
    status = tmp_path / 'status.txt'
    status.write_text('PC_START\nroberta=skip seconds=0\ntrain=2 seconds=1.250\n')
    pc_queue.write_times(status, tmp_path, 'gpu')
    status.write_text('score=0 seconds=2.5\n')
    pc_queue.write_times(status, tmp_path, 'cpu')
    result = json.loads((tmp_path / 'smoke_times.json').read_text())
    assert result['gpu']['steps']['roberta']['exit_code'] == 'skip'
    assert result['gpu']['steps']['train'] == dict(exit_code=2, seconds=1.25)
    assert result['cpu']['steps']['score']['seconds'] == 2.5


@pytest.fixture
def fake_python(tmp_path):
    path = tmp_path / 'python'
    path.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-u': args.pop(0)
if args[0].endswith('pc_queue.py') and args[1] == 'times':
    os.execv(sys.executable, [sys.executable, *args])
kind = Path(args[0]).name
with open(os.environ['PC_EVENTS'], 'a') as stream:
    stream.write(json.dumps(dict(args=args, cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                omp=os.environ.get('OMP_NUM_THREADS'),
                                status=Path(os.environ['PC_STATUS']).read_text()
                                if kind == 'pc_score.py' else None)) + '\\n')
failure = os.environ.get('PC_FAIL', '')
if failure == 'stage0' and kind == 'pc_stage0.py': sys.exit(7)
if failure == 'gate' and '--gate-check' in args: sys.exit(8)
if failure == 'version' and '--version' in args and 'qwen_qwen' in args: sys.exit(9)
sys.exit(0)
''')
    path.chmod(0o755)
    return path


def queue_run(tmp_path, fake_python, queue, failure='', smoke=True):
    root = tmp_path / 'run'
    env = dict(os.environ, PC_PYTHON=str(fake_python), PC_EVENTS=str(tmp_path / 'events'),
               PC_FAIL=failure, CUDA_VISIBLE_DEVICES='', PC_GPU_WAIT='2',
               PC_STATUS=str(root / 'pc_queue_status.txt'))
    args = ['bash', str(HERE / f'pc_queue_{queue}.sh')]
    if queue == 'gpu': args += ['--worker']  # No lock/GPU used by the command stub.
    assert smoke  # Keep all queue writes inside the test's isolated folder.
    args += ['--smoke', str(root)]
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
    events = [json.loads(line) for line in (tmp_path / 'events').read_text().splitlines()]
    status = (root / 'pc_queue_status.txt').read_text()
    times = json.loads((root / 'smoke_times.json').read_text())[queue]['steps']
    return result, events, status, times


def test_gpu_smoke_covers_all_versions_and_arms(tmp_path, fake_python):
    result, events, status, times = queue_run(tmp_path, fake_python, 'gpu')
    assert result.returncode == 0, result.stderr
    assert sum(e['args'][0].endswith('pc_stage1.py') for e in events) == 3
    assert sum(e['args'][0].endswith('pc_stage2.py') for e in events) == 5
    roberta = next(e['args'] for e in events if e['args'][0].endswith('pc_roberta.py'))
    assert roberta[roberta.index('--count') + 1] == '512'
    assert roberta[roberta.index('--outdir') + 1] == str(tmp_path / 'run/roberta')
    assert times['stage2_P5_s0']['exit_code'] == 0
    assert status.count('PC_gpu_DONE') == 1


def test_failed_stage0_keeps_extraction_but_blocks_percept(tmp_path, fake_python):
    result, events, status, times = queue_run(tmp_path, fake_python, 'gpu', 'stage0')
    assert result.returncode == 1
    assert any(e['args'][0].endswith('pc_roberta.py') for e in events)
    assert not any(e['args'][0].endswith(('pc_stage1.py', 'pc_stage2.py')) for e in events)
    assert times['stage1_clip_roberta_s0']['exit_code'] == 'blocked'


def test_stage1_failure_only_blocks_its_arms(tmp_path, fake_python):
    result, events, status, times = queue_run(tmp_path, fake_python, 'gpu', 'version')
    assert result.returncode == 1
    assert times['stage2_P3_s0']['exit_code'] == 'blocked'
    assert times['stage2_P4_s0']['exit_code'] == 'blocked'
    assert times['stage2_P5_s0']['exit_code'] == 0


def test_cpu_smoke_covers_rows_by_aspect(tmp_path, fake_python):
    result, events, status, times = queue_run(tmp_path, fake_python, 'cpu')
    assert result.returncode == 0, result.stderr
    runs = [e for e in events if '--train' in e['args'] and '--aspects' in e['args']]
    assert len(runs) == 8
    assert all(e['omp'] == '8' and e['cuda'] == '' for e in events)
    assert all('--smoke' in e['args'] for e in runs)
    assert 'row10_genre' not in times
    assert 'rows_score' in times and 'sheet' in times
    assert status.count('PC_cpu_DONE') == 1


def test_failed_cpu_gate_still_scores(tmp_path, fake_python):
    result, events, status, times = queue_run(tmp_path, fake_python, 'cpu', 'gate')
    assert result.returncode == 1
    assert not any('--aspects' in e['args'] for e in events)
    assert times['head_rows']['exit_code'] == 'blocked'
    assert times['rows_score']['exit_code'] == 0
    assert times['sheet']['exit_code'] == 0


@pytest.mark.parametrize('gpu_state', ['pending', 'timeout', 'none', 'ok'])
def test_cpu_sheet_waits_for_the_latest_gpu_submission(tmp_path, fake_python, gpu_state):
    root = tmp_path / 'run'
    root.mkdir()
    status_path = root / 'pc_queue_status.txt'
    status_path.write_text('PC_gpu_SUBMIT old\nPC_gpu_DONE old\nPC_gpu_SUBMIT new\n'
                           if gpu_state != 'none' else 'PC_gpu_DONE stale\n')
    if gpu_state == 'ok':
        with status_path.open('a') as stream:
            stream.write('PC_gpu_DONE new\n')
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = executor.submit(queue_run, tmp_path, fake_python, 'cpu')
        if gpu_state == 'pending':
            deadline = time.monotonic() + 10
            while 'rows_score=0' not in status_path.read_text() and time.monotonic() < deadline:
                time.sleep(.05)
            assert 'rows_score=0' in status_path.read_text()
            assert 'sheet=' not in status_path.read_text()
            with status_path.open('a') as stream:
                stream.write('PC_gpu_DONE new\n')
        result, events, status, times = job.result(timeout=15)
    expected = 'ok' if gpu_state == 'pending' else gpu_state
    assert result.returncode == (1 if expected == 'timeout' else 0), result.stderr
    assert status.count('gpu_wait=') == 1
    assert f'gpu_wait={expected} seconds=' in status
    assert times['gpu_wait']['exit_code'] == expected
    sheet = next(e for e in events if e['args'][0].endswith('pc_score.py'))
    if expected == 'ok':
        assert 'PC_gpu_DONE new' in sheet['status']
    assert times['sheet']['exit_code'] == 0
    if expected == 'timeout':
        assert times['gpu_wait']['seconds'] >= 2
        assert 'failed=1' in status


@pytest.mark.parametrize('foreground', [True, False])
@pytest.mark.parametrize('outcome,code', [('timeout', 1), ('died', 137), ('ok', 0), ('stage0', 1)])
def test_gpu_launcher_closes_each_submission_and_preserves_exit_code(
        tmp_path, fake_python, foreground, outcome, code):
    # Stub flock rather than touching the shared GPU lock.
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    flock = bindir / 'flock'
    flock.write_text('''#!/usr/bin/env bash
status="$PC_STATUS"
[[ $(tail -n 1 "$status") == PC_gpu_SUBMIT* ]] || exit 99
case "$PC_FLOCK_OUTCOME" in
    timeout) exit 1 ;;
    died) exit 137 ;;
esac
while [[ $1 != bash ]]; do shift; done
exec "$@"
''')
    flock.chmod(0o755)
    root = tmp_path / 'run'
    root.mkdir()
    status_path = root / 'pc_queue_status.txt'
    status_path.write_text('PC_gpu_SUBMIT old\nPC_gpu_DONE old\n')
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               PC_PYTHON=str(fake_python), PC_EVENTS=str(tmp_path / 'events'),
               PC_STATUS=str(status_path), PC_FLOCK_OUTCOME=outcome,
               PC_FAIL='stage0' if outcome == 'stage0' else '', CUDA_VISIBLE_DEVICES='')
    args = ['bash', str(HERE / 'pc_queue_gpu.sh'), '--smoke', str(root), '--lock-wait', '0']
    if foreground:
        args.append('--foreground')
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == (code if foreground else 0), result.stderr
    deadline = time.monotonic() + 10
    while status_path.read_text().count('PC_gpu_DONE') < 2 and time.monotonic() < deadline:
        time.sleep(.05)
    status = status_path.read_text()
    lines = status.splitlines()
    assert sum(line.startswith('PC_gpu_SUBMIT ') for line in lines) == 2
    assert sum(line.startswith('PC_gpu_DONE ') for line in lines) == 2
    assert lines[2].startswith('PC_gpu_SUBMIT ')
    assert lines[-1].startswith('PC_gpu_DONE ')
    if outcome in ('timeout', 'died'):
        assert f'PC_gpu_DONE lock_or_worker={code} failed=1 ' in lines[-1]
    else:
        assert 'lock_or_worker=' not in lines[-1]
        assert f'failed={code}' in lines[-1]


def test_full_gpu_schedule_skips_finished_store(tmp_path, fake_python):
    import shutil
    folder = tmp_path / 'repo/src/test/round'
    folder.mkdir(parents=True)
    for name in ('pc_queue_gpu.sh', 'pc_queue_common.sh'):
        shutil.copyfile(HERE / name, folder / name)
    events_path = tmp_path / 'events'
    env = dict(os.environ, PC_PYTHON=str(fake_python), PC_EVENTS=str(events_path), PC_FAIL='')
    result = subprocess.run(['bash', str(folder / 'pc_queue_gpu.sh'), '--worker'],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line)['args'] for line in events_path.read_text().splitlines()]
    assert sum(e[0].endswith('pc_stage1.py') for e in events) == 9
    assert sum(e[0].endswith('pc_stage2.py') for e in events) == 15
    assert not any(e[0].endswith('pc_roberta.py') for e in events)
    status = (folder / 'pc_queue_status.txt').read_text()
    assert 'roberta=skip seconds=0' in status
    assert status.count('PC_gpu_DONE') == 1


@pytest.mark.parametrize('emotion,expected_status', [(.1225, 'PASS'), (.1226, 'FAIL')])
def test_stage0_checks_fresh_measured_results_and_exact_schema(tmp_path, monkeypatch,
                                                            emotion, expected_status):
    monkeypatch.setattr(pc_stage0, 'HERE', tmp_path / 'round')
    monkeypatch.setattr(pc_stage0, 'REPO', tmp_path)
    monkeypatch.setattr(pc_stage0, 'export_tree', lambda tree: tree.mkdir(parents=True))
    outdir = tmp_path / 'outputs'
    calls = []

    def fake_run(name, command, folder, cwd, env):
        calls.append(name)
        if name.startswith('run_percept'):
            assert env['PYTHONPATH'].split(os.pathsep)[-1] == str(cwd)
            assert Path(env['PERCEPT_OUTPUT_ROOT']).is_relative_to(cwd)
            assert command == [sys.executable, '-u', str(cwd / pc_stage0.PILOT / f'{name}.py')]
            report = (f'| held-out emotion AMI | 0.1225 | {emotion:.4f} | x | reproduced |\n'
                      '| held-out genre AMI | 0.2274 | 0.2274 | x | reproduced |\n')
            if name == pc_stage0.SCRIPTS[1]:
                report += '| 42 | newly measured | 0.8296 | x | x | x | exceeds |\n'
            (folder / f'{name.removeprefix("run_")}_report.md').write_text(report)
        elif name == 'row2':
            assert env['CUDA_VISIBLE_DEVICES'] == ''
            (outdir / 'row2_check.json').write_text(json.dumps(dict(
                both_mean=dict(emotion=.08855639551158387, style=.17095358715401665,
                               genre=.3930761231906966))))
        elif name == 'fix_tests':
            (folder / 'fix_tests.log').write_text('.....................\n21 passed in 1.26s\n')
        return dict(status='PASS', expected=0, actual=0, seconds=.01)

    monkeypatch.setattr(pc_stage0, 'run_command', fake_run)
    code = pc_stage0.run(outdir)
    result = json.loads((outdir / 'stage0.json').read_text())
    assert result['status'] == expected_status
    assert code == (0 if expected_status == 'PASS' else 1)
    assert result['fix_tests'] == '21 passed in 1.26s'
    assert result['checks']['seed42_macro_auc']['actual'] == .8296
    assert result['checks']['september_1_held-out emotion AMI']['actual'] == emotion
    assert calls == [*pc_stage0.SCRIPTS, 'row2', 'fix_tests']
