"""Tests for pf_queue.sh (ticket 13): `--dry-run` for full and smoke. No GPU, no data, no git writes."""
import shlex
import subprocess
from pathlib import Path

import pytest

QUEUE = Path(__file__).resolve().parent / 'pf_queue.sh'
LOCK = ['flock', '-o', '-w', '3600', '-E', '75', '/tmp/gpu0.lock']
STEPS = ['extract_images', 'extract_texts', 'stage0a', 'stage1', 'commit_sheet_nearcopy', 'stage2', 'commit_stage2',
         'stage3', 'c13_derive', 'c13_compare', 'diag']
GPU_STEPS = ['extract_images', 'extract_texts', 'stage0a', 'stage1', 'stage2']
CPU_STEPS = ['stage3', 'c13_derive', 'c13_compare', 'diag']
DRAWS = ['--draws-clu', '200', '--draws-ret', '500']


def dry_run(*args):
    r = subprocess.run(['bash', str(QUEUE), *args, '--dry-run'], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout


def parse(out):
    """{step: [argv, ...]} in order (None for a skipped step): the commands are the indented lines after `# step`."""
    steps, cur = {}, None
    for line in out.splitlines():
        if line.startswith('# '):
            cur = line[2:].split(' (')[0]
            steps[cur] = []
        elif line.startswith('  # '):
            steps[line[4:].split(' ')[0]] = None
        elif line.startswith('  ') and cur:
            steps[cur].append(shlex.split(line))
    return steps


@pytest.fixture(scope='module')
def full():
    out = dry_run('full')
    return out, parse(out)


@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    out = dry_run('smoke', str(tmp_path_factory.mktemp('scratch')))
    return out, parse(out)


def argv(steps, name):
    assert len(steps[name]) == 1, name
    return steps[name][0]


def test_step_order_full(full):
    assert list(full[1]) == STEPS


def test_step_order_smoke(smoke):
    got = list(smoke[1])
    assert got.index('stage1') < got.index('check_records') < got.index('stage2')    # records exist before stage 2
    assert [s for s in got if s != 'check_records'] == STEPS


@pytest.mark.parametrize('mode', ['full', 'smoke'])
def test_flock_prefix_on_gpu_steps_only(mode, full, smoke):
    steps = {'full': full, 'smoke': smoke}[mode][1]
    for name in GPU_STEPS:
        assert argv(steps, name)[:len(LOCK)] == LOCK, name
    for name in CPU_STEPS:
        assert 'flock' not in argv(steps, name), name


def test_commands_full(full):
    out, steps = full
    assert argv(steps, 'extract_images')[-1] == 'images' and argv(steps, 'extract_texts')[-1] == 'texts'
    assert argv(steps, 'stage0a')[-1] == '0a'
    assert argv(steps, 'stage1')[-1] == '1' and argv(steps, 'stage2')[-1] == '2' and argv(steps, 'stage3')[-1] == '3'
    for flag in ('--smoke', '--rows', '--max-epochs', '--draws-clu', '--draws-ret'):
        assert flag not in out, flag
    assert 'env: PF_SMOKE and PF_OUTPUTS unset' in out
    assert argv(steps, 'c13_derive')[-3:] == ['derive', '--workers', '8']
    assert argv(steps, 'c13_compare')[-1] == 'compare'
    assert argv(steps, 'diag')[-1] == 'diag'


def test_full_commits_and_pushes(full):
    steps = full[1]
    for name, files in (('commit_sheet_nearcopy', ('stage1_sheet.json', 'stage0a_nearcopy.json')),
                        ('commit_stage2', ('stage2_records.json',))):
        cmds = steps[name]
        assert [c[3] for c in cmds] == ['add', 'commit', 'push'], name
        assert cmds[0][4] == '-f'
        for f in files:
            assert any(a.endswith(f) for a in cmds[0]) and any(a.endswith(f) for a in cmds[1]), (name, f)


def test_smoke_flags_and_no_commits(smoke):
    out, steps = smoke
    assert 'git' not in out and 'push' not in out
    assert steps['commit_sheet_nearcopy'] is None and steps['commit_stage2'] is None
    assert argv(steps, 'extract_images')[-2:] == ['images', '--smoke']
    assert argv(steps, 'extract_texts')[-2:] == ['texts', '--smoke']
    assert argv(steps, 'stage0a')[-1] == '0a'               # full draws: the SC replay gate needs them
    assert argv(steps, 'stage1')[-5:] == ['1', '--rows', '1,8', '--max-epochs', '2']
    assert argv(steps, 'stage2')[-3:] == ['2', '--max-epochs', '2']
    assert argv(steps, 'stage3')[-5:] == ['3'] + DRAWS
    assert argv(steps, 'c13_derive')[-7:] == ['derive', '--workers', '8'] + DRAWS
    assert 'env: PF_SMOKE=2000 PF_OUTPUTS=' in out


def test_smoke_n_and_default_outputs_refused(tmp_path):
    assert 'env: PF_SMOKE=123 ' in dry_run('smoke', str(tmp_path), '--smoke-n', '123')
    r = subprocess.run(['bash', str(QUEUE), 'smoke', '/data/SSD2/cosir_r13/20261031_photo_finetune/outputs',
                        '--dry-run'], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and 'must differ' in r.stderr


@pytest.fixture
def queue_harness(tmp_path):
    """Execute the real queue in smoke mode, with only external commands replaced (review B's harness seam)."""
    import os
    import sys
    script = tmp_path / 'pf_queue.sh'
    fake = tmp_path / 'fake.py'
    out = tmp_path / 'out'
    fake.write_text('''import json, os, sys
from pathlib import Path
out = Path(os.environ['PF_OUTPUTS'])
cmd = sys.argv[1]
with (out / 'calls.txt').open('a') as f:
    f.write(cmd + '\\n')
def write(name):
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
if cmd == 'images': write('smoke_views/emotion_focus_train/meta.json')
elif cmd == 'texts': write('smoke_views/emotion_focus_txt/meta.json')
elif cmd == '0a':
    write('stage0a/gates.json')
    write('records/stage0a_nearcopy.json')
elif cmd == '1':
    write('stage1/rows.json')
    write('records/stage1_sheet.json')
elif cmd == '2': write('records/stage2_records.json')
elif cmd == '3': write('stage3/verdict.json')
elif cmd == 'derive': write('rederive/theirs.json')
elif cmd == 'compare': write('rederive/rederive.json')
elif cmd == 'diag': write('diag/diag.json')
else: raise AssertionError(cmd)
''')
    text = QUEUE.read_text()
    for name, replacement in (
            ('GPU=', 'GPU=()'),
            ('PFRUN=', f'PFRUN=({shlex.quote(sys.executable)} {shlex.quote(str(fake))})'),
            ('PFEXTRACT=', f'PFEXTRACT=({shlex.quote(sys.executable)} {shlex.quote(str(fake))})'),
            ('PFREDERIVE=', f'PFREDERIVE=({shlex.quote(sys.executable)} {shlex.quote(str(fake))})')):
        lines = text.splitlines()
        assert sum(line.startswith(name) for line in lines) == 1
        text = '\n'.join(replacement if line.startswith(name) else line for line in lines) + '\n'
    script.write_text(text)

    def run():
        result = subprocess.run(['bash', str(script), 'smoke', str(out)], capture_output=True, text=True,
                                timeout=30, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        assert result.returncode == 0, result.stdout + result.stderr
        return dict(line.split('\t')[1:] for line in (out / 'queue_status.txt').read_text().splitlines())

    return out, run


def test_stage1_seal_needs_sheet_and_output(queue_harness):
    out, run = queue_harness
    assert run()['stage1'] == '0'
    # Stale sheet survives an outputs reset; the output evidence is absent.
    rows = out / 'stage1/rows.json'
    rows.rename(rows.with_suffix('.aside'))
    assert run()['stage1'] == '0'
    assert rows.is_file()
    assert run()['stage1'] == 'skipped'


@pytest.mark.parametrize('newer', ['verdict', 'rederive', 'diag'])
def test_compare_always_runs_and_diag_requires_fresh_inputs(queue_harness, newer):
    import os
    out, run = queue_harness
    run()
    paths = {'verdict': out / 'stage3/verdict.json', 'rederive': out / 'rederive/rederive.json',
             'diag': out / 'diag/diag.json'}
    # Explicit mtimes avoid sleeps and filesystem timestamp resolution dependence.
    for name, path in paths.items():
        stamp = (2_000_000_010 if newer == 'diag' else 1_000_000_010) if name == newer else 1_000_000_000
        os.utime(path, (stamp, stamp))
    status = run()
    assert status['c13_derive'] == 'skipped'
    assert status['c13_compare'] == '0'
    assert status['diag'] == ('skipped' if newer == 'diag' else '0')
