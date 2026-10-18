"""Run the unchanged September regression in an isolated exported tree (CM-1–3)."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COMMIT = '1b70d7f'
PILOT = Path('src/test/20260922_percept_topic_pipeline')
SCRIPTS = ('run_percept_stage1_embedding_snapshot_pilot',
           'run_percept_stage2_extended_seed_pilot')


def export_tree(tree):
    tree.mkdir(parents=True, exist_ok=True)
    # Re-export tracked files so a previous diagnostic cannot change the recipe.
    with subprocess.Popen(['git', 'archive', COMMIT], cwd=REPO,
                          stdout=subprocess.PIPE) as archive:
        subprocess.run(['tar', '-x', '-C', str(tree)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        assert archive.wait() == 0


def report_values(text, stage2=False):
    rows = [line.split('|')[1:-1] for line in text.splitlines() if line.startswith('|')]
    values = {}
    for cells in rows:
        if cells[0].strip() in ('held-out emotion AMI', 'held-out genre AMI'):
            values[cells[0].strip()] = float(cells[2].strip())
        if stage2 and cells[0].strip() == '42':
            values['seed42_macro_auc'] = float(cells[2].strip())
    return values


def compare(actual, expected, decimals=4):
    return dict(status='PASS' if actual is not None and
                f'{actual:.{decimals}f}' == f'{expected:.{decimals}f}' else 'FAIL',
                expected=expected, actual=actual)


def run_command(name, command, folder, cwd, env):
    start = time.monotonic()
    with (folder / f'{name}.log').open('w') as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    return dict(status='PASS' if result.returncode == 0 else 'FAIL',
                expected=0, actual=result.returncode, seconds=time.monotonic() - start)


def run(outdir, smoke=False):
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pending = outdir / 'stage0.pending.json'
    pending.write_text(json.dumps(dict(
        status='FAIL', checks={}, fix_tests='not run', reason='stage 0 in progress')) + '\n')
    pending.replace(outdir / 'stage0.json')
    temporary = outdir / 'tmp'
    temporary.mkdir(exist_ok=True)
    tree = (outdir / 'sept_stage0') if smoke else HERE / 'sept_stage0'
    export_tree(tree)
    folder = tree / 'stage0_outputs' / str(time.time_ns())
    folder.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tree / PILOT), str(tree))),
               PERCEPT_OUTPUT_ROOT=str(folder), PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(temporary), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    checks = {}
    for index, script in enumerate(SCRIPTS):
        command = [sys.executable, '-u', str(tree / PILOT / f'{script}.py')]
        if smoke:
            command = [sys.executable, '-c',
                       f'import {script} as m; assert callable(m.main)']
        checks[f'september_{index + 1}'] = run_command(
            script, command, folder, tree, env)
        if not smoke:
            path = folder / f'{script.removeprefix("run_")}_report.md'
            values = report_values(path.read_text(), stage2=index == 1) if path.exists() else {}
            for key, expected in [('held-out emotion AMI', .1225),
                                  ('held-out genre AMI', .2274)]:
                checks[f'september_{index + 1}_{key}'] = compare(values.get(key), expected)
            if index == 1:
                checks['seed42_macro_auc'] = compare(values.get('seed42_macro_auc'), .8296)
    local_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                     OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TMPDIR=str(temporary))
    checks['row2_command'] = run_command('row2', [sys.executable, str(HERE / 'pc_score.py'),
        '--check-row2', '--outdir', str(outdir)], folder, REPO, local_env)
    row2 = outdir / 'row2_check.json'
    result = json.loads(row2.read_text()) if row2.exists() else {}
    for aspect, expected in [('emotion', .08855639551158387),
                             ('style', .17095358715401665), ('genre', .3930761231906966)]:
        actual = result.get('both_mean', {}).get(aspect)
        checks[f'row2_{aspect}'] = dict(status='PASS' if actual is not None and
            abs(actual - expected) <= 1e-6 else 'FAIL', expected=expected, actual=actual)
    checks['fix_tests'] = run_command('fix_tests', [sys.executable, '-m', 'pytest', '-q',
        str(HERE / 'test_pc_fixes.py')], folder, REPO, local_env)
    lines = (folder / 'fix_tests.log').read_text().splitlines()
    summary = [line for line in lines if re.search(r'\b(?:passed|failed|error)\b', line)]
    passed = all(check['status'] == 'PASS' for check in checks.values())
    result = dict(status=('SMOKE' if smoke else 'PASS') if passed else 'FAIL',
                  checks=checks, fix_tests=summary[-1] if summary else 'no pytest result line',
                  smoke=smoke, commit=COMMIT, reports=str(folder),
                  smoke_scope='September imports only; no numerical regression' if smoke else None)
    pending = outdir / 'stage0.pending.json'
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    pending.replace(outdir / 'stage0.json')
    print(result['status'], flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=Path, help='isolated import/check smoke root')
    args = parser.parse_args()
    return run(args.smoke if args.smoke is not None else HERE / 'outputs',
               smoke=args.smoke is not None)


if __name__ == '__main__':
    raise SystemExit(main())
