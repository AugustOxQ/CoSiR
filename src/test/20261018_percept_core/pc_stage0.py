"""Check the repeatable September artifacts and source copies (A1)."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from pc_fixes import unit_tests_pass
from pc_stage1 import one_thread

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COMMIT = '1b70d7f'
RECORD_COMMITS = ('3b3ce61', '4e4480b')
PILOT = 'src/test/20260922_percept_topic_pipeline/'
BUDDY = 'src/test/20260923_artelingo_buddy_analysis/'
SNAPSHOT = REPO / 'res/percept/percept_stage1_embedding_snapshot.npz'
SOURCE_MAP = {
    **dict.fromkeys(('PRETRAIN_BATCH_SIZE', 'PRETRAIN_EPOCHS', 'MAX_DEC_EPOCHS',
                     'STABILITY_THRESHOLD', 'PRETRAIN_LEARNING_RATE', 'DEC_LEARNING_RATE',
                     'build_autoencoder', 'pretrain_autoencoder', 'soft_assignments',
                     'target_distribution'), 'run_percept_stage1_pilot.py'),
    **dict.fromkeys(('LAMBDA_RECONSTRUCTION', 'LAMBDA_BALANCE',
                     'initialize_cluster_centers', 'train_dec_until_stable'),
                    'run_percept_stage1_cluster_count_sweep_pilot.py'),
    **dict.fromkeys(('N_SURVIVING_CLUSTERS', 'AttentionPoolingMapper', 'evaluate_auc',
                     'auc_summary'), 'run_percept_stage2_sweep_pilot.py'),
}
ALIASES = ('MAX_DEC_EPOCHS', 'STABILITY_THRESHOLD', 'DEC_LEARNING_RATE')
COPY_IMPORTS = ('import sys', 'import numpy as np', 'import torch',
                'from sklearn.cluster import KMeans', 'from sklearn.metrics import roc_auc_score',
                'from torch import nn', 'from torch.nn import functional as F')
SNAPSHOT_EXPECTED = dict(emotion=.12246902855745419, genre=.22741920423971046,
                         heldout_paintings=9365, genre_rows=159)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO).decode()


def python_blobs(commit):
    """A superset of the pilots' import closure, without importing September."""
    blobs = {}
    for entry in git('ls-tree', '-r', '-z', commit, '--', 'src/').split('\0'):
        if not entry:
            continue
        metadata, path = entry.split('\t')
        if path.endswith('.py') and (not path.startswith('src/test/') or
                                    path.startswith((PILOT, BUDDY))):
            _, kind, blob = metadata.split()
            assert kind == 'blob', path
            blobs[path] = blob
    assert blobs, commit
    return blobs


def compare_blobs(record, reference):
    """New reference files are allowed; every record blob must still exist."""
    differences = sorted(path for path, blob in record.items() if reference.get(path) != blob)
    return dict(status='FAIL' if differences else 'PASS', expected=0,
                actual=len(differences), files_compared=len(record), differences=differences)


def source_segments(source, strict=False):
    """Extract exact named segments; validate the copied module's structure."""
    segments, imports, differences = {}, [], []
    lines = source.splitlines(keepends=True)
    for index, node in enumerate(ast.parse(source).body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            invalid = [type(target).__name__ for target in targets
                       if not isinstance(target, ast.Name)]
            if invalid:
                differences.append(f'non-name assignment target at line {node.lineno}: {invalid}')
                continue
            names = [target.id for target in targets]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(source, node))
            continue
        elif (index == 0 and isinstance(node, ast.Expr) and
              isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        else:
            if strict:
                differences.append(f'top-level {type(node).__name__} at line {node.lineno}')
            continue
        segment = ast.get_source_segment(source, node)
        if getattr(node, 'decorator_list', []):
            first = next(line for line in range(node.decorator_list[0].lineno - 1, -1, -1)
                         if lines[line].startswith('@'))
            segment = ''.join(lines[first:node.lineno - 1]) + segment
        for name in names:
            if name in segments:
                differences.append(f'duplicate name {name} at line {node.lineno}')
            else:
                segments[name] = segment
    if strict and tuple(imports) != COPY_IMPORTS:
        differences.append('imports')
    return segments, differences


def check_source_copies(source):
    copied, differences = source_segments(source, strict=True)
    if copied.pop('base', None) != 'base = sys.modules[__name__]':
        differences.append('base')
    originals = {}
    for path in sorted(set(SOURCE_MAP.values())):
        originals[path], errors = source_segments(git('show', f'{COMMIT}:{PILOT}{path}'))
        differences.extend(f'{path}: {error}' for error in errors)
    differences.extend(sorted(set(copied) ^ set(SOURCE_MAP)))
    for name in sorted(set(copied) & set(SOURCE_MAP)):
        if copied[name] != originals[SOURCE_MAP[name]].get(name):
            differences.append(name)
    sweep = originals['run_percept_stage1_cluster_count_sweep_pilot.py']
    for name in ALIASES:
        if sweep.get(name) != f'{name} = base.{name}':
            differences.append(f'sweep alias {name}')
    return dict(status='FAIL' if differences else 'PASS', expected=0,
                actual=len(differences), differences=differences)


def score_snapshot(snapshot):
    emotion, topic, genre = (snapshot[f'heldout_{name}'] for name in ('emotion', 'topic', 'genre'))
    assert emotion.ndim == topic.ndim == genre.ndim == 1
    assert len(emotion) == len(topic) == len(genre) == len(snapshot['heldout_paintings'])
    mask = genre != ''
    return dict(emotion=float(adjusted_mutual_info_score(emotion, topic)),
                genre=float(adjusted_mutual_info_score(genre[mask], topic[mask])),
                heldout_paintings=len(topic), genre_rows=int(mask.sum()))


def check_snapshot(snapshot):
    actual = score_snapshot(snapshot)
    passed = all(abs(actual[name] - expected) <= (1e-6 if name in ('emotion', 'genre') else 0)
                 for name, expected in SNAPSHOT_EXPECTED.items())
    return dict(status='PASS' if passed else 'FAIL', expected=SNAPSHOT_EXPECTED, actual=actual)


def check_kmeans(snapshot):
    assert snapshot['n_initial_clusters'].item() == 60 and snapshot['seed'].item() == 42
    latents, labels = snapshot['train_latent_pre'], snapshot['train_initial_label']
    assert latents.shape == (61402, 128) and labels.shape == (len(latents),)
    start = time.monotonic()
    with one_thread():
        fitted = KMeans(n_clusters=60, n_init=10, random_state=42).fit(latents)
    differences = int(np.count_nonzero(fitted.labels_ != labels))
    return dict(status='FAIL' if differences else 'PASS', expected=0, actual=differences,
                seconds=time.monotonic() - start)


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
    folder = outdir / 'stage0_outputs' / str(time.time_ns())
    folder.mkdir(parents=True)
    reference = python_blobs(COMMIT)
    checks = {f'september_blobs_{commit}': compare_blobs(python_blobs(commit), reference)
              for commit in RECORD_COMMITS}
    checks['source_copies'] = check_source_copies((HERE / 'pc_sept.py').read_text())
    with SNAPSHOT.open('rb') as stream:
        snapshot_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    with np.load(SNAPSHOT, allow_pickle=True) as snapshot:
        checks['snapshot'] = check_snapshot(snapshot)
        checks['kmeans_labels'] = check_kmeans(snapshot)
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
    if not summary or not unit_tests_pass(summary[-1]):
        checks['fix_tests']['status'] = 'FAIL'
    passed = all(check['status'] == 'PASS' for check in checks.values())
    result = dict(status=('SMOKE' if smoke else 'PASS') if passed else 'FAIL',
                  checks=checks, fix_tests=summary[-1] if summary else 'no pytest result line',
                  smoke=smoke, commit=COMMIT, reports=str(folder),
                  snapshot_sha256=snapshot_sha256,
                  files_compared={commit: checks[f'september_blobs_{commit}']['files_compared']
                                  for commit in RECORD_COMMITS},
                  kmeans_seconds=checks['kmeans_labels']['seconds'],
                  kmeans_differing_labels=checks['kmeans_labels']['actual'])
    pending = outdir / 'stage0.pending.json'
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    pending.replace(outdir / 'stage0.json')
    print(result['status'], flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=Path, help='run all checks into an isolated output root')
    args = parser.parse_args()
    return run(args.smoke if args.smoke is not None else HERE / 'outputs',
               smoke=args.smoke is not None)


if __name__ == '__main__':
    raise SystemExit(main())
