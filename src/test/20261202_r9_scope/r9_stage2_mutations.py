"""Ticket 13 review mutation replay. Only scratch copies under /tmp are edited.

Initialize once, then run small batches (each pytest process has a 180s limit):
  python r9_stage2_mutations.py --init
  python r9_stage2_mutations.py --scratch /tmp/r9-t13-mutations-... --mutants m01 m02
No recipe step or feature scoring is executed. Both authorized test files run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REL = HERE.relative_to(ROOT)
DRIVER = 'run_r9_stage2.py'
PINS = 'r9_pins.py'
PY = '/root/miniconda3/envs/CoSiR/bin/python'

# Each replacement is applied to pristine source, once. The review's mutations
# are adapted to the fixed call sites (not to old line numbers).
MUTATIONS = {
    'm01': (DRIVER, "require(bool(diff), f'made {key} equals CLIP')", "pass"),
    'm02': (DRIVER, "require(not diff, f'copied {key} differs from CLIP: {diff}')", "pass"),
    'm03': (DRIVER, 'assert_array_provenance(a, b, [], keys)', 'pass'),
    'm04': (DRIVER, "for role in sorted(BB.FROZEN_ROLES):\n        assert_frozen(role, parts[role], BB.CLIP_PARTS[role])", 'pass'),
    'm05': (DRIVER, "require(want and path.is_file() and record.get('script_sha256') == want == R.sha256_file(path),", 'require(True,'),
    'm06': (DRIVER, 'script = assert_script(rec, stage0b)', "script = rec['script_sha256']"),
    'm07': (DRIVER, "for role in BB.RECIPE_ROLES:\n        assert_script(read_json(str(paths[role]) + '.record.json'), gate)", 'pass'),
    'm08': (DRIVER, 'keys = assert_part(role, path, BB.CLIP_PARTS[role], rec, replay=replay, parts=parts)',
            "keys = {'keys_made': rec['keys_made'], 'keys_copied': rec['keys_copied'], 'keys_tested_equal': rec['keys_copied'], 'keys_tested_differ': rec['keys_made']}"),
    'm09': (DRIVER, 'prov = provenance(backbone, paths, stage0b, replay=dry_run)', 'prov = provenance(backbone, paths, stage0b, replay=True)'),
    'm10': (DRIVER, "require(stage.get('passed') is True and stage.get('backbone') == backbone,", 'require(True,'),
    'm11': (DRIVER, "require(read_json(parts['picks_seed42']).get('passed') is True, 'picks file has not passed')", 'pass'),
    'm12': (DRIVER, "require(path.is_file() and Path(str(path) + '.record.json').is_file(), f'{role}: file or record missing')", 'pass'),
    'm13': (PINS, 'if backbone != "clip_b32" or dry_run:', 'if False:'),
    'm14': (DRIVER, "require(stage['dry_run'] is dry_run, 'dry-run stage 2 cannot authorise a real backbone')", 'pass'),
    'm15': (DRIVER, "require(path_record(paths[role]) == {k: p[k] for k in ('path', 'sha256')}, f'{role}: stage 2 file changed')", 'pass'),
    'm16': (DRIVER, 'if dry_run:\n        # Computation and exact component equality', 'if True:\n        # Computation and exact component equality'),
    'm17': (DRIVER, 'if dry_run:\n        clip_exact(seed, sigma, arrays)', 'if dry_run:\n        pass'),
    'm18': (DRIVER, "require(a.backbone == 'clip_b32' and a.step in ('record', 'pins') and a.replay and a.out,", "require(a.step in ('record', 'pins') and a.replay and a.out,"),
    'm19': (DRIVER, "args = ['flock', '-n', '-o', '-E', '75', '/tmp/gpu0.lock', *args]", 'args = args'),
    'm20': (DRIVER, "CUDA_VISIBLE_DEVICES='0' if step in ('groups', 'a3') else ''", "CUDA_VISIBLE_DEVICES='0'"),
    'm21': (DRIVER, 'if code:\n                return code', 'if code:\n                pass'),
    'm22': (DRIVER, "require(path_record(path) == entry['file'], f'{role}: output changed since it was made')", 'pass'),
    'm23': (DRIVER, "with path.open('x') as f:", "with path.open('w') as f:"),
    'm24': (DRIVER, "EXEMPT = frozenset(('picked', 'picks_seed42', 'cells_seed42', 'baselines_seed42'))", "EXEMPT = frozenset(('picked', 'picks_seed42', 'cells_seed42', 'baselines_seed42', 'rc_tau'))"),
    'm25': (DRIVER, "copied = [k for k in a if k.endswith('/classes_')]", "copied = [k for k in a if k.endswith(('/classes_', '/coef_'))]"),
    'm26': (DRIVER, "S.CELLS = S.cells_from_record(read_json(paths['cells_seed42']))", 'pass'),
    'm27': (DRIVER, "require(np.isfinite(value) and value >= 0, f'{c}: invalid {key}')", 'pass'),
    'm28': (DRIVER, 'import r6_stats as ST', 'import r6_stats as ST\n    import r6_bundle'),
    'm29': (DRIVER, "require(np.isnan(a[k]).all(), f'{role}/{k}: unused head must be NaN')", 'pass'),
    'm30': (DRIVER, "require(features == stores, f'{role}: feature-store provenance is incomplete')", 'pass'),
    'm31': (DRIVER, "'status': 'not run: switched off' if cells['switched_off'] else 'ready'", "'status': 'ready'"),
    'm32': (DRIVER, "require(sha == R.INPUT_SHA256[rel], f'{rel}: frozen input SHA differs')", 'pass'),
    'm33': (DRIVER, "require(stage.get('runner_sha256') == R.sha256_file(__file__) and stage.get('rule_sha256') == R.RULE_SHA256,", 'require(True,'),
    # Additional probes for the guards introduced by this fix round.
    'm34': (DRIVER, "require(role in pins and sha == pins[role]['sha256'], f'frozen {role} differs from pins_clip.json')", 'pass'),
    'm35': (DRIVER, 'assert_modules(read_json(a.stage0b))', 'pass'),
    'm36': (DRIVER, 'assert_modules(stage0b)\n    stores = store_shas(backbone)', 'pass\n    stores = store_shas(backbone)'),
    'm37': (DRIVER, 'modules = assert_modules(gate)\n    require(stage[', 'modules = recipe_module_shas()\n    require(stage['),
    'm38': (DRIVER, "modules = assert_modules(gate)\n    script = STEP_SCRIPTS[step]", "modules = recipe_module_shas()\n    script = STEP_SCRIPTS[step]"),
    'm39': (DRIVER, 'for key in copied:\n            require(key in a', 'for key in []:\n            require(key in a'),
    'm40': (DRIVER, "metadata = [k for k in a if k.endswith(('/n_iter_', '/n_samples_seen_'))]", 'metadata = []'),
    'm41': (DRIVER, "require(means == read_json(paths['picks_seed42'])['mean_r1'], 'picks file mean_r1 differs from seed-42 means')", 'pass'),
    # Second fix: the re-review's three surviving mutants and replay guards.
    'o5': (DRIVER, 'check_receipts(backbone, before=step, module_sha256=modules)', 'pass'),
    'o6': (DRIVER, "stage0b.get('module_sha256', {}).get(module) == sha", "stage0b.get('module_sha256', {}).get(module, sha) == sha"),
    'o8': (DRIVER, 'if not dry_run:\n        check_receipts(backbone, module_sha256=modules)\n    prov =', 'if not dry_run:\n        pass\n    prov ='),
    'n1a': (DRIVER, "if replay and message == 'logical input changed' else (p,)", 'if False else (p,)'),
    'n1b': (DRIVER, "if replay and message == 'logical input changed' else (p,)", "if message == 'logical input changed' else (p,)"),
    'n1c': (DRIVER, 'inputs = _input_files(rec, parts, replay=replay)', 'inputs = _input_files(rec, parts, replay=False)'),
    'n1d': (DRIVER, "_input_files(read_json(str(paths[role]) + '.record.json'), paths, replay=dry_run)", "_input_files(read_json(str(paths[role]) + '.record.json'), paths, replay=False)"),
    'n2a': (DRIVER, 'elif replay and value == file_hashes[role]:', 'elif False:'),
    'n2b': (DRIVER, 'elif replay and value == file_hashes[role]:', 'elif value == file_hashes[role]:'),
    'n2c': (DRIVER, 'skipped_self_inputs[role][key] = value', 'pass'),
}


def copy_sources(scratch):
    (scratch / 'pytest').mkdir(parents=True, exist_ok=True)
    dest = scratch / REL
    dest.mkdir(parents=True, exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ('.py', '.json', '.md', '.sh'):
            shutil.copyfile(p, dest / p.name)
    # git discovery only; no git command mutates this shared metadata.
    (scratch / '.git').write_bytes((ROOT / '.git').read_bytes())
    return dest


def run(scratch, names):
    scratch = scratch.resolve()
    if not scratch.is_relative_to('/tmp'):
        raise ValueError('mutation copies must stay under /tmp')
    results_path = scratch / 'mutation_results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in names:
        dest = copy_sources(scratch)
        source_shas = {n: hashlib.sha256((dest / n).read_bytes()).hexdigest()
                       for n in (DRIVER, PINS, 'test_r9_stage2.py', 'test_r9_pins.py')}
        file, old, new = MUTATIONS[name]
        path = dest / file
        original = path.read_text()
        if original.count(old) != 1:
            raise ValueError(f'{name}: expected one replacement, got {original.count(old)}')
        mutated = original.replace(old, new, 1)
        compile(mutated, str(path), 'exec')
        path.write_text(mutated)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='')
        args = [PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--tb=short',
                '--basetemp', str(scratch / 'pytest' / name), str(dest / 'test_r9_stage2.py'), str(dest / 'test_r9_pins.py')]
        try:
            test = subprocess.run(args, cwd=scratch, env=env, capture_output=True, text=True, timeout=180)
            log = test.stdout + test.stderr
            summary = [line for line in log.splitlines() if ' passed' in line or ' failed' in line or 'ERROR ' in line]
            valid_failure = test.returncode == 1 and any(' failed' in s and ' errors' not in s for s in summary)
            outcome = 'survived' if test.returncode == 0 else 'caught' if valid_failure else 'invalid'
            results[name] = {'result': outcome, 'returncode': test.returncode,
                             'summary': summary[-1] if summary else log[-500:], 'source_sha256': source_shas}
            (scratch / f'{name}.log').write_text(log)
        finally:
            path.write_text(original)  # never leave even the scratch copy mutated
        results_path.write_text(json.dumps(results, indent=1) + '\n')
        print(name, results[name]['result'], results[name]['summary'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--init', action='store_true')
    ap.add_argument('--scratch', type=Path)
    ap.add_argument('--mutants', nargs='+', choices=tuple(MUTATIONS))
    a = ap.parse_args()
    if a.init:
        scratch = Path(tempfile.mkdtemp(prefix='r9-t13-mutations-'))
        copy_sources(scratch)
        print(scratch)
    else:
        if not a.scratch or not a.mutants:
            ap.error('--scratch and --mutants are required')
        run(a.scratch, a.mutants)


if __name__ == '__main__':
    main()
