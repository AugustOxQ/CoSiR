"""Focused fx-inf source mutants; run sequentially, restore every source byte in finally.

Run with the ticket's Python/env prefix and optional --out fresh-results.json.
Results stay beside this script; no input
or frozen module is edited. Every subprocess has a 230-second deadline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
P, H, A, D = ('run_r9_preread.py', 'run_r9_held.py', 'run_r9_apply_rule.py', 'run_r9_descriptive.py')
TP = 'test_r9_preread.py::'
ORDER = 'for kind in ("heldfix1", "fix3", "fix2", "fix1", "crash1"):'
NO_FIX2 = 'for kind in ("heldfix1", "fix3", "fix1", "crash1"):'
STALE = TP + 'test_newer_generation_blocks_older_pass[stale-fix2]'
MUTANTS = [
    ('descriptive fix reverted', D, 'json.dumps(report_plain(report), indent=2',
     'json.dumps(RH._plain(report), indent=2',
     'test_r9_descriptive.py::test_report_writes_infinite_frozen_lambda[H6-True]'),
    ('apply selection omits fix2', A, ORDER, NO_FIX2,
     TP + 'test_numbered_preread_publishes_and_launches[H8-2]'),
    ('held selection omits fix2', H, ORDER, NO_FIX2,
     TP + 'test_numbered_preread_publishes_and_launches[H8-2]'),
    ('launch selection omits fix2', P, "for k in ('fix3', 'fix2', 'fix1', 'crash1')",
     "for k in ('fix3', 'fix1', 'crash1')",
     TP + 'test_numbered_preread_publishes_and_launches[H8-2]'),
    ('apply accepts stale fix2', A,
     '    mods = rec.get("module_sha256") if isinstance(rec.get("module_sha256"), dict) else {}\n',
     '    mods = rec.get("module_sha256") if isinstance(rec.get("module_sha256"), dict) else {}\n'
     '    if path.name == "smoke_record_fix2.json": mods = CM.r9_module_shas(SHA_HERE)\n', STALE),
    ('held accepts stale fix2', H,
     "_refuse_unless(rec.get('module_sha256') == R.r9_module_shas(here),",
     "_refuse_unless(path.name == 'smoke_record_fix2.json' or rec.get('module_sha256') == R.r9_module_shas(here),", STALE),
    ('launch accepts stale fix2', P,
     "require(rec.get('module_sha256') == R.r9_module_shas(here)\n",
     "require((path.name.endswith('_fix2.json') or rec.get('module_sha256') == R.r9_module_shas(here))\n", STALE),
    ('publish accepts stale fix2', P,
     "require(old.get('passed') is True and old.get('module_sha256') == record['module_sha256']\n"
     "                    and old.get('runner_sha256') == record['runner_sha256'],",
     "require(kind == 'fix2' or (old.get('passed') is True and old.get('module_sha256') == record['module_sha256']\n"
     "                    and old.get('runner_sha256') == record['runner_sha256']),",
     TP + 'test_publish_refuses_stale_generation[module_sha256-value0-committed-fix2]'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=HERE / 'fx_inf_mutation_results.json')
    out = ap.parse_args().out
    assert out.resolve().is_relative_to(HERE), 'results must stay inside the round folder'
    assert not out.exists(), 'mutation results already exist; use a fresh --out'
    results = []
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8',
           'MKL_NUM_THREADS': '8', 'CUDA_VISIBLE_DEVICES': ''}
    for name, filename, old, new, test in MUTANTS:
        path = HERE / filename
        raw = path.read_bytes()
        original_sha = hashlib.sha256(raw).hexdigest()
        source = raw.decode()
        assert source.count(old) == 1, (name, 'source anchor')
        try:
            path.write_text(source.replace(old, new))
            proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider',
                                   test, '--tb=short'], cwd=HERE, env=env,
                                  capture_output=True, text=True, timeout=230)
        finally:
            path.write_bytes(raw)
            assert path.read_bytes() == raw, (name, 'restoration')
        failure = next((line for line in proc.stdout.splitlines() if line.startswith('FAILED ')), '')
        caught = proc.returncode == 1 and test.split('::')[1] in failure
        if filename == D:
            caught = caught and 'Out of range float values are not JSON compliant: inf' in proc.stdout
        results.append({'name': name, 'file': filename, 'test': test, 'caught': caught,
                        'exit': proc.returncode, 'failure': failure,
                        'restored': True, 'source_sha256': original_sha})
        print(f'{name}: caught={caught}, restored=True', flush=True)
    with out.open('x') as output:
        output.write(json.dumps(results, indent=2) + '\n')
    return 0 if all(result['caught'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
