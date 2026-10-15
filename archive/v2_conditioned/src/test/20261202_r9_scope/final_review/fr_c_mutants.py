"""Final review C: temporary source mutants of the pre-read and launch gate; each restored in finally.

Usage: fr_c_mutants.py <first> <last> <out.json>. Runs test_r9_preread.py (whole file) per mutant;
caught = pytest exit != 0. Source SHA-256 is checked equal to the original after every mutant.
"""
import hashlib, json, subprocess, sys, os
from pathlib import Path

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[2]
PY = '/root/miniconda3/envs/CoSiR/bin/python'
TMP = '/tmp/claude-0/-project-CoSiR/18057a5a-4e60-48cd-83b0-ff14c8a319ae/scratchpad/fr_c'
P, H, C = 'run_r9_preread.py', 'run_r9_held.py', 'rederive/rd9_compare.py'
MUT = [
    # repeats of t22's guard deletions
    ('del:preread_stage0a_modules', P, "require(recorded == current,", "require(True,"),
    ('del:preread_stage0b', P, "S2.assert_modules(gate)  # guard:preread_stage0b", "pass"),
    ('del:preread_episode_hashes', P, "require(ident['passed'] is True,", "require(True,"),
    ('del:preread_sigma', P, "require(sigma == source['sigma'],", "require(True,"),
    ('del:preread_mutation', P, "require(fired, 'wiring", "require(True, 'wiring"),
    ('del:preread_bytes', P, "require(R.r9_module_shas() == modules and all(R.sha256_file(p) == s for p, s in evidence.items()),", "require(True,"),
    ('del:preread_launch(held call)', H, "        preread_guard(results, read, variant, after_crash, here)  # guard:preread_launch\n", ""),
    ('del:preread_real_inputs', P, "RH.setup_a28(state.env, state.backbone, smoke=False, inputs_only=True)  # guard:preread_real_inputs", "pass"),
    ('del:preread_phase1_pins', P, "require(agr.get('pins_sha256') == R.sha256_file(R.pins_path(state.backbone)),", "require(True,"),
    # sub-condition mutants (not in t22's table)
    ('sub:launch module_sha256 half', P, "require(rec.get('module_sha256') == R.r9_module_shas(here)\n            and rec.get('runner_sha256')", "require(rec.get('runner_sha256')"),
    ('sub:launch read field', P, "and rec.get('read') == read\n", "\n"),
    ('sub:launch backbone field', P, "and rec.get('backbone') == RH.READ_BACKBONES[read]\n", "\n"),
    ('sub:launch all checks passed', P, "and all(rec.get('checks', {}).get(k, {}).get('status') == 'passed' for k in CHECK_ORDER),", ","),
    ('sub:launch non-empty inputs', P, "require(bool(rec.get('input_file_sha256'))\n            and all(", "require(all("),
    ('sub:stage2 switched_off', P, "and stage.get('switched_off') is False and stage.get('dry_run') is False", "and stage.get('dry_run') is False"),
    ('sub:stage2 dry_run', P, " and stage.get('dry_run') is False,", ","),
    ('sub:stage2 backbone', P, "stage.get('passed') is True and stage.get('backbone') == backbone", "stage.get('passed') is True"),
    ('sub:regression fields seed only', P, "fields = ('seed', 'mean_r1', 'checks', 'arrays', 'episodes_sha256', 'coef_sha256')", "fields = ('seed',)"),
    ('sub:phase1 phase==1', P, "and agr.get('phase') == 1 ", ""),
    ('sub:phase1 backbone', P, "and agr.get('backbone') == state.backbone\n", "\n"),
    ('sub:phase1 items equal', P, "\n            and all(v.get('equal') is True for v in items),", ","),
    ('sub:phase1 failed==0', P, "and agr.get('failed') == 0 ", ""),
    ('sub:stage0a runner', P, "require(reg.get('runner_sha256') == S0A.runner_sha256(),", "require(True,"),
    ('sub:clip regression items only', P, "require(len(items) == S0A.N_ITEMS and all(v['equal'] is True for v in items.values())\n                and not S0A.compare_counts(counts, refs[S0A.COUNTS_NAME]),", "require(not S0A.compare_counts(counts, refs[S0A.COUNTS_NAME]),"),
    ('sub:clip regression counts only', P, "\n                and not S0A.compare_counts(counts, refs[S0A.COUNTS_NAME]),", ","),
    ('sub:smoke_guard call in publish', P, "    AR.smoke_guard(Path(results), 'reserve' if kind == 'reserve' else None)\n", ""),
    ('sub:heads not run for clip arm L', P, "RH.head_guard(env, RH.refit_reference(state.backbone))  # guard:preread_arm_l", "pass"),
    ('sub:phase1 independent_identity', P, "require(agr.get('independent_identity') == {", "require(True or agr.get('independent_identity') == {"),
    # comparer (C13) edits of ticket 22
    ('del:preread_compare_pins', C, "    assert own['inputs'].get(str(pins_path),{}).get('sha256')==sha_file(pins_path), 'phase-1 pins binding missing'  # guard:preread_compare_pins\n", ""),
    ('del:preread_compare_sources', C, "    verify_reads(own['inputs'])  # guard:preread_compare_sources\n", ""),
    ('sub:compare new-schema floor cl/pair/parity/cells', C, "        required|={'cl','pair_index','parity','aff_fused_cells','aff_cf_cells'}\n", ""),
    ('sub:compare new-schema floor reader arrays', C, "        required|={f'{prefix}__{side}' for prefix in ('aff_gate','P','margin','pick') for side in ('a','b')}\n", ""),
    ('sub:compare clip B/B0/B1 means', C, "                c.value('seed42.mean_r1.'+name,got,picked.get('mean_r1',{}).get(name))\n", "                pass\n"),
    ('sub:compare new-schema only derived keys', C, "            if name in arrays: hashes[name]=entry", "            if name in arrays and name.startswith('aff_fused__r'): hashes[name]=entry"),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a, b, out):
    res = []
    for i, (name, rel, old, new) in enumerate(MUT[a:b + 1], start=a):
        path = F / rel
        orig = path.read_bytes()
        h0 = sha(path)
        text = orig.decode()
        n = text.count(old)
        if n != 1:
            res.append(dict(i=i, name=name, error=f'anchor count {n}'))
            continue
        try:
            path.write_text(text.replace(old, new))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='')
            r = subprocess.run([PY, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', *os.environ.get('FRC_TEST', str(F / 'test_r9_preread.py')).split(),
                                '--basetemp', f'{TMP}/pm{i}'], cwd=ROOT, env=env, capture_output=True, text=True, timeout=200)
            tail = [l for l in r.stdout.splitlines() if l.startswith(('FAILED', 'ERROR')) or ' passed' in l or ' failed' in l][-3:]
            res.append(dict(i=i, name=name, file=rel, exit=r.returncode, caught=r.returncode != 0, tail=tail))
        finally:
            path.write_bytes(orig)
            assert sha(path) == h0
        print(i, name, res[-1].get('caught'), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
