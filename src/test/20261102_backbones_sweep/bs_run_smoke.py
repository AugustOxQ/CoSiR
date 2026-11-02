"""Smoke of bs_run.py (ticket 12a). Plumbing only, CPU, nothing committed in the real checkout:

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 python bs_run_smoke.py <scratch dir>

Phases (each timed): (1) real steps: `stage0 --only x xi` on the real r13/r12 inputs (CPU, scratch OUT); (2) stubbed steps
(BS_RUN_STUB) with failing gates: pending branch, resolve, failure propagation to status.json, part a states; (3) stage 1 and
stage 3 wiring on synthetic stage 1 files: record written, refusal on a changed file; (4) git: in a throwaway repo
(`*.json` ignored), `commit` adds with -f and scoped, `require` refuses before and passes after; (5) `diag` without bs_diag.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

scratch = Path(sys.argv[1]).resolve()
scratch.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
PY = sys.executable
BASE = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', BS_RUN_NO_COMMIT='1')
GIT = 'git'


def run(tag, out, args, env=None, expect=0):
    e = dict(BASE, BS_OUTPUTS=str(out), BS_RECORDS=str(out / 'records'), **(env or {}))
    t = time.time()
    r = subprocess.run([PY, str(HERE / 'bs_run.py'), *args], env=e, capture_output=True, text=True, cwd=str(HERE))
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f'[{tag}] rc={r.returncode} seconds={time.time() - t:.1f}  {tail[-1] if tail else ""}', flush=True)
    assert r.returncode == expect, (tag, r.returncode, '\n'.join(tail[-8:]))
    return r.stdout + r.stderr


def jl(p):
    return json.loads(Path(p).read_text())


# (1) real gates x and xi
o1 = scratch / 'real'
run('real stage0 x xi', o1, ['stage0', '--only', 'x', 'xi'])
g = jl(o1 / 'stage0/gate.json')
assert g['x']['ok'] and g['xi']['ok'], g

# (2) stubbed stage 0
o2 = scratch / 'stub'
(o2 / 'node').mkdir(parents=True, exist_ok=True)
for enc, done in (('siglip2', [f'{k}__siglip2__s{s}' for s in range(5) for k in ('cstar', 'xphoto')]),
                  ('siglip1', ['cstar__siglip1__s0', 'xphoto__siglip1__s0', 'cstar__siglip1__s1']),
                  ('clip_l14', [f'{k}__clip_l14__s{s}' for s in range(5) for k in ('cstar', 'xphoto')])):
    ok = len(done) == 10
    (o2 / f'node/part_a_{enc}.json').write_text(json.dumps(dict(encoder=enc, status='ok' if ok else 'failed', done=done)))
run('stub stage0 (vii-sha, v fail)', o2, ['stage0'], dict(BS_RUN_STUB='1', BS_RUN_FAIL='vii-sha,v'), expect=1)
assert not (o2 / 'records/status.json').exists(), 'status.json must wait for stage 2'
(o2 / 'stage2').mkdir(parents=True, exist_ok=True)
(o2 / 'stage2/stage2_status.json').write_text(json.dumps(dict(status='ok')))
run('status with pending branch', o2, ['status'], expect=1)
run('resolve vii-sha=ok', o2, ['stage0', '--resolve', 'vii-sha=ok'], dict(BS_RUN_STUB='1'), expect=1)
st = jl(o2 / 'records/status.json')
print(json.dumps(st))
assert st['encoders']['siglip2'] == dict(cstar_runs='ok', photo_runs='ok', stage0=dict(paintings='ok', photos='failed'))
assert st['encoders']['siglip1']['cstar_runs'] == 'failed' and st['encoders']['siglip1']['photo_runs'] == 'failed'
assert st['part_c'] == dict(stage0='ok', stage2='ok')

# (3) stage 1 and stage 3 wiring (synthetic stage 1 files; scoring steps stubbed)
(o2 / 'stage1').mkdir(exist_ok=True)
for enc in ('siglip2', 'clip_l14'):
    for nm in ('artclu_ranges', 'artret_ranges', 'artret_picks', 'affection_ranges'):
        (o2 / f'stage1/{nm}_{enc}.json').write_text(json.dumps(dict(encoder=enc, name=nm)))
run('stub stage1', o2, ['stage1'], dict(BS_RUN_STUB='1'))
assert (o2 / 'records/stage1_siglip2.json').exists() and not (o2 / 'records/stage1_siglip1.json').exists()
(o2 / 'stage3').mkdir(exist_ok=True)
(o2 / 'stage3/outcomes.json').write_text('{}')
(o2 / 'records/stage2_emoset.json').write_text('{}')
run('stub stage3', o2, ['stage3', '--smoke'], dict(BS_RUN_STUB='1'))
assert (o2 / 'records/verdicts.json').exists()
(o2 / 'stage1/artret_picks_siglip2.json').write_text(json.dumps(dict(encoder='siglip2', name='changed')))
out = run('stage3 with changed stage 1 file', o2, ['stage3'], dict(BS_RUN_STUB='1'), expect=1)
assert 'REFUSED' in out

# (4) git: add -f, scoped commit, guard (throwaway repo)
sys.path.insert(0, str(HERE))
os.environ.update(BS_OUTPUTS=str(scratch / 'git_out'))
import bs_paths as BP  # noqa: E402
import bs_run as R  # noqa: E402


def g_(repo, *a):
    return subprocess.run([GIT, '-C', str(repo), *a], capture_output=True, text=True, check=True).stdout


repo = Path(tempfile.mkdtemp(dir=scratch))
g_(repo, 'init', '-q')
(repo / '.gitignore').write_text('*.json\n')
g_(repo, 'config', 'user.email', 'a@b')
g_(repo, 'config', 'user.name', 'smoke')
(repo / 'records').mkdir()
(repo / 'other.json').write_text('{}')
g_(repo, 'add', '-f', 'other.json')
g_(repo, 'commit', '-qm', 'init')
(repo / 'other.json').write_text('{"dirty": 1}')        # an unrelated change must stay out of the commit
BP.REPO = repo
R.NO_COMMIT = False
f = repo / 'records/stage1_x.json'
f.write_text('{"a": 1}\n')
assert R.committed(f), 'must refuse before commit'
try:
    R.require(f, 'stage 3')
    raise AssertionError('require passed on an uncommitted record')
except SystemExit as e:
    assert 'REFUSED' in str(e)
R.commit([f], 'smoke record')
assert R.committed(f) == ''
names = g_(repo, 'show', '--name-only', '--format=', 'HEAD').split()
assert names == ['records/stage1_x.json'], names
f.write_text('{"a": 2}\n')
assert 'differs' in R.committed(f)
print('[git] add -f, scoped commit, guard: ok', flush=True)

# (5) diag without bs_diag
r = subprocess.run([PY, str(HERE / 'bs_run.py'), 'diag', '--smoke'], env=dict(BASE, BS_OUTPUTS=str(scratch / 'diag')),
                   capture_output=True, text=True, cwd=str(HERE))
print(f'[diag] rc={r.returncode} {(r.stdout + r.stderr).strip().splitlines()[-1]}')
print('SMOKE OK')
