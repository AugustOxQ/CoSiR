"""Smoke of bs_run.py (ticket 12a). Plumbing only, CPU, nothing committed in the real checkout:

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 python bs_run_smoke.py <scratch dir>

Phases (each timed): (1) real steps: `stage0 --only x xi` on the real r13/r12 inputs (CPU, scratch OUT); (2) stubbed steps
(BS_RUN_STUB) with failing gates: pending branch, resolve, failure propagation to status.json, part a states; (3) stage 1 and
stage 3 wiring on synthetic stage 1 files: record written, refusal on a changed file; (3b) the new order (final review fix
wave A): exports (steps, skip of valid files, skipped failed arms), stage 1 / 2 / 3 refusals without exports, stage 3 = cells
only + sha256 map, `decide` refusals (no c13 record, pass false, sha256 differs) and its pass; (4) git: in a throwaway repo
(`*.json` ignored), `commit` adds with -f and scoped, `require` refuses before and passes after (also the guard `decide` and the
stage 1 / 2 records use; the stubbed phases run with BS_RUN_NO_COMMIT, where the guard checks the file only), export_valid on real
small files, part_a_state with a photo-only failure; (5) `diag` without bs_diag. Not reachable without real data: the export CLIs
themselves (their own tests and the dry check run them) and bs_outcome decide (stubbed here).
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
def fake_exports(out, files):
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b'x')
        f.with_suffix('.json').write_text('{}')


def photo_f(out, ds, arm, splits=('val', 'test')):
    return [out / 'exports' / ds / arm / f'{sp}__s{s}.npz' for sp in splits for s in range(5)]


def held_f(out, enc):
    return [out / 'cstar' / enc / 'held/k11=40' / f'{a}__s{s}.npz' for s in range(5) for a in ('emotion', 'style', 'genre')]


# (3b-1) exports under stubbed steps: part a of siglip2 (ok/ok), siglip1 (failed/failed: skipped, said so), seeds 0 and 1 only
out = run('stub exports part-a', o2, ['exports', '--set', 'part-a', '--enc', 'siglip2', 'siglip1', '--seeds', '0', '1'], dict(BS_RUN_STUB='1'))
assert 'SKIPPED siglip1' in out, out
steps = [json.loads(x) for x in (o2 / 'exports/steps.jsonl').read_text().splitlines()]
assert [x['step'] for x in steps] == ['held_siglip2', 'affection_photo_siglip2', 'emoset_photo_siglip2'], steps
assert steps[0]['argv'][2:4] == ['export', '--encoder'] and '--device' in steps[0]['argv'] and 'cpu' in steps[0]['argv'], steps[0]
assert 'bs_photo.py' in steps[1]['argv'][1] and '--seeds' in steps[1]['argv'], steps[1]
run('stub exports qwen', o2, ['exports', '--set', 'qwen'], dict(BS_RUN_STUB='1'))
assert [x['step'] for x in [json.loads(l) for l in (o2 / 'exports/steps.jsonl').read_text().splitlines()]][-2:] == \
    ['emoset_photo_qwen', 'emoset_cstar_qwen']
# (3b-2) stage 2 refuses without part a exports (X photo siglip2, clip_l14 are ok) and without the Qwen ones
(o2 / 'stage2/stage2_status.json').write_text(json.dumps(dict(status='ok')))
out = run('stage2 without exports', o2, ['stage2'], dict(BS_RUN_STUB='1'), expect=1)
assert 'REFUSED' in out and 'exports' in out, out
for enc in ('siglip2', 'clip_l14'):
    fake_exports(o2, photo_f(o2, 'emoset', f'photo_{enc}'))
out = run('stage2 without qwen exports', o2, ['stage2'], dict(BS_RUN_STUB='1'), expect=1)
assert 'REFUSED' in out and 'qwen' in out, out
for arm in ('photo_qwen', 'cstar_qwen'):
    fake_exports(o2, photo_f(o2, 'emoset', arm))
run('stage2 with exports (stub: no record is written, so rc 1)', o2, ['stage2'], dict(BS_RUN_STUB='1'), expect=1)
(o2 / 'stage2/stage2_status.json').write_text(json.dumps(dict(status='ok')))       # the stub run above wrote 'failed'

(o2 / 'stage1').mkdir(exist_ok=True)
for enc in ('siglip2', 'clip_l14'):
    for nm in ('artclu_ranges', 'artret_ranges', 'artret_picks', 'affection_ranges'):
        (o2 / f'stage1/{nm}_{enc}.json').write_text(json.dumps(dict(encoder=enc, name=nm)))
run('stub stage1', o2, ['stage1'], dict(BS_RUN_STUB='1'))
assert (o2 / 'records/stage1_siglip2.json').exists() and not (o2 / 'records/stage1_siglip1.json').exists()
(o2 / 'stage3').mkdir(exist_ok=True)
(o2 / 'stage3/outcomes.json').write_text('{}')
(o2 / 'records/stage2_emoset.json').write_text('{}')
out = run('stage3 without held exports', o2, ['stage3'], dict(BS_RUN_STUB='1'), expect=1)
assert 'REFUSED' in out and 'held' in out, out
for enc in ('siglip2', 'clip_l14'):
    fake_exports(o2, held_f(o2, enc))
fake_exports(o2, photo_f(o2, 'affection', 'photo_clip_l14'))
for nm in ('affection', 'artelingo_clu', 'emoset'):
    (o2 / f'stage3/cells_{nm}.json').write_text(json.dumps(dict(cells=nm)))
out = run('stub stage3 (cells only)', o2, ['stage3'], dict(BS_RUN_STUB='1'))
assert 'NO verdict yet' in out and 'sha256 stage3/cells_affection.json' in out and 'sha256 records/stage1_siglip2.json' in out, out
assert not (o2 / 'records/verdicts.json').exists(), 'stage 3 must not write the verdict'
sha = jl(o2 / 'stage3/c13_sha256.json')
assert sorted(sha) == ['records/stage1_clip_l14.json', 'records/stage1_siglip2.json', 'records/stage2_emoset.json',
                       'stage3/cells_affection.json', 'stage3/cells_artelingo_clu.json', 'stage3/cells_emoset.json'], sorted(sha)
out = run('decide without c13', o2, ['decide', '--smoke'], dict(BS_RUN_STUB='1'), expect=1)
assert 'REFUSED' in out and 'c13.json' in out, out
c13 = o2 / 'records/c13.json'
c13.write_text(json.dumps(dict(**{'pass': False}, sha256=sha)))
out = run('decide with c13 pass false', o2, ['decide', '--smoke'], dict(BS_RUN_STUB='1'), expect=1)
assert 'did not pass' in out, out
c13.write_text(json.dumps({'pass': True, 'sha256': dict(sha, **{'stage3/cells_emoset.json': '0' * 64})}))
out = run('decide with c13 sha256 differing', o2, ['decide', '--smoke'], dict(BS_RUN_STUB='1'), expect=1)
assert 'differs' in out and 'stage3/cells_emoset.json' in out, out
c13.write_text(json.dumps({'pass': True, 'sha256': sha}))
run('decide with c13', o2, ['decide', '--smoke'], dict(BS_RUN_STUB='1'))
assert (o2 / 'records/verdicts.json').exists()
(o2 / 'stage3/cells_emoset.json').write_text(json.dumps(dict(cells='changed after C13')))
out = run('decide after a cells file changed', o2, ['decide', '--smoke'], dict(BS_RUN_STUB='1'), expect=1)
assert 'differs' in out, out
(o2 / 'stage1/artret_picks_siglip2.json').write_text(json.dumps(dict(encoder='siglip2', name='changed')))
out = run('stage3 with changed stage 1 file', o2, ['stage3'], dict(BS_RUN_STUB='1'), expect=1)
assert 'REFUSED' in out

# (3c) a scenario with the photos path alive: stage 1 and stage 3 refuse without the Affection exports
o3 = scratch / 'stub_photos'
(o3 / 'node').mkdir(parents=True, exist_ok=True)
(o3 / 'stage2').mkdir(parents=True, exist_ok=True)
(o3 / 'stage1').mkdir(exist_ok=True)
(o3 / 'stage2/stage2_status.json').write_text(json.dumps(dict(status='ok')))
for enc in ('siglip2', 'siglip1', 'clip_l14'):
    (o3 / f'node/part_a_{enc}.json').write_text((o2 / f'node/part_a_{enc}.json').read_text())
    for nm in ('artclu_ranges', 'artret_ranges', 'artret_picks', 'affection_ranges'):
        (o3 / f'stage1/{nm}_{enc}.json').write_text(json.dumps(dict(encoder=enc, name=nm)))
run('stub stage0, nothing failing', o3, ['stage0'], dict(BS_RUN_STUB='1'))
out = run('stage1 without Affection exports', o3, ['stage1'], dict(BS_RUN_STUB='1'), expect=1)
assert 'REFUSED' in out and 'Affection validation exports' in out and 'exports' in out, out
for enc in ('siglip2', 'clip_l14'):
    fake_exports(o3, photo_f(o3, 'affection', f'photo_{enc}', ('val',)))
run('stub stage1 with exports', o3, ['stage1'], dict(BS_RUN_STUB='1'))
(o3 / 'records/stage2_emoset.json').write_text('{}')
for enc in ('siglip2', 'clip_l14'):
    fake_exports(o3, held_f(o3, enc))
out = run('stage3 without Affection test exports', o3, ['stage3'], dict(BS_RUN_STUB='1'), expect=1)
assert 'REFUSED' in out and 'Affection exports' in out, out

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

# (4b) export_valid on real small files; part_a_state with a photo-only failure (F2)
import hashlib  # noqa: E402
import numpy as np  # noqa: E402
ex = scratch / 'valid'
ex.mkdir(exist_ok=True)
pn = ex / 'val__s0.npz'
np.savez(pn, keys=np.array(['a']), emb=np.zeros((1, 4), np.float16))
pn.with_suffix('.json').write_text(json.dumps(dict(sha256=hashlib.sha256(pn.read_bytes()).hexdigest())))
assert R.export_valid(pn, 'photo')
pn.with_suffix('.json').write_text(json.dumps(dict(sha256='0' * 64)))
assert not R.export_valid(pn, 'photo')
hn = ex / 'emotion__s0.npz'
img = np.arange(8, dtype=np.float16).reshape(2, 4)
np.savez(hn, img=img)
hn.with_suffix('.json').write_text(json.dumps(dict(arrays=dict(img=dict(sha256=hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest())))))
assert R.export_valid(hn, 'held')
np.savez(hn, img=img + 1)
assert not R.export_valid(hn, 'held') and not R.export_valid(ex / 'nope.npz', 'held')
(R.OUT / 'node').mkdir(parents=True, exist_ok=True)
(R.OUT / 'node/part_a_siglip2.json').write_text(json.dumps(dict(
    encoder='siglip2', status='failed', done=[f'cstar__siglip2__s{s}' for s in range(5)] + [f'xphoto__siglip2__s{s}' for s in range(3)])))
assert R.part_a_state('siglip2') == ('ok', 'failed')
print('[exports] export_valid, part_a_state photo-only failure: ok', flush=True)

# (5) diag without bs_diag
r = subprocess.run([PY, str(HERE / 'bs_run.py'), 'diag', '--smoke'], env=dict(BASE, BS_OUTPUTS=str(scratch / 'diag')),
                   capture_output=True, text=True, cwd=str(HERE))
print(f'[diag] rc={r.returncode} {(r.stdout + r.stderr).strip().splitlines()[-1]}')
print('SMOKE OK')
