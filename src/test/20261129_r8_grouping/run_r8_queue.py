"""Ticket 13: the overnight run as two queues (GPU, CPU) and the end-to-end smoke. Plumbing only: no number is computed here.

    python run_r8_queue.py gpu|cpu [--smoke] [--dry-run]       (use run_r8_queue_gpu.sh / run_r8_queue_cpu.sh)

Order and stops: rule section 9 and 8. GPU queue: view check and views (03), FCMI FC3 (stage 1), FC1 FC2 FC4 FC5 FC6 (only if
SC2 passed), encoders AL DI SG (04). CPU queue: SC1 check (real) or SC1 smoke (smoke), head runs (05), groupings (07),
placement (08) for every ID and Z<ID>, readers, tau, scoring (09), late-drops file, C13 + trigger (11, 10), NBC chain if
triggered, C13 + gate verdict (11, 10), then labelfree, harm, full (12).
Every step whose output exists is skipped (resume). A failed step is logged; steps that need it are blocked, the rest go on.
SC1 missing or failed: STOP marker, both queues end. Dependencies on GPU outputs are waited for by file presence (poll 60 s).
D21 quiet CPU (rule M7): each heavy CPU job passes `flock -s TURN true`, then runs under `flock -s QUIET`; each FCMI run runs
under `flock -x TURN flock -x QUIET`, waits for the 1-minute load average < 4.0 and logs it. CPU jobs run under nice -n 10.
The load wait is capped (env R8_LOAD_WAIT_MAX, default 7200 s): at the cap the FCMI step exits 4, the GPU queue writes the state
"stopped: load cap", prints it in its summary line and exits 4; the CPU queue counts that state as "GPU done". A restart resumes.
At most 3 CPU slots of 8 threads (readers use all 3).
Smoke: R8_RESULTS (default /data/SSD2/cosir_r8/e2e_smoke/results) must differ from the real results; every step runs on smoke
settings under <R8_RESULTS>/smoke; nothing is written to the real results.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
MAIN_RF = Path('/project/CoSiR/src/test/20261129_r8_grouping')       # r8_common.RF (hard-coded MAIN)
PY = '/root/miniconda3/envs/CoSiR/bin/python'
GPU_LOCK, TURN, QUIET = '/tmp/gpu0.lock', '/tmp/r8_queue_turnstile.lock', '/tmp/r8_queue_quiet.lock'
SMOKE_RES_DEFAULT = '/data/SSD2/cosir_r8/e2e_smoke/results'
VIEWS_REAL = Path('/data/SSD2/pre_extract/artelingo/r8_clip_views')
N_PAINTINGS, N_IMAGES = 36518, 49121
# rule gap: 09, 10, 11 and 12 read r8_groupings.npz at the full scorer-train rows (r8_scoring_io.load_groupings asserts it), so the
# smoke groupings (07) and placement (08) run on all paintings; a 07 --smoke subset could not feed 09 to 12.
SMOKE_N = N_PAINTINGS
FC_SMOKE_N = 2048                            # 06 --smoke covers the first 2,048 paintings only
SMOKE_VIEWS_N = int(os.environ.get('R8_SMOKE_VIEWS_N', N_PAINTINGS))   # 05 reads views of all paintings: V smoke needs full N
SMOKE_ENC_N = int(os.environ.get('R8_SMOKE_ENC_N', 256))               # 07 needs the images of the first 8192 paintings
LOAD_MAX = float(os.environ.get('R8_LOAD_MAX', 4.0))
GPU_WAIT = int(os.environ.get('R8_GPU_WAIT', 8 * 3600))               # seconds `flock -w` waits for the GPU lock
SLOTS = 3
LOAD_WAIT_MAX_DEFAULT = 7200                  # seconds one FCMI run may wait for the load average (env R8_LOAD_WAIT_MAX)
LOAD_CAP_EXIT = 4                             # exit code of the load-wait guard and of the GPU queue that stopped on it
STATE_CAP = 'stopped: load cap'               # GPU state-file value; the CPU queue counts it as "GPU done"
STATE_GRACE = 300                             # seconds a state file older than the CPU queue's start is ignored (restart race)

V_IDS, NB_IDS = ('V100', 'V67', 'V33'), ('NB100', 'NB67', 'NB33')
FC_IDS = tuple(f'FC{i}' for i in range(1, 7))
FC_WEIGHTS = dict(zip(FC_IDS, ('0.05', '0.10', '0.20', '0.40', '0.80', '1.60')))
ENC = {'AL': 'aladin', 'DI': 'dinov2l', 'SG': 'siglip2'}
REFS = ('R0own', 'NRK', 'TP')


def now():
    return datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%F %H:%M:%S')


def jread(p):
    return json.loads(Path(p).read_text())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


class Cfg:
    """Paths of one queue run (real or smoke)."""

    def __init__(self, smoke):
        self.smoke = smoke
        if smoke:
            os.environ.setdefault('R8_RESULTS', SMOKE_RES_DEFAULT)
        self.res = Path(os.environ['R8_RESULTS']) if os.environ.get('R8_RESULTS') else MAIN_RF / 'results'
        if smoke:
            assert self.res.resolve() != (MAIN_RF / 'results').resolve(), 'smoke needs R8_RESULTS other than the real results'
            self.root = self.res / 'smoke'
        else:
            assert self.res.resolve() == (MAIN_RF / 'results').resolve(), 'the real queue writes the real results (unset R8_RESULTS)'
            self.root = self.res
        self.logs = self.root / 'logs'
        self.rule = MAIN_RF / 'DECISION_RULE.md'
        self.views = self.root / 'views' if smoke else VIEWS_REAL
        self.enc_n = SMOKE_ENC_N if smoke else N_IMAGES
        self.views_n = SMOKE_VIEWS_N if smoke else N_PAINTINGS
        # smoke wires V and the encoders through 07/08 only when the smoke extraction covers what 05 and 07 need
        self.v_ok = (not smoke) or SMOKE_VIEWS_N == N_PAINTINGS
        self.enc_ok = (not smoke) or SMOKE_ENC_N == N_IMAGES

    # raw outputs of the steps
    def fc_verdict(self, name):
        if self.smoke:
            return self.root / 'fcmi' / name / ('stage1_fcmi.json' if name == 'FC3' else 'd21.json')
        return self.root / 'stage1_fcmi.json' if name == 'FC3' else self.root / 'fcmi' / name / 'd21.json'

    def enc_record(self, x):
        return self.root / ('enc' if self.smoke else '.') / f'stageE_{x}.json'

    def stage0_raw(self):
        return self.root / ('stage0' if self.smoke else '.') / 'stage0_regression.json'

    def space(self, x):
        return self.root / 'spaces' / f'{x}.json'

    def part(self, x):
        return self.root / 'groupings' / 'parts' / f'{x}.json'

    def head(self, x):
        return self.root / 'heads' / f'head_{x}.json'

    def stop_marker(self):
        return self.logs / 'STOP_SC1.json'

    def gpu_state(self):
        return self.logs / 'gpu_queue_state.json'

    def load_cap_file(self):
        return self.logs / 'load_cap.json'

    def candidates(self):
        # rule gap: 06's smoke labels cover 2,048 paintings, 07/09 need all: in smoke FC1 to FC6 are wired (06, 07 part) but
        # leave the gate as dropped (drops file), so the gate smoke runs without them.
        return list(V_IDS if self.v_ok else ()) + list(NB_IDS) + ([] if self.smoke else list(FC_IDS)) + (list(ENC) if self.enc_ok else [])

    def all_candidates(self):
        return list(V_IDS) + list(NB_IDS) + list(FC_IDS) + list(ENC)


def write_state(cfg, state, **extra):
    """The GPU state file, written atomically (temp file, then os.replace): gpu_done() never reads a half-written file."""
    cfg.logs.mkdir(parents=True, exist_ok=True)
    p = cfg.gpu_state()
    tmp = p.with_name(f'{p.name}.tmp{os.getpid()}')
    tmp.write_text(json.dumps({'state': state, 'at': now(), **extra}))
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------------------ the engine
class Task:
    """ready(): input files are there. cond(): None, or ('skip'|'fail', why); it may read files that ready() guards, so it runs
    after ready(). verify(): checked on resume when the outputs exist; returns None or a failure text. rule_check(): after exit 1
    with the record written, returns the text of a rule-covered outcome (the step still counts as failed; the summary
    lists it as "dropped under the rule" and it alone does not make the exit code 1)."""

    def __init__(self, name, kind, build=None, fn=None, deps=(), after=(), slots=1, outputs=(), ready=None, ready_doc='',
                 cond=None, cond_doc='', verify=None, rule_check=None):
        self.name, self.kind, self.build, self.fn = name, kind, build, fn
        self.deps, self.after, self.slots = list(deps), list(after), (0 if kind in ('light', 'gpu', 'fcmi') else slots)
        self.outputs = [Path(p) for p in outputs]
        self.ready, self.ready_doc, self.cond, self.cond_doc = ready, ready_doc, cond, cond_doc
        self.verify, self.rule_check, self.rule_dropped = verify, rule_check, None
        self.state, self.why, self.t0, self.t1 = 'pending', '', None, None


class Engine:
    def __init__(self, cfg, queue, tasks, dry):
        self.cfg, self.queue, self.tasks, self.dry = cfg, queue, tasks, dry
        self.by = {t.name: t for t in tasks}
        assert len(self.by) == len(tasks)
        for t in tasks:
            assert all(d in self.by for d in t.deps + t.after), t.name
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.stopped = False
        self.cap_written = False
        self.cap = None                      # set when an FCMI run hit the load-wait cap (GPU queue)
        self.seen_errors = {}
        self.t_start = time.time()
        self.poll = 2 if cfg.smoke else 60

    # -- logging
    def log(self, msg):
        line = f'{now()} [{self.queue}] {msg}'
        print(line, flush=True)
        if not self.dry:
            with open(self.cfg.logs / f'queue_{self.queue}.log', 'a') as f:
                f.write(line + '\n')

    # -- plan
    def plan(self):
        print(f'plan: queue={self.queue} mode={"smoke" if self.cfg.smoke else "real"} root={self.cfg.root} logs={self.cfg.logs}')
        print(f'      slots={SLOTS} (8 threads each) poll={self.poll}s gpu_lock_wait={GPU_WAIT}s load_max={LOAD_MAX}')
        print(f'      heavy CPU step = first `flock -s {TURN} true`, then the printed command; FCMI step = `flock -x {TURN}` + `flock -x {QUIET}`'
              f' + load-average wait; light step = no lock')
        for i, t in enumerate(self.tasks, 1):
            done = bool(t.outputs) and all(p.exists() for p in t.outputs)
            flags = [t.kind] + ([f'slots={t.slots}'] if t.slots else [])
            print(f'{i:3d}. {t.name:<22s} {" ".join(flags)}' + ('   [outputs exist: would skip]' if done else ''))
            if t.deps:
                print(f'       needs ok: {", ".join(t.deps)}')
            if t.after:
                print(f'       after (any outcome): {", ".join(t.after)}')
            if t.ready_doc:
                print(f'       waits for file(s): {t.ready_doc}')
            if t.cond_doc:
                print(f'       runs only if: {t.cond_doc}')
            if t.outputs:
                print(f'       skips if exists: {", ".join(str(p) for p in t.outputs)}')
            if t.build:
                b = t.build(True)
                print('       $ ' + ' '.join(self.wrap(t, b['cmd'], b.get('env', {}), True)))
            elif t.fn:
                print(f'       (python step: {t.fn.__name__})')

    # -- command wrapping
    def wrap(self, t, cmd, env, plan=False):
        envs = [f'{k}={v}' for k, v in env.items()]
        if plan:
            envs = ['PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=8', 'MKL_NUM_THREADS=8'] + envs
        pre = ['env'] + envs if envs else []
        if t.kind == 'heavy':
            return ['flock', '-s', QUIET, 'nice', '-n', '10'] + pre + cmd
        if t.kind == 'gpu':
            return ['flock', '-o', '-w', str(GPU_WAIT), '-E', '75', GPU_LOCK] + pre + cmd
        if t.kind == 'fcmi':
            # read_13 #11, accepted: lock order GPU, TURN, QUIET holds the GPU lock while the run waits for running CPU jobs
            # (no deadlock: CPU jobs never ask for the GPU lock).
            inner = [PY, str(HERE / 'run_r8_queue.py'), '_loadwait', '--'] + pre + cmd
            return ['flock', '-o', '-w', str(GPU_WAIT), '-E', '75', GPU_LOCK, 'flock', '-x', TURN, 'flock', '-x', QUIET] + inner
        raise AssertionError(t.kind)

    def run_cmd(self, t):
        b = t.build(False)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   R8_LOAD_POLL='5' if self.cfg.smoke else '60', R8_LOAD_CAP_FILE=str(self.cfg.load_cap_file()))
        argv = self.wrap(t, b['cmd'], b.get('env', {}), False)
        logp = self.cfg.logs / f'{self.queue}_{t.name}.log'
        with open(logp, 'a') as lf:
            lf.write(f'--- {now()} start {t.name}\n$ {" ".join(argv)}\n')
            lf.flush()
            if t.kind == 'heavy':       # turnstile: wait while an FCMI run holds or wants the quiet CPU
                r = subprocess.run(['flock', '-s', TURN, 'true'], env=env, stdout=lf, stderr=lf)
                assert r.returncode == 0
            r = subprocess.run(argv, env=env, cwd=HERE, stdout=lf, stderr=lf)
            lf.write(f'--- {now()} exit {r.returncode}\n')
        return r.returncode, logp

    # -- execution
    def finish(self, t, state, why=''):
        with self.lock:
            t.state, t.why, t.t1 = state, why, time.monotonic()
        self.log(f'{t.name}: {state}' + (f' ({why})' if why else ''))
        self.wake.set()

    def worker(self, t):
        t.t0 = time.monotonic()
        try:
            if t.build:
                rc, logp = self.run_cmd(t)
                if rc == 75 and t.kind in ('gpu', 'fcmi'):
                    return self.finish(t, 'failed', f'GPU lock not obtained in {GPU_WAIT}s (exit 75); see {logp}')
                if rc == LOAD_CAP_EXIT and t.kind == 'fcmi':
                    return self.load_cap_hit(t, logp)
                if rc != 0:
                    if t.rule_check and rc == 1:        # exit 1 after the record was written: a rule-covered outcome
                        why = self.guarded(t, 'rule_check', t.rule_check, None)[0]
                        if why:
                            t.rule_dropped = why
                            return self.finish(t, 'failed', f'exit 1; recorded outcome covered by the rule ({why}); see {logp}')
                    return self.finish(t, 'failed', f'exit {rc}; see {logp}')
            else:
                t.fn()
            if t.outputs and not all(p.exists() for p in t.outputs):
                return self.finish(t, 'failed', 'exit 0 but outputs missing: ' + ', '.join(str(p) for p in t.outputs if not p.exists()))
            self.finish(t, 'ok')
        except BaseException as e:      # a step that raises is a failed step; the queue goes on
            self.finish(t, 'failed', f'{type(e).__name__}: {e}')

    def guarded(self, t, what, fn, default):
        """(value, error text). An exception in fn is logged once per distinct text and returns (default, text)."""
        try:
            return fn(), None
        except Exception as e:
            msg = f'{what}() raised {type(e).__name__}: {e}'
            if self.seen_errors.get((t.name if t else None, what)) != msg:
                self.seen_errors[(t.name if t else None, what)] = msg
                self.log(f'{t.name if t else "queue"}: {msg} (counted as not ready yet)')
            return default, msg

    def load_cap_hit(self, t, logp):
        info = self.guarded(t, 'load_cap_file', lambda: jread(self.cfg.load_cap_file()), {})[0] or {}
        with self.lock:
            self.cap = info or {'note': f'see {logp}'}
        self.finish(t, 'failed', f'load cap: the 1-minute load stayed >= {LOAD_MAX} ({info}); see {logp}')

    def gpu_done(self):
        """The GPU queue ended: state done, or stopped on the load cap. A state file written before this CPU queue started
        counts only after STATE_GRACE seconds (a restart of both queues: the GPU queue rewrites it as running first)."""
        p = self.cfg.gpu_state()
        if not p.is_file():
            return False
        if jread(p).get('state') not in ('done', STATE_CAP):
            return False
        return p.stat().st_mtime >= self.t_start - 2 or time.time() - self.t_start >= STATE_GRACE

    def run(self):
        self.cfg.logs.mkdir(parents=True, exist_ok=True)
        self.log(f'start {"smoke" if self.cfg.smoke else "real"}; {len(self.tasks)} steps; root {self.cfg.root}')
        last = 0.0
        while True:
            with self.lock:
                states = {t.name: t.state for t in self.tasks}
            if all(s in ('ok', 'skipped', 'failed', 'blocked') for s in states.values()):
                break
            if not self.stopped and self.cfg.stop_marker().is_file():      # both queues stop on the SC1 marker
                self.stopped = True
                self.log('STOP marker found (SC1 failed or missing): no further steps')
            if self.cap is not None and not self.cap_written:
                self.cap_written = True
                write_state(self.cfg, STATE_CAP, load_cap=self.cap)
                self.log(f'{STATE_CAP}: no further GPU steps; restart both queues to resume ({self.cap})')
            for t in self.tasks:
                if t.state != 'pending':
                    continue
                if self.stopped:
                    self.finish(t, 'blocked', 'stopped by SC1')
                    continue
                if self.cap is not None:
                    self.finish(t, 'blocked', STATE_CAP)
                    continue
                if any(states[d] in ('failed', 'blocked') for d in t.deps):
                    self.finish(t, 'blocked', 'needs ' + ', '.join(d for d in t.deps if states[d] in ('failed', 'blocked')))
                    continue
                if not all(states[d] in ('ok', 'skipped') for d in t.deps) or \
                        not all(states[d] in ('ok', 'skipped', 'failed', 'blocked') for d in t.after):
                    continue
                # order: outputs exist (resume), then ready() (the input files), then cond() (which may read those files).
                # An exception in verify()/ready()/cond() is "not ready yet" (logged once) in the CPU queue; once the GPU queue
                # is done it fails that one step with the text. The GPU queue has nothing to wait for: it fails the step at once.
                if t.outputs and all(p.exists() for p in t.outputs):
                    bad, err = self.guarded(t, 'verify', t.verify, None) if t.verify else (None, None)
                    bad = bad or err
                    self.finish(t, 'failed' if bad else 'skipped', bad or 'outputs exist')
                    continue
                err = None
                if t.ready:
                    ok, err = self.guarded(t, 'ready', t.ready, False)
                    if not ok:
                        if self.queue == 'gpu' and err:
                            self.finish(t, 'failed', err)
                        elif self.queue == 'cpu' and self.guarded(None, 'gpu_done', self.gpu_done, False)[0]:
                            self.finish(t, 'failed', err or 'its input never appeared and the GPU queue is done: ' + t.ready_doc)
                        continue
                if t.cond:
                    r, err = self.guarded(t, 'cond', t.cond, None)
                    if err:
                        if self.queue == 'gpu' or self.guarded(None, 'gpu_done', self.gpu_done, False)[0]:
                            self.finish(t, 'failed', err)
                        continue
                    if r:
                        kind, why = r
                        self.finish(t, 'skipped' if kind == 'skip' else 'failed', why)
                        continue
                with self.lock:
                    used = sum(x.slots for x in self.tasks if x.state == 'running')
                    if used + t.slots > SLOTS:
                        continue
                    t.state = 'running'
                self.log(f'{t.name}: start ({t.kind})')
                threading.Thread(target=self.worker, args=(t,), daemon=True).start()
                states[t.name] = 'running'
            if self.stopped is False and self.queue == 'cpu' and self.by.get('sc1') and self.by['sc1'].state == 'failed' \
                    and not self.cfg.stop_marker().exists():
                self.cfg.stop_marker().write_text(json.dumps({'stop': 'SC1', 'at': now(), 'why': self.by['sc1'].why}))
                self.stopped = True
            self.wake.wait(timeout=self.poll if not any(t.state == 'running' for t in self.tasks) else 15)
            self.wake.clear()
        return self.summary()

    def summary(self):
        cnt = {}
        for t in self.tasks:
            k = 'dropped under the rule' if t.rule_dropped else t.state
            cnt[k] = cnt.get(k, 0) + 1
        bad = [t.name for t in self.tasks if t.state in ('failed', 'blocked') and not t.rule_dropped]
        rule = [t.name for t in self.tasks if t.rule_dropped]
        extra = ''
        if self.queue == 'gpu':
            walls = []
            for name in FC_IDS:
                p = self.cfg.fc_verdict(name)
                if p.is_file():
                    walls.append(f'{name}={jread(p).get("wall_seconds", float("nan")):.0f}s')
            extra = ' FCMI wall (D21 limit 3600s): ' + (', '.join(walls) or 'none') + '.'
        line = (f'r8 queue {self.queue} [{"smoke" if self.cfg.smoke else "real"}] done {now()}: {len(self.tasks)} steps, '
                + ', '.join(f'{v} {k}' for k, v in sorted(cnt.items())) + (f'; not ok: {", ".join(bad)}' if bad else '; all ok')
                + (f'; dropped under the rule: {", ".join(rule)}' if rule else '')
                + (f'; {STATE_CAP} ({self.cap}); restart both queues to resume' if self.cap is not None else '')
                + '.' + extra + f' Logs: {self.cfg.logs}')
        self.log(line)
        (self.cfg.logs / f'queue_{self.queue}_summary.txt').write_text(line + '\n')
        return line, (3 if self.cfg.stop_marker().is_file() else (LOAD_CAP_EXIT if self.cap is not None else (1 if bad else 0)))



# ------------------------------------------------------------------------------------------------ small file helpers
def exists(*paths):
    return lambda: all(Path(p).is_file() for p in paths)


def check_sc1_real(cfg):
    """Real mode: stage0_regression.json must exist with pass true, smoke false, this rule's SHA-256."""
    p = cfg.root / 'stage0_regression.json'
    assert p.is_file(), f'SC1 record missing: {p}'
    rec = jread(p)
    assert rec.get('pass') is True and rec.get('smoke') is False, f'SC1 did not pass: {p}'
    assert rec.get('rule_sha256') == sha_file(cfg.rule), 'SC1 record carries another rule SHA-256'


def stop_gate(cfg):
    def cond():
        if cfg.stop_marker().is_file():
            return ('fail', 'STOP marker (SC1)')
        if not cfg.smoke:
            try:
                check_sc1_real(cfg)
            except AssertionError as e:
                cfg.logs.mkdir(parents=True, exist_ok=True)
                if not cfg.stop_marker().exists():
                    cfg.stop_marker().write_text(json.dumps({'stop': 'SC1', 'at': now(), 'why': str(e)}))
                return ('fail', f'SC1: {e}')
        return None
    return cond


# ------------------------------------------------------------------------------------------------ GPU queue
def gpu_tasks(cfg):
    sm = cfg.smoke
    T = []
    prev = []

    def add(t):
        t.after = list(prev) + t.after
        T.append(t)
        prev[:] = [t.name]
        return t

    def state(s):
        def fn():
            write_state(cfg, s)
        fn.__name__ = f'write_gpu_state_{s}'
        return fn

    add(Task('gpu_start', 'light', fn=state('running')))
    add(Task('sc1_gate', 'light', fn=lambda: None, cond=stop_gate(cfg),
             cond_doc='stage0_regression.json pass (real); no STOP marker'))
    # views (03)
    vcmd = [PY, 'r8_views.py', '--batch-size', '128'] + (['--smoke', str(cfg.views_n)] if sm else [])
    def views_failed_recorded():
        m = cfg.views / 'meta.json'
        return 'M4 view check failed, recorded in meta.json' if m.is_file() and jread(m)['view_check']['passed'] is False else None
    add(Task('views', 'gpu', build=lambda plan: {'cmd': vcmd}, outputs=[cfg.views / 'meta.json'], deps=[],
             rule_check=views_failed_recorded))
    # FCMI FC3 (stage 1), then the others if SC2 passed
    def fc_cmd(name):
        return [PY, 'r8_fcmi.py', '--weight', FC_WEIGHTS[name], '--device', 'cuda'] + (['--smoke'] if sm else [])
    fenv = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': '0'}
    add(Task('FC3_stage1', 'fcmi', build=lambda plan: {'cmd': fc_cmd('FC3'), 'env': fenv}, outputs=[cfg.fc_verdict('FC3')],
             cond=lambda: None if not cfg.stop_marker().is_file() else ('fail', 'STOP marker')))

    def sc2_cond():
        p = cfg.fc_verdict('FC3')
        if not p.is_file():
            return ('skip', 'no stage-1 record: FC3 did not finish, FC1 FC2 FC4 to FC6 do not run')
        rec = jread(p)
        ok = rec.get('wiring_pass') is True if sm else rec.get('pass') is True
        return None if ok else ('skip', f'SC2 failed ({p.name} {"wiring_pass" if sm else "pass"} is not true): FC1 to FC6 drop out')
    for name in ('FC1', 'FC2', 'FC4', 'FC5', 'FC6'):
        add(Task(f'{name}_d21', 'fcmi', build=lambda plan, n=name: {'cmd': fc_cmd(n), 'env': fenv},
                 outputs=[cfg.fc_verdict(name)], cond=sc2_cond, cond_doc='SC2 passed (stage1_fcmi.json pass true)'))
    # encoders (04)
    for x, e in ENC.items():
        env = {'HF_HUB_OFFLINE': '1'}
        if x == 'AL':
            env['PYTHONPATH'] = '/data/SSD2/third_party/pydeps'
        cmd = [PY, 'r8_enc_extract.py', '--encoder', e] + (['--smoke', str(cfg.enc_n)] if sm else [])
        def esc1_failed_recorded(x=x):
            r = cfg.enc_record(x)
            return f'ESC1 failed, recorded in {r.name}' if r.is_file() and jread(r).get('pass') is False else None
        add(Task(f'enc_{x}', 'gpu', build=lambda plan, c=cmd, v=env: {'cmd': c, 'env': v}, outputs=[cfg.enc_record(x)],
                 rule_check=esc1_failed_recorded))
    add(Task('gpu_done', 'light', fn=state('done')))
    return T


# ------------------------------------------------------------------------------------------------ CPU queue
def write_drops(cfg, batch, names, extra, grouping_files):
    """groupings/r8_drops_<batch>.json = {groupings: {}, dropped: {X: reason}} (index "Late drops")."""
    out = cfg.root / 'groupings' / f'r8_drops_{batch}.json'
    assert not out.exists(), f'{out} exists'
    known = set()
    for g in grouping_files:
        known |= set(jread(g)['dropped'])
    dropped = {}
    for x in names:
        if x in known:
            continue
        reason = None
        if x in extra:
            reason = extra[x]
        hp = cfg.head(x)
        if reason is None and hp.is_file():          # interface: 08 heads/head_<X>.json {dropped, reason}
            h = jread(hp)
            if h.get('dropped') is True:
                reason = f'placement: {h.get("reason")}'
        rp = cfg.root / 'readers' / f'reader_{x}.json'
        if reason is None and rp.is_file():          # run_r8.py reader record {dropped, reason}
            r = jread(rp)
            if r.get('dropped'):
                reason = f'reader: {r.get("reason")}'
        if reason:
            dropped[x] = reason
    for x in names:                                    # a dropped X has no control
        if x.startswith('Z') and x[1:] in names and x not in known and x not in dropped:
            base = x[1:]
            if base in dropped:
                dropped[x] = f'control of dropped {base}: {dropped[base]}'
    assert all(isinstance(v, str) and v.strip() for v in dropped.values())
    payload = json.dumps({'groupings': {}, 'dropped': dropped}, indent=1, sort_keys=True)   # dump first, write once
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x') as f:
        f.write(payload)
    return dropped


def write_smoke_gate_records(cfg):
    """SMOKE ONLY. Ticket 10 in smoke reads stage records with `pass` from the smoke root while stages 0, 1 write
    pass=false + wiring_pass=true in smoke. Write gate records from the wiring results, under the smoke root only."""
    assert cfg.smoke, 'smoke gate records are never written in real mode'
    rule = sha_file(cfg.rule)
    pairs = {'stage0_regression.json': cfg.stage0_raw(), 'stage1_fcmi.json': cfg.fc_verdict('FC3')}
    pairs.update({f'stageE_{x}.json': cfg.enc_record(x) for x in ENC})
    for name, raw in pairs.items():
        out = cfg.root / name
        assert out.resolve().is_relative_to(cfg.root.resolve()) and cfg.root.resolve().is_relative_to((cfg.res / 'smoke').resolve())
        src = jread(raw)
        assert src.get('smoke') is True, f'{raw}: not a smoke record'
        ok = src['wiring_pass'] if 'wiring_pass' in src else src['pass']
        assert isinstance(ok, bool)
        # a full copy of the source record (11 reads evidence, esc1, n, loss_log, labels... as 04 and 06 wrote them; its paths are
        # absolute, so they stay valid) with `pass` set to the wiring result
        rec = {**src, 'pass': ok, 'smoke': True, 'wiring_pass': ok, 'rule_sha256': rule, 'written': now(),
               'source': str(raw), 'source_sha256': sha_file(raw), 'note': 'smoke gate record from the wiring result (ticket 13)'}
        payload = json.dumps(rec, indent=1)          # one write of a prepared string
        with open(out, 'x') as f:
            f.write(payload)


def nbc_dropped(cfg):
    """The reason when NBC is recorded dropped (M16), else None: 07 (r8_groupings_nbc.json), 08 (heads/head_NBC.json),
    09 (readers/reader_NBC.json) or the late-drops file."""
    def read(p):
        return jread(p) if Path(p).is_file() else {}
    g = read(cfg.root / 'groupings' / 'r8_groupings_nbc.json').get('dropped', {})
    if 'NBC' in g:
        return f'07: {g["NBC"]}'
    h = read(cfg.head('NBC'))
    if h.get('dropped') is True:
        return f'08: {h.get("reason")}'
    r = read(cfg.root / 'readers' / 'reader_NBC.json')
    if r.get('dropped'):
        return f'09: {r.get("reason")}'
    d = read(cfg.root / 'groupings' / 'r8_drops_nbc.json').get('dropped', {})
    if 'NBC' in d:
        return f'drops file: {d["NBC"]}'
    return None


def cpu_tasks(cfg):
    sm = cfg.smoke
    R = cfg.root
    T = []
    sflag = ['--smoke'] if sm else []
    rflag = ['--smoke', '--root', str(R)] if sm else []

    def add(*a, **k):
        t = Task(*a, **k)
        T.append(t)
        return t

    def cmd(*parts):
        c = [PY, *[str(p) for p in parts]]
        return lambda plan: {'cmd': c}

    # SC1: real = check the existing record; smoke = run stage 0's wiring smoke (47 s)
    if sm:
        T.append(Task('sc1', 'heavy', build=cmd('run_r8_stage0.py', '--smoke'), outputs=[cfg.stage0_raw()]))
    else:
        def check_sc1_real_step():
            check_sc1_real(cfg)
        T.append(Task('sc1', 'light', fn=check_sc1_real_step))
    gate0 = ['sc1']

    cands = cfg.candidates()
    all_c = cfg.all_candidates()
    place_ids = cands + ['Z' + x for x in cands] + list(REFS)
    names_main = all_c + ['Z' + x for x in all_c] + list(REFS)         # 09's default list; also the drops file's names
    explicit = [] if (cfg.v_ok and cfg.enc_ok and not sm) else ['--groupings', *place_ids]
    part_flag = ['--smoke', str(SMOKE_N)] if sm else []
    fc_part_flag = ['--smoke', str(FC_SMOKE_N)] if sm else []

    # heads (05)
    vmeta = cfg.views / 'meta.json'
    vviews = ['--views', str(cfg.views / 'views.npy')] if sm else []

    def views_passed():
        return jread(vmeta)['view_check']['passed'] is True

    for x in NB_IDS:
        add_t = Task(f'head_{x}', 'heavy', build=cmd('r8_head.py', '--only', x, *sflag), deps=gate0, outputs=[cfg.space(x)])
        T.append(add_t)
    if cfg.v_ok:
        for x in V_IDS:
            T.append(Task(f'head_{x}', 'heavy', build=cmd('r8_head.py', '--only', x, *sflag, *vviews), deps=gate0,
                          outputs=[cfg.space(x)], ready=exists(vmeta), ready_doc=str(vmeta),
                          cond=lambda: None if views_passed() else ('skip', 'M4 view check failed: V spaces are not built (07 records the drop)'),
                          cond_doc='views meta.json view_check.passed is true'))
    # groupings parts (07)
    for x in ('R0own', 'NRK', 'TP'):
        T.append(Task(f'part_{x}', 'heavy', build=cmd('r8_groupings.py', 'part', '--id', x, *part_flag), deps=gate0, outputs=[cfg.part(x)]))
    for x in NB_IDS:
        T.append(Task(f'part_{x}', 'heavy', build=cmd('r8_groupings.py', 'part', '--id', x, *part_flag), deps=[f'head_{x}'], outputs=[cfg.part(x)]))
    if cfg.v_ok:
        for x in V_IDS:      # head_<V> is skipped when the view check failed; 07 then records the drop
            T.append(Task(f'part_{x}', 'heavy', build=cmd('r8_groupings.py', 'part', '--id', x, *part_flag), deps=[f'head_{x}'],
                          outputs=[cfg.part(x)]))
    # read_13 #8, by design (rule section 8, last row: not covered, stop): one crashed FCMI or encoder run (no d21.json or
    # stageE_*.json) fails its part_<X> wait once the GPU queue ends, and collect then blocks the chain. A re-run of both resumes.
    for i, x in enumerate(FC_IDS, 1):
        def fready(x=x, i=i):
            v = cfg.fc_verdict(x)
            if sm or x == 'FC3':
                return v.is_file()
            s1 = cfg.fc_verdict('FC3')
            return s1.is_file() and (jread(s1).get('pass') is False or v.is_file())
        T.append(Task(f'part_{x}', 'heavy', build=cmd('r8_groupings.py', 'part', '--id', x, *fc_part_flag), deps=gate0,
                      outputs=[cfg.part(x)], ready=fready,
                      ready_doc=f'{cfg.fc_verdict(x)} (real: stage1_fcmi.json, and d21.json when SC2 passed)'))
    if cfg.enc_ok:
        for x in ENC:
            T.append(Task(f'part_{x}', 'heavy', build=cmd('r8_groupings.py', 'part', '--id', x, *part_flag), deps=gate0,
                          outputs=[cfg.part(x)], ready=exists(cfg.enc_record(x)), ready_doc=str(cfg.enc_record(x))))
    ids = cands + list(REFS)
    T.append(Task('collect', 'heavy', build=cmd('r8_groupings.py', 'collect', '--ids', *ids, *(['--smoke'] if sm else [])),
                  deps=[f'part_{x}' for x in ids], outputs=[R / 'groupings' / 'r8_groupings.json']))
    # placement (08)   # interface: 08
    for x in place_ids:
        T.append(Task(f'place_{x}', 'heavy', build=cmd('r8_place.py', '--grouping', x, *(['--smoke', str(SMOKE_N)] if sm else [])),   # interface: 08
                      deps=['collect'], outputs=[cfg.head(x)]))
    pl = [f'place_{x}' for x in place_ids]
    # readers, tau, scoring (09)
    T.append(Task('readers_main', 'heavy', slots=SLOTS, build=cmd('run_r8.py', 'readers', '--batch', 'main', '--jobs', '3', *rflag, *explicit),
                  deps=pl, cond=lambda: ('skip', 'arrays_main.npz exists') if (R / 'seed42' / 'arrays_main.npz').is_file() else None,
                  cond_doc='seed42/arrays_main.npz does not exist yet'))
    T.append(Task('taus_main', 'heavy', build=cmd('run_r8.py', 'taus', '--batch', 'main', *rflag, *explicit), deps=['readers_main'],
                  outputs=[R / 'seed42' / 'taus_main.json']))
    T.append(Task('score_main', 'heavy', build=cmd('run_r8.py', 'score', '--batch', 'main', *rflag, *explicit), deps=['taus_main'],
                  outputs=[R / 'seed42' / 'arrays_main.npz', R / 'seed42' / 'index.json']))
    extra_main = {}
    if not cfg.v_ok:
        for x in V_IDS:
            extra_main[x] = f'smoke: views built for {cfg.views_n} of {N_PAINTINGS} paintings'
            extra_main['Z' + x] = f'smoke: control of {x}'
    if sm:
        for x in FC_IDS:
            extra_main[x] = f'smoke: FCMI smoke labels cover {FC_SMOKE_N} paintings, the gate needs {N_PAINTINGS}'
            extra_main['Z' + x] = f'smoke: control of {x}'
    if not cfg.enc_ok:
        for x in ENC:
            extra_main[x] = f'smoke: encoder extraction covers {cfg.enc_n} of {N_IMAGES} images (07 needs all images of the smoke paintings)'
            extra_main['Z' + x] = f'smoke: control of {x}'
    gmain = R / 'groupings' / 'r8_groupings.json'
    dmain = R / 'groupings' / 'r8_drops_main.json'

    def drops_main():
        write_drops(cfg, 'main', names_main, extra_main, [gmain])
    T.append(Task('drops_main', 'light', fn=drops_main, deps=['readers_main'], outputs=[dmain]))
    if sm:
        T.append(Task('smoke_gate_records', 'light', fn=lambda: write_smoke_gate_records(cfg), deps=['sc1'],
                      outputs=[R / n for n in ('stage0_regression.json', 'stage1_fcmi.json', 'stageE_AL.json', 'stageE_DI.json', 'stageE_SG.json')],
                      ready=exists(cfg.stage0_raw(), cfg.fc_verdict('FC3'), *[cfg.enc_record(x) for x in ENC]),
                      ready_doc='smoke stage 0, FC3 and encoder wiring records'))
        T[-1].fn.__name__ = 'write_smoke_gate_records'
    stage_wait = []
    if not sm:
        stage_wait = [cfg.root / 'stage1_fcmi.json'] + [cfg.enc_record(x) for x in ENC]
    gate_dep0 = ['score_main', 'drops_main'] + (['smoke_gate_records'] if sm else [])
    gate_ready = exists(*stage_wait) if stage_wait else None
    gate_ready_doc = ', '.join(str(p) for p in stage_wait)

    inx_main = [R / 'seed42' / 'index.json']
    gr_main = [gmain, dmain]
    rg = ['--smoke', '--results', str(R)] if sm else []

    def gate_args(step, plan, triggered):
        # M16: a triggered NBC that was dropped has no index_nbc.json; 10 and 11 get NBC from the drops/groupings files only
        # (r8_gate.gate_stats and r8_rederive.statistics take a fired trigger with NBC in `dropped`). In the plan: shown with it.
        nbc_idx = triggered and (plan or not nbc_dropped(cfg))
        idx = inx_main + ([R / 'seed42' / 'index_nbc.json'] if nbc_idx else [])
        grp = gr_main + ([R / 'groupings' / 'r8_groupings_nbc.json', R / 'groupings' / 'r8_drops_nbc.json'] if triggered else [])
        return idx, grp

    def triggered():
        return jread(R / 'probe_trigger.json').get('triggered') is True

    def pend_cmd(step, trig):
        def b(plan):
            t = (not plan) and trig()
            idx, grp = gate_args(step, plan, t if not plan else (step == 'gate'))
            return {'cmd': [PY, 'run_r8_gate.py', 'pending', '--step', step, *rg, '--index', *map(str, idx), '--groupings', *map(str, grp)]}
        return b

    def c13_cmd(step, trig):
        def b(plan):
            t = (not plan) and trig()
            idx, grp = gate_args(step, plan, t if not plan else (step == 'gate'))
            return {'cmd': [PY, 'rederive/r8_rederive.py', '--step', step, '--pending', str(R / f'{step}_pending.json'),   # interface: 11
                            *(['--smoke', '--root', str(R), '--aff-reader', str(R / 'seed42' / 'reader_A0.npz')] if sm else []),   # smoke: 09's own A0
                            '--index', *map(str, idx), '--groupings', *map(str, grp)]}
        return b

    fin = lambda step: cmd('run_r8_gate.py', 'finalize', '--step', step, *rg)
    # trigger step
    T.append(Task('trigger_pending', 'heavy', build=pend_cmd('trigger', lambda: False), deps=gate_dep0,
                  outputs=[R / 'trigger_pending.json'], ready=gate_ready, ready_doc=gate_ready_doc))
    def c13_ok(step):
        p = R / 'rederive' / f'{step}_rederive.json'

        def v():       # 11 writes the record also when it disagrees (exit 1): on a resume only "agree" counts as done
            st = jread(p).get('status')
            return None if st == 'agree' else f'{p.name} says status {st!r}, not "agree" (C13 disagreement): the step failed'
        return v
    T.append(Task('trigger_c13', 'heavy', build=c13_cmd('trigger', lambda: False), deps=['trigger_pending'],
                  outputs=[R / 'rederive' / 'trigger_rederive.json'], verify=c13_ok('trigger')))
    T.append(Task('trigger_final', 'heavy', build=fin('trigger'), deps=['trigger_c13'], outputs=[R / 'probe_trigger.json']))
    # NBC chain (if triggered)
    def ntrig_cond():
        return None if triggered() else ('skip', 'probe_trigger.json: not triggered')
    nc = dict(cond=ntrig_cond, cond_doc='probe_trigger.json triggered is true')
    pt = R / 'probe_trigger.json'

    def nbc_head(plan):
        s = '<strength from probe_trigger.json>' if plan else str(jread(pt)['strength'])
        return {'cmd': [PY, 'r8_head.py', '--only', 'NBC', '--strength', s, *sflag]}
    T.append(Task('nbc_head', 'heavy', build=nbc_head, deps=['trigger_final'], outputs=[cfg.space('NBC')], **nc))
    T.append(Task('nbc_part', 'heavy', build=cmd('r8_groupings.py', 'part', '--id', 'NBC', *part_flag), deps=['nbc_head'],
                  outputs=[cfg.part('NBC')], **nc))
    T.append(Task('nbc_collect', 'heavy', build=cmd('r8_groupings.py', 'collect', '--ids', 'NBC', '--name', 'r8_groupings_nbc', *(['--smoke'] if sm else [])),
                  deps=['nbc_part'], outputs=[R / 'groupings' / 'r8_groupings_nbc.json'], **nc))
    for x in ('NBC', 'ZNBC'):
        T.append(Task(f'nbc_place_{x}', 'heavy', build=cmd('r8_place.py', '--grouping', x, '--groupings-name', 'r8_groupings_nbc',    # interface: 08
                                                            *(['--smoke', str(SMOKE_N)] if sm else [])),
                      deps=['nbc_collect'], outputs=[cfg.head(x)], **nc))
    nb = ['--batch', 'nbc', '--groupings', 'NBC', 'ZNBC']
    sd = R / 'seed42'

    def readers_cond():       # read_13 #10: as readers_main, a finished earlier run is not validated again
        r = ntrig_cond()
        if r:
            return r
        if (sd / 'arrays_nbc.npz').is_file() or (sd / 'taus_nbc.json').is_file() or \
                all((R / 'readers' / f'reader_{x}.json').is_file() for x in ('NBC', 'ZNBC')):
            return ('skip', 'NBC readers output exists (arrays_nbc.npz, taus_nbc.json or both reader records)')
        return None

    def undropped_cond():     # final_review M1: NBC dropped after the trigger fired (M16) -> nothing to tau or score
        r = ntrig_cond()
        if r:
            return r
        why = nbc_dropped(cfg)
        return ('skip', f'NBC is dropped under M16, nothing to score; the drop counts as not passing, 10 and 11 get no index_nbc.json ({why})') if why else None
    T.append(Task('nbc_readers', 'heavy', slots=SLOTS, build=cmd('run_r8.py', 'readers', *nb, '--jobs', '3', *rflag),
                  deps=['nbc_place_NBC', 'nbc_place_ZNBC'], cond=readers_cond,
                  cond_doc='probe_trigger.json triggered; no NBC readers output yet'))
    ud = dict(cond=undropped_cond, cond_doc='probe_trigger.json triggered and NBC not recorded dropped (M16)')
    T.append(Task('nbc_taus', 'heavy', build=cmd('run_r8.py', 'taus', *nb, *rflag), deps=['nbc_readers'],
                  outputs=[sd / 'taus_nbc.json'], **ud))
    T.append(Task('nbc_score', 'heavy', build=cmd('run_r8.py', 'score', *nb, *rflag), deps=['nbc_taus'],
                  outputs=[sd / 'arrays_nbc.npz', sd / 'index_nbc.json'], **ud))

    def drops_nbc():
        write_drops(cfg, 'nbc', ['NBC', 'ZNBC'], {}, [R / 'groupings' / 'r8_groupings_nbc.json'])
    # after the readers (not after the score): a dropped NBC has no score step, and the reader records are final here
    T.append(Task('drops_nbc', 'light', fn=drops_nbc, deps=['nbc_readers'], outputs=[R / 'groupings' / 'r8_drops_nbc.json'], **nc))
    # gate step
    T.append(Task('gate_pending', 'heavy', build=pend_cmd('gate', triggered), deps=['trigger_final', 'nbc_score', 'drops_nbc'],
                  outputs=[R / 'gate_pending.json']))
    T.append(Task('gate_c13', 'heavy', build=c13_cmd('gate', triggered), deps=['gate_pending'], outputs=[R / 'rederive' / 'gate_rederive.json'],
                  verify=c13_ok('gate')))
    T.append(Task('gate_final', 'heavy', build=fin('gate'), deps=['gate_c13'], outputs=[R / 'gate_verdict.json']))
    # descriptive (12)
    T.append(Task('desc_labelfree', 'heavy', build=cmd('run_r8_descriptive.py', 'labelfree', *rflag), deps=['gate_final'],
                  outputs=[R / 'descriptive_labelfree.json']))
    T.append(Task('desc_harm', 'heavy', build=cmd('run_r8_descriptive.py', 'harm', *rflag), deps=['desc_labelfree'], outputs=[R / 'harm_run.json']))
    T.append(Task('desc_full', 'heavy', build=cmd('run_r8_descriptive.py', 'full', *rflag), deps=['desc_harm'], outputs=[R / 'descriptive.json']))
    # every CPU task other than sc1 runs only after SC1; tasks with ready= wait for their GPU inputs
    for t in T:
        if t.name != 'sc1' and 'sc1' not in t.deps and not t.deps:
            t.deps.append('sc1')
    return T


def read_load():
    return float(open('/proc/loadavg').read().split()[0])


def loadwait(argv):
    """FCMI start guard: wait (inside the locks, before model construction) for the 1-minute load average < LOAD_MAX, at most
    R8_LOAD_WAIT_MAX seconds (default 2 h). At the cap: log the load, write R8_LOAD_CAP_FILE and return LOAD_CAP_EXIT (4)."""
    assert argv and argv[0] == '--', argv
    cmd = argv[1:]
    poll = int(os.environ.get('R8_LOAD_POLL', 60))
    cap = float(os.environ.get('R8_LOAD_WAIT_MAX', LOAD_WAIT_MAX_DEFAULT))
    waited = 0.0
    while True:
        load = read_load()
        if load < LOAD_MAX:
            break
        if waited >= cap:
            print(f'{now()} LOAD CAP: 1-min load {load:.2f} still >= {LOAD_MAX} after {waited:.0f}s (cap {cap:.0f}s): FCMI run not started, exit {LOAD_CAP_EXIT}',
                  flush=True)
            if os.environ.get('R8_LOAD_CAP_FILE'):
                Path(os.environ['R8_LOAD_CAP_FILE']).write_text(json.dumps(
                    {'load': load, 'load_max': LOAD_MAX, 'waited_seconds': waited, 'cap_seconds': cap, 'at': now(), 'cmd': cmd}))
            return LOAD_CAP_EXIT
        print(f'{now()} load average {load:.2f} >= {LOAD_MAX}: waiting {poll}s (outside D21 wall time; cap {cap:.0f}s, waited {waited:.0f}s)', flush=True)
        step = min(poll, max(cap - waited, 0.0)) if cap - waited < poll else poll
        time.sleep(step)
        waited += step
    print(f'{now()} FCMI start: 1-min load {load:.2f}, waited {waited}s: {" ".join(cmd)}', flush=True)
    env = dict(os.environ)
    i = 0
    if cmd and cmd[0] == 'env':
        i = 1
        while i < len(cmd) and '=' in cmd[i]:
            k, v = cmd[i].split('=', 1)
            env[k] = v
            i += 1
    r = subprocess.run(cmd[i:], env=env)
    print(f'{now()} FCMI end: exit {r.returncode}, 1-min load {read_load():.2f}', flush=True)
    return r.returncode


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_loadwait':
        return loadwait(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('queue', choices=('gpu', 'cpu'))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', help='comma-separated step names: run just these (dependencies on other steps are ignored); for checks')
    a = ap.parse_args()
    cfg = Cfg(a.smoke)
    tasks = gpu_tasks(cfg) if a.queue == 'gpu' else cpu_tasks(cfg)
    if a.only:
        keep = set(a.only.split(','))
        assert keep <= {t.name for t in tasks}, f'unknown steps {keep - {t.name for t in tasks}}'
        tasks = [t for t in tasks if t.name in keep]
        for t in tasks:
            t.deps, t.after = [d for d in t.deps if d in keep], [d for d in t.after if d in keep]
    eng = Engine(cfg, a.queue, tasks, a.dry_run)
    if a.dry_run:
        eng.plan()
        print(f'dry run: {len(tasks)} steps planned for the {a.queue} queue; nothing was run')
        return 0
    cfg.logs.mkdir(parents=True, exist_ok=True)
    line, code = eng.run()
    print(line)
    return code


if __name__ == '__main__':
    sys.exit(main())
