"""r15 stage runner (ticket 07): spec §3.5 stages 0 to 3, §3.7, §6, D6, D17. Plumbing only: every number comes from the
round's modules, called through their command lines (one subprocess per step, so a crash is a failed step, not a dead
runner). Pattern: r14's `bs_run.py`. Each step logs to OUT/stage<k>/logs/<step>.log, appends {step, argv, rc, seconds,
attempt, ...} to OUT/stage<k>/steps.jsonl and prints one line `MT_RUN <stage> <step> rc=<rc> seconds=<s>`. A step whose
exit code is bad, or whose declared output files are missing or hold NaN/Infinity, is retried once; a second failure is
a failed step (rc stays non-zero; status says "failed"); a step whose dependency failed is recorded with rc=-1.
No GPU anywhere. Steps run at most 3 at a time (8 threads each). The runner never commits; it prints the command.

    python mt_run.py stage0 [--enc E ...] [--only STEP ...] [--n-clu N] [--n-ret N]
        gates (i) (v) (ii) (iv) (iii) (vi), then the exports (mt_export, mt_matched), records/cutsets.json; gate evidence in
        OUT/stage0/gate.json; records/status.json (gates; the stage 1 entries are "pending" until stage 1).
        Under MT_SMOKE the replays (iii) and (vi) run at reduced scope (see SMOKE_REPLAY) and their gate entries say
        smoke: true; their status is "ok" when the code path ran and its values agree (wiring), as the module reports.
    python mt_run.py stage1 [--enc E ...]
        ranges (clustering, retrieval) per encoder, the retrieval picks per encoder, the teacher's pick (with Qwen's ranges);
        merged into records/stage1_<enc>.json {encoder, clu_ranges, ret_ranges, picks} and records/stage1_teacher.json;
        records/status.json completed; prints the `git add -f ... && git commit ...` command (the controller commits).
        A missing range or pick is "failed" in status (D17); the other encoders go on.
    python mt_run.py stage2 [--enc E ...] [--n-clu N] [--n-ret N]
        refuses unless records/stage1_*.json, status.json and cutsets.json (also stage1_teacher.json when its pick is ok) are committed unchanged in
        HEAD (smoke: present); scorers per encoder (mt_clu score, mt_ret score; encoders in parallel, at most 3), then
        mt_outcome decide -> OUT/stage2/outcomes.json. An encoder fully voided by status (D17) is skipped, logged.
    python mt_run.py verdicts
        copies OUT/stage2/outcomes.json to records/verdicts.json, only if records/c13.json has "pass": true and the current
        hashes below (outside smoke also committed unchanged in HEAD). C13 is the run chat's Codex job.
    python mt_run.py stage3 [--workers N]
        refuses without records/verdicts.json; runs `mt_diag.py all`.
    python mt_run.py status        reassemble records/status.json from the files on disk

Environment: MT_SMOKE=1 MT_OUTPUTS=<scratch> (records go to <scratch>/records). MT_RUN_STUB=1: steps are not run (rc 0, or 1
for the names in MT_RUN_FAIL, comma list): wiring check only. MT_RUN_TRAILER: lines appended to the printed commit message.

C13 record schema (records/c13.json; sha256 paths relative to OUT, digests of the exact file bytes):
    {"pass": true, "sha256": {"stage2/outcomes.json": "<hex digest>",
                             "stage2/cells_clu.json": "<hex digest>",
                             "stage2/cells_ret.json": "<hex digest>"}}
All three entries must match before verdicts can be copied, including in smoke.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import mt_paths as MP

ROUND = MP.ROUND
OUT = Path(MP.OUT)
SMOKE = bool(MP.SMOKE)
STUB = bool(os.environ.get('MT_RUN_STUB'))
ENCS = tuple(MP.ENCODERS)
ASPECTS = tuple(MP.ASPECTS)
MAX_PARALLEL = 3
OK, FAILED, PENDING = 'ok', 'failed', 'pending'
# Smoke scope of the long replays: a code path each, not the stored intervals (they need the full draws).
SMOKE_REPLAY = dict(clu=['--encs', 'siglip2'], ret=['--parts', 'r14', '--aspects', 'emotion'], vi=['--arms', 'cstar'])
_LOCK = threading.Lock()


def now():
    return time.strftime('%F %H:%M')


def py(module, *args):
    return [sys.executable, str(ROUND / f'{module}.py'), *map(str, args)]


def jload(path):
    return json.loads(Path(path).read_text())


def jdump(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def finite_json(path):
    """'' when `path` is a JSON file without NaN/Infinity, else the reason."""
    def bad(x):
        raise ValueError(f'non-finite constant {x}')
    try:
        json.loads(Path(path).read_text(), parse_constant=bad)
    except (OSError, ValueError) as e:
        return f'{path}: {e}'
    return ''


# ============================================================================================== steps
def _record(stage, rec):
    d = OUT / stage
    with _LOCK:
        with open(d / 'steps.jsonl', 'a') as fh:
            fh.write(json.dumps(rec) + '\n')


def run_step(stage, name, argv, outputs=(), ok_rcs=(0,)):
    """Run one command (twice at most); log, time and record it. `outputs`: files that must exist and be finite JSON.
    `ok_rcs`: exit codes that are a result, not a crash (a gate that reports failure exits 1). Returns the record."""
    d = OUT / stage
    (d / 'logs').mkdir(parents=True, exist_ok=True)
    log = d / 'logs' / f'{name}.log'
    t_all = time.time()
    for attempt in (1, 2):
        t0 = time.time()
        problem = ''
        if STUB:
            rc = 1 if name in os.environ.get('MT_RUN_FAIL', '').split(',') else 0
            log.write_text(f'stub: {argv}\n')
        else:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
            with open(log, 'w' if attempt == 1 else 'a') as fh:
                fh.write(f'# attempt {attempt} {now()}\n')
                fh.flush()
                rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROUND)).returncode
            if rc in ok_rcs:
                problems = [p for p in (finite_json(o) for o in outputs) if p]
                problem = '; '.join(problems)
                if problem:
                    rc = 3
        good = rc in ok_rcs and not problem
        rec = dict(step=name, argv=[str(a) for a in argv], rc=rc, attempt=attempt, seconds=round(time.time() - t0, 1),
                   log=str(log), at=now(), failed=not good, **({'problem': problem} if problem else {}))
        _record(stage, rec)
        if good:
            break
    rec['seconds'] = round(time.time() - t_all, 1)
    print(f'MT_RUN {stage} {name} rc={rc} seconds={rec["seconds"]}', flush=True)
    return rec


def run_graph(stage, tasks, workers=MAX_PARALLEL):
    """tasks: list of dict(name, argv, outputs=(), ok_rcs=(0,), deps=()), in priority order. At most `workers` steps run at once.
    Returns {name: record}; a step whose dependency failed is not run and has rc=-1 (recorded, printed)."""
    (OUT / stage / 'logs').mkdir(parents=True, exist_ok=True)
    pending = {t['name']: t for t in tasks}
    done, running = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            for name, t in list(pending.items()):
                dep_failed = [x for x in t.get('deps', ()) if x in done and done[x]['failed']]
                if dep_failed:
                    rec = dict(step=name, argv=[str(a) for a in t['argv']], rc=-1, attempt=0, seconds=0.0, at=now(),
                               skipped=f'dependency failed: {dep_failed}', failed=True)
                    _record(stage, rec)
                    print(f'MT_RUN {stage} {name} rc=-1 seconds=0.0', flush=True)
                    done[name] = rec
                    del pending[name]
                    continue
                if len(running) < workers and all(x in done for x in t.get('deps', ())):
                    running[pool.submit(run_step, stage, name, t['argv'], t.get('outputs', ()), t.get('ok_rcs', (0,)))] = name
                    del pending[name]
            if running:
                fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for f in fin:
                    done[running.pop(f)] = f.result()
    return done


def skip_record(stage, name, why):
    (OUT / stage / 'logs').mkdir(parents=True, exist_ok=True)
    _record(stage, dict(step=name, rc=-1, attempt=0, seconds=0.0, at=now(), skipped=why, failed=True))
    print(f'MT_RUN {stage} {name} rc=-1 seconds=0.0  # skipped: {why}', flush=True)


def summary(stage, done):
    bad = sorted(n for n, r in done.items() if r['failed'])
    print(f'MT_RUN_SUMMARY {stage} steps={len(done)} failed={len(bad)} {bad}', flush=True)
    return 1 if bad else 0


def check_smoke_env():
    if SMOKE:
        assert os.environ.get('MT_OUTPUTS') and OUT != Path(MP.DEFAULT_OUT), 'smoke needs MT_OUTPUTS=<scratch folder>'


def encs_of(args):
    encs = tuple(args.enc or ENCS)
    assert set(encs) <= set(ENCS), encs
    return encs


# ============================================================================================== stage 0
def stage0_tasks(encs, only, n_clu, n_ret):
    s0 = OUT / 'stage0'
    T = []

    def add(name, argv, outputs=(), ok_rcs=(0,), deps=()):
        T.append(dict(name=name, argv=argv, outputs=[str(o) for o in outputs], ok_rcs=ok_rcs, deps=list(deps)))
    clu_args = ['--n-draws', n_clu or MP.SMOKE_DRAWS['clu']] if (SMOKE or n_clu) else []
    ret_args = ['--n-draws', n_ret or MP.SMOKE_DRAWS['ret']] if (SMOKE or n_ret) else []
    # the long replays first (they hold a slot for 8 to 20 minutes), then the quick gates, then the exports
    add('gate_vi', py('mt_ret', 'replay-vi', *(SMOKE_REPLAY['vi'] if SMOKE else [])), [s0 / 'gate_vi.json'], (0, 1))
    add('gate_iii_clu', py('mt_clu', *clu_args, 'replay-iii', *(SMOKE_REPLAY['clu'] if SMOKE else [])),
        [s0 / 'gate_iii_clu.json'], (0, 1))
    add('gate_iii_ret', py('mt_ret', 'replay-iii', *ret_args, *(SMOKE_REPLAY['ret'] if SMOKE else [])),
        [s0 / 'gate_iii_ret.json'], (0, 1))
    add('gate_i', py('mt_run', '_gate', '--which', 'i'), [s0 / 'gate_i.json'])
    add('gate_v', py('mt_run', '_gate', '--which', 'v'), [s0 / 'gate_v.json'])
    for e in encs:
        add(f'gate_ii_{e}', py('mt_export', 'gate-ii', '--enc', e), [s0 / f'gate_ii_{e}.json'], (0, 1))
    for e in encs:
        add(f'gate_iv_{e}', py('mt_matched', 'gate-iv', '--enc', e), [s0 / f'gate_iv_{e}.json'], (0, 1))
    add('cutsets', py('mt_run', '_cutsets'), [MP.record_path('cutsets.json')], deps=['gate_i'])
    for e in encs:
        add(f'export_{e}', py('mt_export', 'export', '--enc', e))
    for e in encs:
        add(f'matched_export_{e}', py('mt_matched', 'export', '--enc', e))
    if only:
        names = {t['name'] for t in T}
        assert set(only) <= names, f'unknown steps {sorted(set(only) - names)}; known: {sorted(names)}'
        T = [dict(t, deps=[x for x in t['deps'] if x in only]) for t in T if t['name'] in only]
    return T


def _flag(doc, *keys):
    """The first of `keys` present in the gate file (smoke uses the wiring flag, full runs the passed flag)."""
    for k in keys:
        if k in doc:
            return bool(doc[k])
    return False


def _load(path):
    path = Path(path)
    return jload(path) if path.is_file() and not finite_json(path) else None


def collect_gates(encs=ENCS):
    """(gate evidence, status gates) from OUT/stage0 files. A missing or unreadable gate file is "failed"."""
    s0 = OUT / 'stage0'
    st = lambda b: OK if b else FAILED                                            # noqa: E731
    gate, sg = {}, {}
    gi, gv = _load(s0 / 'gate_i.json'), _load(s0 / 'gate_v.json')
    gate['i'] = dict(file='gate_i.json', present=gi is not None, **{'pass': bool(gi and gi['pass'])},
                     parts=gi['parts'] if gi else None)
    gate['v'] = dict(file='gate_v.json', present=gv is not None, **{'pass': bool(gv and gv['pass'])},
                     parts=gv['parts'] if gv else None)
    sg['i_nearcopy'] = st(gi and gi['parts']['i_nearcopy'])
    sg['i_default'] = {e: st(gi and gi['parts']['i_default'].get(e)) for e in ENCS}
    sg['i_teacher'] = {a: st(gi and gi['parts']['i_teacher'].get(a)) for a in ASPECTS}
    sg['v'] = {a: st(gv and gv['parts'].get(a)) for a in ASPECTS}
    for g, key in (('ii', 'pass'), ('iv', 'pass')):
        sg[g], gate[g] = {}, {}
        for e in ENCS:
            d = _load(s0 / f'gate_{g}_{e}.json')
            sg[g][e] = st(d is not None and bool(d.get(key)))
            gate[g][e] = dict(file=f'gate_{g}_{e}.json', present=d is not None, **{'pass': bool(d and d.get(key))},
                              max_abs_diff=(d or {}).get('max_abs_diff'))
    parts = {}
    for name, fn, keys in (('iii_clu', 'gate_iii_clu.json', ('wiring_ok',)), ('iii_ret', 'gate_iii_ret.json', ('values_ok',)),
                           ('vi', 'gate_vi.json', ('values_ok',))):
        d = _load(s0 / fn)
        ok = _flag(d, *(keys if SMOKE else ('passed',))) if d else False
        parts[name] = ok
        gate[name] = dict(file=fn, present=d is not None, smoke=SMOKE, **{'pass': bool(d and d.get('passed'))},
                          ok_for_status=ok, max_abs_diff=(d or {}).get('max_abs_diff'), n_draws=(d or {}).get('n_draws'),
                          scope=SMOKE_REPLAY[name.split('_')[-1]] if SMOKE else 'full')
    sg['iii'] = st(parts['iii_clu'] and parts['iii_ret'])
    sg['vi'] = st(parts['vi'])
    # Failed exports invalidate their gates even if an older gate file passed (D17).
    for e in ENCS:
        for step, g in (('export', 'ii'), ('matched_export', 'iv')):
            rec = last_step('stage0', f'{step}_{e}')
            if rec is not None and rec.get('failed', rec['rc'] != 0):
                sg[g][e] = FAILED
                gate[g][e]['export_step_failed'] = True
    return gate, sg


def last_step(stage, name):
    f = OUT / stage / 'steps.jsonl'
    if not f.is_file():
        return None
    recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r['step'] == name]
    if not recs:
        return None
    r = recs[-1]
    return r


def stage1_status(steps=None):
    s1 = OUT / 'stage1'
    st = lambda b: OK if b else FAILED                                            # noqa: E731

    def good(step, filename):
        rec = steps.get(step) if steps is not None else last_step('stage1', step)
        return rec is not None and rec['rc'] == 0 and not rec.get('failed', False) and _load(s1 / filename) is not None

    ranges = {e: st(good(f'clu_ranges_{e}', f'clu_ranges_{e}.json') and
                    good(f'ret_ranges_{e}', f'ret_ranges_{e}.json')) for e in ENCS}
    picks = {e: st(good(f'ret_pick_{e}', f'picks_{e}.json')) for e in ENCS}
    return dict(ranges=ranges, picks=picks, teacher_pick=st(good('pick_teacher', 'picks_teacher.json')))


def write_status(stage1_done, stage1_steps=None):
    gate, sg = collect_gates()
    jdump(OUT / 'stage0' / 'gate.json', gate)
    s1 = stage1_status(stage1_steps) if stage1_done else dict(ranges={e: PENDING for e in ENCS}, picks={e: PENDING for e in ENCS},
                                                 teacher_pick=PENDING)
    status = dict(gates=sg, stage1=s1)
    jdump(MP.record_path('status.json'), status)
    return status


def run_stage0(args):
    check_smoke_env()
    encs = encs_of(args)
    done = run_graph('stage0', stage0_tasks(encs, args.only, args.n_clu, args.n_ret))
    status = write_status(stage1_done=False)
    bad = sorted(k for k, v in flat(status['gates']).items() if v != OK)
    print(f'gates not ok: {bad or "none"}; wrote {MP.record_path("status.json")} and {OUT / "stage0/gate.json"}')
    return summary('stage0', done)


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f'{prefix}{k}.'))
        else:
            out[f'{prefix}{k}'] = v
    return out


# ============================================================================================== stage 1
def stage1_tasks(encs):
    s1 = OUT / 'stage1'
    T = []
    for e in encs:
        T.append(dict(name=f'clu_ranges_{e}', argv=py('mt_clu', 'ranges', '--enc', e), outputs=[s1 / f'clu_ranges_{e}.json']))
        T.append(dict(name=f'ret_ranges_{e}', argv=py('mt_ret', 'ranges', '--enc', e), outputs=[s1 / f'ret_ranges_{e}.json']))
    for e in encs:
        T.append(dict(name=f'ret_pick_{e}', argv=py('mt_ret', 'pick', '--enc', e), outputs=[s1 / f'picks_{e}.json'],
                      deps=[f'ret_ranges_{e}']))
    if 'qwen' in encs:
        T.append(dict(name='pick_teacher', argv=py('mt_ret', 'pick-teacher'), outputs=[s1 / 'picks_teacher.json'],
                      deps=['ret_ranges_qwen']))
    return T


def merge_records(encs):
    """records/stage1_<enc>.json from the stage 1 files, unchanged inside: clu_ranges (04's file), ret_ranges and picks (05's
    files); records/stage1_teacher.json is 05's teacher file. Parts that do not exist are left out (status says failed)."""
    s1 = OUT / 'stage1'
    written = []
    for e in encs:
        rec = dict(encoder=e)
        for key, fn in (('clu_ranges', f'clu_ranges_{e}.json'), ('ret_ranges', f'ret_ranges_{e}.json'), ('picks', f'picks_{e}.json')):
            d = _load(s1 / fn)
            if d is not None:
                rec[key] = d
        p = MP.record_path(f'stage1_{e}.json')
        jdump(p, rec)
        written.append(p)
    t = _load(s1 / 'picks_teacher.json')
    if t is not None:
        p = MP.record_path('stage1_teacher.json')
        jdump(p, t)
        written.append(p)
    return written


def commit_command(files):
    rel = [str(Path(f).resolve().relative_to(MP.REPO.resolve())) for f in files]
    trailer = os.environ.get('MT_RUN_TRAILER', '')
    msg = 'r15 stage 1: seed ranges, read-out picks, status and cut sets (before any MMArt test number)'
    if trailer:
        msg += '\n\n' + trailer
    return f"git -C {MP.REPO} add -f {' '.join(rel)} && git -C {MP.REPO} commit -m {json.dumps(msg)} -- {' '.join(rel)}"


def run_stage1(args):
    check_smoke_env()
    encs = encs_of(args)
    done = run_graph('stage1', stage1_tasks(encs))
    files = merge_records(encs)
    status = write_status(stage1_done=True, stage1_steps=done)
    bad = sorted(k for k, v in flat(status['stage1']).items() if v != OK)
    print(f'stage 1 not ok (D17: that encoder is inconclusive): {bad or "none"}')
    if SMOKE:
        print(f'smoke: records are in {MP.record_path("")}; nothing to commit')
    else:
        files += [MP.record_path('status.json')] + ([MP.record_path('cutsets.json')] if MP.record_path('cutsets.json').is_file() else [])
        print('COMMIT stage 1 before stage 2 (the runner never commits). Run:')
        print(commit_command(files))
    return summary('stage1', done)


# ============================================================================================== stage 2
def void_plan(status, e):
    """What to run for encoder e from the committed status (spec §3.5 table, D17): None = fully voided (skip), else
    {'clu': scs, 'ret': scs}."""
    g, s1 = status['gates'], status['stage1']
    if g['i_nearcopy'] != OK or g['iii'] != OK or g['i_default']['qwen'] != OK:
        return None
    if g['i_default'][e] != OK or g['ii'][e] != OK:
        return None
    clu = ['SC1'] + (['SC3'] if g['iv'][e] == OK else [])
    if s1['ranges'][e] == OK or _clu_range_exists(e):
        clu.append('SC4')
    ret = []
    if g['vi'] == OK and s1['ranges'][e] == OK and s1['picks'][e] == OK:
        ret = ['SC2', 'SC4'] if s1['teacher_pick'] == OK else ['SC2']
    return dict(clu=clu, ret=ret)


def _clu_range_exists(e):
    """Clustering ranges may exist when only the retrieval ranges failed (status.ranges is one value per encoder)."""
    rec = _load(MP.record_path(f'stage1_{e}.json'))
    return bool(rec and 'clu_ranges' in rec)


def stage2_tasks(encs, status, n_clu, n_ret):
    s2 = OUT / 'stage2'
    T, skipped = [], []
    for e in encs:
        plan = void_plan(status, e)
        if plan is None:
            skipped.append((f'score_{e}', 'voided by status (D17): every cell of this encoder'))
            continue
        prev = None
        for task, scs in (('clu', plan['clu']), ('ret', plan['ret'])):
            if not scs:
                skipped.append((f'score_{task}_{e}', 'voided by status (D17)'))
                continue
            argv = py('mt_run', '_score', '--task', task, '--enc', e, '--scs', *scs)
            n = n_clu if task == 'clu' else n_ret
            if n:
                argv += ['--n-draws', n]
            T.append(dict(name=f'score_{task}_{e}', argv=argv, deps=[prev] if prev else []))
            prev = f'score_{task}_{e}'
    return T, skipped


def refuse_unless_committed(encs):
    problems = []
    try:
        status = MP.read_committed_record('status.json')
    except Exception as ex:
        return [f'status.json: {ex}']
    names = ['cutsets.json'] + [f'stage1_{e}.json' for e in encs]
    if status['stage1']['teacher_pick'] == OK:
        names.append('stage1_teacher.json')
    for name in names:
        try:
            MP.read_committed_record(name)
        except Exception as ex:                                              # missing, differs from HEAD, not in HEAD
            problems.append(f'{name}: {ex}')
    return problems


def run_stage2(args):
    check_smoke_env()
    encs = encs_of(args)
    problems = refuse_unless_committed(encs)
    if problems:
        print('stage 2 refuses: commit the stage 1 records first (run stage 1, then the printed git command):')
        print('\n'.join(problems))
        return 2
    status = jload(MP.record_path('status.json'))
    assert all(v in (OK, FAILED) for v in flat(status).values()), 'status.json has pending entries: run stage 1'
    failed_aspects = [f'gates.{g}.{a}' for g in ('i_teacher', 'v') for a in ASPECTS if status['gates'][g][a] != OK]
    if failed_aspects:
        print(f'stage 2 refuses: failed aspect gates: {", ".join(failed_aspects)}; '
              'the per-aspect SC4 skip of §3.5 is not built; the failure must be traced before scoring')
        return 2
    tasks, skipped = stage2_tasks(encs, status, args.n_clu, args.n_ret)
    (OUT / 'stage2' / 'logs').mkdir(parents=True, exist_ok=True)
    for name, why in skipped:
        skip_record('stage2', name, why)
    done = run_graph('stage2', tasks)
    rc = summary('stage2', done)
    decide = run_step('stage2', 'decide', py('mt_outcome', 'decide'), [OUT / 'stage2/outcomes.json'])
    done['decide'] = decide
    return 1 if (rc or decide['failed']) else 0


def c13_ok():
    """'' when verdicts may be written, else the reason (§3.4: C13 passes before any verdict is written)."""
    p = MP.record_path('c13.json')
    if not p.is_file():
        return f'{p} does not exist: the run chat (Codex) writes it after C13 and the controller commits it'
    try:
        rec = MP.read_committed_record('c13.json')
    except Exception as ex:
        return str(ex)
    if rec.get('pass') is not True:
        return f'{p}: "pass" is not true'
    hashes = rec.get('sha256')
    for name in ('stage2/outcomes.json', 'stage2/cells_clu.json', 'stage2/cells_ret.json'):
        if not isinstance(hashes, dict) or name not in hashes:
            return f'{p}: missing sha256 entry for {name}'
        try:
            digest = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        except OSError as ex:
            return f'C13 checked file {name} cannot be read: {ex}'
        if hashes[name] != digest:
            return f'{p}: sha256 mismatch for {name}; rerun C13 on the current outputs'
    return ''


def run_verdicts(args):
    check_smoke_env()
    src = OUT / 'stage2' / 'outcomes.json'
    if not src.is_file():
        print(f'verdicts refuses: {src} does not exist (stage 2 decide)')
        return 2
    why = c13_ok()
    if why:
        print(f'verdicts refuses: {why}')
        return 2
    dst = MP.record_path('verdicts.json')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f'MT_RUN verdicts copy rc=0 seconds=0.0 -> {dst}')
    if not SMOKE:
        print(f'COMMIT: git -C {MP.REPO} add -f {dst.resolve().relative_to(MP.REPO.resolve())} && git -C {MP.REPO} commit -m "r15 verdicts" -- {dst.resolve().relative_to(MP.REPO.resolve())}')
    return 0


# ============================================================================================== stage 3
def run_stage3(args):
    check_smoke_env()
    if not MP.record_path('verdicts.json').is_file():
        print('stage 3 refuses: records/verdicts.json does not exist (run stage 2, C13, then verdicts)')
        return 2
    extra = ['--workers', args.workers] if args.workers else []
    done = run_graph('stage3', [dict(name='diag_all', argv=py('mt_diag', 'all', *extra))])
    return summary('stage3', done)


# ============================================================================================== internal steps
def cmd_gate(args):
    import mt_data
    res = mt_data.gate_i() if args.which == 'i' else mt_data.gate_v()
    jdump(OUT / 'stage0' / f'gate_{args.which}.json', res)
    print(f'gate ({args.which}) pass={res["pass"]} parts={json.dumps(res["parts"])}')
    return 0


def cmd_cutsets(args):
    import mt_data
    rec = {}
    for name, thr in (('test', MP.CUT_TEST), ('far', MP.CUT_FAR)):
        c = mt_data.cut(thr)
        ks = mt_data.keys(name)
        assert sorted(ks) == sorted(c['kept']), name
        rec[name] = dict(threshold=thr, n=len(ks), n_new=c['n_new'], dropped=dict(exact=len(c['exact']), near=len(c['near']),
                                                                              md5=len(c['md5'])),
                         sha256=hashlib.sha256('\n'.join(sorted(ks)).encode('utf-8')).hexdigest())
    jdump(MP.record_path('cutsets.json'), rec)
    print(json.dumps({k: {x: v[x] for x in ('n', 'sha256')} for k, v in rec.items()}))
    return 0


def cmd_score(args):
    """One encoder's scorer on the test cut. The cells file is merged under a lock, so encoders can run in parallel."""
    path_lock = OUT / 'stage2' / '.cells.lock'
    path_lock.parent.mkdir(parents=True, exist_ok=True)
    n = args.n_draws
    if args.task == 'clu':
        import mt_clu as M
        cells = M.score_encoder(args.enc, 'test', tuple(args.scs), n, 8, write=False)
        with open(path_lock, 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            M.write_cells(M.cells_path('test'), cells)
        path = M.cells_path('test')
    else:
        import mt_ret as M
        n = n or (MP.SMOKE_DRAWS['ret'] if MP.SMOKE else MP.N_DRAWS['ret'])
        cells = M.score_encoder(args.enc, 'test', tuple(args.scs), n, 8)
        path = M.cells_path('test')
        with open(path_lock, 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            M.bs_artret.write_cells(path, cells, [args.enc])
    print(f'{args.enc}: {len(cells)} {args.task} cells -> {path}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, draws=False):
        p.add_argument('--enc', nargs='+', choices=ENCS, default=None, help='encoders (default all four)')
        if draws:
            p.add_argument('--n-clu', type=int, default=None, help='clustering draws (default 2,000; smoke 40)')
            p.add_argument('--n-ret', type=int, default=None, help='retrieval draws (default 10,000; smoke 100)')
    p = sub.add_parser('stage0')
    common(p, True)
    p.add_argument('--only', nargs='+', default=None, help='run only these steps (names as in MT_RUN lines)')
    common(sub.add_parser('stage1'))
    common(sub.add_parser('stage2'), True)
    sub.add_parser('verdicts')
    p = sub.add_parser('stage3')
    p.add_argument('--workers', type=int, default=None)
    sub.add_parser('status')
    p = sub.add_parser('_gate')
    p.add_argument('--which', choices=('i', 'v'), required=True)
    sub.add_parser('_cutsets')
    p = sub.add_parser('_score')
    p.add_argument('--task', choices=('clu', 'ret'), required=True)
    p.add_argument('--enc', choices=ENCS, required=True)
    p.add_argument('--scs', nargs='+', required=True)
    p.add_argument('--n-draws', type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == 'status':
        s1 = (OUT / 'stage1').is_dir() and any((OUT / 'stage1').glob('picks_*.json'))
        print(json.dumps(write_status(bool(s1)), indent=1))
        return 0
    fn = dict(stage0=run_stage0, stage1=run_stage1, stage2=run_stage2, verdicts=run_verdicts, stage3=run_stage3,
              _gate=cmd_gate, _cutsets=cmd_cutsets, _score=cmd_score)[args.cmd]
    return fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
