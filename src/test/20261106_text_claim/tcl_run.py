"""r23 stage runner (ticket 09; spec §3.5, §6, D2, D13). Plumbing only: every number comes from the round's modules, called
as one subprocess per step (a crash is a failed step, not a dead runner). Pattern: r15's `mt_run.py`.

    python tcl_run.py stage0|stage1|stage2|verdicts|stage3|status [--only STEP ...] [--list]

stage0   extraction (13 jobs, resumable by done.json), then gates (i) to (iv); writes records/status.json
stage1   validation scoring (art_val full plan; mm_val per kind), then stage1_art and stage1_mm; writes status.json
stage2   the bar only (art_held, mm_test narrative), then `decide`; refuses unless stage 1 records and status are committed
verdicts copies stage2/outcomes.json to records/verdicts.json after records/c13.json passes and its hashes match
stage3   side rows, the other four MMArt kinds, diagnostics; refuses without records/verdicts.json (smoke excepted)
status   rewrites records/status.json from the step records and gate files

Each step logs to OUT/stage<k>/logs/<step>.log, appends {step, argv, rc, attempt, seconds, log, at, failed} to
OUT/stage<k>/steps.jsonl and prints `TCL_RUN <stage> <step> rc=<rc> seconds=<s>`. A failed step is retried once; a gate that
reports a fail exits 1 and is a result (its output file must exist, or it is a crash). A step skipped for a failed
dependency or a missing stage 1 record has rc=-1. Exit code: non-zero when a step failed or a gate reported a fail.
The runner never commits; it prints the command. Internal subcommands: `_gate_iv`, `_stage1 --set S`, `_wait --est MIN`.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import tcl_paths as P

ROUND = P.ROUND
AMSTERDAM = ZoneInfo('Europe/Amsterdam')
GROUPS = ('art', 'mm')
HELD_OF = {'art': 'art_held', 'mm': 'mm_test'}


def now():
    return datetime.now(AMSTERDAM).strftime('%F %H:%M')


def py(module, *args):
    return [sys.executable, str(ROUND / f'{module}.py'), *map(str, args)]


def heavy_wait_seconds(now, est_minutes):
    """D13: seconds to wait so that a job started now with this estimate does not overlap 10:00 to 19:00 on a weekday
    (Amsterdam time, `now` a datetime in it). If it would, wait until 19:00 of that day. Weekends are free."""
    end = now + timedelta(minutes=est_minutes)
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        if day.weekday() < 5:
            lo, hi = day + timedelta(hours=10), day + timedelta(hours=19)
            if now < hi and end > lo:
                return math.ceil((hi - now).total_seconds())
        day += timedelta(days=1)
    return 0


def stage_dir(stage):
    return P.OUT / stage


def last_step(stage, name):
    f = stage_dir(stage) / 'steps.jsonl'
    recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.is_file() else []
    recs = [r for r in recs if r['step'] == name]
    return recs[-1] if recs else None


def _record(stage, rec):
    with open(stage_dir(stage) / 'steps.jsonl', 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


def run_step(stage, name, argv, outputs=(), ok_rcs=(0,)):
    """Run one command (twice at most); log, time and record it. `outputs` must exist after an ok exit. `ok_rcs`: exit codes
    that are a result, not a crash. Returns the last record (rc, failed, seconds)."""
    d = stage_dir(stage)
    (d / 'logs').mkdir(parents=True, exist_ok=True)
    log = d / 'logs' / f'{name}.log'
    t_all = time.time()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    for attempt in (1, 2):
        t0 = time.time()
        with open(log, 'w' if attempt == 1 else 'a') as fh:
            fh.write(f'# attempt {attempt} {now()}\n')
            fh.flush()
            rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROUND)).returncode
        problem = ''
        if rc in ok_rcs:
            missing = [str(o) for o in outputs if not Path(o).is_file()]
            if missing:
                rc, problem = 3, f'output missing: {missing}'
        good = rc in ok_rcs
        rec = dict(step=name, argv=[str(a) for a in argv], rc=rc, attempt=attempt, seconds=round(time.time() - t0, 1),
                   log=str(log), at=now(), failed=not good, **({'problem': problem} if problem else {}))
        _record(stage, rec)
        if good:
            break
    rec['seconds'] = round(time.time() - t_all, 1)
    print(f'TCL_RUN {stage} {name} rc={rec["rc"]} seconds={rec["seconds"]}', flush=True)
    return rec


def skip_step(stage, name, why):
    stage_dir(stage).joinpath('logs').mkdir(parents=True, exist_ok=True)
    rec = dict(step=name, argv=[], rc=-1, attempt=0, seconds=0.0, log='', at=now(), failed=True, problem=why)
    _record(stage, rec)
    print(f'TCL_RUN {stage} {name} rc=-1 seconds=0.0 skipped: {why}', flush=True)
    return rec


def jload(path):
    return json.loads(Path(path).read_text())


# ================================================================================================== steps
def score_args(set_name, kind=None, rows=None):
    a = ['--set', set_name] + (['--kind', kind] if kind else []) + (['--rows', *rows] if rows else [])
    return py('tcl_score', *a, *(['--smoke'] if P.SMOKE else []))


def gate_path(name):
    return P.OUT / 'stage0' / f'{name}.json'


def stage_steps(stage):
    """Ordered steps of a stage: dicts name, argv, outputs, ok_rcs, plus group where a stage 1 / held group applies."""
    S = lambda name, argv, outputs=(), ok_rcs=(0,), **kw: dict(name=name, argv=argv, outputs=outputs, ok_rcs=ok_rcs, **kw)  # noqa: E731
    bar = ('cstar', 'late', 'plain', 'teacher')
    if stage == 'stage0':
        steps = [S('extract_art_teacher', py('tcl_extract', 'art_teacher')), S('extract_mm_plain', py('tcl_extract', 'mm_plain'))]
        steps += [S(f'extract_mm_teacher_{k}_{s}', py('tcl_extract', 'mm_teacher', '--kind', k, '--split', s))
                  for k in P.KINDS for s in ('val', 'test')]
        steps += [S('extract_roberta', py('tcl_extract', 'roberta')),
                  S('gate_i', py('tcl_extract', 'gate_i'), [gate_path('gate_i')]),
                  S('gate_ii', py('tcl_forward', 'cuda'), [gate_path('gate_ii')]),
                  S('gate_iii', py('tcl_replay', 'cuda'), [gate_path('gate_iii')], (0, 1)),
                  S('gate_iv', py('tcl_run', '_gate_iv'), [gate_path('gate_iv')], (0, 1))]
        return steps
    if stage == 'stage1':
        steps = [S('score_art_val', score_args('art_val'), group='art')]
        steps += [S(f'score_mm_val_{k}', score_args('mm_val', k), group='mm') for k in P.KINDS]
        steps += [S(f'stage1_{g}', py('tcl_run', '_stage1', '--set', f'{g}_val'), [P.record_path(f'stage1_{g}.json')], group=g,
                    final=True) for g in GROUPS]
        return steps
    if stage == 'stage2':
        return [S('score_art_held', score_args('art_held', rows=bar), group='art'),
                S('score_mm_test_narrative', score_args('mm_test', 'narrative', bar), group='mm'),
                S('decide', py('tcl_outcome'), [P.OUT / 'stage2/outcomes.json', P.OUT / 'stage2/cells.json'])]
    if stage == 'stage3':
        steps = [S('score_art_held_side', score_args('art_held', rows=P.SIDE_ROWS), group='art')]
        steps += [S(f'score_mm_test_{k}', score_args('mm_test', k, bar), group='mm') for k in P.KINDS if k != 'narrative']
        steps.append(S('diag', py('tcl_diag', '--device', 'cuda')))
        return steps
    raise AssertionError(stage)


def pick(stage, only):
    steps = stage_steps(stage)
    if only:
        unknown = set(only) - {s['name'] for s in steps}
        assert not unknown, f'unknown steps for {stage}: {sorted(unknown)}; see --list'
        steps = [s for s in steps if s['name'] in only]
    return steps


# ================================================================================================== status
def build_status():
    """records/status.json content from the gate files and step records (ticket 08 reads it)."""
    status = {}
    if gate_path('gate_i').is_file():
        status['gate_i'] = {k: bool(v) for k, v in jload(gate_path('gate_i'))['passed'].items()}
    if gate_path('gate_ii').is_file():
        status['gate_ii'] = {g: bool(jload(gate_path('gate_ii'))['pass'][g]) for g in GROUPS}
    if gate_path('gate_iii').is_file():
        rec = last_step('stage0', 'gate_iii')
        status['gate_iii'] = bool(rec is not None and rec['rc'] == 0 and jload(gate_path('gate_iii'))['overall'])
    if gate_path('gate_iv').is_file():
        status['gate_iv'] = {g: bool(jload(gate_path('gate_iv'))[g]) for g in GROUPS}
    status['stage1'] = {}
    for g in GROUPS:
        rec = last_step('stage1', f'stage1_{g}')
        status['stage1'][g] = bool(rec is not None and rec['rc'] == 0 and P.record_path(f'stage1_{g}.json').is_file())
    old = P.record_path('status.json')
    if old.is_file() and 'c13_void' in jload(old):
        status['c13_void'] = jload(old)['c13_void']
    return status


def write_status():
    path = P.record_path('status.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    status = build_status()
    path.write_text(json.dumps(status, sort_keys=True, indent=1) + '\n')
    return status


def commit_command(files, message):
    rel = [str(Path(f).resolve().relative_to(P.REPO.resolve())) for f in files]
    return f"git -C {P.REPO} add -f {' '.join(rel)} && git -C {P.REPO} commit -m {json.dumps(message)} -- {' '.join(rel)}"


def gate_failed(name, rec):
    """True when a gate step reported a fail (its file says so). rec is the step record."""
    if rec['failed']:
        return False                                                  # a crash is already a failed step
    doc = jload(gate_path(name))
    if name == 'gate_i':
        return not doc['overall']
    if name == 'gate_ii':
        return not all(doc['pass'][g] for g in GROUPS)
    if name == 'gate_iii':
        return rec['rc'] != 0 or not doc['overall']
    return not all(doc[g] for g in GROUPS)


# ================================================================================================== stages
def run_stage(stage, only):
    steps = pick(stage, only)
    (stage_dir(stage) / 'logs').mkdir(parents=True, exist_ok=True)
    bad, failed_groups = [], set()
    for s in steps:
        name, group = s['name'], s.get('group')
        if stage == 'stage1' and group in failed_groups and s.get('final'):
            skip_step(stage, name, 'a score step of this group failed')
            P.record_path(f'stage1_{group}.json').unlink(missing_ok=True)
            continue
        if stage == 'stage0' and name.startswith('gate_'):
            gate_path(name).unlink(missing_ok=True)                    # a crash must not leave an older result
        if stage == 'stage1' and s.get('final'):
            P.record_path(f'stage1_{group}.json').unlink(missing_ok=True)
        if stage == 'stage2' and name == 'decide':
            for o in s['outputs']:
                Path(o).unlink(missing_ok=True)
            P.record_path('verdicts.json').unlink(missing_ok=True)     # old verdicts must not outlive their outcomes
        if stage in ('stage2', 'stage3') and group and not P.record_path(f'stage1_{group}.json').is_file():
            skip_step(stage, name, f'records/stage1_{group}.json is missing: {HELD_OF[group]} verdicts are void (§6)')
            continue
        rec = run_step(stage, name, s['argv'], s['outputs'], s['ok_rcs'])
        if rec['failed']:
            bad.append(name)
            if group:
                failed_groups.add(group)
        elif stage == 'stage0' and name.startswith('gate_') and gate_failed(name, rec):
            bad.append(f'{name} (gate reported a fail)')
    if stage in ('stage0', 'stage1'):
        status = write_status()
        print('status.json:', json.dumps(status, sort_keys=True))
    print(f'{stage}: {"bad steps: " + ", ".join(bad) if bad else "all steps ok"}')
    return 1 if bad else 0


def refuse_unless_committed():
    """Early refusal for stages 2 and 3, through the scoring guard. A group without a stage 1 record is skipped by the
    stage (§6), so only the groups that have one are guarded; status.json is always needed."""
    import tcl_score
    problems = []
    try:
        P.read_committed_record('status.json')
    except Exception as ex:
        problems.append(f'status.json: {ex}')
    for g in GROUPS:
        if P.record_path(f'stage1_{g}.json').is_file():
            try:
                tcl_score.held_guard(HELD_OF[g])
            except Exception as ex:
                problems.append(f'{HELD_OF[g]}: {ex}')
    return problems


def run_held(stage, only):
    if stage == 'stage3' and not P.SMOKE and not P.record_path('verdicts.json').is_file():
        print('stage 3 refuses: records/verdicts.json does not exist (run stage 2, C13, then verdicts)')
        return 2
    problems = refuse_unless_committed()
    if problems:
        print(f'{stage} refuses: commit the stage 1 records and status.json first:')
        print('\n'.join(problems))
        return 2
    return run_stage(stage, only)


def c13_ok():
    """'' when verdicts may be written, else the reason (§3.4; smoke without a C13 record is exempt)."""
    p = P.record_path('c13.json')
    if not p.is_file():
        return '' if P.SMOKE else f'{p} does not exist: the run chat writes it after C13 and the controller commits it'
    try:
        rec = P.read_committed_record('c13.json')
    except Exception as ex:
        return str(ex)
    if rec.get('pass') is not True:
        return f'{p}: "pass" is not true'
    hashes = rec.get('sha256')
    for name in ('stage2/outcomes.json', 'stage2/cells.json'):
        if not isinstance(hashes, dict) or name not in hashes:
            return f'{p}: missing sha256 entry for {name}'
        digest = hashlib.sha256((P.OUT / name).read_bytes()).hexdigest()
        if hashes[name] != digest:
            return f'{p}: sha256 mismatch for {name}; rerun C13 on the current outputs'
    return ''


def run_verdicts():
    src = P.OUT / 'stage2' / 'outcomes.json'
    if not src.is_file():
        print(f'verdicts refuses: {src} does not exist (stage 2 decide)')
        return 2
    why = c13_ok()
    if why:
        print(f'verdicts refuses: {why}')
        return 2
    dst = P.record_path('verdicts.json')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    note = ' (smoke: C13 not required)' if P.SMOKE and not P.record_path('c13.json').is_file() else ''
    print(f'TCL_RUN verdicts copy rc=0 seconds=0.0 -> {dst}{note}')
    if not P.SMOKE:
        print('COMMIT:', commit_command([dst], 'r23 verdicts'))
    return 0


# ================================================================================================== internal steps
def cmd_gate_iv():
    import tcl_data
    path = gate_path('gate_iv')
    path.parent.mkdir(parents=True, exist_ok=True)
    doc, sizes = dict(message={}), {}
    for g, check in (('art', tcl_data.gate_iv_art), ('mm', tcl_data.gate_iv_mm)):
        try:
            sizes.update(check())
            doc[g] = True
        except AssertionError as ex:
            doc[g] = False
            doc['message'][g] = str(ex)
    doc['sizes'] = sizes
    path.write_text(json.dumps(doc, indent=1) + '\n')
    print('gate iv:', {g: doc[g] for g in GROUPS}, doc['message'], sizes)
    return 0 if doc['art'] and doc['mm'] else 1


def cmd_stage1(set_name):
    import tcl_stage1
    tcl_stage1.run(set_name)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('stage0', 'stage1', 'stage2', 'stage3'):
        q = sub.add_parser(name)
        q.add_argument('--only', nargs='+')
        q.add_argument('--list', action='store_true', help='print the stage\'s step names and stop')
    for name in ('verdicts', 'status', '_gate_iv'):
        sub.add_parser(name)
    q = sub.add_parser('_stage1')
    q.add_argument('--set', required=True, choices=('art_val', 'mm_val'), dest='set_name')
    q = sub.add_parser('_wait')
    q.add_argument('--est', type=float, required=True, help='job estimate in minutes')
    a = p.parse_args(argv)
    if a.cmd == '_wait':
        print(heavy_wait_seconds(datetime.now(AMSTERDAM), a.est))
        return 0
    if a.cmd == '_gate_iv':
        return cmd_gate_iv()
    if a.cmd == '_stage1':
        return cmd_stage1(a.set_name)
    if a.cmd in ('stage0', 'stage1', 'stage2', 'stage3') and a.list:
        print('\n'.join(s['name'] for s in stage_steps(a.cmd)))
        return 0
    if a.cmd == 'status':
        print(json.dumps(write_status(), sort_keys=True, indent=1))
        return 0
    if a.cmd == 'verdicts':
        return run_verdicts()
    if a.cmd in ('stage2', 'stage3'):
        return run_held(a.cmd, a.only)
    rc = run_stage(a.cmd, a.only)
    if a.cmd == 'stage1':
        files = [P.record_path(f'stage1_{g}.json') for g in GROUPS if P.record_path(f'stage1_{g}.json').is_file()]
        files.append(P.record_path('status.json'))
        if P.SMOKE:
            print(f'smoke: records are in {P.record_path("")}; nothing to commit')
        else:
            print('COMMIT stage 1 and status.json before stage 2 (the runner never commits). Run:')
            print(commit_command(files, 'r23 stage 1: seed ranges, read-out picks and status (before any held number)'))
    return rc


if __name__ == '__main__':
    sys.exit(main())
