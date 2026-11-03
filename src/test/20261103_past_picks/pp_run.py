"""r19 stage runner. CPU steps only; queue each step with pp_run_cpu.sh.

pp_run.py 0|1|2|3|4 [--step NAME ...] [--enc ENCODER ...] [--list]
Logs and retry-once graph copied from r15 mt_run.py (run_step/run_graph).
The runner never commits. Stage 2 requires records unchanged in HEAD.
"""
import pp_paths as PP
import pp_register  # noqa: F401: register before any inherited imports
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROUND, OUT = PP.ROUND, Path(PP.OUT)
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
        before = {str(o): Path(o).stat().st_mtime_ns if Path(o).exists() else None for o in outputs}
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        with open(log, 'w' if attempt == 1 else 'a') as fh:
            fh.write(f'# attempt {attempt} {now()}\n')
            fh.flush()
            rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROUND)).returncode
        if rc in ok_rcs:
            problems = [p for p in (finite_json(o) for o in outputs) if p]
            if name.startswith('gate_'):
                problems += [f'{o}: stale gate output was not rewritten' for o in outputs
                             if Path(o).exists() and Path(o).stat().st_mtime_ns == before[str(o)]]
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
    print(f'PP_RUN {stage} {name} rc={rc} seconds={rec["seconds"]}', flush=True)
    return rec


def run_graph(stage, tasks, workers=1):
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
                    print(f'PP_RUN {stage} {name} rc=-1 seconds=0.0', flush=True)
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




def load(path):
    path = Path(path)
    if not path.exists():
        return None
    assert not finite_json(path), finite_json(path)
    return jload(path)


def flag(doc, key):
    """Use the writer's exact boolean key. Never infer a gate from rc or wiring_ok."""
    if doc is None:
        return False
    assert key in doc and type(doc[key]) is bool, f'gate requires boolean {key}: {sorted(doc)}'
    return doc[key]


def last_step(stage, name):
    path = OUT / f'stage{stage}' / 'steps.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    return next((r for r in reversed(rows) if r['step'] == name), None)


def collect_gates():
    """Return evidence and the EXACT pp_outcome status schema; missing evidence fails closed."""
    from pp_outcome import STORE_SETS, check_status
    evidence = {}

    def read(name, key='pass', sub=None):
        path = OUT / 'stage0' / name
        doc = load(path)
        part = doc.get(sub) if doc is not None and sub else doc
        ok = flag(part, key)
        step = Path(name).stem
        step = {'gate_iii': f'gate_iii_{sub}', 'gate_iv_r14': 'gate_iv_r14_merge',
                'gate_vii_r14': 'gate_vii_r14_merge', 'gate_vi': 'semart_set'}.get(step, step)
        latest = last_step(0, step)
        if latest and latest['failed']:
            ok = False
        evidence[name + (f'/{sub}' if sub else '')] = dict(file=str(path), present=doc is not None,
                                                        key=key, **{'pass': ok}, evidence=part)
        return 'ok' if ok else 'failed'

    def all_ok(values):
        values = list(values)
        return 'ok' if all(v == 'ok' for v in values) else 'failed'

    g = dict(i_store={e: {s: read(f'gate_i/store_{s}_{e}.json', 'ok') for s in sets}
                      for e, sets in STORE_SETS.items()},
             i_art_parity={e: read(f'gate_i/art_{e}.json') for e in (*PP.NEW_ENCODERS, 'siglip1', 'clip_l14')},
             i_semart_parity={e: read(f'gate_i/semart_{e}.json') for e in ('clip_b32', 'qwen', 'siglip2')},
             i_semart_genre_word=read('gate_i/genre_word.json'),
             i_aff_l14_parity=read('gate_i/aff_clip_l14.json'), ii=read('gate_ii.json'),
             iii={p: read('gate_iii.json', sub=p) for p in ('r10', 'r14')},
             iv=all_ok(read(f'gate_iv_{p}.json', 'passed') for p in ('r14', 'r15', 'semart')),
             v={e: read(f'gate_v_{e}.json') for e in PP.NEW_ENCODERS}, vi=read('gate_vi.json'),
             vii={}, viii=read('gate_viii.json'))
    # MMArt store checks alone do not establish extraction parity of the writing code.
    for e in PP.NEW_ENCODERS:
        g['i_store'][e]['mmart'] = all_ok((g['i_store'][e]['mmart'], read(f'gate_i/mmart_{e}.json')))
    for p in ('clu', 'ret', 'affection'):
        rec = last_step(0, f'gate_iv_r14_{p}')
        if rec and rec['failed']:
            g['iv'] = 'failed'
    g['i_semart_genre_word'] = all_ok((g['i_semart_genre_word'], read('gate_i/store_semart_genre_word.json', 'ok')))
    v14 = read('gate_vii_r14.json', 'passed')
    if any((last_step(0, f'gate_vii_r14_{e}') or {}).get('failed') for e in ('siglip1', 'clip_l14')):
        v14 = 'failed'
    g['vii'] = dict(artelingo=v14, affection=v14, mmart=read('gate_vii_r15.json', 'passed'))
    runs = {}
    for e in PP.NEW_ENCODERS:
        runs[e] = {}
        for kind in ('cstar', 'photo'):
            doc = load(OUT / 'node' / f'{kind}_{e}.json')
            ok = bool(doc and doc.get('status') == 'ok' and sorted(doc.get('seeds', [])) == list(PP.SEEDS))
            runs[e][kind] = 'ok' if ok else 'failed'
        if runs[e]['cstar'] != 'ok':
            runs[e]['photo'] = 'failed'
        runs[e]['matched'] = g['v'][e]
    if g['viii'] == 'failed' and load(OUT / 'stage0/gate_viii.json') is not None:
        # A loaded probe is unresolved, never entered as a failed gate in the voids table.
        timings = [load(p) for p in (OUT / 'probe/batches').glob('*.json')]
        loads = [d[k][0] for d in timings if d for k in ('cpu_load_before', 'cpu_load_after') if k in d]
        assert loads or PP.SMOKE, 'failed probe has no recorded CPU load: investigate before freezing status'
        assert not loads or max(loads) <= 16, 'probe failed above load 16: move OUT/probe aside and rerun quiet before status'
    status = check_status(dict(gates=g, runs=runs))
    return dict(gates=g, sources=evidence,
                notes=['Sample M5 rerun ran on the local 3090; controller accepted its byte-equal weights.']), status


def write_status():
    evidence, status = collect_gates()
    jdump(OUT / 'stage0/gate.json', evidence)
    jdump(PP.record_path('status.json'), status)
    print(evidence['notes'][0])


def assert_semart_stores(doc):
    required = {str(p.relative_to(PP.SEMART_PRE)) for (s, _), p in PP.NEW_STORE.items() if s == 'semart'}
    assert required <= set(doc['stores_checked']), f'missing SemArt stores: {sorted(required - set(doc["stores_checked"]))}'


def merge_records(encs, announce=True):
    """Merge available blocks only; a missing encoder never blocks the others or fabricates a range."""
    names = dict(artclu='artclu_ranges', artret_ranges='artret_ranges', artret_picks='artret_picks',
                 affection='affection_ranges', mmart='mmart')
    paths = []
    for enc in encs:
        if enc not in PP.NEW_ENCODERS:
            continue
        doc = dict(encoder=enc, kind='stage1_record')
        for key, prefix in names.items():
            block = load(OUT / 'stage1' / f'{prefix}_{enc}.json')
            if block is not None:
                assert isinstance(block, dict) and block.get('encoder', enc) == enc, f'{key}: wrong encoder block for {enc}'
                doc[key] = block
        path = PP.record_path(f'stage1_{enc}.json')
        jdump(path, doc)
        paths.append(path)
    semart = load(OUT / 'stage1/semart.json') or {}
    for enc in encs:
        part = load(OUT / 'stage1' / f'semart_{enc}.json')
        if part is not None:
            assert set(part) == {enc}, (enc, part.keys())
            semart.update(part)
    if semart:
        from pp_semart_score import check_stage1
        # Check each available entry. Absent entries remain absent: their own scoring step fails.
        import pp_semart_score as ss
        old = ss.ENCODERS
        try:
            ss.ENCODERS = tuple(semart)
            check_stage1(semart)
        finally:
            ss.ENCODERS = old
        jdump(OUT / 'stage1/semart.json', semart)
        path = PP.record_path('stage1_semart.json')
        jdump(path, semart)
        paths.append(path)
    paths += [PP.record_path(n) for n in ('status.json', 'semart_set.json') if PP.record_path(n).exists()]
    if not announce:
        return paths
    if PP.SMOKE:
        print(f'Smoke records: {PP.record_path("")}')
    else:
        rel = [str(p.relative_to(PP.REPO)) for p in paths]
        msg = ('r19 stage 1: freeze validation ranges, picks, set and status before scoring\n\n'
               'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
               'Claude-Session: https://claude.ai/code/session_013nwCnEBnr7Jcqn7rpm899u')
        print(f'git -C {shlex.quote(str(PP.REPO))} add -f {shlex.join(rel)} && '
              f'git -C {shlex.quote(str(PP.REPO))} commit -m {shlex.quote(msg)} -- {shlex.join(rel)}')


def refuse_unless_committed(encs):
    names = ['status.json', 'semart_set.json', 'stage1_semart.json']
    names += [f'stage1_{e}.json' for e in encs if e in PP.NEW_ENCODERS]
    problems = []
    for name in names:
        try:
            PP.read_committed_record(name)
        except (OSError, ValueError, RuntimeError) as ex:
            problems.append(f'{name}: {ex}')
    return problems


def c13_check():
    """C13 is the run chat's Codex job, before outcomes; bind it to the exact saved inputs."""
    if PP.SMOKE and not PP.record_path('c13.json').exists():
        return
    rec = PP.read_committed_record('c13.json')
    assert rec['pass'] is True, 'C13 did not pass'
    paths = {str(p.relative_to(OUT)): p for d in ('stage2', 'stage3')
             for p in (OUT / d).glob('cells_*.json') if not p.name.endswith('.pending.json')}
    paths.update({str(p.relative_to(OUT)): p for p in (OUT / 'probe').rglob('*.json')})
    paths.update({f'records/{p.name}': p for p in [PP.record_path('status.json'), PP.record_path('semart_set.json'),
                  *PP.record_path('').glob('stage1_*.json')]})
    assert paths and 'stage3/cells_cost.json' in paths, 'C13 needs saved cells'
    for name, path in paths.items():
        assert rec['sha256'].get(name) == hashlib.sha256(path.read_bytes()).hexdigest(), f'C13 hash missing or changed: {name}'


def internal(action, enc=None, encs=PP.ENCODERS):
    if action == 'gate-ii':
        import pp_node as n
        import bs_cstar
        import bs_photo
        root = n.REGRESS_ROOT
        croot = root / 'cstar' / n.REGRESS_ENC
        assert bs_cstar.record_path(croot, bs_cstar.STAR_ARM, 0).is_file(), 'pull gate (ii) c* training first'
        assert (bs_photo.run_dir(n.REGRESS_ENC, bs_photo.REF, 0, root) / 'run.json').is_file(), 'pull gate (ii) photo training first'
        return n.regress_ii('cpu')
    elif action == 'collect':
        write_status()
    elif action == 'semart-set':
        import pp_semart as s
        doc = s.record()
        assert_semart_stores(doc)
        jdump(OUT / 'stage0/gate_vi.json', doc['gate'])
        jdump(PP.record_path('semart_set.json'), doc)
    elif action == 'semart-entry':
        import pp_semart_score as s
        # Development only: no held rows; new records need not be committed until ALL picks are ready.
        original = s.load_record
        s.load_record = lambda e: jload(PP.record_path(f'stage1_{e}.json')) if e in PP.NEW_ENCODERS else original(e)
        jdump(OUT / 'stage1' / f'semart_{enc}.json', s.stage1((enc,)))
    elif action == 'semart-score':
        import pp_semart_score as s
        doc = PP.read_committed_record('stage1_semart.json')
        assert enc in doc, f'no SemArt validation record for {enc}'
        s.ENCODERS = tuple(doc)  # validate every present entry, keep absent encoder paths independent
        s.score('semart_clu', enc)
        s.score('semart_ret', enc)
    elif action in ('merge', 'merge-encoders'):
        merge_records(encs, announce=action == 'merge')
    elif action == 'decide':
        c13_check()
        import mt_data
        import pp_outcome as o
        semart = PP.read_committed_record('semart_set.json')
        status = PP.read_committed_record('status.json')
        assert collect_gates()[1] == status, 'gate or run evidence changed: recollect and commit status before outcomes'
        cells, costs, files = o.load_cells(OUT)
        counts = dict(mmart_n=len(mt_data.keys('test')), semart_n=semart['n_set'], semart_artists=semart['artists_set'])
        res = o.decide(cells, status, costs, counts=counts, smoke=bool(PP.SMOKE))
        res['cell_files'] = files
        jdump(OUT / 'stage4/outcomes.json', res)
        print(o.table(res))
    else:
        raise ValueError(action)
    return 0


def tasks(stage, encs):
    """Each command is a separately queueable unit. GPU/node work is deliberately outside this CPU graph."""
    result = []
    def add(name, module, *args, est='15m', deps=(), outputs=(), gate=False):
        result.append(dict(name=name, argv=py(module, *args), est=est, deps=deps,
                           outputs=[OUT / p for p in outputs], ok_rcs=(0, 1) if gate else (0,)))
    def own(name, action, enc=None, **kw):
        add(name, 'pp_run', '_internal', action, *(['--encoder', enc] if enc else []), **kw)
    if stage == 0:
        own('gate_ii', 'gate-ii', est='2m', gate=True, outputs=['stage0/gate_ii.json'])
        for p in ('r10', 'r14'):
            add(f'gate_iii_{p}', 'pp_export', 'gate-iii', '--part', p, gate=True, outputs=['stage0/gate_iii.json'])
        for p in ('clu', 'ret', 'affection'):
            add(f'gate_iv_r14_{p}', 'pp_gate14', 'iv', p, gate=True, outputs=[f'stage0/gate_iv_r14_{p}.json'])
        add('gate_iv_r14_merge', 'pp_gate14', 'iv', 'merge', est='1m', gate=True, outputs=['stage0/gate_iv_r14.json'])
        add('gate_iv_r15', 'pp_mmart', 'replay-iv', est='30m', gate=True, outputs=['stage0/gate_iv_r15.json'])
        add('gate_iv_semart', 'pp_semart_score', 'gate-iv', est='5m', gate=True, outputs=['stage0/gate_iv_semart.json'])
        for e in PP.NEW_ENCODERS:
            add(f'gate_v_{e}', 'pp_matched', 'gate-v', '--enc', e, est='3m', gate=True, outputs=[f'stage0/gate_v_{e}.json'])
        own('semart_set', 'semart-set', est='4m')
        for e in ('siglip1', 'clip_l14'):
            add(f'gate_vii_r14_{e}', 'pp_gate14', 'vii', e, gate=True, outputs=[f'stage0/gate_vii_r14_{e}.json'])
        add('gate_vii_r14_merge', 'pp_gate14', 'vii', 'merge', est='1m', gate=True, outputs=['stage0/gate_vii_r14.json'])
        add('gate_vii_r15', 'pp_mmart', 'replay-vii', est='17m', gate=True, outputs=['stage0/gate_vii_r15.json'])
        for e in encs:
            if e in PP.NEW_ENCODERS:
                for ds in ('artelingo', 'affection', 'mmart'):
                    add(f'export_{ds}_{e}', 'pp_export', ds, '--encoder', e)
            add(f'export_semart_{e}', 'pp_export', 'semart', '--encoder', e)
            for ds in (('semart', 'mmart_new') if e in PP.NEW_ENCODERS else ('semart',)):
                add(f'matched_{ds}_{e}', 'pp_matched', 'export', '--set', ds, '--enc', e)
        own('collect', 'collect', est='1m')
    elif stage == 1:
        for e in encs:
            if e in PP.NEW_ENCODERS:
                for p in ('artelingo', 'affection'):
                    add(f'stage1_{p}_{e}', 'pp_score14', 'stage1', '--part', p, '--enc', e)
                add(f'stage1_mmart_{e}', 'pp_mmart', 'stage1', '--enc', e, est='20m')
        add('merge_encoders', 'pp_run', '_internal', 'merge-encoders', '--enc', *encs, est='1m')
        for e in encs:
            own(f'stage1_semart_{e}', 'semart-entry', e, est='4m')
        own('collect', 'collect', est='1m')
        add('merge', 'pp_run', '_internal', 'merge', '--enc', *encs, est='1m')
    elif stage == 2:
        for e in encs:
            if e in PP.NEW_ENCODERS:
                for p in ('artelingo_clu', 'artelingo_ret', 'affection'):
                    add(f'score_{p}_{e}', 'pp_score14', 'score', '--part', p, '--enc', e)
                for p in ('clu', 'ret'):
                    add(f'score_mmart_{p}_{e}', 'pp_mmart', 'score', '--enc', e, '--parts', p)
            own(f'score_semart_{e}', 'semart-score', e)
    elif stage == 3:
        add('cells_cost', 'pp_probe', 'cells', '--enc', *encs, est='1m', outputs=['stage3/cells_cost.json'], gate=True)
    else:
        own('decide', 'decide', est='1m', outputs=['stage4/outcomes.json'])
        add('diagnostics', 'pp_diag', 'run', est='30m', deps=('decide',))
    return result


def skip_voided(ts, encs, status):
    """Stage 2: drop (and log as skipped) the steps whose cells are all void in the committed `status`; keep the others."""
    from pp_outcome import Voids, required_cell_keys
    voids = Voids(status)
    kept = []
    for t in ts:
        enc = next(e for e in encs if t['name'].endswith('_' + e))
        if t['name'].startswith('score_semart_'):
            data, task = 'semart_new', None
        elif t['name'].startswith('score_mmart_'):
            data, task = 'mmart_test', ('clu' if '_clu_' in t['name'] else 'ret')
        elif t['name'].startswith('score_affection_'):
            data, task = 'affection_test', None
        else:
            data, task = 'artelingo_held', ('clu' if '_clu_' in t['name'] else 'ret')
        keys = [k for k in required_cell_keys(enc) if k[4] == data and (task is None or k[3] == task)]
        if keys and all(voids.cell(k) for k in keys):
            log = OUT / 'stage2/logs' / (t['name'] + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            why = sorted({reason for k in keys for reason in voids.cell(k)})
            log.write_text('Skipped affected path: ' + '; '.join(why) + '\n')
            _record('stage2', dict(step=t['name'], argv=t['argv'], rc=-1, attempt=0, seconds=0,
                                   failed=True, skipped=why, log=str(log), at=now()))
            print(f'PP_RUN stage2 {t["name"]} skipped: affected path')
        else:
            kept.append(t)
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('0', '1', '2', '3', '4', '_internal', 'status'))
    ap.add_argument('action', nargs='?')
    ap.add_argument('--encoder', choices=PP.ENCODERS)
    ap.add_argument('--enc', nargs='+', choices=PP.ENCODERS, default=list(PP.ENCODERS))
    ap.add_argument('--step', nargs='+')
    ap.add_argument('--list', action='store_true', help='print step and queue estimate, without running or writing')
    args = ap.parse_args(argv)
    if PP.SMOKE:
        assert os.environ.get('PP_OUTPUTS') and OUT.resolve() != PP.DEFAULT_OUT.resolve()
    if args.stage == '_internal':
        return internal(args.action, args.encoder, args.enc)
    if args.stage == 'status':
        return internal('collect')
    stage = int(args.stage)
    ts = tasks(stage, args.enc)
    if args.step:
        names = {t['name'] for t in ts}
        assert set(args.step) <= names, f'unknown steps: {set(args.step) - names}; known: {sorted(names)}'
        ts = [dict(t, deps=[d for d in t['deps'] if d in args.step]) for t in ts if t['name'] in args.step]
    if args.list:
        for t in ts:
            print(t['name'] + '\t' + t['est'])
        return 0
    if stage == 2:
        problems = refuse_unless_committed(args.enc)
        if problems:
            print('stage 2 refuses: commit stage 1 records unchanged in HEAD first\n' + '\n'.join(problems))
            return 2
        from pp_outcome import check_status
        committed_status = check_status(PP.read_committed_record('status.json'))
        assert collect_gates()[1] == committed_status, 'gate or run evidence changed: recollect and commit status before scoring'
    # A separately queued diagnostics command must still require successful current outcomes.
    if stage == 4 and any(t['name'] == 'diagnostics' for t in ts) and not any(t['name'] == 'decide' for t in ts):
        c13_check()
        previous = last_step(4, 'decide')
        assert previous and not previous['failed'], 'diagnostics needs successful outcomes first'
    if stage == 2:
        ts = skip_voided(ts, args.enc, PP.read_committed_record('status.json'))
    done = run_graph(f'stage{stage}', ts)
    bad = [name for name, rec in done.items() if rec['failed']]
    print(f'PP_RUN_SUMMARY stage{stage} steps={len(done)} failed={len(bad)} {bad}')
    return int(bool(bad))


if __name__ == '__main__':
    raise SystemExit(main())
