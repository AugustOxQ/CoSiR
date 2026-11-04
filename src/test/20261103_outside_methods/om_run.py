"""r18 stage runner (ticket 11; spec §3.5 stages 0 to 2, D16, D24, §3.7). Plumbing: no verdict code.

Modelled on r17's `ns_run.py`. Every step writes `OUT/steps/<step>.json` ({step, status, error, seconds, smoke}) whether it
succeeds or not; an exception inside a step is recorded `failed` and the run goes on (D24); the exit code is then 1.
`records/status.json` is built by `status` from the gate files, the stage 1 files and the step files, so it can be rebuilt at
any time; a thing never run is `missing`. Smoke (OM_SMOKE=1, OM_OUTPUTS=<scratch>) sends every record to OUT/records.

  stage0   [--no-run]     gate (i) (CPU, runs om_clay.gate_i) and gate (ii) CPU part (runs om_turtle.gate_ii('cpu')), then
                          `status`, which reads OUT/stage0/gate_{i,ii,iii,iv}.json. Gate (ii) CUDA, gate (iii) and gate (iv)
                          are their own commands (om_queue.sh prints them); this step only reads their files.
  stage1   [--step all|ranges|tpips|clay|records] [--set S] [--workers W]
                          om_stage1's steps (ranges; tpips and clay per set), then `records`: copies
                          OUT/stage1/om_ranges.json, om_picks.json to records/stage1_om_ranges.json, stage1_om_picks.json
                          and rebuilds status. The controller commits the three records before stage 2.
  stage2   [--sc --set --enc --draws --force]
                          refuses unless records/stage1_om_ranges.json, stage1_om_picks.json and status.json are committed
                          unchanged in git HEAD (om_paths.read_committed_record; smoke has no git check there); then om_cells
                          `cells` (a unit that raises is logged and left unscored, the run goes on).
  assemble                om_cells `assemble` -> OUT/stage2/cells.json
  labels                  om_cells `labels` (the guarded step: needs OUT/c13/c13_cells.json) -> records/labels.json
  status | c13-stub (smoke only) | list

A gate counts as `ok` only when its file says pass true AND its code hash is current: gate (ii) parts against om_turtle.py,
gate (iii) parts against `om_gate_iii.code_sha256()` (a part from older code is `failed`, as the gate files say themselves).
Assemble and labels also require committed status to match current gates, stage 1 states and r17's status hash,
including smoke subsets; a changed status must be rebuilt with `status` and committed first (r17's path is ignored).
The CLI requires OM_OUTPUTS and OM_SMOKE=1 together, with a scratch output path different from DEFAULT_OUT.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import om_paths as op  # noqa: E402

OK, FAILED, MISSING = 'ok', 'failed', 'missing'
OUR_GATES = ('i', 'ii', 'iii_a', 'iii_b', 'iii_c', 'iv')
SETS = op.SETS
RECORD_NAMES = ('stage1_om_ranges.json', 'stage1_om_picks.json', 'status.json')


# ------------------------------------------------------------------------------------------------ step files
def step_path(name):
    return op.OUT / 'steps' / f'{name}.json'


def read_step(name):
    p = step_path(name)
    return json.loads(p.read_text()) if p.exists() else None


def write_step(name, status, error=None, seconds=None, **extra):
    op.write_json(step_path(name), dict(step=name, status=status, error=error, seconds=seconds, smoke=bool(op.SMOKE), **extra))


def step_status(name):
    s = read_step(name)
    return MISSING if s is None else s['status']


def run_step(name, fn):
    """Run fn() under a step file; an exception is recorded `failed` (D24) and returns False."""
    t0 = time.monotonic()
    try:
        fn()
        write_step(name, OK, None, time.monotonic() - t0)
        print(f'{name}: ok ({time.monotonic() - t0:.0f}s)', flush=True)
        return True
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        write_step(name, FAILED, f'{type(e).__name__}: {e}', time.monotonic() - t0)
        print(f'{name}: FAILED {type(e).__name__}: {e}', flush=True)
        return False


# ------------------------------------------------------------------------------------------------ gates
def gate_file(name):
    return op.OUT / 'stage0' / f'gate_{name}.json'


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _gate_simple(name, step):
    """ok when the file says pass true; failed when it says anything else or the gate's step raised; else missing."""
    doc = _read(gate_file(name))
    if doc is not None:
        return OK if doc.get('pass') is True else FAILED
    return FAILED if step_status(step) == FAILED else MISSING


def turtle_code_sha256():
    """What om_turtle._code_sha256() is (sha256 of om_turtle.py), without importing torch."""
    return hashlib.sha256((op.ROUND / 'om_turtle.py').read_bytes()).hexdigest()


def gate_ii_status():
    doc = _read(gate_file('ii'))
    if doc is None:
        return FAILED if step_status('gate_ii_cpu') == FAILED else MISSING
    parts = [doc.get(p) for p in ('cpu', 'cuda')]
    if any(p is not None and not p.get('pass') for p in parts):
        return FAILED
    if any(p is not None and p.get('code_sha256') != turtle_code_sha256() for p in parts):
        return FAILED                                       # a part from an older om_turtle.py does not count
    if any(p is None for p in parts):
        return MISSING
    return OK if doc.get('pass') is True else FAILED


def gate_iii_status():
    """{'iii_a', 'iii_b', 'iii_c'}: each part must pass AND carry the current om_gate_iii.code_sha256()."""
    doc = _read(gate_file('iii'))
    out = {}
    if doc is None:
        return {f'iii_{p}': MISSING for p in 'abc'}
    import om_gate_iii
    here = om_gate_iii.code_sha256()
    for p in 'abc':
        part = doc.get(p)
        if part is None:
            out[f'iii_{p}'] = MISSING
        else:
            out[f'iii_{p}'] = OK if (part.get('pass') is True and part.get('code_sha256') == here) else FAILED
    return out


# ------------------------------------------------------------------------------------------------ stage 1 keys
def pick_keys():
    return [f'{kind}/{s}' for s in SETS for kind in ('tpips', 'clay', 'clay_textset/clu', 'clay_textset/ret')]


def stage1_status():
    """{key: s} for exactly the pick keys and range keys a status may name (never `clay_text_sha256`)."""
    import om_stage1 as s1
    picks = _read(op.OUT / 'stage1' / 'om_picks.json') or {}
    ranges = _read(op.OUT / 'stage1' / 'om_ranges.json') or {}
    out = {}
    for k in pick_keys():
        kind, s = k.rsplit('/', 1)
        step = f'stage1_{"tpips" if kind == "tpips" else "clay"}_{s}'
        out[k] = OK if k in picks else (FAILED if step_status(step) == FAILED else MISSING)
    for rk in s1.range_keys():
        k = rk['key']
        if k in ranges and ranges[k].get('r') is not None:
            out[k] = OK
        else:
            out[k] = FAILED if step_status('stage1_ranges') == FAILED else MISSING
    return out


def r17_ref():
    import ns_paths
    path = ns_paths.RECORDS / 'status.json'
    return dict(path=str(path), sha256=op.sha256_file(path))


def build_status(write=True):
    import om_cells
    g = dict(i=_gate_simple('i', 'gate_i'), ii=gate_ii_status(), **gate_iii_status(), iv=_gate_simple('iv', 'gate_iv'))
    status = dict(gates={k: g[k] for k in OUR_GATES}, stage1=stage1_status(), r17=r17_ref())
    om_cells.check_status(status)
    if write:
        op.write_json(op.record_path('status.json'), status)
    return status


def cmd_status(args):
    st = build_status()
    flat = dict(st['gates'], **st['stage1'])
    counts = {s: sum(v == s for v in flat.values()) for s in (OK, FAILED, MISSING)}
    print(f'status -> {op.record_path("status.json")}: {counts}')
    print('gates:', st['gates'])
    for k, v in flat.items():
        if v == FAILED:
            print('  failed:', k)
    return 0


def require_current_status():
    """Refuse a stale committed status, comparing states and the r17 hash, never its checkout-specific path."""
    committed = op.read_committed_record('status.json')
    current = build_status(write=False)
    different = []
    for section in ('gates', 'stage1'):
        old, now = committed.get(section, {}), current[section]
        different.extend(f'{section}.{k}' for k in sorted(old.keys() | now.keys()) if old.get(k) != now.get(k))
    if committed.get('r17', {}).get('sha256') != current['r17']['sha256']:
        different.append('r17.sha256')
    if different:
        raise SystemExit(f'{len(different)} status keys differ from committed status.json: {different[:5]}; '
                         'run status and commit the rebuilt status.json before assemble or labels')


# ------------------------------------------------------------------------------------------------ stage 0
def cmd_stage0(args):
    ok = True
    if not args.no_run:
        def gate_i():
            import om_clay
            res = om_clay.gate_i()
            print('gate (i):', 'pass' if res['pass'] else 'FAIL', res['max_abs'])

        def gate_ii_cpu():
            import om_turtle
            res = om_turtle.gate_ii('cpu')
            print('gate (ii) cpu:', 'pass' if res['cpu']['pass'] else 'FAIL')
        ok &= run_step('gate_i', gate_i)
        ok &= run_step('gate_ii_cpu', gate_ii_cpu)
    st = build_status()
    print('stage 0 status:', st['gates'])
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ stage 1
def cmd_stage1(args):
    import om_stage1 as s1
    steps = ('ranges', 'tpips', 'clay', 'records') if args.step == 'all' else (args.step,)
    sets = SETS if args.set is None else (args.set,)
    ok = True
    data = None
    if set(steps) & {'ranges', 'tpips', 'clay'}:
        data = s1.ValData()
    if 'ranges' in steps:
        def ranges():
            path, table = s1.step_ranges(data)
            nulls = sorted(k for k, v in table.items() if v['r'] is None)
            print(f'ranges: {len(table)} keys, {len(nulls)} with r null -> {path}')
        ok &= run_step('stage1_ranges', ranges)
    for s in sets:
        if 'tpips' in steps:
            def tp(s=s):
                rec = s1.step_tpips(data, s, args.workers)
                print(f'tpips/{s}: pick {rec["pick"]}')
            ok &= run_step(f'stage1_tpips_{s}', tp)
        if 'clay' in steps:
            def cl(s=s):
                recs = s1.step_clay(data, s, args.workers)
                print(f'clay/{s}: ret text set {recs[f"clay_textset/ret/{s}"]["set"]}, clu text set {recs[f"clay_textset/clu/{s}"]["set"]}')
            ok &= run_step(f'stage1_clay_{s}', cl)
    if 'records' in steps:
        ok &= run_step('stage1_records', stage1_records)
        build_status()
    return 0 if ok else 1


def stage1_records():
    allowed = set(pick_keys()) | {'clay_text_sha256'}
    for src, dst in (('om_ranges.json', 'stage1_om_ranges.json'), ('om_picks.json', 'stage1_om_picks.json')):
        path = op.OUT / 'stage1' / src
        if not path.exists():
            print(f'{src}: absent, {dst} not written')
            continue
        doc = json.loads(path.read_text())
        if src == 'om_picks.json':
            extra = sorted(set(doc) - allowed)
            assert not extra, f'{src}: keys no cell uses: {extra[:5]}'
        out = op.record_path(dst)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out)
        print(f'{dst}: {len(doc)} keys -> {out}')
    print('commit records/stage1_om_ranges.json, stage1_om_picks.json and status.json before stage 2')


# ------------------------------------------------------------------------------------------------ stage 2
def require_committed():
    """Stage 2 starts only on stage 1 records and a status committed unchanged (op.read_committed_record raises
    otherwise; outside smoke it compares with git HEAD)."""
    for name in RECORD_NAMES:
        op.read_committed_record(name)


def cmd_stage2(args):
    try:
        require_committed()
    except (RuntimeError, FileNotFoundError) as e:
        raise SystemExit(f'stage 2 refuses: {type(e).__name__}: {e}')
    import om_cells
    ns = argparse.Namespace(sc=args.sc, set=args.set, enc=args.enc, draws=args.draws, force=args.force)
    name = 'stage2_' + '_'.join(str(x) for x in (args.sc, args.set, args.enc) if x) if (args.sc or args.set or args.enc) else 'stage2'
    t0 = time.monotonic()
    rc = om_cells.cmd_cells(ns)             # a failing unit is logged by run_cells and the run goes on (D24)
    write_step(name, OK if rc == 0 else FAILED, None if rc == 0 else 'a unit failed: see the log', time.monotonic() - t0)
    return rc


def _expected(only):
    """Smoke only: the cells of the given units ('SC5/mmart/qwen,...'); None (all 337 cells) otherwise."""
    if not only:
        return None
    assert op.SMOKE, '--only is for the smoke (OM_SMOKE=1): the real run assembles all 337 cells'
    import om_cells
    units = {tuple(u.split('/')) for u in only.split(',')}
    cells = [c for c in om_cells.cell_specs() if (c['sc'], c['set'], c['encoder']) in units]
    assert cells, f'no cell for {only}'
    return cells


def build_subset(expected):
    """om_cells.build_cells for a subset of the cells (same inputs, same assemble)."""
    import om_cells
    ranges = op.read_committed_record('stage1_om_ranges.json')
    picks = op.read_committed_record('stage1_om_picks.json')
    status = om_cells.check_status(op.read_committed_record('status.json'))
    r17 = om_cells.load_r17_status(status)
    raw = []
    for sc, set_, enc in om_cells.units():
        path = om_cells.parts_dir() / f'{sc}__{set_}__{enc}.json'
        if path.exists():
            raw += json.loads(path.read_text())
    return om_cells.assemble(raw, ranges, picks, status, r17, expected=expected)


def cmd_assemble(args):
    require_current_status()
    import om_cells
    expected = _expected(args.only)
    if expected is None:
        return om_cells.cmd_assemble(args)
    cells = build_subset(expected)
    op.write_json(op.OUT / 'stage2' / 'cells.json', cells)
    from collections import Counter
    print(f'{len(cells)} cells (subset) -> {op.OUT / "stage2" / "cells.json"}; labels {dict(Counter(str(c["label"]) for c in cells))}; '
          f'void {sum(bool(c["void"]) for c in cells)}')
    return 0


def cmd_labels(args):
    require_current_status()
    import om_cells
    expected = _expected(args.only)
    if expected is None:
        return om_cells.cmd_labels(args)
    picks = op.read_committed_record('stage1_om_picks.json')
    out = om_cells.run_labels(op.OUT / 'stage2' / 'cells.json', op.OUT / 'c13' / 'c13_cells.json', om_cells.labels_path(), picks,
                              rebuild=lambda: build_subset(expected), expected=expected)
    print(f'labels (subset) -> {out}')
    return 0


def cmd_c13_stub(args):
    """Smoke only: a C13 file made from OUT/stage2/cells.json (so it agrees by construction). Not an independent derivation."""
    assert op.SMOKE, 'c13-stub is for the smoke (OM_SMOKE=1)'
    import om_cells
    cells = json.loads((op.OUT / 'stage2' / 'cells.json').read_text())
    picks = op.read_committed_record('stage1_om_picks.json')
    doc = dict(picks=om_cells.pick_summary(picks), stub=True, cells=[
        {k: c.get(k) for k in ('id', 'cand', 'comp', 'd', 'lo', 'hi', 'label', 'candidate_readout', 'comparator_readout') if k in c}
        for c in cells])
    op.write_json(op.OUT / 'c13' / 'c13_cells.json', doc)
    print(f'stub C13 file with {len(doc["cells"])} cells')
    return 0


def cmd_list(args):
    import om_cells
    print('gates:', ' '.join(OUR_GATES))
    print('records:', ' '.join(RECORD_NAMES), '+ labels.json')
    return om_cells.cmd_list(args)


def main(argv=None):
    from om_env import check_record_environment
    check_record_environment()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('stage0')
    p.add_argument('--no-run', action='store_true', help='read the gate files only (no gate i, ii cpu run)')
    p = sub.add_parser('stage1')
    p.add_argument('--step', choices=('all', 'ranges', 'tpips', 'clay', 'records'), default='all')
    p.add_argument('--set', choices=SETS)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p = sub.add_parser('stage2')
    p.add_argument('--sc')
    p.add_argument('--set', choices=SETS)
    p.add_argument('--enc', choices=op.ENCODERS)
    p.add_argument('--draws', type=int)
    p.add_argument('--force', action='store_true')
    for n in ('assemble', 'labels'):
        p = sub.add_parser(n)
        p.add_argument('--only', help='smoke only: units SC/set/enc,... (assemble and check just those cells)')
    for n in ('status', 'c13-stub', 'list'):
        sub.add_parser(n)
    a = ap.parse_args(argv)
    fn = {'stage0': cmd_stage0, 'stage1': cmd_stage1, 'stage2': cmd_stage2, 'status': cmd_status, 'assemble': cmd_assemble,
          'labels': cmd_labels, 'c13-stub': cmd_c13_stub, 'list': cmd_list}[a.cmd]
    return fn(a)


if __name__ == '__main__':
    sys.exit(main())
