"""r17 runner (ticket 10): one subcommand per queue step, their outputs under OUT/, and the records (spec §3.5, §3.7, §9).

Every step writes a step file `OUT/steps/<step>.json` ({step, status, error, seconds, ...}) whether it succeeds or not; a
gate that raises is recorded `failed` with the message (05 review, note 2), and the exit code is then 1. The status
record `records/status.json` (`ns_paths.record_path`; shape: `ns_outcome.check_status`) is built from the step files and
the stage 1 files by `status`, so it can be rebuilt at any time and a step never run is `missing`.

  stage 0   gate NAME            NAME in GATES (below); results OUT/stage0/<NAME>.json
  stage 1   ranges --set S --enc E     class-count ranges of c* (painting sets) or the photo model -> OUT/stage1/ranges.json
            pick --key ARM/SET/ENC     one read-out pick -> OUT/stage1/picks.json (ns_picks writes it)
            stage1-records             copy both files to records/stage1_ranges.json, records/stage1_picks.json
  stage 2   cells --set S --enc E --task clu|ret [--draws N]   -> OUT/stage2/cells/<task>__<set>__<enc>.json
  result    status | outcomes | c13-check | verdicts
  stage 3   diag ITEM            -> ns_diag.py (ticket 11)
  helpers   list                 steps, claim cells per (set, enc, task)

Smoke (NS_SMOKE=1, NS_OUTPUTS=<scratch>): smoke draws, records under OUT/records, and a gate counts as `ok` when its
values agree (it cannot pass at fewer draws), so the later stages run; a smoke status never leaves OUT/records.
Stage 2 reads the stage 1 records only through `ns_paths.read_committed_record` (inside 07 and 08).
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as P  # noqa: E402

OK, FAILED, MISSING = 'ok', 'failed', 'missing'
OUT = P.OUT
ROUND = P.ROUND

# stage 0 steps -> the status entries they feed
GATES = ('i_a_r14', 'i_a_r15', 'i_a_partitions', 'i_b', 'i_c_affection', 'i_c_emoset', 'i_d_emoset', 'i_d_class_count',
         'ii_scorer', 'ii_crl', 'ii_percept', 'iii', 'iv_export', 'iv', 'v', 'vi')
I_A = ('i_a_r14', 'i_a_r15', 'i_a_partitions')
I_D = ('i_d_emoset', 'i_d_class_count')
GATE_NOTES = {   # from the ticket 06 review
    'i_a_partitions': 'makes no bootstrap draws; its result reports draws 2000 all the same',
    'i_d_emoset': 'makes no bootstrap draws; its result reports draws 2000 all the same',
    'i_d_class_count': "makes no draws (reports 2000); does not replay r15's matched.own per-seed values (that reading is K-free)",
}
# Each queue step is one plain process: ns_clu.paired_cell opens its own fork pool, so cell scoring is never called
# inside a pool worker.
DIAG_ITEMS =('count_sweep', 'per_class', 'qwen_crl', 'told_untold', 'tables')    # ns_diag.py items (ticket 11)


# ------------------------------------------------------------------------------------------------ helpers
def steps_dir():
    return OUT / 'steps'


def step_path(name):
    return steps_dir() / f'{name}.json'


def read_step(name):
    p = step_path(name)
    return json.loads(p.read_text()) if p.exists() else None


def write_step(name, status, error=None, seconds=None, **extra):
    P.write_json(step_path(name), dict(step=name, status=status, error=error, seconds=seconds, smoke=bool(P.SMOKE), **extra))


def lenient_ok(res):
    """A replay result is ok when it passes; in smoke (fewer draws, a gate cannot pass) when its values agree."""
    if res.get('pass'):
        return True
    if P.SMOKE:
        for k in ('wiring_ok', 'values_ok'):
            if k in res:
                return bool(res[k])
        return bool(res.get('max_abs_diff', math.inf) <= P.TOL_REPLAY)
    return False


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def locked_merge(path, new, what):
    """Merge dict `new` into the JSON dict at `path` (locked, atomic); returns the merged dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cur = json.loads(path.read_text()) if path.exists() else {}
        assert isinstance(cur, dict), f'{path}: not a dict'
        dup = [k for k in new if k in cur and cur[k] != new[k]]
        if dup:
            print(f'{what}: replacing {len(dup)} existing keys, e.g. {dup[0]}')
        cur.update(new)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(cur, indent=1, sort_keys=True) + '\n')
        os.replace(tmp, path)
    return cur


def draws_for(task, arg=None):
    if arg:
        return int(arg)
    return P.SMOKE_DRAWS[task] if P.SMOKE else P.N_DRAWS[task]


# ------------------------------------------------------------------------------------------------ stage 0
def run_gate(name):
    """One stage 0 gate. Returns (status value for the step file, full result)."""
    draws = draws_for('clu')
    rdraws = draws_for('ret')
    if name.startswith('i_') and name != 'i_d_emoset' and name != 'i_d_class_count':
        import ns_clu_replay as R
        fn = {'i_a_r14': lambda: R.replay_a_r14(draws), 'i_a_r15': lambda: R.replay_a_r15(draws),
              'i_a_partitions': R.replay_a_partitions, 'i_b': lambda: R.replay_b(draws),
              'i_c_affection': lambda: R.replay_c_affection(draws), 'i_c_emoset': lambda: R.replay_c_emoset(draws)}[name]
        res = fn()
        return (OK if lenient_ok(res) else FAILED), res
    if name in I_D:
        import ns_clu_replay as R
        res = R.replay_d_emoset() if name == 'i_d_emoset' else R.replay_d_class_count()
        return (OK if lenient_ok(res) else FAILED), res
    if name.startswith('ii_'):
        import ns_ret
        res = {'ii_scorer': lambda: ns_ret.gate_ii_scorer(rdraws), 'ii_crl': ns_ret.gate_ii_crl,
               'ii_percept': ns_ret.gate_ii_percept}[name]()
        return (OK if lenient_ok(res) else FAILED), res
    if name == 'iii':
        return _gate_iii()
    if name == 'iv_export':
        import ns_mappers
        paths = ns_mappers.export()
        return OK, dict(files=len(paths))
    if name == 'iv':
        import ns_mappers
        res = ns_mappers.gate_iv()
        assert set(res['status']) == set(P.ENCODERS), res['status']
        return dict(res['status']), res
    if name == 'v':
        import ns_data
        res = ns_data.gate_v()
        return {e: res[e] for e in P.ENCODERS}, res
    if name == 'vi':
        import ns_picks
        res = ns_picks.gate_vi()
        return (OK if lenient_ok(res) else FAILED), {k: v for k, v in res.items() if k != 'items'}
    raise SystemExit(f'unknown gate {name}')


def _gate_iii():
    """(iii): the metric checks, and the class counts of ns_data and ns_photo against ns_paths.CLASS_COUNT."""
    import ns_data
    import ns_metrics
    import ns_photo
    met = ns_metrics.gate_iii_metrics()
    metric = {m: met[m] for m in P.CLASS_METRICS}
    counts, evidence = {}, {}
    for what, fn in (('painting', ns_data.class_counts), ('photo', ns_photo.class_counts)):
        sets = P.PAINTING_SETS if what == 'painting' else P.PHOTO_SETS
        keys = [f'{s}/{a}' for s in sets for a in P.SET_ASPECTS[s]]
        try:
            got = fn()
            for k in keys:
                s, a = k.split('/')
                counts[k] = OK if got.get(k) == P.CLASS_COUNT[(s, a)] else FAILED
            evidence[what] = got
        except Exception as e:      # a held/val count mismatch raises: the whole group fails, with the message
            for k in keys:
                counts[k] = FAILED
            evidence[what] = f'{type(e).__name__}: {e}'
    assert set(counts) == {f'{s}/{a}' for (s, a) in P.CLASS_COUNT}, sorted(counts)
    return dict(metric=metric, count=counts), dict(metrics=met, counts=evidence)


def cmd_gate(args):
    name = args.name
    t0 = time.monotonic()
    try:
        status, res = run_gate(name)
        err = None
    except Exception as e:
        status, res, err = FAILED, None, f'{type(e).__name__}: {e}'
        traceback.print_exc()
    secs = time.monotonic() - t0
    if res is not None:
        P.write_json(OUT / 'stage0' / f'{name}.json', res)
    note = GATE_NOTES.get(name)
    write_step(name, status, err, secs, note=note)
    print(f'gate {name}: {status} {err or ""} ({secs:.0f}s)')
    build_status()
    return 0 if err is None else 1


# ------------------------------------------------------------------------------------------------ stage 1
def range_keys_of(set_, enc):
    import ns_outcome
    return sorted({ns_outcome.range_key(c) for c in ns_outcome.claim_cells()
                   if c['set'] == set_ and c['encoder'] == enc} - {None})


def cmd_ranges(args):
    name = f'ranges__{args.set}__{args.enc}'
    t0 = time.monotonic()
    try:
        import ns_clu_cells
        res = ns_clu_cells.stage1_ranges(args.set, args.enc)
        want = range_keys_of(args.set, args.enc)
        cand = 'photo' if args.set in P.PHOTO_SETS else 'cstar'
        res = {(k if k.count('/') == 4 else f'{cand}/{args.set}/{args.enc}/{k}'): v for k, v in res.items()}
        assert set(res) == set(want), f'ranges differ from the claim cells\' keys: {sorted(set(res) ^ set(want))[:6]}'
        for k, v in res.items():
            assert set(('r', 'per_seed', 'k', 'n_val')) <= set(v) and math.isfinite(v['r']), (k, v)
        locked_merge(OUT / 'stage1' / 'ranges.json', res, 'ranges')
        status, err = OK, None
    except Exception as e:
        status, err = FAILED, f'{type(e).__name__}: {e}'
        traceback.print_exc()
    write_step(name, status, err, time.monotonic() - t0)
    print(f'{name}: {status} {err or ""}')
    build_status()
    return 0 if err is None else 1


def cmd_pick(args):
    name = 'pick__' + args.key.replace('/', '__')
    t0 = time.monotonic()
    try:
        import ns_picks
        arm, set_, enc = args.key.split('/')
        rec = ns_picks.pick(arm, set_, enc, args.workers)
        ns_picks.write_pick(args.key, rec)
        status, err = OK, None
    except Exception as e:
        status, err = FAILED, f'{type(e).__name__}: {e}'
        traceback.print_exc()
    write_step(name, status, err, time.monotonic() - t0)
    print(f'{name}: {status} {err or ""}')
    build_status()
    return 0 if err is None else 1


def cmd_stage1_records(args):
    import ns_outcome
    used_r = {ns_outcome.range_key(c) for c in ns_outcome.claim_cells()} - {None}
    used_p = {ns_outcome.pick_key(c) for c in ns_outcome.claim_cells()} - {None}
    for src, dst, used in (('ranges.json', 'stage1_ranges.json', used_r), ('picks.json', 'stage1_picks.json', used_p)):
        path = OUT / 'stage1' / src
        data = json.loads(path.read_text()) if path.exists() else {}
        extra, absent = sorted(set(data) - used), sorted(used - set(data))
        assert not extra, f'{src}: keys no claim cell uses: {extra[:5]}'
        shutil.copyfile(path, _prep(P.record_path(dst))) if path.exists() else None
        print(f'{dst}: {len(data)} of {len(used)} keys; absent {len(absent)}' + (f', e.g. {absent[:3]}' if absent else ''))
    build_status()
    print('commit records/stage1_ranges.json, stage1_picks.json and status.json before stage 2')
    return 0


def _prep(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------------------------------------ status
def _all(statuses):
    """One status from several steps: ok when all ok; failed when any failed; else missing."""
    if all(s == OK for s in statuses):
        return OK
    return FAILED if FAILED in statuses else MISSING


def _step_status(name):
    s = read_step(name)
    return MISSING if s is None else s['status']


def build_status(write=True):
    """The status record (`ns_outcome.check_status` shape) from the step files and the stage 1 files."""
    import ns_outcome
    g = {}
    g['i_a'] = _all([_step_status(n) for n in I_A])
    g['i_b'] = _step_status('i_b')
    g['i_c'] = {'affection': _step_status('i_c_affection'), 'emoset': _step_status('i_c_emoset')}
    g['i_d'] = _all([_step_status(n) for n in I_D])
    for k in ('ii_scorer', 'ii_crl', 'ii_percept', 'vi'):
        g[k] = _step_status(k)
    iii = read_step('iii')
    keys_count = [f'{s}/{a}' for (s, a) in P.CLASS_COUNT]
    if iii is None or isinstance(iii['status'], str):
        bad = MISSING if iii is None else iii['status']
        g['iii_metric'] = {m: bad for m in P.CLASS_METRICS}
        g['iii_count'] = {k: bad for k in keys_count}
    else:
        g['iii_metric'], g['iii_count'] = iii['status']['metric'], iii['status']['count']
    for k in ('iv', 'v'):
        st = read_step(k)
        if st is None or isinstance(st['status'], str):
            g[k] = {e: (MISSING if st is None else st['status']) for e in P.ENCODERS}
        else:
            g[k] = st['status']
    # stage 1: a key present in the file is ok; else failed if its step failed; else missing
    ranges_f, picks_f = OUT / 'stage1' / 'ranges.json', OUT / 'stage1' / 'picks.json'
    have_r = set(json.loads(ranges_f.read_text())) if ranges_f.exists() else set()
    have_p = set(json.loads(picks_f.read_text())) if picks_f.exists() else set()
    ranges, picks = {}, {}
    for c in ns_outcome.claim_cells():
        rk, pk = ns_outcome.range_key(c), ns_outcome.pick_key(c)
        if rk is not None and rk not in ranges:
            ranges[rk] = OK if rk in have_r else (FAILED if _step_status(f'ranges__{c["set"]}__{c["encoder"]}') == FAILED
                                                    else MISSING)
        if pk is not None and pk not in picks:
            picks[pk] = OK if pk in have_p else (FAILED if _step_status('pick__' + pk.replace('/', '__')) == FAILED
                                                   else MISSING)
    status = dict(gates=g, stage1=dict(ranges=ranges, picks=picks))
    ns_outcome.check_status(status)
    if write:
        P.write_json(_prep(P.record_path('status.json')), status)
    return status


def cmd_status(args):
    st = build_status()
    flat = {}
    for k, v in st['gates'].items():
        for kk, vv in (v.items() if isinstance(v, dict) else [('', v)]):
            flat[f'{k}{"." + kk if kk else ""}'] = vv
    for what in ('ranges', 'picks'):
        flat.update({f'{what}.{k}': v for k, v in st['stage1'][what].items()})
    counts = {s: sum(v == s for v in flat.values()) for s in (OK, FAILED, MISSING)}
    print(f'status -> {P.record_path("status.json")}: {counts}')
    for k, v in flat.items():
        if v == FAILED:
            print('  failed:', k)
    return 0


# ------------------------------------------------------------------------------------------------ stage 2
def claim_scs(set_, enc, task):
    import ns_outcome
    return sorted({c['sc'] for c in ns_outcome.claim_cells() if c['set'] == set_ and c['encoder'] == enc and c['task'] == task})


def cmd_cells(args):
    import ns_outcome
    name = f'cells__{args.task}__{args.set}__{args.enc}'
    t0 = time.monotonic()
    scs = claim_scs(args.set, args.enc, args.task)
    if not scs:
        print(f'{name}: no claim cell for this (set, encoder, task)')
        return 0
    try:
        draws = draws_for(args.task, args.draws)
        if args.task == 'clu':
            import ns_clu_cells
            cells = ns_clu_cells.score_cells(args.set, args.enc, scs, None, draws)
        else:
            import ns_ret
            cells = ns_ret.score_cells(args.set, args.enc, scs, None, draws)
        ids = {c['id'] for c in ns_outcome.claim_cells() if c['set'] == args.set and c['encoder'] == args.enc
               and c['task'] == args.task}
        got = [c['id'] for c in cells]
        assert len(got) == len(set(got)), 'duplicate cell ids'
        assert set(got) <= ids, f'cells outside the claim: {sorted(set(got) - ids)[:3]}'
        if set(got) != ids:
            print(f'{name}: {len(ids - set(got))} claim cells not produced, e.g. {sorted(ids - set(got))[:3]}')
        P.write_json(OUT / 'stage2' / 'cells' / f'{args.task}__{args.set}__{args.enc}.json', cells)
        status, err = OK, None
    except Exception as e:
        status, err = FAILED, f'{type(e).__name__}: {e}'
        traceback.print_exc()
    write_step(name, status, err, time.monotonic() - t0, n_cells=len(cells) if err is None else 0)
    print(f'{name}: {status} {err or ""} ({time.monotonic() - t0:.0f}s)')
    return 0 if err is None else 1


def merged_cells():
    cells, seen = [], {}
    files = sorted((OUT / 'stage2' / 'cells').glob('*.json'))
    for f in files:
        for c in json.loads(f.read_text()):
            assert c['id'] not in seen, f'cell {c["id"]} in {f.name} and {seen[c["id"]]}'
            seen[c['id']] = f.name
            cells.append(c)
    return cells, files


# ------------------------------------------------------------------------------------------------ results
def cmd_outcomes(args):
    import ns_outcome
    status = build_status()
    if not P.SMOKE:
        P.read_committed_record('status.json')          # refuses unless equal to git HEAD
    cells, files = merged_cells()
    P.write_json(OUT / 'stage2' / 'cells.json', cells)
    res = ns_outcome.decide(OUT / 'stage2' / 'cells.json', P.record_path('status.json'), OUT / 'stage2' / 'outcomes.json')
    print(ns_outcome.table(res))
    by = {}
    for r in res['cells']:
        by[r['status']] = by.get(r['status'], 0) + 1
    print(f'{len(res["cells"])} claim cells from {len(cells)} stage 2 cells in {len(files)} files; rows by status {by}')
    miss = [r['id'] for r in res['cells'] if r['status'] == 'missing']
    print(f'missing cells (no stage 2 cell, no failed gate): {len(miss)}' + (f', e.g. {miss[:3]}' if miss else ''))
    dst = _prep(P.record_path('outcomes.json'))
    shutil.copyfile(OUT / 'stage2' / 'outcomes.json', dst)
    print(f'outcomes -> {dst} sha256 {sha256_file(dst)}')
    write_step('outcomes', OK, None, 0.0, n_missing=len(miss))
    return 0


def c13_state():
    path = P.record_path('c13.json')
    if not path.exists():
        return False, f'{path} is absent'
    rec = json.loads(path.read_text())
    outp = P.record_path('outcomes.json')
    if not outp.exists():
        return False, f'{outp} is absent'
    if rec.get('pass') is not True:
        return False, 'c13.json: pass is not true'
    if rec.get('outcomes_sha256') != sha256_file(outp):
        return False, 'c13.json: outcomes_sha256 does not name records/outcomes.json'
    return True, 'c13.json passes and names the outcomes file'


def cmd_c13_check(args):
    ok, why = c13_state()
    print(('c13 ok: ' if ok else 'c13 NOT ok: ') + why)
    return 0 if ok else 1


def cmd_verdicts(args):
    ok, why = c13_state()
    if not ok:
        print('verdicts not written: ' + why)
        return 1
    dst = _prep(P.record_path('verdicts.json'))
    shutil.copyfile(P.record_path('outcomes.json'), dst)
    print(f'verdicts -> {dst}')
    return 0


def cmd_diag(args):
    path = ROUND / 'ns_diag.py'
    if not path.exists():
        print(f'{path} is absent (ticket 11 not merged)')
        return 1
    return subprocess.call([sys.executable, str(path), args.item] + (['--draws', str(args.draws)] if args.draws else []))


def cmd_list(args):
    import ns_outcome
    print('gates:', ' '.join(GATES))
    print('diag items:', ' '.join(DIAG_ITEMS))
    cells = ns_outcome.claim_cells()
    print(f'{len(cells)} claim cells')
    for task in ('clu', 'ret'):
        for s in P.SET_ASPECTS:
            for e in P.ENCODERS:
                scs = claim_scs(s, e, task)
                if scs:
                    n = sum(c['set'] == s and c['encoder'] == e and c['task'] == task for c in cells)
                    print(f'  cells --task {task} --set {s} --enc {e}: {",".join(scs)} ({n} cells)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gate')
    g.add_argument('name', choices=GATES)
    r = sub.add_parser('ranges')
    r.add_argument('--set', required=True, choices=P.SET_ASPECTS)
    r.add_argument('--enc', required=True, choices=P.ENCODERS)
    k = sub.add_parser('pick')
    k.add_argument('--key', required=True)
    k.add_argument('--workers', type=int, default=8)
    sub.add_parser('stage1-records')
    c = sub.add_parser('cells')
    c.add_argument('--set', required=True, choices=P.SET_ASPECTS)
    c.add_argument('--enc', required=True, choices=P.ENCODERS)
    c.add_argument('--task', required=True, choices=('clu', 'ret'))
    c.add_argument('--draws', type=int, default=None)
    for n in ('status', 'outcomes', 'c13-check', 'verdicts', 'list'):
        sub.add_parser(n)
    d = sub.add_parser('diag')
    d.add_argument('item', choices=DIAG_ITEMS)
    d.add_argument('--draws', type=int, default=None)
    args = ap.parse_args(argv)
    fn = {'gate': cmd_gate, 'ranges': cmd_ranges, 'pick': cmd_pick, 'stage1-records': cmd_stage1_records,
          'cells': cmd_cells, 'status': cmd_status, 'outcomes': cmd_outcomes, 'c13-check': cmd_c13_check,
          'verdicts': cmd_verdicts, 'diag': cmd_diag, 'list': cmd_list}[args.cmd]
    return fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
