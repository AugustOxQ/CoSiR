"""r13 stage runner (spec 2026-10-31-photo-finetune-design.md §3.4, §3.5, §3.7, D7, D10; index "Gates", "Sheet").

  python pf_run.py 0a [--parts a,b] [--device cuda] [--workers 8]
  python pf_run.py 1  [--device cuda] [--workers 8]                    (smoke only: --rows 1,8 --max-epochs N)
  python pf_run.py 2  [--device cuda] [--workers 8]                    (smoke only: --max-epochs N)
  python pf_run.py 3  [--workers 8]
  python pf_run.py diag

Wiring only: the logic is in pf_nearcopy, pf_extract, pf_export, pf_graphs, pf_train, pf_sweep, pf_stage2, pf_stage3
(and pf_diag). Files under pf_paths.OUTPUTS:
  stage0a/*.json (evidence per gate part), stage0a/gates.json     {"items": {name: {value, reference, ok}}, "ok", ...}
  runs/<run_id>/, exports/<run_id>/{val,test}/, graphs/           sweep/row{1..9}, final/s{0..4}
  stage3/verdict.json, stage3/perquery/, partitions/
  logs/<stage>.log, step_seconds.txt                              `<stage>\\t<step>\\t<seconds>` per step
Records (git-tracked in a full run; smoke: OUTPUTS/records): NEARCOPY_FILE (0a), SHEET_FILE (1), STAGE2_FILE (2).

The held guard (spec §3.5): stage 2 refuses unless SHEET_FILE is committed (byte-identical to git HEAD); stage 3
refuses unless SHEET_FILE and STAGE2_FILE are. The check is the stage's first statement, before any data or run is
loaded. In smoke the file only has to exist. The queue commits the records, not this runner.

A run that already finished (`runs/<run_id>/run.json`, ok or failed) is reused, not retrained (train_run never
overwrites a checkpoint): to retrain it, move its folder aside. The GPU lock is the queue's job.
"""
import argparse
import contextlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import pf_paths as P  # first: puts r12's round (and every older one) on sys.path
import pf_data
import pf_export
import pf_extract
import pf_graphs
import pf_nearcopy
import pf_stage2
import pf_stage3
import pf_sweep
import pf_train
import rs_vh

STAGES = ('0a', '1', '2', '3', 'diag')
SKIP_NOTE = 'only in smoke (PF_SMOKE)'
N_VAL = P.N['val']


# ---- files, log, step seconds -----------------------------------------------------------------------------------------

def outputs():
    return Path(P.OUTPUTS)


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for t in self.streams:
            t.write(s)
        return len(s)

    def flush(self):
        for t in self.streams:
            t.flush()


@contextlib.contextmanager
def logged(stage):
    """stdout and stderr also go to OUTPUTS/logs/<stage>.log (appended)."""
    path = outputs() / 'logs' / f'{stage}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old[0], f), _Tee(old[1], f)
        try:
            yield
        except BaseException:
            # The uncaught traceback reaches real stderr after these streams are restored.
            traceback.print_exc(file=f)
            raise
        finally:
            sys.stdout.flush()
            sys.stdout, sys.stderr = old


@contextlib.contextmanager
def step(stage, name):
    """Appends `<stage>\\t<step>\\t<seconds>` to OUTPUTS/step_seconds.txt (also when the step raises)."""
    t = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - t
        with open(outputs() / 'step_seconds.txt', 'a') as f:
            f.write(f'{stage}\t{name}\t{seconds:.1f}\n')
        print(f'  [{stage}] {name}: {seconds:.1f}s', flush=True)


# ---- the held guard ---------------------------------------------------------------------------------------------------

def guard_reason(f):
    """'' when the record file `f` passes, else the reason it does not: it exists and (not in smoke) its bytes equal
    `git show HEAD:<file>` (as r12's `ac_run.guard_reason`; r13's records have no outputs copy)."""
    f = Path(f)
    if not f.is_file():
        return f'{f} does not exist'
    if P.SMOKE > 0:
        print(f'guard: smoke run, git check of {f} skipped', flush=True)
        return ''
    r = subprocess.run(['git', '-C', str(f.parent), 'show', f'HEAD:./{f.name}'], capture_output=True)
    if r.returncode:
        return f'{f} is not in git HEAD: commit it first'
    if r.stdout != f.read_bytes():
        return f'{f} differs from its content in git HEAD: commit it first'
    return ''


def require_gates(stage):
    """Spec §3.5: stages 1 to 3 run only after stage 0a passed. SystemExit unless OUTPUTS/stage0a/gates.json exists
    and holds "ok": true for the current smoke setting (a failed 0a leaves no verdict)."""
    f = outputs() / 'stage0a/gates.json'
    if not f.is_file():
        raise SystemExit(f'REFUSED: stage {stage}: {f} does not exist: run stage 0a first')
    doc = read_json(f)
    if doc.get('ok') is not True:
        raise SystemExit(f'REFUSED: stage {stage}: stage 0a did not pass ({f} has no "ok": true)')
    if doc.get('smoke') != P.SMOKE:
        raise SystemExit(f'REFUSED: stage {stage}: gates.json smoke differs from PF_SMOKE={P.SMOKE}')


def require_records(stage):
    """The guard: stage 2 needs the sheet, stage 3 the sheet and the stage 2 record, committed. SystemExit (nonzero)
    before any data is loaded when one fails. The files are read from `pf_paths` at call time."""
    need = {'2': [('sheet', P.SHEET_FILE)], '3': [('sheet', P.SHEET_FILE), ('stage 2 record', P.STAGE2_FILE)]}[stage]
    for kind, f in need:
        why = guard_reason(f)
        if why:
            raise SystemExit(f'REFUSED: stage {stage}: {kind}: {why}')


# ---- stage 0a ---------------------------------------------------------------------------------------------------------

def _gate_parts(args):
    """{part: callable}, in run order. Each writes its evidence file(s) under OUTPUTS/stage0a/."""
    w, dev = args.workers, args.device
    return {
        'nearcopy': lambda: pf_nearcopy.run(),
        'parity_images': lambda: pf_extract.parity_images(dev),
        'parity_texts': lambda: pf_extract.parity_texts(dev),
        'views': lambda: pf_extract.check(),
        'inference': lambda: pf_export.gate_iii(),
        'graphs': lambda: pf_graphs.gate_iv(),
        'loss': lambda: pf_train.gate_vi(device=dev),
        'sweep': lambda: pf_sweep.gate_ii_sweep(workers=w),
        'stage2': lambda: pf_stage2.gate_ii_stage2(workers=w),
        'sc': lambda: pf_stage3.gate_ii_sc(workers=w, draws_clu=args.draws_clu, draws_ret=args.draws_ret),
    }


def _item(path, reference, summary=None, extra_ok=None):
    """A gate item from one evidence file: a missing file is not ok."""
    path = Path(path)
    if not path.is_file():
        return dict(value='missing', reference=reference, ok=False, missing=True)
    doc = read_json(path)
    ok = doc.get('ok') is True and (extra_ok is None or bool(extra_ok(doc)))
    return dict(value=summary(doc) if summary else doc.get('ok'), reference=reference, ok=bool(ok))


def _parity(doc):
    return dict(min_cosine=doc.get('min_cosine'), max_abs_diff=doc.get('max_abs_diff'), n=doc.get('n'))


def _failed_items(doc):
    return sorted(k for k, v in doc['items'].items() if not v['ok'])


def _nearcopy_item(o):
    """(v): the count and histogram are written; no gate on the number. ok = evidence and record agree."""
    path = o / 'nearcopy.json'
    if not path.is_file() or not P.NEARCOPY_FILE.is_file():
        return dict(value='missing', reference='count and histogram written', ok=False, missing=True)
    doc, rec = read_json(path), read_json(P.NEARCOPY_FILE)
    same = all(doc[k] == rec[k] for k in ('n_removed', 'n_train', 'removed_keys', 'hist'))
    return dict(value=dict(n_removed=rec['n_removed'], n_train=rec['n_train'], hist=rec['hist']['counts']),
                reference='count and histogram written; no gate on the number', ok=bool(same))


def gate_items():
    o = outputs() / 'stage0a'
    return {
        '0a(i)_parity_images': _item(o / 'parity_images.json', 'min cosine >= 0.999', _parity),
        '0a(i)_parity_texts': _item(o / 'parity_texts.json', 'min cosine >= 0.999', _parity),
        '0a(i)_views': _item(o / 'views.json', 'one row per train key / explanation; finite; unit norm within 1e-3',
                             lambda d: dict(images=d['images']['rows'], texts=d['texts']['rows'])),
        '0a(ii)_sweep_replay': _item(o / 'sweep_replay.json', 'frozen AMI s0 and c* mAP within 1e-9',
                                     lambda d: dict(ami_diff=d['ami_diff'], map_diff=d['map_diff'])),
        '0a(ii)_stage2_replay': _item(o / 'stage2_replay.json', 'r12 ranges, 25 read-out scores and pick within 1e-9',
                                      _failed_items),
        '0a(ii)_sc_replay': _item(o / 'sc_replay.json', 'r12 SC1-SC4, SC8 within 1e-9 at the full draws',
                                  lambda d: dict(failed=_failed_items(d), full_draws=d['full_draws']),
                                  extra_ok=lambda d: d['full_draws']),
        '0a(iii)_inference': _item(o / 'inference.json', 'c* exports within 1e-3, five seeds, val and test',
                                   lambda d: max(d['max_abs_diff'].values())),
        '0a(iv)_graphs': _item(o / 'graphs.json', 'r10 emotion graphs reproduced exactly',
                               lambda d: {k: v['ok'] for k, v in d['graphs'].items()}),
        '0a(v)_nearcopy': _nearcopy_item(o),
        '0a(vi)_loss': _item(o / 'loss.json', 'c* seed-0 best_dev_loss within 1e-4',
                             lambda d: dict(value=d['value'], abs_diff=d['abs_diff'])),
    }


PART_ITEMS = {'nearcopy': ['0a(v)_nearcopy'], 'parity_images': ['0a(i)_parity_images'],
              'parity_texts': ['0a(i)_parity_texts'], 'views': ['0a(i)_views'], 'inference': ['0a(iii)_inference'],
              'graphs': ['0a(iv)_graphs'], 'loss': ['0a(vi)_loss'], 'sweep': ['0a(ii)_sweep_replay'],
              'stage2': ['0a(ii)_stage2_replay'], 'sc': ['0a(ii)_sc_replay']}


def collect_gates(errors=None):
    """gates.json from every evidence file (also the ones this run did not rerun). A part that raised is not ok."""
    errors = errors or {}
    items = gate_items()
    for part, err in errors.items():
        for n in PART_ITEMS[part]:
            items[n] = dict(value=f'part raised: {err}', reference=items[n]['reference'], ok=False)
    ok = all(v['ok'] for v in items.values())
    doc = dict(items=items, ok=bool(ok), smoke=P.SMOKE, missing=sorted(k for k, v in items.items() if v.get('missing')),
               errors=errors)
    write_json(outputs() / 'stage0a/gates.json', doc)
    return doc


def stage0a(args):
    parts = _gate_parts(args)
    chosen = list(parts) if args.parts is None else args.parts.split(',')
    bad = [p for p in chosen if p not in parts]
    if bad:
        raise SystemExit(f'unknown --parts {bad}; choose from {list(parts)}')
    errors = {}
    for name in [p for p in parts if p in chosen]:
        try:
            with step('0a', name):
                parts[name]()
        except Exception as e:                      # the part's evidence stays missing/old; the item is then not ok
            errors[name] = f'{type(e).__name__}: {e}'
            print(f'  part {name} raised {errors[name]}', file=sys.stderr, flush=True)
    doc = collect_gates(errors)
    failed = sorted(k for k, v in doc['items'].items() if not v['ok'])
    return dict(ok=doc['ok'], summary=f'{len(doc["items"]) - len(failed)}/{len(doc["items"])} gate items ok'
                + (f'; FAILED: {failed}' if failed else ''))


# ---- shared training pieces (stages 1 and 2) --------------------------------------------------------------------------

def load_arrays():
    """fit_keys, then the training arrays of the fit photos and of the 4,251 validation photos (dicts with `keys`)."""
    fk, vk = pf_data.fit_keys(), pf_data.keys('val')
    assert len(vk) == N_VAL, len(vk)
    fit, val = dict(pf_data.photo_arrays(fk), keys=fk), dict(pf_data.photo_arrays(vk), keys=vk)
    print(f'  fit {len(fk)} photos, val {len(vk)} photos', flush=True)
    return fit, val


class Graphs:
    """Builds a (k, correction) graph when its folder is missing, then always loads it through `pf_graphs.load`
    (the key-hash refusal); never the npz path."""

    def __init__(self, fit, val, stage):
        self.fit, self.val, self.stage, self.cache = fit, val, stage, {}

    def get(self, k, correction):
        key = (int(k), bool(correction))
        if key not in self.cache:
            if not (pf_graphs._folder(*key) / 'emotion.npz').is_file():
                with step(self.stage, f'graph k{key[0]}__{"on" if key[1] else "off"}'):
                    pf_graphs.build(*key, self.fit, self.val)
            self.cache[key] = pf_graphs.load(*key, self.fit['keys'], self.val['keys'])
        return self.cache[key]


def train(stage, run_id, settings, cstar_seed, train_seed, graphs, fit, val, args):
    """One run through `pf_train.train_run`, or its finished record when `runs/<run_id>/run.json` exists."""
    run_json = outputs() / 'runs' / run_id / 'run.json'
    if run_json.is_file():
        rec = read_json(run_json)
        if rec.get('fit_keys_sha256') != rs_vh.names_sha(pf_data.fit_keys()):
            raise SystemExit(f'REFUSED: {run_id}: existing run.json has other fit_keys_sha256; '
                             f'move {run_json.parent} aside to retrain')
        assert rec['settings'] == dict(settings) and rec['cstar_seed'] == cstar_seed and rec['train_seed'] == train_seed, \
            f'{run_id}: existing run.json has other settings or seeds; move {run_json.parent} aside to retrain'
        print(f'  {run_id}: reusing the finished run (status {rec["status"]})', flush=True)
        return rec
    graph = graphs.get(settings['k'], settings['correction'])
    with step(stage, f'train {run_id}'):
        rec = pf_train.train_run(run_id, settings, cstar_seed, train_seed, fit, val, graph, args.device,
                                 max_epochs=args.max_epochs)
    print(f'  {run_id}: {rec["status"]}' + (f' ({rec["failure_kind"]}: {rec["error"]})' if rec['status'] != 'ok' else
          f' best_epoch={rec["best_epoch"]} best_dev_loss={rec["best_dev_loss"]:.6f}'), flush=True)
    return rec


# ---- stage 1 ----------------------------------------------------------------------------------------------------------

def parse_rows(text):
    rows = sorted({int(x) for x in text.split(',')})
    assert all(1 <= r <= 8 for r in rows), 'rows must be among 1 to 8'
    return rows


def stage1(args):
    require_gates('1')                              # before any data is loaded
    rows_run = list(range(1, 9)) if args.rows is None else parse_rows(args.rows)
    assert args.rows is None or P.SMOKE > 0, f'--rows {SKIP_NOTE}'    # rows outside enter the sheet as skipped: never in a full run
    assert 1 in rows_run, 'row 1 must run (the sheet and the pick need it)'
    fit, val = load_arrays()
    graphs = Graphs(fit, val, '1')
    entries = {}
    with step('1', 'row 0 frozen c* s0'):
        entries[0] = pf_sweep.entry(None, pf_sweep.REFERENCE, pf_sweep.score(_frozen_cstar_stored(), workers=args.workers))

    def run_row(n, settings):
        run_id = f'sweep/row{n}'
        rec = train('1', run_id, settings, P.SWEEP_SEED, P.SWEEP_SEED, graphs, fit, val, args)
        if rec['status'] != 'ok':
            return pf_sweep.entry(settings, pf_sweep.FAILED)
        with step('1', f'export+score {run_id}'):
            pf_export.export(run_id, 'val')
            scores = pf_sweep.score(pf_export.stored(run_id, 'val'), workers=args.workers)
        print(f'  {run_id}: score {scores["score"]:.4f} (ami {scores["ami"]:.5f}, map {scores["map"]:.5f})', flush=True)
        return pf_sweep.entry(settings, pf_sweep.OK, scores)

    for n in range(1, 9):
        entries[n] = run_row(n, P.GRID[n]) if n in rows_run else pf_sweep.entry(P.GRID[n], pf_sweep.SKIPPED)
        if n == 1 and entries[1]['status'] != pf_sweep.OK:
            write_json(outputs() / 'stage1/rows.json', entries)
            raise SystemExit('STOP: sweep row 1 failed (spec §3.4): no sheet, no pick. See runs/sweep/row1/run.json.')
    settings9, skipped, equals = pf_sweep.row9(entries)
    if skipped:
        print(f'  row 9 equals row {equals}: skipped', flush=True)
    else:
        entries[9] = run_row(9, settings9)
    write_json(outputs() / 'stage1/rows.json', entries)
    doc = pf_sweep.sheet(entries)
    write_json(P.SHEET_FILE, doc)
    bad = [n for n, e in sorted(entries.items()) if e['status'] == pf_sweep.FAILED]
    p = doc['pick']
    return dict(ok=True, summary=f'pick row {p["row"]} (score {p["best_score"]:.4f}, tied {p["tied_rows"]}); '
                f'failed rows {bad}; row 9 {"skipped" if skipped else "run"}; sheet -> {P.SHEET_FILE}')


def _frozen_cstar_stored():
    import ac_data
    return ac_data.stored('cstar', 'val', 0)


# ---- stage 2 ----------------------------------------------------------------------------------------------------------

def final_id(s):
    return f'final/s{s}'


def require_sweep_pick(pick):
    """Reject a stale sheet before loading arrays or training with another outputs root / fit set."""
    run_id = f'sweep/row{pick["row"]}'
    path = outputs() / 'runs' / run_id / 'run.json'
    if not path.is_file():
        raise SystemExit(f'REFUSED: stage 2: {run_id}: {path} does not exist')
    run = read_json(path)
    if (run.get('status') != 'ok' or run.get('settings') != pick['settings']
            or run.get('fit_keys_sha256') != rs_vh.names_sha(pf_data.fit_keys())):
        raise SystemExit(f'REFUSED: stage 2: {run_id}: run must be ok with the sheet pick settings '
                         'and current fit_keys_sha256')


def stage2(args):
    require_gates('2')
    require_records('2')                            # the guard comes before any data is loaded
    assert args.rows is None, '--rows is a stage 1 option'
    sheet = read_json(P.SHEET_FILE)
    pick = sheet['pick']
    settings = {c: pick['settings'][c] for c in P.COLUMNS}
    assert settings == pick['settings'], pick
    require_sweep_pick(pick)
    print(f'  pick: row {pick["row"]}, settings {settings}', flush=True)
    fit, val = load_arrays()
    graphs = Graphs(fit, val, '2')
    for s in P.SEEDS:
        # run s starts from c*'s seed s (a random start ignores it) and trains under seed 100 + s (D10)
        rec = train('2', final_id(s), settings, s, P.train_seed(s), graphs, fit, val, args)
        if rec['status'] != 'ok':
            raise SystemExit(f'STOP: {final_id(s)} failed ({rec["failure_kind"]}: {rec["error"]}); stage 2 needs five runs')
    with step('2', 'val exports'):
        for s in P.SEEDS:
            pf_export.export(final_id(s), 'val')
    with step('2', 'ranges'):
        ranges = pf_stage2.ranges(lambda s: pf_export.stored(final_id(s), 'val'),
                                  lambda s: pf_export.embed(final_id(s), 'val'), workers=args.workers)
    with step('2', 'readout'):
        selected = pf_stage2.readout({s: pf_export.embed(final_id(s), 'val') for s in P.SEEDS},
                                     r_map=ranges['map']['r'], workers=args.workers)    # the photo model's r'_mAP, never r12's
    rec = pf_stage2.record(ranges, selected)
    write_json(P.STAGE2_FILE, rec)
    return dict(ok=True, summary=f'r\'_clu {ranges["clu"]["r"]:.6g}, r\'_mAP {ranges["map"]["r"]:.6g}, '
                f'read-out {selected["pick"]}; record -> {P.STAGE2_FILE}')


# ---- stage 3 ----------------------------------------------------------------------------------------------------------

class RunLoader:
    """`pf_stage3.candidate_from_data`'s loader over the five final runs (sidecar-checked exports)."""

    @staticmethod
    def stored(split, seed):
        return pf_export.stored(final_id(seed), split)

    @staticmethod
    def embed(split, seed):
        return pf_export.embed(final_id(seed), split)


def require_final_runs(sheet, rec, pick, r):
    """D10/D12 and frozen argument checks that need no test data, before any test export."""
    assert pick == rec['readout']['pick'], 'read-out differs from the stage 2 record'
    assert r == {'clu': rec['ranges']['clu']['r'], 'map': rec['ranges']['map']['r']} and \
        rec['ranges'] != rec['r12_ranges'], 'r must be the record\'s photo-model ranges, not r12\'s'
    fit_sha = rs_vh.names_sha(pf_data.fit_keys())
    for s in P.SEEDS:
        run = read_json(outputs() / 'runs' / final_id(s) / 'run.json')
        assert run['status'] == 'ok' and run['settings'] == sheet['pick']['settings'], (final_id(s), 'not the sheet pick')
        assert type(run['cstar_seed']) is int and run['cstar_seed'] == s, (final_id(s), 'cstar_seed')
        assert type(run['train_seed']) is int and run['train_seed'] == P.train_seed(s), (final_id(s), 'train_seed')
        expected_start = (read_json(Path(P.CSTAR_WEIGHTS) / f'emotion__s{s}.json')['weights_sha256']
                          if run['settings']['start'] == 'cstar' else None)
        assert run['cstar_weights_sha256'] == expected_start, (final_id(s), 'cstar_weights_sha256')
        assert run['fit_keys_sha256'] == fit_sha, (final_id(s), 'fit_keys_sha256')


def stage3(args):
    require_gates('3')
    require_records('3')                            # the guard comes before any test export
    sheet, rec = read_json(P.SHEET_FILE), read_json(P.STAGE2_FILE)
    pick = rec['readout']['pick']
    r = {'clu': rec['ranges']['clu']['r'], 'map': rec['ranges']['map']['r']}      # the photo model's r' from the record
    require_final_runs(sheet, rec, pick, r)
    with step('3', 'test exports'):                 # the first test number: only now, after the guard
        for s in P.SEEDS:
            pf_export.export(final_id(s), 'test')
    cand = pf_stage3.candidate_from_data('photo', pick, RunLoader)
    assert cand.readout == pick == rec['readout']['pick'], 'candidate read-out differs from the stage 2 record'
    assert cand.val_embed.shape[:2] == (5, N_VAL), ('validation embeddings', cand.val_embed.shape)
    with step('3', 'SC1-SC5'):
        res = pf_stage3.stage3(cand, r, workers=args.workers, draws_clu=args.draws_clu, draws_ret=args.draws_ret)
    assert res['doc']['r'] == r, 'scored with other ranges than the record\'s'
    pf_stage3.write_stage3(res)
    doc = res['doc']
    return dict(ok=True, summary=' '.join(f'{k}={v["outcome"]}' for k, v in doc['sc'].items())
                + f' verdict={doc["verdict"]} -> {outputs() / "stage3/verdict.json"}')


# ---- diag -------------------------------------------------------------------------------------------------------------

def stagediag(args):
    import pf_diag                                   # ticket 11
    out = pf_diag.main()
    return dict(ok=True, summary=f'diagnostics written ({out if isinstance(out, (str, Path)) else "see pf_diag"})')


# ---- command line -----------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=STAGES)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    ap.add_argument('--parts', help='stage 0a: run only these gate parts (comma list), then recollect gates.json')
    ap.add_argument('--rows', help=f'stage 1, {SKIP_NOTE}: run only these rows of 1 to 8 (others enter the sheet as skipped)')
    ap.add_argument('--max-epochs', type=int, help=f'stage 1, 2, {SKIP_NOTE}')
    ap.add_argument('--draws-clu', type=int, default=2000, help=f'stage 0a (sc), 3; other than 2000 {SKIP_NOTE}')
    ap.add_argument('--draws-ret', type=int, default=10000, help=f'stage 0a (sc), 3; other than 10000 {SKIP_NOTE}')
    args = ap.parse_args(argv)
    if P.SMOKE == 0:
        for flag, bad in (('--rows', args.rows is not None), ('--max-epochs', args.max_epochs is not None),
                          ('--draws-clu', args.draws_clu != 2000), ('--draws-ret', args.draws_ret != 10000)):
            if bad:
                ap.error(f'{flag} is allowed {SKIP_NOTE}, not in a full run')
    if args.parts is not None and args.stage != '0a':
        ap.error('--parts is a stage 0a option')
    if args.rows is not None and args.stage != '1':
        ap.error('--rows is a stage 1 option')
    t = time.perf_counter()
    summary, code = '', 1
    with logged(args.stage):
        print(f'stage {args.stage} (smoke {P.SMOKE}) ...', flush=True)
        try:
            out = globals()['stagediag' if args.stage == 'diag' else f'stage{args.stage}'](args)
            summary, code = out['summary'], 0 if out['ok'] else 1
        except SystemExit as e:                      # a refusal or a stop: the message is the summary
            summary, code = str(e.code), e.code if isinstance(e.code, int) and e.code else 1
        except BaseException as e:
            summary = f'crashed: {type(e).__name__}: {e}'
            print(summary, file=sys.stderr, flush=True)
            raise
        finally:
            print(f'stage {args.stage} {"ok" if code == 0 else "FAILED"} in {time.perf_counter() - t:.1f}s: {summary}',
                  flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
