"""r14 ticket 10a: the job driver of the node402 runs (part a, 0(viii), the graph prep, the photo and knob sweeps,
the smoke, the time estimate). Run from the repo root, on the node inside `scripts/run_bs_node.sh`:

    python src/test/20261102_backbones_sweep/bs_node.py <mode> [args] [--device cuda|cpu]

    smoke                      four tiny units (2 epochs, 5000 fit pairs): X-c* siglip2, a Qwen knob row with scoring, a
                               Qwen photo row with scoring, X photo siglip2 from the smoke X-c*; then `estimate`.
                               Refuses `bs_paths.DEFAULT_OUT`.
    part-a --encoder E         E in siglip2 | siglip1 | clip_l14: for seeds 0 to 4 the X-c* run (keep all, root
                               OUT/cstar/E), then the X photo run (keep all, OUT/photo/E/s<seed>) from that seed's
                               X-c*. Resumes: a verified run is skipped.
    viii                       0(viii): c* `ref` seed 1 retrained on this node (keep none, OUT/stage0/viii); per-head AMI
                               (and per k-means state) against the 3090 value of r10's stored validation export of the
                               same seed; OUT/stage0/viii.json. Records the difference, no gate.
    prep                       builds the missing graph stores once, before parallel workers: c* graphs k = 10 and 80;
                               every (k, correction) of the photo rows.
    sweep --kind photo knob    a worker of the shared queue over `OUT/sweeps/<kind>/<row>/s<seed>`: units in row-major
                               order (the `ref` row first), kinds in the order given. Three workers = three launches.
                               `--reset-claims` first removes the claims of units without `metrics.json` (a restart).
    estimate                   node time left from `OUT/node/units.jsonl`: rows that cannot finish by `bs_paths.NODE_END`.

Rerun rule (spec §3.4): a unit that raises (a non-finite loss included) is run once more with the same seed; a second
failure is recorded and stops that kind of run (part a: `OUT/node/part_a_<enc>.json`: a failed X photo run stops the X photo
units only, a failed X-c* run stops the part) or that row (sweeps:
`OUT/sweeps/<kind>/<row>/failed.json`; its other seeds are skipped). Every unit appends a line to `OUT/node/units.jsonl`.

Claims: `os.mkdir(OUT/sweeps/.claims/<kind>__<row>__s<seed>)` (atomic); a unit is done when its `metrics.json` exists.

Test switches (smoke and the CPU checks only; unset on the node):
    BS_NODE_SMOKE=1        every unit trains in smoke settings (c*: smoke, 2 epochs; photo: 2 epochs). Needs a scratch OUT.
    BS_FAIL_ONCE=<label>   the unit with that label fails on its first BS_FAIL_N attempts (default 1) before training.
    BS_FAIL_N=2            two attempts fail: the second-failure path.
    BS_NODE_ONLY=a,b       smoke: run only these steps (cstar, knob, photo, xphoto); sweep: only these unit labels.
    BS_NODE_TAG            a label stored in each units.jsonl line (the job name).
Unit labels: `cstar__<enc>__s<seed>`, `xphoto__<enc>__s<seed>`, `<kind>__<row>__s<seed>` (sweeps; kind knob or photo).
"""
import argparse
import gc
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')       # the CT queue's setting, before any CUDA work
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bs_paths as P  # noqa: E402

OUT = P.OUT
NODE = OUT / 'node'
UNITS = NODE / 'units.jsonl'
KINDS = ('photo', 'knob')
SMOKE_EPOCHS = 2
SMOKE_ROW_PHOTO = 'lr=0.0003'


def smoke_on():
    return bool(os.environ.get('BS_NODE_SMOKE'))


def now():
    return time.strftime('%F %T')


def jdump(obj):
    import numpy as np

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    return json.dumps(obj, indent=1, allow_nan=False, default=default) + '\n'


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(text)
    tmp.replace(path)


def log_unit(rec):
    NODE.mkdir(parents=True, exist_ok=True)
    rec = dict(rec, host=socket.gethostname(), tag=os.environ.get('BS_NODE_TAG'), at=now(), smoke=smoke_on())
    with open(UNITS, 'a') as f:
        f.write(json.dumps(rec, allow_nan=False, default=str) + '\n')
        f.flush()
        os.fsync(f.fileno())


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:       # noqa: BLE001  (cleanup only)
        pass


# ------------------------------------------------------------------------------------------------ the rerun rule
def attempt(label, fn, meta):
    """Run `fn()` (returns a dict of extra record fields; 'skipped': True when the unit was already done). On any
    exception free the GPU and run once more with the same seed (the unit's own seed is in `fn`); a second failure is
    returned as status 'failed' with the error text. Appends the unit's line to units.jsonl (a skipped unit: none).
    Returns the record."""
    fail_label, fail_n = os.environ.get('BS_FAIL_ONCE'), int(os.environ.get('BS_FAIL_N') or 1)
    errors = []
    for n in (1, 2):
        start = time.monotonic()
        try:
            if fail_label == label and n <= fail_n:
                raise RuntimeError(f'BS_FAIL_ONCE: forced failure of {label}, attempt {n}')
            extra = fn() or {}
            rec = dict(meta, label=label, status='ok', attempts=n, seconds=round(time.monotonic() - start, 1),
                       errors=errors, **extra)
            if not extra.get('skipped'):
                log_unit(rec)
            print(f'BS_NODE unit {label} ok attempts={n} seconds={rec["seconds"]}', flush=True)
            return rec
        except Exception as e:      # noqa: BLE001  (the rerun rule covers any exception)
            text = f'{type(e).__name__}: {e}'
            errors.append(dict(attempt=n, error=text, seconds=round(time.monotonic() - start, 1)))
            print(f'BS_NODE unit {label} attempt {n} failed: {text}', flush=True)
            traceback.print_exc()
            free_gpu()
    rec = dict(meta, label=label, status='failed', attempts=2, seconds=sum(e['seconds'] for e in errors), errors=errors)
    log_unit(rec)
    print(f'BS_NODE unit {label} FAILED twice', flush=True)
    return rec


# ---------------------------------------------------------------------------------------------------- the units
def cstar_args():
    return dict(smoke=True, max_epochs=SMOKE_EPOCHS) if smoke_on() else {}


def photo_epochs():
    return SMOKE_EPOCHS if smoke_on() else None


def history_len(record):
    return len(record.get('history') or [])


def unit_cstar_part_a(enc, seed, device):
    import bs_cstar
    root = P.OUT / 'cstar' / enc

    def fn():
        existed = bs_cstar.record_path(root, bs_cstar.STAR_ARM, seed).is_file()      # verified weights: train() skips
        rec = bs_cstar.train(enc, 'ref', [seed], device, root, keep='all', **cstar_args())[seed].record
        return dict(skipped=existed, best_epoch=rec['best_epoch'], train_seconds=rec['seconds'])
    return attempt(f'cstar__{enc}__s{seed}', fn, dict(kind='cstar', enc=enc, seed=seed))


class PhotoData:
    """Per-process lazy loader of an encoder's photo data and the graph cache."""

    def __init__(self):
        self.cache = {}

    def get(self, enc):
        import bs_photo
        if enc not in self.cache:
            data = bs_photo.load_data(enc)
            self.cache[enc] = (data, bs_photo.Graphs(data[2], data[3], P.OUT))
        return self.cache[enc]


def unit_xphoto_part_a(enc, seed, device, pdata):
    import bs_photo
    cstar_root = P.OUT / 'cstar' / enc

    def fn():
        existed = (bs_photo.run_dir(enc, bs_photo.REF, seed, P.OUT) / 'run.json').is_file()    # train() skips a finished run
        data, graphs = pdata.get(enc)
        out = bs_photo.train(enc, [bs_photo.REF], [seed], device, keep='all', root=P.OUT, max_epochs=photo_epochs(),
                             data=data, graphs=graphs, cstar_root=cstar_root)
        rec = out[(bs_photo.REF, seed)].record
        return dict(skipped=existed, best_epoch=rec['best_epoch'], epochs=history_len(rec), train_seconds=rec['seconds'])
    return attempt(f'xphoto__{enc}__s{seed}', fn, dict(kind='xphoto', enc=enc, seed=seed))


def part_a(enc, device):
    """Seeds 0 to 4: the X-c* run, then the X photo run from it. A run that fails twice stops only its own kind (§6, D18: a
    failed photo run leaves the paintings verdict as it is): after an X photo failure the later X-c* seeds still train and
    the later X photo units are skipped; after an X-c* failure that seed's photo unit cannot run (it starts from X-c*) and
    the part stops (failed X-c* runs mean photo_runs failed, bs_run.part_a_state)."""
    assert enc in P.NEW_ENCODERS, enc
    pdata, done, failed, photo_failed = PhotoData(), [], None, None
    for seed in P.SEEDS:
        rec = unit_cstar_part_a(enc, seed, device)
        if rec['status'] == 'failed':
            failed = dict(label=rec['label'], errors=rec['errors'])
            break
        done.append(rec['label'])
        if photo_failed:
            print(f'BS_NODE part-a {enc}: X photo s{seed} skipped (after {photo_failed["label"]} failed)', flush=True)
            continue
        rec = unit_xphoto_part_a(enc, seed, device, pdata)
        if rec['status'] == 'failed':
            photo_failed = dict(label=rec['label'], errors=rec['errors'])
            continue
        done.append(rec['label'])
    doc = dict(encoder=enc, status='failed' if failed or photo_failed else 'ok', failed=failed, photo_failed=photo_failed,
               done=done, finished_at=now(),
               host=socket.gethostname(), smoke=smoke_on())
    atomic_write(NODE / f'part_a_{enc}.json', jdump(doc))
    print(f'BS_NODE part-a {enc}: {doc["status"]}', flush=True)
    return 1 if failed or photo_failed else 0


# ------------------------------------------------------------------------------------------------- the sweeps
def sweep_rows(kind):
    import bs_cstar
    import bs_photo
    return list(bs_photo.ROWS) if kind == 'photo' else [bs_cstar.REF] + bs_cstar.knob_rows()


def sweep_seeds(kind):
    return P.SWEEP_SEEDS_PHOTO if kind == 'photo' else P.SWEEP_SEEDS_KNOB


def unit_folder(kind, row, seed):
    return P.OUT / 'sweeps' / kind / row / f's{seed}'


def queue(kinds):
    """The units in order: kinds as given, rows row-major (`ref` first), then seeds. `BS_NODE_ONLY` filters labels."""
    only = {x for x in os.environ.get('BS_NODE_ONLY', '').split(',') if x}
    units = [(k, r, s) for k in kinds for r in sweep_rows(k) for s in sweep_seeds(k)]
    if only:
        units = [u for u in units if label_of(*u) in only]
        assert units, f'BS_NODE_ONLY {sorted(only)}: no unit'
    return units


def label_of(kind, row, seed):
    return f'{kind}__{row}__s{seed}'


def done(kind, row, seed):
    return (unit_folder(kind, row, seed) / 'metrics.json').is_file()


def claims_dir():
    return P.OUT / 'sweeps' / '.claims'


def reset_claims(kinds):
    """Remove the claims of units that have no metrics.json (a restart after a crash). Not for use while workers run."""
    removed = []
    for kind, row, seed in queue(kinds):
        c = claims_dir() / label_of(kind, row, seed)
        if c.is_dir() and not done(kind, row, seed):
            shutil.rmtree(c)
            removed.append(c.name)
    print(f'BS_NODE reset-claims: removed {len(removed)}', flush=True)
    return removed


def claim(kind, row, seed):
    claims_dir().mkdir(parents=True, exist_ok=True)
    c = claims_dir() / label_of(kind, row, seed)
    try:
        os.mkdir(c)
    except FileExistsError:
        return False
    (c / 'owner.json').write_text(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), at=now(),
                                                  tag=os.environ.get('BS_NODE_TAG'))))
    return True


def row_failed(kind, row):
    return (P.OUT / 'sweeps' / kind / row / 'failed.json').is_file()


def run_record(rec, meta, kind, row, seed):
    return dict(rec, kind=kind, row=row, seed=seed, unit_status='ok', **meta)


def unit_knob(row, seed, device):
    import bs_cstar
    import bs_sweepscore as SS
    folder = unit_folder('knob', row, seed)

    def fn():
        if (folder / 'metrics.json').is_file():
            return dict(skipped=True)
        t0 = time.monotonic()
        run = bs_cstar.train('qwen', row, [seed], device, folder, keep='none', **cstar_args())[seed]
        t_train = time.monotonic() - t0
        metrics, parts, perq = SS.score_knob(run.dev, run.dev_names)
        SS.write_run(folder, metrics, parts, perq, dict(run.record, kind='knob', row=row, seed=seed,
                                                       train_seconds=t_train, score_seconds=time.monotonic() - t0 - t_train))
        return dict(train_seconds=round(t_train, 1), best_epoch=run.record['best_epoch'])
    return attempt(label_of('knob', row, seed), fn, dict(kind='knob', row=row, seed=seed))


def unit_photo(row, seed, device, pdata):
    import bs_photo
    import bs_sweepscore as SS
    folder = unit_folder('photo', row, seed)

    def fn():
        if (folder / 'metrics.json').is_file():
            return dict(skipped=True)
        (fit, val, _, _), graphs = pdata.get('qwen')
        s = bs_photo.ROWS[row]
        t0 = time.monotonic()
        run = bs_photo.train_run(f'qwen/{row}/s{seed}', s, 2048, seed, bs_photo.train_seed(seed), fit, val,
                                 graphs.get(s['k'], s['correction']), device, keep='none', out_dir=None,
                                 max_epochs=photo_epochs() if s['epochs'] is None else None,
                                 cstar_root=bs_photo.start_root('qwen'), encoder='qwen', row=row)
        t_train = time.monotonic() - t0
        x = bs_photo.infer(run.state, val['I'], 2048)
        metrics, parts, perq = SS.score_photo(x, keys=val['keys'])
        SS.write_run(folder, metrics, parts, perq, dict(run.record, kind='photo', row=row, seed=seed,
                                                       train_seconds=t_train, score_seconds=time.monotonic() - t0 - t_train))
        return dict(train_seconds=round(t_train, 1), best_epoch=run.record['best_epoch'], epochs=history_len(run.record),
                    fixed_epochs=s['epochs'])
    return attempt(label_of('photo', row, seed), fn, dict(kind='photo', row=row, seed=seed))


def sweep(kinds, device, reset=False):
    assert all(k in KINDS for k in kinds), kinds
    if reset:
        reset_claims(kinds)
    pdata, n_done, n_failed = PhotoData(), 0, 0
    for kind, row, seed in queue(kinds):
        if done(kind, row, seed) or row_failed(kind, row):
            continue
        if not claim(kind, row, seed):
            continue
        rec = unit_photo(row, seed, device, pdata) if kind == 'photo' else unit_knob(row, seed, device)
        if rec['status'] == 'failed':
            n_failed += 1
            atomic_write(P.OUT / 'sweeps' / kind / row / 'failed.json',
                         jdump(dict(kind=kind, row=row, seed=seed, errors=rec['errors'], at=now(),
                                    host=socket.gethostname())))
        else:
            n_done += 1
    print(f'BS_NODE sweep {kinds}: units done by this worker {n_done}, failed {n_failed}', flush=True)
    return 1 if n_failed else 0


# --------------------------------------------------------------------------------------------------------- viii
def viii(device):
    """0(viii): the node's c* seed 1 against the 3090's stored validation export (r10), per head and per state."""
    import numpy as np
    import bs_cstar
    import bs_sweepscore as SS
    import rt_data
    root = P.OUT / 'stage0' / 'viii'
    names, labels = SS.validation_names(), SS.rt_data.labels('val')
    out_path = P.OUT / 'stage0' / 'viii.json'
    holder = {}

    def fn():
        t0 = time.monotonic()
        run = bs_cstar.train('qwen', bs_cstar.REF, [1], device, root, keep='none', **cstar_args())[1]
        assert [str(n) for n in run.dev_names] == names
        holder['run'], holder['train_seconds'] = run, time.monotonic() - t0
        return dict(train_seconds=round(holder['train_seconds'], 1))
    rec = attempt('viii__qwen__s1', fn, dict(kind='viii', enc='qwen', seed=1))
    if rec['status'] == 'failed':
        atomic_write(out_path, jdump(dict(status='failed', errors=rec['errors'], at=now())))
        return 1
    run, items = holder['run'], {}
    for aspect in P.ASPECTS:
        node_x = run.dev[aspect]['img']
        stored = rt_data._cstar('val', aspect, 1, names)              # the 3090's export, checksum-verified by rt_data
        y = labels[aspect]
        _, s_node = SS.ami_states(node_x.astype(np.float32), y)
        _, s_3090 = SS.ami_states(stored.astype(np.float32), y)
        diff = np.abs(node_x.astype(np.float32) - stored.astype(np.float32))
        items[aspect] = dict(ami_node=float(np.mean(s_node)), ami_3090=float(np.mean(s_3090)),
                             ami_diff=float(np.mean(s_node) - np.mean(s_3090)), states_node=s_node, states_3090=s_3090,
                             export_max_abs_diff=float(diff.max()), export_mean_abs_diff=float(diff.mean()),
                             best_epoch_node=run.record['best_epoch'][aspect])
    doc = dict(status='ok', seed=1, items=items, gate=None, host=socket.gethostname(),
               seconds=holder['train_seconds'], device=str(device), at=now(), smoke=smoke_on(),
               note='recorded, no gate (0(viii)); spec §3.5')
    atomic_write(out_path, jdump(doc))
    print('BS_NODE viii: ' + ' '.join(f'{a} d_ami={i["ami_diff"]:+.5f}' for a, i in items.items()), flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------- prep
def prep():
    import bs_cstar
    import bs_photo
    for k in (10, 80):
        bs_cstar.main(['graphs', '--k', str(k)])
    data = bs_photo.load_data('qwen')
    graphs = bs_photo.Graphs(data[2], data[3], P.OUT)
    for k, c in sorted({(s['k'], s['correction']) for s in bs_photo.ROWS.values()}):
        graphs.get(k, c)
        print(f'BS_NODE prep: photo graph k={k} correction={c} ready', flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------- estimate
def read_units(path=None):
    path = Path(path or UNITS)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def estimate(now_ts=None, units_path=None, print_out=True):
    """Node time left. Seconds per unit from the ok lines of units.jsonl (smoke lines are ignored when real ones exist):
    X-c* (`cstar`), X photo (`xphoto`), photo sweep (`photo`), knob sweep (`knob`). Fallbacks: knob = X-c* mean; photo
    sweep = X photo mean per epoch x the row's epochs (fixed-epoch rows) or the mean epochs. Simulates 3 GPUs: the
    remaining part a units (one job per encoder), then the queue (photo before knob) over the free GPUs. Returns a dict."""
    import datetime as dt
    import bs_photo
    lines = [u for u in read_units(units_path) if u.get('status') == 'ok']
    real = [u for u in lines if not u.get('smoke')]
    from_smoke = bool(lines) and not real
    lines = real or lines

    def secs(kind):
        return mean(u['seconds'] for u in lines if u['kind'] == kind)

    def per_epoch(kind):
        us = [u for u in lines if u['kind'] == kind and u.get('epochs')]
        return (sum(u['seconds'] for u in us) / sum(u['epochs'] for u in us)) if us else None

    def mean_epochs(kind):
        return mean(u['epochs'] for u in lines if u['kind'] == kind and u.get('epochs'))
    s_cstar, s_xphoto = secs('cstar'), secs('xphoto')
    sec = dict(cstar=s_cstar, xphoto=s_xphoto, knob=secs('knob') or s_cstar, photo=secs('photo') or s_xphoto)
    notes = []
    if from_smoke:
        notes.append('from smoke units (2 epochs, unscaled): full runs take longer')
    if sec['cstar'] is None:
        notes.append('no X-c* unit timed yet; no estimate')
        out = dict(per_unit=sec, notes=notes, late_rows=[], finish=None)
        if print_out:
            print('BS_NODE estimate: ' + '; '.join(notes), flush=True)
        return out
    if sec['xphoto'] is None:
        sec['xphoto'] = sec['photo'] = sec['cstar'] * 0.3
        notes.append('X photo not timed yet: 0.3 x X-c* (guess)')
    if secs('photo') is None:
        notes.append('photo sweep not timed: X photo mean (fixed-epoch rows scaled by epochs)')
    if secs('knob') is None:
        notes.append('knob sweep not timed: X-c* mean')
    pe = per_epoch('photo') or per_epoch('xphoto')
    me = mean_epochs('photo') or mean_epochs('xphoto')

    def photo_secs(row):
        fixed = bs_photo.ROWS[row]['epochs']
        if fixed is not None and pe:
            return pe * fixed
        return secs('photo') or sec['photo']
    now_t = now_ts if now_ts is not None else time.time()
    end = dt.datetime.strptime(P.NODE_END, '%Y-%m-%d %H:%M').timestamp()
    # part a: remaining units per encoder (a job each), sequential; GPUs free when their job ends
    free = []
    for enc in P.NEW_ENCODERS:
        t = now_t
        for seed in P.SEEDS:
            if not (P.OUT / 'cstar' / enc / 'train' / 'k11=40' / f'run__s{seed}.json').is_file():
                t += sec['cstar']
            if not (P.OUT / 'photo' / enc / f's{seed}' / 'run.json').is_file():
                t += sec['xphoto']
        free.append(t)
    free.sort()
    rows_end = {}
    for kind in KINDS:
        for row in sweep_rows(kind):
            for seed in sweep_seeds(kind):
                if done(kind, row, seed):
                    continue
                d = photo_secs(row) if kind == 'photo' else sec['knob']
                i = free.index(min(free))
                free[i] += d
                rows_end[(kind, row)] = max(rows_end.get((kind, row), 0), free[i])
    late = [f'{k}/{r}' for (k, r), t in rows_end.items() if t > end]
    fmt = lambda t: dt.datetime.fromtimestamp(t).strftime('%F %H:%M')
    finish = fmt(max(free))
    out = dict(per_unit={k: (round(v, 1) if v else v) for k, v in sec.items()}, mean_epochs=me, notes=notes,
               late_rows=late, finish=finish, node_end=P.NODE_END, remaining_rows=len(rows_end))
    if print_out:
        print('BS_NODE estimate: seconds/unit ' + json.dumps(out['per_unit']) + f'; finish (3 GPUs) {finish} vs '
              f'{P.NODE_END}; rows after the end: {late if late else "none"}; ' + ' | '.join(notes), flush=True)
    return out


# ------------------------------------------------------------------------------------------------------- smoke
def smoke(device):
    assert P.OUT.resolve() != P.DEFAULT_OUT.resolve(), 'smoke: set BS_OUTPUTS to a scratch folder'
    os.environ['BS_NODE_SMOKE'] = '1'
    import bs_cstar
    only = {x for x in os.environ.get('BS_NODE_ONLY', '').split(',') if x}
    want = lambda step: not only or step in only
    import platform
    import sklearn
    import numpy
    import scipy
    import torch
    print(f'BS_NODE smoke: python {platform.python_version()} torch {torch.__version__} numpy {numpy.__version__} '
          f'scipy {scipy.__version__} sklearn {sklearn.__version__} cuda={torch.cuda.is_available()} '
          f'gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} host={socket.gethostname()}',
          flush=True)
    bad = 0
    pdata = PhotoData()
    # the smoke needs no graph build: the first knob row that does not move k11 (graphs at r10's k = 40)
    knob_row = next(r for r in bs_cstar.knob_rows() if not r.startswith('k11='))
    if want('cstar'):
        bad += unit_cstar_part_a('siglip2', 0, device)['status'] == 'failed'
    if want('knob'):
        rec = unit_knob(knob_row, 1, device)
        bad += rec['status'] == 'failed'
    if want('photo'):
        rec = unit_photo(SMOKE_ROW_PHOTO, 0, device, pdata)
        bad += rec['status'] == 'failed'
    if want('xphoto'):
        bad += unit_xphoto_part_a('siglip2', 0, device, pdata)['status'] == 'failed'
    estimate()
    print(f'BS_NODE smoke: {"FAILED" if bad else "ok"}', flush=True)
    return 1 if bad else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['smoke', 'part-a', 'viii', 'prep', 'sweep', 'estimate'])
    parser.add_argument('--encoder', choices=list(P.NEW_ENCODERS))
    parser.add_argument('--kind', nargs='+', choices=list(KINDS), default=None)
    parser.add_argument('--reset-claims', action='store_true')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    a = parser.parse_args(argv)
    if a.mode == 'part-a':
        assert a.encoder, 'part-a needs --encoder'
        return part_a(a.encoder, a.device)
    if a.mode == 'sweep':
        assert a.kind, 'sweep needs --kind photo and/or knob'
        return sweep(a.kind, a.device, a.reset_claims)
    if a.mode == 'smoke':
        return smoke(a.device)
    if a.mode == 'viii':
        return viii(a.device)
    if a.mode == 'prep':
        return prep()
    estimate()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
