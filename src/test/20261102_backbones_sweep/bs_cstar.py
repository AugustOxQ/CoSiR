"""r14 ticket 03: c*'s trainer for any image encoder, c*'s knob rows, and the ArtELingo exports of the checkpoints.

c* is r10's arm `k11=40` (G with 40 graph neighbours; `rs_sheet.config({'k11': 40})`). The trainer here is r10's
full-scale trainer (`rs_full.train`: `rs_knobs` patches -> `rg_full.train_cohort` -> `rb_student` -> `ct2_d2.run`)
with three differences, none of them in the computation:

  * the encoder is a parameter. `rg_full.ENCODER` (fixed to Qwen in r10) and `rb_encoders.WIDTH` are patched for the
    length of the call, so the image trunk's first layer takes the encoder's width (12-24's `rb_student.build_model`)
    and the image input is the encoder's ArtELingo default image. Qwen's instructed graphs, default captions and
    teachers are untouched, and so is the output width (2048). For Qwen nothing changes at all: same model, data,
    graphs, losses, stopping and seeds as r10's `k11=40` (stage 0 (iv) checks it, `regress-iv`);
  * none of r10's guards runs (frozen-code commit, stage 3 verdict, V4a/V4d, rule commit): they belong to r10's
    decision. What stays is the data hygiene: no output inside another round's folder, smoke never in the real output
    folder, a non-finite loss raises;
  * the graph store of the neighbour rows (k = 10, 80, ...) is built by r10's graph code into this round's output
    folder (never into r10's), after r10's `gates/k11_full.json` (the k = 20 rebuild equals the CT's graphs bit for
    bit, by this graph code) is read and found to pass for the code on disk. r10's own stored k = 40 store is read.

Files (r10's layout, so r10's and r11's readers read them with only the root changed), `--keep all`:
  `<root>/weights/<arm>/<aspect>__s<seed>.pt|json`, `<root>/train/<arm>/R6-5/<aspect>__<row>__s<seed>.npz|json` (the
  validation export, image and caption side, 6,152 paintings), `<root>/train/<arm>/run__s<seed>.json` (run record),
  `<root>/held/<arm>/<aspect>__s<seed>.npz|json` (held export, 12,281 paintings, `export_held`).
`--keep none` (the knob sweep, D15): nothing but the run record is written; the training happens in a scratch folder
that is removed, and the validation outputs are returned in memory (`Run.dev`) for the caller's scoring.

Arms: X-c* is arm `k11=40` (r10's name); a knob row is c* with one knob moved to r10's stage 1 value (graph
neighbours: 10, 20, 80), named `<knob>=<value>` (`k6=0.0005`, `k11=80`); the reference `ref` is c* (arm `k11=40`).
Only Qwen has knob rows (the knob patches build the 2048-wide trunk).

CLI: bs_cstar train --encoder E [--row ROW] [--seeds ..] [--keep all|none] [--root DIR] [--smoke]
     bs_cstar export --encoder E [--row ROW] --kind held val [--seeds ..] [--root DIR] [--out-root DIR]
     bs_cstar graphs --k K          bs_cstar regress-iv          bs_cstar check-export
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import namedtuple
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import bs_paths as P

if str(P.R10_ROUND) not in sys.path:
    sys.path.insert(0, str(P.R10_ROUND))
import rs_paths  # noqa: E402  (r10's folder first; through it the older rounds' folders on sys.path)
import rs_sheet as S  # noqa: E402
import rs_knobs  # noqa: E402
import rs_graphs  # noqa: E402
import rs_full  # noqa: E402  (arm_alias, jsonable, code_sha256, frozen_modules: used, its guards never called)
import ct2_data  # noqa: E402
import eg_train3 as eg  # noqa: E402
import rb_encoders  # noqa: E402
import rb_rows  # noqa: E402
import rb_student  # noqa: E402
import rg_full  # noqa: E402
import rg_variants as V  # noqa: E402
from ab_paths import ms_train  # noqa: E402

ASPECTS = tuple(rs_paths.ASPECTS)
SEEDS = P.SEEDS
STAR = S.config({'k11': 40})                      # c*
assert S.name(STAR) == 'k11=40'
STAR_ARM = S.name(STAR)
REF = 'ref'
GRAPH_ROWS = rs_graphs.ROWS['aspect']             # the graph row of each aspect, in the validation export's file name
KEEP = ('all', 'none')
TOL_TRAIN = 1e-4                                  # 0(iv)
TOL_EXPORT = 1e-3                                 # 0(ix) (r12's bar)
BATCH = rs_knobs.G_BATCH                          # inference batch of every export (G's, as r10)
Run = namedtuple('Run', 'record dev dev_names')   # dev: {aspect: {'img': f16 [n, 2048], 'cap': ...}} (keep none), else None


class CstarNonFiniteLoss(RuntimeError):
    """A train or dev loss that is not finite (the rerun rule is ticket 10's)."""


# ----------------------------------------------------------------------------------------------- encoders and rows
def backbone(enc):
    """The ArtELingo default-image store folder of an encoder key of `bs_paths.ENCODERS`."""
    assert enc in P.ENCODERS, f'encoder {enc!r}: not one of {sorted(P.ENCODERS)}'
    return P.ENCODERS[enc]['art']


def width(enc):
    backbone(enc)
    return P.ENCODERS[enc]['width']


def row_arm(row):
    """The run's arm name: r10's `k11=40` for c* (file names as r10), the row's own name for a knob row."""
    return STAR_ARM if row == REF else row


def row_config(row):
    """The `rs_sheet` configuration of a row: `ref` is c*; `<knob>=<value>` is c* with that one knob moved."""
    if row == REF:
        return dict(STAR)
    knob, _, text = row.partition('=')
    assert knob in S.KNOBS and text, f'row {row!r}: not `ref` or `<knob>=<value>`'
    assert row in knob_rows(), f'row {row!r}: not one of the {len(knob_rows())} knob rows'
    cfg = S.config({'k11': STAR['k11'], knob: float(text)})
    assert [k for k in S.KNOBS if cfg[k] != STAR[k]] == [knob], row
    return cfg


def knob_values(knob):
    """r10's three stage 1 values of a knob; graph neighbours at 10, 20 and 80 around c*'s 40 (D12)."""
    return (10, 20, 80) if knob == 'k11' else S.VALUES[knob]


def row_name(knob, value):
    return f'{knob}={S._fmt(knob, value)}'


def knob_rows():
    """The 39 rows of the knob sweep, r10's sheet order (k1 first, values in listed order)."""
    return [row_name(k, v) for k in S.KNOBS for v in knob_values(k)]


# ---------------------------------------------------------------------------------------------------- hygiene
def check_root(root, smoke=False):
    """A folder this module writes is never inside another round's outputs or the feature stores, and a smoke is
    never in the real output folder."""
    root = Path(root).resolve()
    frozen = [P.R10_OUT, P.R11_OUT, P.R12_OUT, P.R13_OUT, P.PRE, *rs_paths.FROZEN]
    for folder in frozen:
        assert not root.is_relative_to(Path(folder).resolve()), f'{root} is inside {folder}'
    assert not (smoke and root.is_relative_to(P.DEFAULT_OUT.resolve())), f'smoke outputs in the real folder: {root}'
    return root


def code_state():
    """(commit, dirty) of this checkout: HEAD, and whether any tracked or new .py file differs from it."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=P.REPO, capture_output=True, text=True, check=True).stdout.strip()
    return git('rev-parse', 'HEAD'), bool(git('status', '--porcelain', '--', ':/*.py'))


# -------------------------------------------------------------------------------------------------- graph stores
def require_graph_gate():
    """r10's `gates/k11_full.json` passes, for the graph code on disk now (V4c: k = 20 rebuilt equals the CT's graphs)."""
    path = P.R10_OUT / 'gates/k11_full.json'
    gate = json.loads(path.read_text())
    assert gate['pass'] is True and gate['scale'] == 'full' and gate['k'] == 20, f'{path} does not pass'
    assert gate['code_sha256'] == rs_graphs.code_sha256(), f'{path}: written by other graph code than is on disk now'


def graph_source(k):
    """The output folder that holds, or will hold, the full-scale aspect graphs at k neighbours: r10's if its store is
    there (read only), else this round's."""
    store = rs_graphs.store_root(P.R10_OUT, 'full', k, 'aspect')
    return P.R10_OUT if (store / 'd2/graphs/manifest.json').exists() else P.OUT


def graph_data(scale, k, outputs, variant='aspect'):
    """The seam `rs_knobs.store_training_data` is replaced by: the CT's (s, inputs, graphs, manifests) with the aspect
    graphs at k neighbours, built by r10's graph code (`rs_graphs._materialize`, the build/read step its
    `full_training_data` calls after its gates), here after `require_graph_gate`. `outputs` is where a store is built
    when r10 has none."""
    assert scale == 'full' and variant == 'aspect' and k != S.G['k11'], (scale, k, variant)
    require_graph_gate()
    source = graph_source(k)
    if source == P.OUT:
        source = check_root(outputs if outputs is not None else P.OUT)
    return rs_graphs._materialize('full', k, 'aspect', source)[1]


# ------------------------------------------------------------------------------------------ the patched context
Context = namedtuple('Context', 'cfg arm knobs backbone width')


@contextmanager
def training_context(enc, row=REF, *, export=False, graph_out=None):
    """Everything r10's `rs_full.train` enters, in its order, with the encoder as a parameter: the encoder's width and
    folder, the knob patches of the row (the graph seam is `graph_data`; `export`: no graph seam, the exports read no
    graph), and the run's arm name as a copy of G's arm."""
    cfg, arm = row_config(row), row_arm(row)
    name, w = backbone(enc), width(enc)
    assert enc == 'qwen' or row == REF, f'knob rows are Qwen only (the knob patches build the 2048-wide trunk): {enc} {row}'
    assert rb_encoders.WIDTH.get(name, w) == w, (name, rb_encoders.WIDTH[name], w)
    with ExitStack() as stack:
        stack.enter_context(patch.object(rg_full, 'ENCODER', name))
        stack.enter_context(patch.dict(rb_encoders.WIDTH, {name: w}))
        stack.enter_context(patch.object(rs_knobs, 'store_training_data', graph_data))
        knobs = rs_knobs.patches(dict(cfg, k11=S.G['k11']) if export else cfg, stack, scale='full',
                                 outputs=graph_out if graph_out is not None else P.OUT)
        stack.enter_context(rs_full.arm_alias(arm))
        yield Context(cfg, arm, knobs, name, w)


def load_data(ctx):
    """The CT's full-scale training data (the row's graphs under the patches) and the encoder's default image of every
    painting, as `rs_full.load_data` reads them for Qwen."""
    s, inputs, graphs, manifests = rb_student.load_scale(rg_full.SCALE)
    return s, inputs, graphs, manifests, rb_student.training_images(ctx.backbone, s, inputs)


def paths_of(root, arm, aspect, seed):
    """r10's file names under a root: (weights, validation export, held export)."""
    return (rb_student.weights_path(root, arm, aspect, seed),
            ms_train.output_path(Path(root) / 'train' / arm, rg_full.RECIPE, aspect, GRAPH_ROWS[aspect], seed),
            Path(root) / 'held' / arm / f'{aspect}__s{seed}.npz')


def record_path(root, arm, seed):
    return Path(root) / 'train' / arm / f'run__s{seed}.json'


# ------------------------------------------------------------------------------------------------------ training
def _dev_outputs(folder, aspect, seed):
    path = ms_train.output_path(folder, rg_full.RECIPE, aspect, GRAPH_ROWS[aspect], seed)
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['cap', 'img'], (path, z.files)
        return {side: z[side] for side in z.files}


def _record(enc, row, ctx, seed, device, smoke, max_epochs, root, seconds):
    commit, dirty = code_state()
    record = dict(encoder=enc, backbone=ctx.backbone, width=ctx.width, row=row, name=ctx.arm, seed=seed,
                  config=rs_full.jsonable(ctx.cfg), smoke=smoke, max_epochs=max_epochs, device=str(device),
                  status='ok', cause=None, knobs=ctx.knobs, graphs=rs_full.graph_choice(ctx.cfg),
                  code_commit=commit, code_dirty=dirty, code_sha256=rs_full.code_sha256(),
                  bs_cstar_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), host=socket.gethostname(),
                  seconds=seconds)
    infos = {a: json.loads(rb_student.weights_path(root, ctx.arm, a, seed).with_suffix('.json').read_text())
             for a in ASPECTS}
    record.update(weights_sha256={a: i['weights_sha256'] for a, i in infos.items()},
                  best_epoch={a: i['best_epoch'] for a, i in infos.items()},
                  best_dev_loss={a: i['best_dev_loss'] for a, i in infos.items()})
    return record


@threadpool_limits.wrap(limits=8)
def train(enc, row, seeds, device, root, *, keep='all', smoke=False, max_epochs=None, graph_out=None, scratch=None):
    """Train c* (row `ref`) or a knob row on encoder `enc` at full scale, one cohort (three heads) per seed.

    keep 'all': r10's files under `root` (see the module doc); a seed with verified weights is skipped (its record
    must be this run's). keep 'none': training in a scratch folder (under `scratch`, default `root`) that is removed;
    only the run record is written, at r10's record path. Returns {seed: Run}. A non-finite loss raises
    `CstarNonFiniteLoss` (rg_full writes its `failed__s<seed>.json`; the rerun rule is the caller's).
    `graph_out`: where a missing graph store is built (default `bs_paths.OUT`)."""
    seeds = list(seeds)
    assert keep in KEEP and seeds and len(set(seeds)) == len(seeds) and all(s in SEEDS for s in seeds), (keep, seeds)
    assert not smoke or (max_epochs or 2) <= 2, '--smoke: at most 2 epochs'
    assert max_epochs is None or smoke, 'a full run trains to the CT\'s 200 epochs and its patience'
    root, device = check_root(root, smoke), torch.device(device)
    if device.type == 'cuda':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'set CUBLAS_WORKSPACE_CONFIG=:4096:8 (CT queue)'
    torch.set_num_threads(8)
    result = {}
    with training_context(enc, row, graph_out=graph_out) as ctx:
        arm = ctx.arm
        s, inputs, graphs, manifests, images = load_data(ctx)
        rb = rb_student._rb_block(rg_full.SCALE, ctx.backbone, rg_full.RECIPE, rb_rows.recipe(rg_full.RECIPE), images)
        dev_names = [str(n) for n in s['all_painting_ids'][s['dev']]]
        for seed in seeds:
            rg = rg_full.rg_block(arm, seed, len(s['fit']))
            if keep == 'all':
                state = rg_full._cohort_state(root, arm, seed, rb, rg)
                if state == 'done':
                    path = record_path(root, arm, seed)
                    assert path.is_file(), f'{path}: weights without a run record; move them aside'
                    record = json.loads(path.read_text())
                    assert (record['encoder'], record['row'], record['seed'], record['config'], record['smoke']) == (
                        enc, row, seed, rs_full.jsonable(ctx.cfg), smoke), f'{path}: another run'
                    print(f'{arm} s{seed}: verified weights; skip', flush=True)
                    result[seed] = Run(record, None, None)
                    continue
                held = sorted((root / 'held' / arm).rglob('*.npz')) if (root / 'held' / arm).exists() else []
                assert not held, f'no new training of {arm} once held outputs exist: {held[0]}'
                if state == 'broken':
                    rb_student._move_aside(root, arm, seed, rg_full.RECIPE)
                folder, cleanup = root, None
            else:
                root.mkdir(parents=True, exist_ok=True)
                cleanup = tempfile.TemporaryDirectory(dir=scratch or root, prefix='scratch_')
                folder = Path(cleanup.name)
            try:
                start = time.monotonic()
                outcome = rg_full.train_cohort(s, inputs, graphs, manifests, images, arm, seed, device, folder, rb=rb,
                                               smoke=smoke, max_epochs=max_epochs)
                seconds = time.monotonic() - start
                if outcome.get('status') == 'failed':
                    raise CstarNonFiniteLoss(f'{arm} s{seed}: {outcome["cause"]}')
                record = _record(enc, row, ctx, seed, device, smoke, max_epochs, folder, seconds)
                dev = {a: _dev_outputs(folder / 'train' / arm, a, seed) for a in ASPECTS} if keep == 'none' else None
            finally:
                if cleanup is not None:
                    cleanup.cleanup()
            eg.atomic_record(record_path(root, arm, seed), record)
            result[seed] = Run(record, dev, dev_names if dev is not None else None)
    return result


# ------------------------------------------------------------------------------------------------------- exports
def _names_sha(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def _infer_sides(weights, aspect, w, inputs, device):
    """One saved best view on `inputs` {side: float32 [n, in]} as `ct2_d2.save_output` runs it (batches of 512, float16),
    for any image width `w`: the model is `rb_student.build_model`'s, whose image trunk's first layer takes `w`."""
    device = torch.device(device)
    _, views = rb_student.build_model(rb_rows.recipe(rb_rows.REFERENCE)['base'], list(ASPECTS), device, width=w)
    view = views[aspect]
    view.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    view.eval()
    out = {}
    with torch.no_grad():
        for side, x in inputs.items():
            x = np.asarray(x)
            assert x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all(), side
            y = np.empty((len(x), rb_student.OUT), dtype=np.float16)
            for a in range(0, len(x), BATCH):
                xx = torch.from_numpy(np.ascontiguousarray(x[a:a + BATCH])).to(device)
                y[a:a + len(xx)] = view[side](xx).cpu().numpy().astype(np.float16)
            out[side] = y
    return out


def _check_weights(root, arm, aspect, seed, backbone_name, smoke):
    """`rg_full._arm_weights` (checksum, aspect, seed, smoke, 200 epochs, encoder and width, arm and switches) under the
    patched encoder."""
    path, config, weights_sha = rg_full._arm_weights(root, arm, aspect, seed, smoke)
    assert config['rb']['encoder'] == backbone_name
    return path, config, weights_sha


def _validation_inputs(backbone_name):
    """(names, image, caption) of the 6,152 validation paintings in the trainer's dev order: the encoder's default image
    and the default caption mean (the CT's loaders, as `ct2_d2.load_training_data` reads them)."""
    s = ct2_data.split2('full')
    names = [str(n) for n in s['all_painting_ids'][s['dev']]]
    return names, rb_encoders.full_images(backbone_name, names), ct2_data.default_captions(s)[s['dev']]


def export_held(enc, row, seeds, root, *, device=None, out_root=None, smoke=False):
    """The ArtELingo held export (12,281 paintings in the CT's held order, float16 2048, image side) of stored
    checkpoints of any width, in r10's format (`rg_full.infer_held`'s file and sidecar; `device`: the one the weights
    were trained on unless given, e.g. 'cpu'). Files at `out_root` (default `root`) `held/<arm>/<aspect>__s<seed>.npz`."""
    return _export('held', enc, row, seeds, root, device, out_root, smoke)


def export_validation(enc, row, seeds, root, *, device=None, out_root=None, smoke=False):
    """The ArtELingo validation export (6,152 paintings in the trainer's dev order, image and caption side, as
    `ct2_d2.save_output`) from stored checkpoints, files `out_root/val/<arm>/<aspect>__s<seed>.npz`."""
    return _export('val', enc, row, seeds, root, device, out_root, smoke)


def _export(kind, enc, row, seeds, root, device, out_root, smoke):
    seeds = list(seeds)
    assert kind in ('held', 'val') and seeds and len(set(seeds)) == len(seeds) and all(s in SEEDS for s in seeds)
    root = Path(root).resolve()
    out_root = check_root(out_root if out_root is not None else root, smoke)
    result = {}
    with training_context(enc, row, export=True) as ctx:
        arm, w = ctx.arm, ctx.width
        if kind == 'held':
            names, names_sha = rb_student.held_names()
            x = rb_encoders.full_images(ctx.backbone, names)
            assert x.shape == (len(names), w) and x.dtype == np.float32 and np.isfinite(x).all()
            inputs = dict(img=x)
        else:
            names, img, cap = _validation_inputs(ctx.backbone)
            names_sha = _names_sha(names)
            assert img.shape == (len(names), w) and cap.shape == (len(names), 2048)
            inputs = dict(img=img, cap=cap)
        for seed in seeds:
            for aspect in ASPECTS:
                weights, config, weights_sha = _check_weights(root, arm, aspect, seed, ctx.backbone, smoke)
                recorded = torch.device(config['device'])
                used = recorded if device is None else torch.device(device)
                path = (out_root / 'held' / arm if kind == 'held' else out_root / 'val' / arm) / f'{aspect}__s{seed}.npz'
                meta = dict(kind=kind, arm=arm, scale=rg_full.SCALE, aspect=aspect, seed=seed, encoder=ctx.backbone,
                            rows=len(names), width=rb_student.OUT, dtype='float16', list_sha256=names_sha,
                            weights_sha256=weights_sha, image_sha256=rb_student.sha_array(inputs['img']),
                            device=str(recorded), export_device=str(used), recipe=rg_full.RECIPE,
                            switches=V.switches(arm))
                if path.exists():
                    arrays, info = rb_student.read_outputs(path)
                    assert all(info[k] == v for k, v in meta.items()), f'{path}: existing output differs from this run'
                else:
                    out = _infer_sides(weights, aspect, w, inputs, used)
                    for side, y in out.items():
                        problem = rg_full.vc3_problem(y, n_rows=len(names))
                        assert problem is None, f'{path}: {side}: {problem}'
                    rb_student.write_outputs(path, out, dict(meta, sha256=rb_student.sha_array(out['img'])))
                    arrays, info = rb_student.read_outputs(path)
                result[aspect, seed] = arrays
    return result


# --------------------------------------------------------------------------------------------- stage 0 checks
R10_FULL = P.R10_OUT / 'full'


def compare_training(root, reference=R10_FULL, arm=STAR_ARM, seed=0, tol=TOL_TRAIN):
    """0(iv): per head, best epoch equal and best development loss within `tol` of the reference run's sidecar."""
    items = {}
    for aspect in ASPECTS:
        ours, theirs = (json.loads(rb_student.weights_path(r, arm, aspect, seed).with_suffix('.json').read_text())
                        for r in (root, reference))
        diff = abs(ours['best_dev_loss'] - theirs['best_dev_loss'])
        items[aspect] = {'pass': bool(ours['best_epoch'] == theirs['best_epoch'] and diff <= tol),
                         'best_epoch': [ours['best_epoch'], theirs['best_epoch']],
                         'best_dev_loss': [ours['best_dev_loss'], theirs['best_dev_loss']], 'abs_diff': diff}
    return {'pass': all(i['pass'] for i in items.values()), 'tol': tol, 'seed': seed, 'items': items}


def compare_exports(out_root, reference=R10_FULL, arm=STAR_ARM, seeds=SEEDS, tol=TOL_EXPORT):
    """0(ix): the held and validation exports under `out_root` against the reference run's stored ones: same shapes,
    maximum absolute difference of the float16 values at most `tol`."""
    items = {}
    for seed in seeds:
        for aspect in ASPECTS:
            _, theirs_val, theirs_held = paths_of(reference, arm, aspect, seed)
            ours_held = Path(out_root) / 'held' / arm / f'{aspect}__s{seed}.npz'
            ours_val = Path(out_root) / 'val' / arm / f'{aspect}__s{seed}.npz'
            for kind, a, b, sides in (('held', ours_held, theirs_held, ('img',)), ('val', ours_val, theirs_val, ('img', 'cap'))):
                with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
                    for side in sides:
                        u, v = x[side], y[side]
                        ok = u.shape == v.shape and u.dtype == v.dtype
                        diff = float(np.abs(u.astype(np.float64) - v.astype(np.float64)).max()) if ok else None
                        items[f'{kind}/{aspect}__s{seed}/{side}'] = {
                            'pass': bool(ok and diff <= tol), 'max_abs_diff': diff, 'shape': list(u.shape)}
    return {'pass': all(i['pass'] for i in items.values()), 'tol': tol, 'items': items}


# -------------------------------------------------------------------------------------------------- command line
def default_root(enc):
    return P.OUT / 'cstar' / enc


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')    # the CT queue's setting, before any CUDA work
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def common(p, rows=True):
        p.add_argument('--encoder', required=True, choices=sorted(P.ENCODERS))
        if rows:
            p.add_argument('--row', default=REF, help='ref (c*) or a knob row <knob>=<value> (Qwen only)')
        p.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
        p.add_argument('--root', type=Path, default=None, help='default: OUT/cstar/<encoder> (c* rows only)')
        p.add_argument('--smoke', action='store_true', help='2 epochs, 5000 fit pairs; never in the real output folder')

    p = sub.add_parser('train')
    common(p)
    p.add_argument('--keep', choices=KEEP, default='all', help='none: the knob sweep keeps no weights or exports (D15)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-epochs', type=int, default=None)
    p = sub.add_parser('export')
    common(p)
    p.add_argument('--kind', nargs='+', choices=['held', 'val'], default=['held', 'val'])
    p.add_argument('--device', default=None, help='default: the device the weights were trained on')
    p.add_argument('--out-root', type=Path, default=None)
    p = sub.add_parser('graphs', help='build (or read) the full-scale aspect graphs at k neighbours')
    p.add_argument('--k', type=int, required=True)
    p = sub.add_parser('regress-iv', help='stage 0 (iv): Qwen c* seed 0 against r10\'s sidecars')
    p.add_argument('--root', type=Path, default=None)
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('check-export', help='stage 0 (ix): c*\'s five seeds, exported from r10\'s weights, against r10\'s exports')
    p.add_argument('--out-root', type=Path, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args(argv)

    if args.command == 'graphs':
        rs_paths.check_imports()
        graph_data('full', args.k, P.OUT)
        print(f'graphs k={args.k}: {rs_graphs.store_root(graph_source(args.k), "full", args.k, "aspect")}', flush=True)
        return 0
    if args.command == 'regress-iv':
        root = args.root or P.OUT / 'regression/iv'
        train('qwen', REF, [0], args.device, root)
        record = compare_training(root)
        (root / 'regress_iv.json').write_text(json.dumps(record, indent=2) + '\n')
        print(f'0(iv): pass={record["pass"]}', json.dumps(record['items']), flush=True)
        return 0 if record['pass'] else 1
    if args.command == 'check-export':
        out_root = args.out_root or P.OUT / 'regression/ix'
        export_held('qwen', REF, args.seeds, R10_FULL, device=args.device, out_root=out_root)
        export_validation('qwen', REF, args.seeds, R10_FULL, device=args.device, out_root=out_root)
        record = compare_exports(out_root, seeds=args.seeds)
        (Path(out_root) / 'check_ix.json').write_text(json.dumps(record, indent=2) + '\n')
        print(f'0(ix): pass={record["pass"]}; failing: {[k for k, v in record["items"].items() if not v["pass"]]}',
              flush=True)
        return 0 if record['pass'] else 1
    assert args.row == REF or args.root is not None, 'a knob row needs --root (the sweep\'s own folder)'
    root = args.root or default_root(args.encoder)
    if args.command == 'train':
        runs = train(args.encoder, args.row, args.seeds, args.device, root, keep=args.keep, smoke=args.smoke,
                     max_epochs=args.max_epochs)
        for seed, run in runs.items():
            r = run.record
            print(f'BS_CSTAR train {args.encoder} {args.row} s{seed} {r["status"]} seconds={r["seconds"]:.1f} '
                  f'best_epoch={r["best_epoch"]} best_dev_loss={r["best_dev_loss"]}', flush=True)
        return 0
    for kind in args.kind:
        (export_held if kind == 'held' else export_validation)(
            args.encoder, args.row, args.seeds, root, device=args.device, out_root=args.out_root, smoke=args.smoke)
        print(f'BS_CSTAR export {kind} {args.encoder} {args.row} done', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
