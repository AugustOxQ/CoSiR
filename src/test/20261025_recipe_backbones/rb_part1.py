"""Part 1 trainer: a recipe of `rb_rows` at seed 0 on the frozen sample, with exports on P and at epoch 4.

Training is 12-22's: `ab_train.run_cohort` (-> `ms_train.run`), unchanged. Three hooks are added around it and
none touches the training computation or an RNG draw: a wrapper that records the views `build_model` returns, a
wrapper of `ms_train.save_output` (it runs the original first, so the old-5,000 file is the original's) and a
`print` hook that copies the view state with `copy.deepcopy(state_dict())` right after the dev loss of the
snapshot epoch is logged. After the original saver, from the aspect's best state, the wrapper exports the image and
caption outputs on the old 5,000 and on P (under `torch.no_grad()`, eval mode, as the original saver), then the
same from the epoch-4 state.

Files in `<outputs>/part1/train/<recipe>/` (the frozen trainer's own `<aspect>__<graph>__s0.npz|json` stay beside):
  `<aspect>__s0__old5000.npz`, `<aspect>__s0__old5000__e4.npz`, `<aspect>__s0__P.npz`, `<aspect>__s0__P__e4.npz`
  (keys `img`, `cap`, float16 [n, 2048]; old5000 in the sample's `subset_pid` order, P in P.json order),
  `<aspect>__s0.json` (best_epoch, epochs_run, best_dev_loss, history, caption_scored, P sha256, array hashes,
  code hashes), `<aspect>__s0.pt` (best state), `<aspect>__s0__e4.pt` (epoch-4 state, so P can be added later).
P exports run only after a passed stage 0 (V0, `<outputs>/stage0.json`) and check VP2.
"""
import argparse
import builtins
import copy
import hashlib
import json
import re
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import rb_paths
import rb_rows
import ab_train
from ab_rows import ASPECTS
from ab_paths import ms_train

SEED = 0
SNAPSHOT_EPOCH = 4
N_OLD = 5000
N_P = 29232
NORM_TOL = 1e-3
_EPOCH_LINE = re.compile(r'^(\S+)/(\w+) s(\d+): epoch=(\d+) step=')
_CODE = ('rb_part1.py', 'rb_rows.py', 'rb_paths.py')


# ------------------------------------------------------------------ hashes and files
def array_sha256(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def code_hashes():
    """sha256 of the round's trainer code and of the frozen modules it runs (not rb_data: P's code)."""
    paths = {f: rb_paths.ROUND / f for f in _CODE}
    for module in ('ab_train', 'ab_rows', 'ab_paths'):
        paths[module + '.py'] = rb_paths.TEST / '20261021_d2_ablation' / f'{module}.py'
    paths['ms_train.py'] = Path(ms_train.__file__)
    files = {f: hashlib.sha256(p.read_bytes()).hexdigest() for f, p in paths.items()}
    files['ms_code_version'] = ms_train.code_version()
    files['all'] = hashlib.sha256('\n'.join(f'{k}={v}' for k, v in sorted(files.items())).encode()).hexdigest()
    return files


def rb_data_sha256():
    return hashlib.sha256((rb_paths.ROUND / 'rb_data.py').read_bytes()).hexdigest()


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, **arrays)
    pending.replace(path)


def atomic_pt(path, state):
    pending = path.with_name(path.stem + '.pending.pt')
    torch.save({k: v.detach().cpu() for k, v in state.items()}, pending)
    pending.replace(path)


def atomic_json(path, value):
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def describe(arrays):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_sha256(v)) for k, v in arrays.items()}


def check_p(x, name):
    """VP2: 29,232 finite float16 rows with unit norm (|norm - 1| < 1e-3 on the float32 cast)."""
    assert x.dtype == np.float16 and x.shape == (N_P, 2048), (name, x.dtype, x.shape)
    assert np.isfinite(x).all(), f'{name}: nonfinite rows'
    err = np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max()
    assert err < NORM_TOL, f'{name}: norm error {err}'


# ------------------------------------------------------------------ inference
def infer(model, x, device):
    """Image or caption output of float32 rows `x`, batches of the trainer's size, float16 (the original saver's loop)."""
    out = np.empty((len(x), 2048), dtype=np.float16)
    batch = ms_train.CONFIG['batch_size']
    with torch.no_grad():
        for a in range(0, len(x), batch):
            xx = torch.from_numpy(np.ascontiguousarray(x[a:a + batch])).to(device)
            out[a:a + len(xx)] = model(xx).cpu().numpy().astype(np.float16)
    return out


def infer_sides(view, rows, device):
    """{'img', 'cap'} outputs of the view (eval mode) on rows = {'img': x, 'cap': x}."""
    view.eval()
    return {side: infer(view[side], rows[side], device) for side in ('img', 'cap')}


def p_inputs():
    """P's default images and caption means, float32 unit-norm rows in P order (`rb_data`, ticket 01)."""
    import rb_data
    names = rb_data.p_names()
    return dict(img=rb_data.p_default_images(), cap=rb_data.p_default_captions(),
                sha256=rb_data.p_sha256(), names=names, counts=rb_data.p_counts())


def check_p_inputs(p):
    assert p['img'].shape == p['cap'].shape == (N_P, 2048) and p['img'].dtype == p['cap'].dtype == np.float32
    for side in ('img', 'cap'):
        assert np.isfinite(p[side]).all()
        assert np.abs(np.linalg.norm(p[side], axis=1) - 1).max() < NORM_TOL


# ------------------------------------------------------------------ gate
def require_stage0(outputs, smoke):
    """P exports need a passed stage 0 (V0) of this code; a smoke run needs a smoke stage 0 of this code."""
    path = Path(outputs) / 'stage0.json'
    assert path.is_file(), f'{path} missing: run rb_stage0.py first (V0)'
    record = json.loads(path.read_text())
    assert record['code_sha256'] == code_hashes()['all'], 'stage 0 ran other trainer code'
    if smoke:
        assert record['smoke'] is True and record['items'], 'smoke needs a smoke stage 0'
    else:
        assert record['smoke'] is False and record['pass'] is True, f'{path}: V0 did not pass'
    return record


# ------------------------------------------------------------------ one cohort
def final_json(outdir, aspect):
    return Path(outdir) / f'{aspect}__s{SEED}.json'


def _write_aspect(outdir, name, aspect, info, snapshot, files, caption_scored, p, state_files, row_info):
    record = dict(recipe=name, aspect=aspect, seed=SEED, best_epoch=info['best_epoch'],
                  epochs_run=info['epochs_run'], best_dev_loss=info['best_dev_loss'], history=info['history'],
                  caption_scored=caption_scored, snapshot_epoch=snapshot,
                  p_exports=p is not None, p_sha256=None if p is None else p['sha256'],
                  files={f: describe(a) for f, a in files.items()}, state_files=state_files,
                  ms_output_sha256=info['output_sha256'], seconds=info['seconds'],
                  code_sha256=code_hashes(), rb_data_sha256=None if p is None else rb_data_sha256(),
                  **row_info)
    atomic_json(final_json(outdir, aspect), record)
    return record


def _export_set(outdir, aspect, tag, old=None, p_out=None, p=None):
    """Write `__old5000{tag}` and `__P{tag}` npz files; return {file name: arrays}."""
    written = {}
    for label, arrays in (('old5000', old), ('P', p_out)):
        if arrays is None:
            continue
        if label == 'P':
            for side in ('img', 'cap'):
                check_p(arrays[side], f'{aspect} P{tag} {side}')
        fname = f'{aspect}__s{SEED}__{label}{tag}.npz'
        atomic_npz(Path(outdir) / fname, arrays)
        written[fname] = arrays
    return written


def train_cohort(s, inputs, graphs, manifests, recipe, name, cohort, device, outputs, *,
                 p=None, smoke=False, max_epochs=200, snapshot_epoch=SNAPSHOT_EPOCH):
    """Train one cohort through the frozen runner and export. `p` is `p_inputs()` or None (P exports off)."""
    outputs = Path(outputs)
    train_dir = outputs / 'part1' / 'train'
    outdir = train_dir / name
    assert 1 <= snapshot_epoch <= max_epochs
    caption_scored = not (recipe['switches']['endpoints'] == 'image' or not recipe['base']['graph_terms'])
    row_info = dict(switches=recipe['switches'], distill_weight=recipe['base']['distill_weight'],
                    caption_side='scored' if caption_scored else 'not scored')
    ms_files = [ms_train.output_path(train_dir, name, a, recipe['base']['graphs'][a], SEED) for a in cohort]
    done = [final_json(outdir, a).is_file() for a in cohort]
    if all(done):
        return {a: json.loads(final_json(outdir, a).read_text()) for a in cohort}
    assert not any(f.exists() for f in ms_files), \
        f'{name}: trainer outputs exist without exports; move {outdir} aside and rerun'
    captured, snaps, records = {}, {}, {}
    original_build, original_save = ms_train.build_model, ms_train.save_output

    def build(row, aspects, dev):
        models, views = original_build(row, aspects, dev)
        captured['views'] = views
        return models, views

    def hook(*args, **kwargs):
        builtins.print(*args, **kwargs)
        match = _EPOCH_LINE.match(args[0]) if args and isinstance(args[0], str) else None
        if match and match[1] == name and int(match[3]) == SEED and int(match[4]) == snapshot_epoch:
            aspect = match[2]
            assert aspect not in snaps
            snaps[aspect] = copy.deepcopy(captured['views'][aspect].state_dict())

    def save(s_, view, inputs_, state, path, config, start):
        info = original_save(s_, view, inputs_, state, path, config, start)
        aspect = config['aspect']
        dev = torch.device(config['device'])
        assert aspect in snaps, f'{name}/{aspect}: epoch {snapshot_epoch} not reached'
        with np.load(path, allow_pickle=False) as z:
            best_old = {k: z[k] for k in ('img', 'cap')}
        ids = s_['subset_pid']
        old_rows = {k: inputs_[k][ids] for k in ('img', 'cap')}
        assert best_old['img'].shape == (N_OLD, 2048)
        files = _export_set(outdir, aspect, '', old=best_old)
        state_files = {}
        if p is not None:
            files.update(_export_set(outdir, aspect, '', p_out=infer_sides(view, p, dev)))
        atomic_pt(outdir / f'{aspect}__s{SEED}.pt', state['best_state'])
        state_files[f'{aspect}__s{SEED}.pt'] = dict(epoch=info['best_epoch'])
        view.load_state_dict(snaps[aspect])
        files.update(_export_set(outdir, aspect, '__e4', old=infer_sides(view, old_rows, dev)))
        if p is not None:
            files.update(_export_set(outdir, aspect, '__e4', p_out=infer_sides(view, p, dev)))
        atomic_pt(outdir / f'{aspect}__s{SEED}__e4.pt', snaps[aspect])
        state_files[f'{aspect}__s{SEED}__e4.pt'] = dict(epoch=snapshot_epoch)
        records[aspect] = (info, files, state_files)
        return info

    with ExitStack() as stack:
        stack.enter_context(patch.object(ms_train, 'build_model', build))
        stack.enter_context(patch.object(ms_train, 'save_output', save))
        stack.enter_context(patch.object(ms_train, 'print', hook, create=True))
        ab_train.run_cohort(s, inputs, graphs, manifests, recipe, name, cohort, SEED, device,
                            train_dir, smoke, max_epochs)
    assert set(records) == set(cohort)
    for aspect in cohort:
        info, files, state_files = records[aspect]
        records[aspect] = _write_aspect(outdir, name, aspect, info, snapshot_epoch, files, caption_scored,
                                        p, state_files, row_info)
    return records


def add_p(recipe, name, cohort, device, outputs, p):
    """P exports from saved states of a finished P-off run (no retraining); the old-5,000 files stay as they are."""
    outdir = Path(outputs) / 'part1' / 'train' / name
    device = torch.device(device)
    records = {a: json.loads(final_json(outdir, a).read_text()) for a in cohort}
    if all(r['p_exports'] for r in records.values()):
        assert all(r['p_sha256'] == p['sha256'] for r in records.values())
        return records
    assert not any(r['p_exports'] for r in records.values())
    # Rebuilding the modules draws from the global torch RNG: fork it, so nothing later sees the draws.
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        _, views = ms_train.build_model(recipe['base'], cohort, device)
    for aspect in cohort:
        record = records[aspect]
        files = {}
        for tag, pt in (('', f'{aspect}__s{SEED}.pt'), ('__e4', f'{aspect}__s{SEED}__e4.pt')):
            state = torch.load(outdir / pt, map_location=device, weights_only=True)
            views[aspect].load_state_dict(state)
            files.update(_export_set(outdir, aspect, tag, p_out=infer_sides(views[aspect], p, device)))
        old_files = {}
        for fname in record['files']:
            with np.load(outdir / fname, allow_pickle=False) as z:
                old_files[fname] = {k: z[k] for k in z.files}
        old_files.update(files)
        record.update(p_exports=True, p_sha256=p['sha256'], rb_data_sha256=rb_data_sha256(),
                      files={f: describe(a) for f, a in old_files.items()}, code_sha256_add_p=code_hashes())
        atomic_json(final_json(outdir, aspect), record)
    return records


# ------------------------------------------------------------------ all recipes
def resolved(name):
    """12-22's resolved row at seed 0 (graph names formatted)."""
    recipe = rb_rows.recipe(name)
    if recipe['base']['graphs'] is not None:
        recipe['base']['graphs'] = {a: g.format(seed=SEED) for a, g in recipe['base']['graphs'].items()}
    return recipe


@threadpool_limits.wrap(limits=8)
def train(recipes, outputs=rb_paths.OUTPUTS, device='cuda', *, smoke=False, with_p=True, max_epochs=None):
    recipes = list(recipes)
    assert recipes and len(set(recipes)) == len(recipes) and all(r in rb_rows.RECIPES for r in recipes)
    max_epochs = (2 if smoke else ms_train.CONFIG['max_epochs']) if max_epochs is None else max_epochs
    assert not smoke or max_epochs == 2, '--smoke requires 2 epochs'
    snapshot = min(SNAPSHOT_EPOCH, max_epochs)
    outputs = ab_train.validate_outputs(outputs, smoke)
    rb_paths.check_imports()
    p = None
    if with_p:
        require_stage0(outputs, smoke)
        p = p_inputs()
        check_p_inputs(p)
        import rb_data
        rb_data.write_p_json(outputs, p['names'], p['counts'])
        rb_data.read_p_json(outputs, expect=p['names'])
    for name in recipes:
        ab_train.require_graph_store(resolved(name))
    torch.set_num_threads(8)
    device = torch.device(device)
    assert device.type in ('cpu', 'cuda')
    s = ab_train.load_sample()
    fit, dev = ms_train.disjoint_split(s)
    default_inputs = ms_train.head.load_inputs(s, ASPECTS[0], 'default')
    result = {}
    for name in recipes:
        recipe = resolved(name)
        row = recipe['base']
        base = ab_train.require_graph_store(recipe)
        row_inputs = ({a: ms_train.head.load_inputs(s, a, 'prompted') for a in ASPECTS}
                      if row['inputs'] == 'instructed' else {a: default_inputs for a in ASPECTS})
        cohorts = [[a] for a in ASPECTS] if row['share'] == 'separate' else [list(ASPECTS)]
        graphs, manifests = {}, {}
        for aspect in ASPECTS:
            graphs[aspect], manifests[aspect] = ms_train.head.load_graph(s, base, aspect, [row['graphs'][aspect]])
        result[name] = {}
        for cohort in cohorts:
            outdir = Path(outputs) / 'part1' / 'train' / name
            if all(final_json(outdir, a).is_file() for a in cohort):
                if p is not None:
                    result[name].update(add_p(recipe, name, cohort, device, outputs, p))
                else:
                    result[name].update({a: json.loads(final_json(outdir, a).read_text()) for a in cohort})
                continue
            result[name].update(train_cohort(
                s, {a: row_inputs[a] for a in cohort}, graphs, manifests, recipe, name, cohort, device,
                outputs, p=p, smoke=smoke, max_epochs=max_epochs, snapshot_epoch=snapshot))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--recipes', nargs='+', required=True, choices=list(rb_rows.ORDER))
    parser.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-p', action='store_true', help='skip the exports on P (stage 0 runs this way)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args(argv)
    t0 = time.monotonic()
    train(args.recipes, args.outputs, args.device, smoke=args.smoke, with_p=not args.no_p)
    print(f'RB_PART1 DONE {args.recipes} p={not args.no_p} seconds={time.monotonic() - t0:.0f}')


if __name__ == '__main__':
    main()
