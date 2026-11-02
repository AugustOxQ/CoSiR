"""r14 ticket 04: r13's photo fine-tune for any image encoder, the photo sweep rows, and the emotion-pathway export.

Spec 2026-11-02-backbones-sweep-design.md §3.2 (X photo model; the photo sweep), §3.4 Training, §3.5 0(v) and 0(ix),
D14, D15. Nothing of r13 is edited; r13's modules are imported (its round folder goes on `sys.path` as its own modules
do) and used by call, with these differences and only these:

  * the image input has the encoder's width. The model is `rb_student.build_model(..., width=w)` (12-24's factory that
    opens the image trunk's first Linear; at 2048 it is r13's own `ms_train.build_model`, called directly), so only
    `img.trunk.0.weight` changes shape. Graphs, the teacher T, the caption mean D and the train set (69,927 photos after
    r13's near-copy cut) stay r13's and Qwen-built; only the array `I` that enters the loss is the encoder's default
    image (`with_images`). The start point is X-c*'s emotion checkpoint of seed s (Qwen: r10's frozen c*).
  * the epoch loop `fit_photo` is a copy of `pf_train._fit` (its width asserts, its failure handling and its
    best-state rule could not be wrapped): same optimizer, RNG streams, batch order, strict-improvement early stopping
    on the batch-weighted validation loss, best-state copy. Added: a fixed-epoch mode (`settings['epochs'] = N`: no
    stopping, the state after exactly N epochs is the checkpoint; epoch 0, the start point, is never one) and an
    `on_epoch` hook. `pf_train.dev_loss` and `pf_train.checked_load` are r13's, called as they are. A non-finite
    loss raises `NonFiniteLoss` (the rerun rule is the caller's, ticket 10).
  * files and record: `train_run` writes `<out_dir>/emotion.pt` (keep 'all') and `<out_dir>/run.json` (r13's fields
    plus encoder_width, host, fixed_epochs, code sha); keep 'none' writes `run.json` only and returns the state in
    memory (`PhotoRun.state`) for the caller's scoring (D15: no weights, no exports).

Photo model of encoder `enc`, seed s: `OUT/photo/<enc>/s<s>/{emotion.pt, run.json}` (as r13's `runs/final/s<s>/`),
start X-c* seed s, training seed 100 + s. Sweep rows: `ROWS` (`ref` and 14 one-change rows, names `<setting>=<value>`).

Exports (`export_checkpoint`, any emotion checkpoint of the arms `photo_qwen` (r13's five), `cstar_<enc>`,
`photo_<enc>`): `OUT/exports/<dataset>/<arm>/<split>__s<seed>.npz` with `keys` and `emb` (float16 [n, 2048]) and a
`.json` sidecar (`sha256` of the npz file, `emb_sha256`, `keys_sha256`, the source checkpoint and its sha256, ...).
Affection val/test keys are r13's (`pf_data.keys`, sorted); EmoSet keys are the store's own key list, val rows first
and then test (ticket 01), whose evaluation order is ticket 02's job.

CLI:  bs_photo.py train --encoder E [--rows ref ...] [--seeds ...] [--keep all|none] [--root DIR] [--device cuda]
      bs_photo.py export --arm ARM --dataset affection|emoset [--splits val test] [--seeds ...] [--root DIR]
      bs_photo.py regress-v [--root DIR]        0(v): Qwen, r13's final settings, seed 0 against r13's sidecar
      bs_photo.py check-export [--root DIR]     0(ix): r13's five photo models and the frozen c* on Affection val/test
"""
import argparse
import copy
import hashlib
import json
import socket
import sys
import time
import traceback
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bs_paths as P  # noqa: E402

if str(P.R13_ROUND) not in sys.path:
    sys.path.insert(0, str(P.R13_ROUND))
import pf_paths  # noqa: E402  (r13's; through it r12's, r11's, r10's and every older round's folder on sys.path)
import pf_data  # noqa: E402
import pf_graphs  # noqa: E402
import pf_train  # noqa: E402
import ac_data  # noqa: E402
import ab_train  # noqa: E402
import ms_rows  # noqa: E402
import ms_train  # noqa: E402
import rb_rows  # noqa: E402
import rb_student  # noqa: E402
import rs_vh  # noqa: E402

import bs_stores  # noqa: E402

NonFiniteLoss = pf_train.NonFiniteLoss
RECIPE = dict(pf_paths.RECIPE)                          # weight decay 0.05, batch 512 pairs, 200 epochs, patience 15
WIDTH_QWEN = 2048
KEEP = ('all', 'none')
VIEW = {'qwen': 'qwen_default', 'siglip2': 'siglip2', 'siglip1': 'siglip1', 'clip_l14': 'clip_l14'}
SETTING_KEYS = tuple(pf_paths.COLUMNS) + ('epochs',)    # start, lr, k, correction, teacher_weight, epochs (None: stop by patience)
SPLITS = {'affection': ('val', 'test'), 'emoset': ('val', 'test')}
TOL_TRAIN = 1e-4                                        # 0(v)
TOL_EXPORT = 1e-3                                       # 0(ix) (r12's bar)
BATCH = 512                           # inference batch (c*'s)

# ------------------------------------------------------------------------------------------------ the rows (§3.2)
REF = 'ref'
PICK = dict(start='cstar', lr=0.001, k=80, correction=True, teacher_weight=0.25, epochs=None)   # r13's pick (row 5)


def _fmt(v):
    return 'off' if v is False else 'on' if v is True else f'{v:g}' if not isinstance(v, str) else v


def _row(column, value):
    return f'{column}={_fmt(value)}', dict(PICK, **{column: value})


_CHANGES = ([('k', v) for v in (20, 40, 160)] + [('lr', v) for v in (0.0001, 0.0003, 0.003)] +
            [('epochs', v) for v in (2, 5, 20)] + [('teacher_weight', v) for v in (0.0, 0.5, 1.0)] +
            [('start', 'random'), ('correction', False)])
ROWS = {REF: dict(PICK), **dict(_row(c, v) for c, v in _CHANGES)}
SWEEP_ROWS = [n for n in ROWS if n != REF]
assert len(SWEEP_ROWS) == 14 and tuple(PICK) == SETTING_KEYS


def train_seed(s):
    return P.PHOTO_SEED_OFFSET + s


# ------------------------------------------------------------------------------------------------ model and loss
def loss_recipe(teacher_weight):
    """r13's (row, switches) of c*'s switched loss; teacher weight 0 removes the direct teacher term altogether."""
    if teacher_weight > 0:
        return pf_train.loss_recipe(teacher_weight)
    recipe = rb_rows.recipe(pf_train.RECIPE_NAME)
    row = copy.deepcopy(recipe['base'])
    row.update(distill='none', distill_weight=0.0)
    ms_rows.validate(row)
    return row, copy.deepcopy(recipe['switches'])


def build_emotion(device, width, seed=None):
    """r13's `pf_train.build_emotion` with the image trunk's first layer taking `width`: (models, emotion view). At
    2048 the call is r13's own; elsewhere 12-24's factory (3 aspects built in c*'s order, only emotion kept)."""
    if width == WIDTH_QWEN:
        return pf_train.build_emotion(device, seed=seed)
    row, _ = loss_recipe(0.25)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    models, views = rb_student.build_model(row, list(pf_train.ASPECTS), device, width=width)
    return models, views['emotion']


def _tensors(part, device, width):
    """pf_train._tensors with the image width a parameter: I [n, width], D and T [n, 2048]; all float32."""
    out = {}
    for name, key, w in (('img', 'I', width), ('cap', 'D', 2048), ('teacher', 'T', 2048)):
        x = part[key]
        assert x.ndim == 2 and x.shape[1] == w and x.dtype == np.float32, (key, x.shape, x.dtype, w)
        out[name] = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    assert len({len(v) for v in out.values()}) == 1
    return out


def with_images(part, x):
    """`part` (r13's arrays and keys) with the image input `I` replaced by `x` (unit float32 rows of the same keys)."""
    x = np.asarray(x)
    assert x.ndim == 2 and x.dtype == np.float32 and len(x) == len(part['keys']) and np.isfinite(x).all()
    return dict(part, I=x)


# ---------------------------------------------------------------------------------------------------- training
PhotoRun = namedtuple('PhotoRun', 'record state')       # state: CPU state dict of the saved checkpoint (always set)


def fit_photo(run_id, settings, width, cstar_seed, train_seed_, fit, val, graph_pairs, device, *, max_epochs=None,
              cstar_root=None, record=None, on_epoch=None):
    """The training: (state, epoch, dev_loss, history). Standard (`settings['epochs']` None): at most `max_epochs`
    (200) epochs, strict-improvement early stopping with patience 15, the best epoch's state. Fixed (`epochs` = N): exactly
    N epochs, no stopping, the state after epoch N and its validation loss. `pf_train._fit` otherwise (see the module doc)."""
    record = {} if record is None else record
    assert set(settings) == set(SETTING_KEYS), settings
    fixed = settings['epochs']
    assert fixed is None or (isinstance(fixed, int) and 1 <= fixed <= RECIPE['max_epochs'])
    assert fixed is None or max_epochs is None, 'a fixed-epoch row takes no max_epochs'
    max_epochs = fixed if fixed is not None else (RECIPE['max_epochs'] if max_epochs is None else max_epochs)
    assert 1 <= max_epochs <= RECIPE['max_epochs']
    start = settings['start']
    assert start in ('cstar', 'random'), start
    row, switches = loss_recipe(settings['teacher_weight'])
    fit_pairs, dev_pairs = graph_pairs
    n_fit, n_val = len(fit['I']), len(val['I'])
    pf_train._check_pairs(fit_pairs, n_fit)
    pf_train._check_pairs(dev_pairs, n_val)
    # pf_train._fit / ct2_d2.run: seed the global streams, a per-run generator, then build the model.
    np.random.seed(train_seed_)
    torch.manual_seed(train_seed_)
    rng = np.random.default_rng(train_seed_)
    _, view = build_emotion(device, width, seed=train_seed_)
    if start == 'cstar':
        record['cstar_weights_sha256'] = pf_train.checked_load(view, cstar_seed, cstar_root)
    opt = torch.optim.AdamW(view.parameters(), lr=settings['lr'], weight_decay=RECIPE['weight_decay'])
    t_fit, t_dev = _tensors(fit, device, width), _tensors(val, device, width)
    batch = RECIPE['batch_pairs']
    assert batch == ms_train.CONFIG['batch_size'] and RECIPE['weight_decay'] == ms_train.CONFIG['weight_decay']
    patience = RECIPE['patience']
    fit_w = np.ones(len(fit_pairs), dtype=np.float32)
    history, best, best_epoch, best_state, stale = [], float('inf'), 0, None, 0
    for epoch in range(1, max_epochs + 1):                       # epoch 0 (the start point) is never a candidate
        tick = time.monotonic()
        view.train()
        order = rng.permutation(len(fit_pairs))
        total = 0.
        for pos in range(0, len(order), batch):
            ix = order[pos:pos + batch]
            opt.zero_grad(set_to_none=True)
            loss = ab_train.switched_head_loss(view, **t_fit, pairs=fit_pairs[ix], weights=fit_w[ix], row=row,
                                               aspect='emotion', switches=switches)
            if not bool(torch.isfinite(loss)):
                raise NonFiniteLoss('train')
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ix)
        dev = pf_train.dev_loss(view, t_dev, dev_pairs, row, switches)
        history.append(dict(epoch=epoch, train_loss=total / len(fit_pairs), dev_loss=dev,
                            seconds=time.monotonic() - tick))
        if on_epoch is not None:
            on_epoch(epoch, view.state_dict())
        if fixed is not None:
            best, best_epoch = dev, epoch                        # the last epoch is the checkpoint
        elif dev < best:
            best, best_epoch, stale = dev, epoch, 0
            best_state = copy.deepcopy(view.state_dict())
        else:
            stale += 1
        print(f'{run_id}: epoch={epoch} train={history[-1]["train_loss"]:.6f} dev={dev:.6f} '
              f'seconds={history[-1]["seconds"]:.2f}', flush=True)
        if fixed is None and stale >= patience:
            break
    if fixed is not None:
        best_state = copy.deepcopy(view.state_dict())
    assert best_state is not None and tuple(best_state) == pf_train.CKPT_KEYS
    return {k: v.detach().cpu() for k, v in best_state.items()}, best_epoch, best, history


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(obj, indent=1, allow_nan=False))
    pending.replace(path)


def _code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def train_run(run_id, settings, width, cstar_seed, train_seed_, fit, val, graph, device, *, keep='none', out_dir=None,
              max_epochs=None, cstar_root=None, on_epoch=None, encoder=None, row=None):
    """One photo-model run (a row of `ROWS`, or r13's): returns `PhotoRun(record, state)`.

    `fit`, `val`: r13's dicts (`keys`, float32 `I` [n, width], `D`, `T` [n, 2048]; `E` is not used here, it only shapes
    the graph). `graph`: dict of int64 `fit_pairs`, `dev_pairs` (local rows) and optional `manifest_sha256`.
    keep 'all' needs `out_dir` and refuses to overwrite a checkpoint; 'none' writes `out_dir/run.json` if given.
    A non-finite loss writes `failed.json` (keep 'all' only) and raises `NonFiniteLoss`; any exception propagates."""
    assert keep in KEEP and (keep == 'none' or out_dir is not None)
    out_dir = None if out_dir is None else Path(out_dir)
    ckpt = None if out_dir is None else out_dir / 'emotion.pt'
    if keep == 'all':
        assert not ckpt.exists(), f'{ckpt}: exists (the run is done; move it aside to rerun)'
    settings = dict(settings)
    wall = time.monotonic()
    record = dict(run_id=run_id, encoder=encoder, row=row, encoder_width=width, status='failed', settings=settings,
                  fixed_epochs=settings.get('epochs'), cstar_seed=cstar_seed, cstar_weights_sha256=None,
                  train_seed=train_seed_, weights_sha256=None, best_epoch=None, best_dev_loss=None, history=[],
                  graph_manifest_sha256=graph.get('manifest_sha256'), fit_keys_sha256=rs_vh.names_sha(fit['keys']),
                  val_keys_sha256=rs_vh.names_sha(val['keys']), device=str(device), host=socket.gethostname(),
                  bs_photo_sha256=_code_sha(), seconds=None)
    assert len(fit['keys']) == len(fit['I']) and len(val['keys']) == len(val['I'])
    try:
        state, epoch, dev, history = fit_photo(run_id, settings, width, cstar_seed, train_seed_, fit, val,
                                               (graph['fit_pairs'], graph['dev_pairs']), torch.device(device),
                                               max_epochs=max_epochs, cstar_root=cstar_root, record=record,
                                               on_epoch=on_epoch)
    except NonFiniteLoss as e:
        record.update(error=f'non-finite {e.args[0]} loss', seconds=time.monotonic() - wall)
        if keep == 'all':
            _atomic_json(out_dir / 'failed.json', record)
        raise
    if keep == 'all':
        out_dir.mkdir(parents=True, exist_ok=True)
        pending = ckpt.with_name('emotion.pending.pt')
        torch.save(state, pending)
        pending.replace(ckpt)
        record['weights_sha256'] = rs_vh.file_sha(ckpt)
    record.update(status='ok', best_epoch=epoch, best_dev_loss=dev, history=history,
                  seconds=time.monotonic() - wall)
    if out_dir is not None:
        _atomic_json(out_dir / 'run.json', record)
    return PhotoRun(record, state)


# ---------------------------------------------------------------------------------------- data and graphs (CLI)
def images(dataset, enc, keys):
    """Unit float32 default-image rows of `keys` (joined by key, C6). Affection with Qwen is r13's own path."""
    keys = list(keys)
    if dataset == 'affection' and enc == 'qwen':
        return pf_data.default_images(keys)
    return ac_data._unit(bs_stores.load(dataset, VIEW[enc], keys))


def load_data(enc):
    """(fit, val, fit_q, val_q): r13's train set (69,927 photos, near-copy cut) and the 4,251 validation photos.
    `fit`, `val` carry the encoder's image as `I`; `fit_q`, `val_q` are r13's Qwen arrays (the graphs' input)."""
    fk, vk = pf_data.fit_keys(), pf_data.keys('val')
    assert len(vk) == pf_paths.N['val'] and len(fk) == 69927, (len(fk), len(vk))
    fit_q, val_q = dict(pf_data.photo_arrays(fk), keys=fk), dict(pf_data.photo_arrays(vk), keys=vk)
    if enc == 'qwen':
        return fit_q, val_q, fit_q, val_q
    return (with_images(fit_q, images('affection', enc, fk)), with_images(val_q, images('affection', enc, vk)),
            fit_q, val_q)


class Graphs:
    """r13's (k, correction) graphs on the Qwen arrays. A graph r13 already built (its outputs, read only) is loaded
    from there through `pf_graphs.load` (key-hash and code-hash refusals); a missing one is built into
    `<root>/photo_graphs` by r13's `pf_graphs.build` (never into r13's folder) and loaded from there."""

    def __init__(self, fit_q, val_q, root=None):
        self.fit_q, self.val_q, self.cache = fit_q, val_q, {}
        self.root = Path(P.OUT if root is None else root) / 'photo_graphs'

    @staticmethod
    def _name(k, correction):
        return f'k{int(k)}__{"on" if correction else "off"}'

    def _folder(self, k, correction):
        mine, theirs = self.root / self._name(k, correction), P.R13_OUT / 'graphs' / self._name(k, correction)
        return theirs if (theirs / 'emotion.npz').is_file() and not (mine / 'emotion.npz').is_file() else mine

    def get(self, k, correction):
        key = (int(k), bool(correction))
        if key not in self.cache:
            folder = self._folder(*key)
            if not (folder / 'emotion.npz').is_file():
                with patch.object(pf_graphs, '_folder', lambda k_, c_: self._folder(k_, c_)):
                    pf_graphs.build(*key, self.fit_q, self.val_q)
            with patch.object(pf_graphs, '_folder', lambda k_, c_: self._folder(k_, c_)):
                self.cache[key] = pf_graphs.load(*key, self.fit_q['keys'], self.val_q['keys'])
        return self.cache[key]


def start_root(enc, root=None):
    """The folder that holds X-c*'s `emotion__s<seed>.pt` (+ sidecar): r10's frozen c* for Qwen, ticket 03's else."""
    if enc == 'qwen':
        return Path(pf_paths.CSTAR_WEIGHTS)
    return Path(P.OUT if root is None else root) / 'cstar' / enc / 'weights' / 'k11=40'


def run_dir(enc, row, seed, root=None):
    base = Path(P.OUT if root is None else root) / 'photo' / enc
    return base / f's{seed}' if row == REF else base / 'rows' / row / f's{seed}'


def train(enc, rows, seeds, device, *, keep=None, root=None, max_epochs=None, data=None, graphs=None):
    """The CLI's loop: each (row, seed) through `train_run`; a finished run (its run.json) is not retrained. `keep`
    defaults to 'all' for `ref` (the photo model) and 'none' for sweep rows. Returns {(row, seed): PhotoRun}."""
    assert enc in VIEW and all(r in ROWS for r in rows) and all(s in P.SEEDS for s in seeds)
    fit, val, fit_q, val_q = data if data is not None else load_data(enc)
    graphs = graphs if graphs is not None else Graphs(fit_q, val_q, root)
    width = P.ENCODERS[enc]['width']
    out = {}
    for row in rows:
        for seed in seeds:
            k = keep or ('all' if row == REF else 'none')
            folder = run_dir(enc, row, seed, root)
            if (folder / 'run.json').is_file():
                rec = json.loads((folder / 'run.json').read_text())
                assert rec['settings'] == ROWS[row] and rec['train_seed'] == train_seed(seed) and \
                    rec['fit_keys_sha256'] == rs_vh.names_sha(fit['keys']), f'{folder}: another run; move it aside'
                print(f'{enc} {row} s{seed}: finished; skip', flush=True)
                out[(row, seed)] = PhotoRun(rec, None)
                continue
            s = ROWS[row]
            out[(row, seed)] = train_run(f'{enc}/{row}/s{seed}', s, width, seed, train_seed(seed), fit, val,
                                         graphs.get(s['k'], s['correction']), device, keep=k, out_dir=folder,
                                         max_epochs=max_epochs, cstar_root=start_root(enc, root), encoder=enc, row=row)
            r = out[(row, seed)].record
            print(f'{enc} {row} s{seed}: best_epoch={r["best_epoch"]} best_dev_loss={r["best_dev_loss"]:.6f} '
                  f'seconds={r["seconds"]:.1f} host={r["host"]} device={r["device"]}', flush=True)
    return out


# ------------------------------------------------------------------------------------------------------ export
def infer(state, x, width):
    """float16 [n, 2048] emotion image-side output of the checkpoint `state` on float32 unit rows `x` [n, width]:
    `pf_export.infer`'s procedure (CPU, eval, no grad, batches of 512, float16 cast) at any image width."""
    assert tuple(state) == pf_train.CKPT_KEYS, list(state)
    x = np.asarray(x)
    assert x.ndim == 2 and x.shape[1] == width and x.dtype == np.float32 and np.isfinite(x).all()
    torch.set_num_threads(8)
    device = torch.device('cpu')
    _, view = build_emotion(device, width)
    view.load_state_dict(state)                      # strict
    view.eval()
    out = np.empty((len(x), 2048), dtype=np.float16)
    with torch.no_grad():
        for a in range(0, len(x), BATCH):
            xx = torch.from_numpy(np.ascontiguousarray(x[a:a + BATCH])).to(device)
            out[a:a + len(xx)] = view['img'](xx).cpu().numpy().astype(np.float16)
    assert np.isfinite(out).all()
    if len(out):
        assert np.max(np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1)) < 1e-3
    return out


def arm_checkpoint(arm, seed, root=None):
    """(checkpoint path, image width) of an emotion arm: `photo_qwen` (r13's five), `cstar_qwen` (frozen c*),
    `photo_<enc>`, `cstar_<enc>`."""
    kind, _, enc = arm.partition('_')
    assert kind in ('photo', 'cstar') and enc in P.ENCODERS and seed in P.SEEDS, (arm, seed)
    base = Path(P.OUT if root is None else root)
    width = P.ENCODERS[enc]['width']
    if kind == 'photo':
        return (P.R13_OUT / f'runs/final/s{seed}/emotion.pt' if enc == 'qwen' else base / f'photo/{enc}/s{seed}/emotion.pt',
                width)
    return start_root(enc, root) / f'emotion__s{seed}.pt', width


def checkpoint_sha(arm, seed, root=None):
    """The checkpoint's sha256, verified against its own record (run.json `weights_sha256` of a photo run; the
    sidecar `weights_sha256` of a c* checkpoint)."""
    path, _ = arm_checkpoint(arm, seed, root)
    sha = rs_vh.file_sha(path)
    if arm.startswith('photo'):
        run = json.loads((path.parent / 'run.json').read_text())
        assert run['status'] == 'ok' and sha == run['weights_sha256'], f'{path}: differs from its run.json'
    else:
        side = json.loads(path.with_suffix('.json').read_text())
        assert sha == side['weights_sha256'], f'{path}: differs from its sidecar'
    return sha


def export_path(dataset, arm, split, seed, root=None):
    return Path(P.OUT if root is None else root) / 'exports' / dataset / arm / f'{split}__s{seed}.npz'


def _publish(path, keys, emb, side):
    """Write npz + sidecar; an existing file must hold the same keys and embeddings (else AssertionError)."""
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            same = z['keys'].tolist() == keys and np.array_equal(z['emb'], emb)
        assert same, f'{path}: exists with other content (move it aside to re-export)'
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with open(pending, 'wb') as f:
        np.savez(f, keys=np.asarray(keys), emb=emb)
    pending.replace(path)
    _atomic_json(path.with_suffix('.json'), dict(side, sha256=rs_vh.file_sha(path)))
    return path


def export_checkpoint(dataset, arm, split, seed, ckpt, ckpt_sha, width, keys, x, *, root=None):
    """Run the emotion checkpoint `ckpt` (hash `ckpt_sha`, checked) on the unit rows `x` of `keys`; write the export
    and its sidecar; return the npz path. Keys must be unique (rows are joined by key downstream)."""
    keys = [str(k) for k in keys]
    assert len(keys) == len(x) and len(set(keys)) == len(keys) and keys, 'keys: unique, one per row'
    assert rs_vh.file_sha(ckpt) == ckpt_sha, f'{ckpt}: sha256 differs from {ckpt_sha}'
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    emb = infer(state, x, width)
    side = dict(dataset=dataset, arm=arm, split=split, seed=seed, rows=len(keys), width=width,
                keys_sha256=rs_vh.names_sha(keys), emb_sha256=rs_vh.array_sha(emb), checkpoint=str(ckpt),
                checkpoint_sha256=ckpt_sha)
    return _publish(export_path(dataset, arm, split, seed, root), keys, emb, side)


def export_keys(dataset, split):
    """Keys of an export: Affection val/test (r13's, sorted); EmoSet the store's key list, val rows then test rows."""
    assert dataset in SPLITS and split in SPLITS[dataset]
    if dataset == 'affection':
        return pf_data.keys(split)
    keys = bs_stores.folder_keys(bs_stores.folders('emoset', 'qwen_default')[0], 'emoset')
    n_val, n_test = P.EMOSET_SPLITS['val'], P.EMOSET_SPLITS['test']
    assert len(keys) == n_val + n_test, len(keys)
    return keys[:n_val] if split == 'val' else keys[n_val:]


def export_arm(dataset, arm, splits, seeds, *, root=None):
    """The export of arm `arm`'s checkpoints of `seeds` on the dataset's splits; returns the written paths."""
    enc = arm.partition('_')[2]
    written = []
    for split in splits:
        keys = export_keys(dataset, split)
        x = images(dataset, enc, keys)
        for seed in seeds:
            ckpt, width = arm_checkpoint(arm, seed, root)
            written.append(export_checkpoint(dataset, arm, split, seed, ckpt, checkpoint_sha(arm, seed, root), width,
                                             keys, x, root=root))
    return written


# --------------------------------------------------------------------------------------------- stage 0 checks
def regress_v(root=None, device='cuda'):
    """0(v): Qwen, r13's final settings, seed 0 (c* seed 0, training seed 100), on the local GPU, through this module's
    trainer: best epoch and best development loss against r13's final seed-0 run.json within 1e-4."""
    ref = json.loads((P.R13_OUT / 'runs/final/s0/run.json').read_text())
    run = train('qwen', [REF], [0], device, keep='all', root=root)[(REF, 0)].record
    ok = (run['best_epoch'] == ref['best_epoch'] and abs(run['best_dev_loss'] - ref['best_dev_loss']) <= TOL_TRAIN)
    doc = dict(gate='0(v)', ok=bool(ok), best_epoch=run['best_epoch'], best_epoch_r13=ref['best_epoch'],
               best_dev_loss=run['best_dev_loss'], best_dev_loss_r13=ref['best_dev_loss'],
               abs_diff=abs(run['best_dev_loss'] - ref['best_dev_loss']), tol=TOL_TRAIN,
               weights_sha256=run['weights_sha256'], weights_sha256_r13=ref['weights_sha256'],
               device=run['device'], host=run['host'], seconds=run['seconds'])
    _atomic_json(Path(P.OUT if root is None else root) / 'stage0/photo_regress_v.json', doc)
    print(json.dumps(doc, indent=1))
    return 0 if ok else 1


def _stored_img(path):
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img'}
        return z['img']


def check_export(root=None, seeds=P.SEEDS, splits=('val', 'test')):
    """0(ix), emotion pathway: this module's export of r13's five photo models and of the frozen c* on Affection val
    and test against r13's stored exports (`exports/final/s<s>/<split>/emotion.npz`) and r12's stored c* exports
    (`exports/cstar/<split>/emotion__s<s>.npz`): max abs difference within 1e-3 per file."""
    t0 = time.monotonic()
    root = Path(P.OUT if root is None else root)
    items = {}
    for arm, stored in (('photo_qwen', lambda sp, s: P.R13_OUT / f'exports/final/s{s}/{sp}/emotion.npz'),
                        ('cstar_qwen', lambda sp, s: P.R12_OUT / f'exports/cstar/{sp}/emotion__s{s}.npz')):
        paths = export_arm('affection', arm, splits, seeds, root=root)
        for (split, seed), path in zip([(sp, s) for sp in splits for s in seeds], paths):
            with np.load(path, allow_pickle=False) as z:
                mine = z['emb']
                assert z['keys'].tolist() == pf_data.keys(split)
            ref = _stored_img(stored(split, seed))
            assert ref.shape == mine.shape and ref.dtype == mine.dtype
            diff = float(np.max(np.abs(mine.astype(np.float64) - ref.astype(np.float64))))
            items[f'{arm}/{split}/s{seed}'] = dict(max_abs_diff=diff, ok=bool(diff <= TOL_EXPORT))
    doc = dict(gate='0(ix) emotion pathway', ok=all(i['ok'] for i in items.values()), tol=TOL_EXPORT, items=items,
               seconds=time.monotonic() - t0)
    _atomic_json(root / 'stage0/photo_check_export.json', doc)
    print(json.dumps({k: v['max_abs_diff'] for k, v in items.items()}, indent=1), 'ok:', doc['ok'])
    return 0 if doc['ok'] else 1


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--encoder', required=True, choices=sorted(P.ENCODERS))
    t.add_argument('--rows', nargs='+', default=[REF], choices=list(ROWS))
    t.add_argument('--seeds', type=int, nargs='+', default=list(P.SEEDS), choices=P.SEEDS)
    t.add_argument('--keep', choices=KEEP, default=None, help="default: 'all' for ref, 'none' for sweep rows")
    t.add_argument('--device', default='cuda')
    t.add_argument('--max-epochs', type=int, default=None, help='smoke only; not with a fixed-epoch row')
    t.add_argument('--root', type=Path, default=None, help='default: bs_paths.OUT')
    e = sub.add_parser('export')
    e.add_argument('--arm', required=True)
    e.add_argument('--dataset', required=True, choices=sorted(SPLITS))
    e.add_argument('--splits', nargs='+', default=['val', 'test'], choices=['val', 'test'])
    e.add_argument('--seeds', type=int, nargs='+', default=list(P.SEEDS), choices=P.SEEDS)
    e.add_argument('--root', type=Path, default=None)
    for name in ('regress-v', 'check-export'):
        p = sub.add_parser(name)
        p.add_argument('--root', type=Path, default=None)
        if name == 'regress-v':
            p.add_argument('--device', default='cuda')
    a = parser.parse_args(argv)
    if a.cmd == 'train':
        train(a.encoder, a.rows, a.seeds, a.device, keep=a.keep, root=a.root, max_epochs=a.max_epochs)
        return 0
    if a.cmd == 'export':
        for path in export_arm(a.dataset, a.arm, a.splits, a.seeds, root=a.root):
            print(path)
        return 0
    return regress_v(a.root, a.device) if a.cmd == 'regress-v' else check_export(a.root)


if __name__ == '__main__':
    sys.exit(main())
