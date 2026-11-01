"""Emotion-only trainer of r13 (spec §3.4 training, D2, D10, D11) and stage 0a gate (vi).

`train_run`  one fine-tuning (or from-scratch) run of c*'s emotion pathway on Affection photos. Model, loss and
             hyper-parameters are c*'s by import: `ms_train.build_model` (trunk share, as `rb_student` wraps it at width
             2048), `ab_train.switched_head_loss` with R6-5's switches (`rb_rows.recipe('R6-5')`: four buddy terms, own
             caption term 0, direct teacher term), `ms_train.CONFIG` (temperature 0.05, dropout 0.1, batch 512).
             The epoch loop is a single-head copy of `ct2_d2.py:run` (its loop reads the ArtELingo split dict `s`, a
             teacher hash of `s`, three-head graph archives and ArtELingo-bound config and saver, none of which exist
             for photos). Changes against that loop, and only these: inputs are arrays with fit-local pairs; the model
             starts from a checkpoint or from c*'s initializer; the optimizer covers the emotion view only; failures
             are caught and recorded. Optimization order, RNG streams (`np.random.seed`, `torch.manual_seed`,
             `default_rng(train_seed)` permuting the fit pairs every epoch), strict-improvement early stopping on the
             batch-weighted validation loss in fixed pair order, and the best-state copy are `ct2_d2.run`'s.
`dev_loss`   `ct2_d2.run`'s validation pass, shared by the trainer and the gate.
`gate_vi`    stage 0a(vi): c*'s seed-0 emotion checkpoint on c*'s full-scale ArtELingo validation inputs, teacher and
             `Gres` dev pairs gives c*'s `best_dev_loss`. Slow, GPU; the controller runs it.
"""
import copy
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import pf_paths
import ab_train
import ms_rows
import ms_train
import rb_rows
import rs_vh

ASPECTS = ms_train.ASPECTS                      # ('emotion', 'style', 'genre'): c*'s original construction order
RECIPE_NAME = 'R6-5'                            # c*'s switched loss (own caption term off), rb_rows.RECIPES
CKPT_KEYS = ('img.trunk.0.weight', 'img.trunk.0.bias', 'img.output.weight', 'img.output.bias',
             'cap.trunk.0.weight', 'cap.trunk.0.bias', 'cap.output.weight', 'cap.output.bias')


class NonFiniteLoss(Exception):
    """A train or validation loss that is not finite."""


def loss_recipe(teacher_weight):
    """(row, switches) of c*'s switched loss with the given direct-teacher weight (0.25: share 1/5; 1.0: share 1/2)."""
    recipe = rb_rows.recipe(RECIPE_NAME)
    row = copy.deepcopy(recipe['base'])
    row['distill_weight'] = float(teacher_weight)
    ms_rows.validate(row)
    return row, copy.deepcopy(recipe['switches'])


def build_emotion(device, seed=None):
    """c*'s emotion view `{'img','cap'}` and its model. With `seed`, the trainer's initializer: `np.random.seed`,
    `torch.manual_seed`, then `ms_train.build_model` in the original aspect order (emotion, style, genre) so that the
    caption trunk starts at the draws c*'s three-head trainer gave it; only the emotion view is kept and trained."""
    row, _ = loss_recipe(0.25)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    models, views = ms_train.build_model(row, list(ASPECTS), device)
    return models, views['emotion']


def checked_load(view, cstar_seed, root=None):
    """Load c*'s `emotion__s{cstar_seed}.pt` into `view` after checking its sha256 against the sidecar. Returns the
    sha256."""
    root = Path(pf_paths.CSTAR_WEIGHTS if root is None else root)
    path = root / f'emotion__s{cstar_seed}.pt'
    sidecar = json.loads(path.with_suffix('.json').read_text())
    sha = rs_vh.file_sha(path)
    assert sha == sidecar['weights_sha256'], f'{path}: sha256 differs from its sidecar'
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert tuple(state) == CKPT_KEYS, f'{path}: unexpected keys {list(state)}'
    view.load_state_dict(state)                          # strict
    return sha


def _tensors(part, device):
    """Fit or validation arrays as device tensors: img = I, cap = D (c*'s default inputs), teacher = T."""
    out = {}
    for name, key in (('img', 'I'), ('cap', 'D'), ('teacher', 'T')):
        x = part[key]
        assert x.ndim == 2 and x.shape[1] == 2048 and x.dtype == np.float32, (key, x.shape, x.dtype)
        out[name] = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    assert len({len(v) for v in out.values()}) == 1
    return out


def _check_pairs(pairs, n):
    assert pairs.dtype == np.int64 and pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) > 0
    assert (pairs[:, 0] >= 0).all() and (pairs[:, 0] < pairs[:, 1]).all() and (pairs[:, 1] < n).all()


def dev_loss(view, tensors, pairs, row, switches, batch_size=None):
    """`ct2_d2.run`'s validation pass: eval mode, no grad, pairs in the given (fixed) order in batches of
    `batch_size` (c*'s 512), weights 1, loss summed as `float(loss) * len(batch)` and divided by `len(pairs)`."""
    batch_size = ms_train.CONFIG['batch_size'] if batch_size is None else batch_size
    weights = np.ones(len(pairs), dtype=np.float32)
    view.eval()
    total = 0.
    with torch.no_grad():
        for a in range(0, len(pairs), batch_size):
            ps, ws = pairs[a:a + batch_size], weights[a:a + batch_size]
            loss = ab_train.switched_head_loss(view, **tensors, pairs=ps, weights=ws, row=row, aspect='emotion',
                                               include_mask_penalty=False, switches=switches)
            if not bool(torch.isfinite(loss)):
                raise NonFiniteLoss('validation')
            total += float(loss) * len(ps)
    return total / len(pairs)


def _atomic_json(path, obj):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(obj, indent=1, allow_nan=False))
    pending.replace(path)


def _fit(settings, cstar_seed, train_seed, fit, val, graph_pairs, device, max_epochs, cstar_root, record):
    """The training itself; fills `record` as it goes; returns (best_state, best_epoch, best_dev, history)."""
    row, switches = loss_recipe(settings['teacher_weight'])
    start = settings['start']
    assert start in ('cstar', 'random'), start
    fit_pairs, dev_pairs = graph_pairs
    n_fit, n_val = len(fit['I']), len(val['I'])
    _check_pairs(fit_pairs, n_fit)
    _check_pairs(dev_pairs, n_val)
    # ct2_d2.run: seed the global streams, a per-run generator, then build the model.
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    rng = np.random.default_rng(train_seed)
    models, view = build_emotion(device, seed=train_seed)        # c*'s initializer under the run's seed
    if start == 'cstar':
        record['cstar_weights_sha256'] = checked_load(view, cstar_seed, cstar_root)
    opt = torch.optim.AdamW(view.parameters(), lr=settings['lr'], weight_decay=pf_paths.RECIPE['weight_decay'])
    t_fit, t_dev = _tensors(fit, device), _tensors(val, device)
    batch = pf_paths.RECIPE['batch_pairs']
    assert batch == ms_train.CONFIG['batch_size'] and pf_paths.RECIPE['weight_decay'] == ms_train.CONFIG['weight_decay']
    patience = pf_paths.RECIPE['patience']
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
        dev = dev_loss(view, t_dev, dev_pairs, row, switches)
        history.append(dict(epoch=epoch, train_loss=total / len(fit_pairs), dev_loss=dev,
                            seconds=time.monotonic() - tick))
        if dev < best:
            best, best_epoch, stale = dev, epoch, 0
            best_state = copy.deepcopy(view.state_dict())
        else:
            stale += 1
        print(f'{record["run_id"]}: epoch={epoch} train={history[-1]["train_loss"]:.6f} dev={dev:.6f} '
              f'seconds={history[-1]["seconds"]:.2f}', flush=True)
        if stale >= patience:
            break
    return best_state, best_epoch, best, history


def train_run(run_id, settings, cstar_seed, train_seed, fit, val, graph, device, *, max_epochs=None,
              outputs=None, cstar_root=None):
    """One run (index: `OUTPUTS/runs/<run_id>/emotion.pt` and `run.json`); returns the `run.json` dict.

    `settings`: a `pf_paths.GRID` row dict (`start`, `lr`, `k`, `correction`, `teacher_weight`). `fit`, `val`: dicts
    with `keys` (list of `<source>/<image_name>`) and float32 unit arrays `I`, `D`, `T` ([n, 2048], `pf_data.photo_arrays`
    plus the keys; `E` is not used here, it only shapes the graph). `graph`: dict with int64 `fit_pairs`, `dev_pairs`
    (local rows) and optional `manifest_sha256`, or the path of the graph's `emotion.npz` (its `manifest.json` sits
    beside it). `max_epochs`: 200 (c*'s) unless a test lowers it.

    A non-finite train or validation loss (`failure_kind: nonfinite_loss`) or any other exception (`crash`) writes
    `run.json` with `status: failed`, the error, and no checkpoint. A finished run keeps the best epoch's weights.
    Refuses to overwrite an existing checkpoint."""
    outputs = Path(pf_paths.OUTPUTS if outputs is None else outputs)
    folder = outputs / 'runs' / run_id
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = folder / 'emotion.pt'
    assert not ckpt.exists(), f'{ckpt}: exists (the run is done; move it aside to rerun)'
    max_epochs = pf_paths.RECIPE['max_epochs'] if max_epochs is None else max_epochs
    assert 1 <= max_epochs <= pf_paths.RECIPE['max_epochs']
    assert set(settings) == set(pf_paths.COLUMNS), settings
    wall = time.monotonic()
    record = dict(run_id=run_id, status='failed', failure_kind=None, error=None, settings=dict(settings),
                  cstar_seed=cstar_seed, cstar_weights_sha256=None, train_seed=train_seed, weights_sha256=None,
                  best_epoch=None, best_dev_loss=None, history=[], graph_manifest_sha256=None,
                  fit_keys_sha256=rs_vh.names_sha(fit['keys']), val_keys_sha256=rs_vh.names_sha(val['keys']),
                  device=str(device), seconds=None)
    try:
        if isinstance(graph, (str, Path)):
            path = Path(graph)
            with np.load(path, allow_pickle=False) as z:
                graph = dict(fit_pairs=z['fit_pairs'], dev_pairs=z['dev_pairs'])
            graph['manifest_sha256'] = rs_vh.file_sha(path.with_name('manifest.json'))
        record['graph_manifest_sha256'] = graph.get('manifest_sha256')
        assert len(fit['keys']) == len(fit['I']) and len(val['keys']) == len(val['I'])
        best_state, best_epoch, best, history = _fit(settings, cstar_seed, train_seed, fit, val,
                                                     (graph['fit_pairs'], graph['dev_pairs']), torch.device(device),
                                                     max_epochs, cstar_root, record)
        assert best_state is not None and tuple(best_state) == CKPT_KEYS
        pending = ckpt.with_name('emotion.pending.pt')
        torch.save({k: v.detach().cpu() for k, v in best_state.items()}, pending)
        pending.replace(ckpt)
        record.update(status='ok', weights_sha256=rs_vh.file_sha(ckpt), best_epoch=best_epoch, best_dev_loss=best,
                      history=history)
    except NonFiniteLoss as e:
        record.update(failure_kind='nonfinite_loss', error=f'non-finite {e.args[0]} loss')
    except Exception as e:                                       # the caller decides what follows a crash
        record.update(failure_kind='crash', error=''.join(traceback.format_exception_only(type(e), e)).strip() +
                      ' | ' + ' <- '.join(f.name for f in reversed(traceback.extract_tb(e.__traceback__)[-4:])))
    record['seconds'] = time.monotonic() - wall
    _atomic_json(folder / 'run.json', record)
    return record


# --- stage 0a (vi) --------------------------------------------------------------------------------------------------

def gate_vi(outputs=None, device='cuda', cstar_seed=0):
    """Stage 0a(vi): c*'s seed-0 emotion checkpoint, c*'s full-scale ArtELingo validation inputs (default image and
    caption mean, instructed emotion image as teacher) and its `Gres` dev pairs (r10 `graphs/full/aspect__k40`) through
    this module's `dev_loss` give `BEST_DEV_LOSS_S0` within `TOL['vi']`. Also checks the three input arrays against
    the hashes in the checkpoint's sidecar and the graph archive against its manifest. Writes
    `OUTPUTS/stage0a/loss.json`; returns it. GPU for the 6,152 x 39,606-pair pass; the controller runs it."""
    import ct2_data                                                  # ArtELingo loaders (frozen, r10's own)
    wall = time.monotonic()
    outputs = Path(pf_paths.OUTPUTS if outputs is None else outputs)
    device = torch.device(device)
    weights = Path(pf_paths.CSTAR_WEIGHTS) / f'emotion__s{cstar_seed}.pt'
    sidecar = json.loads(weights.with_suffix('.json').read_text())
    s = ct2_data.split2('full')
    dev = s['dev']
    arrays = dict(I=ct2_data.default_images(s), D=ct2_data.default_captions(s),
                  T=ct2_data.instructed_images(s, 'emotion_focus'))
    recorded = sidecar['config']['data_sha256']
    input_ok = {}
    for name, key in (('I', 'img'), ('D', 'cap'), ('T', 'emotion.img')):
        input_ok[name] = rs_vh.array_sha(arrays[name]) == recorded[key]   # = ct2_d2._sha_array
    graph_dir = Path(pf_paths.R10_GRAPHS) / 'full/aspect__k40/d2/graphs'
    manifest = json.loads((graph_dir / 'manifest.json').read_text())
    archive = graph_dir / 'emotion.npz'
    graph_ok = rs_vh.file_sha(archive) == manifest['archives']['emotion']
    with np.load(archive, allow_pickle=False) as z:
        assert np.array_equal(z['dev_node_ids'], dev), 'dev node order differs from the split'
        dev_pairs = z['Gres__dev']
    _check_pairs(dev_pairs, len(dev))
    assert len(dev_pairs) == sidecar['config']['dev_pair_count']
    row, switches = loss_recipe(sidecar['config']['row']['distill_weight'])
    assert sidecar['config']['rb']['switches'] == switches
    _, view = build_emotion(device)
    sha = checked_load(view, cstar_seed)
    tensors = _tensors({k: v[dev] for k, v in arrays.items()}, device)
    value = dev_loss(view, tensors, dev_pairs, row, switches)
    reference = sidecar['best_dev_loss']
    assert cstar_seed != 0 or reference == pf_paths.BEST_DEV_LOSS_S0
    result = dict(gate='0a(vi)', value=value, reference=reference, abs_diff=abs(value - reference),
                  tol=pf_paths.TOL['vi'], inputs_match_sidecar=input_ok, graph_matches_manifest=graph_ok,
                  ok=bool(abs(value - reference) <= pf_paths.TOL['vi'] and all(input_ok.values()) and graph_ok),
                  cstar_seed=cstar_seed, cstar_weights_sha256=sha, n_dev_pairs=int(len(dev_pairs)),
                  device=str(device), seconds=time.monotonic() - wall)
    path = outputs / 'stage0a' / 'loss.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, result)
    return result
