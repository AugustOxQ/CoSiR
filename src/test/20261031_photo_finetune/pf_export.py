"""Photo-model inference, exports and their loader (spec §3.4 exports, §3.5 stage 0a(iii), C6).

`infer`      `rb_student.infer`'s inference (eval, no grad, batches of 512, CPU, float16 cast) on a state dict instead of
             a file: the emotion view built by `pf_train.build_emotion`, image side only.
`export`     run `OUTPUTS/runs/<run_id>/emotion.pt` on the default image of `pf_data.keys(split)` (as
             `ac_export.default_images`) -> `OUTPUTS/exports/<run_id>/<split>/emotion.npz` (+ `.json`). Existing files
             must agree exactly (`ac_export._publish`'s refuse-on-conflict).
`stored`     the checked float16 rows as saved (clustering input, as `ac_data.stored`).
`embed`      the same rows as unit float32 (retrieval input, as `ac_data.embed`).
`gate_iii`   stage 0a(iii): c*'s emotion checkpoints s0 to s4 through `infer` give r12's c* exports within 1e-3. This is
             the one place r13 checks c*'s hashes against c*'s held sidecars (`ac_export._check_weights`).
The loader checks the key-list hash, the array hash, and that the sidecar's `weights_sha256` equals both the run's
`run.json.weights_sha256` and the checkpoint file's own hash.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import pf_paths
import pf_data
import pf_train
import ac_export
import ac_data
import rs_vh

BATCH = 512                                         # c*'s inference batch (rb_student.infer: ms_train.CONFIG batch)
SIDECAR_FIELDS = {'keys_sha256', 'array_sha256', 'weights_sha256', 'run_id', 'split', 'smoke'}


def infer(state, x):
    """float16 [n, 2048] emotion image-side output of the checkpoint `state` (keys `pf_train.CKPT_KEYS`) on float32
    rows `x`. Same procedure as `rb_student.infer` (CPU, eval, no grad, batches of 512)."""
    assert tuple(state) == pf_train.CKPT_KEYS, list(state)
    x = np.asarray(x)
    assert x.ndim == 2 and x.shape[1] == 2048 and x.dtype == np.float32 and np.isfinite(x).all()
    torch.set_num_threads(8)
    device = torch.device('cpu')
    _, view = pf_train.build_emotion(device)
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


def _outputs(outputs):
    return Path(pf_paths.OUTPUTS if outputs is None else outputs)


def _run(run_id, outputs):
    """(checkpoint path, checked sha256): the run finished ok and its file hash is `run.json.weights_sha256`."""
    folder = _outputs(outputs) / 'runs' / run_id
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'ok', f'{run_id}: run status {run["status"]}'
    ckpt = folder / 'emotion.pt'
    sha = rs_vh.file_sha(ckpt)
    assert sha == run['weights_sha256'], f'{run_id}: checkpoint hash differs from run.json'
    return ckpt, sha


def _path(run_id, split, outputs):
    assert split in ('val', 'test'), split
    return _outputs(outputs) / 'exports' / run_id / split / 'emotion.npz'


def export(run_id, split, *, outputs=None):
    """Run `run_id`'s checkpoint on `pf_data.keys(split)` (val or test); write the export and return its path."""
    ckpt, sha = _run(run_id, outputs)
    requested = pf_data.keys(split)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    return _write(run_id, split, requested, infer(state, pf_data.default_images(requested)), sha, outputs)


def _write(run_id, split, requested, img, sha, outputs):
    assert img.shape == (len(requested), 2048) and img.dtype == np.float16
    side = dict(keys_sha256=rs_vh.names_sha(requested), array_sha256=rs_vh.array_sha(img), weights_sha256=sha,
                run_id=run_id, split=split, smoke=pf_paths.SMOKE > 0)
    return ac_export._publish(_path(run_id, split, outputs), 'img', img, side)


def stored(run_id, split, *, outputs=None):
    """Checked float16 export rows in `pf_data.keys(split)` order, without norm repair (clustering input)."""
    path = _path(run_id, split, outputs)
    requested = pf_data.keys(split)
    side = json.loads(path.with_suffix('.json').read_text())
    assert set(side) == SIDECAR_FIELDS, sorted(side)
    # Spec §3.3 (C6): the positional export must attest to the full key list.
    assert side['keys_sha256'] == rs_vh.names_sha(requested), 'export keys hash'
    assert side['run_id'] == run_id and side['split'] == split and side['smoke'] is (pf_paths.SMOKE > 0), \
        'export identity'
    _, sha = _run(run_id, outputs)
    assert side['weights_sha256'] == sha, 'export weights hash differs from the run'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img'}, 'export array keys'
        x = ac_data._array(z['img'], len(requested), 2048, np.float16, unit=True)
    assert side['array_sha256'] == rs_vh.array_sha(x), 'export array hash'
    return x


def embed(run_id, split, *, outputs=None):
    """Unit float32 rows in `pf_data.keys(split)` order (retrieval input)."""
    return ac_data._unit(stored(run_id, split, outputs=outputs))


def gate_iii(seeds=pf_paths.SEEDS, splits=('val', 'test'), outputs=None):
    """Stage 0a(iii), CPU. c*'s emotion checkpoint of each seed (hash against c*'s held sidecar) through `infer` on the
    default images; max abs difference to r12's `exports/cstar/<split>/emotion__s<seed>.npz` per array. Writes
    `OUTPUTS/stage0a/inference.json` and returns its dict; `ok` means every difference is within `TOL['iii']`."""
    start = time.monotonic()
    diffs = {}
    for split in splits:
        requested = pf_data.keys(split)
        x = pf_data.default_images(requested)
        for s in seeds:
            weights, sha = ac_export._check_weights('emotion', s)
            state = torch.load(weights, map_location='cpu', weights_only=True)
            out = infer(state, x)
            ref = pf_paths.R12_OUT / 'exports/cstar' / split / f'emotion__s{s}.npz'
            side = json.loads(ref.with_suffix('.json').read_text())
            assert side['weights_sha256'] == sha and side['keys_sha256'] == rs_vh.names_sha(requested)
            with np.load(ref, allow_pickle=False) as z:
                assert set(z.files) == {'img'}
                r12 = z['img']
            assert r12.shape == out.shape and r12.dtype == out.dtype
            diffs[f'{split}/emotion__s{s}'] = float(np.max(np.abs(out.astype(np.float64) - r12.astype(np.float64))))
    tol = pf_paths.TOL['iii']
    result = dict(max_abs_diff=diffs, tol=tol, ok=bool(all(d <= tol for d in diffs.values())),
                  seconds=time.monotonic() - start)
    folder = _outputs(outputs) / 'stage0a'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'inference.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
