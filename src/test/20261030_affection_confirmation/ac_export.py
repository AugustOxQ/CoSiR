"""Frozen exports of r12 (spec 2026-10-30 §3.2, §3.5 0a(iv) and 0a(v), §3.7): c* heads, PercepT core, Impressions.

  python ac_export.py cstar     [--split val|test|both] [--smoke N]      # CPU: 15 heads per split
  python ac_export.py impressions [--smoke N]                            # CPU: c* seed-0 emotion head, 1,442 images
  python ac_export.py reinfer   [--smoke N]                              # CPU, slow: stage 0a(iv)
  python ac_export.py percept   --device cuda [--split ...] [--smoke N]  # GPU: P3 seeds 0 to 4
  python ac_export.py percept-check --device cuda                        # GPU: stage 0a(v)

Writes under ac_paths.OUTPUTS: exports/cstar/{split}/{aspect}__s{seed}.npz (key img), exports/percept/{split}/P3__s{seed}.npz
(key prob), exports/cstar/impressions/emotion__s0.npz, each with a .json sidecar; stage0a/cstar_reinfer.json and
stage0a/percept_loader.json. An existing file is kept only when its content agrees exactly; nothing is overwritten.
Smoke runs (--smoke N: the first N non-FI keys) need an AC_OUTPUTS other than the default and write sidecars with
smoke=true, which ac_data rejects.
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ac_paths
import ac_data
import rs_vh
import rt_data
import rt_paths

SEEDS, ASPECTS = ac_paths.SEEDS, ac_paths.ASPECTS
AMI_PERCEPT_OWN = 0.08742922158584242          # r10's frozen own-reading emotion AMI (rt_stage0.AMI_REFERENCES)
CSTAR_WEIGHTS = rt_data.CSTAR_ROOT / 'weights/k11=40'
IMPRESSIONS_N = 1442


# ---- shared helpers ---------------------------------------------------------------------------------------------

def _outputs(smoke):
    out = Path(ac_paths.OUTPUTS)
    default = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs')
    assert smoke is None or out.resolve() != default.resolve(), 'smoke needs a separate AC_OUTPUTS'
    return out


def _write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')


def _publish(path, name, x, side):
    """Write array and sidecar; an existing pair must agree exactly (no overwrite, no half pairs)."""
    path = Path(path)
    meta = path.with_suffix('.json')
    text = json.dumps(side, indent=2, allow_nan=False) + '\n'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            same = set(z.files) == {name} and z[name].dtype == x.dtype and np.array_equal(z[name], x)
        if not same:
            raise FileExistsError(f'Refusing to overwrite {path}: different content')
    if meta.exists() and json.loads(meta.read_text()) != json.loads(text):
        raise FileExistsError(f'Refusing to overwrite {meta}: different content')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('xb') as handle:
            np.savez_compressed(handle, **{name: x})
    if not meta.exists():
        with meta.open('x') as handle:
            handle.write(text)
    return path


def _requested(split, smoke):
    """keys(split), or with smoke the first N non-FI keys (FI images enter the store only in the run chat)."""
    keys = ac_data.keys(split)
    if smoke is None:
        return keys
    return [k for k in keys if k.split('/', 1)[0] != ac_paths.FI_SOURCE][:smoke]


def default_images(requested):
    """The default Qwen image exactly as r11's re-inference feeds it (rb_encoders.full_images -> ct2_data.default_images):
    float16 store rows gathered by key, cast to float32, divided by the float32 row norm. Same operations as
    ac_data.embed('plain')."""
    return ac_data._unit(ac_data._plain(list(requested)))


def _held_sidecar(path):
    return json.loads(Path(path).read_text())


def _check_weights(aspect, seed):
    """Frozen c* weights: file hash equals the held sidecar's weights_sha256 (r11's 0b(iv) item)."""
    weights = CSTAR_WEIGHTS / f'{aspect}__s{seed}.pt'
    sha = rs_vh.file_sha(weights)
    ref = _held_sidecar(rt_data.CSTAR_ROOT / f'held/k11=40/{aspect}__s{seed}.json')['weights_sha256']
    assert sha == ref, f'c* weights hash {aspect} s{seed}'
    return weights, sha


def cstar_infer(x, aspect, seed):
    """c*'s CPU inference on float32 unit rows x: r10's rb_student.infer, float16 [n, 2048]."""
    import rb_student
    weights, sha = _check_weights(aspect, seed)
    out = rb_student.infer(weights, aspect, 2048, x, 'cpu')
    assert out.shape == (len(x), 2048) and out.dtype == np.float16
    return out, sha


# ---- c* exports -------------------------------------------------------------------------------------------------

def export_cstar(splits=('val', 'test'), smoke=None):
    out = _outputs(smoke)
    written = []
    for split in splits:
        requested = _requested(split, smoke)
        x = default_images(requested)
        digest = rs_vh.names_sha(requested)
        for aspect in ASPECTS:
            for seed in SEEDS:
                img, sha = cstar_infer(x, aspect, seed)
                side = dict(keys_sha256=digest, array_sha256=rs_vh.array_sha(img), weights_sha256=sha, split=split,
                            seed=seed, aspect=aspect, rows=len(requested), smoke=smoke is not None)
                written.append(_publish(out / 'exports/cstar' / split / f'{aspect}__s{seed}.npz', 'img', img, side))
    return written


def export_impressions(smoke=None):
    """c*'s seed-0 emotion head on the Impressions default store: all 1,442 images, keyed by file name."""
    out = _outputs(smoke)
    store = Path(ac_paths.IMPRESSIONS_STORE)
    index = json.loads((store / 'index.json').read_text())
    assert index['dataset'] == 'impressions' and index['backbone'] == 'qwen3vl_emb_2b' and not index['smoke']
    assert index['dtype'] == 'float16' and index['dim'] == 2048 and index['n_images'] == IMPRESSIONS_N
    files = [r['image'] for r in index['images']]
    assert len(files) == len(set(files)) == IMPRESSIONS_N and all(isinstance(f, str) and f for f in files)
    raw = np.load(store / 'img.npy', mmap_mode='r', allow_pickle=False)
    assert raw.shape == (IMPRESSIONS_N, 2048) and raw.dtype == np.float16
    names = files if smoke is None else files[:smoke]
    rows = np.asarray([files.index(n) for n in names], dtype=np.int64)     # by key, not by position
    x = np.asarray(raw[rows], dtype=np.float32)
    assert np.isfinite(x).all() and (np.linalg.norm(x, axis=1) > 0).all()
    x = ac_data._unit(x)                                                   # float32 division, as the default store path
    img, sha = cstar_infer(x, 'emotion', 0)
    side = dict(keys_sha256=rs_vh.names_sha(names), array_sha256=rs_vh.array_sha(img), weights_sha256=sha,
                split='impressions', seed=0, aspect='emotion', rows=len(names), keys=names, smoke=smoke is not None)
    return _publish(out / 'exports/cstar/impressions/emotion__s0.npz', 'img', img, side)


# ---- stage 0a(iv) -----------------------------------------------------------------------------------------------

def reinfer(smoke=None):
    """Re-infer ArtELingo validation outputs for all 15 heads from the plain Qwen image (r11's 0b(iv) path)."""
    out = _outputs(smoke)
    start = time.perf_counter()
    x = rt_data.embed('plain', 'val', 'emotion')
    saved_n = len(x) if smoke is None else smoke
    x = x[:saved_n]
    items = {}
    for aspect in ASPECTS:
        for seed in SEEDS:
            pred, sha = cstar_infer(x, aspect, seed)
            saved = rt_data.raw('cstar', 'val', aspect, seed)[:saved_n]
            assert pred.shape == saved.shape and pred.dtype == saved.dtype == np.float16
            diff = float(np.max(np.abs(pred.astype(np.float32) - saved.astype(np.float32))))
            items[f'{aspect}/s{seed}'] = dict(value=diff, reference=0.0, tolerance=1e-3, ok=bool(diff <= 1e-3),
                                              weights_sha256=sha)
    doc = dict(items=items, rows=saved_n, smoke=smoke, seconds=time.perf_counter() - start,
               ok=all(i['ok'] for i in items.values()))
    _write_json(out / 'stage0a/cstar_reinfer.json', doc)
    print(f'0a(iv): {"PASS" if doc["ok"] else "FAIL"}; max diff {max(i["value"] for i in items.values()):.3g}')
    return 0 if doc['ok'] else 1


# ---- PercepT core -----------------------------------------------------------------------------------------------

def _percept_paths(seed):
    base = rt_paths.PERCEPT_OUT
    source = base / f'percept/core/stage2/k40/P3__s{seed}.npz'
    held = base / f'held/percept/core__k40/P3__s{seed}.json'
    return source, held


def percept_infer(seed, x, device):
    """Frozen full-scale P3 mapper on float32 unit rows x [n, 2048]; returns (prob float32 [n, 40], weights_sha256).

    Same steps as ct_percept.infer_stage2 after its image load (second L2 division included), but the images come from
    the caller. `device` must equal the mapper's recorded training device ('cuda')."""
    import torch
    import ct_percept
    source, held = _percept_paths(seed)
    meta = json.loads(source.with_suffix('.json').read_text())
    comparator = _held_sidecar(held)
    config = meta['config']
    assert (config['settings'], config['cut'], config['arm'], config['seed']) == ('core', 'k40', 'P3', seed)
    assert not config['smoke'] and meta['K'] == 40 and config['input_dim'] == 2048
    assert config['device'] == comparator['inference_device'] == 'cuda', 'recorded training device'
    assert str(device) == config['device'], 'inference device differs from training device'
    with patch.object(ct_percept, 'OUTPUTS', rt_paths.PERCEPT_OUT):
        assert ct_percept.stage2_path('core', 'k40', 'P3', seed) == source
        ct_percept.verify_output(source)
        sha = rs_vh.file_sha(source.with_suffix('.pt'))
        assert sha == meta['weights_sha256'] == comparator['weights_sha256'], 'held mapper weights hash'
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == 2048 and x.dtype == np.float32 and np.isfinite(x).all()
        ct_percept.determinism(seed)
        mapper = ct_percept.build_mapper(config, meta['K'])
        state = torch.load(source.with_suffix('.pt'), map_location='cpu', weights_only=True)
        mapper.load_state_dict(state['mapper'])
        mapper.to(device).eval()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)     # core image_inputs normalizes again
        with torch.no_grad():
            prob = mapper(torch.from_numpy(np.ascontiguousarray(x)).to(device)).sigmoid().cpu().numpy()
    assert prob.shape == (len(x), 40) and prob.dtype == np.float32
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    return prob, sha


def export_percept(device, splits=('val', 'test'), smoke=None):
    out = _outputs(smoke)
    written = []
    for split in splits:
        requested = _requested(split, smoke)
        x = default_images(requested)
        digest = rs_vh.names_sha(requested)
        for seed in SEEDS:
            prob, sha = percept_infer(seed, x, device)
            side = dict(keys_sha256=digest, array_sha256=rs_vh.array_sha(prob), weights_sha256=sha, split=split,
                        seed=seed, device=str(device), rows=len(requested), smoke=smoke is not None)
            written.append(_publish(out / 'exports/percept' / split / f'P3__s{seed}.npz', 'prob', prob, side))
    return written


def _own_ami_task(args):
    prob, labels = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return rs_vh.own_partition(prob)


def percept_check(device, workers=5):
    """Stage 0a(v): the loader on ArtELingo held default images gives r11's held P3 arrays (hash, all five seeds;
    the gate is seed 0) and own-reading emotion AMI (r10's frozen functions, as rt_stage0.held_ami)."""
    import ct_held
    out = _outputs(None)
    start = time.perf_counter()
    names = rt_data.names('held')
    x = ct_held.qwen_default_images(names)        # the image path of ct_percept.infer_stage2
    probs, hashes = {}, {}
    for seed in SEEDS:
        probs[seed], _ = percept_infer(seed, x, device)
        held_side = _held_sidecar(_percept_paths(seed)[1])
        hashes[seed] = dict(value=rs_vh.array_sha(probs[seed]), reference=held_side['array_sha256s']['prob'])
        hashes[seed]['ok'] = hashes[seed]['value'] == hashes[seed]['reference']
    labels = rt_data.labels('held')['emotion']
    with mp.get_context('fork').Pool(workers) as pool:
        parts = dict(zip(SEEDS, pool.map(_own_ami_task, [(probs[s], None) for s in SEEDS])))
    table = rs_vh.point_table(labels, {'percept': {s: {'own': p} for s, p in parts.items()}})
    ami = rs_vh._mean_over(rs_vh._point_val(table), 'percept', sorted(parts), 'own')
    doc = dict(device=str(device), seeds=hashes, ami=dict(value=ami, reference=AMI_PERCEPT_OWN, tolerance=1e-9,
                                                         ok=bool(abs(ami - AMI_PERCEPT_OWN) <= 1e-9)),
               seconds=time.perf_counter() - start)
    doc['ok'] = hashes[0]['ok'] and doc['ami']['ok']
    _write_json(out / 'stage0a/percept_loader.json', doc)
    print(f'0a(v): {"PASS" if doc["ok"] else "FAIL"}; seed-0 hash {hashes[0]["ok"]}; AMI {ami!r}; '
          f'hashes ok {[hashes[s]["ok"] for s in SEEDS]}')
    return 0 if doc['ok'] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('part', choices=('cstar', 'impressions', 'reinfer', 'percept', 'percept-check'))
    p.add_argument('--split', choices=('val', 'test', 'both'), default='both')
    p.add_argument('--device', help="PercepT only; must equal the mapper's training device ('cuda')")
    p.add_argument('--smoke', type=int)
    a = p.parse_args(argv)
    splits = ac_paths.SPLITS if a.split == 'both' else (a.split,)
    if a.part in ('percept', 'percept-check'):
        assert a.device, '--device is required for PercepT parts'
    if a.part == 'cstar':
        print(*export_cstar(splits, a.smoke), sep='\n')
    elif a.part == 'impressions':
        print(export_impressions(a.smoke))
    elif a.part == 'reinfer':
        return reinfer(a.smoke)
    elif a.part == 'percept':
        print(*export_percept(a.device, splits, a.smoke), sep='\n')
    else:
        return percept_check(a.device)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
