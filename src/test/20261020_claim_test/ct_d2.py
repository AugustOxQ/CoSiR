"""Frozen D2 cohorts with per-aspect best weights and image inference."""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261019_model_sheet'))
import ms_train
from ms_rows import ASPECTS, ROWS
import ct_held

OUTPUTS = HERE / 'outputs'
SAMPLE = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')


def _weight_path(outdir, aspect, seed):
    assert aspect in ASPECTS and seed in range(5)
    return Path(outdir) / 'd2/weights' / f'{aspect}__s{seed}.pt'


def _artifact_info(path, kind):
    info = json.loads(path.with_suffix('.json').read_text())
    assert ms_train.head.sha_file(path) == info[f'{kind}_sha256'], f'{kind} checksum mismatch'
    assert isinstance(info['config'], dict)
    return info


def _weights_info(outdir, aspect, seed):
    info = _artifact_info(_weight_path(outdir, aspect, seed), 'weights')
    config = info['config']
    assert config['aspect'] == aspect and config['seed'] == seed
    assert config['row'] == ROWS['D2'] and config['name'] == 'D2'
    assert config['active_aspects'] == list(ASPECTS) and not config['smoke']
    assert config['max_epochs'] == ms_train.CONFIG['max_epochs']
    return info


def _verified_cohort(s, manifests, seed, device, outdir):
    result, incomplete = {}, False
    corrupt = (AssertionError, OSError, ValueError, KeyError, zipfile.BadZipFile)
    for aspect in ASPECTS:
        config = ms_train.make_config(s, ROWS['D2'], 'D2', aspect, list(ASPECTS),
                                      manifests, seed, device, False, ms_train.CONFIG['max_epochs'])
        path = ms_train.output_path(outdir / 'd2/train', 'D2', aspect,
                                    config['graph_row'], seed)
        records = {}
        for kind, artifact in (('output', path), ('weights', _weight_path(outdir, aspect, seed))):
            try:
                info = _artifact_info(artifact, kind)
            except corrupt:
                incomplete = True
                continue
            # A valid artifact from another recipe/device must never be replaced.
            assert info['config'] == config, f'{kind} resume config mismatch'
            records[kind] = info
        if len(records) != 2:
            continue
        try:
            output = ms_train.head.verify_output(path, config)
            weights = records['weights']
            for key in ('output_sha256', 'best_epoch', 'best_dev_loss'):
                assert weights[key] == output[key], f'weights/output {key} mismatch'
            result[aspect] = output
        except corrupt:
            incomplete = True
    return None if incomplete else result


def _preserve_cohort(outdir, seed):
    """Move incomplete artifacts aside so the unchanged sheet cannot skip or reject them."""
    base = outdir / 'd2'
    backup = base / 'replaced' / f's{seed}-{time.time_ns()}'
    for aspect in ASPECTS:
        output = ms_train.output_path(base / 'train', 'D2', aspect, ROWS['D2']['graphs'][aspect], seed)
        weight = _weight_path(outdir, aspect, seed)
        for path in (output, output.with_suffix('.json'), weight, weight.with_suffix('.json')):
            if path.exists():
                target = backup / path.relative_to(base)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)


def load_training_data():
    """Use the sheet's setup and the main checkout's frozen sample and graphs."""
    with np.load(SAMPLE, allow_pickle=False) as z:
        s = {key: z[key] for key in z.files}
    assert (len(s['sample_ids']), len(s['first']), len(s['subset_pid'])) == (99999, 19889, 5000)
    assert len(np.unique(s['sample_ids'])) == 99999
    assert len(np.unique(s['subset_pid'])) == 5000
    assert np.array_equal(s['all_painting_ids'][s['pid']], s['painting_ids'])
    assert np.array_equal(s['all_painting_ids'][s['subset_pid']], s['subset_painting_ids'])
    ms_train.disjoint_split(s)
    inputs = ms_train.head.load_inputs(s, ASPECTS[0], 'default')
    graphs, manifests = {}, {}
    for aspect in ASPECTS:
        graphs[aspect], manifests[aspect] = ms_train.head.load_graph(
            s, ms_train.GRAPH_BASE, aspect, [ROWS['D2']['graphs'][aspect]])
    return s, {a: inputs for a in ASPECTS}, graphs, manifests


@threadpool_limits.wrap(limits=8)
def train(seeds=range(5), device='cuda', outdir=OUTPUTS):
    """Train one unchanged three-aspect D2 cohort per seed, retaining best states."""
    seeds = list(seeds)
    assert seeds and len(set(seeds)) == len(seeds) and all(s in range(5) for s in seeds)
    device, outdir = torch.device(device), Path(outdir)
    torch.set_num_threads(8)
    s, inputs, graphs, manifests = load_training_data()
    original_save = ms_train.save_output

    def save_weights(s, view, inputs, state, path, config, start):
        info = original_save(s, view, inputs, state, path, config, start)
        weight = _weight_path(outdir, config['aspect'], config['seed'])
        weight.parent.mkdir(parents=True, exist_ok=True)
        pending = weight.with_suffix('.pending.pt')
        torch.save({key: value.detach().cpu() for key, value in state['best_state'].items()}, pending)
        pending.replace(weight)
        ms_train.head.atomic_json(weight.with_suffix('.json'), dict(
            weights_sha256=ms_train.head.sha_file(weight), output_sha256=info['output_sha256'],
            config=config, best_epoch=info['best_epoch'], best_dev_loss=info['best_dev_loss']))
        return info

    result = {}
    with patch.object(ms_train, 'save_output', save_weights):
        for seed in seeds:
            verified = _verified_cohort(s, manifests, seed, device, outdir)
            if verified is not None:
                result[seed] = verified
                continue
            _preserve_cohort(outdir, seed)
            result[seed] = ms_train.run(s, inputs, graphs, manifests, ROWS['D2'], 'D2',
                                       list(ASPECTS), seed, device, outdir / 'd2/train')
    return result


def _inference_device(weights, device):
    recorded = torch.device(weights['config']['device'])
    assert device is None or torch.device(device) == recorded, 'inference device differs from training device'
    return recorded


@threadpool_limits.wrap(limits=8)
def infer(aspect, seed, names, side='img', *, device=None, outdir=OUTPUTS):
    """Apply one saved best view to default images in the requested name order."""
    assert side == 'img', 'held inference requires the image tower'
    path = _weight_path(outdir, aspect, seed)
    weights = _weights_info(outdir, aspect, seed)
    torch.set_num_threads(8)
    device = _inference_device(weights, device)
    _, views = ms_train.build_model(ROWS['D2'], list(ASPECTS), device)
    view = views[aspect]
    view.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    view.eval()
    x = ct_held.qwen_default_images(names)
    assert x.shape == (len(names), 2048) and x.dtype == np.float32 and np.isfinite(x).all()
    out = np.empty((len(names), 2048), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(names), ms_train.CONFIG['batch_size']):
            xx = torch.from_numpy(np.ascontiguousarray(x[start:start + ms_train.CONFIG['batch_size']])).to(device)
            out[start:start + len(xx)] = view[side](xx).cpu().numpy().astype(np.float16)
    assert np.isfinite(out).all()
    if len(out):
        assert np.max(np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1)) < 1e-3
    return out


def write_held(seeds=range(5), device=None, outdir=OUTPUTS):
    """Write all three best image views in the frozen held-list order."""
    seeds = list(seeds)
    assert seeds and len(set(seeds)) == len(seeds) and all(s in range(5) for s in seeds)
    outdir = Path(outdir)
    held_list = outdir / 'held/held_paintings.json'
    if not held_list.exists():
        ct_held.write_held_list(held_list)
    names, held_hash = ct_held.read_held_list(held_list)
    result = {}
    for seed in seeds:
        for aspect in ASPECTS:
            weights = _weights_info(outdir, aspect, seed)
            _inference_device(weights, device)
            path = outdir / 'held/d2' / f'{aspect}__s{seed}.npz'
            identity = dict(held_sha256=held_hash, weights_sha256=weights['weights_sha256'], rows=len(names))
            if path.exists():
                info = json.loads(path.with_suffix('.json').read_text())
                assert all(info[key] == value for key, value in identity.items()), 'held identity mismatch'
                assert ms_train.head.sha_file(path) == info['output_sha256'], 'held output checksum mismatch'
                with np.load(path, allow_pickle=False) as z:
                    assert z.files == ['img']
                    x = z['img']
                    assert x.shape == (len(names), 2048) and x.dtype == np.float16
                    assert np.isfinite(x).all()
                    assert np.max(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1)) < 1e-3
            else:
                x = infer(aspect, seed, names, device=device, outdir=outdir)
                path.parent.mkdir(parents=True, exist_ok=True)
                pending = path.with_suffix('.pending.npz')
                np.savez(pending, img=x)
                info = dict(identity, output_sha256=ms_train.head.sha_file(pending))
                ms_train.head.atomic_json(path.with_suffix('.json'), info)
                pending.replace(path)
            result[aspect, seed] = info
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    p.add_argument('--device', choices=['cpu', 'cuda'],
                   help='training defaults to cuda; held inference uses the weights device')
    p.add_argument('--train', action='store_true')
    p.add_argument('--held', action='store_true', help='held inference; run after the stage 0 gate')
    p.add_argument('--outdir', type=Path, default=OUTPUTS)
    args = p.parse_args(argv)
    assert args.train or args.held, 'choose --train and/or --held'
    assert len(set(args.seeds)) == len(args.seeds)
    if args.held:
        import ct_stage0
        ct_stage0.require_pass()  # V1: no held inference without a Stage 0 pass
    if args.train:
        train(args.seeds, args.device or 'cuda', args.outdir)
    if args.held:
        write_held(args.seeds, args.device, args.outdir)


if __name__ == '__main__':
    main()
