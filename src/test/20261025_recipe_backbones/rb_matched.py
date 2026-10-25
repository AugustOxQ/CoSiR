"""Matched arm (M2, M3, M5) and the CT's B1 and B6 on ArtELingo-28 (rule §2 "M2, M3, M5", stage 5; spec §3.2, D22).

A matched arm is PercepT core P3 (K = 40, the MLP Linear(w, 1024), ReLU, Linear(1024, 40), 100 epochs of full-batch
Adam lr .003 on the mean BCE) trained on a candidate's encoder image with the CT's saved Qwen Stage 1 topics of the
same seed and scale (no Stage 1 rerun). Only the input width differs from P3. Training and inference call the CT's own
code (`ct2_percept.mapper_config`, `caption_targets`, `ct_percept.train_mapper_with_state`, `build_mapper`); the image
provider is the only new piece. Outputs, relative to the outputs folder:
  part2/<scale>/matched/<M>/P3__s<seed>.{npz,json,pt}   Stage 2 mapper (the CT's writer)
  part2/<scale>/held/<M>/P3__s<seed>.npz                prob float32 [12281, 40], held list order
  part2/<scale>/a28/<M|B1|B6>/P<n>__s<seed>.npz         prob float32 [746, 40], `a28_names()` order
each with a sidecar .json (index "Output sidecars"). Regressions: `--regress stage2|stage3a` (see `regress`).
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import rb_paths
import ct_held
import ct_percept as base
import ct2_data
import ct2_percept
import pc_sept
import rb_encoders
import rb_score

MATCHED = {'M2': 'siglip2_so400m', 'M3': 'clip_b32', 'M5': 'dinov2_large'}
CANDIDATE = {'M2': 'K2', 'M3': 'K3', 'M5': 'K5'}
CT_ARMS = {'B1': (3, 'qwen3vl_emb_2b'), 'B6': (2, 'clip_b32')}   # B1 = core P3, B6 = core P2: arm number, A28 store
REGRESSION = {'stage2': 'sample', 'stage3a': 'full'}
SEEDS = list(base.SEEDS)
QWEN = 'qwen3vl_emb_2b'


@dataclass
class World:
    """What a scale needs from the frozen CT and the stores; tests build a tiny one."""
    scale: str
    ct_root: Path                 # CT_OUT or CT_FULL
    s: dict                       # the CT's split at this scale (`ct2_data.split2`)
    topics: Callable              # seed -> (topic int16 per caption, k, sha256 of the Stage 1 file)
    held_names: list
    held_sha256: str
    full_images: Callable         # (backbone, names) -> float32 [n, w], unit norm
    a28_names: list
    a28_sha256: str
    a28_images: Callable          # backbone -> float32 [746, w], unit norm
    clip_images: Callable         # names -> the CT's P2 input for those paintings (regression only)


def stage1_topics(s, ct_root, seed):
    """The CT's saved Qwen Stage 1 of this seed at this scale, checked as the CT does; returns topic, k, file sha256."""
    with ct2_percept.runtime(ct_root, s) as view:
        path = base.stage1_path('core', 'qwen_qwen', seed)
        meta = base.verify_output(path)
        assert meta['config']['version'] == 'qwen_qwen' and meta['config']['seed'] == seed
        assert not meta['config']['smoke'], 'smoke Stage 1'
        topic, k = base.load_stage1(view, path, 'qwen_qwen', seed)
        return topic, k, base.file_sha256(path)


def load_world(scale):
    assert scale in ('sample', 'full'), scale
    ct_root = rb_score.CT_ROOT[scale]
    s = ct2_data.split2(scale)
    names, sha = ct_held.read_held_list(rb_paths.CT_OUT / 'held/held_paintings.json')
    assert sha == rb_paths.HELD_SHA256
    return World(scale=scale, ct_root=ct_root, s=s, topics=lambda seed: stage1_topics(s, ct_root, seed),
                 held_names=names, held_sha256=sha, full_images=rb_encoders.full_images,
                 a28_names=rb_encoders.a28_names(), a28_sha256=rb_encoders.a28_sha256(),
                 a28_images=rb_encoders.a28_images, clip_images=ct_held.clip_global_images)


# ---- Stage 2 --------------------------------------------------------------------------------------------------

def image_inputs(all_images, ids, device):
    """pc_stage2.image_inputs for a global vector: divide by the float32 norm, then join by painting row."""
    norms = np.linalg.norm(all_images, axis=1, keepdims=True)
    assert (norms > 0).all()
    x = all_images / norms
    tensors = {}
    for part, pids in ids.items():
        rows = x[pids]
        assert rows.shape == (len(pids), x.shape[1]) and rows.dtype == np.float32 and np.isfinite(rows).all()
        tensors[part] = torch.from_numpy(np.ascontiguousarray(rows)).to(device)
    return tensors


def train_stage2(s, topic, k, all_images, backbone, seed, device, smoke, epochs=None, arm='P3'):
    """Train one mapper of arm P3 (P2 only for test stand-ins) on `all_images` (one row per painting of `s`, unit norm
    float32). `epochs` shortens the run (smoke and tests); None keeps P3's 100. Returns arrays, state, config, meta."""
    assert all_images.dtype == np.float32 and all_images.shape[0] == len(s['all_painting_ids']) == len(s['first'])
    assert all_images.ndim == 2 and np.isfinite(all_images).all()
    start = time.perf_counter()
    config = ct2_percept.mapper_config(s, arm, seed, smoke, device)
    config.update(settings='core', cut='k40', input_dim=int(all_images.shape[1]))
    if arm == 'P3':
        config['input'] = f'{backbone} default image'
    if epochs is not None:
        config['epochs'] = epochs
    base.determinism(seed)
    ids, targets = ct2_percept.caption_targets(s, topic, k, smoke)
    with threadpool_limits(limits=8):
        inputs = image_inputs(all_images, ids, device)
        eval_prob, dev_prob, history, epoch_seconds, state = base.train_mapper_with_state(inputs, targets, config, k)
    for part, prob in (('eval', eval_prob), ('dev', dev_prob)):
        assert prob.shape == (len(ids[part]), k) and prob.dtype == np.float32
        assert np.isfinite(prob).all() and base.in_unit_interval(prob)
    meta = dict(config=config, K=k, eval_names_sha256=ct_held.names_sha256(s['all_painting_ids'][ids['eval']].tolist()),
                dev_auc=None, dev_valid_topics=0, loss_history=history, epoch_seconds=epoch_seconds)
    positives = targets['dev'].sum(axis=0)
    if ((positives > 0) & (positives < len(targets['dev']))).any():
        aucs, skipped = pc_sept.evaluate_auc(dev_prob, targets['dev'], lambda line: print(line, flush=True),
                                             arm + ' dev')
        meta.update(dev_auc=pc_sept.auc_summary(aucs)['macro'], dev_valid_topics=len(aucs), dev_skipped_topics=skipped)
    else:
        meta.update(dev_auc_note='No dev topic has both positive and negative labels',
                    dev_skipped_topics=list(range(k)))
    meta['seconds'] = time.perf_counter() - start
    return dict(eval_prob=eval_prob, dev_prob=dev_prob, state=state, config=config, targets=targets, ids=ids,
                k=k, meta=meta)


def save_stage2(path, result, extra=None):
    """Seal the mapper as the CT does (arrays, CPU weights, hash sidecar) and verify it by reading it back."""
    path = Path(path)
    meta = dict(result['meta'], **(extra or {}))
    base.save_output(path, dict(eval_prob=result['eval_prob']), dict(mapper=result['state']), meta)
    base.verify_output(path)
    return path


def renormalizes(input_kind):
    """P3-style mappers see images divided by the norm twice (the store loader, then `image_inputs`); P2 once."""
    return input_kind != 'clip_image'


def infer(path, images, device=None):
    """float32 sigmoid probabilities of a saved Stage 2 mapper, one batch in caller order, on its training device.

    `images` are the store loader's rows (unit norm float32, width = the mapper's input width)."""
    meta = base.verify_output(path)
    config = meta['config']
    device = config['device'] if device is None else device
    assert str(device) == config['device'], f'inference device {device} differs from training device {config["device"]}'
    assert images.dtype == np.float32 and images.shape == (len(images), config['input_dim']) and len(images) > 0
    assert np.isfinite(images).all()
    x = images / np.linalg.norm(images, axis=1, keepdims=True) if renormalizes(config['input']) else images
    base.determinism(config['seed'])
    with threadpool_limits(limits=8):
        mapper = base.build_mapper(config, meta['K'])
        state = torch.load(Path(path).with_suffix('.pt'), map_location='cpu', weights_only=True)
        mapper.load_state_dict(state['mapper'])
        mapper.to(device).eval()
        with torch.no_grad():
            prob = mapper(torch.from_numpy(np.ascontiguousarray(x)).to(device)).sigmoid().cpu().numpy()
    assert prob.shape == (len(images), meta['K']) and prob.dtype == np.float32
    assert np.isfinite(prob).all() and base.in_unit_interval(prob)
    return prob


# ---- outputs --------------------------------------------------------------------------------------------------

def _code():
    return {name: rb_score.sha_file(Path(__file__).with_name(name)) for name in ('rb_matched.py', 'rb_encoders.py')}


def write_output(path, prob, *, list_sha256, scale, arm, seed, extra=None):
    """Save `prob` (key `prob`) and its sidecar (index "Output sidecars"); pending file first."""
    path = Path(path)
    assert prob.ndim == 2 and prob.dtype == np.float32 and prob.shape[0] > 0, (prob.shape, prob.dtype)
    assert np.isfinite(prob).all() and base.in_unit_interval(prob), 'probabilities outside [0, 1] or nonfinite'
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, prob=prob)
    pending.replace(path)
    meta = dict(list_sha256=list_sha256, rows=int(prob.shape[0]), width=int(prob.shape[1]), dtype=str(prob.dtype),
                sha256=rb_score.array_sha256(prob), scale=scale, arm=arm, seed=seed, aspect='percept',
                code_sha256=_code(), **(extra or {}))
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    pending.replace(path.with_suffix('.json'))
    return path


def stage2_path(outputs, scale, arm, seed):
    return Path(outputs) / f'part2/{scale}/matched/{arm}/P3__s{seed}.npz'


def a28_path(outputs, scale, arm, seed):
    number = 3 if arm in MATCHED else CT_ARMS[arm][0]
    return Path(outputs) / f'part2/{scale}/a28/{arm}/P{number}__s{seed}.npz'


def _gate(outputs, scale, candidates):
    """V1: held and ArtELingo-28 outputs of these arms are computed only after the scale's regression passed."""
    rb_score.require_gate(Path(outputs), 3 if scale == 'sample' else 4, list(candidates))


def run_matched(world, arm, seeds, outputs, device, smoke, epochs=None, gate=True):
    """Train `arm` (M2, M3 or M5) for each seed at the world's scale; write held and ArtELingo-28 outputs."""
    backbone = MATCHED[arm]
    if gate:
        _gate(outputs, world.scale, [CANDIDATE[arm]])
    s = world.s
    all_images = world.full_images(backbone, s['all_painting_ids'])
    held_images = world.full_images(backbone, world.held_names)
    a28_images = world.a28_images(backbone)
    paths = []
    for seed in seeds:
        topic, k, stage1_sha256 = world.topics(seed)
        result = train_stage2(s, topic, k, all_images, backbone, seed, device, smoke, epochs=epochs)
        path = save_stage2(stage2_path(outputs, world.scale, arm, seed), result,
                           dict(stage1_sha256=stage1_sha256, backbone=backbone, arm=arm, code_sha256=_code()))
        weights = base.verify_output(path)['weights_sha256']
        extra = dict(backbone=backbone, stage2_weights_sha256=weights, stage1_sha256=stage1_sha256, smoke=smoke)
        held_path = rb_score.output_path(world.scale, arm, seed, outputs=outputs)
        write_output(held_path, infer(path, held_images), list_sha256=world.held_sha256, scale=world.scale,
                     arm=arm, seed=seed, extra=extra)
        write_output(a28_path(outputs, world.scale, arm, seed), infer(path, a28_images),
                     list_sha256=world.a28_sha256, scale=world.scale, arm=arm, seed=seed, extra=extra)
        paths.append(held_path)
        print(f'{arm} {world.scale} s{seed}: wrote {held_path.parent} and a28', flush=True)
    return paths


def run_ct_a28(world, arm, seeds, outputs, gate=True):
    """B1 (core P3) or B6 (core P2) on ArtELingo-28 from the CT's saved mapper of each seed at this scale."""
    number, backbone = CT_ARMS[arm]
    if gate:
        _gate(outputs, world.scale, [])
    images = world.a28_images(backbone)
    paths = []
    for seed in seeds:
        source = world.ct_root / f'percept/core/stage2/k40/P{number}__s{seed}.npz'
        meta = base.verify_output(source)
        assert meta['config']['arm'] == f'P{number}' and meta['config']['seed'] == seed
        prob = infer(source, images)
        paths.append(write_output(a28_path(outputs, world.scale, arm, seed), prob, list_sha256=world.a28_sha256,
                                  scale=world.scale, arm=arm, seed=seed,
                                  extra=dict(backbone=backbone, source_weights_sha256=meta['weights_sha256'])))
    return paths


# ---- regressions ------------------------------------------------------------------------------------------------

def _same(ours, theirs):
    ours, theirs = np.ascontiguousarray(ours), np.ascontiguousarray(theirs)
    return dict(equal=bool(ours.shape == theirs.shape and ours.dtype == theirs.dtype and (ours == theirs).all()),
                ours_sha256=rb_score.array_sha256(ours), ct_sha256=rb_score.array_sha256(theirs))


def regress(world, outputs, device, epochs=None):
    """Matched-arm regression at the world's scale (stage 2 = sample, stage 3a = full), seed 0, bit for bit.

    The matched code on the default Qwen image (the 'M' arm with the CT's own input) must reproduce the CT's P3 evaluation
    probabilities, mapper weights and held output; the shared inference must reproduce the CT's saved P3 and P2 held
    outputs from the CT's saved mappers. Needs the device the CT trained on. Items `matched/...` are merged into
    `part2/regression/<stage>.json`.
    """
    stage = {v: k for k, v in REGRESSION.items()}[world.scale]
    outputs, seed = Path(outputs), 0
    ct = world.ct_root
    source = {arm: ct / f'percept/core/stage2/k40/{arm}__s{seed}.npz' for arm in ('P3', 'P2')}
    held = {arm: ct / f'held/percept/core__k40/{arm}__s{seed}.npz' for arm in ('P3', 'P2')}
    meta = base.verify_output(source['P3'])
    assert str(device) == meta['config']['device'], \
        f'regression device {device} differs from the CT\'s training device {meta["config"]["device"]}'
    s = world.s
    topic, k, stage1_sha256 = world.topics(seed)
    result = train_stage2(s, topic, k, world.full_images(QWEN, s['all_painting_ids']), QWEN, seed, device,
                          smoke=False, epochs=epochs)
    path = save_stage2(outputs / f'part2/regression/matched_{world.scale}/P3__s{seed}.npz', result,
                       dict(stage1_sha256=stage1_sha256))
    qwen_held = world.full_images(QWEN, world.held_names)
    with np.load(source['P3'], allow_pickle=False) as z:
        ct_eval = z['eval_prob']
    with np.load(held['P3'], allow_pickle=False) as z:
        ct_p3 = z['prob']
    with np.load(held['P2'], allow_pickle=False) as z:
        ct_p2 = z['prob']
    ct_state = torch.load(source['P3'].with_suffix('.pt'), map_location='cpu', weights_only=True)['mapper']
    state_equal = (list(ct_state) == list(result['state']) and
                   all(torch.equal(ct_state[key], result['state'][key]) for key in ct_state))
    # 'input' names the image; 'mode' and 'names_sha256' are provenance the sample-scale CT sidecar predates.
    different = sorted(key for key in set(meta['config']) | set(result['config'])
                       if key not in ('input', 'mode', 'names_sha256')
                       and meta['config'].get(key) != result['config'].get(key))
    items = {
        f'matched/qwen_s{seed}/eval_prob': _same(result['eval_prob'], ct_eval),
        f'matched/qwen_s{seed}/held_prob': _same(infer(path, qwen_held), ct_p3),
        f'matched/qwen_s{seed}/mapper_state': dict(equal=bool(state_equal)),
        f'matched/qwen_s{seed}/config': dict(equal=not different, different_keys=different),
        f'matched/ct_mapper_infer/P3_s{seed}': _same(infer(source['P3'], qwen_held), ct_p3),
        f'matched/ct_mapper_infer/P2_s{seed}': _same(infer(source['P2'], world.clip_images(world.held_names)), ct_p2),
    }
    items = {name: dict(item, **{'pass': item['equal']}) for name, item in items.items()}
    return rb_score.write_regression_items(outputs / f'part2/regression/{stage}.json', items)


# ---- CLI --------------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--regress', choices=sorted(REGRESSION), help='seed 0 regression against the CT\'s saved files')
    p.add_argument('--scale', choices=('sample', 'full'))
    p.add_argument('--arm', choices=sorted(MATCHED) + sorted(CT_ARMS))
    p.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    p.add_argument('--device', default=None)
    args = p.parse_args(argv)
    if args.smoke:
        assert not args.regress, 'a regression check is never shrunk: no --smoke with --regress'
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs --outputs under /tmp/claude-0/'
    assert len(args.seeds) == len(set(args.seeds)), 'duplicate seeds'
    if args.regress:
        world = load_world(REGRESSION[args.regress])
        record = regress(world, args.outputs, args.device or 'cuda')
        bad = [name for name, item in record['items'].items() if not item['pass'] and name.startswith('matched/')]
        print(f'{args.regress}: pass={record["pass"]}; failing matched items: {bad}', flush=True)
        raise SystemExit(0 if not bad else 1)
    assert args.scale and args.arm, 'give --regress, or --scale and --arm'
    seeds = [0] if args.smoke else args.seeds
    device = args.device or ('cuda' if torch.cuda.is_available() or not args.smoke else 'cpu')
    assert args.smoke or device.startswith('cuda'), 'full Stage 2 requires GPU'
    world = load_world(args.scale)
    if args.arm in MATCHED:
        run_matched(world, args.arm, seeds, args.outputs, device, args.smoke, epochs=2 if args.smoke else None,
                    gate=not args.smoke)
    else:
        run_ct_a28(world, args.arm, seeds, args.outputs, gate=not args.smoke)


if __name__ == '__main__':
    main()
