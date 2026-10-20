"""Core PercepT training artifacts and image-only inference for the claim test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
for folder in ('20261017_aspect_premise_checks', '20261018_percept_core'):
    sys.path.insert(0, str(HERE.parent / folder))

import ct_held
import pc_data
import pc_fixes
import pc_sept
import pc_stage1
import pc_stage2

OUTPUTS = HERE / 'outputs'
VERSIONS = ('clip_roberta', 'qwen_qwen')
ARMS = {arm: pc_stage2.ARMS[arm] for arm in ('P1', 'P2', 'P3', 'P4')}
SEEDS = tuple(range(5))


def check_path(settings, cut='k40'):
    """Reserve the paper/strict paths for ticket 07."""
    assert settings == 'core' and cut == 'k40'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stage1_path(settings, version, seed):
    check_path(settings)
    assert version in VERSIONS and seed in SEEDS
    return OUTPUTS / 'percept' / settings / 'stage1' / f'{version}__s{seed}.npz'


def stage2_path(settings, cut, arm, seed):
    check_path(settings, cut)
    assert arm in ARMS and seed in SEEDS
    return OUTPUTS / 'percept' / settings / 'stage2' / cut / f'{arm}__s{seed}.npz'


def save_output(path, arrays, state, meta):
    """Commit arrays and CPU weights first, then their hash sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, **arrays)
    pending.replace(path)
    weights = path.with_suffix('.pt')
    pending = weights.with_name(weights.stem + '.pending.pt')
    torch.save(state, pending)
    pending.replace(weights)
    meta.update(array_sha256s={key: pc_stage1.array_sha256(x) for key, x in arrays.items()},
                weights_sha256=file_sha256(weights))
    pc_stage2.write_sidecar(path.with_suffix('.json'), meta)
    return path


def verify_output(path):
    """Verify payload hashes; existence alone never marks a run complete."""
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    config = meta['config']
    check_path(config['settings'], config['cut'])
    assert config['seed'] in SEEDS and meta['K'] == 40
    stage2 = 'arm' in config
    identity = config['arm'] if stage2 else config['version']
    assert path.stem == f"{identity}__s{config['seed']}"
    assert file_sha256(path.with_suffix('.pt')) == meta['weights_sha256'], 'weights hash mismatch'
    with np.load(path, allow_pickle=False) as arrays:
        keys = {'eval_prob'} if stage2 else {'center_norms', 'fit_counts', 'kept', 'topic', 'q_paint_eval'}
        assert set(arrays.files) == keys
        assert set(arrays.files) == set(meta['array_sha256s'])
        for key in arrays.files:
            assert pc_stage1.array_sha256(arrays[key]) == meta['array_sha256s'][key], 'array hash mismatch'
            assert np.isfinite(arrays[key]).all()
        prob = arrays['eval_prob' if stage2 else 'q_paint_eval']
        assert prob.shape == (config['eval_paintings'], 40) and prob.dtype == np.float32
        assert ((prob >= 0) & (prob <= 1)).all()
        if not stage2:
            assert arrays['center_norms'].shape == (100,) and arrays['center_norms'].dtype == np.float32
            assert arrays['fit_counts'].shape == (100,) and arrays['fit_counts'].dtype == np.int64
            assert (arrays['fit_counts'] >= 0).all() and arrays['fit_counts'].sum() == config['fit_pairs']
            assert arrays['kept'].dtype == np.bool_
            assert np.array_equal(arrays['kept'], pc_fixes.top_used(arrays['fit_counts'], 40))
            assert arrays['topic'].shape == (config['captions'],) and arrays['topic'].dtype == np.int16
            assert ((arrays['topic'] >= 0) & (arrays['topic'] < 40)).all()
            assert np.allclose(prob.sum(axis=1), 1, atol=1e-6)
    state = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
    with torch.random.fork_rng(devices=[]):
        if stage2:
            assert set(state) == {'mapper'}
            module = build_mapper(config, meta['K'])
            weights = state['mapper']
        else:
            assert set(state) == {'encoder', 'centers'}
            assert state['centers'].shape == (100, 128) and state['centers'].dtype == torch.float32
            assert torch.isfinite(state['centers']).all()
            module, _ = pc_sept.build_autoencoder(config['input_dim'])
            weights = state['encoder']
        assert weights and all(value.dtype == torch.float32 and torch.isfinite(value).all()
                               for value in weights.values())
        module.load_state_dict(weights)
    return meta


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def run_stage1(s, settings, version, seed, *, smoke=False, device='cpu'):
    """Train one version/seed and seal its arrays and complete config sidecar."""
    check_path(settings)
    assert version in VERSIONS and seed in SEEDS
    total_start = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(8)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    fit_rows = pc_fixes.pair_rows(s['pid'], pc_data.partitions(s))
    if smoke:
        assert len(fit_rows) >= 2000
        fit_rows = fit_rows[:2000]
    config = pc_stage1.make_config(s, version, seed, smoke, fit_rows, device)
    with threadpool_limits(limits=8):
        h, provenance = pc_stage1.prepare_inputs(s, version, fit_rows)
        seconds = provenance['seconds']
        inputs = torch.from_numpy(h[fit_rows])
        start = time.perf_counter()
        encoder, decoder = pc_sept.build_autoencoder(h.shape[1])
        encoder.to(device)
        decoder.to(device)
        # Keep the copied functions verbatim; smoke changes only their limits.
        names = ('PRETRAIN_EPOCHS', 'MAX_DEC_EPOCHS', 'STABILITY_THRESHOLD')
        previous = {name: getattr(pc_sept, name) for name in names}
        pc_sept.PRETRAIN_EPOCHS = config['pretrain_epochs']
        pc_sept.MAX_DEC_EPOCHS = config['max_joint_steps']
        pc_sept.STABILITY_THRESHOLD = config['stability_threshold']
        try:
            pretrain_losses = pc_sept.pretrain_autoencoder(
                encoder, decoder, inputs, device, lambda line: print(line, flush=True))
            pc_stage1.record_seconds(seconds, 'pretrain', start)
            start = time.perf_counter()
            centers = pc_stage1.kmeans_start(encoder, inputs, device, 100, seed)
            pc_stage1.record_seconds(seconds, 'kmeans', start)
            start = time.perf_counter()
            history, stop_reason, steps = pc_sept.train_dec_until_stable(
                encoder, decoder, centers, inputs, device,
                lambda line: print(line, flush=True), 100)
            pc_stage1.record_seconds(seconds, 'joint', start)
        finally:
            for name, value in previous.items():
                setattr(pc_sept, name, value)
        del decoder
        start = time.perf_counter()
        center_norms = torch.linalg.vector_norm(centers.detach(), dim=1).cpu().numpy()
        fit_counts = pc_stage1.count_fit_pairs(encoder, centers, inputs, device)
        del inputs
        kept = pc_fixes.top_used(fit_counts, 40)
        assert kept.sum() == 40
        counts = {gamma: int(pc_fixes.kept_centers(center_norms, gamma=float(gamma)).sum())
                  for gamma in ('1.5', '2', '3', '5')}
        topic, q_paint_eval = pc_stage1.assign_captions(
            encoder, centers, h, s['pid'], s['subset_pid'], kept, device)
        kept_topics = pc_stage1.kept_topic_stats(topic[fit_rows], fit_counts, kept)
        pc_stage1.record_seconds(seconds, 'assignment', start)
    path = stage1_path(settings, version, seed)
    pc_stage1.record_seconds(seconds, 'total', total_start)
    meta = dict(config=config, K=40, K_gamma=counts, kept_topics=kept_topics,
                epochs=dict(pretrain=len(pretrain_losses), joint=steps),
                final_change_fraction=history[-1]['fraction_changed'], stop_reason=stop_reason,
                pretrain_losses=pretrain_losses, joint_history=history, seconds=seconds,
                sample_sha256=pc_data.fingerprint(s),
                input_sha256s=provenance['input_sha256s'], maps=provenance['maps'])
    meta['config'].update(settings=settings, cut='k40', eval_paintings=len(s['subset_pid']))
    arrays = dict(center_norms=center_norms, fit_counts=fit_counts, kept=kept,
                  topic=topic, q_paint_eval=q_paint_eval)
    state = dict(encoder=cpu_state(encoder), centers=centers.detach().cpu().clone())
    return save_output(path, arrays, state, meta)


def determinism(seed):
    """Use core's seeds, thread limits and deterministic torch settings."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'


def train_mapper_with_state(inputs, targets, config, k):
    """Run core unchanged; retain the mapper when it evaluates its full batch."""
    states = []

    def capture(module, args, output):
        # train_mapper does not return its mapper. Observe only its outer module
        # on the evaluation tensor; the hook changes neither outputs nor RNG.
        if (isinstance(module, (torch.nn.Sequential, pc_sept.AttentionPoolingMapper))
                and not module.training and args[0] is inputs['eval']):
            states.append(cpu_state(module))

    handle = torch.nn.modules.module.register_module_forward_hook(capture)
    try:
        result = pc_stage2.train_mapper(inputs, targets, config, k)
    finally:
        handle.remove()
    assert len(states) == 1, 'core mapper was not captured on its evaluation batch'
    return (*result, states[0])


def run_stage2(s, settings, cut, arm, seed, *, smoke=False, device='cpu'):
    """Train core's mapper and save its full evaluation batch and final state."""
    path = stage2_path(settings, cut, arm, seed)
    start = time.perf_counter()
    version, _, _ = ARMS[arm]
    stage1 = stage1_path(settings, version, seed)
    previous = verify_output(stage1)
    assert previous['config']['smoke'] == smoke, 'Stage 1 smoke mismatch'
    topic, k = pc_stage2.load_stage1(s, stage1, version, seed)
    config = pc_stage2.make_config(s, arm, seed, smoke, device)
    config.update(settings=settings, cut=cut)
    determinism(seed)
    ids, targets = pc_stage2.caption_targets(s, topic, k, smoke)
    with threadpool_limits(limits=8):
        inputs = pc_stage2.image_inputs(s, arm, ids, device)
        eval_prob, dev_prob, history, epoch_seconds, state = train_mapper_with_state(
            inputs, targets, config, k)
    for part, prob in (('eval', eval_prob), ('dev', dev_prob)):
        assert prob.shape == (len(ids[part]), k) and prob.dtype == np.float32
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    meta = dict(config=config, K=k, sample_sha256=pc_data.fingerprint(s),
                stage1_sha256=file_sha256(stage1),
                stage1_array_sha256s=previous['array_sha256s'],
                stage1_weights_sha256=previous['weights_sha256'],
                eval_names_sha256=ct_held.names_sha256(s['all_painting_ids'][ids['eval']].tolist()),
                dev_auc=None, dev_valid_topics=0,
                loss_history=history, epoch_seconds=epoch_seconds)
    positives = targets['dev'].sum(axis=0)
    valid = (positives > 0) & (positives < len(targets['dev']))
    if valid.any():
        aucs, skipped = pc_sept.evaluate_auc(dev_prob, targets['dev'],
                                            lambda line: print(line, flush=True), arm + ' dev')
        meta.update(dev_auc=pc_sept.auc_summary(aucs)['macro'], dev_valid_topics=len(aucs),
                    dev_skipped_topics=skipped)
    else:
        meta.update(dev_auc_note='No dev topic has both positive and negative labels',
                    dev_skipped_topics=list(range(k)))
    meta['seconds'] = time.perf_counter() - start
    return save_output(path, dict(eval_prob=eval_prob), dict(mapper=state), meta)


def build_mapper(config, k):
    """Rebuild the exact core architecture before loading its state."""
    if config['mapper'] == 'attention':
        return pc_sept.AttentionPoolingMapper(d_model=config['input_dim'], n_topics=k)
    return torch.nn.Sequential(torch.nn.Linear(config['input_dim'], config['hidden_dim']),
                               torch.nn.ReLU(), torch.nn.Linear(config['hidden_dim'], k))


def infer_stage2(settings, cut, arm, seed, names, *, device=None):
    """Return float32 sigmoid probabilities in caller order, in one batch.

    The controller enforces V1 before calling this with held painting names.
    By default use the training device, so evaluation replay uses the same kernels.
    """
    path = stage2_path(settings, cut, arm, seed)
    meta = verify_output(path)
    config = meta['config']
    assert (config['settings'], config['cut'], config['arm'], config['seed']) == (settings, cut, arm, seed)
    assert len(names) == len(set(names)) > 0 and all(isinstance(name, str) for name in names)
    device = config['device'] if device is None else device
    determinism(seed)
    with threadpool_limits(limits=8):
        mapper = build_mapper(config, meta['K'])
        state = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        mapper.load_state_dict(state['mapper'])
        mapper.to(device).eval()
        kind = ARMS[arm][1]
        if kind == 'patch':
            x = ct_held.clip_patches(names)
        elif kind == 'clip_image':
            x = ct_held.clip_global_images(names)
        else:
            x = ct_held.qwen_default_images(names)
            # default_images already L2-normalizes; core image_inputs does so again.
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        shape = (len(names), 50, config['input_dim']) if kind == 'patch' else (len(names), config['input_dim'])
        assert x.shape == shape and x.dtype == np.float32 and np.isfinite(x).all()
        with torch.no_grad():
            prob = mapper(torch.from_numpy(np.ascontiguousarray(x)).to(device)).sigmoid().cpu().numpy()
    assert prob.shape == (len(names), meta['K']) and prob.dtype == np.float32
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    return prob


def load_frozen():
    """Read the main checkout's frozen sample; never create a worktree sample."""
    path = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')
    with np.load(path, allow_pickle=False) as z:
        s = {key: z[key] for key in z.files}
    assert (len(s['sample_ids']), len(s['first']), len(s['subset_pid'])) == (99999, 19889, 5000)
    assert len(np.unique(s['sample_ids'])) == 99999
    assert len(np.unique(s['subset_pid'])) == 5000
    assert np.array_equal(s['all_painting_ids'][s['pid']], s['painting_ids'])
    assert np.array_equal(s['all_painting_ids'][s['subset_pid']], s['subset_painting_ids'])
    return s


INCOMPLETE = (OSError, ValueError, KeyError, AssertionError, RuntimeError,
              EOFError, TypeError, zipfile.BadZipFile)


def complete_output(path, expected, sample_sha256, previous=None):
    """Check the requested run identity, payloads and Stage 1 dependency."""
    try:
        meta = verify_output(path)
        assert all(meta['config'][key] == value for key, value in expected.items()), 'run identity mismatch'
        assert meta['sample_sha256'] == sample_sha256, 'sample mismatch'
        if previous is not None:
            assert meta['stage1_array_sha256s'] == previous['array_sha256s'], 'Stage 1 arrays changed'
            assert meta['stage1_weights_sha256'] == previous['weights_sha256'], 'Stage 1 weights changed'
        return True
    except INCOMPLETE as error:
        if path.exists():
            print(f'Incomplete {path}: {error}', flush=True)
        return False


def verify_held_output(path, source, names, sha256, device):
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['config'] == source['config']
    assert meta['weights_sha256'] == source['weights_sha256'], 'held checkpoint changed'
    assert meta['held_sha256'] == sha256 == ct_held.names_sha256(names), 'held list mismatch'
    assert meta['rows'] == len(names) == 12281
    assert meta['inference_device'] == str(device)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'prob'}
        prob = z['prob']
        assert prob.shape == (len(names), source['K']) and prob.dtype == np.float32
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
        assert meta['array_sha256s'] == dict(prob=pc_stage1.array_sha256(prob)), 'held array hash mismatch'
    return meta


def run_held(settings, cut, arm, seed, *, device=None, smoke=False):
    """Save one frozen-list inference, or reuse a verified complete held artifact.

    V1 is the queue/controller's prerequisite, before invoking this function.
    """
    source = verify_output(stage2_path(settings, cut, arm, seed))
    assert source['config']['smoke'] == smoke, 'Stage 2 smoke mismatch'
    names, sha256 = ct_held.read_held_list(OUTPUTS / 'held/held_paintings.json')
    assert len(names) == 12281 and ct_held.names_sha256(names) == sha256
    device = source['config']['device'] if device is None else device
    path = OUTPUTS / 'held/percept' / f'{settings}__{cut}' / f'{arm}__s{seed}.npz'
    try:
        verify_held_output(path, source, names, sha256, device)
        print(f'Verified complete: {path}', flush=True)
        return path
    except INCOMPLETE as error:
        if path.exists():
            print(f'Incomplete {path}: {error}', flush=True)
    prob = infer_stage2(settings, cut, arm, seed, names, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, prob=prob)
    pending.replace(path)
    meta = dict(config=source['config'], K=source['K'], held_sha256=sha256,
                rows=len(names), weights_sha256=source['weights_sha256'],
                inference_device=str(device), array_sha256s=dict(prob=pc_stage1.array_sha256(prob)))
    pc_stage2.write_sidecar(path.with_suffix('.json'), meta)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', choices=('core',), default='core')
    parser.add_argument('--cut', choices=('k40',), default='k40')
    parser.add_argument('--version', choices=VERSIONS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--stage', choices=('1', '2', 'held'), required=True)
    parser.add_argument('--device', default='cpu', help='training/inference device (default: cpu)')
    parser.add_argument('--smoke', action='store_true', help='core smoke limits; never reused as a full run')
    args = parser.parse_args(argv)
    if args.stage == '1':
        if args.version is None or args.arm is not None:
            parser.error('Stage 1 requires --version and no --arm')
    elif args.arm is None or args.version is not None:
        parser.error('Stage 2/held requires --arm and no --version')
    assert len(args.seeds) == len(set(args.seeds)), 'duplicate seeds'
    s = load_frozen() if args.stage != 'held' else None
    paths = []
    for seed in args.seeds:
        if args.stage == 'held':
            paths.append(run_held(args.settings, args.cut, args.arm, seed,
                                  device=args.device, smoke=args.smoke))
            continue
        expected = dict(settings=args.settings, cut=args.cut, seed=seed,
                        smoke=args.smoke, device=args.device)
        previous = None
        if args.stage == '1':
            expected['version'] = args.version
            path = stage1_path(args.settings, args.version, seed)
        else:
            expected['arm'] = args.arm
            version = ARMS[args.arm][0]
            path = stage2_path(args.settings, args.cut, args.arm, seed)
            previous = verify_output(stage1_path(args.settings, version, seed))
            assert previous['config']['smoke'] == args.smoke, 'Stage 1 smoke mismatch'
        if complete_output(path, expected, pc_data.fingerprint(s), previous):
            print(f'Verified complete: {path}', flush=True)
        elif args.stage == '1':
            run_stage1(s, args.settings, args.version, seed, smoke=args.smoke, device=args.device)
        else:
            run_stage2(s, args.settings, args.cut, args.arm, seed, smoke=args.smoke, device=args.device)
        paths.append(path)
    return paths


if __name__ == '__main__':
    main()
