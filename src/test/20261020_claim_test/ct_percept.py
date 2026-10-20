"""Core and paper PercepT artifacts and image-only inference for the claim test.

Paper Stage 1 saves both cuts in one artifact: kept_<cut>, topic_<cut>,
fit_topics_<cut> and q_paint_eval_<cut>. The unsuffixed core-compatible arrays
alias K=40; K_by_cut in the sidecar records both widths. stop_epoch is a
one-element int64 array, also recorded in JSON. Full runs finalize R-COL once
both versions have all five seeds; smoke artifacts never enter that file.
"""
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
STAGE1_KEYS = dict(core={'center_norms', 'fit_counts', 'kept', 'topic', 'q_paint_eval',
                         'centers', 'fit_latent', 'kept_k40', 'fit_topics_k40'})

STAGE1_KEYS['paper'] = STAGE1_KEYS['core'] | {
    'kept_strict', 'fit_topics_strict', 'topic_strict', 'topic_k40',
    'q_paint_eval_strict', 'q_paint_eval_k40', 'stop_epoch'}


def check_path(settings, cut='k40'):
    """Keep core on A3; paper exposes both cuts."""
    assert settings in ('core', 'paper')
    assert cut in (('k40',) if settings == 'core' else ('strict', 'k40'))


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
    assert settings != 'paper' or arm in ('P1', 'P4')
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
    assert config['seed'] in SEEDS
    assert type(meta['K']) is int
    if config['settings'] == 'core' or 'arm' not in config:
        assert meta['K'] == 40
    else:
        assert config['arm'] in ('P1', 'P4')
        assert meta['K'] == 40 if config['cut'] == 'k40' else 2 <= meta['K'] <= 100
        verify_stage1_width(path, config, meta)
    stage2 = 'arm' in config
    identity = config['arm'] if stage2 else config['version']
    assert path.stem == f"{identity}__s{config['seed']}"
    assert file_sha256(path.with_suffix('.pt')) == meta['weights_sha256'], 'weights hash mismatch'
    with np.load(path, allow_pickle=False) as arrays:
        keys = {'eval_prob'} if stage2 else STAGE1_KEYS[config['settings']]
        assert set(arrays.files) == keys
        assert set(arrays.files) == set(meta['array_sha256s'])
        for key in arrays.files:
            assert pc_stage1.array_sha256(arrays[key]) == meta['array_sha256s'][key], 'array hash mismatch'
            assert np.isfinite(arrays[key]).all()
        prob = arrays['eval_prob' if stage2 else 'q_paint_eval']
        assert prob.shape == (config['eval_paintings'], meta['K']) and prob.dtype == np.float32
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
            centers = arrays['centers']
            assert centers.shape == (100, 128) and centers.dtype == np.float32
            assert np.allclose(arrays['center_norms'], np.linalg.norm(centers.astype(np.float64), axis=1),
                               rtol=1e-6)
            assert arrays['fit_latent'].shape == (config['fit_pairs'], 128)
            assert arrays['fit_latent'].dtype == np.float32
            assert arrays['kept_k40'].dtype == np.bool_
            assert np.array_equal(arrays['kept_k40'], arrays['kept'])
            assert arrays['fit_topics_k40'].shape == (config['fit_pairs'],)
            assert arrays['fit_topics_k40'].dtype == np.int16
            assert ((arrays['fit_topics_k40'] >= 0) & (arrays['fit_topics_k40'] < 40)).all()
        if not stage2 and config['settings'] == 'paper':
            verify_paper_cuts(arrays, config, meta)
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
            assert np.array_equal(state['centers'].numpy(), centers), 'checkpoint centers mismatch'
            if config['settings'] == 'paper':
                import ct_paper_sept
                module, _ = ct_paper_sept.build_autoencoder(config['input_dim'])
            else:
                module, _ = pc_sept.build_autoencoder(config['input_dim'])
            weights = state['encoder']
        assert weights and all(value.dtype == torch.float32 and torch.isfinite(value).all()
                               for value in weights.values())
        module.load_state_dict(weights)
    return meta


def verify_stage1_width(path, config, meta):
    """Paper Stage 2 width must be its Stage 1 kept count for that cut."""
    stage1 = stage1_path(config['settings'], ARMS[config['arm']][0], config['seed'])
    stage1_meta = json.loads(stage1.with_suffix('.json').read_text())
    with np.load(stage1, allow_pickle=False) as z:
        kept = int(z[f"kept_{config['cut']}"].sum())
    assert meta['K'] == kept == stage1_meta['K_by_cut'][config['cut']], \
        f"{path}: K {meta['K']} differs from Stage 1 kept count {kept}"


def verify_paper_cuts(arrays, config, meta):
    """Check both cuts against the rule, including their saved local topic IDs."""
    import ct_rcol
    masks = dict(strict=ct_rcol.strict_mask(arrays['center_norms']),
                 k40=ct_rcol.k40_mask(arrays['fit_counts']))
    assert meta['K_by_cut'] == {cut: int(mask.sum()) for cut, mask in masks.items()}
    assert arrays['stop_epoch'].shape == (1,) and arrays['stop_epoch'].dtype == np.int64
    assert arrays['stop_epoch'].item() == meta['stop_epoch'] == meta['epochs']['joint']
    assert 1 <= meta['stop_epoch'] <= config['max_joint_steps']
    for cut, mask in masks.items():
        assert arrays[f'kept_{cut}'].dtype == np.bool_
        assert np.array_equal(arrays[f'kept_{cut}'], mask)
        k = int(mask.sum())
        for key, shape in ((f'topic_{cut}', (config['captions'],)),
                           (f'fit_topics_{cut}', (config['fit_pairs'],))):
            assert arrays[key].shape == shape and arrays[key].dtype == np.int16
            assert ((arrays[key] >= 0) & (arrays[key] < k)).all()
        q = arrays[f'q_paint_eval_{cut}']
        assert q.shape == (config['eval_paintings'], k) and q.dtype == np.float32
        assert ((q >= 0) & (q <= 1)).all() and np.allclose(q.sum(axis=1), 1, atol=1e-6)
    assert np.array_equal(arrays['topic_k40'], arrays['topic'])
    assert np.array_equal(arrays['q_paint_eval_k40'], arrays['q_paint_eval'])


def load_paper_cut(s, path, meta, cut):
    """Read the chosen cut from the single shared paper Stage 1 artifact."""
    check_path('paper', cut)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == STAGE1_KEYS['paper']
        verify_paper_cuts(z, meta['config'], meta)
        k = meta['K_by_cut'][cut]
        topic = z[f'topic_{cut}']
        assert topic.shape == s['pid'].shape
        assert z[f'q_paint_eval_{cut}'].shape == (len(s['subset_pid']), k)
        return topic, k


def fit_probabilities(fit_latent, centers, device, block_size=pc_stage1.BLOCK_SIZE):
    """Student-t Q of every fit pair over all 100 centers, in one float32 pass."""
    q = np.empty((len(fit_latent), len(centers)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(fit_latent), block_size):
            z = torch.from_numpy(fit_latent[start:start + block_size]).to(device)
            q[start:start + block_size] = pc_sept.soft_assignments(z, centers.detach()).cpu().numpy()
    assert np.isfinite(q).all()
    return q


def paper_fit_assignments(q, strict):
    """Paper fit counts and both cuts' topics from one Q, as ct_rcol's docstring defines them.

    Counts: argmax over all 100 centers. Topics: argmax of Q over the kept columns,
    renormalized (in float64, so renormalizing creates no new ties), lowest index on a
    tie. A pair counted at a kept center keeps that center, so no post-cut count is lost.
    """
    assert q.ndim == 2 and q.shape[1] == 100 and strict.shape == (100,)
    counts = np.bincount(q.argmax(axis=1), minlength=100).astype(np.int64)
    kept = pc_fixes.top_used(counts, 40)
    topics = {}
    for cut, mask in (('strict', strict), ('k40', kept)):
        cut_q = q[:, mask].astype(np.float64)
        cut_q /= cut_q.sum(axis=1, keepdims=True)
        topics[cut] = cut_q.argmax(axis=1).astype(np.int16)
    return counts, kept, topics


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def run_stage1(s, settings, version, seed, *, smoke=False, device=None):
    """Train one version/seed and seal its arrays and complete config sidecar."""
    check_path(settings)
    assert version in VERSIONS and seed in SEEDS
    device = device or 'cuda'
    assert smoke or str(device).startswith('cuda'), 'full Stage 1 requires GPU'
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
    if settings == 'paper':
        import ct_paper_sept
        ct_paper_sept.stage1_config(config)
    with threadpool_limits(limits=8):
        h, provenance = pc_stage1.prepare_inputs(s, version, fit_rows)
        seconds = provenance['seconds']
        inputs = torch.from_numpy(h[fit_rows])
        start = time.perf_counter()
        builder = ct_paper_sept if settings == 'paper' else pc_sept
        encoder, decoder = builder.build_autoencoder(h.shape[1])
        encoder.to(device)
        decoder.to(device)
        if settings == 'paper':
            pretrain_losses = ct_paper_sept.pretrain_autoencoder(
                encoder, decoder, inputs, device, epochs=config['pretrain_epochs'])
            pc_stage1.record_seconds(seconds, 'pretrain', start)
            start = time.perf_counter()
            centers = pc_stage1.kmeans_start(encoder, inputs, device, 100, seed)
            pc_stage1.record_seconds(seconds, 'kmeans', start)
            start = time.perf_counter()
            history, stop_reason, steps = ct_paper_sept.train_dec_until_stable(
                encoder, decoder, centers, inputs, device, max_epochs=config['max_joint_steps'],
                stability_threshold=config['stability_threshold'])
            pc_stage1.record_seconds(seconds, 'joint', start)
        else:
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
        fit_latent = np.empty((len(inputs), 128), dtype=np.float32)
        encoder.eval()
        with torch.no_grad():
            for start_row in range(0, len(inputs), pc_stage1.BLOCK_SIZE):
                stop = start_row + pc_stage1.BLOCK_SIZE
                fit_latent[start_row:stop] = encoder(inputs[start_row:stop].to(device)).cpu().numpy()
        del inputs
        if settings == 'paper':
            # One fit Q: counts and topics then agree by construction, even on near-ties.
            import ct_rcol
            strict = ct_rcol.strict_mask(center_norms)
            fit_counts, kept, fit_topics = paper_fit_assignments(
                fit_probabilities(fit_latent, centers, device), strict)
        else:
            kept = pc_fixes.top_used(fit_counts, 40)
        assert kept.sum() == 40
        counts = {gamma: int(pc_fixes.kept_centers(center_norms, gamma=float(gamma)).sum())
                  for gamma in ('1.5', '2', '3', '5')}
        topic, q_paint_eval = pc_stage1.assign_captions(
            encoder, centers, h, s['pid'], s['subset_pid'], kept, device)
        if settings == 'paper':
            topic_strict, q_strict = pc_stage1.assign_captions(
                encoder, centers, h, s['pid'], s['subset_pid'], strict, device)
        kept_topics = pc_stage1.kept_topic_stats(
            fit_topics['k40'] if settings == 'paper' else topic[fit_rows], fit_counts, kept)
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
                  topic=topic, q_paint_eval=q_paint_eval, centers=centers.detach().cpu().numpy(),
                  fit_latent=fit_latent, kept_k40=kept, fit_topics_k40=topic[fit_rows])
    if settings == 'paper':
        arrays.update(kept_strict=strict, fit_topics_strict=fit_topics['strict'],
                      fit_topics_k40=fit_topics['k40'],
                      topic_strict=topic_strict, topic_k40=topic,
                      q_paint_eval_strict=q_strict, q_paint_eval_k40=q_paint_eval,
                      stop_epoch=np.asarray([steps], dtype=np.int64))
        meta.update(K_by_cut=dict(strict=int(strict.sum()), k40=40), stop_epoch=steps)
    state = dict(encoder=cpu_state(encoder), centers=centers.detach().cpu().clone())
    save_output(path, arrays, state, meta)
    if settings == 'paper' and not smoke:
        finalize_stage1()
    return path


def finalize_stage1():
    """Write R-COL once both paper versions have all five full-run artifacts."""
    paths = [stage1_path('paper', version, seed) for version in VERSIONS for seed in SEEDS]
    if not all(path.exists() for path in paths):
        return None
    for path in paths:
        meta = verify_output(path)
        assert meta['config']['settings'] == 'paper' and not meta['config']['smoke'], \
            'R-COL requires full Stage 1 runs'
        assert meta['config']['fit_pairs'] == 67327
    import ct_rcol
    return ct_rcol.write_stage1(paths[0].parent, OUTPUTS / 'percept/rcol_stage1.json')


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


def load_stage1(s, path, version, seed, *, cut='k40'):
    """Validate Stage 1 identity and load caption topics for the selected cut."""
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['sample_sha256'] == pc_data.fingerprint(s), 'Stage 1 sample mismatch'
    assert meta['config']['version'] == version, 'Stage 1 version mismatch'
    assert meta['config']['seed'] == seed, 'Stage 1 seed mismatch'
    assert set(meta['K_gamma']) == {'1.5', '2', '3', '5'}
    if meta['config']['settings'] == 'paper':
        return load_paper_cut(s, path, meta, cut)
    assert cut == 'k40'
    k = meta['K']
    assert type(k) is int and k == 40
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == STAGE1_KEYS[meta['config']['settings']]
        assert z['center_norms'].shape == z['fit_counts'].shape == z['kept'].shape == (100,)
        assert z['center_norms'].dtype == np.float32 and np.isfinite(z['center_norms']).all()
        assert z['fit_counts'].dtype == np.int64 and (z['fit_counts'] >= 0).all()
        assert z['kept'].dtype == np.bool_ and k == 40 == int(z['kept'].sum())
        assert np.array_equal(z['kept'], pc_fixes.top_used(z['fit_counts'], 40)), 'Stage 1 kept mask mismatch'
        topic = z['topic']
        assert topic.shape == s['pid'].shape and topic.dtype == np.int16
        assert ((topic >= 0) & (topic < k)).all()
        q = z['q_paint_eval']
        assert q.shape == (len(s['subset_pid']), k) and q.dtype == np.float32
        assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
        assert np.allclose(q.sum(axis=1), 1, atol=1e-6)
    return topic, k


def run_stage2(s, settings, cut, arm, seed, *, smoke=False, device=None):
    """Train the selected recipe's mapper, saving evaluation probabilities and weights."""
    path = stage2_path(settings, cut, arm, seed)
    device = device or 'cuda'
    assert smoke or str(device).startswith('cuda'), 'full Stage 2 requires GPU'
    start = time.perf_counter()
    version, _, _ = ARMS[arm]
    stage1 = stage1_path(settings, version, seed)
    previous = verify_output(stage1)
    assert previous['config']['smoke'] == smoke, 'Stage 1 smoke mismatch'
    topic, k = load_stage1(s, stage1, version, seed, cut=cut)
    if k < 2:
        print(f'Skipping {settings}/{cut}/{arm} seed={seed}: only {k} topic', flush=True)
        return None
    config = pc_stage2.make_config(s, arm, seed, smoke, device)
    config.update(settings=settings, cut=cut)
    if settings == 'paper':
        import ct_paper_sept
        ct_paper_sept.stage2_config(config)
    determinism(seed)
    ids, targets = pc_stage2.caption_targets(s, topic, k, smoke)
    with threadpool_limits(limits=8):
        inputs = pc_stage2.image_inputs(s, arm, ids, device)
        trainer = ct_paper_sept.train_mapper if settings == 'paper' else train_mapper_with_state
        eval_prob, dev_prob, history, epoch_seconds, state = trainer(inputs, targets, config, k)
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
    """Rebuild the shared mapper architecture before loading its state."""
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
    assert str(device) == config['device'], 'inference device differs from training device'
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
            if meta['config']['settings'] == 'paper':
                assert meta['K'] == previous['K_by_cut'][meta['config']['cut']], \
                    'Stage 2 K differs from Stage 1 kept count'
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
    if settings == 'paper':
        previous = verify_output(stage1_path(settings, ARMS[arm][0], seed))
        assert previous['config']['smoke'] == smoke, 'Stage 1 smoke mismatch'
        if previous['K_by_cut'][cut] < 2:
            print(f'Skipping held {settings}/{cut}/{arm} seed={seed}: only 1 topic', flush=True)
            return None
    source = verify_output(stage2_path(settings, cut, arm, seed))
    assert source['config']['smoke'] == smoke, 'Stage 2 smoke mismatch'
    if settings == 'paper':
        assert source['K'] == previous['K_by_cut'][cut], 'Stage 2 K differs from Stage 1 kept count'
    names, sha256 = ct_held.read_held_list(OUTPUTS / 'held/held_paintings.json')
    assert len(names) == 12281 and ct_held.names_sha256(names) == sha256
    inference_device = source['config']['device'] if device is None else str(device)
    assert inference_device == source['config']['device'], 'inference device differs from training device'
    path = OUTPUTS / 'held/percept' / f'{settings}__{cut}' / f'{arm}__s{seed}.npz'
    try:
        verify_held_output(path, source, names, sha256, inference_device)
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
                inference_device=inference_device, array_sha256s=dict(prob=pc_stage1.array_sha256(prob)))
    pc_stage2.write_sidecar(path.with_suffix('.json'), meta)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', choices=('core', 'paper'), default='core')
    parser.add_argument('--cut', choices=('k40', 'strict'), default='k40',
                        help='Stage 2/held cut; paper Stage 1 saves both cuts')
    parser.add_argument('--version', choices=VERSIONS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--stage', choices=('1', '2', 'held'), required=True)
    parser.add_argument('--device', default=None,
                        help='training defaults to cuda; held uses the recorded training device')
    parser.add_argument('--smoke', action='store_true', help='reduced training limits; never reused as a full run')
    args = parser.parse_args(argv)
    check_path(args.settings, args.cut)
    if args.settings == 'paper' and args.arm is not None and args.arm not in ('P1', 'P4'):
        parser.error('Paper settings support P1 and P4 only')
    if args.stage == '1':
        if args.version is None or args.arm is not None:
            parser.error('Stage 1 requires --version and no --arm')
    elif args.arm is None or args.version is not None:
        parser.error('Stage 2/held requires --arm and no --version')
    assert len(args.seeds) == len(set(args.seeds)), 'duplicate seeds'
    device = args.device
    if args.stage != 'held':
        device = args.device or 'cuda'
    s = load_frozen() if args.stage != 'held' else None
    paths = []
    for seed in args.seeds:
        if args.stage == 'held':
            path = run_held(args.settings, args.cut, args.arm, seed,
                            device=args.device, smoke=args.smoke)
            if path is not None:
                paths.append(path)
            continue
        expected = dict(settings=args.settings, cut='k40' if args.stage == '1' else args.cut, seed=seed,
                        smoke=args.smoke, device=device)
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
            if args.settings == 'paper' and previous['K_by_cut'][args.cut] < 2:
                print(f'Skipping {path}: only 1 topic', flush=True)
                continue
        if complete_output(path, expected, pc_data.fingerprint(s), previous):
            print(f'Verified complete: {path}', flush=True)
        elif args.stage == '1':
            run_stage1(s, args.settings, args.version, seed, smoke=args.smoke, device=device)
        else:
            run_stage2(s, args.settings, args.cut, args.arm, seed, smoke=args.smoke, device=device)
        paths.append(path)
    if args.stage == '1' and args.settings == 'paper' and not args.smoke:
        finalize_stage1()
    return paths


if __name__ == '__main__':
    main()
