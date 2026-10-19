"""Fit pair-level PercepT core topics with the September training recipe."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

import pc_data
import pc_fixes
import pc_sept

HERE = Path(__file__).resolve().parent
VERSIONS = dict(clip_roberta=('clip_image', 'clip_text', 'roberta', 512),
                qwen_qwen=('qwen_default', 'qwen_default', 'qwen_emotion_focus', 2048),
                qwen_roberta=('qwen_default', 'qwen_default', 'roberta', 2048))
BLOCK_SIZE = 4096


@contextmanager
def one_thread():
    """Limit already-loaded BLAS and OpenMP pools, including sklearn's."""
    with threadpool_limits(limits=1):
        assert _openmp_effective_n_threads() == 1
        assert all(pool['num_threads'] == 1 for pool in threadpool_info())
        yield


def kmeans_start(encoder, inputs, device, n_clusters, seed):
    """Run the verbatim September start with repeatable center sums."""
    with one_thread():
        return pc_sept.initialize_cluster_centers(encoder, inputs, device, n_clusters, seed)


def array_sha256(x):
    """SHA256 of C-order array bytes in joined frozen order, before transforms."""
    digest = hashlib.sha256()
    for start in range(0, len(x), BLOCK_SIZE):
        digest.update(np.ascontiguousarray(x[start:start + BLOCK_SIZE]).tobytes())
    return digest.hexdigest()


def transform_emotion(emotion, *, mean, projection):
    """Apply centered fit PCA; both fit mean and projection are mandatory."""
    assert emotion.ndim == projection.ndim == 2 and mean.ndim == 1
    assert emotion.shape[1] == len(mean) == projection.shape[0]
    assert emotion.dtype == mean.dtype == projection.dtype == np.float32
    return (emotion - mean) @ projection


def prepare_inputs(s, version, fit_rows, block_size=BLOCK_SIZE):
    """Join features, fit E on fit pairs only, and fuse all captions in blocks."""
    assert version in VERSIONS and block_size > 0
    assert fit_rows.ndim == 1 and fit_rows.dtype == np.int64
    assert len(fit_rows) > 0 and ((fit_rows >= 0) & (fit_rows < len(s['pid']))).all()
    image_kind, text_kind, emotion_kind, width = VERSIONS[version]
    start = time.perf_counter()
    image = pc_data.image_features(s, image_kind)
    text = pc_data.caption_features(s, text_kind)
    emotion = pc_data.caption_features(s, emotion_kind)
    inputs = dict(image=array_sha256(image), text=array_sha256(text),
                  emotion=array_sha256(emotion))
    seconds = dict(load_features=time.perf_counter() - start)
    print(f"load_features: {seconds['load_features']:.3f} seconds", flush=True)
    start = time.perf_counter()
    mean = projection = orthogonal = None
    maps = dict(pca_mean_sha256=None, pca_projection_sha256=None, orthogonal_map_sha256=None)
    if version == 'clip_roberta':
        mean, projection = pc_fixes.pca_map(emotion[fit_rows], 512)
        maps.update(pca_mean_sha256=array_sha256(mean),
                    pca_projection_sha256=array_sha256(projection))
    elif version == 'qwen_roberta':
        orthogonal = pc_fixes.orthogonal_map(768, 2048, seed=0)
        maps['orthogonal_map_sha256'] = array_sha256(orthogonal)
    seconds['emotion_map'] = time.perf_counter() - start
    print(f"emotion_map: {seconds['emotion_map']:.3f} seconds", flush=True)
    start = time.perf_counter()
    h = np.empty((len(s['pid']), width), dtype=np.float32)
    for start_row in range(0, len(h), block_size):
        stop = start_row + block_size
        e = emotion[start_row:stop]
        if version == 'clip_roberta':
            e = transform_emotion(e, mean=mean, projection=projection)
        elif version == 'qwen_roberta':
            norms = np.linalg.norm(e, axis=1, keepdims=True)
            assert (norms > 0).all()
            e = (e / norms) @ orthogonal.T
        h[start_row:stop] = pc_fixes.fuse_eq2(image[s['pid'][start_row:stop]],
                                            text[start_row:stop], e)
    seconds['fusion'] = time.perf_counter() - start
    print(f"fusion: {seconds['fusion']:.3f} seconds", flush=True)
    return h, dict(input_sha256s=inputs, maps=maps, seconds=seconds)


def count_fit_pairs(encoder, centers, inputs, device, block_size=BLOCK_SIZE):
    """Count fit argmax assignments over every trained center, before A3's cut."""
    assert inputs.ndim == centers.ndim == 2 and len(inputs) > 0 and len(centers) == 100
    assert inputs.dtype == centers.dtype == torch.float32 and block_size > 0
    counts = np.zeros(len(centers), dtype=np.int64)
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), block_size):
            z = encoder(inputs[start:start + block_size].to(device))
            q = pc_sept.soft_assignments(z, centers)
            assert torch.isfinite(q).all()
            counts += np.bincount(q.argmax(dim=1).cpu().numpy(), minlength=len(centers))
    assert counts.sum() == len(inputs)
    return counts


def assign_captions(encoder, centers, h, pid, eval_pid, kept, device, block_size=BLOCK_SIZE):
    """Frozen Q over ascending kept centers; average per evaluation painting."""
    assert h.ndim == 2 and h.dtype == np.float32 and len(h) == len(pid)
    assert pid.ndim == eval_pid.ndim == kept.ndim == 1
    assert np.issubdtype(pid.dtype, np.integer) and (pid >= 0).all()
    assert np.issubdtype(eval_pid.dtype, np.integer) and (eval_pid >= 0).all()
    assert len(np.unique(eval_pid)) == len(eval_pid) > 0
    assert centers.ndim == 2 and len(centers) == len(kept)
    assert centers.dtype == torch.float32 and kept.dtype == np.bool_ and kept.any()
    assert block_size > 0
    lookup = np.full(max(pid.max(), eval_pid.max()) + 1, -1, dtype=np.int64)
    lookup[eval_pid] = np.arange(len(eval_pid))
    q_sum = np.zeros((len(eval_pid), int(kept.sum())), dtype=np.float32)
    counts = np.zeros(len(eval_pid), dtype=np.int64)
    topic = np.empty(len(h), dtype=np.int16)
    encoder.eval()
    with torch.no_grad():
        surviving = centers.detach()[torch.from_numpy(kept).to(device)]
        for start in range(0, len(h), block_size):
            z = encoder(torch.from_numpy(h[start:start + block_size]).to(device))
            q = pc_sept.soft_assignments(z, surviving).cpu().numpy()
            assert np.isfinite(q).all()
            topic[start:start + block_size] = q.argmax(axis=1)
            local = lookup[pid[start:start + block_size]]
            selected = local >= 0
            np.add.at(q_sum, local[selected], q[selected])
            np.add.at(counts, local[selected], 1)
    assert (counts > 0).all(), 'evaluation painting without captions'
    return topic, (q_sum / counts[:, None]).astype(np.float32)


def record_seconds(seconds, phase, start):
    seconds[phase] = time.perf_counter() - start
    print(f'{phase}: {seconds[phase]:.3f} seconds', flush=True)


def make_config(s, version, seed, smoke, fit_rows, device):
    image, text, emotion, width = VERSIONS[version]
    return dict(version=version, seed=seed, smoke=bool(smoke), device=str(device),
                source_commit='1b70d7f', rows='image-caption pairs; fit only',
                image=image, text=text, emotion=emotion, input_dim=width,
                captions=len(s['pid']), paintings=len(s['first']),
                fit_pairs=len(fit_rows), fit_paintings=len(np.unique(s['pid'][fit_rows])),
                fit_rows_sha256=array_sha256(fit_rows),
                smoke_fit_selection='first 2000 fit captions in frozen order' if smoke else None,
                fusion='Eq.2; normalized I,T,E; unnormalized h',
                emotion_map={'clip_roberta': 'fit-centered full-SVD PCA to 512; fixed signs',
                             'qwen_qwen': 'identity',
                             'qwen_roberta': 'l2 E; QR orthonormal columns 2048x768; fixed signs'}[version],
                orthogonal_seed=0, pca_fit_partition='fit',
                encoder=[width, 500, 500, 2000, 128],
                decoder=[128, 2000, 500, 500, width], activation='ReLU; linear endpoints',
                pretrain_epochs=2 if smoke else 100, pretrain_batch_size=1024,
                pretrain_shuffle='torch.randperm each epoch; include tail',
                pretrain_lr=.001, joint_lr=.0001, optimizer='Adam',
                adam_betas=[.9, .999], adam_eps=1e-8, weight_decay=0,
                initial_centers=100, kmeans_n_init=10, kmeans_random_state=seed,
                kmeans_init='k-means++', kmeans_max_iter=300, kmeans_tol=.0001,
                kmeans_algorithm='lloyd', kmeans_threads=1,
                max_joint_steps=5 if smoke else 500, joint_batch='full fit pairs',
                stability_threshold=0.0 if smoke else .001, full_stability_threshold=.001,
                stop_rule='post-step hard-label change fraction < threshold or epoch ceiling',
                p_refresh_steps=1, p_detached=True, student_t_alpha=1,
                joint_loss='KL(P||Q) batchmean + mean MSE + balance_weight KL(U||mean Q) sum',
                reconstruction_weight=1, balance_weight=1000, balance_log_clamp=1e-8,
                reported_gammas=['1.5', '2', '3', '5'],
                pruning='A3; top 40 of 100 by fit-pair argmax count; ties to lower index',
                topic_order='ascending original kept center index',
                heldout_assignment='frozen encoder; Q renormalized over kept centers',
                painting_q='caption Q mean in subset_pid order',
                block_size=BLOCK_SIZE, array_hash='sha256 C-order array bytes; before transforms',
                numpy_seed=seed, torch_seed=seed, cuda_seed=seed,
                cudnn_deterministic=True, cudnn_benchmark=False,
                deterministic_algorithms=True, deterministic_warn_only=True,
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'], cpu_threads=8)


def run(s, version, seed, smoke=None, device=None, outroot=None):
    """Train one version/seed and seal its arrays and complete config sidecar."""
    assert version in VERSIONS and seed in (0, 1, 2)
    assert smoke is None or outroot is None, '--outroot cannot be used with --smoke'
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert smoke is not None or str(device).startswith('cuda'), 'full Stage 1 requires GPU'
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
    if smoke is not None:
        assert len(fit_rows) >= 2000
        fit_rows = fit_rows[:2000]
    config = make_config(s, version, seed, smoke is not None, fit_rows, device)
    with threadpool_limits(limits=8):
        h, provenance = prepare_inputs(s, version, fit_rows)
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
            record_seconds(seconds, 'pretrain', start)
            start = time.perf_counter()
            centers = kmeans_start(encoder, inputs, device, 100, seed)
            record_seconds(seconds, 'kmeans', start)
            start = time.perf_counter()
            history, stop_reason, steps = pc_sept.train_dec_until_stable(
                encoder, decoder, centers, inputs, device,
                lambda line: print(line, flush=True), 100)
            record_seconds(seconds, 'joint', start)
        finally:
            for name, value in previous.items():
                setattr(pc_sept, name, value)
        del decoder
        start = time.perf_counter()
        center_norms = torch.linalg.vector_norm(centers.detach(), dim=1).cpu().numpy()
        fit_counts = count_fit_pairs(encoder, centers, inputs, device)
        del inputs
        kept = pc_fixes.top_used(fit_counts, 40)
        assert kept.sum() == 40
        counts = {gamma: int(pc_fixes.kept_centers(center_norms, gamma=float(gamma)).sum())
                  for gamma in ('1.5', '2', '3', '5')}
        topic, q_paint_eval = assign_captions(
            encoder, centers, h, s['pid'], s['subset_pid'], kept, device)
        final_counts = np.bincount(topic[fit_rows], minlength=40)
        kept_topics = dict(fit_pairs=len(fit_rows), with_fit_pair=int((final_counts > 0).sum()),
                           with_1pct=int((final_counts * 100 >= len(fit_rows)).sum()),
                           share_before_cut=float(fit_counts[kept].sum() / len(fit_rows)))
        record_seconds(seconds, 'assignment', start)
    folder = (Path(smoke) if smoke is not None else
              Path(outroot) if outroot is not None else HERE / 'outputs') / 'stage1'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{version}__s{seed}.npz'
    start = time.perf_counter()
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, center_norms=center_norms, fit_counts=fit_counts, kept=kept,
                        topic=topic, q_paint_eval=q_paint_eval)
    pending.replace(path)
    record_seconds(seconds, 'save_arrays', start)
    record_seconds(seconds, 'total', total_start)
    meta = dict(config=config, K=40, K_gamma=counts, kept_topics=kept_topics,
                epochs=dict(pretrain=len(pretrain_losses), joint=steps),
                final_change_fraction=history[-1]['fraction_changed'], stop_reason=stop_reason,
                pretrain_losses=pretrain_losses, joint_history=history, seconds=seconds,
                sample_sha256=pc_data.fingerprint(s),
                input_sha256s=provenance['input_sha256s'], maps=provenance['maps'])
    sidecar = path.with_suffix('.json')
    pending_json = sidecar.with_name(sidecar.stem + '.pending.json')
    pending_json.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    pending_json.replace(sidecar)
    print(f'Wrote {path}; K=40; K_gamma={counts}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=VERSIONS, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument('--smoke', type=Path,
                        help='output root; 2000 fit pairs, 2 pretrain epochs, exactly 5 joint steps')
    outputs.add_argument('--outroot', type=Path, help='full-run output root (default: HERE/outputs)')
    args = parser.parse_args()
    run(pc_data.load_frozen(), args.version, args.seed, args.smoke, outroot=args.outroot)


if __name__ == '__main__':
    main()
