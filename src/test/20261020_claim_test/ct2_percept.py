"""PercepT at sample/full scale, with unchanged recipes and optionally checkpointed DEC.

Public runners take mode, recipe, seed and an artifact root. Full evaluation
probabilities are on validation (also the mapper's dev AUC set); sample keeps
stage 1's frozen evaluation order.

DEC row chunks (`dec_chunk_rows`): None keeps stage 1's own soft_assignments
(bitwise identical to stage 1); an int replaces it by a row-chunked, per-chunk
checkpointed version whose centers' gradient sums in a different order (about
1e-8 at 5,000 rows, enough to drift over up to 500 DEC epochs). Default: None in
sample mode (stage 1 must match by hash, 67,327 pairs), 1,024 in full mode
(216,107 pairs do not fit unchunked). `--dec-chunk-rows N` overrides it; the
config records the value. Use a separate --root for such an extra run.

V1 (stage 2a gate): full-mode training (Stage 1 and 2, not smoke) and full-mode
non-smoke held writes first call ct2_stage2a.require_pass; sample mode never does.
"""
import argparse
from contextlib import contextmanager, ExitStack
from functools import partial
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

import ct_percept as base
import ct_held
import ct_score
import ct2_data as data

pc_data = base.pc_data
pc_stage1 = base.pc_stage1
pc_stage2 = base.pc_stage2
pc_sept = base.pc_sept
pc_fixes = base.pc_fixes
VERSIONS = base.VERSIONS
SEEDS = base.SEEDS
ARMS = base.ARMS
DEC_CHUNK_ROWS = 1024
ORIGINAL_SOFT_ASSIGNMENTS = pc_sept.soft_assignments


MODE_DEFAULT = 'mode default'


def default_chunk_rows(mode):
    """None (stage 1's function, bitwise) in sample mode; DEC_CHUNK_ROWS in full mode."""
    assert mode in ('sample', 'full'), mode
    return None if mode == 'sample' else DEC_CHUNK_ROWS


def chunked_soft_assignments(latent, centers, chunk_rows=None):
    """Identical Student-t rows; checkpoint each fixed-size distance broadcast.

    Only the centers' gradient sums change order across chunks. Latent rows
    keep their original reductions; Q, global P and the full-batch loss do not
    change. Checkpointing retains no (rows, topics, latent-width) intermediates.
    """
    chunk_rows = DEC_CHUNK_ROWS if chunk_rows is None else chunk_rows
    assert latent.ndim == centers.ndim == 2 and len(latent) > 0 and chunk_rows > 0
    chunks = []
    for start in range(0, len(latent), chunk_rows):
        rows = latent[start:start + chunk_rows]
        if torch.is_grad_enabled():
            q = checkpoint(ORIGINAL_SOFT_ASSIGNMENTS, rows, centers,
                           use_reentrant=False, preserve_rng_state=False)
        else:
            q = ORIGINAL_SOFT_ASSIGNMENTS(rows, centers)
        chunks.append(q)
    return torch.cat(chunks)


CLIP = data.CLIP
ROBERTA = data.ROBERTA
ROBERTA_S2 = ROBERTA.with_name(ROBERTA.name + '_s2')
PATCHES = pc_data.PATCHES
ANNOTATIONS = pc_data.ANNOTATIONS
FROZEN_HELD_LIST = ct_score.OUTPUTS / 'held/held_paintings.json'
_CLIP_FEATURES = pc_data.clip_features
_PATCH_TOKENS = pc_data.patch_tokens


def clip_features(sample_ids, feature):
    """Reuse the extraction-ID joined CLIP reader at either scale."""
    with patch.object(pc_data, 'CLIP', CLIP):
        return _CLIP_FEATURES(sample_ids, feature)


def caption_features(s, kind):
    """Raw float32 captions; old RoBERTa rows win in the gated store union."""
    if kind == 'clip_text':
        return clip_features(s['sample_ids'], 'txt_features')
    if kind == 'qwen_default':
        return data.default_captions(s, per_caption=True)
    if kind == 'qwen_emotion_focus':
        return data.instructed_captions(s, 'emotion_focus', per_caption=True)
    assert kind == 'roberta', kind
    wanted = s['sample_ids']
    assert wanted.dtype == np.int64 and len(np.unique(wanted)) == len(wanted)
    lookup = {int(sid): row for row, sid in enumerate(wanted)}
    x = np.empty((len(wanted), 768), dtype=np.float32)
    seen = np.zeros(len(wanted), dtype=bool)
    # The frozen sample is fully covered by the old store: sample mode never opens the s2 store.
    paths = [ROBERTA] + ([ROBERTA_S2] if s['mode'] == 'full' else [])
    for path in paths:
        if path == ROBERTA:
            meta = json.loads((path / 'done.json').read_text())
            assert not meta.get('smoke', False), 'smoke RoBERTa store'
            assert meta['snapshot_revision'] == 'd75048347613a25d77de8cf6412eaae9fa7b26be'
        else:
            meta = data.check_s2_store(path)  # parity, revision, counts, ID hash, not smoke
        assert meta['model_id'] == 'SamLowe/roberta-base-go_emotions'
        with np.load(path / 'ids.npz', allow_pickle=False) as ids:
            stored, names = ids['sample_ids'], ids['painting_ids']
        assert stored.dtype == np.int64 and stored.ndim == 1
        assert len(np.unique(stored)) == len(stored), 'duplicate RoBERTa IDs'
        assert names.shape == stored.shape and meta['count'] == len(stored)
        raw = np.load(path / 'txt.npy', mmap_mode='r')
        assert raw.shape == (len(stored), 768) and raw.dtype == np.float32
        source = np.asarray([row for row, sid in enumerate(stored) if int(sid) in lookup], dtype=np.int64)
        target = np.asarray([lookup[int(stored[row])] for row in source], dtype=np.int64)
        assert np.array_equal(names[source], s['painting_ids'][target]), 'RoBERTa painting mismatch'
        cosines = []
        for start in range(0, len(source), 4096):
            src, dst = source[start:start + 4096], target[start:start + 4096]
            values = np.asarray(raw[src], dtype=np.float32)
            assert np.isfinite(values).all() and (np.linalg.norm(values, axis=1) > 0).all()
            overlap = seen[dst]
            if overlap.any():
                cosines.append(np.sum(data.l2(x[dst[overlap]]) * data.l2(values[overlap]), axis=1))
            x[dst[~overlap]] = values[~overlap]
            seen[dst] = True
        if cosines:
            cosine = np.concatenate(cosines)
            assert cosine.mean() > .999 and cosine.min() > .99, 'RoBERTa overlap cosine gate failed'
    assert seen.all(), f'missing RoBERTa captions: {int((~seen).sum())}'
    return x


def image_features(s, kind):
    """Painting-joined float32 globals, with stage 1 normalization unchanged."""
    if kind == 'qwen_default':
        return data.default_images(s)
    assert kind == 'clip_image', kind
    first = s['first']
    assert np.array_equal(s['painting_ids'][first], s['all_painting_ids'])
    return clip_features(s['sample_ids'][first], 'img_features')


def patch_tokens(s, pids):
    """Reuse the sorted-name patch join and its bounded CPU gather."""
    with patch.object(pc_data, 'PATCHES', PATCHES), patch.object(pc_data, 'ANNOTATIONS', ANNOTATIONS):
        return _PATCH_TOKENS(s, pids)


def check_arm(mode, settings, cut, arm):
    assert mode in ('sample', 'full'), mode
    base.check_path(settings, cut)
    allowed = ('P1', 'P4') if settings == 'paper' else (tuple(ARMS) if mode == 'sample' else ('P2', 'P3'))
    assert arm in allowed, f'unsupported arm {arm} for {mode}/{settings}'


def evaluation_ids(s):
    return s['subset_pid'] if s['mode'] == 'sample' else s['dev']


def partitions(s):
    """Fit/dev roles; full validation has no independent evaluation role."""
    part = np.full(len(s['first']), '', dtype='<U4')
    roles = [('fit', s['fit']), ('dev', s['dev'])]
    if s['mode'] == 'sample':
        roles.append(('eval', s['subset_pid']))
    for name, rows in roles:
        assert rows.dtype == np.int64 and len(rows) == len(np.unique(rows)) > 0
        assert (part[rows] == '').all(), 'overlapping PercepT roles'
        part[rows] = name
    assert (part != '').all()
    return part


def caption_targets(s, topic, k, smoke=False):
    """Union all captions of each fit/dev painting, preserving core row order."""
    partition = partitions(s)
    assert partition.shape == (len(s['first']),)
    ids = {part: np.flatnonzero(partition == part).astype(np.int64) for part in ('fit', 'dev')}
    ids['eval'] = evaluation_ids(s).copy()
    # Sample evaluates its own role; full evaluates the dev role (no independent eval role).
    assert (partition[ids['eval']] == ('eval' if s['mode'] == 'sample' else 'dev')).all()
    if smoke:
        assert len(ids['fit']) >= 200
        ids['fit'] = ids['fit'][:200]
    targets = {}
    for part in ('fit', 'dev'):
        rows = pc_fixes.pair_rows(s['pid'], partition, part)
        targets[part] = pc_fixes.multi_hot(s['pid'][rows], topic[rows], ids[part], k)
    return ids, targets


# pc_stage2.make_config, with its inline sample-size gate generalized.
def mapper_config(s, arm, seed, smoke, device):
    version, kind, width = ARMS[arm]
    partition = partitions(s)
    counts = {part: int((partition == part).sum()) for part in ('fit', 'dev', 'eval')}
    counts['eval'] = len(evaluation_ids(s))
    return dict(mode=s['mode'], names_sha256=s.get('names_sha256', {}),
                arm=arm, version=version, seed=seed, smoke=smoke, device=str(device),
                source_commit='1b70d7f', input=kind, input_dim=width,
                mapper='attention' if kind == 'patch' else 'mlp',
                hidden_dim=None if kind == 'patch' else 1024,
                activation=None if kind == 'patch' else 'ReLU',
                normalization='stored patches' if kind == 'patch' else 'l2 global vector',
                targets='multi-hot union of own partition caption topics; A3 K=40',
                fit_paintings=min(200, counts['fit']) if smoke else counts['fit'],
                dev_paintings=counts['dev'], eval_paintings=counts['eval'],
                smoke_fit_selection='first 200 fit paintings in frozen order' if smoke else None,
                epochs=3 if smoke else 100, lr=.003, optimizer='Adam',
                adam_betas=[.9, .999], adam_eps=1e-8, weight_decay=0,
                batch='full fit paintings', loss='mean BCEWithLogitsLoss',
                scheduler=None, early_stopping=False,
                dev_metric='September per-topic ROC AUC; skip constant; unweighted macro',
                eval_order='subset_pid' if s['mode'] == 'sample' else 'dev',
                numpy_seed=seed, torch_seed=seed, cuda_seed=seed,
                cudnn_deterministic=True, cudnn_benchmark=False,
                deterministic_algorithms=True, deterministic_warn_only=True,
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'], cpu_threads=8)


_ORIGINAL_STAGE1_CONFIG = pc_stage1.make_config
_FINGERPRINT = pc_data.fingerprint


def stage1_config(s, version, seed, smoke, fit_rows, device, dec_chunk_rows=MODE_DEFAULT):
    if dec_chunk_rows == MODE_DEFAULT:
        dec_chunk_rows = default_chunk_rows(s['mode'])
    config = _ORIGINAL_STAGE1_CONFIG(s, version, seed, smoke, fit_rows, device)
    config.update(mode=s['mode'], names_sha256=s.get('names_sha256', {}),
                  eval_order='subset_pid' if s['mode'] == 'sample' else 'dev',
                  dec_chunk_rows=dec_chunk_rows,
                  dec_checkpoint=('none; stage 1 soft_assignments' if dec_chunk_rows is None else
                                  'per row chunk; centers gradient sums reordered'))
    return config


@contextmanager
def runtime(root, s=None, dec_chunk_rows=MODE_DEFAULT):
    """Temporarily redirect the shared trainers; always restore stage 1 globals.

    A private full-mode view aliases subset_pid to dev for legacy array shapes.
    The caller's split and its fingerprint stay canonical. R-COL finalization
    belongs to ct2's verdict ticket: the legacy finalizer requires 67,327 pairs.
    dec_chunk_rows: see the module docstring; the default depends on s['mode']
    (full mode's when no split is given).
    """
    if dec_chunk_rows == MODE_DEFAULT:
        dec_chunk_rows = default_chunk_rows('full' if s is None else s['mode'])
    with ExitStack() as stack:
        stack.enter_context(patch.object(base, 'OUTPUTS', Path(root)))
        if dec_chunk_rows is not None:
            stack.enter_context(patch.object(pc_sept.base, 'soft_assignments',
                                             partial(chunked_soft_assignments, chunk_rows=dec_chunk_rows)))
        stack.enter_context(patch.object(pc_data, 'image_features', image_features))
        stack.enter_context(patch.object(pc_data, 'caption_features', caption_features))
        stack.enter_context(patch.object(pc_data, 'patch_tokens', patch_tokens))
        stack.enter_context(patch.object(pc_data, 'partitions', partitions))
        stack.enter_context(patch.object(pc_stage1, 'make_config',
                                         partial(stage1_config, dec_chunk_rows=dec_chunk_rows)))
        stack.enter_context(patch.object(pc_stage2, 'make_config', mapper_config))
        stack.enter_context(patch.object(pc_stage2, 'caption_targets', caption_targets))
        stack.enter_context(patch.object(base, 'finalize_stage1', lambda: None))
        view = s
        if s is not None:
            identity = _FINGERPRINT(s)
            stack.enter_context(patch.object(pc_data, 'fingerprint', lambda _: identity))
            if s['mode'] == 'full':
                view = dict(s, subset_pid=s['dev'])
        yield view


def stage1_path(settings, version, seed, *, root):
    with runtime(root):
        return base.stage1_path(settings, version, seed)


def stage2_path(settings, cut, arm, seed, *, root):
    with runtime(root):
        return base.stage2_path(settings, cut, arm, seed)


def verify_output(path, *, root):
    with runtime(root):
        return base.verify_output(path)


def require_stage2a(mode, smoke, device):
    """V1: only full-mode, non-smoke work needs the stage 2a pass (sample mode is stage 2a)."""
    if mode == 'full' and not smoke:
        import ct2_stage2a
        ct2_stage2a.require_pass(device=str(device))


def run_stage1(mode, settings, version, seed, *, root, device=None, smoke=False,
               dec_chunk_rows=MODE_DEFAULT):
    """Train all captions of the mode's fit paintings; paper saves both cuts."""
    assert mode in ('sample', 'full'), mode
    require_stage2a(mode, smoke, device or 'cuda')
    s = data.split2(mode)
    with runtime(root, s, dec_chunk_rows) as view:
        return base.run_stage1(view, settings, version, seed, device=device, smoke=smoke)


def run_stage2(mode, settings, cut, arm, seed, *, root, device=None, smoke=False):
    """Train a core/paper mapper; dev AUC uses the mode's dev paintings."""
    check_arm(mode, settings, cut, arm)
    require_stage2a(mode, smoke, device or 'cuda')
    s = data.split2(mode)
    with runtime(root, s) as view:
        previous = base.verify_output(base.stage1_path(settings, ARMS[arm][0], seed))
        assert previous['config']['mode'] == mode, 'Stage 1 mode mismatch'
        return base.run_stage2(view, settings, cut, arm, seed, device=device, smoke=smoke)


def infer_stage2(mode, settings, cut, arm, seed, names, *, root, device=None):
    """Image-only inference in caller order, using the recorded mapper device."""
    check_arm(mode, settings, cut, arm)
    with runtime(root):
        meta = base.verify_output(base.stage2_path(settings, cut, arm, seed))
        assert meta['config']['mode'] == mode, 'Stage 2 mode mismatch'
        return base.infer_stage2(settings, cut, arm, seed, names, device=device)


def run_held(mode, settings, cut, arm, seed, *, root, device=None, smoke=False):
    """Save ct_percept's held layout/sidecars, after the V1 gate and the frozen-list check."""
    check_arm(mode, settings, cut, arm)
    with runtime(root):
        if settings == 'paper':
            meta = base.verify_output(base.stage1_path(settings, ARMS[arm][0], seed))
        else:
            meta = base.verify_output(base.stage2_path(settings, cut, arm, seed))
        assert meta['config']['mode'] == mode, 'held source mode mismatch'
        # Held inference runs on the device the weights were trained on.
        require_stage2a(mode, smoke, meta['config']['device'])
        held_sha256 = ct_held.read_held_list(Path(root) / 'held/held_paintings.json')[1]
        frozen_sha256 = ct_held.read_held_list(FROZEN_HELD_LIST)[1]
        assert held_sha256 == frozen_sha256, 'held list differs from the stage 1 frozen held list'
        return base.run_held(settings, cut, arm, seed, device=device, smoke=smoke)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('sample', 'full'), required=True)
    parser.add_argument('--settings', choices=('core', 'paper'), default='core')
    parser.add_argument('--stage', choices=('1', '2', 'held'), required=True)
    parser.add_argument('--version', choices=VERSIONS)
    parser.add_argument('--cut', choices=('strict', 'k40'), default='k40')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--device', default=None)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dec-chunk-rows', type=int, metavar='N',
                        help='Stage 1 only: DEC row-chunk size (default: none in sample mode, 1024 in full)')
    args = parser.parse_args(argv)
    if args.dec_chunk_rows is not None and (args.stage != '1' or args.dec_chunk_rows < 1):
        parser.error('--dec-chunk-rows is a positive integer and applies to --stage 1 only')
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('duplicate seeds')
    try:
        base.check_path(args.settings, args.cut)
        if args.stage == '1':
            assert args.version is not None and args.arm is None, 'Stage 1 requires --version and no --arm'
        else:
            assert args.arm is not None and args.version is None, 'Stage 2/held requires --arm and no --version'
            check_arm(args.mode, args.settings, args.cut, args.arm)
    except AssertionError as error:
        parser.error(str(error))
    paths = []
    for seed in args.seeds:
        kwargs = dict(root=args.root, device=args.device, smoke=args.smoke)
        if args.stage == '1':
            path = run_stage1(args.mode, args.settings, args.version, seed, **kwargs,
                              dec_chunk_rows=MODE_DEFAULT if args.dec_chunk_rows is None else args.dec_chunk_rows)
        else:
            runner = run_stage2 if args.stage == '2' else run_held
            path = runner(args.mode, args.settings, args.cut, args.arm, seed, **kwargs)
        if path is not None:
            paths.append(path)
    return paths


if __name__ == '__main__':
    main()
