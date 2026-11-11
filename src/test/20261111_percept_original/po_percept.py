"""r26 PercepT at paper settings on CLIP ViT-L/14 + ModernBERT-GoEmotions (arm L1: Stage 2 on L/14 patch tokens).

Stage 1 is arm B4's Stage 1 (`ct_percept.run_stage1(view, 'paper', ...)` inside `ct2_percept.runtime`) with the new
version `P.VERSION` registered and the new feature kinds dispatched (others fall through to ct2's). No emotion map: E
(ModernBERT, 768) enters Eq. 2 unchanged, because it already has CLIP L/14's width.

Stage 2 is its own trainer (`train_mapper`), the paper optimisation of `ct_paper_sept.train_mapper` unchanged, with the
fit tokens (42,969 x 257 x 768) resident on the GPU as float16 and cast to float32 per batch (exact), and evaluation in
chunks from the float16 memmap. Held inference reads the same store by painting name.

Every reader joins by the IDs the stores hold (painting names, sample IDs), never by position (constitution C6).
Store paths are module globals (or arguments) so tests can point them at small synthetic stores.
"""
import hashlib
import json
import os
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import po_paths as P

for _folder in (P.CT_ROUND, P.PC_ROUND):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))
if str(P.REPO) not in sys.path:
    sys.path.insert(0, str(P.REPO))

def add_scorer_paths():
    """r17's folder first on sys.path, then the older rounds it needs (as r25's pb_paths.add_scorer_paths)."""
    if str(P.NS_ROUND) not in sys.path:
        sys.path.insert(0, str(P.NS_ROUND))
    import ns_paths
    ns_paths.add_rounds()
    return ns_paths


# r17's scorer checks that its helper modules (apc_, pav_, itp_, ...) come from the main checkout. Import it before the
# claim test's code, which would otherwise cache the worktree copies of the same files first.
add_scorer_paths()
import ns_data                       # noqa: E402,F401

import ct_percept as base            # noqa: E402
import ct_paper_sept                 # noqa: E402
import ct_held                       # noqa: E402
import ct2_percept as ct2            # noqa: E402
import ct2_data                      # noqa: E402

pc_data, pc_stage1, pc_sept = base.pc_data, base.pc_stage1, base.pc_sept

CLIP_GLOBAL = P.CLIP_L14_GLOBAL
CLIP_PATCH = P.CLIP_L14_PATCH
MODERNBERT = P.MODERNBERT
CHUNK_FIT_LOAD = 256         # paintings per host block while filling the GPU tensor
CHUNK_EVAL = 512             # paintings per evaluation chunk
INCOMPLETE = base.INCOMPLETE


# ====================================================================================================== readers
def join_rows(stored, wanted, what):
    """Row of each wanted key in the stored key list (dict join); every wanted key must exist, stored keys unique."""
    stored = [str(x) if not isinstance(x, (int, np.integer)) else int(x) for x in np.asarray(stored).tolist()]
    assert len(set(stored)) == len(stored), f'{what}: duplicate stored IDs'
    lookup = {key: row for row, key in enumerate(stored)}
    wanted = np.asarray(wanted).tolist()
    missing = [k for k in wanted if k not in lookup]
    assert not missing, f'{what}: {len(missing)} requested IDs missing from the store, e.g. {missing[:3]}'
    return np.asarray([lookup[k] for k in wanted], dtype=np.int64)


def _check_float(x, shape, what):
    assert x.shape == shape and x.dtype == np.float32, (what, x.shape, x.dtype)
    for a in range(0, len(x), 65536):
        assert np.isfinite(x[a:a + 65536]).all(), f'{what}: non-finite values'
    return x


def _clip_index(root, n_images, n_texts, width):
    root = Path(root)
    index = json.loads((root / 'index.json').read_text())
    assert index['backbone'] == 'clip_l14' and not index['smoke'] and index['dtype'] == 'float16', 'CLIP L/14 store identity'
    assert index['dim'] == width and index['n_images'] == n_images == len(index['images']), 'CLIP L/14 image count'
    assert index['n_texts'] == n_texts == len(index['texts']), 'CLIP L/14 text count'
    if (root / 'progress.json').exists():
        assert json.loads((root / 'progress.json').read_text())['gate']['ok'] is True, 'CLIP L/14 store gate'
    return index


def clipl14_image(s, root=None, *, n_images=P.N_PAINTINGS, n_texts=P.N_CAPTIONS, width=P.TEXT_WIDTH):
    """float32 [len(s['first']), 768]: L/14 global image vectors by painting name, in s['all_painting_ids'] order."""
    root = Path(CLIP_GLOBAL if root is None else root)
    first = s['first']
    assert np.array_equal(s['painting_ids'][first], s['all_painting_ids'])
    index = _clip_index(root, n_images, n_texts, width)
    raw = np.load(root / 'img.npy', mmap_mode='r')
    assert raw.shape == (n_images, width) and raw.dtype == np.float16
    rows = join_rows([r['painting'] for r in index['images']], s['all_painting_ids'], 'CLIP L/14 images')
    x = np.asarray(raw[rows]).astype(np.float32)
    return _check_float(x, (len(first), width), 'CLIP L/14 image')


def clipl14_text(s, root=None, *, n_images=P.N_PAINTINGS, n_texts=P.N_CAPTIONS, width=P.TEXT_WIDTH):
    """float32 [len(s['sample_ids']), 768]: L/14 caption vectors by sample_id, in s['sample_ids'] order."""
    root = Path(CLIP_GLOBAL if root is None else root)
    index = _clip_index(root, n_images, n_texts, width)
    raw = np.load(root / 'txt.npy', mmap_mode='r')
    assert raw.shape == (n_texts, width) and raw.dtype == np.float16
    texts = index['texts']
    stored = np.asarray([r['sample_id'] for r in texts], dtype=np.int64)
    rows = join_rows(stored, np.asarray(s['sample_ids'], dtype=np.int64), 'CLIP L/14 texts')
    got = np.asarray([texts[r]['painting'] for r in rows])
    assert np.array_equal(got, s['painting_ids']), 'CLIP L/14 caption painting mismatch'
    x = np.asarray(raw[rows]).astype(np.float32)
    return _check_float(x, (len(rows), width), 'CLIP L/14 text')


def _done(root, what):
    done = json.loads((Path(root) / 'done.json').read_text())
    assert not done.get('smoke', False), f'{what}: smoke store'
    return done


def modernbert(s, root=None, *, n_captions=P.N_CAPTIONS, width=P.TEXT_WIDTH):
    """float32 [len(s['sample_ids']), 768]: ModernBERT-GoEmotions caption vectors by sample_id, in s['sample_ids'] order."""
    root = Path(MODERNBERT if root is None else root)
    done = _done(root, 'ModernBERT')
    assert done['model_id'] == P.MODERNBERT_MODEL and done['revision'] == P.MODERNBERT_REVISION, 'ModernBERT identity'
    assert done['count'] == n_captions and done['shape'] == [n_captions, width] and done['dtype'] == 'float32'
    assert done['parity']['ok'] is True, 'ModernBERT parity gate'
    with np.load(root / 'ids.npz', allow_pickle=False) as z:
        stored, names = z['sample_ids'], z['painting_ids']
    assert stored.dtype == np.int64 and stored.shape == names.shape == (n_captions,)
    raw = np.load(root / 'txt.npy', mmap_mode='r')
    assert raw.shape == (n_captions, width) and raw.dtype == np.float32
    rows = join_rows(stored, np.asarray(s['sample_ids'], dtype=np.int64), 'ModernBERT')
    assert np.array_equal(names[rows], s['painting_ids']), 'ModernBERT painting mismatch'
    return _check_float(np.asarray(raw[rows]), (len(rows), width), 'ModernBERT')


class PatchStore:
    """L/14 patch tokens: float16 memmap [N, 257, 768] and its painting names (ids.npz painting_names)."""

    def __init__(self, root=None, *, n=P.N_PAINTINGS, tokens=P.PATCH_TOKENS, width=P.PATCH_WIDTH):
        root = Path(CLIP_PATCH if root is None else root)
        self.root = root
        self.done = _done(root, 'CLIP L/14 patch')
        assert self.done['model_id'] == P.CLIP_L14_MODEL and self.done['revision'] == P.CLIP_L14_REVISION
        assert self.done['count'] == n and self.done['shape'] == [n, tokens, width] and self.done['dtype'] == 'float16'
        assert self.done['parity_gate']['ok'] is True, 'CLIP L/14 patch parity gate'
        with np.load(root / 'ids.npz', allow_pickle=False) as z:
            self.names = z['painting_names']
        assert self.names.ndim == 1 and len(self.names) == n
        self.raw = np.load(root / 'tokens.npy', mmap_mode='r')
        assert self.raw.shape == (n, tokens, width) and self.raw.dtype == np.float16

    def rows(self, names):
        return join_rows(self.names, names, 'CLIP L/14 patch tokens')

    def fetch(self, rows):
        """float16 numpy block of the given store rows (copy), finite."""
        block = np.ascontiguousarray(self.raw[rows])
        assert block.dtype == np.float16 and np.isfinite(block).all(), 'patch tokens: dtype or non-finite values'
        return block


def load_tokens_to_device(store, rows, device, chunk=CHUNK_FIT_LOAD):
    """float16 tensor [len(rows), T, D] on `device`, filled block by block (no full host copy)."""
    out = torch.empty((len(rows), *store.raw.shape[1:]), dtype=torch.float16, device=device)
    for a in range(0, len(rows), chunk):
        block = store.fetch(rows[a:a + chunk])
        out[a:a + len(block)] = torch.from_numpy(block).to(device)
    return out


# ============================================================================================ Stage 1 patches
def image_features(s, kind):
    if kind == 'clipl14_image':
        return clipl14_image(s)
    return ct2.image_features(s, kind)


def caption_features(s, kind):
    if kind == 'clipl14_text':
        return clipl14_text(s)
    if kind == 'modernbert':
        return modernbert(s)
    return ct2.caption_features(s, kind)


def _file_sha(path):
    return base.file_sha256(path)


def stage1_config(s, version, seed, smoke, fit_rows, device):
    """ct2's Stage 1 config with this round's version: no emotion map (E enters Eq. 2 unchanged)."""
    assert version == P.VERSION
    config = ct2.stage1_config(s, 'qwen_qwen', seed, smoke, fit_rows, device)   # any registered version; all overridden
    image, text, emotion, width = P.VERSION_KINDS
    config.update(version=version, image=image, text=text, emotion=emotion, input_dim=width,
                  encoder=[width, 500, 500, 2000, 128], decoder=[128, 2000, 500, 500, width],
                  emotion_map='none: ModernBERT (768) has CLIP L/14 width (768); E enters Eq. 2 as stored '
                              '(fuse_eq2 normalises I, T and E itself); no PCA, no rotation',
                  encoders=dict(image=f'{P.CLIP_L14_MODEL}@{P.CLIP_L14_REVISION} (global, {CLIP_GLOBAL})',
                                text=f'{P.CLIP_L14_MODEL}@{P.CLIP_L14_REVISION} (global, {CLIP_GLOBAL})',
                                emotion=f'{P.MODERNBERT_MODEL}@{P.MODERNBERT_REVISION} ({MODERNBERT})'),
                  emotion_store_done_sha256=_file_sha(Path(MODERNBERT) / 'done.json') if (Path(MODERNBERT) / 'done.json').exists() else None)
    config.pop('orthogonal_seed', None)
    config.pop('pca_fit_partition', None)
    return config


@contextmanager
def runtime(root, s=None):
    """ct2's runtime plus this round's version, feature kinds and Stage 1 config."""
    with ExitStack() as stack:
        view = stack.enter_context(ct2.runtime(root, s))
        stack.enter_context(patch.object(base, 'VERSIONS', base.VERSIONS + (P.VERSION,)))
        stack.enter_context(patch.dict(pc_stage1.VERSIONS, {P.VERSION: P.VERSION_KINDS}))
        stack.enter_context(patch.object(pc_data, 'image_features', image_features))
        stack.enter_context(patch.object(pc_data, 'caption_features', caption_features))
        stack.enter_context(patch.object(pc_stage1, 'make_config', stage1_config))
        yield view


# ============================================================================================== paths, Stage 1
def stage1_path(root, seed):
    return Path(root) / 'percept' / 'paper' / 'stage1' / f'{P.VERSION}__s{seed}.npz'


def stage2_path(root, cut, seed):
    assert cut in P.CUTS and seed in P.SEEDS
    return Path(root) / 'percept' / 'paper' / 'stage2' / cut / f'{P.ARM}__s{seed}.npz'


def held_path(root, cut, seed):
    assert cut in P.CUTS and seed in P.SEEDS
    return Path(root) / 'held' / f'{P.ARM}__{cut}__s{seed}.npz'


def verify_stage1(root, seed):
    with runtime(root):
        assert stage1_path(root, seed) == base.stage1_path('paper', P.VERSION, seed)
        return base.verify_output(stage1_path(root, seed))


def complete_stage1(s, root, seed, smoke):
    with runtime(root, s):
        expected = dict(settings='paper', cut='k40', seed=seed, smoke=smoke, version=P.VERSION)
        return base.complete_output(stage1_path(root, seed), expected, pc_data.fingerprint(s))


def run_stage1(s, root, seed, *, device=None, smoke=False):
    """Stage 1, both cuts, chunked DEC (full-mode default): ct_percept.run_stage1 on the full view."""
    with runtime(root, s) as view:
        assert s['mode'] == 'full' and len(s['fit']) == P.N_TRAIN and len(s['dev']) == P.N_VAL
        return base.run_stage1(view, 'paper', P.VERSION, seed, device=device, smoke=smoke)


# ================================================================================================== Stage 2
def stage2_config(s, cut, seed, smoke, device, ids, k):
    return dict(mode=s['mode'], names_sha256=s.get('names_sha256', {}), settings='paper', cut=cut,
                arm=P.ARM, version=P.VERSION, seed=seed, smoke=bool(smoke), device=str(device),
                input='patch', input_dim=P.PATCH_WIDTH, n_tokens=P.PATCH_TOKENS,
                mapper='attention', hidden_dim=None, activation=None,
                normalization='stored L/14 tokens: visual_projection(last_hidden_state), 257 tokens, none applied',
                token_store=str(CLIP_PATCH), token_store_model=f'{P.CLIP_L14_MODEL}@{P.CLIP_L14_REVISION}',
                targets='multi-hot union of own partition caption topics; selected cut',
                fit_paintings=len(ids['fit']), dev_paintings=len(ids['dev']), eval_paintings=len(ids['eval']),
                smoke_fit_selection='first 200 fit paintings in frozen order' if smoke else None,
                epochs=3 if smoke else 100, lr=.001, optimizer='Adam', adam_betas=[.9, .999], adam_eps=1e-8,
                weight_decay=0, batch='shuffled fit minibatches; include tail', batch_size=64,
                loss='mean BCEWithLogitsLoss', scheduler='cosine per epoch', eta_min=1e-5, early_stopping=False,
                dev_metric='September per-topic ROC AUC; skip constant; unweighted macro',
                eval_order='dev', fit_token_residency='float16 on the training device; each batch cast to float32 '
                                                      'before the mapper (exact cast of the stored values)',
                eval_residency=f'chunks of {CHUNK_EVAL} paintings from the CPU float16 memmap, cast to float32',
                numpy_seed=seed, torch_seed=seed, cuda_seed=seed, cudnn_deterministic=True, cudnn_benchmark=False,
                deterministic_algorithms=True, deterministic_warn_only=True,
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'], cpu_threads=8)


def predict(mapper, fetch, n, device, chunk=CHUNK_EVAL):
    """Sigmoid probabilities float32 [n, K], in chunks; fetch(a, b) returns the float16/float32 rows a..b."""
    mapper.eval()
    k = mapper.classifier.out_features
    out = np.empty((n, k), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            x = torch.as_tensor(fetch(a, b)).to(device).float()
            out[a:b] = mapper(x).sigmoid().cpu().numpy()
    return out


def train_mapper(fit_tokens, fit_target, config, k):
    """ct_paper_sept.train_mapper's optimisation, on tokens of any dtype (cast to float32 per batch).

    Returns (mapper in eval mode, loss history, epoch seconds, CPU state). Non-finite losses do not stop training
    (paper settings: the artifact is saved and marked).
    """
    seed, device = config['seed'], config['device']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mapper = pc_sept.AttentionPoolingMapper(d_model=config['input_dim'], n_topics=k).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=.001)
    scheduler = ct_paper_sept.cosine_scheduler(optimizer, config['epochs'])
    target = torch.from_numpy(fit_target).to(device)
    assert len(target) == len(fit_tokens)
    history, epoch_seconds = [], []
    mapper.train()
    for epoch in range(1, config['epochs'] + 1):
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        tick = time.perf_counter()
        permutation = torch.randperm(len(target), device=fit_tokens.device)
        total = 0.0
        for start in range(0, len(target), 64):
            rows = permutation[start:start + 64]
            loss = F.binary_cross_entropy_with_logits(mapper(fit_tokens[rows].float()), target[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        history.append(total / len(target))
        scheduler.step()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        epoch_seconds.append(time.perf_counter() - tick)
        print(f'Paper {P.ARM} epoch={epoch} mean_bce={history[-1]:.6f}', flush=True)
    mapper.eval()
    return mapper, history, epoch_seconds, base.cpu_state(mapper)


def load_fit_tokens(s, store, device, smoke=False):
    """GPU float16 tokens of the fit paintings, in s['fit'] order (smoke: the first 200)."""
    fit = s['fit'][:200] if smoke else s['fit']
    rows = store.rows(s['all_painting_ids'][fit])
    t = time.perf_counter()
    tokens = load_tokens_to_device(store, rows, device)
    print(f'fit tokens on {device}: {tuple(tokens.shape)} {tokens.dtype}, {time.perf_counter() - t:.1f} s', flush=True)
    return tokens


def run_stage2(s, root, cut, seed, *, device=None, smoke=False, fit_tokens=None, store=None):
    """Train one (cut, seed) mapper on the L/14 patch tokens; save dev probabilities, weights and the sidecar."""
    device = device or 'cuda'
    assert smoke or str(device).startswith('cuda'), 'full Stage 2 requires GPU'
    start = time.perf_counter()
    store = store or PatchStore()
    with runtime(root, s) as view:
        stage1 = stage1_path(root, seed)
        previous = base.verify_output(stage1)
        assert previous['config']['smoke'] == smoke, 'Stage 1 smoke mismatch'
        topic, k = base.load_stage1(view, stage1, P.VERSION, seed, cut=cut)
        sample_sha = pc_data.fingerprint(view)
    if k < 2:
        print(f'Skipping {cut} seed={seed}: only {k} topic', flush=True)
        return None
    ids, targets = ct2.caption_targets(s, topic, k, smoke)
    config = stage2_config(s, cut, seed, smoke, device, ids, k)
    base.determinism(seed)
    if fit_tokens is None:
        fit_tokens = load_fit_tokens(s, store, device, smoke)
    assert len(fit_tokens) == len(ids['fit']) and fit_tokens.dtype == torch.float16
    # Review S1: the tokens follow load_fit_tokens' painting order; the targets must follow the same paintings.
    assert np.array_equal(ids['fit'], s['fit'][:200] if smoke else s['fit']), 'fit token / target painting order differs'
    with threadpool_limits(limits=8):
        mapper, history, epoch_seconds, state = train_mapper(fit_tokens, targets['fit'], config, k)
        rows = store.rows(s['all_painting_ids'][ids['dev']])
        dev_prob = predict(mapper, lambda a, b: store.fetch(rows[a:b]), len(rows), device)
    assert dev_prob.shape == (len(ids['dev']), k) and dev_prob.dtype == np.float32
    assert base.in_unit_interval(dev_prob, True)
    nonfinite = (not np.isfinite(dev_prob).all() or not all(np.isfinite(h) for h in history)
                 or not all(torch.isfinite(v).all() for v in state.values()))
    meta = dict(config=config, K=k, sample_sha256=sample_sha, stage1_sha256=base.file_sha256(stage1),
                stage1_array_sha256s=previous['array_sha256s'], stage1_weights_sha256=previous['weights_sha256'],
                eval_names_sha256=ct_held.names_sha256(s['all_painting_ids'][ids['eval']].tolist()),
                dev_auc=None, dev_valid_topics=0, nonfinite=bool(nonfinite),
                loss_history=[h if np.isfinite(h) else None for h in history], epoch_seconds=epoch_seconds)
    positives = targets['dev'].sum(axis=0)
    valid = (positives > 0) & (positives < len(targets['dev']))
    if nonfinite:
        meta.update(dev_auc_note='Non-finite outputs; dev AUC not computed (collapsed)', dev_skipped_topics=list(range(k)))
    elif valid.any():
        aucs, skipped = pc_sept.evaluate_auc(dev_prob, targets['dev'], lambda line: print(line, flush=True),
                                             f'{P.ARM} dev')
        meta.update(dev_auc=pc_sept.auc_summary(aucs)['macro'], dev_valid_topics=len(aucs), dev_skipped_topics=skipped)
    else:
        meta.update(dev_auc_note='No dev topic has both positive and negative labels', dev_skipped_topics=list(range(k)))
    meta['seconds'] = time.perf_counter() - start
    return base.save_output(stage2_path(root, cut, seed), dict(eval_prob=dev_prob), dict(mapper=state), meta)


def verify_stage2(root, cut, seed):
    """Payload hashes, identity, weights loadable, K equal to Stage 1's kept count; returns the sidecar."""
    path = stage2_path(root, cut, seed)
    meta = json.loads(path.with_suffix('.json').read_text())
    config = meta['config']
    assert (config['settings'], config['cut'], config['arm'], config['version'], config['seed']) == \
        ('paper', cut, P.ARM, P.VERSION, seed)
    assert type(meta['K']) is int and type(meta['nonfinite']) is bool and path.stem == f'{P.ARM}__s{seed}'
    assert base.file_sha256(path.with_suffix('.pt')) == meta['weights_sha256'], 'weights hash mismatch'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'eval_prob'} == set(meta['array_sha256s'])
        prob = z['eval_prob']
        assert pc_stage1.array_sha256(prob) == meta['array_sha256s']['eval_prob'], 'array hash mismatch'
    assert prob.shape == (config['dev_paintings'], meta['K']) and prob.dtype == np.float32
    assert base.in_unit_interval(prob, meta['nonfinite'])
    assert meta['nonfinite'] or np.isfinite(prob).all()
    state = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
    assert set(state) == {'mapper'}
    mapper = pc_sept.AttentionPoolingMapper(d_model=config['input_dim'], n_topics=meta['K'])
    mapper.load_state_dict(state['mapper'])
    previous = verify_stage1(root, seed)
    assert meta['K'] == previous['K_by_cut'][cut], 'Stage 2 K differs from Stage 1 kept count'
    assert meta['stage1_array_sha256s'] == previous['array_sha256s'], 'Stage 1 arrays changed'
    assert meta['stage1_weights_sha256'] == previous['weights_sha256'], 'Stage 1 weights changed'
    return meta


def complete_stage2(root, cut, seed, smoke, sample_sha256):
    try:
        meta = verify_stage2(root, cut, seed)
        assert meta['config']['smoke'] == smoke and meta['sample_sha256'] == sample_sha256, 'run identity mismatch'
        return True
    except INCOMPLETE as error:
        if stage2_path(root, cut, seed).exists():
            print(f'Incomplete {stage2_path(root, cut, seed)}: {error}', flush=True)
        return False


# ============================================================================================== held inference
def held_names():
    """r17's held painting names (the 12,281 test paintings), checked against ct_held's split rebuild."""
    names = ns_data.keys('artelingo', 'held')
    assert len(names) == len(set(names)) == P.N_TEST
    assert set(names) == set(ct_held.held_names()), 'r17 held names differ from the claim test held set'
    return names


def verify_held(path, source, names, device):
    meta = json.loads(path.with_suffix('.json').read_text())
    sha = ct_held.names_sha256(names)
    assert meta['config'] == source['config'] and meta['weights_sha256'] == source['weights_sha256']
    assert meta['held_sha256'] == sha and meta['rows'] == len(names) == P.N_TEST
    assert meta['inference_device'] == str(device)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'prob'}
        prob = z['prob']
    assert prob.shape == (len(names), source['K']) and prob.dtype == np.float32
    assert meta['nonfinite'] is bool(not np.isfinite(prob).all()) and base.in_unit_interval(prob, True)
    assert meta['array_sha256s'] == dict(prob=pc_stage1.array_sha256(prob)), 'held array hash mismatch'
    return meta


def infer_held(root, cut, seed, names, *, device=None, store=None):
    """float32 probabilities [len(names), K] in `names` order from the saved mapper (image-only)."""
    meta = verify_stage2(root, cut, seed)
    config = meta['config']
    device = config['device'] if device is None else device
    assert str(device) == config['device'], 'inference device differs from training device'
    assert len(names) == len(set(names)) > 0
    store = store or PatchStore()
    rows = store.rows(names)
    base.determinism(seed)
    with threadpool_limits(limits=8):
        mapper = pc_sept.AttentionPoolingMapper(d_model=config['input_dim'], n_topics=meta['K'])
        mapper.load_state_dict(torch.load(stage2_path(root, cut, seed).with_suffix('.pt'), map_location='cpu',
                                          weights_only=True)['mapper'])
        mapper.to(device).eval()
        prob = predict(mapper, lambda a, b: store.fetch(rows[a:b]), len(rows), device)
    assert prob.shape == (len(names), meta['K']) and prob.dtype == np.float32 and base.in_unit_interval(prob, True)
    return prob


def run_held(root, cut, seed, names, *, device=None, smoke=False, store=None):
    previous = verify_stage1(root, seed)
    assert previous['config']['smoke'] == smoke, 'Stage 1 smoke mismatch'
    if previous['K_by_cut'][cut] < 2:
        print(f'Skipping held {cut} seed={seed}: only {previous["K_by_cut"][cut]} topic', flush=True)
        return None
    source = verify_stage2(root, cut, seed)
    assert source['config']['smoke'] == smoke, 'Stage 2 smoke mismatch'
    dev = source['config']['device'] if device is None else str(device)
    assert dev == source['config']['device'], 'inference device differs from training device'
    path = held_path(root, cut, seed)
    try:
        verify_held(path, source, names, dev)
        print(f'Verified complete: {path}', flush=True)
        return path
    except INCOMPLETE as error:
        if path.exists():
            print(f'Incomplete {path}: {error}', flush=True)
    prob = infer_held(root, cut, seed, names, device=device, store=store)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez_compressed(pending, prob=prob)
    pending.replace(path)
    meta = dict(config=source['config'], K=source['K'], held_sha256=ct_held.names_sha256(names), rows=len(names),
                weights_sha256=source['weights_sha256'], inference_device=dev, nonfinite=bool(not np.isfinite(prob).all()),
                array_sha256s=dict(prob=pc_stage1.array_sha256(prob)))
    base.pc_stage2.write_sidecar(path.with_suffix('.json'), meta)
    return path


def split():
    """The full-mode split (fit = 42,969 train paintings, dev = 6,152 validation paintings)."""
    s = ct2_data.split2('full')
    assert (len(s['fit']), len(s['dev'])) == (P.N_TRAIN, P.N_VAL)
    return s
