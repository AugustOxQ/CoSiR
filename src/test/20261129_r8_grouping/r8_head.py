"""CPU contrastive spaces V*, NB*, NBC (decision-rule M5/M6).

Array computations are independent of r8_common; file-level build imports its
shared M1/M2 joins lazily. No item labels, selection fit, or held inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IDS = ('V100', 'V67', 'V33', 'NB100', 'NB67', 'NB33', 'NBC')
OWN_COUNTS = {100: 128, 67: 86, 33: 42}
VIEWS = Path('/data/SSD2/pre_extract/artelingo/r8_clip_views/views.npy')
CSD = Path('/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy')
CSD_SHA256 = '7eefe522383b753bad32f404651103471f9d5b6c41926ee57bfc7386c84818ad'
SHARED_INPUTS = {
    '/data/SSD2/pre_extract/artelingo/features/metadata.json':
        '43963ca92f46ee638ace94f8ff72807de29ec87471aa5b68904bdc4afb7e8011',
    '/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json':
        '6013b2f801eb1e6ae104e9fb3c71a377890bb9aab58e4cb27991c269febdf13b',
    '/data/SSD2/pre_extract/artelingo/style_csd_vitl/row_to_image.npy':
        '914f4e621ffecd4f0bcb7bc13465544553f7d48646668f02a0ffe9438050ab10',
    'src/test/20261031_pseudo_partitions/results/partitions.npz':
        'cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa',
    'src/test/20261117_reader_fix_csd/results/rb_halves.npz':
        'ed2d359e6517b4237f8aa4de5e78f787699b366a975ac993eaef0e7c4491c7da',
}


def neighbours(x, content, block_size=512):
    """Fixed other-content argmax in float64; ascending M1 index breaks ties."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all()
    assert content.dtype == np.int64 and content.shape == (len(x),)
    assert len(np.unique(content)) >= 2 and block_size > 0
    result = np.empty(len(x), dtype=np.int64)
    for start in range(0, len(x), block_size):
        stop = min(start + block_size, len(x))
        scores = x[start:stop] @ x.T
        scores[content[start:stop, None] == content[None, :]] = -np.inf
        result[start:stop] = np.argmax(scores, axis=1)
    assert np.all(content[result] != content)
    return result


def negative_pools(content):
    """Ascending M1 pools, cached per group; exclusions are query-specific."""
    return {int(c): (np.flatnonzero(content == c), np.flatnonzero(content != c))
            for c in np.unique(content)}


def batch_draws(rng, batch, content, n, n_own, pools=None):
    """M6 RNG order, also consuming view draws for NB/NBC batches."""
    assert 0 <= n_own <= 128
    assert content.dtype == n.dtype == np.int64 and content.shape == n.shape
    assert np.all((n >= 0) & (n < len(content)))
    assert np.all(content[n] != content)
    pools = negative_pools(content) if pools is None else pools
    views = rng.integers(0, 5, size=len(batch))
    negative = np.empty((len(batch), 128), dtype=np.int64)
    for j, p in enumerate(batch):
        own, other = pools[int(content[p])]
        own = own[(own != p) & (own != n[p])]
        other = other[(other != p) & (other != n[p])]
        assert len(own) >= n_own and len(other) >= 128 - n_own
        negative[j, :n_own] = rng.choice(own, n_own, replace=False)
        negative[j, n_own:] = rng.choice(other, 128 - n_own, replace=False)
    return views, negative


def info_nce(query, positive, negative):
    """Mean M6 InfoNCE, including gradients through every negative."""
    assert query.ndim == 2 and positive.shape == query.shape
    assert negative.ndim == 3 and negative.shape[0] == len(query)
    assert negative.shape[2] == query.shape[1]
    pos = (query * positive).sum(dim=-1, keepdim=True)
    neg = torch.einsum('bd,bnd->bn', query, negative)
    logits = torch.cat((pos, neg), dim=1) / 0.1
    return F.cross_entropy(logits, torch.zeros(len(query), dtype=torch.long))


class Head(nn.Module):
    """M6's default-initialised 512 → 512 → 128 head, followed by unit."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))

    def forward(self, x):
        return F.normalize(self.layers(x), dim=-1)


def assert_unit(x, shape):
    assert x.dtype == np.float32 and x.shape == shape
    assert np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x, axis=-1), 1, atol=1e-6, rtol=0)


def embed(head, x, batch_size=1024):
    """Evaluate f(x) in input order, including selection images after fitting."""
    assert_unit(x, (len(x), 512))
    assert batch_size > 0 and len(x) > 0
    head.eval()
    with torch.no_grad():
        phi = np.concatenate([head(torch.from_numpy(x[j:j + batch_size])).numpy()
                              for j in range(0, len(x), batch_size)])
    assert_unit(phi, (len(x), 128))
    return phi


def train_head(x, content, n, n_own, views=None, epochs=30, batch_size=512,
               anchors=None, progress=False):
    """Fit M6 from arrays; optional anchors/batch size are for synthetic/smoke runs.

    NB/NBC still consume the unused view draws, as M6 specifies one RNG sequence.
    Every occurrence of a negative is embedded with gradient.
    """
    assert_unit(x, (len(x), 512))
    assert content.dtype == n.dtype == np.int64 and content.shape == n.shape == (len(x),)
    assert np.all((n >= 0) & (n < len(x))) and np.all(content[n] != content)
    assert n_own in (128, 86, 42) and epochs > 0 and batch_size > 0
    if views is not None:
        assert_unit(views, (len(x), 5, 512))
    anchors = np.arange(len(x), dtype=np.int64) if anchors is None else np.asarray(anchors)
    assert anchors.ndim == 1 and len(anchors) > 0
    assert anchors.dtype == np.int64 and np.all((anchors >= 0) & (anchors < len(x)))
    assert np.all(np.diff(anchors) > 0)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    start = perf_counter()
    head = Head().cpu()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    pools = negative_pools(content)
    xt = torch.from_numpy(x)
    vt = torch.from_numpy(views) if views is not None else None
    losses, seconds = [], []
    for epoch in range(epochs):
        epoch_start = perf_counter()
        order = anchors[rng.permutation(len(anchors))]
        total = 0.0
        head.train()
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            v, negative = batch_draws(rng, batch, content, n, n_own, pools)
            optimizer.zero_grad()
            query = head(xt[batch])
            positive = head(vt[batch, v] if vt is not None else xt[n[batch]])
            neg = head(xt[negative.reshape(-1)]).reshape(len(batch), 128, 128)
            loss = info_nce(query, positive, neg)
            assert torch.isfinite(loss).item(), 'nonfinite InfoNCE'
            loss.backward()
            optimizer.step()
            # rule gap: "epoch mean" does not specify partial-batch weighting;
            # use the mean over queries, weighting each batch by its size.
            total += loss.item() * len(batch)
        losses.append(total / len(order))
        seconds.append(perf_counter() - epoch_start)
        if progress:
            print(f'epoch {epoch + 1}/{epochs}: {seconds[-1]:.1f}s', flush=True)
    return head, {'last_epoch_mean_loss': losses[-1], 'epoch_losses': losses,
                  'epoch_seconds': seconds, 'training_wall_seconds': perf_counter() - start,
                  'epochs': epochs, 'batch_size': batch_size,
                  'batches_per_epoch': (len(anchors) + batch_size - 1) // batch_size,
                  'n_queries': len(anchors), 'n_pool_paintings': len(x),
                  'torch_seed': 0, 'numpy_seed': 0, 'threads': 8,
                  'deterministic_algorithms': True}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def read_views(path, painting_ids):
    """Check ticket 03's output hash, M1 painting alignment, and M4 view check.

    Expected metadata: painting_ids, views_sha256, view_check.passed.
    """
    path = Path(path)
    meta_path = path.with_name('meta.json')
    meta = json.loads(meta_path.read_text())
    # rule gap: ticket 03 specifies metadata contents but no JSON key schema;
    # this boundary uses painting_ids, views_sha256, and view_check.passed.
    assert {'painting_ids', 'views_sha256', 'view_check'} <= meta.keys()
    assert meta['view_check']['passed'] is True, 'M4 failed: V spaces must not run'
    assert np.array_equal(np.asarray(meta['painting_ids']), painting_ids), 'views M1 alignment'
    digest = sha_file(path)
    assert digest == meta['views_sha256'], 'views output SHA-256'
    views = np.load(path, mmap_mode='r', allow_pickle=False)
    assert views.dtype == np.float32 and views.shape == (len(painting_ids), 5, 512)
    # Materialise once: writable arrays avoid PyTorch's read-only mmap warning.
    views = np.array(views)
    norms = np.linalg.norm(views, axis=-1, keepdims=True)
    assert np.isfinite(views).all() and np.all(norms > 0)
    views /= norms  # M6 explicitly embeds unit views; M4 may store raw features.
    assert_unit(views, (len(painting_ids), 5, 512))
    return views, {str(path): digest, str(meta_path): sha_file(meta_path)}


def read_csd(painting_image):
    """Read only scorer-train CSD vectors via the shared M1 image mapping."""
    digest = sha_file(CSD)
    assert digest == CSD_SHA256, 'CSD embeddings SHA-256 (rule section 10)'
    source = np.load(CSD, mmap_mode='r', allow_pickle=False)
    assert source.dtype == np.float32 and source.shape == (61402, 768)
    x = np.array(source[painting_image], dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and np.all(norm > 0)
    x /= norm
    assert_unit(x, (len(painting_image), 768))
    return x, {str(CSD): digest}


def check_nbc_trigger(path, strength):
    """NBC cannot start before its controller's C13-checked trigger record."""
    record = json.loads(Path(path).read_text())
    # rule gap: section 6.3 does not define strength's JSON representation;
    # require the numeric fraction (1.0, 0.67, 0.33), as the CLI argument.
    assert record['triggered'] is True and record['strength'] == strength
    assert record['rule_sha256'] == sha_file(Path(__file__).with_name('DECISION_RULE.md'))
    return {str(path): sha_file(path)}


def build(only=('NB100',), strength=None, smoke=False, out=None, views_path=VIEWS,
          time_one_epoch=False):
    """Write spaces and their training/provenance records through r8_common."""
    # rule gap: ticket 01 was unavailable for this ticket's file-level smoke
    # and real-epoch timing; the controller must run both after integration.
    import r8_common as C

    start = perf_counter()
    C.assert_rule()
    C.assert_inputs()
    input_hashes = {}
    for key, expected in SHARED_INPUTS.items():
        path = Path(key) if key.startswith('/') else Path(C.MAIN) / key
        digest = sha_file(path)
        assert digest == expected, f'{path}: section 10 SHA-256'
        input_hashes[str(path)] = digest
    assert only and len(set(only)) == len(only) and set(only) <= set(IDS)
    assert not (smoke and time_one_epoch)
    assert ('NBC' in only) == (strength is not None), 'NBC requires --strength; others do not'
    if strength is not None:
        assert strength in (1.0, 0.67, 0.33)
    is_smoke = smoke or time_one_epoch
    root = (Path(out) if out is not None else (C.SMOKE if is_smoke else C.RESULTS)).resolve()
    # Explicit worktree output is supported without writing to MAIN during implementation.
    allowed = (Path(C.SMOKE), Path(__file__).parent / 'results' / 'smoke')
    if is_smoke:
        assert any(root.resolve().is_relative_to(p.resolve()) for p in allowed), 'smoke output root'
    else:
        allowed = (Path(C.RESULTS), Path(__file__).parent / 'results')
        assert any(root.resolve().is_relative_to(p.resolve()) for p in allowed), 'results output root'
        assert 'smoke' not in root.relative_to(next(
            p.resolve() for p in allowed if root.is_relative_to(p.resolve()))).parts
    targets = [(root / 'spaces' / f'{name}.npz', root / 'spaces' / f'{name}.json') for name in only]
    assert all(not path.exists() for pair in targets for path in pair), 'write-once space output exists'
    ctx = C.ctx()
    assert ctx.n_paintings == 36518
    assert ctx.scorer_train.dtype == np.int64 and ctx.scorer_train.shape == (183694,)
    assert ctx.g_st.dtype == np.int64 and ctx.g_st.shape == ctx.scorer_train.shape
    assert ctx.first_rows.shape == (36518,)
    assert np.array_equal(ctx.g_st[ctx.first_rows], np.arange(36518)), 'M1 painting order'
    assert ctx.content.dtype == np.int64 and ctx.content.shape == (36518,)
    assert len(np.unique(ctx.content)) == 64
    assert min(np.sum(ctx.content == c) for c in np.unique(ctx.content)) >= 129
    painting_ids = np.asarray(ctx.painting_ids)
    assert painting_ids.shape == (36518,) and len(np.unique(painting_ids)) == 36518
    x = C.x_paintings()
    assert_unit(x, (36518, 512))
    image_index = C.nonheld_images()
    assert image_index.dtype == np.int32 and image_index.shape == (49121,)
    assert np.all(np.diff(image_index) > 0), 'ascending non-held image order'
    painting_image = C.painting_image()
    assert painting_image.dtype == np.int32 and painting_image.shape == (36518,)
    row_image = C.row_to_image()
    assert row_image.dtype == np.int32 and row_image.shape == (308723,)
    assert np.array_equal(painting_image, row_image[ctx.scorer_train[ctx.first_rows]])
    pos = np.searchsorted(image_index, painting_image)
    assert np.all(pos < len(image_index)) and np.array_equal(image_index[pos], painting_image)
    images = C.x_images()
    assert_unit(images, (49121, 512))
    assert np.array_equal(images[pos], x), 'CLIP painting/image first-row alignment'
    inputs = {**input_hashes, 'x_paintings_array': sha_array(x),
              'x_images_array': sha_array(images), 'content_array': sha_array(ctx.content),
              'painting_image_array': sha_array(painting_image),
              'image_index_array': sha_array(image_index)}
    view_array = None
    if any(name.startswith('V') for name in only):
        view_array, hashes = read_views(views_path, painting_ids)
        inputs.update(hashes)
    if 'NBC' in only:
        inputs.update(check_nbc_trigger(Path(C.RESULTS) / 'probe_trigger.json', strength))
    neighbour_arrays = {}
    if any(name != 'NBC' for name in only):
        neighbour_arrays['CLIP'] = neighbours(x, ctx.content)
    if 'NBC' in only:
        csd, hashes = read_csd(painting_image)
        inputs.update(hashes)
        neighbour_arrays['CSD'] = neighbours(csd, ctx.content)
    # rule gap: first-2048 smoke groups cannot supply 128 own negatives;
    # restrict queries to the prefix, retaining full M1 pools/neighbours.
    anchors = np.arange(2048, dtype=np.int64) if smoke else None
    records = {}
    for name, (space_path, record_path) in zip(only, targets):
        run_start = perf_counter()
        percent = int(round(100 * strength)) if name == 'NBC' else int(name.lstrip('VNB'))
        n_own = OWN_COUNTS[percent]
        neighbour_source = 'CSD' if name == 'NBC' else 'CLIP'
        n = neighbour_arrays[neighbour_source]
        print(f'{name}: CPU head, {2048 if smoke else 36518} queries', flush=True)
        head, training = train_head(x, ctx.content, n, n_own,
                                   views=view_array if name.startswith('V') else None,
                                   epochs=1 if is_smoke else 30, anchors=anchors, progress=True)
        assert training['batches_per_epoch'] == (4 if smoke else 72)
        phi = embed(head, images)
        C.write_npz_once(space_path, phi=phi, image_index=image_index)
        record = {'builder': 'r8_head', 'setting': name, 'smoke': is_smoke,
                  'timing_only': time_one_epoch, 'strength': percent / 100,
                  'n_own': n_own, 'n_other': 128 - n_own,
                  'positive': 'view' if name.startswith('V') else 'neighbour',
                  'neighbour_source': neighbour_source, 'neighbour_array_sha256': sha_array(n),
                  'input_sha256s': inputs, 'space_sha256': sha_file(space_path),
                  'phi_shape': list(phi.shape), 'training': training,
                  'run_wall_seconds': perf_counter() - run_start,
                  'invocation_wall_seconds': perf_counter() - start,
                  'loss_epoch_average': 'query_weighted'}
        records[name] = C.write_json_once(record_path, record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=IDS, default=['NB100'])
    parser.add_argument('--strength', type=float, choices=(1.0, 0.67, 0.33))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_true', help='1 epoch, first 2048 queries')
    modes.add_argument('--time-one-epoch', action='store_true',
                       help='1 full-data epoch; outputs are smoke/timing objects only')
    parser.add_argument('--out', type=Path, help='results root (smoke requires results/smoke)')
    parser.add_argument('--views', type=Path, default=VIEWS)
    args = parser.parse_args()
    build(args.only, args.strength, args.smoke, args.out, args.views, args.time_one_epoch)


if __name__ == '__main__':
    main()
