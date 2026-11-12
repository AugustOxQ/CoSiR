"""r29 photo model F1: the Affection SigLIP 1 caption store and the photo trainer (ticket 12; spec §3.2 F1, D14 to D16,
§3.5 gates (vii) trainer part and (viii)).

F1 is r14's photo model (`bs_photo`: r13's final photo settings, r13's train set and graphs, training seed 100 + s) with
ONE input changed: the caption mean D that enters the loss is the L2-normalised mean of a photo's default explanations
encoded by SigLIP 1 (width 1,152) in K*'s caption form (D16), instead of Qwen3's (2,048). Seed s starts from K*'s seed-s
emotion checkpoint `OUT/cstar/siglip1/weights/<K* arm>/emotion__s<s>.pt`. Nothing of r13 or r14 is edited: the changes are
`unittest.mock.patch`es that unwind on success or failure (as `r29_trainer`, `r28_trainer`). Where each sits:

  * the model: `bs_photo.build_emotion` (and so `bs_photo.infer` of the exports) builds the caption trunk at 1,152 through
    `r29_model.build_model`; `bs_photo._tensors` checks the caption array at 1,152. At width 2,048 (the Qwen3 caption
    input, used by gate (vii) only) nothing is patched: r14's own objects;
  * the start point: `bs_photo.start_root` is K*'s arm folder; `pf_train.checked_load` first asserts the start
    checkpoint's sidecar (`cap_width`, `cap_source`, `cap_store_sha256`, `width`) against this model;
  * the run record: `run.json` gets `cap_source`, `cap_width`, `cap_store_sha256` (the Affection store's index hash),
    `cap_sha256` and this code's hash.
r13's graphs, the near-copy cut, the teacher T and the image I are read as r14 reads them (Qwen3-built graphs; SigLIP 1
default image). Exports are r14's: `OUT/photo/siglip1/s<s>/{emotion.pt, run.json}` and
`OUT/exports/affection/photo_siglip1/<split>__s<s>.npz` (+ sidecar).

The store: `r29_paths.AFFECTION/siglip_so400m_txt/` (D15, a new folder), one SigLIP 1 text row per row of the Qwen3
Affection store (526,749 explanations), same `index.json` text rows (`img`, `csv_row`, `split`, ...), K*'s form (plain or
prefixed: `records/prefix_pick.json` `kstar_form`). Rows join to the Qwen3 store by `csv_row` with the photo checked.

CLI (after the merge, from the integration worktree so records land there; set HF_HOME=/data/SSD2/HF_home
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1):
  r29_photo store     [--device cuda] [--out DIR]     build the store in K*'s form and write records/gates/viii_affection.json
  r29_photo gate-viii [--store DIR]                    check an existing store again (one more try)
  r29_photo gate-vii  [--device cpu] [--n-fit N]       photo trainer against r14's on a fixed batch: records/gates/vii_trainer.json
  r29_photo train     [--seeds ..] [--device cuda]     F1, seeds from K*'s checkpoints (refuses unless pass_rule.json says pass)
  r29_photo export    [--seeds ..] [--splits val test] F1's Affection exports
  r29_photo smoke     [--kstar siglip1] [--device cuda] a small end-to-end run under SMOKE_OUT/t12 (writes no records)
"""
import argparse
import hashlib
import json
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P  # noqa: E402
P.use_older_rounds()
import r29_captions as C  # noqa: E402
import r29_extract as X  # noqa: E402
import r29_model as M  # noqa: E402
import r29_trainer as T  # noqa: E402  (puts r14's folder and the older rounds on sys.path; check_root)
import bs_photo as BP  # noqa: E402  (r14's photo trainer)
import pf_data as PD  # noqa: E402  (r13's photo data provider)
import pf_graphs  # noqa: E402
import rs_vh  # noqa: E402
P.check_older_imports()  # every older-round module loaded so far is the main checkout's copy

ENC = 'siglip1'                                           # bs_paths' encoder key of F1's image tower
CAP_WIDTH = 1152
IMG_WIDTH = 1152                                          # bs_paths.ENCODERS['siglip1']['width'] (asserted below)
FORMS = {'siglip1': 'plain', 'siglip1_prefix': 'prefixed'}  # K* arm -> its caption form
STORE = P.AFFECTION / 'siglip_so400m_txt'                 # D15: a new folder
DEFAULT_INDEX = P.AFFECTION / 'qwen3vl_emb_2b' / 'index.json'   # the Qwen3 default store: keys and explanation list
AFFECTION_CSV = Path('/data/SSD/affection/language/version_0/affection_preprocessed_public_version_0.csv')
PASS_RULE = P.RECORDS / 'pass_rule.json'
PICK_FILE = P.RECORDS / 'prefix_pick.json'
GATE_VIII = 'viii_affection'
GATE_VII = 'vii_trainer'
MAX_TRIES = 3
SMOKE_ROOT = P.SMOKE_OUT / 't12'
assert BP.P.ENCODERS[ENC]['width'] == IMG_WIDTH
_BUILD_EMOTION = BP.build_emotion                          # r14's, kept for width 2,048 and for unwinding


def code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def store_hash(folder):
    """sha256 of the store's `index.json` bytes (as `r29_captions.store_hash` does for the ArtELingo stores)."""
    return X._sha(Path(folder) / 'index.json')


def _json(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------------------------------------------- guards
def read_prefix_pick(path=None):
    """Ticket 05's `records/prefix_pick.json`: the keys and their consistency are asserted (index, Shared interfaces)."""
    doc = _json(PICK_FILE if path is None else path)
    missing = [k for k in ('kstar', 'kstar_form', 'p1_form', 'flag') if k not in doc]
    assert not missing, f'prefix_pick.json lacks {missing}'
    assert doc['kstar'] in FORMS, f"kstar {doc['kstar']!r}: not one of {sorted(FORMS)}"
    assert doc['kstar_form'] == FORMS[doc['kstar']], f"kstar {doc['kstar']!r} is the {FORMS[doc['kstar']]} form"
    assert doc['p1_form'] in ('plain', 'prefixed'), doc['p1_form']
    return doc


def require_pass(path=None):
    """D3: F1 is stage 2: only after `records/pass_rule.json` says pass."""
    path = Path(PASS_RULE if path is None else path)
    assert path.is_file() and _json(path).get('status') == 'pass', \
        f'F1 is a stage 2 piece: needs {path} with status "pass" (D3)'


# ----------------------------------------------------------------------------------- the store and gate (viii)
def affection_texts(reference, csv_path=AFFECTION_CSV):
    """The raw utterance of every text row of `reference`, in its order, joined by `csv_row`: the row of the preprocessed
    CSV must be the row's photo (source and image name) and split. Only these fields are read: no label, no worker."""
    import pandas as pd
    pre = pd.read_csv(csv_path, usecols=['image_name', 'dataset', 'utterance', 'split'], dtype=str)
    images = reference['images']
    out = []
    for row in reference['texts']:
        c, image = row['csv_row'], images[row['img']]
        assert 0 <= c < len(pre), f'csv_row {c} is outside the CSV ({len(pre)} rows)'
        assert (pre.dataset[c], pre.image_name[c]) == (image['source'], image['image_name']), \
            f'photo mismatch at csv_row {c}'
        assert pre.split[c] == row['split'], f'split mismatch at csv_row {c}'
        text = pre.utterance[c]
        assert isinstance(text, str) and text.strip(), f'empty utterance at csv_row {c}'
        out.append(text)
    return out


def build_store(reference, texts, enc, out, *, form, smoke=False):
    """The SigLIP 1 text store of `reference`'s explanations in K*'s form, through ticket 02's extractor (the prefix goes on
    the raw text once; the extractor owns the 64-token cut, its count and the float16 rows)."""
    assert form in ('plain', 'prefixed')
    return X.extract_store(reference, texts, enc, out, prefix=P.PREFIX if form == 'prefixed' else '', smoke=smoke,
                           scheme='affection')


def _photo_counts(index):
    counts = {}
    for owner in X._keys(index, 'affection').values():
        counts[owner] = counts.get(owner, 0) + 1
    return counts


def gate_viii(store, reference, *, form, record_folder=None, allow_smoke=False):
    """Gate (viii) on the Affection store: one row per key, finite, unit norm within 1e-3, the keys join to the Qwen3
    store's by `csv_row` and photo, the row and per-photo counts equal the Qwen3 store's, the stored prefix is K*'s form,
    truncation counted (D14: cut at 64 tokens, counted and flagged, rows kept). Writes `viii_affection.json`
    (`record_folder`, default records/gates); at most three tries."""
    folder = Path(P.GATES if record_folder is None else record_folder)
    path = folder / f'{GATE_VIII}.json'
    tries = _json(path)['tries'] + 1 if path.exists() else 1
    assert tries <= MAX_TRIES, f'gate {GATE_VIII} exhausted its three tries'
    details, status = dict(form=form, store=str(store)), 'fail'
    try:
        check = X.check_store(store, reference, expected_prefix=P.PREFIX if form == 'prefixed' else None,
                              allow_smoke=allow_smoke, scheme='affection')
        index = _json(Path(store) / 'index.json')
        assert index['images'] == reference['images'], 'photo list differs from the reference'
        counts, ref_counts = _photo_counts(index), _photo_counts(reference)
        assert counts == ref_counts, 'per-photo explanation counts differ from the Qwen3 store'
        n = check['n_truncated']
        details.update(prefix=index['prefix'], n_texts=check['n_texts'], n_texts_reference=reference['n_texts'],
                       photo_counts_equal=True, n_photos=len(counts), n_truncated=n,
                       truncation_limit=index['truncation_limit'], index_sha256=check['index_sha256'],
                       max_norm_dev=check.get('max_norm_dev'),
                       truncation_flag=None if n == 0 else
                       f'{n} of {check["n_texts"]} explanations are over {index["truncation_limit"]} tokens: cut, counted, kept (D14)')
        status = 'pass'
    except Exception as error:
        details['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        C.write_gate_record(folder, dict(status=status, details=details), gate=GATE_VIII)
        # `write_gate_record` counted the file's earlier runs itself: it agrees with `tries`
    return _json(path)


# ------------------------------------------------------------------------------------------- the caption mean (D16)
def caption_means(keys, store, *, width=CAP_WIDTH, chunk=4096):
    """D of F1: per photo key (in the given order), the mean of its eligible default explanations' rows of `store`, L2
    normalised; float32 [n, width]. Eligible rows are r13's (`pf_data._eligible_rows`: the row's own split is not
    "ignored"); each is found in the store by `csv_row`. The arithmetic is r13's `_group_mean` (float64 sum in row order,
    divided by the count, cast float32, then the CT's L2) at this width."""
    keys = list(keys)
    rows = PD._eligible_rows(keys)
    folder = Path(store)
    index = _json(folder / 'index.json')
    where = {t['csv_row']: i for i, t in enumerate(index['texts'])}
    assert len(where) == len(index['texts']), 'duplicate csv_row in the store'
    txt = np.load(folder / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert txt.dtype == np.float16 and txt.shape == (len(where), width), f'caption store width/shape {txt.shape}'
    csv = [int(c) for c in rows['csv_row']]
    missing = [c for c in csv if c not in where]
    assert not missing, f'csv_row not in the caption store: {missing[:3]}'
    pos = np.asarray([where[c] for c in csv], dtype=np.int64)
    slot = {k: i for i, k in enumerate(keys)}
    kidx = np.asarray([slot[k] for k in rows['key']], dtype=np.int64)
    count = np.bincount(kidx, minlength=len(keys))
    assert (count > 0).all(), f'a key has no eligible row: {[keys[i] for i in np.flatnonzero(count == 0)[:3]]}'
    total = np.zeros((len(keys), width), dtype=np.float64)
    for a in range(0, len(kidx), chunk):
        np.add.at(total, kidx[a:a + chunk], np.asarray(txt[pos[a:a + chunk]]).astype(np.float64))
    mean = (total / count[:, None]).astype(np.float32)
    out = PD.rt_data.ct2_data.l2(mean)
    assert out.shape == (len(keys), width) and out.dtype == np.float32 and np.isfinite(out).all()
    assert np.abs(np.linalg.norm(out, axis=1) - 1).max() < 1e-4, 'caption mean not unit'
    return out


def load_data(source, store=STORE):
    """r14's `load_data('siglip1')` (train set of 69,927 photos after the near-copy cut and the 4,251 validation photos:
    SigLIP 1 default image `I`, instructed teacher `T`, Qwen3 `D`; `fit_q`, `val_q` are r13's Qwen arrays for the graphs),
    with `D` replaced by the store's caption means when `source` is not `qwen`."""
    fit, val, fit_q, val_q = BP.load_data(ENC)
    if source != 'qwen':
        fit, val = (dict(x, D=caption_means(x['keys'], store)) for x in (fit, val))
    return fit, val, fit_q, val_q


# ------------------------------------------------------------------------------------------------ the trainer patches
def build_emotion(device, width, cap_width, seed=None):
    """`bs_photo.build_emotion(device, width, seed)` with the caption trunk's first layer taking `cap_width`. At 2,048 it
    is r14's call itself."""
    if cap_width == P.OUTPUT_WIDTH:
        return _BUILD_EMOTION(device, width, seed)
    row, _ = BP.loss_recipe(0.25)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    models, views = M.build_model(row, list(BP.pf_train.ASPECTS), device, width=width, cap_width=cap_width)
    return models, views['emotion']


def build_state(width, cap_width, seed):
    """CPU state dict of a freshly initialised emotion view (the checkpoint layout, `pf_train.CKPT_KEYS`)."""
    _, view = build_emotion(torch.device('cpu'), width, cap_width, seed)
    return {k: v.detach().cpu() for k, v in view.state_dict().items()}


def _tensors(cap_width):
    def tensors(part, device, width):
        """`bs_photo._tensors` with the caption array at `cap_width`: I [n, width], D [n, cap_width], T [n, 2048]."""
        out = {}
        for name, key, w in (('img', 'I', width), ('cap', 'D', cap_width), ('teacher', 'T', P.OUTPUT_WIDTH)):
            x = part[key]
            assert x.ndim == 2 and x.shape[1] == w and x.dtype == np.float32, (key, x.shape, x.dtype, w)
            out[name] = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        assert len({len(v) for v in out.values()}) == 1
        return out
    return tensors


def check_start_sidecar(path, source, cap_width, width=IMG_WIDTH):
    """The start checkpoint's sidecar (ticket 03's `rb` block) must say it was trained on this caption source at this
    caption width, against the caption store K*'s ArtELingo model read, with this image width."""
    rb = _json(Path(path).with_suffix('.json'))['config']['rb']
    missing = [k for k in ('width', 'cap_width', 'cap_source', 'cap_store_sha256') if k not in rb]
    assert not missing, f'{path}: sidecar rb block lacks {missing}'
    assert rb['cap_width'] == cap_width, f'{path}: sidecar cap_width {rb["cap_width"]} is not {cap_width}'
    assert rb['cap_source'] == source, f'{path}: sidecar cap_source {rb["cap_source"]!r} is not {source!r}'
    assert rb['cap_store_sha256'] == C.store_hash(source), f'{path}: sidecar cap_store_sha256 is not the store\'s'
    assert rb['width'] == width, f'{path}: sidecar width {rb["width"]} is not {width}'


@contextmanager
def caption_patches(source, cap_info=None):
    """The patches of this module's photo training for caption `source` (`qwen`, `siglip1`, `siglip1_prefix`; module doc).
    `cap_info` (dict) is written into every run.json this context finishes."""
    assert source in P.CAPTION_SOURCES, source
    cap_width = P.CAPTION_SOURCES[source][1]
    base_load, base_run = BP.pf_train.checked_load, BP.train_run

    def build(device, width, seed=None):
        return build_emotion(device, width, cap_width, seed)

    def start_root(enc, cstar_root=None):
        return Path(P.OUT / 'cstar' / enc if cstar_root is None else cstar_root) / 'weights' / source

    def checked_load(view, cstar_seed, root=None):
        check_start_sidecar(Path(root) / f'emotion__s{cstar_seed}.pt', source, cap_width)
        return base_load(view, cstar_seed, root)

    def train_run(*args, **kwargs):
        run = base_run(*args, **kwargs)
        out_dir = kwargs.get('out_dir')
        if out_dir is not None and (Path(out_dir) / 'run.json').is_file():
            fields = dict(cap_source=source, cap_width=cap_width, **cap_info, r29_photo_sha256=code_sha())
            run.record.update(fields)
            BP._atomic_json(Path(out_dir) / 'run.json', run.record)
        return run

    with ExitStack() as stack:
        if cap_width != P.OUTPUT_WIDTH:         # at 2,048 r14's own objects
            stack.enter_context(patch.object(BP, 'build_emotion', build))
            stack.enter_context(patch.object(BP, '_tensors', _tensors(cap_width)))
        if source != 'qwen':
            stack.enter_context(patch.object(BP, 'start_root', start_root))
            stack.enter_context(patch.object(BP.pf_train, 'checked_load', checked_load))
        if cap_info is not None:
            stack.enter_context(patch.object(BP, 'train_run', train_run))
        yield


# --------------------------------------------------------------------------------------------- gate (vii) trainer part
def _batch_loss_and_grads(view, part, pairs, width, settings, seed, device):
    """Loss and parameter gradients of one batch (the first 512 pairs) in the training mode, the RNG seeded."""
    row, switches = BP.loss_recipe(settings['teacher_weight'])
    t = BP._tensors(part, device, width)
    pairs = pairs[:BP.RECIPE['batch_pairs']]
    view.train()
    torch.manual_seed(seed)
    loss = BP.ab_train.switched_head_loss(view, **t, pairs=pairs, weights=np.ones(len(pairs), dtype=np.float32),
                                          row=row, aspect='emotion', switches=switches)
    loss.backward()
    return loss.detach().clone(), [p.grad.detach().clone() for p in view.parameters()]


def _trainer_side(fit, val, graph, settings, width, device, seed, cstar_root):
    """One trainer's observations: the model state at the start, the loss and gradients of the first batch, and the
    state, epoch, development loss and history of the training itself."""
    _, view = BP.build_emotion(device, width, seed=seed)
    if settings['start'] == 'cstar':
        BP.pf_train.checked_load(view, 0, cstar_root)
    start = {k: v.detach().cpu().clone() for k, v in view.state_dict().items()}
    loss, grads = _batch_loss_and_grads(view, fit, graph[0], width, settings, seed, device)
    state, epoch, dev, history = BP.fit_photo('compare', settings, width, 0, seed, fit, val, graph, device,
                                              cstar_root=cstar_root)
    losses = [(h['epoch'], h['train_loss'], h['dev_loss']) for h in history]
    return dict(start=start, loss=loss, grads=grads, state=state, trained=(epoch, dev, losses))


def compare_trainers(fit, val, graph, settings, width, device, *, seed, cstar_root):
    """Gate (vii), trainer part: r14's photo trainer (unpatched) and this module's with the Qwen3 caption input, on the same
    arrays, graph pairs, start and seed: model state at the start, loss and gradients on a fixed batch, the trained state
    and the training history, each exactly equal. Returns {check: bool}."""
    a = _trainer_side(fit, val, graph, settings, width, device, seed, cstar_root)
    with caption_patches('qwen'):
        b = _trainer_side(fit, val, graph, settings, width, device, seed, cstar_root)
    same_state = lambda x, y: tuple(x) == tuple(y) and all(torch.equal(x[k], y[k]) for k in x)   # noqa: E731
    return dict(start_state=same_state(a['start'], b['start']), loss=bool(torch.equal(a['loss'], b['loss'])),
                gradients=len(a['grads']) == len(b['grads']) and all(map(torch.equal, a['grads'], b['grads'])),
                trained_state=same_state(a['state'], b['state']), history=a['trained'] == b['trained'])


def write_gate_vii(results, details, folder=None):
    status = 'pass' if results and all(results.values()) else 'fail'
    return C.write_gate_record(P.GATES if folder is None else folder,
                               dict(status=status, details=dict(details, results=results)), gate=GATE_VII)


def random_pairs(n, count, seed):
    """Sorted unique local pairs i < j of n rows (a fixed stand-in graph for a fixed-batch comparison)."""
    rng = np.random.default_rng(seed)
    found = {tuple(sorted(map(int, rng.choice(n, 2, replace=False)))) for _ in range(count)}
    return np.array(sorted(found), dtype=np.int64)


def gate_vii(device='cpu', n_fit=1024, n_val=256, write=True):
    """Gate (vii), trainer part, on real data: the first `n_fit` sorted fit photos and `n_val` validation photos (SigLIP 1
    image, Qwen3 caption mean, instructed teacher), random fixed pairs, K0's SigLIP 1 seed-0 emotion checkpoint as the start
    (r14's own start point), r13's final settings for exactly one epoch, on `device` (CPU by default: no CUDA
    nondeterminism in the comparison)."""
    folder = P.GATES
    if write and (folder / f'{GATE_VII}.json').exists():
        assert _json(folder / f'{GATE_VII}.json')['tries'] < MAX_TRIES, f'gate {GATE_VII} exhausted its three tries'
    fk, vk = PD.fit_keys()[:n_fit], PD.keys('val')[:n_val]
    part = lambda keys: dict(keys=keys, I=BP.images('affection', ENC, keys), D=PD._default_texts(keys),   # noqa: E731
                             T=PD._teacher_images(keys))
    fit, val = part(fk), part(vk)
    graph = (random_pairs(n_fit, 4 * n_fit, 1), random_pairs(n_val, 4 * n_val, 2))
    settings = dict(BP.PICK, epochs=1)
    start = P.K0_ROOT / 'weights' / P.K0_ARM
    results = compare_trainers(fit, val, graph, settings, IMG_WIDTH, torch.device(device),
                               seed=BP.train_seed(0), cstar_root=start)
    details = dict(n_fit=n_fit, n_val=n_val, device=device, start=str(start), settings=settings, seed=BP.train_seed(0),
                   n_fit_pairs=len(graph[0]), n_dev_pairs=len(graph[1]))
    print(json.dumps(results))
    return write_gate_vii(results, details) if write else dict(details=dict(details, results=results))


# ----------------------------------------------------------------------------------------------- train, export
def train_arm(kstar, seeds, device, root, *, data=None, graphs=None, max_epochs=None, cap_info=None, smoke=False):
    """F1's photo models for `seeds`, start K*'s checkpoints under `<root>/cstar/siglip1/weights/<kstar>/`; r14's layout
    under `root`. Refuses unless the pass rule says pass (a smoke, under the smoke folder only, is exempt)."""
    assert kstar in FORMS, kstar
    if not smoke:
        require_pass()
    root = T.check_root(root, smoke)
    if data is None:
        data = load_data(kstar)
    fit, val, fit_q, val_q = data
    if cap_info is None:
        cap_info = dict(cap_store_sha256=store_hash(STORE), cap_sha256=dict(fit=rs_vh.array_sha(fit['D']),
                                                                             val=rs_vh.array_sha(val['D'])))
    graphs = graphs if graphs is not None else BP.Graphs(fit_q, val_q, root)
    with caption_patches(kstar, cap_info):
        return BP.train(ENC, [BP.REF], list(seeds), device, keep='all', root=root, max_epochs=max_epochs, data=data,
                        graphs=graphs, cstar_root=root / 'cstar' / ENC)


def export_arm(kstar, seeds, splits, root, device=None):
    """r14's Affection export of the photo models (arm `photo_siglip1`) at the caption width of `kstar`."""
    require_pass()
    root = T.check_root(root)
    with caption_patches(kstar):
        return BP.export_arm('affection', f'photo_{ENC}', splits, list(seeds), root=root)


# ------------------------------------------------------------------------------------------------------- smoke
class SmokeGraphs:
    """r13's real graphs (read only), restricted to the pairs inside the first `n_fit` fit and `n_val` validation photos
    (pairs are i < j, so both ends are inside when j is)."""

    def __init__(self, n_fit, n_val, root):
        self.n_fit, self.n_val = n_fit, n_val
        self.full = BP.Graphs(dict(keys=PD.fit_keys()), dict(keys=PD.keys('val')), root)

    def get(self, k, correction):
        g = self.full.get(k, correction)
        fit, dev = g['fit_pairs'], g['dev_pairs']
        return dict(fit_pairs=fit[fit[:, 1] < self.n_fit], dev_pairs=dev[dev[:, 1] < self.n_val],
                    manifest_sha256=g['manifest_sha256'])


def make_standin(root, kstar, seed=0):
    """Smoke only: K0's SigLIP 1 emotion checkpoint with the caption trunk's first layer cut to 1,152 inputs, written as a
    K* checkpoint (ticket 03's layout and sidecar) under the smoke root. Not a trained K*."""
    src = P.K0_ROOT / 'weights' / P.K0_ARM / f'emotion__s{seed}.pt'
    state = torch.load(src, map_location='cpu', weights_only=True)
    state['cap.trunk.0.weight'] = state['cap.trunk.0.weight'][:, :CAP_WIDTH].clone()
    folder = Path(root) / 'cstar' / ENC / 'weights' / kstar
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'emotion__s{seed}.pt'
    torch.save(state, path)
    path.with_suffix('.json').write_text(json.dumps(dict(
        weights_sha256=rs_vh.file_sha(path), standin_of=str(src),
        config=dict(rb=dict(width=IMG_WIDTH, cap_width=CAP_WIDTH, cap_source=kstar,
                            cap_store_sha256=C.store_hash(kstar))))))
    return path


def smoke(kstar='siglip1', device='cuda', n_fit=6000, n_val=1500, root=SMOKE_ROOT):
    """End to end on a slice: the explanations of the first `n_fit` fit and `n_val` validation photos are encoded into a
    store under the smoke root, checked by gate (viii) (record under the smoke root), a K0-based stand-in start checkpoint,
    two epochs of F1 seed 0 from it, and an export of the validation slice. Writes nothing outside the smoke root."""
    from src.data.feature_extract import load_encoder
    root = Path(root)
    assert root.resolve().is_relative_to(P.SMOKE_OUT.resolve()), f'smoke outputs must be under {P.SMOKE_OUT}'
    fk, vk = PD.fit_keys()[:n_fit], PD.keys('val')[:n_val]
    reference = _json(DEFAULT_INDEX)
    want = set(fk) | set(vk)
    images = reference['images']
    texts = [t for t in reference['texts'] if f"{images[t['img']]['source']}/{images[t['img']]['image_name']}" in want]
    ref = dict(reference, texts=texts, n_texts=len(texts))
    store = root / 'store'
    enc = load_encoder(X.extractor.SIGLIP1, device)
    form = FORMS[kstar]
    print(f'smoke: {len(texts)} explanations of {len(want)} photos', flush=True)
    build_store(ref, affection_texts(ref), enc, store, form=form, smoke=True)
    record = gate_viii(store, ref, form=form, record_folder=root / 'records/gates', allow_smoke=True)
    make_standin(root, kstar)
    fit = dict(keys=fk, I=BP.images('affection', ENC, fk), D=caption_means(fk, store), T=PD._teacher_images(fk))
    val = dict(keys=vk, I=BP.images('affection', ENC, vk), D=caption_means(vk, store), T=PD._teacher_images(vk))
    cap = dict(cap_store_sha256=store_hash(store), cap_sha256=dict(fit=rs_vh.array_sha(fit['D']),
                                                                    val=rs_vh.array_sha(val['D'])))
    out = train_arm(kstar, [0], device, root, data=(fit, val, fit, val), graphs=SmokeGraphs(n_fit, n_val, root),
                    max_epochs=2, cap_info=cap, smoke=True)
    run = out[(BP.REF, 0)].record
    ckpt = root / f'photo/{ENC}/s0/emotion.pt'
    with caption_patches(kstar):
        path = BP.export_checkpoint('affection', f'photo_{ENC}', 'val', 0, ckpt, BP.checkpoint_sha(f'photo_{ENC}', 0, root),
                                    IMG_WIDTH, vk, val['I'], root=root)
    with np.load(path, allow_pickle=False) as z:
        emb = z['emb']
    assert emb.shape == (len(vk), P.OUTPUT_WIDTH) and np.isfinite(emb.astype(np.float32)).all()
    print(json.dumps(dict(status='ok', gate_viii=record['status'], n_truncated=record['details']['n_truncated'],
                          best_epoch=run['best_epoch'], best_dev_loss=run['best_dev_loss'], seconds=run['seconds'],
                          n_fit_pairs=len(SmokeGraphs(n_fit, n_val, root).get(80, True)['fit_pairs']),
                          export=str(path), export_shape=list(emb.shape))))


# ------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('store')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, default=STORE)
    p = sub.add_parser('gate-viii')
    p.add_argument('--store', type=Path, default=STORE)
    p = sub.add_parser('gate-vii')
    p.add_argument('--device', default='cpu')
    p.add_argument('--n-fit', type=int, default=1024)
    p.add_argument('--no-record', action='store_true')
    for name in ('train', 'export'):
        p = sub.add_parser(name)
        p.add_argument('--seeds', type=int, nargs='+', choices=P.SEEDS, default=list(P.SEEDS))
        p.add_argument('--root', type=Path, default=P.OUT)
        if name == 'train':
            p.add_argument('--device', default='cuda')
        else:
            p.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    p = sub.add_parser('smoke')
    p.add_argument('--kstar', choices=sorted(FORMS), default='siglip1')
    p.add_argument('--device', default='cuda')
    p.add_argument('--root', type=Path, default=SMOKE_ROOT)
    p.add_argument('--n-fit', type=int, default=6000)
    p.add_argument('--n-val', type=int, default=1500)
    a = parser.parse_args(argv)
    if a.cmd == 'train':
        require_pass()
        train_arm(read_prefix_pick()['kstar'], a.seeds, a.device, a.root)
    elif a.cmd == 'export':
        for path in export_arm(read_prefix_pick()['kstar'], a.seeds, a.splits, a.root):
            print(path)
    elif a.cmd == 'gate-vii':
        gate_vii(a.device, a.n_fit, write=not a.no_record)
    elif a.cmd == 'smoke':
        smoke(a.kstar, a.device, a.n_fit, a.n_val, root=a.root)
    else:
        form = read_prefix_pick()['kstar_form']
        reference = _json(DEFAULT_INDEX)
        X._keys(reference, 'affection')
        if a.cmd == 'store':
            from src.data.feature_extract import load_encoder
            texts = affection_texts(reference)
            enc = load_encoder(X.extractor.SIGLIP1, a.device)
            print(json.dumps(build_store(reference, texts, enc, a.out, form=form)))
            print(json.dumps(gate_viii(a.out, reference, form=form)['status']))
        else:
            print(json.dumps(gate_viii(a.store, reference, form=form)['status']))


if __name__ == '__main__':
    main()
