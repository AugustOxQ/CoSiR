"""Pure functions for the attention-map files: token types, rollout, suffix rollout, band maps, writer, reader.

Numpy only at import time (torch is imported inside the rollout functions), so the viewer and the analysis
can import this module without a GPU or a model.

File layout of one painting (`p{k:04d}.npz`), keys joined with '__':
  painting: k, subset_pid, painting_id, image_index, image_size, grid_hw, cap_rows, cap_sample_ids, cap_emo, sty, gen
  per view v, modality m in img/txt/joint: {v}__{m}__off/ids/ttype/roll/layers/emb
  joint only: {v}__joint__cross (float16, per pass (3, n_caption_tokens, n_image_tokens) flattened: the raw head-mean
  attention of each caption-token query row over the image columns, summed over each band of BANDS), {v}__joint__crossoff
  `roll` (pooled-token rollout) is a record only; the maps used are raw attention (`layers`, `cross`) summed over a band.
`load_painting(root, k)` returns {'painting': {...fields}, view: {modality: [pass dict, ...]}}.
"""
import os
from pathlib import Path
import numpy as np

VIEWS = ('default', 'emotion_focus', 'style_word', 'genre_word')
MODS = ('img', 'txt', 'joint')
N_LAYERS = 28
EMB_DIM = 2048
# Qwen3-VL special token ids (checked against the tokenizer by att_extract at start-up)
IM_START, IM_END, ENDOFTEXT = 151644, 151645, 151643
VISION_START, VISION_END, IMAGE_PAD = 151652, 151653, 151655
TAIL = 6   # <|im_end|> \n <|im_start|> assistant \n <|endoftext|>
# (name, first layer, end layer exclusive); layer index 0..27 = the order of out.attentions
BANDS = (('layers 16-27', 16, 28), ('last layer', 27, 28), ('all layers', 0, 28))
PAINTING_KEYS = ('k', 'subset_pid', 'painting_id', 'image_index', 'image_size', 'grid_hw', 'cap_rows',
                 'cap_sample_ids', 'cap_emo', 'sty', 'gen')


# ---------------------------------------------------------------- token types
def token_types(ids, kind):
    """int8 per token: 0 template/special, 1 instruction text, 2 image, 3 caption.

    kind: 'img' | 'txt' | 'joint'. ids: 1-D int array of one full input.
    """
    ids = np.asarray(ids)
    t = np.zeros(len(ids), np.int8)
    assert ids[0] == IM_START
    e1 = int(np.nonzero(ids == IM_END)[0][0])          # end of the system turn
    t[3:e1] = 1                                        # <|im_start|> system \n | instruction | <|im_end|>
    t[ids == IMAGE_PAD] = 2
    if kind in ('txt', 'joint'):
        if kind == 'joint':
            start = int(np.nonzero(ids == VISION_END)[0][0]) + 1
        else:
            start = int(np.nonzero(ids == IM_START)[0][1]) + 3   # <|im_start|> user \n
        ends = np.nonzero(ids == IM_END)[0]
        end = int(ends[-1]) if ends[-1] > start else len(ids)
        t[start:end] = 3
    return t


# ---------------------------------------------------------------- rollout (torch, imported lazily)
def head_mean_norm(a):
    """(B, H, Lq, Lk) attention -> head mean, float32, rows renormalised to sum 1 (removes bf16 rounding)."""
    m = a.float().mean(1)
    return m / m.sum(-1, keepdim=True)


def rollout_full(att, keep_all=False):
    """Rollout of Abnar and Zuidema, R_0 = I, R_l = (0.5 A_l + 0.5 I) R_{l-1}.

    att: list of 28 tensors (B, L, L) float32 (head-mean, row-normalised). Returns R_28 (B, L, L), or with
    keep_all the list R_0..R_28.
    """
    import torch
    B, L, _ = att[0].shape
    eye = torch.eye(L, device=att[0].device, dtype=torch.float32)
    R = eye.expand(B, L, L).contiguous()
    keep = [R]
    for A in att:
        R = torch.bmm(0.5 * A + 0.5 * eye, R)
        keep.append(R)
    return keep if keep_all else R


def suffix_rollout(att, RP, P):
    """Rollout rows of the suffix positions, given the prefix rollout blocks.

    att: list of 28 tensors (B, Ls, P+Ls) float32 (head-mean, row-normalised; suffix queries over prefix+suffix keys).
    RP: list R^P_0..R^P_28, each (P, P) (single prefix shared by the batch).
    Row s of layer l: R_l[s, :] = sum_j (0.5 A_l[s, j] + 0.5 delta_sj) R_{l-1}[j, :], where rows j in the prefix are
    R^P_{l-1} (zero in the suffix columns) and rows j in the suffix are the rows already computed for layer l-1.
    Returns R_28 suffix rows (B, Ls, P+Ls).
    """
    import torch
    B, Ls, L = att[0].shape
    assert L == P + Ls
    eye = torch.eye(Ls, device=att[0].device, dtype=torch.float32)
    R = torch.zeros(B, Ls, L, device=att[0].device, dtype=torch.float32)
    R[:, :, P:] = eye
    for l, A in enumerate(att, start=1):
        M = 0.5 * A
        M[:, :, P:] += 0.5 * eye
        new = torch.empty_like(R)
        new[:, :, :P] = M[:, :, :P] @ RP[l - 1] + M[:, :, P:] @ R[:, :, :P]
        new[:, :, P:] = M[:, :, P:] @ R[:, :, P:]
        R = new
    return R


def l1_norm_maps(a, b, eps=1e-12):
    """L1 distance between two non-negative vectors/rows (last axis) after renormalising each to sum 1 (0..2)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a / np.maximum(a.sum(-1, keepdims=True), eps)
    b = b / np.maximum(b.sum(-1, keepdims=True), eps)
    return np.abs(a - b).sum(-1)


def band_map(p, band, kind='image'):
    """Raw-attention map of the pooled token over one band of layers.

    p: pass dict (layers (28, L), ttype (L,)); band: one entry of BANDS; kind 'image' (ttype 2) or 'caption' (ttype 3).
    Returns (map renormalised to sum 1, share): share = band-summed attention on those positions / number of layers.
    """
    _, a, b = band
    s = np.asarray(p['layers'][a:b], np.float64).sum(0)
    pos = np.asarray(p['ttype']) == (2 if kind == 'image' else 3)
    m = s[pos]
    tot = float(m.sum())
    return m / max(tot, 1e-300), tot / (b - a)


# ---------------------------------------------------------------- per-pass checks (gate c)
def check_pass(p, kind, grid_hw=None, tol=1e-3):
    """Finite values, rollout row sums to 1, token-type counts match the ids. Raises AssertionError."""
    ids, tt = p['ids'], p['ttype']
    L = len(ids)
    assert tt.shape == (L,) and p['roll'].shape == (L,) and p['layers'].shape == (N_LAYERS, L)
    for key in ('roll', 'layers', 'emb') + (('cross',) if 'cross' in p else ()):
        assert np.isfinite(np.asarray(p[key], np.float32)).all(), f'non-finite {key}'
    assert abs(float(p['roll'].sum()) - 1.0) < tol, f"rollout row sum {float(p['roll'].sum())}"
    assert int((tt == 2).sum()) == int((ids == IMAGE_PAD).sum())
    if grid_hw is not None and kind != 'txt':
        assert int((tt == 2).sum()) == int(grid_hw[0]) * int(grid_hw[1])
    if kind == 'txt':
        assert int((tt == 2).sum()) == 0
    else:
        assert np.array_equal(tt == 1, token_types(ids, kind) == 1)
    assert np.array_equal(tt, token_types(ids, kind))
    if kind == 'joint':
        assert p['cross'].shape == (len(BANDS), int((tt == 3).sum()), int((tt == 2).sum()))
    if kind == 'img':
        assert int((tt == 3).sum()) == 0
    if kind == 'joint':
        assert int((tt == 3).sum()) == L - (int(np.nonzero(ids == VISION_END)[0][0]) + 1) - TAIL
    if kind == 'txt' and L < 512:
        assert int((tt == 3).sum()) == L - (int(np.nonzero(ids == IM_START)[0][1]) + 3) - TAIL
    if kind != 'txt' or L < 512:
        assert ids[-1] == ENDOFTEXT and ids[-TAIL] == IM_END


# ---------------------------------------------------------------- writer / reader
def painting_path(root, k):
    return Path(root) / f'p{int(k):04d}.npz'


def is_complete(root, k, views=VIEWS):
    path = painting_path(root, k)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
    except Exception:
        return False
    need = set(PAINTING_KEYS)
    for v in views:
        for m in MODS:
            need |= {f'{v}__{m}__{f}' for f in ('off', 'ids', 'ttype', 'roll', 'layers', 'emb')}
        need |= {f'{v}__joint__cross', f'{v}__joint__crossoff'}
    return need <= keys


def pack_view(v, views_data):
    """views_data: {modality: [pass dict]} -> flat dict of npz arrays for view v."""
    out = {}
    for m in MODS:
        ps = views_data[m]
        lens = np.array([len(p['ids']) for p in ps], np.int64)
        out[f'{v}__{m}__off'] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        out[f'{v}__{m}__ids'] = np.concatenate([p['ids'] for p in ps]).astype(np.int32)
        out[f'{v}__{m}__ttype'] = np.concatenate([p['ttype'] for p in ps]).astype(np.int8)
        out[f'{v}__{m}__roll'] = np.concatenate([p['roll'] for p in ps]).astype(np.float32)
        out[f'{v}__{m}__layers'] = np.concatenate([p['layers'] for p in ps], axis=1).astype(np.float16)
        out[f'{v}__{m}__emb'] = np.stack([p['emb'] for p in ps]).astype(np.float16)
    cr = [np.asarray(p['cross']).astype(np.float16).reshape(-1) for p in views_data['joint']]
    out[f'{v}__joint__cross'] = np.concatenate(cr)
    out[f'{v}__joint__crossoff'] = np.concatenate([[0], np.cumsum([len(c) for c in cr])]).astype(np.int64)
    return out


def write_painting(root, k, painting, data):
    """Atomic write. painting: dict of PAINTING_KEYS; data: {view: {modality: [pass dict]}}."""
    flat = {}
    for key in PAINTING_KEYS:
        flat[key] = np.asarray(painting[key])
    for v, d in data.items():
        flat.update(pack_view(v, d))
    path = painting_path(root, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **flat)
    os.replace(tmp, path)
    return path


def load_painting(root, k):
    """Read one painting file. Returns {'painting': {fields}, view: {modality: [pass dict]}}.

    Pass dict: ids (int32), ttype (int8), roll (float32, L), layers (float16, 28 x L), emb (float16, 2048); the joint
    passes also have cross (float16, 3 bands x n_caption_tokens x n_image_tokens).
    """
    out = {}
    with np.load(painting_path(root, k), allow_pickle=False) as z:
        out['painting'] = {key: z[key] for key in PAINTING_KEYS}
        for v in VIEWS:
            if f'{v}__img__off' not in z.files:
                continue
            out[v] = {}
            for m in MODS:
                off = z[f'{v}__{m}__off']
                ids, tt = z[f'{v}__{m}__ids'], z[f'{v}__{m}__ttype']
                roll, layers, emb = z[f'{v}__{m}__roll'], z[f'{v}__{m}__layers'], z[f'{v}__{m}__emb']
                passes = []
                for i in range(len(off) - 1):
                    a, b = int(off[i]), int(off[i + 1])
                    p = dict(ids=ids[a:b], ttype=tt[a:b], roll=roll[a:b], layers=layers[:, a:b], emb=emb[i])
                    passes.append(p)
                if m == 'joint':
                    cross, co = z[f'{v}__joint__cross'], z[f'{v}__joint__crossoff']
                    for i, p in enumerate(passes):
                        p['cross'] = cross[int(co[i]):int(co[i + 1])].reshape(len(BANDS), int((p['ttype'] == 3).sum()),
                                                                              int((p['ttype'] == 2).sum()))
                out[v][m] = passes
    return out
