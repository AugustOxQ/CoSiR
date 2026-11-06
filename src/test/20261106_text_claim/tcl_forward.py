"""r23 forward passes, late fusion and gate (ii) (ticket 04; spec §3.2, D1, D8, gate (ii) in §3.5).

    rows(set, kind, row, head, seed, device)   (text, image): float32 rows in `tcl_data.frame` order
        cstar    caption side of the aspect's head on the plain text, image side on the plain image
        free     the image side on both (captions through the image side); ArtELingo only
        late     (sqrt(1 - A) c*, sqrt(A) plain) for text and for image, same head and seed; width 4096, no further normalising
        plain, teacher, clip_l14, siglip2_so400m, siglip_so400m, roberta: `tcl_data`, head 'all' (teacher: the aspect), seed 'x'
    fuse(c, plain, a)                          the late vector of one side
    gate_ii(device)                            15 weight files against the stored exports; writes OUT/stage0/gate_ii.json

At raw cosine, the cosine of two late vectors is (1 - a) c* cosine + a plain cosine. This module reads no label directly.
rows() delegates through tcl_data.frame, which loads held labels on held sets; no returned vector depends on them.
Gate (ii) reads no held-set label or frame.
"""
import json
import sys
from pathlib import Path

import numpy as np

import tcl_paths as P
import tcl_data
import tx_data
import tx_model
import rt_data
import mt_data

TOL = 1e-3


# ------------------------------------------------------------------------------------------------ rows
def fuse(c, plain, a=P.A):
    """The late vector of one side: (sqrt(1 - a) c*, sqrt(a) plain), float32, width = width(c) + width(plain)."""
    assert c.shape == plain.shape and c.dtype == plain.dtype == np.float32 and 0 <= a <= 1
    w_c, w_p = float(np.sqrt(1 - a)), float(np.sqrt(a))
    out = np.concatenate((w_c * c, w_p * plain), axis=1).astype(np.float32)
    assert out.shape == (len(c), c.shape[1] + plain.shape[1]) and np.isfinite(out).all()
    return out


def rows(set_name, kind, row, head, seed, device):
    """(text, image) float32 rows of one row of the study in `tcl_data.frame` order. The head is an aspect for the head rows
    (cstar, late, free, teacher) and 'all' for the others; the seed is 0 to 4 for cstar, late and free, else 'x'."""
    assert set_name in P.SETS and kind in P.kinds(P.GROUP[set_name]), (set_name, kind)
    assert row in P.BAR_ROWS or row in P.SIDE_ROWS, row
    assert (head in P.ASPECTS) == (row in P.HEAD_ROWS), (row, head)
    assert (seed in P.SEEDS) == (row in P.SEEDED_ROWS), (row, seed)
    if row == 'plain':
        return tcl_data.plain(set_name, kind)
    if row == 'teacher':
        return tcl_data.teacher(set_name, kind, head)
    if row not in P.SEEDED_ROWS:
        return tcl_data.side(set_name, row)
    text, image = tcl_data.plain(set_name, kind)
    image_side, caption_side = tx_model.load_sides(head, seed, device)
    if row == 'free':
        assert P.GROUP[set_name] == 'art', 'the free variant is an ArtELingo side row'
        return image_side(text), image_side(image)
    c_t, c_i = caption_side(text), image_side(image)
    if row == 'cstar':
        return c_t, c_i
    assert row == 'late'
    return fuse(c_t, text), fuse(c_i, image)


# ------------------------------------------------------------------------------------------------ gate (ii)
def _diff(got, saved):
    """Largest absolute difference, float64; the shapes must agree and everything be finite."""
    saved = np.asarray(saved, dtype=np.float32)
    assert got.shape == saved.shape and got.ndim == 2 and np.isfinite(got).all() and np.isfinite(saved).all()
    return float(np.max(np.abs(got.astype(np.float64) - saved.astype(np.float64))))


def _stored_caption(aspect, seed):
    """The stored validation caption side of (aspect, seed), float16 [6152, 2048]. `rt_data.raw` checks the file's sidecar
    first (it reads the same file's `img` array); then the `cap` array is read from that file."""
    rt_data.raw('cstar', 'val', aspect, seed)
    path = rt_data.CSTAR_ROOT / f'train/k11=40/R6-5/{aspect}__{rt_data.GRAPHS[aspect]}__s{seed}.npz'
    with np.load(path, allow_pickle=False) as z:
        cap = z['cap']
    assert cap.dtype == np.float16 and cap.shape == (P.N_PAINTINGS['art_val'], 2048), (path, cap.shape, cap.dtype)
    return cap


def gate_ii(device='cpu', out=None):
    """Spec §3.5 (ii), on the full sets (TCL_SMOKE does not apply). Per file (aspect, seed), the largest absolute difference:
    art_val_image, art_held_image, mm_test_image against the stored image exports (mm_test by key, from r15's `new` export of
    16,782 keys) and art_val_caption (L2-normalised caption means through the caption side) against the stored `cap`. A set
    group passes when all its differences are at most 1e-3: `art` (art_val_image, art_held_image, art_val_caption)
    and `mm` (mm_test_image, art_val_caption).
    MMArt text rows use the same caption side, so its caption check serves both groups.
    No held label or held frame is read: images come from the stores by painting name or key."""
    import mt_export
    captions = tx_data.val_captions()
    assert np.array_equal(captions['pid'], tcl_data._art_captions('val')['pid']), 'pid differs from tcl_data'
    n_val = P.N_PAINTINGS['art_val']
    means = tx_data.caption_mean(tx_data.text_rows('plain', raw=True), captions['pid'], n_val)
    images = dict(art_val=rt_data.embed('plain', 'val', 'emotion'), art_held=rt_data.embed('plain', 'held', 'emotion'))
    assert len(images['art_val']) == n_val and len(images['art_held']) == P.N_PAINTINGS['art_held']
    test_keys = mt_data.keys('test')
    assert len(test_keys) == P.N_PAINTINGS['mm_test']
    images['mm_test'] = mt_data.features('qwen_default', test_keys)
    files = {}
    for aspect in P.ASPECTS:
        for seed in P.SEEDS:
            image_side, caption_side = tx_model.load_sides(aspect, seed, device)
            stored = dict(
                art_val=rt_data.embed('cstar', 'val', aspect, seed), art_held=rt_data.embed('cstar', 'held', aspect, seed),
                mm_test=mt_export.load('new', 'qwen', aspect, seed, test_keys, root=P.R15_OUT).astype(np.float32))
            item = {f'{s}_image': _diff(image_side(images[s]), stored[s]) for s in ('art_val', 'art_held', 'mm_test')}
            item['art_val_caption'] = _diff(caption_side(means), _stored_caption(aspect, seed))
            files[f'{aspect}__s{seed}'] = item
            print('gate ii', aspect, seed, {k: round(v, 6) for k, v in item.items()}, flush=True)
            del image_side, caption_side
    assert len(files) == 15
    groups = dict(art=('art_val_image', 'art_held_image', 'art_val_caption'), mm=('mm_test_image', 'art_val_caption'))
    largest = {k: max(f[k] for f in files.values()) for k in groups['art'] + groups['mm']}
    passed = {g: bool(all(largest[k] <= TOL for k in keys)) for g, keys in groups.items()}
    result = dict(tolerance=TOL, files=files, largest=largest, **{'pass': passed})
    path = Path(P.OUT if out is None else out) / 'stage0/gate_ii.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    res = gate_ii(device)
    print(json.dumps(dict(largest=res['largest'], **{'pass': res['pass']}), indent=2))
