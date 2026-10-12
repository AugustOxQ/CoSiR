"""Round-8 placement (DECISION_RULE.md M11, M12, M13, M14, M16; ticket 08): own map, cross-fitted bank inputs, text heads.

For one grouping ID X (a candidate, R0own, NRK, TP) or one control Z<X>, from the grouping file
(`groupings/r8_groupings.npz`, ticket 07) and X's space file (`spaces/X.npz`) to, in R7's format,
  heads/post_<ID>.npz    img, txt [308,723, K] float32, rows = selection, NaN outside (R7 `_posterior_ok`)
  heads/xheads_<ID>.npz  img, txt [183,694, K], filled_by_half, classes_half0/1 (rows = scorer-train)
  heads/head_<ID>.json   dropped, reason, n_groups, label_sha256, drop checks, provenance (written last)
Image side (M11, M12): one-hot own map m(i) = argmax_g phi_i . mu_g (float64, ties to the lowest group). Selection rows:
centres from all scorer-train paintings. Scorer-train rows: centres from the other painting half (mu^(1-h)), labels of
the full grouping, phi not refit. Z<X> (M13): the painting draws of `r8_groupings.control_draws` (never repeated here).
Caption side (M14): X's CLIP text head, standard and cross-fitted (`r8_recipe.fit_clip_heads`, R7 L10/L11); R0own takes
R7's `post_R0` / `xheads_R0` txt (M11, M12). `--h-arm` also writes the CLIP-head arm H<X> (R7's image and text heads on
X's labels, from the same fit; for ticket 12).
Drops (M12, M16): a group with no scorer-train painting in a half, or a group missing from a head draw (R7 L10/L11),
drops X; its head_X.json says why and no post/xheads file is written. A dropped X has no control: Z<X> is dropped too.

CPU only. `--smoke [N]` uses the first N paintings of the smoke grouping file (made by `r8_groupings part --smoke N` and
`collect --smoke`) and writes under results/smoke/ only.

  python r8_place.py --grouping V100            # also AL, ..., R0own, NRK, TP, ZV100, ...
  python r8_place.py --grouping V100 --h-arm    # plus post/xheads/head_HV100
  python r8_place.py --grouping V100 --h-only   # after V100 is placed: only post/xheads/head_HV100 (ticket 12, after the verdict)
  python r8_place.py --grouping NBC --groupings-name r8_groupings_nbc
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import r8_groupings as G

SMOKE_PAINTINGS = 8192          # smoke default; head draws of 5,000 + 1,000 rows need >= 6,000 rows per half
R7_R0_POST = '20261126_design_l_style_source/results/heads/post_R0.npz'
R7_R0_XHEADS = '20261126_design_l_style_source/results/heads/xheads_R0.npz'
CHUNK = 8192


# ====================================================================== pure functions (no r8_common)

def onehot(labels, k):
    labels = np.asarray(labels)
    assert labels.ndim == 1 and labels.size and labels.min() >= 0 and labels.max() < k, 'one-hot: label out of range'
    out = np.zeros((len(labels), k), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1
    return out


def unit_centres(phi_p, labels_p, k, members=None):
    """M11: mu_g = unit(mean of phi_p over the member paintings of group g), float64 [k, d]. `members` (bool per
    painting) restricts the paintings (M12's half-(1-h) centres); every group needs at least one member."""
    phi_p = np.asarray(phi_p)
    labels_p = np.asarray(labels_p)
    assert phi_p.ndim == 2 and labels_p.shape == (len(phi_p),), 'centres: phi/labels shape'
    use = np.ones(len(phi_p), dtype=bool) if members is None else np.asarray(members, dtype=bool)
    assert use.shape == labels_p.shape, 'centres: members shape'
    mu = np.empty((k, phi_p.shape[1]), dtype=np.float64)
    for g in range(k):
        sel = (labels_p == g) & use
        assert sel.any(), f'centres: group {g} has no member'
        m = phi_p[sel].astype(np.float64).mean(axis=0)
        norm = np.linalg.norm(m)
        assert norm > 0 and np.isfinite(norm), f'centres: group {g} has a zero mean vector'
        mu[g] = m / norm
    return mu


def nearest_centre(phi, mu):
    """M11: argmax_g phi_i . mu_g in float64; ties go to the lowest group (argmax returns the first maximum)."""
    mu = np.asarray(mu, dtype=np.float64)
    out = np.empty(len(phi), dtype=np.int64)
    for a in range(0, len(phi), CHUNK):
        out[a:a + CHUNK] = np.argmax(np.asarray(phi[a:a + CHUNK], dtype=np.float64) @ mu.T, axis=1)
    return out


def own_map(phi, phi_p, labels_p, k):
    """The own map of images phi with centres from all scorer-train paintings (selection images, M11)."""
    return nearest_centre(phi, unit_centres(phi_p, labels_p, k))


def crossfit_centres(phi_p, labels_p, halves_p, k):
    """M12: [mu^(1-h) for h = 0, 1]: the centres that half-h paintings are mapped with, from the half-(1-h) members of
    each group (labels of the full grouping)."""
    halves_p = np.asarray(halves_p)
    return [unit_centres(phi_p, labels_p, k, members=halves_p == 1 - h) for h in (0, 1)]


def crossfit_painting_labels(phi_p, labels_p, halves_p, k):
    """M12: the own-map group of every scorer-train painting with the other half's centres (phi not refit)."""
    halves_p = np.asarray(halves_p)
    mus = crossfit_centres(phi_p, labels_p, halves_p, k)
    out = np.empty(len(phi_p), dtype=np.int64)
    for h in (0, 1):
        idx = np.flatnonzero(halves_p == h)
        out[idx] = nearest_centre(phi_p[idx], mus[h])
    return out


def m12_reason(labels_p, halves_p, k):
    """M12: None, or the reason X is dropped: a group with no scorer-train painting in a half."""
    # rule gap: M12 says "a group with no member in a half"; read as no scorer-train painting (not row) of the group in
    # that half, by the full grouping's labels. Both halves are checked (half h is mapped with the half-(1-h) centres).
    labels_p, halves_p = np.asarray(labels_p), np.asarray(halves_p)
    for h in (0, 1):
        counts = np.bincount(labels_p[halves_p == h], minlength=k)
        if (counts == 0).any():
            g = int(np.flatnonzero(counts == 0)[0])
            return f'M12: group {g} has no scorer-train painting in half {h} (the half-{1 - h} centres cannot be built)'
    return None


def image_side(phi_p, labels_p, halves_p, g_st, phi_sel):
    """X's image side: {'img_sel' [n_sel, K] one-hot own map with all-painting centres, 'img_cf' [rows, K] cross-fitted
    one-hots per scorer-train row, agreement diagnostics}, or {'dropped': True, 'reason'} (M12)."""
    k = int(np.max(labels_p)) + 1
    reason = m12_reason(labels_p, halves_p, k)
    if reason is not None:
        return {'dropped': True, 'reason': reason}
    cf_p = crossfit_painting_labels(phi_p, labels_p, halves_p, k)
    fit_p = own_map(phi_p, phi_p, labels_p, k)
    return {'img_sel': onehot(own_map(phi_sel, phi_p, labels_p, k), k), 'img_cf': onehot(cf_p[g_st], k),
            'agreement_fit': float((fit_p == labels_p).mean()), 'agreement_crossfit': float((cf_p == labels_p).mean()),
            'sizes_cf_paintings': np.bincount(cf_p, minlength=k).tolist()}


def control_image_labels(labels_p, i, g_st, sel_codes):
    """M13's image side: (group per scorer-train row, group per selection row) from `G.control_draws`. The scorer-train
    paintings draw once for both halves; the selection paintings are the distinct leakage-group codes of the selection
    rows (ascending), a row takes its painting's draw."""
    # rule gap: M13's n_sel is read as the number of distinct selection paintings (the "n_sel paintings" in ascending
    # artelingo_splits().groups code), not the number of selection rows; a row takes its painting's draw.
    uniq, inv = np.unique(np.asarray(sel_codes), return_inverse=True)
    train, sel = G.control_draws(np.asarray(labels_p), i, len(uniq))
    assert len(train) == len(labels_p) and len(sel) == len(uniq), 'control draws: size'
    return train[g_st], sel[inv]


def control_side(labels_p, i, g_st, sel_codes):
    k = int(np.max(labels_p)) + 1
    train_rows, sel_rows = control_image_labels(labels_p, i, g_st, sel_codes)
    return {'img_sel': onehot(sel_rows, k), 'img_cf': onehot(train_rows, k)}


def assert_halves_agree(halves_p, g_st, half_of_local_row):
    """C6: the half of each painting (indexed by the `g_st` code) equals the half of each scorer-train row."""
    halves_p, g_st, half_row = np.asarray(halves_p), np.asarray(g_st), np.asarray(half_of_local_row)
    assert g_st.shape == half_row.shape and g_st.max() < len(halves_p), 'halves: painting codes / row halves shape'
    assert np.array_equal(halves_p[g_st], half_row), 'halves: painting halves differ from the scorer-train row halves (C6)'


def join_selection(space_idx, sel, sel_img, smoke):
    """C6: each selection row's row in the space file (`space_idx`: sorted image ids). All of them; in smoke only the rows
    whose image is in the space file. Returns (selection rows, space rows)."""
    # rule gap: a smoke space file may hold only some images; smoke then places the selection rows whose image is there
    # (a real run asserts all of them).
    space_idx, sel, sel_img = np.asarray(space_idx), np.asarray(sel), np.asarray(sel_img)
    assert sel.shape == sel_img.shape and (np.diff(space_idx) > 0).all(), 'selection join: shapes / space ids not sorted'
    pos = np.searchsorted(space_idx, sel_img)
    ok = (pos < len(space_idx)) & (space_idx[np.minimum(pos, len(space_idx) - 1)] == sel_img)
    assert smoke or ok.all(), 'a selection image is not in the space file'
    assert ok.any(), 'no selection image is in the space file'
    return sel[ok], pos[ok]


def listed_drop(name, X, control, x_in_file, name_in_file, listed, gname):
    """M16: the reason when ticket 07 left X (or Z<X>) out of the grouping file, else None. A control of a dropped X has
    no control (M13): 07's own record for Z<X> is kept as written, else X's reason gets run_r8.split_dropped's prefix."""
    if x_in_file and (not control or name_in_file):
        return None
    if not x_in_file:
        why = listed.get(X) or f'{X} is not in {gname}'
        return (listed.get(name) or f'control of dropped {X}: {why}') if control else why
    return listed.get(name) or f'{name} is not in {gname}'


def x_drop(X, control, labels_p, halves_p, k, draw_fn):
    """The drops that decide X and its control alike: M12 (centres), then R7 L10/L11 (head draws on X's labels, `draw_fn()`
    -> (draws, reason or None); R0own takes R7's heads, so none). Returns (draws, reason or None); a control's reason gets
    the prefix `control of dropped X:` (as run_r8.split_dropped)."""
    # rule gap: M13 says a dropped X has no control; read as every drop of X (ticket 07, M12, a head draw of X), so Z<X> is
    # dropped with X's reason. Z<X>'s own head draw (R7 L10/L11 on Z's labels) can drop Z<X> alone; Z has no centres, so
    # M12 does not apply to it.
    reason, draws = m12_reason(labels_p, halves_p, k), None
    if reason is None and X != 'R0own':
        draws, reason = draw_fn()
    if reason is not None and control:
        reason = f'control of dropped {X}: {reason}'
    return draws, reason


def assert_tp_agreement(side):
    """M9: TP's images are its own map; the one-hot space gives that only if every painting maps to its own label, in the
    fit and cross-fitted."""
    assert side['agreement_fit'] == 1.0 and side['agreement_crossfit'] == 1.0, \
        f"TP: own map differs from the TP labels (agreement {side['agreement_fit']}, cross-fitted {side['agreement_crossfit']})"


X_IMAGE_SIDE_KEYS = ('space', 'space_sha256', 'phi_shape', 'agreement_fit', 'agreement_crossfit', 'sizes_cf_paintings')


def h_record(x_rec, X, text_rec, draws):
    """head_H<X>.json: X's record without X's own-map image-side keys (H<X>'s image side is the CLIP image head), with the
    text record and draws of the fit that gave H<X>."""
    rec = {k: v for k, v in x_rec.items() if k not in X_IMAGE_SIDE_KEYS + ('rule_sha256', 'written')}
    rec.update(grouping=f'H{X}', matched_grouping=X, text=text_rec, draws=draws,
               image_side='R7 L10 CLIP image head on X labels (CLIP image features, not the own map)')
    return rec


def assert_same_text(txt_g, stored_g, txt_cf, stored_cf):
    """--h-only: the refit text heads equal X's stored ones bit for bit (global: NaN where NaN; cross-fitted rows)."""
    for what, a, b in (('post txt', txt_g, stored_g), ('xheads txt', txt_cf, stored_cf)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, f'--h-only: refit {what} dtype/shape differs from X\'s'
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        assert same_nan and a[~np.isnan(a)].tobytes() == b[~np.isnan(b)].tobytes(), \
            f'--h-only: refit {what} differs from X\'s stored one (bytes of the finite values, NaN places)'


def post_arrays(n_all, selection, img_sel, txt):
    """post_<ID>.npz arrays (R7 format): img one-hot on the selection rows, NaN elsewhere; txt is the head's global
    posterior (finite on the selection, NaN elsewhere); rows = selection."""
    selection = np.asarray(selection)
    k = img_sel.shape[1]
    assert selection.dtype == np.int64 and img_sel.shape == (len(selection), k), 'post: image side shape'
    assert txt.shape == (n_all, k) and txt.dtype == np.float32, 'post: txt shape/dtype'
    inside = np.zeros(n_all, dtype=bool)
    inside[selection] = True
    assert np.isfinite(txt[selection]).all() and np.isnan(txt[~inside]).all(), 'post: txt must be finite on the selection only'
    img = np.full((n_all, k), np.nan, dtype=np.float32)
    img[selection] = img_sel
    assert np.allclose(img[selection].sum(1), 1, rtol=0, atol=1e-5) and np.allclose(txt[selection].sum(1), 1, atol=1e-5), \
        'post: rows must sum to 1'
    return {'img': img, 'txt': txt, 'rows': selection}


def xheads_arrays(img_cf, txt_cf, half_of_local_row, k):
    """xheads_<ID>.npz arrays (R7 format, M12): filled_by_half = 1 - half, classes_half0/1 = arange(K)."""
    half = np.asarray(half_of_local_row)
    n = len(half)
    for name, a in (('img', img_cf), ('txt', txt_cf)):
        assert a.dtype == np.float32 and a.shape == (n, k), f'xheads {name}: shape/dtype'
        assert np.isfinite(a).all() and (a >= 0).all() and np.allclose(a.sum(1), 1, rtol=0, atol=1e-5), \
            f'xheads {name}: rows must be finite, non-negative and sum to 1'
    classes = np.arange(k, dtype=np.int64)
    return {'img': img_cf, 'txt': txt_cf, 'filled_by_half': (1 - half).astype(np.int8),
            'classes_half0': classes, 'classes_half1': classes.copy()}


def drop_record(name, reason, smoke, **extra):
    assert reason, 'a drop needs a reason'
    return {'grouping': name, 'smoke': bool(smoke), 'dropped': True, 'reason': str(reason), **extra}


def sub_halves(h, g_st):
    """Round 1's halves dict restricted to the scorer-train rows of the first paintings (smoke): `g_st` is the sub-set's
    painting codes, ordered as the full scorer-train rows with the rest removed."""
    keep = np.flatnonzero(h['g_st'] < (g_st.max() + 1))
    assert np.array_equal(h['g_st'][keep], g_st), 'smoke halves: painting codes differ'
    st = h['scorer_train'][keep]
    half = h['half_of_local_row'][keep]
    out = {'scorer_train': st, 'half_of_local_row': half, 'g_st': g_st,
           'painting_of_local_row': h['painting_of_local_row'][keep]}
    for j in (0, 1):
        loc = np.flatnonzero(half == j).astype(np.int64)
        out[f'local_rows_half{j}'], out[f'global_rows_half{j}'] = loc, st[loc]
    return out, keep


# ====================================================================== file steps (import r8_common lazily)

def candidates():
    import r8_scoring_io as S
    return S.CANDIDATES, S.REFERENCES


def parse_id(name):
    """(X, is_control): 'ZV100' -> ('V100', True)."""
    cand, refs = candidates()
    if name in cand or name in refs:
        return name, False
    assert name.startswith('Z') and name[1:] in cand, f'unknown grouping ID {name!r} (candidates, R0own, NRK, TP, Z<candidate>)'
    return name[1:], True


def load_grouping(root, gname, names, smoke):
    """The grouping file's labels for `names` (None when absent) and the reason from the json when X was dropped.
    Returns (labels {name: row labels or None}, st, g_st, drop_reasons {name: reason})."""
    import r8_common as C
    path, jpath = root / 'groupings' / f'{gname}.npz', root / 'groupings' / f'{gname}.json'
    assert path.is_file() and jpath.is_file(), f'{path}: no grouping file (run ticket 07 collect)'
    rec = json.loads(jpath.read_text())
    assert bool(rec.get('smoke')) == bool(smoke), f'{jpath}: smoke flag differs'
    c = C.ctx()
    with np.load(path, allow_pickle=False) as z:
        assert {'scorer_train', 'g_st'} <= set(z.files), f'{path}: keys {z.files}'
        st, g_st = np.asarray(z['scorer_train']), np.asarray(z['g_st'])
        labels = {n: (np.asarray(z[n]) if n in z.files else None) for n in names}
    n_p = int(g_st.max()) + 1
    assert n_p == rec['n_paintings'] and (smoke or n_p == c.n_paintings), f'{path}: painting count'
    keep = c.g_st < n_p
    assert np.array_equal(st, c.scorer_train[keep]) and np.array_equal(g_st, c.g_st[keep]), \
        f'{path}: scorer-train rows or painting codes differ from r8_common (C6)'
    for n, lab in labels.items():
        if lab is None:
            assert n in rec['dropped'], f'{path}: {n} is neither in the file nor in the json drops'
        else:
            assert lab.dtype == np.int64 and lab.shape == st.shape, f'{n}: label dtype/shape'
            G.painting_labels(lab, g_st)
            assert np.array_equal(np.unique(lab), np.arange(lab.max() + 1)) and lab.max() >= 1, f'{n}: classes'
    return labels, st, g_st, {n: str(rec['dropped'][n]) for n in names if n in rec['dropped']}, rec, n_p


def selection_for(space_idx, smoke):
    """The selection rows (all of them; in smoke only those whose image is in the space file) and each row's space row."""
    import r8_common as C
    return join_selection(space_idx, C.ctx().selection, C.selection_image(), smoke)


def head_sizes(smoke, head_rows=None, check_rows=None):
    import r8_common as C
    H = C.load_r7('r7_heads')
    hr = (H.SMOKE_HEAD_ROWS if smoke else H.HEAD_ROWS) if head_rows is None else head_rows
    cr = (H.SMOKE_CHECK_ROWS if smoke else H.CHECK_ROWS) if check_rows is None else check_rows
    assert smoke or (hr, cr) == (H.HEAD_ROWS, H.CHECK_ROWS), 'real head sizes are fixed'
    return hr, cr


def draw_check(local, st, halves, hr, cr):
    """R7 L10/L11: every group must be in the standard and in both cross-fit draws. Returns (draws, reason or None)."""
    import r8_common as C
    H = C.load_r7('r7_heads')
    pools = {'standard': st, **{f'half{j}': halves[f'global_rows_half{j}'] for j in (0, 1)}}
    draws = {}
    for key, pool in pools.items():
        assert len(pool) >= hr + cr, f'{key}: insufficient draw/check rows'
        draws[key] = H._draw_record(local, st, pool, hr)
    bad = [f'{key} misses groups {d["missing_groups"]}' for key, d in draws.items() if not d['all_groups_in_draw']]
    return draws, ('missing groups in a head draw (R7 L10/L11): ' + '; '.join(bad) if bad else None)


def r0own_text(labels, keep, selection):
    """R0own's txt: R7's post_R0 / xheads_R0 (same grouping and recipe; M11, M12). Returns (txt global, txt cf, shas)."""
    import r8_common as C
    c = C.ctx()
    # rule gap: M11 "R0own's txt is R7's post_R0.npz txt (same grouping and recipe)": the same grouping is asserted (R0own
    # labels equal R7's R0 labels row by row), so no head is refit and no draw check is run for R0own.
    keys = (R7_R0_POST, R7_R0_XHEADS)
    shas = C.assert_inputs(keys=keys, r7=False)
    with np.load(C.path_of(R7_R0_POST), allow_pickle=False) as z:
        assert np.array_equal(z['rows'], c.selection), 'R7 post_R0: rows differ from the selection'
        txt = np.asarray(z['txt'])
    with np.load(C.path_of(R7_R0_XHEADS), allow_pickle=False) as z:
        txt_cf = np.asarray(z['txt'])[keep]
        assert np.array_equal(z['filled_by_half'][keep], 1 - c.halves_dict['half_of_local_row'][keep]), 'R7 xheads_R0 half'
    with np.load(C.path_of(G.R7_GROUPINGS_KEY), allow_pickle=False) as z:
        assert np.array_equal(z['R0'][keep], labels), 'R0own labels differ from R7 R0 labels (txt columns would misalign)'
    assert txt.shape[1] == 17 and txt_cf.shape == (len(keep), 17) and labels.max() == 16, 'R0own: R7 R0 has 17 classes'
    out = np.full_like(txt, np.nan)
    out[selection] = txt[selection]
    return out, txt_cf, {k: shas[k] for k in keys}


def write_outputs(root, name, post, xheads, record):
    """post, xheads (write once; smoke overwrites), then head_<ID>.json last."""
    import r8_common as C
    d = root / 'heads'
    C.write_npz_once(d / f'post_{name}.npz', **post)
    C.write_npz_once(d / f'xheads_{name}.npz', **xheads)
    return C.write_json_once(d / f'head_{name}.json', record)


def assert_free(root, names):
    import r8_common as C
    for name in names:
        for stem, e in (('post', 'npz'), ('xheads', 'npz'), ('head', 'json')):
            p = root / 'heads' / f'{stem}_{name}.{e}'
            if p.exists() and not C.in_smoke(p):
                raise SystemExit(f'{p} exists; refusing to overwrite (use fix_path for a corrected run)')


def require_placed(root, X):
    """--h-only: X's post, xheads and head record must exist and X must not be dropped. Returns X's head record."""
    d = root / 'heads'
    for stem, e in (('post', 'npz'), ('xheads', 'npz'), ('head', 'json')):
        if not (d / f'{stem}_{X}.{e}').is_file():
            raise SystemExit(f'--h-only: {d / f"{stem}_{X}.{e}"} is missing; place {X} first (r8_place.py --grouping {X})')
    rec = json.loads((d / f'head_{X}.json').read_text())
    if rec.get('dropped') is not False:
        raise SystemExit(f'--h-only: {X} is dropped ({rec.get("reason")}); there is no H{X}')
    return rec


def halves_for(c, st, g_st, n_p):
    """Round 1's halves dict for the scorer-train rows of the first n_p paintings, and the rows kept (all in a real run)."""
    if n_p == c.n_paintings:
        halves, keep = c.halves_dict, np.arange(len(st))
    else:
        halves, keep = sub_halves(c.halves_dict, g_st)
    assert np.array_equal(halves['scorer_train'], st), 'halves: scorer-train rows differ'
    return halves, keep


def write_h(root, X, x_rec, post_h, cross_h, text_rec, draws, selection, halves, k, n_all, built_by):
    """post/xheads/head_H<X>: R7's CLIP image and text heads on X's labels (the fit that gave X's text side)."""
    import r8_common as C
    import r8_bundle as B
    hname = f'H{X}'
    hpost = post_arrays(n_all, selection, post_h['img'][selection], post_h['txt'])
    hx = xheads_arrays(cross_h['img'], cross_h['txt'], halves['half_of_local_row'], k)
    B._posterior_ok({m: hpost[m] for m in ('img', 'txt')}, SimpleNamespace(groups=C.ctx().groups, selection=selection), hname)
    return write_outputs(root, hname, hpost, hx, {**h_record(x_rec, X, text_rec, draws), 'built_by': built_by})


def build_h_only(X, smoke=False, n_smoke=None, gname='r8_groupings', head_rows=None, check_rows=None):
    """H<X> after X is placed (rule 6.6, after the verdict): refit X's CLIP heads (the same `fit_clip_heads` call as X's),
    assert the text heads equal X's stored ones bit for bit, and write only post/xheads/head_H<X>."""
    import r8_common as C
    import r8_recipe as RC
    import r8_scoring_io as S
    C.assert_rule()
    C.assert_threads()
    assert parse_id(X) == (X, False) and f'H{X}' in S.GROUPINGS, f'no H arm for {X}'
    root = C.SMOKE if smoke else C.RESULTS
    assert_free(root, [f'H{X}'])
    x_rec = require_placed(root, X)                   # SystemExit when X's files are missing or X is dropped
    assert bool(x_rec['smoke']) == bool(smoke), f'head_{X}.json: smoke flag differs'
    c = C.ctx()
    labels, st, g_st, _, grec, n_p = load_grouping(root, gname, [X], smoke)
    if smoke:
        assert n_smoke in (None, n_p), f'--smoke {n_smoke}, but {gname} holds {n_p} paintings'
    local = labels[X]
    assert local is not None, f'{X} is not in {gname}'
    assert x_rec['label_sha256'] == C.load_prior('rb_build').sha_array(local), f'{X}: labels differ from the placed ones'
    k = int(G.painting_labels(local, g_st).max()) + 1
    halves, _ = halves_for(c, st, g_st, n_p)
    hr, cr = head_sizes(smoke, head_rows, check_rows)
    assert (x_rec['head_rows'], x_rec['check_rows']) == (hr, cr), f'head_{X}.json: head sizes differ'
    with np.load(root / 'heads' / f'post_{X}.npz', allow_pickle=False) as z:
        selection, stored_g = np.asarray(z['rows']), np.asarray(z['txt'])
    with np.load(root / 'heads' / f'xheads_{X}.npz', allow_pickle=False) as z:
        stored_cf = np.asarray(z['txt'])
    assert smoke or np.array_equal(selection, c.selection), f'post_{X}: rows differ from the selection'
    post_h, cross_h, prov = RC.fit_clip_heads(c.data, local, st, selection, halves, smoke=smoke, head_rows=hr, check_rows=cr)
    assert_same_text(post_h['txt'], stored_g, cross_h['txt'], stored_cf)
    text_rec = {'source': 'r8_recipe.fit_clip_heads (R7 L10/L11, M14)', 'standard': prov['standard'], 'halves': prov['halves']}
    out = write_h(root, X, x_rec, post_h, cross_h, text_rec, prov['draws'], selection, halves, k, len(c.groups), 'h-only')
    print(f'H{X}: placed (K={k}, smoke={smoke}; text heads equal {X}\'s stored ones)')
    return out


def build(name, smoke=False, n_smoke=None, gname='r8_groupings', h_arm=False, head_rows=None, check_rows=None):
    """Place one grouping ID. Returns the head_<ID>.json record (written), dropped or not."""
    import r8_common as C
    import r8_bundle as B
    import r8_recipe as RC
    C.assert_rule()
    C.assert_threads()
    X, control = parse_id(name)
    if h_arm:
        import r8_scoring_io as S
        assert not control and f'H{X}' in S.GROUPINGS, f'no H arm for {name}'
    root = C.SMOKE if smoke else C.RESULTS
    assert_free(root, [name] + ([f'H{X}'] if h_arm else []))
    c = C.ctx()
    labels, st, g_st, listed, grec, n_p = load_grouping(root, gname, [X, name] if control else [X], smoke)
    if smoke:
        assert n_smoke in (None, n_p), f'--smoke {n_smoke}, but {gname} holds {n_p} paintings'
    gfile = root / 'groupings' / f'{gname}.npz'
    common = {'smoke': bool(smoke), 'grouping_file': str(gfile), 'grouping_file_sha256': C.sha_file(gfile),
              'n_paintings': n_p}

    def drop(reason):
        rec = C.write_json_once(root / 'heads' / f'head_{name}.json', drop_record(name, reason, smoke, **{k: v for k, v in common.items() if k != 'smoke'}))
        print(f'{name}: dropped ({reason})')
        return rec

    reason = listed_drop(name, X, control, labels[X] is not None, labels.get(name) is not None, listed, gname)
    if reason is not None:                           # dropped by ticket 07 (M16); a dropped X has no control
        return drop(reason)
    labels_p = G.painting_labels(labels[X], g_st)
    k = int(labels_p.max()) + 1
    halves_p = c.halves[:n_p]
    halves, keep = halves_for(c, st, g_st, n_p)
    assert_halves_agree(halves_p, g_st, halves['half_of_local_row'])
    hr, cr = head_sizes(smoke, head_rows, check_rows)
    # drops that decide X and its control alike: M12 (centres) and R7 L10/L11 (head draws on X's labels)
    draws, x_reason = x_drop(X, control, labels_p, halves_p, k, lambda: draw_check(labels[X], st, halves, hr, cr))
    if x_reason is not None:
        return drop(x_reason)
    local = labels[name]
    if control:
        assert np.array_equal(local, G.random_control_labels(labels[X], g_st, G.CONTROL_INDEX[X])), \
            f'{name}: labels differ from M13 (R7 L8) on {X}'
        draws, z_reason = draw_check(local, st, halves, hr, cr)
        if z_reason is not None:
            return drop(z_reason)
    # --- image side
    if control:
        side = control_side(labels_p, G.CONTROL_INDEX[X], g_st, c.groups[c.selection])
        selection, space_rec = c.selection, {}
    else:
        sp = root / 'spaces' / f'{X}.npz'
        phi, idx = G.read_space(sp, smoke)
        selection, spos = selection_for(idx, smoke)
        phi_p = G.painting_vectors(phi, idx, n_p)
        side = image_side(phi_p, labels_p, halves_p, g_st, phi[spos])
        assert 'dropped' not in side
        if X == 'TP':
            assert_tp_agreement(side)
        space_rec = {'space': str(sp), 'space_sha256': C.sha_file(sp), 'phi_shape': list(phi.shape),
                     'agreement_fit': side['agreement_fit'], 'agreement_crossfit': side['agreement_crossfit'],
                     'sizes_cf_paintings': side['sizes_cf_paintings']}
    # --- text side
    fit = (None, None)
    if X == 'R0own' and not control:
        txt_g, txt_cf, r7_shas = r0own_text(labels[X], keep, selection)
        text_rec = {'source': 'R7 post_R0 / xheads_R0 txt (M11, M12)', 'input_sha256': r7_shas}
    else:
        post_h, cross_h, prov = RC.fit_clip_heads(c.data, local, st, selection, halves, smoke=smoke,
                                                  head_rows=hr, check_rows=cr)
        txt_g, txt_cf = post_h['txt'], cross_h['txt']
        text_rec = {'source': 'r8_recipe.fit_clip_heads (R7 L10/L11, M14)', 'standard': prov['standard'],
                    'halves': prov['halves']}
        fit = (post_h, cross_h)
    n_all = len(c.groups)
    post = post_arrays(n_all, selection, side['img_sel'], txt_g)
    xheads = xheads_arrays(side['img_cf'], txt_cf, halves['half_of_local_row'], k)
    B._posterior_ok({m: post[m] for m in ('img', 'txt')}, SimpleNamespace(groups=c.groups, selection=selection), name)
    # rule gap: head_<ID>.json is not in the ticket's file list; r8_bundle.load_posteriors reads n_groups from it and
    # r8_reader._record reads rule_sha256/smoke/dropped, so it is written (last) for every ID, dropped or placed.
    rec = {'grouping': name, 'smoke': bool(smoke), 'dropped': False, 'reason': None, 'n_groups': k,
           'label_sha256': C.load_prior('rb_build').sha_array(local), 'n_selection_rows': int(len(selection)),
           'selection_basis': 'full selection' if len(selection) == len(c.selection) else 'smoke subset (space images)',
           'head_rows': hr, 'check_rows': cr, 'draws': draws, 'text': text_rec,
           'image_side': 'M13 painting draws (control_draws)' if control else 'M11/M12 own map',
           'sizes_paintings': np.bincount(labels_p, minlength=k).tolist(), **common, **space_rec}
    if control:
        i = G.CONTROL_INDEX[X]
        rec.update(seeds={'labels': 7100 + i, 'scorer_train': 7200 + i, 'selection': 7300 + i}, control_index=i,
                   matched_grouping=X)
    out = write_outputs(root, name, post, xheads, rec)
    # rule gap: the rule defines H<X> only as "the CLIP-head arm" (§2, §6.6); built as R7 L10/L11 on X's labels, image and
    # text heads both, from the fit that gave X's text side (no second fit).
    if h_arm:                                          # R7's CLIP image and text heads on X's labels (descriptive arm)
        write_h(root, X, rec, *fit, text_rec, draws, selection, halves, k, n_all, 'h-arm')
    print(f'{name}: placed (K={k}, smoke={smoke})')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grouping', required=True, help='X (candidate, R0own, NRK, TP) or Z<candidate>')
    ap.add_argument('--smoke', nargs='?', const=SMOKE_PAINTINGS, type=int, metavar='N',
                    help=f'first N paintings of the smoke grouping file (default {SMOKE_PAINTINGS}); writes under results/smoke/')
    ap.add_argument('--groupings-name', default='r8_groupings', help='grouping file stem (NBC: r8_groupings_nbc)')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--h-arm', action='store_true', help='place X and also write post/xheads/head_H<X> (CLIP-head arm, ticket 12)')
    mode.add_argument('--h-only', action='store_true',
                      help='X is placed already: refit its CLIP heads, check the text equals X\'s stored text, write only post/xheads/head_H<X>')
    a = ap.parse_args()
    if a.h_only:
        build_h_only(a.grouping, a.smoke is not None, a.smoke, a.groupings_name)
    else:
        build(a.grouping, a.smoke is not None, a.smoke, a.groupings_name, a.h_arm)


if __name__ == '__main__':
    main()
