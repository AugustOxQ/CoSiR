"""r19 registration of DINOv2, CLIP B/32 and CLIP B/16 into r14's and r15's encoder tables (no frozen file is edited).

`import pp_register` (after `pp_paths`) registers once and is idempotent. r14's and r15's functions read their encoder
tables at call time, so adding entries is enough for the code that goes through a table. The functions that read a
feature store through `bs_stores` / `mp_stores` are wrapped: a call for one of the new three goes to `pp_stores`, any
other call goes to the original function (kept as `<function>._pp_original`).

Registered (see the index of the r19 tickets, "Shared interfaces: Registration"):
  bs_paths   ENCODERS, NEW_ENCODERS (appended), MATCHED, FROZEN_ARM
  bs_stores  VIEWS, WIDTH, folders (new views: ArtELingo store, Affection new folder; not EmoSet)
  bs_photo   VIEW;  bs_affection  ENCODERS, VIEW;  bs_artret  ENC (r11_arm None);  bs_artclu  ENC
  rb_encoders.WIDTH (through `eg_encoders.extend_rb_maps()`)
  mt_paths   DEFAULT_VIEW, WIDTH, MATCHED, CSTAR_ROOT, ENCODERS (a tuple, rebound)
  wrapped    bs_artret.frozen_embed, bs_affection.frozen_embed, bs_artclu.store_loader, mt_data.features

Notes for users of this module.
- A module that copied a table at its own import (`bs_outcome.ENCODERS`, `mt_outcome.ENCODERS`, `bs_run.ENCS`,
  `mt_diag.ENCODERS`, `bs_extract.VIEWS`) keeps the old tuple if it was imported before `pp_register`: import
  `pp_register` first, or do not read those copies.
- `mt_paths.VIEWS` (the prep round's seven views, used by r15's gate (i)) is left as it is.
- dinov2_large's `bs_artclu.ENC` root is `M5_ROOT/part2`, the folder whose `full/{held,matched}/M5` holds r19's M5
  (10-26's layout: `EG_PART2/full/...`); it has no stored K5 / F5 (`cand`, `frozen` are None).
"""
import functools
from pathlib import Path

import pp_paths as PP
import pp_stores as S

import bs_affection
import bs_artclu
import bs_artret
import bs_paths
import bs_photo
import bs_stores
import eg_encoders
import mt_data
import mt_paths
import rb_encoders

NEW = tuple(PP.NEW_ENCODERS)                      # dinov2_large, clip_b32, clip_b16
FROZEN_ARM = {'clip_b32': 'F3', 'clip_b16': 'F6', 'dinov2_large': 'F5'}   # 10-25 / 10-26 names (F5 is not stored)


def _extend_tuple(old, new):
    return tuple(dict.fromkeys(tuple(old) + tuple(new)))


def _wrap(module, name, make):
    """Replace module.<name> by make(original) once; the original stays at `.<name>._pp_original`."""
    current = getattr(module, name)
    if hasattr(current, '_pp_original'):
        return
    wrapper = make(current)
    wrapper._pp_original = current
    setattr(module, name, wrapper)


# ------------------------------------------------------------------------------------------------ the wrappers
def _folders(original):
    @functools.wraps(original)
    def folders(dataset, view):
        if view in NEW:
            assert dataset in ('artelingo', 'affection'), f'no {dataset} store for {view} (EmoSet is out of scope)'
            return S.sources(dataset, view)
        return original(dataset, view)
    return folders


def _frozen_embed_art(original):
    @functools.wraps(original)
    def frozen_embed(view, split):
        if view in NEW:
            import rt_data
            return S.load('artelingo', view, rt_data.names(split))
        return original(view, split)
    return frozen_embed


def _frozen_embed_aff(original):
    @functools.wraps(original)
    def frozen_embed(enc, keys):
        if enc in NEW:
            return S.load('affection', enc, keys)
        return original(enc, keys)
    return frozen_embed


def _store_loader(original):
    @functools.wraps(original)
    def store_loader(enc, names):
        if enc in NEW:
            return S.load('artelingo', enc, names)
        return original(enc, names)
    return store_loader


def _features(original):
    @functools.wraps(original)
    def features(view, keys, *, index=None, root=None, art_root=None):
        if view in NEW:
            assert root is None and art_root is None, 'new views: roots are pp_stores.*_ROOT'
            return S.load('mmart', view, keys, index=index)
        return original(view, keys, index=index, root=root, art_root=art_root)
    return features


# ------------------------------------------------------------------------------------------------ register
def register():
    """Add the new three to the tables above; safe to call again."""
    for e in NEW:
        w = PP.WIDTH[e]
        bs_paths.ENCODERS[e] = dict(width=w, art=PP.ART_STORE[e], aff=(PP.ART_STORE[e],), emo=None)
        bs_paths.MATCHED[e] = PP.MATCHED[e]
        bs_paths.FROZEN_ARM[e] = FROZEN_ARM[e]
        bs_stores.WIDTH[e] = w
        bs_photo.VIEW[e] = e
        bs_affection.VIEW[e] = e
        bs_artret.ENC[e] = dict(view=e, r11_arm=None)
        mt_paths.DEFAULT_VIEW[e] = e
        mt_paths.WIDTH[e] = w
        mt_paths.MATCHED[e] = PP.MATCHED[e]
        mt_paths.CSTAR_ROOT[e] = PP.CSTAR_ROOT[e]
    bs_paths.NEW_ENCODERS = _extend_tuple(bs_paths.NEW_ENCODERS, NEW)
    bs_stores.VIEWS = _extend_tuple(bs_stores.VIEWS, NEW)
    bs_affection.ENCODERS = _extend_tuple(bs_affection.ENCODERS, NEW)
    mt_paths.ENCODERS = _extend_tuple(mt_paths.ENCODERS, NEW)
    bs_artclu.ENC['clip_b32'] = dict(store='clip_b32', root=PP.RB_PART2, cand='K3', matched='M3', frozen='F3')
    bs_artclu.ENC['clip_b16'] = dict(store='clip_b16', root=PP.EG_PART2, cand='K6', matched='M6', frozen='F6')
    bs_artclu.ENC['dinov2_large'] = dict(store='dinov2_large', root=PP.M5_ROOT / 'part2', cand=None, matched='M5',
                                         frozen=None)
    eg_encoders.extend_rb_maps()                                  # rb_encoders.WIDTH: clip_b16 (and K5 to K8, M5 to M8)
    _wrap(bs_stores, 'folders', _folders)
    _wrap(bs_artret, 'frozen_embed', _frozen_embed_art)
    _wrap(bs_affection, 'frozen_embed', _frozen_embed_aff)
    _wrap(bs_artclu, 'store_loader', _store_loader)
    _wrap(mt_data, 'features', _features)


# ------------------------------------------------------------------------------------------------ check
def check():
    """Assert every registered entry (value, not only presence); returns {'entries': n}. Names the entry that fails."""
    n = 0

    def ok(cond, what):
        nonlocal n
        assert cond, f'pp_register: {what}'
        n += 1

    for e in NEW:
        w = PP.WIDTH[e]
        ok(bs_paths.ENCODERS.get(e) == dict(width=w, art=PP.ART_STORE[e], aff=(PP.ART_STORE[e],), emo=None),
           f'bs_paths.ENCODERS[{e}]')
        ok(e in bs_paths.NEW_ENCODERS, f'bs_paths.NEW_ENCODERS has {e}')
        ok(bs_paths.MATCHED.get(e) == PP.MATCHED[e], f'bs_paths.MATCHED[{e}]')
        ok(e in bs_stores.VIEWS and bs_stores.WIDTH.get(e) == w, f'bs_stores.VIEWS / WIDTH[{e}]')
        ok(bs_photo.VIEW.get(e) == e, f'bs_photo.VIEW[{e}]')
        ok(e in bs_affection.ENCODERS and bs_affection.VIEW.get(e) == e, f'bs_affection ENCODERS / VIEW[{e}]')
        ok(bs_artret.ENC.get(e) == dict(view=e, r11_arm=None), f'bs_artret.ENC[{e}]')
        ok(rb_encoders.WIDTH.get(PP.ART_STORE[e]) == w, f'rb_encoders.WIDTH[{e}]')
        ok(e in mt_paths.ENCODERS, f'mt_paths.ENCODERS has {e}')
        ok(mt_paths.DEFAULT_VIEW.get(e) == e, f'mt_paths.DEFAULT_VIEW[{e}]')
        ok(mt_paths.WIDTH.get(e) == w, f'mt_paths.WIDTH[{e}]')
        ok(mt_paths.MATCHED.get(e) == PP.MATCHED[e], f'mt_paths.MATCHED[{e}]')
        ok(mt_paths.CSTAR_ROOT.get(e) == PP.CSTAR_ROOT[e], f'mt_paths.CSTAR_ROOT[{e}]')
        ok(bs_artclu.ENC.get(e, {}).get('store') == PP.ART_STORE[e], f'bs_artclu.ENC[{e}] store')
        ok(bs_artclu.ENC[e].get('matched') == PP.MATCHED[e], f'bs_artclu.ENC[{e}] matched arm')
        ok(PP.WIDTH[e] == S.WIDTH[e], f'pp_stores.WIDTH[{e}]')
        ok(S.sources('artelingo', e) == [Path(S.ART_ROOT) / PP.ART_STORE[e]], f'pp_stores artelingo folder of {e}')
        for dataset in ('mmart', 'affection'):
            ok(S.NEW_STORE.get((dataset, e)) == PP.NEW_STORE[(dataset, e)], f'pp_stores.NEW_STORE[({dataset}, {e})]')
            ok(S.sources(dataset, e) == [Path(PP.NEW_STORE[(dataset, e)])], f'pp_stores {dataset} folder of {e}')
    ok(bs_artclu.ENC.get('clip_b32', {}).get('root') == PP.RB_PART2 and bs_artclu.ENC['clip_b32']['cand'] == 'K3'
       and bs_artclu.ENC['clip_b32']['frozen'] == 'F3', 'bs_artclu.ENC[clip_b32] K3/M3/F3 at RB_PART2')
    ok(bs_artclu.ENC.get('clip_b16', {}).get('root') == PP.EG_PART2 and bs_artclu.ENC['clip_b16']['cand'] == 'K6'
       and bs_artclu.ENC['clip_b16']['frozen'] == 'F6', 'bs_artclu.ENC[clip_b16] K6/M6/F6 at EG_PART2')
    ok(bs_artclu.ENC.get('dinov2_large', {}).get('root') == PP.M5_ROOT / 'part2'
       and bs_artclu.ENC['dinov2_large']['cand'] is None and bs_artclu.ENC['dinov2_large']['frozen'] is None,
       'bs_artclu.ENC[dinov2_large] M5 under M5_ROOT, no stored K5 / F5')
    for module, name in ((bs_stores, 'folders'), (bs_artret, 'frozen_embed'), (bs_affection, 'frozen_embed'),
                         (bs_artclu, 'store_loader'), (mt_data, 'features')):
        fn = getattr(module, name)
        ok(hasattr(fn, '_pp_original') and not hasattr(fn._pp_original, '_pp_original'),
           f'{module.__name__}.{name} is wrapped once')
    return {'entries': n}


register()
