"""Round 5: the placement extension of a bundle (DECISION_RULE.md of this folder: D5, D11; §4 item 3; §5 items 4 and 5;
§10 list A item 4).

  extend(bundle, placement)               D5: post_Q (a new dict), stack_Q, F_Q, B'_Q and its per-anchor arrays, from a
                                          freshly built round-4 bundle (r4_bundle.build_bundle) and a placement object
  positive_check(bundle, ext, placement)  D5's positive check (with Q_GE, §5 item 5): booleans only
  same_as_bundle(bundle, ext)             §5 item 4's first bullet (with Q_CLIP the extension is the bundle's own stack,
                                          F and B'(A0)): booleans only
  require_ext / check_pair / validate_ext the extension's guard, its pairing with a bundle and its invariants
  save_ext / load_ext / ext_path          the per-seed extension cache, bound by SHA-256 to round 4's two cache files
  exts_equal / shared_state / post_q      the cache round-trip comparison, D5's fingerprints, post_Q

Every function that takes a placement, or an extension built from one, calls r5_guard.require first (rule D11): a 'ge'
placement is refused until the guard is released. The extension never assigns into ``bundle.post`` or into round 3's
process cache of the affect heads (``r3_bundle._HEADS``, whose caption-posterior dict IS ``bundle.post["affect"]``):
fingerprints of round 3's fields (r4_bundle.r3_fingerprint), of round 4's A1 fields, of the whole of ``bundle.post``
(csd included) and of ``_HEADS``, and the identities of the shared dicts and arrays, are taken before and after, and a
difference stops the run. Nothing here prints a value.
"""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import r5_common as R5
import r5_guard as R5G

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import (_require_condition_free, crossfit_condition_free,  # noqa: E402
                                          uniform_probe_scores)

R3, RB3, RB4, C = R5.R3, R5.RB3, R5.RB4, R5.C
rbe = R3.rbe
A0, A1 = R3.A0, RB4.A1
K_CAND = RB3.K_CAND                         # 13
N_FEATURES = RB3.N_FEATURES                 # 18
AFFECT_COLS = slice(0, 6)                   # D5: affect S, C, Delta, the two spreads, the arg-max match share
OTHER_COLS = slice(6, N_FEATURES)           # image and caption features
EXT_FORMAT = "r5_ext_cache_v1"


def log(msg):
    C.log(f"[r5_bundle] {msg}")


# ---------------------------------------------------------------- the shared state (fingerprints and identities)

def _ids(bundle) -> dict:
    """Identities of the shared objects: bundle.post, its affect dict and arrays, and every _HEADS entry with its post
    dict and arrays (the affect dict of a freshly built bundle IS _HEADS[60000]["post"])."""
    post = bundle.post
    heads = []
    for k, v in RB3._HEADS.items():
        hp = v.get("post") if isinstance(v, dict) else None
        arrays = tuple(sorted((str(s), id(a)) for s, a in hp.items())) if isinstance(hp, dict) else ()
        heads.append((repr(k), id(v), id(hp), arrays))
    return {"post": id(post), "post.affect": id(post["affect"]), "post.affect.img": id(post["affect"]["img"]),
            "post.affect.txt": id(post["affect"]["txt"]),
            "post.entries": tuple((h, id(post[h])) for h in post), "heads": tuple(heads),
            "affect_is_a_cached_head": tuple(post["affect"] is (v.get("post") if isinstance(v, dict) else None)
                                             for v in RB3._HEADS.values())}


def shared_state(bundle) -> dict:
    """{part: fingerprint}: round 3's fields (r4_bundle.r3_fingerprint), round 4's A1 fields (the same fingerprint),
    the whole of bundle.post (csd included), round 3's _HEADS cache, and the identities of the shared objects."""
    return {"round3_fields": RB4.r3_fingerprint(bundle),
            "round4_a1_fields": {k: RB4._fp(getattr(bundle, k), 0, set()) for k in RB4.A1_FIELDS},
            "bundle_post": RB4._fp(bundle.post, 0, set()),
            "r3_bundle_HEADS": RB4._fp(RB3._HEADS, 0, set()),
            "identities": _ids(bundle)}


def _assert_unchanged(before, after, what):
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise AssertionError(f"{what} changed {changed} (rule D5: nothing is assigned into bundle.post, round 3's "
                             f"_HEADS or a field of round 3's or round 4's bundle)")


# ---------------------------------------------------------------- D5

def post_q(post, Q) -> dict:
    """D5: a NEW dict in A0 order, {"affect": {"img": post["affect"]["img"], "txt": Q}, "image": post["image"],
    "caption": post["caption"]}; nothing is assigned into ``post``."""
    return {"affect": {"img": post["affect"]["img"], "txt": Q}, "image": post["image"], "caption": post["caption"]}


def _check_bundle(bundle, what):
    if getattr(bundle, "ctx", None) is None or getattr(bundle, "post", None) is None:
        raise ValueError(f"{what} needs a freshly built round-4 bundle (with ctx and post), not a cache")
    missing = [k for k in RB4.R3_FIELDS + RB4.A1_FIELDS if not hasattr(bundle, k)]
    if missing:
        raise ValueError(f"{what} takes round 4's extended bundle (r4_bundle.build_bundle); missing {missing}")
    if tuple(bundle.post) != A1:
        raise ValueError(f"{what}: bundle.post must hold round 4's groupings {A1} in that order")


def _check_q(Q, bundle):
    """D4's shape and finiteness: float32 (rows, 41), finite on the selection rows, NaN on every other row."""
    p_img = bundle.post["affect"]["img"]
    rows = int(np.asarray(p_img).shape[0])
    if Q.dtype != np.float32 or Q.shape != (rows, R5.N_CLASSES) or np.asarray(p_img).shape[1] != R5.N_CLASSES:
        raise ValueError(f"the placement must be float32 ({rows}, {R5.N_CLASSES}) like the affect image posterior")
    sel = np.zeros(rows, dtype=bool)
    sel[np.asarray(bundle.ctx.selection)] = True
    if not np.isfinite(Q[sel]).all():
        raise ValueError("the placement is not finite on every selection row (D4)")
    if not np.isnan(Q[~sel]).all():
        raise ValueError("the placement must be NaN outside the selection rows (D4)")


def extend(bundle, placement) -> SimpleNamespace:
    """D5 for a placement object (r5_guard.require first: a 'ge' placement is refused before release). From a freshly
    built round-4 bundle: post_Q = post_q(bundle.post, Q) (a new dict), then exactly as round 3's build_bundle forms
    its own fields,
      stack = common.grouping_stack(post_Q, ep, A0)                       {d: (n, 3, 13) float32}
      F     = rb_eval.seed42_features(ns(ctx, post_Q), A0)[0]             {c: (n, 18) float64}
      Bp    = crossfit_condition_free(ctx.cos, t_n1u, uniform_probe_scores(post_Q, ep, A0), ctx.parity)[0]
      pBp   = per_anchor(Bp)
    Asserted: the image and caption slices of stack and columns 6 to 17 of F equal the bundle's exactly; Delta^b =
    -Delta^a; Bp condition-free with condition gain 0 on every episode; round 3's and round 4's fields, the whole of
    bundle.post, _HEADS and the shared identities unchanged. -> SimpleNamespace(kind, placement, placement_sha256,
    seed, smoke, n, stack, F, Bp, pBp, Bp_picks, checks)."""
    R5G.require(placement, "r5_bundle.extend")
    _check_bundle(bundle, "r5_bundle.extend")
    Q = placement.Q
    _check_q(Q, bundle)
    before = shared_state(bundle)
    ctx, ep = bundle.ctx, bundle.ctx.pooled
    pq = post_q(bundle.post, Q)
    if tuple(pq) != A0 or pq is bundle.post or pq["affect"] is bundle.post["affect"]:
        raise AssertionError("post_Q must be a new dict in A0 order (D5)")
    checks = {"post_Q_new_dict_in_A0_order": True}
    stack = C.grouping_stack(pq, ep, A0)
    F, fchecks = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=pq), A0)
    checks.update({f"F_{k}": v for k, v in fchecks.items()})
    Bp, Bp_picks = crossfit_condition_free(ctx.cos, bundle.t_n1u, uniform_probe_scores(pq, ep, A0), ctx.parity)
    pBp = per_anchor(Bp)
    del pq
    _assert_unchanged(before, shared_state(bundle), "the placement extension")
    checks["round3_round4_fields_post_and_HEADS_unchanged"] = True
    ext = SimpleNamespace(kind=placement.kind, placement=placement, placement_sha256=placement.sha256,
                          seed=int(bundle.seed), smoke=bool(bundle.smoke), n=int(bundle.n), stack=stack, F=F, Bp=Bp,
                          pBp=pBp, Bp_picks={int(h): [float(x) for x in v] for h, v in Bp_picks.items()},
                          checks=checks, from_cache=False)
    checks.update(validate_ext(ext))
    checks.update(check_pair(bundle, ext))
    log(f"{placement.kind} extension done: stack {tuple(stack['i2t'].shape)}, F {tuple(F['a'].shape)}")
    return ext


# ---------------------------------------------------------------- the extension's guard and invariants

def require_ext(ext, what):
    """r5_guard.require on the placement the extension was built from, and the extension must carry that placement's
    kind and SHA-256."""
    pl = getattr(ext, "placement", None)
    R5G.require(pl, what)
    if getattr(ext, "kind", None) != pl.kind or getattr(ext, "placement_sha256", None) != pl.sha256:
        raise R5G.GuardError(f"{what}: the extension was not built from this placement")
    return ext


def validate_ext(ext) -> dict:
    """Shapes, dtypes and D5's identities of an extension: stack {d: (n, 3, 13) finite float32}; F {c: (n, 18) finite
    float64} with Delta^b = -Delta^a for every grouping; Bp finite float32 (n, 13), condition-free; pBp = per_anchor(Bp)
    with condition gain 0 on every episode. Raises ValueError."""
    require_ext(ext, "r5_bundle.validate_ext")
    n = int(ext.n)
    if set(ext.stack) != set(DIRECTIONS):
        raise ValueError("stack: keys must be the directions")
    for d in DIRECTIONS:
        x = np.asarray(ext.stack[d])
        if x.shape != (n, len(A0), K_CAND) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"stack/{d}: must be finite float32 of shape ({n}, {len(A0)}, {K_CAND})")
    if set(ext.F) != set(CONDITIONS):
        raise ValueError("F: keys must be the conditions")
    for c in CONDITIONS:
        x = np.asarray(ext.F[c])
        if x.shape != (n, N_FEATURES) or x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError(f"F/{c}: must be finite float64 of shape ({n}, {N_FEATURES})")
    for j in range(len(A0)):
        if not np.array_equal(ext.F["b"][:, 6 * j + 2], -ext.F["a"][:, 6 * j + 2]):
            raise ValueError(f"F: Delta^b != -Delta^a for {A0[j]} (D5)")
    RB3._scores_ok(ext.Bp, n, "Bp")
    _require_condition_free(ext.Bp, "B'_Q")                               # ValueError
    if set(ext.pBp) != set(METRICS) or any(np.asarray(ext.pBp[m]).shape != (n,) for m in METRICS):
        raise ValueError(f"pBp: per-anchor arrays of every metric, shape ({n},)")
    pa = per_anchor(ext.Bp)
    if not all(np.array_equal(np.asarray(ext.pBp[m]), pa[m]) for m in METRICS):
        raise ValueError("pBp is not per_anchor(Bp)")
    if not (np.asarray(ext.pBp["gain"]) == 0).all():
        raise ValueError("B'_Q must have condition gain 0 on every episode (D5)")
    return {"ext_shapes_dtypes_ok": True, "ext_delta_b_equals_minus_delta_a": True, "Bprime_Q_condition_free": True,
            "Bprime_Q_gain_zero": True}


def check_pair(bundle, ext) -> dict:
    """The extension belongs to this bundle: same seed, smoke flag and n; the image and caption slices of the stack and
    columns 6 to 17 of F equal the bundle's exactly (D5). Raises AssertionError."""
    require_ext(ext, "r5_bundle.check_pair")
    if (int(ext.seed), bool(ext.smoke), int(ext.n)) != (int(bundle.seed), bool(bundle.smoke), int(bundle.n)):
        raise AssertionError("the extension was built for another seed, smoke flag or episode count")
    for d in DIRECTIONS:
        x, y = np.asarray(ext.stack[d])[:, 1:], np.asarray(bundle.stack[d])[:, 1:]
        if x.dtype != y.dtype or not np.array_equal(x, y):
            raise AssertionError(f"stack_Q/{d}: the image and caption slices differ from the bundle's (D5)")
    for c in CONDITIONS:
        x, y = np.asarray(ext.F[c])[:, OTHER_COLS], np.asarray(bundle.F[c])[:, OTHER_COLS]
        if x.dtype != y.dtype or not np.array_equal(x, y):
            raise AssertionError(f"F_Q/{c}: columns 6 to 17 differ from the bundle's (D5)")
    return {"stack_image_caption_slices_equal_bundle": True, "F_columns_6_to_17_equal_bundle": True}


def _equal(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y))


def same_as_bundle(bundle, ext) -> dict:
    """§5 item 4's first bullet: {name: bool} that stack_Q, F_Q, B'_Q and its per-anchor arrays equal the bundle's
    stack, F, B'(A0) and pBp exactly (value, shape and dtype), and "all_pass". Booleans only."""
    require_ext(ext, "r5_bundle.same_as_bundle")
    out = {}
    for d in DIRECTIONS:
        out[f"stack.{d}"] = _equal(ext.stack[d], bundle.stack[d])
    for c in CONDITIONS:
        out[f"F.{c}"] = _equal(ext.F[c], bundle.F[c])
        for d in DIRECTIONS:
            out[f"Bp.{c}.{d}"] = _equal(ext.Bp[c][d], bundle.Bp[c][d])
    for m in METRICS:
        out[f"pBp.{m}"] = _equal(ext.pBp[m], bundle.pBp[m])
    out["all_pass"] = all(out.values())
    return out


def positive_check(bundle, ext, placement) -> dict:
    """D5's positive check (with Q = Q_GE, §5 item 5, after release). -> {name: bool}: the placement is the 'ge' one
    and the extension was built from it; the affect slice of stack_Q equals, exactly, the independent recomputation
    einsum('nc,nkc->nk', p_img[anchor], Q[candidates]) (i2t) and einsum('nc,nkc->nk', Q[anchor], p_img[candidates])
    (t2i), p_img being the affect image posterior; it differs from the bundle's affect slice on at least one episode;
    F_Q's columns 0 to 5 differ from F's on at least one episode; "all_pass". Booleans only; nothing is printed."""
    R5G.require(placement, "r5_bundle.positive_check")
    require_ext(ext, "r5_bundle.positive_check")
    ep = bundle.ctx.pooled
    p_img, Q = bundle.post["affect"]["img"], placement.Q
    ind = {"i2t": np.einsum("nc,nkc->nk", p_img[ep.anchor], Q[ep.candidates]),
           "t2i": np.einsum("nc,nkc->nk", Q[ep.anchor], p_img[ep.candidates])}
    out = {"placement_is_ge": placement.kind == "ge",
           "extension_built_from_this_placement": bool(ext.kind == placement.kind
                                                       and ext.placement_sha256 == placement.sha256),
           "affect_slice_equals_independent_einsum": all(_equal(np.asarray(ext.stack[d])[:, 0], ind[d])
                                                         for d in DIRECTIONS),
           "affect_slice_differs_from_bundle_on_an_episode": any(
               bool(np.any(np.asarray(ext.stack[d])[:, 0] != np.asarray(bundle.stack[d])[:, 0])) for d in DIRECTIONS),
           "F_columns_0_to_5_differ_from_bundle_on_an_episode": any(
               bool(np.any(np.asarray(ext.F[c])[:, AFFECT_COLS] != np.asarray(bundle.F[c])[:, AFFECT_COLS]))
               for c in CONDITIONS)}
    out["all_pass"] = all(out.values())
    return out


# ---------------------------------------------------------------- the per-seed cache (§4 item 3)

def ext_path(r4_path, kind="ge") -> Path:
    """The extension file beside round 4's two cache files: <stem>__<kind>.npz (round 4's A1 file is <stem>__a1.npz)."""
    p = Path(r4_path)
    if kind not in ("clip", "ge"):
        raise ValueError(f"kind {kind!r} is not 'clip' or 'ge'")
    return p.with_name(f"{p.stem}__{kind}.npz")


def _out_dir(smoke) -> Path:
    return Path(R5.SMOKE if smoke else R5.RESULTS)


def _check_r4_files(r4_path, r4_shas):
    """Round 4's two cache files (round 3's file and its A1 file beside it) exist and have the given SHA-256s."""
    if not isinstance(r4_shas, dict) or set(r4_shas) != {"r3", "a1"}:
        raise ValueError("r4_shas must be r4_bundle.save_bundle's {'r3': ..., 'a1': ...}")
    r4_path = Path(r4_path)
    for key, p in (("r3", r4_path), ("a1", RB4.a1_path(r4_path))):
        if not p.is_file():
            raise SystemExit(f"{p.name}: round 4's cache file is missing")
        if R5.sha256_file(p) != r4_shas[key]:
            raise SystemExit(f"{p.name}: SHA-256 differs from round 4's recorded one")


def _ext_arrays(ext) -> dict:
    out = {f"stack__{d}": np.asarray(ext.stack[d]) for d in DIRECTIONS}
    out.update({f"F__{c}": np.asarray(ext.F[c]) for c in CONDITIONS})
    for c in CONDITIONS:
        for d in DIRECTIONS:
            out[f"Bp__{c}__{d}"] = np.asarray(ext.Bp[c][d])
    out.update({f"pBp__{m}": np.asarray(ext.pBp[m]) for m in METRICS})
    return out


def save_ext(ext, r4_path, r4_shas) -> str:
    """Write the extension (stack, F, Bp, pBp) with a JSON record to ext_path(r4_path, ext.kind), beside round 4's two
    cache files, which must already be in this round's results/ (smoke: results/smoke/) and have the SHA-256s
    ``r4_shas`` (r4_bundle.save_bundle's {"r3", "a1"}); the record holds both SHA-256s and the placement's, which bind
    the three files. A non-smoke file is never overwritten. Returns the file's SHA-256."""
    require_ext(ext, "r5_bundle.save_ext")
    validate_ext(ext)
    r4_path = Path(r4_path)
    out_dir = _out_dir(ext.smoke)
    if r4_path.resolve().parent != out_dir.resolve():
        raise ValueError(f"round 4's cache files of this round live in {out_dir}, not {r4_path.parent}")
    _check_r4_files(r4_path, r4_shas)
    pg = ext_path(r4_path, ext.kind)
    R5.refuse_existing([pg], ext.smoke)
    arrays = _ext_arrays(ext)
    meta = {"format": EXT_FORMAT, "kind": ext.kind, "placement_sha256": ext.placement_sha256, "seed": int(ext.seed),
            "smoke": bool(ext.smoke), "n": int(ext.n), "groupings": list(A0), "rule_sha256": R5.RULE_SHA,
            "r4_rule_sha256": R5.R4C.RULE_SHA, "r3_rule_sha256": R3.RULE_SHA,
            "r3_cache_file": r4_path.name, "r3_cache_sha256": r4_shas["r3"],
            "a1_cache_file": RB4.a1_path(r4_path).name, "a1_cache_sha256": r4_shas["a1"],
            "Bp_picks": {str(h): v for h, v in ext.Bp_picks.items()}, "checks": ext.checks,
            "provenance": R5.provenance(ext.smoke)}
    meta = C.jsonable(meta)
    C.assert_finite_tree(meta)
    arrays["meta"] = np.array(json.dumps(meta))
    tmp = pg.with_name(pg.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(pg)
    return R5.sha256_file(pg)


def load_ext(r4_path, r4_shas, sha256, placement, bundle) -> SimpleNamespace:
    """The extension from save_ext's file for ``placement`` (r5_guard.require first) and ``bundle`` (round 4's bundle of
    the same seed, built or loaded from round 4's cache files at ``r4_path``). Refused: round 4's files or the extension
    file with another SHA-256; another rule, format, groupings, placement, seed, smoke flag or n; a record bound to
    other round-4 files; an extension whose image and caption slices differ from the bundle's."""
    R5G.require(placement, "r5_bundle.load_ext")
    r4_path = Path(r4_path)
    _check_r4_files(r4_path, r4_shas)
    pg = ext_path(r4_path, placement.kind)
    if not pg.is_file():
        raise SystemExit(f"{pg.name}: the extension file beside {r4_path.name} is missing")
    data = pg.read_bytes()                                  # one read: the bytes hashed are the bytes loaded
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{pg.name}: SHA-256 differs from the recorded one")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][()]))
        g = {k: z[k] for k in z.files if k != "meta"}
    if (meta.get("format") != EXT_FORMAT or meta.get("rule_sha256") != R5.RULE_SHA
            or meta.get("r4_rule_sha256") != R5.R4C.RULE_SHA or meta.get("r3_rule_sha256") != R3.RULE_SHA):
        raise SystemExit(f"{pg.name}: written under another rule or cache format")
    if tuple(meta["groupings"]) != A0:
        raise SystemExit(f"{pg.name}: other groupings {meta['groupings']}")
    if meta["kind"] != placement.kind or meta["placement_sha256"] != placement.sha256:
        raise SystemExit(f"{pg.name}: written for another placement")
    if (meta["r3_cache_file"], meta["r3_cache_sha256"], meta["a1_cache_file"], meta["a1_cache_sha256"]) != \
            (r4_path.name, r4_shas["r3"], RB4.a1_path(r4_path).name, r4_shas["a1"]):
        raise SystemExit(f"{pg.name}: recorded for other round-4 cache files than {r4_path.name}")
    if (meta["seed"], meta["smoke"], meta["n"]) != (int(bundle.seed), bool(bundle.smoke), int(bundle.n)):
        raise SystemExit(f"{pg.name}: seed, smoke flag or n differ from the bundle's")
    ext = SimpleNamespace(
        kind=meta["kind"], placement=placement, placement_sha256=meta["placement_sha256"], seed=int(meta["seed"]),
        smoke=bool(meta["smoke"]), n=int(meta["n"]), stack={d: g[f"stack__{d}"] for d in DIRECTIONS},
        F={c: g[f"F__{c}"] for c in CONDITIONS},
        Bp={c: {d: g[f"Bp__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS},
        pBp={m: g[f"pBp__{m}"] for m in METRICS}, Bp_picks={int(h): v for h, v in meta["Bp_picks"].items()},
        checks=meta["checks"], from_cache=True, cache_meta=meta)
    validate_ext(ext)
    check_pair(bundle, ext)
    return ext


def exts_equal(a, b) -> dict:
    """{array name: exactly equal (value, shape and dtype)} over two extensions' arrays, plus kind, placement, seed,
    smoke, n and the B'_Q cross-fit picks."""
    require_ext(a, "r5_bundle.exts_equal")
    require_ext(b, "r5_bundle.exts_equal")
    x, y = _ext_arrays(a), _ext_arrays(b)
    out = {k: bool(k in y and _equal(x[k], y[k])) for k in x}
    out["kind_placement_seed_smoke_n"] = ((a.kind, a.placement_sha256, a.seed, a.smoke, a.n)
                                          == (b.kind, b.placement_sha256, b.seed, b.smoke, b.n))
    out["Bp_picks"] = {int(h): list(v) for h, v in a.Bp_picks.items()} == {int(h): list(v) for h, v in b.Bp_picks.items()}
    return out
