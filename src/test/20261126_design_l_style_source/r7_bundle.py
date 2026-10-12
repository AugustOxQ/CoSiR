"""Round 7 per-seed bundle: DECISION_RULE.md §7.3, L11, L14 and G4; R3 D2 to D5, D10, D11.

Diff-able changes from r3_bundle.py's build path:
  imports/modules: reused modules come through r7_common.load, from MAIN only;
  build_bundle guard: r7_common.check_seed and assert_rule replace R3's guards;
  build_bundle: after the unchanged A0 sequence, load R0 and requested X from heads/post_{X}.npz;
  stack_and_features: expose R3's grouping_stack/seed42_features for any parts tuple;
  condition_free: expose L14's per-seed cross-fit, returning (per-anchor dict, picks).
R3's affect-head fit, input hashes, B, Bp, A0 stack/features and A0 readers keep their call sequence.
The build sequence is copied; the SHA-verified R3 affect-head and A0 validation helpers are reused unchanged.
R3's cache, CLI and writers are omitted. This module writes nothing.
"""
import r7_common as R

import time
from types import SimpleNamespace

import numpy as np

A0 = ("affect", "image", "caption")
A1 = A0 + ("R0",)
K_CAND = 13


def modules():
    """The same single instances as R3, loaded through the R7 absolute-path boundary."""
    rc, n6, rto, rg = (R.load(name) for name in ("run_checks", "run_n6", "run_told_oracle", "run_gonogo"))
    assert n6.rc is rc and rto.rc is rc and rto.n6 is n6 and n6.rg is rg and rto.rg is rg, \
        "run_checks / run_n6 / run_gonogo loaded more than once"
    return SimpleNamespace(rc=rc, n6=n6, rto=rto, rg=rg)


def _posterior_ok(post, ctx, name):
    n = len(ctx.groups)
    rows = np.asarray(ctx.selection)
    assert rows.ndim == 1 and rows.dtype == np.int64 and len(rows), "selection: nonempty int64 rows required"
    assert rows[0] >= 0 and rows[-1] < n and np.all(np.diff(rows) > 0), "selection: sorted unique global rows"
    inside = np.zeros(n, dtype=bool)
    inside[rows] = True
    assert set(post) == {"img", "txt"}, f"{name}: img/txt keys required"
    shape = post["img"].shape
    assert len(shape) == 2 and shape[0] == n and shape[1] >= 2, f"{name}: global row count/class shape differs"
    for side, x in post.items():
        assert x.shape == shape and x.dtype == np.float32, f"{name}/{side}: shape or float32 dtype differs"
        assert np.isfinite(x[rows]).all(), f"{name}/{side}: non-finite posterior on selection"
        assert np.isnan(x[~inside]).all(), f"{name}/{side}: posterior is not NaN outside selection"


def load_posteriors(ctx, name):
    """Read one standard head at the global-row boundary, rejecting misaligned or unmasked arrays."""
    assert name in ("R0", "T", "N") or name in tuple(f"{p}{k}" for p in ("V", "Z") for k in range(1, 7)), \
        f"unknown extra grouping {name!r}"
    path = R.RESULTS / "heads" / f"post_{name}.npz"
    with np.load(path, allow_pickle=False) as z:
        assert {"img", "txt", "rows"} <= set(z.files), f"{path}: missing posterior keys"
        rows = z["rows"]
        assert rows.dtype == np.int64 and np.array_equal(rows, ctx.selection), f"{path}: rows differ from selection"
        post = {side: z[side] for side in ("img", "txt")}
    _posterior_ok(post, ctx, name)
    if name == "R0":
        assert post["img"].shape[1] == 17, "R0 must have its frozen 17 classes"
    return post


def stack_and_features(bundle, parts):
    """Return (direction stacks, condition features) in ``parts`` order, using R3's exact definitions."""
    parts = tuple(parts)
    assert len(parts) >= 2 and len(set(parts)) == len(parts), "parts must be distinct groupings"
    C, rbe, rf = (R.load(name) for name in ("common", "rb_eval", "rb_features"))
    expected = [f"{h}__{f}" for h in parts for f in
                ("S", "C", "Delta", "sd_support", "sd_contrast", "argmax_match")]
    assert rf.feature_names(parts) == expected, "reader feature names/order differ from L11"
    for h in parts:
        assert h in bundle.post, f"missing posterior for {h}"
        _posterior_ok(bundle.post[h], bundle.ctx, h)
    stack = C.grouping_stack(bundle.post, bundle.ctx.pooled, parts)
    F, checks = rbe.seed42_features(bundle, parts)
    assert all(checks.values()), "reader features differ from shared definitions"
    for d in ("i2t", "t2i"):
        x = stack[d]
        assert x.shape == (bundle.n, len(parts), K_CAND) and x.dtype == np.float32 and np.isfinite(x).all(), \
            f"stack/{d}: shape, dtype or finite values differ"
    for c in ("a", "b"):
        x = F[c]
        assert x.shape == (bundle.n, len(expected)) and x.dtype == np.float64 and np.isfinite(x).all(), \
            f"features/{c}: shape, dtype or finite values differ"
    return stack, F


def build_bundle(seed, smoke, extra=("R0",)):
    """R3's exact A0 sequence, then R0 and ``extra``. B/Bp are scores; pB/pBp are per-anchor B/B0.

    stack/F/readers remain A0's; call stack_and_features for an extra configuration and condition_free for B1/BX.
    No reader is fitted and no file is written. Seed-42 inputs are hashed by the unchanged R3 checks.
    """
    seed, smoke = int(seed), bool(smoke)
    R.check_seed(seed, smoke)  # before any rule or input read
    R.assert_rule()
    R3, old = (R.load(name) for name in ("r3_common", "r3_bundle"))
    C, rb, rbe, rf = (R.load(name) for name in ("common", "rb_build", "rb_eval", "rb_features"))
    metrics = R.load("src.eval.aspect_metrics")
    quick = R.load("src.eval.aspect_quick_checks")
    shas = dict(C.verify_inputs())  # D15: on every seed, as R3
    shas.update(R3.assert_inputs(old.BUILD_INPUTS + (old.SEED42_INPUTS if seed == 42 and not smoke else [])))
    assert tuple(C.CONFIGS["A0"]) == A0, "round 1's A0 differs from this round's A0"
    M = modules()
    t0 = time.time()
    ctx = M.rg.EvalContext(seed, smoke)
    ep = ctx.pooled
    scorer_train = np.asarray(M.rg.artelingo_splits(ctx.data).scorer_train)
    inp = M.rc.model_inputs(ctx, "A3", scorer_train, False)[0]  # never the smoke checkpoint
    t_n1u = quick.centered_term(inp, ep, uniform=True)
    del inp
    post_e2 = M.n6.load_posteriors(C.INPUT_FILES["n6_posteriors"], ctx)
    B = quick.crossfit_condition_free(ctx.cos, t_n1u, M.n6.n6_terms(post_e2, ep)[2], ctx.parity)[0]  # D10
    post_aff, prov, reused = old._affect_heads(M, ctx, scorer_train)
    checks = {"affect_head_equals_told_oracle_arm_L": old._assert_head_identity(M, prov)}
    post = {"affect": post_aff, "image": post_e2["image"], "caption": post_e2["caption"]}
    del post_e2
    for h in A0:
        _posterior_ok(post[h], ctx, h)
    checks["posteriors_finite_on_selection"] = True
    Bp = quick.crossfit_condition_free(ctx.cos, t_n1u, quick.uniform_probe_scores(post, ep, A0), ctx.parity)[0]  # D11
    stack = C.grouping_stack(post, ep, A0)
    F, fchecks = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post), A0)
    checks.update(fchecks)
    readers, readers_rec, readers_npz = rb.load_readers("A0", False)  # never smoke
    if readers_npz is not None:
        readers_npz.close()
    assert readers["feature_names"] == rf.feature_names(A0), "A0 reader feature names/order differ"
    bundle = SimpleNamespace(
        seed=seed, smoke=smoke, n=int(ctx.n), ctx=ctx, cl=ctx.anchor_group, parity=ctx.parity,
        pair_index=ctx.pair_index, anchor=np.asarray(ep.anchor), cos=ctx.cos, B=B, pB=metrics.per_anchor(B), Bp=Bp,
        pBp=metrics.per_anchor(Bp), t_n1u=t_n1u, post=post, stack=stack, F=F, affect_head=prov,
        affect_heads_reused=bool(reused), readers=readers,
        readers_record={"rule_sha256": readers_rec["rule_sha256"], "smoke": readers_rec["smoke"]},
        episodes_sha256=dict(ctx.shas), input_sha256=shas, checks=checks, from_cache=False,
        build_s=time.time() - t0)
    checks.update(old.validate(bundle))
    # The only extension to the evaluation sequence: posteriors, with no new seed-42 scoring.
    for name in dict.fromkeys(("R0",) + tuple(extra)):
        assert name not in A0, f"extra {name!r} would replace an A0 head"
        path = R.RESULTS / "heads" / f"post_{name}.npz"
        shas[str(path)] = R.sha_file(path)
        post[name] = load_posteriors(ctx, name)
        checks[f"posterior_{name}_rows_shape_mask_ok"] = True
    bundle.build_s = time.time() - t0
    C.log(f"[r7_bundle] seed {seed}: n {bundle.n}, stack {stack['i2t'].shape}, features {F['a'].shape} "
          f"[{bundle.build_s:.0f}s]")
    return bundle


def condition_free(bundle, parts):
    """L14: (per-anchor metrics, parity-half picks) for B0=A0, B1=A1, or BX=A0+(X,)."""
    parts = tuple(parts)
    assert len(parts) >= 2 and len(set(parts)) == len(parts), "parts must be distinct groupings"
    for h in parts:
        assert h in bundle.post, f"missing posterior for {h}"
        _posterior_ok(bundle.post[h], bundle.ctx, h)
    quick = R.load("src.eval.aspect_quick_checks")
    scores, picks = quick.crossfit_condition_free(
        bundle.cos, bundle.t_n1u, quick.uniform_probe_scores(bundle.post, bundle.ctx.pooled, parts), bundle.parity)
    pa = R.load("src.eval.aspect_metrics").per_anchor(scores)
    assert all(x.shape == (bundle.n,) and x.dtype == np.float64 and np.isfinite(x).all() for x in pa.values()), \
        "condition-free per-anchor shape/dtype/values differ"
    assert (pa["gain"] == 0).all(), "condition-free gain must be zero"
    return pa, picks
