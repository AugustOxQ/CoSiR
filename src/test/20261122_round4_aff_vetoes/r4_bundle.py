"""Round 4: the seed-parameterised bundle (DECISION_RULE.md of this folder: §4 item 1, D2 to D5, D11; §5 item 1; §10).

  build_bundle(seed, smoke)            round 3's bundle (r3_bundle.build_bundle under round 4's seed guard), then the A1
                                       extension
  extend_a1(bundle)                    the A1 extension of a pure round-3 bundle: post["csd"] (D2), F1 (D3, 24 features),
                                       readers_a1 (round 1's A1 half-readers), Bp1 / pBp1 (D4, B'(A1)), v (D5); never
                                       changes a round-3 field (asserted by fingerprints taken before and after)
  validate_a1(bundle)                  shapes, dtypes and D3 to D5's identities of an extended bundle (build, save, load)
  compare_a1_with_round1(bundle, r1)   seed 42 only: §5 item 1's A1 comparisons against round 1's common.load_bundle()
  check_v75(bundle)                    seed 42 only: v75 recomputed equals the rule's, and its keep count
  save_bundle / load_bundle_cache      round 3's per-seed cache plus an A1 npz beside it, both SHA-256-recorded

Seed guard (rule §4 item 1): r4_common sets r3_common.TEST_SEEDS = (52, 53, 54) on import; _check_seed asserts that
this is still so and then calls round 3's guard, which admits 42, 52, 53 and 54, and the smoke seeds 9001 to 9003 only
with smoke=True. The extension reads no seed-dependent file: the CSD heads, the A1 readers and run_step1.py are the
same for every episode seed, and the smoke flag is never passed to load_readers or the head files.

Dry check on seed 42 (writes only to results/smoke/; prints only CHECK <name> PASS|FAIL lines, the final
R4_BUNDLE_DRY42 line and the builders' progress lines with counts, shapes and seconds):
    cd src/test/20261122_round4_aff_vetoes && \
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python r4_bundle.py --dry-check-42 > results/smoke/r4_bundle_dry42.log 2>&1
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r4_common as R4  # noqa: E402  (imports round 3's modules by path and sets R3.TEST_SEEDS)

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import (_require_condition_free, crossfit_condition_free,  # noqa: E402
                                          uniform_probe_scores)

R3, RB3, C = R4.R3, R4.RB3, R4.C
rb, rbe, rf = R3.rb, R3.rbe, R3.rf
A0, A1 = R3.A0, R4.A1
N_FEATURES_A1 = 6 * len(A1)                    # D3: 24 features
N_SEED42 = 12288
A1_CACHE_FORMAT = "r4_bundle_a1_cache_v1"

# D11 inputs the extension reads or imports (keys of r4_common.INPUTS, paths relative to src/test/)
STEP1_PY = "20261116_grouping_step1_style/run_step1.py"
HEADS = "20261116_grouping_step1_style/results/step1_heads_style.npz"
READER_A1_INPUTS = ["20261117_reader_fix_csd/results/rb_reader_A1.pkl",
                    "20261117_reader_fix_csd/results/rb_reader_A1.json",
                    "20261117_reader_fix_csd/results/rb_reader_A1.npz"]
A1_INPUTS = ["20261121_round3_affect_gate/DECISION_RULE.md", "20261121_round3_affect_gate/r3_common.py",
             "20261121_round3_affect_gate/r3_bundle.py", STEP1_PY, HEADS] + READER_A1_INPUTS

# the fields of r3_bundle.build_bundle's SimpleNamespace (a test checks this against a real round-3 build)
R3_FIELDS = ("seed", "smoke", "n", "ctx", "cl", "parity", "pair_index", "anchor", "cos", "B", "pB", "Bp", "pBp",
             "t_n1u", "post", "stack", "F", "affect_head", "affect_heads_reused", "readers", "readers_record",
             "episodes_sha256", "input_sha256", "checks", "from_cache", "build_s")
A1_FIELDS = ("F1", "readers_a1", "readers_a1_record", "Bp1", "pBp1", "v", "checks_a1", "input_sha256_a1")


def log(msg):
    C.log(f"[r4_bundle] {msg}")


# ---------------------------------------------------------------- seed guard (rule §4 item 1)

def _check_seed(seed, smoke):
    """Round 3's guard with round 4's seeds: admits 42, 52, 53, 54 (smoke False) and 9001 to 9003 (smoke True)."""
    if tuple(R3.TEST_SEEDS) != tuple(R4.TEST_SEEDS) or tuple(R3.SMOKE_SEEDS) != tuple(R4.SMOKE_SEEDS):
        raise ValueError("round 3's seed guard does not hold round 4's seeds (r4_common sets r3_common.TEST_SEEDS)")
    RB3._check_seed(int(seed), bool(smoke))


# ---------------------------------------------------------------- the guards (T1-1)

def check_heads_selection(z_selection, ctx_selection):
    """D2: the step-1 heads were computed on the context's selection rows (same rows, same order)."""
    if not np.array_equal(np.asarray(z_selection), np.asarray(ctx_selection)):
        raise AssertionError("step-1 heads were computed on another selection than the context's (D2)")
    return True


def check_finite_on_selection(post, selection, parts):
    """D2: every posterior of ``parts`` is finite on the selection rows (rows outside selection may be NaN)."""
    for h in parts:
        for side in ("img", "txt"):
            if not np.isfinite(np.asarray(post[h][side])[selection]).all():
                raise AssertionError(f"{h}/{side}: non-finite posteriors on selection rows")
    return True


def check_parity(parity, n):
    """The parity halves are the episode-index parity (round 3's validate; the cross-fits rely on it)."""
    parity = np.asarray(parity)
    if parity.shape != (n,) or not np.array_equal(parity, np.arange(n) % 2):
        raise ValueError("parity must be the episode-index parity")
    return True


# ---------------------------------------------------------------- the round-3 field fingerprint

_MAX_OBJ_DEPTH = 2


def _fp(x, depth, seen):
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        h = (hashlib.sha256(repr(a.tolist()).encode()) if a.dtype.hasobject
             else hashlib.sha256(a.reshape(-1).view(np.uint8)))          # the array's own buffer, no copy
        return ("nd", x.dtype.str, x.shape, h.hexdigest())
    if isinstance(x, (bool, int, float, complex, str, bytes, type(None), np.generic)):
        return ("v", type(x).__name__, repr(x))
    if isinstance(x, dict):
        return ("dict", tuple((repr(k), _fp(v, depth, seen)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(_fp(v, depth, seen) for v in x))
    if id(x) in seen or depth >= _MAX_OBJ_DEPTH or not hasattr(x, "__dict__"):
        return ("obj", type(x).__name__, id(x))
    seen.add(id(x))
    return ("obj", type(x).__name__, tuple((k, _fp(v, depth + 1, seen)) for k, v in sorted(vars(x).items())))


def r3_fingerprint(bundle) -> dict:
    """{round-3 field: fingerprint}: every array by dtype, shape and SHA-256 of its bytes (inside dicts, lists and
    objects such as ctx, two object levels deep), scalars by value, deeper objects by identity; post on A0's keys."""
    out = {}
    for k in R3_FIELDS:
        x = getattr(bundle, k)
        if k == "post":
            x = {h: x[h] for h in A0}
        out[k] = _fp(x, 0, set())
    return out


# ---------------------------------------------------------------- inputs of the extension

def step1():
    """run_step1 (D11 SHA-256 asserted), loaded by path under its own name after round 3's quick-check modules, so
    run_checks / run_told_oracle stay one instance; its import prints nothing to the console."""
    R4.assert_inputs([STEP1_PY])
    RB3.modules()
    with RB3._quiet():
        return RB3._load_by_path("run_step1", R4.input_path(STEP1_PY))


def _load_csd_heads():
    """{"selection", "style_csd__img", "style_csd__txt"} of step 1's heads file (round 1's common.INPUT_FILES path,
    D11 SHA-256 asserted). The same file for every episode seed and for smoke runs."""
    p = Path(C.INPUT_FILES["step1_heads_style"]).resolve()
    if p != R4.input_path(HEADS).resolve():
        raise AssertionError("round 1's step1_heads_style path is not the rule's D11 file")
    R4.assert_inputs([HEADS])
    with np.load(p) as z:
        return {k: np.asarray(z[k]) for k in ("selection", "style_csd__img", "style_csd__txt")}


def load_readers_a1():
    """Round 1's two A1 half-readers (rb_build.load_readers("A1", False); D11 SHA-256s asserted, never smoke)."""
    R4.assert_inputs(READER_A1_INPUTS)
    pk, rec, rz = rb.load_readers("A1", False)
    if rz is not None:
        rz.close()
    if (pk["config"] != "A1" or tuple(pk["groupings"]) != A1 or list(pk["feature_names"]) != rf.feature_names(A1)
            or len(pk["halves"]) != 2):
        raise SystemExit("rb_reader_A1.pkl: not round 1's two A1 half-readers over (affect, image, caption, csd)")
    return pk, {"rule_sha256": rec["rule_sha256"], "smoke": rec["smoke"]}


def image_abstention_signal(F):
    """D5: v = min(S_image^a, C_image^a), the A0 feature columns 6 and 7 of condition a (float64)."""
    return np.minimum(np.asarray(F["a"])[:, 6], np.asarray(F["a"])[:, 7])


# ---------------------------------------------------------------- the A1 extension

def extend_a1(bundle):
    """Rule §4 item 1's A1 extension of a freshly built round-3 bundle (with ctx): adds post["csd"] (post becomes A1's
    four groupings in A1 order; its A0 entries are round 3's objects), F1, readers_a1, readers_a1_record, Bp1, pBp1, v,
    checks_a1 and input_sha256_a1. Asserted: the heads' selection equals ctx.selection; the four groupings' posteriors
    are finite on selection rows; the parity halves are the episode-index parity; the affect head equals
    told_oracle.json arm L's (round 3's D2 check, re-run); F1[c][:, :18] == F[c]; v^b == v^a; B'(A1) condition-free;
    every round-3 field unchanged. Returns the same bundle object."""
    if bundle.ctx is None:
        raise ValueError("extend_a1 needs a freshly built round-3 bundle (with ctx), not a cache")
    if set(vars(bundle)) != set(R3_FIELDS) or tuple(bundle.post) != A0:
        raise ValueError("extend_a1 takes a pure round-3 bundle (already extended, or another object)")
    t0 = time.time()
    R4.assert_rule()
    shas = R4.assert_inputs(A1_INPUTS)
    ctx, ep, n = bundle.ctx, bundle.ctx.pooled, int(bundle.n)
    before = r3_fingerprint(bundle)
    checks = {"parity_is_episode_index_parity": check_parity(bundle.parity, n) and check_parity(ctx.parity, n)}
    M = RB3.modules()
    checks["affect_head_equals_told_oracle_arm_L"] = RB3._assert_head_identity(M, bundle.affect_head)
    rs1 = step1()
    z = _load_csd_heads()
    checks["csd_heads_selection_equals_ctx_selection"] = check_heads_selection(z["selection"], ctx.selection)
    csd = {"img": rs1.full_post(z, "style_csd__img", ctx), "txt": rs1.full_post(z, "style_csd__txt", ctx)}   # D2
    del z
    post = {h: bundle.post[h] for h in A0}
    post["csd"] = csd
    if tuple(post) != A1:
        raise AssertionError("post must hold A1's groupings in A1 order")
    checks["posteriors_finite_on_selection_A1"] = check_finite_on_selection(post, ctx.selection, A1)
    F1, fchecks = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post), A1)                           # D3
    checks.update({f"F1_{k}": v for k, v in fchecks.items()})
    v = image_abstention_signal(bundle.F)                                                                # D5
    readers_a1, readers_a1_record = load_readers_a1()                                                    # D3
    t6u = uniform_probe_scores(post, ep, A1)                                                             # D4
    Bp1 = crossfit_condition_free(ctx.cos, bundle.t_n1u, t6u, ctx.parity)[0]
    del t6u
    pBp1 = per_anchor(Bp1)
    after = r3_fingerprint(bundle)
    changed = [k for k in R3_FIELDS if before[k] != after[k]]
    if changed:
        raise AssertionError(f"the A1 extension changed round 3's fields {changed}")
    checks["round3_fields_unchanged"] = True
    bundle.post = post
    bundle.F1, bundle.v, bundle.Bp1, bundle.pBp1 = F1, v, Bp1, pBp1
    bundle.readers_a1, bundle.readers_a1_record = readers_a1, readers_a1_record
    bundle.checks_a1, bundle.input_sha256_a1 = checks, shas
    checks.update(validate_a1(bundle))
    log(f"A1 extension done: F1 {tuple(F1['a'].shape)}, Bp1 {tuple(Bp1['a']['i2t'].shape)}, v {tuple(v.shape)} "
        f"[{time.time() - t0:.0f}s]")
    return bundle


def build_bundle(seed: int, smoke: bool) -> SimpleNamespace:
    """Rule §4 item 1 for episode seed ``seed``: round 3's bundle (every field as r3_bundle.build_bundle documents it),
    then extend_a1. The seed guard runs before anything is read."""
    seed, smoke = int(seed), bool(smoke)
    _check_seed(seed, smoke)                                   # before anything is read
    R4.assert_rule()
    return extend_a1(RB3.build_bundle(seed, smoke))


# ---------------------------------------------------------------- invariants of an extended bundle

def validate_a1(b) -> dict:
    """Round 3's validate plus the A1 fields: F1 {c: (n, 24) finite float64} with F1[c][:, :18] == F[c] and
    Delta^b = -Delta^a for the four groupings; v (n,) float64 equal to min(F^a[:, 6], F^a[:, 7]) and to
    min(F1^b[:, 6], F1^b[:, 7]); Bp1 finite float32 (n, 13) and condition-free; pBp1 = per_anchor(Bp1) with condition
    gain 0 on every episode. Raises ValueError."""
    RB3.validate(b)
    n = int(b.n)
    if set(b.F1) != set(CONDITIONS):
        raise ValueError("F1: keys must be the conditions")
    for c in CONDITIONS:
        x = np.asarray(b.F1[c])
        if x.shape != (n, N_FEATURES_A1) or x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError(f"F1/{c}: must be finite float64 of shape ({n}, {N_FEATURES_A1})")
        if not np.array_equal(x[:, :6 * len(A0)], np.asarray(b.F[c])):
            raise ValueError(f"F1/{c}: the first 18 columns differ from the A0 features (D3)")
    for j in range(len(A1)):
        if not np.array_equal(b.F1["b"][:, 6 * j + 2], -b.F1["a"][:, 6 * j + 2]):
            raise ValueError(f"F1: Delta^b != -Delta^a for {A1[j]} (D3)")
    v = np.asarray(b.v)
    if v.shape != (n,) or v.dtype != np.float64:
        raise ValueError(f"v: must be float64 of shape ({n},)")
    if not np.array_equal(v, image_abstention_signal(b.F)):
        raise ValueError("v differs from min(S_image^a, C_image^a) (D5)")
    if not np.array_equal(np.minimum(b.F1["b"][:, 6], b.F1["b"][:, 7]), v):
        raise ValueError("v^b != v^a (D5)")
    RB3._scores_ok(b.Bp1, n, "Bp1")
    _require_condition_free(b.Bp1, "Bp1")                                  # ValueError
    if set(b.pBp1) != set(METRICS) or any(np.asarray(b.pBp1[m]).shape != (n,) for m in METRICS):
        raise ValueError(f"pBp1: per-anchor arrays of every metric, shape ({n},)")
    pa = per_anchor(b.Bp1)
    if not all(np.array_equal(np.asarray(b.pBp1[m]), pa[m]) for m in METRICS):
        raise ValueError("pBp1 is not per_anchor(Bp1)")
    if not (np.asarray(b.pBp1["gain"]) == 0).all():
        raise ValueError("pBp1: a condition-free scorer must have condition gain 0 on every episode")
    return {"F1_shapes_dtypes_ok": True, "F1_first18_equal_A0_features": True,
            "F1_delta_b_equals_minus_delta_a": True, "v_equals_min_image_S_C_a": True, "v_b_equals_v_a": True,
            "Bprime_A1_condition_free": True, "Bprime_A1_gain_zero": True}


# ---------------------------------------------------------------- §5 item 1 (A1 part) and v75 on seed 42

def compare_a1_with_round1(bundle, r1=None) -> dict:
    """§5 item 1 on seed 42, the A1 extension against round 1's common.load_bundle() exactly: post["csd"] (image and
    caption sides), B'(A1) (scores Bp["A1"] and per-anchor pBp["A1"]), the 24 A1 features of both conditions against
    rb_eval.seed42_features(<round 1's bundle>, A1), their first 18 columns against the A0 features, v against round
    1's image columns, round 1's own checks, and B'(A1)'s mean R@1 equal to the rule's 18.804931640625. ``r1`` is a
    round-1 bundle already loaded (seed 42 loads it once for both compare functions); otherwise it is loaded here.
    Round 1's printing is discarded (rule §10); exceptions propagate. -> {"checks", "n_checks", "all_pass"}; raises
    r3_bundle.BundleMismatch (an AssertionError carrying the same dict) on any failure."""
    if (bundle.seed != 42 or bundle.smoke or bundle.ctx is None or getattr(bundle, "F1", None) is None
            or bundle.post is None or "csd" not in bundle.post):
        raise ValueError("compare_a1_with_round1 takes the freshly built, extended, non-smoke seed-42 bundle only")
    with RB3._quiet():                        # round 1 prints seed-42 margins; discarded (rule §10)
        if r1 is None:
            r1 = C.load_bundle(smoke=False)   # asserts round 1's rule, inputs and the step-1 arrays itself
        F1_r1, _ = rbe.seed42_features(r1, A1)
    checks = {}

    def same(name, x, y, nan=False):
        x, y = np.asarray(x), np.asarray(y)
        checks[name] = bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=nan))

    for side in ("img", "txt"):
        same(f"post.csd.{side}", bundle.post["csd"][side], r1.post["csd"][side], nan=True)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            same(f"Bprime_A1.{c}.{d}", bundle.Bp1[c][d], r1.Bp["A1"][c][d])
    for m in METRICS:
        same(f"pBprime_A1.{m}", bundle.pBp1[m], r1.pBp["A1"][m])
    for c in CONDITIONS:
        same(f"features_A1.{c}", bundle.F1[c], F1_r1[c])
        same(f"features_A1.{c}.first18_equal_A0", np.asarray(bundle.F1[c])[:, :6 * len(A0)], bundle.F[c])
    same("v_equals_round1_image_S_C", bundle.v, np.minimum(F1_r1["a"][:, 6], F1_r1["a"][:, 7]))
    checks["round1_load_bundle_own_checks"] = all(v is True for v in r1.checks.values())
    checks["Bprime_A1_mean_r1_equals_rule"] = bool(100 * float(np.mean(bundle.pBp1["r1"])) == R4.BPA1_MEAN_42)
    failed = [k for k, v in checks.items() if v is not True]
    result = {"checks": checks, "n_checks": len(checks), "all_pass": not failed}
    if failed:
        raise RB3.BundleMismatch(f"seed-42 A1 extension differs from round 1's in {len(failed)} of {len(checks)} "
                                 f"comparisons: {failed}", result)
    return result


def check_v75(bundle) -> dict:
    """D5 on seed 42: numpy.percentile(v, 75) recomputed equals the rule's v75 exactly, 1[v < v75] = 1 on the rule's
    keep count, and the episode count is seed 42's. -> {"checks", "n_checks", "all_pass"}; raises BundleMismatch."""
    if bundle.seed != 42 or bundle.smoke:
        raise ValueError("check_v75 takes the non-smoke seed-42 bundle only")
    v = np.asarray(bundle.v)
    checks = {"v75_recomputed_equals_rule": bool(float(np.percentile(v, 75)) == R4.V75),
              "v75_keep_count_equals_rule": bool(int(np.count_nonzero(v < R4.V75)) == R4.V75_KEEP_42),
              "n_episodes_equals_seed42": bool(int(bundle.n) == N_SEED42 and v.shape == (N_SEED42,))}
    failed = [k for k, x in checks.items() if x is not True]
    result = {"checks": checks, "n_checks": len(checks), "all_pass": not failed}
    if failed:
        raise RB3.BundleMismatch(f"v75 check failed: {failed}", result)
    return result


# ---------------------------------------------------------------- the per-seed cache (round 3's plus an A1 npz)

def a1_path(path) -> Path:
    """The A1 npz beside round 3's cache file: <stem>__a1.npz."""
    path = Path(path)
    return path.with_name(path.stem + "__a1.npz")


def _a1_arrays(b) -> dict:
    out = {f"F1__{c}": np.asarray(b.F1[c]) for c in CONDITIONS}
    out["v"] = np.asarray(b.v)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            out[f"Bp1__{c}__{d}"] = np.asarray(b.Bp1[c][d])
    for m in METRICS:
        out[f"pBp1__{m}"] = np.asarray(b.pBp1[m])
    return out


def save_bundle(bundle, path) -> dict:
    """Round 3's cache at ``path`` (r3_bundle.save_bundle) and the A1 arrays (F1, v, Bp1, pBp1) with a JSON record at
    a1_path(path); the A1 record holds the round-3 file's SHA-256, which binds the pair. A non-smoke cache is never
    overwritten (both paths checked before anything is written). -> {"r3": SHA-256, "a1": SHA-256}."""
    path = Path(path)
    p1 = a1_path(path)
    validate_a1(bundle)
    R4.refuse_existing([path, p1], bundle.smoke)
    sha_r3 = RB3.save_bundle(bundle, path)
    arrays = _a1_arrays(bundle)
    meta = {"format": A1_CACHE_FORMAT, "seed": int(bundle.seed), "smoke": bool(bundle.smoke), "n": int(bundle.n),
            "groupings": list(A1), "rule_sha256": R4.RULE_SHA, "r3_rule_sha256": R3.RULE_SHA,
            "r3_cache_file": path.name, "r3_cache_sha256": sha_r3, "checks_a1": bundle.checks_a1,
            "readers_a1_record": bundle.readers_a1_record, "input_sha256_a1": bundle.input_sha256_a1,
            "provenance": R4.provenance(bundle.smoke)}
    meta = C.jsonable(meta)
    C.assert_finite_tree(meta)
    arrays["meta"] = np.array(json.dumps(meta))
    tmp = p1.with_name(p1.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(p1)
    return {"r3": sha_r3, "a1": R4.sha_file(p1)}


def load_bundle_cache(path, seed, smoke, sha256, readers=True) -> SimpleNamespace:
    """An extended bundle from save_bundle's two files (ctx and post are None). ``sha256`` = save_bundle's
    {"r3", "a1"}; both are required and checked. Refused: either SHA-256 differs; another rule, format or groupings;
    a seed, smoke flag or n other than given or than round 3's file; an A1 record bound to another round-3 file.
    readers=True reloads round 1's A0 and A1 half-readers (SHA-256s asserted)."""
    if not isinstance(sha256, dict) or set(sha256) != {"r3", "a1"}:
        raise ValueError("sha256 must be save_bundle's {'r3': ..., 'a1': ...}")
    path = Path(path)
    p1 = a1_path(path)
    b = RB3.load_bundle_cache(path, seed=seed, smoke=smoke, sha256=sha256["r3"], readers=readers)
    if not p1.exists():
        raise SystemExit(f"{p1.name}: the A1 cache beside {path.name} is missing")
    if R4.sha_file(p1) != sha256["a1"]:
        raise SystemExit(f"{p1.name}: SHA-256 differs from the recorded one")
    with np.load(p1, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][()]))
        if meta.get("format") != A1_CACHE_FORMAT or meta.get("rule_sha256") != R4.RULE_SHA \
                or meta.get("r3_rule_sha256") != R3.RULE_SHA:
            raise SystemExit(f"{p1.name}: written under another rule or cache format")
        if tuple(meta["groupings"]) != A1:
            raise SystemExit(f"{p1.name}: other groupings {meta['groupings']}")
        if (meta["seed"], meta["smoke"], meta["n"]) != (int(seed), bool(smoke), b.n):
            raise SystemExit(f"{p1.name}: seed, smoke flag or n differ from the requested or round 3's file")
        if meta["r3_cache_sha256"] != sha256["r3"]:
            raise SystemExit(f"{p1.name}: recorded for another round-3 cache file than {path.name}")
        g = {k: z[k] for k in z.files if k != "meta"}
    b.F1 = {c: g[f"F1__{c}"] for c in CONDITIONS}
    b.v = g["v"]
    b.Bp1 = {c: {d: g[f"Bp1__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS}
    b.pBp1 = {m: g[f"pBp1__{m}"] for m in METRICS}
    b.checks_a1, b.input_sha256_a1 = meta["checks_a1"], meta["input_sha256_a1"]
    b.readers_a1, b.readers_a1_record = None, None
    b.a1_cache_meta = meta
    validate_a1(b)
    if readers:
        b.readers_a1, b.readers_a1_record = load_readers_a1()
    return b


def bundles_equal(a, b) -> dict:
    """{array name: exactly equal (value, shape and dtype)} over round 3's cached arrays and the A1 arrays."""
    out = RB3.bundles_equal(a, b)
    x, y = _a1_arrays(a), _a1_arrays(b)
    out.update({f"a1.{k}": bool(k in y and x[k].shape == y[k].shape and x[k].dtype == y[k].dtype
                                and np.array_equal(x[k], y[k])) for k in x})
    return out


# ---------------------------------------------------------------- dry check on seed 42 (prints only pass or fail)

def dry_check_42() -> bool:
    """§5 item 1 on seed 42 through this round's code path: round 3's comparisons (r3_bundle.compare_with_round1),
    the A1 comparisons, v75 and its keep count, and the cache round trip. Round 1's bundle is loaded once. Prints only
    pass or fail; the record in results/smoke/ holds only booleans and counts; the cache files are deleted."""
    R4.assert_rule()
    out_dir = R4.res_dir(True)
    t0 = time.time()
    bundle = build_bundle(42, False)
    print("CHECK build_bundle_seed42 PASS", flush=True)
    try:
        with RB3._quiet():
            r1 = C.load_bundle(smoke=False)
    except BaseException as e:                # round 1's own stop: marker, re-raise
        print(f"CHECK round1_load_bundle FAIL ({type(e).__name__})", flush=True)
        print("R4_BUNDLE_DRY42 FAIL", flush=True)
        raise
    print("CHECK round1_load_bundle PASS", flush=True)
    rec, ok = {}, True
    for part, fn in (("r3", lambda: RB3.compare_with_round1(bundle, r1)),
                     ("a1", lambda: compare_a1_with_round1(bundle, r1)),
                     ("v75", lambda: check_v75(bundle))):
        try:
            res = fn()
        except RB3.BundleMismatch as e:
            res = e.result
        except BaseException as e:
            print(f"CHECK {part} FAIL ({type(e).__name__})", flush=True)
            print("R4_BUNDLE_DRY42 FAIL", flush=True)
            raise
        for k, v in res["checks"].items():
            print(f"CHECK {part}.{k} {'PASS' if v is True else 'FAIL'}", flush=True)
        rec[part] = res["checks"]
        ok &= bool(res["all_pass"])
    cache = out_dir / "r4_bundle_dry42_cache.npz"
    cache.unlink(missing_ok=True)
    a1_path(cache).unlink(missing_ok=True)
    try:
        shas = save_bundle(bundle, cache)
        eq = bundles_equal(bundle, load_bundle_cache(cache, 42, False, shas))
    finally:
        cache.unlink(missing_ok=True)
        a1_path(cache).unlink(missing_ok=True)
    cache_ok = all(eq.values())
    print(f"CHECK cache_round_trip_exact {'PASS' if cache_ok else 'FAIL'}", flush=True)
    ok &= cache_ok
    n_checks = 2 + sum(len(v) for v in rec.values()) + 1
    R4.write_json_once(out_dir / "r4_bundle_dry42.json",
                       {"what": "r4_bundle dry check on seed 42 (rule §5 item 1, D5's v75); smoke output, not a "
                                "result; booleans only",
                        "checks": rec, "cache_round_trip": eq, "n_checks": n_checks, "all_pass": bool(ok),
                        "seconds": round(time.time() - t0)}, smoke=True)
    print(f"R4_BUNDLE_DRY42 {'PASS' if ok else 'FAIL'} ({n_checks} checks)", flush=True)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-check-42", action="store_true", help="§5 item 1 on seed 42; writes only to results/smoke/")
    args = ap.parse_args()
    if not args.dry_check_42:
        ap.error("nothing to do (use --dry-check-42)")
    sys.exit(0 if dry_check_42() else 1)


if __name__ == "__main__":
    main()
