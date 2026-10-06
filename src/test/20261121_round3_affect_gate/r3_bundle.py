"""Round 3: the seed-parameterised bundle (DECISION_RULE.md of this folder: §4 items 1 and 2, D1 to D5, D7, D10, D11,
D15; §5 item 1; the per-seed cache of §6.4).

  build_bundle(seed, smoke)         everything one episode seed's evaluation needs, in rule §4 item 1's call sequence
  load_external(bundle)             cosine and RCA per anchor from run_baselines.py's per_anchor_seed{s}.npz (§4 item 2)
  redundancy(bundle)                D7: mean per-row Pearson correlation of z(s_h) with z(B) (the brainstorm's row_corr)
  compare_with_round1(bundle)       seed 42 only: every comparison of §5 item 1 against round 1's common.load_bundle()
  save_bundle / load_bundle_cache   the per-seed cache (§6.4)

No step of build_bundle reads seed-42 arrays (C.verify_inputs only hashes them); run_sweep.setup and
common.load_bundle are called only by compare_with_round1, a separate layer on top. The smoke flag changes only the
episode source (run_gonogo.EvalContext(s, True) reads src/test/20261030_aspect_baselines/results/smoke/); the A3
checkpoint, the 60,000-row head draw, the posteriors and the readers are the real ones in every mode (the flag is never
passed to model_inputs or load_readers).

The two quick-check modules named run_checks: this file loads src/test/20261108_new_method_quick_checks/run_checks.py
by path (importlib) under the name run_checks, which is the module run_n6, run_told_oracle and round 1's run_step1 all
import under that name; a different module already registered as run_checks stops the import.

Dry check on seed 42 (writes only to results/smoke/). Its output is CHECK <name> PASS|FAIL lines, the final
R3_BUNDLE_DRY42 line and r3_bundle's own progress lines (counts, shapes, seconds). compare_with_round1 discards
everything round 1's load_bundle and run_sweep.setup print, so no seed-42 value appears (rule §10); exceptions still
propagate:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261121_round3_affect_gate/r3_bundle.py --dry-check-42 \
        > src/test/20261121_round3_affect_gate/results/smoke/r3_bundle_dry42.log 2>&1
"""
import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r3_common as R3  # noqa: E402  (puts round 1, round 2 and the repo root on sys.path)

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import (_require_condition_free, centered_term,  # noqa: E402
                                          crossfit_condition_free, uniform_probe_scores)

C, rb, rbe = R3.C, R3.rb, R3.rbe
A0 = R3.A0
TEST = R3.TEST
QC = TEST / "20261108_new_method_quick_checks"
TOLD_DIR = TEST / "20261111_community_told_oracle"
GONOGO = TEST / "20261101_aspect_factor_gonogo"
K_CAND = 13
N_FEATURES = 6 * len(A0)                       # D5: 18 reader features
CACHE_FORMAT = "r3_bundle_cache_v1"

# D15 inputs every build reads or imports (keys of r3_common.INPUTS, paths relative to src/test/)
TOLD_JSON = "20261111_community_told_oracle/results/told_oracle.json"
BUILD_INPUTS = [
    "20261117_reader_fix_csd/DECISION_RULE.md",
    "20261117_reader_fix_csd/common.py",
    "20261117_reader_fix_csd/rc_core.py",
    "20261117_reader_fix_csd/rb_build.py",
    "20261117_reader_fix_csd/rb_eval.py",
    "20261117_reader_fix_csd/rb_features.py",
    "20261117_reader_fix_csd/results/rb_reader_A0.pkl",
    "20261117_reader_fix_csd/results/rb_reader_A0.json",
    TOLD_JSON,
    "20261111_community_told_oracle/results/per_anchor_told_oracle.npz",
    "20261031_pseudo_partitions/results/partitions.npz",
    "20261108_new_method_quick_checks/results/n6_posteriors.npz",
    "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt",
]
READER_INPUTS = BUILD_INPUTS[6:8]
SEED42_INPUTS = ["20261030_aspect_baselines/results/episodes_seed42.npz",
                 "20261030_aspect_baselines/results/baselines_seed42.json"]
EXTERNAL_42 = "20261030_aspect_baselines/results/per_anchor_seed42.npz"


def log(msg):
    C.log(f"[r3_bundle] {msg}")


# ---------------------------------------------------------------- the older exploratory modules, by path

_MODS = None


def _load_by_path(name, path):
    """Import ``path`` as module ``name``; if ``name`` is already registered it must be this very file."""
    path = Path(path).resolve()
    mod = sys.modules.get(name)
    if mod is not None:
        if Path(mod.__file__).resolve() != path:
            raise ImportError(f"module {name!r} is already {mod.__file__}, not {path}")
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def modules():
    """SimpleNamespace(rc, n6, rto, rg): the quick-check run_checks, run_n6, run_told_oracle and run_gonogo, each one
    instance shared by every importer (asserted)."""
    global _MODS
    if _MODS is None:
        rc = _load_by_path("run_checks", QC / "run_checks.py")
        n6 = _load_by_path("run_n6", QC / "run_n6.py")
        rto = _load_by_path("run_told_oracle", TOLD_DIR / "run_told_oracle.py")
        rg = rc.rg
        if Path(rg.__file__).resolve() != (GONOGO / "run_gonogo.py").resolve():
            raise ImportError(f"run_gonogo resolved to {rg.__file__}")
        if not (n6.rc is rc and rto.rc is rc and rto.n6 is n6 and n6.rg is rg and rto.rg is rg):
            raise ImportError("run_checks / run_n6 / run_gonogo are loaded twice under different instances")
        _MODS = SimpleNamespace(rc=rc, n6=n6, rto=rto, rg=rg)
    return _MODS


# ---------------------------------------------------------------- build

_HEADS = {}


def _check_seed(seed, smoke):
    allowed = tuple(R3.SMOKE_SEEDS) if smoke else (42,) + tuple(R3.TEST_SEEDS)
    if seed not in allowed:
        raise ValueError(f"episode seed {seed} (smoke {smoke}) is not one of this round's seeds {allowed}")


def _affect_heads(M, ctx, scorer_train):
    """D2: the affect heads refit with run_told_oracle.fit_one_head on partition_L (60,000-row draw). They depend on
    no episode seed, so one fit serves every bundle of this process (same rows asserted; arrays read-only)."""
    key = int(M.n6.HEAD_ROWS)
    hit = _HEADS.get(key)
    if hit is not None:
        if not (np.array_equal(hit["selection"], ctx.selection) and np.array_equal(hit["scorer_train"], scorer_train)
                and hit["n_rows_all"] == len(ctx.groups)):
            raise AssertionError("the cached affect heads were fitted for other rows")
        log("affect heads: reused from this process")
        return hit["post"], hit["prov"], True
    with np.load(C.INPUT_FILES["affect_L_partition(per_anchor_told_oracle)"]) as z:
        partition_L = np.asarray(z["partition_L"], dtype=np.int64)
    t = time.time()
    post, prov = M.rto.fit_one_head(ctx, M.rto.global_labels(partition_L, scorer_train, len(ctx.groups)),
                                    scorer_train, M.n6.HEAD_ROWS)
    for side in post:
        post[side].flags.writeable = False
    _HEADS[key] = {"post": post, "prov": prov, "selection": np.array(ctx.selection, copy=True),
                   "scorer_train": np.array(scorer_train, copy=True), "n_rows_all": len(ctx.groups)}
    log(f"affect heads: fitted on {key} rows [{time.time() - t:.0f}s]")
    return post, prov, False


def _assert_head_identity(M, prov):
    """D2: the refit head equals told_oracle.json arm L's head (SHA-256 of told_oracle.json asserted)."""
    R3.assert_inputs([TOLD_JSON])
    stored = json.loads(R3.input_path(TOLD_JSON).read_text())
    if M.rto.roundtrip(prov) != stored["arms"]["L"]["head"]:
        raise SystemExit("affect head differs from told_oracle.json arm L's head (D2)")
    return True


def build_bundle(seed: int, smoke: bool) -> SimpleNamespace:
    """Rule §4 item 1 for episode seed ``seed``. Attributes: seed, smoke, n, ctx, cl, parity, pair_index, anchor, cos,
    B, pB, Bp (B'(A0)), pBp, t_n1u, post (A0: {h: {"img", "txt"}}), stack ({d: (n, 3, 13)}), F ({c: (n, 18)}),
    affect_head, affect_heads_reused, readers (round 1's A0 half-readers), episodes_sha256, input_sha256, checks
    (every value True). Scores are {c: {d: (n, 13) float32}}; per-anchor dicts {metric: (n,)}."""
    seed, smoke = int(seed), bool(smoke)
    _check_seed(seed, smoke)                                   # before anything is read
    R3.assert_rule()
    shas = dict(C.verify_inputs())                             # D15: on every seed
    shas.update(R3.assert_inputs(BUILD_INPUTS + (SEED42_INPUTS if seed == 42 and not smoke else [])))
    if tuple(C.CONFIGS["A0"]) != A0:
        raise AssertionError("round 1's A0 differs from this round's A0")
    from src.data.artelingo_splits import artelingo_splits

    M = modules()
    t0 = time.time()
    ctx = M.rg.EvalContext(seed, smoke)                        # episodes, parity halves, paintings, pair index, cosine
    ep = ctx.pooled
    log(f"seed {seed}{' (smoke)' if smoke else ''}: {ctx.n} episodes ({ctx.n_per_pair} per pair)")
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    inp = M.rc.model_inputs(ctx, "A3", scorer_train, False)[0]   # never the smoke checkpoint
    t_n1u = centered_term(inp, ep, uniform=True)
    del inp
    post_e2 = M.n6.load_posteriors(C.INPUT_FILES["n6_posteriors"], ctx)
    B = crossfit_condition_free(ctx.cos, t_n1u, M.n6.n6_terms(post_e2, ep)[2], ctx.parity)[0]          # D10
    log("B built")
    post_aff, prov, reused = _affect_heads(M, ctx, scorer_train)
    checks = {"affect_head_equals_told_oracle_arm_L": _assert_head_identity(M, prov)}
    post = {"affect": post_aff, "image": post_e2["image"], "caption": post_e2["caption"]}                # D2
    del post_e2
    for h in A0:
        for side in ("img", "txt"):
            if not np.isfinite(post[h][side][ctx.selection]).all():
                raise AssertionError(f"{h}/{side}: non-finite posteriors on selection rows")
    checks["posteriors_finite_on_selection"] = True
    Bp = crossfit_condition_free(ctx.cos, t_n1u, uniform_probe_scores(post, ep, A0), ctx.parity)[0]      # D11
    log("B'(A0) built")
    stack = C.grouping_stack(post, ep, A0)                                                                # D4
    F, fchecks = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post), A0)                            # D5
    checks.update(fchecks)
    readers, readers_rec, readers_npz = rb.load_readers("A0", False)                                     # never smoke
    if readers_npz is not None:
        readers_npz.close()
    bundle = SimpleNamespace(
        seed=seed, smoke=smoke, n=int(ctx.n), ctx=ctx, cl=ctx.anchor_group, parity=ctx.parity,
        pair_index=ctx.pair_index, anchor=np.asarray(ep.anchor), cos=ctx.cos, B=B, pB=per_anchor(B), Bp=Bp,
        pBp=per_anchor(Bp), t_n1u=t_n1u, post=post, stack=stack, F=F, affect_head=prov,
        affect_heads_reused=bool(reused), readers=readers,
        readers_record={"rule_sha256": readers_rec["rule_sha256"], "smoke": readers_rec["smoke"]},
        episodes_sha256=dict(ctx.shas), input_sha256=shas, checks=checks, from_cache=False,
        build_s=time.time() - t0)
    checks.update(validate(bundle))
    log(f"bundle done: n {bundle.n}, stack {tuple(stack['i2t'].shape)}, features {tuple(F['a'].shape)} "
        f"[{bundle.build_s:.0f}s]")
    return bundle


# ---------------------------------------------------------------- invariants of a bundle (build, save, load)

def _scores_ok(s, n, what):
    if set(s) != set(CONDITIONS) or any(set(s[c]) != set(DIRECTIONS) for c in CONDITIONS):
        raise ValueError(f"{what}: keys must be conditions x directions")
    for c in CONDITIONS:
        for d in DIRECTIONS:
            x = np.asarray(s[c][d])
            if x.shape != (n, K_CAND) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f"{what}/{c}/{d}: must be finite float32 of shape ({n}, {K_CAND})")


def validate(b) -> dict:
    """Shapes, dtypes and the rule's identities: cosine, B, B' and T_N1u condition-free, B and B' with condition gain
    exactly 0 on every episode, Delta^b = -Delta^a in the features (D3). Raises ValueError."""
    n = int(b.n)
    for k in ("cl", "parity", "pair_index", "anchor"):
        if np.asarray(getattr(b, k)).shape != (n,):
            raise ValueError(f"{k}: shape must be ({n},)")
    if not np.array_equal(np.asarray(b.parity), np.arange(n) % 2):
        raise ValueError("parity must be the episode-index parity")
    for k in ("cos", "B", "Bp", "t_n1u"):
        _scores_ok(getattr(b, k), n, k)
        _require_condition_free(getattr(b, k), k)                         # ValueError
    for k in ("pB", "pBp"):
        pa = getattr(b, k)
        if set(pa) != set(METRICS) or any(np.asarray(pa[m]).shape != (n,) for m in METRICS):
            raise ValueError(f"{k}: per-anchor arrays of every metric, shape ({n},)")
        if not (np.asarray(pa["gain"]) == 0).all():
            raise ValueError(f"{k}: a condition-free scorer must have condition gain 0 on every episode")
    if set(b.stack) != set(DIRECTIONS):
        raise ValueError("stack: keys must be the directions")
    for d in DIRECTIONS:
        x = np.asarray(b.stack[d])
        if x.shape != (n, len(A0), K_CAND) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"stack/{d}: must be finite float32 of shape ({n}, {len(A0)}, {K_CAND})")
    if set(b.F) != set(CONDITIONS):
        raise ValueError("F: keys must be the conditions")
    for c in CONDITIONS:
        x = np.asarray(b.F[c])
        if x.shape != (n, N_FEATURES) or x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError(f"F/{c}: must be finite float64 of shape ({n}, {N_FEATURES})")
    for j in range(len(A0)):
        if not np.array_equal(b.F["b"][:, 6 * j + 2], -b.F["a"][:, 6 * j + 2]):
            raise ValueError(f"F: Delta^b != -Delta^a for {A0[j]} (D3)")
    return {"shapes_dtypes_ok": True, "cos_B_Bprime_Tn1u_condition_free": True, "B_Bprime_gain_zero": True,
            "delta_b_equals_minus_delta_a": True}


# ---------------------------------------------------------------- external baselines (§4 item 2)

def load_external(bundle, path=None) -> dict:
    """{"cosine": per_anchor, "rca": per_anchor} (float64) from run_baselines.py's per_anchor_seed{s}.npz (its smoke
    folder in smoke mode). Asserted: its anchor_group and pair_index equal the bundle's and its cosine__* arrays equal
    per_anchor(bundle.cos) exactly. ``path`` overrides the file (tests only)."""
    if path is None:
        path = R3.AB / "results" / ("smoke" if bundle.smoke else "") / f"per_anchor_seed{bundle.seed}.npz"
        if bundle.seed == 42 and not bundle.smoke:
            R3.assert_inputs([EXTERNAL_42])
    cos_pa = per_anchor(bundle.cos)
    with np.load(path) as z:
        ok_g = bool(np.array_equal(z["anchor_group"], bundle.cl))
        ok_p = bool(np.array_equal(z["pair_index"], bundle.pair_index))
        ok_c = all(bool(np.array_equal(cos_pa[m], z[f"cosine__{m}"])) for m in METRICS)
        if not (ok_g and ok_p and ok_c):
            raise AssertionError(f"{Path(path).name}: misaligned with the bundle (anchor_group equal {ok_g}, "
                                 f"pair_index equal {ok_p}, cosine equal {ok_c})")
        out = {k: {m: np.asarray(z[f"{k}__{m}"], dtype=np.float64) for m in METRICS} for k in ("cosine", "rca")}
    for k in out:
        for m in METRICS:
            if out[k][m].shape != (bundle.n,) or not np.isfinite(out[k][m]).all():
                raise AssertionError(f"{Path(path).name}: {k}__{m} is not a finite ({bundle.n},) array")
    return out


# ---------------------------------------------------------------- D7 redundancy

def row_corr(x, y):
    """The brainstorm's bs_05_aff.row_corr: mean over rows of the Pearson correlation; rows whose denominator is 0 are
    left out."""
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x * x).sum(1) * (y * y).sum(1))
    ok = den > 0
    return float(np.mean((x * y).sum(1)[ok] / den[ok]))


def _zrows64(x):
    """D8's per-row z-score (zscore_rows, float32), then cast to float64 (the brainstorm's bs_lib.zrows)."""
    import torch
    from src.model.aspect_rule import zscore_rows
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy().astype(np.float64)


def redundancy(bundle) -> dict:
    """D7: {h: {d: mean over the seed's ranking rows of direction d of corr(z(s_h), z(B))}} for h in A0 (one row per
    episode; B must be condition-free)."""
    _require_condition_free(bundle.B, "B")
    zB = {d: _zrows64(bundle.B["a"][d]) for d in DIRECTIONS}
    return {h: {d: row_corr(_zrows64(bundle.stack[d][:, j]), zB[d]) for d in DIRECTIONS} for j, h in enumerate(A0)}


def affect_least_redundant(red) -> bool:
    """D7's order: affect has the strictly smallest redundancy in both directions."""
    return all(red["affect"][d] < min(red[h][d] for h in A0 if h != "affect") for d in DIRECTIONS)


# ---------------------------------------------------------------- §5 item 1: seed 42 against round 1

@contextlib.contextmanager
def _quiet():
    """Discard stdout and stderr of round-1 / step-1 code: round 1's load_bundle and run_sweep.setup print seed-42
    margin values, which rule §10 keeps out of every console and log. Exceptions propagate unchanged."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


class BundleMismatch(AssertionError):
    """compare_with_round1 found a difference; ``result`` holds every check's pass or fail."""

    def __init__(self, msg, result):
        super().__init__(msg)
        self.result = result


def compare_with_round1(bundle, r1=None) -> dict:
    """§5 item 1 on seed 42: the bundle equals round 1's common.load_bundle() exactly (episodes, parity, anchor
    paintings, pair index, cosine, B and B'(A0) scores and per-anchor arrays, T_N1u, the A0 posteriors, the grouping
    scores via common.grouping_stack on round 1's bundle, the 18 features via rb_eval.seed42_features on round 1's
    bundle, the affect head); per_anchor_seed42.npz passes load_external's assertions; D7's six values equal the
    rule's exactly and affect is the least redundant in both directions. -> {"checks", "redundancy", "n_checks",
    "all_pass"}; raises BundleMismatch (an AssertionError carrying the same dict) on any failure."""
    if bundle.seed != 42 or bundle.smoke or bundle.ctx is None:
        raise ValueError("compare_with_round1 takes the freshly built non-smoke seed-42 bundle only")
    with _quiet():                            # round 1 prints seed-42 margins; discarded (rule §10)
        if r1 is None:
            r1 = C.load_bundle(smoke=False)   # asserts round 1's rule, inputs and the step-1 arrays itself
        st1 = C.grouping_stack(r1.post, r1.ctx.pooled, A0)
        F1, _ = rbe.seed42_features(r1, A0)
    checks = {}

    def same(name, x, y, nan=False):
        x, y = np.asarray(x), np.asarray(y)
        checks[name] = bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=nan))

    ep, ep1 = bundle.ctx.pooled, r1.ctx.pooled
    for f in fields(ep):
        a, b = getattr(ep, f.name), getattr(ep1, f.name)
        if isinstance(a, np.ndarray):
            same(f"episodes.{f.name}", a, b)
        else:
            checks[f"episodes.{f.name}"] = bool(a == b)
    checks["episodes.sha256"] = dict(bundle.episodes_sha256) == dict(r1.ctx.shas)
    same("anchor", bundle.anchor, ep1.anchor)
    same("parity", bundle.parity, r1.parity)
    same("anchor_paintings", bundle.cl, r1.cl)
    same("pair_index", bundle.pair_index, r1.pair_index)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            same(f"cos.{c}.{d}", bundle.cos[c][d], r1.ctx.cos[c][d])
            same(f"B.{c}.{d}", bundle.B[c][d], r1.B[c][d])
            same(f"Bprime_A0.{c}.{d}", bundle.Bp[c][d], r1.Bp["A0"][c][d])
            same(f"T_N1u.{c}.{d}", bundle.t_n1u[c][d], r1.t_n1u[c][d])
    for m in METRICS:
        same(f"pB.{m}", bundle.pB[m], r1.pB[m])
        same(f"pBprime_A0.{m}", bundle.pBp[m], r1.pBp["A0"][m])
    for h in A0:
        for side in ("img", "txt"):
            same(f"post.{h}.{side}", bundle.post[h][side], r1.post[h][side], nan=True)
    for d in DIRECTIONS:
        same(f"stack.{d}", bundle.stack[d], st1[d])
    for c in CONDITIONS:
        same(f"features.{c}", bundle.F[c], F1[c])
    checks["affect_head"] = json.loads(json.dumps(bundle.affect_head)) == json.loads(json.dumps(r1.affect_head))
    checks["round1_load_bundle_own_checks"] = all(v is True for v in r1.checks.values())
    try:
        load_external(bundle)
        checks["external_per_anchor_seed42_aligned"] = True
    except AssertionError:
        checks["external_per_anchor_seed42_aligned"] = False
    red = redundancy(bundle)
    for h in A0:
        for d in DIRECTIONS:
            checks[f"redundancy.{h}.{d}_equals_rule"] = bool(red[h][d] == R3.REDUNDANCY_42[h][d])
    checks["redundancy.affect_least_redundant_both_directions"] = affect_least_redundant(red)
    failed = [k for k, v in checks.items() if v is not True]
    result = {"checks": checks, "redundancy": red, "n_checks": len(checks), "all_pass": not failed}
    if failed:
        raise BundleMismatch(f"seed-42 bundle differs from round 1's in {len(failed)} of {len(checks)} comparisons: "
                             f"{failed}", result)
    return result


# ---------------------------------------------------------------- the per-seed cache (§6.4)

def _arrays_of(b) -> dict:
    out = {k: np.asarray(getattr(b, k)) for k in ("cl", "parity", "pair_index", "anchor")}
    for k in ("cos", "B", "Bp", "t_n1u"):
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[f"{k}__{c}__{d}"] = np.asarray(getattr(b, k)[c][d])
    for k in ("pB", "pBp"):
        for m in METRICS:
            out[f"{k}__{m}"] = np.asarray(getattr(b, k)[m])
    for d in DIRECTIONS:
        out[f"stack__{d}"] = np.asarray(b.stack[d])
    for c in CONDITIONS:
        out[f"F__{c}"] = np.asarray(b.F[c])
    return out


def save_bundle(bundle, path) -> str:
    """Write the bundle's arrays (everything but ctx, the posteriors and the readers) and a JSON record to ``path``
    (.npz). A non-smoke cache is never overwritten (rule §10). Returns the file's SHA-256."""
    path = Path(path)
    validate(bundle)
    R3.refuse_existing([path], bundle.smoke)
    arrays = _arrays_of(bundle)
    meta = {"format": CACHE_FORMAT, "seed": int(bundle.seed), "smoke": bool(bundle.smoke), "n": int(bundle.n),
            "groupings": list(A0), "rule_sha256": R3.RULE_SHA, "affect_head": bundle.affect_head,
            "episodes_sha256": bundle.episodes_sha256, "input_sha256": bundle.input_sha256, "checks": bundle.checks,
            "provenance": R3.provenance(bundle.smoke)}
    meta = C.jsonable(meta)
    C.assert_finite_tree(meta)
    arrays["meta"] = np.array(json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return R3.sha_file(path)


def load_bundle_cache(path, seed=None, smoke=None, sha256=None, readers=True) -> SimpleNamespace:
    """A bundle from save_bundle's file (ctx and post are None). Refused: another rule, format or groupings; a seed or
    smoke flag other than the given ones; a SHA-256 other than ``sha256``. readers=True reloads round 1's A0
    half-readers (rb_build.load_readers("A0", False), SHA-256s asserted)."""
    path = Path(path)
    if sha256 is not None and R3.sha_file(path) != sha256:
        raise SystemExit(f"{path.name}: SHA-256 differs from the recorded one")
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][()]))
        if meta.get("format") != CACHE_FORMAT or meta.get("rule_sha256") != R3.RULE_SHA:
            raise SystemExit(f"{path.name}: written under another rule or cache format")
        if tuple(meta["groupings"]) != A0:
            raise SystemExit(f"{path.name}: other groupings {meta['groupings']}")
        if seed is not None and meta["seed"] != int(seed):
            raise SystemExit(f"{path.name}: seed {meta['seed']}, expected {seed}")
        if smoke is not None and meta["smoke"] != bool(smoke):
            raise SystemExit(f"{path.name}: smoke {meta['smoke']}, expected {smoke}")
        g = {k: z[k] for k in z.files if k != "meta"}
    b = SimpleNamespace(
        seed=int(meta["seed"]), smoke=bool(meta["smoke"]), n=int(meta["n"]), ctx=None, cl=g["cl"],
        parity=g["parity"], pair_index=g["pair_index"], anchor=g["anchor"],
        cos={c: {d: g[f"cos__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS},
        B={c: {d: g[f"B__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS},
        pB={m: g[f"pB__{m}"] for m in METRICS},
        Bp={c: {d: g[f"Bp__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS},
        pBp={m: g[f"pBp__{m}"] for m in METRICS},
        t_n1u={c: {d: g[f"t_n1u__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS},
        post=None, stack={d: g[f"stack__{d}"] for d in DIRECTIONS}, F={c: g[f"F__{c}"] for c in CONDITIONS},
        affect_head=meta["affect_head"], readers=None, readers_record=None,
        episodes_sha256=meta["episodes_sha256"], input_sha256=meta["input_sha256"], checks=meta["checks"],
        from_cache=True, cache_meta=meta)
    validate(b)
    if readers:
        R3.assert_inputs(READER_INPUTS)
        pk, rec, rz = rb.load_readers("A0", False)
        if rz is not None:
            rz.close()
        b.readers, b.readers_record = pk, {"rule_sha256": rec["rule_sha256"], "smoke": rec["smoke"]}
    return b


def bundles_equal(a, b) -> dict:
    """{array name: exactly equal (value, shape and dtype)} over the cached arrays of two bundles."""
    x, y = _arrays_of(a), _arrays_of(b)
    out = {k: bool(k in y and x[k].shape == y[k].shape and x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k]))
           for k in x}
    out["seed_smoke_n"] = (a.seed, a.smoke, a.n) == (b.seed, b.smoke, b.n)
    return out


# ---------------------------------------------------------------- dry check on seed 42 (prints only pass or fail)

def dry_check_42() -> bool:
    R3.assert_rule()
    out_dir = R3.res_dir(True)
    t0 = time.time()
    bundle = build_bundle(42, False)
    print("CHECK build_bundle_seed42 PASS", flush=True)
    try:
        res = compare_with_round1(bundle)
    except BundleMismatch as e:
        res = e.result
    except BaseException as e:                # round 1's own stop (SystemExit, AssertionError, ...): marker, re-raise
        print(f"CHECK compare_with_round1 FAIL ({type(e).__name__})", flush=True)
        print("R3_BUNDLE_DRY42 FAIL", flush=True)
        raise
    for k, v in res["checks"].items():
        print(f"CHECK {k} {'PASS' if v is True else 'FAIL'}", flush=True)
    cache = out_dir / "r3_bundle_dry42_cache.npz"
    cache.unlink(missing_ok=True)
    sha = save_bundle(bundle, cache)
    eq = bundles_equal(bundle, load_bundle_cache(cache, seed=42, smoke=False, sha256=sha))
    cache.unlink()
    cache_ok = all(eq.values())
    print(f"CHECK cache_round_trip_exact {'PASS' if cache_ok else 'FAIL'}", flush=True)
    ok = bool(res["all_pass"] and cache_ok)
    R3.write_json_once(out_dir / "r3_bundle_dry42.json",
                       {"what": "r3_bundle dry check on seed 42 (rule §5 item 1); smoke output, not a result",
                        "checks": res["checks"], "redundancy_D7": res["redundancy"], "cache_round_trip": eq,
                        "n_checks": res["n_checks"] + 1, "all_pass": ok, "seconds": time.time() - t0}, smoke=True)
    print(f"R3_BUNDLE_DRY42 {'PASS' if ok else 'FAIL'} ({res['n_checks'] + 1} checks)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-check-42", action="store_true", help="§5 item 1 on seed 42; writes only to results/smoke/")
    args = ap.parse_args()
    if not args.dry_check_42:
        ap.error("nothing to do (use --dry-check-42)")
    sys.exit(0 if dry_check_42() else 1)


if __name__ == "__main__":
    main()
