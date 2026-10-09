"""Round 6 bundle (DECISION_RULE.md of this folder: section 5 items 4 and 5, section 6 items 1 and 4, section 11;
contracts section 4; ticket 05): every scorer's terms for one seed's episodes, on a RowContext (r6_context), held or
selection. No pick is made here (ticket 06 assembles the frozen picks).

  fit_pm(data, scorer_train)          the PCA basis and pair scaler of run_baselines.py, refit on the same 60,000
                                      scorer-train rows (rule section 11; fit_rows_sha256 asserted against
                                      baselines_seed42.json, the unsorted draw's convention)
  load_readers()                      round 1's two A0 half-readers (rb_build.load_readers("A0", False), never smoke)
  build_bundle_r6(ctx, readers, pm)   round 3's bundle call sequence (r3_bundle.build_bundle, cpu_path section 3 items 1
                                      to 9) on the context, with the changes below

The call sequence, against round 3's build_bundle:
  1. scorer_train: the split's (load_split asserted it equal to prepare.npz's), not a second artelingo_splits.
  2. T_N1u = centered_term(run_checks.model_inputs(ctx, "A3", scorer_train, False)[0], pooled, uniform=True): the A3
     checkpoint (SHA asserted against picked.json by model_inputs), encoded through ctx.encode on exactly ctx.rows.
  3. B's T_6u = uniform_probe_scores over run_n6.PARTS of the refit E2 heads {"affect": affect-km, "image", "caption"}
     (in round 3: run_n6.n6_terms(load_posteriors(...))[2], the same function on the stored posteriors). Never
     run_n6.load_posteriors.
  4, 5. post = A0's posteriors (D1 affect, 41 classes; image; caption), all from the refit heads.
  6. B0's T_6u = uniform_probe_scores(post, pooled, A0).
     B1's T_6u = uniform_probe_scores(post + csd, pooled, A1), A1 = (affect, image, caption, csd) in round 4's order,
     csd from the refit csd head (rule section 6 item 1), never r4_bundle.extend_a1's stored-posterior load. No round 4
     or round 5 code is copied (B1 is uniform_probe_scores plus ticket 06's nested assembly).
  7. stack = common.grouping_stack(post, pooled, A0).
  8. F = rb_eval.seed42_features(SimpleNamespace(ctx=ctx, post=post), A0) (its own exactness checks).
  9. readers: given (load_readers()), checked for A0's feature layout.
  B and B0 themselves (the cross-fitted or frozen scores) are not built here.
  RCA and the nine PM terms: run_baselines.py's term calls (its order), with base = EvalInputs(ctx.img, ctx.txt), on
  this seed's pooled episodes only (hazard 6: xing and wang depend on the episode count through Adam's eps), torch
  at 8 threads as run_baselines set it.

Never called: r3_bundle.build_bundle, r3_bundle._check_seed, run_baselines.py, run_n6.load_posteriors (hazards 14, 17).
Round 3's final review N11: cl == groups[anchor] asserted.

Guards carry a `# guard:<name>` marker; test_r6_context.py deletes each on a copy and shows that its scenario then
goes through.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_context as X  # noqa: E402
import r6_heads as H  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS  # noqa: E402
from src.eval.aspect_quick_checks import _require_condition_free, centered_term, uniform_probe_scores  # noqa: E402
from src.eval.aspect_scorers import EvalInputs  # noqa: E402
from src.eval.pair_metric_baselines import (bilinear_agreement_term, diag_agreement_term, fit_pair_scaler,  # noqa: E402
                                            fit_pca_basis, kissme_term, pair_probe_term, rca_term,
                                            tip_adapter_term, value_prototype_term, wang_term, xing_term)

C, rb, rbe, rf = R.C, R.rb, R.rbe, R.rf
rc, n6 = R.run_checks, R.run_n6

A0 = tuple(R.R3.A0)                                # ("affect", "image", "caption"); affect = D1's (41 classes)
A1 = A0 + ("csd",)                                 # round 4's D4 order
E2_PARTS = tuple(n6.PARTS)                         # ("affect", "image", "caption"); this affect is affect-km (E2)
E2_FROM = {"affect": "affect_km", "image": "image", "caption": "caption"}   # run_n6 part -> r6 head
PM_NAMES = ("diag", "diag_relu", "bilinear", "kissme", "xing", "wang", "probe", "tip", "value_prototype")  # rule 2
TERM_ORDER = ("diag", "diag_relu", "bilinear", "kissme", "rca", "xing", "wang", "probe", "tip", "value_prototype")
PM_TERM_NAMES = ("rca",) + PM_NAMES                # pm_terms' keys
K_CAND = 13
N_FEATURES = 6 * len(A0)
FIT_ROWS = 60_000
FIT_SEED = 0
TORCH_THREADS = 8
BASELINES42_REL = "20261030_aspect_baselines/results/baselines_seed42.json"
A3_REL = "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt"
READER_RELS = ("20261117_reader_fix_csd/results/rb_reader_A0.pkl", "20261117_reader_fix_csd/results/rb_reader_A0.json")
FIELDS = ("n", "cl", "parity", "pair_index", "anchor", "cos", "t_n1u", "t6u_B", "t6u_B0", "t6u_B1", "post", "stack",
          "F", "pm_terms", "episodes_sha256", "module_sha256")          # contracts section 4

if not (A0 == ("affect", "image", "caption") and E2_PARTS == ("affect", "image", "caption")
        and set(PM_TERM_NAMES) == set(TERM_ORDER) and len(TERM_ORDER) == 10):
    raise ImportError("A0, run_n6.PARTS or the PM names differ from the ones this module was written for")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def log(msg):
    rc.log(f"[r6_bundle] {msg}")


# ---------------------------------------------------------------- PM fits (rule section 11)

def fit_rows(scorer_train) -> np.ndarray:
    """run_baselines.py's draw: default_rng(0).choice(scorer_train, 60000, replace=False) (unsorted)."""
    return np.random.default_rng(FIT_SEED).choice(np.asarray(scorer_train), FIT_ROWS, replace=False)


def fit_pm(data, scorer_train) -> SimpleNamespace:
    """run_baselines.py:126-132: the PCA basis and pair scaler on the unit-normalised raw features of the 60,000-row
    draw. The draw's SHA-256 (int64 bytes of the unsorted draw) must equal baselines_seed42.json's fit_rows_sha256.
    -> SimpleNamespace(basis, scaler, fit_rows_sha256)."""
    H.require_threads()
    R.assert_input(BASELINES42_REL)
    rec = json.loads(Path(R.INPUT_PATHS[BASELINES42_REL]).read_text())
    rows = fit_rows(scorer_train)
    sha = R.sha256_bytes(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    _require(int(rec["fit_rows"]) == FIT_ROWS and sha == rec["fit_rows_sha256"],
             f"PM fit rows: SHA-256 {sha} differs from baselines_seed42.json's "
             f"{rec['fit_rows_sha256']} (unsorted draw)")  # guard:fit_rows_sha
    fi, ft = (np.asarray(x[rows], np.float32) for x in (data.img_features, data.txt_features))
    _require(bool(np.isfinite(fi).all() and np.isfinite(ft).all()),
             "PM fit: non-finite features on the fit rows; the fits use raw scorer-train features, never a masked "
             "array (hazard 18)")  # guard:pm_raw
    fi, ft = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (fi, ft))   # unit rows, like EvalInputs
    return SimpleNamespace(basis=fit_pca_basis(fi, ft), scaler=fit_pair_scaler(fi, ft), fit_rows_sha256=sha)


# ---------------------------------------------------------------- readers

def load_readers() -> dict:
    """Round 1's A0 half-readers: the pickle dict of rb_build.load_readers("A0", False) (file SHA-256s asserted)."""
    for rel in READER_RELS:
        R.assert_input(rel)
    pk, _rec, z = rb.load_readers("A0", False)
    if z is not None:
        z.close()
    check_readers(pk)
    return pk


def check_readers(readers):
    _require(isinstance(readers, dict) and readers.get("config") == "A0"
             and list(readers.get("feature_names", ())) == rf.feature_names(A0),
             "readers: not round 1's A0 half-readers over A0's 18 features")  # guard:readers


# ---------------------------------------------------------------- the context's checks

def check_context(ctx):
    """The context fields the bundle relies on: n, pair index and parity as EvalContext's; every episode member a row
    of the context (rule section 5 item 5); cl = groups[anchor] (round 3's final review N11)."""
    n = int(ctx.n)
    ep = ctx.pooled
    _require(n == len(R.PAIRS) * int(ctx.n_per_pair) == len(ep.anchor) and ep is ctx.eps.pooled
             and X.ADMITTED.get((ctx.mode, int(ctx.seed))) == int(ctx.n_per_pair),
             f"{n} episodes of {ctx.mode} seed {ctx.seed}: one seed's pooled episodes (3 x its admitted "
             f"episodes per pair) are expected")  # guard:per_seed
    _require(np.array_equal(ctx.pair_index, np.repeat(np.arange(len(R.PAIRS)), ctx.n_per_pair))
             and np.array_equal(ctx.parity, np.arange(n) % 2),
             "pair_index or parity is not EvalContext's")
    _require(np.array_equal(ctx.anchor, ep.anchor), "ctx.anchor is not the pooled episodes' anchor")
    in_rows = np.zeros(R.N_ROWS, dtype=bool)
    in_rows[np.asarray(ctx.rows)] = True
    _require(np.array_equal(in_rows, ctx.in_rows) and np.array_equal(ctx.selection, ctx.rows),
             "ctx.in_rows or ctx.selection is not ctx.rows")
    _require(bool(in_rows[np.asarray(ep.rows())].all()),
             f"an episode member lies outside the {ctx.mode} rows")  # guard:bundle_rows
    cl = np.asarray(ctx.anchor_group)
    _require(cl.shape == (n,) and np.array_equal(cl, np.asarray(ctx.split.groups)[ep.anchor]),
             "cl differs from groups[anchor] (round 3 final review N11)")  # guard:cl_groups
    return cl


# ---------------------------------------------------------------- the terms

def pm_term_scores(ctx, pm) -> dict:
    """{name: term {c: {d: (n, 13)}}} for rca and the nine PM scorers: run_baselines.py's calls (:157-172) on this
    seed's pooled episodes, base = EvalInputs(ctx.img, ctx.txt). Computed in run_baselines' order."""
    check_context(ctx)
    torch.set_num_threads(TORCH_THREADS)                     # as run_baselines.py at import
    _require(torch.get_num_threads() == TORCH_THREADS, "torch must run with 8 threads")
    base = EvalInputs(ctx.img, ctx.txt)
    ep = ctx.pooled
    calls = {
        "diag": lambda: diag_agreement_term(base, ep, relu=False),
        "diag_relu": lambda: diag_agreement_term(base, ep, relu=True),
        "bilinear": lambda: bilinear_agreement_term(base, ep, pm.basis),
        "kissme": lambda: kissme_term(base, ep, pm.basis),
        "rca": lambda: rca_term(base, ep, pm.basis),
        "xing": lambda: xing_term(base, ep, pm.basis),
        "wang": lambda: wang_term(base, ep),
        "probe": lambda: pair_probe_term(base, ep, pm.scaler),
        "tip": lambda: tip_adapter_term(base, ep),
        "value_prototype": lambda: value_prototype_term(base, ep),
    }
    _require(tuple(calls) == TERM_ORDER, "term calls out of run_baselines' order")
    out = {}
    for name in TERM_ORDER:
        t = time.time()
        out[name] = calls[name]()
        check_scores(out[name], ctx.n, f"pm term {name}")
        log(f"{name} term [{time.time() - t:.0f}s]")
    return {name: out[name] for name in PM_TERM_NAMES}


def check_scores(s, n, what, dtype=None):
    """{c: {d: (n, 13)}} with finite entries (float32 when ``dtype`` is given)."""
    _require(set(s) == set(CONDITIONS) and all(set(s[c]) == set(DIRECTIONS) for c in CONDITIONS),
             f"{what}: keys must be conditions x directions")
    for c in CONDITIONS:
        for d in DIRECTIONS:
            x = np.asarray(s[c][d])
            _require(x.shape == (n, K_CAND) and (dtype is None or x.dtype == dtype) and bool(np.isfinite(x).all()),
                     f"{what}/{c}/{d}: {x.dtype} {x.shape}, must be finite "
                     f"{'' if dtype is None else np.dtype(dtype).name + ' '}({n}, {K_CAND})")  # guard:scores


def build_bundle_r6(ctx, readers, pm) -> SimpleNamespace:
    """Contracts section 4. Fields: n, cl, parity, pair_index, anchor, cos, t_n1u, t6u_B (affect-km, image, caption),
    t6u_B0 (A0), t6u_B1 (A1), post (A0 posteriors, {h: {"img", "txt"}}), stack, F, pm_terms ({rca, nine PM}: term),
    episodes_sha256, module_sha256; plus mode, seed, smoke, pooled, readers, coef_sha256, fit_rows_sha256,
    input_sha256, checks and build_s. Scores are {c: {d: (n, 13) float32}}."""
    t0 = time.time()
    cl = check_context(ctx)
    check_readers(readers)
    H.require_threads()
    shas = R.assert_inputs()
    ep, n = ctx.pooled, int(ctx.n)
    log(f"{ctx.mode} seed {ctx.seed}: {n} episodes ({ctx.n_per_pair} per pair)")

    # T_N1u (R3 rule D10): A3 codes of ctx.rows only, centred per episode
    inp = rc.model_inputs(ctx, "A3", np.asarray(ctx.split.scorer_train), False)[0]   # never the smoke checkpoint
    t_n1u = centered_term(inp, ep, uniform=True)
    del inp

    # T_6u of B (D10): the refit E2 heads, run_n6.PARTS order (affect = affect-km)
    post_e2 = {part: ctx.post[E2_FROM[part]] for part in E2_PARTS}
    t6u_B = uniform_probe_scores(post_e2, ep, E2_PARTS)
    # A0 posteriors (D2) and B0's T_6u (D11); B1's over A1 with the refit csd head (round 4's D4, rule section 6.1)
    post = {h: ctx.post[h] for h in A0}
    for h in A0:
        for side in ("img", "txt"):
            _require(bool(np.isfinite(post[h][side][ctx.rows]).all()), f"{h}/{side}: non-finite posteriors on rows")
    t6u_B0 = uniform_probe_scores(post, ep, A0)
    post_a1 = {**post, "csd": ctx.post["csd"]}
    _require(tuple(post_a1) == A1, "A1 posteriors out of A1 order")
    t6u_B1 = uniform_probe_scores(post_a1, ep, A1)
    stack = C.grouping_stack(post, ep, A0)                                                      # D4
    F, checks = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post), A0)                    # D5
    log("terms and features built")

    pm_terms = pm_term_scores(ctx, pm)
    bundle = SimpleNamespace(
        n=n, cl=cl, parity=np.asarray(ctx.parity), pair_index=np.asarray(ctx.pair_index),
        anchor=np.asarray(ep.anchor), cos=ctx.cos, t_n1u=t_n1u, t6u_B=t6u_B, t6u_B0=t6u_B0, t6u_B1=t6u_B1,
        post=post, stack=stack, F=F, pm_terms=pm_terms, episodes_sha256=dict(ctx.episode_sha),
        module_sha256=R.r6_module_shas(),
        mode=ctx.mode, seed=int(ctx.seed), smoke=bool(ctx.smoke), pooled=ep, readers=readers,
        coef_sha256=ctx.coef_sha256, fit_rows_sha256=pm.fit_rows_sha256, input_sha256=shas, checks=dict(checks),
        build_s=None)
    bundle.checks.update(validate(bundle))
    bundle.build_s = time.time() - t0
    log(f"bundle done [{bundle.build_s:.0f}s]")
    return bundle


# ---------------------------------------------------------------- invariants

def validate(b) -> dict:
    """Shapes, dtypes and identities of a bundle (round 3's validate for the fields kept): cos, T_N1u and the three
    T_6u terms finite float32 and condition-free; stack (n, 3, 13) float32; F (n, 18) float64 with Delta^b =
    -Delta^a (D3); pm_terms the ten terms, finite; per-episode fields of length n."""
    n = int(b.n)
    _require(all(hasattr(b, f) for f in FIELDS), f"bundle lacks {[f for f in FIELDS if not hasattr(b, f)]}")
    for k in ("cl", "parity", "pair_index", "anchor"):
        _require(np.asarray(getattr(b, k)).shape == (n,), f"{k}: shape must be ({n},)")
    _require(np.array_equal(np.asarray(b.parity), np.arange(n) % 2), "parity must be the episode-index parity")
    for k in ("cos", "t_n1u", "t6u_B", "t6u_B0", "t6u_B1"):
        check_scores(getattr(b, k), n, k, np.float32)
        _require_condition_free(getattr(b, k), k)                                   # ValueError
    _require(set(b.post) == set(A0), f"post must hold A0's groupings, not {sorted(b.post)}")
    _require(set(b.stack) == set(DIRECTIONS), "stack: keys must be the directions")
    for d in DIRECTIONS:
        x = np.asarray(b.stack[d])
        _require(x.shape == (n, len(A0), K_CAND) and x.dtype == np.float32 and bool(np.isfinite(x).all()),
                 f"stack/{d}: must be finite float32 of shape ({n}, {len(A0)}, {K_CAND})")
    _require(set(b.F) == set(CONDITIONS), "F: keys must be the conditions")
    for c in CONDITIONS:
        x = np.asarray(b.F[c])
        _require(x.shape == (n, N_FEATURES) and x.dtype == np.float64 and bool(np.isfinite(x).all()),
                 f"F/{c}: must be finite float64 of shape ({n}, {N_FEATURES})")
    for j in range(len(A0)):
        _require(np.array_equal(b.F["b"][:, 6 * j + 2], -b.F["a"][:, 6 * j + 2]),
                 f"F: Delta^b != -Delta^a for {A0[j]} (D3)")
    _require(tuple(b.pm_terms) == PM_TERM_NAMES, f"pm_terms must be {PM_TERM_NAMES}")
    for k, t in b.pm_terms.items():
        check_scores(t, n, f"pm term {k}")
    _require(tuple(b.episodes_sha256) == R.PAIR_NAMES, "episodes_sha256 must hold the three pairs")
    return {"shapes_dtypes_ok": True, "cos_tn1u_t6u_condition_free": True, "delta_b_equals_minus_delta_a": True}
