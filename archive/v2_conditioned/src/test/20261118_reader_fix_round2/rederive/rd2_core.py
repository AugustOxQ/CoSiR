"""Round-2 independent re-derivation: shared core (decides nothing by itself).

Written from DECISION_RULE.md of the parent folder (SHA-256 368bec11...) alone. It never reads or imports the round-2
implementation files (r2_fusion.py, run_r2_fusion.py, r2_apply_rule.py, r2_readers.py, run_r2_readers.py and their
tests). It imports only: r2_common.py (rule SHA, D14 SHA-256 table, round-1 module handles), round 1's
common.load_bundle (the bundle, B, B', standard heads: rule §4.1), round 1's rb_eval.seed42_features (only to check
that our own seed-42 features equal it), the stored round-1 readers (pickles, loaded as data), src.eval.aspect_metrics
.cluster_bootstrap (rule §7) and src.model.aspect_rule.zscore_rows (rule D6). Every round-2 quantity (features,
probabilities, T, margins, thresholds, gates, top-k sets, restriction, cells, integer cross-fits, counterpart, EM,
re-standardisation, impure banks, SMDs, D(k), k*, R3 training, comparators, bar margins, gain statistics) is our own
code here or in the rd2_* scripts. CPU only.

Phase guard: until the controller says the regression check has passed in the main run, only k_top = 13 cells
(cells 0 to 223) may be scored on real data (rule §4.7, §7). `allowed_ktops()` enforces it through the marker file
out/PHASE2_AUTHORISED, which only the controller's message justifies creating.
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R2DIR = HERE.parent
if str(R2DIR) not in sys.path:
    sys.path.insert(0, str(R2DIR))
import r2_common as RC  # noqa: E402  (also puts the repo root and round 1's folder on sys.path)

import torch  # noqa: E402

from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "rd2_seed42_cache.npz"
PHASE2_MARKER = OUT / "PHASE2_AUTHORISED"

COND = ("a", "b")
DIRS = ("i2t", "t2i")
METRICS = ("r1", "gain", "other", "swap", "strict")
CONFIGS = {"A0": ("affect", "image", "caption"), "A1": ("affect", "image", "caption", "csd")}
TOLD = {"A0": {"emotion": "affect", "style": "image", "genre": "image"},
        "A1": {"emotion": "affect", "style": "csd", "genre": "image"}}
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
PAIR_NAMES = tuple(f"{a}__{b}" for a, b in PAIRS)
FEATS = ("S", "C", "Delta", "sd_support", "sd_contrast", "argmax_match")

# rule §4.5 / D6, written out here and checked against the base library and the rule's list
KTOPS = (13, 5, 3, 2)
NU = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NA = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CONTROL_SUMS = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 3, 4, 4.25, 4.5, 5, 6, 8, 8.25, 8.5, 9, 10, 12, 16,
                16.25, 16.5, 17, 18, 20, 24, 32)
PCTS = (0, 25, 50, 75)
N_TAU = 4
CELLS_PER_KTOP = N_TAU * len(NU) * len(NA)      # 224

if sorted({u + a for u in NU for a in NA}) != [float(x) for x in CONTROL_SUMS]:
    raise AssertionError("own control sums differ from the rule's list")


def check_base_constants():
    from src.eval.aspect_nested import NESTED_A, NESTED_U, control_sums, nested_cells
    ok = (tuple(NESTED_U) == NU and tuple(NESTED_A) == NA and list(control_sums()) == [float(x) for x in CONTROL_SUMS]
          and nested_cells() == [(u, a) for u in NU for a in NA])
    if not ok:
        raise AssertionError("own lambda grid / control sums differ from src.eval.aspect_nested")
    return ok


def cell_id(kappa, t, u, a):
    """Rule §4.5 item 7: ((kappa*4 + t)*7 + u)*8 + a, zero-based indices."""
    return ((kappa * N_TAU + t) * len(NU) + u) * len(NA) + a


def cell_decode(i):
    a = i % len(NA)
    u = (i // len(NA)) % len(NU)
    t = (i // (len(NA) * len(NU))) % N_TAU
    kappa = i // CELLS_PER_KTOP
    return kappa, t, u, a


def cell_desc(i, taus=None):
    kappa, t, u, a = cell_decode(i)
    d = {"cell": int(i), "k_top": KTOPS[kappa], "tau_index": int(t), "lambda_u": NU[u], "lambda_a": NA[a]}
    if taus is not None:
        d["tau"] = float(taus[t])
    return d


def allowed_ktops():
    """Phase 1: k_top = 13 only (cells 0..223). Phase 2 only after the controller's message (marker file)."""
    return KTOPS if PHASE2_MARKER.exists() else (13,)


# ---------------------------------------------------------------- small utilities

def sha_arr(a, dtype=None):
    a = np.asarray(a)
    if dtype is not None:
        if a.dtype != np.dtype(dtype):
            raise AssertionError(f"array dtype {a.dtype} is not {dtype}")
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_json(path, obj):
    obj = jsonable(obj)
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=True))
    return obj


def log(msg, t0=None):
    pre = f"[{time.strftime('%H:%M:%S')}]" + (f" [{time.time() - t0:.0f}s]" if t0 is not None else "")
    print(f"{pre} {msg}", flush=True)


def point_ci(values, clusters):
    """Percent point and 95% painting-bootstrap interval: cluster_bootstrap itself, 5,000 resamples, seed 42, chunk
    250 (rule §2, §7)."""
    r = cluster_bootstrap(np.asarray(values, dtype=np.float64), clusters, n_boot=5000, seed=42, chunk=250)
    return {"point": 100 * r["point"], "ci95": [100 * r["ci95"][0], 100 * r["ci95"][1]]}


def provenance():
    return {"rule_sha256": RC.RULE_SHA, "written_amsterdam": RC.now_ams(), "git_head": RC.C.git_head()}


# ---------------------------------------------------------------- features (rule §4.1; own code)

def features_one(post, parts, si, st, ci, ct):
    """(E, 6H) float64: per grouping in configuration order S, C, Delta (float32 agreements p(i).p(t) and float32 means,
    as aspect_deltas computes them), the ddof-1 standard deviations of the four support / contrast agreements (float64
    of the float32 agreements) and the share of the 4 support pairs whose image and caption arg-max groups coincide."""
    cols = []
    for h in parts:
        pi, pt = post[h]["img"], post[h]["txt"]
        a_i, a_t = pi[si], pt[st]
        sup = np.einsum("nsc,nsc->ns", a_i, a_t)
        con = np.einsum("nsc,nsc->ns", pi[ci], pt[ct])
        s_mean, c_mean = sup.mean(axis=1), con.mean(axis=1)
        match = (a_i.argmax(axis=-1) == a_t.argmax(axis=-1)).mean(axis=1)
        cols += [s_mean.astype(np.float64), c_mean.astype(np.float64), (s_mean - c_mean).astype(np.float64),
                 sup.astype(np.float64).std(axis=1, ddof=1), con.astype(np.float64).std(axis=1, ddof=1),
                 np.asarray(match, np.float64)]
    return np.stack(cols, axis=1)


def features_both(post, parts, pai, pat, pbi, pbt):
    """{'a','b'}: condition a = supports pairs_a, contrasts pairs_b; condition b swapped."""
    return {"a": features_one(post, parts, pai, pat, pbi, pbt), "b": features_one(post, parts, pbi, pbt, pai, pat)}


def feature_names(parts):
    return [f"{h}__{f}" for h in parts for f in FEATS]


# ---------------------------------------------------------------- readers (as data)

def load_reader_pickle(config):
    import sklearn
    name = f"results/rb_reader_{config}.pkl"
    RC.assert_inputs([name, f"results/rb_reader_{config}.json"])
    with open(RC.r1_path(name), "rb") as f:
        pk = pickle.load(f)
    if tuple(pk["groupings"]) != CONFIGS[config]:
        raise AssertionError("reader groupings differ")
    if pk["sklearn_version"] != sklearn.__version__:
        raise SystemExit(f"reader pickle written with scikit-learn {pk['sklearn_version']}, running {sklearn.__version__}")
    if pk["feature_names"] != feature_names(CONFIGS[config]):
        raise AssertionError("reader feature layout differs")
    return pk


def half_probs_scaled(model, Xs):
    p = model.predict_proba(Xs)
    if not np.array_equal(model.classes_, np.arange(p.shape[1])):
        raise AssertionError("half-reader classes are not 0..H-1")
    return np.asarray(p, np.float64)


def mean_two(p0, p1):
    return (np.asarray(p0, np.float64) + np.asarray(p1, np.float64)) / 2.0


def picks_margins(P):
    """Pick = arg max (ties to the first grouping); margin = largest minus second-largest."""
    P = np.asarray(P, np.float64)
    pick = np.argmax(P, axis=1).astype(np.int64)
    srt = np.sort(P, axis=1)
    return pick, srt[:, -1] - srt[:, -2]


def thresholds(m_a, m_b):
    allm = np.concatenate([np.asarray(m_a, np.float64), np.asarray(m_b, np.float64)])
    return [float(v) for v in np.percentile(allm, PCTS)]


def weighted_term(stack, P, idx):
    """T^c = sum_h P^c(h) s_h over the configuration's groupings (float64 sum, cast to float32). stack: {dir: (E, G, K)
    float32} of all groupings; idx: their positions in it."""
    out = {}
    for c in COND:
        Pc = np.asarray(P[c], np.float64)
        out[c] = {}
        for d in DIRS:
            s = stack[d][:, idx, :].astype(np.float64)
            out[c][d] = np.einsum("nh,nhk->nk", Pc, s).astype(np.float32)
    return out


# ---------------------------------------------------------------- scores and metrics (own)

def z(x):
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy()


def combine(zb, term, lu, la):
    """float32: z(B) + lu z(B) + la term, terms of weight 0 left out (aspect_nested._combine's order)."""
    s = zb
    if lu > 0:
        s = s + np.float32(lu) * zb
    if la > 0:
        s = s + np.float32(la) * term
    return np.asarray(s, np.float32)


def first_place(s, col):
    """bool (E,): column col strictly above every other candidate; non-finite rows miss."""
    s = np.asarray(s)
    t = s[:, col:col + 1]
    less = s < t
    less[:, col] = True
    return less.all(axis=1) & np.isfinite(s).all(axis=1)


def counts(scores):
    """Integer per-episode counts over the 4 rankings: hits (target strictly first), other (the other aspect's candidate
    strictly first), swap2 (0..2), strict2 (0..2)."""
    E = len(scores["a"]["i2t"])
    hit = np.zeros(E, np.int64)
    oth = np.zeros(E, np.int64)
    sw = np.zeros(E, np.int64)
    stc = np.zeros(E, np.int64)
    for d in DIRS:
        sa, sb = np.asarray(scores["a"][d]), np.asarray(scores["b"][d])
        haa, hbb = first_place(sa, 0), first_place(sb, 1)
        hab, hba = first_place(sb, 0), first_place(sa, 1)
        hit += haa.astype(np.int64) + hbb
        oth += hab.astype(np.int64) + hba
        fin = np.isfinite(sa).all(1) & np.isfinite(sb).all(1)
        sw += ((sa[:, 0] > sa[:, 1]) & (sb[:, 1] > sb[:, 0]) & fin)
        stc += (haa & hbb)
    return hit, oth, sw, stc


def metrics_from_counts(hit, oth, sw, stc):
    return {"r1": hit / 4.0, "gain": (hit - oth) / 4.0, "other": oth / 4.0, "swap": sw / 2.0, "strict": stc / 2.0}


def topk_positions(b):
    """For one ranking row block b (E, 13) of B: pos[n, j] = position of candidate j in o = argsort(-b, stable)."""
    o = np.argsort(-np.asarray(b), axis=1, kind="stable")
    pos = np.empty_like(o)
    rows = np.arange(o.shape[0])[:, None]
    pos[rows, o] = np.arange(o.shape[1])[None, :]
    return o, pos


def restrict(S, pos, k):
    """Rule §4.5 item 6: members of K keep S; the candidate at position p >= k gets min_K S - 1 - (p - k) (float64)."""
    if k == 13:
        return S
    S64 = np.asarray(S, np.float32).astype(np.float64)
    inK = pos < k
    minK = np.where(inK, S64, np.inf).min(axis=1)
    return np.where(inK, S64, (minK[:, None] - 1.0) - (pos - k).astype(np.float64))


# ---------------------------------------------------------------- comparators and evaluation

def bar_comparator(pBp, pc, pB):
    """D10: largest mean R@1 among B', counterpart, B (full precision); ties to the earliest in that order."""
    order = (("B_prime", pBp), ("counterpart", pc), ("B", pB))
    means = [float(np.mean(np.asarray(p["r1"], np.float64))) for _, p in order]
    best = 0
    for i in (1, 2):
        if means[i] > means[best]:
            best = i
    return order[best][0], order[best][1], {n: 100 * m for (n, _), m in zip(order, means)}


def either(p):
    return np.asarray(p["r1"], np.float64) + np.asarray(p["other"], np.float64)


def diff3(pa, pb, cl):
    return {"r1": point_ci(np.asarray(pa["r1"], np.float64) - np.asarray(pb["r1"], np.float64), cl),
            "gain": point_ci(np.asarray(pa["gain"], np.float64) - np.asarray(pb["gain"], np.float64), cl),
            "either": point_ci(either(pa) - either(pb), cl)}


def evaluate(pn, pc, pB, pBp, cl, pair_index, per_pair=True):
    """D10 to D12 from per-anchor arrays."""
    comp_name, comp, means = bar_comparator(pBp, pc, pB)
    bar_v = np.asarray(pn["r1"], np.float64) - np.asarray(comp["r1"], np.float64)
    gain_v = np.asarray(pn["gain"], np.float64) - np.asarray(pc["gain"], np.float64)
    bar = point_ci(bar_v, cl)
    gain = point_ci(gain_v, cl)
    clauses = {"c1_bar_point_ge_0.5": bool(bar["point"] >= 0.5), "c2_bar_lower_gt_0": bool(bar["ci95"][0] > 0),
               "c3_gain_lower_gt_0": bool(gain["ci95"][0] > 0)}
    clauses["clears_bar"] = all(clauses.values())
    out = {"r1_means": {"fused": 100 * float(np.mean(np.asarray(pn["r1"], np.float64))), **{f"{k}": v for k, v in means.items()}},
           "bar_comparator": comp_name, "bar_margin": bar, "gain_statistic": gain, "clauses": clauses,
           "margin": diff3(pn, pc, cl), "fused_vs_B": diff3(pn, pB, cl), "fused_vs_Bprime": diff3(pn, pBp, cl),
           "counterpart_vs_B": diff3(pc, pB, cl)}
    if per_pair:
        out["per_pair"] = {}
        for i, p in enumerate(PAIR_NAMES):
            m = pair_index == i
            sub = lambda x: {k: np.asarray(v)[m] for k, v in x.items()}  # noqa: E731
            out["per_pair"][p] = {"bar_margin_r1": point_ci(bar_v[m], cl[m]),
                                  "gain_statistic": point_ci(gain_v[m], cl[m]),
                                  "margin": diff3(sub(pn), sub(pc), cl[m]),
                                  "fused_vs_B": diff3(sub(pn), sub(pB), cl[m]),
                                  "fused_vs_Bprime": diff3(sub(pn), sub(pBp), cl[m]),
                                  "counterpart_vs_B": diff3(sub(pc), sub(pB), cl[m])}
    return out, bar_v, gain_v


# ---------------------------------------------------------------- the seed-42 cache

def load_cache():
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing: run rd2_prep.py first")
    z_ = np.load(CACHE)
    c = {"parity": z_["parity"], "cl": z_["anchor_group"], "pair_index": z_["pair_index"],
         "B": {cc: {d: z_[f"B__{cc}__{d}"] for d in DIRS} for cc in COND},
         "pB": {m: z_[f"pB__{m}"] for m in METRICS},
         "pBp": {cfg: {m: z_[f"pBp_{cfg}__{m}"] for m in METRICS} for cfg in CONFIGS},
         "F": {cfg: {cc: z_[f"F_{cfg}__{cc}"] for cc in COND} for cfg in CONFIGS},
         "stack": {d: z_[f"stack__{d}"] for d in DIRS},
         "stack_groupings": tuple(str(x) for x in z_["stack_groupings"])}
    return c


def stack_idx(cache, config):
    return [cache["stack_groupings"].index(h) for h in CONFIGS[config]]
