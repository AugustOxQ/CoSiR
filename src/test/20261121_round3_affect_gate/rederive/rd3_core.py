"""Round-3 independent re-derivation: shared core (decides nothing by itself).

Written from ../DECISION_RULE.md (SHA-256 asserted) alone. It never opens or imports the round-3 implementation
(r3_common.py, r3_bundle.py, r3_fusion.py, r3_stats.py, run_r3_*.py, r3_apply_rule.py, test_r3_*.py), round 1's
common.py / rc_core.py / rb_eval.py / rb_features.py (by name), or round 2's r2_fusion.py. Allowed imports (rule §8):
EvalContext, run_checks.model_inputs, centered_term, run_n6.load_posteriors, run_told_oracle.fit_one_head and
global_labels, rb_build.load_readers("A0", False), zscore_rows, crossfit_condition_free, uniform_probe_scores,
cluster_bootstrap (rb_build imports round 1's common at module level; no function of it is called here).

Own code here: SHA-256 checks, features, reader probabilities, picks and margins, the weighted term, grouping scores,
gates, z-score wrapper, float32 combine, integer rank counts, per-anchor metrics (R@1 strict: ties miss), comparators,
intervals, redundancy (D7) and the sensitivity projection (§6.1). Parts are copied from round 2's own re-derivation
(rederive/rd2_core.py), never imported. CPU only.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
R3DIR = HERE.parent
ROOT = R3DIR.parents[2]
T = ROOT / "src/test"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
PHASE2_MARKER = OUT / "PHASE2_AUTHORISED"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402
from src.model.aspect_rule import zscore_rows  # noqa: E402

# ---------------------------------------------------------------- the rule and its inputs (D15)
RULE = R3DIR / "DECISION_RULE.md"
RULE_SHA = "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925"   # fab5ae1, the run log
INPUT_SHA = {   # rule D1, D2, D10, D15, §2 (paths under src/test/)
    "20261030_aspect_baselines/results/episodes_seed42.npz":
        "12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986",
    "20261111_community_told_oracle/results/per_anchor_told_oracle.npz":
        "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366",
    "20261031_pseudo_partitions/results/partitions.npz":
        "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa",
    "20261108_new_method_quick_checks/results/n6_posteriors.npz":
        "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0",
    "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt":
        "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2",
    "20261111_community_told_oracle/results/told_oracle.json":
        "76d9ec896b941a518e7db6684805fe0bc329f6afe4a48e1993992fbb221d70d2",
    "20261117_reader_fix_csd/DECISION_RULE.md": "613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c",
    "20261117_reader_fix_csd/rb_build.py": "63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b",
    "20261117_reader_fix_csd/results/rb_reader_A0.pkl":
        "6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c",
    "20261117_reader_fix_csd/results/rb_reader_A0.json":
        "cd90da922967cf1b62e437d3927c97cd9cfdae2c139ee392744538d3e3e5c6a8",
    "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz":
        "628b21aeaf6d306f82bd9abb68e6c257348535c3aff0fad0a2d065fc48302981",
    "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json":
        "c6e2f83b73c47b9c054e6381a5d020a05618728cb820861b89f0b80de5a8a16e",
    "20261117_reader_fix_csd/results/rc_tau.json": "e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf",
    "20261120_r1_levers_brainstorm/results/bs_04_readers.json":
        "42bf6f204598c011bfada46e5eabf42dc7759fc6f6750721527d16239141691e",
    "20261120_r1_levers_brainstorm/results/bs_05_aff.json":
        "a96719ba505b72883bfa2eeaca66ab872da128314654ba8f74580998b2a72b10",
    "20261030_aspect_baselines/results/per_anchor_seed42.npz":
        "a4818ba0fa5f7249355afe2d2483404dcd34d22cae26984b76be787bb6e9e59d",
}

COND = ("a", "b")
DIRS = ("i2t", "t2i")
METRICS = ("r1", "gain", "other", "swap", "strict")
A0 = ("affect", "image", "caption")
E2_PARTS = ("affect", "image", "caption")      # E2's k-means groupings (affect = affect-km), n6 posteriors
PAIR_NAMES = ("emotion__style", "emotion__genre", "style__genre")
TOLD_IDX = {"emotion__style": (0, 1), "emotion__genre": (0, 1), "style__genre": (1, 1)}   # D14 (a, b) -> A0 index
FEATS = ("S", "C", "Delta", "sd_support", "sd_contrast", "argmax_match")

# D6 (written in the rule; rc_tau.json asserted equal)
RULE_TAUS = (3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211)
# D8
NU = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NA = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CONTROL_SUMS = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 3, 4, 4.25, 4.5, 5, 6, 8, 8.25, 8.5, 9, 10, 12, 16,
                16.25, 16.5, 17, 18, 20, 24, 32)
N_TAU = 4
N_CELLS = N_TAU * len(NU) * len(NA)     # 224
if sorted({u + a for u in NU for a in NA}) != [float(x) for x in CONTROL_SUMS]:
    raise AssertionError("own control sums differ from the rule's list (D8 item 5)")

FORBIDDEN_SEEDS = (49, 50, 51, 9001, 9002, 9003)


def cell_id(t, u, a):
    """D8 item 4: (t*7 + u)*8 + a, zero-based."""
    return (t * len(NU) + u) * len(NA) + a


def cell_decode(i):
    return i // (len(NU) * len(NA)), (i // len(NA)) % len(NU), i % len(NA)


def cell_desc(i, taus):
    t, u, a = cell_decode(int(i))
    return {"cell": int(i), "tau_index": int(t), "tau": float(taus[t]), "lambda_u": NU[u], "lambda_a": NA[a]}


# ---------------------------------------------------------------- utilities

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def assert_rule():
    got = sha_file(RULE)
    if got != RULE_SHA:
        raise SystemExit(f"DECISION_RULE.md SHA-256 {got} is not the committed {RULE_SHA}")
    return got


def assert_inputs(names):
    for n in names:
        got = sha_file(T / n)
        if got != INPUT_SHA[n]:
            raise SystemExit(f"{n}: SHA-256 {got} differs from the rule ({INPUT_SHA[n]})")
    return {n: INPUT_SHA[n] for n in names}


def guard_seed(seed):
    if int(seed) in FORBIDDEN_SEEDS and not PHASE2_MARKER.exists():
        raise SystemExit(f"seed {seed} may not be touched in phase 1")


def now_ams():
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M")


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


def import_path(name, path):
    """Import a module from an explicit file under a fixed module name (asserting no other copy is loaded)."""
    path = Path(path).resolve()
    if name in sys.modules:
        f = Path(sys.modules[name].__file__).resolve()
        if f != path:
            raise AssertionError(f"module {name} already loaded from {f}, wanted {path}")
        return sys.modules[name]
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def point_ci(values, clusters):
    """Percentage-point mean and 95% painting-bootstrap percentile interval (rule §2: 5,000, seed 42, chunk 250)."""
    r = cluster_bootstrap(np.asarray(values, dtype=np.float64), clusters, n_boot=5000, seed=42, chunk=250)
    return {"point": 100 * r["point"], "ci95": [100 * r["ci95"][0], 100 * r["ci95"][1]]}


# ---------------------------------------------------------------- features, grouping scores (D3, D4, D5)

def features_one(post, parts, si, st, ci, ct):
    """(E, 6H) float64, per grouping in configuration order: S, C, Delta (float32 agreements and means, as
    aspect_deltas computes them, cast to float64), ddof-1 sd of the 4 support / 4 contrast agreements (float64 of the
    float32 agreements), share of support pairs whose image and caption arg-max groups coincide."""
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


def features_both(post, parts, ep):
    """Condition a: supports = pairs_a, contrasts = pairs_b; condition b swapped."""
    return {"a": features_one(post, parts, ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt),
            "b": features_one(post, parts, ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt)}


def feature_names(parts):
    return [f"{h}__{f}" for h in parts for f in FEATS]


def grouping_scores(post, ep, parts):
    """D4: {dir: (E, H, 13) float32}, s_h = p_h(query) . p_h(candidate), each from its own modality's head."""
    return {"i2t": np.stack([np.einsum("nc,nkc->nk", post[h]["img"][ep.anchor], post[h]["txt"][ep.candidates])
                             for h in parts], axis=1),
            "t2i": np.stack([np.einsum("nc,nkc->nk", post[h]["txt"][ep.anchor], post[h]["img"][ep.candidates])
                             for h in parts], axis=1)}


# ---------------------------------------------------------------- reader (D5)

def reader_probs(pk, F):
    """P^c = mean over the two half-readers of predict_proba(scaler.transform(x)) (float64); also the halves."""
    if tuple(pk["groupings"]) != A0 or list(pk["feature_names"]) != feature_names(A0):
        raise AssertionError("reader groupings or feature layout differ from A0's 18 features")
    halves, P = {}, {}
    for c in COND:
        hs = []
        for h in pk["halves"]:
            p = h["model"].predict_proba(h["scaler"].transform(F[c]))
            if not np.array_equal(h["model"].classes_, np.arange(p.shape[1])):
                raise AssertionError("half-reader classes are not 0..H-1")
            hs.append(np.asarray(p, np.float64))
        halves[c] = hs
        P[c] = (hs[0] + hs[1]) / 2.0
        if not np.allclose(P[c].sum(1), 1.0, rtol=0, atol=1e-12):
            raise AssertionError("reader probabilities do not sum to 1")
    return P, halves


def picks_margins(P):
    """pi = arg max (ties to the first grouping, numpy.argmax); m = largest minus second-largest (float64)."""
    P = np.asarray(P, np.float64)
    srt = np.sort(P, axis=1)
    return np.argmax(P, axis=1).astype(np.int64), srt[:, -1] - srt[:, -2]


def thresholds(m_a, m_b):
    allm = np.concatenate([np.asarray(m_a, np.float64), np.asarray(m_b, np.float64)])
    return [float(v) for v in np.percentile(allm, [0, 25, 50, 75])]


def weighted_term(stack, P):
    """T^c = sum_h P^c(h) s_h (float64 einsum of the float32 s_h, cast to float32)."""
    return {c: {d: np.einsum("nh,nhk->nk", np.asarray(P[c], np.float64), stack[d].astype(np.float64))
                .astype(np.float32) for d in DIRS} for c in COND}


# ---------------------------------------------------------------- scores and metrics (own)

def z(x):
    return zscore_rows(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy()


def combine(zb, term, lu, la):
    """D8 item 3 (float32): z(B) + lu z(B) + la term, terms of weight 0 left out, in aspect_nested._combine's order."""
    s = zb
    if lu > 0:
        s = s + np.float32(lu) * zb
    if la > 0:
        s = s + np.float32(la) * term
    return np.asarray(s, np.float32)


def first_place(s, col):
    """bool (E,): column col strictly above every other candidate (ties miss); a non-finite row misses."""
    s = np.asarray(s)
    t = s[:, col:col + 1]
    less = s < t
    less[:, col] = True
    return less.all(axis=1) & np.isfinite(s).all(axis=1)


def counts(scores):
    """Per-episode integers over the 4 rankings: hits, other-aspect wins, swap (0..2), strict (0..2)."""
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


def metrics(scores):
    return metrics_from_counts(*counts(scores))


def either(p):
    return np.asarray(p["r1"], np.float64) + np.asarray(p["other"], np.float64)


def mean_pp(p, m="r1"):
    return 100 * float(np.mean(np.asarray(p[m], np.float64)))


# ---------------------------------------------------------------- comparators and evaluation (D12, D13)

def bar_comparator(pBp, pc, pB, mask=None):
    """D12: largest mean R@1 among B'(A0), counterpart, B over the episodes considered; ties to the earliest."""
    order = (("B_prime", pBp), ("counterpart", pc), ("B", pB))
    sel = (lambda x: x) if mask is None else (lambda x: np.asarray(x)[mask])
    means = [float(np.mean(sel(np.asarray(p["r1"], np.float64)))) for _, p in order]
    best = 0
    for i in (1, 2):
        if means[i] > means[best]:
            best = i
    return order[best][0], order[best][1], {n: 100 * m for (n, _), m in zip(order, means)}


def evaluate(pn, pc, pB, pBp, cl, pair_index):
    comp_name, comp, means = bar_comparator(pBp, pc, pB)
    if not np.all(np.asarray(pc["gain"]) == 0):
        raise AssertionError("counterpart gain is not 0 on every episode")
    bar_v = np.asarray(pn["r1"], np.float64) - np.asarray(comp["r1"], np.float64)
    gain_v = np.asarray(pn["gain"], np.float64) - np.asarray(pc["gain"], np.float64)
    marg_v = np.asarray(pn["r1"], np.float64) - np.asarray(pc["r1"], np.float64)
    bar, gain, marg = point_ci(bar_v, cl), point_ci(gain_v, cl), point_ci(marg_v, cl)
    eith = point_ci(either(pn) - either(pc), cl)
    clauses = {"c1_bar_point_ge_0.5": bool(bar["point"] >= 0.5), "c2_bar_lower_gt_0": bool(bar["ci95"][0] > 0),
               "c3_gain_lower_gt_0": bool(gain["ci95"][0] > 0)}
    clauses["clears_bar"] = all(clauses.values())
    per_pair = {}
    for i, p in enumerate(PAIR_NAMES):
        m = np.asarray(pair_index) == i
        per_pair[p] = {"bar_margin": point_ci(bar_v[m], cl[m]), "margin": point_ci(marg_v[m], cl[m]),
                       "gain_statistic": point_ci(gain_v[m], cl[m])}
    out = {"r1_means": {"fused": mean_pp(pn), **means}, "bar_comparator": comp_name, "bar_margin": bar,
           "margin_vs_counterpart": marg, "gain_statistic": gain, "either_change_vs_counterpart": eith,
           "D13_clauses": clauses, "per_pair": per_pair}
    return out, bar_v, gain_v


# ---------------------------------------------------------------- D7 redundancy

def redundancy(stack, B, parts):
    """Mean over the ranking rows of one direction (one row per episode) of the Pearson correlation over the 13
    candidates between z(s_h) and z(B) (zscore_rows in float32, then float64); rows with a zero denominator left out."""
    out, kept = {}, {}
    for d in DIRS:
        zb = z(B["a"][d]).astype(np.float64)
        yc = zb - zb.mean(axis=1, keepdims=True)
        for j, h in enumerate(parts):
            x = z(stack[d][:, j, :]).astype(np.float64)
            xc = x - x.mean(axis=1, keepdims=True)
            num = (xc * yc).sum(axis=1)
            den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
            ok = den > 0
            out.setdefault(h, {})[d] = float(np.mean(num[ok] / den[ok]))
            kept.setdefault(h, {})[d] = int(ok.sum())
    return out, kept


# ---------------------------------------------------------------- §6.1 sensitivity projection

def sensitivity(diff_pp, groups):
    """One-way random-effects split of a per-episode difference (percentage points) by anchor painting; projected
    pooled SE over three seeds, half-width 1.96 SE, detectable margin x = 2.80 SE."""
    v = np.asarray(diff_pp, np.float64)
    _, idx = np.unique(np.asarray(groups), return_inverse=True)
    P = int(idx.max()) + 1
    n = len(v)
    m = np.bincount(idx, minlength=P).astype(np.float64)
    sums = np.bincount(idx, weights=v, minlength=P)
    means = sums / m
    grand = v.mean()
    ss_within = float(((v - means[idx]) ** 2).sum())
    ss_between = float((m * (means - grand) ** 2).sum())
    ms_within = ss_within / (n - P)
    ms_between = ss_between / (P - 1)
    sum_m2 = float((m ** 2).sum())
    n0 = (n - sum_m2 / n) / (P - 1)
    sig_e2 = ms_within
    sig_a2 = max(0.0, (ms_between - sig_e2) / n0)
    se2 = (sig_a2 * (9 * sum_m2 - 6 * n) + sig_e2 * 3 * n) / (3 * n) ** 2
    se = float(np.sqrt(se2))
    return {"n": n, "P": P, "sum_m2": sum_m2, "n0": n0, "ms_within": ms_within, "ms_between": ms_between,
            "sigma_eps2": sig_e2, "sigma_a2": sig_a2, "SE": se, "half_width": 1.96 * se, "x": 2.80 * se,
            "point_seed42": float(grand)}
