"""Throwaway spike: do support-set baselines on raw CLIP features match SE on the same selection episodes?

Run from the repo root:  python src/test/20261022_support_baseline_spike/run_spike.py
Only selection rows are read (all other rows NaN). Nothing here is part of the pipeline.
"""
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
_p = ROOT / "src/test/20261018_affect_factor_learning/run_affect.py"
_s = importlib.util.spec_from_file_location("run_affect", _p)
ra = importlib.util.module_from_spec(_s)
sys.modules["run_affect"] = ra
_s.loader.exec_module(ra)
grid, sel = ra.grid, ra.sel

from src.eval.condition_eval import paired_bootstrap  # noqa: E402
from src.eval.label_episodes import label_episodes_sha256, tie_aware_rank  # noqa: E402
from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes  # noqa: E402

def KEY(*a):
    return "__".join(str(x) for x in a)


LABELS, DIRS = ("emotion", "art_style"), ("i2t", "t2i")
BETA = 0.3
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, float("inf")]
BETAS_RAW = [0.0, 0.03, 0.1, 0.3, 1.0]
OUT = HERE / "results"
T0 = perf_counter()


def log(*a):
    print(f"[{perf_counter() - T0:7.1f}s]", *a, flush=True)


def tt(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


# ------------------------------------------------------------------ setup
data = ra.load_artelingo()
cache, prep, meta, _ = grid.load_grid()
prep_record = json.loads((ra.CACHE / "affect_prepare.json").read_text())
sl = cache["selection"]
in_sel = grid._row_mask(len(cache["groups"]), sl)
episodes, _, eps_meta = ra.selection_episodes(data, cache)
stored = json.loads((ra.RESULTS / "selection_results.json").read_text())
refs = dict(np.load(ra.RESULTS / "selection_ranks.npz"))
img_raw, txt_raw = (sel.masked(x, sl) for x in (data.img_features, data.txt_features))
for lab in LABELS:
    e = episodes[lab]
    rows = np.column_stack([e.anchor, e.positive, e.supports, e.contrasts, e.distractors])
    assert in_sel[rows].all(), "episode row outside selection"
    assert np.isfinite(img_raw[rows]).all() and np.isfinite(txt_raw[rows]).all()
assert np.isnan(img_raw[~in_sel]).all() and np.isnan(txt_raw[~in_sel]).all()
log("row scope asserted: all episode rows are selection rows; all other feature rows NaN")

checks = {}
# check 1: SHA
for lab in LABELS:
    mine, theirs = label_episodes_sha256(episodes[lab]), stored["episodes"][lab]["sha256"]
    assert mine == theirs, f"SHA mismatch {lab}"
    checks[f"sha_{lab}"] = mine
log("CHECK 1 passed: episode SHA-256 equals stored for both labels", checks)

SE_ic, SE_tc, _ = ra.model_codes("SE", data, cache, prep_record)
C0_ic, C0_tc, _ = ra.model_codes("C0", data, cache, prep_record)
codes = {"SE": (SE_ic, SE_tc), "C0": (C0_ic, C0_tc)}
wts = {m: {lab: ra.label_episode_weights(codes[m][0], codes[m][1], episodes[lab]) for lab in LABELS} for m in codes}


def cand_idx(e):
    return np.concatenate([e.positive[:, None], e.distractors], axis=1)


# checks 2 and 3, exactly through the original scoring path (raw features, float32)
def old_path(lab, d, ic, tc, w, beta):
    e = episodes[lab]
    c = cand_idx(e)
    if d == "i2t":
        qf, cf, qc, cc = img_raw, txt_raw, ic, tc
    else:
        qf, cf, qc, cc = txt_raw, img_raw, tc, ic
    return tie_aware_rank(conditional_score(tt(qf[e.anchor]), tt(cf[c]), tt(qc[e.anchor]), tt(cc[c]), w, beta)).numpy()


for lab in LABELS:
    for d in DIRS:
        zero = torch.zeros(len(episodes[lab].anchor), 32)
        r = old_path(lab, d, SE_ic, SE_tc, zero, BETA)
        ref = refs[f"clip_only__-__0.3__{lab}__{d}"]
        assert np.array_equal(r, ref), f"clip_only mismatch {lab} {d}"
        for m in ("SE", "C0"):
            r = old_path(lab, d, *codes[m], wts[m][lab], BETA)
            ref = refs[f"{KEY('naive', m, BETA)}__{lab}__{d}"]
            assert np.array_equal(r, ref), f"{m} naive mismatch {lab} {d}"
log("CHECK 2 passed: CLIP only reproduces stored clip_only__-__0.3 ranks exactly (both labels, both directions)")
log("CHECK 3 passed: SE and C0 naive beta 0.3 reproduce stored ranks exactly (both labels, both directions)")
checks.update(check2="exact", check3="exact")

# ------------------------------------------------------------------ normalized features, episode tensors
def nrm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


img_n, txt_n = nrm(img_raw), nrm(txt_raw)   # NaN rows stay NaN; never indexed


class Ep:
    pass


EP = {}
for lab in LABELS:
    e, c = episodes[lab], cand_idx(episodes[lab])
    o = Ep()
    o.n = len(e.anchor)
    o.img = {"a": tt(img_n[e.anchor]), "c": tt(img_n[c]), "s": tt(img_n[e.supports]), "k": tt(img_n[e.contrasts])}
    o.txt = {"a": tt(txt_n[e.anchor]), "c": tt(txt_n[c]), "s": tt(txt_n[e.supports]), "k": tt(txt_n[e.contrasts])}
    EP[lab] = o


def zs(t):
    t = t.double()
    mu, sd = t.mean(1, keepdim=True), t.std(1, unbiased=False, keepdim=True)
    return torch.where(sd > 0, (t - mu) / sd.clamp_min(1e-300), torch.zeros_like(t))


def cos_q(q, c):
    return F.cosine_similarity(q[:, None, :], c, dim=-1)


def mod(o, d):
    """(query modality dict, candidate modality dict) for direction d."""
    return (o.img, o.txt) if d == "i2t" else (o.txt, o.img)


# ---- batched L2 logistic regression in the dual (representer) form, intercept unpenalized
def fit_dual(X, y, C=1.0, iters=60):
    """X (n,m,D) double, y (m,) in {0,1}. Minimizes 0.5||w||^2 + C*sum logloss, w = X^T a. Returns a (n,m)."""
    n, m, _ = X.shape
    s = (2 * y - 1).double()
    K = X @ X.transpose(1, 2) + 1e-10 * torch.eye(m, dtype=X.dtype)
    a = torch.zeros(n, m, dtype=X.dtype)
    b = torch.zeros(n, dtype=X.dtype)

    def obj(a, b):
        f = (K @ a[..., None])[..., 0] + b[:, None]
        return 0.5 * (a * (K @ a[..., None])[..., 0]).sum(1) + C * F.softplus(-s * f).sum(1)

    cur = obj(a, b)
    for _ in range(iters):
        f = (K @ a[..., None])[..., 0] + b[:, None]
        p = torch.sigmoid(s * f)
        g = -s * (1 - p)                          # d loss / d f
        h = (p * (1 - p)).clamp_min(1e-12)
        ga = (K @ (a + C * g)[..., None])[..., 0]
        gb = C * g.sum(1)
        Hh = C * h
        Haa = K + C * K @ (h[..., None] * K)
        Hab = C * (K @ h[..., None])[..., 0]
        Hbb = C * h.sum(1)
        H = torch.zeros(n, m + 1, m + 1, dtype=X.dtype)
        H[:, :m, :m], H[:, :m, m], H[:, m, :m], H[:, m, m] = Haa, Hab, Hab, Hbb
        H = H + 1e-9 * torch.eye(m + 1, dtype=X.dtype)
        step = torch.linalg.solve(H, torch.cat([ga, gb[:, None]], 1)[..., None])[..., 0]
        best, ba, bb = cur.clone(), a.clone(), b.clone()
        for t in (1.0, 0.5, 0.25, 0.125, 0.0625):
            na, nb = a - t * step[:, :m], b - t * step[:, m]
            o = obj(na, nb)
            better = o < best - 1e-14
            best = torch.where(better, o, best)
            ba = torch.where(better[:, None], na, ba)
            bb = torch.where(better, nb, bb)
        a, b, cur = ba, bb, best
    return a


YLAB = torch.tensor([1] * 4 + [0] * 4)


def probe_scores(o, d, both):
    qm, cm = mod(o, d)
    if both:
        X = torch.cat([o.img["s"], o.img["k"], o.txt["s"], o.txt["k"]], 1).double()
        y = torch.cat([YLAB, YLAB])
    else:
        X = torch.cat([cm["s"], cm["k"]], 1).double()
        y = YLAB
    a = fit_dual(X, y)
    w = torch.einsum("nm,nmd->nd", a, X)
    return torch.einsum("nkd,nd->nk", cm["c"].double(), w)


def sk_scores(o, d, both, idx, tol):
    qm, cm = mod(o, d)
    out = []
    for i in idx:
        if both:
            X = torch.cat([o.img["s"][i], o.img["k"][i], o.txt["s"][i], o.txt["k"][i]]).numpy()
            y = np.array([1] * 4 + [0] * 4 + [1] * 4 + [0] * 4)
        else:
            X = torch.cat([cm["s"][i], cm["k"][i]]).numpy()
            y = np.array([1] * 4 + [0] * 4)
        lr = LogisticRegression(C=1.0, max_iter=1000, tol=tol).fit(X.astype(np.float64), y)
        out.append(lr.decision_function(cm["c"][i].numpy().astype(np.float64)))
    return np.array(out)


# sanity: torch dual vs sklearn on 50 random episodes (ranking within episode)
rng = np.random.default_rng(0)
probe_check = {}
for lab in LABELS:
    o = EP[lab]
    idx = rng.choice(o.n, 25, replace=False)
    sub = Ep()
    sub.img = {k: v[idx] for k, v in o.img.items()}
    sub.txt = {k: v[idx] for k, v in o.txt.items()}
    for both in (False, True):
        for d in DIRS:
            mine = probe_scores(sub, d, both).numpy()
            for tol in (1e-4, 1e-10):
                sk = sk_scores(sub, d, both, range(len(idx)), tol)
                same = (np.argsort(-mine, 1) == np.argsort(-sk, 1)).all(1).mean()
                probe_check[f"{lab}/{d}/{'x' if both else 'm'}/tol{tol:g}"] = float(same)
bad = {k: v for k, v in probe_check.items() if "tol1e-10" in k and v < 1.0}
log("probe torch-vs-sklearn identical-ordering share:", probe_check)
assert not bad, f"batched probe ordering differs from sklearn (tight tol): {bad}"

# ------------------------------------------------------------------ terms
def term(name, lab, d):
    o = EP[lab]
    qm, cm = mod(o, d)
    q, c = qm["a"], cm["c"]
    if name in ("proto", "proto_pos"):
        mp = cm["s"].mean(1)
        t = cos_q(mp, c)
        if name == "proto":
            t = t - cos_q(cm["k"].mean(1), c)
        return t
    if name == "probe":
        return probe_scores(o, d, False)
    if name == "probe_x":
        return probe_scores(o, d, True)
    if name == "dir":
        pair = lambda s: 0.5 * (o.img[s] + o.txt[s])
        dd = F.normalize(pair("s").mean(1) - pair("k").mean(1), dim=-1)
        return (q * dd).sum(-1, keepdim=True) * (c * dd[:, None, :]).sum(-1)
    if name in ("SE_term", "C0_term"):
        m = name[:2] if name.startswith("SE") else "C0"
        ic, tc = codes[m]
        e = episodes[lab]
        qc, cc = (ic, tc) if d == "i2t" else (tc, ic)
        return conditional_score(tt(q), tt(c), tt(qc[e.anchor]), tt(cc[cand_idx(e)]), wts[m][lab], 0.0)
    raise KeyError(name)


def hits(r):
    return (np.asarray(r) <= 1).astype(np.float64)


ranks = {}          # "scorer__setting__label__dir" -> ranks
SETTINGS = {}       # scorer -> list of setting strings
TERMS = ["proto", "proto_pos", "probe", "probe_x", "dir", "SE_term", "C0_term"]
for name in TERMS:
    t1 = perf_counter()
    SETTINGS[name] = [f"lam{l:g}" for l in LAMS]
    for lab in LABELS:
        for d in DIRS:
            T = term(name, lab, d)
            qm, cm = mod(EP[lab], d)
            base = zs(cos_q(qm["a"], cm["c"]))
            for l in LAMS:
                s = zs(T) if l == float("inf") else base + l * zs(T)
                ranks[f"{name}__lam{l:g}__{lab}__{d}"] = tie_aware_rank(s).numpy()
    log(f"term {name} done ({perf_counter() - t1:.1f}s)")

# raw_naive
SETTINGS["raw_naive"] = [f"beta{b:g}" for b in BETAS_RAW]
for lab in LABELS:
    o = EP[lab]
    w = naive_condition_weights(pair_codes(o.img["s"], o.txt["s"]), pair_codes(o.img["k"], o.txt["k"]))
    for d in DIRS:
        qm, cm = mod(o, d)
        for b in BETAS_RAW:
            s = conditional_score(qm["a"], cm["c"], qm["a"], cm["c"], w, b)
            ranks[f"raw_naive__beta{b:g}__{lab}__{d}"] = tie_aware_rank(s).numpy()
log("raw_naive done")

# stored references
REF = {"SE": KEY("naive", "SE", BETA), "C0": KEY("naive", "C0", BETA), "R3": KEY("naive", "R3", BETA),
       "CLIP_only": KEY("clip_only", "-", BETA)}
for nm, k in REF.items():
    for lab in LABELS:
        for d in DIRS:
            ranks[f"{nm}__ref__{lab}__{d}"] = refs[f"{k}__{lab}__{d}"]


# ------------------------------------------------------------------ metrics
def H(prefix, lab, d):
    return hits(ranks[f"{prefix}__{lab}__{d}"])


def r1_block(getter):
    """getter(lab, d) -> hit array. Returns R@1 % per scope x (i2t, t2i, mean)."""
    out = {}
    for scope in LABELS + ("pooled",):
        labs = LABELS if scope == "pooled" else (scope,)
        v = {d: float(np.concatenate([getter(l, d) for l in labs]).mean() * 100) for d in DIRS}
        v["mean"] = 0.5 * (v["i2t"] + v["t2i"])
        out[scope] = v
    return out


results = {"episodes_sha256": checks, "probe_vs_sklearn_ordering": probe_check, "r1": {}, "cv_picks": {}, "diffs": {}}
for scorer, sets in SETTINGS.items():
    for s in sets:
        results["r1"][f"{scorer}__{s}"] = r1_block(lambda l, d, p=f"{scorer}__{s}": H(p, l, d))
for nm in REF:
    results["r1"][nm] = r1_block(lambda l, d, p=f"{nm}__ref": H(p, l, d))

# cross-fitting by episode-index parity
for scorer, sets in SETTINGS.items():
    folds = {}
    cv = {(l, d): np.zeros(EP[l].n) for l in LABELS for d in DIRS}
    for tune_par in (0, 1):
        def pooled_mean(s, par):
            tot = []
            for d in DIRS:
                tot.append(np.concatenate([H(f"{scorer}__{s}", l, d)[np.arange(EP[l].n) % 2 == par] for l in LABELS]).mean())
            return float(np.mean(tot)) * 100
        scores = [pooled_mean(s, tune_par) for s in sets]
        pick = sets[int(np.argmax(scores))]            # first in grid order on ties
        folds[f"tuned_on_parity{tune_par}"] = {"picked": pick, "tune_pooled_r1": max(scores), "all": dict(zip(sets, scores))}
        for l in LABELS:
            m = np.arange(EP[l].n) % 2 == 1 - tune_par
            for d in DIRS:
                cv[(l, d)][m] = ranks[f"{scorer}__{pick}__{l}__{d}"][m]
    for (l, d), r in cv.items():
        ranks[f"{scorer}_cv__cv__{l}__{d}"] = r
    results["cv_picks"][scorer] = folds
    results["r1"][f"{scorer}_cv"] = r1_block(lambda l, d, p=f"{scorer}_cv__cv": H(p, l, d))

# paired differences
CV_SCORERS = [f"{s}_cv" for s in SETTINGS]
for scorer in CV_SCORERS:
    for base in ("SE", "C0"):
        res = {}
        pa = f"{scorer}__cv" if scorer in [f"{s}_cv" for s in SETTINGS] else None
        per = {l: 0.5 * ((H(pa, l, "i2t") - H(f"{base}__ref", l, "i2t")) + (H(pa, l, "t2i") - H(f"{base}__ref", l, "t2i")))
               for l in LABELS}
        per["pooled"] = np.concatenate([per[l] for l in LABELS])
        for scope, v in per.items():
            b = paired_bootstrap(v, 5000, 42)
            res[scope] = {"point_pp": b["point"] * 100, "ci95_pp": [b["ci95"][0] * 100, b["ci95"][1] * 100]}
        results["diffs"][f"{scorer}_minus_{base}"] = res

np.savez_compressed(OUT / "spike_ranks.npz", **{k: np.asarray(v) for k, v in ranks.items()})
results["runtime_seconds"] = perf_counter() - T0
(OUT / "spike_results.json").write_text(json.dumps(results, indent=1))

# ------------------------------------------------------------------ compact table
log("FINAL TABLE: R@1 % (mean of directions): pooled / emotion / style")
rows = [f"{s}_cv" for s in SETTINGS] + ["SE", "C0", "R3", "CLIP_only"]
for r in rows:
    v = results["r1"][r]
    log(f"  {r:14s} {v['pooled']['mean']:6.2f} {v['emotion']['mean']:6.2f} {v['art_style']['mean']:6.2f}")
log("PICKS per fold (tuned on parity p, applied to the other):")
for s, f in results["cv_picks"].items():
    log(f"  {s:10s} p0 -> {f['tuned_on_parity0']['picked']}, p1 -> {f['tuned_on_parity1']['picked']}")
log("DIFFS vs SE (pp, 95% CI): pooled / emotion / style")
for k, v in results["diffs"].items():
    if k.endswith("_minus_SE"):
        log("  " + f"{k:24s} " + "  ".join(f"{v[s]['point_pp']:+.2f} [{v[s]['ci95_pp'][0]:+.2f},{v[s]['ci95_pp'][1]:+.2f}]" for s in ("pooled", "emotion", "art_style")))
log("DIFFS vs C0:")
for k, v in results["diffs"].items():
    if k.endswith("_minus_C0"):
        log("  " + f"{k:24s} " + "  ".join(f"{v[s]['point_pp']:+.2f} [{v[s]['ci95_pp'][0]:+.2f},{v[s]['ci95_pp'][1]:+.2f}]" for s in ("pooled", "emotion", "art_style")))
log(f"runtime {perf_counter() - T0:.1f}s")
