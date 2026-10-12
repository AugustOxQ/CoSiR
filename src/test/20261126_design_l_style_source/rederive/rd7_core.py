"""Independent R7 arrays: R3 D3–D9 and R7 L11–L14, §9.

Feature, probability, term and metric code copied from R3's rd3_core, never
imported. Only frozen components are loaded through r7_common (R3 §8).
"""
import numpy as np
import torch
import r7_common as R

zscore_rows = R.load("src.model.aspect_rule").zscore_rows

COND = ("a", "b")
DIRS = ("i2t", "t2i")
METRICS = ("r1", "gain", "other", "swap", "strict")
A0 = ("affect", "image", "caption")
E2_PARTS = ("affect", "image", "caption")      # E2's k-means groupings (affect = affect-km), n6 posteriors
PAIR_NAMES = ("emotion__style", "emotion__genre", "style__genre")
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


def cell_id(t, u, a):
    """D8 item 4: (t*7 + u)*8 + a, zero-based."""
    return (t * len(NU) + u) * len(NA) + a


def cell_decode(i):
    return i // (len(NU) * len(NA)), (i // len(NA)) % len(NU), i % len(NA)


def cell_desc(i, taus):
    t, u, a = cell_decode(int(i))
    return {"cell": int(i), "tau_index": int(t), "tau": float(taus[t]), "lambda_u": NU[u], "lambda_a": NA[a]}


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

def reader_probs(pk, F, parts=A0):
    """P^c = mean over the two half-readers of predict_proba(scaler.transform(x)) (float64); also the halves."""
    if tuple(pk["groupings"]) != tuple(parts) or list(pk["feature_names"]) != feature_names(parts):
        raise AssertionError("reader groupings or feature layout differ from the configuration")
    assert len(pk["halves"]) == 2, "two half-readers expected"
    halves, P = {}, {}
    for c in COND:
        hs = []
        for h in pk["halves"]:
            p = h["model"].predict_proba(h["scaler"].transform(F[c]))
            assert p.shape == (len(F[c]), len(parts)), "half-reader probability shape"
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "half-reader probability range"
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
