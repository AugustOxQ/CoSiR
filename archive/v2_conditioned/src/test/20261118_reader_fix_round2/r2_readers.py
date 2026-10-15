"""Pure-array pieces of round 2's readers R2 and R3 (DECISION_RULE.md of this folder, §4.3 and §4.4). No data is loaded
and no file is written here; run_r2_readers.py does both. Round 1's rb_features (via r2_common) supplies the feature
code, labels, SMD, probability averaging and picks, exactly as the rule names them.

R2 (§4.3), the adapted learned reader:
  restandardise_stats   a: mu42 = X.mean(0), sigma42 = X.std(0) (ddof 0) of the 24,576 seed-42 rows; exact 0 -> 1
  half_probs            a: model_j.predict_proba((x - m_j) / s_j) for given per-half statistics (m_j, s_j)
  reader_probs          a: P^c(h) = mean over the two half-readers (rb_features.average_probs)
  scaler_stats          e: each half's own scaler statistics (mean_, scale_), for the code check
  code_check            e: R2's code with own statistics and no EM equals R1 to 1e-12 with identical picks, else stop
  adapt                 b: P'(h) = P(h) pi(h) / pi_train(h) / sum_h' P(h') pi(h') / pi_train(h')
  em_prior              b: Saerens, Latinne and Decaestecker (2002) EM for the class prior, stop rule as written

R3 (§4.4), the realistic-practice learned reader:
  replacement_draws     b: positions order and replacement rows img, cap from default_rng(s_j), in the rule's order
  replaced_mask         c: position p of side s is replaced at purity k iff p is among order[n, s, :4 - k]
  impure_pairs          c: the bank's four pair arrays with the replaced positions swapped for (img, cap)
  bank_matrix           d, e, g: features of both conditions (rb_features.both_conditions), stacked as stage_train does
  d_table, choose_k     f: SMD per feature (rb_features.smd), D(k) = mean |SMD|, k* = argmin, exact ties to larger k
  require_equal, require_smd_equal   d, f: the purity-4 checks (exact), a failure stops
  mean_abs_delta        i: diagnostic mean |Delta_h| per grouping
"""
import hashlib
from types import SimpleNamespace

import numpy as np

import r2_common as R

rf = R.rf
CONDITIONS = ("a", "b")                       # src.eval.aspect_metrics.CONDITIONS; condition a's rows first

# --------------------------------------------------------------- constants written in the rule
EM_TOL, EM_MAX_ITER = 1e-10, 10_000           # §4.3 b
CODE_CHECK_ATOL = 1e-12                       # §4.3 e (absolute)
R3_SEEDS = (21_700, 21_800)                   # §4.4 b: s_0, s_1
PURITIES = (1, 2, 3, 4)                       # §4.4 c, f
N_PAIRS = 4
PAIR_KEYS = (("pairs_a_img", "pairs_a_txt"), ("pairs_b_img", "pairs_b_txt"))   # side 0, side 1
HALF_ROWS = (91_949, 91_745)                  # §4.4 a: rows of half 0 and half 1
BANK_N = {"A0": 49_152, "A1": 98_304}         # §4.4 a, §4.8: episodes per half
BLOCK_N = 16_384                              # episodes per block
SMOKE_PER_BLOCK = 1_000                       # smoke: the first 1,000 episodes of each block (draws still for all N)

if rf.N_PAIRS != N_PAIRS or tuple(R.C.CONDITIONS) != CONDITIONS:
    raise AssertionError("round 1's pair count or condition order differs from round 2's")


def sha_array(a) -> str:
    """SHA-256 of an array's bytes in C order (rb_build.sha_array)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# =============================================================== R2 (§4.3)

def restandardise_stats(Fa, Fb):
    """(mu42, sigma42, replaced): X = numpy.vstack([F_a, F_b]) in float64, mu42 = X.mean(axis=0), sigma42 =
    X.std(axis=0) (ddof 0). An entry of sigma42 that is exactly 0 is replaced by 1; its index is returned."""
    X = np.vstack([np.asarray(Fa, dtype=np.float64), np.asarray(Fb, dtype=np.float64)])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    zero = np.flatnonzero(sd == 0)
    sd = sd.copy()
    sd[zero] = 1.0
    return mu, sd, [int(i) for i in zero]


def scaler_stats(halves):
    """[(scaler_j.mean_, scaler_j.scale_)] of the half-readers (the code check's statistics, §4.3 e)."""
    return [(np.asarray(h["scaler"].mean_, dtype=np.float64), np.asarray(h["scaler"].scale_, dtype=np.float64))
            for h in halves]


def half_probs(halves, X, stats, n_classes):
    """Per half-reader j: model_j.predict_proba((x - m_j) / s_j), with (m_j, s_j) = stats[j]. R2 passes (mu42, sigma42)
    for both halves; the code check passes each half's own scaler statistics."""
    if len(halves) != len(stats):
        raise ValueError("one (mean, scale) pair per half-reader")
    X = np.asarray(X, dtype=np.float64)
    out = []
    for h, (m, s) in zip(halves, stats):
        if not np.array_equal(h["model"].classes_, np.arange(n_classes)):
            raise AssertionError("a half-reader's classes differ from the configuration's groupings")
        p = h["model"].predict_proba((X - np.asarray(m, dtype=np.float64)) / np.asarray(s, dtype=np.float64))
        if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise AssertionError("half-reader probabilities do not sum to 1")
        out.append(p)
    return out


def reader_probs(halves, F, stats, n_classes):
    """{c: P^c} = mean over the half-readers (rb_features.average_probs) of half_probs on condition c's features."""
    return {c: rf.average_probs(half_probs(halves, F[c], stats, n_classes)) for c in CONDITIONS}


def code_check(P_r1, P_own, atol=CODE_CHECK_ATOL):
    """§4.3 e: R2's code with each half's own scaler statistics and without EM (P_own) must reproduce R1's P^c (P_r1)
    to within ``atol`` (absolute, every entry) with identical picks (ties to the first grouping). Returns the record;
    a failure stops (SystemExit) before any R2 probability exists."""
    rec, ok = {}, True
    for c in CONDITIONS:
        a, b = np.asarray(P_own[c], dtype=np.float64), np.asarray(P_r1[c], dtype=np.float64)
        if a.shape != b.shape:
            raise SystemExit(f"R2 code check: condition {c} shapes differ ({a.shape} vs {b.shape})")
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        same = bool(np.array_equal(rf.picks_and_margins(a)[0], rf.picks_and_margins(b)[0]))
        rec[c] = {"max_abs_diff": d, "picks_identical": same, "n": int(len(a))}
        ok &= (d <= atol) and same
    rec["tolerance_abs"] = atol
    rec["passed"] = bool(ok)
    if not ok:
        raise SystemExit(f"R2 code check failed (DECISION_RULE.md §4.3 e): {rec}")
    return rec


def adapt(P, pi_hat, pi_train):
    """P'(h) = P(h) pi_hat(h) / pi_train(h) / sum_h' P(h') pi_hat(h') / pi_train(h'), row by row, float64."""
    P = np.asarray(P, dtype=np.float64)
    num = P * np.asarray(pi_hat, dtype=np.float64) / np.asarray(pi_train, dtype=np.float64)
    return num / num.sum(axis=1, keepdims=True)


def em_prior(P, pi_train, tol=EM_TOL, max_iter=EM_MAX_ITER):
    """§4.3 b. pi^(0) = pi_train; for s = 0, 1, ...: P'^(s) = adapt(P, pi^(s), pi_train) and pi^(s+1) = mean of P'^(s)
    over the rows. Stop at the first s with max_h |pi^(s+1) - pi^(s)| < tol, or when s + 1 = max_iter; pi_hat =
    pi^(s+1). Returns (pi_hat, record) with n_iter = s + 1 (the number of updates), whether the criterion stopped it,
    whether the cap did, and the last change."""
    P = np.asarray(P, dtype=np.float64)
    pt = np.asarray(pi_train, dtype=np.float64)
    pi = pt.copy()
    s = 0
    while True:
        new = adapt(P, pi, pt).mean(axis=0)
        change = float(np.max(np.abs(new - pi)))
        n_iter = s + 1
        if change < tol:
            converged = True
            break
        if n_iter == max_iter:
            converged = False
            break
        pi = new
        s += 1
    return new, {"n_iter": int(n_iter), "converged_by_criterion": bool(converged), "cap_reached": not converged,
                 "last_max_abs_change": change, "tol": tol, "max_iter": int(max_iter),
                 "pi_start": pt.tolist(), "n_rows": int(len(P))}


# =============================================================== R3 (§4.4)

def replacement_draws(anchor, rows, paint, seed):
    """§4.4 b for one half: one generator g = numpy.random.default_rng(seed), used in this order:
      1. U = g.random((N, 2, 4)); order = argsort(U, axis=2, kind="stable");
      2. img = R[g.integers(0, len(R), size=(N, 2, 4))]; while some entry has the anchor's painting, redraw exactly
         those entries (row-major) with R[g.integers(0, len(R), size=bad.sum())];
      3. cap likewise, bad = anchor's painting or the painting of img at the same slot.
    anchor: (N,) local rows; rows: R = the half's local rows; paint: painting of each local row.
    Returns (order, img, cap, info), each array (N, 2, 4) int64."""
    A = np.asarray(anchor, dtype=np.int64)
    Rw = np.asarray(rows, dtype=np.int64)
    paint = np.asarray(paint)
    N = len(A)
    g = np.random.default_rng(seed)
    U = g.random((N, 2, N_PAIRS))
    order = np.argsort(U, axis=2, kind="stable").astype(np.int64)
    pa = paint[A][:, None, None]
    img = Rw[g.integers(0, len(Rw), size=(N, 2, N_PAIRS))]
    rounds_img, redrawn_img = 0, 0
    while True:
        bad = paint[img] == pa
        if not bad.any():
            break
        rounds_img += 1
        redrawn_img += int(bad.sum())
        img[bad] = Rw[g.integers(0, len(Rw), size=bad.sum())]
    cap = Rw[g.integers(0, len(Rw), size=(N, 2, N_PAIRS))]
    rounds_cap, redrawn_cap = 0, 0
    while True:
        bad = (paint[cap] == pa) | (paint[cap] == paint[img])
        if not bad.any():
            break
        rounds_cap += 1
        redrawn_cap += int(bad.sum())
        cap[bad] = Rw[g.integers(0, len(Rw), size=bad.sum())]
    info = {"seed": int(seed), "n_episodes": int(N), "n_rows": int(len(Rw)),
            "img_redraw_rounds": rounds_img, "img_entries_redrawn": redrawn_img,
            "cap_redraw_rounds": rounds_cap, "cap_entries_redrawn": redrawn_cap}
    return order, img.astype(np.int64), cap.astype(np.int64), info


def replaced_mask(order, k):
    """(N, 2, 4) bool: True at the positions order[n, s, 0] .. order[n, s, 3 - k] (the first 4 - k entries of
    order[n, s]), the pairs replaced at purity k. Purity 4 replaces nothing."""
    if k not in PURITIES:
        raise ValueError(f"purity must be one of {PURITIES}, got {k}")
    order = np.asarray(order)
    mask = np.zeros(order.shape, dtype=bool)
    np.put_along_axis(mask, order[:, :, :N_PAIRS - k], True, axis=2)
    return mask


def impure_pairs(pairs, order, img, cap, k):
    """§4.4 c: copies of the bank's four pair arrays in which, for every episode n and side s (0: pairs_a, 1: pairs_b),
    pairs_{a|b}_img[n, p] becomes img[n, s, p] and pairs_{a|b}_txt[n, p] becomes cap[n, s, p] at each replaced position
    p (replaced_mask). Nested across k by construction: the same (img, cap) at every purity that replaces p."""
    mask = replaced_mask(order, k)
    out = {}
    for s, (ki, kt) in enumerate(PAIR_KEYS):
        m = mask[:, s, :]
        out[ki] = np.where(m, img[:, s, :], np.asarray(pairs[ki], dtype=np.int64))
        out[kt] = np.where(m, cap[:, s, :], np.asarray(pairs[kt], dtype=np.int64))
    return out


def bank_matrix(post, parts, pairs, ya, yb):
    """(X, y, episode): features of both conditions (rb_features.both_conditions; condition b from the swapped sets),
    stacked with rb_features.stack_conditions (rows 0..N-1 condition a, N..2N-1 condition b), as rb_build.bank_features
    computes them for a bank."""
    ep = SimpleNamespace(**{k: pairs[k] for kk in PAIR_KEYS for k in kk})
    F = rf.both_conditions(post, parts, ep)
    X, y, epi = rf.stack_conditions(F["a"], F["b"], ya, yb)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite bank features")
    return X, y, epi


def subset_index(n_blocks, block_n, per_block):
    """The first ``per_block`` episodes of each block (smoke subset), ascending."""
    return np.concatenate([np.arange(i * block_n, i * block_n + per_block, dtype=np.int64) for i in range(n_blocks)])


def stacked_rows(idx, n):
    """Row indices of a stack_conditions matrix that belong to episodes ``idx`` of an n-episode bank."""
    idx = np.asarray(idx, dtype=np.int64)
    return np.concatenate([idx, n + idx])


def d_table(x42, bank_by_k):
    """§4.4 f: {k: (smd (F,), D(k))}, SMD = rb_features.smd(seed-42 rows, bank rows of purity k), D(k) = mean over the
    features of |SMD|. An SMD that is not finite stops the step."""
    out = {}
    for k in sorted(bank_by_k):
        s = rf.smd(x42, bank_by_k[k])
        if not np.isfinite(s).all():
            raise SystemExit(f"R3: purity {k} has a non-finite SMD (features {np.flatnonzero(~np.isfinite(s)).tolist()}); "
                             "stopping (DECISION_RULE.md §4.4 f)")
        out[k] = (s, float(np.mean(np.abs(s))))
    return out


def choose_k(D):
    """k* = the purity with the smallest D(k) at full precision; exact ties go to the larger k."""
    if not D or not all(np.isfinite(v) for v in D.values()):
        raise SystemExit(f"R3: D(k) must be finite for every purity: {D}")
    best = None
    for k in sorted(D):
        if best is None or D[k] <= D[best]:
            best = k
    return best


def require_equal(what, got, want):
    """Exact equality (shape, dtype and every value); a failure stops the step (§4.4 d)."""
    got, want = np.asarray(got), np.asarray(want)
    if not (got.shape == want.shape and got.dtype == want.dtype and np.array_equal(got, want)):
        raise SystemExit(f"R3 purity-4 check failed: {what} differs from round 1's (DECISION_RULE.md §4.4 d)")
    return True


def require_smd_equal(smd, names, stored):
    """§4.4 f: the purity-4 SMDs equal round 1's shift report (c_shift_report.smd, name -> value) exactly."""
    if list(stored) != list(names):
        raise SystemExit("R3 purity-4 SMD check failed: feature names differ from round 1's shift report")
    for n, v in zip(names, np.asarray(smd, dtype=np.float64)):
        if stored[n] is None or float(v) != float(stored[n]):
            raise SystemExit(f"R3 purity-4 SMD check failed: {n} {float(v)!r} != round 1's {stored[n]!r} "
                             "(DECISION_RULE.md §4.4 f)")
    return True


def mean_abs_delta(X, parts):
    """§4.4 i diagnostic: per grouping, the mean of |Delta_h| (feature column 6 j + 2) over the rows of X."""
    j = {f: i for i, f in enumerate(rf.FEATURES)}["Delta"]
    return {h: float(np.mean(np.abs(np.asarray(X, dtype=np.float64)[:, len(rf.FEATURES) * i + j])))
            for i, h in enumerate(parts)}


def episode_painting_reuse(base, img, cap, paint):
    """Diagnostic (§4.4 b's remark): the share of drawn replacement slots (N x 2 x 4) whose image or caption comes from
    a painting already in the episode (its anchor, candidates and the 16 pair rows of the base bank)."""
    paint = np.asarray(paint)
    ep = np.concatenate([np.asarray(base["anchor"])[:, None], np.asarray(base["candidates"])]
                        + [np.asarray(base[k]) for kk in PAIR_KEYS for k in kk], axis=1)
    ep_p = paint[ep]                                               # (N, 30) paintings of the episode
    n = len(ep_p)
    hit = np.zeros((n, img[0].size), dtype=bool)
    for arr in (img, cap):
        rp = paint[np.asarray(arr)].reshape(n, -1)                 # (N, 8)
        hit |= (rp[:, :, None] == ep_p[:, None, :]).any(axis=2)
    return float(hit.mean())
