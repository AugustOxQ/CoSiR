"""Round 5 re-derivation: the placement function (D4), the GE input and head with its fallback, and the placement
swap (D5: a new post dict, never an assignment into a shared one). scikit-learn's LogisticRegression is called
directly; nothing of rounds 2 to 5 is imported."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from rd5_paths import sha_array

DRAW_SEED, CHECK_SEED, N_DRAW, N_CHECK = 0, 1, 60_000, 10_000
MAX_ITER, FALLBACK_MAX_ITER = 300, 3_000
DRAW_SHA = "7be956c09bf716547df20264435388bd3645ae963df728636774e80359cdef5c"     # told_oracle.json arm L


def unit(x) -> np.ndarray:
    """Unit-normalised rows in float32 (the CLIP heads' transform)."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def identity(x) -> np.ndarray:
    return np.asarray(x)


def draw_rows(scorer_train, n_draw=N_DRAW, n_check=N_CHECK):
    draw = np.random.default_rng(DRAW_SEED).choice(scorer_train, n_draw, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(CHECK_SEED).choice(rest, min(n_check, len(rest)), replace=False)
    return draw, check


def placement(F, transform, lab, scorer_train, rows, max_iter, n_draw=N_DRAW, n_check=N_CHECK) -> dict:
    """Fit LogisticRegression(C=1, max_iter) on transform(F[draw]) -> lab[draw]; posterior on `rows` scattered into
    a NaN float32 (N, K) array; held-out accuracy on the check rows."""
    draw, check = draw_rows(scorer_train, n_draw, n_check)
    Xd = transform(F[draw])
    Xc = transform(F[check])
    if not (np.isfinite(Xd).all() and np.isfinite(Xc).all()):
        raise AssertionError("draw or check rows are not finite")
    clf = LogisticRegression(C=1.0, max_iter=max_iter).fit(Xd, lab[draw])
    K = len(clf.classes_)
    if not np.array_equal(clf.classes_, np.arange(K)):
        raise AssertionError(f"classes_ must be 0..{K - 1}, got {clf.classes_}")
    post = np.full((len(F), K), np.nan, dtype=np.float32)
    post[rows] = clf.predict_proba(transform(F[rows]))
    return {"post": post, "classes": clf.classes_.copy(), "n_iter": int(np.max(clf.n_iter_)),
            "accuracy": 100 * float(clf.score(Xc, lab[check])), "draw": draw, "check": check,
            "draw_sha": sha_array(np.sort(draw)), "max_iter": max_iter}


def global_labels(local, scorer_train, n_total) -> np.ndarray:
    lab = np.full(n_total, -1, dtype=np.int64)
    lab[scorer_train] = local
    return lab


def ge_input(n_total, scorer_train, affect_probs, rows, probs) -> np.ndarray:
    """X: float32 (N, 28), NaN except X[scorer_train[i]] = affect_probs[i] and X[rows] = probs (raw, no scaler)."""
    st = np.asarray(scorer_train)
    if not (np.all(np.diff(st) > 0) and len(st) == len(affect_probs)):
        raise AssertionError("scorer_train must be ascending and aligned with affect_probs")
    rows = np.asarray(rows)
    if len(np.intersect1d(st, rows)):
        raise AssertionError("selection rows overlap scorer-train rows")
    X = np.full((n_total, affect_probs.shape[1]), np.nan, dtype=np.float32)
    X[st] = affect_probs
    X[rows] = probs
    return X


def check_mapping(scorer_train, draw) -> bool:
    """The draw's positions in scorer-train order satisfy scorer_train[pos] == draw."""
    pos = np.searchsorted(scorer_train, draw)
    return bool(np.array_equal(np.asarray(scorer_train)[pos], draw))


def ge_head(X, lab, scorer_train, rows, n_draw=N_DRAW, n_check=N_CHECK, max_iter=MAX_ITER,
            fallback_max_iter=FALLBACK_MAX_ITER) -> dict:
    """D4: the GE head on the raw probabilities with the fallback (refit once with 3,000 if 300 is reached)."""
    first = placement(X, identity, lab, scorer_train, rows, max_iter, n_draw, n_check)
    rec = {"n_iter_first": first["n_iter"], "fallback_used": False, "n_iter_fallback": None}
    head = first
    if first["n_iter"] >= max_iter:
        second = placement(X, identity, lab, scorer_train, rows, fallback_max_iter, n_draw, n_check)
        rec.update(fallback_used=True, n_iter_fallback=second["n_iter"])
        if second["n_iter"] >= fallback_max_iter:
            rec["stop"] = "the refit reached its cap: no GE posterior; the user decides"
            return {**rec, "head": None}
        head = second
    sel = head["post"][rows]
    if not (np.isfinite(sel).all() and np.isnan(np.delete(head["post"], rows, axis=0)).all()):
        raise AssertionError("Q_GE must be finite on selection rows and NaN elsewhere")
    if not np.allclose(sel.sum(axis=1, dtype=np.float64), 1.0, atol=1e-5, rtol=0):
        raise AssertionError("Q_GE rows must sum to 1 within 1e-5")
    return {**rec, "head": head}


# ---------------------------------------------------------------- the placement swap (D5)

def swap_caption_side(post, Q) -> dict:
    """A NEW post dict with the affect caption side replaced by Q; nothing is assigned into `post`."""
    return {"affect": {"img": post["affect"]["img"], "txt": Q}, "image": post["image"], "caption": post["caption"]}


def fingerprint(post) -> dict:
    """Identity and value fingerprint of every array of a post dict (and of the dict objects themselves)."""
    out = {"ids": {"post": id(post)}}
    for h, sides in post.items():
        out["ids"][h] = id(sides)
        for m, arr in sides.items():
            out[f"{h}/{m}"] = (id(arr), sha_array(arr), arr.dtype.str, arr.shape)
    return out
