"""Round 5 placement (rule DECISION_RULE.md D4, section 5 item 3): one placement function for the CLIP heads and the GE head.

``place`` does exactly what ``run_told_oracle.fit_one_head`` does for one modality: the draw
``default_rng(0).choice(scorer_train, 60_000, replace=False)``, the check rows ``default_rng(1).choice(rest, 10_000)``,
``LogisticRegression(C=1.0, max_iter=max_iter)`` on ``transform(F[draw])``, ``predict_proba(transform(F[rows]))`` scattered
into a NaN float32 (n_rows, K) array and the held-out accuracy 100 * score on the check rows. ``fit_ge_head`` adds the
fallback (one refit with max_iter 3,000; a second cap stops). No GoEmotions value is printed by this module.
"""
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import r5_common as R5

N_DRAW = 60_000          # run_n6.HEAD_ROWS
N_CHECK = 10_000         # run_n6.CHECK_ROWS
DRAW_SEED, CHECK_SEED = 0, 1   # run_checks.PROBE_SEED, and fit_one_head's literal 1


class PlacementError(RuntimeError):
    """A placement input, class set or mapping is not what rule D4 requires, or the GE head did not converge."""


def draw_positions(scorer_train, draw):
    """Positions of ``draw`` in ``scorer_train`` order with ``scorer_train[pos] == draw`` (rule section 5 item 3); a
    draw row that is not a scorer-train row refuses."""
    st, draw = np.asarray(scorer_train), np.asarray(draw)
    order = np.argsort(st, kind="stable")
    srt = st[order]
    idx = np.searchsorted(srt, draw)
    if (idx >= len(srt)).any() or not np.array_equal(srt[np.minimum(idx, len(srt) - 1)], draw):
        raise PlacementError("a draw row is not a scorer-train row")
    pos = order[idx]
    if not np.array_equal(st[pos], draw):
        raise PlacementError("scorer_train[pos] != draw")
    return pos


def identity(x):
    return np.asarray(x)


def place(F, transform, lab, scorer_train, rows, max_iter, n_draw=N_DRAW, n_check=N_CHECK, n_classes=R5.N_CLASSES):
    """Returns {"post" (len(F), K) float32 NaN outside ``rows``, "classes", "n_iter", "acc", "draw", "check"}."""
    scorer_train, rows, lab = np.asarray(scorer_train), np.asarray(rows), np.asarray(lab)
    draw = np.random.default_rng(DRAW_SEED).choice(scorer_train, n_draw, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(CHECK_SEED).choice(rest, min(n_check, len(rest)), replace=False)
    draw_positions(scorer_train, draw)                       # every draw row is a scorer-train row
    for name, r in (("draw", draw), ("check", check)):
        if not np.isfinite(F[r]).all():
            raise PlacementError(f"{name} rows hold non-finite features (they must be scorer-train rows)")
        if (lab[r] < 0).any():
            raise PlacementError(f"{name} rows without a label")
    clf = LogisticRegression(C=1.0, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # reaching the cap is detected through n_iter_
        clf.fit(transform(F[draw]), lab[draw])
    if not np.array_equal(clf.classes_, np.arange(n_classes)):
        raise PlacementError(f"classes_ {clf.classes_.tolist()} are not 0..{n_classes - 1}")
    if not np.isfinite(F[rows]).all():
        raise PlacementError("selection rows hold non-finite features")
    post = np.full((len(F), n_classes), np.nan, dtype=np.float32)
    post[rows] = clf.predict_proba(transform(F[rows]))
    outside = np.ones(len(F), bool)
    outside[rows] = False
    if not (np.isfinite(post[rows]).all() and np.isnan(post[outside]).all()):
        raise PlacementError("posteriors must be finite on the rows and NaN elsewhere")
    acc = 100 * float(clf.score(transform(F[check]), lab[check]))
    return {"post": post, "classes": clf.classes_, "n_iter": int(np.max(clf.n_iter_)), "acc": acc,
            "draw": draw, "check": check}


def head_record(res, lab, n_classes=R5.N_CLASSES):
    """The record fields shared by the CLIP-head and GE-head records."""
    counts = np.bincount(lab[res["check"]], minlength=n_classes)
    return {"heldout_accuracy": res["acc"], "classes": res["classes"].tolist(), "n_iter": res["n_iter"],
            "check_majority_share": 100 * float(counts.max() / counts.sum()), "uniform": 100.0 / n_classes}


def fit_ge_head(X, lab, scorer_train, rows, caps=(R5.GE_MAX_ITER, R5.GE_FALLBACK_MAX_ITER), **kw):
    """D4's fallback: fit at caps[0]; when n_iter_ reaches it, refit once at caps[1] (everything else equal; the refit
    is the head); when the refit reaches caps[1] too, nothing is returned (PlacementError). Returns (res, record)."""
    res = place(X, identity, lab, scorer_train, rows, caps[0], **kw)
    used, first = False, res["n_iter"]
    if first >= caps[0]:
        used = True
        res = place(X, identity, lab, scorer_train, rows, caps[1], **kw)
        if res["n_iter"] >= caps[1]:
            raise PlacementError(f"the GE head reached its fallback cap ({caps[1]}); no GE posterior is written")
    rec = head_record(res, lab, **({"n_classes": kw["n_classes"]} if "n_classes" in kw else {}))
    rec.update({"n_iter": first if not used else res["n_iter"], "fallback_used": used,
                "n_iter_fallback": res["n_iter"] if used else None, "n_iter_first": first})
    return res, rec


def load_goemo(path):
    """The GoEmotions file (keys probs, rows, sample_ids): its SHA-256 must equal r5_common.GOEMO_FILE_SHA (refused
    while that is None)."""
    if R5.GOEMO_FILE_SHA is None:
        raise PlacementError("r5_common.GOEMO_FILE_SHA is not set: the GoEmotions file is not yet an accepted input")
    sha = R5.sha256_file(path)
    if sha != R5.GOEMO_FILE_SHA:
        raise PlacementError(f"{Path(path).name}: SHA-256 {sha} differs from r5_common.GOEMO_FILE_SHA")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def ge_input(ctx, goemo_npz, scorer_train, affect_probs, n_goemo=R5.N_GOEMO):
    """D4's input X: float32 (n_rows, 28), NaN except X[scorer_train[i]] = affect_probs[i] and X[rows] = probs."""
    z = load_goemo(goemo_npz) if isinstance(goemo_npz, (str, Path)) else goemo_npz
    scorer_train, probs, rows = np.asarray(scorer_train), np.asarray(z["probs"]), np.asarray(z["rows"])
    affect_probs = np.asarray(affect_probs)
    if affect_probs.shape != (len(scorer_train), n_goemo):
        raise PlacementError(f"affect_probs {affect_probs.shape} does not match scorer_train ({len(scorer_train)}) x {n_goemo}")
    if probs.shape != (len(rows), n_goemo):
        raise PlacementError(f"probs {probs.shape} does not match rows ({len(rows)}) x {n_goemo}")
    if not np.array_equal(rows, np.asarray(ctx.selection)) or not (np.diff(rows) > 0).all():
        raise PlacementError("the GoEmotions rows are not the context's selection rows (ascending)")
    if len(np.intersect1d(rows, scorer_train)):
        raise PlacementError("selection rows overlap scorer-train rows")
    if not (np.isfinite(probs).all() and np.isfinite(affect_probs).all()):
        raise PlacementError("non-finite GoEmotions probabilities")
    X = np.full((len(ctx.groups), n_goemo), np.nan, dtype=np.float32)
    X[scorer_train] = affect_probs
    X[rows] = probs
    return X


def write_ge_posterior(path, post_sel, rows, classes, n_rows=R5.N_ROWS, n_sel=R5.N_SELECTION, n_classes=R5.N_CLASSES):
    """cache/r5_ge_posterior.npz (keys post_sel, rows, classes), written once; returns the file's SHA-256."""
    path = Path(path)
    R5.refuse_existing([path], False)
    post_sel, rows, classes = np.asarray(post_sel), np.asarray(rows), np.asarray(classes)
    if post_sel.dtype != np.float32 or post_sel.shape != (n_sel, n_classes):
        raise PlacementError(f"post_sel is {post_sel.dtype} {post_sel.shape}, not float32 ({n_sel}, {n_classes})")
    if rows.dtype != np.int64 or rows.shape != (n_sel,) or not (np.diff(rows) > 0).all() or rows[0] < 0 or rows[-1] >= n_rows:
        raise PlacementError("rows must be int64, ascending, unique and inside the dataset")
    if not np.array_equal(classes, np.arange(n_classes)):
        raise PlacementError("classes must be 0..K-1")
    if not np.isfinite(post_sel).all() or not (np.abs(post_sel.sum(axis=1, dtype=np.float64) - 1.0) <= 1e-5).all():
        raise PlacementError("post_sel must be finite with rows summing to 1 within 1e-5")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, post_sel=post_sel, rows=rows, classes=classes.astype(np.int64))
    tmp.rename(path)
    return R5.sha256_file(path)
