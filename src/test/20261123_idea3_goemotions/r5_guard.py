"""Round 5 guard (rule D11): placement objects and the release of GE-placement results.

Every function of this round that can receive a placement Q takes a Placement whose kind is 'clip' or 'ge' and calls
require() first. A 'clip' object is built only from the bundle's own Q_CLIP (clip_from_bundle: a read-only view of
``bundle.post["affect"]["txt"]`` plus its fingerprint, never a copy that could be mutated apart from the owner); a 'ge'
object only from the GE file (ge_from_file: the file's SHA-256 must equal r5_common.GE_POST_SHA). require() refuses a
'ge' object until release() has read a results/regression_check.json that records items 1 to 4 as passed under this
rule's SHA-256. require_carry() is the further gate of the measured diagnostics. State is per process.
"""
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import r5_common as R5


class GuardError(RuntimeError):
    """A GE-placement result was requested before the regression checks released it (rule D11), or a placement,
    a regression record or a carry record is not what the rule requires."""


_STATE = {"released": False}
_MINT = object()          # module-private: only clip_from_bundle and ge_from_file hold it
_GE_FPS = set()           # fingerprints of every GE Q minted in this process


def _reset_for_tests():
    _STATE["released"] = False


def is_released() -> bool:
    return bool(_STATE["released"])


def fingerprint(Q) -> str:
    """SHA-256 of the shape, dtype and bytes of Q (NaN pattern included)."""
    a = np.ascontiguousarray(Q)
    h = hashlib.sha256()
    h.update(f"{a.shape}|{a.dtype}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Placement:
    """kind 'clip' or 'ge'; Q a float32 (rows, 41) read-only array; sha256 = Q's fingerprint (clip) or the GE file's
    SHA-256 (ge)."""
    kind: str
    Q: np.ndarray
    sha256: str
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._token is not _MINT:
            raise GuardError("a Placement is built only by clip_from_bundle or ge_from_file")
        if self.kind not in ("clip", "ge"):
            raise GuardError(f"placement kind {self.kind!r} is not 'clip' or 'ge'")
        Q = self.Q
        if not isinstance(Q, np.ndarray) or Q.dtype != np.float32 or Q.ndim != 2 or Q.shape[1] != R5.N_CLASSES:
            raise GuardError(f"a placement is a float32 (rows, {R5.N_CLASSES}) array")


def clip_from_bundle(bundle) -> Placement:
    """Q_CLIP = ``bundle.post["affect"]["txt"]`` as a read-only view (shares memory with the bundle's array) with its
    fingerprint; where round 3's cached affect heads exist, Q must equal their caption posterior by value."""
    Q = bundle.post["affect"]["txt"]
    if not isinstance(Q, np.ndarray) or Q.dtype != np.float32 or Q.ndim != 2 or Q.shape[1] != R5.N_CLASSES:
        raise GuardError(f"the bundle's affect caption posterior is not a float32 (rows, {R5.N_CLASSES}) array")
    if not any(np.array_equal(hit["post"]["affect"]["txt"], Q, equal_nan=True) for hit in R5.RB3._HEADS.values()):
        raise GuardError("the bundle's affect caption posterior equals none of round 3's cached heads "
                         "(the cache is empty or holds other arrays)")
    view = Q.view()
    view.flags.writeable = False
    fp = fingerprint(view)
    if fp in _GE_FPS:
        raise GuardError("the bundle's affect caption posterior is the GE placement, not Q_CLIP")
    return Placement("clip", view, fp, _MINT)


def ge_from_file(path) -> Placement:
    """Q_GE from cache/r5_ge_posterior.npz (keys post_sel, rows, classes): the file's SHA-256 must equal
    r5_common.GE_POST_SHA (refused while it is None). Scatters post_sel into a NaN float32 (308,723, 41) array."""
    n_rows, n_sel = R5.N_ROWS, R5.N_SELECTION
    if R5.GE_POST_SHA is None:
        raise GuardError("r5_common.GE_POST_SHA is not set: the GE posterior file has not been committed as an input")
    data = Path(path).read_bytes()                       # one read: the bytes hashed are the bytes loaded
    sha = hashlib.sha256(data).hexdigest()
    if sha != R5.GE_POST_SHA:
        raise GuardError(f"{Path(path).name}: SHA-256 {sha} differs from r5_common.GE_POST_SHA")
    with np.load(io.BytesIO(data)) as z:
        post_sel, rows, classes = z["post_sel"], z["rows"], z["classes"]
    K = R5.N_CLASSES
    if post_sel.dtype != np.float32 or post_sel.shape != (n_sel, K):
        raise GuardError(f"post_sel is {post_sel.dtype} {post_sel.shape}, not float32 ({n_sel}, {K})")
    if rows.dtype != np.int64 or rows.shape != (n_sel,) or not (np.diff(rows) > 0).all() \
            or rows[0] < 0 or rows[-1] >= n_rows:
        raise GuardError("rows must be int64, ascending, unique and inside the dataset")
    if not np.array_equal(classes, np.arange(K)):
        raise GuardError("classes must be 0..40 (partition_L label order)")
    if not np.isfinite(post_sel).all() or not (np.abs(post_sel.sum(axis=1, dtype=np.float64) - 1.0) <= 1e-5).all():
        raise GuardError("post_sel must be finite with rows summing to 1 within 1e-5")
    Q = np.full((n_rows, K), np.nan, np.float32)
    Q[rows] = post_sel
    Q.flags.writeable = False
    _GE_FPS.add(fingerprint(Q))
    return Placement("ge", Q, sha, _MINT)


def require(placement, what) -> Placement:
    """Called first by every function that can receive a placement: refuses a 'ge' placement before release(); a
    'clip' placement must still match its fingerprint (nobody wrote into it)."""
    if not isinstance(placement, Placement):
        raise GuardError(f"{what}: a Placement is required, got {type(placement).__name__}")
    if placement._token is not _MINT:
        raise GuardError(f"{what}: this Placement was not minted by clip_from_bundle or ge_from_file")
    if placement.kind == "ge":
        if R5.GE_POST_SHA is None or placement.sha256 != R5.GE_POST_SHA:
            raise GuardError(f"{what}: the GE placement is not bound to r5_common.GE_POST_SHA")
        if not _STATE["released"]:
            raise GuardError(f"{what}: a GE-placement result is refused before rule section 5 items 1 to 4 have "
                             f"passed and results/regression_check.json is released (rule D11)")
    else:
        fp = fingerprint(placement.Q)
        if fp != placement.sha256:
            raise GuardError(f"{what}: the CLIP placement changed since it was taken from the bundle")
        if fp in _GE_FPS:
            raise GuardError(f"{what}: a clip placement carries the GE array")
    return placement


def _read_record(path, name, what):
    p = Path(path)
    if p.name != name or p.parent.name != "results":
        raise GuardError(f"{what}: only a results/{name} is accepted, got {p}")
    if not p.is_file():
        raise GuardError(f"{what}: {p} does not exist")
    try:
        rec = json.loads(p.read_text())
    except ValueError as e:
        raise GuardError(f"{what}: {p.name} is not valid JSON ({e})") from None
    if not isinstance(rec, dict) or rec.get("rule_sha256") != R5.RULE_SHA:
        raise GuardError(f"{what}: {p.name} does not carry this rule's SHA-256")
    return rec


def release(path):
    """Releases the guard from results/regression_check.json: this rule's SHA-256, items '1' to '4' each with
    passed True, all_passed True. Anything else raises GuardError and leaves the guard closed."""
    rec = _read_record(path, "regression_check.json", "release")
    items = rec.get("items")
    ok = (isinstance(items, dict) and set(items) == {"1", "2", "3", "4"}
          and all(isinstance(v, dict) and v.get("passed") is True for v in items.values())
          and rec.get("all_passed") is True)
    if not ok:
        raise GuardError("release: regression_check.json does not record items 1 to 4 as passed")
    _STATE["released"] = True


def require_carry(path):
    """The measured diagnostics (rule §5) are refused until results/carry.json exists and carries this rule's SHA-256."""
    return _read_record(path, "carry.json", "require_carry")
