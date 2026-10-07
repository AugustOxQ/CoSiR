"""Round 4 A1 reader and the candidates' gates (DECISION_RULE.md D3, D5, D6, section 5 items 3 and 5).

Pure functions on a bundle (a SimpleNamespace; interface of r4_bundle): F1 ({c: (n, 24) float64} A1 features) and
readers_a1 (round 1's pickle dict of the two A1 half-readers). Round 3's reader, family and gates are reused through
r4_common (RF3), never reimplemented. Every gate is a float32 array of 0s and 1s per condition, as RF3.gates_aff
returns it, and every candidate's gate is closed wherever AFF's is closed.
"""
import numpy as np

import r4_common as R4

RF3, R3, C = R4.RF3, R4.R3, R4.C
rbe, rf = R3.rbe, R3.rf
from src.eval.aspect_metrics import CONDITIONS  # noqa: E402

N_GROUPINGS_A1 = len(R4.A1)          # 4
AFFECT = R3.AFFECT                   # 0, the first grouping of A1 as of A0


# ---------------------------------------------------------------- the A1 reader (D3)

def reader_a1(bundle, readers=None):
    """Round 1's two A1 half-readers on the bundle's 24 A1 features (D3). -> {"P": {c: (n, 4) float64}, "pick": {c: (n,)
    int64}}; pick = numpy.argmax, so an exact tie goes to the first grouping in A1 order (index 0 = affect).
    readers: the pickle dict (injected by tests); default bundle.readers_a1."""
    if readers is None:
        readers = bundle.readers_a1
    if list(readers["feature_names"]) != rf.feature_names(R4.A1):
        raise AssertionError("the A1 readers were trained on another feature layout")
    P = {}
    for c in CONDITIONS:
        X = np.asarray(bundle.F1[c], dtype=np.float64)
        P[c] = rf.average_probs(rbe.half_reader_probs(readers, X, N_GROUPINGS_A1))
        if P[c].shape != (len(X), N_GROUPINGS_A1) or not np.allclose(P[c].sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise AssertionError("averaged A1 probabilities do not have shape (n, 4) summing to 1")
    return {"P": P, "pick": {c: np.asarray(P[c]).argmax(axis=1).astype(np.int64) for c in CONDITIONS}}


# ---------------------------------------------------------------- abstention and gates (D5, D6)

def abstain(v, v75=R4.V75):
    """a_v = 1[v < v75], float32 (n,), the same for both conditions (D5)."""
    return (np.asarray(v, dtype=np.float64) < float(v75)).astype(np.float32)


def _check_01(g, what):
    for t, gt in enumerate(g):
        for c in CONDITIONS:
            x = np.asarray(gt[c])
            if x.dtype != np.float32:
                raise AssertionError(f"{what}: tau {t} condition {c} is {x.dtype}, not float32")
            if not np.all((x == 0) | (x == 1)):
                raise AssertionError(f"{what}: tau {t} condition {c} is not 0/1")


def apply_keep(g, keep):
    """The one gate-factor function behind V4's abstention (and the IMGABST path of section 5 item 3): g_t^c * keep, for
    every tau index and condition. keep: (n,) 0/1 float32 (a_v), the same for both conditions."""
    keep = np.asarray(keep)
    for gt in g:
        for c in CONDITIONS:
            if keep.shape != np.asarray(gt[c]).shape:
                raise AssertionError("the keep factor's shape differs from the gate's")
    if keep.dtype != np.float32 or not np.all((keep == 0) | (keep == 1)):
        raise AssertionError("the keep factor must be a float32 0/1 array")
    return [{c: (np.asarray(gt[c]) * keep).astype(np.float32) for c in CONDITIONS} for gt in g]


def apply_affect_pick(g, pick_a1):
    """g_t^c * 1[pi_A1^c = affect] (V2's factor)."""
    for gt in g:
        for c in CONDITIONS:
            if np.asarray(pick_a1[c]).shape != np.asarray(gt[c]).shape:
                raise AssertionError("the A1 pick's shape differs from the gate's")
    return [{c: (np.asarray(gt[c]) * (np.asarray(pick_a1[c]) == AFFECT).astype(np.float32)).astype(np.float32)
             for c in CONDITIONS} for gt in g]


def gates_candidate(name, g_aff, pick_a1=None, keep=None):
    """D6: AFF's gate times the candidate's factors. name in AFF, V4 (a_v), V2 (1[pi_A1 = affect]), V24 (both).
    -> list of 4 {c: (n,) float32 0/1}; asserts float32, 0/1 and every gate closed wherever AFF's is closed."""
    if name not in ("AFF", "V4", "V2", "V24"):
        raise ValueError(f"unknown candidate {name!r}")
    if name in ("V4", "V24") and keep is None:
        raise ValueError(f"{name} needs the abstention factor keep")
    if name in ("V2", "V24") and pick_a1 is None:
        raise ValueError(f"{name} needs the A1 picks")
    if name == "AFF" and (pick_a1 is not None or keep is not None):
        raise ValueError("AFF takes no veto factors")
    _check_01(g_aff, "AFF gate")
    g = [{c: np.array(gt[c], copy=True) for c in CONDITIONS} for gt in g_aff]
    if name in ("V4", "V24"):
        g = apply_keep(g, keep)
    if name in ("V2", "V24"):
        g = apply_affect_pick(g, pick_a1)
    _check_01(g, f"{name} gate")
    for t in range(len(g)):
        for c in CONDITIONS:
            if np.any(g[t][c] > np.asarray(g_aff[t][c])):
                raise AssertionError(f"{name}: gate open where AFF's is closed (tau {t}, condition {c})")
    return g


def gates_imgabst_r1(g_r1, keep):
    """Section 5 item 3: R1's gates times a_v, by the same factor function V4 uses (apply_keep)."""
    _check_01(g_r1, "R1 gate")
    g = apply_keep(g_r1, keep)
    _check_01(g, "IMGABST gate")
    return g


def run_candidate(bundle, T, name, g_aff, pick_a1=None, keep=None, fused_only=False, return_gates=False):
    """The candidate's family (D7): its gates (gates_candidate) in a single RF3.run_family call, so the matched
    counterpart G_cf is built from the candidate's own gates, never AFF's. return_gates=True returns (family, gates),
    the gates being the very list the family was run from (final review S6: the runner checks them against item 5's)."""
    g = gates_candidate(name, g_aff, pick_a1=pick_a1, keep=keep)
    fam = RF3.run_family(bundle, T, g, fused_only=fused_only)
    return (fam, g) if return_gates else fam
