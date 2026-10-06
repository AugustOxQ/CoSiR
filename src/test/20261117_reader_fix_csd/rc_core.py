"""R-c's computation (DECISION_RULE.md section 4.3) as pure functions on z-scored torch dicts, so that the unit tests can
run it on synthetic data. run_rc.py wires it to the bundle and the parent candidate."""
import numpy as np
import torch

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.eval.aspect_nested import _combine, _zdict, control_sums, nested_cells
from src.eval.aspect_quick_checks import _require_condition_free

PCTS = (0, 25, 50, 75)


def thresholds(margins):
    """tau_0..tau_3 = numpy.percentile (linear) 0, 25, 50, 75 of the parent's margins over both conditions."""
    m = np.concatenate([np.asarray(margins["a"], np.float64), np.asarray(margins["b"], np.float64)])
    return [float(x) for x in np.percentile(m, PCTS)], int(len(m))


def gates(margins, taus):
    """{k: {cond: (E,) float32}}: g^c = 1 if m^c >= tau_k else 0."""
    return {k: {c: (np.asarray(margins[c]) >= t).astype(np.float32) for c in CONDITIONS} for k, t in enumerate(taus)}


def gated_terms(zT, g):
    """{cond: {dir: torch (E,K)}}: g^c * z(T^c); z-scoring first, the gate multiplies after."""
    return {c: {d: torch.as_tensor(g[c])[:, None] * zT[c][d] for d in DIRECTIONS} for c in CONDITIONS}


def g_cf(gated):
    """G_cf = (g^a z(T^a) + g^b z(T^b)) / 2, identical under both conditions (not re-z-scored). Averaged in float64 as
    diagnose_counterparts.cf_version does, then float32."""
    out = {}
    for d in DIRECTIONS:
        m = (0.5 * (gated["a"][d].numpy().astype(np.float64) + gated["b"][d].numpy().astype(np.float64))).astype(np.float32)
        out[d] = m
    G = {c: {d: torch.as_tensor(out[d].copy()) for d in DIRECTIONS} for c in CONDITIONS}
    _require_condition_free({c: {d: G[c][d].numpy() for d in DIRECTIONS} for c in CONDITIONS}, "G_cf")
    return G


def rc_cells():
    """224 cells (tau index, lambda_u, lambda_a): tau outer, then lambda_u, then lambda_a."""
    return [(k, u, a) for k in range(len(PCTS)) for (u, a) in nested_cells()]


def _vecs(scores):
    pa = per_anchor(scores)
    return np.asarray(pa["r1"], np.float64), np.asarray(pa["gain"], np.float64)


def cell_statistics(zB, gated, G, cells):
    """Per cell: per-episode R@1 and gain of the gated fused reader, per-episode R@1 of the gated counterpart."""
    fr1, fg, cr1 = [], [], []
    for (k, u, a) in cells:
        r, g = _vecs(_combine(zB, zB, gated[k], u, a))
        fr1.append(r), fg.append(g)
        cr1.append(_vecs(_combine(zB, zB, G[k], u, a))[0])
    return np.array(fr1), np.array(fg), np.array(cr1)


def control_choice(zB, parity):
    """Per tune half, sigma maximising R@1 over control_sums() of (1+sigma) z(B) (ties to the smallest sigma)."""
    r1 = {s: _vecs(_combine(zB, zB, zB, s, 0.0))[0] for s in control_sums()}
    out = {}
    for half in (0, 1):
        tune = parity == half
        sigma = max(control_sums(), key=lambda s: float(r1[s][tune].mean()))
        out[half] = (sigma, float(r1[sigma][tune].mean()))
    return out


def select_fused(fr1, fg, ctrl, parity, allowed):
    """Min-margin pick: on each tune half the allowed cell maximising min(R@1 - control R@1, gain) (ties to the
    first). -> {half: cell index}."""
    picks = {}
    for half in (0, 1):
        tune = parity == half
        r_ctrl = ctrl[half][1]
        crit = lambda i: min(float(fr1[i][tune].mean()) - r_ctrl, float(fg[i][tune].mean()))      # noqa: E731
        picks[half] = max(allowed, key=crit)
    return picks


def select_cf(cr1, parity, allowed):
    """Each half picks the allowed cell with the highest R@1 (ties to the first)."""
    return {half: max(allowed, key=lambda i: float(cr1[i][parity == half].mean())) for half in (0, 1)}


def assemble(zB, terms, cells, picks, parity):
    """Scores of each half's picked cell applied to the other half (as crossfit_nested fills them)."""
    shape = {c: {d: zB[c][d].shape for d in DIRECTIONS} for c in CONDITIONS}
    out = {c: {d: np.empty(shape[c][d], np.float32) for d in DIRECTIONS} for c in CONDITIONS}
    for half in (0, 1):
        apply = parity != half
        k, u, a = cells[picks[half]]
        s = _combine(zB, zB, terms[k], u, a)
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[c][d][apply] = s[c][d][apply]
    return out
