"""Score matrices for aspect episodes: backbone cosine, the agreement rule on factor codes (and its uniform-weight
control), z-fusion and lambda cross-fitting (CVPR plan spec §5.1, §8)."""

from dataclasses import dataclass

import numpy as np
import torch

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.model.aspect_rule import agreement_weights, zfuse

LAMBDA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, float("inf")]
EDGE_EXTENSION = [32.0, 64.0]


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@dataclass
class EvalInputs:
    img: np.ndarray
    txt: np.ndarray
    img_codes: np.ndarray | None = None
    txt_codes: np.ndarray | None = None

    def __post_init__(self):
        self.img, self.txt = _unit(self.img), _unit(self.txt)


def _query_side(inputs, ep, d):
    """(query features, candidate features, query codes, candidate codes) for direction d."""
    cand = ep.candidates
    if d == "i2t":
        return (inputs.img[ep.anchor], inputs.txt[cand],
                None if inputs.img_codes is None else inputs.img_codes[ep.anchor],
                None if inputs.txt_codes is None else inputs.txt_codes[cand])
    return (inputs.txt[ep.anchor], inputs.img[cand],
            None if inputs.txt_codes is None else inputs.txt_codes[ep.anchor],
            None if inputs.img_codes is None else inputs.img_codes[cand])


def cosine_scores(inputs: EvalInputs, ep) -> dict:
    out = {c: {} for c in CONDITIONS}
    for d in DIRECTIONS:
        q, c, _, _ = _query_side(inputs, ep, d)
        s = np.einsum("nd,nkd->nk", q, c)
        for cond in CONDITIONS:
            out[cond][d] = s                                   # identical under both conditions by construction
    return out


def _weights(inputs, ep, cond, uniform):
    si, st, ci, ct, _ = ep.condition(cond)
    if uniform:
        f = inputs.img_codes.shape[1]
        return torch.full((len(ep.anchor), f), 1.0 / f)
    return agreement_weights(_t(inputs.img_codes[si]), _t(inputs.txt_codes[st]), _t(inputs.img_codes[ci]),
                             _t(inputs.txt_codes[ct]))


def agreement_term(inputs: EvalInputs, ep, uniform: bool = False) -> dict:
    """Factor term sum_l w_l q_l c_l with agreement-rule weights (or uniform weights: the condition removed)."""
    out = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        w = _weights(inputs, ep, cond, uniform)
        for d in DIRECTIONS:
            _, _, qc, cc = _query_side(inputs, ep, d)
            out[cond][d] = (w[:, None, :] * _t(qc)[:, None, :] * _t(cc)).sum(-1).numpy()
    return out


def fixed_beta_scores(inputs: EvalInputs, ep, beta: float, uniform: bool = False) -> dict:
    """beta * cos + factor term: the score used in training (spec §6)."""
    cos, term = cosine_scores(inputs, ep), agreement_term(inputs, ep, uniform)
    return {c: {d: beta * cos[c][d] + term[c][d] for d in DIRECTIONS} for c in CONDITIONS}


def fused_scores(cos: dict, term: dict, lam: float) -> dict:
    return {c: {d: zfuse(_t(cos[c][d]), _t(term[c][d]), lam).numpy() for d in DIRECTIONS} for c in CONDITIONS}


def _criterion(scores, rows):
    m = per_anchor({c: {d: scores[c][d][rows] for d in DIRECTIONS} for c in CONDITIONS})
    return 0.5 * (m["r1"].mean() + m["gain"].mean())


def crossfit_lambda(cos: dict, term: dict, parity: np.ndarray):
    """Pick lambda on each parity half by mean(R@1, condition gain), apply it to the other half; one edge extension."""
    parity = np.asarray(parity)
    fused = {lam: fused_scores(cos, term, lam) for lam in LAMBDA_GRID}
    picks = {}
    out = {c: {d: np.empty_like(cos[c][d]) for d in DIRECTIONS} for c in CONDITIONS}
    for half in (0, 1):
        tune, apply = parity == half, parity != half
        best = max(fused, key=lambda lam: _criterion(fused[lam], tune))
        if best == LAMBDA_GRID[-2]:                                  # 16 picked: extend the grid once
            for lam in EDGE_EXTENSION:
                fused[lam] = fused_scores(cos, term, lam)
            best = max(fused, key=lambda lam: _criterion(fused[lam], tune))
        picks[half] = best
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[c][d][apply] = fused[best][c][d][apply]
    return out, picks
