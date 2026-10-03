"""Tier-1 raw-feature baselines for aspect episodes (CVPR plan spec §8): similarity estimated directly from the 4+4
example pairs, plus few-shot classics adapted to pairs. Each returns a term-score dict scores[cond][dir] (n, 13)."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


@dataclass
class PcaBasis:
    mean_img: np.ndarray
    mean_txt: np.ndarray
    components: np.ndarray     # (r, D)
    scale: np.ndarray          # (r,), so that projections have unit variance on the fit rows

    def project(self, x, modality):
        mean = self.mean_img if modality == "img" else self.mean_txt
        return ((np.asarray(x, np.float32) - mean) @ self.components.T) / self.scale


def fit_pca_basis(img_rows, txt_rows, r: int = 32, seed: int = 42) -> PcaBasis:
    """Unsupervised: centre each modality, PCA on the stacked rows, whiten. Fit on training rows only."""
    mi, mt = img_rows.mean(0), txt_rows.mean(0)
    stacked = np.concatenate([img_rows - mi, txt_rows - mt]).astype(np.float64)
    rng = np.random.default_rng(seed)
    if len(stacked) > 50_000:
        stacked = stacked[rng.choice(len(stacked), 50_000, replace=False)]
    _, s, vt = np.linalg.svd(stacked, full_matrices=False)
    comps = vt[:r].astype(np.float32)
    scale = (s[:r] / np.sqrt(len(stacked) - 1)).astype(np.float32)
    return PcaBasis(mi.astype(np.float32), mt.astype(np.float32), comps, scale)


def _sides(inputs, ep, d):
    if d == "i2t":
        return inputs.img[ep.anchor], inputs.txt[ep.candidates], "img", "txt"
    return inputs.txt[ep.anchor], inputs.img[ep.candidates], "txt", "img"


def _pairs(inputs, ep, cond):
    si, st, ci, ct, _ = ep.condition(cond)
    return inputs.img[si], inputs.txt[st], inputs.img[ci], inputs.txt[ct]   # (n, 4, D) each


def _each(fn):
    """Build scores[cond][dir] by calling fn(cond, d) -> (n, 13) numpy."""
    return {c: {d: fn(c, d) for d in DIRECTIONS} for c in CONDITIONS}


def diag_agreement_term(inputs, ep, relu: bool):
    def fn(cond, d):
        sx, sy, cx, cy = _pairs(inputs, ep, cond)
        w = (sx * sy).mean(1) - (cx * cy).mean(1)
        if relu:
            w = np.maximum(w, 0.0)
        q, c, _, _ = _sides(inputs, ep, d)
        return np.einsum("nd,nkd->nk", q * w, c)
    return _each(fn)


def _proj_pairs(inputs, ep, cond, basis):
    sx, sy, cx, cy = _pairs(inputs, ep, cond)
    p = lambda a, m: basis.project(a.reshape(-1, a.shape[-1]), m).reshape(a.shape[0], a.shape[1], -1)  # noqa: E731
    return p(sx, "img"), p(sy, "txt"), p(cx, "img"), p(cy, "txt")


def _proj_sides(inputs, ep, d, basis):
    q, c, qm, cm = _sides(inputs, ep, d)
    n, k, dim = c.shape
    return basis.project(q, qm), basis.project(c.reshape(-1, dim), cm).reshape(n, k, -1)


def bilinear_agreement_term(inputs, ep, basis):
    def fn(cond, d):
        sx, sy, cx, cy = _proj_pairs(inputs, ep, cond, basis)
        m = np.einsum("nsi,nsj->nij", sx, sy) / sx.shape[1] - np.einsum("nsi,nsj->nij", cx, cy) / cx.shape[1]
        m = 0.5 * (m + m.transpose(0, 2, 1))
        q, c = _proj_sides(inputs, ep, d, basis)
        return np.einsum("ni,nij,nkj->nk", q, m, c)
    return _each(fn)


def _cov(delta):                      # (n, s, r) -> (n, r, r) + I
    r = delta.shape[-1]
    return np.einsum("nsi,nsj->nij", delta, delta) / delta.shape[1] + np.eye(r, dtype=np.float32)


def kissme_term(inputs, ep, basis):
    def fn(cond, d):
        sx, sy, cx, cy = _proj_pairs(inputs, ep, cond, basis)
        m = np.linalg.inv(_cov(sx - sy)) - np.linalg.inv(_cov(cx - cy))
        q, c = _proj_sides(inputs, ep, d, basis)
        diff = q[:, None, :] - c
        return -np.einsum("nki,nij,nkj->nk", diff, m, diff)
    return _each(fn)


def rca_term(inputs, ep, basis):
    def fn(cond, d):
        sx, sy, _, _ = _proj_pairs(inputs, ep, cond, basis)
        vals, vecs = np.linalg.eigh(_cov(sx - sy))
        w = np.einsum("nij,nj,nkj->nik", vecs, 1.0 / np.sqrt(vals), vecs)       # Sigma^(-1/2)
        q, c = _proj_sides(inputs, ep, d, basis)
        qw, cw = np.einsum("nij,nj->ni", w, q), np.einsum("nij,nkj->nki", w, c)
        qw /= np.linalg.norm(qw, axis=-1, keepdims=True)
        cw /= np.linalg.norm(cw, axis=-1, keepdims=True)
        return np.einsum("ni,nki->nk", qw, cw)
    return _each(fn)


def xing_term(inputs, ep, basis, steps: int = 100):
    def fn(cond, d):
        sx, sy, cx, cy = (_t(a) for a in _proj_pairs(inputs, ep, cond, basis))
        ds, dc = (sx - sy) ** 2, (cx - cy) ** 2
        a = torch.ones(ds.shape[0], ds.shape[-1], requires_grad=True)
        opt = torch.optim.Adam([a], lr=0.05)
        for _ in range(steps):
            loss = ((ds * a[:, None]).sum(-1).sum(-1)
                    - torch.log(torch.sqrt((dc * a[:, None]).sum(-1) + 1e-8).sum(-1))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                a.clamp_(min=0.0)
        q, c = (_t(v) for v in _proj_sides(inputs, ep, d, basis))
        return (-(((q[:, None] - c) ** 2) * a.detach()[:, None]).sum(-1)).numpy()
    return _each(fn)


def wang_term(inputs, ep, steps: int = 50):
    def fn(cond, d):
        sx, sy, cx, cy = (_t(a) for a in _pairs(inputs, ep, cond))
        ps, pc = sx * sy, cx * cy                                        # (n, 4, D)
        w = torch.ones(ps.shape[0], ps.shape[-1], requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.01)
        for _ in range(steps):
            sim_s = (ps * (w ** 2)[:, None]).sum(-1)                     # (n, 4)
            sim_c = (pc * (w ** 2)[:, None]).sum(-1)
            loss = F.softplus(sim_c[:, None, :] - sim_s[:, :, None]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        q, c, _, _ = _sides(inputs, ep, d)
        return (_t(q)[:, None] * _t(c) * (w.detach() ** 2)[:, None]).sum(-1).numpy()
    return _each(fn)


@dataclass
class PairScaler:
    mean: np.ndarray           # (D,) mean of z = x*y over same-row image-caption products of training rows
    std: np.ndarray            # (D,)


def fit_pair_scaler(img_rows, txt_rows, seed: int = 42) -> PairScaler:
    """Unsupervised statistics of z = x*y (rows unit-normalized first, like EvalInputs); fit on training rows only."""
    x = np.asarray(img_rows, np.float64)
    y = np.asarray(txt_rows, np.float64)
    if len(x) > 50_000:
        keep = np.random.default_rng(seed).choice(len(x), 50_000, replace=False)
        x, y = x[keep], y[keep]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    z = x * y
    return PairScaler(z.mean(0).astype(np.float32), np.maximum(z.std(0), 1e-8).astype(np.float32))


def pair_probe_term(inputs, ep, scaler, steps: int = 300, l2: float = 1.0, return_grad_norm: bool = False):
    """Standardized L2 logistic probe on z = x*y (S = 1, C = 0), fit per episode with Adam (lr 0.05) on the SUM over
    episodes of [mean BCE + 0.5*l2*||beta||^2/8], so each episode's optimisation does not depend on n."""
    mean, std = _t(scaler.mean), _t(scaler.std)
    grad_norms = []

    def fn(cond, d):
        sx, sy, cx, cy = (_t(a) for a in _pairs(inputs, ep, cond))
        z = (torch.cat([sx * sy, cx * cy], dim=1) - mean) / std           # (n, 8, D)
        y = torch.cat([torch.ones(sx.shape[:2]), torch.zeros(cx.shape[:2])], dim=1)
        beta = torch.zeros(z.shape[0], z.shape[-1], requires_grad=True)
        bias = torch.zeros(z.shape[0], requires_grad=True)
        opt = torch.optim.Adam([beta, bias], lr=0.05)
        for _ in range(steps):
            logits = (z * beta[:, None]).sum(-1) + bias[:, None]
            per_ep = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(1) \
                + 0.5 * l2 * (beta ** 2).sum(-1) / z.shape[1]
            opt.zero_grad(); per_ep.sum().backward(); opt.step()
        logits = (z * beta[:, None]).sum(-1) + bias[:, None]
        per_ep = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(1) \
            + 0.5 * l2 * (beta ** 2).sum(-1) / z.shape[1]
        g_beta, g_bias = torch.autograd.grad(per_ep.sum(), [beta, bias])
        grad_norms.append(torch.sqrt((g_beta ** 2).sum(-1) + g_bias ** 2).mean().item())
        q, c, _, _ = _sides(inputs, ep, d)
        qc = (_t(q)[:, None] * _t(c) - mean) / std
        return ((qc * beta.detach()[:, None]).sum(-1) + bias.detach()[:, None]).numpy()
    out = _each(fn)
    return (out, float(np.mean(grad_norms))) if return_grad_norm else out


def tip_adapter_term(inputs, ep, gamma: float = 5.0):
    def fn(cond, d):
        sx, sy, cx, cy = _pairs(inputs, ep, cond)
        q, c, _, _ = _sides(inputs, ep, d)
        qc = q[:, None] * c                                               # (n, 13, D)
        qc /= np.linalg.norm(qc, axis=-1, keepdims=True) + 1e-12
        def aff(px, py):
            k = px * py
            k /= np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12
            return np.exp(-gamma * (1.0 - np.einsum("nkd,nsd->nks", qc, k))).sum(-1)
        return aff(sx, sy) - aff(cx, cy)
    return _each(fn)


def value_prototype_term(inputs, ep):
    def fn(cond, d):
        si, st, ci, ct, _ = ep.condition(cond)
        q, c, _, cm = _sides(inputs, ep, d)
        feats = inputs.txt if cm == "txt" else inputs.img
        sup = np.concatenate([feats[si], feats[st]], axis=1).mean(1)
        con = np.concatenate([feats[ci], feats[ct]], axis=1).mean(1)
        sup /= np.linalg.norm(sup, axis=1, keepdims=True)
        con /= np.linalg.norm(con, axis=1, keepdims=True)
        return np.einsum("nkd,nd->nk", c, sup) - np.einsum("nkd,nd->nk", c, con)
    return _each(fn)
