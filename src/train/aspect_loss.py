"""Pseudo-aspect episode loss (CVPR plan spec §6): cross-entropy on the correct target under each condition and
direction, plus a swap term that the target outranks the other aspect's candidate."""

import torch
from torch import Tensor
from torch.nn import functional as F

from src.model.aspect_rule import agreement_weights
from src.model.conditioning import conditional_score

CONDITIONS = ("a", "b")
DIRECTIONS = ("i2t", "t2i")
TARGET = {"a": 0, "b": 1}


def aspect_episode_scores(img_feat, txt_feat, img_codes, txt_codes, idx: dict, beta: float) -> dict:
    """idx holds row-local index tensors: anchor (E,), candidates (E,13), pa_img/pa_txt/pb_img/pb_txt (E,4)."""
    roles = {"a": (idx["pa_img"], idx["pa_txt"], idx["pb_img"], idx["pb_txt"]),
             "b": (idx["pb_img"], idx["pb_txt"], idx["pa_img"], idx["pa_txt"])}
    out = {}
    for cond, (si, st, ci, ct) in roles.items():
        w = agreement_weights(img_codes[si], txt_codes[st], img_codes[ci], txt_codes[ct])
        a, c = idx["anchor"], idx["candidates"]
        out[(cond, "i2t")] = conditional_score(img_feat[a], txt_feat[c], img_codes[a], txt_codes[c], w, beta)
        out[(cond, "t2i")] = conditional_score(txt_feat[a], img_feat[c], txt_codes[a], img_codes[c], w, beta)
    return out


def aspect_episode_loss(scores: dict, log_tau: Tensor, lambda_swap: float) -> Tensor:
    tau = log_tau.exp()
    ce, swap = 0.0, 0.0
    for cond in CONDITIONS:
        t, o = TARGET[cond], 1 - TARGET[cond]
        for d in DIRECTIONS:
            s = scores[(cond, d)] / tau
            target = torch.full((s.shape[0],), t, dtype=torch.long, device=s.device)
            ce = ce + F.cross_entropy(s, target)
            swap = swap + F.softplus(-(s[:, t] - s[:, o])).mean()
    return (ce + lambda_swap * swap) / 4.0
