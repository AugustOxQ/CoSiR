"""Naive-rule condition-episode loss for training factor encoders (factor-learning spec §5).

The zero-parameter naive rule turns an episode's supports and contrasts into factor weights. The episode's
candidates are scored with ``conditional_score`` at a fixed beta in both retrieval directions, and a
multi-positive softmax rewards ranking the positives first. Gradients reach the factor codes through both the
weights and the candidate scores. The only free parameter is the temperature, which does not change rankings.
"""

import torch
from torch import Tensor

from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes
from src.train.train_scorer import multi_positive_nce

DIRECTIONS = ("i2t", "t2i")


def naive_episode_scores(img_feat: Tensor, txt_feat: Tensor, img_codes: Tensor, txt_codes: Tensor,
                         anchor: Tensor, supports: Tensor, contrasts: Tensor, candidates: Tensor,
                         beta: float) -> dict[str, Tensor]:
    """Scores ``(E, K)`` per direction for E episodes over R encoded rows.

    ``img_feat`` / ``txt_feat`` are the rows' CLIP features ``(R, D)``; ``img_codes`` / ``txt_codes`` their factor
    codes ``(R, F)``. ``anchor (E,)``, ``supports (E, S)``, ``contrasts (E, C)`` and ``candidates (E, K)`` index
    those rows. i2t scores the anchor's image against the candidates' captions; t2i the anchor's caption against
    the candidates' images.
    """
    weights = naive_condition_weights(pair_codes(img_codes[supports], txt_codes[supports]),
                                      pair_codes(img_codes[contrasts], txt_codes[contrasts]))
    return {
        "i2t": conditional_score(img_feat[anchor], txt_feat[candidates], img_codes[anchor], txt_codes[candidates],
                                 weights, beta),
        "t2i": conditional_score(txt_feat[anchor], img_feat[candidates], txt_codes[anchor], img_codes[candidates],
                                 weights, beta),
    }


def naive_episode_loss(img_feat: Tensor, txt_feat: Tensor, img_codes: Tensor, txt_codes: Tensor,
                       anchor: Tensor, supports: Tensor, contrasts: Tensor, candidates: Tensor,
                       positive_mask: Tensor, beta: float, log_tau: Tensor) -> Tensor:
    """Mean over both directions of the multi-positive softmax loss of ``naive_episode_scores / exp(log_tau)``."""
    scores = naive_episode_scores(img_feat, txt_feat, img_codes, txt_codes, anchor, supports, contrasts,
                                  candidates, beta)
    tau = log_tau.exp()
    return 0.5 * sum(multi_positive_nce(scores[d] / tau, positive_mask) for d in DIRECTIONS)
