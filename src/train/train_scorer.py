"""Train the stage-(d) conditional scorer on frozen factor codes (spec §4).

Episodes are mined fresh every step from a condition source built on scorer-train rows, so no episode is
reused. The loss is a multi-positive ranking loss in both directions, plus an optional swap term.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.model.condition_interface import ConditionalScorer, ResidualConditionInterface
from src.train.condition_episodes import mine_condition_episodes, mine_swap_episodes, pair_feature_units


@dataclass
class ScorerTrainingConfig:
    steps: int = 3000
    batch_episodes: int = 64
    episodes_per_condition: int = 4
    lr: float = 1e-3
    swap: bool = False
    lambda_swap: float = 1.0
    beta_init: float = 0.3
    hidden: int = 16
    hard_pool: int = 2048
    seed: int = 42


def multi_positive_nce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every episode needs at least one positive")
    positives = logits.masked_fill(~positive_mask, float("-inf"))
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positives, dim=1)).mean()


def _t(values, device):
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)


def episode_logits(scorer, anchor, supports, contrasts, candidates, img_feat, txt_feat, img_codes, txt_codes,
                   device) -> dict:
    support_pair = 0.5 * (_t(img_codes[supports], device) + _t(txt_codes[supports], device))
    contrast_pair = 0.5 * (_t(img_codes[contrasts], device) + _t(txt_codes[contrasts], device))
    weights = scorer.weights(support_pair, contrast_pair)
    out = {}
    for direction, qf, cf, qc, cc in (("i2t", img_feat, txt_feat, img_codes, txt_codes),
                                      ("t2i", txt_feat, img_feat, txt_codes, img_codes)):
        scores = scorer.score(_t(qf[anchor], device), _t(cf[candidates], device),
                              _t(qc[anchor], device), _t(cc[candidates], device), weights)
        out[direction] = scores / scorer.tau
    return out


def _rank_loss(scorer, ep, data, device):
    logits = episode_logits(scorer, ep.anchor, ep.supports, ep.contrasts, ep.candidates, *data, device)
    mask = torch.as_tensor(ep.positive_mask, device=device)
    return 0.5 * (multi_positive_nce(logits["i2t"], mask) + multi_positive_nce(logits["t2i"], mask))


def _swap_loss(scorer, sw, data, device):
    total = 0.0
    for supports, contrasts, mask in ((sw.supports_a, sw.contrasts_a, sw.positive_mask_a),
                                      (sw.supports_b, sw.contrasts_b, sw.positive_mask_b)):
        logits = episode_logits(scorer, sw.anchor, supports, contrasts, sw.candidates, *data, device)
        m = torch.as_tensor(mask, device=device)
        total = total + 0.5 * (multi_positive_nce(logits["i2t"], m) + multi_positive_nce(logits["t2i"], m))
    return total / 2


def train_scorer(source, img_feat, txt_feat, img_codes, txt_codes, keys, factor_scale, config: ScorerTrainingConfig,
                 device=None, log_every=100):
    if config.swap and not source.swap_capable:
        raise ValueError(f"{source.name} cannot form swap pairs; train it without the swap term")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    units = pair_feature_units(img_feat, txt_feat)
    data = (img_feat, txt_feat, img_codes, txt_codes)
    scorer = ConditionalScorer(ResidualConditionInterface(factor_scale, config.hidden),
                               beta_init=config.beta_init).to(device)

    def mine():
        return mine_condition_episodes(source, units, keys, config.batch_episodes, rng,
                                       config.episodes_per_condition, hard_pool=config.hard_pool)

    first = mine()
    with torch.no_grad():                                   # tau := std of step-0 scores (unit-scale logits)
        logits = episode_logits(scorer, first.anchor, first.supports, first.contrasts, first.candidates,
                                *data, device)
        scorer.set_tau(float(torch.cat([logits["i2t"].ravel(), logits["t2i"].ravel()]).std()))
    optimizer = torch.optim.Adam(scorer.parameters(), lr=config.lr)
    history = {"step": [], "loss": [], "loss_rank": [], "loss_swap": [], "beta": [], "tau": []}
    for step in range(config.steps):
        episodes = first if step == 0 else mine()
        loss_rank = _rank_loss(scorer, episodes, data, device)
        loss = loss_rank
        loss_swap = None
        if config.swap:
            swaps = mine_swap_episodes(source, units, keys, config.batch_episodes, rng,
                                       config.episodes_per_condition, hard_pool=config.hard_pool)
            loss_swap = _swap_loss(scorer, swaps, data, device)
            loss = loss + config.lambda_swap * loss_swap
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == config.steps - 1:
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["loss_rank"].append(float(loss_rank))
            if loss_swap is not None:
                history["loss_swap"].append(float(loss_swap))
            history["beta"].append(float(scorer.beta))
            history["tau"].append(float(scorer.tau))
    return scorer.eval(), history


def save_scorer_checkpoint(scorer: ConditionalScorer, config: ScorerTrainingConfig, path) -> None:
    torch.save({"state_dict": scorer.state_dict(), "config": asdict(config),
                "num_factors": int(scorer.interface.factor_scale.numel())}, path)


def load_scorer_checkpoint(path, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    config = ScorerTrainingConfig(**payload["config"])
    scorer = ConditionalScorer(ResidualConditionInterface(torch.ones(payload["num_factors"]), config.hidden),
                               beta_init=config.beta_init)
    scorer.load_state_dict(payload["state_dict"])
    return scorer.to(device).eval(), config
