"""Verbatim September training functions (commit 1b70d7f).

Source: src/test/20260922_percept_topic_pipeline/
  run_percept_stage1_pilot.py:138-186 (autoencoder, pretrain, Q, P)
  run_percept_stage1_cluster_count_sweep_pilot.py:55-162 (init, joint loop)
Constants: base:38-46, sweep:27-28. The sweep's `base` reference below
points to this module's verbatim base functions, with no September imports.
"""
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

PRETRAIN_BATCH_SIZE = 1024
PRETRAIN_EPOCHS = 100
MAX_DEC_EPOCHS = 500
STABILITY_THRESHOLD = 0.001
PRETRAIN_LEARNING_RATE = 1e-3
DEC_LEARNING_RATE = 1e-4
LAMBDA_RECONSTRUCTION = 1
LAMBDA_BALANCE = 1000

base = sys.modules[__name__]


def build_autoencoder(input_dim: int) -> tuple[nn.Sequential, nn.Sequential]:
    """Build the paper-specified 128-dimensional PercepT Stage 1 autoencoder."""
    encoder = nn.Sequential(
        nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(),
        nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, 128),
    )
    decoder = nn.Sequential(
        nn.Linear(128, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(),
        nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim),
    )
    return encoder, decoder


def pretrain_autoencoder(encoder: nn.Sequential, decoder: nn.Sequential, inputs: torch.Tensor, device: str, log) -> list[float]:
    """Pretrain reconstruction with shuffled 1,024-node batches for 100 epochs."""
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=PRETRAIN_LEARNING_RATE)
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        permutation = torch.randperm(len(inputs))
        total_loss = 0.0
        total_nodes = 0
        for start in range(0, len(inputs), PRETRAIN_BATCH_SIZE):
            batch = inputs[permutation[start:start + PRETRAIN_BATCH_SIZE]].to(device)
            reconstruction_loss = F.mse_loss(decoder(encoder(batch)), batch)
            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()
            total_loss += reconstruction_loss.item() * len(batch)
            total_nodes += len(batch)
        mean_loss = total_loss / total_nodes
        losses.append(mean_loss)
        if epoch % 10 == 0:
            log(f"Pretrain epoch {epoch:03d}/{PRETRAIN_EPOCHS}: mean reconstruction={mean_loss:.6f}")
    return losses


def soft_assignments(latent: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Student's-t DEC assignments (alpha=1), normalized per node."""
    squared_distances = torch.sum((latent.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
    q = 1.0 / (1.0 + squared_distances)
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Return the detached DEC self-sharpened target distribution P."""
    weights = q.pow(2) / q.sum(dim=0, keepdim=True)
    return (weights / weights.sum(dim=1, keepdim=True)).detach()


def initialize_cluster_centers(
    encoder: nn.Sequential,
    inputs: torch.Tensor,
    device: str,
    n_initial_clusters: int,
    seed: int,
) -> nn.Parameter:
    """Initialize the requested number of DEC centers with fresh K-means."""
    encoder.eval()
    with torch.no_grad():
        latent = encoder(inputs.to(device)).cpu().numpy()
    kmeans = KMeans(n_clusters=n_initial_clusters, n_init=10, random_state=seed)
    kmeans.fit(latent)
    return nn.Parameter(
        torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    )


def train_dec_until_stable(
    encoder: nn.Sequential,
    decoder: nn.Sequential,
    centers: nn.Parameter,
    inputs: torch.Tensor,
    device: str,
    log,
    n_initial_clusters: int,
) -> tuple[list[dict[str, float]], str, int]:
    """Train balanced DEC with the original reconstruction weight."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=DEC_LEARNING_RATE,
    )
    inputs = inputs.to(device)
    encoder.eval()
    with torch.no_grad():
        previous_assignments = base.soft_assignments(encoder(inputs), centers).argmax(
            dim=1
        )
    log(
        f"Starting K={n_initial_clusters} joint DEC training "
        f"(up to {MAX_DEC_EPOCHS} full-batch epochs, lr={DEC_LEARNING_RATE:g}, "
        f"stability threshold={STABILITY_THRESHOLD:.3f}, "
        f"balance lambda={LAMBDA_BALANCE:g})..."
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, MAX_DEC_EPOCHS + 1):
        latent = encoder(inputs)
        reconstruction = decoder(latent)
        q = base.soft_assignments(latent, centers)
        p = base.target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean", log_target=False)
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        mean_q = q.mean(dim=0)
        uniform = torch.full_like(mean_q, 1.0 / mean_q.shape[0])
        balance_loss = F.kl_div(
            mean_q.clamp_min(1e-8).log(), uniform, reduction="sum"
        )
        total_loss = (
            kl_loss
            + LAMBDA_RECONSTRUCTION * reconstruction_loss
            + LAMBDA_BALANCE * balance_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        encoder.eval()
        with torch.no_grad():
            current_assignments = base.soft_assignments(
                encoder(inputs), centers
            ).argmax(dim=1)
        num_changed = (current_assignments != previous_assignments).sum().item()
        fraction_changed = num_changed / len(inputs)
        previous_assignments = current_assignments
        encoder.train()
        checkpoint = {
            "epoch": float(epoch),
            "total": total_loss.item(),
            "kl": kl_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "balance": balance_loss.item(),
            "fraction_changed": fraction_changed,
        }
        losses.append(checkpoint)
        stopped_for_stability = fraction_changed < STABILITY_THRESHOLD
        reached_epoch_ceiling = epoch == MAX_DEC_EPOCHS
        stopping = stopped_for_stability or reached_epoch_ceiling
        if epoch % 25 == 0 or stopping:
            log(
                f"K={n_initial_clusters} DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}, "
                f"balance={checkpoint['balance']:.6f}"
            )
        if epoch % 10 == 0 or stopping:
            log(
                f"K={n_initial_clusters} DEC epoch {epoch:03d}/{MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes)"
            )
        if stopping:
            stop_reason = (
                "stability criterion" if stopped_for_stability else "epoch ceiling"
            )
            log(f"Stopping K={n_initial_clusters} DEC at epoch {epoch}: {stop_reason}.")
            return losses, stop_reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


N_SURVIVING_CLUSTERS = 40


# Source: src/test/20260922_percept_topic_pipeline/run_percept_stage2_sweep_pilot.py:61-78; commit 1b70d7f.
class AttentionPoolingMapper(nn.Module):
    """Pool 50 CLIP tokens with one learned query, then score all P-Topics."""

    def __init__(self, d_model: int = 512, n_topics: int = N_SURVIVING_CLUSTERS):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.classifier = nn.Linear(d_model, n_topics)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Return one multi-label logit per frozen topic."""
        query = self.query.expand(patch_tokens.shape[0], -1, -1)
        attn_weights = (
            query @ patch_tokens.transpose(1, 2)
        ) / (patch_tokens.shape[-1] ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        pooled = (attn_weights @ patch_tokens).squeeze(1)
        return self.classifier(pooled)



# Source: src/test/20260922_percept_topic_pipeline/run_percept_stage2_sweep_pilot.py:131-165; commit 1b70d7f.
def evaluate_auc(
    scores: np.ndarray,
    targets: np.ndarray,
    log,
    scorer_name: str,
) -> tuple[dict[int, float], list[int]]:
    """Score valid per-topic held-out AUCs and explicitly log invalid topics."""
    aucs = {}
    skipped = []
    for topic in range(targets.shape[1]):
        topic_targets = targets[:, topic]
        positives = int(topic_targets.sum())
        negatives = len(topic_targets) - positives
        if positives == 0 or negatives == 0:
            log(
                f"Skipping {scorer_name} AUC for topic {topic}: "
                f"{positives} positives and {negatives} negatives in held-out."
            )
            skipped.append(topic)
            continue
        aucs[topic] = float(roc_auc_score(topic_targets, scores[:, topic]))
    if not aucs:
        raise RuntimeError("No held-out topics have both positive and negative labels.")
    return aucs, skipped


def auc_summary(aucs: dict[int, float]) -> dict[str, float]:
    """Return macro/min/median/max aggregate statistics."""
    values = np.asarray(list(aucs.values()))
    return {
        "macro": float(np.mean(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }
