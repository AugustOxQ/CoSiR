"""Bug-fixed re-run of the Stage 2 pilot (macro AUC 0.5690 originally cited
against buddy's 0.5978 in the master report).

Same two bugs as `run_percept_stage1_faithful_recipe_fixed_pilot.py` (see
that file's docstring and
docs/reports/2026-09-27_agy_independent_percept_review.md for the full
derivation against the paper, arXiv:2606.03345):

1. `run_percept_stage1_cluster_count_sweep_pilot.py::prune_centers` (which
   this Stage-2 pilot's own Stage-1 re-fit depends on, both directly and
   via `sweep.evaluate_run`'s internal call) sorted DEC center norms
   DESCENDING and kept the highest-norm centers -- backwards relative to
   the paper's Algorithm 1 (keep LOW-norm, populated centers). Fixed here
   via a module-attribute override (`sweep.prune_centers = ...`), the same
   safe pattern already used elsewhere in this investigation to override a
   bare module-global without editing the shared file.
2. `train_dec_until_stable` (also in the sweep module) used
   `F.mse_loss(reconstruction, inputs)` with the default 'mean' reduction,
   dividing by the 2,816-D feature dimension. Since this loss line is
   local to that function's body (not a rebindable global), this file
   copies that function with only the loss line corrected -- everything
   else (optimizer, schedule, stopping rule, the LAMBDA_BALANCE=1000 term)
   is left exactly as in the original, so this isolates the two verified
   bugs without also removing or changing the balance term.

Original (buggy) file and report are left untouched for provenance:
`run_percept_stage2_pilot.py`, `percept_stage2_pilot_report.md`.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
SWEEP_PILOT_PATH = os.path.join(
    OUT_DIR, "run_percept_stage1_cluster_count_sweep_pilot.py"
)
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "percept_stage2_fixed_pilot_report.md")
PERCEPT_FEATURE_ROOT = os.environ.get("PERCEPT_FEATURE_ROOT", "/data/SSD2/pre_extract")
PATCH_FEATURE_DIR = f"{PERCEPT_FEATURE_ROOT}/artelingo_percept_patch_features"
TRAIN_PATCH_FEATURE_PATH = os.path.join(PATCH_FEATURE_DIR, "train_patch_features.pt")
HELDOUT_PATCH_FEATURE_PATH = os.path.join(
    PATCH_FEATURE_DIR, "heldout_patch_features.pt"
)

N_INITIAL_CLUSTERS = 60
N_SURVIVING_CLUSTERS = 40
LAMBDA_BALANCE = 1000
LAMBDA_RECONSTRUCTION = 1
SEED = 42
MULTI_LABEL_THRESHOLD = 2.0 / N_SURVIVING_CLUSTERS
MAPPER_LEARNING_RATE = 1e-3
MAPPER_EPOCHS = 100
ORIGINAL_HELDOUT_EMOTION_AMI = 0.1238
ORIGINAL_HELDOUT_GENRE_AMI = 0.2617
ORIGINAL_MACRO_AUC = 0.5690
BUDDY_STAGE2_MACRO_AUC = 0.5978


def load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("percept_stage1_base_for_stage2_fixed", BASE_PILOT_PATH)
sweep = load_module("percept_stage1_sweep_for_stage2_fixed", SWEEP_PILOT_PATH)


def prune_centers_fixed(centers: torch.Tensor, n_surviving_clusters: int) -> tuple[torch.Tensor, np.ndarray]:
    """Paper-faithful direction: keep the N LOWEST-norm centers, not highest."""
    norms = torch.linalg.vector_norm(centers.detach(), dim=1)
    surviving = torch.argsort(norms, descending=False)[:n_surviving_clusters]
    return centers.detach()[surviving], surviving.cpu().numpy()


# Module-attribute override: evaluate_run's internal `prune_centers(...)` call
# resolves this bare global at call time, so this also fixes evaluate_run
# without editing the shared file. Verified safe (same pattern as this
# investigation's D_SHARED override): prune_centers is only ever referenced
# as a bare module-global inside sweep's own function bodies.
sweep.prune_centers = prune_centers_fixed


def summed_reconstruction_loss(reconstruction: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Paper-matching L_R = ||h - h_hat||^2: sum over feature dim, mean over batch."""
    return F.mse_loss(reconstruction, inputs, reduction="none").sum(dim=1).mean()


def train_dec_until_stable_fixed(encoder, decoder, centers, inputs, device, log, n_initial_clusters):
    """Copy of sweep.train_dec_until_stable with only the reconstruction-loss
    reduction corrected. Balance and reconstruction weights, optimizer,
    schedule, and stopping rule are otherwise identical."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [centers],
        lr=sweep.DEC_LEARNING_RATE,
    )
    inputs = inputs.to(device)
    encoder.eval()
    with torch.no_grad():
        previous_assignments = base.soft_assignments(encoder(inputs), centers).argmax(dim=1)
    log(
        f"Starting K={n_initial_clusters} joint DEC training (fixed reconstruction "
        f"loss scale) (up to {sweep.MAX_DEC_EPOCHS} full-batch epochs, "
        f"lr={sweep.DEC_LEARNING_RATE:g}, stability threshold="
        f"{sweep.STABILITY_THRESHOLD:.3f}, balance lambda={LAMBDA_BALANCE:g})..."
    )
    losses = []
    encoder.train()
    decoder.train()
    for epoch in range(1, sweep.MAX_DEC_EPOCHS + 1):
        latent = encoder(inputs)
        reconstruction = decoder(latent)
        q = base.soft_assignments(latent, centers)
        p = base.target_distribution(q)
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean", log_target=False)
        reconstruction_loss = summed_reconstruction_loss(reconstruction, inputs)
        mean_q = q.mean(dim=0)
        uniform = torch.full_like(mean_q, 1.0 / mean_q.shape[0])
        balance_loss = F.kl_div(mean_q.clamp_min(1e-8).log(), uniform, reduction="sum")
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
            current_assignments = base.soft_assignments(encoder(inputs), centers).argmax(dim=1)
        num_changed = (current_assignments != previous_assignments).sum().item()
        fraction_changed = num_changed / len(inputs)
        previous_assignments = current_assignments
        encoder.train()
        checkpoint = {
            "epoch": float(epoch), "total": total_loss.item(), "kl": kl_loss.item(),
            "reconstruction": reconstruction_loss.item(), "balance": balance_loss.item(),
            "fraction_changed": fraction_changed,
        }
        losses.append(checkpoint)
        stopped_for_stability = fraction_changed < sweep.STABILITY_THRESHOLD
        reached_epoch_ceiling = epoch == sweep.MAX_DEC_EPOCHS
        stopping = stopped_for_stability or reached_epoch_ceiling
        if epoch % 25 == 0 or stopping:
            log(
                f"K={n_initial_clusters} DEC epoch {epoch:03d}/{sweep.MAX_DEC_EPOCHS}: "
                f"total={checkpoint['total']:.6f}, KL={checkpoint['kl']:.6f}, "
                f"reconstruction={checkpoint['reconstruction']:.6f}, "
                f"balance={checkpoint['balance']:.6f}"
            )
        if epoch % 10 == 0 or stopping:
            log(
                f"K={n_initial_clusters} DEC epoch {epoch:03d}/{sweep.MAX_DEC_EPOCHS}: "
                f"fraction_changed={fraction_changed:.6f} ({num_changed:,} nodes)"
            )
        if stopping:
            stop_reason = "stability criterion" if stopped_for_stability else "epoch ceiling"
            log(f"Stopping K={n_initial_clusters} DEC at epoch {epoch}: {stop_reason}.")
            return losses, stop_reason, epoch
    raise RuntimeError("DEC training exited without a stopping condition.")


class AttentionPoolingMapper(nn.Module):
    def __init__(self, d_model: int = 512, n_topics: int = N_SURVIVING_CLUSTERS):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.classifier = nn.Linear(d_model, n_topics)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(patch_tokens.shape[0], -1, -1)
        attn_weights = (query @ patch_tokens.transpose(1, 2)) / (patch_tokens.shape[-1] ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        pooled = (attn_weights @ patch_tokens).squeeze(1)
        return self.classifier(pooled)


def load_patch_features(path: str, expected_paintings: int, split_name: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing {split_name} patch-feature cache at {path}. "
            "Run run_percept_patch_feature_extraction.py first."
        )
    patch_features = torch.load(path, map_location="cpu")
    if not isinstance(patch_features, torch.Tensor):
        raise RuntimeError(f"{split_name} patch cache is not a tensor: {path}")
    expected_shape = (expected_paintings, 50, 512)
    if tuple(patch_features.shape) != expected_shape:
        raise RuntimeError(
            f"{split_name} patch cache has shape {tuple(patch_features.shape)}; "
            f"expected {expected_shape} in load_dedup_features() painting order."
        )
    if patch_features.dtype != torch.float32:
        raise RuntimeError(f"{split_name} patch cache has dtype {patch_features.dtype}; expected float32.")
    return patch_features


def multi_hot_targets(encoder, centers, inputs, device):
    encoder.eval()
    with torch.no_grad():
        q = base.soft_assignments(encoder(inputs.to(device)), centers)
        targets = torch.zeros_like(q)
        targets.scatter_(1, q.argmax(dim=1, keepdim=True), 1.0)
        targets[q > MULTI_LABEL_THRESHOLD] = 1.0
    return targets.cpu()


def label_statistics(targets: torch.Tensor) -> dict[str, float]:
    counts = targets.sum(dim=1).cpu().numpy()
    return {
        "mean": float(np.mean(counts)), "median": float(np.median(counts)),
        "max": int(np.max(counts)), "fraction_multi_labeled": float(np.mean(counts > 1)),
    }


def evaluate_auc(scores, targets, log, scorer_name):
    aucs = {}
    skipped = []
    for topic in range(targets.shape[1]):
        topic_targets = targets[:, topic]
        positives = int(topic_targets.sum())
        negatives = len(topic_targets) - positives
        if positives == 0 or negatives == 0:
            log(f"Skipping {scorer_name} AUC for topic {topic}: {positives} positives and {negatives} negatives in held-out.")
            skipped.append(topic)
            continue
        aucs[topic] = float(roc_auc_score(topic_targets, scores[:, topic]))
    if not aucs:
        raise RuntimeError("No held-out topics have both positive and negative labels.")
    return aucs, skipped


def auc_summary(aucs):
    values = np.asarray(list(aucs.values()))
    return {"macro": float(np.mean(values)), "min": float(np.min(values)),
            "median": float(np.median(values)), "max": float(np.max(values))}


def write_report(refit_metrics, occupancy, train_target_stats, heldout_target_stats,
                  mapper_losses, model_aucs, baseline_aucs, skipped_topics):
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    model_summary = auc_summary(model_aucs)
    baseline_summary = auc_summary(baseline_aucs)
    improvement = model_summary["macro"] - baseline_summary["macro"]
    meaningful = improvement >= 0.01
    lines = [
        "# ArtELingo PercepT Stage 2 patch-attention pilot (BUG-FIXED re-run)\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "Fixes the same two bugs as "
        "`run_percept_stage1_faithful_recipe_fixed_pilot.py` (center-pruning "
        "direction, reconstruction-loss scale), applied here to the K=60/40 "
        "Stage-1 configuration this Stage-2 pilot actually depends on -- NOT "
        "the K=100/67 faithful-recipe configuration, which is a separate, "
        "unrelated Stage-1 fit. See "
        "[`docs/reports/2026-09-27_agy_independent_percept_review.md`]"
        "(../../../docs/reports/2026-09-27_agy_independent_percept_review.md) "
        "for the full derivation. The LAMBDA_BALANCE=1000 term is left "
        "unchanged from the original, to isolate the effect of the two "
        "verified bugs alone. Original (buggy) file and report are left "
        "unmodified for provenance: `run_percept_stage2_pilot.py`, "
        "`percept_stage2_pilot_report.md`.\n\n",
        "## Stage-1 K=60/40 seed-42 re-fit (fixed pruning + loss scale)\n\n",
        f"Held-out emotion AMI: **{emotion_ami:.4f}** (original buggy citation: "
        f"{ORIGINAL_HELDOUT_EMOTION_AMI:.4f}). Held-out genre AMI: "
        f"**{genre_ami:.4f}** (original buggy citation: "
        f"{ORIGINAL_HELDOUT_GENRE_AMI:.4f}). This does not attempt to "
        "reproduce the original numbers -- they came from the buggy pruning "
        "direction and loss scale, so a different result is expected and is "
        "the point of this re-run.\n\n",
        f"Surviving-center occupancy (of {N_SURVIVING_CLUSTERS}): "
        f"min={occupancy['min']}, max={occupancy['max']}, "
        f"median={occupancy['median']:.1f}, "
        f"below 1%={occupancy['below_one_percent']}/{N_SURVIVING_CLUSTERS}.\n\n",
        "## Frozen multi-label target statistics\n\n",
        "| split | mean labels | median labels | max labels | fraction multi-labeled |\n",
        "|---|---:|---:|---:|---:|\n",
        f"| train | {train_target_stats['mean']:.3f} | {train_target_stats['median']:.3f} | "
        f"{train_target_stats['max']} | {train_target_stats['fraction_multi_labeled']:.2%} |\n",
        f"| held-out | {heldout_target_stats['mean']:.3f} | "
        f"{heldout_target_stats['median']:.3f} | {heldout_target_stats['max']} | "
        f"{heldout_target_stats['fraction_multi_labeled']:.2%} |\n\n",
        "## Held-out macro AUC\n\n",
        "| scorer | macro AUC | min | median | max | skipped topics |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| patch-attention mapper (fixed Stage 1) | {model_summary['macro']:.4f} | "
        f"{model_summary['min']:.4f} | {model_summary['median']:.4f} | "
        f"{model_summary['max']:.4f} | {len(skipped_topics)} |\n",
        f"| train-marginal baseline | {baseline_summary['macro']:.4f} | "
        f"{baseline_summary['min']:.4f} | {baseline_summary['median']:.4f} | "
        f"{baseline_summary['max']:.4f} | {len(skipped_topics)} |\n\n",
        "## Comparison\n\n",
        f"- Original buggy PercepT Stage 2 macro AUC: {ORIGINAL_MACRO_AUC:.4f}\n",
        f"- Fixed PercepT Stage 2 macro AUC: **{model_summary['macro']:.4f}**\n",
        f"- Buddy's own Stage 2 macro AUC: {BUDDY_STAGE2_MACRO_AUC:.4f}\n",
        (
            f"- **Fixed PercepT ({model_summary['macro']:.4f}) beats buddy "
            f"({BUDDY_STAGE2_MACRO_AUC:.4f})**\n"
            if model_summary["macro"] > BUDDY_STAGE2_MACRO_AUC else
            f"- **Buddy ({BUDDY_STAGE2_MACRO_AUC:.4f}) still beats fixed PercepT "
            f"({model_summary['macro']:.4f})**\n"
        ),
        (
            "**The image-only attention-pooling mapper meaningfully beats the "
            "marginal-frequency baseline.** "
            f"Its macro AUC is higher by {improvement:.4f}, exceeding the "
            "predeclared 0.01 practical margin.\n"
            if meaningful else
            "**The image-only attention-pooling mapper does not meaningfully "
            f"beat the marginal-frequency baseline.** Its macro-AUC difference "
            f"is {improvement:.4f}, below the 0.01 practical margin.\n"
        ),
    ]
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pipeline = base.load_sibling_module("artelingo_run_pipeline_percept_stage2_fixed_train", base.PIPELINE_PATH)
    affect_pilot = base.load_sibling_module("artelingo_run_affect_pilot_percept_stage2_fixed", base.AFFECT_PILOT_PATH)
    cca_audit = base.load_sibling_module("artelingo_run_cca_audit_percept_stage2_fixed", base.CCA_AUDIT_PATH)
    heldout_pipeline = base.load_sibling_module("artelingo_run_pipeline_percept_stage2_fixed_heldout", base.PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    log("Loading and deduplicating train CLIP features for Stage-1 re-fit...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log("Loading and deduplicating held-out CLIP features for Stage-1 re-fit...")
    heldout_paintings, heldout_img_nodes, heldout_txt_nodes, heldout_emotion_counts = heldout_pipeline.load_dedup_features()
    heldout_emotions = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for Stage-1 re-fit (fixed) and Stage-2 mapper training.")
    affect_train = base.extract_affect_embedding_nodes(pipeline.TRAIN_JSON, paintings, device, log)
    affect_heldout = base.extract_affect_embedding_nodes(base.HELDOUT_JSON, heldout_paintings, device, log)
    train_h = base.fused_embeddings(img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot)
    heldout_h = base.fused_embeddings(heldout_img_nodes, heldout_txt_nodes, affect_heldout, cca_audit, affect_pilot)
    if train_h.shape[1] != heldout_h.shape[1]:
        raise RuntimeError(f"Train/held-out fused dimensions differ: {train_h.shape[1]} vs {heldout_h.shape[1]}.")

    train_inputs = torch.from_numpy(train_h)
    encoder, decoder = base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
    centers = sweep.initialize_cluster_centers(encoder, train_inputs, device, N_INITIAL_CLUSTERS, SEED)
    train_dec_until_stable_fixed(encoder, decoder, centers, train_inputs, device, log, N_INITIAL_CLUSTERS)
    surviving_centers, surviving_indices = prune_centers_fixed(centers, N_SURVIVING_CLUSTERS)
    log(
        f"Pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers (fixed direction: "
        f"lowest-norm kept); retained original indices {surviving_indices.tolist()}."
    )
    _, heldout_refit_metrics, _ = sweep.evaluate_run(
        encoder, centers, train_inputs, heldout_h, pipeline, heldout_pipeline,
        paintings, train_emotions, heldout_paintings, heldout_emotions, device,
        N_SURVIVING_CLUSTERS,
    )
    occupancy = {
        "min": int(heldout_refit_metrics["cluster_sizes"].min()),
        "max": int(heldout_refit_metrics["cluster_sizes"].max()),
        "median": float(np.median(heldout_refit_metrics["cluster_sizes"])),
        "below_one_percent": heldout_refit_metrics["small_clusters"],
    }
    log(
        f"Fixed Stage-1 re-fit: held-out emotion AMI={heldout_refit_metrics['emotion']['AMI']:.4f}, "
        f"genre AMI={heldout_refit_metrics['genre']['AMI']:.4f}, "
        f"below_1pct={occupancy['below_one_percent']}/{N_SURVIVING_CLUSTERS}."
    )

    train_targets = multi_hot_targets(encoder, surviving_centers, train_inputs, device)
    heldout_targets = multi_hot_targets(encoder, surviving_centers, torch.from_numpy(heldout_h), device)
    train_target_stats = label_statistics(train_targets)
    heldout_target_stats = label_statistics(heldout_targets)
    log(
        f"Train label counts: mean={train_target_stats['mean']:.3f}, "
        f"multi={train_target_stats['fraction_multi_labeled']:.2%}."
    )
    log(
        f"Held-out label counts: mean={heldout_target_stats['mean']:.3f}, "
        f"multi={heldout_target_stats['fraction_multi_labeled']:.2%}."
    )

    train_patch_features = load_patch_features(TRAIN_PATCH_FEATURE_PATH, len(paintings), "train")
    heldout_patch_features = load_patch_features(HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out")
    mapper = AttentionPoolingMapper().to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=MAPPER_LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    train_patch_features = train_patch_features.to(device)
    train_targets = train_targets.to(device)
    mapper_losses = []
    mapper.train()
    for epoch in range(1, MAPPER_EPOCHS + 1):
        loss = loss_fn(mapper(train_patch_features), train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mapper_losses.append(loss.item())
        if epoch % 10 == 0:
            log(f"Stage-2 epoch {epoch:03d}/{MAPPER_EPOCHS}: mean BCE={loss.item():.6f}")

    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patch_features.to(device))).cpu().numpy()
    heldout_targets_np = heldout_targets.cpu().numpy()
    model_aucs, skipped_topics = evaluate_auc(heldout_scores, heldout_targets_np, log, "patch-attention mapper")
    train_marginals = train_targets.float().mean(dim=0).cpu().numpy()
    baseline_scores = np.broadcast_to(train_marginals, heldout_targets_np.shape)
    baseline_aucs, baseline_skipped_topics = evaluate_auc(baseline_scores, heldout_targets_np, log, "train-marginal baseline")
    if skipped_topics != baseline_skipped_topics:
        raise RuntimeError("AUC validity differs between model and identical-target baseline.")
    write_report(
        heldout_refit_metrics, occupancy, train_target_stats, heldout_target_stats,
        mapper_losses, model_aucs, baseline_aucs, skipped_topics,
    )
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
