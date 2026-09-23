"""Stress-test PercepT Stage 2 mapper seeds, target thresholds, and learning rates.

This standalone GPU script deterministically re-fits the fixed K=60/40
Stage-1 configuration once, freezes its encoder and surviving centers, then
reuses their targets and cached CLIP patch features for every Stage-2 run.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


OUT_DIR = os.path.dirname(__file__)
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
SWEEP_PILOT_PATH = os.path.join(
    OUT_DIR, "run_percept_stage1_cluster_count_sweep_pilot.py"
)
REPORT_PATH = os.path.join(OUT_DIR, "percept_stage2_sweep_pilot_report.md")
PATCH_FEATURE_DIR = "/data/SSD2/pre_extract/artelingo_percept_patch_features"
TRAIN_PATCH_FEATURE_PATH = os.path.join(PATCH_FEATURE_DIR, "train_patch_features.pt")
HELDOUT_PATCH_FEATURE_PATH = os.path.join(
    PATCH_FEATURE_DIR, "heldout_patch_features.pt"
)

N_INITIAL_CLUSTERS = 60
N_SURVIVING_CLUSTERS = 40
SEED = 42
SEEDS = (42, 7, 123, 2024)
THRESHOLD_MULTIPLIERS = (2.0, 1.5, 1.2)
MAPPER_LEARNING_RATE = 1e-3
MAPPER_EPOCHS = 100
LEARNING_RATES = (3e-4, 1e-3, 3e-3)
EXPECTED_HELDOUT_EMOTION_AMI = 0.1238
EXPECTED_HELDOUT_GENRE_AMI = 0.2617
REPRODUCTION_TOLERANCE = 0.002
PRACTICAL_AUC_MARGIN = 0.01
BASELINE_MACRO_AUC = 0.5000


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("percept_stage1_base_for_stage2_sweep", BASE_PILOT_PATH)
sweep = load_module("percept_stage1_sweep_for_stage2_sweep", SWEEP_PILOT_PATH)


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


def load_patch_features(path: str, expected_paintings: int, split_name: str) -> torch.Tensor:
    """Load and validate the Script-1 cache aligned to deduplicated paintings."""
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
        raise RuntimeError(
            f"{split_name} patch cache has dtype {patch_features.dtype}; expected float32."
        )
    return patch_features


def multi_hot_targets(
    encoder: nn.Module,
    centers: torch.Tensor,
    inputs: torch.Tensor,
    device: str,
    threshold: float,
) -> torch.Tensor:
    """Freeze argmax-plus-threshold DEC assignments as multi-label targets."""
    encoder.eval()
    with torch.no_grad():
        q = base.soft_assignments(encoder(inputs.to(device)), centers)
        targets = torch.zeros_like(q)
        targets.scatter_(1, q.argmax(dim=1, keepdim=True), 1.0)
        targets[q > threshold] = 1.0
    return targets.cpu()


def label_statistics(targets: torch.Tensor) -> dict[str, float]:
    """Summarize labels per painting for the required report."""
    counts = targets.sum(dim=1).cpu().numpy()
    return {
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "max": int(np.max(counts)),
        "fraction_multi_labeled": float(np.mean(counts > 1)),
    }


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


def threshold_label(multiplier: float) -> str:
    """Format a threshold as the explainable multiple of uniform probability."""
    return f"{multiplier:.1f}/{N_SURVIVING_CLUSTERS}"


def train_and_evaluate_mapper(
    train_patch_features: torch.Tensor,
    heldout_patch_features: torch.Tensor,
    train_targets: torch.Tensor,
    heldout_targets: torch.Tensor,
    mapper_seed: int,
    learning_rate: float,
    device: str,
    log,
    run_name: str,
) -> dict:
    """Train one full-batch mapper draw and score its frozen held-out targets."""
    # This must be immediately before mapper construction: Stage-1 remains fixed.
    torch.manual_seed(mapper_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mapper_seed)
    mapper = AttentionPoolingMapper().to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    mapper.train()
    for epoch in range(1, MAPPER_EPOCHS + 1):
        loss = loss_fn(mapper(train_patch_features), train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            log(
                f"{run_name} epoch {epoch:03d}/{MAPPER_EPOCHS}: "
                f"mean BCE={loss.item():.6f}"
            )

    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patch_features)).cpu().numpy()
    heldout_targets_np = heldout_targets.cpu().numpy()
    model_aucs, skipped_topics = evaluate_auc(
        heldout_scores, heldout_targets_np, log, run_name
    )
    train_marginals = train_targets.float().mean(dim=0).cpu().numpy()
    baseline_scores = np.broadcast_to(train_marginals, heldout_targets_np.shape)
    baseline_aucs, baseline_skipped_topics = evaluate_auc(
        baseline_scores, heldout_targets_np, log, f"{run_name} train-marginal baseline"
    )
    if skipped_topics != baseline_skipped_topics:
        raise RuntimeError("AUC validity differs between model and identical-target baseline.")
    return {
        "losses": losses,
        "aucs": model_aucs,
        "summary": auc_summary(model_aucs),
        "baseline_summary": auc_summary(baseline_aucs),
        "skipped_topics": skipped_topics,
    }


def select_part_c_threshold(
    threshold_results: dict[float, dict],
    target_statistics: dict[float, dict],
) -> float:
    """Prefer a genuinely multi-label threshold unless it is materially worse."""
    original = threshold_results[2.0]["summary"]["macro"]
    viable = [
        multiplier
        for multiplier in (1.5, 1.2)
        if target_statistics[multiplier]["train"]["fraction_multi_labeled"] > 0
        and target_statistics[multiplier]["heldout"]["fraction_multi_labeled"] > 0
        and threshold_results[multiplier]["summary"]["macro"]
        >= original - PRACTICAL_AUC_MARGIN
    ]
    if not viable:
        return 2.0
    return max(viable, key=lambda multiplier: threshold_results[multiplier]["summary"]["macro"])


def write_report(
    refit_metrics: dict,
    reproduced: bool,
    part_a: dict[int, dict] | None = None,
    target_statistics: dict[float, dict] | None = None,
    threshold_results: dict[float, dict] | None = None,
    selected_threshold: float | None = None,
    part_c: dict[float, dict] | None = None,
) -> None:
    """Write the reproducibility gate or complete three-part sweep report."""
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    lines = [
        "# ArtELingo PercepT Stage 2 sweep pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Shared Stage-1 K=60/40 seed-42 reproduction\n\n",
        "Stage 1 was re-fit exactly once, before every frozen target and mapper "
        "variant below. All Stage-2 runs share that one frozen encoder, 40 surviving "
        "centers, and cached patch features.\n\n",
        "| metric | established | re-fit | absolute difference | status |\n",
        "|---|---:|---:|---:|---|\n",
        f"| held-out emotion AMI | {EXPECTED_HELDOUT_EMOTION_AMI:.4f} | "
        f"{emotion_ami:.4f} | {abs(emotion_ami - EXPECTED_HELDOUT_EMOTION_AMI):.4f} | "
        f"{'reproduced' if reproduced else 'FAILED'} |\n",
        f"| held-out genre AMI | {EXPECTED_HELDOUT_GENRE_AMI:.4f} | "
        f"{genre_ami:.4f} | {abs(genre_ami - EXPECTED_HELDOUT_GENRE_AMI):.4f} | "
        f"{'reproduced' if reproduced else 'FAILED'} |\n\n",
    ]
    if not reproduced:
        lines.append(
            "**Reproducibility failure.** The shared Stage-1 re-fit was not within "
            f"the absolute AMI tolerance of {REPRODUCTION_TOLERANCE:.3f}; no Stage-2 "
            "mapper was trained, so the sweep cannot silently use a different clustering.\n"
        )
        with open(REPORT_PATH, "w") as report_file:
            report_file.writelines(lines)
        return

    assert part_a is not None
    assert target_statistics is not None
    assert threshold_results is not None
    assert selected_threshold is not None
    assert part_c is not None
    seed_macros = [part_a[seed]["summary"]["macro"] for seed in SEEDS]
    robust = all(macro > BASELINE_MACRO_AUC for macro in seed_macros)
    lines.extend([
        "## Part A — mapper-init seed stress test\n\n",
        "Each row uses the original `q > 2.0/40` target threshold, `lr=1e-3`, and "
        "100 epochs. Seed 42 is a new mapper-initialization draw, not a citation of "
        "the earlier smoke-test number.\n\n",
        "| mapper-init seed | held-out macro AUC | min per-topic AUC | median per-topic AUC | max per-topic AUC | skipped topics |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ])
    lines.extend(
        f"| {seed} | {part_a[seed]['summary']['macro']:.4f} | "
        f"{part_a[seed]['summary']['min']:.4f} | "
        f"{part_a[seed]['summary']['median']:.4f} | "
        f"{part_a[seed]['summary']['max']:.4f} | "
        f"{len(part_a[seed]['skipped_topics'])} |\n"
        for seed in SEEDS
    )
    lines.extend([
        "\n| seed-summary statistic | held-out macro AUC |\n",
        "|---|---:|\n",
        f"| mean | {np.mean(seed_macros):.4f} |\n",
        f"| min | {np.min(seed_macros):.4f} |\n",
        f"| max | {np.max(seed_macros):.4f} |\n\n",
        (
            "**Verdict:** The baseline-beating result is seed-robust across these four "
            "mapper initializations: every held-out macro AUC exceeds the 0.5000 "
            "train-marginal baseline.\n\n"
            if robust
            else "**Verdict:** The baseline-beating result is seed-dependent: at least "
            "one of the four mapper initializations does not exceed the 0.5000 "
            "train-marginal baseline.\n\n"
        ),
        "## Part B — multi-label threshold comparison\n\n",
        "All targets below were derived from the single shared frozen Stage-1 fit. "
        "The `2.0/40` AUC cites Part A's seed-42 run and was not retrained.\n\n",
        "| threshold | train mean / median / max labels | train fraction multi-labeled | held-out mean / median / max labels | held-out fraction multi-labeled | held-out macro AUC |\n",
        "|---|---|---:|---|---:|---:|\n",
    ])
    for multiplier in THRESHOLD_MULTIPLIERS:
        train_stats = target_statistics[multiplier]["train"]
        heldout_stats = target_statistics[multiplier]["heldout"]
        lines.append(
            f"| q > {threshold_label(multiplier)} | "
            f"{train_stats['mean']:.3f} / {train_stats['median']:.3f} / {train_stats['max']} | "
            f"{train_stats['fraction_multi_labeled']:.2%} | "
            f"{heldout_stats['mean']:.3f} / {heldout_stats['median']:.3f} / {heldout_stats['max']} | "
            f"{heldout_stats['fraction_multi_labeled']:.2%} | "
            f"{threshold_results[multiplier]['summary']['macro']:.4f} |\n"
        )
    genuine_thresholds = [
        threshold_label(multiplier)
        for multiplier in THRESHOLD_MULTIPLIERS
        if target_statistics[multiplier]["train"]["fraction_multi_labeled"] > 0
        or target_statistics[multiplier]["heldout"]["fraction_multi_labeled"] > 0
    ]
    if genuine_thresholds:
        lines.append(
            "\n**Multi-label result:** Genuine multi-label targets occur at "
            f"{', '.join(genuine_thresholds)} (at least one split has nonzero "
            "multi-labeled paintings). The table shows whether their macro-AUC change "
            "is material relative to the original threshold.\n\n"
        )
    else:
        lines.append(
            "\n**Multi-label result:** None of the tested thresholds produced genuine "
            "multi-label targets in either split; Stage 2 remains effectively "
            "single-label for this frozen fit.\n\n"
        )

    selected_result = threshold_results[selected_threshold]
    lines.extend([
        "## Part C — learning-rate mini-sweep\n\n",
        f"The selected threshold is `q > {threshold_label(selected_threshold)}`. "
        "Selection prefers genuine multi-label targets whose macro AUC is within the "
        f"predeclared {PRACTICAL_AUC_MARGIN:.2f} practical margin of the original "
        "threshold; otherwise it retains `q > 2.0/40`. The `1e-3` result is cited "
        "from Part A or Part B and was not retrained.\n\n",
        "| learning rate | held-out macro AUC | source |\n",
        "|---:|---:|---|\n",
    ])
    for learning_rate in LEARNING_RATES:
        if learning_rate == MAPPER_LEARNING_RATE:
            source = "Part A seed-42" if selected_threshold == 2.0 else "Part B"
            result = selected_result
        else:
            source = "new Part C run"
            result = part_c[learning_rate]
        lines.append(
            f"| {learning_rate:.0e} | {result['summary']['macro']:.4f} | {source} |\n"
        )

    all_configurations = [
        (part_a[seed]["summary"]["macro"], seed, 2.0, MAPPER_LEARNING_RATE)
        for seed in SEEDS
    ]
    all_configurations.extend(
        (threshold_results[multiplier]["summary"]["macro"], 42, multiplier, MAPPER_LEARNING_RATE)
        for multiplier in (1.5, 1.2)
    )
    all_configurations.extend(
        (part_c[learning_rate]["summary"]["macro"], 42, selected_threshold, learning_rate)
        for learning_rate in LEARNING_RATES
        if learning_rate != MAPPER_LEARNING_RATE
    )
    best_macro, best_seed, best_threshold, best_lr = max(all_configurations)
    best_margin = best_macro - BASELINE_MACRO_AUC
    robustness = "robust across the four original-threshold seeds" if robust else "still provisional because the original-threshold seed test was seed-dependent"
    lines.extend([
        "\n## Recommendation\n\n",
        "The single best observed Stage-2 configuration is mapper-init seed "
        f"{best_seed}, `q > {threshold_label(best_threshold)}`, `lr={best_lr:.0e}`, "
        f"and 100 epochs, with held-out macro AUC {best_macro:.4f}. Its improvement "
        f"over the 0.5000 train-marginal baseline is {best_margin:.4f}. This is "
        f"{robust}; the selected maximum should nevertheless be interpreted as a "
        "sweep result rather than as an independently replicated estimate.\n",
    ])
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Re-fit Stage 1 once, then run all specified Stage-2 sweep points."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_stage2_sweep_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_percept_stage2_sweep", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_percept_stage2_sweep", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_percept_stage2_sweep_heldout", base.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    log("Loading and deduplicating train CLIP features for the shared Stage-1 re-fit...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log("Loading and deduplicating held-out CLIP features for the shared Stage-1 re-fit...")
    (
        heldout_paintings,
        heldout_img_nodes,
        heldout_txt_nodes,
        heldout_emotion_counts,
    ) = heldout_pipeline.load_dedup_features()
    heldout_emotions = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for the shared Stage-1 re-fit and Stage-2 mapper sweep.")
    affect_train = base.extract_affect_embedding_nodes(
        pipeline.TRAIN_JSON, paintings, device, log
    )
    affect_heldout = base.extract_affect_embedding_nodes(
        base.HELDOUT_JSON, heldout_paintings, device, log
    )
    train_h = base.fused_embeddings(
        img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot
    )
    heldout_h = base.fused_embeddings(
        heldout_img_nodes,
        heldout_txt_nodes,
        affect_heldout,
        cca_audit,
        affect_pilot,
    )
    if train_h.shape[1] != heldout_h.shape[1]:
        raise RuntimeError(
            f"Train/held-out fused dimensions differ: {train_h.shape[1]} vs "
            f"{heldout_h.shape[1]}."
        )

    train_inputs = torch.from_numpy(train_h)
    encoder, decoder = base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
    centers = sweep.initialize_cluster_centers(
        encoder, train_inputs, device, N_INITIAL_CLUSTERS, SEED
    )
    sweep.train_dec_until_stable(
        encoder,
        decoder,
        centers,
        train_inputs,
        device,
        log,
        N_INITIAL_CLUSTERS,
    )
    surviving_centers, surviving_indices = sweep.prune_centers(
        centers, N_SURVIVING_CLUSTERS
    )
    log(
        f"Pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers; retained "
        f"original indices {surviving_indices.tolist()}."
    )
    _, heldout_refit_metrics, _ = sweep.evaluate_run(
        encoder,
        centers,
        train_inputs,
        heldout_h,
        pipeline,
        heldout_pipeline,
        paintings,
        train_emotions,
        heldout_paintings,
        heldout_emotions,
        device,
        N_SURVIVING_CLUSTERS,
    )
    reproduced = (
        abs(heldout_refit_metrics["emotion"]["AMI"] - EXPECTED_HELDOUT_EMOTION_AMI)
        < REPRODUCTION_TOLERANCE
        and abs(heldout_refit_metrics["genre"]["AMI"] - EXPECTED_HELDOUT_GENRE_AMI)
        < REPRODUCTION_TOLERANCE
    )
    if not reproduced:
        write_report(heldout_refit_metrics, reproduced=False)
        log(f"Stage-1 reproducibility failure; wrote report to {REPORT_PATH}.")
        return

    target_sets = {}
    target_statistics = {}
    heldout_inputs = torch.from_numpy(heldout_h)
    for multiplier in THRESHOLD_MULTIPLIERS:
        threshold = multiplier / N_SURVIVING_CLUSTERS
        train_targets = multi_hot_targets(
            encoder, surviving_centers, train_inputs, device, threshold
        )
        heldout_targets = multi_hot_targets(
            encoder, surviving_centers, heldout_inputs, device, threshold
        )
        target_sets[multiplier] = (train_targets, heldout_targets)
        target_statistics[multiplier] = {
            "train": label_statistics(train_targets),
            "heldout": label_statistics(heldout_targets),
        }
        log(
            f"q > {threshold_label(multiplier)} targets: train "
            f"multi={target_statistics[multiplier]['train']['fraction_multi_labeled']:.2%}; "
            f"held-out multi="
            f"{target_statistics[multiplier]['heldout']['fraction_multi_labeled']:.2%}."
        )

    train_patch_features = load_patch_features(
        TRAIN_PATCH_FEATURE_PATH, len(paintings), "train"
    ).to(device)
    heldout_patch_features = load_patch_features(
        HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out"
    ).to(device)

    original_train_targets, original_heldout_targets = target_sets[2.0]
    original_train_targets = original_train_targets.to(device)
    original_heldout_targets = original_heldout_targets.to(device)
    part_a = {}
    for mapper_seed in SEEDS:
        part_a[mapper_seed] = train_and_evaluate_mapper(
            train_patch_features,
            heldout_patch_features,
            original_train_targets,
            original_heldout_targets,
            mapper_seed,
            MAPPER_LEARNING_RATE,
            device,
            log,
            f"Part A seed {mapper_seed}",
        )

    threshold_results = {2.0: part_a[42]}
    for multiplier in (1.5, 1.2):
        train_targets, heldout_targets = target_sets[multiplier]
        threshold_results[multiplier] = train_and_evaluate_mapper(
            train_patch_features,
            heldout_patch_features,
            train_targets.to(device),
            heldout_targets.to(device),
            SEED,
            MAPPER_LEARNING_RATE,
            device,
            log,
            f"Part B q > {threshold_label(multiplier)}",
        )

    selected_threshold = select_part_c_threshold(threshold_results, target_statistics)
    log(f"Part C selected q > {threshold_label(selected_threshold)}.")
    selected_train_targets, selected_heldout_targets = target_sets[selected_threshold]
    part_c = {}
    for learning_rate in LEARNING_RATES:
        if learning_rate == MAPPER_LEARNING_RATE:
            continue
        part_c[learning_rate] = train_and_evaluate_mapper(
            train_patch_features,
            heldout_patch_features,
            selected_train_targets.to(device),
            selected_heldout_targets.to(device),
            SEED,
            learning_rate,
            device,
            log,
            f"Part C q > {threshold_label(selected_threshold)} lr={learning_rate:.0e}",
        )

    write_report(
        heldout_refit_metrics,
        reproduced=True,
        part_a=part_a,
        target_statistics=target_statistics,
        threshold_results=threshold_results,
        selected_threshold=selected_threshold,
        part_c=part_c,
    )
    log(f"Wrote report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
