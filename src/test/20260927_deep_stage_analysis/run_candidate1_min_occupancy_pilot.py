"""Candidate 1: minimum-occupancy handling for buddy's Stage 2 targets.

Motivated by deep_stage_analysis_report.md's Finding A: buddy's Stage 2
macro AUC is significantly correlated with per-topic occupancy (Pearson
r=0.531, p=0.019). Train-side occupancy is much healthier than held-out
(only 1/19 train communities fall below the established 1%-of-train
threshold: community 18 at 430/61,402), so rather than apply that generic
threshold (which would only catch one of the three topics the deep
analysis already flagged as problematic -- held-out topics 16/17/18,
sizes 87/69/65), this pilot directly targets the three smallest-by-
train-size communities (16: 1,235; 17: 708; 18: 430), which are exactly
the ones the prior analysis implicated.

Two variants against the reproduced 0.5978 baseline:
  A. Class-balanced (inverse-train-frequency) per-sample loss weighting,
     same 19 topics, no merging.
  B. Merge the three smallest train communities into their nearest larger
     neighbor (by L2-normalized centroid cosine similarity), producing a
     16-topic vocabulary; held-out labels re-derived via the same k=20
     transfer onto this merged vocabulary.

Reuses (imports, does not modify): buddy's frozen snapshot,
run_heldout_label_transfer_pilot.py::assign_to_train_communities,
run_percept_stage2_pilot.py's AttentionPoolingMapper/load_patch_features/
evaluate_auc/auc_summary.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn


HERE = Path(__file__).resolve().parent
PERCEPT_DIR = HERE.parents[1] / "test/20260922_percept_topic_pipeline"
BUDDY_DIR = HERE.parents[1] / "test/20260923_artelingo_buddy_analysis"
SNAPSHOT = BUDDY_DIR / "attention_h1_embedding_snapshot.npz"
REPORT_PATH = HERE / "candidate1_min_occupancy_pilot_report.md"

SEED = 42
N_TOPICS = 19
TRANSFER_K = 20
BASELINE_MACRO_AUC = 0.5978
PRACTICAL_MARGIN = 0.005
SMALL_COMMUNITIES = (16, 17, 18)  # smallest by train size; see module docstring


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_hot(labels: np.ndarray, n_topics: int) -> torch.Tensor:
    targets = torch.zeros((len(labels), n_topics), dtype=torch.float32)
    targets.scatter_(1, torch.as_tensor(labels, dtype=torch.long)[:, None], 1.0)
    return targets


def merge_small_communities(
    train_embedding: np.ndarray, train_labels: np.ndarray, small: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, int]]:
    """Relabel each small community's members to its nearest larger
    neighbor's label (by L2-normalized centroid cosine similarity), then
    compact the remaining labels to a contiguous 0..K-1 range."""
    large = [t for t in range(train_labels.max() + 1) if t not in small]
    centroids = {}
    for topic in range(train_labels.max() + 1):
        members = train_embedding[train_labels == topic]
        centroid = members.mean(axis=0)
        centroids[topic] = centroid / max(np.linalg.norm(centroid), 1e-12)

    merge_target = {}
    for topic in small:
        sims = {t: float(centroids[topic] @ centroids[t]) for t in large}
        merge_target[topic] = max(sims, key=sims.get)
        log(f"Merging small train community {topic} (n={int((train_labels == topic).sum())}) "
            f"into {merge_target[topic]} (cosine sim={sims[merge_target[topic]]:.4f})")

    relabeled = train_labels.copy()
    for topic, target in merge_target.items():
        relabeled[train_labels == topic] = target

    remaining = sorted(set(relabeled.tolist()))
    compact_map = {old: new for new, old in enumerate(remaining)}
    compacted = np.array([compact_map[label] for label in relabeled], dtype=np.int64)
    full_map = dict(compact_map)
    for topic, target in merge_target.items():
        full_map[topic] = compact_map[target]
    return compacted, full_map


def train_and_eval(
    name: str, n_topics: int, train_labels: np.ndarray, heldout_labels: np.ndarray,
    train_patches: torch.Tensor, heldout_patches: torch.Tensor, stage2_ref, device: str,
    sample_weights: np.ndarray | None = None,
) -> dict:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    train_targets = one_hot(train_labels, n_topics).to(device)
    heldout_targets = one_hot(heldout_labels, n_topics).cpu().numpy()

    mapper = stage2_ref.AttentionPoolingMapper(n_topics=n_topics).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=stage2_ref.MAPPER_LEARNING_RATE)
    weights_tensor = (
        torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
        if sample_weights is not None else None
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mapper.train()
    for epoch in range(1, stage2_ref.MAPPER_EPOCHS + 1):
        logits = mapper(train_patches)
        per_sample_loss = bce(logits, train_targets).mean(dim=1)
        loss = (
            (per_sample_loss * weights_tensor).mean()
            if weights_tensor is not None else per_sample_loss.mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            log(f"{name}: epoch {epoch}/{stage2_ref.MAPPER_EPOCHS}: loss={loss.item():.6f}")

    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patches)).cpu().numpy()
    model_aucs, skipped = stage2_ref.evaluate_auc(heldout_scores, heldout_targets, log, name)
    summary = stage2_ref.auc_summary(model_aucs)

    predicted = heldout_scores.argmax(axis=1)
    top1_acc = float((predicted == heldout_labels).mean())

    occupancy = np.bincount(heldout_labels, minlength=n_topics)
    per_topic_auc = np.full(n_topics, np.nan)
    for topic, auc in model_aucs.items():
        per_topic_auc[topic] = auc
    valid = ~np.isnan(per_topic_auc)
    if valid.sum() >= 3:
        pear_r, pear_p = pearsonr(occupancy[valid].astype(float), per_topic_auc[valid])
        spear_r, spear_p = spearmanr(occupancy[valid].astype(float), per_topic_auc[valid])
    else:
        pear_r = pear_p = spear_r = spear_p = float("nan")

    return {
        "name": name, "n_topics": n_topics, "macro": summary["macro"],
        "min": summary["min"], "median": summary["median"], "max": summary["max"],
        "top1_acc": top1_acc, "skipped": len(skipped),
        "pearson_r": pear_r, "pearson_p": pear_p, "spearman_r": spear_r, "spearman_p": spear_p,
    }


def main() -> None:
    stage2_ref = load_module("stage2_ref_for_candidate1", PERCEPT_DIR / "run_percept_stage2_pilot.py")
    transfer = load_module("transfer_for_candidate1", BUDDY_DIR / "run_heldout_label_transfer_pilot.py")

    with np.load(SNAPSHOT, allow_pickle=True) as source:
        train_paintings = source["train_paintings"]
        train_embedding = source["train_embedding_post"]
        train_labels = source["train_community_post"].astype(np.int64)
        heldout_paintings = source["heldout_paintings"]
        heldout_embedding = source["heldout_embedding_post"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_patches = stage2_ref.load_patch_features(
        stage2_ref.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train"
    ).to(device)
    heldout_patches = stage2_ref.load_patch_features(
        stage2_ref.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out"
    ).to(device)

    baseline_heldout_labels = transfer.assign_to_train_communities(
        train_embedding, train_labels, heldout_embedding, k=TRANSFER_K
    ).astype(np.int64)

    log("=== Baseline (reproduction check) ===")
    baseline = train_and_eval(
        "baseline", N_TOPICS, train_labels, baseline_heldout_labels,
        train_patches, heldout_patches, stage2_ref, device,
    )
    log(f"Baseline macro AUC: {baseline['macro']:.4f} (expected {BASELINE_MACRO_AUC:.4f})")
    sanity_ok = abs(baseline["macro"] - BASELINE_MACRO_AUC) <= 0.001
    if not sanity_ok:
        REPORT_PATH.write_text(
            f"# Candidate 1 sanity check FAILED\n\nBaseline macro AUC "
            f"{baseline['macro']:.4f} does not match expected "
            f"{BASELINE_MACRO_AUC:.4f} within 0.001. Stopping before "
            "testing variants.\n", encoding="utf-8",
        )
        log("SANITY CHECK FAILED. Stopping.")
        raise SystemExit(1)

    log("=== Variant A: class-balanced loss weighting ===")
    train_counts = np.bincount(train_labels, minlength=N_TOPICS)
    inverse_freq = (len(train_labels) / N_TOPICS) / train_counts[train_labels]
    inverse_freq = inverse_freq / inverse_freq.mean()  # keep mean weight at 1
    variant_a = train_and_eval(
        "variant_a", N_TOPICS, train_labels, baseline_heldout_labels,
        train_patches, heldout_patches, stage2_ref, device, sample_weights=inverse_freq,
    )
    log(f"Variant A macro AUC: {variant_a['macro']:.4f}")

    log("=== Variant B: merge small communities ===")
    merged_train_labels, label_map = merge_small_communities(train_embedding, train_labels, SMALL_COMMUNITIES)
    n_merged_topics = len(set(label_map.values()))
    merged_heldout_labels_raw = transfer.assign_to_train_communities(
        train_embedding, train_labels, heldout_embedding, k=TRANSFER_K
    ).astype(np.int64)
    merged_heldout_labels = np.array([label_map[label] for label in merged_heldout_labels_raw], dtype=np.int64)
    variant_b = train_and_eval(
        "variant_b", n_merged_topics, merged_train_labels, merged_heldout_labels,
        train_patches, heldout_patches, stage2_ref, device,
    )
    log(f"Variant B macro AUC: {variant_b['macro']:.4f} (K={n_merged_topics})")

    write_report(baseline, variant_a, variant_b, label_map)
    log(f"Wrote {REPORT_PATH}")


def write_report(baseline: dict, variant_a: dict, variant_b: dict, label_map: dict) -> None:
    def row(result: dict) -> str:
        return (
            f"| {result['name']} | {result['n_topics']} | {result['macro']:.4f} | "
            f"{result['min']:.4f} | {result['median']:.4f} | {result['max']:.4f} | "
            f"{result['top1_acc']:.4f} | {result['spearman_r']:.3f} (p={result['spearman_p']:.3f}) |\n"
        )

    best = max((baseline, variant_a, variant_b), key=lambda r: r["macro"])
    beats_baseline = best["name"] != "baseline" and (best["macro"] - baseline["macro"]) >= PRACTICAL_MARGIN

    lines = [
        "# Candidate 1 — minimum-occupancy handling for buddy's Stage 2 targets\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; seed {SEED}. Motivated by "
        "[`deep_stage_analysis_report.md`](deep_stage_analysis_report.md) Finding A "
        "(macro AUC correlated with topic occupancy, Pearson r=0.531, p=0.019).\n\n",
        "Baseline reproduction check passed "
        f"({baseline['macro']:.4f} vs. expected {BASELINE_MACRO_AUC:.4f}, within 0.001).\n\n",
        "## Results\n\n",
        "| variant | K | macro AUC | min | median | max | top-1 acc | AUC-occupancy Spearman |\n",
        "|---|---:|---:|---:|---:|---:|---:|---|\n",
        row(baseline), row(variant_a), row(variant_b),
        f"\nMerge map (Variant B): small train communities {list(SMALL_COMMUNITIES)} "
        f"relabeled, producing {variant_b['n_topics']} surviving topics "
        f"(from {N_TOPICS}). Original-to-final label map: "
        f"{ {k: v for k, v in sorted(label_map.items())} }.\n\n",
        "## Verdict\n\n",
    ]
    if beats_baseline:
        lines.append(
            f"**{best['name']} meaningfully beats the baseline**: macro AUC "
            f"{best['macro']:.4f} vs. {baseline['macro']:.4f} "
            f"({best['macro'] - baseline['macro']:+.4f}), exceeding the "
            f"predeclared {PRACTICAL_MARGIN:.3f} practical margin. Recommended: "
            "adopt this variant and proceed to seed-stress before reporting it "
            "as a settled improvement (candidate 5 in the updated list).\n"
        )
    else:
        lines.append(
            f"**Neither variant meaningfully beats the baseline** (best: "
            f"{best['name']} at {best['macro']:.4f}, "
            f"{best['macro'] - baseline['macro']:+.4f} vs. the "
            f"{PRACTICAL_MARGIN:.3f} margin). Candidate 1 is closed as tested "
            "-- move to candidate 2 (mapper LR/capacity sweep).\n"
        )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
