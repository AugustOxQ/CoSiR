"""Candidate 4: k=20 cosine-vote multi-label targets for buddy Stage 2.

The frozen Stage 1 snapshot and K=16 merge are the candidate 1+2 setup.
Only the mapper is trained; seed 42 screens the three specified cutoffs.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_candidate1_min_occupancy_pilot import (
    PERCEPT_DIR, PRACTICAL_MARGIN, SMALL_COMMUNITIES, SNAPSHOT, TRANSFER_K,
    load_module, merge_small_communities, one_hot,
)
from run_candidate1_stress_pilot import STRESS_SEEDS
from run_candidate2_mapper_sweep_pilot import log


HERE = Path(__file__).resolve().parent
REPORT_PATH = HERE / "candidate4_rich_multilabel_pilot_report.md"
BASELINE_MACRO_AUC = 0.8461
THRESHOLDS = (0.5, 0.3, 0.15)
MAPPER_LR = 1e-2
MAPPER_EPOCHS = 400
QUERY_BATCH_SIZE = 128


def cosine_vote_fractions(
    train_embeddings: np.ndarray, train_labels: np.ndarray,
    query_embeddings: np.ndarray, n_topics: int, device: str,
) -> np.ndarray:
    """Return merged-label fractions among the exact k=20 cosine neighbors.

    This local helper is needed because the shared transfer function discards
    neighbor counts when it returns the hard majority label. Queries are
    batched just as in that function; train queries include their own point.
    """
    reference = torch.as_tensor(train_embeddings, dtype=torch.float32, device=device)
    reference = nn.functional.normalize(reference, dim=1)
    queries = torch.as_tensor(query_embeddings, dtype=torch.float32, device=device)
    queries = nn.functional.normalize(queries, dim=1)
    labels = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    fractions = np.empty((len(query_embeddings), n_topics), dtype=np.float32)
    for start in range(0, len(query_embeddings), QUERY_BATCH_SIZE):
        stop = min(start + QUERY_BATCH_SIZE, len(query_embeddings))
        similarity = queries[start:stop] @ reference.T
        neighbors = similarity.topk(TRANSFER_K, dim=1).indices
        neighbor_labels = labels[neighbors].cpu().numpy().ravel()
        votes = one_hot(neighbor_labels, n_topics)
        fractions[start:stop] = votes.reshape(stop - start, TRANSFER_K, n_topics).mean(dim=1).numpy()
    if not np.allclose(fractions.sum(axis=1), 1.0):
        raise RuntimeError("Every k-NN vote distribution must sum to one")
    return fractions


def threshold_targets(fractions: np.ndarray, cutoff: float) -> np.ndarray:
    """Keep topics with vote fraction strictly above cutoff times row max."""
    return (fractions > cutoff * fractions.max(axis=1, keepdims=True)).astype(np.float32)


def label_stats(targets: np.ndarray) -> dict:
    counts = targets.sum(axis=1)
    return {
        "mean": float(counts.mean()), "median": float(np.median(counts)),
        "max": int(counts.max()), "multi": float(np.mean(counts > 1)),
    }


def train_and_eval(
    stage2_ref, train_targets: np.ndarray, heldout_targets: np.ndarray,
    train_patches: torch.Tensor, heldout_patches: torch.Tensor,
    device: str, seed: int, name: str,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n_topics = train_targets.shape[1]
    mapper = stage2_ref.AttentionPoolingMapper(n_topics=n_topics).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=MAPPER_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    train_tensor = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    mapper.train()
    for epoch in range(1, MAPPER_EPOCHS + 1):
        loss = loss_fn(mapper(train_patches), train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            log(f"{name}: epoch {epoch}/{MAPPER_EPOCHS}, BCE={loss.item():.6f}")
    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patches)).cpu().numpy()
    per_topic, skipped = stage2_ref.evaluate_auc(
        heldout_scores, heldout_targets, log, name,
    )
    summary = stage2_ref.auc_summary(per_topic)
    return {**summary, "skipped": len(skipped)}


def write_report(rows: list[dict], stress: list[tuple[int, dict]], winner: float | None) -> None:
    lines = [
        "# Candidate 4 — richer multi-label buddy Stage 2 targets\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Frozen buddy Stage 1 "
        "snapshot; candidate 1's three-smallest-community merge to K=16; "
        "candidate 2's AttentionPoolingMapper, lr=1e-2, 400 epochs. "
        f"Seed 42 screen; practical margin +{PRACTICAL_MARGIN:.3f} over the "
        f"candidate 2 four-seed mean macro AUC {BASELINE_MACRO_AUC:.4f}.\n\n",
        "## Targets and training\n\n",
        "The shared `assign_to_train_communities` returns only a hard winning "
        "label, not the neighbor vote fractions. A local batched helper "
        "therefore computes the same k=20 cosine nearest-neighbor vote "
        "fractions, counting neighbors in the merged K=16 vocabulary. "
        "Train paintings are queried against the train embeddings, including "
        "their own embedding; held-out paintings are queried against that "
        "same train vocabulary. A topic is positive when its fraction is "
        "strictly greater than the stated cutoff times that painting's "
        "largest fraction.\n\n",
        "The loss is **unweighted BCEWithLogitsLoss** on multi-hot targets. "
        "Candidate 1's single-sample class-balanced weighting is dropped "
        "because its inverse hard-class frequency definition does not "
        "carry over to multi-hot paintings.\n\n",
        "## Seed 42 screen\n\n",
        "| relative cutoff | train mean / median / max labels | train multi-labeled | "
        "held-out mean / median / max labels | held-out multi-labeled | "
        "macro AUC | min / median / max topic AUC | skipped topics |\n",
        "|---:|---|---:|---|---:|---:|---|---:|\n",
    ]
    for row in rows:
        train, heldout, result = row["train"], row["heldout"], row["result"]
        lines.append(
            f"| {row['cutoff']:.2f} | {train['mean']:.3f} / {train['median']:.3f} / {train['max']} | "
            f"{train['multi']:.2%} | {heldout['mean']:.3f} / {heldout['median']:.3f} / {heldout['max']} | "
            f"{heldout['multi']:.2%} | {result['macro']:.4f} | "
            f"{result['min']:.4f} / {result['median']:.4f} / {result['max']:.4f} | "
            f"{result['skipped']} |\n"
        )
    best = max(rows, key=lambda row: row["result"]["macro"])
    lines.append(
        "\nThe baseline uses strict single-label held-out targets, so its AUC "
        "and these AUCs evaluate different target definitions. Their "
        "difference is a practical screen, not a like-for-like metric "
        "improvement on identical labels. The experiment also drops the "
        "single-label loss weighting, so this screen does not isolate the "
        "effect of multi-label targets alone.\n\n"
    )
    if winner is None:
        reason = (
            f"the best seed 42 macro AUC was {best['result']['macro']:.4f} "
            f"at cutoff {best['cutoff']:.2f}, below the "
            f"{BASELINE_MACRO_AUC + PRACTICAL_MARGIN:.4f} stress gate"
            if best["result"]["macro"] < BASELINE_MACRO_AUC + PRACTICAL_MARGIN
            else "no cutoff both cleared the stress gate and gave nondegenerate multi-label targets"
        )
        lines.append(
            "## Verdict\n\n"
            f"**Candidate 4 is closed as tested under the specified screen "
            "with a negative result.** "
            f"Four-seed stress was skipped because {reason}. "
            "No additional cutoffs were tested.\n"
        )
    else:
        macros = np.array([result["macro"] for _, result in stress])
        lines.extend([
            f"\n## Four-seed stress: cutoff {winner:.2f}\n\n",
            "Only the mapper seed changes; the frozen Stage 1 snapshot and "
            "targets are reused.\n\n",
            "| seed | macro AUC | min / median / max topic AUC | skipped topics |\n",
            "|---:|---:|---|---:|\n",
        ])
        for seed, result in stress:
            lines.append(
                f"| {seed} | {result['macro']:.4f} | {result['min']:.4f} / "
                f"{result['median']:.4f} / {result['max']:.4f} | {result['skipped']} |\n"
            )
        mean = float(macros.mean())
        if mean >= BASELINE_MACRO_AUC + PRACTICAL_MARGIN:
            verdict = f"**Cutoff {winner:.2f} meaningfully beats the baseline**"
        else:
            verdict = "**Candidate 4 is closed as tested with a negative result**"
        lines.append(
            f"\n## Verdict\n\n{verdict}: four-seed mean macro AUC "
            f"{mean:.4f} (min {macros.min():.4f}, max {macros.max():.4f}) "
            f"versus the {BASELINE_MACRO_AUC:.4f} single-label baseline "
            f"({mean - BASELINE_MACRO_AUC:+.4f}); required margin "
            f"+{PRACTICAL_MARGIN:.3f}. No additional cutoffs were tested.\n"
        )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires a local CUDA GPU")
    device = "cuda"
    stage2_ref = load_module(
        "stage2_ref_for_candidate4", PERCEPT_DIR / "run_percept_stage2_pilot.py",
    )
    with np.load(SNAPSHOT, allow_pickle=True) as source:
        train_paintings = source["train_paintings"]
        train_embeddings = source["train_embedding_post"]
        hard_train_labels = source["train_community_post"].astype(np.int64)
        heldout_paintings = source["heldout_paintings"]
        heldout_embeddings = source["heldout_embedding_post"]
    merged_train_labels, label_map = merge_small_communities(
        train_embeddings, hard_train_labels, SMALL_COMMUNITIES,
    )
    n_topics = len(set(label_map.values()))
    if n_topics != 16:
        raise RuntimeError(f"Expected candidate 1's merged K=16, got K={n_topics}")
    log("Computing k=20 merged-community cosine vote fractions")
    train_fractions = cosine_vote_fractions(
        train_embeddings, merged_train_labels, train_embeddings, n_topics, device,
    )
    heldout_fractions = cosine_vote_fractions(
        train_embeddings, merged_train_labels, heldout_embeddings, n_topics, device,
    )
    train_patches = stage2_ref.load_patch_features(
        stage2_ref.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train",
    ).to(device)
    heldout_patches = stage2_ref.load_patch_features(
        stage2_ref.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out",
    ).to(device)
    rows = []
    for cutoff in THRESHOLDS:
        train_targets = threshold_targets(train_fractions, cutoff)
        heldout_targets = threshold_targets(heldout_fractions, cutoff)
        log(f"Screening relative cutoff {cutoff:.2f} at seed 42")
        result = train_and_eval(
            stage2_ref, train_targets, heldout_targets, train_patches,
            heldout_patches, device, 42, f"cutoff_{cutoff:.2f}_seed42",
        )
        rows.append({
            "cutoff": cutoff, "train": label_stats(train_targets),
            "heldout": label_stats(heldout_targets), "result": result,
        })
        log(f"Cutoff {cutoff:.2f}: macro AUC={result['macro']:.4f}")
    eligible = [
        row for row in rows
        if row["train"]["multi"] > 0 and row["heldout"]["multi"] > 0
        and row["train"]["mean"] < n_topics and row["heldout"]["mean"] < n_topics
        and row["result"]["macro"] >= BASELINE_MACRO_AUC + PRACTICAL_MARGIN
    ]
    winner = max(eligible, key=lambda row: row["result"]["macro"]) if eligible else None
    stress = []
    if winner is not None:
        cutoff = winner["cutoff"]
        train_targets = threshold_targets(train_fractions, cutoff)
        heldout_targets = threshold_targets(heldout_fractions, cutoff)
        log(f"Cutoff {cutoff:.2f} cleared margin; running four-seed stress")
        for seed in STRESS_SEEDS:
            result = train_and_eval(
                stage2_ref, train_targets, heldout_targets, train_patches,
                heldout_patches, device, seed, f"cutoff_{cutoff:.2f}_seed{seed}",
            )
            stress.append((seed, result))
            log(f"Stress seed {seed}: macro AUC={result['macro']:.4f}")
    else:
        log("No threshold cleared the +0.005 seed-42 screen; skipping stress")
    write_report(rows, stress, winner["cutoff"] if winner else None)
    log(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
