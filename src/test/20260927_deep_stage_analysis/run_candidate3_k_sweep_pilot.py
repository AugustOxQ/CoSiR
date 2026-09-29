"""Candidate 3: buddy topic-count (K) sweep.

Important simplification (verified by reading run_learned_student_arch_sweep_pilot.py):
buddy's community count is a DOWNSTREAM Leiden clustering choice applied
to the already-trained embedding's own mutual-kNN graph
(`detect_communities(final_graph, ...)`), not something baked into the
InfoNCE training itself. So sweeping K requires NO retraining of buddy's
embedding at all -- only re-clustering the already-saved, frozen
`train_embedding_post` at different Leiden resolutions (same
RBConfigurationVertexPartition technique used for the RedCaps diagnostic
earlier this session), then re-running the already-cheap Stage 2 pipeline
(candidate 1+2's adopted recipe: merge smallest topics below 1% train
share, class-balanced loss, lr=1e-2, epochs=400) at each K. This is why
this candidate does NOT need DAS6 despite initially looking like it would.

Screens at seed 42 (mapper seed; Stage 1 embedding and Leiden partition
are deterministic given a fixed resolution and seed=42 Leiden call).
"""

import time
from pathlib import Path

import sys

import igraph as ig
import leidenalg
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_mutual_info_score
from torch import nn

from run_candidate1_min_occupancy_pilot import PERCEPT_DIR, load_module, one_hot


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.conditional_buddy.buddy_graph import mutual_knn as project_mutual_knn

BUDDY_DIR = HERE.parents[1] / "test/20260923_artelingo_buddy_analysis"
SNAPSHOT = BUDDY_DIR / "attention_h1_embedding_snapshot.npz"
REPORT_PATH = HERE / "candidate3_k_sweep_pilot_report.md"

SEED = 42
GRAPH_K = 20  # mutual-kNN neighbors, matches this project's established buddy graph convention
TRANSFER_K = 20
MIN_TOPIC_SHARE = 0.01  # merge below 1% of train, matching candidate 1's convention
RESOLUTIONS = (2.0, 1.0, 0.5, 0.25, 0.1, 0.05)
EMOTION_BAR = 0.1236
GENRE_BAR = 0.1954
CURRENT_K = 19
CURRENT_STAGE2_AUC = 0.8461
MAPPER_LR = 1e-2
MAPPER_EPOCHS = 400


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def mutual_knn_graph(embeddings: np.ndarray, k: int, device: str) -> csr_matrix:
    """Reuse the project's own memory-safe, batched mutual-kNN builder
    rather than a naive O(N^2) dense implementation (61k train nodes would
    need ~15GB for a dense float32 similarity matrix)."""
    return project_mutual_knn(embeddings.astype(np.float32), K=k, device=device)


def leiden_partition(adjacency: csr_matrix, resolution: float, seed: int = SEED) -> np.ndarray:
    coo = adjacency.tocoo()
    mask = coo.row < coo.col
    edges = list(zip(coo.row[mask].tolist(), coo.col[mask].tolist()))
    graph = ig.Graph(n=adjacency.shape[0], edges=edges)
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    return np.array(partition.membership, dtype=np.int64)


def merge_small(train_embedding: np.ndarray, labels: np.ndarray, min_share: float) -> tuple[np.ndarray, int]:
    n_topics = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_topics)
    small = [t for t in range(n_topics) if counts[t] < min_share * len(labels)]
    if not small:
        return labels, n_topics
    large = [t for t in range(n_topics) if t not in small]
    if not large:
        return labels, n_topics  # degenerate: everything is "small"; leave as-is
    centroids = {}
    for topic in range(n_topics):
        members = train_embedding[labels == topic]
        centroid = members.mean(axis=0)
        centroids[topic] = centroid / max(np.linalg.norm(centroid), 1e-12)
    relabeled = labels.copy()
    for topic in small:
        sims = {t: float(centroids[topic] @ centroids[t]) for t in large}
        target = max(sims, key=sims.get)
        relabeled[labels == topic] = target
    remaining = sorted(set(relabeled.tolist()))
    compact = {old: new for new, old in enumerate(remaining)}
    return np.array([compact[x] for x in relabeled], dtype=np.int64), len(remaining)


def train_stage2(n_topics, train_labels, heldout_labels, train_patches, heldout_patches, stage2_ref, device):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    train_targets = one_hot(train_labels, n_topics).to(device)
    heldout_targets = one_hot(heldout_labels, n_topics).cpu().numpy()
    counts = np.bincount(train_labels, minlength=n_topics)
    inverse_freq = (len(train_labels) / n_topics) / counts[train_labels]
    inverse_freq = inverse_freq / inverse_freq.mean()
    weights = torch.as_tensor(inverse_freq, dtype=torch.float32, device=device)

    mapper = stage2_ref.AttentionPoolingMapper(n_topics=n_topics).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=MAPPER_LR)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mapper.train()
    for _ in range(MAPPER_EPOCHS):
        logits = mapper(train_patches)
        loss = (bce(logits, train_targets).mean(dim=1) * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(heldout_patches)).cpu().numpy()
    model_aucs, _ = stage2_ref.evaluate_auc(heldout_scores, heldout_targets, log, f"K={n_topics}")
    return stage2_ref.auc_summary(model_aucs)["macro"]


def main() -> None:
    stage2_ref = load_module("stage2_ref_for_candidate3", PERCEPT_DIR / "run_percept_stage2_pilot.py")
    transfer = load_module("transfer_for_candidate3", BUDDY_DIR / "run_heldout_label_transfer_pilot.py")

    with np.load(SNAPSHOT, allow_pickle=True) as source:
        train_paintings = source["train_paintings"]
        train_embedding = source["train_embedding_post"]
        heldout_paintings = source["heldout_paintings"]
        heldout_embedding = source["heldout_embedding_post"]
        train_emotion = source["train_emotion"]
        train_genre = source["train_genre"]
        heldout_emotion = source["heldout_emotion"]
        heldout_genre = source["heldout_genre"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_patches = stage2_ref.load_patch_features(
        stage2_ref.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train"
    ).to(device)
    heldout_patches = stage2_ref.load_patch_features(
        stage2_ref.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out"
    ).to(device)

    log(f"Building train mutual-kNN graph (K={GRAPH_K}) on the frozen embedding...")
    train_graph = mutual_knn_graph(train_embedding, GRAPH_K, device)

    genre_mask_train = np.array([str(g) not in ("", "nan") for g in train_genre])
    genre_mask_heldout = np.array([str(g) not in ("", "nan") for g in heldout_genre])

    results = []
    for resolution in RESOLUTIONS:
        log(f"=== resolution={resolution} ===")
        raw_labels = leiden_partition(train_graph, resolution)
        raw_k = int(raw_labels.max()) + 1
        merged_labels, k = merge_small(train_embedding, raw_labels, MIN_TOPIC_SHARE)
        if k < 2:
            log(f"resolution={resolution}: degenerate (K={k} after merge), skipping.")
            continue
        heldout_labels_raw = transfer.assign_to_train_communities(
            train_embedding, merged_labels, heldout_embedding, k=TRANSFER_K
        )
        emotion_ami = adjusted_mutual_info_score(heldout_emotion, heldout_labels_raw)
        genre_ami = (
            adjusted_mutual_info_score(heldout_genre[genre_mask_heldout], heldout_labels_raw[genre_mask_heldout])
            if genre_mask_heldout.sum() > 0 else float("nan")
        )
        clears_bar = emotion_ami > EMOTION_BAR and genre_ami > GENRE_BAR
        macro_auc = train_stage2(
            k, merged_labels, heldout_labels_raw, train_patches, heldout_patches, stage2_ref, device
        )
        log(f"resolution={resolution}: raw_K={raw_k}, merged_K={k}, emotion_AMI={emotion_ami:.4f}, "
            f"genre_AMI={genre_ami:.4f}, clears_bar={clears_bar}, Stage2_macro_AUC={macro_auc:.4f}")
        results.append({
            "resolution": resolution, "raw_k": raw_k, "k": k,
            "emotion_ami": emotion_ami, "genre_ami": genre_ami,
            "clears_bar": clears_bar, "macro_auc": macro_auc,
        })

    write_report(results)
    log(f"Wrote {REPORT_PATH}")


def write_report(results: list[dict]) -> None:
    lines = [
        "# Candidate 3 — buddy topic-count (K) sweep\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Current adopted K={CURRENT_K}, "
        f"Stage 2 macro AUC {CURRENT_STAGE2_AUC:.4f} (candidate 2's result). "
        "K here is swept via Leiden resolution on the already-trained, frozen "
        "embedding (no InfoNCE retraining) -- see module docstring for why "
        "this makes the candidate cheap enough for local GPU, no DAS6 needed.\n\n",
        "Each row: raw Leiden K, K after merging any topic below 1% train "
        "share, held-out emotion/genre AMI (Pareto bar: emotion > 0.1236 AND "
        "genre > 0.1954), and Stage 2 macro AUC using the candidate 1+2 "
        f"adopted recipe (class-balanced loss, lr={MAPPER_LR:g}, "
        f"epochs={MAPPER_EPOCHS}).\n\n",
        "| resolution | raw K | merged K | emotion AMI | genre AMI | clears bar | Stage 2 macro AUC |\n",
        "|---:|---:|---:|---:|---:|---|---:|\n",
    ]
    for r in results:
        lines.append(
            f"| {r['resolution']:g} | {r['raw_k']} | {r['k']} | {r['emotion_ami']:.4f} | "
            f"{r['genre_ami']:.4f} | {'yes' if r['clears_bar'] else 'no'} | {r['macro_auc']:.4f} |\n"
        )

    best = max(results, key=lambda r: r["macro_auc"]) if results else None
    lines.append("\n## Verdict\n\n")
    if best and best["macro_auc"] - CURRENT_STAGE2_AUC >= 0.005 and best["clears_bar"]:
        lines.append(
            f"**K={best['k']} (resolution={best['resolution']:g}) beats the current "
            f"K={CURRENT_K} baseline**: Stage 2 macro AUC {best['macro_auc']:.4f} vs. "
            f"{CURRENT_STAGE2_AUC:.4f} ({best['macro_auc'] - CURRENT_STAGE2_AUC:+.4f}), "
            "while still clearing the Pareto bar. Recommended: 4-seed stress "
            "this K before adopting.\n"
        )
    else:
        lines.append(
            f"**No tested K meaningfully beats the current K={CURRENT_K} configuration** "
            f"on Stage 2 macro AUC while also clearing the Pareto bar "
            f"(best: K={best['k'] if best else 'n/a'} at "
            f"{best['macro_auc'] if best else float('nan'):.4f}). Candidate 3 is closed "
            "as tested with a negative/neutral result; K=19 (merged to 16 for "
            "Stage 2) remains the adopted configuration. Move to candidate 4 "
            "(richer multi-label targets) or consider this investigation's "
            "Stage 2 improvements complete for now.\n"
        )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
