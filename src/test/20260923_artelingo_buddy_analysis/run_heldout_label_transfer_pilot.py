"""Transfer held-out Attention-h1 points onto the frozen train Leiden labels.

This CPU-only pilot reads the saved embedding snapshot; it does not fit a
student model or recompute either split's Leiden communities.
"""

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "attention_h1_embedding_snapshot.npz"
REPORT = HERE / "heldout_label_transfer_pilot_report.md"
K_VALUES = (5, 10, 20, 50)
CITED_HELDOUT_EMOTION_AMI = 0.1249
CITED_HELDOUT_GENRE_AMI = 0.2404
BATCH_SIZE = 128


def assign_to_train_communities(
    train_embeddings: np.ndarray,
    train_community: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Assign queries to train Leiden labels by cosine k-NN majority vote.

    A tied vote takes the closest train point among the tied winners. If all k
    labels are distinct, that choice is counted as a degenerate fallback in
    ``assign_to_train_communities.last_fallback_count``. Query batches cap
    the cosine-distance matrix passed to scikit-learn at 128 x N_train.
    """
    train_embeddings = np.asarray(train_embeddings)
    train_community = np.asarray(train_community)
    query_embeddings = np.asarray(query_embeddings)
    if train_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")
    if train_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("Train and query embedding dimensions must match")
    if len(train_embeddings) != len(train_community):
        raise ValueError("Each train embedding needs one community label")
    if not 1 <= k <= len(train_embeddings):
        raise ValueError("k must be between 1 and the number of train points")

    neighbors = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=1)
    neighbors.fit(train_embeddings)
    assigned = np.empty(len(query_embeddings), dtype=train_community.dtype)
    fallback_count = 0
    for start in range(0, len(query_embeddings), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(query_embeddings))
        # Passing a batch explicitly prevents a full query-by-train matrix.
        indices = neighbors.kneighbors(query_embeddings[start:stop], return_distance=False)
        for offset, row in enumerate(indices):
            labels = train_community[row]
            values, counts = np.unique(labels, return_counts=True)
            highest = counts.max()
            winners = values[counts == highest]
            if len(winners) == 1:
                assigned[start + offset] = winners[0]
            else:
                assigned[start + offset] = next(label for label in labels if label in winners)
                if highest == 1 and len(values) > k / 2:
                    fallback_count += 1
    assign_to_train_communities.last_fallback_count = fallback_count
    return assigned


assign_to_train_communities.last_fallback_count = 0


def load_pipeline():
    """Use the sibling pipeline's exact metric and genre-map definitions."""
    path = HERE / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("artelingo_label_transfer_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not SNAPSHOT.is_file():
        raise FileNotFoundError(f"Required saved snapshot is missing: {SNAPSHOT}")
    pipeline = load_pipeline()
    with np.load(SNAPSHOT, allow_pickle=True) as snapshot:
        train_embeddings = snapshot["train_embedding_post"]
        train_community = snapshot["train_community_post"]
        query_embeddings = snapshot["heldout_embedding_post"]
        independent = snapshot["heldout_community_post"]
        emotions = snapshot["heldout_emotion"]
        paintings = snapshot["heldout_paintings"]
        saved_genres = snapshot["heldout_genre"]

    genre_map = pipeline.load_genre_map()
    genre_indices = np.asarray(
        [i for i, painting in enumerate(paintings) if painting in genre_map],
        dtype=np.int64,
    )
    genres = [genre_map[paintings[i]] for i in genre_indices]
    if not np.array_equal(saved_genres[genre_indices], np.asarray(genres, dtype=object)):
        raise RuntimeError("Saved held-out genre labels disagree with pipeline genre map")

    def score(labels: np.ndarray) -> tuple[float, float]:
        emotion_ami = pipeline.external_metrics(labels, emotions)["AMI"]
        genre_ami = pipeline.external_metrics(labels[genre_indices], genres)["AMI"]
        return float(emotion_ami), float(genre_ami)

    reference_emotion, reference_genre = score(independent)
    # Citations have four decimals, so half a unit at the fifth decimal
    # is the precise acceptance window for reproducing their rounded values.
    if (abs(reference_emotion - CITED_HELDOUT_EMOTION_AMI) >= 0.00005 or
            abs(reference_genre - CITED_HELDOUT_GENRE_AMI) >= 0.00005):
        raise RuntimeError(
            "Independent-Leiden sanity check disagrees with cited baselines: "
            f"emotion={reference_emotion:.8f}, genre={reference_genre:.8f}"
        )
    train_labels = set(np.unique(train_community).tolist())
    original_count = len(np.unique(independent))
    rows = []
    for k in K_VALUES:
        transferred = assign_to_train_communities(
            train_embeddings, train_community, query_embeddings, k=k
        )
        assigned_labels = set(np.unique(transferred).tolist())
        if not assigned_labels.issubset(train_labels):
            raise RuntimeError(f"k={k} produced a label outside the train vocabulary")
        emotion_ami, genre_ami = score(transferred)
        rows.append((k, emotion_ami, genre_ami, len(assigned_labels),
                     assign_to_train_communities.last_fallback_count))
        print(f"k={k}: emotion AMI={emotion_ami:.6f}, genre AMI={genre_ami:.6f}, "
              f"coverage={len(assigned_labels)}/{len(train_labels)}, "
              f"fallbacks={rows[-1][4]}", flush=True)

    lines = [
        "# Held-out label transfer onto the train Leiden vocabulary\n\n",
        "## Sanity check\n\n",
        "Using the saved post-training independent held-out Leiden labels, "
        f"emotion AMI = **{reference_emotion:.8f}** and genre AMI = "
        f"**{reference_genre:.8f}**. These round to the cited "
        "**0.1249** and **0.2404**, respectively: both match. "
        f"Genre AMI uses the {len(genre_indices)} held-out paintings in "
        "`pipeline.load_genre_map()`; their saved genre values all match that map.\n\n",
        "## k sweep\n\n",
        f"Post-training train vocabulary: {len(train_labels)} communities; "
        f"held-out points: {len(query_embeddings)}. Cosine `NearestNeighbors` "
        f"queries at most {BATCH_SIZE} held-out points per batch, so this script "
        "never passes the full held-out-by-train matrix to scikit-learn. "
        "A tied top vote uses the nearest train point among the tied "
        "winners (the interpretation of the brief's nearest-neighbor "
        "tie-break); fallback count records the all-distinct-label case. "
        f"With only {len(train_labels)} train labels, that fallback is "
        "impossible when k exceeds 19; ordinary tied top votes are not "
        "counted.\n\n",
        "| method | emotion AMI | genre AMI | train coverage | tie fallbacks | distinct held-out labels |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| Original independent Leiden | {reference_emotion:.4f} | "
        f"{reference_genre:.4f} | n/a | n/a | {original_count} |\n",
    ]
    for k, emotion_ami, genre_ami, coverage, fallbacks in rows:
        lines.append(
            f"| k={k} | {emotion_ami:.4f} | {genre_ami:.4f} | "
            f"{coverage}/{len(train_labels)} | {fallbacks} | {coverage} |\n"
        )
    emotion_values = [row[1] for row in rows]
    genre_values = [row[2] for row in rows]
    best_emotion = max(emotion_values)
    best_genre = max(genre_values)
    k20 = next(row for row in rows if row[0] == 20)
    emotion_delta = k20[1] - reference_emotion
    genre_delta = k20[2] - reference_genre
    genre_range = best_genre - min(genre_values)
    emotion_range = best_emotion - min(emotion_values)
    sensitivity = ("Genre is more sensitive to k in this sweep."
                   if genre_range > emotion_range else
                   "Emotion is at least as sensitive to k in this sweep.")
    if emotion_delta >= 0 and genre_delta >= 0:
        comparison = (
            "Both AMIs are higher at k=20, so k-NN transfer preserves "
            "external-label agreement well enough to be a usable Stage-2 "
            "target-construction mechanism at that setting; the observed "
            "comparison shows no degradation."
        )
    else:
        comparison = (
            "At least one AMI is lower at k=20; assess that degradation "
            "before using these assignments as Stage-2 targets."
        )
    lines.extend([
        "\n## Verdict\n\n",
        f"Across k=5–50, emotion AMI ranges {min(emotion_values):.4f}–"
        f"{best_emotion:.4f} and genre AMI ranges {min(genre_values):.4f}–"
        f"{best_genre:.4f}. Emotion varies by {emotion_range:.4f}; "
        f"genre varies by {genre_range:.4f} on {len(genre_indices)} "
        f"genre-labeled points. {sensitivity}\n\n",
        f"At the project-convention k=20, emotion AMI changes by "
        f"{emotion_delta:+.4f} and genre AMI by {genre_delta:+.4f} relative "
        f"to independent held-out Leiden. {comparison} Genre AMI uses only "
        f"{len(genre_indices)} of {len(query_embeddings)} held-out points, "
        "so its differences are less secure than the full-split emotion "
        "AMI. This is an observed comparison, not a formal equivalence "
        "test.\n\n",
        "The specific structural blocker—the absence of a shared "
        "train/held-out topic vocabulary—is resolved for hard-label "
        "assignment: every transferred held-out label belongs to the "
        "frozen train vocabulary. Future Stage-2 wiring should import "
        "`assign_to_train_communities` from "
        "`run_heldout_label_transfer_pilot.py` and reuse it, rather than "
        "reimplementing the transfer. This pilot evaluates hard labels; "
        "it does not construct or evaluate soft [N, K] targets.\n",
    ])
    REPORT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
