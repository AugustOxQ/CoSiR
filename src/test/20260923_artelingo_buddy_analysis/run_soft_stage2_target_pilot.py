"""Does a soft, uncertainty-preserving Stage 2 target beat a hard topic label?

Candidate 5 from the brainstorm memo: buddy's k=20 neighbor vote already
carries more information than the argmax hard label it collapses to. This
pilot compares three image-only mapper targets on the same frozen buddy
train communities and patch-feature architecture as
run_buddy_percept_downstream_probe_pilot.py: (a) the hard one-hot label
(cited from that pilot, not rerun), (b) the k=20 vote-frequency
distribution as a soft cross-entropy target, and (c) a sparse multi-hot
threshold of that same distribution (>= 2/20 votes, mirroring PercepT's
own "at least 2 votes" multi-label convention at its own topic count).
Each mapper's held-out prediction is scored by the same fixed-capacity
downstream probe against real human labels.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from torch import nn


HERE = Path(__file__).resolve().parent
PERCEPT_DIR = HERE.parent / "20260922_percept_topic_pipeline"
BUDDY_SNAPSHOT = HERE / "attention_h1_embedding_snapshot.npz"
REPORT = HERE / "soft_stage2_target_pilot_report.md"
SEED = 42
N_TRAIN = 61402
N_HELDOUT = 9365
N_TOPICS = 19
K = 20
MULTI_LABEL_THRESHOLD = 2.0 / K  # Mirrors PercepT's own "at least 2 votes" convention.
# Cited rows from buddy_percept_downstream_probe_pilot_report.md (not rerun).
HARD_LABEL_RESULT = {
    "emotion_ami": 0.0231, "emotion_accuracy": 0.3350,
    "genre_ami": 0.2625, "genre_accuracy": 0.4403,
}
CONTROL_RESULT = {
    "emotion_ami": 0.0685, "emotion_accuracy": 0.3882,
    "genre_ami": 0.3399, "genre_accuracy": 0.5660,
}


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_matching_paintings(actual, expected, split, expected_count):
    actual_set = set(np.asarray(actual).tolist())
    expected_set = set(np.asarray(expected).tolist())
    if len(actual) != expected_count or len(expected) != expected_count:
        raise RuntimeError(f"{split} painting count mismatch: {len(actual)} vs {len(expected)}")
    if actual_set != expected_set:
        raise RuntimeError(f"{split} painting sets differ.")


def align_to_order(snapshot, canonical_paintings, split, keys):
    saved_paintings = snapshot[f"{split}_paintings"]
    assert_matching_paintings(
        canonical_paintings, saved_paintings, split,
        N_TRAIN if split == "train" else N_HELDOUT,
    )
    positions = {painting: index for index, painting in enumerate(saved_paintings)}
    order = np.asarray([positions[p] for p in canonical_paintings], dtype=np.int64)
    if not np.array_equal(saved_paintings[order], canonical_paintings):
        raise RuntimeError(f"{split} alignment failed.")
    return {key: snapshot[key][order] for key in keys}


def train_vote_frequency(train_embeddings, train_community, k=K, n_topics=N_TOPICS):
    """Self-referential k-NN vote frequency: each train point's own k nearest
    OTHER train points' community distribution, excluding itself."""
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=1)
    neighbors.fit(train_embeddings)
    frequencies = np.zeros((len(train_embeddings), n_topics), dtype=np.float32)
    batch_size = 2048
    for start in range(0, len(train_embeddings), batch_size):
        stop = min(start + batch_size, len(train_embeddings))
        indices = neighbors.kneighbors(train_embeddings[start:stop], return_distance=False)
        for offset, row in enumerate(indices):
            # Drop the query's own index (guaranteed to be its own nearest
            # neighbor at cosine distance 0 since it is itself a train point).
            self_index = start + offset
            neighbor_rows = row[row != self_index][:k]
            labels = train_community[neighbor_rows]
            counts = np.bincount(labels, minlength=n_topics).astype(np.float32)
            frequencies[self_index] = counts / counts.sum()
    return frequencies


def train_mapper_soft(stage2, patch_features, target_distribution, device, log):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    mapper = stage2.AttentionPoolingMapper(n_topics=N_TOPICS).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=stage2.MAPPER_LEARNING_RATE)
    target = torch.as_tensor(target_distribution, dtype=torch.float32, device=device)
    final_loss = float("nan")
    for epoch in range(1, stage2.MAPPER_EPOCHS + 1):
        optimizer.zero_grad()
        log_probs = torch.log_softmax(mapper(patch_features), dim=-1)
        loss = -(target * log_probs).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch % 10 == 0:
            log(f"soft mapper epoch {epoch:03d}/{stage2.MAPPER_EPOCHS}: soft cross-entropy={final_loss:.6f}")
    mapper.eval()
    return mapper, final_loss


def train_mapper_multi_hot(stage2, patch_features, multi_hot_targets, device, log):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    mapper = stage2.AttentionPoolingMapper(n_topics=N_TOPICS).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=stage2.MAPPER_LEARNING_RATE)
    target = torch.as_tensor(multi_hot_targets, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss()
    final_loss = float("nan")
    for epoch in range(1, stage2.MAPPER_EPOCHS + 1):
        optimizer.zero_grad()
        loss = criterion(mapper(patch_features), target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch % 10 == 0:
            log(f"multi-hot mapper epoch {epoch:03d}/{stage2.MAPPER_EPOCHS}: BCE={final_loss:.6f}")
    mapper.eval()
    return mapper, final_loss


def probe_scores(features, emotion, genre, emotion_indices, genre_indices, genre_cv, pipeline):
    emotion_train, emotion_eval = emotion_indices
    emotion_probe = LogisticRegression(max_iter=2000)
    emotion_probe.fit(features[emotion_train], emotion[emotion_train])
    emotion_predictions = emotion_probe.predict(features[emotion_eval])
    emotion_truth = emotion[emotion_eval]
    genre_truth = genre[genre_indices]
    genre_predictions = cross_val_predict(
        LogisticRegression(max_iter=2000), features[genre_indices], genre_truth, cv=genre_cv,
    )
    return {
        "emotion_ami": float(pipeline.external_metrics(emotion_predictions, emotion_truth)["AMI"]),
        "emotion_accuracy": float(accuracy_score(emotion_truth, emotion_predictions)),
        "genre_ami": float(pipeline.external_metrics(genre_predictions, genre_truth)["AMI"]),
        "genre_accuracy": float(accuracy_score(genre_truth, genre_predictions)),
    }


def write_report(results, losses, multi_hot_stats):
    names = ("hard one-hot (cited)", "soft vote-frequency", "sparse multi-hot", "mean-pooled control (cited)")
    lines = [
        "# Soft, uncertainty-preserving Stage 2 target pilot\n\n",
        "## Method\n\n",
        "Uses the same frozen buddy train communities, patch-feature cache, "
        "AttentionPoolingMapper architecture, Adam lr/epoch budget, and "
        "downstream logistic-regression probe protocol as "
        "`run_buddy_percept_downstream_probe_pilot.py`. The hard one-hot and "
        "mean-pooled control rows are cited from that pilot's report, not "
        "rerun. Two new mapper targets are compared:\n\n",
        f"- **Soft vote-frequency**: each train painting's own k={K} nearest "
        "OTHER train paintings' (cosine, excluding itself) community-label "
        "frequency distribution over the 19 frozen train communities, used "
        "as a soft cross-entropy target (`-(target * log_softmax(logits))"
        ".sum(dim=1).mean()`).\n",
        f"- **Sparse multi-hot**: the same distribution thresholded at "
        f">= {MULTI_LABEL_THRESHOLD:.4f} ({int(round(MULTI_LABEL_THRESHOLD * K))}/{K} "
        "votes, mirroring PercepT's own \"at least 2 votes\" multi-label "
        "convention at its own topic count), trained with "
        "`BCEWithLogitsLoss` as PercepT's original Stage 2 pilot does for "
        "its own multi-hot DEC targets.\n\n",
        f"Multi-hot label statistics: mean {multi_hot_stats['mean']:.3f} labels "
        f"per painting, {multi_hot_stats['fraction_multi']:.1%} of paintings "
        "multi-labeled.\n\n",
        "## Results\n\n",
        "| target | emotion AMI | emotion accuracy | genre AMI | genre accuracy |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for name in names:
        score = results[name]
        lines.append(
            f"| {name} | {score['emotion_ami']:.4f} | {score['emotion_accuracy']:.4f} | "
            f"{score['genre_ami']:.4f} | {score['genre_accuracy']:.4f} |\n"
        )
    lines.append(f"\nFinal train loss: soft={losses['soft']:.6f}; multi-hot={losses['multi_hot']:.6f}.\n\n")
    lines.append("## Verdict\n\n")
    hard = results[names[0]]
    soft = results["soft vote-frequency"]
    multi = results["sparse multi-hot"]
    soft_better = sum(soft[k] > hard[k] for k in ("emotion_ami", "emotion_accuracy", "genre_ami", "genre_accuracy"))
    multi_better = sum(multi[k] > hard[k] for k in ("emotion_ami", "emotion_accuracy", "genre_ami", "genre_accuracy"))
    lines.append(
        f"Soft vote-frequency beats the hard one-hot target on {soft_better}/4 metrics; "
        f"sparse multi-hot beats it on {multi_better}/4 metrics. "
    )
    if soft_better >= 3 or multi_better >= 3:
        lines.append(
            "At least one soft/uncertainty-preserving target is a clear improvement "
            "over the hard label on most metrics — worth adopting for a real Stage 2 "
            "wiring, not just this probe.\n"
        )
    else:
        lines.append(
            "Neither soft target is a clear, consistent improvement over the plain "
            "hard label by this single-seed/single-split screen. Softening the "
            "Stage 2 target is not a high-value lever on this evidence.\n"
        )
    with open(REPORT, "w") as report_file:
        report_file.writelines(lines)


def main():
    stage2 = load_module("percept_stage2_for_soft_target", PERCEPT_DIR / "run_percept_stage2_pilot.py")
    pipeline = load_module("pipeline_for_soft_target", HERE / "run_pipeline.py")
    heldout_pipeline = load_module("pipeline_for_soft_target_heldout", HERE / "run_pipeline.py")
    heldout_pipeline.STORAGE_DIR = stage2.base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = stage2.base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    train_paintings, _, _, train_emotion_counts = pipeline.load_dedup_features()
    heldout_paintings, _, _, heldout_emotion_counts = heldout_pipeline.load_dedup_features()
    if len(train_paintings) != N_TRAIN or len(heldout_paintings) != N_HELDOUT:
        raise RuntimeError("Fresh deduplicated painting counts differ from patch caches.")
    log("Loading patch features in the fresh deduplicated painting order...")
    train_patches = stage2.load_patch_features(stage2.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train")
    heldout_patches = stage2.load_patch_features(stage2.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out")

    with np.load(BUDDY_SNAPSHOT, allow_pickle=True) as source:
        buddy = {key: source[key] for key in (
            "train_paintings", "train_embedding_post", "train_community_post",
            "heldout_paintings", "heldout_emotion", "heldout_genre", "seed",
        )}
    if int(buddy["seed"]) != SEED:
        raise RuntimeError("Buddy snapshot must be the seed-42 run.")
    train_aligned = align_to_order(
        buddy, train_paintings, "train", ("train_embedding_post", "train_community_post"),
    )
    heldout_aligned = align_to_order(
        buddy, heldout_paintings, "heldout", ("heldout_emotion", "heldout_genre"),
    )
    train_community = np.asarray(train_aligned["train_community_post"], dtype=np.int64)
    if not np.array_equal(np.unique(train_community), np.arange(N_TOPICS)):
        raise RuntimeError("Buddy train vocabulary is not 0..18.")

    log(f"Computing self-referential k={K} vote-frequency targets on {N_TRAIN:,} train paintings...")
    vote_frequency = train_vote_frequency(train_aligned["train_embedding_post"], train_community, k=K)
    multi_hot = (vote_frequency >= MULTI_LABEL_THRESHOLD).astype(np.float32)
    empty_rows = np.flatnonzero(multi_hot.sum(axis=1) == 0)
    if len(empty_rows):
        # Guarantee at least the argmax label survives thresholding.
        multi_hot[empty_rows, vote_frequency[empty_rows].argmax(axis=1)] = 1.0
    multi_hot_stats = {
        "mean": float(multi_hot.sum(axis=1).mean()),
        "fraction_multi": float(np.mean(multi_hot.sum(axis=1) > 1)),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Training soft and multi-hot mappers on {device}...")
    train_patches = train_patches.to(device)
    soft_mapper, soft_loss = train_mapper_soft(stage2, train_patches, vote_frequency, device, log)
    multi_mapper, multi_loss = train_mapper_multi_hot(stage2, train_patches, multi_hot, device, log)
    del train_patches
    heldout_patches = heldout_patches.to(device)
    with torch.no_grad():
        soft_features = torch.softmax(soft_mapper(heldout_patches), dim=-1).cpu().numpy()
        multi_features = torch.sigmoid(multi_mapper(heldout_patches)).cpu().numpy()
    del soft_mapper, multi_mapper

    emotion = heldout_aligned["heldout_emotion"]
    genre_all = heldout_aligned["heldout_genre"]
    emotion_indices = train_test_split(
        np.arange(N_HELDOUT), test_size=0.5, stratify=emotion, random_state=SEED,
    )
    genre_indices = np.flatnonzero(genre_all != "")
    genre_values, genre_counts = np.unique(genre_all[genre_indices], return_counts=True)
    n_splits = min(5, int(genre_counts.min()))
    genre_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    log(f"Genre probe: {len(genre_indices)} paintings; {n_splits} folds.")

    results = {
        "hard one-hot (cited)": HARD_LABEL_RESULT,
        "mean-pooled control (cited)": CONTROL_RESULT,
        "soft vote-frequency": probe_scores(
            soft_features, emotion, genre_all, emotion_indices, genre_indices, genre_cv, pipeline
        ),
        "sparse multi-hot": probe_scores(
            multi_features, emotion, genre_all, emotion_indices, genre_indices, genre_cv, pipeline
        ),
    }
    write_report(results, {"soft": soft_loss, "multi_hot": multi_loss}, multi_hot_stats)
    for name in ("soft vote-frequency", "sparse multi-hot"):
        log(f"{name}: {results[name]}")
    log(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
