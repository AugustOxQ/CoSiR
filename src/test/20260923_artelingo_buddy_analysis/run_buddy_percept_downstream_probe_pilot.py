"""Compare frozen buddy and PercepT topics as image-only human-label features.

Both systems provide one hard train topic per painting. We deliberately train
their shared attention-pooling architecture with single-label cross-entropy,
instead of the original Stage 2 pilot's BCE-with-logits objective for DEC's
soft multi-hot targets. This makes the two hard-label mapper tasks symmetric.
The held-out human labels are used only to fit and evaluate the downstream
probes; they never enter either mapper's training.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from torch import nn


HERE = Path(__file__).resolve().parent
PERCEPT_DIR = HERE.parent / "20260922_percept_topic_pipeline"
BUDDY_SNAPSHOT = HERE / "attention_h1_embedding_snapshot.npz"
PERCEPT_SNAPSHOT = PERCEPT_DIR / "percept_stage1_faithful_recipe_snapshot.npz"
REPORT = HERE / "buddy_percept_downstream_probe_pilot_report.md"
SEED = 42
N_TRAIN = 61402
N_HELDOUT = 9365


def load_module(module_name, path):
    """Import a sibling pilot without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_snapshot(path, keys):
    """Read only the saved identities and labels needed for this pilot."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as source:
        return {key: source[key] for key in keys}


def align_snapshot(snapshot, canonical_paintings, split, keys, audit):
    """Reindex snapshot rows by painting ID into the patch cache's order."""
    saved_paintings = snapshot[f"{split}_paintings"]
    audit.assert_matching_paintings(
        canonical_paintings, saved_paintings, split,
        N_TRAIN if split == "train" else N_HELDOUT,
    )
    positions = {painting: index for index, painting in enumerate(saved_paintings)}
    aligned = np.asarray(
        [positions[painting] for painting in canonical_paintings], dtype=np.int64
    )
    if not np.array_equal(saved_paintings[aligned], canonical_paintings):
        raise RuntimeError(f"{split} painting alignment failed for snapshot")
    return {key: snapshot[key][aligned] for key in keys}


def check_topic_labels(labels, n_topics, system):
    """Require valid hard integer targets in the fixed train vocabulary."""
    if labels.shape != (N_TRAIN,) or not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError(f"{system} train topics must be {N_TRAIN} integer labels")
    if np.any(labels < 0) or np.any(labels >= n_topics):
        raise RuntimeError(f"{system} train topics fall outside 0..{n_topics - 1}")


def train_mapper(stage2, patch_features, labels, n_topics, device, log, system):
    """Fit the sibling architecture with the Stage 2 full-batch budget."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    mapper = stage2.AttentionPoolingMapper(n_topics=n_topics).to(device)
    optimizer = torch.optim.Adam(
        mapper.parameters(), lr=stage2.MAPPER_LEARNING_RATE
    )
    target = torch.as_tensor(labels, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()
    mapper.train()
    final_loss = float("nan")
    for epoch in range(1, stage2.MAPPER_EPOCHS + 1):
        optimizer.zero_grad()
        loss = criterion(mapper(patch_features), target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch % 10 == 0:
            log(
                f"{system} mapper epoch {epoch:03d}/{stage2.MAPPER_EPOCHS}: "
                f"train cross-entropy={final_loss:.6f}"
            )
    mapper.eval()
    return mapper, final_loss


def probe_scores(features, labels, emotion_indices, genre_indices, genre_cv,
                 pipeline):
    """Fit fixed logistic probes and score their held-out predictions."""
    emotion_train, emotion_eval = emotion_indices
    emotion_probe = LogisticRegression(max_iter=2000)
    emotion_probe.fit(features[emotion_train], labels["emotion"][emotion_train])
    emotion_predictions = emotion_probe.predict(features[emotion_eval])
    emotion_truth = labels["emotion"][emotion_eval]

    genre_truth = labels["genre"][genre_indices]
    genre_predictions = cross_val_predict(
        LogisticRegression(max_iter=2000), features[genre_indices], genre_truth,
        cv=genre_cv,
    )
    return {
        "emotion_ami": float(
            pipeline.external_metrics(emotion_predictions, emotion_truth)["AMI"]
        ),
        "emotion_accuracy": float(accuracy_score(emotion_truth, emotion_predictions)),
        "genre_ami": float(
            pipeline.external_metrics(genre_predictions, genre_truth)["AMI"]
        ),
        "genre_accuracy": float(accuracy_score(genre_truth, genre_predictions)),
    }


def write_report(results, losses, emotion_indices, genre_count, genre_splits,
                 genre_min_count, baselines):
    """Write the shared comparison and a numeric, outcome-specific verdict."""
    names = ("buddy topic softmax", "PercepT topic softmax", "mean-pooled control")
    buddy, percept, control = (results[name] for name in names)
    lines = [
        "# Buddy versus PercepT: image-only downstream probe pilot\n\n",
        "## Method\n\n",
        "The train and held-out patch caches follow the fresh painting order "
        "returned by each split's `pipeline.load_dedup_features()`. Both saved "
        "snapshots were checked for identical unique painting-ID sets and "
        "reindexed by painting ID into those cache orders before labels were "
        "used. Aligned held-out emotions and genres from both snapshots were "
        "checked against labels computed from the held-out pipeline. Buddy's "
        "independently reclustered held-out communities were not used; "
        "PercepT's held-out topics were checked but were not mapper targets.\n\n",
        "Two unchanged `AttentionPoolingMapper` instances from the sibling "
        "PercepT Stage 2 pilot were trained independently on the full "
        "61,402-painting patch cache, with 19 buddy outputs or 67 PercepT "
        "outputs. Both used Adam, learning rate 1e-3, 100 full-batch epochs, "
        "and **single-label cross-entropy**. This is a deliberate change "
        "from the original Stage 2 pilot's BCE-with-logits objective: both "
        "systems here supply one hard integer label per painting, whereas "
        "the original DEC targets were soft multi-hot. Cross-entropy gives "
        "these two systems the same hard-label task. Each frozen mapper's "
        "held-out softmax is predicted from image patches alone. The control "
        "is the untrained mean of each painting's 50 patch tokens (512-D).\n\n",
        "Each feature set got the same `LogisticRegression(max_iter=2000)` "
        "probe with default settings. Emotion used a seed-42 stratified 50/50 "
        f"split: {len(emotion_indices[0]):,} probe-train and "
        f"{len(emotion_indices[1]):,} probe-eval paintings. Genre used all "
        f"{genre_count} annotated paintings in seed-42 shuffled "
        f"`StratifiedKFold(n_splits={genre_splits})`, with "
        "`cross_val_predict` and pooled out-of-fold scoring. The smallest "
        f"genre class had {genre_min_count} paintings, so the actual "
        f"`n_splits` was {genre_splits} (the largest feasible value up to 5). "
        "AMI uses `pipeline.external_metrics`; accuracy is the fraction of "
        "correct predictions.\n\n",
        "## Results\n\n",
        "| feature set | emotion AMI | emotion accuracy | genre AMI | genre accuracy |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for name in names:
        score = results[name]
        lines.append(
            f"| {name} | {score['emotion_ami']:.4f} | "
            f"{score['emotion_accuracy']:.4f} | {score['genre_ami']:.4f} | "
            f"{score['genre_accuracy']:.4f} |\n"
        )
    lines.extend([
        "\nFinal full-batch train cross-entropy: buddy "
        f"**{losses['buddy']:.6f}**; PercepT **{losses['percept']:.6f}**.\n\n",
        "## Verdict\n\n",
        "The matched audit measured buddy occupancy at **0/19 empty, 3/19 "
        "below 1%** and PercepT occupancy at **21/67 empty, 50/67 below "
        "1%**. Those are frozen-topic occupancy numbers, not probe scores.\n\n",
    ])
    for outcome in ("emotion", "genre"):
        majority = baselines[outcome]
        control_ami = control[f"{outcome}_ami"]
        control_lift = control[f"{outcome}_accuracy"] - majority
        lines.append(
            f"**{outcome.capitalize()}:** mean-pooled control AMI "
            f"{control_ami:.4f}, accuracy {control[f'{outcome}_accuracy']:.4f}; "
            f"majority-class accuracy {majority:.4f}. "
        )
        for name, score in (("Buddy", buddy), ("PercepT", percept)):
            ami = score[f"{outcome}_ami"]
            lift = score[f"{outcome}_accuracy"] - majority
            if control_ami > 0 and control_lift > 0:
                ami_fraction = ami / control_ami
                lift_fraction = lift / control_lift
                retained = ami_fraction >= 0.5 and lift_fraction >= 0.5
                lines.append(
                    f"{name} retains {ami_fraction:.1%} of control AMI and "
                    f"{lift_fraction:.1%} of its accuracy lift over the "
                    f"majority class: {'most' if retained else 'less than most'} "
                    "by the stated half-of-both rule. "
                )
            else:
                lines.append(
                    f"{name} AMI {ami:.4f}, accuracy {score[f'{outcome}_accuracy']:.4f}; "
                    "control has no positive AMI and accuracy lift to retain. "
                )
        lines.append("\n\n")
    differences = {
        metric: buddy[metric] - percept[metric]
        for metric in (
            "emotion_ami", "emotion_accuracy", "genre_ami", "genre_accuracy"
        )
    }
    if all(value > 0 for value in differences.values()):
        direction = "Buddy leads PercepT on both metrics for both outcomes"
    elif all(value < 0 for value in differences.values()):
        direction = "PercepT leads buddy on both metrics for both outcomes"
    else:
        direction = "Neither bottleneck leads on both metrics for both outcomes"
    outcome_winners = {}
    for outcome in ("emotion", "genre"):
        ami_gap = differences[f"{outcome}_ami"]
        accuracy_gap = differences[f"{outcome}_accuracy"]
        if ami_gap > 0 and accuracy_gap > 0:
            outcome_winners[outcome] = "buddy"
        elif ami_gap < 0 and accuracy_gap < 0:
            outcome_winners[outcome] = "PercepT"
        else:
            outcome_winners[outcome] = "mixed"
    consistent = (
        outcome_winners["emotion"] == outcome_winners["genre"]
        and outcome_winners["emotion"] != "mixed"
    )
    lines.append(
        f"{direction}. Buddy minus PercepT is "
        f"{differences['emotion_ami']:+.4f} emotion AMI, "
        f"{differences['emotion_accuracy']:+.4f} emotion accuracy, "
        f"{differences['genre_ami']:+.4f} genre AMI, and "
        f"{differences['genre_accuracy']:+.4f} genre accuracy. "
        f"Emotion favors {outcome_winners['emotion']}; genre favors "
        f"{outcome_winners['genre']}. The direction is "
        f"{'consistent' if consistent else 'not consistent'} between outcomes. "
        "This single seed and single probe split establish the observed "
        "direction, but cannot establish reliability across mapper seeds or "
        "probe splits.\n"
    )
    REPORT.write_text("".join(lines), encoding="utf-8")


def main():
    """Align snapshots to caches, train mappers, and write the probe report."""
    stage2 = load_module(
        "percept_stage2_for_downstream_probe",
        PERCEPT_DIR / "run_percept_stage2_pilot.py",
    )
    audit = load_module(
        "matched_audit_for_downstream_probe",
        HERE / "run_buddy_percept_matched_silhouette_audit_pilot.py",
    )
    pipeline = load_module("pipeline_for_downstream_train", HERE / "run_pipeline.py")
    heldout_pipeline = load_module(
        "pipeline_for_downstream_heldout", HERE / "run_pipeline.py"
    )
    heldout_pipeline.STORAGE_DIR = stage2.base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = stage2.base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    train_paintings, _, _, train_emotion_counts = pipeline.load_dedup_features()
    heldout_paintings, _, _, heldout_emotion_counts = (
        heldout_pipeline.load_dedup_features()
    )
    if len(train_paintings) != N_TRAIN or len(heldout_paintings) != N_HELDOUT:
        raise RuntimeError("Fresh deduplicated painting counts differ from patch caches")
    log("Loading patch features in the fresh deduplicated painting order...")
    train_patches = stage2.load_patch_features(
        stage2.TRAIN_PATCH_FEATURE_PATH, len(train_paintings), "train"
    )
    heldout_patches = stage2.load_patch_features(
        stage2.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out"
    )
    log(f"Patch cache directory: {stage2.PATCH_FEATURE_DIR}")

    common_keys = (
        "train_paintings", "train_emotion", "train_genre",
        "heldout_paintings", "heldout_emotion", "heldout_genre", "seed",
    )
    buddy = load_snapshot(BUDDY_SNAPSHOT, (*common_keys, "train_community_post"))
    percept = load_snapshot(
        PERCEPT_SNAPSHOT,
        (*common_keys, "train_topic", "heldout_topic", "n_surviving_clusters"),
    )
    if int(buddy["seed"]) != SEED or int(percept["seed"]) != SEED:
        raise RuntimeError("Both snapshots must be seed-42 runs")
    if int(percept["n_surviving_clusters"]) != 67:
        raise RuntimeError("PercepT snapshot must have 67 surviving centers")
    buddy_train = align_snapshot(
        buddy, train_paintings, "train",
        ("train_community_post", "train_emotion", "train_genre"), audit,
    )
    percept_train = align_snapshot(
        percept, train_paintings, "train",
        ("train_topic", "train_emotion", "train_genre"), audit,
    )
    buddy_heldout = align_snapshot(
        buddy, heldout_paintings, "heldout",
        ("heldout_emotion", "heldout_genre"), audit,
    )
    percept_heldout = align_snapshot(
        percept, heldout_paintings, "heldout",
        ("heldout_topic", "heldout_emotion", "heldout_genre"), audit,
    )
    for split, canonical_counts, first, second in (
        ("train", train_emotion_counts, buddy_train, percept_train),
        ("heldout", heldout_emotion_counts, buddy_heldout, percept_heldout),
    ):
        canonical_emotions = np.asarray(
            [pipeline.majority(counts) for counts in canonical_counts]
        )
        for system, aligned in (("buddy", first), ("PercepT", second)):
            if not np.array_equal(aligned[f"{split}_emotion"], canonical_emotions):
                raise RuntimeError(f"{system} {split} emotions disagree with pipeline")
    genre_map = pipeline.load_genre_map()
    canonical_train_genres = np.asarray(
        [genre_map.get(painting, "") for painting in train_paintings]
    )
    canonical_genres = np.asarray(
        [genre_map.get(painting, "") for painting in heldout_paintings]
    )
    for system, aligned in (("buddy", buddy_train), ("PercepT", percept_train)):
        if not np.array_equal(aligned["train_genre"], canonical_train_genres):
            raise RuntimeError(f"{system} train genres disagree with pipeline")
    for system, aligned in (("buddy", buddy_heldout), ("PercepT", percept_heldout)):
        if not np.array_equal(aligned["heldout_genre"], canonical_genres):
            raise RuntimeError(f"{system} held-out genres disagree with pipeline")
    check_topic_labels(buddy_train["train_community_post"], 19, "buddy")
    check_topic_labels(percept_train["train_topic"], 67, "PercepT")
    if not np.array_equal(np.unique(buddy_train["train_community_post"]), np.arange(19)):
        raise RuntimeError("Buddy train vocabulary is not 0..18")
    if np.any(percept_heldout["heldout_topic"] < 0) or np.any(
        percept_heldout["heldout_topic"] >= 67
    ):
        raise RuntimeError("PercepT held-out topics fall outside 0..66")

    labels = {"emotion": buddy_heldout["heldout_emotion"], "genre": canonical_genres}
    emotion_indices = train_test_split(
        np.arange(N_HELDOUT), test_size=0.5, stratify=labels["emotion"],
        random_state=SEED,
    )
    genre_indices = np.flatnonzero(canonical_genres != "")
    genre_values, genre_counts = np.unique(
        labels["genre"][genre_indices], return_counts=True
    )
    if len(genre_values) < 2 or len(genre_indices) == 0:
        raise RuntimeError("Genre probe needs at least two annotated classes")
    genre_min_count = int(genre_counts.min())
    n_splits = min(5, genre_min_count)
    if n_splits < 2:
        raise RuntimeError("Stratified genre CV requires at least two per class")
    genre_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=SEED
    )
    log(f"Genre probe: {len(genre_indices)} paintings; {n_splits} folds")

    mean_pooled = heldout_patches.mean(dim=1).numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Training two full-batch image-only mappers on {device}...")
    train_patches = train_patches.to(device)
    features = {"mean-pooled control": mean_pooled}
    losses = {}
    mappers = {}
    for name, key, n_topics, loss_key in (
        ("buddy topic softmax", "train_community_post", 19, "buddy"),
        ("PercepT topic softmax", "train_topic", 67, "percept"),
    ):
        train_labels = buddy_train[key] if loss_key == "buddy" else percept_train[key]
        mappers[name], losses[loss_key] = train_mapper(
            stage2, train_patches, train_labels, n_topics, device, log, name
        )
    del train_patches
    heldout_patches = heldout_patches.to(device)
    for name, mapper in mappers.items():
        with torch.no_grad():
            features[name] = torch.softmax(
                mapper(heldout_patches), dim=-1
            ).cpu().numpy()
    del mappers

    results = {
        name: probe_scores(
            values, labels, emotion_indices, genre_indices, genre_cv, pipeline
        )
        for name, values in features.items()
    }
    emotion_eval = labels["emotion"][emotion_indices[1]]
    baselines = {
        "emotion": float(np.unique(emotion_eval, return_counts=True)[1].max())
        / len(emotion_eval),
        "genre": float(genre_counts.max()) / len(genre_indices),
    }
    write_report(
        results, losses, emotion_indices, len(genre_indices), n_splits,
        genre_min_count, baselines,
    )
    for name, scores in results.items():
        log(f"{name}: {scores}")
    log(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
