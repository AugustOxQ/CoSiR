"""Compare deployable buddy and faithful PercepT labels on matched paintings.

Reads frozen snapshots and writes the audit report; it does no training.
"""

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score


HERE = Path(__file__).resolve().parent
PERCEPT_DIR = HERE.parent / "20260922_percept_topic_pipeline"
BUDDY_SNAPSHOT = HERE / "attention_h1_embedding_snapshot.npz"
PERCEPT_SNAPSHOT = PERCEPT_DIR / "percept_stage1_faithful_recipe_snapshot.npz"
REPORT = HERE / "buddy_percept_matched_silhouette_audit_pilot_report.md"
EMOTION_PARETO_BAR = 0.1236
GENRE_PARETO_BAR = 0.1954


def load_module(module_name, path):
    """Load a sibling pilot's helpers without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_matching_paintings(buddy_paintings, percept_paintings, split, expected_count):
    """Require exactly the same unique painting ids in both snapshots."""
    buddy_ids = set(np.asarray(buddy_paintings).tolist())
    percept_ids = set(np.asarray(percept_paintings).tolist())
    if len(buddy_paintings) != expected_count or len(percept_paintings) != expected_count:
        raise RuntimeError(
            f"{split} painting count mismatch: buddy={len(buddy_paintings)}, "
            f"PercepT={len(percept_paintings)}, expected={expected_count}."
        )
    if len(buddy_ids) != expected_count or len(percept_ids) != expected_count:
        raise RuntimeError(f"{split} painting ids are not unique in both snapshots.")
    if buddy_ids != percept_ids:
        raise RuntimeError(
            f"{split} painting sets differ: buddy-only={len(buddy_ids - percept_ids)}, "
            f"PercepT-only={len(percept_ids - buddy_ids)}."
        )


def occupancy(labels, vocabulary):
    """Count every train label, including those empty on held-out data."""
    labels = np.asarray(labels)
    vocabulary = np.asarray(vocabulary)
    if labels.ndim != 1 or vocabulary.ndim != 1:
        raise ValueError("Labels and vocabulary must be one-dimensional.")
    if not np.isin(labels, vocabulary).all():
        raise RuntimeError("Held-out assignment contains a label outside the train vocabulary.")
    counts = np.asarray([np.count_nonzero(labels == label) for label in vocabulary])
    below_one_percent = int(np.sum(counts < 0.01 * len(labels)))
    return {
        "vocabulary": vocabulary,
        "counts": counts,
        "min": int(counts.min()),
        "max": int(counts.max()),
        "median": float(np.median(counts)),
        "zero": int(np.sum(counts == 0)),
        "below_one_percent": below_one_percent,
        "collapsed": below_one_percent > len(vocabulary) / 2,
    }


def score(pipeline, genre_map, paintings, emotions, saved_genres,
          embeddings, labels, vocabulary, sample_indices):
    """Use the same external labels and silhouette protocol for each system."""
    if len(embeddings) != len(paintings) or len(labels) != len(paintings):
        raise RuntimeError("Embeddings, labels, and paintings are not row-aligned.")
    if len(emotions) != len(paintings) or len(saved_genres) != len(paintings):
        raise RuntimeError("External labels and paintings are not row-aligned.")
    emotion_ami = float(pipeline.external_metrics(labels, emotions)["AMI"])
    genre_indices = np.asarray(
        [index for index, painting in enumerate(paintings) if painting in genre_map],
        dtype=np.int64,
    )
    if not len(genre_indices):
        raise RuntimeError("No genre-labelled paintings overlap the held-out split.")
    genres = np.asarray([genre_map[paintings[index]] for index in genre_indices])
    if not np.array_equal(saved_genres[genre_indices], genres):
        raise RuntimeError("Saved held-out genre labels disagree with the pipeline genre map.")
    genre_ami = float(pipeline.external_metrics(labels[genre_indices], genres)["AMI"])
    sampled_labels = labels[sample_indices]
    if 1 < len(np.unique(sampled_labels)) < len(sampled_labels):
        silhouette = float(silhouette_score(
            embeddings[sample_indices], sampled_labels,
            sample_size=min(4000, len(sample_indices)), random_state=42,
        ))
    else:
        silhouette = float("nan")
    return {
        "emotion_ami": emotion_ami,
        "genre_ami": genre_ami,
        "genre_count": len(genre_indices),
        "silhouette": silhouette,
        "emotion_bar": emotion_ami > EMOTION_PARETO_BAR,
        "genre_bar": genre_ami > GENRE_PARETO_BAR,
        "occupancy": occupancy(labels, vocabulary),
    }


def write_report(buddy, percept, heldout_count, sample_count,
                 saved_percept_full_silhouette):
    """Write the matched-protocol results after both snapshots are scored."""
    def status(value):
        return "pass" if value else "fail"

    def silhouette_text(value):
        return f"{value:.4f}" if np.isfinite(value) else "n/a (one cluster)"

    lines = [
        "# Buddy versus PercepT: matched held-out silhouette and occupancy audit\n\n",
        "## Population and protocol\n\n",
        "Painting-set identity **passed**: both snapshots contain the same "
        "61,402 unique train paintings and 9,365 unique held-out paintings. "
        "The PercepT held-out rows were aligned to buddy's painting order "
        "before a common seed-42 sample was drawn.\n\n",
        "Buddy labels are k=20 cosine-neighbor majority votes onto its "
        "19 frozen train Leiden communities, using the unchanged "
        "`assign_to_train_communities` helper. PercepT labels are its saved "
        "held-out assignments to 67 surviving train-fitted DEC centers. "
        "The independently reclustered buddy held-out communities were not "
        "used.\n\n",
        f"Both silhouettes use the same {sample_count:,} held-out paintings "
        f"out of {heldout_count:,}: `np.random.default_rng(42).choice` "
        "followed by `silhouette_score(sample_size=min(4000, len(idx)), "
        "random_state=42)` with default Euclidean distance. "
        "Scores are still measured in each model's own space: 32-D buddy "
        "fused embeddings versus 128-D PercepT post-DEC latents. Thus the "
        "sampling and distance rule are matched, while the representation "
        "geometry is not fully controlled.\n\n",
        "## Combined held-out comparison\n\n",
        "The predeclared AMI bar requires emotion > 0.1236 and genre > "
        "0.1954, both strict. Collapse means more than half of train labels "
        "receive fewer than 1% of held-out paintings. Zero occupancy is "
        "counted separately.\n\n",
        "| system | emotion AMI / bar | genre AMI / bar | both AMI bars | "
        "silhouette | occupancy min / median / max | zero labels | "
        "below 1% | collapse rule |\n",
        "|---|---:|---:|---|---:|---:|---:|---:|---|\n",
    ]
    for name, result in (("Buddy Attention-h1", buddy), ("PercepT Variant A", percept)):
        sizes = result["occupancy"]
        lines.append(
            f"| {name} | {result['emotion_ami']:.4f} / "
            f"{status(result['emotion_bar'])} | {result['genre_ami']:.4f} / "
            f"{status(result['genre_bar'])} | "
            f"{status(result['emotion_bar'] and result['genre_bar'])} | "
            f"{silhouette_text(result['silhouette'])} | "
            f"{sizes['min']:,} / {sizes['median']:.1f} / {sizes['max']:,} | "
            f"{sizes['zero']}/{len(sizes['vocabulary'])} | "
            f"{sizes['below_one_percent']}/{len(sizes['vocabulary'])} | "
            f"{'Collapsed' if sizes['collapsed'] else 'not collapsed'} |\n"
        )
    lines.extend([
        "\nEmotion AMI uses all held-out paintings. Genre AMI uses the "
        f"{buddy['genre_count']} paintings with genre annotation for buddy "
        f"and {percept['genre_count']} for PercepT; saved genre values were "
        "checked against the pipeline genre map.\n\n",
        "## Full held-out occupancy histograms\n\n",
    ])
    for name, result in (("Buddy: frozen train communities", buddy),
                         ("PercepT: surviving train centers", percept)):
        sizes = result["occupancy"]
        lines.extend([
            f"### {name}\n\n",
            f"Min {sizes['min']:,}; max {sizes['max']:,}; median "
            f"{sizes['median']:.1f}; zero-count {sizes['zero']}; "
            f"below 1% {sizes['below_one_percent']}/"
            f"{len(sizes['vocabulary'])}.\n\n",
            "| train label | held-out count |\n",
            "|---:|---:|\n",
        ])
        for label, count in zip(sizes["vocabulary"], sizes["counts"]):
            lines.append(f"| {int(label)} | {int(count):,} |\n")
        lines.append("\n")

    lines.extend([
        "## Interpretation and verdict\n\n",
        "The published faithful-recipe Variant-A seed-42 run had held-out "
        "emotion AMI **0.1092**, genre AMI **0.3288**, and full-split "
        "silhouette **0.5120**. Its verdict was **Collapsed**: **50/67** "
        "surviving held-out centers were below 1% occupancy, with minimum "
        "surviving-center count **0**. It missed the project's own joint "
        "AMI Pareto bar as well as its non-collapse criterion. The matched "
        "sampled silhouette above is a newly computed value and need not "
        "equal the published full-split 0.5120.\n\n",
        f"The faithful snapshot records full-split held-out silhouette "
        f"**{saved_percept_full_silhouette:.4f}**. "
        + (
            "This agrees with the published 0.5120 at four decimals.\n\n"
            if np.isfinite(saved_percept_full_silhouette)
            and abs(saved_percept_full_silhouette - 0.5120) <= 0.00005
            else "**WARNING:** this differs from the published 0.5120 "
                 "beyond four-decimal rounding; inspect the snapshot run.\n\n"
        ),
        "The earlier held-out label-transfer pilot reported **19/19** "
        "buddy train communities covered at k=20; the per-label counts "
        "above are recomputed from that same validated transfer rule.\n\n",
    ])
    if np.isfinite(percept["silhouette"]) and np.isfinite(buddy["silhouette"]):
        if percept["silhouette"] > buddy["silhouette"]:
            lines.append(
                f"PercepT retains a numerical silhouette advantage under "
                f"matched sampling ({percept['silhouette']:.4f} versus "
                f"{buddy['silhouette']:.4f}). "
            )
        else:
            lines.append(
                "PercepT does not retain a numerical silhouette advantage "
                "under matched sampling. "
            )
    else:
        lines.append("A matched silhouette is undefined for at least one system. ")
    percept_sizes = percept["occupancy"]
    buddy_sizes = buddy["occupancy"]
    lines.append(
        f"The occupancy picture is different: PercepT has "
        f"{percept_sizes['zero']} zero-occupancy surviving centers and "
        f"{percept_sizes['below_one_percent']}/"
        f"{len(percept_sizes['vocabulary'])} below 1%, versus buddy's "
        f"{buddy_sizes['zero']} zero-occupancy communities and "
        f"{buddy_sizes['below_one_percent']}/"
        f"{len(buddy_sizes['vocabulary'])} below 1%. "
        "This means a larger silhouette alone is weak evidence of better "
        "topics: part of the gap is consistent with degenerate, "
        "over-confident clustering rather than better coverage or human-label "
        "agreement. Occupancy and silhouette do not prove the cause of that "
        "gap.\n\n",
    )
    lines.append(
        "This audit does **not** test whether either system's labels are "
        "more useful downstream. Candidate 1's shared image-only mapper "
        "probe is deliberately out of scope for this pilot and remains a "
        "follow-up decision.\n"
    )
    REPORT.write_text("".join(lines), encoding="utf-8")


def main():
    if not BUDDY_SNAPSHOT.is_file() or not PERCEPT_SNAPSHOT.is_file():
        raise FileNotFoundError(
            f"Both snapshots are required: {BUDDY_SNAPSHOT} and {PERCEPT_SNAPSHOT}"
        )
    with np.load(BUDDY_SNAPSHOT, allow_pickle=True) as source:
        buddy = {key: source[key] for key in (
            "train_paintings", "train_embedding_post", "train_community_post",
            "heldout_paintings", "heldout_embedding_post", "heldout_emotion",
            "heldout_genre", "seed",
        )}
    with np.load(PERCEPT_SNAPSHOT, allow_pickle=True) as source:
        percept = {key: source[key] for key in (
            "train_paintings", "heldout_paintings", "heldout_latent",
            "heldout_topic", "heldout_emotion", "heldout_genre",
            "n_initial_clusters", "n_surviving_clusters", "seed",
            "held_out_silhouette",
        )}

    # Check populations before comparing scores or assigning buddy labels.
    assert_matching_paintings(
        buddy["train_paintings"], percept["train_paintings"], "train", 61402
    )
    assert_matching_paintings(
        buddy["heldout_paintings"], percept["heldout_paintings"], "held-out", 9365
    )
    if int(buddy["seed"]) != 42 or int(percept["seed"]) != 42:
        raise RuntimeError("Both snapshots must be the seed-42 runs.")
    if int(percept["n_initial_clusters"]) != 100 or int(percept["n_surviving_clusters"]) != 67:
        raise RuntimeError("PercepT snapshot does not contain the faithful 100/67 recipe.")

    # A common index sample only means common paintings after row alignment.
    percept_positions = {
        painting: index for index, painting in enumerate(percept["heldout_paintings"])
    }
    aligned = np.asarray(
        [percept_positions[painting] for painting in buddy["heldout_paintings"]],
        dtype=np.int64,
    )
    for key in ("heldout_paintings", "heldout_latent", "heldout_topic",
                "heldout_emotion", "heldout_genre"):
        percept[key] = percept[key][aligned]
    if not np.array_equal(buddy["heldout_paintings"], percept["heldout_paintings"]):
        raise RuntimeError("Held-out row alignment failed after painting-set assertion.")
    if not np.array_equal(buddy["heldout_emotion"], percept["heldout_emotion"]):
        raise RuntimeError("Held-out emotion labels disagree for matched paintings.")
    if not np.array_equal(buddy["heldout_genre"], percept["heldout_genre"]):
        raise RuntimeError("Held-out genre labels disagree for matched paintings.")
    if buddy["heldout_embedding_post"].shape != (9365, 32):
        raise RuntimeError("Buddy held-out embeddings must have shape (9365, 32).")
    if percept["heldout_latent"].shape != (9365, 128):
        raise RuntimeError("PercepT held-out latents must have shape (9365, 128).")

    transfer = load_module(
        "artelingo_heldout_label_transfer_for_matched_audit",
        HERE / "run_heldout_label_transfer_pilot.py",
    )
    pipeline = load_module(
        "artelingo_run_pipeline_matched_audit", HERE / "run_pipeline.py"
    )
    buddy_labels = transfer.assign_to_train_communities(
        train_embeddings=buddy["train_embedding_post"],
        train_community=buddy["train_community_post"],
        query_embeddings=buddy["heldout_embedding_post"],
        k=20,
    )
    buddy_vocabulary = np.unique(buddy["train_community_post"])
    if len(buddy_vocabulary) != 19:
        raise RuntimeError(f"Expected 19 frozen buddy train communities, got {len(buddy_vocabulary)}.")
    percept_vocabulary = np.arange(int(percept["n_surviving_clusters"]))
    heldout_count = len(buddy["heldout_paintings"])
    sample_indices = np.random.default_rng(42).choice(
        heldout_count, size=min(6000, heldout_count), replace=False
    )
    genre_map = pipeline.load_genre_map()
    buddy_result = score(
        pipeline, genre_map, buddy["heldout_paintings"], buddy["heldout_emotion"],
        buddy["heldout_genre"], buddy["heldout_embedding_post"], buddy_labels,
        buddy_vocabulary, sample_indices,
    )
    percept_result = score(
        pipeline, genre_map, percept["heldout_paintings"], percept["heldout_emotion"],
        percept["heldout_genre"], percept["heldout_latent"], percept["heldout_topic"],
        percept_vocabulary, sample_indices,
    )
    write_report(
        buddy_result, percept_result, heldout_count, len(sample_indices),
        float(percept["held_out_silhouette"]),
    )
    for name, result in (("Buddy", buddy_result), ("PercepT", percept_result)):
        print(
            f"{name}: emotion AMI={result['emotion_ami']:.4f}, "
            f"genre AMI={result['genre_ami']:.4f}, "
            f"matched silhouette={result['silhouette']:.4f}, "
            f"below 1%={result['occupancy']['below_one_percent']}/"
            f"{len(result['occupancy']['vocabulary'])}, "
            f"zero={result['occupancy']['zero']}"
        )
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
