"""Measure the ArtELingo-native BERT ceiling on held-out val+test paintings.

Run manually in a GPU-capable environment.  This redirects the reusable
pipeline helpers to the held-out feature store and JSON before loading any
data, so it is a generalization check rather than a train-split ceiling.
"""

import importlib.util
import os
import sys
import time

import torch


OUT_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
SINGLE_MODALITY_PATH = os.path.join(OUT_DIR, "run_single_modality_pilot.py")
AFFECT_PILOT_PATH = os.path.join(OUT_DIR, "run_affect_pilot.py")
BERT_CEILING_PATH = os.path.join(OUT_DIR, "run_bert_ceiling_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "bert_heldout_pilot_report.md")

HELDOUT_STORAGE_DIR = "/data/SSD2/pre_extract/artelingo_heldout/features"
HELDOUT_JSON = "/data/PDD/artelingo/artelingo_val_test.json"
HELDOUT_ROWS = 46_813
HELDOUT_PAINTINGS = 9_365
SEED = 42

GOEMOTIONS_EMOTION_AMI = 0.1180
GOEMOTIONS_GENRE_AMI = 0.0396
TRAIN_CAPTION_ACCURACY = 0.9367
TRAIN_EMOTION_AMI = 0.2897
TRAIN_GENRE_AMI = 0.0582
TRAIN_LABEL_MAPPING = {
    0: "amusement",
    1: "awe",
    2: "contentment",
    3: "excitement",
    4: "anger",
    5: "disgust",
    6: "fear",
    7: "sadness",
    8: "something else",
}

REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.conditional_buddy.prototype_seed import detect_communities


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_sibling_module(module_name: str, path: str):
    """Import a sibling standalone script without executing its ``main`` block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mapping_matches_train(heldout_mapping: dict) -> bool:
    """Return whether every recovered held-out checkpoint label matches train."""
    return all(
        heldout_mapping[index]["emotion"] == TRAIN_LABEL_MAPPING[index]
        for index in range(9)
    )


def write_report(
    caption_accuracy: dict,
    emotion_metrics: dict,
    genre_metrics: dict,
    genre_count: int,
    n_paintings: int,
    k: int,
    alpha: float,
) -> None:
    """Write held-out results with leakage and supervision-scope conclusions."""
    emotion_ami = emotion_metrics["AMI"]
    accuracy_drop = TRAIN_CAPTION_ACCURACY - caption_accuracy["accuracy"]
    emotion_ami_drop = TRAIN_EMOTION_AMI - emotion_ami
    labels_match = mapping_matches_train(caption_accuracy["mapping"])
    mapping_statement = (
        "matches the train-run recovered mapping label-for-label"
        if labels_match
        else "does not match the train-run recovered mapping label-for-label"
    )
    mostly_generalization = accuracy_drop <= 0.05 and emotion_ami_drop <= 0.05
    generalization_statement = (
        "This supports mostly genuine generalization: both held-out drops are small "
        "under the predeclared 5 percentage-point / 0.05-AMI practical thresholds."
        if mostly_generalization
        else "This supports mostly memorization (or a material generalization gap): at "
        "least one held-out drop exceeds the 5 percentage-point / 0.05-AMI practical "
        "thresholds, rather than remaining close to the train-split result."
    )

    lines = [
        "# ArtELingo-native BERT ceiling on held-out val+test data\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Held-out setup\n\n",
        f"This run uses ArtELingo val+test combined: **{HELDOUT_ROWS:,} rows** and "
        f"**{HELDOUT_PAINTINGS:,} unique paintings** (loaded as {n_paintings:,} "
        "deduplicated nodes). The set has verified **zero painting-level overlap** "
        "with the train split. The same ArtELingo-native BERT checkpoint is evaluated "
        "on these held-out captions, then its mean-pooled 9-way softmax vectors form a "
        f"single-modality mutual-kNN graph (K={k}), repaired with alpha={alpha}, with "
        f"Leiden seed={SEED}.\n\n",
        "## Per-caption classifier sanity check\n\n",
        "Anonymous checkpoint indices are re-mapped independently on held-out captions "
        "by empirical per-index majority vote; the train mapping is not reused for the "
        f"accuracy calculation. The recovered held-out mapping {mapping_statement}.\n\n",
        "| checkpoint index | held-out recovered emotion | supporting caption rows | "
        "train recovered emotion |\n",
        "|---|---|---:|---|\n",
    ]
    for index in range(9):
        recovered = caption_accuracy["mapping"][index]
        lines.append(
            f"| LABEL_{index} | {recovered['emotion'] or 'unobserved'} | "
            f"{recovered['support']:,} | {TRAIN_LABEL_MAPPING[index]} |\n"
        )
    lines.extend([
        f"\n**Held-out recovered-mapping top-1 accuracy: "
        f"{caption_accuracy['accuracy']:.2%} ({caption_accuracy['correct']:,}/"
        f"{caption_accuracy['total']:,} English caption rows).**\n\n",
        "## Comparison\n\n",
        "Genre metrics use only paintings that overlap the genre-labelled diagnostic "
        f"set. The held-out genre subset has n={genre_count:,}, so this diagnostic has "
        "low sample size and low statistical power here; unlike emotion, it does not "
        "use the full held-out graph.\n\n",
        "| signal / split | per-caption accuracy | emotion AMI | emotion V-measure | "
        "genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        "| GoEmotions-only (off-the-shelf, out-of-domain) | — | 0.1180 | — | 0.0396 | — |\n",
        "| ArtELingo-native BERT, TRAIN split (likely leaky) | 93.67% | 0.2897 | — | "
        "0.0582 | — |\n",
        f"| ArtELingo-native BERT, HELD-OUT val+test (this run; genre n={genre_count:,}) | "
        f"{caption_accuracy['accuracy']:.2%} | {emotion_ami:.4f} | "
        f"{emotion_metrics['V_measure']:.4f} | {genre_metrics['AMI']:.4f} | "
        f"{genre_metrics['V_measure']:.4f} |\n",
        "\n## Conclusion\n\n",
        f"Compared with the likely-leaky train split, held-out caption accuracy changed "
        f"by {accuracy_drop:+.2%} ({TRAIN_CAPTION_ACCURACY:.2%} to "
        f"{caption_accuracy['accuracy']:.2%}) and held-out emotion AMI changed by "
        f"{emotion_ami_drop:+.4f} ({TRAIN_EMOTION_AMI:.4f} to {emotion_ami:.4f}). "
        f"{generalization_statement}\n\n",
        "Regardless of that generalization result, this is evidence only that a "
        "well-matched **supervised** emotion signal can survive routing through "
        "buddy-graph clustering. It is not evidence that buddy-graph/DEC can discover "
        "affect structure **unsupervised** from raw CLIP features. That scope limit is "
        "important because RedCaps, CoSiR's other main dataset, has no emotion labels "
        "with which to supervise a comparable classifier; even a perfectly strong "
        "ArtELingo held-out result therefore would not transfer to RedCaps.\n",
    ])
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    pipeline = load_sibling_module("artelingo_run_pipeline_heldout", PIPELINE_PATH)
    pipeline.STORAGE_DIR = HELDOUT_STORAGE_DIR
    pipeline.TRAIN_JSON = HELDOUT_JSON
    single_modality = load_sibling_module(
        "artelingo_run_single_modality_pilot_heldout", SINGLE_MODALITY_PATH
    )
    affect_pilot = load_sibling_module(
        "artelingo_run_affect_pilot_heldout", AFFECT_PILOT_PATH
    )
    bert_ceiling = load_sibling_module(
        "artelingo_run_bert_ceiling_pilot_heldout", BERT_CEILING_PATH
    )

    log("Verifying held-out feature extraction is complete...")
    pipeline.assert_extraction_complete()
    log("Loading and deduplicating held-out cached features for painting order and labels...")
    paintings, _img_nodes, _txt_nodes, emotion_counts = pipeline.load_dedup_features()
    if len(paintings) != HELDOUT_PAINTINGS:
        raise RuntimeError(
            f"Expected {HELDOUT_PAINTINGS:,} held-out paintings, got {len(paintings):,}."
        )
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for held-out BERT extraction and graph construction.")

    caption_accuracy = {}
    bert_nodes = bert_ceiling.extract_bert_ceiling_nodes(
        HELDOUT_JSON, paintings, device, caption_accuracy
    )
    if bert_nodes.shape != (len(paintings), 9):
        raise RuntimeError(
            f"Expected BERT nodes with shape ({len(paintings)}, 9), got {bert_nodes.shape}."
        )

    graph = single_modality.build_single_modality_graph(
        "ArtELingo-native-BERT-heldout-only",
        bert_nodes,
        pipeline,
        affect_pilot,
        device,
        expected_nodes=len(paintings),
    )
    log(f"ArtELingo-native-BERT-heldout-only: running Leiden (seed={SEED})...")
    community = detect_communities(graph, seed=SEED)
    emotion_metrics = pipeline.external_metrics(community, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [i for i, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the held-out node set.")
    genre_metrics = pipeline.external_metrics(
        [community[i] for i in genre_indices],
        [genre_map[paintings[i]] for i in genre_indices],
    )
    log(
        f"ArtELingo-native-BERT-heldout-only: emotion AMI={emotion_metrics['AMI']:.4f}, "
        f"genre AMI={genre_metrics['AMI']:.4f} (genre n={len(genre_indices):,})."
    )
    write_report(
        caption_accuracy,
        emotion_metrics,
        genre_metrics,
        len(genre_indices),
        len(paintings),
        pipeline.K,
        pipeline.ALPHA,
    )
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
