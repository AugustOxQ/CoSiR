"""Measure the ArtELingo-native BERT emotion-signal buddy-graph ceiling.

Run manually in a GPU-capable environment. This script intentionally evaluates
the ArtELingo authors' checkpoint on nodes from its likely training split, so
the resulting ceiling can include memorization and is not a generalization test.
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


OUT_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
SINGLE_MODALITY_PATH = os.path.join(OUT_DIR, "run_single_modality_pilot.py")
AFFECT_PILOT_PATH = os.path.join(OUT_DIR, "run_affect_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "bert_ceiling_pilot_report.md")
MODEL_PATH = (
    "/data/PDD/artelingo/ArtELingo/ArtELingo/saved_models/Emotion Prediction/"
    "single_head/bert_english"
)
TOKENIZER_NAME = "bert-base-uncased"
BATCH_SIZE = 256
MAX_LENGTH = 64
SEED = 42
GOEMOTIONS_EMOTION_AMI = 0.1180
GOEMOTIONS_GENRE_AMI = 0.0396

REPO_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.conditional_buddy.prototype_seed import detect_communities


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_sibling_module(module_name: str, path: str):
    """Import a sibling standalone script without running its ``main`` block."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def caption_text(record: dict) -> str:
    """Match ArtELingo's string-or-singleton-list caption convention."""
    caption = record["caption"]
    return caption if isinstance(caption, str) else caption[0]


def recover_caption_accuracy(predicted_indices: list[int], true_emotions: list[str]) -> dict:
    """Recover the checkpoint's anonymized label names by per-index majority vote."""
    if len(predicted_indices) != len(true_emotions):
        raise ValueError("Prediction and ground-truth counts differ.")

    supports = {}
    for predicted_index, emotion in zip(predicted_indices, true_emotions):
        supports.setdefault(predicted_index, Counter())[emotion] += 1

    mapping = {}
    mapped_correct = 0
    for predicted_index in range(9):
        counts = supports.get(predicted_index, Counter())
        if not counts:
            mapping[predicted_index] = {"emotion": None, "support": 0}
            continue
        emotion, support = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        mapping[predicted_index] = {"emotion": emotion, "support": support}
        mapped_correct += support

    return {
        "mapping": mapping,
        "correct": mapped_correct,
        "total": len(true_emotions),
        "accuracy": mapped_correct / len(true_emotions),
    }


def extract_bert_ceiling_nodes(
    train_json_path: str,
    paintings: list[str],
    device: str,
    caption_accuracy: dict | None = None,
) -> np.ndarray:
    """Mean-pool 9-way BERT softmax vectors by painting in pipeline order.

    When supplied, ``caption_accuracy`` is populated with the empirically
    recovered anonymous-label mapping and caption-level top-1 accuracy. The
    returned value is always a ``(num_paintings, 9)`` probability matrix.
    """
    log(f"Loading ArtELingo-native BERT checkpoint from {MODEL_PATH} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    if model.config.num_labels != 9:
        raise RuntimeError(
            f"Expected the ArtELingo checkpoint to have 9 labels, got {model.config.num_labels}."
        )

    with open(train_json_path) as train_file:
        train = json.load(train_file)

    painting_to_idx = {painting: index for index, painting in enumerate(paintings)}
    row_node_indices = []
    captions = []
    true_emotions = []
    for record in train:
        if record.get("language", "english").lower() != "english":
            continue
        try:
            row_node_indices.append(painting_to_idx[record["painting"]])
        except KeyError as exc:
            raise RuntimeError(
                "Training row references painting absent from deduplicated features: "
                f"{record['painting']}"
            ) from exc
        captions.append(caption_text(record))
        true_emotions.append(record["emotion"])

    if not captions:
        raise RuntimeError("No English caption rows found in TRAIN_JSON.")

    total_batches = (len(captions) + BATCH_SIZE - 1) // BATCH_SIZE
    log(
        f"Running per-caption BERT classification sanity check for {len(captions):,} "
        f"English rows in {total_batches} batches (batch_size={BATCH_SIZE}, "
        f"max_length={MAX_LENGTH})..."
    )
    probability_batches = []
    predicted_indices = []
    with torch.no_grad():
        for start in range(0, len(captions), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(captions))
            encoded = tokenizer(
                captions[start:end],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            probabilities = (
                torch.softmax(model(**encoded).logits, dim=-1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            probability_batches.append(probabilities)
            predicted_indices.extend(probabilities.argmax(axis=1).tolist())
            if (start // BATCH_SIZE + 1) % 100 == 0 or end == len(captions):
                log(f"BERT classification progress: {end:,}/{len(captions):,} caption rows.")

    accuracy = recover_caption_accuracy(predicted_indices, true_emotions)
    log("Per-caption classification sanity check (before painting pooling):")
    for predicted_index in range(9):
        recovered = accuracy["mapping"][predicted_index]
        log(
            f"  LABEL_{predicted_index} -> {recovered['emotion'] or 'unobserved'} "
            f"(support={recovered['support']:,})"
        )
    log(
        f"Recovered-mapping top-1 accuracy: {accuracy['accuracy']:.4%} "
        f"({accuracy['correct']:,}/{accuracy['total']:,})."
    )
    if caption_accuracy is not None:
        caption_accuracy.update(accuracy)

    bert_sums = np.zeros((len(paintings), 9), dtype=np.float32)
    bert_counts = np.zeros(len(paintings), dtype=np.int32)
    for batch_index, probabilities in enumerate(probability_batches):
        start = batch_index * BATCH_SIZE
        end = start + len(probabilities)
        node_indices = np.asarray(row_node_indices[start:end], dtype=np.intp)
        np.add.at(bert_sums, node_indices, probabilities)
        np.add.at(bert_counts, node_indices, 1)

    missing = np.flatnonzero(bert_counts == 0)
    if len(missing):
        raise RuntimeError(f"{len(missing)} deduplicated paintings have no English BERT vectors.")
    return bert_sums / bert_counts[:, None]


def evaluate_graph(pipeline, community: np.ndarray, paintings: list[str], majority_emotion: list[str]):
    """Evaluate full-graph emotion and genre-subset labels with pipeline helpers."""
    emotion_metrics = pipeline.external_metrics(community, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [index for index, painting in enumerate(paintings) if painting in genre_map]
    if not genre_indices:
        raise RuntimeError("No genre-labelled paintings overlap the deduplicated node set.")
    genre_metrics = pipeline.external_metrics(
        [community[index] for index in genre_indices],
        [genre_map[paintings[index]] for index in genre_indices],
    )
    return emotion_metrics, genre_metrics, len(genre_indices)


def write_report(
    caption_accuracy: dict,
    emotion_metrics: dict,
    genre_metrics: dict,
    genre_count: int,
    n_paintings: int,
    k: int,
    alpha: float,
) -> None:
    """Write the leakage-qualified caption and graph ceiling results."""
    emotion_ami = emotion_metrics["AMI"]
    if emotion_ami <= 0.15:
        interpretation = (
            "This run supports the method-bottleneck reading: despite a high direct "
            "per-caption classification result and in-domain training, the graph ceiling "
            "remains modest (in the roughly 0.12–0.15 range)."
        )
    else:
        interpretation = (
            "This run supports the signal-quality/domain-mismatch reading: the in-domain "
            "BERT graph ceiling is substantially above the GoEmotions-only ceiling, so "
            "the out-of-domain affect signal was likely a main limiter."
        )

    lines = [
        "# ArtELingo-native BERT emotion-classifier ceiling pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Important interpretation caveat: likely train-split leakage\n\n",
        "**This BERT checkpoint was almost certainly trained on ArtELingo's official "
        "train split, the same split used to build `artelingo_train.json` and these "
        "painting nodes. Its caption accuracy and graph ceiling may therefore partly "
        "reflect memorization rather than generalization. This is not a fair "
        "apples-to-apples generalization comparison with GoEmotions. It answers the "
        "narrower question: given the best available in-domain emotion signal, what "
        "ceiling does buddy-graph/Leiden clustering reach at all?**\n\n",
        "## Per-caption classifier sanity check\n\n",
        "The anonymous `LABEL_n` outputs were empirically mapped by assigning each "
        "predicted index to its most common ground-truth emotion among captions with "
        "that prediction. This check occurs before painting-level pooling.\n\n",
        "| checkpoint index | recovered emotion | supporting caption rows |\n",
        "|---|---|---:|\n",
    ]
    for predicted_index in range(9):
        recovered = caption_accuracy["mapping"][predicted_index]
        lines.append(
            f"| LABEL_{predicted_index} | {recovered['emotion'] or 'unobserved'} | "
            f"{recovered['support']:,} |\n"
        )
    lines.extend([
        f"\n**Recovered-mapping top-1 accuracy: {caption_accuracy['accuracy']:.2%} "
        f"({caption_accuracy['correct']:,}/{caption_accuracy['total']:,} English caption rows).**\n\n",
        "## Graph-based ceiling\n\n",
        f"Setup: {n_paintings:,} deduplicated painting nodes, mean-pooled 9-way BERT "
        f"softmax probabilities, single-modality mutual-kNN graph (K={k}), graph "
        f"repair with alpha={alpha}, and Leiden seed={SEED}. Genre metrics use the "
        f"{genre_count:,}-painting genre-labelled overlap.\n\n",
        "| signal / ceiling | emotion AMI | emotion V-measure | genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|\n",
        "| GoEmotions-only (off-the-shelf, out-of-domain) Leiden ceiling | 0.1180 | — | 0.0396 | — |\n",
        f"| ArtELingo-native BERT (in-domain, train-set, likely leaky) Leiden ceiling | "
        f"{emotion_ami:.4f} | {emotion_metrics['V_measure']:.4f} | "
        f"{genre_metrics['AMI']:.4f} | {genre_metrics['V_measure']:.4f} |\n",
        "\n## Interpretation\n\n",
        "If this ceiling is also modest (similar to 0.12–0.15) despite high per-caption "
        "accuracy and in-domain training, the clustering method—not signal quality—is "
        "the likely bottleneck. If it is substantially higher, the GoEmotions result was "
        "mainly limited by domain mismatch and signal quality. "
        f"{interpretation}\n",
    ])
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    pipeline = load_sibling_module("artelingo_run_pipeline", PIPELINE_PATH)
    single_modality = load_sibling_module(
        "artelingo_run_single_modality_pilot", SINGLE_MODALITY_PATH
    )
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot", AFFECT_PILOT_PATH)

    pipeline.assert_extraction_complete()
    log("Loading and deduplicating cached CLIP features for painting order and labels...")
    paintings, _img_nodes, _txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for BERT extraction and single-modality graph construction.")

    caption_accuracy = {}
    bert_nodes = extract_bert_ceiling_nodes(
        pipeline.TRAIN_JSON, paintings, device, caption_accuracy
    )
    if bert_nodes.shape != (len(paintings), 9):
        raise RuntimeError(
            f"Expected BERT nodes with shape ({len(paintings)}, 9), got {bert_nodes.shape}."
        )

    graph = single_modality.build_single_modality_graph(
        "ArtELingo-native-BERT-only",
        bert_nodes,
        pipeline,
        affect_pilot,
        device,
        expected_nodes=len(paintings),
    )
    log(f"ArtELingo-native-BERT-only: running Leiden community detection (seed={SEED})...")
    community = detect_communities(graph, seed=SEED)
    emotion_metrics, genre_metrics, genre_count = evaluate_graph(
        pipeline, community, paintings, majority_emotion
    )
    log(
        f"ArtELingo-native-BERT-only: emotion AMI={emotion_metrics['AMI']:.4f}, "
        f"genre AMI={genre_metrics['AMI']:.4f}."
    )
    write_report(
        caption_accuracy,
        emotion_metrics,
        genre_metrics,
        genre_count,
        len(paintings),
        pipeline.K,
        pipeline.ALPHA,
    )
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
