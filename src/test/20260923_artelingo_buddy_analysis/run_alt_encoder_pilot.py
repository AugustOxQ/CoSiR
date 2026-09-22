"""Test a broader-domain 7-way affect encoder as an ArtELingo graph signal.

This deliberately holds the single-modality graph construction, Leiden
clustering, K, and seed fixed relative to the GoEmotions ceiling pilot. Run it
manually in a GPU-capable environment; it recomputes affect features and writes
the results report after completion.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


OUT_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
AFFECT_PILOT_PATH = os.path.join(OUT_DIR, "run_affect_pilot.py")
SINGLE_MODALITY_PILOT_PATH = os.path.join(OUT_DIR, "run_single_modality_pilot.py")
REPORT_PATH = os.path.join(OUT_DIR, "alt_encoder_pilot_report.md")

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
BATCH_SIZE = 256
MAX_LENGTH = 64
SEED = 42
GOEMOTIONS_EMOTION_AMI = 0.1180
GOEMOTIONS_GENRE_AMI = 0.0396
REAL_IMPROVEMENT_BAR = 0.177


# Make this standalone script importable from any working directory, matching
# run_pipeline.py's repository-root import convention.
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


def caption_text(record: dict) -> str:
    """Match the dataset's caption convention while tolerating singleton lists."""
    caption = record["caption"]
    return caption if isinstance(caption, str) else caption[0]


def extract_alt_affect_nodes(train_json: str, paintings: list[str], device: str) -> np.ndarray:
    """Mean-pool 7-way softmax affect probabilities by painting in pipeline order."""
    log(f"Loading alternate affect encoder ({MODEL_NAME}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    with open(train_json) as train_file:
        train = json.load(train_file)

    painting_to_idx = {painting: i for i, painting in enumerate(paintings)}
    row_node_indices = []
    captions = []
    for record in train:
        # artelingo_train.json is the English-caption split. Retain an explicit
        # language guard for compatible exports that carry a language field.
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

    if not captions:
        raise RuntimeError("No English caption rows found in TRAIN_JSON.")

    affect_sums = None
    affect_counts = np.zeros(len(paintings), dtype=np.int32)
    total_batches = (len(captions) + BATCH_SIZE - 1) // BATCH_SIZE
    log(
        f"Extracting alternate affect probabilities for {len(captions)} English caption rows "
        f"in {total_batches} batches (batch_size={BATCH_SIZE}, max_length={MAX_LENGTH})..."
    )
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
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy().astype(
                np.float32
            )
            if affect_sums is None:
                affect_sums = np.zeros((len(paintings), probabilities.shape[1]), dtype=np.float32)
            node_indices = np.asarray(row_node_indices[start:end], dtype=np.intp)
            np.add.at(affect_sums, node_indices, probabilities)
            np.add.at(affect_counts, node_indices, 1)
            if (start // BATCH_SIZE + 1) % 100 == 0 or end == len(captions):
                log(f"Alternate affect encoder progress: {end:,}/{len(captions):,} caption rows.")

    missing = np.flatnonzero(affect_counts == 0)
    if len(missing):
        raise RuntimeError(f"{len(missing)} deduplicated paintings have no English affect vectors.")
    return affect_sums / affect_counts[:, None]


def write_report(emotion_metrics: dict, genre_metrics: dict, genre_count: int, k: int) -> None:
    """Write the predeclared comparison and its data-dependent interpretation."""
    emotion_ami = emotion_metrics["AMI"]
    genre_ami = genre_metrics["AMI"]
    if emotion_ami > REAL_IMPROVEMENT_BAR:
        conclusion = "yes, real improvement"
    elif emotion_ami > GOEMOTIONS_EMOTION_AMI:
        conclusion = "smaller improvement, does not clear the bar"
    else:
        conclusion = "no improvement or worse"

    if genre_ami > GOEMOTIONS_GENRE_AMI:
        genre_direction = "improved"
    elif genre_ami < GOEMOTIONS_GENRE_AMI:
        genre_direction = "dropped"
    else:
        genre_direction = "held"

    lines = [
        "# ArtELingo alternate affect-encoder pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "**Setup:** `j-hartmann/emotion-english-distilroberta-base` supplies a 7-way "
        "softmax distribution over Ekman's six basic emotions (anger, disgust, fear, "
        "joy, sadness, surprise) plus neutral. Its mixed training domains are broader "
        "than Reddit-only GoEmotions and include Crowdflower, MELD dialogue transcripts, "
        "ISEAR personal narratives, SemEval-2018, and GoEmotions. **This is not a fully "
        "independent comparison: GoEmotions is one ingredient in the model's training mix, "
        "so any gain should be read as a broader-domain-trained encoder result, not proof "
        "from a completely unrelated encoder.** Mean-pooled per-painting affect vectors "
        f"are L2-normalized, then used alone to build a mutual-kNN graph (K={k}) with the "
        "same repairs as the reference pilot; Leiden uses seed=42. "
        f"Genre metrics use the {genre_count}-painting genre-labelled overlap.\n\n",
        "| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |\n",
        "|---|---:|---:|---:|---:|\n",
        "| GoEmotions-affect-only (Leiden, reference) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |\n",
        "| GoEmotions-affect-only (DEC v2, reference — different clustering method, not directly comparable to this pilot's Leiden-only rows) | 0.1492 | 0.1498 | 0.0554 | 0.0930 |\n",
        "| ArtELingo-native BERT, held-out (reference — in-domain supervised, upper bound) | 0.1693 | 0.1730 | 0.0138 | 0.2518 |\n",
        f"| j-hartmann-affect-only (Leiden, this run) | {emotion_ami:.4f} | "
        f"{emotion_metrics['V_measure']:.4f} | {genre_ami:.4f} | "
        f"{genre_metrics['V_measure']:.4f} |\n",
        "\n## Conclusion\n\n",
        f"The new encoder's Leiden emotion AMI is {emotion_ami:.4f}; {conclusion}. "
        f"The predeclared bar is AMI > {REAL_IMPROVEMENT_BAR:.3f}, a 50% relative "
        f"improvement over the GoEmotions single-modality ceiling of {GOEMOTIONS_EMOTION_AMI:.4f}. "
        f"Genre AMI {genre_direction} relative to the GoEmotions Leiden reference "
        f"({genre_ami:.4f} vs. {GOEMOTIONS_GENRE_AMI:.4f}). GoEmotions is part of this "
        "encoder's training mix, so this remains a non-independent comparison and any "
        "improvement must be interpreted cautiously as evidence for a broader-domain-trained "
        "encoder, not a completely unrelated one.\n",
    ]
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    pipeline = load_sibling_module("artelingo_run_pipeline", PIPELINE_PATH)
    affect_pilot = load_sibling_module("artelingo_run_affect_pilot", AFFECT_PILOT_PATH)
    single_modality_pilot = load_sibling_module(
        "artelingo_run_single_modality_pilot", SINGLE_MODALITY_PILOT_PATH
    )

    pipeline.assert_extraction_complete()
    log("Loading and deduplicating cached CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    if img_nodes.shape[0] != len(paintings) or txt_nodes.shape[0] != len(paintings):
        raise RuntimeError("Deduplicated CLIP node arrays do not align with the painting list.")
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for alternate affect extraction and graph construction.")

    affect_nodes = extract_alt_affect_nodes(pipeline.TRAIN_JSON, paintings, device)
    if affect_nodes.shape != (len(paintings), 7):
        raise RuntimeError(
            f"Alternate affect encoder returned {affect_nodes.shape}; expected ({len(paintings)}, 7)."
        )
    normalized_affect_nodes = affect_pilot.l2_normalize(affect_nodes)

    graph = single_modality_pilot.build_single_modality_graph(
        "j-hartmann-affect-only",
        normalized_affect_nodes,
        pipeline,
        affect_pilot,
        device,
        expected_nodes=len(paintings),
    )
    log(f"j-hartmann-affect-only: running Leiden community detection (seed={SEED})...")
    community = detect_communities(graph, seed=SEED)
    emotion_metrics, genre_metrics, genre_count = single_modality_pilot.evaluate_graph(
        pipeline, community, paintings, majority_emotion
    )
    log(
        f"j-hartmann-affect-only: emotion AMI={emotion_metrics['AMI']:.4f}, "
        f"genre AMI={genre_metrics['AMI']:.4f}."
    )
    write_report(emotion_metrics, genre_metrics, genre_count, pipeline.K)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
