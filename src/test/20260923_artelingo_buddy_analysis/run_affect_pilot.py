"""Test affect-aware text fusion for ArtELingo buddy-graph communities.

This script is intentionally standalone.  It reuses the committed CLIP feature
deduplication and evaluation helpers from ``run_pipeline.py``, then adds a
GoEmotions probability vector pooled over each painting's English captions.
Run it manually in a GPU-capable environment; it can take several minutes.
"""

import importlib.util
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


OUT_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(OUT_DIR, "run_pipeline.py")
REPORT_PATH = os.path.join(OUT_DIR, "affect_pilot_report.md")
COMMUNITIES_PATH = os.path.join(OUT_DIR, "affect_pilot_communities.npz")

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
BATCH_SIZE = 256
MAX_LENGTH = 64
AFFECT_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)
EMOTION_BASELINE_AMI = 0.0593
NON_TIED_EMOTION_BASELINE_AMI = 0.0758
GENRE_BASELINE_AMI = 0.4384
REPRODUCTION_TOLERANCE = 0.005


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_pipeline_module():
    """Load the sibling standalone pipeline without triggering its main function."""
    spec = importlib.util.spec_from_file_location("artelingo_run_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pipeline helpers from {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l2_normalize(nodes: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(nodes, axis=1, keepdims=True)
    return nodes / np.maximum(norms, 1e-12)


def caption_text(record: dict) -> str:
    """Match the dataset's caption convention while tolerating singleton lists."""
    caption = record["caption"]
    return caption if isinstance(caption, str) else caption[0]


def extract_affect_nodes(train_json: str, paintings: list[str], device: str) -> np.ndarray:
    """Mean-pool GoEmotions sigmoid probabilities by painting in pipeline order."""
    log(f"Loading GoEmotions encoder ({MODEL_NAME}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    with open(train_json) as f:
        train = json.load(f)

    painting_to_idx = {painting: i for i, painting in enumerate(paintings)}
    row_node_indices = []
    captions = []
    for record in train:
        # artelingo_train.json is the English-caption split.  Retain an explicit
        # language guard for compatible exports that carry a language field.
        if record.get("language", "english").lower() != "english":
            continue
        try:
            row_node_indices.append(painting_to_idx[record["painting"]])
        except KeyError as exc:
            raise RuntimeError(
                f"Training row references painting absent from deduplicated features: "
                f"{record['painting']}"
            ) from exc
        captions.append(caption_text(record))

    if not captions:
        raise RuntimeError("No English caption rows found in TRAIN_JSON.")

    affect_sums = None
    affect_counts = np.zeros(len(paintings), dtype=np.int32)
    total_batches = (len(captions) + BATCH_SIZE - 1) // BATCH_SIZE
    log(
        f"Extracting GoEmotions probabilities for {len(captions)} English caption rows "
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
            probabilities = torch.sigmoid(model(**encoded).logits).cpu().numpy().astype(np.float32)
            if affect_sums is None:
                affect_sums = np.zeros((len(paintings), probabilities.shape[1]), dtype=np.float32)
            node_indices = np.asarray(row_node_indices[start:end], dtype=np.intp)
            np.add.at(affect_sums, node_indices, probabilities)
            np.add.at(affect_counts, node_indices, 1)
            if (start // BATCH_SIZE + 1) % 100 == 0 or end == len(captions):
                log(f"Affect encoder progress: {end:,}/{len(captions):,} caption rows.")

    missing = np.flatnonzero(affect_counts == 0)
    if len(missing):
        raise RuntimeError(f"{len(missing)} deduplicated paintings have no English affect vectors.")
    return affect_sums / affect_counts[:, None]


def evaluate_communities(pipeline, community: np.ndarray, paintings: list[str], majority_emotion: list[str]):
    """Return full-graph emotion and genre-subset metrics using pipeline helpers."""
    emotion_metrics = pipeline.external_metrics(community, majority_emotion)
    genre_map = pipeline.load_genre_map()
    genre_indices = [i for i, painting in enumerate(paintings) if painting in genre_map]
    genre_metrics = pipeline.external_metrics(
        [community[i] for i in genre_indices],
        [genre_map[paintings[i]] for i in genre_indices],
    )
    return emotion_metrics, genre_metrics, len(genre_indices)


def write_report(results: list[dict], genre_count: int) -> None:
    baseline = results[0]
    reproduced = (
        abs(baseline["emotion"]["AMI"] - EMOTION_BASELINE_AMI) <= REPRODUCTION_TOLERANCE
        and abs(baseline["genre"]["AMI"] - GENRE_BASELINE_AMI) <= REPRODUCTION_TOLERANCE
    )
    candidates = [
        result
        for result in results
        if result["weight"] > 0.0
        and result["emotion"]["AMI"] > 0.09
        # A 20% genre-AMI reduction is the operational meaning of "badly" here.
        and result["genre"]["AMI"] >= GENRE_BASELINE_AMI * 0.8
    ]

    lines = [
        "# ArtELingo affect-aware buddy-graph pilot\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "**Setup:** GoEmotions (`SamLowe/roberta-base-go_emotions`) sigmoid logits are "
        "mean-pooled over each painting's English captions, L2-normalized, and "
        "concatenated with L2-normalized CLIP text features. Image features remain "
        "pure CLIP. Buddy graph: K=20, alpha=0.5, Leiden seed=42.\n\n",
        "| affect_weight | community_vs_emotion_AMI | community_vs_emotion_Vmeasure | "
        "community_vs_genre_AMI | community_vs_genre_Vmeasure |\n",
        "|---:|---:|---:|---:|---:|\n",
    ]
    for result in results:
        lines.append(
            f"| {result['weight']:.1f} | {result['emotion']['AMI']:.4f} | "
            f"{result['emotion']['V_measure']:.4f} | {result['genre']['AMI']:.4f} | "
            f"{result['genre']['V_measure']:.4f} |\n"
        )

    lines.append("\n## CLIP-only reproduction check\n\n")
    if reproduced:
        lines.append(
            f"The affect_weight=0.0 control reproduced the original CLIP-only baseline "
            f"within ±{REPRODUCTION_TOLERANCE:.3f}: emotion AMI={baseline['emotion']['AMI']:.4f} "
            f"vs. {EMOTION_BASELINE_AMI:.4f}, genre AMI={baseline['genre']['AMI']:.4f} "
            f"vs. {GENRE_BASELINE_AMI:.4f} (genre n={genre_count}).\n"
        )
    else:
        lines.append(
            f"**WARNING:** the affect_weight=0.0 control did not closely reproduce the "
            f"original CLIP-only baseline: emotion AMI={baseline['emotion']['AMI']:.4f} "
            f"vs. {EMOTION_BASELINE_AMI:.4f}, genre AMI={baseline['genre']['AMI']:.4f} "
            f"vs. {GENRE_BASELINE_AMI:.4f}. Treat the affect sweep cautiously; this "
            f"discrepancy may exceed normal Leiden variation.\n"
        )

    lines.append("\n## Conclusion\n\n")
    if candidates:
        weights = ", ".join(f"{result['weight']:.1f}" for result in candidates)
        lines.append(
            f"Yes. Affect weight(s) {weights} achieved the predeclared meaningful emotion "
            f"threshold (AMI > 0.09, at least a 50% relative improvement over the 0.0593 "
            f"full-graph baseline) while retaining at least 80% of the 0.4384 genre AMI "
            f"baseline. The non-tied emotion reference is AMI={NON_TIED_EMOTION_BASELINE_AMI:.4f}.\n"
        )
    else:
        lines.append(
            f"No. No affect weight achieved AMI > 0.09 (the predeclared 50% relative "
            f"improvement threshold over the 0.0593 full-graph baseline) while retaining "
            f"at least 80% of the 0.4384 genre AMI baseline. The non-tied emotion reference "
            f"is AMI={NON_TIED_EMOTION_BASELINE_AMI:.4f}; inspect the table for trade-offs.\n"
        )

    with open(REPORT_PATH, "w") as f:
        f.writelines(lines)


def main() -> None:
    pipeline = load_pipeline_module()
    pipeline.assert_extraction_complete()
    log("Loading and deduplicating cached CLIP features...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for affect extraction and buddy-graph construction.")
    affect_nodes = extract_affect_nodes(pipeline.TRAIN_JSON, paintings, device)

    normalized_txt = l2_normalize(txt_nodes)
    normalized_affect = l2_normalize(affect_nodes)
    communities = {}
    results = []
    genre_count = 0
    for weight in AFFECT_WEIGHTS:
        # The control intentionally uses the original, unnormalized CLIP text matrix
        # so it reproduces run_pipeline.py's graph-building call exactly.
        fused_txt = txt_nodes if weight == 0.0 else np.concatenate(
            (normalized_txt, weight * normalized_affect), axis=1
        ).astype(np.float32, copy=False)
        log(
            f"Weight {weight:.1f}: building buddy graph (K={pipeline.K}, "
            f"alpha={pipeline.ALPHA}, device={device})..."
        )
        _, _, edges = pipeline.build_buddy_graphs(
            img_nodes,
            fused_txt,
            K=pipeline.K,
            alpha=pipeline.ALPHA,
            device=device,
            connect_components=True,
        )
        log(f"Weight {weight:.1f}: running Leiden community detection (seed={pipeline.SEED})...")
        community = pipeline.detect_communities(edges, seed=pipeline.SEED)
        emotion_metrics, genre_metrics, genre_count = evaluate_communities(
            pipeline, community, paintings, majority_emotion
        )
        communities[f"community_labels_weight_{weight:.1f}"] = community
        results.append({"weight": weight, "emotion": emotion_metrics, "genre": genre_metrics})
        log(
            f"Weight {weight:.1f}: emotion AMI={emotion_metrics['AMI']:.4f}, "
            f"genre AMI={genre_metrics['AMI']:.4f}."
        )

    np.savez_compressed(
        COMMUNITIES_PATH,
        paintings=np.asarray(paintings, dtype=str),
        **communities,
    )
    log(f"Saved community assignments to {COMMUNITIES_PATH}")
    write_report(results, genre_count)
    log(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
