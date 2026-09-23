"""Qualitatively inspect train paintings' secondary PercepT topic memberships.

This read-only GPU diagnostic deterministically re-fits the established
K=60/40 Stage-1 space, checks it against the established held-out AMIs, and
then prints caption-level comparisons for a reproducible sample of topics.
It deliberately performs no Stage-2 mapper training.
"""

import importlib.util
import json
import os
import random
import time

import numpy as np
import torch


OUT_DIR = os.path.dirname(__file__)
REPORT_OUT_DIR = os.environ.get("PERCEPT_OUTPUT_ROOT") or OUT_DIR
BASE_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage1_pilot.py")
SWEEP_PILOT_PATH = os.path.join(
    OUT_DIR, "run_percept_stage1_cluster_count_sweep_pilot.py"
)
STAGE2_SWEEP_PILOT_PATH = os.path.join(OUT_DIR, "run_percept_stage2_sweep_pilot.py")
REPORT_PATH = os.path.join(REPORT_OUT_DIR, "percept_topic_qualitative_check_report.md")

N_INITIAL_CLUSTERS = 60
N_SURVIVING_CLUSTERS = 40
SEED = 42
EXPECTED_HELDOUT_EMOTION_AMI = 0.1238
EXPECTED_HELDOUT_GENRE_AMI = 0.2617
REPRODUCTION_TOLERANCE = 0.002
SAMPLED_TOPIC_COUNT = 5
EXAMPLES_PER_GROUP = 5
PRIMARY_COMPARISON_EXAMPLES = 2
# Must match the Stage-2 "genuine multi-label" threshold established in the
# sweep report (q > 2.0/40 and 1.5/40 both produced ~0% multi-labeled
# paintings; 1.2/40 was the first to produce real multi-label targets).
MULTI_LABEL_THRESHOLD_MULTIPLIER = 1.2


def load_module(module_name: str, path: str):
    """Load a standalone sibling script without running its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("percept_stage1_base_for_qualitative_check", BASE_PILOT_PATH)
sweep = load_module("percept_stage1_sweep_for_qualitative_check", SWEEP_PILOT_PATH)
stage2_sweep = load_module(
    "percept_stage2_sweep_for_qualitative_check", STAGE2_SWEEP_PILOT_PATH
)


def first_english_captions(train_json: str, paintings: list[str]) -> dict[str, str]:
    """Return the first English caption row for every deduplicated train painting."""
    with open(train_json) as train_file:
        records = json.load(train_file)
    available_paintings = set(paintings)
    captions = {}
    for record in records:
        if record.get("language", "english").lower() != "english":
            continue
        painting = record["painting"]
        if painting in available_paintings and painting not in captions:
            captions[painting] = base.caption_text(record)
    missing = [painting for painting in paintings if painting not in captions]
    if missing:
        raise RuntimeError(
            f"{len(missing)} deduplicated train paintings have no English caption row."
        )
    return captions


def markdown_caption(painting: str, caption: str) -> str:
    """Format one caption without permitting a caption to break Markdown rows."""
    safe_caption = caption.replace("\n", " ").replace("|", "\\|")
    return f"- `{painting}`: {safe_caption}\n"


def append_topic_section(
    lines: list[str],
    topic: int,
    paintings: list[str],
    captions: dict[str, str],
    targets: np.ndarray,
    primary_topics: np.ndarray,
    single_indices_by_topic: dict[int, list[int]],
) -> None:
    """Append one sampled topic's core and secondary-caption comparisons."""
    label_counts = targets.sum(axis=1)
    single_indices = single_indices_by_topic[topic]
    secondary_indices = [
        index
        for index in range(len(paintings))
        if targets[index, topic] == 1 and primary_topics[index] != topic
    ]
    lines.extend([
        f"## Topic {topic}\n\n",
        f"- Single-labeled core members: **{len(single_indices):,}**\n",
        f"- Secondary memberships: **{len(secondary_indices):,}**\n\n",
        "### Single-labeled example captions\n\n",
    ])
    if single_indices:
        lines.extend(
            markdown_caption(paintings[index], captions[paintings[index]])
            for index in single_indices[:EXAMPLES_PER_GROUP]
        )
    else:
        lines.append("_No single-labeled train paintings for this topic._\n")

    lines.append("\n### Secondary-membership example captions and primary-topic comparisons\n\n")
    if not secondary_indices:
        lines.append("_No train paintings carry this topic as a non-primary label._\n\n")
        return
    for number, index in enumerate(secondary_indices[:EXAMPLES_PER_GROUP], start=1):
        primary_topic = int(primary_topics[index])
        lines.extend([
            f"#### Secondary example {number}: topic {topic} with primary topic {primary_topic}\n\n",
            markdown_caption(paintings[index], captions[paintings[index]]),
            f"\nPrimary topic {primary_topic}'s single-labeled comparison examples "
            f"(population: {len(single_indices_by_topic[primary_topic]):,}):\n\n",
        ])
        primary_examples = single_indices_by_topic[primary_topic][
            :PRIMARY_COMPARISON_EXAMPLES
        ]
        if primary_examples:
            lines.extend(
                markdown_caption(paintings[example], captions[paintings[example]])
                for example in primary_examples
            )
        else:
            lines.append("_No single-labeled examples are available for this primary topic._\n")
        lines.append("\n")


def write_report(
    refit_metrics: dict,
    reproduced: bool,
    sampled_topics: list[int] | None = None,
    paintings: list[str] | None = None,
    captions: dict[str, str] | None = None,
    targets: np.ndarray | None = None,
    primary_topics: np.ndarray | None = None,
    single_indices_by_topic: dict[int, list[int]] | None = None,
) -> None:
    """Write the reproduction gate and, if valid, qualitative comparisons."""
    emotion_ami = refit_metrics["emotion"]["AMI"]
    genre_ami = refit_metrics["genre"]["AMI"]
    lines = [
        "# PercepT topic qualitative check\n\n",
        f"Generated automatically, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "## Stage-1 K=60/40 seed-42 reproduction\n\n",
        "| metric | established | re-fit | absolute difference | status |\n",
        "|---|---:|---:|---:|---|\n",
        f"| held-out emotion AMI | {EXPECTED_HELDOUT_EMOTION_AMI:.4f} | "
        f"{emotion_ami:.4f} | {abs(emotion_ami - EXPECTED_HELDOUT_EMOTION_AMI):.4f} | "
        f"{'reproduced' if reproduced else 'FAILED'} |\n",
        f"| held-out genre AMI | {EXPECTED_HELDOUT_GENRE_AMI:.4f} | "
        f"{genre_ami:.4f} | {abs(genre_ami - EXPECTED_HELDOUT_GENRE_AMI):.4f} | "
        f"{'reproduced' if reproduced else 'FAILED'} |\n\n",
    ]
    if not reproduced:
        lines.append(
            "**Reproducibility failure.** The K=60/40 seed-42 re-fit was not "
            f"within the required absolute AMI tolerance of {REPRODUCTION_TOLERANCE:.3f}; "
            "the qualitative target inspection was stopped rather than silently "
            "using a different topic space.\n"
        )
        os.makedirs(REPORT_OUT_DIR, exist_ok=True)
        with open(REPORT_PATH, "w") as report_file:
            report_file.writelines(lines)
        return

    assert sampled_topics is not None
    assert paintings is not None
    assert captions is not None
    assert targets is not None
    assert primary_topics is not None
    assert single_indices_by_topic is not None
    lines.extend([
        "The target is the frozen Stage-2 rule: argmax topic plus every topic "
        "whose DEC assignment satisfies `q > 1.2/40`. The five topics below "
        f"are sampled without replacement with `random.seed({SEED})`: "
        f"{', '.join(str(topic) for topic in sampled_topics)}.\n\n",
    ])
    for topic in sampled_topics:
        append_topic_section(
            lines,
            topic,
            paintings,
            captions,
            targets,
            primary_topics,
            single_indices_by_topic,
        )

    lines.extend([
        "## Qualitative reading\n\n",
        "This five-topic sample is a caption-level observation, not a statistical "
        "test. The paired examples should be read for recurring shared subject "
        "matter, style, or mood language between each secondary member and the "
        "topic's single-labeled core. A mixed or unclear pattern is itself the "
        "honest outcome here: neither apparent overlap nor apparent loose affinity "
        "in this small sample establishes that all threshold-derived memberships "
        "are semantically meaningful.\n",
    ])
    os.makedirs(REPORT_OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)


def main() -> None:
    """Re-fit frozen topics, reproduce the gate, then inspect train captions."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_qualitative_check_train", base.PIPELINE_PATH
    )
    affect_pilot = base.load_sibling_module(
        "artelingo_run_affect_pilot_qualitative_check", base.AFFECT_PILOT_PATH
    )
    cca_audit = base.load_sibling_module(
        "artelingo_run_cca_audit_qualitative_check", base.CCA_AUDIT_PATH
    )
    heldout_pipeline = base.load_sibling_module(
        "artelingo_run_pipeline_qualitative_check_heldout", base.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = base.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = base.HELDOUT_JSON
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    log = pipeline.log

    log("Loading and deduplicating train CLIP features for Stage-1 re-fit...")
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotions = [pipeline.majority(counts) for counts in emotion_counts]
    log("Loading and deduplicating held-out CLIP features for Stage-1 re-fit...")
    (
        heldout_paintings,
        heldout_img_nodes,
        heldout_txt_nodes,
        heldout_emotion_counts,
    ) = heldout_pipeline.load_dedup_features()
    heldout_emotions = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} for the read-only Stage-1 re-fit and caption inspection.")
    affect_train = base.extract_affect_embedding_nodes(
        pipeline.TRAIN_JSON, paintings, device, log
    )
    affect_heldout = base.extract_affect_embedding_nodes(
        base.HELDOUT_JSON, heldout_paintings, device, log
    )
    train_h = base.fused_embeddings(
        img_nodes, txt_nodes, affect_train, cca_audit, affect_pilot
    )
    heldout_h = base.fused_embeddings(
        heldout_img_nodes,
        heldout_txt_nodes,
        affect_heldout,
        cca_audit,
        affect_pilot,
    )
    if train_h.shape[1] != heldout_h.shape[1]:
        raise RuntimeError(
            f"Train/held-out fused dimensions differ: {train_h.shape[1]} vs "
            f"{heldout_h.shape[1]}."
        )

    train_inputs = torch.from_numpy(train_h)
    encoder, decoder = base.build_autoencoder(train_h.shape[1])
    encoder.to(device)
    decoder.to(device)
    base.pretrain_autoencoder(encoder, decoder, train_inputs, device, log)
    centers = sweep.initialize_cluster_centers(
        encoder, train_inputs, device, N_INITIAL_CLUSTERS, SEED
    )
    sweep.train_dec_until_stable(
        encoder,
        decoder,
        centers,
        train_inputs,
        device,
        log,
        N_INITIAL_CLUSTERS,
    )
    surviving_centers, surviving_indices = sweep.prune_centers(
        centers, N_SURVIVING_CLUSTERS
    )
    log(
        f"Pruned {N_INITIAL_CLUSTERS - N_SURVIVING_CLUSTERS} centers; retained "
        f"original indices {surviving_indices.tolist()}."
    )
    _, heldout_refit_metrics, _ = sweep.evaluate_run(
        encoder,
        centers,
        train_inputs,
        heldout_h,
        pipeline,
        heldout_pipeline,
        paintings,
        train_emotions,
        heldout_paintings,
        heldout_emotions,
        device,
        N_SURVIVING_CLUSTERS,
    )
    reproduced = (
        abs(heldout_refit_metrics["emotion"]["AMI"] - EXPECTED_HELDOUT_EMOTION_AMI)
        < REPRODUCTION_TOLERANCE
        and abs(heldout_refit_metrics["genre"]["AMI"] - EXPECTED_HELDOUT_GENRE_AMI)
        < REPRODUCTION_TOLERANCE
    )
    if not reproduced:
        write_report(heldout_refit_metrics, reproduced=False)
        log(f"Stage-1 reproducibility failure; wrote report to {REPORT_PATH}.")
        return

    multi_label_threshold = MULTI_LABEL_THRESHOLD_MULTIPLIER / N_SURVIVING_CLUSTERS
    targets = stage2_sweep.multi_hot_targets(
        encoder, surviving_centers, train_inputs, device, multi_label_threshold
    ).numpy().astype(np.int8, copy=False)
    encoder.eval()
    with torch.no_grad():
        primary_topics = (
            base.soft_assignments(encoder(train_inputs.to(device)), surviving_centers)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
    label_counts = targets.sum(axis=1)
    single_indices_by_topic = {
        topic: [
            index
            for index in range(len(paintings))
            if label_counts[index] == 1 and primary_topics[index] == topic
        ]
        for topic in range(N_SURVIVING_CLUSTERS)
    }
    captions = first_english_captions(pipeline.TRAIN_JSON, paintings)
    random.seed(SEED)
    sampled_topics = random.sample(range(N_SURVIVING_CLUSTERS), SAMPLED_TOPIC_COUNT)
    write_report(
        heldout_refit_metrics,
        reproduced=True,
        sampled_topics=sampled_topics,
        paintings=paintings,
        captions=captions,
        targets=targets,
        primary_topics=primary_topics,
        single_indices_by_topic=single_indices_by_topic,
    )
    log(f"Wrote qualitative caption report to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
