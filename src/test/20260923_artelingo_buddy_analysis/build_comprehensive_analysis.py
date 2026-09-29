"""Build the saved-data ArtELingo buddy/PercepT analysis and its figures.

The only fit performed here is the seed-42, CPU Stage 2 image mapper needed
to obtain example-level buddy scores. Stage 1 models remain frozen.
Run with the CoSiR environment and PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PERCEPT = HERE.parent / "20260922_percept_topic_pipeline"
REPORTS = ROOT / "docs" / "reports"
ASSETS = REPORTS / "assets" / "comprehensive_analysis"
OUTPUT = REPORTS / "2026-09-27_comprehensive_analysis.md"
CSV_PATH = Path("/data/PDD/artelingo/ArtELingo/ArtELingo/Dataset/artelingo_release_lite.csv")
SEEDS = (42, 7, 123, 2024)
METRICS = ("emotion AMI", "genre AMI", "silhouette")

VARIANTS = [
    ("Attention-h1 baseline", "attention_h1_baseline_seed_stress_pilot_report.md", "held-out silhouette"),
    ("Cosine schedule (noise=0)", "attention_h1_noise_schedule_pilot_report.md", "held-out silhouette"),
    ("Schedule + pseudo-contrastive", "attention_h1_noise_schedule_pseudo_contrastive_stress_pilot_report.md", "held-out silhouette"),
    ("Euclidean DEC", "attention_h1_dec_hybrid_pilot_report.md", "held-out silhouette"),
    ("vMF DEC", "attention_h1_vmf_dec_hybrid_pilot_report.md", "held-out silhouette"),
    ("Decoupled, un-detached", "attention_h1_decoupled_cluster_head_pilot_report.md", "held-out fused silhouette"),
    ("Decoupled, detached", "attention_h1_decoupled_cluster_head_detached_pilot_report.md", "held-out fused silhouette"),
    ("Reconstruction-anchored", "reconstruction_anchored_cluster_head_pilot_report.md", "held-out fused silhouette"),
]


def cells(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def markdown_table_after(text: str, heading: str) -> tuple[list[str], list[list[str]]]:
    """Return the first complete Markdown table after an exact heading."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    header_at = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("|"))
    header = cells(lines[header_at])
    rows = []
    for line in lines[header_at + 2 :]:
        if not line.startswith("|"):
            break
        row = cells(line)
        if len(row) != len(header):
            raise ValueError(f"Malformed table after {heading}: {line}")
        rows.append(row)
    return header, rows


def seed_values(path: Path, silhouette_column: str) -> dict[str, np.ndarray]:
    """Parse real per-seed held-out rows, including fused-space DEC silhouettes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    required = {"seed", "held-out emotion AMI", "held-out genre AMI", silhouette_column}
    for i, line in enumerate(lines):
        if not line.startswith("| seed |"):
            continue
        header = cells(line)
        if not required.issubset(header):
            continue
        rows = []
        for candidate in lines[i + 2 :]:
            if not candidate.startswith("|"):
                break
            row = cells(candidate)
            if row[header.index("seed")] in {str(seed) for seed in SEEDS}:
                rows.append(row)
        by_seed = {int(row[header.index("seed")]): row for row in rows}
        if set(by_seed) != set(SEEDS):
            continue
        return {
            metric: np.array(
                [float(by_seed[seed][header.index(column)]) for seed in SEEDS], dtype=float
            )
            for metric, column in (
                ("emotion AMI", "held-out emotion AMI"),
                ("genre AMI", "held-out genre AMI"),
                ("silhouette", silhouette_column),
            )
        }
    raise ValueError(f"No complete four-seed held-out table in {path}")


def bootstrap(values: np.ndarray) -> tuple[float, float, float, float]:
    draws = np.random.default_rng(42).choice(values, size=(10_000, 4), replace=True)
    low, high = np.percentile(draws.mean(axis=1), [2.5, 97.5])
    return float(values.mean()), float(values.std(ddof=1)), float(low), float(high)


def reported_mean(values: np.ndarray) -> str:
    """Round the four-decimal source entries without binary-float tie drift."""
    exact = sum((Decimal(f"{value:.4f}") for value in values), Decimal(0)) / Decimal(4)
    return str(exact.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def parse_occupancy(path: Path, heading: str, expected: int) -> np.ndarray:
    _, rows = markdown_table_after(path.read_text(encoding="utf-8"), heading)
    labels = np.array([int(row[0]) for row in rows])
    counts = np.array([int(row[1].replace(",", "")) for row in rows])
    assert np.array_equal(labels, np.arange(expected))
    assert counts.sum() == 9_365
    return counts


def parse_aucs(path: Path, expected: int) -> dict[int, float]:
    _, rows = markdown_table_after(path.read_text(encoding="utf-8"), "## Held-out per-topic AUC")
    result = {int(row[0]): float(row[1]) for row in rows}
    assert set(result) == set(range(expected))
    return result


def parse_schedule_recall(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, rows = markdown_table_after(path.read_text(encoding="utf-8"), "### noise_std=0")
    return tuple(np.array([float(row[i]) for row in rows]) for i in (0, 2, 3))


def parse_percept_losses(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("- Joint DEC losses and LR:")
                and "epoch 154" in line)
    matches = re.findall(r"epoch (\d+): total=[\d.]+, KL=([\d.]+), recon=([\d.]+)", line)
    assert len(matches) >= 10
    return tuple(np.array([float(row[i]) for row in matches]) for i in range(3))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sorted_json_paintings(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return sorted({row["painting"] for row in json.load(handle)})


def mapper_scores(buddy: dict[str, np.ndarray], buddy_aucs: dict[int, float],
                  audit_counts: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the original Stage 2 BCE mapper at seed 42, forced onto CPU."""
    stage2_buddy = load_module("buddy_stage2_for_comprehensive", HERE / "run_buddy_stage2_pilot.py")
    stage2 = load_module("percept_stage2_for_comprehensive", PERCEPT / "run_percept_stage2_pilot.py")
    pipeline = load_module("pipeline_for_comprehensive", HERE / "run_pipeline.py")
    train_order = sorted_json_paintings(pipeline.TRAIN_JSON)
    heldout_order = sorted_json_paintings(stage2.base.HELDOUT_JSON)
    assert len(train_order) == 61_402 and len(heldout_order) == 9_365
    assert set(train_order) == set(buddy["train_paintings"])
    assert set(heldout_order) == set(buddy["heldout_paintings"])
    train_index = {name: i for i, name in enumerate(buddy["train_paintings"])}
    heldout_index = {name: i for i, name in enumerate(buddy["heldout_paintings"])}
    train_pos = np.array([train_index[name] for name in train_order])
    heldout_pos = np.array([heldout_index[name] for name in heldout_order])
    train_labels = buddy["train_community_post"][train_pos]
    assert np.array_equal(np.unique(train_labels), np.arange(19))
    transfer = load_module("transfer_for_comprehensive", HERE / "run_heldout_label_transfer_pilot.py")
    heldout_labels = transfer.assign_to_train_communities(
        buddy["train_embedding_post"][train_pos], train_labels,
        buddy["heldout_embedding_post"][heldout_pos], k=stage2_buddy.TRANSFER_K)
    assert np.array_equal(np.bincount(heldout_labels, minlength=19), audit_counts)

    train_patches = stage2.load_patch_features(stage2.TRAIN_PATCH_FEATURE_PATH, len(train_order), "train")
    heldout_patches = stage2.load_patch_features(stage2.HELDOUT_PATCH_FEATURE_PATH, len(heldout_order), "held-out")
    torch.set_num_threads(min(16, torch.get_num_threads()))
    np.random.seed(42)
    torch.manual_seed(42)
    mapper = stage2.AttentionPoolingMapper(n_topics=19).cpu()
    optimizer = torch.optim.Adam(mapper.parameters(), lr=stage2.MAPPER_LEARNING_RATE)
    target = stage2_buddy.one_hot_targets(train_labels)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    mapper.train()
    for epoch in range(1, stage2.MAPPER_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(mapper(train_patches), target)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"CPU buddy Stage 2 mapper epoch {epoch}/{stage2.MAPPER_EPOCHS}: BCE={loss.item():.6f}", flush=True)
    mapper.eval()
    with torch.no_grad():
        ordered_scores = torch.sigmoid(mapper(heldout_patches)).numpy()
    assert abs(float(loss.item()) - 0.200270) < 0.00005
    measured = {topic: roc_auc_score((heldout_labels == topic).astype(int), ordered_scores[:, topic])
                for topic in range(19)}
    max_delta = max(abs(measured[topic] - buddy_aucs[topic]) for topic in range(19))
    print(f"Fresh CPU mapper versus saved per-topic AUC: maximum absolute difference={max_delta:.6f}",
          flush=True)
    if max_delta > 0.005:
        raise ValueError("Fresh mapper AUCs drift too far from the cited Stage 2 per-topic AUCs")
    scores = np.empty_like(ordered_scores)
    scores[heldout_pos] = ordered_scores
    return scores, max_delta


def select_across_topics(candidates: list[int], top: np.ndarray, scores: np.ndarray,
                         paintings: np.ndarray, n: int) -> list[int]:
    """Choose one example per eligible topic before taking a second from any."""
    groups: dict[int, list[int]] = {}
    for index in candidates:
        groups.setdefault(int(top[index]), []).append(index)
    for topic, indices in groups.items():
        indices.sort(key=lambda i: (-float(scores[i, topic]), str(paintings[i])))
    result = []
    while len(result) < n and any(groups.values()):
        ordered_topics = sorted((topic for topic, items in groups.items() if items),
                                key=lambda topic: (-float(scores[groups[topic][0], topic]), topic))
        for topic in ordered_topics:
            result.append(groups[topic].pop(0))
            if len(result) == n:
                break
    return result


def dominant(labels: np.ndarray, emotions: np.ndarray) -> dict[int, str]:
    counts: dict[int, Counter] = {}
    for label, emotion in zip(labels, emotions):
        counts.setdefault(int(label), Counter())[str(emotion)] += 1
    return {label: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
            for label, counter in counts.items()}


def first_captions(paintings: set[str]) -> dict[str, dict[str, str]]:
    found = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["art_style", "emotion", "language", "painting", "split", "utterance"]
        for row in reader:
            painting = row["painting"]
            if painting in paintings and painting not in found:
                found[painting] = row
    if set(found) != paintings:
        raise ValueError(f"Missing CSV captions for {len(paintings) - len(found)} held-out paintings")
    return found


def safe_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def example_rows(indices: list[int], buddy: dict[str, np.ndarray], scores: np.ndarray,
                 aucs: dict[int, float], captions: dict[str, dict[str, str]],
                 buddy_dom: dict[int, str], percept_topic: np.ndarray,
                 percept_dom: dict[int, str]) -> list[str]:
    rows = []
    for i in indices:
        painting = str(buddy["heldout_paintings"][i])
        caption = captions[painting]
        topic = int(scores[i].argmax())
        ptopic = int(percept_topic[i])
        fields = [painting, caption["art_style"], str(buddy["heldout_emotion"][i]),
                  caption["emotion"], str(buddy["heldout_genre"][i]) or "—",
                  topic, buddy_dom[topic], f"{aucs[topic]:.4f}",
                  ptopic, percept_dom.get(ptopic, "—"), caption["utterance"]]
        rows.append("| " + " | ".join(safe_cell(value) for value in fields) + " |")
    return rows


def save(fig, filename: str) -> None:
    fig.savefig(ASSETS / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def tournament_plot(variants, percept_metrics):
    names = [item[0] for item in variants] + ["PercepT faithful (seed 42)"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), sharex=True)
    x = np.arange(len(names))
    colors = ["#547aa5"] * len(variants) + ["#dc694b"]
    for ax, metric in zip(axes, METRICS):
        means = [data[metric].mean() for _, _, data in variants] + [percept_metrics[metric]]
        ax.bar(x, means, color=colors, alpha=0.83)
        for j, (_, _, data) in enumerate(variants):
            ax.scatter(j + np.linspace(-0.15, 0.15, 4), data[metric], s=18,
                       color="#152536", zorder=3)
        if metric in ("emotion AMI", "genre AMI"):
            bar = 0.1236 if metric == "emotion AMI" else 0.1954
            ax.axhline(bar, color="#a13a35", linestyle="--", linewidth=1.5,
                       label=f"Pareto bar {bar:.4f}")
            ax.legend(loc="upper left", fontsize=8)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xticks(x, names, rotation=62, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Held-out Stage 1 tournament: bars are means; dots are individual seeds")
    fig.tight_layout()
    save(fig, "tournament.png")


def occupancy_plot(buddy_counts, percept_counts):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax, counts, title, color in ((axes[0], buddy_counts, "Buddy: 19 train communities", "#547aa5"),
                                     (axes[1], percept_counts, "PercepT: 67 surviving centers", "#dc694b")):
        ax.bar(np.arange(len(counts)), counts, color=color)
        ax.set_title(title)
        ax.set_xlabel("Frozen train label")
        ax.set_ylabel("Held-out painting count")
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].text(0.98, 0.96, f"{np.count_nonzero(percept_counts == 0)} empty centers",
                 ha="right", va="top", transform=axes[1].transAxes, fontsize=9)
    fig.tight_layout()
    save(fig, "occupancy.png")


def projection_plot(snapshot, embedding_key, topic_key, title, filename):
    emb = snapshot[embedding_key]
    xy = PCA(n_components=2, random_state=42).fit_transform(emb)
    labels = [("Topic / community", snapshot[topic_key]),
              ("Majority emotion", snapshot["heldout_emotion"]),
              ("Genre (annotated only)", snapshot["heldout_genre"])]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, (heading, values) in zip(axes, labels):
        values = np.asarray(values).astype(str)
        if heading.startswith("Genre"):
            absent = values == ""
            ax.scatter(xy[absent, 0], xy[absent, 1], s=2, color="#d4d7da", alpha=0.22,
                       rasterized=True)
            shown = np.flatnonzero(~absent)
        else:
            shown = np.arange(len(values))
        categories = sorted(set(values[shown]))
        palette = plt.get_cmap("tab20" if len(categories) <= 20 else "nipy_spectral", len(categories))
        for index, category in enumerate(categories):
            rows = shown[values[shown] == category]
            ax.scatter(xy[rows, 0], xy[rows, 1], s=3 if heading.startswith("Topic") else 5,
                       color=palette(index), alpha=0.62, label=category, rasterized=True)
        ax.set_title(heading)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if len(categories) <= 20:
            ax.legend(markerscale=3, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1),
                      frameon=False, ncol=1)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, filename)
    return len(emb), int(np.count_nonzero(np.asarray(snapshot["heldout_genre"]).astype(str) != ""))


def auc_plot(buddy_aucs, percept_aucs):
    fig, axes = plt.subplots(2, 1, figsize=(17, 8))
    for ax, aucs, name, color in ((axes[0], buddy_aucs, "Buddy (19 topics)", "#547aa5"),
                                  (axes[1], percept_aucs, "PercepT (40 topics)", "#dc694b")):
        ordered = sorted(aucs.items(), key=lambda item: (-item[1], item[0]))
        ax.bar(np.arange(len(ordered)), [value for _, value in ordered], color=color)
        ax.set_xticks(np.arange(len(ordered)), [str(topic) for topic, _ in ordered], fontsize=7)
        ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.2)
        ax.set_title(name)
        ax.set_ylabel("Held-out AUC")
        ax.set_xlabel("Topic ID, sorted by AUC descending within system")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "topic_auc.png")


def curves_plot(schedule, percept_losses):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs, content, affect = schedule
    axes[0].plot(epochs, content, label="Content held-out recall", color="#42688d")
    axes[0].plot(epochs, affect, label="Affect held-out recall", color="#c27035")
    axes[0].set(title="Buddy zero-noise cosine schedule (baseline trajectory unavailable)",
                xlabel="Buddy training epoch", ylabel="Held-out recall")
    axes[0].legend(fontsize=8)
    ep, kl, recon = percept_losses
    axes[1].plot(ep, kl, color="#95558b", label="DEC KL")
    axes[1].set(title="PercepT faithful Variant A, joint DEC", xlabel="Joint DEC epoch",
                ylabel="DEC KL loss")
    other = axes[1].twinx()
    other.plot(ep, recon, color="#438d75", label="Reconstruction MSE")
    other.set_ylabel("Reconstruction MSE")
    axes[1].legend(loc="upper left", fontsize=8)
    other.legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "training_curves.png")


def source_link(path: Path) -> str:
    return f"[source](../../src/test/{path.parent.name}/{path.name})"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    variants = []
    for name, filename, silhouette_column in VARIANTS:
        source = HERE / filename
        variants.append((name, source, seed_values(source, silhouette_column)))
    faithful = (PERCEPT / "percept_stage1_faithful_recipe_pilot_report.md").read_text(encoding="utf-8")
    row = next(cells(line) for line in faithful.splitlines()
               if line.startswith("| A (pretrain-only noise) | 42 | held-out |"))
    percept_metrics = dict(zip(METRICS, map(float, row[3:6])))

    audit = HERE / "buddy_percept_matched_silhouette_audit_pilot_report.md"
    buddy_counts = parse_occupancy(audit, "### Buddy: frozen train communities", 19)
    percept_counts = parse_occupancy(audit, "### PercepT: surviving train centers", 67)
    buddy_auc_file = HERE / "buddy_stage2_pilot_report.md"
    percept_auc_file = PERCEPT / "percept_stage2_pilot_report.md"
    buddy_aucs = parse_aucs(buddy_auc_file, 19)
    percept_aucs = parse_aucs(percept_auc_file, 40)
    schedule_file = HERE / "attention_h1_noise_schedule_pilot_report.md"
    faithful_file = PERCEPT / "percept_stage1_faithful_recipe_pilot_report.md"
    schedule = parse_schedule_recall(schedule_file)
    percept_losses = parse_percept_losses(faithful_file)

    with np.load(HERE / "attention_h1_embedding_snapshot.npz", allow_pickle=True) as source:
        buddy = {key: source[key] for key in source.files}
    with np.load(PERCEPT / "percept_stage1_faithful_recipe_snapshot.npz", allow_pickle=True) as source:
        percept = {key: source[key] for key in source.files}
    assert set(buddy["heldout_paintings"]) == set(percept["heldout_paintings"])

    tournament_plot(variants, percept_metrics)
    occupancy_plot(buddy_counts, percept_counts)
    buddy_n, buddy_genre_n = projection_plot(
        buddy, "heldout_embedding_post", "heldout_community_post",
        "Buddy Attention-h1 held-out embedding: PCA of all 9,365 paintings", "buddy_pca.png")
    percept_n, percept_genre_n = projection_plot(
        percept, "heldout_latent", "heldout_topic",
        "PercepT faithful held-out latent: PCA of all 9,365 paintings", "percept_pca.png")
    auc_plot(buddy_aucs, percept_aucs)
    curves_plot(schedule, percept_losses)

    # The faithful 67-center snapshot has hard topic assignments but no saved
    # Stage 2 mapper scores. Its topics are distinct from the 40-topic AUC run.
    percept_positions = {name: i for i, name in enumerate(percept["heldout_paintings"])}
    aligned_percept_topic = np.array([percept["heldout_topic"][percept_positions[name]]
                                      for name in buddy["heldout_paintings"]])
    buddy_dom = dominant(buddy["train_community_post"], buddy["train_emotion"])
    percept_dom = dominant(percept["train_topic"], percept["train_emotion"])
    scores, auc_delta = mapper_scores(buddy, buddy_aucs, buddy_counts)
    top = scores.argmax(axis=1)
    truth = np.asarray(buddy["heldout_emotion"]).astype(str)
    good_candidates = [i for i, topic in enumerate(top)
                       if buddy_aucs[int(topic)] >= 0.75 and truth[i] == buddy_dom[int(topic)]]
    good = select_across_topics(good_candidates, top, scores, buddy["heldout_paintings"], 5)
    miss_candidates = [i for i, topic in enumerate(top)
                       if truth[i] != buddy_dom[int(topic)]
                       and percept_dom.get(int(aligned_percept_topic[i])) == truth[i]]
    miss = select_across_topics(miss_candidates, top, scores, buddy["heldout_paintings"], 3)
    assert len(good) == 5 and len(miss) == 3
    chosen = {str(buddy["heldout_paintings"][i]) for i in good + miss}
    captions = first_captions(chosen)
    assert all(captions[str(buddy["heldout_paintings"][i])]["split"] in ("val", "test")
               for i in good + miss)

    lines = [
        "# Comprehensive ArtELingo buddy versus PercepT analysis\n\n",
        "This report analyzes saved Stage 1 snapshots and source pilot reports for 9,365 held-out paintings. "
        "The only new fit is the seed-42 image-only buddy Stage 2 mapper on CPU, needed for individual example scores. "
        "**No raw painting images are present in this environment**; examples therefore show painting IDs, labels, and caption text, not thumbnails. "
        "The Stage 1 results favor different systems by different measures: PercepT has much higher matched-sample silhouette "
        "(0.4973 versus 0.0416), while buddy has far more balanced occupancy. The separate image-only downstream probe "
        "found PercepT slightly stronger for genre (AMI 0.2973 versus 0.2625) and near-tied for emotion "
        "(0.0221 versus 0.0231); its mean-pooled image-feature control exceeded both "
        "(emotion 0.0685, genre 0.3399). See the [investigation](2026-09-26_artelingo_buddy_vs_percept_stage1_report.md), "
        f"{source_link(audit)}, and {source_link(HERE / 'buddy_percept_downstream_probe_pilot_report.md')}.\n\n",
        "## Tournament\n\n",
        "![Held-out tournament](assets/comprehensive_analysis/tournament.png)\n\n",
        "Bars show four-seed held-out means for eight buddy variants; dots show all four actual seed results. "
        "The PercepT faithful recipe is one seed (42), so its bar has no dots or uncertainty estimate. "
        "The dashed AMI lines are the predeclared strict Pareto thresholds (emotion 0.1236, genre 0.1954). "
        "DEC-head silhouettes are measured in the original fused 32-D buddy embedding. "
        "These bars combine buddy pilots' sampled silhouettes with PercepT's published full-split "
        "128-D silhouette (0.5120), so this panel is an overview, not a matched silhouette test; "
        "the matched audit gives 0.0416 versus 0.4973. The faithful PercepT snapshot is a fresh "
        "refit with a different full-split silhouette (0.4840); its assignments feed the PCA, "
        "occupancy, and example panels. "
        "Sources: " + ", ".join(f"[{name}](../../src/test/20260923_artelingo_buddy_analysis/{path.name})"
                                   for name, path, _ in variants) + f", and {source_link(faithful_file)}.\n\n",
        "## Occupancy histograms\n\n",
        "![Held-out occupancy](assets/comprehensive_analysis/occupancy.png)\n\n",
        "These are the exact held-out counts in the matched audit, assigned to frozen train labels: "
        "buddy has 19/19 occupied communities (median 523; 3/19 below 1%), while PercepT has "
        "21/67 empty centers (median 13; 50/67 below 1%). PercepT uses a symmetric-log y-axis "
        f"so zero counts remain visible. Source: {source_link(audit)}.\n\n",
        "## Embedding visualizations\n\n",
        "![Buddy PCA](assets/comprehensive_analysis/buddy_pca.png)\n\n",
        f"PCA projects all {buddy_n:,} buddy held-out 32-D embeddings into 2D. Colors show independently "
        f"computed held-out Leiden communities, majority caption emotion, and the {buddy_genre_n} annotated genres; "
        "unannotated paintings are gray in the genre panel. These independently reclustered held-out "
        "community IDs differ from the frozen train vocabulary used by buddy Stage 2. This is a "
        "visualization, not a clustering quality metric. "
        f"Source: {source_link(HERE / 'attention_h1_embedding_snapshot_pilot_report.md')} and its saved NPZ.\n\n",
        "![PercepT PCA](assets/comprehensive_analysis/percept_pca.png)\n\n",
        f"PCA projects all {percept_n:,} faithful-recipe held-out 128-D latents into 2D. Colors show "
        f"saved center assignments, majority emotion, and {percept_genre_n} annotated genres; "
        f"unannotated paintings are gray. These center labels use the train-fitted vocabulary. "
        f"Source: {source_link(faithful_file)} and its saved NPZ.\n\n",
        "## Per-topic AUC comparison\n\n",
        "![Sorted per-topic AUC](assets/comprehensive_analysis/topic_auc.png)\n\n",
        "Each panel sorts its own reported held-out Stage 2 mapper AUCs, with a 0.5 chance line. "
        "Buddy has 19 topics (macro AUC 0.5978); PercepT has 40 (0.5690). Topic IDs and "
        "the underlying Stage 1 runs differ, so bars at the same rank or ID are not matched topics. "
        f"Sources: {source_link(buddy_auc_file)} and {source_link(percept_auc_file)}.\n\n",
        "## Training curves\n\n",
        "![Logged training curves](assets/comprehensive_analysis/training_curves.png)\n\n",
        "Left: logged content and affect held-out recall at checkpoints for the zero-noise cosine-schedule "
        "Attention-h1 run. The plain fixed-LR baseline's report and snapshot contain no saved checkpoint "
        "trajectory, so that requested exact curve remains unavailable without another training run. "
        "Right: PercepT faithful "
        "Variant A joint DEC KL and reconstruction MSE at its logged checkpoints, on separate y-axes. "
        "Epoch axes are separate because the objectives and phases differ. "
        f"Sources: {source_link(schedule_file)} and {source_link(faithful_file)}.\n\n",
        "## Statistics\n\n",
        "Each four-seed row uses seeds 42, 7, 123, and 2024 in that order. Standard deviations are "
        "sample SD (ddof=1); 95% confidence intervals are percentile bootstrap intervals for the "
        "mean from 10,000 size-four resamples with replacement using `np.random.default_rng(42)`. "
        "With n=4, these intervals are indicative and cannot support a rigorous significance claim. "
        "The PercepT faithful recipe has only seed 42 and has no four-seed CI.\n\n",
        "| Variant | Metric | Seeds 42 / 7 / 123 / 2024 | Mean | Sample SD | 95% bootstrap CI |\n",
        "|---|---|---|---:|---:|---:|\n",
    ]
    for name, _, data in variants:
        for metric in METRICS:
            vals = data[metric]
            _mean, sd, low, high = bootstrap(vals)
            lines.append(f"| {name} | {metric} | {' / '.join(f'{x:.4f}' for x in vals)} | "
                         f"{reported_mean(vals)} | {sd:.4f} | [{low:.4f}, {high:.4f}] |\n")
    lines.extend([
        "\nPercepT faithful seed-42 held-out values: emotion AMI "
        f"{percept_metrics['emotion AMI']:.4f}, genre AMI {percept_metrics['genre AMI']:.4f}, "
        f"silhouette {percept_metrics['silhouette']:.4f} ({source_link(faithful_file)}). "
        "The four-seed source rows supersede rounded or inconsistent summary prose: notably, "
        "the investigation overview lists vMF DEC as 0.1210/0.1567/0.0303, while its own "
        "four-seed table gives 0.1215/0.1504/0.0298. The baseline stress report's seed-42 "
        "silhouette is 0.0377; a later schedule report's reference row lists 0.0392, so the "
        "tournament uses the direct stress-table value.\n\n",
        "## Qualitative examples\n\n",
        "The seed-42 buddy Stage 2 mapper was fitted fresh on CPU with the sibling pilot's "
        "100-epoch full-batch BCE recipe and aligned patch-cache order; no mapper checkpoint or "
        "individual held-out score cache existed. Its final BCE matched the saved 0.200270 and its "
        f"19 per-topic AUCs differed from the saved table by at most {auc_delta:.6f}. "
        "Top-1 means the mapper's highest sigmoid score. "
        "Topic dominant emotion is the most common majority emotion among that topic's train paintings "
        "(alphabetical tie break). `True emotion` is the held-out snapshot's painting-majority label; "
        "`first-caption emotion` and caption come from the first CSV row for that painting. "
        "The PercepT assignment is its saved faithful 67-center hard assignment, whose train-topic "
        "dominant emotion is computed the same way. Its own per-painting Stage 2 scores cannot be "
        "reproduced from the snapshot alone; the 40-topic AUC report is from a different model. "
        f"Only {len(set(int(top[i]) for i in good_candidates))} high-AUC topics produced eligible "
        f"emotion-matching top-1 candidates, and {len(set(int(top[i]) for i in miss_candidates))} "
        "buddy topics produced contrasting-miss candidates; the tables reflect that concentration. "
        "These selected examples are illustrative, not random or representative, and offer qualitative "
        "color rather than evidence of general performance.\n\n",
        "### Five high-AUC, emotion-matching buddy examples\n\n",
        "Candidates have a top-1 buddy topic with reported AUC ≥0.75 and a majority true emotion "
        "matching that topic's dominant emotion; five are selected one per eligible topic first, "
        "then by descending top-1 score where another row is needed. "
        f"AUC source: {source_link(buddy_auc_file)}.\n\n",
    ])
    header = "| Painting ID | Art style | True emotion | First-caption emotion | True genre | Buddy topic | Buddy dominant emotion | Buddy topic AUC | PercepT center | PercepT dominant emotion | First caption |\n"
    divider = "|---|---|---|---|---|---:|---|---:|---:|---|---|\n"
    lines += [header, divider] + [row + "\n" for row in example_rows(
        good, buddy, scores, buddy_aucs, captions, buddy_dom, aligned_percept_topic, percept_dom)]
    lines += [
        "\n### Three buddy misses with a contrasting PercepT assignment\n\n",
        "Candidates have a buddy top-1 topic whose dominant emotion differs from the painting's "
        "majority true emotion, while the saved PercepT center's dominant emotion matches it; "
        "three are selected across different eligible buddy topics first, then by descending "
        "top-1 score if another row is needed. This comparison uses category summaries, "
        "not unavailable PercepT mapper probabilities.\n\n",
        header, divider,
    ] + [row + "\n" for row in example_rows(
        miss, buddy, scores, buddy_aucs, captions, buddy_dom, aligned_percept_topic, percept_dom)]
    lines += [
        "\n## Reproduce\n\n",
        "Run from the repository root with "
        "`PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python "
        "src/test/20260923_artelingo_buddy_analysis/build_comprehensive_analysis.py`. "
        "This script was actually executed to create the six PNGs embedded above and this report. "
        "The tournament reads per-seed rows from the eight cited variant reports and the PercepT "
        "faithful seed-42 row. Occupancy reads the audit's exact count tables. Both PCA plots read "
        "the two saved snapshots. The AUC panels read the two Stage 2 per-topic tables. Training "
        "curves read the zero-noise schedule checkpoint table and PercepT Variant A joint DEC log. "
        "The example tables refit only the buddy image mapper on CPU, align its patch cache by "
        "painting ID, and read the first caption rows from the ArtELingo CSV. UMAP was unavailable "
        "in the CoSiR environment, so PCA was used on all held-out points; plotting uses matplotlib.\n",
    ]
    OUTPUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    for path in sorted(ASSETS.glob("*.png")):
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
