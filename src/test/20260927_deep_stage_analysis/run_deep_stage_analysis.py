"""Deep Stage 1/2 analysis: topic characterization, per-topic AUC vs
occupancy, cross-system topic alignment, emotion-conditioned error
analysis, and a synthesis of concrete next ideas for buddy's own Stage 2.

Consumes two snapshots:
- buddy: src/test/20260923_artelingo_buddy_analysis/attention_h1_embedding_snapshot.npz
- PercepT (bug-fixed): percept_fixed_snapshot.npz (this directory, from
  run_percept_fixed_snapshot_pilot.py)

Read-only against all existing shared/frozen files; imports their
functions rather than reimplementing them.
"""

import csv
import importlib.util
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch import nn


HERE = Path(__file__).resolve().parent
PERCEPT_DIR = HERE.parents[1] / "test/20260922_percept_topic_pipeline"
BUDDY_DIR = HERE.parents[1] / "test/20260923_artelingo_buddy_analysis"
ASSETS_DIR = HERE / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

BUDDY_SNAPSHOT = BUDDY_DIR / "attention_h1_embedding_snapshot.npz"
PERCEPT_SNAPSHOT = HERE / "percept_fixed_snapshot.npz"
CAPTIONS_CSV = Path(
    "/data/PDD/artelingo/ArtELingo/ArtELingo/Dataset/artelingo_release_lite.csv"
)
SEED = 42
N_BUDDY_TOPICS = 19
TRANSFER_K = 20
REPORT_PATH = HERE / "deep_stage_analysis_report.md"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def load_english_captions(csv_path: Path) -> dict[str, list[str]]:
    """painting -> list of English utterances (multiple annotators)."""
    captions: dict[str, list[str]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("language", "").strip().lower() != "english":
                continue
            painting = row.get("painting", "").strip()
            utterance = row.get("utterance", "").strip()
            if painting and utterance:
                captions[painting].append(utterance)
    return captions


def shannon_entropy(labels: list[str]) -> float:
    if not labels:
        return float("nan")
    counts = Counter(labels)
    total = sum(counts.values())
    return float(-sum((c / total) * np.log2(c / total) for c in counts.values()))


def occupancy_from_labels(labels: np.ndarray, n_topics: int) -> np.ndarray:
    return np.bincount(labels, minlength=n_topics)


def topic_characterization(
    name: str, topics: np.ndarray, paintings: np.ndarray, emotions: np.ndarray,
    genres: np.ndarray, n_topics: int, captions: dict[str, list[str]], rng: random.Random,
) -> list[dict]:
    rows = []
    for topic in range(n_topics):
        mask = topics == topic
        size = int(mask.sum())
        if size == 0:
            rows.append({
                "system": name, "topic": topic, "size": 0,
                "majority_emotion": "n/a", "emotion_share": float("nan"),
                "majority_genre": "n/a", "genre_share": float("nan"),
                "genre_n": 0, "entropy": float("nan"), "captions": [],
            })
            continue
        topic_emotions = [str(e) for e in emotions[mask]]
        emotion_counts = Counter(topic_emotions)
        maj_emotion, maj_emotion_count = emotion_counts.most_common(1)[0]
        topic_genres = [str(g) for g in genres[mask] if str(g) not in ("", "nan", "None")]
        if topic_genres:
            genre_counts = Counter(topic_genres)
            maj_genre, maj_genre_count = genre_counts.most_common(1)[0]
            genre_share = maj_genre_count / len(topic_genres)
        else:
            maj_genre, genre_share = "n/a", float("nan")
        entropy = shannon_entropy(topic_emotions)
        topic_paintings = list(paintings[mask])
        rng.shuffle(topic_paintings)
        sampled_captions = []
        for painting in topic_paintings:
            caption_list = captions.get(str(painting))
            if caption_list:
                sampled_captions.append(rng.choice(caption_list))
            if len(sampled_captions) == 3:
                break
        rows.append({
            "system": name, "topic": topic, "size": size,
            "majority_emotion": maj_emotion, "emotion_share": maj_emotion_count / size,
            "majority_genre": maj_genre, "genre_share": genre_share,
            "genre_n": len(topic_genres), "entropy": entropy, "captions": sampled_captions,
        })
    return rows


def auc_vs_occupancy_plot(
    name: str, per_topic_auc: np.ndarray, occupancy: np.ndarray, out_path: Path,
) -> dict:
    valid = ~np.isnan(per_topic_auc)
    x = occupancy[valid].astype(float)
    y = per_topic_auc[valid]
    if len(x) < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan"), "n": len(x)}
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("topic size (held-out occupancy)")
    plt.ylabel("per-topic AUC")
    plt.title(f"{name}: per-topic AUC vs. occupancy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return {
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p), "n": len(x),
    }


def top1_accuracy_by_group(
    scores: np.ndarray, true_topic: np.ndarray, groups: np.ndarray,
) -> dict[str, tuple[float, int]]:
    predicted = scores.argmax(axis=1)
    correct = predicted == true_topic
    result = {}
    for group in sorted(set(str(g) for g in groups)):
        mask = np.array([str(g) == group for g in groups])
        if mask.sum() == 0:
            continue
        result[group] = (float(correct[mask].mean()), int(mask.sum()))
    return result


def main() -> None:
    rng = random.Random(SEED)

    log("Loading buddy snapshot...")
    buddy = load_npz(BUDDY_SNAPSHOT)
    log("Loading PercepT fixed snapshot...")
    percept = load_npz(PERCEPT_SNAPSHOT)

    audit = load_module("audit_for_deep_analysis", BUDDY_DIR / "run_buddy_percept_matched_silhouette_audit_pilot.py")
    transfer = load_module("transfer_for_deep_analysis", BUDDY_DIR / "run_heldout_label_transfer_pilot.py")
    stage2_ref = load_module("stage2_ref_for_deep_analysis", PERCEPT_DIR / "run_percept_stage2_pilot.py")

    buddy_heldout_paintings = buddy["heldout_paintings"]
    percept_heldout_paintings = percept["heldout_paintings"]
    audit.assert_matching_paintings(buddy_heldout_paintings, percept_heldout_paintings, "held-out", len(buddy_heldout_paintings))
    log(f"Confirmed matching held-out painting ID sets: {len(buddy_heldout_paintings)} paintings.")

    # --- Buddy: compute held-out topic labels via the same k=20 transfer
    # buddy_stage2_pilot.py uses, and retrain its Stage-2 mapper to capture
    # predictions (never persisted by that pilot).
    train_embeddings = buddy["train_embedding_post"]
    train_labels = buddy["train_community_post"].astype(np.int64)
    heldout_embeddings = buddy["heldout_embedding_post"]
    buddy_heldout_topic = transfer.assign_to_train_communities(
        train_embeddings, train_labels, heldout_embeddings, k=TRANSFER_K
    ).astype(np.int64)
    log(f"Buddy held-out topics assigned via k={TRANSFER_K} transfer; "
        f"{len(np.unique(buddy_heldout_topic))}/{N_BUDDY_TOPICS} train communities represented.")

    def one_hot(labels: np.ndarray, n_topics: int) -> torch.Tensor:
        targets = torch.zeros((len(labels), n_topics), dtype=torch.float32)
        targets.scatter_(1, torch.as_tensor(labels, dtype=torch.long)[:, None], 1.0)
        return targets

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_patches = stage2_ref.load_patch_features(
        stage2_ref.TRAIN_PATCH_FEATURE_PATH, len(buddy["train_paintings"]), "train"
    ).to(device)
    heldout_patches = stage2_ref.load_patch_features(
        stage2_ref.HELDOUT_PATCH_FEATURE_PATH, len(buddy_heldout_paintings), "held-out"
    ).to(device)
    buddy_train_targets = one_hot(train_labels, N_BUDDY_TOPICS).to(device)
    buddy_heldout_targets = one_hot(buddy_heldout_topic, N_BUDDY_TOPICS).cpu().numpy()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    buddy_mapper = stage2_ref.AttentionPoolingMapper(n_topics=N_BUDDY_TOPICS).to(device)
    optimizer = torch.optim.Adam(buddy_mapper.parameters(), lr=stage2_ref.MAPPER_LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    buddy_mapper.train()
    for epoch in range(1, stage2_ref.MAPPER_EPOCHS + 1):
        loss = loss_fn(buddy_mapper(train_patches), buddy_train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            log(f"Buddy Stage-2 mapper epoch {epoch}/{stage2_ref.MAPPER_EPOCHS}: BCE={loss.item():.6f}")
    buddy_mapper.eval()
    with torch.no_grad():
        buddy_heldout_scores = torch.sigmoid(buddy_mapper(heldout_patches)).cpu().numpy()
    buddy_model_aucs, buddy_skipped = stage2_ref.evaluate_auc(
        buddy_heldout_scores, buddy_heldout_targets, log, "buddy mapper"
    )
    buddy_macro_auc = stage2_ref.auc_summary(buddy_model_aucs)["macro"]
    log(f"Buddy Stage-2 macro AUC (this run): {buddy_macro_auc:.4f} "
        "(cross-check against buddy_stage2_pilot_report.md's 0.5978).")

    percept_topic_count = int(percept["heldout_topic"].max()) + 1
    percept_heldout_scores = percept["heldout_stage2_scores"]
    percept_heldout_targets = percept["heldout_stage2_targets"]
    percept_per_topic_auc = percept["per_topic_auc"]

    buddy_per_topic_auc = np.full(N_BUDDY_TOPICS, np.nan)
    for topic, auc in buddy_model_aucs.items():
        buddy_per_topic_auc[topic] = auc

    captions = load_english_captions(CAPTIONS_CSV)
    log(f"Loaded English captions for {len(captions)} paintings.")

    # --- Analysis 1: topic characterization ---
    log("Analysis 1: topic characterization...")
    buddy_topics_char = topic_characterization(
        "buddy", buddy_heldout_topic, buddy_heldout_paintings, buddy["heldout_emotion"],
        buddy["heldout_genre"], N_BUDDY_TOPICS, captions, rng,
    )
    percept_topics_char = topic_characterization(
        "percept", percept["heldout_topic"], percept_heldout_paintings, percept["heldout_emotion"],
        percept["heldout_genre"], percept_topic_count, captions, rng,
    )

    # --- Analysis 2: AUC vs occupancy ---
    log("Analysis 2: per-topic AUC vs occupancy...")
    buddy_occupancy = occupancy_from_labels(buddy_heldout_topic, N_BUDDY_TOPICS)
    percept_occupancy = occupancy_from_labels(percept["heldout_topic"], percept_topic_count)
    buddy_corr = auc_vs_occupancy_plot("buddy", buddy_per_topic_auc, buddy_occupancy, ASSETS_DIR / "buddy_auc_vs_occupancy.png")
    percept_corr = auc_vs_occupancy_plot("percept", percept_per_topic_auc, percept_occupancy, ASSETS_DIR / "percept_auc_vs_occupancy.png")
    log(f"Buddy AUC-occupancy Spearman r={buddy_corr['spearman_r']:.3f}; "
        f"PercepT Spearman r={percept_corr['spearman_r']:.3f}")

    # --- Analysis 3: cross-system alignment ---
    log("Analysis 3: cross-system topic alignment...")
    # Align both label arrays to the same painting order (buddy's held-out order).
    percept_position = {p: i for i, p in enumerate(percept_heldout_paintings)}
    percept_topic_aligned = np.array(
        [percept["heldout_topic"][percept_position[p]] for p in buddy_heldout_paintings]
    )
    nmi = normalized_mutual_info_score(buddy_heldout_topic, percept_topic_aligned)
    contingency = np.zeros((N_BUDDY_TOPICS, percept_topic_count), dtype=np.int64)
    for b, p in zip(buddy_heldout_topic, percept_topic_aligned):
        contingency[b, p] += 1
    buddy_to_percept = {}
    for b in range(N_BUDDY_TOPICS):
        row = contingency[b]
        if row.sum() == 0:
            continue
        top = np.argsort(-row)[:3]
        buddy_to_percept[b] = [(int(t), int(row[t])) for t in top if row[t] > 0]
    percept_to_buddy = {}
    for p in range(percept_topic_count):
        col = contingency[:, p]
        if col.sum() == 0:
            continue
        top = np.argsort(-col)[:3]
        percept_to_buddy[p] = [(int(t), int(col[t])) for t in top if col[t] > 0]
    log(f"Cross-system NMI: {nmi:.4f}")

    # --- Analysis 4: emotion-conditioned error analysis ---
    log("Analysis 4: emotion-conditioned error analysis...")
    buddy_acc_by_emotion = top1_accuracy_by_group(buddy_heldout_scores, buddy_heldout_topic, buddy["heldout_emotion"])
    percept_acc_by_emotion = top1_accuracy_by_group(percept_heldout_scores, percept["heldout_topic"], percept["heldout_emotion"])
    buddy_acc_by_genre = top1_accuracy_by_group(
        buddy_heldout_scores[[i for i, g in enumerate(buddy["heldout_genre"]) if str(g) not in ("", "nan")]],
        buddy_heldout_topic[[i for i, g in enumerate(buddy["heldout_genre"]) if str(g) not in ("", "nan")]],
        buddy["heldout_genre"][[i for i, g in enumerate(buddy["heldout_genre"]) if str(g) not in ("", "nan")]],
    )
    percept_acc_by_genre = top1_accuracy_by_group(
        percept_heldout_scores[[i for i, g in enumerate(percept["heldout_genre"]) if str(g) not in ("", "nan")]],
        percept["heldout_topic"][[i for i, g in enumerate(percept["heldout_genre"]) if str(g) not in ("", "nan")]],
        percept["heldout_genre"][[i for i, g in enumerate(percept["heldout_genre"]) if str(g) not in ("", "nan")]],
    )

    write_report(
        buddy_topics_char, percept_topics_char, buddy_corr, percept_corr,
        nmi, buddy_to_percept, percept_to_buddy, contingency,
        buddy_acc_by_emotion, percept_acc_by_emotion,
        buddy_acc_by_genre, percept_acc_by_genre,
        buddy_macro_auc, N_BUDDY_TOPICS, percept_topic_count,
    )
    log(f"Wrote {REPORT_PATH}")


def write_report(
    buddy_topics_char, percept_topics_char, buddy_corr, percept_corr,
    nmi, buddy_to_percept, percept_to_buddy, contingency,
    buddy_acc_by_emotion, percept_acc_by_emotion,
    buddy_acc_by_genre, percept_acc_by_genre,
    buddy_macro_auc, n_buddy_topics, n_percept_topics,
) -> None:
    lines = [
        "# Deep Stage 1 / Stage 2 analysis: buddy vs. fixed PercepT\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Companion to "
        "[`docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`]"
        "(../../../docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md) "
        "§6b. Buddy has 19 held-out topics (k=20 transfer); PercepT (bug-fixed, "
        f"K=60/40) has {n_percept_topics} surviving held-out topics.\n\n",
        f"Buddy Stage-2 macro AUC in this run: {buddy_macro_auc:.4f} "
        "(cross-check against `buddy_stage2_pilot_report.md`'s 0.5978 -- minor "
        "differences are expected from a fresh mapper-init draw at the same seed "
        "if any upstream randomness source differs; treat close agreement as a "
        "sanity pass).\n\n",
        "## 1. Topic characterization\n\n",
        "Majority emotion/genre and share are of the topic's held-out members; "
        "genre share is only over the genre-labelled subset (thin sample, see "
        "`genre_n`). Entropy is the Shannon entropy (bits) of the topic's "
        "emotion-label distribution -- lower means more emotionally homogeneous. "
        "Captions are a seed-42 sample of up to 3 held-out members' first "
        "English caption.\n\n",
        "### Buddy (19 topics)\n\n",
        "| topic | size | majority emotion | share | majority genre | genre share | genre n | entropy | example captions |\n",
        "|---:|---:|---|---:|---|---:|---:|---:|---|\n",
    ]
    for row in buddy_topics_char:
        captions_str = " / ".join(c[:80] for c in row["captions"]) or "n/a"
        lines.append(
            f"| {row['topic']} | {row['size']} | {row['majority_emotion']} | "
            f"{row['emotion_share']:.2f} | {row['majority_genre']} | "
            f"{row['genre_share']:.2f} | {row['genre_n']} | {row['entropy']:.2f} | "
            f"{captions_str} |\n"
        )
    lines.append(f"\n### PercepT, bug-fixed ({n_percept_topics} topics)\n\n")
    lines.append("| topic | size | majority emotion | share | majority genre | genre share | genre n | entropy | example captions |\n")
    lines.append("|---:|---:|---|---:|---|---:|---:|---:|---|\n")
    for row in percept_topics_char:
        captions_str = " / ".join(c[:80] for c in row["captions"]) or "n/a"
        lines.append(
            f"| {row['topic']} | {row['size']} | {row['majority_emotion']} | "
            f"{row['emotion_share']:.2f} | {row['majority_genre']} | "
            f"{row['genre_share']:.2f} | {row['genre_n']} | {row['entropy']:.2f} | "
            f"{captions_str} |\n"
        )

    lines.extend([
        "\n## 2. Per-topic AUC vs. occupancy\n\n",
        f"Buddy: Pearson r={buddy_corr['pearson_r']:.3f} (p={buddy_corr['pearson_p']:.3f}), "
        f"Spearman r={buddy_corr['spearman_r']:.3f} (p={buddy_corr['spearman_p']:.3f}), "
        f"n={buddy_corr['n']} topics. "
        f"![buddy AUC vs occupancy](assets/buddy_auc_vs_occupancy.png)\n\n",
        f"PercepT: Pearson r={percept_corr['pearson_r']:.3f} (p={percept_corr['pearson_p']:.3f}), "
        f"Spearman r={percept_corr['spearman_r']:.3f} (p={percept_corr['spearman_p']:.3f}), "
        f"n={percept_corr['n']} topics. "
        f"![percept AUC vs occupancy](assets/percept_auc_vs_occupancy.png)\n\n",
    ])

    lines.extend([
        "## 3. Cross-system topic alignment\n\n",
        f"Normalized mutual information (buddy 19-way vs. PercepT "
        f"{n_percept_topics}-way, same {contingency.sum()} held-out paintings): "
        f"**NMI = {nmi:.4f}**.\n\n",
        "Top PercepT topics each buddy community maps onto most heavily "
        "(topic: shared-painting count):\n\n",
    ])
    for b, targets in buddy_to_percept.items():
        targets_str = ", ".join(f"P{t}:{c}" for t, c in targets)
        lines.append(f"- Buddy {b} -> {targets_str}\n")
    lines.append("\nTop buddy communities each PercepT topic maps onto most heavily:\n\n")
    for p, targets in percept_to_buddy.items():
        targets_str = ", ".join(f"B{t}:{c}" for t, c in targets)
        lines.append(f"- PercepT {p} -> {targets_str}\n")

    lines.extend([
        "\n## 4. Emotion-conditioned error analysis\n\n",
        "Top-1 accuracy = predicted topic (argmax mapper score) equals the "
        "painting's true hard-assigned topic. Full held-out set (9,365 "
        "paintings), grouped by ground-truth majority emotion.\n\n",
        "| emotion | buddy top-1 acc | buddy n | PercepT top-1 acc | PercepT n |\n",
        "|---|---:|---:|---:|---:|\n",
    ])
    all_emotions = sorted(set(buddy_acc_by_emotion) | set(percept_acc_by_emotion))
    for emotion in all_emotions:
        b_acc, b_n = buddy_acc_by_emotion.get(emotion, (float("nan"), 0))
        p_acc, p_n = percept_acc_by_emotion.get(emotion, (float("nan"), 0))
        lines.append(f"| {emotion} | {b_acc:.3f} | {b_n} | {p_acc:.3f} | {p_n} |\n")

    lines.extend([
        "\n### Genre-conditioned (thin sample, indicative only -- ~159 "
        "genre-labelled held-out paintings)\n\n",
        "| genre | buddy top-1 acc | buddy n | PercepT top-1 acc | PercepT n |\n",
        "|---|---:|---:|---:|---:|\n",
    ])
    all_genres = sorted(set(buddy_acc_by_genre) | set(percept_acc_by_genre))
    for genre in all_genres:
        b_acc, b_n = buddy_acc_by_genre.get(genre, (float("nan"), 0))
        p_acc, p_n = percept_acc_by_genre.get(genre, (float("nan"), 0))
        lines.append(f"| {genre} | {b_acc:.3f} | {b_n} | {p_acc:.3f} | {p_n} |\n")

    lines.append(
        "\n## 5. Synthesis -- concrete next ideas\n\n"
        "(Written after inspecting the tables above -- see the companion "
        "report for how each idea was prioritized into the candidate list.)\n"
    )
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
