"""ArtELingo buddy-graph vs. ground-truth-label analysis — full unattended pipeline.

Runs standalone (no CoSiRModel/cuml import — reuses the same pure numpy/scipy
buddy-graph primitives as src/test/20260623_redcaps_buddy/extract_features.py
and src/conditional_buddy/prototype_seed.py, both already cuml-free).

Sequence:
  1. Wait for the ArtELingo CLIP feature extraction
     (/data/SSD2/pre_extract/artelingo/features) to finish.
  2. Load cached img/txt features + artelingo_train.json, dedup from
     308,723 caption-rows down to 61,402 unique-painting nodes (image
     embedding as-is; text embedding = mean of that painting's captions).
  3. Build the buddy graph (mutual-kNN, K=20 — chosen smaller than the
     validated K=30 default because this corpus, at ~61k unique nodes, is
     smaller than any scale Experiment 16.1 tuned K for; this is a judgment
     call, not a validated choice, and should be revisited if results look
     off) and run Leiden community detection — same primitives as
     Experiment 18's buddy-seeding path.
  4. Build one master per-painting table: community_id, majority English
     emotion (+ full count dict, for disagreement/rare-class analysis),
     Arabic/Chinese majority emotion (joined from the raw multilingual CSV,
     for the cross-lingual consistency test), and genre where available
     (from the 1,144-painting train-split overlap with the ArtELingo-28
     diagnostic set).
  5. Run all 5 analyses proposed to the user on 2026-09-22 and write a
     single markdown findings report.

This script is the reliability-critical path — it does NOT depend on Codex
being available or on any confirmation gate resolving correctly, since the
user is offline overnight. A separate Codex dispatch (independent
cross-check + plots) runs in parallel and is explicitly non-blocking.
"""

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy.sparse import save_npz
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.conditional_buddy.compute_buddies import build_buddy_graphs
from src.conditional_buddy.prototype_seed import detect_communities
from src.utils import FeatureManager

PERCEPT_FEATURE_ROOT = os.environ.get("PERCEPT_FEATURE_ROOT", "/data/SSD2/pre_extract")
PERCEPT_RAW_JSON_ROOT = os.environ.get("PERCEPT_RAW_JSON_ROOT", "/data/PDD/artelingo")
STORAGE_DIR = f"{PERCEPT_FEATURE_ROOT}/artelingo/features"
TRAIN_JSON = f"{PERCEPT_RAW_JSON_ROOT}/artelingo_train.json"
GENRE_JSON = f"{PERCEPT_RAW_JSON_ROOT}/artelingo_genre_emotion_eng.json"
MAIN_CSV = "/data/PDD/artelingo/ArtELingo/ArtELingo/Dataset/artelingo_release_lite.csv"
OUT_DIR = os.path.dirname(__file__)

K = 20
ALPHA = 0.5
SEED = 42

POSITIVE = {"amusement", "awe", "contentment", "excitement"}
NEGATIVE = {"anger", "disgust", "fear", "sadness"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def assert_extraction_complete():
    """Fail fast (not silently proceed on a partial store) if extraction isn't done.

    metadata.json is written at open_for_writing() time (extraction START), not
    just at finalize_writing() — so its mere existence is NOT a valid completion
    signal. This script is only ever launched by the orchestrating session after
    it has received the actual background-task completion notification for the
    extraction run; this check is a defensive correctness assertion, not the
    primary synchronization mechanism.
    """
    meta_path = os.path.join(STORAGE_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"{meta_path} does not exist — extraction has not started.")
    expected = len(json.load(open(TRAIN_JSON)))
    fm = FeatureManager(STORAGE_DIR)
    actual = len(fm.get_all_sample_ids())
    if actual != expected:
        raise RuntimeError(
            f"Feature store at {STORAGE_DIR} has {actual} sample ids, expected "
            f"{expected} — extraction looks incomplete or failed partway. "
            f"Check /tmp/artelingo_extract.log before re-running this pipeline."
        )
    log(f"Extraction verified complete: {actual} sample ids match {TRAIN_JSON}.")


def load_dedup_features():
    train = json.load(open(TRAIN_JSON))
    fm = FeatureManager(STORAGE_DIR)
    sample_ids = fm.get_all_sample_ids()
    feats = fm.load_all_to_ram(["img_features", "txt_features"])
    img_all = feats["img_features"].numpy()
    txt_all = feats["txt_features"].numpy()
    # sample_ids are row indices into train.json (see cosir_datamodule.py:34)
    id_to_row = {sid: i for i, sid in enumerate(sample_ids)}

    by_painting = defaultdict(list)
    for row_idx, rec in enumerate(train):
        by_painting[rec["painting"]].append(row_idx)

    paintings = sorted(by_painting.keys())
    img_dim = img_all.shape[1]
    txt_dim = txt_all.shape[1]
    img_nodes = np.zeros((len(paintings), img_dim), dtype=np.float32)
    txt_nodes = np.zeros((len(paintings), txt_dim), dtype=np.float32)
    emotion_counts = []
    img_consistency_checked = 0
    img_consistency_min_cos = 1.0

    for i, painting in enumerate(paintings):
        row_idxs = by_painting[painting]
        feat_rows = [id_to_row[r] for r in row_idxs if r in id_to_row]
        if not feat_rows:
            raise RuntimeError(f"No extracted features found for painting {painting}")
        imgs = img_all[feat_rows]
        txts = txt_all[feat_rows]
        img_nodes[i] = imgs[0]
        txt_nodes[i] = txts.mean(axis=0)
        if len(feat_rows) > 1 and img_consistency_checked < 500:
            a = imgs[0] / (np.linalg.norm(imgs[0]) + 1e-8)
            for j in range(1, len(feat_rows)):
                b = imgs[j] / (np.linalg.norm(imgs[j]) + 1e-8)
                cos = float(np.dot(a, b))
                img_consistency_min_cos = min(img_consistency_min_cos, cos)
            img_consistency_checked += 1
        emotion_counts.append(Counter(train[r]["emotion"] for r in row_idxs))

    log(
        f"Dedup: {len(train)} rows -> {len(paintings)} unique paintings. "
        f"Image-embedding self-consistency check (same painting, different caption "
        f"rows, n={img_consistency_checked}): min cosine sim = {img_consistency_min_cos:.4f} "
        f"(should be ~1.0 — same image every time; low value would mean a data bug)."
    )
    return paintings, img_nodes, txt_nodes, emotion_counts


def load_genre_map():
    genre_data = json.load(open(GENRE_JSON))
    genre_by_painting = {}
    for rec in genre_data:
        genre_by_painting.setdefault(rec["painting"], Counter())[rec["genre"]] += 1
    return {p: c.most_common(1)[0][0] for p, c in genre_by_painting.items()}


def load_other_language_emotions(paintings_set):
    ar = defaultdict(Counter)
    zh = defaultdict(Counter)
    with open(MAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["painting"] not in paintings_set:
                continue
            if row["language"] == "arabic":
                ar[row["painting"]][row["emotion"]] += 1
            elif row["language"] == "chinese":
                zh[row["painting"]][row["emotion"]] += 1
    ar_majority = {p: c.most_common(1)[0][0] for p, c in ar.items()}
    zh_majority = {p: c.most_common(1)[0][0] for p, c in zh.items()}
    return ar_majority, zh_majority


def majority(counter: Counter):
    return counter.most_common(1)[0][0]


def coarse_valence(emotion: str) -> str:
    if emotion in POSITIVE:
        return "positive"
    if emotion in NEGATIVE:
        return "negative"
    return "something else"


def external_metrics(labels_a, labels_b):
    return {
        "AMI": adjusted_mutual_info_score(labels_a, labels_b),
        "NMI": normalized_mutual_info_score(labels_a, labels_b),
        "V_measure": v_measure_score(labels_a, labels_b),
        "ARI": adjusted_rand_score(labels_a, labels_b),
    }


def main():
    assert_extraction_complete()

    log("Loading + deduping features...")
    paintings, img_nodes, txt_nodes, emotion_counts_list = load_dedup_features()
    n = len(paintings)
    painting_to_idx = {p: i for i, p in enumerate(paintings)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Building buddy graph (K={K}, alpha={ALPHA}, device={device}) on {n} nodes...")
    if device == "cpu":
        log("CUDA unavailable (NVML init failure observed earlier this session) — "
            "falling back to CPU. Confirmed correct and feasible on a synthetic "
            "N=61402 timing test before this run started.")
    A_img, A_txt, E = build_buddy_graphs(
        img_nodes, txt_nodes, K=K, alpha=ALPHA, device=device, connect_components=True
    )
    save_npz(os.path.join(OUT_DIR, "buddy_graph_E.npz"), E)

    log("Running Leiden community detection...")
    community = detect_communities(E, seed=SEED)
    n_communities = int(community.max()) + 1
    log(f"Found {n_communities} communities over {n} painting-nodes.")

    majority_emotion = [majority(c) for c in emotion_counts_list]
    valence = [coarse_valence(e) for e in majority_emotion]
    n_annotations = [sum(c.values()) for c in emotion_counts_list]

    genre_map = load_genre_map()
    genre = [genre_map.get(p) for p in paintings]
    has_genre = np.array([g is not None for g in genre])
    log(f"Genre available for {has_genre.sum()} / {n} paintings (diagnostic-set overlap).")

    log("Joining Arabic/Chinese emotion labels for cross-lingual test...")
    ar_majority, zh_majority = load_other_language_emotions(set(paintings))
    arabic_emotion = [ar_majority.get(p) for p in paintings]
    chinese_emotion = [zh_majority.get(p) for p in paintings]
    log(
        f"Arabic labels for {sum(1 for x in arabic_emotion if x)} paintings, "
        f"Chinese labels for {sum(1 for x in chinese_emotion if x)} paintings."
    )

    # --- Persist the master table ---
    table = []
    for i, p in enumerate(paintings):
        table.append({
            "painting": p,
            "community_id": int(community[i]),
            "n_annotations": n_annotations[i],
            "emotion_counts": dict(emotion_counts_list[i]),
            "majority_emotion": majority_emotion[i],
            "valence": valence[i],
            "genre": genre[i],
            "arabic_emotion": arabic_emotion[i],
            "chinese_emotion": chinese_emotion[i],
        })
    with open(os.path.join(OUT_DIR, "painting_community_table.json"), "w") as f:
        json.dump(table, f)
    log("Saved painting_community_table.json")

    # ================= ANALYSES =================
    report = []
    report.append("# ArtELingo buddy-graph vs. ground-truth-label analysis\n")
    report.append(f"Generated automatically overnight, {time.strftime('%Y-%m-%d %H:%M:%S')}.\n")
    report.append(
        f"\n**Setup:** {n} unique painting-nodes (deduped from {sum(n_annotations)} English "
        f"caption-rows), buddy graph K={K}, alpha={ALPHA}, Leiden community detection "
        f"({n_communities} communities found), seed={SEED}.\n"
    )
    report.append(
        "**Caveat on K:** chosen as a smaller-than-default judgment call for this node "
        "count (~61k, below any scale Experiment 16.1 validated K for) — not itself "
        "validated. Revisit if results look off.\n"
    )

    community_labels = community.tolist()

    # 1. Genre <-> Emotion baseline correlation (on the genre-labeled subset)
    report.append("\n## 1. Genre <-> Emotion ground-truth correlation (prerequisite baseline)\n")
    idx_g = np.where(has_genre)[0]
    genre_sub = [genre[i] for i in idx_g]
    emo_sub = [majority_emotion[i] for i in idx_g]
    m = external_metrics(genre_sub, emo_sub)
    report.append(f"n = {len(idx_g)} paintings with genre labels.\n")
    report.append(f"AMI(genre, emotion) = {m['AMI']:.4f}, V-measure = {m['V_measure']:.4f}, "
                   f"ARI = {m['ARI']:.4f}\n")
    contingency = np.array(
        [[sum(1 for gg, ee in zip(genre_sub, emo_sub) if gg == g and ee == e)
          for e in sorted(set(emo_sub))] for g in sorted(set(genre_sub))]
    )
    chi2, p, dof, _ = chi2_contingency(contingency)
    report.append(f"Chi-square(genre, emotion): chi2={chi2:.2f}, dof={dof}, p={p:.2e}\n")
    report.append(
        "Interpretation guide: if AMI here is already substantial, any 'buddy predicts "
        "emotion' result below must be read alongside 'buddy predicts genre' before "
        "concluding buddy structure carries affect specifically.\n"
    )

    # 2. Buddy community vs. genre and vs. emotion, full graph + genre subset
    report.append("\n## 2. Buddy community vs. genre / emotion / valence\n")
    m_genre = external_metrics(community_labels_sub := [community_labels[i] for i in idx_g], genre_sub)
    report.append(f"Community vs. genre (n={len(idx_g)}): AMI={m_genre['AMI']:.4f}, "
                   f"V-measure={m_genre['V_measure']:.4f}, ARI={m_genre['ARI']:.4f}\n")
    m_emo_full = external_metrics(community_labels, majority_emotion)
    report.append(f"Community vs. emotion (n={n}, full graph): AMI={m_emo_full['AMI']:.4f}, "
                   f"V-measure={m_emo_full['V_measure']:.4f}, ARI={m_emo_full['ARI']:.4f}\n")
    m_val_full = external_metrics(community_labels, valence)
    report.append(f"Community vs. coarse valence (n={n}, 3-way pos/neg/other): "
                   f"AMI={m_val_full['AMI']:.4f}, V-measure={m_val_full['V_measure']:.4f}, "
                   f"ARI={m_val_full['ARI']:.4f}\n")

    # 3. Content-stratified confound control: within-genre emotion separation
    report.append("\n## 3. Content-stratified test: does community still separate emotion WITHIN genre?\n")
    report.append(
        "(The sharpest test — controls for genre so a positive result can't just be "
        "content leaking through.)\n\n"
    )
    genre_groups = defaultdict(list)
    for i in idx_g:
        genre_groups[genre[i]].append(i)
    report.append("| genre | n | AMI(community, emotion) within genre |\n|---|---:|---:|\n")
    for g, idxs in sorted(genre_groups.items(), key=lambda kv: -len(kv[1])):
        if len(idxs) < 15 or len(set(majority_emotion[i] for i in idxs)) < 2:
            report.append(f"| {g} | {len(idxs)} | skipped (too few samples / labels) |\n")
            continue
        sub_comm = [community_labels[i] for i in idxs]
        sub_emo = [majority_emotion[i] for i in idxs]
        ami = adjusted_mutual_info_score(sub_comm, sub_emo)
        report.append(f"| {g} | {len(idxs)} | {ami:.4f} |\n")

    # 4. Rare-class (anger) behavior
    report.append("\n## 4. Rare-class (anger) behavior under buddy smoothing\n")
    anger_idx = [i for i in range(n) if majority_emotion[i] == "anger"]
    report.append(f"n(anger, full graph majority-vote) = {len(anger_idx)} / {n}\n")
    if anger_idx:
        anger_communities = Counter(community_labels[i] for i in anger_idx)
        top_comm, top_count = anger_communities.most_common(1)[0]
        comm_sizes = Counter(community_labels)
        expected_frac = comm_sizes[top_comm] / n
        report.append(
            f"Anger paintings' most common community: id={top_comm}, holds "
            f"{top_count}/{len(anger_idx)} of all anger paintings "
            f"({top_count/len(anger_idx)*100:.1f}%), while that community holds "
            f"{expected_frac*100:.1f}% of ALL paintings — "
            f"{'concentrated (above base rate)' if top_count/len(anger_idx) > expected_frac * 1.5 else 'roughly diffuse'}.\n"
        )
        report.append(
            "Interpretation: concentration well above the community's overall size share "
            "means buddy structure is NOT just diluting the rare class into arbitrary "
            "neighbors — worth a closer look either way.\n"
        )
    else:
        report.append("No anger-majority paintings found in this node set.\n")

    # 5. Cross-lingual emotion consistency
    report.append("\n## 5. Cross-lingual emotion consistency (English buddy graph vs. Arabic/Chinese labels)\n")
    for lang_name, lang_list in [("Arabic", arabic_emotion), ("Chinese", chinese_emotion)]:
        idx_l = [i for i, v in enumerate(lang_list) if v is not None]
        if len(idx_l) < 30:
            report.append(f"{lang_name}: only {len(idx_l)} labeled paintings, skipped.\n")
            continue
        comm_l = [community_labels[i] for i in idx_l]
        emo_l = [lang_list[i] for i in idx_l]
        m_l = external_metrics(comm_l, emo_l)
        # Also: does the English-graph community predict the OTHER language's emotion
        # AS WELL AS it predicts English emotion, on the same painting subset?
        emo_en_same_subset = [majority_emotion[i] for i in idx_l]
        m_en_same_subset = external_metrics(comm_l, emo_en_same_subset)
        report.append(
            f"{lang_name} (n={len(idx_l)}): community vs. {lang_name} emotion "
            f"AMI={m_l['AMI']:.4f}, V-measure={m_l['V_measure']:.4f}  |  "
            f"community vs. English emotion on SAME subset: AMI={m_en_same_subset['AMI']:.4f}\n"
        )
    report.append(
        "\nInterpretation: if AMI(community, other-language emotion) is comparable to "
        "AMI(community, English emotion) on the same paintings, that's evidence buddy "
        "structure captures something language-independent about the image, not an "
        "English-caption-vocabulary artifact.\n"
    )

    report_path = os.path.join(OUT_DIR, "findings_report.md")
    with open(report_path, "w") as f:
        f.writelines(report)
    log(f"Wrote {report_path}")
    log("Pipeline complete.")


if __name__ == "__main__":
    main()
