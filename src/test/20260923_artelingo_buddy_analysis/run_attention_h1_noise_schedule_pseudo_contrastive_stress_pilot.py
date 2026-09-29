"""Stress-test the combined noise-schedule + pseudo-contrastive pilot's
seed-42 result at three more seeds, even though seed 42 narrowly missed the
held-out Pareto bar (by 0.0026 on emotion AMI), because it had the best
genre AMI/silhouette profile of any buddy variant tried this session.

Reuses run_attention_h1_noise_schedule_pseudo_contrastive_pilot.py's own
context-building and run_seed function unchanged; only forces stress seeds
to run regardless of the seed-42 Pareto-bar outcome.
"""

import importlib.util
import os
import time

import numpy as np
import torch
from sklearn.decomposition import PCA


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_PATH = os.path.join(
    OUT_DIR, "run_attention_h1_noise_schedule_pseudo_contrastive_pilot.py"
)
REPORT_PATH = os.path.join(
    OUT_DIR, "attention_h1_noise_schedule_pseudo_contrastive_stress_pilot_report.md"
)
SEEDS = (7, 123, 2024)
# From attention_h1_noise_schedule_pseudo_contrastive_pilot_report.md's own
# seed-42 "This combination" row.
SEED42_KNOWN = {"seed": 42, "emotion": 0.1210, "genre": 0.2623, "silhouette": 0.0789}
EMOTION_BAR = 0.1236
GENRE_BAR = 0.1954


def load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


combined = load_module("attention_h1_combined_for_stress", COMBINED_PATH)


def clears(row):
    return row["emotion"] > EMOTION_BAR and row["genre"] > GENRE_BAR


def build_context(SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    arch = combined.load_module("attention_h1_stress_arch_sweep", combined.ARCH_SWEEP_PATH)
    pipeline = arch.load_sibling_module("artelingo_run_pipeline_attn_stress", arch.PIPELINE_PATH)
    affect_pilot = arch.load_sibling_module("artelingo_run_affect_pilot_attn_stress", arch.AFFECT_PILOT_PATH)
    single_modality = arch.load_sibling_module(
        "artelingo_run_single_modality_attn_stress", arch.SINGLE_MODALITY_PATH
    )
    cca_audit = arch.load_sibling_module("artelingo_run_cca_audit_attn_stress", arch.CCA_AUDIT_PATH)
    arch.cca_audit = cca_audit
    heldout_pipeline = arch.load_sibling_module(
        "artelingo_run_pipeline_attn_stress_heldout", arch.PIPELINE_PATH
    )
    heldout_pipeline.STORAGE_DIR = arch.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = arch.HELDOUT_JSON
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = arch.log
    log(f"Using {device} for the combined-pilot stress re-run.")

    log("Verifying and loading train CLIP features...")
    pipeline.assert_extraction_complete()
    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    majority_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    affect_train = np.asarray(
        affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, str(device)),
        dtype=np.float64,
    )
    heldout_pipeline.assert_extraction_complete()
    heldout_paintings, heldout_img, heldout_txt, heldout_emotion_counts = (
        heldout_pipeline.load_dedup_features()
    )
    heldout_majority_emotion = [
        heldout_pipeline.majority(counts) for counts in heldout_emotion_counts
    ]
    affect_heldout = np.asarray(
        affect_pilot.extract_affect_nodes(arch.HELDOUT_JSON, heldout_paintings, str(device)),
        dtype=np.float64,
    )

    content_train = cca_audit.content_features(img_nodes, txt_nodes, affect_pilot)
    content_heldout = cca_audit.content_features(heldout_img, heldout_txt, affect_pilot)
    pca = PCA(n_components=arch.CONTENT_PCA_DIM, random_state=SEED)
    content_train = pca.fit_transform(content_train).astype(np.float32)
    content_heldout = pca.transform(content_heldout).astype(np.float32)
    _img_graph, _txt_graph, content_teacher_graph = pipeline.build_buddy_graphs(
        img_nodes, txt_nodes, K=pipeline.K, alpha=pipeline.ALPHA, device=str(device),
        connect_components=True,
    )
    affect_teacher_graph = single_modality.build_single_modality_graph(
        "train-affect-teacher", affect_train, pipeline, affect_pilot, str(device),
        expected_nodes=len(paintings),
    )
    content_edges = arch.upper_triangle_edges(content_teacher_graph)
    affect_edges = arch.upper_triangle_edges(affect_teacher_graph)
    heldout_content_graph = single_modality.build_single_modality_graph(
        "held-out-content-reference", content_heldout, heldout_pipeline, affect_pilot,
        str(device), expected_nodes=len(heldout_paintings),
    )
    heldout_affect_graph = single_modality.build_single_modality_graph(
        "held-out-affect-reference", affect_heldout, heldout_pipeline, affect_pilot,
        str(device), expected_nodes=len(heldout_paintings),
    )
    diagnostic_rng = np.random.default_rng(SEED)
    sampled_nodes = diagnostic_rng.choice(
        len(heldout_paintings), size=arch.EDGE_SAMPLE_SIZE, replace=False
    )
    rank_nodes = diagnostic_rng.choice(
        len(heldout_paintings), size=arch.EFFECTIVE_RANK_SAMPLE_SIZE, replace=False
    )
    return {
        "arch": arch, "pipeline": pipeline, "heldout_pipeline": heldout_pipeline,
        "affect_pilot": affect_pilot, "single_modality": single_modality,
        "device": device, "paintings": paintings,
        "majority_emotion": majority_emotion,
        "heldout_paintings": heldout_paintings,
        "heldout_majority_emotion": heldout_majority_emotion,
        "genre_map": pipeline.load_genre_map(),
        "content_edges": content_edges, "affect_edges": affect_edges,
        "heldout_content_graph": heldout_content_graph,
        "heldout_affect_graph": heldout_affect_graph,
        "sampled_nodes": sampled_nodes, "rank_nodes": rank_nodes,
        "train_content_t": torch.as_tensor(content_train, dtype=torch.float32, device=device),
        "train_affect_t": torch.as_tensor(affect_train, dtype=torch.float32, device=device),
        "heldout_content_t": torch.as_tensor(content_heldout, dtype=torch.float32, device=device),
        "heldout_affect_t": torch.as_tensor(affect_heldout, dtype=torch.float32, device=device),
    }


def main():
    assert combined.NOISE_STD == 0.0, "Combined pilot's NOISE_STD changed; re-check before reusing."
    context = build_context(combined.SEED)
    rows = [SEED42_KNOWN]
    for seed in SEEDS:
        result = combined.run_seed(seed, combined.NOISE_STD, context)
        heldout = result["heldout_post"]
        rows.append({
            "seed": seed, "emotion": heldout["emotion_ami"],
            "genre": heldout["genre_ami"], "silhouette": heldout["silhouette"],
        })
        print(f"[stress seed {seed}] {rows[-1]}", flush=True)

    clear_count = sum(clears(row) for row in rows)
    lines = [
        "# Combined noise-schedule + pseudo-contrastive: four-seed stress\n\n",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n",
        "Seed 42's row is copied from "
        "`attention_h1_noise_schedule_pseudo_contrastive_pilot_report.md` "
        "(not re-run); seeds 7, 123, 2024 are freshly fit here using the same "
        "`run_seed` function, NOISE_STD, and context-building code as that "
        "pilot, unchanged.\n\n",
        "| seed | held-out emotion AMI | held-out genre AMI | held-out silhouette | Pareto bar |\n",
        "|---:|---:|---:|---:|---|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['emotion']:.4f} | {row['genre']:.4f} | "
            f"{row['silhouette']:.4f} | {'clears' if clears(row) else 'does not clear'} |\n"
        )
    emotion_vals = [r["emotion"] for r in rows]
    genre_vals = [r["genre"] for r in rows]
    silhouette_vals = [r["silhouette"] for r in rows]
    lines.extend([
        "\n## Summary\n\n",
        f"- Emotion AMI: mean={np.mean(emotion_vals):.4f}; "
        f"min={np.min(emotion_vals):.4f}; max={np.max(emotion_vals):.4f}.\n",
        f"- Genre AMI: mean={np.mean(genre_vals):.4f}; "
        f"min={np.min(genre_vals):.4f}; max={np.max(genre_vals):.4f}.\n",
        f"- Silhouette: mean={np.mean(silhouette_vals):.4f}; "
        f"min={np.min(silhouette_vals):.4f}; max={np.max(silhouette_vals):.4f}.\n",
        f"- Both held-out Pareto bars clear simultaneously in "
        f"**{clear_count}/4 seeds**.\n",
    ])
    with open(REPORT_PATH, "w") as report_file:
        report_file.writelines(lines)
    print(f"Wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
