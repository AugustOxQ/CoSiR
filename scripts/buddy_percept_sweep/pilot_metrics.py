"""Pilot-method held-out metrics for the confirmation checks (QC1/QC2).

The sweep harness scores held-out paintings by a k-NN transfer to merged
train topics; the investigation's standing bar used *independent* held-out
re-clustering (mutual-kNN graph + modularity Leiden). This module exposes
the pilots' own graph + Leiden code so both yardsticks can be applied to the
same embedding. Loaded via the same sibling-module-import pattern as
real_data.py (never edits the originals).
"""
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

_BUDDY_DIR = Path(__file__).resolve().parents[2] / "src/test/20260923_artelingo_buddy_analysis"


@dataclass
class PilotModules:
    pipeline: object            # train-split run_pipeline.py module
    heldout_pipeline: object    # held-out copy with STORAGE_DIR/TRAIN_JSON routed
    affect_pilot: object
    single_modality: object


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pilot_modules() -> PilotModules:
    arch = _load_module("arch_for_metrics", _BUDDY_DIR / "run_learned_student_arch_sweep_pilot.py")
    pipeline = arch.load_sibling_module("pipeline_for_metrics", arch.PIPELINE_PATH)
    affect_pilot = arch.load_sibling_module("affect_for_metrics", arch.AFFECT_PILOT_PATH)
    single_modality = arch.load_sibling_module("single_modality_for_metrics", arch.SINGLE_MODALITY_PATH)
    heldout_pipeline = arch.load_sibling_module("heldout_pipeline_for_metrics", arch.PIPELINE_PATH)
    # Same env-var routing of the hardcoded held-out paths as real_data.py.
    feature_root = os.environ.get("PERCEPT_FEATURE_ROOT", "/data/SSD2/pre_extract")
    json_root = os.environ.get("PERCEPT_RAW_JSON_ROOT", "/data/PDD/artelingo")
    heldout_pipeline.STORAGE_DIR = f"{feature_root}/artelingo_heldout/features"
    heldout_pipeline.TRAIN_JSON = f"{json_root}/artelingo_val_test.json"
    return PilotModules(pipeline=pipeline, heldout_pipeline=heldout_pipeline,
                        affect_pilot=affect_pilot, single_modality=single_modality)


def independent_partition(embedding: np.ndarray, modules: PilotModules,
                          split: str, seed: int, device: str) -> np.ndarray:
    """The pilots' exact independent re-clustering of `embedding`: repaired
    mutual-kNN graph, then modularity Leiden. `split` picks which pipeline
    module the graph builder uses ("train" or "heldout")."""
    from src.conditional_buddy.prototype_seed import detect_communities

    if split == "train":
        pipeline = modules.pipeline
    elif split == "heldout":
        pipeline = modules.heldout_pipeline
    else:
        raise ValueError(f"split must be 'train' or 'heldout', got {split!r}")
    graph = modules.single_modality.build_single_modality_graph(
        f"{split}-independent", embedding, pipeline, modules.affect_pilot,
        device, expected_nodes=len(embedding),
    )
    return np.asarray(detect_communities(graph, seed=seed), dtype=np.int64)


def ami_emotion_genre(labels, emotion, genre) -> tuple[float, float]:
    """Emotion AMI over all rows; genre AMI over rows with a genre."""
    labels = np.asarray(labels)
    emotion = np.asarray(emotion)
    genre = np.asarray(genre)
    emotion_ami = float(adjusted_mutual_info_score(emotion, labels))
    covered = genre != ""
    if covered.any() and len(set(labels[covered].tolist())) > 1:
        genre_ami = float(adjusted_mutual_info_score(genre[covered], labels[covered]))
    else:
        genre_ami = 0.0
    return emotion_ami, genre_ami
