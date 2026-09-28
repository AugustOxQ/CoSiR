"""Real ArtELingo/GoEmotions/patch-feature loader -- the `raw_loader`
passed into `FixedInputCache.get` outside of tests. Reuses the exact
loading functions this investigation already validated, via the
established sibling-module-import pattern (never edits the originals).
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

_BUDDY_DIR = Path(__file__).resolve().parents[2] / "src/test/20260923_artelingo_buddy_analysis"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_real_raw_inputs():
    from scripts.buddy_percept_sweep.cache import RawInputs

    arch = _load_module("arch_for_sweep", _BUDDY_DIR / "run_learned_student_arch_sweep_pilot.py")
    pipeline = arch.load_sibling_module("pipeline_for_sweep", arch.PIPELINE_PATH)
    affect_pilot = arch.load_sibling_module("affect_for_sweep", arch.AFFECT_PILOT_PATH)
    single_modality = arch.load_sibling_module("single_modality_for_sweep", arch.SINGLE_MODALITY_PATH)
    cca_audit = arch.load_sibling_module("cca_for_sweep", arch.CCA_AUDIT_PATH)
    heldout_pipeline = arch.load_sibling_module("heldout_pipeline_for_sweep", arch.PIPELINE_PATH)
    heldout_pipeline.STORAGE_DIR = arch.HELDOUT_STORAGE_DIR
    heldout_pipeline.TRAIN_JSON = arch.HELDOUT_JSON

    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paintings, img_nodes, txt_nodes, emotion_counts = pipeline.load_dedup_features()
    train_emotion = [pipeline.majority(counts) for counts in emotion_counts]
    heldout_paintings, heldout_img, heldout_txt, heldout_emotion_counts = heldout_pipeline.load_dedup_features()
    heldout_emotion = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]

    train_affect = np.asarray(affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, device), dtype=np.float32)
    heldout_affect = np.asarray(affect_pilot.extract_affect_nodes(arch.HELDOUT_JSON, heldout_paintings, device), dtype=np.float32)

    train_content_raw = cca_audit.content_features(img_nodes, txt_nodes, affect_pilot)
    heldout_content_raw = cca_audit.content_features(heldout_img, heldout_txt, affect_pilot)

    genre_map = pipeline.load_genre_map()
    train_genre = np.array([genre_map.get(p, "") for p in paintings], dtype=object)
    heldout_genre = np.array([genre_map.get(p, "") for p in heldout_paintings], dtype=object)

    stage2 = arch.load_sibling_module("stage2_for_sweep_patches", _BUDDY_DIR / "run_buddy_stage2_pilot.py")
    train_patches = stage2.load_patch_features(stage2.TRAIN_PATCH_FEATURE_PATH, len(paintings), "train")
    heldout_patches = stage2.load_patch_features(stage2.HELDOUT_PATCH_FEATURE_PATH, len(heldout_paintings), "held-out")

    return RawInputs(
        train_content_raw=train_content_raw.astype(np.float32),
        heldout_content_raw=heldout_content_raw.astype(np.float32),
        train_affect=train_affect, heldout_affect=heldout_affect,
        train_emotion=train_emotion, heldout_emotion=heldout_emotion,
        train_genre=train_genre, heldout_genre=heldout_genre,
        train_patches=train_patches, heldout_patches=heldout_patches,
    )
