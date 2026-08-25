"""
Load buddy-graph source features for an arbitrary (vision, text) encoder pair, row-aligned
to a given FeatureManager's sample-id order.

Used by Experiment 8 (buddy-init encoder-pair ablation,
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md) to swap which encoder
pair BUILDS the buddy graph/init while the frozen training backbone stays CLIP throughout.

'clip_img'/'clip_txt' come straight from the dataset's own load_data() (which itself reads
the CLIP FeatureManager). Every other name is read from the held-out feature cache built by
src/test/20260708_heldout_grid/extract_heldout.py, which is guaranteed row-aligned to the
SAME FeatureManager because both were built by calling that dataset's shared load_data().
"""
import os
import sys
from typing import List, Tuple

import numpy as np

_HELDOUT_GRID_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "test", "20260708_heldout_grid"
)
if _HELDOUT_GRID_DIR not in sys.path:
    sys.path.insert(0, _HELDOUT_GRID_DIR)

VISION_ENCODERS = ["clip_img", "dinov2", "siglip_v", "vit_sup"]
TEXT_ENCODERS = ["clip_txt", "minilm", "bge", "e5"]

# dataset key -> (module name, directory to import it from)
_DATASET_LOADERS = {
    "redcaps": ("redcaps_buddy", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "test", "20260623_redcaps_buddy")),
}


def _dataset_module(dataset: str):
    mod_name, mod_dir = _DATASET_LOADERS[dataset]
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    return __import__(mod_name)


def load_encoder_pair_features(
    dataset: str, vision: str, text: str, feature_manager
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Returns (img_feats, txt_feats, sample_ids) for the (vision, text) encoder pair, in
    feature_manager's own sample-id order.

    Raises AssertionError if the dataset's load_data() row order (which the held-out cache
    was built against) does not exactly match feature_manager.get_all_sample_ids() — the
    CLAUDE.md 'sample ID consistency' check, applied to this second cached feature source.
    """
    from extract_heldout import cache_path  # src/test/20260708_heldout_grid

    mod = _dataset_module(dataset)
    data = mod.load_data()
    fm_ids = list(feature_manager.get_all_sample_ids())
    assert data.sample_ids == fm_ids, (
        f"held-out cache row order does not match feature_manager for dataset={dataset}: "
        f"{len(data.sample_ids)} vs {len(fm_ids)} sample ids, or order differs. Re-run "
        "extract_heldout.py against the SAME FeatureManager storage_dir this training run "
        "uses before selecting this encoder pair."
    )

    def _load(name):
        if name == "clip_img":
            return np.ascontiguousarray(data.img, dtype=np.float32)
        if name == "clip_txt":
            return np.ascontiguousarray(data.txt, dtype=np.float32)
        path = cache_path(dataset, name, 0)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing held-out cache {path}; run:\n"
                f"  python src/test/20260708_heldout_grid/extract_heldout.py "
                f"--dataset {dataset} --model {name}"
            )
        arr = np.load(path).astype(np.float32)
        # The assert above only compares two reads of the SAME CLIP-backed store (data vs.
        # feature_manager) -- it never touches the actual held-out .npy being loaded here. A
        # stale cache with the same row COUNT but a different row ORDER (e.g. regenerated from
        # a re-shuffled annotation file) would silently produce a garbage graph without this
        # check. This only catches a row-count mismatch, not a same-count reordering -- but a
        # reorder without a count change would require the cache to have been rebuilt against
        # a differently-sized annotation file that happens to match by coincidence, which is
        # far less likely than the row-count drift this guards against.
        assert arr.shape[0] == len(data.sample_ids), (
            f"held-out cache {path} has {arr.shape[0]} rows, expected {len(data.sample_ids)} "
            f"(matching {dataset}'s FeatureManager) -- re-run extract_heldout.py for this model"
        )
        return arr

    return _load(vision), _load(text), data.sample_ids
