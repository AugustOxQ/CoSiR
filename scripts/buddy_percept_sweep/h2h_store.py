"""On-disk store of every hyperparameter-independent input both systems
need (dedup CLIP nodes, content features, 28-d affect, PercepT fused input,
labels), built once under a file lock so sweep agents on a node load it
instead of re-extracting GoEmotions features per trial. Patch features are
large and already on disk as .pt files, so they are loaded, never cached.
"""
import fcntl
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

_ARRAY_KEYS = (
    "train_paintings", "heldout_paintings",
    "train_img", "train_txt", "heldout_img", "heldout_txt",
    "train_content_raw", "heldout_content_raw",
    "train_affect28", "heldout_affect28",
    "train_percept_h", "heldout_percept_h",
    "train_emotion", "heldout_emotion", "train_genre", "heldout_genre",
)
_OBJECT_KEYS = ("train_paintings", "heldout_paintings", "train_emotion",
                "heldout_emotion", "train_genre", "heldout_genre")
_EXPECTED_TRAIN, _EXPECTED_HELDOUT = 61_402, 9_365

_BUILD_LOCK = threading.Lock()
_PERCEPT_DIR = Path(__file__).resolve().parents[2] / "src/test/20260922_percept_topic_pipeline"


@dataclass
class H2HStore:
    train_paintings: np.ndarray; heldout_paintings: np.ndarray       # object
    train_img: np.ndarray; train_txt: np.ndarray                     # dedup CLIP nodes, dtype as returned by load_dedup_features
    heldout_img: np.ndarray; heldout_txt: np.ndarray
    train_content_raw: np.ndarray; heldout_content_raw: np.ndarray   # cca_audit.content_features, dtype as returned
    train_affect28: np.ndarray; heldout_affect28: np.ndarray         # float64 (N, 28), the pilot's asarray dtype
    train_percept_h: np.ndarray; heldout_percept_h: np.ndarray       # PercepT fused inputs, dtype as returned by fused_embeddings
    train_emotion: np.ndarray; heldout_emotion: np.ndarray           # object, majority emotion
    train_genre: np.ndarray; heldout_genre: np.ndarray               # object, "" when missing
    train_patches: torch.Tensor; heldout_patches: torch.Tensor       # loaded from the .pt files, never cached


def default_cache_dir() -> Path:
    override = os.environ.get("H2H_CACHE_DIR")
    if override:
        return Path(override)
    root = os.environ.get("PERCEPT_FEATURE_ROOT", "/data/SSD2/pre_extract")
    return Path(root) / "artelingo_h2h_cache"


def _keep(x) -> np.ndarray:
    """No dtype cast: the pilot's Stage 1 must see exactly what the pilot
    functions returned (bit-for-bit reproduction of its snapshot)."""
    return np.asarray(x)


def _obj(x) -> np.ndarray:
    out = np.empty(len(x), dtype=object)
    out[:] = list(x)
    return out


_SEED = 42


def build_arrays(pilot, percept_base=None) -> dict:
    """The expensive part (real data). Mirrors real_data.load_real_raw_inputs
    and the PercepT Stage-1 input flow of fit_stage1_and_get_targets.

    Runs under the snapshot pilot's determinism settings so the stored inputs
    are bit-identical to what the pilot computes in-process; previous torch
    flags are restored afterwards. `percept_base` (default: load the PercepT
    stage-1 pilot) is injectable for tests."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prev = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        return _build_arrays(pilot, percept_base)
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        torch.backends.cudnn.deterministic = prev[2]
        torch.backends.cudnn.benchmark = prev[3]


def _build_arrays(pilot, percept_base) -> dict:
    import torch as _torch

    arch, pipeline, heldout_pipeline = pilot.arch, pilot.pipeline, pilot.heldout_pipeline
    affect_pilot, cca_audit = pilot.affect_pilot, pilot.cca_audit
    pipeline.assert_extraction_complete()
    heldout_pipeline.assert_extraction_complete()
    device = "cuda" if _torch.cuda.is_available() else "cpu"

    paintings, img, txt, counts = pipeline.load_dedup_features()
    h_paintings, h_img, h_txt, h_counts = heldout_pipeline.load_dedup_features()
    if (len(paintings), len(h_paintings)) != (_EXPECTED_TRAIN, _EXPECTED_HELDOUT) and percept_base is None:
        raise RuntimeError(f"unexpected painting counts {len(paintings)}/{len(h_paintings)}; "
                           f"expected {_EXPECTED_TRAIN}/{_EXPECTED_HELDOUT}")
    train_emotion = _obj([pipeline.majority(c) for c in counts])
    heldout_emotion = _obj([heldout_pipeline.majority(c) for c in h_counts])

    affect28 = np.asarray(affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings, device), dtype=np.float64)
    h_affect28 = np.asarray(affect_pilot.extract_affect_nodes(arch.HELDOUT_JSON, h_paintings, device), dtype=np.float64)
    content = _keep(cca_audit.content_features(img, txt, affect_pilot))
    h_content = _keep(cca_audit.content_features(h_img, h_txt, affect_pilot))

    genre_map = pipeline.load_genre_map()
    train_genre = _obj([genre_map.get(p, "") for p in paintings])
    heldout_genre = _obj([genre_map.get(p, "") for p in h_paintings])

    base = percept_base or arch.load_sibling_module(
        "percept_stage1_base_for_h2h", str(_PERCEPT_DIR / "run_percept_stage1_pilot.py"))
    log = pipeline.log
    aff768 = base.extract_affect_embedding_nodes(pipeline.TRAIN_JSON, paintings, device, log)
    h_aff768 = base.extract_affect_embedding_nodes(base.HELDOUT_JSON, h_paintings, device, log)
    percept_h = _keep(base.fused_embeddings(img, txt, aff768, cca_audit, affect_pilot))
    h_percept_h = _keep(base.fused_embeddings(h_img, h_txt, h_aff768, cca_audit, affect_pilot))

    return dict(
        train_paintings=_obj(paintings), heldout_paintings=_obj(h_paintings),
        train_img=_keep(img), train_txt=_keep(txt), heldout_img=_keep(h_img), heldout_txt=_keep(h_txt),
        train_content_raw=content, heldout_content_raw=h_content,
        train_affect28=affect28, heldout_affect28=h_affect28,
        train_percept_h=percept_h, heldout_percept_h=h_percept_h,
        train_emotion=train_emotion, heldout_emotion=heldout_emotion,
        train_genre=train_genre, heldout_genre=heldout_genre,
    )


def load_patches(n_train: int, n_heldout: int) -> tuple:
    from scripts.buddy_percept_sweep.pilot_metrics import _load_module

    stage2 = _load_module("stage2_for_h2h_patches", _PERCEPT_DIR / "run_percept_stage2_pilot.py")
    train = stage2.load_patch_features(stage2.TRAIN_PATCH_FEATURE_PATH, n_train, "train")
    heldout = stage2.load_patch_features(stage2.HELDOUT_PATCH_FEATURE_PATH, n_heldout, "held-out")
    return train, heldout


def _default_builder() -> dict:
    from scripts.buddy_percept_sweep.pilot_metrics import load_pilot_modules
    return build_arrays(load_pilot_modules())


def load_or_build_store(cache_dir: Optional[Path] = None,
                        builder: Optional[Callable[[], dict]] = None,
                        patch_loader: Optional[Callable[[int, int], tuple]] = None) -> H2HStore:
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / "h2h_store.npz"
    tmp = cache_dir / "h2h_store.npz.tmp"
    builder = builder or _default_builder
    patch_loader = patch_loader or load_patches

    with _BUILD_LOCK:
        with open(cache_dir / "h2h_store.lock", "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if not final.exists():
                    arrays = builder()
                    with open(tmp, "wb") as fh:   # a file object stops savez appending ".npz"
                        np.savez(fh, **{k: arrays[k] for k in _ARRAY_KEYS})
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, final)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    with np.load(final, allow_pickle=True) as data:
        arrays = {k: data[k] for k in _ARRAY_KEYS}
    for k in _OBJECT_KEYS:
        arrays[k] = arrays[k].astype(object)
    train_patches, heldout_patches = patch_loader(len(arrays["train_paintings"]),
                                                  len(arrays["heldout_paintings"]))
    return H2HStore(**arrays, train_patches=train_patches, heldout_patches=heldout_patches)
