"""Independent check of the code review's two critical claims (no training).

The review probe (``src/test/20260929_factor_collapse_probe/probe.py``) measures
variance share / participation ratio of the cached seed-42 codes and compares
PCA baselines against the *Task 6 report's* reconstruction numbers. This script
tests the two inferential steps that probe leaves open:

1. Information, not just variance: fit the best linear (ridge) readout from the
   cached codes to CLIP features on train rows and score held rows, using the
   top-k code principal components (k = 1..32). If k=1 already matches k=32,
   the codes carry ~one dimension of *information*, not just of variance.
   Context: participation ratio of centered CLIP features themselves.
2. Leakage by two independent keys: exact image-vector hash and the annotation
   ``painting`` field; plus exact text-vector reuse.

Run from the repository root:

    /root/miniconda3/envs/CoSiR/bin/python src/test/20260929_factor_collapse_verification/verify.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASK9 = ROOT / "src/test/20261007_naive_rule_mechanism_analysis"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK9))

from run_mechanism import EXPECTED_SAMPLES, load_real_features, split_items  # noqa: E402
from q3_emotion import _load_labels  # noqa: E402

FIT_ROWS = 100_000  # train rows used to fit readouts/PCA (seeded subsample)
RIDGE = 1e-3


def relative_l2(target: np.ndarray, reconstruction: np.ndarray) -> float:
    """Same per-row metric as the review probe and the Task 6 report."""
    return float((np.linalg.norm(target - reconstruction, axis=1)
                  / np.linalg.norm(target, axis=1)).mean())


def spectrum(values: np.ndarray) -> tuple[float, float]:
    """PC1 variance share and participation ratio of centered rows."""
    singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    share = singular**2 / (singular**2).sum()
    return float(share[0]), float(1 / (share**2).sum())


def ridge_readout(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Affine least-squares map x -> y with a small ridge term."""
    x_mean, y_mean = x_fit.mean(axis=0), y_fit.mean(axis=0)
    xc = x_fit - x_mean
    gram = xc.T @ xc + RIDGE * len(xc) * np.eye(xc.shape[1])
    weights = np.linalg.solve(gram, xc.T @ (y_fit - y_mean))
    return (x_eval - x_mean) @ weights + y_mean


def readout_curve(codes: np.ndarray, features: np.ndarray, fit: np.ndarray,
                  held: np.ndarray, ks=(1, 2, 3, 5, 10, 32)) -> dict:
    """Held relative L2 of CLIP reconstructed from the top-k code PCs."""
    code_mean = codes[fit].mean(axis=0)
    _, _, components = np.linalg.svd(codes[fit] - code_mean, full_matrices=False)
    out = {}
    for k in ks:
        basis = components[:k].T
        z_fit, z_held = (codes[fit] - code_mean) @ basis, (codes[held] - code_mean) @ basis
        out[k] = relative_l2(features[held], ridge_readout(z_fit, features[fit], z_held))
    return out


def pca_curve(features: np.ndarray, fit: np.ndarray, held: np.ndarray,
              ks=(1, 2, 3, 5, 10, 32)) -> dict:
    mean = features[fit].mean(axis=0)
    _, _, components = np.linalg.svd(features[fit] - mean, full_matrices=False)
    out = {"mean_only": relative_l2(features[held], np.broadcast_to(mean, features[held].shape))}
    for k in ks:
        basis = components[:k]
        out[k] = relative_l2(features[held], (features[held] - mean) @ basis.T @ basis + mean)
    return out


def row_hashes(values: np.ndarray) -> np.ndarray:
    return np.array([hashlib.md5(row.tobytes()).hexdigest() for row in values])


def main() -> None:
    img_codes = np.load(TASK9 / "cache/factor42_img.npy")
    txt_codes = np.load(TASK9 / "cache/factor42_txt.npy")
    img_features, txt_features = load_real_features()
    train, held = split_items(EXPECTED_SAMPLES)
    fit = np.sort(np.random.default_rng(42).choice(train, FIT_ROWS, replace=False))
    result = {}

    for name, codes, features in (("img", img_codes, img_features), ("txt", txt_codes, txt_features)):
        clip_pc1, clip_pr = spectrum(features[held])
        code_pc1, code_pr = spectrum(codes[held])
        result[name] = {
            "clip_centered_pc1_share": clip_pc1, "clip_centered_participation_ratio": clip_pr,
            "code_pc1_share": code_pc1, "code_participation_ratio": code_pr,
            "clip_pca_rel_l2": pca_curve(features, fit, held),
            "code_pc_readout_rel_l2": readout_curve(codes, features, fit, held),
        }
        print(name, json.dumps(result[name], indent=1), flush=True)

    emotions, paintings = _load_labels(EXPECTED_SAMPLES)
    del emotions
    img_hash, txt_hash = row_hashes(img_features), row_hashes(txt_features)
    train_paint, train_img, train_txt = set(paintings[train]), set(img_hash[train]), set(txt_hash[train])
    result["leakage"] = {
        "unique_paintings": int(len(set(paintings))),
        "unique_image_vectors": int(len(set(img_hash))),
        "held_rows_painting_in_train": float(np.isin(paintings[held], list(train_paint)).mean()),
        "held_rows_image_vector_in_train": float(np.mean([h in train_img for h in img_hash[held]])),
        "held_rows_text_vector_in_train": float(np.mean([h in train_txt for h in txt_hash[held]])),
        "held_unique_paintings_seen_in_train": float(np.mean(
            [p in train_paint for p in set(paintings[held])])),
    }
    print("leakage", json.dumps(result["leakage"], indent=1), flush=True)
    print("RESULT_JSON=" + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
