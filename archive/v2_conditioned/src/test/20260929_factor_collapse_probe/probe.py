"""Code-review probe: factor-space collapse and train/held image leakage.

Reads the cached seed-42 factor codes written by the Task 9 mechanism run
(``src/test/20261007_naive_rule_mechanism_analysis/cache/``) and the real
ArtELingo features. No training. Run from the repository root:

    /root/miniconda3/envs/CoSiR/bin/python src/test/20260929_factor_collapse_probe/probe.py
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASK9 = ROOT / "src/test/20261007_naive_rule_mechanism_analysis"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK9))

from run_mechanism import EXPECTED_SAMPLES, load_real_features, split_items  # noqa: E402


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def relative_l2(target: np.ndarray, reconstruction: np.ndarray) -> float:
    return float((np.linalg.norm(target - reconstruction, axis=1)
                  / np.linalg.norm(target, axis=1)).mean())


def reconstruction_baselines(features: np.ndarray, train: np.ndarray, held: np.ndarray) -> dict:
    """Held-row relative L2 of mean-only and rank-k PCA reconstructions (fit on train)."""
    mean = features[train].mean(axis=0)
    _, _, components = np.linalg.svd(features[train][::5] - mean, full_matrices=False)
    centered = features[held] - mean
    out = {"mean_only": relative_l2(features[held], np.broadcast_to(mean, centered.shape))}
    for k in (1, 4, 32):
        basis = components[:k]
        out[f"pca_{k}"] = relative_l2(features[held], centered @ basis.T @ basis + mean)
    return out


def code_geometry(codes: np.ndarray) -> dict:
    """Sparsity and effective dimensionality of one modality's factor codes."""
    singular = np.linalg.svd(codes - codes.mean(axis=0), compute_uv=False)
    share = singular**2 / (singular**2).sum()
    return {
        "zero_fraction": float((codes == 0).mean()),
        "active_per_row": float((codes > 0).sum(axis=1).mean()),
        "pc_share_top1_top3_top5": [float(share[:k].sum()) for k in (1, 3, 5)],
        "participation_ratio": float(1 / (share**2).sum()),
    }


def main() -> None:
    img_codes = np.load(TASK9 / "cache/factor42_img.npy")
    txt_codes = np.load(TASK9 / "cache/factor42_txt.npy")
    img_features, txt_features = load_real_features()
    train, held = split_items(EXPECTED_SAMPLES)

    print("== Reconstruction baselines (held rows; factor model reported 0.5606 img / 0.5154 txt)")
    for name, features in (("img", img_features), ("txt", txt_features)):
        print(name, {k: round(v, 4) for k, v in reconstruction_baselines(features, train, held).items()})

    print("== Factor-code geometry (held rows)")
    for name, codes in (("img", img_codes), ("txt", txt_codes)):
        print(name, code_geometry(codes[held]))

    pair = 0.5 * (img_codes + txt_codes)
    corr = np.abs(np.corrcoef(pair[train].T))[np.triu_indices(pair.shape[1], 1)]
    print(f"pair-code factor pairs with |r|>=.9: {(corr >= .9).sum()}/{len(corr)}; mean |r| {corr.mean():.3f}")

    shuffled = np.random.default_rng(0).permutation(len(img_codes))
    img_unit, txt_unit = unit_rows(img_codes), unit_rows(txt_codes)
    print(f"img-txt code cosine: matched {(img_unit * txt_unit).sum(1).mean():.3f}, "
          f"shuffled {(img_unit * txt_unit[shuffled]).sum(1).mean():.3f}")

    print("== Split leakage (split_items is by annotation row)")
    hashes = np.array([hashlib.md5(row.tobytes()).hexdigest() for row in img_features])
    train_hashes = set(hashes[train])
    print(f"unique image vectors: {len(set(hashes)):,} of {len(hashes):,}")
    print(f"held rows whose exact image vector is in train: "
          f"{np.mean([h in train_hashes for h in hashes[held]]):.4f}")


if __name__ == "__main__":
    main()
