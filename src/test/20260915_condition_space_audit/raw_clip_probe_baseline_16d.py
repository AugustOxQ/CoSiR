"""Fix (Exp. 17.1 final-review): dimension-matched raw-CLIP supervised-probe
selectivity baseline for the `warmth` / `register` axes on redcaps_150k image
features.

Experiment 17.2 compares trained 16-D condition-vector probe selectivity
against Exp. 17.1's raw-CLIP baseline. The original raw baseline used the
full 512-D CLIP representation, which confounds that comparison with a 32x
capacity difference. This script closes that final-review gap: it reuses
`raw_clip_audit.load_features` (no reimplementation) to obtain the same raw,
L2-normalized 512-D CLIP image features and reuses
`checkpoint_probe.probe_selectivity` (no reimplementation), after reducing
the raw features to a dimension-matched 16-D representation with PCA and a
Gaussian random projection.
"""
import json
import os
import sys

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

sys.path.insert(0, os.path.dirname(__file__))
import axis_definitions as ax
import checkpoint_probe as cp
import raw_clip_audit as rca


def main():
    redcaps_feats, redcaps_records = rca.load_features(rca.REDCAPS_STORAGE, rca.REDCAPS_ANNOT)
    raw_img_features = redcaps_feats["img"]

    results = {}
    for axis_name in ("warmth", "register"):
        keep, labels = ax.redcaps_binary_labels(redcaps_records, axis_name)
        feats = raw_img_features[keep]

        # Explicit caveat: these transforms are fit on the FULL feature set for
        # the axis, rather than inside each cross-validation fold of
        # probe_selectivity. Fold-local fitting would be more statistically
        # rigorous, but is out of scope for this one-shot diagnostic.
        feats_pca16 = PCA(n_components=16, random_state=42).fit_transform(feats)
        feats_randproj16 = GaussianRandomProjection(
            n_components=16, random_state=42
        ).fit_transform(feats)

        results[axis_name] = {
            "pca16": cp.probe_selectivity(feats_pca16, labels),
            "randproj16": cp.probe_selectivity(feats_randproj16, labels),
        }

    out_path = os.path.join(os.path.dirname(__file__), "raw_clip_probe_baseline_16d.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    raw_baseline_path = os.path.join(os.path.dirname(__file__), "raw_clip_probe_baseline.json")
    with open(raw_baseline_path) as f:
        raw_baseline = json.load(f)

    checkpoint_results_path = os.path.join(
        os.path.dirname(__file__), "checkpoint_probe_results.json"
    )
    with open(checkpoint_results_path) as f:
        checkpoint_results = json.load(f)

    print("\nSelectivity comparison")
    print(
        f"{'axis':<10} {'raw-512D':>10} {'PCA-16':>10} "
        f"{'RandomProj-16':>14} {'trained-checkpoint range':>26}"
    )
    for axis_name in ("warmth", "register"):
        trained_selectivities = [
            run_results[axis_name]["selectivity"]
            for run_results in checkpoint_results["redcaps_150k"].values()
        ]
        print(
            f"{axis_name:<10} "
            f"{raw_baseline[axis_name]['selectivity']:>10.4f} "
            f"{results[axis_name]['pca16']['selectivity']:>10.4f} "
            f"{results[axis_name]['randproj16']['selectivity']:>14.4f} "
            f"{min(trained_selectivities):>10.4f} to {max(trained_selectivities):>10.4f}"
        )


if __name__ == "__main__":
    main()
