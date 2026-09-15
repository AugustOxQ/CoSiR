"""Fix (Exp. 17.1 final-review wave): raw-CLIP supervised-probe selectivity
baseline for the `warmth` / `register` axes on redcaps_150k image features.

Experiment 17.2's success criterion compares a *trained* condition vector's
probe selectivity against "17.1's raw-CLIP baseline selectivity" — this
number was never computed by the original audit (which only ran the
text-anchor direction check on raw CLIP, and the selectivity probe on
*trained* condition vectors). This script closes that gap: it reuses
`raw_clip_audit.load_features` (no reimplementation) to get the same raw,
L2-normalized 512-D CLIP image features already used by Phase A, and reuses
`checkpoint_probe.probe_selectivity` (no reimplementation) — the exact same
Hewitt & Liang selectivity probe already used for trained checkpoints in
Phase B — fed raw CLIP features instead of a trained 16-D condition vector.
"""
import json
import os
import sys

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
        result = cp.probe_selectivity(raw_img_features[keep], labels)
        results[axis_name] = result

    out_path = os.path.join(os.path.dirname(__file__), "raw_clip_probe_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
