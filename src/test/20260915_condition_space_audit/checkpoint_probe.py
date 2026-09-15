"""Phase 17.1(b): does a linear probe on already-trained buddy-condition
vectors already recover a proxy-label axis, beyond a matched random-relabeling
control (Hewitt & Liang selectivity)? No retraining — reads existing
checkpoints' final_embeddings only.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import axis_definitions as ax

Z_BAR = 2.0
SELECTIVITY_FLOOR = 0.10


def load_checkpoint(run_dir):
    emb_dir = os.path.join(run_dir, "final_embeddings")
    emb = np.load(os.path.join(emb_dir, "embeddings.npy"))
    sample_ids = np.load(os.path.join(emb_dir, "sample_ids.npy"))
    return emb, sample_ids


def probe_selectivity(X, y, seed=42, n_shuffles=20, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    real_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=skf)
    real_acc = float(real_scores.mean())

    rng = np.random.RandomState(seed)
    shuffle_accs = []
    for i in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        skf_i = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + i + 1)
        s = cross_val_score(LogisticRegression(max_iter=1000), X, y_shuffled, cv=skf_i)
        shuffle_accs.append(s.mean())
    shuffle_accs = np.array(shuffle_accs)

    z = float((real_acc - shuffle_accs.mean()) / (shuffle_accs.std(ddof=1) + 1e-8))
    selectivity = float(real_acc - shuffle_accs.mean())
    return {
        "n": int(len(y)),
        "real_acc": real_acc,
        "control_mean": float(shuffle_accs.mean()),
        "control_std": float(shuffle_accs.std(ddof=1)),
        "selectivity": selectivity,
        "z": z,
        "verdict": "positive" if z >= Z_BAR and selectivity >= SELECTIVITY_FLOOR else "null",
    }


def probe_checkpoint(run_dir, records_by_sample_id, axes, binary_labels_fn):
    emb, sample_ids = load_checkpoint(run_dir)
    records = [records_by_sample_id[int(s)] for s in sample_ids]
    result = {}
    for axis_name in axes:
        keep, labels = binary_labels_fn(records, axis_name)
        if len(np.unique(labels)) < 2 or len(labels) < 20:
            result[axis_name] = {"verdict": "skipped", "reason": "insufficient samples for this axis in this checkpoint"}
            continue
        result[axis_name] = probe_selectivity(emb[keep], labels)
    return result


def main():
    checkpoints_path = os.path.join(os.path.dirname(__file__), "checkpoints.json")
    checkpoints = json.load(open(checkpoints_path))

    redcaps_meta = json.load(open("/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"))
    impressions_meta = json.load(open("/project/Impressions/metadata/impressions_train.json"))

    results = {"redcaps_150k": {}, "impressions": {}}
    for run_dir in checkpoints["redcaps_150k"]:
        results["redcaps_150k"][run_dir] = probe_checkpoint(
            run_dir, redcaps_meta, ax.REDCAPS_AXES.keys(), ax.redcaps_binary_labels
        )
    for run_dir in checkpoints["impressions"]:
        results["impressions"][run_dir] = probe_checkpoint(
            run_dir, impressions_meta, ax.IMPRESSIONS_AXES.keys(), ax.impressions_binary_labels
        )

    out_path = os.path.join(os.path.dirname(__file__), "checkpoint_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
