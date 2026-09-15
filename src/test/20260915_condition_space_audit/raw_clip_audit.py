"""Phase 17.1(a): does a CLIP-text-anchor semantic direction separate RedCaps
subreddit or Impressions caption_type proxy labels, beyond a matched
random-prompt control? Training-free — reuses already-cached CLIP features.
"""
import json
import os
import sys

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import axis_definitions as ax
import text_anchor as ta
from src.utils import FeatureManager

REDCAPS_STORAGE = "/data/SSD2/pre_extract/redcaps_150k/features"
REDCAPS_ANNOT = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"
IMPRESSIONS_STORAGE = "/data/SSD2/pre_extract/impressions/features"
IMPRESSIONS_ANNOT = "/project/Impressions/metadata/impressions_train.json"

POSITIVE_AUC_FLOOR = 0.60
Z_BAR = 2.0


def load_features(storage_dir, annotation_path):
    fm = FeatureManager(storage_dir)
    d = fm.load_all_to_ram(["img_features", "txt_features"])
    img = F.normalize(d["img_features"].float(), dim=1).numpy().astype(np.float32)
    txt = F.normalize(d["txt_features"].float(), dim=1).numpy().astype(np.float32)
    sample_ids = [int(x) for x in d["sample_ids"]]
    meta = json.load(open(annotation_path))
    records = [meta[s] for s in sample_ids]
    return {"img": img, "txt": txt}, records


def evaluate_direction(features, labels, direction):
    scores = features @ direction
    return float(roc_auc_score(labels, scores))


def fold_auc(auc):
    return max(auc, 1.0 - auc)


def decision_rule(real_auc_folded, z):
    if z >= Z_BAR and real_auc_folded >= POSITIVE_AUC_FLOOR:
        return "positive"
    if z >= Z_BAR:
        return "partial"
    return "null"


def audit_axis(features_by_modality, keep, labels, prompts_a, prompts_b, model, tokenizer, n_control=20, seed=42):
    real_dir = ta.build_direction(model, tokenizer, prompts_a, prompts_b, device="cpu")
    control_dirs = ta.build_control_directions(model, tokenizer, n=n_control, seed=seed, device="cpu")

    result = {}
    for modality, features in features_by_modality.items():
        feats = features[keep]
        real_auc = evaluate_direction(feats, labels, real_dir)
        control_aucs = np.array([fold_auc(evaluate_direction(feats, labels, d)) for d in control_dirs])
        real_auc_folded = fold_auc(real_auc)
        z = float((real_auc_folded - control_aucs.mean()) / (control_aucs.std(ddof=1) + 1e-8))
        result[modality] = {
            "n_pos": int(labels.sum()),
            "n_neg": int((1 - labels).sum()),
            "real_auc": real_auc,
            "real_auc_folded": real_auc_folded,
            "control_mean": float(control_aucs.mean()),
            "control_std": float(control_aucs.std(ddof=1)),
            "z": z,
            "verdict": decision_rule(real_auc_folded, z),
        }
    return result


def main():
    model, tokenizer = ta.load_clip_text_tower(device="cpu")

    redcaps_feats, redcaps_records = load_features(REDCAPS_STORAGE, REDCAPS_ANNOT)
    impressions_feats, impressions_records = load_features(IMPRESSIONS_STORAGE, IMPRESSIONS_ANNOT)

    results = {"redcaps_150k": {}, "impressions": {}}

    for axis_name, spec in ax.REDCAPS_AXES.items():
        keep, labels = ax.redcaps_binary_labels(redcaps_records, axis_name)
        results["redcaps_150k"][axis_name] = audit_axis(
            redcaps_feats, keep, labels, spec["prompts_a"], spec["prompts_b"], model, tokenizer
        )

    for axis_name, spec in ax.IMPRESSIONS_AXES.items():
        keep, labels = ax.impressions_binary_labels(impressions_records, axis_name)
        results["impressions"][axis_name] = audit_axis(
            impressions_feats, keep, labels, spec["prompts_a"], spec["prompts_b"], model, tokenizer
        )

    out_path = os.path.join(os.path.dirname(__file__), "raw_clip_audit_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
