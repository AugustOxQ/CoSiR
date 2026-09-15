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


def audit_axis(
    features_by_modality,
    keep,
    labels,
    prompts_a,
    prompts_b,
    model,
    tokenizer,
    n_control=20,
    seed=42,
    records=None,
    content_control_prompts_a=None,
    content_control_prompts_b=None,
):
    real_dir = ta.build_direction(model, tokenizer, prompts_a, prompts_b, device="cpu")
    control_dirs = ta.build_control_directions(model, tokenizer, n=n_control, seed=seed, device="cpu")
    content_control_dir = None
    if content_control_prompts_a is not None and content_control_prompts_b is not None:
        content_control_dir = ta.build_direction(
            model,
            tokenizer,
            content_control_prompts_a,
            content_control_prompts_b,
            device="cpu",
        )

    img_ids = None
    if records is not None:
        candidate_img_ids = [records[i].get("ImgId") for i in keep]
        present_img_ids = sum(img_id is not None for img_id in candidate_img_ids)
        if present_img_ids and present_img_ids * 2 >= len(candidate_img_ids):
            img_ids = candidate_img_ids

    result = {}
    for modality, features in features_by_modality.items():
        feats = features[keep]
        real_auc = evaluate_direction(feats, labels, real_dir)
        control_aucs = np.array([fold_auc(evaluate_direction(feats, labels, d)) for d in control_dirs])
        real_auc_folded = fold_auc(real_auc)
        z = float((real_auc_folded - control_aucs.mean()) / (control_aucs.std(ddof=1) + 1e-8))
        content_control_auc = None
        content_control_auc_folded = None
        beats_content_control = None
        if content_control_dir is not None:
            content_control_auc = evaluate_direction(feats, labels, content_control_dir)
            content_control_auc_folded = fold_auc(content_control_auc)
            beats_content_control = real_auc_folded > content_control_auc_folded
        result[modality] = {
            "n_pos": int(labels.sum()),
            "n_neg": int((1 - labels).sum()),
            "real_auc": real_auc,
            "real_auc_folded": real_auc_folded,
            "control_mean": float(control_aucs.mean()),
            "control_std": float(control_aucs.std(ddof=1)),
            "control_rank": f"{int((control_aucs < real_auc_folded).sum())} of {len(control_aucs)} controls exceeded",
            "z": z,
            "verdict": decision_rule(real_auc_folded, z),
            "content_control_auc": content_control_auc,
            "content_control_auc_folded": content_control_auc_folded,
            "beats_content_control": beats_content_control,
        }
        if modality == "img" and img_ids is not None:
            pole_a_img_ids = {img_id for img_id, label in zip(img_ids, labels) if label == 1 and img_id is not None}
            pole_b_img_ids = {img_id for img_id, label in zip(img_ids, labels) if label == 0 and img_id is not None}
            if pole_a_img_ids and pole_b_img_ids:
                overlap = len(pole_a_img_ids & pole_b_img_ids) / len(pole_a_img_ids | pole_b_img_ids)
                if overlap > 0.9:
                    result[modality]["verdict"] = "n/a_same_images"
    return result


def main():
    model, tokenizer = ta.load_clip_text_tower(device="cpu")

    redcaps_feats, redcaps_records = load_features(REDCAPS_STORAGE, REDCAPS_ANNOT)
    impressions_feats, impressions_records = load_features(IMPRESSIONS_STORAGE, IMPRESSIONS_ANNOT)

    results = {"redcaps_150k": {}, "impressions": {}}

    for axis_name, spec in ax.REDCAPS_AXES.items():
        keep, labels = ax.redcaps_binary_labels(redcaps_records, axis_name)
        results["redcaps_150k"][axis_name] = audit_axis(
            redcaps_feats, keep, labels, spec["prompts_a"], spec["prompts_b"], model, tokenizer,
            records=redcaps_records,
            content_control_prompts_a=spec.get("content_control_prompts_a"),
            content_control_prompts_b=spec.get("content_control_prompts_b"),
        )

    for axis_name, spec in ax.IMPRESSIONS_AXES.items():
        keep, labels = ax.impressions_binary_labels(impressions_records, axis_name)
        results["impressions"][axis_name] = audit_axis(
            impressions_feats, keep, labels, spec["prompts_a"], spec["prompts_b"], model, tokenizer,
            records=impressions_records,
            content_control_prompts_a=spec.get("content_control_prompts_a"),
            content_control_prompts_b=spec.get("content_control_prompts_b"),
        )

    # Pipeline sanity check, not a formal experimental axis.
    sanity_keep = np.array(
        [i for i, record in enumerate(redcaps_records) if ax._subreddit_of(record) in ("cats", "carporn")],
        dtype=np.int64,
    )
    sanity_labels = np.array(
        [1 if ax._subreddit_of(redcaps_records[i]) == "cats" else 0 for i in sanity_keep], dtype=np.int64
    )
    results["sanity_check"] = audit_axis(
        {"txt": redcaps_feats["txt"]},
        sanity_keep,
        sanity_labels,
        ["a caption about a cat"],
        ["a caption about a car"],
        model,
        tokenizer,
        records=redcaps_records,
    )

    out_path = os.path.join(os.path.dirname(__file__), "raw_clip_audit_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
