"""Find completed CoSiR training runs whose condition-vector checkpoints match
the current default architecture (asymmetric, combine_side='img', buddy-init) —
candidates for Experiment 17.1(b)'s existing-checkpoint probe.
"""
import ast
import glob
import json
import os


def find_candidate_checkpoints(res_globs):
    candidates = []
    for pattern in res_globs:
        for exp_dir in sorted(glob.glob(pattern)):
            meta_path = os.path.join(exp_dir, "experiment_metadata.json")
            emb_path = os.path.join(exp_dir, "final_embeddings", "embeddings.npy")
            if not (os.path.isfile(meta_path) and os.path.isfile(emb_path)):
                continue
            meta = json.load(open(meta_path))
            try:
                cfg = ast.literal_eval(meta["config"])
            except (KeyError, ValueError, SyntaxError):
                continue
            model_cfg = cfg.get("model", {})
            train_cfg = cfg.get("train", {})
            if (
                model_cfg.get("combine_side") == "img"
                and model_cfg.get("conditioning_mode", "asymmetric") == "asymmetric"
                and train_cfg.get("initialization_strategy") == "buddies"
            ):
                candidates.append(exp_dir)
    return candidates


REDCAPS_150K_GLOBS = [
    "res/CoSiR_init_ablation/redcaps_150k/init_buddies/*_CoSiR_Experiment",
    "res/CoSiR_condition_freeze_ablation/redcaps_150k/*_CoSiR_Experiment",
]
IMPRESSIONS_GLOBS = [
    "res/CoSiR_init_ablation/impressions/init_buddies/*_CoSiR_Experiment",
]


def main():
    result = {
        "redcaps_150k": find_candidate_checkpoints(REDCAPS_150K_GLOBS),
        "impressions": find_candidate_checkpoints(IMPRESSIONS_GLOBS),
    }
    out_path = os.path.join(os.path.dirname(__file__), "checkpoints.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"redcaps_150k: {len(result['redcaps_150k'])} candidates")
    print(f"impressions: {len(result['impressions'])} candidates")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
