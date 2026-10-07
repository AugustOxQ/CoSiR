"""Stage the inputs of the CLIP fine-tuning comparator (scripts/run_clipft.sh) on a DAS6 node through the cluster CLI's
own data-sync functions (plan_data_sync / run_data_sync, used as a library; cluster.py is not modified).

The job is a `bash scripts/run_*.sh` job, so `cluster check --sync-data` cannot see its paths. Follows
scripts/das6_sync_mllm_probe_8b.py. Four transfers, all resolved through DATA_MAP:
- the uint8 224x224 image cache (/data/SSD2/pre_extract/artelingo_clip224, ~7.4 GB; must be complete);
- the ArtELingo train annotations and the CLIP feature cache;
- the openai/clip-vit-base-patch32 hub repo (whole dir, symlinks kept).

Run with the system Python, NOT the CoSiR conda env (conda's OpenSSL breaks the system ssh that rsync calls):
    /usr/bin/python3 scripts/das6_sync_clipft.py [--node node405]        # print the plan and exit
    /usr/bin/python3 scripts/das6_sync_clipft.py [--node node405] --run  # perform it
--node defaults to cluster.conf's NODE.
"""
import argparse
import sys
from pathlib import Path

SKILL = Path("/root/.claude/skills/cluster-run")
sys.path.insert(0, str(SKILL))
import cluster  # type: ignore  # noqa: E402

# Node paths: the same defaults as scripts/run_clipft.sh.
IMAGE_CACHE = "/local/wding/Dataset/pre_extract/artelingo_clip224"
ANNOTATIONS = "/local/wding/Dataset/artelingo/artelingo_train.json"
FEATURES = "/local/wding/pre_extract/artelingo/features"
HF_MODEL = "/var/scratch/wding/cache/hub/models--openai--clip-vit-base-patch32"
CACHE_FILES = ("images_uint8.npy", "paintings.json", "cache_record.json")


def build_actions(cfg, need_complete):
    mapped = {"clipft_image_cache": IMAGE_CACHE, "artelingo_annotations": ANNOTATIONS,
              "artelingo_features": FEATURES, "hf_model_clip_vit_b32": HF_MODEL}
    actions, unresolved = cluster.plan_data_sync(cfg, mapped, mapped)
    if unresolved:
        sys.exit(f"UNRESOLVED: {unresolved}")
    kinds = {a["key"]: a["kind"] for a in actions}
    for key in ("clipft_image_cache", "hf_model_clip_vit_b32", "artelingo_features"):
        if kinds.get(key) != "dir":
            sys.exit(f"{key} must be a whole-dir transfer, planned as {kinds.get(key)}")
    cache_local = next(a["local"] for a in actions if a["key"] == "clipft_image_cache")
    absent = [f for f in CACHE_FILES if not (Path(cache_local) / f).is_file()]
    if absent:
        msg = f"image cache at {cache_local} is incomplete, missing {absent}"
        if need_complete:
            sys.exit(f"REFUSING --run: {msg}")
        print(f"WARNING: {msg} (the plan includes partial files; --run refuses until it is built)")
    return actions


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", default=None, help="default: NODE from cluster.conf")
    ap.add_argument("--run", action="store_true", help="perform the transfers (default: print the plan only)")
    args = ap.parse_args()
    cfg = cluster.load_config(SKILL / "cluster.conf", project="CoSiR")
    node = cluster.validate_node(args.node or cfg["NODE"], cfg)
    actions = build_actions(cfg, need_complete=args.run)

    max_gb = float(cfg["DATA_MAX_GB"])
    print(f"Plan for {node} ({len(actions)} actions):")
    for a in actions:
        print(f"  {a['key']}: {a['kind']}, {a['bytes'] / 1024**3:.3f} GB, {a['local']} -> {a['remote']}")
    total_gb = sum(a["bytes"] for a in actions) / 1024**3
    print(f"Total: {total_gb:.2f} GB (DATA_MAX_GB {max_gb:g})")
    if any(a["bytes"] / 1024**3 > max_gb for a in actions) or total_gb > max_gb:
        sys.exit(f"REFUSING: over DATA_MAX_GB={max_gb:g}")
    if not args.run:
        print("Plan only; pass --run to transfer.")
        return
    done = cluster.run_data_sync(cfg, node, actions)
    print("Done:", done)


if __name__ == "__main__":
    main()
