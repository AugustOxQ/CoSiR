"""Copy the inputs of the 8B MLLM probe (scripts/run_mllm_probe_8b.sh) to a DAS6 node through the cluster CLI's own
data-sync functions (plan_data_sync / run_data_sync, used as a library; cluster.py is not modified).

The probe is a `bash scripts/run_*.sh` job, so `cluster check --sync-data` cannot see its paths (no Hydra config).
This script builds the transfers by hand, following src/test/20260928_buddy_percept_sweep/force_data_sync_das6.py
(commit e8fc714):
- the Qwen3-VL-8B-Instruct hub repo (whole dir, symlinks kept) -> HF hub cache on DAS6 (DATA_MAP);
- the ArtELingo train annotations and CLIP feature cache (re-sync is cheap with rsync; fixes a stale or missing copy);
- the WikiArt genre CSVs (no DATA_MAP prefix covers the node path, so a manual whole-dir action);
- only the WikiArt images the seed-46 episodes use, from scripts/mllm_probe_8b_images.txt (manual 'selected' action).

Run with the system Python, NOT the CoSiR conda env (conda's OpenSSL breaks the system ssh that rsync calls):
    /usr/bin/python3 scripts/das6_sync_mllm_probe_8b.py            # print the plan and exit
    /usr/bin/python3 scripts/das6_sync_mllm_probe_8b.py --run      # copy to node404 (or --node <node>)
"""
import argparse
import sys
from pathlib import Path

SKILL = Path("/root/.claude/skills/cluster-run")
sys.path.insert(0, str(SKILL))
import cluster  # type: ignore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGE_LIST = ROOT / "scripts" / "mllm_probe_8b_images.txt"

# Node paths: the same defaults as scripts/run_mllm_probe_8b.sh.
HF_MODEL = "/var/scratch/wding/cache/hub/models--Qwen--Qwen3-VL-8B-Instruct"
ANNOTATIONS = "/local/wding/Dataset/artelingo/artelingo_train.json"
FEATURES = "/local/wding/pre_extract/artelingo/features"
WIKIART = "/local/wding/Dataset/wikiart_proj/wikiart"
GENRE_REMOTE, GENRE_LOCAL = "/local/wding/Dataset/wikiart_genre", "/data/SSD/wikiart_genre"


def build_actions(cfg):
    mapped = {"hf_model_qwen3_vl_8b": HF_MODEL, "artelingo_annotations": ANNOTATIONS, "artelingo_features": FEATURES}
    actions, unresolved = cluster.plan_data_sync(cfg, mapped, mapped)
    if unresolved:
        sys.exit(f"UNRESOLVED: {unresolved}")

    genre = Path(GENRE_LOCAL)
    if not all((genre / name).is_file() for name in ("genre_train.csv", "genre_val.csv")):
        sys.exit(f"genre CSVs missing under {genre}")
    actions.append({"key": "wikiart_genre", "kind": "dir", "remote": GENRE_REMOTE, "local": GENRE_LOCAL,
                    "bytes": cluster.dir_bytes(genre)})

    local_images = cluster.map_to_local(WIKIART, cluster.parse_data_map(cfg["DATA_MAP"]))
    if local_images != "/data/PDD/wikiart_proj/wikiart":
        sys.exit(f"DATA_MAP maps {WIKIART} to {local_images}, expected /data/PDD/wikiart_proj/wikiart")
    files = [line for line in IMAGE_LIST.read_text().splitlines() if line]
    if len(set(files)) != len(files) or files != sorted(files):
        sys.exit(f"{IMAGE_LIST} must be sorted and unique")
    missing = [f for f in files if not (Path(local_images) / f).is_file()]
    if missing:
        sys.exit(f"{len(missing)} listed images missing locally, e.g. {missing[:3]}")
    actions.append({"key": "wikiart_probe_images", "kind": "selected", "remote": WIKIART, "local": local_images,
                    "files": files, "bytes": sum((Path(local_images) / f).stat().st_size for f in files),
                    "from_list": str(IMAGE_LIST.relative_to(ROOT))})
    return actions


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", default="node404")
    ap.add_argument("--run", action="store_true", help="perform the copy (default: print the plan only)")
    args = ap.parse_args()
    cfg = cluster.load_config(SKILL / "cluster.conf", project="CoSiR")
    node = cluster.validate_node(args.node, cfg)
    actions = build_actions(cfg)

    max_gb = float(cfg["DATA_MAX_GB"])
    print(f"Plan for {node} ({len(actions)} actions):")
    for a in actions:
        extra = f", {len(a['files'])} files" if a["kind"] == "selected" else ""
        print(f"  {a['key']}: {a['kind']}{extra}, {a['bytes'] / 1024**3:.3f} GB, {a['local']} -> {a['remote']}")
    total_gb = sum(a["bytes"] for a in actions) / 1024**3
    print(f"Total: {total_gb:.2f} GB (DATA_MAX_GB {max_gb:g})")
    if any(a["bytes"] / 1024**3 > max_gb for a in actions) or total_gb > max_gb:
        sys.exit(f"REFUSING: over DATA_MAX_GB={max_gb:g}")
    if not args.run:
        print("Plan only; pass --run to copy.")
        return
    done = cluster.run_data_sync(cfg, node, actions)
    print("Done:", done)


if __name__ == "__main__":
    main()
