"""Copy round 9 inputs through cluster.py's plan_data_sync / run_data_sync (cluster.py is not modified).

First stage the ArtELingo row order with the CoSiR Python (CPU only):
    python scripts/das6_sync_r9.py --node node406 --make-sample-ids
Then use the system Python for plans and copies (conda's OpenSSL breaks rsync's system ssh):
    /usr/bin/python3 scripts/das6_sync_r9.py --node node406 --models --artelingo --a28 --semart
    /usr/bin/python3 scripts/das6_sync_r9.py --node node406 --models facebook/nllb-200-3.3B --run

Each selected group must fit DATA_MAX_GB; the combined total may exceed it. HF repos travel whole, keeping
snapshot symlinks. The sample IDs are staged beside an annotation symlink inside this worktree, then copied
as a manual file action because their local source is outside DATA_MAP. Nothing goes to the node's /tmp.
"""
import argparse
import sys
from pathlib import Path

SKILL = Path("/root/.claude/skills/cluster-run")
sys.path.insert(0, str(SKILL))
import cluster  # type: ignore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / ".scratch" / "r9-sync" / "artelingo"
FEATURES = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_LOCAL = "/data/PDD/artelingo/artelingo_train.json"
ARTELINGO = "/var/scratch/wding/Dataset/artelingo"
A28 = ARTELINGO + "/ArtELingo/semeval-artelingo28"
HF_HUB = "/var/scratch/wding/cache/hub"
MODELS = ("google/siglip2-so400m-patch14-384", "facebook/nllb-200-3.3B", "Qwen/Qwen3-VL-Embedding-2B")


def make_sample_ids():
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from src.utils.feature_manager import FeatureManager

    ids = np.asarray(FeatureManager(storage_dir=FEATURES).get_all_sample_ids(), dtype=np.int64)
    assert ids.shape == (308_723,), ids.shape
    assert len(np.unique(ids)) == len(ids) and ids.min() >= 0
    assert Path(ANNOTATIONS_LOCAL).is_file(), ANNOTATIONS_LOCAL
    STAGING.mkdir(parents=True, exist_ok=True)
    annotations = STAGING / "artelingo_train.json"
    if annotations.is_symlink() or annotations.exists():
        assert annotations.is_symlink() and annotations.resolve() == Path(ANNOTATIONS_LOCAL)
    else:
        annotations.symlink_to(ANNOTATIONS_LOCAL)
    path = STAGING / "r9_sample_ids.npy"
    np.save(path, ids, allow_pickle=False)
    print(f"Staged {path}: {len(ids):,} rows, {ids.dtype}, {path.stat().st_size} bytes")


def mapped_actions(cfg, mapped):
    actions, unresolved = cluster.plan_data_sync(cfg, mapped, mapped)
    assert not unresolved, f"UNRESOLVED: {unresolved}"
    return actions


def build_groups(cfg, args):
    groups = {}
    if args.models is not None:
        repos = args.models.split(",")
        assert len(set(repos)) == len(repos) and all(repo in MODELS for repo in repos), repos
        mapped = {"hf_model_" + repo.replace("/", "--"): HF_HUB + "/models--" + repo.replace("/", "--")
                  for repo in repos}
        groups["models"] = mapped_actions(cfg, mapped)
    if args.artelingo:
        path = STAGING / "r9_sample_ids.npy"
        assert path.is_file(), f"run --make-sample-ids with the CoSiR Python first: {path}"
        groups["artelingo"] = mapped_actions(cfg, {"artelingo_annotations": ARTELINGO + "/artelingo_train.json"})
        groups["artelingo"].append({"key": "r9_sample_ids", "kind": "file", "local": str(path),
                                    "remote": ARTELINGO + "/r9_sample_ids.npy", "bytes": path.stat().st_size})
    if args.a28:
        groups["a28"] = mapped_actions(cfg, {"a28_csv": A28 + "/artelingo28_public_noworkerid_split.csv",
                                             "a28_images_zip": A28 + "/images.zip"})
    if args.semart:
        data_map = cluster.parse_data_map(cfg["DATA_MAP"])
        for dataset in ("semart", "cub"):
            remote = f"/var/scratch/wding/Dataset/{dataset}"
            local = f"/data/SSD/{dataset}"
            assert cluster.map_to_local(remote, data_map) == local, f"DATA_MAP must map {remote} to {local}"
        groups["semart"] = mapped_actions(cfg, {"semart": "/var/scratch/wding/Dataset/semart/SemArt"})
    for actions in groups.values():
        for a in actions:
            assert a["remote"].startswith(("/var/scratch/wding/", "/local/wding/")), a["remote"]
            assert ".." not in Path(a["remote"]).parts, a["remote"]
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", required=True)
    ap.add_argument("--models", nargs="?", const=",".join(MODELS), metavar="REPO,REPO",
                    help="all three HF repos, or a comma list of exact repo IDs")
    ap.add_argument("--artelingo", action="store_true")
    ap.add_argument("--a28", action="store_true")
    ap.add_argument("--semart", action="store_true")
    ap.add_argument("--make-sample-ids", action="store_true", help="stage row order with the CoSiR Python, then exit")
    ap.add_argument("--config", type=Path, default=SKILL / "cluster.conf", help="machine config (default: cluster.conf)")
    ap.add_argument("--run", action="store_true", help="perform the copy (default: print the plan only)")
    args = ap.parse_args()
    selected = args.models is not None or args.artelingo or args.a28 or args.semart
    if args.make_sample_ids:
        if selected or args.run:
            ap.error("--make-sample-ids is a separate local step; do not combine with transfer flags or --run")
        make_sample_ids()
        return
    if not selected:
        ap.error("select --models, --artelingo, --a28 and/or --semart")
    cfg = cluster.load_config(args.config, project="CoSiR", projects_dir=SKILL / "projects")
    node = cluster.validate_node(args.node, cfg)
    groups = build_groups(cfg, args)
    max_gb = float(cfg["DATA_MAX_GB"])
    print(f"Plan for {node} (DATA_MAX_GB {max_gb:g} per group):")
    oversized = []
    for name, actions in groups.items():
        size_gb = sum(a["bytes"] for a in actions) / 1024**3
        print(f"  {name}: {len(actions)} actions, {size_gb:.3f} GB")
        for a in actions:
            print(f"    {a['key']}: {a['kind']}, {a['bytes'] / 1024**3:.3f} GB, {a['local']} -> {a['remote']}")
        if size_gb > max_gb:
            oversized.append(name)
    total_gb = sum(a["bytes"] for actions in groups.values() for a in actions) / 1024**3
    print(f"Total: {total_gb:.3f} GB (cap applies separately to each group)")
    if oversized:
        sys.exit(f"REFUSING: groups {oversized} over DATA_MAX_GB={max_gb:g}; select fewer repos with --models REPO,REPO")
    if not args.run:
        print("Plan only; pass --run to copy.")
        return
    for name, actions in groups.items():
        print(f"Copying {name}:", flush=True)
        print("Done:", cluster.run_data_sync(cfg, node, actions))


if __name__ == "__main__":
    main()
