"""Copy round 6 GPU job inputs to a DAS6 node through the cluster CLI's own data-sync functions (plan_data_sync /
run_data_sync, used as a library, as scripts/das6_sync_mllm_probe_8b.py does; cluster.py is not modified).

Actions (each chosen by its flag):
- --job-dir DIR: a job-input folder (r6_gpu_inputs.py's verbaliser folder, a listing folder with listing_input.jsonl,
  or another round-6 job folder) -> /local/wding/r6_jobs/<folder name>, the default R6_JOB_ROOT of the wrappers. The
  folder is refused if it holds an episodes file or any array named like one (anchor, candidates, pair names, episode
  hashes) or a label: GPU jobs receive row ids, image paths and captions only (rule section 8 item 3).
- --images: the WikiArt images listed in DIR/images.txt (or --image-list FILE), a 'selected' action into the node's
  WikiArt tree (DATA_MAP). A list larger than DATA_MAX_GB is split into consecutive chunks of the sorted list, each at
  most DATA_MAX_GB; the plan shows them and --chunk K copies one (rsync skips files already on the node).
- --model: the Qwen3-VL-8B-Instruct hub repo into the node HF cache (HF_HUB_REMOTE, via DATA_MAP), only when the
  wrapper's snapshot check finds it missing (the cache is shared by the nodes and already holds it from the 8B probe).
Every remote path except the model's lies under /local/wding/ (asserted); nothing goes to a node's /tmp.

Run with the system Python, NOT the CoSiR conda env (conda's OpenSSL breaks the system ssh that rsync calls):
    /usr/bin/python3 scripts/das6_sync_r6.py --node node401 --job-dir <dir> [--images [--chunk K]] [--model]   # plan
    /usr/bin/python3 scripts/das6_sync_r6.py --node node401 --job-dir <dir> --images --chunk 0 --run           # copy
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

SKILL = Path("/root/.claude/skills/cluster-run")
sys.path.insert(0, str(SKILL))
import cluster  # type: ignore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/local/wding/"
JOBS_REMOTE = "/local/wding/r6_jobs"
WIKIART = "/local/wding/Dataset/wikiart_proj/wikiart"
WIKIART_LOCAL = "/data/PDD/wikiart_proj/wikiart"
HF_REPO = "models--Qwen--Qwen3-VL-8B-Instruct"
JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
JOB_INPUTS = ("rows_manifest.npz", "verbalise_input.npz", "listing_input.jsonl", "rerank_input.npz", "ft_rows.npz")
FORBIDDEN_ARRAYS = {"anchor", "candidates", "pair_names", "pair_index", "labels", "label", "emotion", "style", "genre",
                    "art_style", "aspect", "aspects", "painting", "paintings"}


def forbidden_in_job(job: Path) -> list:
    """Files of the job folder that look like episodes or labels: an npz array named like an episode-file or label
    field (npz members are '<name>.npy'; read with zipfile, so no numpy is needed), or a file named like an episodes
    file."""
    bad = []
    for f in sorted(job.rglob("*")):
        if not f.is_file():
            continue
        if "episode" in f.name:
            bad.append(f"{f.relative_to(job)}: an episodes file")
        if f.suffix == ".npz":
            try:
                with zipfile.ZipFile(f) as z:
                    names = {Path(n).stem for n in z.namelist()}
            except zipfile.BadZipFile:
                bad.append(f"{f.relative_to(job)}: not a readable npz")
                continue
            named = sorted(n for n in names if n in FORBIDDEN_ARRAYS or n.startswith("sha__"))
            if named:
                bad.append(f"{f.relative_to(job)}: arrays {named}")
    return bad


def image_chunks(files, sizes, max_bytes) -> list:
    """Consecutive slices of the sorted list, each at most max_bytes (one file larger than that is refused)."""
    chunks, cur, cur_bytes = [], [], 0
    for f, b in zip(files, sizes):
        if b > max_bytes:
            sys.exit(f"REFUSING: {f} alone is over DATA_MAX_GB")
        if cur and cur_bytes + b > max_bytes:
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(f)
        cur_bytes += b
    if cur:
        chunks.append(cur)
    return chunks


def build_actions(cfg, args):
    actions, notes = [], []
    max_bytes = float(cfg["DATA_MAX_GB"]) * 1024**3
    job = None
    if args.job_dir:
        job = Path(args.job_dir).resolve()
        if not job.is_dir() or not JOB_NAME.fullmatch(job.name):
            sys.exit(f"--job-dir {job}: not a folder with a plain name")
        if not any((job / f).is_file() for f in JOB_INPUTS):
            sys.exit(f"--job-dir {job}: holds none of {', '.join(JOB_INPUTS)}")
        bad = forbidden_in_job(job)
        if bad:
            sys.exit("REFUSING: the job folder holds episode or label data: " + "; ".join(bad))
        actions.append({"key": "r6_job_" + job.name, "kind": "dir", "remote": f"{JOBS_REMOTE}/{job.name}",
                        "local": str(job), "bytes": cluster.dir_bytes(job)})
    if args.images or args.image_list:
        listing = Path(args.image_list) if args.image_list else (job / "images.txt" if job else None)
        if listing is None or not listing.is_file():
            sys.exit("--images needs --job-dir with an images.txt, or --image-list FILE")
        local_images = cluster.map_to_local(WIKIART, cluster.parse_data_map(cfg["DATA_MAP"]))
        if local_images != WIKIART_LOCAL:
            sys.exit(f"DATA_MAP maps {WIKIART} to {local_images}, expected {WIKIART_LOCAL}")
        files = [line for line in listing.read_text().splitlines() if line]
        if not files or len(set(files)) != len(files) or files != sorted(files):
            sys.exit(f"{listing} must be non-empty, sorted and unique")
        if any(f.startswith("/") or ".." in Path(f).parts for f in files):
            sys.exit(f"{listing}: every path must be relative to the WikiArt root")
        missing = [f for f in files if not (Path(local_images) / f).is_file()]
        if missing:
            sys.exit(f"{len(missing)} listed images missing locally, e.g. {missing[:3]}")
        sizes = [(Path(local_images) / f).stat().st_size for f in files]
        chunks = image_chunks(files, sizes, max_bytes)
        size_of = dict(zip(files, sizes))
        for k, ch in enumerate(chunks):
            notes.append(f"  images chunk {k}: {len(ch)} files, {sum(size_of[f] for f in ch) / 1024**3:.3f} GB "
                         f"({ch[0]} .. {ch[-1]})")
        if len(chunks) > 1 and args.chunk is None:
            notes.append(f"  {len(chunks)} image chunks: pass --chunk K (0 to {len(chunks) - 1}) to copy one")
            if args.run:
                sys.exit("REFUSING: the image list is over DATA_MAX_GB; copy it chunk by chunk with --chunk K")
        pick = range(len(chunks)) if args.chunk is None else [args.chunk]
        if args.chunk is not None and not 0 <= args.chunk < len(chunks):
            sys.exit(f"--chunk {args.chunk}: the list has {len(chunks)} chunks")
        for k in pick:
            actions.append({"key": f"wikiart_r6_images_chunk{k}", "kind": "selected", "remote": WIKIART,
                            "local": local_images, "files": chunks[k], "bytes": sum(size_of[f] for f in chunks[k]),
                            "from_list": str(listing)})
    if args.model:
        remote = f"{cfg['HF_HUB_REMOTE'].rstrip('/')}/{HF_REPO}"
        mapped = {"hf_model_qwen3_vl_8b": remote}
        planned, unresolved = cluster.plan_data_sync(cfg, mapped, mapped)
        if unresolved:
            sys.exit(f"UNRESOLVED: {unresolved}")
        actions += planned
    for a in actions:
        if a["key"] == "hf_model_qwen3_vl_8b":
            if not a["remote"].startswith(cfg["HF_HUB_REMOTE"].rstrip("/") + "/"):
                sys.exit(f"REFUSING: model remote {a['remote']} outside HF_HUB_REMOTE")
        elif not a["remote"].startswith(REMOTE_ROOT) or ".." in Path(a["remote"]).parts:
            sys.exit(f"REFUSING: remote {a['remote']} outside {REMOTE_ROOT}")
    return actions, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", required=True, help="node401, node402 or node408 (each node has its own /local/wding)")
    ap.add_argument("--job-dir", help="a local job-input folder")
    ap.add_argument("--images", action="store_true", help="also copy the images listed in <job-dir>/images.txt")
    ap.add_argument("--image-list", help="an image list (relative to the WikiArt root) in place of <job-dir>/images.txt")
    ap.add_argument("--chunk", type=int, default=None, help="copy only this chunk of the image list")
    ap.add_argument("--model", action="store_true", help="also copy the 8B model repo into the node HF cache")
    ap.add_argument("--run", action="store_true", help="perform the copy (default: print the plan only)")
    args = ap.parse_args()
    if not (args.job_dir or args.images or args.image_list or args.model):
        ap.error("nothing to copy: give --job-dir, --images / --image-list or --model")
    cfg = cluster.load_config(SKILL / "cluster.conf", project="CoSiR")
    node = cluster.validate_node(args.node, cfg)
    actions, notes = build_actions(cfg, args)

    max_gb = float(cfg["DATA_MAX_GB"])
    print(f"Plan for {node} ({len(actions)} actions):")
    for a in actions:
        extra = f", {len(a['files'])} files" if a["kind"] == "selected" else ""
        print(f"  {a['key']}: {a['kind']}{extra}, {a['bytes'] / 1024**3:.3f} GB, {a['local']} -> {a['remote']}")
    for line in notes:
        print(line)
    total_gb = sum(a["bytes"] for a in actions) / 1024**3
    print(f"Total: {total_gb:.2f} GB (DATA_MAX_GB {max_gb:g})")
    if not args.run:
        print("Plan only; pass --run to copy.")
        return
    if any(a["bytes"] / 1024**3 > max_gb for a in actions) or total_gb > max_gb:
        sys.exit(f"REFUSING: over DATA_MAX_GB={max_gb:g}; copy the actions one at a time")
    done = cluster.run_data_sync(cfg, node, actions)
    print("Done:", done)


if __name__ == "__main__":
    main()
