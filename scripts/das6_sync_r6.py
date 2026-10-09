"""Copy round 6 GPU job inputs to a DAS6 node through the cluster CLI's own functions (plan and rsync helpers of
cluster.py, used as a library as scripts/das6_sync_mllm_probe_8b.py does; cluster.py is not modified).

A GPU job receives row ids, images and captions only (rule section 8 item 3). WikiArt paths begin with the style
folder, so images travel under neutral names: r6_gpu_inputs.py writes, beside each job folder, a CPU-only map
<job>.image_map.json ({neutral name: WikiArt path}) and stages the images as symlinks <staging>/<name>.

Actions (each chosen by its flag):
- --job-dir DIR: the job folder -> /local/wding/r6_jobs/<folder name>. Refused if it holds an episodes file, an array
  named like an episode or label field, an image map, a name in images.txt that is not neutral, or any WikiArt path
  ("<style folder>/" in UTF-8 or UTF-32, the npz string encoding, searched in every file and npz member).
- --images: the images of DIR's map (or --image-map FILE) -> /local/wding/r6_jobs/images/<name>. Each staged link
  must resolve to the map's WikiArt file and its name must be that path's neutral name; rsync follows the links
  (cluster.rsync_argv plus --copy-links), so the node holds plain files under neutral names only. A list over
  DATA_MAX_GB is split into consecutive chunks of the sorted names; --chunk K copies one (rsync skips files present).
Every remote path lies under /local/wding/ (asserted); nothing goes to a node's /tmp. The 8B model is not copied
here: the wrappers read it from the node HF cache (/var/scratch/wding/cache/hub, which holds the pinned snapshot from
the 8B probe) and stop if it is missing; scripts/das6_sync_mllm_probe_8b.py's model action copies it.

Run with the system Python, NOT the CoSiR conda env (conda's OpenSSL breaks the system ssh that rsync calls):
    /usr/bin/python3 scripts/das6_sync_r6.py --node node401 --job-dir <dir> [--images [--chunk K]]   # plan
    /usr/bin/python3 scripts/das6_sync_r6.py --node node401 --job-dir <dir> --images --run           # copy
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

SKILL = Path("/root/.claude/skills/cluster-run")
sys.path.insert(0, str(SKILL))
import cluster  # type: ignore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/local/wding/"
JOBS_REMOTE = "/local/wding/r6_jobs"
IMAGES_REMOTE = "/local/wding/r6_jobs/images"
WIKIART_LOCAL = "/data/PDD/wikiart_proj/wikiart"
JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
NEUTRAL = re.compile(r"[0-9a-f]{20}\.[a-z0-9]{1,5}")
JOB_INPUTS = ("rows_manifest.npz", "verbalise_input.npz", "listing_input.jsonl", "rerank_input.npz", "ft_rows.npz")
FORBIDDEN_ARRAYS = {"anchor", "candidates", "pair_names", "pair_index", "labels", "label", "emotion", "style", "genre",
                    "art_style", "aspect", "aspects", "painting", "paintings", "image_relpath", "sample_id"}


def style_folders(image_root=WIKIART_LOCAL) -> list:
    folders = sorted(p.name for p in Path(image_root).iterdir() if p.is_dir())
    if not folders:
        sys.exit(f"no style folders under {image_root}")
    return folders


def path_pattern(folders):
    """Bytes regex of '<style folder>/' in UTF-8 and in UTF-32-LE (numpy's unicode arrays)."""
    alts = [re.escape((f + "/").encode("utf-8")) for f in folders]
    alts += [re.escape((f + "/").encode("utf-32-le")) for f in folders]
    return re.compile(b"|".join(alts))


def forbidden_in_job(job: Path, folders) -> list:
    """What in the job folder could carry episodes or labels (see the module docstring)."""
    bad = []
    pattern = path_pattern(folders)
    for f in sorted(job.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(job)
        if "episode" in f.name:
            bad.append(f"{rel}: an episodes file")
        if f.name.endswith(".image_map.json"):
            bad.append(f"{rel}: an image map (CPU-only)")
        if f.name == "images.txt":
            odd = [x for x in f.read_text().splitlines() if x and not NEUTRAL.fullmatch(x)]
            if odd:
                bad.append(f"{rel}: names that are not neutral, e.g. {odd[:2]}")
        blobs = []
        if f.suffix == ".npz":
            try:
                with zipfile.ZipFile(f) as z:
                    names = {Path(n).stem for n in z.namelist()}
                    blobs = [(n, z.read(n)) for n in z.namelist()]
            except zipfile.BadZipFile:
                bad.append(f"{rel}: not a readable npz")
                continue
            named = sorted(n for n in names if n in FORBIDDEN_ARRAYS or n.startswith("sha__"))
            if named:
                bad.append(f"{rel}: arrays {named}")
        else:
            blobs = [(f.name, f.read_bytes())]
        for n, b in blobs:
            m = pattern.search(b)
            if m:
                bad.append(f"{rel} ({n}): a WikiArt path")
    return bad


def neutral_name(relpath) -> str:
    ext = Path(relpath).suffix.lower().lstrip(".")
    return f"{hashlib.sha256(relpath.encode('utf-8')).hexdigest()[:20]}.{ext}"


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


def image_actions(cfg, args, job, notes) -> list:
    max_bytes = float(cfg["DATA_MAX_GB"]) * 1024**3
    map_file = Path(args.image_map) if args.image_map else (job.with_name(job.name + ".image_map.json") if job else None)
    if map_file is None or not map_file.is_file():
        sys.exit("--images needs --job-dir with its <job>.image_map.json beside it, or --image-map FILE")
    m = json.loads(map_file.read_text())
    staging, image_root, images = Path(m["staging"]), Path(m["image_root"]), m["images"]
    names = sorted(images)
    if not names:
        sys.exit(f"{map_file}: no images")
    wrong = [n for n in names if not NEUTRAL.fullmatch(n) or neutral_name(images[n]) != n]
    if wrong:
        sys.exit(f"{map_file}: names that are not their path's neutral name, e.g. {wrong[:2]}")
    unstaged = [n for n in names if os.path.realpath(staging / n) != os.path.realpath(image_root / images[n])
                or not (staging / n).is_file()]
    if unstaged:
        sys.exit(f"{len(unstaged)} images are not staged in {staging} as links to their files, e.g. {unstaged[:2]}")
    sizes = [(staging / n).stat().st_size for n in names]
    chunks = image_chunks(names, sizes, max_bytes)
    size_of = dict(zip(names, sizes))
    for k, ch in enumerate(chunks):
        notes.append(f"  images chunk {k}: {len(ch)} files, {sum(size_of[f] for f in ch) / 1024**3:.3f} GB")
    if len(chunks) > 1 and args.chunk is None:
        notes.append(f"  {len(chunks)} image chunks: pass --chunk K (0 to {len(chunks) - 1}) to copy one")
        if args.run:
            sys.exit("REFUSING: the image list is over DATA_MAX_GB; copy it chunk by chunk with --chunk K")
    if args.chunk is not None and not 0 <= args.chunk < len(chunks):
        sys.exit(f"--chunk {args.chunk}: the list has {len(chunks)} chunks")
    pick = range(len(chunks)) if args.chunk is None else [args.chunk]
    return [{"key": f"r6_images_chunk{k}", "kind": "selected", "remote": IMAGES_REMOTE, "local": str(staging),
             "files": chunks[k], "bytes": sum(size_of[f] for f in chunks[k]), "copy_links": True,
             "from_map": str(map_file)} for k in pick]


def build_actions(cfg, args):
    actions, notes = [], []
    job = None
    if args.job_dir:
        job = Path(args.job_dir).resolve()
        if not job.is_dir() or not JOB_NAME.fullmatch(job.name) or job.name == "images":
            sys.exit(f"--job-dir {job}: not a folder with a plain name")
        if not any((job / f).is_file() for f in JOB_INPUTS):
            sys.exit(f"--job-dir {job}: holds none of {', '.join(JOB_INPUTS)}")
        bad = forbidden_in_job(job, style_folders())
        if bad:
            sys.exit("REFUSING: the job folder holds episode, path or label data: " + "; ".join(bad))
        actions.append({"key": "r6_job_" + job.name, "kind": "dir", "remote": f"{JOBS_REMOTE}/{job.name}",
                        "local": str(job), "bytes": cluster.dir_bytes(job)})
    if args.images or args.image_map:
        actions += image_actions(cfg, args, job, notes)
    for a in actions:
        if not a["remote"].startswith(REMOTE_ROOT) or ".." in Path(a["remote"]).parts:
            sys.exit(f"REFUSING: remote {a['remote']} outside {REMOTE_ROOT}")
    return actions, notes


def copy_images(node, action):
    """cluster.run_data_sync's 'selected' transfer with --copy-links, so the staged links arrive as plain files."""
    cluster.require_remote(cluster.remote(node, 'mkdir -p -- "$DIR"', {"DIR": action["remote"]}, 30),
                           f"create {action['remote']}")
    print(f"data sync: {action['key']} {len(action['files'])} files {action['bytes'] / 1024**2:.0f} MB "
          f"-> {action['remote']}", flush=True)
    with tempfile.NamedTemporaryFile("w", suffix=".files", delete=False) as listing:
        listing.write("\n".join(action["files"]) + "\n")
    try:
        argv = cluster.rsync_argv(node, action, listing.name)
        assert argv[:2] == ["rsync", "-a"], argv
        cluster.require_local(argv[:2] + ["--copy-links"] + argv[2:], f"sync {action['key']}", 6 * 3600)
    finally:
        os.unlink(listing.name)
    return {k: v for k, v in action.items() if k != "files"} | {"files": len(action["files"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", required=True, help="node401, node402 or node408 (each node has its own /local/wding)")
    ap.add_argument("--job-dir", help="a local job folder (r6_gpu_inputs.py)")
    ap.add_argument("--images", action="store_true", help="also copy the images of <job-dir>.image_map.json")
    ap.add_argument("--image-map", help="an image map in place of <job-dir>.image_map.json")
    ap.add_argument("--chunk", type=int, default=None, help="copy only this chunk of the images")
    ap.add_argument("--run", action="store_true", help="perform the copy (default: print the plan only)")
    args = ap.parse_args()
    if not (args.job_dir or args.images or args.image_map):
        ap.error("nothing to copy: give --job-dir and/or --images / --image-map")
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
    done = [copy_images(node, a) if a.get("copy_links") else cluster.run_data_sync(cfg, node, [a])[0]
            for a in actions]
    print("Done:", done)


if __name__ == "__main__":
    main()
