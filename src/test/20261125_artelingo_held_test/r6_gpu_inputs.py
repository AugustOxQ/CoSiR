"""Round 6 GPU job inputs, CPU side (DECISION_RULE.md of this folder: section 7 item 1, section 8 item 3; contracts
section 9): job folders that hold row ids, neutral image names and captions only, and, for the verbaliser, the job
input built from a full episodes file (r6_episodes.save_episodes, CPU-internal).

Neutral image names. A WikiArt path begins with the style folder (a label), so no shipped file and no file name on a
node carries it: each image is named <first 20 hex of SHA-256 of its WikiArt path>.<extension> (the same name for the
same image in every job). The name -> path map is a CPU-only file next to the job folder, <job>.image_map.json, never
shipped. The images are staged here as symlinks <staging>/<name> -> <WikiArt root>/<path> (default
results/gpu_images/ of this folder; the local GPU reads that folder), and scripts/das6_sync_r6.py copies the files
they point to into /local/wding/r6_jobs/images/<name> on a node.

A job folder (write_job_folder; written whole into <out>.partial, then renamed; nothing is ever overwritten):
- rows_manifest.npz: `rows` (int64, sorted unique), `image_name`, `caption`. Built through the C6 join: row ->
  data.sample_ids[row] -> annotations[sample id]'s `image` and `caption`, the only two annotation fields read, and
  asserted against the data's own join (each image file's stem is data.paintings[row]). No path, no sample id.
- the job's own input files (npz of row ids), e.g. verbalise_input.npz: `seed`, `episode_index` (n,) (positions in
  the seed's concatenated episodes), `pairs_a_img`, `pairs_a_txt`, `pairs_b_img`, `pairs_b_txt` (n, 4). No anchor,
  candidates, pair names or labels.
- images.txt: the neutral names of the images the job opens (for the verbaliser, those of the pairs' image rows).
- job_record.json: SHA-256 of each file, counts, the job's own record fields, module SHA-256s and the time.

Run (verbaliser):
    python r6_gpu_inputs.py --episodes <episodes npz> --out <job folder> [--first-per-pair 1024]
                            [--image-staging <folder>] [--image-root <WikiArt root>]
--first-per-pair N keeps the first N episodes of each pair (rule section 7 item 4's tuning subset: 1,024).

Guards carry a `# guard:<name>` marker; the tests delete each on a copy and show that its scenario then passes.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (before anything that imports src)
import r6_episodes as E  # noqa: E402
import r6_gpu_common as G  # noqa: E402

import numpy as np  # noqa: E402

WIKIART = Path("/data/PDD/wikiart_proj/wikiart")
IMAGE_STAGING = R.RESULTS / "gpu_images"
MANIFEST, IMAGES_TXT, RECORD = "rows_manifest.npz", "images.txt", "job_record.json"
JOB_FILES = (MANIFEST, "verbalise_input.npz", IMAGES_TXT, RECORD)


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def image_name(relpath) -> str:
    """The neutral file name of a WikiArt image: <first 20 hex of SHA-256 of its path>.<lower-case extension>."""
    ext = Path(relpath).suffix.lower().lstrip(".")
    _require(re.fullmatch(r"[a-z0-9]{1,5}", ext) is not None, f"{relpath!r}: no plain file extension")
    return f"{hashlib.sha256(str(relpath).encode('utf-8')).hexdigest()[:20]}.{ext}"


def image_map_path(job) -> Path:
    """The CPU-only name -> WikiArt path map of a job folder: <job>.image_map.json beside it (never shipped)."""
    job = Path(job)
    return job.with_name(job.name + ".image_map.json")


# ---------------------------------------------------------------- the manifest (C6) and the staged images

def build_manifest(rows, sample_ids, paintings, annotations):
    """rows -> sample id (C6: data.sample_ids[row]) -> annotations[sample id]'s image path and caption.
    -> (manifest {rows, image_name, caption}, the WikiArt paths aligned with rows; the paths stay on the CPU)."""
    rows = np.asarray(rows, dtype=np.int64)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    _require(rows.ndim == 1 and rows.size and bool((np.diff(rows) > 0).all()) and rows[0] >= 0
             and rows[-1] < R.N_ROWS, "rows must be sorted unique row positions")
    _require(len(sample_ids) == len(paintings) == len(annotations) == R.N_ROWS
             and len(np.unique(sample_ids)) == R.N_ROWS and sample_ids.min() >= 0
             and sample_ids.max() < len(annotations),
             "sample ids must uniquely index the annotation list (C6)")  # guard:sample_ids
    sid = sample_ids[rows]
    paths, caption = [], []
    for s in sid.tolist():
        a = annotations[s]
        paths.append(a["image"])
        caption.append(a["caption"])
    _require(all(isinstance(c, str) and c.strip() for c in caption), "every caption must be a non-empty string")
    stems = np.asarray([Path(i).stem for i in paths])
    bad = int((stems != np.asarray(paintings)[rows]).sum())
    _require(bad == 0, f"{bad} rows: the image file is not the data's painting for that row (C6 join)")  # guard:c6_join
    names = [image_name(p) for p in paths]
    _require(len(set(zip(names, paths))) == len(set(names)) == len(set(paths)), "two images share a neutral name")
    man = {"rows": rows, "image_name": np.asarray(names, dtype=str), "caption": np.asarray(caption, dtype=str)}
    _require(man["caption"].tolist() == caption and man["image_name"].tolist() == names,
             "a caption or name changed when stored as a unicode array")
    return man, paths


def stage_images(images: dict, staging, image_root=WIKIART) -> int:
    """Symlinks <staging>/<name> -> <image_root>/<path> for {name: path}; an existing entry must be the same link."""
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    for name, rel in sorted(images.items()):
        target = Path(image_root) / rel
        _require(target.is_file(), f"image missing: {target}")
        link = staging / name
        if link.is_symlink() or link.exists():
            _require(link.is_symlink() and os.readlink(link) == str(target),
                     f"{link} exists and is not the link to {target}")  # guard:staging_target
        else:
            os.symlink(target, link)
    return len(images)


def write_job_folder(out, rows, image_rows, sample_ids, paintings, annotations, arrays: dict, record: dict,
                     staging=None, image_root=WIKIART, check=None) -> dict:
    """A GPU job folder ``out`` for any job: rows_manifest.npz for ``rows``, one npz per entry of ``arrays``
    ({file name: {key: array}}), images.txt (the neutral names of the images of ``image_rows``, the ones the job
    opens) and job_record.json (``record`` plus counts, file and module SHA-256s, time). Beside it, never shipped,
    <out>.image_map.json ({name: WikiArt path} of those images, the image root, the staging folder). The images are
    staged as symlinks in ``staging``. ``check(folder)`` runs on the written folder before it is renamed into place."""
    out = Path(out)
    tmp, mp = out.with_name(out.name + ".partial"), image_map_path(out)
    mp_tmp = mp.with_name(mp.name + ".partial")
    _require(not any(p.exists() for p in (out, tmp, mp, mp_tmp)),
             f"{out} or its image map (or a .partial) exists; job folders are never overwritten")  # guard:no_overwrite
    _require(all(f.endswith(".npz") and f != MANIFEST for f in arrays), "job input files must be npz files")
    rows = np.asarray(rows, dtype=np.int64)
    image_rows = np.unique(np.asarray(image_rows, dtype=np.int64))
    _require(bool(np.isin(image_rows, rows).all()), "every image row must be a manifest row")
    man, paths = build_manifest(rows, sample_ids, paintings, annotations)
    pos = np.searchsorted(rows, image_rows)
    shown = {str(man["image_name"][i]): paths[i] for i in pos.tolist()}
    staging = Path(staging or IMAGE_STAGING).resolve()
    tmp.mkdir(parents=True)
    np.savez_compressed(tmp / MANIFEST, **man)
    for f, d in arrays.items():
        np.savez(tmp / f, **d)
    (tmp / IMAGES_TXT).write_text("\n".join(sorted(shown)) + "\n", encoding="utf-8")
    G.Manifest(tmp / MANIFEST)                                        # reads back with the GPU job's own loader
    if check is not None:
        check(tmp)
    stage_images(shown, staging, image_root)
    rec = dict(record, n_rows=int(len(rows)), n_images=len(shown),
               files_sha256={p.name: G.sha256_file(p) for p in sorted(tmp.iterdir())},
               module_sha256=R.r6_module_shas(), time=R.amsterdam_now())
    G.write_json(tmp / RECORD, rec)
    G.write_json(mp_tmp, {"what": "CPU-only map of a round-6 GPU job folder; never shipped to a GPU job",
                          "job": out.name, "image_root": str(image_root), "staging": str(staging),
                          "images": dict(sorted(shown.items()))})
    os.replace(tmp, out)
    os.replace(mp_tmp, mp)
    return rec


# ---------------------------------------------------------------- the verbaliser's job

def select_episodes(eps, first_per_pair=None) -> np.ndarray:
    """Positions (in the seed's concatenated episodes) of every episode, or of the first N of each pair."""
    if first_per_pair is None:
        return np.arange(eps.n, dtype=np.int64)
    n = int(first_per_pair)
    out = []
    for k in range(len(R.PAIRS)):
        pos = np.flatnonzero(eps.pair_index == k)
        _require(0 < n <= len(pos), f"--first-per-pair {n}: pair {k} has {len(pos)} episodes")
        out.append(pos[:n])
    return np.concatenate(out).astype(np.int64)


def verbalise_arrays(eps, positions) -> dict:
    """The contract's verbalise_input arrays: the example pairs of the episodes at ``positions``, nothing else."""
    positions = np.asarray(positions, dtype=np.int64)
    out = {"seed": np.int64(eps.seed), "episode_index": positions}
    for f in G.PAIR_FIELDS:
        out[f] = np.ascontiguousarray(getattr(eps.pooled, f)[positions], dtype=np.int64)
    return out


def write_job(eps, sample_ids, paintings, annotations, out, first_per_pair=None, episodes_file_sha256=None,
              staging=None, image_root=WIKIART) -> dict:
    """The verbaliser's job folder (rows_manifest.npz, verbalise_input.npz, images.txt, job_record.json) and its
    CPU-only image map; -> job_record."""
    positions = select_episodes(eps, first_per_pair)
    vin = verbalise_arrays(eps, positions)
    rows = np.unique(np.concatenate([vin[f].ravel() for f in G.PAIR_FIELDS]))
    img_rows = np.unique(np.concatenate([vin[f].ravel() for f in ("pairs_a_img", "pairs_b_img")]))

    def check(folder):
        back = G.load_verbalise_input(folder / "verbalise_input.npz", G.Manifest(folder / MANIFEST))
        _require(all(np.array_equal(back[f], vin[f]) for f in ("episode_index", *G.PAIR_FIELDS))
                 and back["seed"] == int(eps.seed), "the job input does not read back as written")

    record = {"what": "round 6 verbaliser job inputs (r6_gpu_inputs.py)", "seed": int(eps.seed),
              "n_episodes": int(len(positions)), "first_per_pair": first_per_pair,
              "episodes_file_sha256": episodes_file_sha256,
              "episodes_sha256_in_pair_order": [eps.sha[p] for p in R.PAIR_NAMES]}
    return write_job_folder(out, rows, img_rows, sample_ids, paintings, annotations, {"verbalise_input.npz": vin},
                            record, staging, image_root, check)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Round 6 verbaliser job inputs (CPU); prints no metric")
    ap.add_argument("--episodes", required=True, help="a full episodes file (r6_episodes.save_episodes)")
    ap.add_argument("--out", required=True, help="the job folder to create")
    ap.add_argument("--first-per-pair", type=int, default=None, help="keep the first N episodes of each pair")
    ap.add_argument("--image-staging", default=str(IMAGE_STAGING), help="the local folder of neutral image links")
    ap.add_argument("--image-root", default=str(WIKIART), help="the WikiArt root")
    args = ap.parse_args(argv)
    from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo
    eps = E.load_episodes(args.episodes)
    data = load_artelingo()
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        annotations = json.load(f)
    rec = write_job(eps, data.sample_ids, data.paintings, annotations, args.out, args.first_per_pair,
                    G.sha256_file(args.episodes), args.image_staging, args.image_root)
    print(f"job folder {args.out}: seed {rec['seed']}, {rec['n_episodes']} episodes, {rec['n_rows']} rows, "
          f"{rec['n_images']} images staged in {args.image_staging}; image map {image_map_path(args.out)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
