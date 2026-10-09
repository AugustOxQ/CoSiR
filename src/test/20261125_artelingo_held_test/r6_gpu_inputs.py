"""Round 6 GPU job inputs, CPU side (DECISION_RULE.md of this folder: section 7 item 1, section 8 item 3; contracts
section 9): from a full episodes file (r6_episodes.save_episodes, CPU-internal) and the data, a job folder for the
verbaliser holding only row ids, image paths and captions.

The folder (written whole into <out>.partial, then renamed to <out>; an existing folder is never overwritten):
- rows_manifest.npz: `rows` (int64, sorted unique: every row of the selected episodes' example pairs), `sample_id` =
  data.sample_ids[rows] (C6), and `image_relpath`, `caption` read from annotations[sample_id], the only two annotation
  fields read. The join is asserted against the data's own: each image file's stem is data.paintings[row].
- verbalise_input.npz: `seed`, `episode_index` (n,) (positions in the seed's concatenated episodes), `pairs_a_img`,
  `pairs_a_txt`, `pairs_b_img`, `pairs_b_txt` (n, 4) row ids. No anchor, candidates, pair names or labels.
- images.txt: the sorted unique image paths, relative to the WikiArt root, for scripts/das6_sync_r6.py.
- job_record.json: SHA-256 of each file and of the source episodes file, the seed, counts, `first_per_pair`, module
  SHA-256s and the time (no aspect name, label or path).
`image_relpath` begins with WikiArt's style folder (the tree's layout). The job uses it only to open the file; the
processor turns it into pixels and the path never enters the prompt text.

Run:
    python r6_gpu_inputs.py --episodes <episodes npz> --out <job folder> [--first-per-pair 1024]
--first-per-pair N keeps the first N episodes of each pair (rule section 7 item 4's tuning subset: 1,024).

Guards carry a `# guard:<name>` marker; the tests delete each on a copy and show that its scenario then passes.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (before anything that imports src)
import r6_episodes as E  # noqa: E402
import r6_gpu_common as G  # noqa: E402

import numpy as np  # noqa: E402

JOB_FILES = ("rows_manifest.npz", "verbalise_input.npz", "images.txt", "job_record.json")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


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


def build_manifest(rows, sample_ids, paintings, annotations) -> dict:
    """rows -> sample id (C6: data.sample_ids[row]) -> annotations[sample id]'s image path and caption."""
    rows = np.asarray(rows, dtype=np.int64)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    _require(rows.ndim == 1 and rows.size and bool((np.diff(rows) > 0).all()) and rows[0] >= 0
             and rows[-1] < R.N_ROWS, "rows must be sorted unique row positions")
    _require(len(sample_ids) == len(paintings) == len(annotations) == R.N_ROWS
             and len(np.unique(sample_ids)) == R.N_ROWS and sample_ids.min() >= 0
             and sample_ids.max() < len(annotations),
             "sample ids must uniquely index the annotation list (C6)")  # guard:sample_ids
    sid = sample_ids[rows]
    image, caption = [], []
    for s in sid.tolist():
        a = annotations[s]
        image.append(a["image"])
        caption.append(a["caption"])
    _require(all(isinstance(c, str) and c.strip() for c in caption), "every caption must be a non-empty string")
    stems = np.asarray([Path(i).stem for i in image])
    bad = int((stems != np.asarray(paintings)[rows]).sum())
    _require(bad == 0, f"{bad} rows: the image file is not the data's painting for that row (C6 join)")  # guard:c6_join
    man = {"rows": rows, "sample_id": sid, "image_relpath": np.asarray(image, dtype=str),
           "caption": np.asarray(caption, dtype=str)}
    _require(man["caption"].tolist() == caption and man["image_relpath"].tolist() == image,
             "a caption or path changed when stored as a unicode array")
    return man


def write_job(eps, sample_ids, paintings, annotations, out, first_per_pair=None, episodes_file_sha256=None) -> dict:
    """Write the verbaliser job folder ``out`` (see the module docstring) and read it back with the GPU job's own
    loaders; -> job_record."""
    out = Path(out)
    tmp = out.with_name(out.name + ".partial")
    _require(not out.exists() and not tmp.exists(), f"{out} (or {tmp.name}) exists; job folders are never "
                                                    "overwritten")  # guard:no_overwrite
    positions = select_episodes(eps, first_per_pair)
    vin = verbalise_arrays(eps, positions)
    rows = np.unique(np.concatenate([vin[f].ravel() for f in G.PAIR_FIELDS]))
    man = build_manifest(rows, sample_ids, paintings, annotations)
    tmp.mkdir(parents=True)
    np.savez_compressed(tmp / "rows_manifest.npz", **man)
    np.savez(tmp / "verbalise_input.npz", **vin)
    images = sorted(set(man["image_relpath"].tolist()))
    (tmp / "images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    manifest = G.Manifest(tmp / "rows_manifest.npz")
    back = G.load_verbalise_input(tmp / "verbalise_input.npz", manifest)
    _require(all(np.array_equal(back[f], vin[f]) for f in ("episode_index", *G.PAIR_FIELDS))
             and back["seed"] == int(eps.seed), "the job input does not read back as written")
    rec = {"what": "round 6 verbaliser job inputs (r6_gpu_inputs.py)", "seed": int(eps.seed),
           "n_episodes": int(len(positions)), "first_per_pair": first_per_pair, "n_rows": int(len(rows)),
           "n_images": len(images), "episodes_file_sha256": episodes_file_sha256,
           "episodes_sha256_in_pair_order": [eps.sha[p] for p in R.PAIR_NAMES],
           "files_sha256": {f: G.sha256_file(tmp / f) for f in JOB_FILES[:3]},
           "module_sha256": R.r6_module_shas(), "time": R.amsterdam_now()}
    G.write_json(tmp / "job_record.json", rec)
    os.replace(tmp, out)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Round 6 verbaliser job inputs (CPU); prints no metric")
    ap.add_argument("--episodes", required=True, help="a full episodes file (r6_episodes.save_episodes)")
    ap.add_argument("--out", required=True, help="the job folder to create")
    ap.add_argument("--first-per-pair", type=int, default=None, help="keep the first N episodes of each pair")
    args = ap.parse_args(argv)
    from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo
    eps = E.load_episodes(args.episodes)
    data = load_artelingo()
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        annotations = json.load(f)
    rec = write_job(eps, data.sample_ids, data.paintings, annotations, args.out, args.first_per_pair,
                    G.sha256_file(args.episodes))
    print(f"job folder {args.out}: seed {rec['seed']}, {rec['n_episodes']} episodes, {rec['n_rows']} rows, "
          f"{rec['n_images']} images", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
