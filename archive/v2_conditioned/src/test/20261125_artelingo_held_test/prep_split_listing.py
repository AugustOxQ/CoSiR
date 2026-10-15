"""Round 6 run chat (r6 run, 2026-10-10): split one listing job folder (run_r6_dts.py --stage list-input) into N
shard folders, so the listing runs on several GPUs. Not an r6 module (not hashed by r6_common.r6_module_shas).

Each shard gets a contiguous slice of the parent's listing_input.jsonl, byte for byte. r6_gpu_listing.py copies the
cached (phrase, K) items first and batches the rest in input order, so every slice boundary is put where the number of
uncached items before it is a multiple of the listing batch size (16, dts_settings.json
generation.listing.batch_size): each shard's batches are exactly the batches the unsplit job would run with the same
caches (same prompts, same order, same left padding). Without --cache every item counts, as before (r6 run, the
tuning listing). Each --cache NAME=FOLDER (an earlier listing output folder: listings.jsonl, provenance.json) is
copied into every shard's cache/NAME/; the listing job checks its fingerprint. The DTS stages merge listing folders by
(phrase, K) (r6_dts.merge_listings). Each shard folder gets split_record.json (parent, parent listing_input SHA-256,
line range, uncached range, the caches' SHA-256s, its own SHA-256). Refuses existing shard folders; writes nothing
else.

    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/prep_split_listing.py <job folder> <N>
        [--cache NAME=FOLDER]...
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = json.loads((HERE / "dts_settings.json").read_text(encoding="utf-8"))["generation"]["listing"]["batch_size"]
CACHE_FILES = ("listings.jsonl", "provenance.json")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def key(line: bytes) -> tuple:
    rec = json.loads(line)
    return rec["phrase"], rec["K"]


def main(job: str, n: int, caches: list) -> int:
    job = Path(job)
    raw = (job / "listing_input.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    named = []
    for c in caches:
        name, _, folder = c.partition("=")
        folder = Path(folder)
        if not name or any(not (folder / f).is_file() for f in CACHE_FILES):
            print(f"refused: --cache {c} is not NAME=FOLDER with {', '.join(CACHE_FILES)}")
            return 2
        named.append((name, folder))
    if len({nm for nm, _ in named}) != len(named):
        print("refused: repeated cache name")
        return 2
    cached = set()
    for _, folder in named:
        cached.update(key(x) for x in (folder / "listings.jsonl").read_bytes().splitlines() if x.strip())
    todo = [i for i, x in enumerate(lines) if key(x) not in cached]
    batches = -(-len(todo) // BATCH)
    per = -(-batches // n) if batches else 0
    tb = [min(i * per * BATCH, len(todo)) for i in range(n + 1)] if per else [0] + [len(todo)] * n
    bounds = [0] + [todo[t] if t < len(todo) else len(lines) for t in tb[1:-1]] + [len(lines)]
    outs = [job.with_name(f"{job.name}_part{i}") for i in range(n)]
    for o in outs:
        if o.exists():
            print(f"refused: {o} exists")
            return 4
    cache_sha = {nm: sha((folder / "listings.jsonl").read_bytes()) for nm, folder in named}
    for i, o in enumerate(outs):
        a, b = bounds[i], bounds[i + 1]
        if a >= b:
            continue
        o.mkdir()
        body = b"".join(lines[a:b])
        (o / "listing_input.jsonl").write_bytes(body)
        for nm, folder in named:
            (o / "cache" / nm).mkdir(parents=True)
            for f in CACHE_FILES:
                shutil.copyfile(folder / f, o / "cache" / nm / f)
        (o / "split_record.json").write_text(json.dumps(
            {"parent": job.name, "parent_listing_input_sha256": sha(raw), "lines": [a, b], "n_parent_lines": len(lines),
             "batch_size": BATCH, "uncached": [tb[i], tb[i + 1]], "n_parent_uncached": len(todo),
             "caches": {nm: {"folder": str(folder), "listings_sha256": cache_sha[nm]} for nm, folder in named},
             "listing_input_sha256": sha(body)}, indent=1) + "\n", encoding="utf-8")
        print(f"{o.name}: lines [{a}, {b}), uncached [{tb[i]}, {tb[i + 1]})")
    assert b"".join(lines[bounds[0]:bounds[-1]]) == raw
    assert all(t % BATCH == 0 for t in tb if t < len(todo))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("n", type=int)
    ap.add_argument("--cache", action="append", default=[], help="NAME=FOLDER, an earlier listing output (repeatable)")
    args = ap.parse_args()
    sys.exit(main(args.job, args.n, args.cache))
