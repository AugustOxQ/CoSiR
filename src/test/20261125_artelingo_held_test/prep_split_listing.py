"""Round 6 run chat (r6 run, 2026-10-10): split one listing job folder (run_r6_dts.py --stage list-input) into N
shard folders, so the listing runs on several GPUs. Not an r6 module (not hashed by r6_common.r6_module_shas).

Each shard gets a contiguous slice of the parent's listing_input.jsonl, byte for byte, with every slice boundary at a
multiple of the listing batch size (16, dts_settings.json generation.listing.batch_size): r6_gpu_listing.py batches
its input in order, so each shard's batches are exactly the batches the unsplit job would run (same prompts, same
order, same left padding). The DTS stages merge listing folders by (phrase, K) (r6_dts.merge_listings). Each shard
folder gets split_record.json (parent, parent listing_input SHA-256, line range, its own SHA-256). Refuses existing
shard folders; writes nothing else.

    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/prep_split_listing.py <job folder> <N>
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = json.loads((HERE / "dts_settings.json").read_text(encoding="utf-8"))["generation"]["listing"]["batch_size"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(job: str, n: int) -> int:
    job = Path(job)
    raw = (job / "listing_input.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    batches = -(-len(lines) // BATCH)
    per = -(-batches // n)
    bounds = [min(i * per * BATCH, len(lines)) for i in range(n + 1)]
    outs = [job.with_name(f"{job.name}_part{i}") for i in range(n)]
    for o in outs:
        if o.exists():
            print(f"refused: {o} exists")
            return 4
    for i, o in enumerate(outs):
        a, b = bounds[i], bounds[i + 1]
        if a >= b:
            continue
        o.mkdir()
        body = b"".join(lines[a:b])
        (o / "listing_input.jsonl").write_bytes(body)
        (o / "split_record.json").write_text(json.dumps(
            {"parent": job.name, "parent_listing_input_sha256": sha(raw), "lines": [a, b], "n_parent_lines": len(lines),
             "batch_size": BATCH, "listing_input_sha256": sha(body)}, indent=1) + "\n", encoding="utf-8")
        print(f"{o.name}: lines [{a}, {b})")
    assert b"".join(lines[bounds[0]:bounds[-1]]) == raw
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
