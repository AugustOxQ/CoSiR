"""
Build a uniform-random N-sample of RedCaps-medium for the buddy run.

The source file is grouped by subreddit, so a prefix slice would be badly biased.
We sample uniformly at random (seed 42 by default), preserving natural subreddit
frequencies.

Usage
-----
  python build_subsample.py                                    # original 150k invocation
  python build_subsample.py --n 300000 --out .../redcaps_300k_diverse.json
  python build_subsample.py --n 500000 --out .../redcaps_500k_diverse.json

Refuses to overwrite an existing OUT file unless --force is given (protects
redcaps_150k.json, already relied on by several other experiments, from accidental
regeneration/overwrite).
"""
import argparse
import json
import os
import random
from collections import Counter

SRC = "/data/PDD/redcaps/redcaps_plus/redcaps_medium.json"
DEFAULT_OUT = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"
DEFAULT_N = 150_000
SEED = 42


def subreddit(rec) -> str:
    parts = rec["image"].split("/")
    return parts[2] if len(parts) > 2 else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite --out if it already exists")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} already exists — nothing to do (pass --force to overwrite).")
        return

    print(f"Loading {args.src} …")
    data = json.load(open(args.src))
    print(f"  {len(data):,} records")

    random.seed(args.seed)
    idx = random.sample(range(len(data)), args.n)
    idx.sort()
    sub = [data[i] for i in idx]

    subs = Counter(subreddit(r) for r in sub)
    print(f"Sampled {len(sub):,} records across {len(subs)} subreddits (seed={args.seed})")
    print("  top 10:", subs.most_common(10))

    json.dump(sub, open(args.out, "w"))
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
