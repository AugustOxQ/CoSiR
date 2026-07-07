"""
Build a uniform-random 150K subsample of RedCaps-medium for the buddy run.

The source file is grouped by subreddit, so a prefix slice would be badly biased.
We sample uniformly at random (seed 42), preserving natural subreddit frequencies.
"""
import json
import os
import random
from collections import Counter

SRC = "/data/PDD/redcaps/redcaps_plus/redcaps_medium.json"
OUT = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"
N = 150_000
SEED = 42


def subreddit(rec) -> str:
    parts = rec["image"].split("/")
    return parts[2] if len(parts) > 2 else "?"


def main():
    print(f"Loading {SRC} …")
    data = json.load(open(SRC))
    print(f"  {len(data):,} records")

    random.seed(SEED)
    idx = random.sample(range(len(data)), N)
    idx.sort()
    sub = [data[i] for i in idx]

    subs = Counter(subreddit(r) for r in sub)
    print(f"Sampled {len(sub):,} records across {len(subs)} subreddits")
    print("  top 10:", subs.most_common(10))

    json.dump(sub, open(OUT, "w"))
    print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
