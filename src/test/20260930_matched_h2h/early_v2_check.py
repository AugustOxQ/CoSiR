"""Early V2 check (controller, before Task 6): does the buddy pilot port
(`h2h_buddy.fit_buddy_stage1`, impl="pilot", default config, full held-out
monitor) reproduce the unchanged pilot's snapshot on the same DAS6 GPU?

Compares against the QC2 snapshot the pilot itself wrote on this node
(`--snapshot`), both the raw embeddings and the independent-re-clustering
AMIs / community counts.

Usage (DAS6 via scripts/run_h2h_early_v2.sh):
    python early_v2_check.py --seed 42 --snapshot /local/wding/jobs/qc2-s42/code/src/test/20260930_harness_confirmation/snapshots/seed42/snapshot_seed42.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from scripts.buddy_percept_sweep.h2h_buddy import BuddyStage1Config, fit_buddy_stage1
from scripts.buddy_percept_sweep.h2h_store import load_or_build_store
from scripts.buddy_percept_sweep.pilot_metrics import (
    ami_emotion_genre, independent_partition, load_pilot_modules,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = load_or_build_store()
    pilot = load_pilot_modules()
    start = time.monotonic()
    out = fit_buddy_stage1(BuddyStage1Config(), store, args.seed,
                           np.arange(len(store.heldout_paintings)), pilot, device)
    fit_seconds = time.monotonic() - start
    heldout_labels = independent_partition(out.heldout_embedding, pilot, "heldout", args.seed, device)
    train_labels = independent_partition(out.train_embedding, pilot, "train", args.seed, device)
    emotion, genre = ami_emotion_genre(heldout_labels, store.heldout_emotion, store.heldout_genre)

    with np.load(args.snapshot, allow_pickle=True) as snap:
        pilot_heldout = snap["heldout_embedding_post"]
        pilot_train = snap["train_embedding_post"]
        pilot_communities = snap["heldout_community_post"]
        same_paintings = bool(np.array_equal(snap["heldout_paintings"], store.heldout_paintings))
    pilot_emotion, pilot_genre = ami_emotion_genre(pilot_communities, store.heldout_emotion, store.heldout_genre)

    result = {
        "seed": args.seed,
        "stop_reason": out.info.get("stop_reason"),
        "epochs_run": out.info.get("epochs_run"),
        "fit_seconds": round(fit_seconds, 1),
        "same_heldout_painting_order": same_paintings,
        "heldout_embedding_bit_equal": bool(np.array_equal(out.heldout_embedding, pilot_heldout)),
        "train_embedding_bit_equal": bool(np.array_equal(out.train_embedding, pilot_train)),
        "heldout_embedding_max_abs_diff": float(np.max(np.abs(out.heldout_embedding - pilot_heldout))),
        "port": {"ind_emo": emotion, "ind_genre": genre,
                 "ind_k": int(len(np.unique(heldout_labels))), "train_k": int(len(np.unique(train_labels)))},
        "pilot": {"ind_emo": pilot_emotion, "ind_genre": pilot_genre,
                  "ind_k": int(len(np.unique(pilot_communities)))},
    }
    print("EARLY_V2 " + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
