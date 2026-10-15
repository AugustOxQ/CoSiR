"""CPU smoke for ticket 04: joins and frozen split, with per-backbone timings.

Run from any checkout; inputs and src imports resolve to the main checkout.
No scores, fitting, or output files are made.
"""

import argparse
import gc
import hashlib
import json
from time import perf_counter

from r9_backbone import BACKBONES, LEDGER_NAMES, MAIN, STORE_PATHS
from r9_data import load_data, load_split
import src.data.artelingo as artelingo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, help="default: check all three")
    args = parser.parse_args()
    assert artelingo.__file__.startswith(str(MAIN / "src")), artelingo.__file__
    reference_hash = None
    for backbone in (BACKBONES if args.backbone is None else (args.backbone,)):
        start = perf_counter()
        data = load_data(backbone)
        loaded = perf_counter()
        split = load_split(data)
        end = perf_counter()
        digest = hashlib.sha256()
        for name in ("groups", "train", "val", "held", "scorer_train", "selection"):
            digest.update(getattr(split, name).tobytes())
        split_hash = digest.hexdigest()
        if reference_hash is None:
            reference_hash = split_hash
        assert split_hash == reference_hash, "split differs between backbones"
        print(json.dumps({
            "backbone": backbone, "ledger_name": LEDGER_NAMES[backbone],
            "store": str(STORE_PATHS[backbone]), "rows": len(data.sample_ids),
            "dim": data.img_features.shape[1], "dtype": str(data.img_features.dtype),
            "sizes": {name: len(getattr(split, name)) for name in (
                "train", "val", "held", "scorer_train", "selection")},
            "split_arrays_sha256": split_hash,
            "load_s": round(loaded - start, 3), "split_s": round(end - loaded, 3),
            "total_s": round(end - start, 3), "passed": True,
        }), flush=True)
        del data, split
        gc.collect()


if __name__ == "__main__":
    main()
