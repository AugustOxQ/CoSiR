"""Stage 1, last step (rule section 3.3 "Pins"): write pins_a28.json in this folder.

    python src/test/20261202_r9_scope/run_r9_a28_pins.py [--backbones clip_b32 qwen3vl_emb_2b siglip2_so400m]

SHA-256 of translations.parquet and coverage.json of each of the four DAS6 translation shards (shard_0 to shard_3 of
res/r9/a28_translate, the list in the merged folder's meta.json) and of the merged folder nllb200_3.3b_en, of each backbone's ArtELingo-28 image
store, caption store and index.json, and of value_sets_a28.json, language_flags_a28.json and read_set_a28.parquet.
A missing store or file fails before anything is written; an existing pins file is never overwritten. The file is
checked back with check_pins, the function every read runner calls (rule section 6 item 1).
Labels and hashes only: no feature is read and nothing is scored.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_a28_data as D  # noqa: E402
import r9_backbone as B  # noqa: E402

check_pins = D.check_pins                      # the runners import it from here or from r9_a28_data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbones", nargs="+", default=list(B.BACKBONES), choices=B.BACKBONES)
    ap.add_argument("--store-root", type=Path, default=D.STORE_ROOT)
    ap.add_argument("--out", type=Path, default=D.PINS_PATH)
    a = ap.parse_args()
    pins = D.write_pins(a.out, a.store_root, D.results_dir(), tuple(a.backbones))
    got = check_pins(a.out, tuple(a.backbones))
    assert got == {k: v["sha256"] for k, v in pins["files"].items()}
    print(f"passed: {a.out} ({len(got)} files, backbones {pins['backbones']})")


if __name__ == "__main__":
    main()
