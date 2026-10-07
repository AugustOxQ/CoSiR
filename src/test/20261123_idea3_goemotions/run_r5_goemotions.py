"""Round 5 GoEmotions runner (rule DECISION_RULE.md of this folder: D2, §8 step 3). Run by the main session only.
    flock -n -o -E 75 /tmp/gpu0.lock env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        PYTHONDONTWRITEBYTECODE=1 <python> run_r5_goemotions.py --device cuda
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 <python> \
        run_r5_goemotions.py --device cpu
Prints the pass or fail of item 2, counts and the SHA-256 of the written files; never a probability.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r5_common as R5  # noqa: E402
import r5_goemo as G  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = ap.parse_args(argv)

    R5.assert_rule()
    R5.assert_modules()
    shas = R5.assert_inputs(G.INPUT_NAMES)
    G.assert_snapshot_ref()
    G.assert_annotation_source()
    R5.refuse_existing([R5.CACHE / G.NPZ_NAME, R5.CACHE / G.JSON_NAME, R5.CACHE / G.FAIL_NAME], smoke=False)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda given but CUDA is not available")
    if args.device == "cpu" and torch.cuda.is_available():
        raise SystemExit("--device cpu given but CUDA is visible; launch with CUDA_VISIBLE_DEVICES=")
    loaded = G.load_goemotions(device="cuda:0" if args.device == "cuda" else "cpu")
    G.assert_device(loaded, args.device)

    ctx = R5.RG.EvalContext(42, False)
    file_shas = {Path(k).name: v for k, v in shas.items() if k.startswith(R5.SNAP)}
    rec = G.run(args.device, loaded, R5.CACHE, ctx, file_shas=file_shas)

    item2 = rec["item2"]
    print(f"item 2: {'PASS' if item2['passed'] else 'FAIL'} (sample {item2['sample_size']} captions)")
    if not item2["passed"]:
        print(f"no selection caption was passed to the model; failure record {R5.CACHE / G.FAIL_NAME}")
        print(f"sha256 {G.FAIL_NAME} {R5.sha256_file(R5.CACHE / G.FAIL_NAME)}")
        raise SystemExit(1)
    print(f"selection rows {rec['n_rows']}, captions {rec['n_captions']}, device {rec['device']}")
    print(f"captions over 64 tokens {rec['n_captions_over_64_tokens']}, largest token count {rec['max_token_count']}")
    for name in (G.NPZ_NAME, G.JSON_NAME):
        print(f"sha256 {name} {R5.sha256_file(R5.CACHE / name)}")


if __name__ == "__main__":
    main()
