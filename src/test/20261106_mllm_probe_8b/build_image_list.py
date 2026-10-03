"""Write scripts/mllm_probe_8b_images.txt: the WikiArt images (relative to run_probe.WIKIART) of every row the
seed-46, 600-per-pair probe episodes use, built with run_probe's own build_episodes + row_lookups. Cross-checks each
pair's episodes SHA-256 against the locally saved results/episodes_seed46.npz (only that file is read).

Run (CPU only): CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python \
    src/test/20261106_mllm_probe_8b/build_image_list.py <scratch dir for the rebuilt episodes npz>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEED, N = 46, 600
OUT = ROOT / "scripts" / "mllm_probe_8b_images.txt"

spec = importlib.util.spec_from_file_location("run_probe", ROOT / "src/test/20261102_mllm_probe/run_probe.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
from src.eval.aspect_episodes import AspectEpisodes, episodes_sha256   # noqa: E402  (run_probe put ROOT on sys.path)


def main():
    scratch = Path(sys.argv[1])
    scratch.mkdir(parents=True, exist_ok=True)
    assert str(rp.WIKIART) == "/data/PDD/wikiart_proj/wikiart", "build the list against the local default paths"
    data = rp.load_artelingo()
    splits = rp.artelingo_splits(data)
    labels = rp.artelingo_aspect_labels(data)
    annotations = json.load(open(rp.ANNOTATIONS_PATH))
    parts, hashes = rp.build_episodes(data, splits, labels, N, SEED, scratch)

    saved = np.load(HERE / "results" / f"episodes_seed{SEED}.npz")
    assert [str(x) for x in saved["pair_order"]] == list(hashes), "pair order differs from the saved episodes"
    for a, b, _ in rp.PAIRS:
        key = f"{a}__{b}"
        old = AspectEpisodes(a, b, *(saved[f"{key}__{f}"] for f in
                                     ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")))
        assert episodes_sha256(old) == hashes[key], f"{key}: rebuilt episodes differ from the saved ones"
        print(f"{key}: sha256 {hashes[key]} (matches results/episodes_seed{SEED}.npz)")

    ep = rp.concat_episodes(parts)
    path, _ = rp.row_lookups(data, annotations, ep.rows())
    rel = sorted({str(Path(p).relative_to(rp.WIKIART)) for p in path.values()})
    missing = [r for r in rel if not (rp.WIKIART / r).is_file()]
    assert not missing, f"{len(missing)} images missing locally, e.g. {missing[:3]}"
    size = sum((rp.WIKIART / r).stat().st_size for r in rel)
    OUT.write_text("\n".join(rel) + "\n")
    print(f"{len(path)} rows, {len(rel)} unique images, {size} bytes ({size / 1024**3:.3f} GB) -> {OUT}")


if __name__ == "__main__":
    main()
