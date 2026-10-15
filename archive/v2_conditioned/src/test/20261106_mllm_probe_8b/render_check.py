"""Render-equivalence check for the 8B probe: CPU only, processor only (no model weights, no scores). Builds the
seed-46 episodes as run_probe.py does, renders the first 4 episodes of each pair (both conditions, both directions,
the run's permutations) through the reranker's processor path, and writes hashes of everything the model would see.
Run under two transformers versions and compare with compare_render.py.

Run: python src/test/20261106_mllm_probe_8b/render_check.py --out <json>"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/test/20261102_mllm_probe"))

import run_probe as rp                                                                           # noqa: E402
from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo                                  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits                  # noqa: E402
from src.eval.aspect_episodes import AspectEpisodes, concat_episodes, episodes_sha256            # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS                                       # noqa: E402
from src.eval.mllm_reranker import LETTERS, build_messages                                       # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SEED, N, PER_PAIR = 46, 600, 4
LOCAL_NPZ = Path(__file__).parent / "results" / "episodes_seed46.npz"
sha = lambda b: hashlib.sha256(b).hexdigest()


def settings(proc):
    ip = proc.image_processor
    keep = ("size", "min_pixels", "max_pixels", "patch_size", "temporal_patch_size", "merge_size", "do_resize",
            "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb", "resample")
    d = {k: getattr(ip, k) for k in keep if hasattr(ip, k)}
    d["image_processor_class"] = type(ip).__name__
    return json.loads(json.dumps(d, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import transformers
    from transformers import AutoProcessor
    data = load_artelingo()
    splits = artelingo_splits(data)
    labels = artelingo_aspect_labels(data)
    annotations = json.load(open(ANNOTATIONS_PATH))
    with tempfile.TemporaryDirectory() as tmp:                      # build_episodes also writes an npz; keep it out of results/
        parts, hashes = rp.build_episodes(data, splits, labels, N, SEED, Path(tmp))
    checked = False
    if LOCAL_NPZ.exists():
        z = np.load(LOCAL_NPZ)
        for k, (a, b, _) in enumerate(rp.PAIRS):
            e = AspectEpisodes(a, b, **{f.split("__", 2)[2]: z[f] for f in z.files if f.startswith(f"{a}__{b}__")})
            assert episodes_sha256(e) == hashes[f"{a}__{b}"], f"{a}__{b}: episodes differ from {LOCAL_NPZ}"
        checked = True
    ep = concat_episodes(parts)
    path, cap = rp.row_lookups(data, annotations, ep.rows())
    root = str(rp.WIKIART).rstrip("/") + "/"
    rel = {r: (p[len(root):] if p.startswith(root) else p) for r, p in path.items()}
    assert all(not v.startswith("/") for v in rel.values()), "image path outside the WikiArt root"
    perms = np.stack([np.random.default_rng([SEED, i]).permuted(
        np.tile(np.arange(13), (len(CONDITIONS), len(DIRECTIONS), 1)), axis=-1) for i in range(len(ep.anchor))])

    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=rp.MAX_PIXELS)      # as QwenReranker.__init__
    tok = proc.tokenizer
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    img_tok = proc.image_token_id if hasattr(proc, "image_token_id") else tok.convert_tokens_to_ids(proc.image_token)
    prompts, abs_in_ids = {}, 0
    for k in range(len(rp.PAIRS)):
        for j in range(PER_PAIR):
            i = k * N + j
            for ci, cond in enumerate(CONDITIONS):
                for di, d in enumerate(DIRECTIONS):
                    perm = perms[i, ci, di]
                    msgs = rp.make_prompt(build_messages, ep, i, cond, d, perm, path, cap)
                    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt")
                    ids = inp["input_ids"][0].tolist()
                    pv = inp["pixel_values"].to(torch.bfloat16).contiguous().view(torch.int16).numpy()
                    imgs = [c["image"] for c in msgs[0]["content"] if c["type"] == "image"]
                    abs_in_ids += int(str(rp.WIKIART) in tok.decode(ids))
                    before = {}
                    for p in range(1, len(ids) - 1):
                        for li, L in enumerate(LETTERS):
                            if ids[p] == letter_ids[li] and tok.decode([ids[p + 1]]).startswith(".") \
                                    and "\n" in tok.decode([ids[p - 1]]):
                                before.setdefault(L, ids[p - 1])
                    prompts[f"{rp.PAIRS[k][0]}__{rp.PAIRS[k][1]}|ep{j}|{cond}|{d}"] = {
                        "images_rel": [rel[[r for r, v in path.items() if v == im][0]] for im in imgs],
                        "input_ids_sha256": sha(np.asarray(ids, np.int64).tobytes()),
                        "seq_len": len(ids),
                        "n_image_tokens": int(sum(t == img_tok for t in ids)),
                        "image_grid_thw": inp["image_grid_thw"].tolist(),
                        "pixel_values_bf16_sha256": sha(pv.tobytes()),
                        "pixel_values_shape": list(inp["pixel_values"].shape),
                        "letter_prev_token_ids": [before.get(L) for L in LETTERS]}
    out = {"transformers_version": transformers.__version__, "torch_version": torch.__version__, "model": MODEL,
           "max_pixels": rp.MAX_PIXELS, "processor_image_settings": settings(proc),
           "image_token_id": img_tok, "letter_token_ids": letter_ids,
           "letter_token_strings": [tok.decode([t]) for t in letter_ids],
           "episodes_sha256": hashes, "episodes_checked_against_local_npz": checked,
           "absolute_root_in_decoded_prompt": abs_in_ids, "n_prompts": len(prompts), "prompts": prompts}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"transformers {transformers.__version__}: {len(prompts)} prompts, local npz check {checked}, "
          f"absolute root in decoded text: {abs_in_ids}")


if __name__ == "__main__":
    main()
