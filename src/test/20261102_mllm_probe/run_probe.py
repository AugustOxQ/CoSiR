"""Early MLLM probe (CVPR plan Task 14, spec §6): does an in-context Qwen3-VL-2B reranker beat plain CLIP cosine on
seed-44 selection-row aspect episodes? Pre-registered rule: the MLLM works iff compare(mllm, cosine) has a 95% CI
lower bound > 0 on both r1 and gain.

Run: flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python src/test/20261102_mllm_probe/run_probe.py
         [--n 300] [--seed 44] [--model <hf id>] [--out <dir>]
Resumes from <out>/probe_partial.npz (checkpoint every 50 episodes)."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.artelingo import ANNOTATIONS_PATH, join_captions, load_artelingo          # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits         # noqa: E402
from src.eval.aspect_episodes import (PaintingValueIndex, build_aspect_episodes, concat_episodes,   # noqa: E402
                                      episodes_sha256, validate_aspect_episodes)
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, compare, per_anchor, summarize           # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores                            # noqa: E402

WIKIART = Path(os.environ.get("COSIR_WIKIART_DIR") or "/data/PDD/wikiart_proj/wikiart")   # env: cluster runs
PAIRS = (("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion"))
CHECKPOINT_EVERY = 50
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MAX_PIXELS = 256 * 28 * 28


def build_episodes(data, splits, labels, n, seed, out_dir):
    index = PaintingValueIndex(labels, splits.groups)
    parts, saved, hashes = [], {}, {}
    for a, b, third in PAIRS:
        ep = build_aspect_episodes(labels, splits.groups, splits.selection, a, b, n, seed, third=third, index=index)
        validate_aspect_episodes(ep, labels, splits.groups, index, third=third)
        assert np.isin(ep.rows(), splits.selection).all(), f"{a}__{b}: a row outside the selection split"
        hashes[f"{a}__{b}"] = episodes_sha256(ep)
        for field in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt"):
            saved[f"{a}__{b}__{field}"] = getattr(ep, field)
        parts.append(ep)
    saved["pair_order"] = np.asarray([f"{a}__{b}" for a, b, _ in PAIRS])
    np.savez(out_dir / f"episodes_seed{seed}.npz", **saved)
    return parts, hashes


def row_lookups(data, annotations, rows):
    """image path and caption for each global row in ``rows`` (annotation index = data.sample_ids[row])."""
    rows = np.unique(rows)
    sid = data.sample_ids[rows]
    caps = join_captions(sid, annotations)
    return ({int(r): str(WIKIART / annotations[int(s)]["image"]) for r, s in zip(rows, sid)},
            {int(r): str(c) for r, c in zip(rows, caps)})


def cosine_baseline(data, selection, ep):
    img = np.full(data.img_features.shape, np.nan, np.float32)
    txt = np.full(data.txt_features.shape, np.nan, np.float32)
    img[selection], txt[selection] = data.img_features[selection], data.txt_features[selection]
    return cosine_scores(EvalInputs(img, txt), ep)


def make_prompt(build_messages, ep, i, cond, d, perm, path, cap):
    si, st, ci, ct, _ = ep.condition(cond)
    supports = [(path[int(x)], cap[int(y)]) for x, y in zip(si[i], st[i])]
    contrasts = [(path[int(x)], cap[int(y)]) for x, y in zip(ci[i], ct[i])]
    cand = ep.candidates[i][perm]                                   # letter j shows candidate column perm[j]
    anchor = int(ep.anchor[i])
    if d == "i2t":
        return build_messages((path[anchor], None), [cap[int(c)] for c in cand], supports, contrasts, d)
    return build_messages((None, cap[anchor]), [path[int(c)] for c in cand], supports, contrasts, d)


def fingerprint(ep, perms, path, cap, hashes, model_id=MODEL_ID):
    """Everything that must be identical for a partial file to be resumable."""
    from src.eval.mllm_reranker import INSTRUCTION, build_messages
    sha = lambda x: hashlib.sha256(x.encode()).hexdigest()
    sample = {d: sha(json.dumps(make_prompt(build_messages, ep, 0, "a", d, np.arange(13), path, cap)))
              for d in DIRECTIONS}
    return json.dumps({"model": model_id, "max_pixels": MAX_PIXELS, "episodes_sha256": hashes,
                       "instruction": sha(INSTRUCTION), "sample_prompt": sample,
                       "perms": sha(perms.tobytes().hex())}, sort_keys=True)


def run_mllm(ep, perms, path, cap, out_dir, rec, fp, model_id=MODEL_ID):
    from src.eval.mllm_reranker import QwenReranker, build_messages, unpermute
    n = len(ep.anchor)
    scores = np.full((len(CONDITIONS), len(DIRECTIONS), n, 13), np.nan)
    done, times = 0, []
    partial = out_dir / "probe_partial.npz"
    if partial.exists():
        z = np.load(partial)
        assert str(z["fingerprint"]) == fp, "partial file was made with other episodes, prompts, model or permutations"
        scores, done, times = z["scores"], int(z["done"]), list(z["times"])
        print(f"resuming at episode {done}/{n}", flush=True)
    rr = None
    if done < n:
        t_load = time.time()
        rr = QwenReranker(model_id, "cuda", MAX_PIXELS)
        rec["model_load_s"] = time.time() - t_load

    def save():
        tmp = out_dir / "probe_partial.tmp.npz"
        np.savez(tmp, scores=scores, perms=perms, done=done, times=np.asarray(times), fingerprint=fp)
        os.replace(tmp, partial)

    for i in range(done, n):
        t0 = time.time()
        for ci, cond in enumerate(CONDITIONS):
            for di, d in enumerate(DIRECTIONS):
                perm = perms[i, ci, di]
                msgs = make_prompt(build_messages, ep, i, cond, d, perm, path, cap)
                logits = rr.score(msgs)
                scores[ci, di, i] = unpermute(logits, perm)           # back to the episode's column order
        times.append(time.time() - t0)
        done = i + 1
        if done % CHECKPOINT_EVERY == 0 or done == n:
            save()
            print(f"episode {done}/{n}, {np.mean(times[-CHECKPOINT_EVERY:]):.2f} s/episode (4 prompts)", flush=True)
    rec["episode_s_mean"], rec["prompt_s_mean"] = float(np.mean(times)), float(np.mean(times)) / 4
    rec["prompt_layout"] = "explicit line breaks (v2)"
    return {c: {d: scores[ci, di] for di, d in enumerate(DIRECTIONS)} for ci, c in enumerate(CONDITIONS)}


def sub(scores, sl):
    return {c: {d: scores[c][d][sl] for d in DIRECTIONS} for c in CONDITIONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    t_start = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_artelingo()
    splits = artelingo_splits(data)
    labels = artelingo_aspect_labels(data)
    annotations = json.load(open(ANNOTATIONS_PATH))
    parts, hashes = build_episodes(data, splits, labels, args.n, args.seed, out_dir)
    ep = concat_episodes(parts)
    assert np.isin(ep.rows(), splits.selection).all()
    path, cap = row_lookups(data, annotations, ep.rows())
    perms = np.stack([np.random.default_rng([args.seed, i]).permuted(
        np.tile(np.arange(13), (len(CONDITIONS), len(DIRECTIONS), 1)), axis=-1) for i in range(len(ep.anchor))])
    rec = {"model": args.model, "max_pixels": MAX_PIXELS, "n_per_pair": args.n, "seed": args.seed,
           "episodes_sha256": hashes}
    t0 = time.time()
    mllm = run_mllm(ep, perms, path, cap, out_dir, rec, fingerprint(ep, perms, path, cap, hashes, args.model),
                    args.model)
    rec["probe_total_s"] = time.time() - t0
    rec["wall_total_s"] = time.time() - t_start
    import torch
    rec["peak_gpu_mem_bytes"] = int(torch.cuda.max_memory_allocated())
    sha_file = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    rec["runner_sha256"] = sha_file(__file__)
    rec["mllm_reranker_sha256"] = sha_file(ROOT / "src/eval/mllm_reranker.py")
    cos = cosine_baseline(data, splits.selection, ep)
    clusters = splits.groups[ep.anchor]
    pa_m, pa_c = per_anchor(mllm), per_anchor(cos)
    result = dict(rec, pairs={}, pooled={"mllm": summarize(pa_m, clusters), "cosine": summarize(pa_c, clusters)})
    for k, (a, b, _) in enumerate(PAIRS):
        sl = slice(k * args.n, (k + 1) * args.n)
        pm, pc = per_anchor(sub(mllm, sl)), per_anchor(sub(cos, sl))
        result["pairs"][f"{a}__{b}"] = {"mllm": summarize(pm, clusters[sl]), "cosine": summarize(pc, clusters[sl])}
    result["compare_pooled"] = {m: compare(pa_m, pa_c, clusters, m) for m in ("r1", "gain")}
    result["mllm_works"] = bool(all(result["compare_pooled"][m]["ci95"][0] > 0 for m in ("r1", "gain")))
    json.dump(result, open(out_dir / "probe.json", "w"), indent=2)
    for name, s in result["pooled"].items():
        print(name, {m: round(v["point"], 2) for m, v in s.items()})
    print("compare", result["compare_pooled"], "\nMLLM works (pre-registered rule):", result["mllm_works"])
    np.savez(out_dir / "per_anchor.npz", **{f"mllm__{m}": v for m, v in pa_m.items()},
             **{f"cosine__{m}": v for m, v in pa_c.items()}, anchor_group=clusters)


if __name__ == "__main__":
    main()
