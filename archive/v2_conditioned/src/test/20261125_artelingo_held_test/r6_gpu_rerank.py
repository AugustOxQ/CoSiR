"""Round 6 in-context MLLM reranker, the reported baseline "MLLM" (DECISION_RULE.md of this folder: section 2, section 8
item 3, section 10 item 2; contracts section 9; gpu_path section 1, job J6): Qwen3-VL-8B-Instruct reads the example and
counter-example pairs, the query and 13 lettered candidates, and its next-token scores for the letters A..M are the
candidate scores. The prompt and the scorer are src/eval/mllm_reranker.py's (build_messages, QwenReranker, fp32 letter
scores from the last hidden state, max_pixels 256*28*28); that file is loaded by its path from the code checkout, so this
module imports neither src nor r6_common.

Inputs. The job folder: rows_manifest.npz and rerank_input.npz (r6_gpu_inputs.py): row ids, neutral image names and
captions; per episode the query row, the four example pairs a and b, and `cand_shown` (n, 2 cond, 2 dir, 13), the
candidate row ids already permuted on the CPU. The job scores the candidates in the order shown (letter j = shown
column j, the identity permutation) and cannot know the candidate order, the target, the anchor's aspect or a label.

Prompts. Condition a: supports = pairs_a, contrasts = pairs_b; condition b swaps them. Direction i2t: the query is the
image of query_row, the candidates are the captions of the shown rows; t2i: the query is the caption of query_row, the
candidates are the images of the shown rows. Pair k of a group is the image of row pairs_x_img[e, k] with the caption of
row pairs_x_txt[e, k]. Images reach the processor as files <image dir>/<neutral name>.

Output (<out>, outputs/r6_rerank/<job>/): scores.npz {episode_index (m,), scores_shown (m, 2, 2, 13) float32} for the
episodes done so far (rewritten atomically every 50 episodes; a rerun into the same folder resumes), and provenance.json
(model, snapshot, max_pixels, prompt and script hashes, versions, timings). The CPU side un-permutes with its own
permutations and merges shards. No metric is computed or printed. On DAS6 each launch has its own worktree: a relaunch
passes the earlier launch's output folder as --prior (read only, same fingerprint).

Run (DAS6 through scripts/run_r6_rerank.sh; the local GPU only under its lock):
    python r6_gpu_rerank.py --job-dir <job> --out <dir> [--start i] [--stop j] [--prior <dir>]
    ... --check-only             inputs, images, snapshot, then 2 episodes (8 prompts) scored with the model
    ... --check-only --no-model  the same without loading the model (no GPU): prompts are built for 2 episodes
Guards carry a `# guard:<name>` marker; the tests delete each on a copy.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_gpu_common as G  # noqa: E402
import r6_gpu_t12 as T  # noqa: E402

RERANKER_FILE = HERE.parents[1] / "eval" / "mllm_reranker.py"
DIRECTIONS = ("i2t", "t2i")
OUT_KEYS = ("episode_index", "scores_shown")


def load_reranker_module(path=None):
    """src/eval/mllm_reranker.py loaded by its path (it imports only numpy and torch)."""
    path = Path(path or RERANKER_FILE)
    spec = importlib.util.spec_from_file_location("r6_mllm_reranker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the prompts

def episode_prompt(inp, manifest, pos, ci, di, image_dir, build_messages):
    """The chat messages of input position ``pos``, condition index ``ci``, direction index ``di``; the candidates in
    the order shown (inp['cand_shown'])."""
    cond = G.CONDITIONS[ci]
    own, other = (("a", "b") if cond == "a" else ("b", "a"))
    pairs = lambda x: manifest.items(inp[f"pairs_{x}_img"][pos], inp[f"pairs_{x}_txt"][pos], image_dir)
    supports, contrasts = pairs(own), pairs(other)
    q = int(inp["query_row"][pos])
    shown = inp["cand_shown"][pos, ci, di]
    if DIRECTIONS[di] == "i2t":
        query = (manifest.items([q], [q], image_dir)[0][0], None)
        cands = [manifest.items([r], [r], image_dir)[0][1] for r in shown.tolist()]
    else:
        query = (None, manifest.items([q], [q], image_dir)[0][1])
        cands = [manifest.items([r], [r], image_dir)[0][0] for r in shown.tolist()]
    return build_messages(query, cands, supports, contrasts, DIRECTIONS[di])


def episode_image_names(inp, manifest, start, stop) -> list:
    """Neutral names of every image the episodes [start, stop) show."""
    rows = set(inp["query_row"][start:stop].tolist())
    rows.update(inp["cand_shown"][start:stop].ravel().tolist())
    for k in ("pairs_a_img", "pairs_b_img"):
        rows.update(inp[k][start:stop].ravel().tolist())
    return sorted({str(manifest.image_name[i]) for i in manifest.index(sorted(rows)).tolist()})


def check_images(inp, manifest, start, stop, image_dir) -> int:
    names = episode_image_names(inp, manifest, start, stop)
    missing = [x for x in names if not (Path(image_dir) / x).is_file()]
    G._require(not missing, f"{len(missing)} of {len(names)} images missing under {image_dir}, "
                            f"e.g. {missing[:3]}")  # guard:images
    return len(names)


# ---------------------------------------------------------------- scores, checkpoints and resume

def save_scores(out_dir, index, scores):
    """scores.npz with the episodes done so far (atomic)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(index)
    tmp = out_dir / "scores.tmp.npz"
    np.savez(tmp, episode_index=np.asarray(index, dtype=np.int64)[order],
             scores_shown=np.asarray(scores, dtype=np.float32).reshape(-1, 2, 2, 13)[order])
    os.replace(tmp, out_dir / "scores.npz")


def load_scores(path, allowed=None) -> dict:
    """{episode index: (2, 2, 13) float32} of a scores.npz; its keys, shapes and finiteness checked, and each episode
    must be one the run plans (``allowed``)."""
    path = Path(path)
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as z:
        G._require(sorted(z.files) == sorted(OUT_KEYS), f"{path}: keys {sorted(z.files)} differ from {sorted(OUT_KEYS)}")
        idx, sc = z["episode_index"], z["scores_shown"]
    G._require(idx.dtype == np.int64 and sc.dtype == np.float32 and sc.shape == (len(idx), 2, 2, 13)
               and len(set(idx.tolist())) == len(idx), f"{path}: episode_index / scores_shown layout")
    G._require(bool(np.isfinite(sc).all()), f"{path}: non-finite scores")
    G._require(allowed is None or set(idx.tolist()) <= set(allowed),
               f"{path}: holds episodes this run does not plan")  # guard:planned
    return {int(i): sc[k] for k, i in enumerate(idx.tolist())}


def load_prior(dirs, fingerprint, planned) -> dict:
    """Scores of earlier output folders (read only, same fingerprint) for the planned episodes."""
    out = {}
    for d in dirs:
        d = Path(d)
        prov = json.loads((d / "provenance.json").read_text())
        G._require(prov.get("fingerprint") == fingerprint,
                   f"{d}: written by another model, settings, script or input; not reusable")  # guard:prior_fingerprint
        for i, s in load_scores(d / "scores.npz").items():
            if i in planned:
                G._require(i not in out or np.array_equal(out[i], s), f"earlier outputs disagree on episode {i}")
                out[i] = s
    return out


def run_episodes(inp, manifest, out_dir, score, build_messages, image_dir, start, stop, every=G.CHECKPOINT_EVERY,
                 max_episodes=None, log=print, prior=None) -> dict:
    """Score every episode of [start, stop) not yet in <out_dir>/scores.npz nor in ``prior`` ({episode: scores}):
    ``score(messages) -> 13 letter scores`` for the 4 prompts (cond, dir) of an episode in the order shown; checkpoint
    every ``every`` episodes. ``max_episodes`` stops early (tests)."""
    out_dir = Path(out_dir)
    planned = {int(inp["episode_index"][p]): p for p in range(start, stop)}
    done = load_scores(out_dir / "scores.npz", planned)
    n_before, n_prior = len(done), 0
    for e, s in (prior or {}).items():
        if e in planned and e not in done:
            done[e] = s
            n_prior += 1
    todo = [e for e in planned if e not in done]
    log(f"reranker: {len(planned)} episodes planned in [{start}, {stop}), {n_before} already written, {n_prior} copied "
        f"from earlier outputs, {len(todo)} to run")

    def save():
        save_scores(out_dir, list(done), [done[e] for e in done])

    n, t0 = 0, time.time()
    if n_prior:
        save()
    for e in todo:
        pos = planned[e]
        sc = np.empty((2, 2, 13), dtype=np.float32)
        for ci in range(2):
            for di in range(2):
                v = np.asarray(score(episode_prompt(inp, manifest, pos, ci, di, image_dir, build_messages)))
                G._require(v.shape == (13,) and bool(np.isfinite(v).all()), f"episode {e}: the scorer returned {v.shape}")
                sc[ci, di] = v          # the shown order: no un-permuting here
        done[e] = sc
        n += 1
        if n % every == 0:
            save()  # guard:rerank_checkpoint
            log(f"reranker: {n}/{len(todo)} episodes, {(time.time() - t0) / n:.2f} s per episode")
        if max_episodes is not None and n >= max_episodes:
            break
    save() if (done or n_prior) else None
    return {"n_planned": len(planned), "n_done_before": n_before, "n_prior": n_prior, "n_scored": n,
            "complete": all(e in done for e in planned), "s_per_episode": (time.time() - t0) / n if n else None}


# ---------------------------------------------------------------- command line

def sha_text(s) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Round 6 in-context MLLM reranker (reported baseline); prints no metric")
    ap.add_argument("--job-dir", required=True, help="folder with rows_manifest.npz and rerank_input.npz")
    ap.add_argument("--out", required=True, help="output folder (scores.npz, provenance.json)")
    ap.add_argument("--start", type=int, default=0, help="first input position (default 0)")
    ap.add_argument("--stop", type=int, default=None, help="end input position, exclusive (default: all)")
    ap.add_argument("--check-only", action="store_true", help="check inputs, images and snapshot, then score 2 episodes")
    ap.add_argument("--no-model", action="store_true", help="with --check-only: stop before loading the model")
    ap.add_argument("--prior", action="append", default=[],
                    help="an earlier output folder of this job (read only, same fingerprint) whose scores are reused")
    ap.add_argument("--hub-cache", default=None, help="HF hub cache (default $HF_HUB_CACHE)")
    ap.add_argument("--image-dir", default=os.environ.get("R6_IMAGE_DIR") or G.DEFAULT_IMAGE_DIR,
                    help="folder of the images under their neutral names (default $R6_IMAGE_DIR, else the node's)")
    args = ap.parse_args(argv)
    if args.no_model and not args.check_only:
        ap.error("--no-model needs --check-only")
    if args.start < 0 or (args.stop is not None and args.stop <= args.start):
        ap.error("need 0 <= --start < --stop")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    settings, settings_sha = G.load_settings()
    job = Path(args.job_dir)
    manifest = G.Manifest(job / "rows_manifest.npz")
    inp = T.load_rerank_input(job / "rerank_input.npz", manifest)
    n = len(inp["episode_index"])
    stop = n if args.stop is None else args.stop
    G._require(stop <= n, f"--stop {stop} is beyond the input's {n} episodes")
    n_images = check_images(inp, manifest, args.start, stop, args.image_dir)
    mm = settings["model"]
    max_pixels = int(mm["processor_kwargs"]["max_pixels"])
    G._require(max_pixels == 256 * 28 * 28, "settings: max_pixels is not the probe's 256*28*28")
    snap = G.snapshot_dir(G.hub_cache(args.hub_cache), mm["id"], mm["snapshot"])
    snap_info = G.check_snapshot(snap)
    rr = load_reranker_module()
    fingerprint = {"job": "r6_gpu_rerank", "model_id": mm["id"], "snapshot": mm["snapshot"], "max_pixels": max_pixels,
                   "instruction_sha256": sha_text(rr.INSTRUCTION),
                   "scripts_sha256": G.script_shas(__file__, G.__file__, RERANKER_FILE),
                   "inputs_sha256": {f: G.sha256_file(job / f) for f in ("rows_manifest.npz", "rerank_input.npz")}}
    print(f"reranker inputs ok: seed {inp['seed']}, {n} episodes, range [{args.start}, {stop}), "
          f"{len(manifest.rows)} manifest rows, {n_images} images, snapshot {snap_info['files']} files", flush=True)
    run = {"args": {k: v for k, v in vars(args).items()}, "stop": stop, "seed": inp["seed"], "n_input": n,
           "snapshot": snap_info}
    if args.check_only:
        first = min(args.start + 2, stop)
        for p in range(args.start, first):
            for ci in range(2):
                for di in range(2):
                    episode_prompt(inp, manifest, p, ci, di, args.image_dir, rr.build_messages)
        v = G.check_imports()
        print(f"imports ok: torch {v['torch']}, transformers {v['transformers']}", flush=True)
        if args.no_model:
            print("check-only --no-model: prompts built; stopping before the model", flush=True)
            return 0
        out = Path(args.out) / f"check_only_{G.amsterdam_now().replace(' ', '_').replace(':', '')}"
        prov = G.begin_provenance(out, fingerprint, dict(run, check_only=True))
        t0 = time.time()
        model = rr.QwenReranker(str(snap), "cuda", max_pixels)
        load_s = time.time() - t0
        res = run_episodes(inp, manifest, out, model.score, rr.build_messages, args.image_dir, args.start, first,
                           every=1)
        import torch
        G.end_provenance(out, prov, status="check-only", model_load_s=load_s, versions=G.versions(),
                         peak_gpu_mem_bytes=int(torch.cuda.max_memory_allocated()), **res)
        print(f"check-only: model loaded in {load_s:.0f} s, {res['n_scored']} episodes, "
              f"{res['s_per_episode']:.2f} s per episode, outputs in {out}", flush=True)
        return 0
    out = Path(args.out)
    prior = load_prior(args.prior, fingerprint, set(inp["episode_index"][args.start:stop].tolist()))
    prov = G.begin_provenance(out, fingerprint, run)
    t0 = time.time()
    model = rr.QwenReranker(str(snap), "cuda", max_pixels)
    load_s = time.time() - t0
    G.update_provenance(out, prov, model_load_s=load_s, versions=G.versions())
    res = run_episodes(inp, manifest, out, model.score, rr.build_messages, args.image_dir, args.start, stop,
                       prior=prior)
    import torch
    G.end_provenance(out, prov, status="complete" if res["complete"] else "partial",
                     peak_gpu_mem_bytes=int(torch.cuda.max_memory_allocated()), **res)
    print(f"reranker: {'complete' if res['complete'] else 'partial'}, outputs in {out}", flush=True)
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
