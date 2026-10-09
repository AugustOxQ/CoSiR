"""Round 6 describe-then-score verbaliser (DECISION_RULE.md of this folder: section 7 item 1, section 8 item 3;
contracts section 9; settings in dts_settings.json): one greedy call of Qwen3-VL-8B-Instruct per (episode, condition,
wording), at most 32 new tokens, batch size 1.

The message. One user turn: "Group A:", its 4 pairs, "Group B:", its 4 pairs, then the wording (settings
verbaliser.message). Pair i of a group is the image of row pairs_x_img[e, i-1] and the caption of row
pairs_x_txt[e, i-1], in episode column order. Condition a shows pairs_a as Group A (the supports) and pairs_b as Group B
(the contrasts); condition b swaps them. Images reach the processor as file paths under the WikiArt root (the processor
loads the pixels; the path is never in the text).

Inputs and outputs. The job folder holds rows_manifest.npz and verbalise_input.npz (r6_gpu_inputs.py); nothing else is
read. Each call writes one line {"seed", "episode_index", "condition", "wording", "answer"} (the raw answer,
unnormalised) to <out>/phrases_<wording>.jsonl; the files are rewritten atomically every 50 calls, and a rerun into the
same folder resumes, skipping the keys already written (provenance.json's fingerprint must match). No metric is
computed or printed.

Run (DAS6 through scripts/run_r6_verbalise.sh; the local GPU only under its lock):
    python r6_gpu_verbalise.py --job-dir <job folder> --out <output folder> --wordings W1,W2 [--start i] [--stop j]
    ... --check-only            inputs, images, snapshot, then the first wording on two episodes, both conditions
    ... --check-only --no-model the same without loading the model (no GPU)
--start/--stop are positions in verbalise_input.npz (a shard is [start, stop)).
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_gpu_common as G  # noqa: E402

FIELDS = ("seed", "episode_index", "condition", "wording", "answer")
KEY_FIELDS = FIELDS[:4]
DEFAULT_IMAGE_ROOT = "/data/PDD/wikiart_proj/wikiart"


# ---------------------------------------------------------------- the message (rule section 7 item 1)

def _text(s) -> dict:
    return {"type": "text", "text": s}


def group_rows(inp, pos, condition, settings):
    """((image rows, caption rows) of Group A, (image rows, caption rows) of Group B) of input position ``pos``."""
    G._require(condition in G.CONDITIONS, f"condition {condition!r} is not a or b", ValueError)
    groups = settings["verbaliser"]["conditions"][condition]
    return tuple((inp[f"{groups[g]}_img"][pos], inp[f"{groups[g]}_txt"][pos]) for g in ("group_a", "group_b"))


def build_messages(group_a, group_b, wording, settings) -> list:
    """The chat messages of one call: group_a and group_b are 4 (image path, caption) pairs each."""
    tm = settings["verbaliser"]["message"]
    content = []
    for name, pairs in (("A", group_a), ("B", group_b)):
        G._require(len(pairs) == G.NUM_PAIRS, f"Group {name} has {len(pairs)} pairs, not {G.NUM_PAIRS}")
        content.append(_text(tm["group_header"].format(group=name)))
        for i, (image, caption) in enumerate(pairs, 1):
            content += [_text(tm["pair_label"].format(i=i)), {"type": "image", "image": str(image)},
                        _text(tm["pair_caption"].format(caption=caption))]
    content.append(_text(tm["wording"].format(wording=wording)))
    return [{"role": tm["role"], "content": content}]


def episode_messages(inp, manifest, pos, condition, wording_id, settings, image_root) -> list:
    (ai, at), (bi, bt) = group_rows(inp, pos, condition, settings)
    return build_messages(manifest.items(ai, at, image_root), manifest.items(bi, bt, image_root),
                          settings["verbaliser"]["wordings"][wording_id], settings)


# ---------------------------------------------------------------- the calls, checkpoints and resume

def plan(inp, start, stop, wordings) -> list:
    """(position, wording, condition) of every call of [start, stop), episode-major."""
    return [(p, w, c) for p in range(start, stop) for w in wordings for c in G.CONDITIONS]


def call_key(inp, pos, wording, condition) -> tuple:
    return (int(inp["seed"]), int(inp["episode_index"][pos]), condition, wording)


def run_calls(inp, manifest, settings, out_dir, wordings, start, stop, generate, image_root,
              every=G.CHECKPOINT_EVERY, max_calls=None, log=print) -> dict:
    """Call ``generate(messages) -> str`` for every planned key not yet in <out_dir>/phrases_<w>.jsonl; checkpoint
    every ``every`` calls. ``max_calls`` stops early (tests)."""
    out_dir = Path(out_dir)
    calls = plan(inp, start, stop, wordings)
    allowed = {w: {call_key(inp, p, w, c) for p, ww, c in calls if ww == w} for w in wordings}
    stores = {w: G.KeyedJsonl(out_dir / f"phrases_{w}.jsonl", FIELDS, KEY_FIELDS) for w in wordings}
    for w, st in stores.items():
        st.load(allowed[w])
    todo = [(p, w, c) for p, w, c in calls if call_key(inp, p, w, c) not in stores[w]]
    log(f"verbaliser: {len(calls)} calls planned in [{start}, {stop}), {len(calls) - len(todo)} already written, "
        f"{len(todo)} to run")
    n, t0 = 0, time.time()
    for p, w, c in todo:
        answer = generate(episode_messages(inp, manifest, p, c, w, settings, image_root))
        G._require(isinstance(answer, str), f"the model returned {type(answer).__name__}, not text")
        seed, ep, _, _ = call_key(inp, p, w, c)
        stores[w].add({"seed": seed, "episode_index": ep, "condition": c, "wording": w, "answer": answer}, allowed[w])
        n += 1
        if n % every == 0:
            for st in stores.values():
                st.save()  # guard:checkpoint
            log(f"verbaliser: {n}/{len(todo)} calls, {(time.time() - t0) / n:.2f} s per call")
        if max_calls is not None and n >= max_calls:
            break
    for st in stores.values():
        st.save()
    complete = all(len(stores[w]) == len(allowed[w]) for w in wordings)
    return {"n_planned": len(calls), "n_done_before": len(calls) - len(todo), "n_called": n, "complete": complete,
            "s_per_call": (time.time() - t0) / n if n else None}


def make_generate(processor, model, settings):
    max_new = settings["generation"]["verbaliser"]["max_new_tokens"]

    def generate(messages):
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        return G.generate_texts(model, processor, inputs, max_new, settings)[0]
    return generate


# ---------------------------------------------------------------- command line

def check_images(inp, manifest, start, stop, image_root) -> int:
    """Every image the calls of [start, stop) show exists under ``image_root``."""
    rows = set()
    for k in ("pairs_a_img", "pairs_b_img"):
        rows.update(inp[k][start:stop].ravel().tolist())
    rels = sorted({str(manifest.image_relpath[i]) for i in manifest.index(sorted(rows)).tolist()})
    missing = [r for r in rels if not (Path(image_root) / r).is_file()]
    G._require(not missing, f"{len(missing)} of {len(rels)} images missing under {image_root}, "
                            f"e.g. {missing[:3]}")  # guard:images
    return len(rels)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Round 6 DTS verbaliser (rule section 7 item 1); prints no metric")
    ap.add_argument("--job-dir", required=True, help="folder with rows_manifest.npz and verbalise_input.npz")
    ap.add_argument("--out", required=True, help="output folder (phrases_<wording>.jsonl, provenance.json)")
    ap.add_argument("--wordings", required=True, help="comma-separated, from W1, W2, W3, W4")
    ap.add_argument("--start", type=int, default=0, help="first input position (default 0)")
    ap.add_argument("--stop", type=int, default=None, help="end input position, exclusive (default: all)")
    ap.add_argument("--check-only", action="store_true",
                    help="check inputs, images and snapshot, then run the first wording on two episodes")
    ap.add_argument("--no-model", action="store_true", help="with --check-only: stop before loading the model")
    ap.add_argument("--hub-cache", default=None, help="HF hub cache (default $HF_HUB_CACHE)")
    ap.add_argument("--image-root", default=os.environ.get("COSIR_WIKIART_DIR") or DEFAULT_IMAGE_ROOT)
    args = ap.parse_args(argv)
    args.wordings = [w for w in args.wordings.split(",") if w]
    if not args.wordings or len(set(args.wordings)) != len(args.wordings) \
            or not set(args.wordings) <= set(G.WORDING_IDS):
        ap.error(f"--wordings must be distinct ids from {', '.join(G.WORDING_IDS)}")
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
    inp = G.load_verbalise_input(job / "verbalise_input.npz", manifest)
    n = len(inp["episode_index"])
    stop = n if args.stop is None else args.stop
    G._require(stop <= n, f"--stop {stop} is beyond the input's {n} episodes")
    n_images = check_images(inp, manifest, args.start, stop, args.image_root)
    snap = G.snapshot_dir(G.hub_cache(args.hub_cache), settings["model"]["id"], settings["model"]["snapshot"])
    snap_info = G.check_snapshot(snap)
    fingerprint = {"job": "r6_gpu_verbalise", "model_id": settings["model"]["id"],
                   "snapshot": settings["model"]["snapshot"], "settings_sha256": settings_sha,
                   "scripts_sha256": G.script_shas(__file__, G.__file__),
                   "inputs_sha256": {f: G.sha256_file(job / f) for f in ("rows_manifest.npz", "verbalise_input.npz")}}
    print(f"verbaliser inputs ok: seed {inp['seed']}, {n} episodes, range [{args.start}, {stop}), "
          f"{len(manifest.rows)} manifest rows, {n_images} images, snapshot {snap_info['files']} files, "
          f"settings {settings_sha[:12]}", flush=True)
    run = {"args": {k: v for k, v in vars(args).items()}, "stop": stop, "seed": inp["seed"], "n_input": n,
           "snapshot": snap_info}
    if args.check_only:
        first = min(args.start + 2, stop)
        for p, w, c in plan(inp, args.start, first, args.wordings[:1]):
            episode_messages(inp, manifest, p, c, w, settings, args.image_root)
        if args.no_model:
            print("check-only --no-model: messages built; stopping before the model", flush=True)
            return 0
        out = Path(args.out) / f"check_only_{G.amsterdam_now().replace(' ', '_').replace(':', '')}"
        prov = G.begin_provenance(out, fingerprint, dict(run, check_only=True))
        t0 = time.time()
        processor, model = G.load_model(snap, settings)
        load_s = time.time() - t0
        res = run_calls(inp, manifest, settings, out, args.wordings[:1], args.start, first,
                        make_generate(processor, model, settings), args.image_root, every=1)
        import torch
        G.end_provenance(out, prov, status="check-only", model_load_s=load_s, versions=G.versions(),
                         peak_gpu_mem_bytes=int(torch.cuda.max_memory_allocated()), **res)
        print(f"check-only: model loaded in {load_s:.0f} s, {res['n_called']} calls, {res['s_per_call']:.2f} s per "
              f"call, outputs in {out}", flush=True)
        return 0
    out = Path(args.out)
    prov = G.begin_provenance(out, fingerprint, run)
    t0 = time.time()
    processor, model = G.load_model(snap, settings)
    load_s = time.time() - t0
    res = run_calls(inp, manifest, settings, out, args.wordings, args.start, stop,
                    make_generate(processor, model, settings), args.image_root)
    import torch
    G.end_provenance(out, prov, status="complete" if res["complete"] else "partial", model_load_s=load_s,
                     versions=G.versions(), peak_gpu_mem_bytes=int(torch.cuda.max_memory_allocated()), **res)
    print(f"verbaliser: {'complete' if res['complete'] else 'partial'}, outputs in {out}", flush=True)
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
